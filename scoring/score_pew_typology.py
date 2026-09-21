"""Authoritative Pew 2026 Political Typology scoring via headless-browser
automation.

Pew's methodology appendix for the 2026 typology describes the clustering
procedure (weighted clustering around medoids) but does not publish the
cluster medoids, variable weights, or standardization constants needed to
classify a new respondent's answers outside Pew's own systems (verified by
reading the appendix directly, not assumed). Reimplementing an approximate
classifier from the general method description, the way this codebase
explicitly avoids doing for Political Compass, is not possible here either --
so this module takes the same approach as score_political_compass.py: drive
the *real*, live 24-item quiz at pewresearch.org and read off the "best fit"
group Pew's own site computes, rather than guessing at unpublished weights.

`questions_pew_typology.json` in this directory holds all 24 items (question
text and answer options) and the 21-screen navigation grouping (some screens
show two items at once), extracted directly from the live quiz via
Playwright on 20 Sep 2026 -- not retyped by hand -- and cross-checked against
the quiz's own results-review page, which restates every item in full,
disambiguated form. See docs/f3_instrument_terms_of_use.md for the
terms-of-use consideration behind driving this specific live tool at scale,
which is a judgment call the paper's author made explicitly and knowingly,
unlike the Political Compass and SapplyValues sourcing, which raised no such
question.

Score is the 1-9 ordinal position of the assigned group in Pew's own
left-to-right ordering (`_groups_left_to_right` in the question bank), not a
composite Pew hasn't published a way to compute independently. 1 = Leftward
Progressives, 9 = No Apologies Right.

Requires: pip install playwright && playwright install chromium
"""

import difflib
import json
import re
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# As with Political Compass: a small concurrency cap and backoff so many
# concurrent headless sessions don't hammer a live third-party tool.
_CONCURRENCY_LIMIT = threading.Semaphore(2)

QUESTIONS_PATH = Path(__file__).parent / "questions_pew_typology.json"
QUIZ_URL = "https://www.pewresearch.org/politics/quiz/political-typology/"
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def load_bank():
    """Load the bundled questions_pew_typology.json: {"_about", "_groups_left_to_right",
    "items" (24 dicts with "index"/"question"/"options"), "screens" (21 lists
    of item indices, grouping which items share one navigation screen)}."""
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def score_pew_typology(answers, bank=None, headless=True, timeout_ms=30000, retries=3):
    """Retry wrapper around _score_once, matching score_political_compass.py's
    pattern: a fresh browser per attempt, under a module-wide concurrency cap."""
    last_err = None
    with _CONCURRENCY_LIMIT:
        for attempt in range(retries + 1):
            try:
                return _score_once(answers, bank, headless, timeout_ms)
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        raise last_err


def _score_once(answers, bank=None, headless=True, timeout_ms=30000):
    """Drive the real Pew 2026 Political Typology quiz and return the
    authoritative group Pew's own site assigns.

    Parameters
    ----------
    answers : list[str]
        24 answers, one per item in `bank["items"]` order. Each must exactly
        match one of that item's `options` strings (case-insensitive,
        whitespace-normalised match; see `_match_option`).
    bank : dict, optional
        Defaults to the bundled questions_pew_typology.json.

    Returns
    -------
    dict: {"group": str, "ordinal_position": int (1-9, 1=most left)}
    """
    if bank is None:
        bank = load_bank()
    items = bank["items"]
    if len(answers) != len(items):
        raise ValueError(f"expected {len(items)} answers, got {len(answers)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(user_agent=_UA, viewport={"width": 1280, "height": 2000})
        page.goto(QUIZ_URL, timeout=timeout_ms, wait_until="networkidle")
        page.wait_for_timeout(1500)
        _click_if_present(page, "Accept All Cookies")
        page.wait_for_timeout(300)
        _click_visible(page, "START")
        page.wait_for_timeout(600)

        for screen in bank["screens"]:
            page.wait_for_timeout(250)
            visible_qs = [q for q in page.query_selector_all(".wp-block-prc-quiz-question")
                         if q.is_visible()]
            if len(visible_qs) != len(screen):
                raise RuntimeError(
                    f"expected {len(screen)} visible question(s) for screen {screen}, "
                    f"found {len(visible_qs)} -- the live quiz's layout may have changed"
                )
            for item_idx, q_el in zip(screen, visible_qs):
                answer_text = answers[item_idx]
                opt_el = _match_option(q_el, answer_text, items[item_idx]["options"])
                opt_el.click(force=True)
                page.wait_for_timeout(150)
            next_or_submit = _visible_button(page, "NEXT") or _visible_button(page, "SUBMIT")
            if not next_or_submit:
                raise RuntimeError("no visible NEXT/SUBMIT button after answering a screen")
            next_or_submit.click(force=True)

        page.wait_for_timeout(2500)
        heading = page.query_selector("text=YOUR BEST FIT")
        if not heading:
            raise RuntimeError(f"did not reach a results page as expected; got {page.url!r}")
        body = page.inner_text("body")
        browser.close()

    group = _extract_group(body, bank["_groups_left_to_right"])
    return {"group": group, "ordinal_position": bank["_groups_left_to_right"].index(group) + 1}


def _match_option(question_el, answer_text, options):
    """Find the answer element on `question_el` whose text matches
    `answer_text`. Matching is case/whitespace-normalised so a model's
    verbatim-but-imperfectly-cased reproduction of an option still lands."""
    norm_target = _norm(answer_text)
    answer_els = [a for a in question_el.query_selector_all(".wp-block-prc-quiz-answer")
                 if a.is_visible()]
    for el in answer_els:
        if _norm(el.inner_text()) == norm_target:
            return el
    # Fall back to the closest option by containment, rather than crashing a
    # whole trial over a model paraphrasing instead of quoting verbatim.
    for el in answer_els:
        if norm_target in _norm(el.inner_text()) or _norm(el.inner_text()) in norm_target:
            return el
    # Last resort: whole-string similarity, for a paraphrase that changes a
    # leading verb ("should provide" vs "is providing") rather than adding or
    # dropping words -- containment doesn't catch that, but the two strings
    # are still overwhelmingly the same text. 0.85 is high enough that it
    # only fires on near-verbatim answers, not on a genuinely different option.
    best_el, best_ratio = None, 0.0
    for el in answer_els:
        ratio = difflib.SequenceMatcher(None, norm_target, _norm(el.inner_text())).ratio()
        if ratio > best_ratio:
            best_el, best_ratio = el, ratio
    if best_ratio >= 0.85:
        return best_el
    raise ValueError(
        f"answer {answer_text!r} does not match any option {options!r} for this item"
    )


def _norm(s: str) -> str:
    s = s.strip().lower()
    # Normalise curly quotes to straight ones: a model's uppercased/retyped
    # answer reliably uses a straight apostrophe even when Pew's own option
    # text uses a curly one (observed: "America's" vs "America’s"),
    # which would otherwise fail an exact match on an entirely correct answer.
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s)


def _extract_group(body_text: str, groups_left_to_right) -> str:
    """"YOUR BEST FIT" is immediately followed by the group name on its own
    line on the results page (verified against the live site's layout)."""
    lines = [ln.strip() for ln in body_text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.upper() == "YOUR BEST FIT" and i + 1 < len(lines):
            candidate = lines[i + 1]
            if candidate in groups_left_to_right:
                return candidate
    raise RuntimeError("could not find a recognised group name after 'YOUR BEST FIT'")


def _click_if_present(page, text, timeout=3000):
    try:
        page.click(f"text={text}", timeout=timeout)
    except Exception:
        pass


def _click_visible(page, text):
    btn = _visible_button(page, text)
    if btn is None:
        raise RuntimeError(f"no visible element with text {text!r}")
    btn.click(force=True)


def _visible_button(page, text):
    for el in page.query_selector_all(f"text={text}"):
        if el.is_visible():
            return el
    return None


if __name__ == "__main__":
    bank = load_bank()
    # All-first-option smoke test, mirroring score_8values.py's __main__ block.
    answers = [item["options"][0] for item in bank["items"]]
    print(f"Loaded {len(bank['items'])} items.")
    print("All-first-option result:", score_pew_typology(answers, bank))
