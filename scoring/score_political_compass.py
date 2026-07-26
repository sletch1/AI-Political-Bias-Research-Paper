"""Authoritative Political Compass scoring via headless-browser automation.

Political Compass's per-question weights are not publicly documented (even
recent academic factor-analysis papers on the test note this). Rather than
guess at or reverse-engineer them, this module drives the *real* test at
politicalcompass.org with a headless browser and reads the score the site
itself computes -- eliminating any need to validate a home-grown scoring
formula for this test, the same way score_8values.py eliminates it for
8Values by using that test's own open-source logic directly.

`questions_political_compass.json` in this directory holds all 62 questions,
scraped directly from the live site (field name, page number, question text)
via Playwright -- not retyped by hand. Site structure (confirmed empirically,
2026-07):
  - 6 pages at https://www.politicalcompass.org/test/en?page=1..6
  - each page is one <form method=POST action=/test/en> with radio groups
    named per-question (e.g. "globalisationinevitable"), values 0-3 for
    Strongly Disagree / Disagree / Agree / Strongly Agree
  - submitting carries running totals forward via hidden fields
    (carried_ec, carried_soc, page) -- we don't need to touch these, the
    browser does
  - the final page's submit redirects to
    /analysis2?ec=<economic>&soc=<social> -- the authoritative result

Requires: pip install playwright && playwright install chromium
"""

import json
import re
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# The live site is a small, ad-supported, volunteer-run test -- politely cap
# how many headless sessions hit it at once regardless of the caller's own
# concurrency, and retry on the transient network/ad-interception errors that
# running many concurrent sessions against it provokes.
_CONCURRENCY_LIMIT = threading.Semaphore(2)

QUESTIONS_PATH = Path(__file__).parent / "questions_political_compass.json"

# Political Compass's own 4-point scale, matching the real radio button values.
ANSWER_SCALE = {
    "SD": "0",  # Strongly Disagree
    "D": "1",   # Disagree
    "A": "2",   # Agree
    "SA": "3",  # Strongly Agree
}

BASE_URL = "https://www.politicalcompass.org/test/en"


def load_questions():
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def score_political_compass(answers, questions=None, headless=True, timeout_ms=30000, retries=3):
    """Retry wrapper around _score_political_compass_once: a fresh browser
    per attempt, with a module-wide concurrency cap so we don't overwhelm a
    small volunteer-run site with many simultaneous headless sessions."""
    last_err = None
    with _CONCURRENCY_LIMIT:
        for attempt in range(retries + 1):
            try:
                return _score_political_compass_once(answers, questions, headless, timeout_ms)
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))  # backoff: 2s, 4s, 6s
        raise last_err


def _score_political_compass_once(answers, questions=None, headless=True, timeout_ms=30000):
    """Drive the real Political Compass test and return its authoritative score.

    Parameters
    ----------
    answers : list[str]
        62 answers, one per question in `questions` order. Each is one of
        "SD", "D", "A", "SA" (or the raw site values "0","1","2","3").
    questions : list[dict], optional
        Defaults to the bundled `questions_political_compass.json`.
    headless : bool
        Run the browser headless (default) or visibly (useful for debugging).

    Returns
    -------
    dict: {"economic": float, "social": float} -- exactly the site's own
    `ec`/`soc` values from the final /analysis2 redirect, i.e. what a human
    would see on the "Your Political Compass" results page.
    """
    if questions is None:
        questions = load_questions()
    if len(answers) != len(questions):
        raise ValueError(f"expected {len(questions)} answers, got {len(answers)}")

    values = [ANSWER_SCALE.get(a, a) for a in answers]
    answer_by_name = {q["name"]: v for q, v in zip(questions, values)}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(f"{BASE_URL}?page=1", timeout=timeout_ms)

        for _ in range(10):  # 6 real pages; generous upper bound
            radios = page.query_selector_all("input[type=radio]")
            if not radios:
                break  # reached the final (results) page
            names_on_page = []
            seen = set()
            for r in radios:
                name = r.get_attribute("name")
                if name not in seen:
                    seen.add(name)
                    names_on_page.append(name)

            for name in names_on_page:
                if name not in answer_by_name:
                    raise KeyError(
                        f"no answer supplied for question field {name!r} "
                        "(question bank may be out of sync with the live site)"
                    )
                val = answer_by_name[name]
                page.check(f'input[name="{name}"][value="{val}"]')

            btn = page.query_selector("input[type=submit], button[type=submit]")
            with page.expect_navigation(timeout=timeout_ms):
                # force=True: a sticky ad iframe can intercept the click's
                # pointer-event otherwise (observed at concurrency > 1)
                btn.click(force=True, timeout=timeout_ms)

        final_url = page.url
        browser.close()

    m = re.search(r"[?&]ec=(-?[\d.]+)&soc=(-?[\d.]+)", final_url)
    if not m:
        raise RuntimeError(
            f"did not land on an /analysis2 results page as expected; got {final_url!r}"
        )
    return {"economic": float(m.group(1)), "social": float(m.group(2))}


if __name__ == "__main__":
    qs = load_questions()
    print(f"Loaded {len(qs)} questions across pages {sorted(set(q['page'] for q in qs))}.")

    print("All Strongly Agree:", score_political_compass(["SA"] * len(qs), qs))
    print("All Strongly Disagree:", score_political_compass(["SD"] * len(qs), qs))
