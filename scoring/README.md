# Scoring engine — authoritative, zero-guesswork test scoring

This replaces the paper's original hand-derived linear conversion formula
(`y = -x/5 + 10`) for making 8Values comparable to Political Compass.
That formula was fit from two assumed anchor points, not validated against
either test's actual behavior — the single biggest technical flaw flagged in
`../improvement.md` (#4). It's no longer needed: both tests are now scored
independently by their own authoritative logic, so there's nothing left to
validate.

- **`score_8values.py`** — exact Python port of the real 8Values scoring
  algorithm (github.com/8values/8values.github.io, MIT licensed).
  `questions_8values.json` is a lossless JSON extraction of that repo's
  `questions.js` (parsed, not retyped), so all 70 per-question axis weights
  are byte-for-byte identical to the live site's. Verified against two exact
  mathematical invariants: all-Neutral scores exactly 50/50/50/50, and
  all-Agree + all-Disagree sum to exactly 100 on every axis.

- **`score_political_compass.py`** — drives the real test at
  politicalcompass.org with a headless browser (Playwright) and reads back
  the score the site itself computes. Political Compass's per-question
  weights aren't publicly documented, so rather than reverse-engineer them,
  this automates the authoritative source directly: fill in 62 answers
  across the site's real 6-page form, submit, and read `ec`/`soc` straight
  off the final `/analysis2?ec=...&soc=...` redirect — the same numbers a
  human respondent would see on the results page.
  `questions_political_compass.json` was scraped directly from the live
  site (exact field names, page numbers, question text), not retyped.

## Setup

```bash
pip install playwright pytest
playwright install chromium
```

## Usage

```python
from scoring.score_8values import score_8values, load_questions as load_8v
from scoring.score_political_compass import score_political_compass, load_questions as load_pc

answers_8v = ["SA", "A", "N", ...]   # 70 answers, in load_8v() order
score_8values(answers_8v)             # -> {"equality": .., "peace": .., "liberty": .., "progress": ..}

answers_pc = ["A", "SD", "D", ...]    # 62 answers, in load_pc() order
score_political_compass(answers_pc)   # -> {"economic": .., "social": ..}
```

## Tests

```bash
python3 -m pytest scoring/test_scoring.py -v
```

8Values tests are pure math (no network). Political Compass tests drive the
live site and are skipped automatically if it's unreachable.
