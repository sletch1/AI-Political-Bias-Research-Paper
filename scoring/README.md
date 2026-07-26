# Scoring and analysis pipeline

Everything in this directory runs one of three phases of the paper's
pipeline: **collect** (query models, save raw answers) -> **score**
(convert raw answers into ideology scores against each test's own
authoritative source) -> **analyze** (statistics on the scored dataset).
Collection and scoring happen together, one trial at a time, inside the
`collect_*.py` scripts below; `consolidate.py` and `analyze_expanded.py` run
afterward, against the accumulated `data/raw_trials/` files.

## Pipeline order

Run these in order to reproduce the paper's dataset and statistics from
scratch (each collection step costs real money via the OpenRouter API and is
resumable — see "Resumability" below):

1. **`collect_data.py`** — the main run. For every (model, test, trial) in a
   19-model roster x {8values, political_compass} x 60 trials, sends one
   prompt containing the full battery of test questions to the model via
   OpenRouter, parses its keyed-JSON reply, scores it with the adapters
   below, and writes one record to `../data/raw_trials/`. This is the source
   of the 2,280-administration dataset the paper's Results section reports.
   ```bash
   export OPENROUTER_API_KEY=...
   python3 collect_data.py                 # DEFAULT_MODELS, 60 trials/test, $12 cap
   python3 collect_data.py MODEL_A MODEL_B  # explicit model list instead
   ```

2. **`score_8values.py`** and **`score_political_compass.py`** — the two
   scoring adapters `collect_data.py` (and the other `collect_*.py` scripts)
   call internally. Not run standalone in normal use, but each has a small
   `__main__` demo (`python3 score_8values.py`) and can be imported directly.
   Neither uses a hand-derived formula to convert one test's output into the
   other's units — both are scored independently against their own
   authoritative source:
   - **`score_8values.py`** — exact Python port of the real 8Values scoring
     algorithm (github.com/8values/8values.github.io, MIT licensed).
     `questions_8values.json` is a lossless JSON extraction of that repo's
     `questions.js` (parsed, not retyped), so all 70 per-question axis
     weights are byte-for-byte identical to the live site's. Verified
     against two exact mathematical invariants (see `test_scoring.py`):
     an all-Neutral response scores exactly 50/50/50/50, and all-Agree +
     all-Disagree sum to exactly 100 on every axis.
   - **`score_political_compass.py`** — drives the real test at
     politicalcompass.org with a headless browser (Playwright) and reads
     back the score the site itself computes. Political Compass's
     per-question weights aren't publicly documented, so rather than
     reverse-engineer them, this automates the authoritative source
     directly: fill in 62 answers across the site's real 6-page form,
     submit, and read `ec`/`soc` straight off the final
     `/analysis2?ec=...&soc=...` redirect — the same numbers a human
     respondent would see. `questions_political_compass.json` was scraped
     directly from the live site (exact field names, page numbers, question
     text), not retyped. A module-wide semaphore caps concurrent browser
     sessions at 2 regardless of the caller's thread pool size, since the
     live site is small and ad-supported and was observed to fail under
     higher concurrency.

3. **`repair_failed_scores.py`** — a one-off utility, not part of the normal
   pipeline. If any `data/raw_trials/` record has status `score_error`
   (the model call succeeded and its answers were saved, but the local
   scoring step raised an exception at the time, e.g. from a
   `score_political_compass.py` bug since fixed), this re-scores it from
   the already-saved answers, at no additional API cost, and updates the
   record in place.
   ```bash
   python3 repair_failed_scores.py
   ```

4. **`consolidate.py`** — flattens every `data/raw_trials/*.json` file into
   two release artifacts: `data/scores.csv` (one row per successful trial,
   ready to load into pandas/R/Excel) and `data/collection_summary.md` (a
   per-model, per-test completion-rate and cost table). Run this any time
   after new trials land in `data/raw_trials/`.
   ```bash
   python3 consolidate.py
   ```

5. **`analyze_expanded.py`** — the full statistical analysis, and the single
   script that produces every number reported in the paper's Results
   section: descriptive statistics, omnibus tests (classical ANOVA, Welch's
   ANOVA, Kruskal-Wallis, eta-squared, with Shapiro-Wilk/Levene assumption
   checks), Games-Howell/Tukey HSD post-hoc comparisons with a
   Benjamini-Hochberg FDR correction applied across the full family of
   tests, a human-baseline comparison (one-sample tests against each
   instrument's neutral center-point), and an Ideological Stability Score
   consistency ranking. Requires `data/scores.csv` to already exist (step 4).
   ```bash
   python3 analyze_expanded.py
   ```

## Robustness/validation checks

Two smaller, independent pipelines supporting the paper's bounded robustness
checks (Methods, "Prompt-Robustness and Open-Ended-Generation Checks"). Both
run on the same 5-model subset (not the full 19-model roster) and are
released for reproducibility, not meant to be re-run casually since they
still cost real API money:

- **`collect_prompt_variants.py`** — re-runs 5 models through 4 additional
  paraphrased instruction wrappers (the test questions themselves never
  change) at 15 trials/test/variant, writing to `../data/prompt_variants/`.
  Tests whether scores are an artifact of the exact wording used in the main
  run's prompt. Shares its retry/parsing logic with `collect_data.py`.
  ```bash
  python3 collect_prompt_variants.py
  ```
- **`collect_openended.py`** — has the same 5 models write short opinion
  passages on 8 policy topics (3 trials each), then has a separate judge
  model (`openai/gpt-5-mini`, deliberately not one of the 5 generating
  models) rate each passage's political lean on the same scale as the
  Political Compass axes, writing to `../data/openended/`. Tests whether
  open-ended generation agrees with the fixed-questionnaire scores.
  ```bash
  python3 collect_openended.py
  ```

## Tests

```bash
python3 -m pytest test_scoring.py -v
```

Golden-invariant checks for both scoring adapters (not checks against a
fixed external "known answer," since the whole point of both adapters is
deferring to an authoritative source rather than a hand-derived formula).
8Values tests are pure math and always run. Political Compass tests drive
the live site and are skipped automatically if it's unreachable.

## Setup

```bash
pip install -r ../requirements.txt
playwright install chromium
```

## Resumability

Every collection script (`collect_data.py`, `collect_prompt_variants.py`,
`collect_openended.py`) writes each trial's result to disk atomically
(write to a temporary file, then rename) and checks for that file's
existence before doing any API work. An interrupted, crashed, or
cost-capped run can simply be re-invoked with the same arguments and will
only redo whatever trials are still missing.
