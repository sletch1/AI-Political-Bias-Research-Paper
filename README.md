# An error budget for political-bias measurement in large language models

Code, data, and analysis supporting the paper **"An error budget for
political-bias measurement in large language models"** (`main.tex` / `main.pdf`,
supplementary information in `supplementary.tex` / `supplementary.pdf`).
Target venue: see `impr.md` (private, not part of this repository) for the
current submission plan; the paper is written to be adaptable across a short
list of target journals rather than tied to one.

**Research question:** Large language models are widely reported to lean
left of centre on standardized political instruments. How much of a reported
political score is actually about the model, and how much is about the
measurement apparatus itself (which instrument was used, and ordinary
trial-to-trial noise)?

## Key findings

- **19 models across 11 organizations** (OpenAI, Anthropic, DeepSeek, Google,
  Meta, Mistral, Alibaba, xAI, Cohere, Amazon, NVIDIA) were each administered
  the Political Compass and 8Values tests 60 times per instrument (2,280 total
  administrations), collected programmatically via the OpenRouter API.
- **Instrument and axis choice accounts for 52% of variance in a measured
  political score, and model identity for only 13%**: which questionnaire an
  auditor picks moves a reported score roughly four times more than which
  model is being audited, and trial-to-trial noise alone (35%) exceeds the
  model term.
- **The two instruments fail the standard psychometric test of convergent
  validity**: constructs they both claim to measure correlate at $r=-0.26$
  across models, versus $r=+0.52$ for distinct constructs measured by the
  same instrument (a multi-trait multi-method analysis). Their scores are
  therefore not interchangeable, even in principle.
- The direction of the lean reproduces (every model left-of-centre
  economically and libertarian-of-centre socially on every axis), but the
  paper's claim is about the apparatus, not that the lean doesn't exist: the
  published *magnitudes* are not comparable across studies that used
  different instruments.
- A previously reported range-normalized stability metric is shown to be
  invalid (it assigns one of the worst "instability" scores in the cohort to
  a model whose answers varied by six hundredths of a point) and replaced
  with a variance-components estimate.
- The paper proposes a minimum reporting standard for political-bias audits:
  name the instrument, state the asker condition, report repetition count and
  dispersion, report a variance decomposition or error budget, run a
  contamination check, and report item-level results.

Both instruments are scored against **authoritative sources**, not an
approximate cross-test formula: 8Values via a lossless port of its own
open-source scoring algorithm, and Political Compass via a headless browser
reading the site's own results page. A third instrument, SapplyValues, has a
scoring adapter built to the same standard (`scoring/score_sapplyvalues.py`)
but is not yet part of the collected dataset (see `docs/f3_instrument_terms_of_use.md`).

## What's collected vs. not yet collected

The main 2,280-administration baseline (Political Compass + 8Values, 19
models, 60 trials each) is complete and analyzed. Several further measurement
factors have collection pipelines and tests already built, but no data
collected yet: asker-identity framing, contamination controls (item inversion
and paraphrase), response-format variation, and the IssueBench and
ballot-proposition behavioural-validity arms. `results/w1/w1_results.json`
and `results/w2/w2_results.json` record which of these are and are not
available. `docs/item_variant_review.md` is the required sign-off record for
the contamination-control item banks before they are used to collect data.

## Repository contents

| Path | Description |
|---|---|
| `main.tex` / `main.pdf` | The paper. |
| `supplementary.tex` / `supplementary.pdf` | Supplementary information (material moved out of the main text under length limits). |
| `references.bib` | Bibliography. |
| `lit_review/` | Literature review notes, one file per paper, on competing and related work. |
| `data/raw_trials/` | One JSON file per trial from the main 2,280-administration run: the model's structured answers, resulting score, and exact API cost. |
| `data/scores.csv` | Flattened, analysis-ready table built from `raw_trials/`. |
| `data/collection_summary.md` | Per-model completion rate and cost for the main run. |
| `data/prompt_variants/` | Raw trials from the prompt-robustness check (5 models x 4 paraphrased prompts x 15 trials). |
| `data/openended/` | Raw trials from the open-ended-generation validation arm (5 models x 8 topics x 3 trials, judge-scored). |
| `results/w1/`, `results/w2/` | Re-analysis outputs (IRT results; status of the not-yet-collected validity-battery arms). |
| `results/w4/` | Variance decomposition, MTMM, item-entropy, and ISS-repair outputs behind the main text's display items. |
| `docs/item_variant_review.md` | Sign-off record for the contamination-control (inverted/paraphrased) item banks. |
| `docs/f3_instrument_terms_of_use.md` | Terms-of-use and adapter-priority notes for candidate additional instruments. |
| `scoring/collect_data.py` | Main collection pipeline (keyed-JSON prompting, concurrency, retries, resumable). |
| `scoring/score_8values.py` | Authoritative 8Values scoring (ported line-for-line from the official site's own algorithm). |
| `scoring/score_political_compass.py` | Authoritative Political Compass scoring (Playwright browser automation against the live site). |
| `scoring/score_sapplyvalues.py` | Authoritative SapplyValues scoring (ported line-for-line; adapter built, data not yet collected). |
| `scoring/variance_components.py` | Shared statistical model: crossed random-effects variance decomposition, cluster-bootstrap confidence intervals, MTMM, and related tests. |
| `scoring/analyze_w4.py` | Re-analysis of the existing baseline: item-level stability, ISS repair, variance decomposition with bootstrap CIs, MTMM, Overton envelope. |
| `scoring/irt_analysis.py` | Item-response-theory reanalysis (graded-response model, avoidance model). |
| `scoring/consolidate.py` | Builds `data/scores.csv` and `data/collection_summary.md` from `data/raw_trials/`. |
| `scoring/analyze_expanded.py` | The original omnibus-test battery (ANOVA/Welch/Kruskal-Wallis, Games-Howell post-hoc, human-baseline tests) reported in the Supplementary Information. |
| `scoring/collect_asker_identity.py`, `collect_item_variants.py`, `collect_format_factorial.py`, `collect_issuebench.py`, `collect_ballot.py` | Collection pipelines for measurement factors not yet run (see above). |
| `mech/` | Mechanistic-interpretability exploration (probes, steering); not part of the current paper's scope. |

## Running the analysis

Requires Python 3.9+ and:

```bash
pip install requests numpy scipy statsmodels pandas pingouin playwright
playwright install chromium
```

Regenerate the analysis-ready tables and rerun the full statistical analysis
from already-collected data:

```bash
cd scoring
python3 consolidate.py
python3 analyze_expanded.py   # omnibus tests (Supplementary Information)
python3 analyze_w4.py         # error budget, MTMM, ISS repair, item entropy (main text)
```

Re-running data collection requires an `OPENROUTER_API_KEY` environment
variable (never commit this key) and will incur API costs:

```bash
export OPENROUTER_API_KEY=...
python3 collect_data.py               # main 19-model run, 8values + political_compass (~$9)
python3 collect_prompt_variants.py    # prompt-robustness check (~$1)
python3 collect_openended.py          # open-ended-generation arm (~$0.15)
```

All collection scripts are resumable: each trial is written atomically and
skipped on re-run if it already exists, so an interrupted run can simply be
restarted.

## Limitations

See the paper's Discussion and "What this leaves open" sections for the full
account. In brief: the error budget here quantifies the instrument and trial
terms only; asker-identity accommodation, instrument contamination, and
forced-choice framing have each been demonstrated in the literature but are
not yet estimated in this design; and the paper does not claim that a
questionnaire score predicts real-world model behaviour (a separate,
behavioural-validity question).

## Citation

If you use this code or data, please cite the accompanying paper (see
`main.tex`/`main.pdf` for full reference details).
