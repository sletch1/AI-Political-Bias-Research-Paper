# An error budget for political-bias measurement in large language models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22968158.svg)](https://doi.org/10.5281/zenodo.22968158)

Code, data, and analysis supporting the paper **"An error budget for
political-bias measurement in large language models"**, submitted as a
Research Article to *Political Analysis*: `pa_article.tex` / `pa_article.pdf`
(the manuscript), with full methods and extended results in the web
appendix, submitted as supplementary material, `pa_appendix.tex` /
`pa_appendix.pdf`. Target venue: see `impr.md` (private, not part of this
repository) for the current submission plan.

This snapshot (release `v0.4.1`) is archived at Zenodo with DOI
[10.5281/zenodo.22968158](https://doi.org/10.5281/zenodo.22968158).

**Research question:** Large language models are widely reported to lean
left of centre on standardized political instruments. How much of a reported
political score is actually about the model, and how much is about the
measurement apparatus itself (which instrument was used, how the question was
asked, whether the instrument is contaminated, how the response was elicited,
and ordinary trial-to-trial noise)?

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
- **Crossing two further instruments** (SapplyValues; Pew Research Center's
  2026 Political Typology, scored by driving its live public quiz) into the
  same variance-components model collapses the pooled model-identity share
  further, to **1.89%**.
- **The two original instruments fail the standard psychometric test of
  convergent validity**: constructs they both claim to measure correlate at
  $r=-0.26$ across models, versus $r=+0.52$ for distinct constructs measured
  by the same instrument (a multi-trait multi-method analysis). Their scores
  are therefore not interchangeable, even in principle.
- **Three further measurement factors, each estimated in its own factorial
  design, are larger than model identity too**: asker identity ($\eta^2=0.79$
  vs. $0.56$–$0.73$ for model identity), instrument contamination (17 of 18
  models shift under item inversion), and response format (39% vs. 23% for
  model identity).
- The direction of the lean reproduces (every model left-of-centre
  economically and libertarian-of-centre socially on every axis), but the
  paper's claim is about the apparatus, not that the lean doesn't exist: the
  published *magnitudes* are not comparable across studies that used
  different instruments, askers, contamination levels, or elicitation formats.
- A previously reported range-normalized stability metric is shown to be
  invalid (it assigns one of the worst "instability" scores in the cohort to
  a model whose answers varied by six hundredths of a point) and replaced
  with a variance-components estimate.
- The paper proposes a minimum reporting standard for political-bias audits:
  name the instrument, state the asker condition, report repetition count and
  dispersion, report a variance decomposition or error budget, run a
  contamination check, and report item-level results.

All instruments are scored against **authoritative sources**, not an
approximate cross-test formula: 8Values and SapplyValues via a lossless port
of each instrument's own open-source scoring algorithm, and Political
Compass and Pew's 2026 Political Typology via a headless browser reading
each site's own results page.

## What's collected vs. not yet collected

Everything reported in the paper is collected and analyzed: the main
2,280-administration baseline, the asker-identity/contamination/format
factorials, the SapplyValues and Pew Political Typology instruments (crossed
into the shared variance-components model), the prompt-robustness and
open-ended-generation validation arms, and the IRT reanalysis. Two further
behavioural-validity checks have collection pipelines built but no data
collected (`data/ballot/README.md`, `results/w2/w2_results.json`): an
IssueBench-based arm and a ballot-proposition/referendum comparison in the
style of Barmettler (2026). These are noted as future work in the paper, not
required for its current claims.

## Repository contents

| Path | Description |
|---|---|
| `pa_article.tex` / `pa_article.pdf` | The manuscript (Political Analysis Research Article submission). |
| `pa_appendix.tex` / `pa_appendix.pdf` | Web appendix / supplementary material: full methods, extended discussion, all supporting results. |
| `pa_cover_letter.md` | Submission cover letter. |
| `references.bib` | Bibliography. |
| `.zenodo.json` | Metadata for the GitHub→Zenodo archival DOI. |
| `lit_review/` | Literature review notes, one file per paper, on competing and related work. |
| `data/raw_trials/` | One JSON file per trial from the main run plus SapplyValues (filenames contain `sapplyvalues`): structured answers, score, exact API cost. |
| `data/asker_identity/`, `data/item_variants/`, `data/format_factorial/` | Raw trials for the asker-identity, contamination, and response-format factorials. |
| `data/pew_typology/` | Raw trials for the Pew 2026 Political Typology instrument. |
| `data/prompt_variants/`, `data/openended/` | Raw trials for the prompt-robustness and open-ended-generation validation arms. |
| `data/scores.csv` | Flattened, analysis-ready table built from `raw_trials/`. |
| `results/w1/`, `results/w2/`, `results/w4/` | Re-analysis outputs: IRT, item entropy, ISS repair, MTMM, and status of the two uncollected behavioural-validity arms. |
| `results/f3_variance_components/` | The four-instrument crossed variance-components model. |
| `results/prompt_robustness/`, `results/posthoc/` | Prompt-robustness/open-ended-generation and Games-Howell/FDR post-hoc results. |
| `docs/item_variant_review.md` | Sign-off record for the contamination-control (inverted/paraphrased) item banks. |
| `docs/f3_instrument_terms_of_use.md` | Terms-of-use analysis and design notes for the third and fourth instruments. |
| `scoring/collect_*.py` | Collection pipelines (main run, asker identity, contamination items, response format, prompt variants, open-ended, Pew quiz). |
| `scoring/score_*.py` | Authoritative scoring adapters (8Values, Political Compass, SapplyValues, Pew Typology). |
| `scoring/variance_components.py` | Shared statistical model: crossed random-effects variance decomposition, cluster-bootstrap CIs, MTMM. |
| `scoring/analyze_f3_variance_components.py` | Crosses all four instruments into the shared variance-components model. |
| `scoring/analyze_w1.py`, `analyze_w2.py`, `analyze_w4.py` | Re-analyses: validity-battery factorials, behavioural-validity status, baseline item/ISS/MTMM. |
| `scoring/analyze_prompt_robustness.py`, `analyze_posthoc.py` | Prompt-robustness/open-ended stats and the Games-Howell/FDR post-hoc family. |
| `scoring/irt_analysis.py` | Item-response-theory reanalysis (graded-response model, avoidance model). |
| `scoring/consolidate.py` | Builds `data/scores.csv` and `data/collection_summary.md` from `data/raw_trials/`. |
| `mech/` | Mechanistic-interpretability exploration (probes, steering); not part of the current paper's scope. |

## Running the analysis

Requires Python 3.9+ and:

```bash
pip install -r requirements.txt
playwright install chromium
```

Regenerate the analysis-ready tables and rerun the full statistical analysis
from already-collected data:

```bash
cd scoring
python3 consolidate.py
python3 analyze_w4.py                       # error budget, MTMM, ISS repair, item entropy
python3 analyze_f3_variance_components.py   # four-instrument crossing
python3 analyze_posthoc.py                  # Games-Howell / FDR post-hoc family
python3 analyze_prompt_robustness.py        # prompt-robustness + open-ended checks
python3 analyze_w1.py                       # asker-identity, contamination, format factorials
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

See the paper's Discussion for the full account. In brief: the design does
not cross all measurement factors (instrument, asker, contamination, format)
simultaneously, only pairwise against the field's standard baseline; the
IssueBench and ballot-proposition behavioural-validity arms have pipelines
built but no data collected; and the paper does not claim that a
questionnaire score predicts real-world model behaviour, a separate,
behavioural-validity question a dual-instrument study (Barmettler 2026)
addresses directly.

## Citation

If you use this code or data, please cite the accompanying paper (see
`pa_article.tex` for full reference details).
