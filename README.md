# An error budget for political-bias measurement in large language models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22968157.svg)](https://doi.org/10.5281/zenodo.22968157)

Code, data, and analysis supporting the paper **"An error budget for
political-bias measurement in large language models"**, prepared as a Research
Article for *Research & Politics* (Sage; double-anonymized): `manuscript.tex` /
`manuscript.pdf` (anonymized article, 4,000-word limit), `title_page.tex`
(authors and declarations, not sent to reviewers), and the online appendix
`main.tex` / `main.pdf`.

The release `v0.5.0` snapshot, which matches the article, is archived at Zenodo. The concept DOI
[10.5281/zenodo.22968157](https://doi.org/10.5281/zenodo.22968157) always resolves to the latest version.

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
- **Scores are direction-aligned before any cross-instrument estimate.** The
  instruments score the left in opposite directions (high on 8Values, negative
  on the Political Compass), so every axis is oriented so that higher means
  further left, libertarian or progressive (`variance_components.orient`). An
  earlier draft estimated the model without this alignment and reported a
  transferable share of 0.0% and a convergent correlation of -0.26; both were
  artefacts of the opposite scoring directions and are superseded.
- **A nested variance-components model** (model, model x instrument,
  model x axis within instrument, trial noise) on the 2,280 administrations of
  the two original instruments gives: model identity that transfers across
  instruments and axes **17.2% (95% CI 0.0-40.0%)**; model position specific to
  the instrument **27.7% (11.5-41.3%)**; specific to the axis within the
  instrument **21.1% (11.4-34.0%)**; trial noise **34.0% (20.0-50.7%)**. The
  two-term specification (instrument folded into trait) gives 30.2% to model
  identity and is kept for comparison. Intervals are similar when resampling
  the 11 providers instead of the 19 models.
- **Crossing two further instruments** (SapplyValues; Pew Research Center's
  2026 Political Typology, scored by driving its live public quiz) puts the
  transferable share at **2.0% (0.0-18.4%)**, with model x instrument 31.8%,
  model x axis 31.3% and noise 34.9%. Across leave-one-instrument-out fits the
  transferable share is 0.0-8.6% and across leave-one-provider-out fits
  0.0-7.5%.
- **Convergence between the two original instruments is partial**: the social
  constructs correlate at r = +0.51 across models and the economic constructs
  at r = +0.005; the mean convergent correlation is +0.26 (SE 0.17, 95% CI
  [-0.07, 0.57]) against +0.52 (SE 0.11) for distinct constructs on one
  instrument (difference -0.26, SE 0.14). SapplyValues' economic axis agrees
  with 8Values equality (rho = +0.53) and disagrees with the Political Compass
  economic axis (rho = -0.49).
- **Positive control**: every model answers further left for a self-described
  progressive than for a conservative on every axis of both instruments (108
  of 108 cells), and the size of that shift correlates at r = 0.86 (95% CI
  [0.63, 0.94]) across the two instruments, against 0.06 for no-asker
  positions in the same design. The instruments agree on a large common
  signal; what transfers poorly is the smaller difference between models'
  default positions.
- **Recovery simulation** on the real design (`scoring/simulate_recovery.py`)
  shows what transferable share the nested estimator can detect; see the
  Online Appendix.
- **Three further measurement factors, each estimated in its own factorial
  design**: asker identity (partial $\eta^2=0.79$ vs. $0.23$ for model identity
  in the same design), instrument contamination (17 of 18 models shift under
  item inversion), and response format (elicitation $0.39$ vs. $0.23$ for model
  identity on average, though not on every axis).
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
| `manuscript.tex` / `manuscript.pdf` | The anonymized article (Research & Politics Research Article). |
| `title_page.tex` | Title page, acknowledgments and Statements and Declarations. |
| `main.tex` / `main.pdf` | The web appendix (standalone: author block on page 1; full methods, extended discussion, all supporting results). |
| `cover_letter_research_and_politics.md` | Submission cover letter (generic AI statement). |
| `references.bib` | Bibliography. |
| `.zenodo.json` | Metadata for the GitHub→Zenodo archival DOI. |
| `data/raw_trials/` | One JSON file per trial from the main run plus SapplyValues (filenames contain `sapplyvalues`): structured answers, score, exact API cost. |
| `data/asker_identity/`, `data/item_variants/`, `data/format_factorial/` | Raw trials for the asker-identity, contamination, and response-format factorials. |
| `data/pew_typology/` | Raw trials for the Pew 2026 Political Typology instrument. |
| `data/prompt_variants/`, `data/openended/` | Raw trials for the prompt-robustness and open-ended-generation validation arms. |
| `data/scores.csv` | Flattened, analysis-ready table built from `raw_trials/`. |
| `results/w1/`, `results/w2/`, `results/w4/` | Re-analysis outputs: IRT, item entropy, ISS repair, MTMM, and status of the two uncollected behavioural-validity arms. |
| `results/f3_variance_components/`, `results/nested_design/`, `results/se_supplement/` | Two-term four-instrument model; nested model with provider bootstrap and leave-one-out; SEs for every correlation. |
| `results/prompt_robustness/`, `results/posthoc/` | Prompt-robustness/open-ended-generation and Games-Howell/FDR post-hoc results. |
| `docs/item_variant_review.md` | Sign-off record for the contamination-control (inverted/paraphrased) item banks. |
| `docs/f3_instrument_terms_of_use.md` | Sourcing and design notes for the third and fourth instruments. |
| `scoring/collect_*.py` | Collection pipelines (main run, asker identity, contamination items, response format, prompt variants, open-ended, Pew quiz). |
| `scoring/score_*.py` | Authoritative scoring adapters (8Values, Political Compass, SapplyValues, Pew Typology). |
| `scoring/variance_components.py` | Shared statistical model: crossed random-effects variance decomposition, cluster-bootstrap CIs, MTMM. |
| `scoring/analyze_nested_design.py` | Nested model x instrument x axis variance components, provider-clustered bootstrap, leave-one-instrument/provider-out. |
| `scoring/analyze_se_supplement.py` | Standard errors and intervals for every correlation, and the neutral-point tests. |
| `scoring/analyze_known_groups.py` | Known-groups positive control: progressive-versus-conservative asker shift and its agreement across instruments. |
| `scoring/simulate_recovery.py` | Recovery of a known transferable share by the nested estimator on the real design. |
| `scoring/make_paper_numbers.py` | Writes every quoted number and table (`results/paper/`) from the result files; both TeX files `\input` them. |
| `scoring/make_figures.py` | Submission figures (vector PDF and 300 dpi TIFF). |
| `run_all.sh`, `REPLICATION.md` | One-command offline reproduction and replication guide. |
| `scoring/analyze_f3_variance_components.py` | Crosses all four instruments into the shared variance-components model. |
| `scoring/analyze_w1.py`, `analyze_w2.py`, `analyze_w4.py` | Re-analyses: validity-battery factorials, behavioural-validity status, baseline item/ISS/MTMM. |
| `scoring/analyze_prompt_robustness.py`, `analyze_posthoc.py` | Prompt-robustness/open-ended stats and the Games-Howell/FDR post-hoc family. |
| `scoring/irt_analysis.py` | Item-response-theory reanalysis (graded-response model, avoidance model). |
| `scoring/consolidate.py` | Builds `data/scores.csv` and `data/collection_summary.md` from `data/raw_trials/`. |

## Running the analysis

The one-command route is `./run_all.sh` (offline, no API key); see
[REPLICATION.md](REPLICATION.md) for what can and cannot be re-run, the
display-item-to-script map, and licensing notes. The archived snapshot above
predates the nested-design analysis; a new release is needed before the
submission is final.

Individual steps:

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
`manuscript.tex` for full reference details).
