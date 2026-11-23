# Replication guide

This repository reproduces every table, figure and in-text number of the article
(`manuscript.tex`) and its web appendix (`main.tex`) from the archived raw trials.

## What can and cannot be re-run

| Step | Re-runnable? | Notes |
|---|---|---|
| Scoring, consolidation, all statistics, all figures | **Yes**, offline | No API key, no network. `./run_all.sh` |
| Data collection (`scoring/collect_*.py`) | **No, not identically** | Hosted models are updated and withdrawn (`mistralai/mistral-large-2512` was removed mid-study), the upstream host behind OpenRouter's default routing was not recorded, and per-call timestamps were not logged. The archived raw trials are the replication object. Re-running collection needs `OPENROUTER_API_KEY`, costs money, and will give different numbers. |
| Political Compass scoring (`score_political_compass.py`) | Only against the live site | Scores in the archive were read off politicalcompass.org's own results page. |

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # pinned versions are in requirements.lock.txt
./run_all.sh            # 1,000 bootstrap resamples; pass a number to change
```

The bootstrap code uses all available cores minus one automatically (override with
`BOOT_WORKERS=n`) and gives identical results to a serial run for a given seed.
`playwright` is needed only for the collection scripts, not for `run_all.sh`.

Software and run time are in `RUN_LOG.md` (written by the clean-environment test).

## Which script produces which display item

| Item in article / appendix | Script | Output |
|---|---|---|
| Fig. 1 error budget (nested shares, CIs; within-axis shares) | `scoring/analyze_nested_design.py`, `scoring/make_figures.py` | `results/nested_design/nested_design.json`, `figures/fig_error_budget.{pdf,tiff}` |
| Table 1 variance shares, original and nested designs; provider bootstrap; leave-one-out | `scoring/analyze_f3_variance_components.py`, `scoring/analyze_nested_design.py` | `results/f3_variance_components/`, `results/nested_design/` |
| Online Appendix per-axis factorial tables (asker, format, contamination) | `scoring/analyze_w1.py` | `results/w1/w1_results.json` |
| Table 2 MTMM with SEs (and Online Appendix MTMM figure) | `scoring/analyze_se_supplement.py`, `scoring/analyze_w4.py` | `results/se_supplement/`, `results/w4/` |
| Spearman checks, SapplyValues and Pew | `scoring/analyze_sapplyvalues.py`, `scoring/analyze_pew_typology.py`, `scoring/analyze_se_supplement.py` | `results/sapplyvalues/`, `results/pew_typology/` |
| Neutral-point tests and effect sizes | `scoring/analyze_se_supplement.py` | `results/se_supplement/se_supplement.json` |
| Known-groups positive control | `scoring/analyze_known_groups.py` | `results/known_groups/known_groups.json` |
| Recovery simulation | `scoring/simulate_recovery.py` | `results/recovery_simulation/` |
| Direction convention and its sensitivity | `scoring/variance_components.py` (`orient`), `scoring/analyze_nested_design.py` | `results/nested_design/nested_design.json` |
| Numbers and tables quoted in the article and appendix | `scoring/make_paper_numbers.py` | `results/paper/` |
| IRT reanalysis (not reported) | `scoring/irt_analysis.py` | `results/w1/irt_results.json` |
| Item entropy, ISS (not reported), appendix tables | `scoring/analyze_w4.py` | `results/w4/` |
| Post-hoc comparisons (not reported) | `scoring/analyze_posthoc.py` | `results/posthoc/` |
| Temperature sensitivity (8Values, 8 models, T = 0 / 0.7 / 1.0) | `scoring/collect_temperature.py` (needs key), `scoring/analyze_temperature_sweep.py` | `data/temperature_sweep/`, `results/temperature_sweep/` |
| Inter-judge agreement, open-ended arm (not reported) | `scoring/validate_judge_agreement.py` (re-judging needs key; cached in `results/judge_agreement/rejudged.json`) | `results/judge_agreement/` |
| Citation audit (institutional affiliation of first/last authors) (not reported) | `scoring/audit_citations.py` (network) | `results/citation_audit/` |
| Prompt-robustness and open-ended arms (not reported) | `scoring/analyze_prompt_robustness.py` | `results/prompt_robustness/` |

## Data

`data/raw_trials/` etc. hold one JSON file per administration (structured answers,
score, cost, status). `data/README.md` describes each directory. Model identifiers and
collection windows are in appendix Table S1.

## Licensing and redistribution

* Code: MIT (`LICENSE`), added 2026-11-10 at the maintainer's discretion; change it if you prefer another.
* 8Values and SapplyValues item banks and scoring code: MIT-licensed upstream, notice retained in
  the adapters.
* **Political Compass:** `scoring/questions_political_compass.json` holds the item text scraped from the site and the archived trials hold the scores it returned; the author has decided no further permission step is needed for this research replication package.
* **Pew Typology:** only item text and the classification Pew's public quiz returned are
  stored; no respondent microdata. See `docs/f3_instrument_terms_of_use.md`.
