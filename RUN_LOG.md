# Clean-environment test log

Run on 2026-11-10, macOS (Darwin 24.2), Apple silicon, 14 cores, system Python 3.9.6, fresh virtualenv
built from `requirements.txt` (exact versions in `requirements.lock.txt`), a copy of the repository
without `.env` and with no `OPENROUTER_API_KEY` in the environment, and no network use by the scripts.

| Step | Result | Time |
|---|---|---|
| `pytest` in `scoring/` | 137 passed | 3 min 17 s |
| `./run_all.sh 40` (40 bootstrap resamples, for speed) | all steps completed (two script bugs found and fixed first: `consolidate.py` and `irt_analysis.py` failed on the SapplyValues records stored in `data/raw_trials/`) | about 31 min to the IRT step, a few minutes after |
| Paper-grade run (1,000 resamples for `analyze_nested_design.py`; 5,000 for `analyze_se_supplement.py`) | `analyze_nested_design.py` took about 2 hours on 12 workers (shared machine) | |

Comparison with the committed results: `data/scores.csv` is byte-identical; the SE supplement,
IRT, post-hoc, SapplyValues and Pew results are byte-identical; the w1, w4, prompt-robustness and w2
JSON files differ only in absolute file paths; point estimates of every variance share agree to
better than 1e-9 (bootstrap intervals depend on the number of resamples and are those from the
1,000-resample run).

Data collection (`collect_*.py`) was not re-run and cannot be re-run identically.

## Added after the clean run (2026-11-10)
`analyze_temperature_sweep.py` and `validate_judge_agreement.py` (offline, from archived data and cached re-judgements) were
added to `run_all.sh` after the clean-environment test; they were run in the development environment, not re-run in the
clean copy. The data for both were collected with `collect_temperature.py` (320 calls, $0.29) and the re-judging
(240 calls, $0.26) on 2026-11-10.

## Final clean run (2026-11-10, after the retarget to Research & Politics)
`./run_all.sh 12` on a copy of the final repository, system Python 3.9.6, fresh virtualenv, no API key: exit 0 in about 13.5 min
(12 bootstrap resamples for speed; the reported intervals are from the 1,000-resample runs). The temperature-sweep, judge-agreement
(from cached re-judgements), SE-supplement and citation-audit steps all ran; point estimates agree with the committed results.

## Direction alignment and added checks (2026-11-14)
The instruments score the left in opposite directions (8Values high, Political Compass negative), and the cross-instrument
estimates had been computed without aligning them. `variance_components.orient` now orients every axis so that higher means
further left, libertarian or progressive before any cross-instrument estimate. Rerun after the change, in a fresh virtualenv
(Python 3.14, packages from `requirements.txt`), offline:

| Step | Setting | Result |
|---|---|---|
| `pytest` in `scoring/` | | 137 passed, 2 skipped (browser tests) |
| `analyze_se_supplement.py` | 5,000 resamples | completed |
| `analyze_w4.py`, `make_w4_figures.py` | 2,000 resamples | completed |
| `analyze_f3_variance_components.py` | 1,000 resamples | completed |
| `analyze_nested_design.py` | 1,000 resamples, 12 workers | 1,000 of 1,000 converged in each of the eight bootstraps |
| `analyze_known_groups.py` (new) | 5,000 resamples | completed |
| `simulate_recovery.py` (new) | 200 runs per condition | completed, about 45 min on a shared machine |
| `make_paper_numbers.py` (new), `make_figures.py` | | completed |

Effect on the headline numbers: transferable share 0.0% -> 17.2% (two instruments) and 0.0% -> 2.0% (four); mean convergent
correlation -0.26 -> +0.26. Within-axis shares, partial eta-squared values, trial-noise shares and item-level results are
unchanged.

## Clean end-to-end rerun (2026-11-15)

`run_all.sh` was run from start to finish in a fresh `git clone` of the repository (Python 3.14, current library versions,
`NSIM=200`, default 1,000 bootstrap resamples) and exited with status 0. The test suite passed (137 passed, 2 skipped).
Every file in `results/paper/` (all quoted numbers and tables) was reproduced identically. The rerun also showed that
`data/collection_summary.md` had not been regenerated since SapplyValues was added, that the temperature-sweep intervals
had been computed with 500 rather than 1,000 resamples, and that two messages in `results/w2/w2_results.json` held an
absolute local path. All three were corrected; the appendix temperature table now carries the 1,000-resample intervals
(point estimates unchanged). `make_result_figures.py` (appendix Figures 3-5) was added to `run_all.sh`. The remaining
differences from the committed files are floating-point noise in the 12th decimal and PDF metadata.
