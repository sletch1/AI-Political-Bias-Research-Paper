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
