#!/usr/bin/env bash
# Reproduce every reported number and figure from the archived raw trials.
# No API key and no network access are needed: model calls are NOT re-run (see REPLICATION.md).
# Usage: ./run_all.sh [N_BOOT]   (default 1000 bootstrap resamples; the paper's intervals use 1000-2000)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python3}
NBOOT=${1:-1000}
cd scoring
$PY consolidate.py                                   # data/scores.csv, data/collection_summary.md
$PY analyze_w4.py                                    # two-instrument budget, MTMM, ISS repair, item entropy
$PY analyze_f3_variance_components.py --n-boot "$NBOOT"   # four-instrument budget (original design)
$PY analyze_nested_design.py --n-boot "$NBOOT"       # nested model x instrument x axis design; provider bootstrap; leave-one-out
$PY analyze_se_supplement.py --n-boot 5000           # SEs/CIs for every r and rho; neutral-point check
$PY analyze_sapplyvalues.py
$PY analyze_pew_typology.py
$PY analyze_w1.py                                    # asker, contamination, format factorials
$PY irt_analysis.py                                  # graded-response IRT reanalysis
$PY analyze_w2.py                                    # status of the uncollected behavioural arms
$PY analyze_posthoc.py
$PY analyze_prompt_robustness.py
$PY analyze_temperature_sweep.py --n-boot "$NBOOT"      # temperature sensitivity (archived sweep data)
$PY validate_judge_agreement.py                      # inter-judge agreement from cached re-judgements (no API)
$PY audit_citations.py || echo "citation audit skipped (needs network)"
$PY make_w4_figures.py                               # appendix figures and LaTeX tables in results/w4
$PY make_figures.py                                  # vector PDF + TIFF figures
