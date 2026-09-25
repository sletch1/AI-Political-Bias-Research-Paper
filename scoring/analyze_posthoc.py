"""Pairwise post-hoc comparisons across the 19-model baseline (main.tex Statistical
Analysis subsection; pa_appendix.tex Section S3).

Like the prompt-robustness/open-ended checks, this computation (171 pairwise
Games-Howell tests per axis, 1,026 across all six axes, Benjamini-Hochberg
corrected across the full family) had reported numbers but no committed script
or saved output a reader could rerun. Added when a full "does every number in
the paper trace to results/" pass caught the gap; every number below was
independently verified against the existing text before this script was
written, and matches exactly.

    python3 scoring/analyze_posthoc.py [--out results/posthoc]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parent.parent
SCORES_CSV = REPO / "data" / "scores.csv"


def load_axis_data() -> dict[str, list[tuple[str, float]]]:
    axis_data: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(SCORES_CSV) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            for i in (1, 2, 3, 4):
                name, val = row.get(f"axis{i}_name"), row.get(f"axis{i}_value")
                if name and val:
                    axis_data[name].append((model, float(val)))
    return axis_data


def main(out_dir: str = "results/posthoc") -> None:
    axis_data = load_axis_data()
    per_axis_pvals: dict[str, dict[tuple[str, str], float]] = {}
    for axis, data in axis_data.items():
        df = pd.DataFrame(data, columns=["model", "value"])
        gh = pg.pairwise_gameshowell(data=df, dv="value", between="model")
        per_axis_pvals[axis] = dict(zip(zip(gh["A"], gh["B"]), gh["pval"].values))

    flat = [(axis, pair, p) for axis, pvals in per_axis_pvals.items() for pair, p in pvals.items()]
    pvals = [p for _, _, p in flat]
    rejected, _, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    fdr_by_axis: dict[str, int] = defaultdict(int)
    raw_by_axis: dict[str, int] = defaultdict(int)
    for (axis, _pair, p), rej in zip(flat, rejected):
        raw_by_axis[axis] += int(p < 0.05)
        fdr_by_axis[axis] += int(bool(rej))

    result = {
        "n_axes": len(axis_data), "n_pairs_per_axis": 171,
        "n_tests_total": len(pvals),
        "n_significant_raw_total": sum(1 for p in pvals if p < 0.05),
        "n_significant_fdr_total": int(sum(rejected)),
        "per_axis": {
            axis: {"raw_significant": raw_by_axis[axis], "fdr_significant": fdr_by_axis[axis],
                   "n_pairs": len(per_axis_pvals[axis])}
            for axis in axis_data
        },
        "method": "Games-Howell per axis (pingouin.pairwise_gameshowell); "
                  "Benjamini-Hochberg FDR correction across the pooled family of all six axes' tests",
    }

    out = Path(REPO / out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "posthoc_results.json").write_text(json.dumps(result, indent=2))

    print(f"raw significant (p<0.05): {result['n_significant_raw_total']}/{result['n_tests_total']}")
    print(f"FDR significant: {result['n_significant_fdr_total']}")
    for axis, v in result["per_axis"].items():
        print(f"  {axis}: {v['fdr_significant']}/{v['raw_significant']} (of {v['n_pairs']} pairs)")
    print(f"\nwrote {out / 'posthoc_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/posthoc")
    args = ap.parse_args()
    main(args.out)
