"""Full statistical analysis of the expanded 12-model dataset (data/scores.csv).

Mirrors extended_stats.py's rigor (assumption checks, robustness tests, effect
sizes) but generalized from 3 groups to 12, across all 6 scored axes (4 from
8Values, 2 from Political Compass).
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import f_oneway, iqr, kruskal, levene, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

DATA_DIR = Path(__file__).parent.parent / "data"

AXES = [
    ("8values", "equality"), ("8values", "peace"), ("8values", "liberty"), ("8values", "progress"),
    ("political_compass", "economic"), ("political_compass", "social"),
]


def load_scores():
    by_axis = defaultdict(lambda: defaultdict(list))  # (test,axis) -> model -> [values]
    with open(DATA_DIR / "scores.csv") as f:
        for row in csv.DictReader(f):
            test = row["test"]
            for i in (1, 2, 3, 4):
                name = row[f"axis{i}_name"]
                val = row[f"axis{i}_value"]
                if name and val:
                    by_axis[(test, name)][row["model"]].append(float(val))
    return by_axis


def relative_sd(vals):
    vals = np.asarray(vals, dtype=float)
    rng = vals.max() - vals.min()
    return 100 * vals.std(ddof=1) / rng if rng > 0 else 0.0


def relative_iqr(vals):
    vals = np.asarray(vals, dtype=float)
    rng = vals.max() - vals.min()
    return 100 * iqr(vals) / rng if rng > 0 else 0.0


def iss(vals, w_rsd=0.7, w_riqr=0.3):
    return w_rsd * relative_sd(vals) + w_riqr * relative_iqr(vals)


def eta_squared(groups):
    all_vals = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return ss_between / ss_total


def main():
    by_axis = load_scores()

    print("=" * 70)
    print("DESCRIPTIVE STATISTICS BY MODEL AND AXIS")
    print("=" * 70)
    for (test, axis), by_model in by_axis.items():
        print(f"\n--- {test} / {axis} ---")
        for model in sorted(by_model):
            vals = np.array(by_model[model])
            print(f"  {model:42s} n={len(vals):3d}  mean={vals.mean():7.2f}  "
                  f"sd={vals.std(ddof=1):5.2f}  median={np.median(vals):7.2f}")

    print("\n" + "=" * 70)
    print("OMNIBUS TESTS (12 models) + ROBUSTNESS + EFFECT SIZE, PER AXIS")
    print("=" * 70)
    sig_summary = []
    for (test, axis), by_model in by_axis.items():
        models = sorted(by_model)
        groups = [by_model[m] for m in models]
        print(f"\n--- {test} / {axis} ---")

        f_stat, p_classic = f_oneway(*groups)
        welch = anova_oneway(groups, use_var="unequal")
        h_stat, p_kw = kruskal(*groups)
        eta2 = eta_squared(groups)
        print(f"  Classic ANOVA:  F={f_stat:.3f}  p={p_classic:.5f}")
        print(f"  Welch's ANOVA:  F={welch.statistic:.3f}  p={welch.pvalue:.5f}")
        print(f"  Kruskal-Wallis: H={h_stat:.3f}  p={p_kw:.5f}")
        print(f"  eta^2 = {eta2:.4f} "
              f"({'small' if eta2 < 0.06 else 'medium' if eta2 < 0.14 else 'large'})")

        normal_violations = sum(1 for g in groups if shapiro(g).pvalue < 0.05)
        _, levene_p = levene(*groups)
        print(f"  Shapiro-Wilk violations: {normal_violations}/{len(groups)} groups non-normal")
        print(f"  Levene's p={levene_p:.5f} ({'variances differ' if levene_p < 0.05 else 'homogeneous'})")

        all_agree = (p_classic < 0.05) == (welch.pvalue < 0.05) == (p_kw < 0.05)
        sig_summary.append((test, axis, p_classic, welch.pvalue, p_kw, eta2, all_agree))

    print("\n" + "=" * 70)
    print("SIGNIFICANCE SUMMARY (all three tests agree?)")
    print("=" * 70)
    for test, axis, pc, pw, pk, eta2, agree in sig_summary:
        flag = "ALL AGREE" if agree else "DISAGREE -- fragile"
        print(f"  {test:20s} {axis:10s} classic={pc:.4f} welch={pw:.4f} kw={pk:.4f} eta2={eta2:.4f}  [{flag}]")

    print("\n" + "=" * 70)
    print("POST-HOC COMPARISONS: TUKEY HSD (classical) vs GAMES-HOWELL (robust)")
    print("=" * 70)
    print("Tukey HSD assumes equal variances -- already shown to be violated here.")
    print("Games-Howell does not assume equal variances or equal n; it is the")
    print("appropriate post-hoc test for this data and is treated as primary.")
    posthoc_agreement = []
    for (test, axis), by_model in by_axis.items():
        models = sorted(by_model)
        all_vals, labels = [], []
        for m in models:
            all_vals.extend(by_model[m])
            labels.extend([m] * len(by_model[m]))
        df = pd.DataFrame({"score": all_vals, "model": labels})

        tukey_res = pairwise_tukeyhsd(np.array(all_vals), np.array(labels), alpha=0.05)
        tukey_sig = {frozenset([row[0], row[1]]) for row in tukey_res.summary().data[1:] if row[-1] == True}

        gh = pg.pairwise_gameshowell(data=df, dv="score", between="model")
        gh_sig = gh[gh["pval"] < 0.05]

        print(f"\n--- {test} / {axis} ---")
        print(f"  Tukey HSD significant pairs:     {len(tukey_sig)} / {len(gh)}")
        print(f"  Games-Howell significant pairs:  {len(gh_sig)} / {len(gh)}")
        only_tukey = tukey_sig - {frozenset([r["A"], r["B"]]) for _, r in gh_sig.iterrows()}
        if only_tukey:
            print(f"  Pairs significant under Tukey but NOT Games-Howell (i.e. an artifact "
                  f"of assuming equal variances): {len(only_tukey)}")
            for pair in only_tukey:
                print(f"    {' vs '.join(pair)}")
        posthoc_agreement.append((test, axis, len(tukey_sig), len(gh_sig), len(only_tukey)))

        for _, row in gh_sig.iterrows():
            print(f"  [Games-Howell] {row['A']:35s} vs {row['B']:35s}  "
                  f"meandiff={row['diff']:+.3f}  p={row['pval']:.4f}  hedges_g={row['hedges']:+.3f}")

    print("\n" + "=" * 70)
    print("ISS CONSISTENCY RANKING (ascending = most consistent first)")
    print("=" * 70)
    iss_rows = []
    for (test, axis), by_model in by_axis.items():
        for model in by_model:
            vals = by_model[model]
            iss_rows.append((f"{model} ({test}/{axis})", iss(vals), relative_sd(vals), relative_iqr(vals)))
    iss_rows.sort(key=lambda r: r[1])
    for label, i, rsd, riqr in iss_rows[:10]:
        print(f"  MOST consistent: {label:55s} ISS={i:6.2f}")
    print("  ...")
    for label, i, rsd, riqr in iss_rows[-10:]:
        print(f"  LEAST consistent: {label:55s} ISS={i:6.2f}")


if __name__ == "__main__":
    main()
