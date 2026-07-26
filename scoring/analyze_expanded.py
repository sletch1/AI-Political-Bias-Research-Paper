"""Full statistical analysis of the 19-model dataset (data/scores.csv).

This is the single script that produces every statistic reported in the
paper's Results section. Run after consolidate.py has built data/scores.csv:

    python3 analyze_expanded.py

For each of the 6 scored axes (equality, peace, liberty, progress from
8Values; economic, social from Political Compass), across all 19 models, it
prints, in order:

  1. Descriptive statistics (n, mean, SD, median) per model.
  2. Omnibus tests: classical one-way ANOVA, Welch's ANOVA, Kruskal-Wallis,
     plus eta-squared effect size and Shapiro-Wilk/Levene assumption checks.
  3. A significance-agreement summary across the three omnibus tests.
  4. Post-hoc pairwise comparisons: Tukey HSD vs. Games-Howell (171 pairs per
     axis), including a Benjamini-Hochberg FDR correction applied across the
     full 1,026-test family (all 6 axes together), not axis-by-axis.
  5. A human-baseline check: one-sample t-tests of each model's mean score
     against that instrument's neutral center-point (0 for Political
     Compass, 50 for 8Values), FDR-corrected across all 114 model-axis tests.
  6. An Ideological Stability Score (ISS) consistency ranking across all 114
     model-axis combinations (most and least consistent 10).

Output is printed to stdout for inspection; the numbers reported in main.tex
were transcribed from a run of this script against the released dataset.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import f_oneway, iqr, kruskal, levene, shapiro, ttest_1samp, wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

DATA_DIR = Path(__file__).parent.parent / "data"

AXES = [
    ("8values", "equality"), ("8values", "peace"), ("8values", "liberty"), ("8values", "progress"),
    ("political_compass", "economic"), ("political_compass", "social"),
]


def load_scores():
    """Load data/scores.csv into {(test, axis_name): {model: [values]}},
    e.g. by_axis[("political_compass", "economic")]["openai/gpt-4o"] is the
    list of that model's economic scores across all its trials. Each row of
    scores.csv carries up to 4 axes (8values rows use all 4; political_compass
    rows use only axis1/axis2 and leave axis3/axis4 blank), so this reads
    whichever axis1..axis4_name/value columns are non-empty."""
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
    """Standard deviation of `vals`, expressed as a percentage of that same
    sample's own range (max - min). 0 if every value is identical (range 0),
    rather than dividing by zero."""
    vals = np.asarray(vals, dtype=float)
    rng = vals.max() - vals.min()
    return 100 * vals.std(ddof=1) / rng if rng > 0 else 0.0


def relative_iqr(vals):
    """Interquartile range of `vals`, expressed as a percentage of that same
    sample's own range. 0 if the range is 0."""
    vals = np.asarray(vals, dtype=float)
    rng = vals.max() - vals.min()
    return 100 * iqr(vals) / rng if rng > 0 else 0.0


def iss(vals, w_rsd=0.7, w_riqr=0.3):
    """Ideological Stability Score: a weighted average of relative_sd and
    relative_iqr for one model's trials on one axis. Lower = more consistent.
    Because both inputs are normalized by the sample's own range rather than
    the instrument's theoretical scale, a model whose answers are clustered
    into a very narrow absolute range can still register a high ISS (see the
    range-normalization caveat in main.tex's Statistical Analysis section)."""
    return w_rsd * relative_sd(vals) + w_riqr * relative_iqr(vals)


def eta_squared(groups):
    """Omnibus eta-squared effect size (fraction of total variance explained
    by group membership) for a one-way design, given `groups` as a list of
    per-group value lists. Conventional benchmarks: ~0.01 small, ~0.06
    medium, ~0.14 large."""
    all_vals = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return ss_between / ss_total


def main():
    """Entry point: runs every analysis described in the module docstring,
    in order, printing results to stdout. See the module docstring for the
    full list of what's computed and in what order."""
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
    print("OMNIBUS TESTS (19 models) + ROBUSTNESS + EFFECT SIZE, PER AXIS")
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
    gh_by_axis = {}
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
        gh_by_axis[(test, axis)] = gh

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
    print("BENJAMINI-HOCHBERG FDR CORRECTION ACROSS THE FULL FAMILY OF")
    print("GAMES-HOWELL COMPARISONS (all 6 axes x 171 pairs = 1,026 tests)")
    print("=" * 70)
    all_axes_order = list(gh_by_axis.keys())
    all_pvals = []
    axis_boundaries = []
    for key in all_axes_order:
        gh = gh_by_axis[key]
        axis_boundaries.append((key, len(all_pvals), len(all_pvals) + len(gh)))
        all_pvals.extend(gh["pval"].tolist())
    all_pvals = np.array(all_pvals)
    reject, pvals_corrected, _, _ = multipletests(all_pvals, alpha=0.05, method="fdr_bh")
    print(f"  Total pairwise tests in family: {len(all_pvals)}")
    print(f"  Significant at raw p<0.05:      {int((all_pvals < 0.05).sum())}")
    print(f"  Significant after BH-FDR (q<0.05): {int(reject.sum())}")
    for key, start, end in axis_boundaries:
        test, axis = key
        raw_sig = int((all_pvals[start:end] < 0.05).sum())
        fdr_sig = int(reject[start:end].sum())
        print(f"  {test:20s} {axis:10s} raw-sig={raw_sig:3d}/171  BH-FDR-sig={fdr_sig:3d}/171")

    print("\n" + "=" * 70)
    print("HUMAN-BASELINE COMPARISON: ONE-SAMPLE TESTS AGAINST EACH")
    print("INSTRUMENT'S OWN NEUTRAL CENTER-POINT (0 for Political Compass,")
    print("50 for 8Values), as a bounded proxy for a human reference point")
    print("(Gallup 2024 Values and Beliefs poll finds the US public within a")
    print("few points of an implied neutral center on both economic and")
    print("social self-identification -- see main.tex for the exact figures")
    print("and the caveats on treating 0/50 as a human-population proxy)")
    print("=" * 70)
    NEUTRAL = {"equality": 50.0, "peace": 50.0, "liberty": 50.0, "progress": 50.0,
               "economic": 0.0, "social": 0.0}
    baseline_pvals = []
    baseline_labels = []
    n_reject_raw = 0
    for (test, axis), by_model in sorted(by_axis.items()):
        center = NEUTRAL[axis]
        for model in sorted(by_model):
            vals = np.asarray(by_model[model], dtype=float)
            if vals.std(ddof=1) == 0:
                # constant response: trivially "different from center" if the
                # constant itself isn't the center, undefined t-test otherwise
                p_t = 0.0 if vals[0] != center else 1.0
            else:
                _, p_t = ttest_1samp(vals, center)
            baseline_pvals.append(p_t)
            baseline_labels.append(f"{model} ({test}/{axis})")
            if p_t < 0.05:
                n_reject_raw += 1
    baseline_pvals = np.array(baseline_pvals)
    reject_b, _, _, _ = multipletests(baseline_pvals, alpha=0.05, method="fdr_bh")
    print(f"  Total one-sample tests (19 models x 6 axes): {len(baseline_pvals)}")
    print(f"  Significantly different from neutral center at raw p<0.05: {n_reject_raw}")
    print(f"  Significantly different after BH-FDR correction:            {int(reject_b.sum())}")
    not_sig = [lbl for lbl, r in zip(baseline_labels, reject_b) if not r]
    if not_sig:
        print(f"  NOT significantly different from center after correction ({len(not_sig)}):")
        for lbl in not_sig:
            print(f"    {lbl}")
    else:
        print("  Every one of the 114 model-axis combinations differs significantly "
              "from the instrument's neutral center-point after correction.")

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
