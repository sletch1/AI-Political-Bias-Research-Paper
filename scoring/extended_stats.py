"""Extended statistical analysis addressing improvement.md #5 and #7.

Re-analyzes the *existing* trial data already embedded in
econ_one_way_anova_test_+_turkey_hsd_test.py and
social_one_way_anova_test_+_turkey_hsd_test.py -- no new data collection.

#7 (statistical toolkit too shallow): adds effect sizes (eta-squared for the
omnibus ANOVA, Cohen's d for each pairwise comparison) and assumption checks
(Shapiro-Wilk normality per group, Levene's test for homogeneity of variance)
that the original analysis omitted.

#5 (ISS weighting is ad hoc): recomputes the Ideological Stability Score
under three weightings (the paper's 0.7/0.3, plus 0.5/0.5 and 0.9/0.1) and
checks whether the model/test consistency ranking is stable.
"""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import f_oneway, iqr, kruskal, levene, shapiro
from statsmodels.stats.oneway import anova_oneway

REPO_ROOT = Path(__file__).parent.parent


def _load_scores(filename):
    src = open(REPO_ROOT / filename).read()
    cut = src.index("# 1) One-Way ANOVA") if "# 1) One-Way ANOVA" in src else src.index("def run_one_way_anova")
    ns = {}
    exec(src[:cut], ns)
    return {"GPT-4": ns["gpt_scores"], "Claude": ns["claude_scores"], "DeepSeek": ns["deepseek_scores"]}


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_sd


def eta_squared(groups):
    all_vals = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return ss_between / ss_total


def assumption_checks(groups_dict):
    print("  Shapiro-Wilk normality (p < 0.05 => reject normality):")
    for name, vals in groups_dict.items():
        stat, p = shapiro(vals)
        flag = "NON-NORMAL" if p < 0.05 else "ok"
        print(f"    {name:10s} n={len(vals):3d}  W={stat:.4f}  p={p:.4f}  [{flag}]")
    stat, p = levene(*groups_dict.values())
    flag = "VARIANCES DIFFER" if p < 0.05 else "ok (homogeneous)"
    print(f"  Levene's test (equal variances): stat={stat:.4f}  p={p:.4f}  [{flag}]")


def robustness_check(groups_dict):
    """Assumption-free alternatives to the paper's one-way ANOVA, run only
    because assumption_checks() found both normality and equal-variance
    violated -- exactly the scenario where a flat ANOVA's p-value is least
    trustworthy."""
    vals = list(groups_dict.values())
    f_stat, p_classic = f_oneway(*vals)
    print(f"  Classic one-way ANOVA (paper's original test): F={f_stat:.4f}  p={p_classic:.4f}")

    welch = anova_oneway(vals, use_var="unequal")
    print(f"  Welch's ANOVA (does not assume equal variances): "
          f"F={welch.statistic:.4f}  p={welch.pvalue:.4f}")

    h_stat, p_kw = kruskal(*vals)
    print(f"  Kruskal-Wallis (does not assume normality): H={h_stat:.4f}  p={p_kw:.4f}")

    agree = (p_classic < 0.05) == (welch.pvalue < 0.05) == (p_kw < 0.05)
    print(f"  All three agree on significance at alpha=0.05: {agree}")


def effect_sizes(groups_dict):
    names = list(groups_dict.keys())
    vals = list(groups_dict.values())
    eta2 = eta_squared(vals)
    print(f"  Omnibus effect size: eta^2 = {eta2:.4f} "
          f"({'small' if eta2 < 0.06 else 'medium' if eta2 < 0.14 else 'large'})")
    print("  Pairwise Cohen's d:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = cohens_d(vals[i], vals[j])
            mag = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
            print(f"    {names[i]:10s} vs {names[j]:10s}: d = {d:+.3f}  ({mag})")


def relative_sd(vals):
    vals = np.asarray(vals, dtype=float)
    return 100 * vals.std(ddof=1) / (vals.max() - vals.min())


def relative_iqr(vals):
    vals = np.asarray(vals, dtype=float)
    return 100 * iqr(vals) / (vals.max() - vals.min())


def iss(vals, w_rsd=0.7, w_riqr=0.3):
    return w_rsd * relative_sd(vals) + w_riqr * relative_iqr(vals)


def iss_sensitivity(all_groups):
    """all_groups: dict of label -> values, one entry per (model,test,axis) cell."""
    weightings = [(0.7, 0.3), (0.5, 0.5), (0.9, 0.1)]
    rows = {}
    for label, vals in all_groups.items():
        rows[label] = [iss(vals, w_rsd, w_riqr) for w_rsd, w_riqr in weightings]

    print(f"  {'label':30s}" + "".join(f"  ISS(0.7/0.3)  ISS(0.5/0.5)  ISS(0.9/0.1)" for _ in [0])[:0])
    header = f"  {'':30s}" + "".join(f"{'w='+str(w):>16s}" for w in weightings)
    print(header)
    for label, iss_vals in rows.items():
        print(f"  {label:30s}" + "".join(f"{v:16.2f}" for v in iss_vals))

    ranks = {w: sorted(rows.keys(), key=lambda k: rows[k][i]) for i, w in enumerate(weightings)}
    base_rank = ranks[weightings[0]]
    print("\n  Ranking (least to most inconsistent) under paper's weighting (0.7/0.3):")
    print("   ", " < ".join(base_rank))
    stable = all(ranks[w] == base_rank for w in weightings)
    print(f"\n  Ranking IDENTICAL across all three weightings: {stable}")
    if not stable:
        for w in weightings[1:]:
            if ranks[w] != base_rank:
                print(f"    Under {w}: ", " < ".join(ranks[w]))
    return stable


if __name__ == "__main__":
    econ = _load_scores("econ_one_way_anova_test_+_turkey_hsd_test.py")
    social = _load_scores("social_one_way_anova_test_+_turkey_hsd_test.py")

    for label, groups in [("ECONOMIC", econ), ("SOCIAL", social)]:
        print(f"\n=== {label} SCORES ===")
        print(" Assumption checks (improvement.md #7):")
        assumption_checks(groups)
        print(" Robustness check (improvement.md #7):")
        robustness_check(groups)
        print(" Effect sizes (improvement.md #7):")
        effect_sizes(groups)

    print("\n=== ISS WEIGHTING SENSITIVITY (improvement.md #5) ===")
    all_groups = {}
    for axis_name, groups in [("Econ", econ), ("Social", social)]:
        for model, vals in groups.items():
            all_groups[f"{model} {axis_name}"] = vals
    iss_sensitivity(all_groups)
