"""The paper's primary statistical model (updates/04_analysis.md), in one place.

updates/04_analysis.md replaces the manuscript's ANOVA / Kruskal-Wallis / Games-Howell battery
with a variance-components model, because the old battery answers "do models
differ?" -- which nobody doubts -- while the new question is "of the variance in
a measured political position, how much is the model and how much is the
measurement?". That ratio is the paper's headline number and the error budget
is its decomposition.

    score_ijklm = mu
                + model_i            (random)
                + asker_j            (fixed, 6 levels)          [W1.1]
                + format_k           (fixed, fractional design) [W1.3]
                + instrument_l       (fixed, 2 levels)
                + item_m             (random, nested in instrument)
                + model_i x asker_j  (random interaction)
                + eps_ijklm

Every workstream's analysis calls into here rather than rolling its own, so the
error budget is computed one way throughout, and the fallbacks and edge cases
are debugged once.

Reported quantities:
  * ICC for model identity -- the fraction of measured political variance that
    is actually about the model
  * variance share per measurement factor -- the error budget itself
  * residual variance -- the defensible replacement for the ISS
  * partial eta-squared for fixed factors, and BH-corrected p-values

Deliberately absent: pairwise Hedges' g. updates/04_analysis.md drops it, and section 4.3 of
the current manuscript already documents why its own g values are inflated.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# variance components
# --------------------------------------------------------------------------
def variance_decomposition(df: pd.DataFrame, value: str, random_effects,
                           fixed_effects=(), z_within=None) -> dict:
    """Crossed random-effects decomposition of `value`.

    Parameters
    ----------
    df : DataFrame with one row per observation.
    value : name of the numeric column being decomposed.
    random_effects : column names entering as crossed random effects. Use a
        "a:b" name to request the interaction of two columns; it is built here.
    fixed_effects : column names entering the mean structure. Their effect is
        removed before the components are estimated, so a variance share is
        always a share of what the fixed part did not explain.
    z_within : optional column to z-score `value` within before fitting. Pass
        the trait/instrument column whenever the data span two instruments with
        different units; without it the larger-scaled instrument dominates every
        component and the decomposition is meaningless.

    Returns a dict of components, shares, the ICC for each random effect, and
    which estimator produced them.
    """
    import statsmodels.formula.api as smf

    d = df.copy()
    d["_y"] = d[value].astype(float)
    if z_within:
        d["_y"] = d.groupby(z_within)["_y"].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
        )

    for name in random_effects:
        if ":" in name:
            parts = name.split(":")
            d[_safe(name)] = d[parts[0]].astype(str)
            for p in parts[1:]:
                d[_safe(name)] = d[_safe(name)] + "|" + d[p].astype(str)
        else:
            d[_safe(name)] = d[name].astype(str)

    d["_grp"] = 1
    vc = {_safe(name): f"0 + C({_safe(name)})" for name in random_effects}
    formula = "_y ~ 1"
    if fixed_effects:
        formula += " + " + " + ".join(f"C({f})" for f in fixed_effects)

    out = {"method": "mixedlm_vc", "n_observations": int(len(d)),
           "fixed_effects": list(fixed_effects), "z_within": z_within}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(formula, d, groups=d["_grp"], vc_formula=vc).fit(reml=True)
        comps = {k: float(v) for k, v in fit.vcomp_and_names()} if hasattr(
            fit, "vcomp_and_names") else dict(zip(list(vc), [float(x) for x in fit.vcomp]))
        comps["residual"] = float(fit.scale)
    except Exception as exc:  # numerical fallback, reported not hidden
        out["method"] = "method_of_moments_fallback"
        out["mixedlm_error"] = str(exc)
        comps = _moments(d, [_safe(n) for n in random_effects])

    # Map the mangled names back to what the caller asked for.
    rename = {_safe(n): n for n in random_effects}
    comps = {rename.get(k, k): v for k, v in comps.items()}
    total = sum(comps.values())
    out["variance_components"] = comps
    out["variance_share"] = {k: (v / total if total else 0.0) for k, v in comps.items()}
    out["total_variance"] = total
    out["icc"] = {k: (v / total if total else 0.0)
                  for k, v in comps.items() if k != "residual"}
    return out


def _safe(name: str) -> str:
    return "vc_" + name.replace(":", "_x_").replace(".", "_")


def _moments(d: pd.DataFrame, factors) -> dict:
    """Method-of-moments decomposition: each factor's between-group variance of
    cell means, with whatever is left over as residual. Crude, and used only
    when the likelihood fails to converge, but it never returns nothing."""
    comps = {}
    explained = pd.Series(0.0, index=d.index)
    for f in factors:
        eff = d.groupby(f)["_y"].transform("mean") - d["_y"].mean()
        comps[f] = float(np.var(eff.groupby(d[f]).first(), ddof=0))
        explained = explained + eff
    comps["residual"] = float(np.var(d["_y"] - explained, ddof=0))
    return comps


# --------------------------------------------------------------------------
# fixed-effect sizes
# --------------------------------------------------------------------------
def partial_eta_squared(df: pd.DataFrame, value: str, factors) -> dict:
    """Partial eta-squared per factor from a Type-II ANOVA.

    Gate A (updates/04_analysis.md) is stated in these units (asker or format eta-squared
    above 0.10 confirms the thesis), so this is the quantity that decides
    whether the project proceeds to Phase 3.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # A factor that is constant in this slice has no effect to estimate, and
    # patsy raises rather than returning a zero. Drop it and say so: a design
    # cell where a factor never varied is a fact about the data, not an error.
    usable = [f for f in factors if df[f].nunique() > 1]
    dropped = [f for f in factors if f not in usable]
    if not usable:
        return {"_r_squared": float("nan"), "_constant_factors": dropped}

    formula = f"{value} ~ " + " + ".join(f"C({f})" for f in usable)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, df).fit()
        table = sm.stats.anova_lm(model, typ=2)
    ss_resid = float(table.loc["Residual", "sum_sq"])
    out = {}
    if dropped:
        out["_constant_factors"] = dropped
    for f in usable:
        row = f"C({f})"
        if row not in table.index:
            continue
        ss = float(table.loc[row, "sum_sq"])
        out[f] = {
            "partial_eta_squared": ss / (ss + ss_resid) if (ss + ss_resid) else float("nan"),
            "F": float(table.loc[row, "F"]),
            "p_value": float(table.loc[row, "PR(>F)"]),
            "df": float(table.loc[row, "df"]),
        }
    out["_r_squared"] = float(model.rsquared)
    return out


# --------------------------------------------------------------------------
# multiple comparisons and intervals
# --------------------------------------------------------------------------
def benjamini_hochberg(p_values, alpha: float = 0.05) -> dict:
    """BH step-up FDR control. The manuscript's existing implementation is
    correct and updates/04_analysis.md says to keep it; this is the same procedure exposed
    for the new analyses so the whole family is corrected together."""
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    if n == 0:
        return {"adjusted": [], "rejected": [], "n_rejected": 0}
    order = np.argsort(p)
    ranked = p[order]
    adjusted_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(adjusted_sorted, 0, 1)
    rejected = adjusted <= alpha
    return {"adjusted": adjusted.tolist(), "rejected": rejected.tolist(),
            "n_rejected": int(rejected.sum()), "alpha": alpha}


def bootstrap_ci(x, y=None, statistic=None, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 20260909) -> dict:
    """Percentile bootstrap CI for a statistic of one or two paired samples.

    Defaults to Pearson r when `y` is given, which is what Gates B and C are
    stated in. n = 19 models is small enough that a point estimate without an
    interval would be actively misleading, and updates/04_analysis.md makes that
    explicit.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if statistic is None:
        if y is None:
            statistic = np.mean
        else:
            def statistic(a, b):
                return float(np.corrcoef(a, b)[0, 1])
    if y is None:
        point = float(statistic(x))
        draws = [float(statistic(rng.choice(x, len(x), replace=True))) for _ in range(n_boot)]
    else:
        y = np.asarray(y, dtype=float)
        if len(x) != len(y):
            raise ValueError(f"paired samples must match: {len(x)} vs {len(y)}")
        point = float(statistic(x, y))
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        draws = [float(statistic(x[i], y[i])) for i in idx]
    draws = np.asarray(draws)
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": point, "ci_lower": float(lo), "ci_upper": float(hi),
            "n_boot": int(len(draws)), "alpha": alpha}


def _cluster_resample(df: pd.DataFrame, cluster: str, clusters, rng) -> pd.DataFrame:
    """One cluster-bootstrap resample: draw cluster IDs with replacement and
    keep every row belonging to a drawn ID, repeated for each time it was
    drawn. Shared by every cluster-bootstrap function below so the resampling
    itself is implemented once."""
    picked = rng.choice(clusters, size=len(clusters), replace=True)
    counts = pd.Series(picked).value_counts()
    parts = [df[df[cluster] == cid] for cid in counts.index for _ in range(counts[cid])]
    return pd.concat(parts, ignore_index=True)


def cluster_bootstrap_variance_shares(df: pd.DataFrame, value: str, random_effects,
                                      cluster: str, fixed_effects=(), z_within=None,
                                      n_boot: int = 2000, alpha: float = 0.05,
                                      seed: int = 20260919) -> dict:
    """95% CIs on every variance share, by resampling whole clusters.

    updates/03_experiments.md Task A / oct_fix.md F4 asks for a cluster
    bootstrap over models (n = 19, so intervals will be wide -- report them
    anyway), not an i.i.d. bootstrap over rows: rows within a model are not
    independent, so resampling rows would understate the interval. Each
    resample draws cluster IDs with replacement, keeps every row belonging to
    a drawn ID (with its repeats, so a cluster picked twice contributes twice),
    and reruns `variance_decomposition` on the result. A resample that fails to
    converge is dropped and counted rather than allowed to crash the run.

    Point estimates come from the original, unresampled fit; the bootstrap
    only supplies the interval around them.
    """
    rng = np.random.default_rng(seed)
    point = variance_decomposition(df, value=value, random_effects=random_effects,
                                   fixed_effects=fixed_effects, z_within=z_within)
    clusters = df[cluster].unique()
    n = len(clusters)

    draws: dict = {k: [] for k in point["variance_share"]}
    n_failed = 0
    for _ in range(n_boot):
        resample = _cluster_resample(df, cluster, clusters, rng)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = variance_decomposition(resample, value=value, random_effects=random_effects,
                                             fixed_effects=fixed_effects, z_within=z_within)
        except Exception:
            n_failed += 1
            continue
        for k, v in fit["variance_share"].items():
            draws.setdefault(k, []).append(v)

    ci = {}
    for k, vals in draws.items():
        arr = np.asarray([v for v in vals if np.isfinite(v)])
        if len(arr) < 10:
            ci[k] = {"ci_lower": float("nan"), "ci_upper": float("nan"), "n_boot": int(len(arr))}
            continue
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci[k] = {"ci_lower": float(lo), "ci_upper": float(hi), "n_boot": int(len(arr))}

    return {
        "point_estimate": point,
        "cluster_column": cluster,
        "n_clusters": int(n),
        "variance_share_ci": ci,
        "n_boot_requested": n_boot,
        "n_boot_failed": n_failed,
        "alpha": alpha,
        "seed": seed,
    }


def cluster_bootstrap_eta_squared(df: pd.DataFrame, value: str, factors,
                                  cluster: str, n_boot: int = 2000, alpha: float = 0.05,
                                  seed: int = 20260920) -> dict:
    """95% CIs on partial eta-squared for each factor, by resampling whole
    clusters (see `_cluster_resample`) and refitting `partial_eta_squared` on
    each resample.

    The asker-identity, contamination, and response-format arms
    (oct_fix.md checklist: "every variance share has a 95% CI") report
    partial eta-squared from a fixed-effects ANOVA rather than the
    random-effects variance shares `cluster_bootstrap_variance_shares`
    handles, so this is a separate function rather than a variant call --
    the two statistics are estimated by different models and are not
    interchangeable inputs to one bootstrap loop.
    """
    rng = np.random.default_rng(seed)
    point = partial_eta_squared(df, value=value, factors=factors)
    clusters = df[cluster].unique()
    n = len(clusters)

    usable = [f for f in factors if isinstance(point.get(f), dict)]
    draws: dict = {f: [] for f in usable}
    n_failed = 0
    for _ in range(n_boot):
        resample = _cluster_resample(df, cluster, clusters, rng)
        try:
            fit = partial_eta_squared(resample, value=value, factors=factors)
        except Exception:
            n_failed += 1
            continue
        for f in usable:
            entry = fit.get(f)
            if isinstance(entry, dict) and "partial_eta_squared" in entry:
                draws[f].append(entry["partial_eta_squared"])

    ci = {}
    for f, vals in draws.items():
        arr = np.asarray([v for v in vals if np.isfinite(v)])
        if len(arr) < 10:
            ci[f] = {"ci_lower": float("nan"), "ci_upper": float("nan"), "n_boot": int(len(arr))}
            continue
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci[f] = {"ci_lower": float(lo), "ci_upper": float(hi), "n_boot": int(len(arr))}

    return {
        "point_estimate": point,
        "cluster_column": cluster,
        "n_clusters": int(n),
        "eta_squared_ci": ci,
        "n_boot_requested": n_boot,
        "n_boot_failed": n_failed,
        "alpha": alpha,
        "seed": seed,
    }


# --------------------------------------------------------------------------
# multi-trait multi-method
# --------------------------------------------------------------------------
def mtmm(df: pd.DataFrame, unit: str, trait: str, method: str, value: str) -> dict:
    """Campbell-Fiske multi-trait multi-method matrix.

    `unit` is what the correlations run across (models, here). Convergent
    coefficients are same-trait/different-method; discriminant are
    different-trait/same-method.
    """
    cell = df.groupby([unit, method, trait])[value].mean().reset_index()
    wide = cell.pivot_table(index=unit, columns=[method, trait], values=value)
    cols = list(wide.columns)
    corr = wide.corr(method="pearson")

    convergent, disc_same_method, hetero = [], [], []
    for a, b in combinations(cols, 2):
        r = float(corr.loc[a, b])
        entry = {"a": f"{a[0]}:{a[1]}", "b": f"{b[0]}:{b[1]}", "r": r}
        if a[1] == b[1] and a[0] != b[0]:
            convergent.append(entry)
        elif a[0] == b[0] and a[1] != b[1]:
            disc_same_method.append(entry)
        else:
            hetero.append(entry)

    mean_conv = float(np.mean([e["r"] for e in convergent])) if convergent else float("nan")
    mean_disc = float(np.mean([e["r"] for e in disc_same_method])) if disc_same_method else float("nan")
    return {
        "n_units": int(wide.shape[0]),
        "convergent": convergent,
        "discriminant_same_method": disc_same_method,
        "heterotrait_heteromethod": hetero,
        "convergent_mean_r": mean_conv,
        "discriminant_mean_r": mean_disc,
        # Campbell-Fiske's first two criteria: convergent coefficients should be
        # significantly non-zero, and should exceed the discriminant ones.
        "campbell_fiske_passes": bool(mean_conv > mean_disc and mean_conv > 0),
    }
