"""Standard errors and intervals for every correlation and effect size quoted in the text.

Every reported correlation coefficient needs a standard error, and every number needs its
uncertainty. The earlier scripts gave SEs for the four individual MTMM Pearson
correlations only; this script adds, all by bootstrap over models (the unit of analysis, n = 19
or 18):

  * the mean convergent r and mean discriminant r, each with SE and 95% CI, and their difference,
  * a bootstrap SE for each Spearman rho in the four-instrument convergence checks,
  * the mean within-8Values correlation with SE and CI,
  * the neutral-point check: per model-axis one-sample t-tests (BH-FDR), Cohen's d, and a
    model-clustered bootstrap CI for the median |d|.

    python3 scoring/analyze_se_supplement.py [--n-boot 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_f3_variance_components import load_pooled_long  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SEED = 20260919

PC_E, PC_S = "political_compass:economic", "political_compass:social"
EQ, PR = "8values:equality", "8values:progress"
CONVERGENT = [(EQ, PC_E), (PR, PC_S)]
DISCRIMINANT = [(EQ, PR), (PC_E, PC_S)]
SPEARMAN_PAIRS = [
    ("sapplyvalues:right", PC_E), ("sapplyvalues:right", EQ),
    ("sapplyvalues:auth", PC_S), ("sapplyvalues:auth", "8values:liberty"),
    ("sapplyvalues:prog", PR),
    ("pew_typology:position", PC_E), ("pew_typology:position", EQ),
]
NEUTRAL = {"8values:equality": 50, "8values:peace": 50, "8values:liberty": 50,
           "8values:progress": 50, "political_compass:economic": 0, "political_compass:social": 0}


def pearson_se(r: float, n: int) -> float:
    return float(np.sqrt((1 - r ** 2) / (n - 2)))


def boot_stat(means: pd.DataFrame, fn, n_boot: int, rng) -> np.ndarray:
    n = len(means)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(means.iloc[idx])
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def summarize(point: float, draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(point), "se": float(draws.std(ddof=1)),
            "ci_lower": float(lo), "ci_upper": float(hi), "n_boot_ok": int(len(draws))}


def mean_r(m: pd.DataFrame, pairs) -> float:
    return float(np.mean([np.corrcoef(m[a], m[b])[0, 1] for a, b in pairs]))


def main(n_boot: int, out_dir: str = "results/se_supplement") -> None:
    rng = np.random.default_rng(SEED)
    df = load_pooled_long()
    means = df.groupby(["model", "trait"])["value"].mean().unstack("trait")
    two = means.dropna(subset=[PC_E, PC_S, EQ, PR])
    n2 = len(two)
    res: dict = {"seed": SEED, "n_boot": n_boot}

    # --- MTMM -----------------------------------------------------------------------------
    pairs = []
    for kind, plist in (("convergent", CONVERGENT), ("discriminant", DISCRIMINANT)):
        for a, b in plist:
            r = float(np.corrcoef(two[a], two[b])[0, 1])
            z, se_z = np.arctanh(r), 1 / np.sqrt(n2 - 3)
            pairs.append({"type": kind, "a": a, "b": b, "r": r, "se_r": pearson_se(r, n2),
                          "ci_fisher": [float(np.tanh(z - 1.96 * se_z)), float(np.tanh(z + 1.96 * se_z))]})
    res["mtmm_pairs"] = {"n_models": n2, "pairs": pairs}
    conv = boot_stat(two, lambda m: mean_r(m, CONVERGENT), n_boot, rng)
    disc = boot_stat(two, lambda m: mean_r(m, DISCRIMINANT), n_boot, rng)
    diffs = boot_stat(two, lambda m: mean_r(m, CONVERGENT) - mean_r(m, DISCRIMINANT), n_boot, rng)
    res["mtmm_means"] = {
        "mean_convergent": summarize(mean_r(two, CONVERGENT), conv),
        "mean_discriminant": summarize(mean_r(two, DISCRIMINANT), disc),
        "convergent_minus_discriminant": summarize(mean_r(two, CONVERGENT) - mean_r(two, DISCRIMINANT), diffs),
        "share_of_resamples_convergent_below_discriminant": float(np.mean(diffs < 0)),
        "note": ("bootstrap over models; the two convergent coefficients share the same 19 "
                 "models so the SE of their mean is not the mean of their SEs"),
    }

    # --- 8Values internal structure -------------------------------------------------------
    ev = [c for c in means.columns if c.startswith("8values:")]
    ev_pairs = list(combinations(ev, 2))
    ev_m = means.dropna(subset=ev)
    res["eightvalues_internal"] = {
        "n_models": len(ev_m),
        "mean_r": summarize(mean_r(ev_m, ev_pairs),
                            boot_stat(ev_m, lambda m: mean_r(m, ev_pairs), n_boot, rng)),
        "pairs": {f"{a} vs {b}": float(np.corrcoef(ev_m[a], ev_m[b])[0, 1]) for a, b in ev_pairs},
    }

    # --- Spearman rho SEs -----------------------------------------------------------------
    sp = []
    for a, b in SPEARMAN_PAIRS:
        sub = means[[a, b]].dropna()
        rho = float(stats.spearmanr(sub[a], sub[b])[0])
        draws = boot_stat(sub, lambda m: stats.spearmanr(m[a], m[b])[0], n_boot, rng)
        sp.append({"a": a, "b": b, "n_models": len(sub), **summarize(rho, draws)})
    res["spearman"] = sp

    # --- neutral-point check --------------------------------------------------------------
    rows = []
    for (model, trait), g in df[df.trait.isin(NEUTRAL)].groupby(["model", "trait"]):
        v = g["value"].to_numpy(float)
        sd = v.std(ddof=1)
        if sd < 1e-6:
            p = 0.0 if abs(v[0] - NEUTRAL[trait]) > 1e-6 else 1.0
            d = np.nan
        else:
            p = stats.ttest_1samp(v, NEUTRAL[trait])[1]
            d = (v.mean() - NEUTRAL[trait]) / sd
        rows.append({"model": model, "trait": trait, "p": p, "d": d, "n": len(v)})
    cells = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    rej = multipletests(cells.p, alpha=0.05, method="fdr_bh")[0]
    absd = cells.d.abs()
    models = cells.model.unique()
    meds = []
    for _ in range(n_boot):
        pick = rng.choice(models, len(models), replace=True)
        vals = np.concatenate([absd[cells.model == m].dropna().to_numpy() for m in pick])
        meds.append(np.median(vals))
    res["neutral_point_check"] = {
        "n_cells": int(len(cells)), "n_rejected_bh_fdr_05": int(rej.sum()),
        "n_cells_with_finite_d": int(absd.notna().sum()),
        "min_abs_d": float(absd.min()), "max_abs_d": float(absd.max()),
        "median_abs_d": summarize(float(absd.median()), np.array(meds)),
        "test": "one-sample t-test per model x axis cell against the instrument midpoint, "
                "BH-FDR across all cells; the midpoint is a convention, not a population estimate",
    }

    out = REPO / out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "se_supplement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("mtmm_means", "neutral_point_check")}, indent=1))
    for s in sp:
        print(f"{s['a']:26s} {s['b']:28s} rho={s['estimate']:+.3f} SE={s['se']:.3f} [{s['ci_lower']:+.2f},{s['ci_upper']:+.2f}]")
    print("8V mean r", res["eightvalues_internal"]["mean_r"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    main(a.n_boot)
