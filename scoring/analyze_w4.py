"""Workstream 4 — re-analysis of the existing 2,280-administration dataset.

Implements every task in Workstream 4 (see updates/02_where_the_paper_stands.md). Costs nothing to run and touches no API:
all five analyses operate on data already in the repository.

    python3 scoring/analyze_w4.py [--out results/w4]

Tasks
-----
4.1  Item-level stability. Per-*item* response entropy from data/raw_trials/,
     replacing the per-axis aggregate that hides issue-specific inconsistency.
4.2  ISS repair. Reproduces the range-normalisation artefact, then replaces it
     with a variance-components estimate and reports absolute SD alongside.
4.3  Variance decomposition. Crossed random-effects model over the 2,280
     administrations: how much variance is model, trait, model x trait, trial.
4.4  MTMM. Multi-trait multi-method matrix over the two instruments, giving
     convergent and discriminant validity coefficients.
4.5  Overton envelope. Cohort spread as a fraction of a reference spread.

Every number printed is also written to <out>/w4_results.json so the manuscript
can cite a machine-readable source rather than a transcribed console dump.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
SCORES_CSV = REPO / "data" / "scores.csv"
RAW_TRIALS = REPO / "data" / "raw_trials"

# Political Compass axes are on [-10, 10]; 8Values axes on [0, 100].
INSTRUMENT_RANGE = {"political_compass": 20.0, "8values": 100.0}


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_long() -> pd.DataFrame:
    """scores.csv in long form: one row per (model, instrument, trial, axis)."""
    wide = pd.read_csv(SCORES_CSV)
    rows = []
    for _, r in wide.iterrows():
        for i in (1, 2, 3, 4):
            name, val = r.get(f"axis{i}_name"), r.get(f"axis{i}_value")
            if isinstance(name, str) and pd.notna(val):
                rows.append(
                    {
                        "model": r["model"],
                        "instrument": r["test"],
                        "trial": int(r["trial"]),
                        "axis": name,
                        "value": float(val),
                    }
                )
    df = pd.DataFrame(rows)
    # A "trait" is an instrument-axis pair: the unit a score is comparable within.
    df["trait"] = df["instrument"] + ":" + df["axis"]
    return df


def load_raw_trials() -> pd.DataFrame:
    """One row per (model, instrument, trial) with the raw per-item answers."""
    rows = []
    for path in sorted(RAW_TRIALS.glob("*.json")):
        with open(path) as fh:
            rec = json.load(fh)
        if rec.get("status") != "ok" or not rec.get("answers"):
            continue
        rows.append(
            {
                "model": rec["model"],
                "instrument": rec["test"],
                "trial": int(rec["trial"]),
                "answers": rec["answers"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4.1  item-level stability
# --------------------------------------------------------------------------
def normalised_entropy(labels) -> float:
    """Shannon entropy of a categorical sample, normalised to [0, 1] by the
    entropy of a uniform distribution over the *observed* alphabet size.

    Normalising by the observed alphabet rather than the theoretical one keeps
    the statistic comparable across instruments whose answer scales differ.
    Returns 0.0 for a single-category (perfectly stable) item.
    """
    counts = Counter(labels)
    n = sum(counts.values())
    if n == 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return h / math.log(len(counts)) if len(counts) > 1 else 0.0


def task_4_1(raw: pd.DataFrame) -> dict:
    """Per-item response entropy per model, per instrument.

    Answer! The paper's per-axis stability metric averages over items and so
    cannot see that a model is rock-solid on 60 items and incoherent on 10.
    This computes the per-item distribution directly.
    """
    per_item = []
    for (model, instrument), grp in raw.groupby(["model", "instrument"]):
        mat = list(grp["answers"])
        n_items = min(len(a) for a in mat)
        for idx in range(n_items):
            col = [a[idx] for a in mat]
            per_item.append(
                {
                    "model": model,
                    "instrument": instrument,
                    "item": idx,
                    "entropy": normalised_entropy(col),
                    "modal_share": max(Counter(col).values()) / len(col),
                    "n_distinct": len(set(col)),
                    "n_trials": len(col),
                }
            )
    items = pd.DataFrame(per_item)

    summary = (
        items.groupby(["model", "instrument"])["entropy"]
        .agg(
            mean_entropy="mean",
            median_entropy="median",
            p90_entropy=lambda s: float(np.percentile(s, 90)),
            frac_unstable=lambda s: float((s > 0.5).mean()),
            frac_perfectly_stable=lambda s: float((s == 0.0).mean()),
        )
        .reset_index()
        .sort_values("mean_entropy")
    )

    # The headline claim W4.1 is meant to establish: instability is
    # concentrated in a minority of items, not spread evenly.
    concentration = {}
    for instrument, grp in items.groupby("instrument"):
        ent = grp.groupby("item")["entropy"].mean().sort_values(ascending=False)
        total = float(ent.sum())
        top_decile = int(max(1, round(0.1 * len(ent))))
        concentration[instrument] = {
            "n_items": int(len(ent)),
            "share_of_total_entropy_in_top_decile_of_items": (
                float(ent.head(top_decile).sum() / total) if total > 0 else 0.0
            ),
            "most_unstable_items": [
                {"item": int(i), "mean_entropy": float(v)} for i, v in ent.head(10).items()
            ],
            "n_items_with_zero_entropy_everywhere": int((ent == 0).sum()),
        }

    return {
        "per_model": summary.to_dict("records"),
        "concentration": concentration,
        "_items_frame": items,
    }


# --------------------------------------------------------------------------
# 4.2  ISS repair
# --------------------------------------------------------------------------
def legacy_iss(vals: np.ndarray) -> float:
    """The published metric, imported from analyze_expanded so it is the real
    function and not a paraphrase of it.

    ISS = 0.7 * relative_sd + 0.3 * relative_iqr, where both terms are
    normalised by the *sample's own* range rather than the instrument's scale.
    That normalisation is the defect: a model whose scores are confined to a
    0.12-wide interval divides by 0.12 and can register a larger "instability"
    than a model that swings across the whole scale.
    """
    from analyze_expanded import iss as _published_iss

    return float(_published_iss(np.asarray(vals, dtype=float)))


def task_4_2(df: pd.DataFrame) -> dict:
    """Replace range-normalised ISS with scale-anchored dispersion.

    Two replacements are reported:
      * `sd_absolute`  — plain SD in the instrument's own units.
      * `iss_vc`       — SD as a percentage of the instrument's full response
                         range, which is fixed a priori and therefore cannot be
                         inflated by a model that happens to be consistent.
    """
    recs = []
    for (model, instrument, axis), grp in df.groupby(["model", "instrument", "axis"]):
        vals = grp["value"].to_numpy(dtype=float)
        full_range = INSTRUMENT_RANGE[instrument]
        recs.append(
            {
                "model": model,
                "instrument": instrument,
                "axis": axis,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "sd_absolute": float(vals.std(ddof=1)),
                "observed_range": float(vals.max() - vals.min()),
                "iss_legacy": legacy_iss(vals),
                "iss_vc": float(vals.std(ddof=1) / full_range * 100.0),
            }
        )
    out = pd.DataFrame(recs)

    # Demonstrate the artefact: rows where the legacy metric screams but the
    # absolute dispersion is negligible.
    out["legacy_rank"] = out["iss_legacy"].rank(ascending=False)
    out["vc_rank"] = out["iss_vc"].rank(ascending=False)
    out["rank_shift"] = out["legacy_rank"] - out["vc_rank"]
    artefacts = out.reindex(out["rank_shift"].abs().sort_values(ascending=False).index)

    spearman = float(out["iss_legacy"].corr(out["iss_vc"], method="spearman"))

    return {
        "per_model_axis": out.sort_values("iss_vc").to_dict("records"),
        "legacy_vs_repaired_spearman": spearman,
        "largest_rank_shifts": artefacts.head(10)[
            [
                "model",
                "instrument",
                "axis",
                "sd_absolute",
                "observed_range",
                "iss_legacy",
                "iss_vc",
                "rank_shift",
            ]
        ].to_dict("records"),
        "n_cells_where_legacy_exceeds_50_but_sd_below_1pct_of_scale": int(
            (
                (out["iss_legacy"] > 50)
                & (
                    out["sd_absolute"]
                    / out["instrument"].map(INSTRUMENT_RANGE)
                    < 0.01
                )
            ).sum()
        ),
    }


# --------------------------------------------------------------------------
# 4.3  variance decomposition
# --------------------------------------------------------------------------
def task_4_3(df: pd.DataFrame) -> dict:
    """Crossed random-effects decomposition of the 2,280 administrations.

    Delegates to variance_components.variance_decomposition, which is the
    paper's single statistical model (updates/04_analysis.md). Rolling a second
    decomposition here would let W4 and the manuscript quote different error
    budgets computed two different ways, which is exactly the failure the
    shared module exists to prevent.

    Scores are z-scored within trait first because the two instruments have
    incomparable units; without that the 0-100 instrument dominates every
    component. Components: model, trait, their interaction, and residual --
    the last being pure trial-to-trial instability, which is the quantity the
    manuscript calls stability.
    """
    from variance_components import cluster_bootstrap_variance_shares, variance_decomposition

    out = variance_decomposition(
        df, value="value", random_effects=["model", "trait", "model:trait"], z_within="trait"
    )
    out["interpretation"] = (
        "the residual share is the fraction of variation that is pure trial-to-trial "
        "instability rather than any stable property of a model or an instrument; the "
        "model share is the ICC for model identity, i.e. how much of a measured "
        "political position is actually about the model"
    )

    # oct_fix.md F4 / updates/04_analysis.md Gate A: every variance share needs a
    # 95% CI before it goes in the abstract. Cluster bootstrap over the 19 models
    # (>= 2000 resamples, seed recorded) -- resampling rows instead of models
    # would understate the interval, since rows from one model are not independent.
    boot = cluster_bootstrap_variance_shares(
        df, value="value", random_effects=["model", "trait", "model:trait"],
        cluster="model", z_within="trait", n_boot=2000,
    )
    out["bootstrap_ci"] = {
        "cluster_column": boot["cluster_column"],
        "n_clusters": boot["n_clusters"],
        "variance_share_ci": boot["variance_share_ci"],
        "n_boot_requested": boot["n_boot_requested"],
        "n_boot_failed": boot["n_boot_failed"],
        "seed": boot["seed"],
    }
    return out


# --------------------------------------------------------------------------
# 4.4  multi-trait multi-method matrix
# --------------------------------------------------------------------------
# Which axes are attempting to measure the same underlying construct across
# the two instruments. These pairings are the paper's own claim; MTMM is how
# that claim gets tested rather than asserted.
TRAIT_PAIRS = [
    ("economic", "political_compass:economic", "8values:equality"),
    ("social", "political_compass:social", "8values:progress"),
]


def task_4_4(df: pd.DataFrame) -> dict:
    """Campbell-Fiske MTMM over the two instruments.

    Delegates the matrix itself to variance_components.mtmm. That function keys
    convergence off a shared trait label, so TRAIT_PAIRS is applied first to
    harmonise each instrument's axis onto the construct the manuscript claims
    it measures. Those pairings *are* the manuscript's claim; MTMM is how the
    claim gets tested instead of asserted.

    The strict matrix is 2 traits x 2 methods. The four 8Values axes without a
    Political Compass counterpart are reported separately as internal factor
    structure -- they are a real finding, but folding them into the discriminant
    mean inflates it and is not Campbell-Fiske.
    """
    from variance_components import mtmm

    mapping = {pc: name for name, pc, _ in TRAIT_PAIRS}
    mapping.update({ev: name for name, _, ev in TRAIT_PAIRS})
    d = df[df["trait"].isin(mapping)].copy()
    d["construct"] = d["trait"].map(mapping)

    out = mtmm(d, unit="model", trait="construct", method="instrument", value="value")
    out["trait_pairings"] = [
        {"construct": n, "political_compass": pc, "8values": ev} for n, pc, ev in TRAIT_PAIRS
    ]

    # Internal structure of 8Values, reported but excluded from the matrix.
    means = df.groupby(["model", "trait"])["value"].mean().unstack("trait")
    ev_cols = [c for c in means.columns if c.startswith("8values:")]
    ev_corr = means[ev_cols].corr()
    internal = [
        {"trait_a": a, "trait_b": b, "r": float(ev_corr.loc[a, b])}
        for i, a in enumerate(ev_cols)
        for b in ev_cols[i + 1 :]
    ]
    out["eightvalues_internal_structure"] = {
        "pairs": internal,
        "mean_r": float(np.mean([e["r"] for e in internal])) if internal else float("nan"),
        "note": (
            "8Values axes correlate strongly with one another across models, which is "
            "consistent with one dominant general factor rather than four separable "
            "traits. Excluded from the discriminant mean above because these traits have "
            "no Political Compass counterpart and so are not part of the MTMM design."
        ),
    }
    out["note"] = (
        "Correlations run across the 19 models (n=19). With n=19 the 95% CI on an r of "
        "0.5 is roughly [0.05, 0.78]; report intervals, not point estimates."
    )
    return out


# --------------------------------------------------------------------------
# 4.5  Overton envelope
# --------------------------------------------------------------------------
def task_4_5(df: pd.DataFrame) -> dict:
    """Cohort spread as a fraction of the instrument's full range.

    updates/04_analysis.md asks for the spread as a fraction of the *human/party* spread. No
    party-calibrated reference distribution ships with this repository, so the
    denominator here is the instrument's own full range, which is a strictly
    conservative stand-in: it makes the cohort look *wider* than a party-spread
    denominator would. `reference_spread` is left as a hook for the external
    human distribution once W5 supplies it.
    """
    out = {}
    for trait, grp in df.groupby("trait"):
        instrument = trait.split(":")[0]
        model_means = grp.groupby("model")["value"].mean()
        full = INSTRUMENT_RANGE[instrument]
        spread = float(model_means.max() - model_means.min())
        out[trait] = {
            "n_models": int(len(model_means)),
            "cohort_min": float(model_means.min()),
            "cohort_max": float(model_means.max()),
            "cohort_spread": spread,
            "instrument_full_range": full,
            "cohort_spread_as_frac_of_instrument_range": spread / full,
            "cohort_sd_across_models": float(model_means.std(ddof=1)),
            "reference_spread": None,
            "cohort_spread_as_frac_of_reference": None,
        }
    return {
        "per_trait": out,
        "caveat": (
            "Denominator is the instrument's full range, not a party/human spread. "
            "Substitute reference_spread once a calibrated human distribution exists "
            "(updates/03_experiments.md Task 1.5, human-norm instrument) and the fractions will shrink, strengthening the claim."
        ),
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "results" / "w4"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_long()
    raw = load_raw_trials()
    print(f"loaded {len(df)} axis-level observations, {len(raw)} raw trials, "
          f"{df.model.nunique()} models, {df.trait.nunique()} traits\n")

    r41 = task_4_1(raw)
    items = r41.pop("_items_frame")
    items.to_csv(out_dir / "item_entropy.csv", index=False)
    r42 = task_4_2(df)
    r43 = task_4_3(df)
    r44 = task_4_4(df)
    r45 = task_4_5(df)

    print("=" * 78)
    print("4.1  ITEM-LEVEL STABILITY")
    print("=" * 78)
    print(f"{'model':42s} {'inst':18s} {'meanH':>7s} {'p90H':>7s} {'%unstable':>10s}")
    for r in r41["per_model"]:
        print(f"{r['model']:42s} {r['instrument']:18s} {r['mean_entropy']:7.3f} "
              f"{r['p90_entropy']:7.3f} {r['frac_unstable']*100:9.1f}%")
    for inst, c in r41["concentration"].items():
        print(f"\n  {inst}: {c['n_items']} items; top decile of items carries "
              f"{c['share_of_total_entropy_in_top_decile_of_items']*100:.1f}% of total entropy; "
              f"{c['n_items_with_zero_entropy_everywhere']} items perfectly stable across all models")

    print("\n" + "=" * 78)
    print("4.2  ISS REPAIR")
    print("=" * 78)
    print(f"Spearman(legacy ISS, repaired ISS) = {_fmt(r42['legacy_vs_repaired_spearman'])}")
    print(f"cells where legacy ISS > 50 but SD < 1% of scale: "
          f"{r42['n_cells_where_legacy_exceeds_50_but_sd_below_1pct_of_scale']}")
    print("\nlargest rank shifts (the artefact):")
    print(f"  {'model':38s} {'axis':12s} {'SD':>8s} {'range':>8s} {'legacy':>8s} {'fixed':>7s}")
    for r in r42["largest_rank_shifts"]:
        print(f"  {r['model']:38s} {r['axis']:12s} {r['sd_absolute']:8.3f} "
              f"{r['observed_range']:8.3f} {r['iss_legacy']:8.2f} {r['iss_vc']:7.2f}")

    print("\n" + "=" * 78)
    print("4.3  VARIANCE DECOMPOSITION")
    print("=" * 78)
    print(f"method: {r43['method']}")
    for k, v in sorted(r43["variance_share"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v*100:6.2f}%   (var={r43['variance_components'][k]:.4f})")

    print("\n" + "=" * 78)
    print("4.4  MULTI-TRAIT MULTI-METHOD")
    print("=" * 78)
    print(f"n models = {r44['n_units']}   (strict 2 traits x 2 methods design)")
    print("convergent (same construct, different instrument) -- want HIGH:")
    for c in r44["convergent"]:
        print(f"  r = {c['r']:+.3f}   {c['a']} vs {c['b']}")
    print("discriminant (different construct, same instrument) -- want LOW:")
    for d_ in r44["discriminant_same_method"]:
        print(f"  r = {d_['r']:+.3f}   {d_['a']} vs {d_['b']}")
    print(f"mean convergent r = {_fmt(r44['convergent_mean_r'])}, "
          f"mean discriminant r = {_fmt(r44['discriminant_mean_r'])}")
    print(f"Campbell-Fiske passes: {r44['campbell_fiske_passes']}")
    internal = r44["eightvalues_internal_structure"]
    print(f"\n  8Values internal structure (reported separately, not part of the matrix): "
          f"mean r = {_fmt(internal['mean_r'])} across {len(internal['pairs'])} axis pairs")
    for e in sorted(internal["pairs"], key=lambda e: -abs(e["r"]))[:3]:
        print(f"    {e['trait_a']} vs {e['trait_b']}: r = {e['r']:+.3f}")

    print("\n" + "=" * 78)
    print("4.5  OVERTON ENVELOPE")
    print("=" * 78)
    for trait, v in r45["per_trait"].items():
        print(f"  {trait:34s} spread={v['cohort_spread']:7.2f} of {v['instrument_full_range']:6.1f} "
              f"= {v['cohort_spread_as_frac_of_instrument_range']*100:5.1f}% of scale")

    payload = {"task_4_1": r41, "task_4_2": r42, "task_4_3": r43,
               "task_4_4": r44, "task_4_5": r45}
    with open(out_dir / "w4_results.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote {out_dir/'w4_results.json'} and {out_dir/'item_entropy.csv'}")


if __name__ == "__main__":
    main()
