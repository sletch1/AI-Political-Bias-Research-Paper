"""Workstream 1 analysis -- the validity battery.

Consumes whatever W1 data exists and produces the error-budget numbers that
Gate A (updates/04_analysis.md) turns on. Every arm is optional: an arm whose data has not
been collected is reported as absent rather than silently skipped, so the
output doubles as a progress report on the workstream.

    python3 scoring/analyze_w1.py [--out results/w1]

Arms
----
1.1  Asker identity. Two-way ANOVA (model x condition) per axis with partial
     eta-squared, the conservative/progressive accommodation asymmetry ratio
     that tests whether Tornberg's 8.0x replicates on our roster, and the
     anchor check that the `none` cell reproduces data/raw_trials/.
1.2  Contamination. Paired original-vs-inverted comparison per (model, axis),
     the Contamination Index, BH-corrected significance across models, and the
     verbatim-completion rate from the memorisation probe.
1.3  Response format. Variance components per format factor, and each model's
     position reported as a design-averaged interval rather than a point.

The primary decomposition for all three comes from variance_components.py, so
the error budget is computed the same way everywhere.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from variance_components import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_ci,
    partial_eta_squared,
    variance_decomposition,
)
from variant_scoring import INSTRUMENT_RANGE, contamination_index  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DIRS = {
    "asker": REPO / "data" / "asker_identity",
    "variants": REPO / "data" / "item_variants",
    "format": REPO / "data" / "format_factorial",
}
BASELINE = REPO / "data" / "raw_trials"

AXES = {"8values": ("equality", "peace", "liberty", "progress"),
        "political_compass": ("economic", "social")}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_trials(directory: Path, extra_keys=()) -> pd.DataFrame:
    """Long-form scored trials from a collection directory.

    One row per (trial, axis). Failed trials are dropped from the analysis
    frame but counted, because a condition that a model refuses more often is
    itself a result and must not vanish into a completion rate nobody prints.
    """
    rows, failures = [], []
    if not directory.exists():
        return pd.DataFrame(rows), pd.DataFrame(failures)
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith(("probe__", "memprobe__")):
            continue
        rec = json.loads(path.read_text())
        base = {"model": rec.get("model"), "instrument": rec.get("test"),
                "trial": rec.get("trial")}
        for k in extra_keys:
            base[k] = rec.get(k)
        if rec.get("status") != "ok":
            failures.append({**base, "status": rec.get("status")})
            continue
        for axis, value in rec["scores"].items():
            if axis in ("n_neutral", "bracket_width"):
                base[axis] = value
                continue
            rows.append({**base, "axis": axis, "value": float(value)})
    df = pd.DataFrame(rows)
    if len(df):
        df["trait"] = df["instrument"] + ":" + df["axis"]
    return df, pd.DataFrame(failures)


def load_baseline() -> pd.DataFrame:
    """The published 2,280-administration run, in the same long form."""
    rows = []
    for path in sorted(BASELINE.glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        for axis, value in rec["scores"].items():
            rows.append({"model": rec["model"], "instrument": rec["test"],
                         "trial": rec["trial"], "axis": axis, "value": float(value)})
    df = pd.DataFrame(rows)
    df["trait"] = df["instrument"] + ":" + df["axis"]
    return df


# --------------------------------------------------------------------------
# 1.1  asker identity
# --------------------------------------------------------------------------
def anchor_check(asker: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """Does the `none` condition reproduce the published run?

    The whole factorial hangs on this. If the anchor cell drifts from
    data/raw_trials/ then either the prompt is not byte-identical after all or
    the model versions have moved under us (updates/03_experiments.md Task 3.1 (drift)), and in either case the
    new conditions cannot be compared to the published baseline.
    """
    anchor = asker[asker.condition == "none"]
    if anchor.empty:
        return {"available": False, "reason": "no `none` condition collected"}
    a = anchor.groupby(["model", "trait"])["value"].mean()
    b = baseline.groupby(["model", "trait"])["value"].mean()
    common = a.index.intersection(b.index)
    if len(common) < 4:
        return {"available": False, "reason": "too few overlapping cells"}
    diff = (a.loc[common] - b.loc[common])
    per_trait = []
    for trait in sorted({t for _, t in common}):
        sel = [i for i in common if i[1] == trait]
        d = diff.loc[sel]
        scale = INSTRUMENT_RANGE[trait.split(":")[0]]
        per_trait.append({"trait": trait, "n_cells": len(sel),
                          "mean_drift": float(d.mean()),
                          "max_abs_drift": float(d.abs().max()),
                          "max_abs_drift_as_frac_of_scale": float(d.abs().max() / scale)})
    worst = max(p["max_abs_drift_as_frac_of_scale"] for p in per_trait)
    return {"available": True, "per_trait": per_trait,
            "worst_drift_as_frac_of_scale": worst,
            "anchor_holds": bool(worst < 0.05),
            "note": ("A drift above 5% of scale means the anchor cell no longer "
                     "reproduces data/raw_trials/; investigate model version drift "
                     "(updates/03_experiments.md Task 3.1 (drift)) before interpreting any condition effect.")}


def asker_arm(asker: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    if asker.empty:
        return {"available": False, "reason": f"no data in {DIRS['asker']}"}

    out = {"available": True, "n_administrations": int(len(asker) // 1),
           "conditions": sorted(asker.condition.unique()),
           "models": int(asker.model.nunique()),
           "anchor_check": anchor_check(asker, baseline)}

    per_axis = []
    for trait, grp in asker.groupby("trait"):
        eta = partial_eta_squared(grp, "value", ["model", "condition"])
        entry = {"trait": trait,
                 "eta_sq_model": eta.get("model", {}).get("partial_eta_squared"),
                 "eta_sq_condition": eta.get("condition", {}).get("partial_eta_squared"),
                 "p_condition": eta.get("condition", {}).get("p_value")}
        means = grp.groupby("condition")["value"].mean()
        entry["condition_means"] = {c: float(v) for c, v in means.items()}
        # Tornberg's asymmetry: rightward accommodation was 8.0x leftward.
        if {"none", "conservative", "progressive"} <= set(means.index):
            right = abs(means["conservative"] - means["none"])
            left = abs(means["progressive"] - means["none"])
            entry["shift_conservative"] = float(means["conservative"] - means["none"])
            entry["shift_progressive"] = float(means["progressive"] - means["none"])
            entry["asymmetry_ratio"] = float(right / left) if left > 0 else float("inf")
        per_axis.append(entry)
    out["per_axis"] = per_axis

    out["variance_decomposition"] = variance_decomposition(
        asker, "value", random_effects=["model", "model:condition"],
        fixed_effects=["condition"], z_within="trait")

    etas = [e["eta_sq_condition"] for e in per_axis if e["eta_sq_condition"] is not None]
    out["gate_a"] = {
        "max_condition_eta_squared": float(np.max(etas)) if etas else None,
        "mean_condition_eta_squared": float(np.mean(etas)) if etas else None,
        "verdict": _gate_a(etas),
    }
    return out


def _gate_a(etas) -> str:
    """updates/04_analysis.md, Gate A, stated in its own thresholds."""
    if not etas:
        return "no data"
    m = float(np.max(etas))
    if m > 0.10:
        return ("PROCEED: a measurement factor explains a non-trivial share of variance; "
                "the error-budget thesis is confirmed and NMI remains the target")
    if m < 0.05:
        return ("REFRAME: measurement factors are small; the finding becomes "
                "'questionnaire audits are more robust than the 2026 critiques imply'. "
                "Drop W3, target npj AI or Nature Communications")
    return "AMBIGUOUS (0.05-0.10): report with intervals and decide on the format arm too"


# --------------------------------------------------------------------------
# 1.2  contamination
# --------------------------------------------------------------------------
def contamination_arm(variants: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    if variants.empty:
        return {"available": False, "reason": f"no data in {DIRS['variants']}"}

    out = {"available": True, "conditions": sorted(variants.condition.unique()),
           "models": int(variants.model.nunique())}

    # If the `original` condition was not re-collected, fall back to the
    # published baseline, and say so: the comparison is then across time as
    # well as across condition, which is a weaker design.
    if "original" in set(variants.condition):
        original = variants[variants.condition == "original"]
        out["original_source"] = "re-collected alongside the variants"
    else:
        original = baseline.assign(condition="original")
        out["original_source"] = ("data/raw_trials/ baseline -- comparison spans "
                                  "time as well as condition; see updates/03_experiments.md Task 3.1 (drift)")

    rows, p_values = [], []
    for (model, trait), grp in variants.groupby(["model", "trait"]):
        base = original[(original.model == model) & (original.trait == trait)]["value"]
        if base.empty:
            continue
        instrument = trait.split(":")[0]
        entry = {"model": model, "trait": trait,
                 "mean_original": float(base.mean()), "n_original": int(len(base))}
        for condition in ("inverted", "paraphrased"):
            sel = grp[grp.condition == condition]["value"]
            if sel.empty:
                continue
            entry[f"mean_{condition}"] = float(sel.mean())
            entry[f"ci_{condition}"] = contamination_index(
                base.mean(), sel.mean(), instrument)
            t, p = stats.ttest_ind(base, sel, equal_var=False)
            entry[f"t_{condition}"] = float(t)
            entry[f"p_{condition}"] = float(p)
            if condition == "inverted":
                p_values.append(p)
        rows.append(entry)

    if not rows:
        return {"available": False, "reason": "no overlapping (model, trait) cells"}

    bh = benjamini_hochberg(p_values)
    inverted_rows = [r for r in rows if "p_inverted" in r]
    for r, adj, rej in zip(inverted_rows, bh["adjusted"], bh["rejected"]):
        r["p_inverted_bh"] = adj
        r["inverted_differs_after_fdr"] = bool(rej)

    cis = [r["ci_inverted"] for r in inverted_rows if "ci_inverted" in r]
    n_models_flagged = len({r["model"] for r in inverted_rows
                            if r.get("inverted_differs_after_fdr")})
    out.update({
        "per_cell": sorted(rows, key=lambda r: -r.get("ci_inverted", 0)),
        "mean_contamination_index": float(np.mean(cis)) if cis else None,
        "max_contamination_index": float(np.max(cis)) if cis else None,
        "n_cells_flagged_after_fdr": bh["n_rejected"],
        "n_models_flagged_after_fdr": n_models_flagged,
        "n_models": int(variants.model.nunique()),
    })
    return out


def memorisation_arm(directory: Path) -> dict:
    """Verbatim-completion rate from the Task 1.2 memorisation probe.

    Similarity is computed here rather than at collection time so the metric
    can be revised without re-spending API budget. Two thresholds are reported:
    an exact normalised match, and a 0.9 sequence-similarity near-match that
    tolerates punctuation and casing drift.
    """
    files = sorted(directory.glob("memprobe__*.json")) if directory.exists() else []
    if not files:
        return {"available": False, "reason": "no memorisation probe collected"}

    per_model = []
    for path in files:
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            per_model.append({"model": rec["model"], "status": rec.get("status")})
            continue
        exact = near = 0
        details = []
        for item, completion in zip(rec["items"], rec["completions"]):
            truth = _normalise(item["full"])
            got = _normalise(str(completion))
            ratio = difflib.SequenceMatcher(None, truth, got).ratio()
            exact += truth == got
            near += ratio >= 0.90
            details.append({"instrument": item["instrument"], "item": item["item_index"],
                            "similarity": round(ratio, 4)})
        n = len(rec["items"])
        per_model.append({"model": rec["model"], "trial": rec["trial"], "n_items": n,
                          "exact_rate": exact / n, "near_rate": near / n,
                          "mean_similarity": float(np.mean([d["similarity"] for d in details])),
                          "items": details})
    scored = [m for m in per_model if "exact_rate" in m]
    return {"available": bool(scored), "per_model": sorted(
        scored, key=lambda m: -m["near_rate"]),
        "mean_exact_rate": float(np.mean([m["exact_rate"] for m in scored])) if scored else None,
        "mean_near_rate": float(np.mean([m["near_rate"] for m in scored])) if scored else None,
        "note": ("A high verbatim-completion rate is direct evidence the item bank is "
                 "in pretraining data, independent of any score shift.")}


def _normalise(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# --------------------------------------------------------------------------
# 1.3  response format
# --------------------------------------------------------------------------
def format_arm(fmt: pd.DataFrame) -> dict:
    if fmt.empty:
        return {"available": False, "reason": f"no data in {DIRS['format']}"}

    for factor in ("scale", "item_order", "option_order", "elicitation"):
        fmt[factor] = fmt["config"].map(lambda c: c.get(factor) if isinstance(c, dict) else None)

    out = {"available": True, "n_configs": int(fmt.config_id.nunique()),
           "models": int(fmt.model.nunique())}

    out["variance_decomposition"] = variance_decomposition(
        fmt, "value",
        random_effects=["model", "model:scale", "model:elicitation"],
        fixed_effects=["scale", "item_order", "option_order", "elicitation"],
        z_within="trait")

    per_axis = []
    for trait, grp in fmt.groupby("trait"):
        eta = partial_eta_squared(
            grp, "value", ["model", "scale", "item_order", "option_order", "elicitation"])
        per_axis.append({"trait": trait,
                         **{k: v["partial_eta_squared"] for k, v in eta.items()
                            if isinstance(v, dict)}})
    out["partial_eta_squared_per_axis"] = per_axis

    # Debevc et al.'s point: a model's position under a varied design is an
    # interval, not a point. Report it as one.
    intervals = []
    for (model, trait), grp in fmt.groupby(["model", "trait"]):
        by_config = grp.groupby("config_id")["value"].mean()
        intervals.append({
            "model": model, "trait": trait, "n_configs": int(len(by_config)),
            "design_averaged_mean": float(by_config.mean()),
            "min_config_mean": float(by_config.min()),
            "max_config_mean": float(by_config.max()),
            "range_as_frac_of_scale": float(
                (by_config.max() - by_config.min()) / INSTRUMENT_RANGE[trait.split(":")[0]]),
            "bootstrap_ci": bootstrap_ci(by_config.to_numpy(), n_boot=2000),
        })
    out["design_averaged_positions"] = sorted(
        intervals, key=lambda r: -r["range_as_frac_of_scale"])

    if "bracket_width" in fmt.columns:
        widths = fmt["bracket_width"].dropna()
        if len(widths):
            out["forced_choice_bracket"] = {
                "mean_width": float(widths.mean()), "max_width": float(widths.max()),
                "note": ("Width of the Political Compass bracket induced by responses "
                         "that were neutral on a scale the instrument has no neutral "
                         "for; see format_design.py.")}
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "results" / "w1"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline()
    asker, asker_fail = load_trials(DIRS["asker"], extra_keys=("condition",))
    variants, variant_fail = load_trials(DIRS["variants"], extra_keys=("condition",))
    fmt, fmt_fail = load_trials(DIRS["format"], extra_keys=("config", "config_id"))

    results = {
        "baseline": {"n_rows": int(len(baseline)), "models": int(baseline.model.nunique())},
        "task_1_1_asker_identity": asker_arm(asker, baseline),
        "task_1_2_contamination": contamination_arm(variants, baseline),
        "task_1_2_memorisation": memorisation_arm(DIRS["variants"]),
        "task_1_3_format": format_arm(fmt),
        "failures": {"asker": len(asker_fail), "variants": len(variant_fail),
                     "format": len(fmt_fail)},
    }

    print("=" * 78)
    print("W1 VALIDITY BATTERY")
    print("=" * 78)
    print(f"baseline: {results['baseline']['n_rows']} axis-level observations, "
          f"{results['baseline']['models']} models\n")

    a = results["task_1_1_asker_identity"]
    print("-- 1.1 asker identity " + "-" * 55)
    if not a["available"]:
        print(f"   not collected ({a['reason']})")
    else:
        anchor = a["anchor_check"]
        if anchor.get("available"):
            print(f"   anchor cell reproduces baseline: {anchor['anchor_holds']} "
                  f"(worst drift {anchor['worst_drift_as_frac_of_scale']*100:.1f}% of scale)")
        for e in a["per_axis"]:
            print(f"   {e['trait']:28s} eta2_condition={e['eta_sq_condition']:.3f}  "
                  f"asymmetry={e.get('asymmetry_ratio', float('nan')):.2f}x")
        print(f"   GATE A: {a['gate_a']['verdict']}")

    c = results["task_1_2_contamination"]
    print("\n-- 1.2 contamination " + "-" * 56)
    if not c["available"]:
        print(f"   not collected ({c['reason']})")
    else:
        print(f"   original from: {c['original_source']}")
        print(f"   mean Contamination Index = {c['mean_contamination_index']:.4f}, "
              f"max = {c['max_contamination_index']:.4f}")
        print(f"   {c['n_models_flagged_after_fdr']}/{c['n_models']} models differ "
              f"under inversion after BH correction")
    m = results["task_1_2_memorisation"]
    if m["available"]:
        print(f"   memorisation: mean verbatim rate {m['mean_exact_rate']*100:.1f}%, "
              f"near-verbatim {m['mean_near_rate']*100:.1f}%")
    else:
        print(f"   memorisation probe: {m['reason']}")

    f = results["task_1_3_format"]
    print("\n-- 1.3 response format " + "-" * 54)
    if not f["available"]:
        print(f"   not collected ({f['reason']})")
    else:
        for k, v in sorted(f["variance_decomposition"]["variance_share"].items(),
                           key=lambda kv: -kv[1]):
            print(f"   {k:20s} {v*100:6.2f}%")
        widest = f["design_averaged_positions"][0]
        print(f"   widest design range: {widest['model']} on {widest['trait']} spans "
              f"{widest['range_as_frac_of_scale']*100:.1f}% of scale across configs")

    path = out_dir / "w1_results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
