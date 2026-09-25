"""Prompt-robustness and open-ended-generation checks (main.tex, Prompt-Robustness
and Open-Ended-Generation Checks subsection; pa_appendix.tex Sections S6-S7).

These two bounded validation arms (data/prompt_variants/, data/openended/) were
collected by scoring/collect_prompt_variants.py and scoring/collect_openended.py,
but neither had an analysis script committed alongside them, unlike every other
result in this paper -- the reported numbers (29 of 30 model-axis combinations
significant, eta^2 range and median, the largest single shift, and the two
open-ended correlations) had no script + saved output a reader could rerun. This
script is that missing piece, added when a full "does every number in the paper
trace to results/" pass caught the gap.

    python3 scoring/analyze_prompt_robustness.py [--out results/prompt_robustness]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RAW_TRIALS = REPO / "data" / "raw_trials"
PROMPT_VARIANTS = REPO / "data" / "prompt_variants"
OPENENDED = REPO / "data" / "openended"

SUBSET_MODELS = (
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-flash",
)
AXES = (
    ("political_compass", "economic"), ("political_compass", "social"),
    ("8values", "equality"), ("8values", "liberty"),
    ("8values", "peace"), ("8values", "progress"),
)


def _slug(model: str) -> str:
    return model.replace("/", "_")


def load_baseline(model: str, test: str, axis: str) -> list[float]:
    vals = []
    for path in RAW_TRIALS.glob(f"{_slug(model)}__{test}__trial*.json"):
        rec = json.loads(path.read_text())
        if rec.get("status") == "ok":
            vals.append(float(rec["scores"][axis]))
    return vals


def load_variant(model: str, test: str, axis: str, variant: str) -> list[float]:
    vals = []
    for path in PROMPT_VARIANTS.glob(f"{variant}__{_slug(model)}__{test}__trial*.json"):
        rec = json.loads(path.read_text())
        if rec.get("status") == "ok":
            vals.append(float(rec["scores"][axis]))
    return vals


def prompt_robustness() -> dict:
    variants = ["v2_direct", "v3_formal", "v4_casual", "v5_thirdperson"]
    per_cell = []
    for model in SUBSET_MODELS:
        for test, axis in AXES:
            baseline = load_baseline(model, test, axis)
            groups = [baseline] + [load_variant(model, test, axis, v) for v in variants]
            groups = [g for g in groups if g]
            if len(groups) < 2:
                continue
            f_stat, p_value = stats.f_oneway(*groups)
            grand_mean = np.mean([x for g in groups for x in g])
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_total = sum((x - grand_mean) ** 2 for g in groups for x in g)
            eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")
            means = {"baseline": np.mean(baseline)}
            means.update({v: np.mean(load_variant(model, test, axis, v)) for v in variants})
            per_cell.append({
                "model": model, "test": test, "axis": axis,
                "f_stat": f_stat, "p_value": p_value, "eta_sq": eta_sq,
                "n_per_group": [len(g) for g in groups],
                "condition_means": means,
                "range_across_conditions": max(means.values()) - min(means.values()),
            })
    n_significant = sum(1 for c in per_cell if c["p_value"] < 0.05)
    n_large_effect = sum(1 for c in per_cell if c["p_value"] < 0.05 and c["eta_sq"] >= 0.14)
    eta_sqs = [c["eta_sq"] for c in per_cell]
    largest_range = max(per_cell, key=lambda c: c["range_across_conditions"])
    return {
        "n_cells": len(per_cell),
        "n_significant_p_lt_05": n_significant,
        "n_large_effect_given_significant": n_large_effect,
        "eta_sq_min": min(eta_sqs), "eta_sq_max": max(eta_sqs),
        "eta_sq_median": float(np.median(eta_sqs)),
        "largest_shift_cell": largest_range,
        "per_cell": per_cell,
    }


def open_ended() -> dict:
    axis_scores = {"economic": {}, "social": {}}
    for path in OPENENDED.glob("*.json"):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok" or "lean_score" not in rec:
            continue
        axis = rec["axis"]
        if axis not in axis_scores:
            continue
        axis_scores[axis].setdefault(rec["model"], []).append(float(rec["lean_score"]))

    result = {}
    for axis in ("economic", "social"):
        pc_axis = "economic" if axis == "economic" else "social"
        models = sorted(axis_scores[axis])
        judged_means = [np.mean(axis_scores[axis][m]) for m in models]
        pc_means = [np.mean(load_baseline(m, "political_compass", pc_axis)) for m in models]
        r, p = stats.pearsonr(pc_means, judged_means)
        result[axis] = {
            "n_models": len(models), "models": models,
            "pearson_r": r, "p_value": p,
        }
    return result


def main(out_dir: str = "results/prompt_robustness") -> None:
    result = {"prompt_robustness": prompt_robustness(), "open_ended": open_ended()}
    out = Path(REPO / out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_robustness_results.json").write_text(json.dumps(result, indent=2, default=str))

    pr = result["prompt_robustness"]
    print(f"prompt-robustness: {pr['n_significant_p_lt_05']}/{pr['n_cells']} cells significant "
          f"(p<0.05); {pr['n_large_effect_given_significant']} of those large (eta2>=0.14)")
    print(f"  eta2 range [{pr['eta_sq_min']:.3f}, {pr['eta_sq_max']:.3f}], median {pr['eta_sq_median']:.3f}")
    c = pr["largest_shift_cell"]
    print(f"  largest shift: {c['model']} {c['test']}/{c['axis']}, range {c['range_across_conditions']:.2f}")
    for axis, r in result["open_ended"].items():
        print(f"open-ended {axis}: r={r['pearson_r']:.2f}, p={r['p_value']:.2f}, n={r['n_models']}")
    print(f"\nwrote {out / 'prompt_robustness_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/prompt_robustness")
    args = ap.parse_args()
    main(args.out)
