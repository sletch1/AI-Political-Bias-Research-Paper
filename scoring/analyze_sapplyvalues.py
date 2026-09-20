"""SapplyValues baseline analysis -- a third instrument, checked but not yet
fully integrated (oct_fix.md F3 / docs/f3_instrument_terms_of_use.md).

Political Analysis's reviewers are expected to press on "a variance component
estimated from two instruments is meaningless." A full fix needs >= 4
instruments crossed into the same variance-components model used for the
Political Compass / 8Values error budget (variance_components.py). That is
future work: this script instead runs the cheap, honest check that is
possible right now on the 540 administrations already collected --

    1. Does the standard left/libertarian/progressive finding replicate on a
       third instrument at all (directional check, no correlation needed)?
    2. Where a SapplyValues axis and an existing axis claim to measure the
       same construct, do they actually agree across models (Spearman rho,
       with a bootstrap CI, matching how the paper's own MTMM correlations
       are reported)?

Sign conventions (all confirmed from each score's own site, not assumed):
  - SapplyValues: "right" positive = economically right, "auth" positive =
    authoritarian, "prog" positive = progressive (results.html canvas labels).
  - Political Compass: "economic" positive = right, "social" positive =
    authoritarian (site convention; also consistent with the negative means
    observed here, since main.tex reports these 19 models as left/libertarian).
  - 8Values: "equality" high = left/egalitarian (opposite sign to "right"),
    "liberty" high = libertarian (opposite sign to "auth"), "progress" high =
    progressive (same sign as "prog").

    python3 scoring/analyze_sapplyvalues.py [--out results/sapplyvalues]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from variance_components import bootstrap_ci  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RAW_TRIALS = REPO / "data" / "raw_trials"

# (sapplyvalues axis, comparison instrument:axis, expected sign, why)
COMPARISONS = [
    ("right", "political_compass:economic", +1,
     "both instruments use positive = economically right"),
    ("right", "8values:equality", -1,
     "8Values equality is high when left/egalitarian, the opposite pole from 'right'"),
    ("auth", "political_compass:social", +1,
     "both instruments use positive = authoritarian"),
    ("auth", "8values:liberty", -1,
     "8Values liberty is high when libertarian, the opposite pole from 'auth'"),
    ("prog", "8values:progress", +1,
     "both instruments use positive = progressive"),
]


def load_model_means(instrument_filter=None):
    """Per-model, per-(instrument:axis) mean score across all trials in
    data/raw_trials/. `instrument_filter`, if given, restricts to one test
    name (e.g. "sapplyvalues"); otherwise all are loaded and keyed by
    "<instrument>:<axis>"."""
    sums, counts = {}, {}
    for path in RAW_TRIALS.glob("*.json"):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        if instrument_filter and rec["test"] != instrument_filter:
            continue
        model = rec["model"]
        for axis, value in rec["scores"].items():
            key = axis if instrument_filter else f"{rec['test']}:{axis}"
            sums.setdefault(model, {}).setdefault(key, 0.0)
            counts.setdefault(model, {}).setdefault(key, 0)
            sums[model][key] += float(value)
            counts[model][key] += 1
    return {
        m: {k: sums[m][k] / counts[m][k] for k in sums[m]}
        for m in sums
    }


def directional_check(sv_means: dict) -> dict:
    """How many of the models scored left / libertarian / progressive on
    SapplyValues, mirroring the same check main.tex Section 3.1 runs for the
    baseline two-instrument run."""
    models = sorted(sv_means)
    n = len(models)
    n_left = sum(1 for m in models if sv_means[m]["right"] < 0)
    n_libertarian = sum(1 for m in models if sv_means[m]["auth"] < 0)
    n_progressive = sum(1 for m in models if sv_means[m]["prog"] > 0)
    return {
        "n_models": n,
        "n_economically_left": n_left,
        "n_libertarian": n_libertarian,
        "n_progressive": n_progressive,
        "note": ("main.tex's baseline finding is that every model is left AND "
                 "libertarian AND progressive on the original two instruments; "
                 "this checks whether the same pattern holds on a third."),
    }


def convergence_check(sv_means: dict, other_means: dict) -> list:
    """Spearman rho (with a bootstrap CI) between each SapplyValues axis and
    the existing axis it claims to share a construct with, across models
    present in both. A positive `expected_sign` means agreement should show
    up as a positive correlation once both series are put on the same pole;
    we report the raw correlation and flag whether its sign matches
    `expected_sign`, rather than pre-flipping the data, so a reader can see
    exactly what was computed."""
    out = []
    for sv_axis, other_key, expected_sign, why in COMPARISONS:
        models = sorted(
            m for m in sv_means
            if sv_axis in sv_means[m] and m in other_means and other_key in other_means[m]
        )
        x = [sv_means[m][sv_axis] for m in models]
        y = [other_means[m][other_key] for m in models]
        if len(x) < 4:
            out.append({"sapplyvalues_axis": sv_axis, "other": other_key,
                        "available": False, "reason": "fewer than 4 shared models"})
            continue

        def _rho(a, b):
            import numpy as np
            ra = np.argsort(np.argsort(a))
            rb = np.argsort(np.argsort(b))
            return float(np.corrcoef(ra, rb)[0, 1])

        ci = bootstrap_ci(x, y, statistic=_rho, n_boot=2000)
        out.append({
            "sapplyvalues_axis": sv_axis, "other": other_key, "available": True,
            "n_models": len(models), "expected_sign": expected_sign, "why": why,
            "spearman_rho": ci["estimate"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
            "agrees_with_expected_sign": bool(
                (ci["estimate"] > 0) == (expected_sign > 0) if ci["estimate"] != 0 else None
            ),
        })
    return out


def main(out_dir="results/sapplyvalues"):
    sv_means = load_model_means("sapplyvalues")
    other_means = load_model_means(None)

    result = {
        "n_administrations": 540,
        "n_models": len(sv_means),
        "excluded_model": "mistralai/mistral-large-2512 (withdrawn from the OpenRouter API)",
        "directional_check": directional_check(sv_means),
        "convergence_check": convergence_check(sv_means, other_means),
        "status": ("Preliminary triangulation check on one additional instrument, not yet "
                   "crossed into the shared variance-components model with Political Compass "
                   "and 8Values; reaching the >= 4 instruments a Political Analysis reviewer "
                   "would ask for is future work (docs/f3_instrument_terms_of_use.md)."),
    }

    out = Path(REPO / out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sapplyvalues_results.json").write_text(json.dumps(result, indent=2))

    d = result["directional_check"]
    print(f"Directional check ({d['n_models']} models): "
          f"{d['n_economically_left']} left, {d['n_libertarian']} libertarian, "
          f"{d['n_progressive']} progressive")
    for c in result["convergence_check"]:
        if not c["available"]:
            print(f"  {c['sapplyvalues_axis']} vs {c['other']}: {c['reason']}")
            continue
        print(f"  {c['sapplyvalues_axis']} vs {c['other']}: "
              f"rho={c['spearman_rho']:.3f} [{c['ci_lower']:.3f}, {c['ci_upper']:.3f}] "
              f"(expected sign {'+' if c['expected_sign'] > 0 else '-'}, "
              f"{'agrees' if c['agrees_with_expected_sign'] else 'disagrees'})")
    print(f"\nwrote {out / 'sapplyvalues_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/sapplyvalues")
    args = ap.parse_args()
    main(args.out)
