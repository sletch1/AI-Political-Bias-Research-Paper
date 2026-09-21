"""Pew 2026 Political Typology analysis -- the fourth instrument
(oct_fix.md F3; see docs/f3_instrument_terms_of_use.md).

Unlike Political Compass / 8Values / SapplyValues, this instrument has no
continuous per-axis score to decompose: each administration returns one of
Pew's own nine typology groups, authoritative because it comes from driving
the live quiz (score_pew_typology.py), not from reimplementing Pew's
unpublished classification. The natural score is therefore the group's 1-9
ordinal position in Pew's own left-to-right ordering, and the natural
questions are (1) does the standard left-lean finding replicate on this
instrument, using Pew's own classification rather than any score this
project computed, and (2) does that ordinal position correlate with the
existing instruments' economic axes across the 18 shared models.

    python3 scoring/analyze_pew_typology.py [--out results/pew_typology]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from variance_components import bootstrap_ci  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PEW_DIR = REPO / "data" / "pew_typology"
RAW_TRIALS = REPO / "data" / "raw_trials"

GROUPS_LEFT_TO_RIGHT = [
    "Leftward Progressives", "Loyal Liberals", "Left-Out Left",
    "Order and Opportunity Left", "Tuned-Out Middle",
    "Pragmatic and Polite Right", "Unconventional Right",
    "Faith First Conservatives", "No Apologies Right",
]

# Same convergence pairing logic as analyze_sapplyvalues.py: an existing axis
# that claims to measure the same construct as Pew's overall left-right
# placement, with the sign that predicts agreement if the two are really
# measuring the same thing. Higher Pew ordinal position = further right.
COMPARISONS = [
    ("political_compass:economic", +1, "both use higher = more economically right"),
    ("8values:equality", -1, "8Values equality is high when left/egalitarian, opposite pole from 'right'"),
]


def load_pew_positions():
    """Per-model list of ordinal positions (1-9), one per trial."""
    by_model = {}
    for path in PEW_DIR.glob("*.json"):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        by_model.setdefault(rec["model"], []).append(rec["scores"]["ordinal_position"])
    return by_model


def load_other_axis_means():
    """Per-model mean for every <instrument>:<axis> in data/raw_trials/
    (Political Compass, 8Values, SapplyValues alike), matching
    analyze_sapplyvalues.py's loader so the two triangulation checks are
    computed the same way."""
    sums, counts = {}, {}
    for path in RAW_TRIALS.glob("*.json"):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        model = rec["model"]
        for axis, value in rec["scores"].items():
            key = f"{rec['test']}:{axis}"
            sums.setdefault(model, {}).setdefault(key, 0.0)
            counts.setdefault(model, {}).setdefault(key, 0)
            sums[model][key] += float(value)
            counts[model][key] += 1
    return {m: {k: sums[m][k] / counts[m][k] for k in sums[m]} for m in sums}


def directional_check(positions: dict) -> dict:
    """How many of the 18 models' modal (most common) Pew group falls left
    of centre (position <= 4), centre (5), or right (>= 6)."""
    modal = {m: Counter(v).most_common(1)[0][0] for m, v in positions.items()}
    n_left = sum(1 for p in modal.values() if p <= 4)
    n_centre = sum(1 for p in modal.values() if p == 5)
    n_right = sum(1 for p in modal.values() if p >= 6)
    return {
        "n_models": len(modal),
        "n_left_of_centre": n_left,
        "n_centre": n_centre,
        "n_right_of_centre": n_right,
        "modal_group_by_model": {m: GROUPS_LEFT_TO_RIGHT[p - 1] for m, p in modal.items()},
        "note": ("main.tex's baseline finding is that every model leans left on the original "
                 "two instruments; this checks whether that holds on Pew's own authoritative "
                 "classification, which this project did not compute."),
    }


def convergence_check(positions: dict, other_means: dict) -> list:
    """Spearman rho (bootstrap CI) between each model's mean Pew ordinal
    position and its mean score on an existing axis claiming the same
    construct, mirroring analyze_sapplyvalues.py's convergence_check."""
    mean_position = {m: sum(v) / len(v) for m, v in positions.items()}
    out = []
    for other_key, expected_sign, why in COMPARISONS:
        models = sorted(m for m in mean_position
                        if m in other_means and other_key in other_means[m])
        x = [mean_position[m] for m in models]
        y = [other_means[m][other_key] for m in models]
        if len(x) < 4:
            out.append({"other": other_key, "available": False,
                       "reason": "fewer than 4 shared models"})
            continue

        def _rho(a, b):
            import numpy as np
            ra = np.argsort(np.argsort(a))
            rb = np.argsort(np.argsort(b))
            return float(np.corrcoef(ra, rb)[0, 1])

        ci = bootstrap_ci(x, y, statistic=_rho, n_boot=2000)
        out.append({
            "other": other_key, "available": True, "n_models": len(models),
            "expected_sign": expected_sign, "why": why,
            "spearman_rho": ci["estimate"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
            "agrees_with_expected_sign": bool(
                (ci["estimate"] > 0) == (expected_sign > 0) if ci["estimate"] != 0 else None
            ),
        })
    return out


def main(out_dir="results/pew_typology"):
    positions = load_pew_positions()
    other_means = load_other_axis_means()

    result = {
        "n_administrations": sum(len(v) for v in positions.values()),
        "n_models": len(positions),
        "excluded_model": "mistralai/mistral-large-2512 (withdrawn from the OpenRouter API)",
        "groups_left_to_right": GROUPS_LEFT_TO_RIGHT,
        "directional_check": directional_check(positions),
        "convergence_check": convergence_check(positions, other_means),
        "status": ("Fourth-instrument check using Pew's own live classification (not a formula "
                   "this project computed). Ordinal-position scoring (1-9) is coarser than the "
                   "continuous axes of the other three instruments and is not crossed into the "
                   "shared variance-components model; see docs/f3_instrument_terms_of_use.md."),
    }

    out = Path(REPO / out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pew_typology_results.json").write_text(json.dumps(result, indent=2))

    d = result["directional_check"]
    print(f"Directional check ({d['n_models']} models): "
          f"{d['n_left_of_centre']} left-of-centre, {d['n_centre']} centre, "
          f"{d['n_right_of_centre']} right-of-centre")
    for m, g in sorted(d["modal_group_by_model"].items()):
        print(f"  {m:42s} {g}")
    for c in result["convergence_check"]:
        if not c["available"]:
            print(f"  {c['other']}: {c['reason']}")
            continue
        print(f"  ordinal position vs {c['other']}: rho={c['spearman_rho']:.3f} "
              f"[{c['ci_lower']:.3f}, {c['ci_upper']:.3f}] "
              f"(expected sign {'+' if c['expected_sign'] > 0 else '-'}, "
              f"{'agrees' if c['agrees_with_expected_sign'] else 'disagrees'})")
    print(f"\nwrote {out / 'pew_typology_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/pew_typology")
    args = ap.parse_args()
    main(args.out)
