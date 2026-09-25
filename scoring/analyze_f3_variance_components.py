"""F3/F4 crossing: fold SapplyValues and Pew's 2026 Political Typology into the
shared variance-components model (oct_fix.md F3 + F4; docs/f3_instrument_terms_of_use.md).

analyze_w4.py task 4.3 already crosses Political Compass and 8Values through
variance_components.variance_decomposition; analyze_sapplyvalues.py and
analyze_pew_typology.py each collected a third and fourth instrument but explicitly
stopped short of this step -- both say in their own `status` field that they are "not
... crossed into the shared variance-components model". This script is that crossing.

It adds no new statistical machinery. It builds one pooled long-format frame over all
four instruments and hands it to the exact same `variance_decomposition` /
`cluster_bootstrap_variance_shares` functions analyze_w4.py uses, per
variance_components.py's own rule that every workstream goes through the one shared
model rather than rolling a second implementation that could quote a different error
budget computed a different way.

Design notes
------------
* Each instrument's own axis (political_compass:economic, 8values:progress,
  sapplyvalues:auth, pew_typology:position, ...) enters as its own "trait" level,
  exactly like the original two-instrument design: instrument identity is folded into
  trait rather than fit as a separate term. That mirrors how the existing "instrument/
  axis 52%" headline figure (analyze_w4.py task 4.3) is already defined, so this
  extension changes which traits feed the model, not what the model estimates.
  Separating instrument-level variance from axis-within-instrument variance is a
  further extension, not attempted here -- see `note` in the output.
* z_within="trait" is required and applied throughout: Political Compass and
  SapplyValues run +/-10, 8Values runs 0-100, and Pew's ordinal typology position runs
  1-9. Without it the 0-100 axis dominates every component (see
  test_statistics.py::test_z_within_puts_two_scales_on_a_common_footing).
* Political Compass and 8Values were collected for 19 models; SapplyValues and Pew for
  18 (mistralai/mistral-large-2512 was withdrawn from OpenRouter partway through
  collection). The crossed design does not require a balanced panel, but this is a
  real asymmetry, not an artefact, and is reported in the output rather than silently
  dropping the 19th model from the two original instruments.
* Pew contributes one trait (its own composite left-right placement, ordinal 1-9),
  not per-axis scores like the other three -- it is Pew's own classification, not a
  score this project computed (see analyze_pew_typology.py).

    python3 scoring/analyze_f3_variance_components.py [--out results/f3_variance_components]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from variance_components import cluster_bootstrap_variance_shares, variance_decomposition  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RAW_TRIALS = REPO / "data" / "raw_trials"
PEW_DIR = REPO / "data" / "pew_typology"

# Full response range per instrument axis, for reporting alongside the model, not for
# the decomposition itself (z_within handles the scale difference statistically).
INSTRUMENT_RANGE = {
    "political_compass": 20.0,  # -10..+10
    "8values": 100.0,           # 0..100
    "sapplyvalues": 20.0,       # -10..+10
    "pew_typology": 8.0,        # ordinal position 1..9
}

RAW_TRIALS_INSTRUMENTS = ("political_compass", "8values", "sapplyvalues")


def load_pooled_long(instruments=("political_compass", "8values", "sapplyvalues", "pew_typology")) -> pd.DataFrame:
    """One row per (model, instrument, axis, trial), restricted to `instruments`.

    Political Compass, 8Values and SapplyValues come from data/raw_trials/'s "scores"
    dict (one key per axis); Pew's own single composite score is its ordinal
    left-right position, from data/pew_typology/. `instruments` defaults to all four
    but accepts a subset, which is how this module's own diagnostic run isolates which
    added instrument moves the decomposition.
    """
    rows = []
    for path in sorted(RAW_TRIALS.glob("*.json")):
        rec = json.loads(path.read_text())
        if (rec.get("status") != "ok" or rec.get("test") not in RAW_TRIALS_INSTRUMENTS
                or rec["test"] not in instruments):
            continue
        for axis, value in rec["scores"].items():
            rows.append({
                "model": rec["model"], "instrument": rec["test"],
                "trial": int(rec["trial"]), "axis": axis, "value": float(value),
            })
    if "pew_typology" in instruments:
        for path in sorted(PEW_DIR.glob("*.json")):
            rec = json.loads(path.read_text())
            if rec.get("status") != "ok":
                continue
            rows.append({
                "model": rec["model"], "instrument": "pew_typology",
                "trial": int(rec["trial"]), "axis": "position",
                "value": float(rec["scores"]["ordinal_position"]),
            })
    df = pd.DataFrame(rows)
    df["trait"] = df["instrument"] + ":" + df["axis"]
    return df


def panel_summary(df: pd.DataFrame) -> dict:
    per_instrument = (
        df.groupby("instrument")
        .agg(n_administrations=("trial", "count"), n_models=("model", "nunique"),
             n_axes=("axis", "nunique"))
        .to_dict("index")
    )
    models_by_instrument = {i: set(g["model"]) for i, g in df.groupby("instrument")}
    common = set.intersection(*models_by_instrument.values()) if models_by_instrument else set()
    return {
        "per_instrument": per_instrument,
        "n_models_common_to_all_four": len(common),
        "models_missing_from_at_least_one_instrument": sorted(
            set.union(*models_by_instrument.values()) - common
        ) if models_by_instrument else [],
    }


def main(out_dir: str = "results/f3_variance_components", n_boot: int = 2000,
        instruments=("political_compass", "8values", "sapplyvalues", "pew_typology")) -> None:
    df = load_pooled_long(instruments)

    decomposition = variance_decomposition(
        df, value="value", random_effects=["model", "trait", "model:trait"], z_within="trait",
    )
    decomposition["interpretation"] = (
        "same quantities as analyze_w4.py task 4.3 (model share = ICC for model "
        "identity, residual = pure trial-to-trial instability), now estimated over all "
        "four instruments' traits instead of two"
    )

    boot = cluster_bootstrap_variance_shares(
        df, value="value", random_effects=["model", "trait", "model:trait"],
        cluster="model", z_within="trait", n_boot=n_boot,
    )

    result = {
        "instruments": sorted(df["instrument"].unique()),
        "traits": sorted(df["trait"].unique()),
        "panel": panel_summary(df),
        "decomposition": decomposition,
        "bootstrap_ci": {
            "cluster_column": boot["cluster_column"],
            "n_clusters": boot["n_clusters"],
            "variance_share_ci": boot["variance_share_ci"],
            "n_boot_requested": boot["n_boot_requested"],
            "n_boot_failed": boot["n_boot_failed"],
            "seed": boot["seed"],
        },
        "note": (
            "Instrument identity is folded into 'trait' (instrument:axis), the same "
            "design analyze_w4.py task 4.3 uses for the two-instrument headline figure; "
            "this does not separately estimate an instrument-level term distinct from "
            "axis-within-instrument. The model:trait cells are unbalanced (19 models on "
            "Political Compass/8Values, 18 on SapplyValues/Pew) because "
            "mistralai/mistral-large-2512 was withdrawn from OpenRouter partway through "
            "collection; see panel.models_missing_from_at_least_one_instrument."
        ),
    }

    out = Path(REPO / out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "f3_variance_components.json").write_text(json.dumps(result, indent=2, default=str))

    print(f"instruments: {result['instruments']}")
    print(f"traits ({len(result['traits'])}): {result['traits']}")
    print(f"models common to all four instruments: {result['panel']['n_models_common_to_all_four']}")
    if result["panel"]["models_missing_from_at_least_one_instrument"]:
        print("  missing from at least one instrument: "
              f"{result['panel']['models_missing_from_at_least_one_instrument']}")
    print(f"\nmethod: {decomposition['method']}")
    for k, v in sorted(decomposition["variance_share"].items(), key=lambda kv: -kv[1]):
        ci = result["bootstrap_ci"]["variance_share_ci"].get(k, {})
        lo, hi = ci.get("ci_lower"), ci.get("ci_upper")
        ci_str = f"[{lo*100:.1f}%, {hi*100:.1f}%]" if lo is not None else "n/a"
        print(f"  {k:14s} {v*100:6.2f}%   95% CI {ci_str}   (var={decomposition['variance_components'][k]:.4f})")
    print(f"\nwrote {out / 'f3_variance_components.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/f3_variance_components")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--instruments", nargs="+",
                    default=["political_compass", "8values", "sapplyvalues", "pew_typology"],
                    help="subset of instruments to pool, for diagnosing which addition "
                         "moves the decomposition (e.g. --instruments political_compass "
                         "8values sapplyvalues to check SapplyValues alone)")
    args = ap.parse_args()
    main(args.out, args.n_boot, tuple(args.instruments))
