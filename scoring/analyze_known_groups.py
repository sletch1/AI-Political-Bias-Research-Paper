"""Known-groups (positive-control) check on the asker-identity factorial.

The main design finds that little of a model's default position transfers across instruments. One
reading is that the instruments cannot register the same thing at all. This script tests that reading
with a contrast whose direction is known in advance: the same model asked by a self-described
progressive Democrat and by a self-described conservative Republican (data/asker_identity/, both
original instruments, 10 trials per model x instrument x condition).

All scores are direction-aligned first (variance_components.orient: high = left / libertarian /
progressive), so a positive gap means the model answers further left for the progressive asker.

Reported, each with a bootstrap over models:
  * the progressive-minus-conservative gap per axis, in instrument units, as a share of the
    instrument's range, and in units of the between-model SD of the no-asker condition;
  * the share of model x axis cells whose gap is positive;
  * the correlation across models of the gap on one instrument with the gap on the other
    (the two construct pairs used in the multitrait-multimethod matrix, and all eight
    cross-instrument axis pairs), next to the same correlations for the no-asker positions.

Offline: reads archived trials only.

    python3 analyze_known_groups.py --n-boot 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_se_supplement import CONVERGENT, boot_stat, mean_r, summarize  # noqa: E402
from variance_components import orient  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ASKER_DIR = REPO / "data" / "asker_identity"
SEED = 20261114
RANGE = {"political_compass": 20.0, "8values": 100.0}


def load_asker() -> pd.DataFrame:
    rows = []
    for path in sorted(ASKER_DIR.glob("*__trial*.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok" or not rec.get("scores"):
            continue
        for axis, value in rec["scores"].items():
            rows.append({"model": rec["model"], "instrument": rec["test"],
                         "condition": rec["condition"], "trial": int(rec["trial"]),
                         "axis": axis, "value": float(value)})
    df = pd.DataFrame(rows)
    df["trait"] = df["instrument"] + ":" + df["axis"]
    return orient(df)


def main(n_boot: int, out_dir: str = "results/known_groups") -> None:
    rng = np.random.default_rng(SEED)
    df = load_asker()
    cell = df.groupby(["model", "trait", "condition"])["value"].mean().unstack("condition")
    gap = (cell["progressive"] - cell["conservative"]).unstack("trait").dropna()
    base = cell["none"].unstack("trait").loc[gap.index]
    traits = list(gap.columns)
    ev = [t for t in traits if t.startswith("8values:")]
    pc = [t for t in traits if t.startswith("political_compass:")]
    cross = list(product(ev, pc))

    res: dict = {"seed": SEED, "n_boot": n_boot, "n_models": int(len(gap)),
                 "direction": "high = left / libertarian / progressive on every axis",
                 "contrast": "progressive-asker mean minus conservative-asker mean, per model and axis"}

    per_axis = {}
    for t in traits:
        rng_units = RANGE[t.split(":")[0]]
        sd0 = float(base[t].std(ddof=1))
        per_axis[t] = {
            "mean_gap": summarize(float(gap[t].mean()),
                                  boot_stat(gap[[t]], lambda m, t=t: m[t].mean(), n_boot, rng)),
            "mean_gap_share_of_range": float(gap[t].mean() / rng_units),
            "between_model_sd_no_asker": sd0,
            "mean_gap_in_between_model_sd": float(gap[t].mean() / sd0),
            "min_gap": float(gap[t].min()),
            "share_of_models_positive": float((gap[t] > 0).mean()),
        }
    res["per_axis"] = per_axis
    res["share_of_cells_positive"] = float((gap.to_numpy() > 0).mean())
    res["n_cells"] = int(gap.size)
    ratios = [per_axis[t]["mean_gap_in_between_model_sd"] for t in traits]
    res["gap_in_between_model_sd_range"] = [float(min(ratios)), float(max(ratios))]

    def block(m: pd.DataFrame) -> dict:
        return {
            "convergent_pairs": {f"{a} vs {b}": float(np.corrcoef(m[a], m[b])[0, 1]) for a, b in CONVERGENT},
            "mean_convergent": summarize(mean_r(m, CONVERGENT),
                                         boot_stat(m, lambda x: mean_r(x, CONVERGENT), n_boot, rng)),
            "mean_all_cross_instrument": summarize(mean_r(m, cross),
                                                   boot_stat(m, lambda x: mean_r(x, cross), n_boot, rng)),
            "cross_instrument_range": [float(min(np.corrcoef(m[a], m[b])[0, 1] for a, b in cross)),
                                       float(max(np.corrcoef(m[a], m[b])[0, 1] for a, b in cross))],
        }

    res["asker_shift_correlations"] = block(gap)
    res["no_asker_position_correlations"] = block(base)
    diff = boot_stat(pd.concat([gap.add_prefix("g|"), base.add_prefix("b|")], axis=1),
                     lambda x: mean_r(x, [("g|" + a, "g|" + b) for a, b in CONVERGENT])
                     - mean_r(x, [("b|" + a, "b|" + b) for a, b in CONVERGENT]), n_boot, rng)
    res["convergent_shift_minus_position"] = summarize(
        mean_r(gap, CONVERGENT) - mean_r(base, CONVERGENT), diff)

    out = REPO / out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "known_groups.json").write_text(json.dumps(res, indent=1))
    print(f"{res['n_models']} models, {res['n_cells']} model x axis cells; "
          f"share with a positive gap {res['share_of_cells_positive']:.3f}")
    for t in traits:
        a = per_axis[t]
        print(f"  {t:28s} gap {a['mean_gap']['estimate']:+7.2f} "
              f"[{a['mean_gap']['ci_lower']:+.2f}, {a['mean_gap']['ci_upper']:+.2f}]  "
              f"{100 * a['mean_gap_share_of_range']:.1f}% of range, "
              f"{a['mean_gap_in_between_model_sd']:.1f} between-model SD")
    for name in ("asker_shift_correlations", "no_asker_position_correlations"):
        b = res[name]
        print(name, "convergent", {k: round(v, 3) for k, v in b["convergent_pairs"].items()},
              "mean", round(b["mean_convergent"]["estimate"], 3),
              [round(b["mean_convergent"]["ci_lower"], 2), round(b["mean_convergent"]["ci_upper"], 2)],
              "| all cross", round(b["mean_all_cross_instrument"]["estimate"], 3),
              [round(x, 2) for x in b["cross_instrument_range"]])
    d = res["convergent_shift_minus_position"]
    print("shift minus position (mean convergent r):", round(d["estimate"], 3),
          [round(d["ci_lower"], 2), round(d["ci_upper"], 2)])
    print("wrote", out / "known_groups.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=5000)
    main(ap.parse_args().n_boot)
