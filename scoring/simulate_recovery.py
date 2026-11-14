"""Recovery simulation for the transferable (model) share in the nested design.

Question: with this few models and instruments, could the nested REML estimate miss a transferable
share that is really there, or produce one that is not?

For each panel (two instruments, four instruments) the simulation keeps the real design exactly: the
same models, instruments, axes and number of trials in every cell as the archived data. It then draws
scores from Equation (1) with a known transferable share (0, 2, 5, 10, 20, 30 per cent), splitting the
remainder between model x instrument, model x axis(instrument) and trial noise in the proportions
estimated from the real data (results/nested_design/nested_design.json), and refits the same model with
the same code (variance_components.variance_decomposition, z-scored within trait).

Reported per panel and true share: mean, median and 5th-95th percentiles of the estimate, the share of
simulations with an estimate below 1 per cent, and the share with an estimate at or below (and at or
above) the value observed in the real data.

Offline. Run after analyze_nested_design.py.

    python3 simulate_recovery.py --reps 200
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_f3_variance_components import load_pooled_long  # noqa: E402
from analyze_nested_design import DESIGNS, default_workers  # noqa: E402
from variance_components import variance_decomposition  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SEED = 20261114
TRUTHS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30)
_SKELETON: dict = {}


def skeleton(panel: str) -> pd.DataFrame:
    """The real design (one row per model x instrument x axis x trial), without the scores."""
    if panel not in _SKELETON:
        df = load_pooled_long() if panel == "four_instruments" else load_pooled_long(("political_compass", "8values"))
        _SKELETON[panel] = df[["model", "instrument", "trait"]].reset_index(drop=True)
    return _SKELETON[panel]


def simulate_once(job) -> float:
    panel, truth, rest, seed = job
    rng = np.random.default_rng(seed)
    sk = skeleton(panel)
    k = (1.0 - truth) / sum(rest)
    s_mi, s_ma, s_e = (np.sqrt(x * k) for x in rest)
    m_code, m_n = pd.factorize(sk["model"])
    mi_code, mi_n = pd.factorize(sk["model"] + "|" + sk["instrument"])
    ma_code, ma_n = pd.factorize(sk["model"] + "|" + sk["trait"])
    y = (rng.normal(0, np.sqrt(truth), len(m_n))[m_code]
         + rng.normal(0, s_mi, len(mi_n))[mi_code]
         + rng.normal(0, s_ma, len(ma_n))[ma_code]
         + rng.normal(0, s_e, len(sk)))
    d = sk.assign(value=y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = variance_decomposition(d, value="value", random_effects=DESIGNS["nested"], z_within="trait")
    return float(r["variance_share"]["model"])


def main(reps: int, workers: int, out_dir: str = "results/recovery_simulation") -> None:
    nested = json.loads((REPO / "results" / "nested_design" / "nested_design.json").read_text())
    jobs, meta = [], {}
    for p_i, panel in enumerate(("two_instruments", "four_instruments")):
        share = nested[panel]["nested"]["point"]["share"]
        rest = (share["model:instrument"], share["model:trait"], share["residual"])
        meta[panel] = {"observed_model_share": share["model"],
                       "remainder_split_model_instrument_axis_noise": [x / sum(rest) for x in rest]}
        for t_i, truth in enumerate(TRUTHS):
            for r in range(reps):
                jobs.append((panel, truth, rest, SEED + 1_000_000 * p_i + 10_000 * t_i + r))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        est = list(ex.map(simulate_once, jobs, chunksize=2))
    df = pd.DataFrame({"panel": [j[0] for j in jobs], "truth": [j[1] for j in jobs], "estimate": est})

    res = {"seed": SEED, "reps": reps, "true_shares": list(TRUTHS), "panels": {}}
    for panel, g in df.groupby("panel", sort=False):
        obs = meta[panel]["observed_model_share"]
        rows = {}
        for truth, e in g.groupby("truth")["estimate"]:
            rows[f"{truth:.2f}"] = {
                "mean": float(e.mean()), "median": float(e.median()),
                "q05": float(e.quantile(0.05)), "q95": float(e.quantile(0.95)),
                "share_below_1pct": float((e < 0.01).mean()),
                "share_at_or_below_observed": float((e <= obs).mean()),
                "share_at_or_above_observed": float((e >= obs).mean()),
            }
        res["panels"][panel] = {**meta[panel], "by_true_share": rows}

    out = REPO / out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "recovery_simulation.json").write_text(json.dumps(res, indent=1))
    df.to_csv(out / "recovery_draws.csv", index=False)
    for panel, p in res["panels"].items():
        print(f"{panel}: observed model share {100 * p['observed_model_share']:.1f}%")
        for truth, r in p["by_true_share"].items():
            print(f"  true {100 * float(truth):4.0f}%  mean {100 * r['mean']:5.1f}  median {100 * r['median']:5.1f}  "
                  f"5-95% [{100 * r['q05']:.1f}, {100 * r['q95']:.1f}]  <1%: {r['share_below_1pct']:.2f}  "
                  f"<=obs: {r['share_at_or_below_observed']:.2f}  >=obs: {r['share_at_or_above_observed']:.2f}")
    print("wrote", out / "recovery_simulation.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.reps, a.workers or default_workers())
