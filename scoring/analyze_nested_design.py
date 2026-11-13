"""Nested-design variance decomposition and robustness checks (Plan B, item 2a-2c).

The earlier decomposition (analyze_f3_variance_components.py) folds instrument into
"trait" (instrument:axis), so its model x trait term cannot say whether the
reordering of models comes from the *instrument* or merely from different *axes*
being different constructs. This script fits the nested design

    z_ijkt = mu + model_i + model x instrument_ij + model x axis(instrument)_ijk + e

where axis is nested in instrument, and reports every share with a cluster-bootstrap
CI. It also gives:
  * provider-clustered bootstrap CIs (19 models come from 11 providers; models from
    one provider are not independent draws),
  * leave-one-instrument-out and leave-one-provider-out point estimates,
  * the same quantities for the original (model, model x trait) design, so the
    two are directly comparable.

Bootstrap resamples relabel repeated clusters as distinct units (a cluster drawn
twice contributes two independent random-effect levels), which is the correct
treatment for random-effect models; the earlier script merges duplicates under one
label, so CIs for the original design may differ slightly from the manuscript's
headline intervals. Both are reported.

    python3 scoring/analyze_nested_design.py [--n-boot 1000] [--workers N]  (workers default to available cores - 1)
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the bootstrap parallelizes across resamples, and letting every
# worker also spawn BLAS threads oversubscribes the CPUs (load average >> core count).
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
from variance_components import variance_decomposition  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SEED = 20260919

DESIGNS = {
    "original": ["model", "model:trait"],
    "nested": ["model", "model:instrument", "model:trait"],
}


def default_workers() -> int:
    """Use the cores actually available to this process (respects affinity / cgroup limits where
    the OS reports them), leaving one free for the parent and the rest of the system."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        n = os.cpu_count() or 1
    return max(1, n - 1)


def provider_of(model: str) -> str:
    return model.split("/")[0]


def fit(df: pd.DataFrame, design: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = variance_decomposition(df, value="value", random_effects=DESIGNS[design],
                                   z_within="trait")
    return {"share": r["variance_share"], "method": r["method"]}


def resample_clusters(df: pd.DataFrame, cluster: str, rng) -> pd.DataFrame:
    ids = df[cluster].unique()
    picked = rng.choice(ids, size=len(ids), replace=True)
    parts = []
    for j, cid in enumerate(picked):
        part = df[df[cluster] == cid].copy()
        # distinct model labels for repeated draws, and for every model of a drawn provider
        part["model"] = part["model"] + f"#{j}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _boot_one(args):
    df, design, cluster, seed = args
    rng = np.random.default_rng(seed)
    try:
        return fit(resample_clusters(df, cluster, rng), design)["share"]
    except Exception:
        return None


def bootstrap(df, design, cluster, n_boot, workers, tag):
    seeds = np.random.SeedSequence(SEED).spawn(n_boot)
    jobs = [(df, design, cluster, int(s.generate_state(1)[0])) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_boot_one, jobs, chunksize=4))
    ok = [r for r in res if r]
    keys = ok[0].keys() if ok else []
    ci = {}
    for k in keys:
        arr = np.array([r.get(k, np.nan) for r in ok], float)
        arr = arr[np.isfinite(arr)]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        ci[k] = {"ci_lower": float(lo), "ci_upper": float(hi)}
    print(f"  [{tag}] {len(ok)}/{n_boot} resamples ok", flush=True)
    return {"cluster": cluster, "n_boot": n_boot, "n_ok": len(ok), "ci": ci}


def main(n_boot: int, workers: int, out_dir: str = "results/nested_design") -> None:
    two = load_pooled_long(("political_compass", "8values"))
    four = load_pooled_long()
    for d in (two, four):
        d["provider"] = d["model"].map(provider_of)

    result = {"seed": SEED, "n_boot": n_boot, "n_providers": int(two.provider.nunique())}
    result["note"] = (
        "Pew Political Typology has one axis, so for Pew the model x instrument and "
        "model x axis(instrument) terms coincide; in the four-instrument panel they are "
        "separately identified only through the other three instruments."
    )

    panels = {"two_instruments": two, "four_instruments": four}
    for pname, df in panels.items():
        result[pname] = {}
        for design in DESIGNS:
            print(f"{pname} / {design}", flush=True)
            entry = {"point": fit(df, design)}
            entry["model_bootstrap"] = bootstrap(df, design, "model", n_boot, workers,
                                                 f"{pname}/{design}/model")
            entry["provider_bootstrap"] = bootstrap(df, design, "provider", n_boot, workers,
                                                    f"{pname}/{design}/provider")
            result[pname][design] = entry

    # Leave-one-instrument-out (from the four-instrument panel) and leave-one-provider-out.
    loo_inst, loo_prov = {}, {}
    for inst in sorted(four.instrument.unique()):
        sub = four[four.instrument != inst]
        loo_inst[inst] = {d: fit(sub, d)["share"] for d in DESIGNS}
        print("LOIO", inst, loo_inst[inst]["original"], flush=True)
    for prov in sorted(two.provider.unique()):
        sub = four[four.provider != prov]
        loo_prov[prov] = {"n_models": int(sub.model.nunique()),
                          **{d: fit(sub, d)["share"] for d in DESIGNS}}
        print("LOPO", prov, loo_prov[prov]["original"], flush=True)
    result["leave_one_instrument_out_from_four"] = loo_inst
    result["leave_one_provider_out_four_instruments"] = loo_prov

    out = REPO / out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "nested_design.json").write_text(json.dumps(result, indent=2))
    print("wrote", out / "nested_design.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: available cores minus one)")
    a = ap.parse_args()
    main(a.n_boot, a.workers or default_workers())
