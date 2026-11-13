"""Temperature sensitivity of the 8Values variance decomposition (Plan B 2b).

Compares temperature 0, 0.7 (first 20 trials of the main run, same 8 models) and 1.0
(data/temperature_sweep/, collected by collect_temperature.py). Per temperature: variance shares from
the model / model x axis / residual model on the four 8Values axes (z-scored within axis), with a
cluster bootstrap over the 8 models, plus the within-axis share of variance due to model identity and the
rank agreement of model means across temperatures. Small design (8 models, one instrument): intervals are wide.

    python3 scoring/analyze_temperature_sweep.py [--n-boot 1000]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from variance_components import variance_decomposition, _cluster_resample  # noqa: E402

# The swept models, read from the collected files so this script needs no API key.
MODELS = sorted({json.loads(p.read_text())["model"]
                 for p in (Path(__file__).resolve().parent.parent / "data/temperature_sweep/T0.0").glob("*.json")})

REPO = Path(__file__).resolve().parent.parent
SEED = 20260919


def load(temp):
    rows = []
    if temp == 0.7:
        files = [p for m in MODELS for p in (REPO / "data/raw_trials").glob(f"{m.replace('/', '_')}__8values__trial*.json")]
    else:
        files = list((REPO / f"data/temperature_sweep/T{temp}").glob("*.json"))
    for p in files:
        r = json.loads(p.read_text())
        if r.get("status") != "ok" or r["model"] not in MODELS or int(r["trial"]) > 20:
            continue
        for ax, v in r["scores"].items():
            rows.append({"model": r["model"], "trial": int(r["trial"]), "axis": ax, "trait": "8values:" + ax, "value": float(v)})
    return pd.DataFrame(rows)


def shares(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pooled = variance_decomposition(df, "value", ["model", "model:trait"], z_within="trait")["variance_share"]
        within = {}
        for ax, g in df.groupby("axis"):
            within[ax] = variance_decomposition(g, "value", ["model"], z_within="trait")["variance_share"]["model"]
    return pooled, within


def main(n_boot):
    rng = np.random.default_rng(SEED)
    out = {"models": MODELS, "n_trials_per_cell": 20, "n_boot": n_boot, "temperatures": {}}
    data = {t: load(t) for t in (0.0, 0.7, 1.0)}
    for t, df in data.items():
        pooled, within = shares(df)
        draws = {k: [] for k in pooled}; wdraws = []
        ids = df.model.unique()
        for _ in range(n_boot):
            rs = _cluster_resample(df, "model", ids, rng)
            try:
                p, w = shares(rs)
            except Exception:
                continue
            for k, v in p.items(): draws[k].append(v)
            wdraws.append(np.mean(list(w.values())))
        ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]
        out["temperatures"][str(t)] = {
            "n_administrations": int(df.groupby(["model", "trial"]).ngroups),
            "pooled_share": {k: float(v) for k, v in pooled.items()},
            "pooled_share_ci": {k: ci(v) for k, v in draws.items()},
            "within_axis_model_share_mean": float(np.mean(list(within.values()))),
            "within_axis_model_share_mean_ci": ci(wdraws),
            "within_axis_model_share_by_axis": within,
        }
    # rank agreement of model means across temperatures, per axis then averaged
    means = {t: d.groupby(["model", "axis"])["value"].mean().unstack() for t, d in data.items()}
    agree = {}
    for a, b in ((0.0, 0.7), (0.7, 1.0), (0.0, 1.0)):
        rhos = [spearmanr(means[a][ax], means[b][ax])[0] for ax in means[a].columns]
        agree[f"{a}_vs_{b}"] = {"mean_spearman_over_axes": float(np.mean(rhos)), "by_axis": dict(zip(means[a].columns, map(float, rhos)))}
    out["rank_agreement_of_model_means"] = agree
    out["mean_axis_score_by_temperature"] = {str(t): {ax: float(v) for ax, v in m.mean().items()} for t, m in means.items()}
    (REPO / "results").joinpath("temperature_sweep").mkdir(exist_ok=True)
    (REPO / "results/temperature_sweep/temperature_sweep.json").write_text(json.dumps(out, indent=2))
    for t, r in out["temperatures"].items():
        print(t, {k: round(v * 100, 1) for k, v in r["pooled_share"].items()}, "within-axis model share", round(r["within_axis_model_share_mean"] * 100, 1), [round(x * 100, 1) for x in r["within_axis_model_share_mean_ci"]])
    print(json.dumps(agree, indent=1)[:600])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=1000)
    main(ap.parse_args().n_boot)
