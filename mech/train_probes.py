"""Task 3.1, steps 3-4 -- ridge probes, and the generalisation test that matters.

    python3 mech/train_probes.py --model meta-llama/Llama-3.1-8B-Instruct

Fits a ridge probe per layer to predict DW-NOMINATE from activations, reports
held-out R-squared by layer, and -- the part Gate C actually turns on -- applies
the legislator-trained probe to activations from news-outlet prompts with
published slant scores.

Why the split is grouped by legislator, not random: a random split puts the same
legislator's ten topic prompts on both sides of it, and the probe then scores
well by recognising the person. Grouping by legislator forces it to generalise
across people, which is the weakest form of the claim that still means anything.

Writes results/w3/probes__{model_slug}.json and the fitted probe weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_activations import slug  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ACT_DIR = REPO / "data" / "activations"
OUT_DIR = REPO / "results" / "w3"

# the retired NMI plan Gate C: held-out R-squared above this, plus off-distribution
# generalisation, or the arm is cut cleanly rather than reported weakly.
GATE_C_R2 = 0.4


def load(model: str, source: str):
    path = ACT_DIR / f"{slug(model)}__{source}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run extract_activations.py --source {source}")
    d = np.load(path)
    meta = json.loads((ACT_DIR / f"{slug(model)}__{source}.meta.json").read_text())
    return d["X"].astype("float32"), d["y"].astype("float32"), meta


def grouped_folds(groups, n_folds: int = 5, seed: int = 20260909):
    """Fold assignment by group, so no group spans the split."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(groups)))
    rng.shuffle(unique)
    fold_of = {g: i % n_folds for i, g in enumerate(unique)}
    return np.array([fold_of[g] for g in groups])


def fit_layer(X, y, folds, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0)):
    """Ridge with alpha chosen inside each training fold, never on the test fold."""
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler

    preds = np.zeros_like(y)
    for f in sorted(set(folds)):
        train, test = folds != f, folds == f
        scaler = StandardScaler().fit(X[train])
        inner = RidgeCV(alphas=alphas).fit(scaler.transform(X[train]), y[train])
        model = Ridge(alpha=inner.alpha_).fit(scaler.transform(X[train]), y[train])
        preds[test] = model.predict(scaler.transform(X[test]))
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    r = float(np.corrcoef(preds, y)[0, 1]) if np.std(preds) > 0 else float("nan")

    scaler = StandardScaler().fit(X)
    full = RidgeCV(alphas=alphas).fit(scaler.transform(X), y)
    return {"r2_heldout": r2, "pearson_r_heldout": r,
            "alpha": float(full.alpha_)}, (scaler, full), preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-generalisation", action="store_true")
    args = ap.parse_args()

    X, y, meta = load(args.model, "legislators")
    groups = [m["name"] for m in meta]
    folds = grouped_folds(groups, args.folds)
    n_layers = X.shape[1]
    print(f"{args.model}: {X.shape[0]} examples, {n_layers} layers, "
          f"{len(set(groups))} legislators, {args.folds} grouped folds")

    per_layer, fitted = [], {}
    for layer in range(n_layers):
        stats, model, _ = fit_layer(X[:, layer, :], y, folds)
        per_layer.append({"layer": layer, **stats})
        fitted[layer] = model
        print(f"  layer {layer:3d}  R2 = {stats['r2_heldout']:+.3f}  "
              f"r = {stats['pearson_r_heldout']:+.3f}")

    best = max(per_layer, key=lambda p: p["r2_heldout"])
    results = {"model": args.model, "n_examples": int(X.shape[0]),
               "n_legislators": len(set(groups)), "per_layer": per_layer,
               "best_layer": best, "gate_c_r2_threshold": GATE_C_R2}

    if not args.skip_generalisation:
        try:
            Xn, yn, nmeta = load(args.model, "news")
        except SystemExit as exc:
            results["generalisation"] = {"available": False, "reason": str(exc)}
        else:
            scaler, ridge = fitted[best["layer"]]
            pred = ridge.predict(scaler.transform(Xn[:, best["layer"], :]))
            r = float(np.corrcoef(pred, yn)[0, 1])
            ss_res = float(((yn - pred) ** 2).sum())
            ss_tot = float(((yn - yn.mean()) ** 2).sum())
            results["generalisation"] = {
                "available": True, "layer": best["layer"], "n_examples": int(len(yn)),
                "pearson_r": r, "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
                "note": ("Predictions are on the legislator scale and the targets on the "
                         "outlet scale, so the correlation is the meaningful statistic; "
                         "R-squared is reported but will be poor by construction unless "
                         "the two scales happen to align."),
                "per_outlet": [{"outlet": m["outlet"], "topic": m["topic"],
                                "predicted": float(p), "slant": float(t)}
                               for m, p, t in zip(nmeta, pred, yn)],
            }

    gen = results.get("generalisation", {})
    passes = best["r2_heldout"] > GATE_C_R2 and (
        gen.get("available") and abs(gen.get("pearson_r", 0)) > 0.5)
    results["gate_c"] = {
        "passes": bool(passes),
        "verdict": ("INCLUDE the mechanistic arm; submit to NMI." if passes else
                    "CUT the mechanistic arm cleanly and submit to rung 2-3. the retired NMI plan Gate C: "
                    "a half-working probe analysis is worse than none, because it invites a "
                    "methods reviewer with a specific objection."),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"probes__{slug(args.model)}.json"
    path.write_text(json.dumps(results, indent=2))
    np.savez_compressed(OUT_DIR / f"probe_weights__{slug(args.model)}.npz",
                        **{f"layer_{l}_coef": fitted[l][1].coef_ for l in fitted},
                        **{f"layer_{l}_mean": fitted[l][0].mean_ for l in fitted},
                        **{f"layer_{l}_scale": fitted[l][0].scale_ for l in fitted})
    print(f"\nbest layer {best['layer']}: held-out R2 = {best['r2_heldout']:+.3f}")
    if gen.get("available"):
        print(f"off-distribution (news slant): r = {gen['pearson_r']:+.3f}")
    print(f"GATE C: {results['gate_c']['verdict']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
