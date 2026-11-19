"""Reliability-corrected convergent and discriminant correlations (answers the range-restriction / attenuation objection).

For each trait the reliability of a model's mean over its k repeated administrations is ICC(k) = (MSB - MSW) / MSB from a
one-way ANOVA over models. Across-model correlations are disattenuated, r* = r / sqrt(rel_x * rel_y), and bootstrapped over models
(reliabilities recomputed within each resample). Also reports the correlation between model means in two random halves of the
trials as a direct check. Writes results/reliability_corrected/reliability_corrected.json.   python analyze_reliability_corrected.py [n_boot]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_f3_variance_components import load_pooled_long
from analyze_se_supplement import CONVERGENT, DISCRIMINANT, PC_E, PC_S, EQ, PR

SEED = 20261119
TRAITS = [PC_E, PC_S, EQ, PR]


def icc_k(wide_trials):
    """wide_trials: models x trials array. ICC(k) of the model mean."""
    n, k = wide_trials.shape
    mb = k * wide_trials.mean(1).var(ddof=1)
    mw = ((wide_trials - wide_trials.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float(np.clip((mb - mw) / mb, 0.0, 1.0)) if mb > 0 else 0.0


def stats_from(arr):  # arr: dict trait -> models x trials
    means = {t: a.mean(1) for t, a in arr.items()}
    rel = {t: icc_k(a) for t, a in arr.items()}
    def r(a, b, corrected):
        x = np.corrcoef(means[a], means[b])[0, 1]
        return x / np.sqrt(rel[a] * rel[b]) if corrected else x
    out = {}
    for corrected in (False, True):
        conv = [r(a, b, corrected) for a, b in CONVERGENT]; disc = [r(a, b, corrected) for a, b in DISCRIMINANT]
        out["corrected" if corrected else "raw"] = dict(conv=conv, disc=disc, conv_mean=float(np.mean(conv)), disc_mean=float(np.mean(disc)),
                                                         diff=float(np.mean(conv) - np.mean(disc)))
    return out, rel


def main(n_boot=2000):
    df = load_pooled_long(("political_compass", "8values"))
    models = sorted(df.model.unique())
    arr = {}
    for t in TRAITS:
        w = df[df.trait == t].pivot_table(index="model", columns="trial", values="value").loc[models]
        arr[t] = w.dropna(axis=1).to_numpy()
    k = {t: arr[t].shape[1] for t in TRAITS}
    point, rel = stats_from(arr)
    rng = np.random.default_rng(SEED); draws = {"raw": [], "corrected": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(models), len(models))
        try:
            s, _r = stats_from({t: a[idx] for t, a in arr.items()})
        except Exception:
            continue
        for c in draws:
            v = [s[c]["conv_mean"], s[c]["disc_mean"], s[c]["diff"], *s[c]["conv"]]
            if np.all(np.isfinite(v)): draws[c].append(v)
    res = dict(seed=SEED, n_boot=n_boot, n_models=len(models), trials_per_trait=k, reliability_of_model_means=rel, point=point, ci={})
    for c, d in draws.items():
        d = np.array(d); q = np.percentile(d, [2.5, 97.5], axis=0)
        res["ci"][c] = {n: [float(q[0, i]), float(q[1, i])] for i, n in enumerate(["conv_mean", "disc_mean", "diff", "conv_pair_economic", "conv_pair_social"])}
    # split-half check
    sh = []
    for _ in range(500):
        rs = {}
        for t in TRAITS:
            perm = rng.permutation(arr[t].shape[1]); h = arr[t].shape[1] // 2
            rs[t] = (arr[t][:, perm[:h]].mean(1), arr[t][:, perm[h:2 * h]].mean(1))
        sh.append({t: np.corrcoef(*rs[t])[0, 1] for t in TRAITS})
    res["split_half_r_of_model_means"] = {t: float(np.mean([s[t] for s in sh])) for t in TRAITS}
    out = Path(__file__).resolve().parent.parent / "results" / "reliability_corrected"; out.mkdir(parents=True, exist_ok=True)
    (out / "reliability_corrected.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: res[k_] for k_ in ("reliability_of_model_means", "split_half_r_of_model_means", "point", "ci")}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
