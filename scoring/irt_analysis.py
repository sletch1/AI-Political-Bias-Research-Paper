"""Workstream 1, Task 1.4 -- item-response-theory re-analysis of existing data.

Zero API calls: this runs entirely on the 2,280 administrations already in
data/raw_trials/. updates/03_experiments.md Task 1.4 covers writing it up. It is the highest value-per-dollar task in the
plan, and it answers two distinct reviewer objections:

  * Wachter et al.: the published scoring treats every item as equally
    informative. A graded response model estimates each item's discrimination
    instead, and shows how concentrated the measurement really is.
  * Wachter et al. again: a model that avoids political items and a model that
    sits at the centre of the scale score identically under the published
    formula. The two-stage avoidance model separates them.

    python3 scoring/irt_analysis.py [--out results/w1]

Design decisions worth knowing before reading the numbers
---------------------------------------------------------
*One unidimensional GRM per axis*, not one per instrument. Both instruments are
multidimensional by construction, and a single latent trait fitted across all
items would be a nuisance factor, not an ideology estimate.

*Item-to-axis assignment differs by instrument, of necessity.* 8Values
publishes its per-item axis weights, so items are assigned and oriented by that
published key. The Political Compass does not publish its weights at all --
that is precisely why score_political_compass.py drives the live site rather
than reimplementing a formula -- so its items are assigned to the axis they
correlate with most strongly across trials and oriented by the sign of that
correlation. That empirical key is reported alongside the results; it is a
derived artefact of this analysis, not a claim about the site's internals.

*Respondent = one model-trial.* The 60 trials per model per instrument are what
make an IRT fit possible at all here: 1,140 respondents per instrument, which
is a normal-sized calibration sample. A design with 1-5 trials per model, as
most of this literature uses, could not support this analysis.

*Estimation is marginal maximum likelihood with an EM loop over Gauss-Legendre
quadrature*, implemented here rather than taken from a package. `girth`,
`py-irt` and `mirt` all bring either a heavy dependency or an R bridge for what
is, for a graded response model, about a hundred lines of scipy. Keeping it in
the repository also means the estimator is auditable by a reviewer alongside
the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
RAW_TRIALS = REPO / "data" / "raw_trials"
SCORES_CSV = REPO / "data" / "scores.csv"

# Response categories, ordered from most-disagree to most-agree, per instrument.
CATEGORIES = {
    "8values": ["SD", "D", "N", "A", "SA"],
    "political_compass": ["SD", "D", "A", "SA"],
}

# The 8Values axis whose published weights key each reported score.
EIGHTVALUES_AXES = {"equality": "econ", "peace": "dipl", "liberty": "govt", "progress": "scty"}
COMPASS_AXES = ("economic", "social")

# 8Values' only true neutral. The Political Compass is forced-choice and has
# none, so the avoidance analysis is 8Values-only and says so.
NEUTRAL = {"8values": "N"}

N_QUAD = 41
QUAD_LIMIT = 4.0
MAX_EM_ITERS = 200
EM_TOL = 1e-4

# The responses are close to deterministic within a model: 19 models answer 60
# times each and mostly answer identically. That is the paper's stability
# finding, and it is also a hard problem for an unpenalised IRT fit, because a
# Guttman-perfect item has infinite discrimination. Three guards, all standard
# and all reported:
#   * items with essentially no response variation are dropped, not fitted
#   * log-discrimination carries a weak N(0, 0.7) ridge penalty
#   * discrimination and thresholds are box-bounded
MIN_ITEM_VARIATION = 0.01        # fraction of responses that must be off the modal category
LOG_A_PRIOR_SD = 0.7
A_BOUNDS = (0.05, 8.0)
B_BOUND = 6.0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_responses() -> dict:
    """Item-level response matrices, one per instrument.

    Returns {instrument: (codes, meta)} where `codes` is an (n_respondents,
    n_items) integer array of category indices and `meta` is a DataFrame with
    the model and trial each row came from.
    """
    rows = defaultdict(list)
    for path in sorted(RAW_TRIALS.glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok" or not rec.get("answers"):
            continue
        rows[rec["test"]].append(rec)

    out = {}
    for instrument, recs in rows.items():
        cats = CATEGORIES[instrument]
        index = {c: i for i, c in enumerate(cats)}
        n_items = len(recs[0]["answers"])
        codes = np.full((len(recs), n_items), -1, dtype=int)
        for r, rec in enumerate(recs):
            answers = rec["answers"]
            if len(answers) != n_items:
                continue
            for c, a in enumerate(answers):
                codes[r, c] = index.get(str(a).strip().upper(), -1)
        meta = pd.DataFrame({"model": [r["model"] for r in recs],
                             "trial": [r["trial"] for r in recs]})
        out[instrument] = (codes, meta)
    return out


def load_axis_scores() -> pd.DataFrame:
    """Published axis scores per (model, instrument, trial), long form."""
    wide = pd.read_csv(SCORES_CSV)
    rows = []
    for _, r in wide.iterrows():
        for i in (1, 2, 3, 4):
            name, val = r.get(f"axis{i}_name"), r.get(f"axis{i}_value")
            if isinstance(name, str) and pd.notna(val):
                rows.append({"model": r["model"], "instrument": r["test"],
                             "trial": int(r["trial"]), "axis": name, "value": float(val)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# item -> axis keys
# --------------------------------------------------------------------------
def eightvalues_key() -> dict:
    """{axis: [(item_index, orientation), ...]} from the published weights."""
    from score_8values import load_questions

    bank = load_questions()
    key = {}
    for axis, effect_name in EIGHTVALUES_AXES.items():
        entries = [(i, 1 if q["effect"][effect_name] > 0 else -1)
                   for i, q in enumerate(bank) if q["effect"][effect_name] != 0]
        key[axis] = entries
    return key


def compass_key(codes: np.ndarray, meta: pd.DataFrame, axis_scores: pd.DataFrame) -> dict:
    """Empirical item-to-axis key for the Political Compass.

    The site does not publish its weights, so each item is assigned to the axis
    its responses correlate with most strongly across the 1,140 administrations
    and oriented by the sign of that correlation. Items whose strongest
    correlation is below `min_r` are dropped rather than forced onto an axis
    they do not load on.
    """
    min_r = 0.10
    scores = {}
    for axis in COMPASS_AXES:
        sub = axis_scores[(axis_scores.instrument == "political_compass")
                          & (axis_scores.axis == axis)]
        lookup = {(r.model, r.trial): r.value for r in sub.itertuples()}
        scores[axis] = np.array([lookup.get((m, t), np.nan)
                                 for m, t in zip(meta.model, meta.trial)])

    key = {axis: [] for axis in COMPASS_AXES}
    diagnostics = []
    for item in range(codes.shape[1]):
        col = codes[:, item].astype(float)
        col[codes[:, item] < 0] = np.nan
        best, best_r = None, 0.0
        per_axis = {}
        for axis in COMPASS_AXES:
            ok = ~np.isnan(col) & ~np.isnan(scores[axis])
            if ok.sum() < 30 or np.std(col[ok]) == 0:
                per_axis[axis] = 0.0
                continue
            r = float(np.corrcoef(col[ok], scores[axis][ok])[0, 1])
            per_axis[axis] = 0.0 if np.isnan(r) else r
            if abs(per_axis[axis]) > abs(best_r):
                best, best_r = axis, per_axis[axis]
        diagnostics.append({"item": item + 1, "assigned": best if abs(best_r) >= min_r else None,
                            "r": round(best_r, 4),
                            **{f"r_{a}": round(v, 4) for a, v in per_axis.items()}})
        if best is not None and abs(best_r) >= min_r:
            key[best].append((item, 1 if best_r > 0 else -1))
    return key, diagnostics


# --------------------------------------------------------------------------
# graded response model
# --------------------------------------------------------------------------
def _thresholds(free: np.ndarray) -> np.ndarray:
    """Monotone-increasing thresholds from an unconstrained vector.

    b_1 free, then each subsequent threshold is the previous plus a positive
    increment. Reparameterising rather than constraining keeps the M-step an
    unconstrained optimisation, which is both faster and far less likely to
    stall on a boundary.
    """
    # Clip before exponentiating: an unbounded increment overflows on the
    # near-Guttman items this dataset is full of, and a threshold beyond the
    # quadrature range is meaningless anyway.
    increments = np.exp(np.clip(free[1:], -12.0, np.log(2 * B_BOUND)))
    first = np.clip(free[:1], -B_BOUND, B_BOUND)
    return np.concatenate([first, first + np.cumsum(increments)])


def _category_probs(a: float, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(n_theta, n_categories) response probabilities under Samejima's GRM."""
    cum = expit(a * (theta[:, None] - b[None, :]))          # P(X >= k), k = 1..K-1
    ones = np.ones((len(theta), 1))
    zeros = np.zeros((len(theta), 1))
    upper = np.hstack([ones, cum])
    lower = np.hstack([cum, zeros])
    return np.clip(upper - lower, 1e-12, 1.0)


def fit_grm(codes: np.ndarray, n_cat: int, max_iters: int = MAX_EM_ITERS,
            tol: float = EM_TOL) -> dict:
    """Marginal-ML graded response model via EM over quadrature.

    `codes` is (n_respondents, n_items) with category indices in 0..n_cat-1 and
    -1 for missing. Returns discriminations, thresholds, EAP thetas and fit
    diagnostics.
    """
    n_resp, n_items = codes.shape
    nodes, weights = np.polynomial.legendre.leggauss(N_QUAD)
    theta = nodes * QUAD_LIMIT
    prior = np.exp(-0.5 * theta ** 2) * weights * QUAD_LIMIT
    prior /= prior.sum()

    a = np.ones(n_items)
    # Start thresholds at the empirical category boundaries, which puts the EM
    # loop near the answer and avoids the flat region a zero start sits in.
    b_free = np.zeros((n_items, n_cat - 1))
    for j in range(n_items):
        obs = codes[:, j][codes[:, j] >= 0]
        if len(obs) == 0:
            b_free[j] = np.linspace(-1, 1, n_cat - 1)[0], *np.zeros(n_cat - 2)
            continue
        cum = np.cumsum(np.bincount(obs, minlength=n_cat)) / len(obs)
        q = np.clip(cum[:-1], 0.02, 0.98)
        start = np.log(q / (1 - q))          # crude threshold estimates
        start = np.sort(start)
        gaps = np.maximum(np.diff(start), 1e-2)
        b_free[j] = np.concatenate([start[:1], np.log(gaps)])

    loglik_prev = -np.inf
    history = []
    for it in range(max_iters):
        # ---- E step: posterior over theta for each respondent
        loglik_nodes = np.zeros((n_resp, N_QUAD))
        for j in range(n_items):
            probs = _category_probs(a[j], _thresholds(b_free[j]), theta)  # (Q, K)
            obs = codes[:, j]
            seen = obs >= 0
            loglik_nodes[seen] += np.log(probs[:, obs[seen]]).T
        weighted = np.exp(loglik_nodes - loglik_nodes.max(axis=1, keepdims=True)) * prior
        marginal = weighted.sum(axis=1, keepdims=True)
        post = weighted / marginal
        loglik = float((np.log(marginal[:, 0]) + loglik_nodes.max(axis=1)).sum())
        history.append(loglik)

        # ---- M step: expected counts per (item, node, category), then fit
        for j in range(n_items):
            obs = codes[:, j]
            seen = obs >= 0
            counts = np.zeros((N_QUAD, n_cat))
            for k in range(n_cat):
                rows = seen & (obs == k)
                if rows.any():
                    counts[:, k] = post[rows].sum(axis=0)

            def negll(params, counts=counts):
                log_a = np.clip(params[0], np.log(A_BOUNDS[0]), np.log(A_BOUNDS[1]))
                aj = np.exp(log_a)
                probs = _category_probs(aj, _thresholds(params[1:]), theta)
                ridge = 0.5 * (log_a / LOG_A_PRIOR_SD) ** 2   # weak prior, keeps a finite
                return -float((counts * np.log(probs)).sum()) + ridge

            x0 = np.concatenate([[np.log(max(a[j], 1e-3))], b_free[j]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(negll, x0, method="Nelder-Mead",
                               options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
            a[j] = float(np.clip(np.exp(res.x[0]), *A_BOUNDS))
            b_free[j] = res.x[1:]

        if abs(loglik - loglik_prev) < tol * max(1.0, abs(loglik_prev)):
            break
        loglik_prev = loglik

    eap = (post * theta[None, :]).sum(axis=1)
    eap_sd = np.sqrt((post * (theta[None, :] - eap[:, None]) ** 2).sum(axis=1))
    b = np.array([_thresholds(row) for row in b_free])
    return {"a": a, "b": b, "theta": eap, "theta_se": eap_sd,
            "loglik": history[-1], "n_iter": len(history), "converged": len(history) < max_iters,
            "loglik_history": history}


def test_information(a: np.ndarray, b: np.ndarray, theta_grid: np.ndarray) -> np.ndarray:
    """Per-item Fisher information summed over the theta grid.

    Used to answer "does the headline lean rest on a handful of items?" -- the
    question the plan asks and the published equal-weight scoring cannot.
    """
    info = np.zeros(len(a))
    for j in range(len(a)):
        probs = _category_probs(a[j], b[j], theta_grid)
        cum = np.cumsum(probs[:, ::-1], axis=1)[:, ::-1][:, 1:]
        d = a[j] * cum * (1 - cum)
        num = np.diff(np.hstack([np.zeros((len(theta_grid), 1)), d,
                                 np.zeros((len(theta_grid), 1))]), axis=1) ** 2
        info[j] = float((num / probs).sum())
    return info


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------
def degenerate_items(codes: np.ndarray, min_variation: float = MIN_ITEM_VARIATION) -> np.ndarray:
    """Boolean mask of items with essentially no response variation.

    An item every administration answers the same way carries no information
    about any latent trait and cannot have its discrimination identified. How
    many such items an instrument has is itself a result -- it is the sharpest
    possible statement of "these 70 items are not 70 independent measurements".
    """
    mask = np.zeros(codes.shape[1], dtype=bool)
    for j in range(codes.shape[1]):
        obs = codes[:, j][codes[:, j] >= 0]
        if len(obs) == 0:
            mask[j] = True
            continue
        modal_share = np.bincount(obs).max() / len(obs)
        mask[j] = (1.0 - modal_share) < min_variation
    return mask


def orient(codes: np.ndarray, entries, n_cat: int) -> np.ndarray:
    """Sub-matrix for one axis, with reverse-keyed items flipped.

    Flipping is done on the category index, which is exactly the mirror the
    instruments' own scales define, so a reverse-keyed item contributes in the
    same direction as a forward-keyed one and the fitted theta has a single
    interpretable sign.
    """
    cols, signs = zip(*entries)
    sub = codes[:, list(cols)].copy()
    for c, sign in enumerate(signs):
        if sign < 0:
            flip = sub[:, c] >= 0
            sub[flip, c] = (n_cat - 1) - sub[flip, c]
    return sub


def bland_altman(x: np.ndarray, y: np.ndarray) -> dict:
    """Agreement between two measurements of the same quantity, on z-scores.

    Correlation says the two measures rank models the same way; Bland-Altman
    says whether they agree in level, which is the question when one is
    proposed as a substitute for the other.
    """
    xz = (x - x.mean()) / x.std(ddof=1)
    yz = (y - y.mean()) / y.std(ddof=1)
    diff = xz - yz
    return {"mean_difference": float(diff.mean()), "sd_difference": float(diff.std(ddof=1)),
            "loa_lower": float(diff.mean() - 1.96 * diff.std(ddof=1)),
            "loa_upper": float(diff.mean() + 1.96 * diff.std(ddof=1)),
            "mean_values": [float(v) for v in (xz + yz) / 2],
            "differences": [float(v) for v in diff]}


def avoidance_model(codes: np.ndarray, meta: pd.DataFrame, instrument: str) -> dict:
    """Wachter's two-stage separation of avoidance from position.

    Stage 1 models the probability that a response is the scale's neutral
    category, per model. Stage 2 re-fits position on engaged responses only.
    Only 8Values can support this: the Political Compass is forced-choice with
    no neutral option, so 'avoidance' has no observable form there, and
    reporting one anyway would be inventing a measurement.
    """
    neutral_label = NEUTRAL.get(instrument)
    if neutral_label is None:
        return {"applicable": False,
                "reason": "instrument is forced-choice with no neutral category"}
    k = CATEGORIES[instrument].index(neutral_label)
    is_neutral = (codes == k)
    valid = codes >= 0

    per_model = []
    for model in sorted(meta.model.unique()):
        rows = (meta.model == model).to_numpy()
        n_valid = int(valid[rows].sum())
        n_neutral = int(is_neutral[rows].sum())
        per_model.append({"model": model, "n_responses": n_valid,
                          "n_neutral": n_neutral,
                          "neutral_rate": n_neutral / n_valid if n_valid else float("nan")})
    per_item = [{"item": j + 1,
                 "neutral_rate": float(is_neutral[valid[:, j], j].mean())
                 if valid[:, j].any() else float("nan")}
                for j in range(codes.shape[1])]
    rates = np.array([m["neutral_rate"] for m in per_model])
    return {"applicable": True,
            "per_model": sorted(per_model, key=lambda m: -m["neutral_rate"]),
            "per_item_top": sorted(per_item, key=lambda i: -i["neutral_rate"])[:10],
            "overall_neutral_rate": float(np.nanmean(rates)),
            "range_across_models": [float(np.nanmin(rates)), float(np.nanmax(rates))]}


def run_axis(instrument: str, axis: str, codes: np.ndarray, entries, meta: pd.DataFrame,
             axis_scores: pd.DataFrame, engaged_only: bool = False) -> dict:
    """Fit one axis and compare its theta against the published axis score."""
    n_cat = len(CATEGORIES[instrument])
    sub = orient(codes, entries, n_cat)
    if engaged_only and instrument in NEUTRAL:
        k = CATEGORIES[instrument].index(NEUTRAL[instrument])
        sub = sub.copy()
        sub[sub == k] = -1          # treat neutrals as missing, not as a position

    dead = degenerate_items(sub)
    kept = np.where(~dead)[0]
    if len(kept) < 4:
        return {"instrument": instrument, "axis": axis, "skipped": True,
                "reason": f"only {len(kept)} of {sub.shape[1]} items vary enough to fit",
                "n_degenerate_items": int(dead.sum())}
    entries = [entries[i] for i in kept]
    sub = sub[:, kept]

    fit = fit_grm(sub, n_cat)
    per_trial = meta.copy()
    per_trial["theta"] = fit["theta"]
    by_model = per_trial.groupby("model")["theta"].mean()

    published = (axis_scores[(axis_scores.instrument == instrument)
                             & (axis_scores.axis == axis)]
                 .groupby("model")["value"].mean())
    common = sorted(set(by_model.index) & set(published.index))
    x = by_model.loc[common].to_numpy()
    y = published.loc[common].to_numpy()
    rho, p = spearmanr(x, y)

    theta_grid = np.linspace(-3, 3, 61)
    info = test_information(fit["a"], fit["b"], theta_grid)
    order = np.argsort(-info)
    total = info.sum()
    n_top = max(1, round(0.1 * len(info)))

    return {
        "instrument": instrument,
        "axis": axis,
        "n_items": int(sub.shape[1]),
        "n_degenerate_items_dropped": int(dead.sum()),
        "n_respondents": int(sub.shape[0]),
        "converged": bool(fit["converged"]),
        "n_iter": int(fit["n_iter"]),
        "loglik": float(fit["loglik"]),
        "discrimination": {
            "median": float(np.median(fit["a"])),
            "min": float(fit["a"].min()),
            "max": float(fit["a"].max()),
            "n_below_0.5": int((fit["a"] < 0.5).sum()),
        },
        "information_concentration": {
            "share_in_top_decile_of_items": float(info[order[:n_top]].sum() / total)
            if total else float("nan"),
            "n_items_for_half_of_information": int(
                np.searchsorted(np.cumsum(info[order]) / total, 0.5) + 1) if total else -1,
        },
        "top_items": [{"item": int(entries[i][0]) + 1, "orientation": int(entries[i][1]),
                       "a": float(fit["a"][i]), "information": float(info[i])}
                      for i in order[:8]],
        "theta_vs_published": {
            "spearman_rho": float(rho), "p_value": float(p), "n_models": len(common),
            "bland_altman": bland_altman(x, y),
            "per_model": [{"model": m, "theta": float(xi), "published": float(yi)}
                          for m, xi, yi in zip(common, x, y)],
        },
        "_theta_by_model": {m: float(v) for m, v in by_model.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "results" / "w1"))
    ap.add_argument("--instruments", nargs="*", default=["8values", "political_compass"])
    ap.add_argument("--skip-avoidance-stage2", action="store_true",
                    help="skip the engaged-responses-only refit (halves runtime)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    responses = load_responses()
    axis_scores = load_axis_scores()
    results = {"axes": [], "avoidance": {}, "compass_empirical_key": None}

    for instrument in args.instruments:
        codes, meta = responses[instrument]
        print(f"\n{instrument}: {codes.shape[0]} administrations x {codes.shape[1]} items, "
              f"{meta.model.nunique()} models")

        if instrument == "8values":
            key = eightvalues_key()
        else:
            key, diagnostics = compass_key(codes, meta, axis_scores)
            results["compass_empirical_key"] = diagnostics
            n_assigned = sum(len(v) for v in key.values())
            print(f"  empirical key assigned {n_assigned}/{codes.shape[1]} items to an axis")

        results["avoidance"][instrument] = avoidance_model(codes, meta, instrument)

        for axis, entries in key.items():
            if len(entries) < 4:
                print(f"  {axis}: only {len(entries)} items -- skipped")
                continue
            print(f"  fitting {axis} ({len(entries)} items)...", flush=True)
            res = run_axis(instrument, axis, codes, entries, meta, axis_scores)
            if res.get("skipped"):
                print(f"    skipped: {res['reason']}")
                results["axes"].append(res)
                continue
            if not args.skip_avoidance_stage2 and instrument in NEUTRAL:
                engaged = run_axis(instrument, axis, codes, entries, meta, axis_scores,
                                   engaged_only=True)
                a = np.array([r["theta"] for r in res["theta_vs_published"]["per_model"]])
                b = np.array([r["theta"] for r in engaged["theta_vs_published"]["per_model"]])
                res["engaged_only"] = {
                    "spearman_rho_vs_published":
                        engaged["theta_vs_published"]["spearman_rho"],
                    "spearman_rho_vs_all_responses": float(spearmanr(a, b).statistic),
                    "n_items": engaged["n_items"],
                }
            results["axes"].append(res)
            print(f"    rho(theta, published) = {res['theta_vs_published']['spearman_rho']:+.3f} "
                  f"(p = {res['theta_vs_published']['p_value']:.4f}), "
                  f"median a = {res['discrimination']['median']:.2f}, "
                  f"half the information in "
                  f"{res['information_concentration']['n_items_for_half_of_information']} items")

    print("\n" + "=" * 78)
    print("SUMMARY -- IRT theta vs published axis score, per axis")
    print("=" * 78)
    print(f"{'instrument':18s} {'axis':10s} {'items':>6s} {'rho':>7s} {'p':>8s} "
          f"{'med a':>7s} {'items for 50% info':>19s}")
    for r in results["axes"]:
        if r.get("skipped"):
            print(f"{r['instrument']:18s} {r['axis']:10s} skipped -- {r['reason']}")
            continue
        tp = r["theta_vs_published"]
        print(f"{r['instrument']:18s} {r['axis']:10s} {r['n_items']:6d} "
              f"{tp['spearman_rho']:+7.3f} {tp['p_value']:8.4f} "
              f"{r['discrimination']['median']:7.2f} "
              f"{r['information_concentration']['n_items_for_half_of_information']:19d}")

    for instrument, av in results["avoidance"].items():
        if not av["applicable"]:
            print(f"\navoidance ({instrument}): not applicable -- {av['reason']}")
            continue
        print(f"\navoidance ({instrument}): overall neutral rate "
              f"{av['overall_neutral_rate']*100:.1f}%, "
              f"range across models {av['range_across_models'][0]*100:.1f}%"
              f"-{av['range_across_models'][1]*100:.1f}%")
        for m in av["per_model"][:5]:
            print(f"    {m['model']:44s} {m['neutral_rate']*100:5.1f}%")

    path = out_dir / "irt_results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
