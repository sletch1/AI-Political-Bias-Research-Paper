"""Workstream 2 analysis -- ecological validity, and Gate B (updates/04_analysis.md).

    python3 scoring/analyze_w2.py [--out results/w2]

Task 2.1  Does questionnaire score predict behaviour on realistic prompts?
          Per-model mean stance on the IssueBench arm against per-model mean
          questionnaire score, per axis, n = 19 models, with a bootstrap CI.
          Judge quality is reported first: without Cohen's kappa on the human
          validation subsample the correlation is uninterpretable, so a missing
          validation file is stated loudly rather than passed over.

Task 2.2  Do model votes on real ballot measures track real outcomes and party
          endorsements? Agreement with the certified result, agreement with each
          party's endorsement, and the correlation between a model's
          questionnaire lean and its partisan agreement rate.

Gate B (updates/04_analysis.md) is evaluated and printed. Both directions are wins:
|r| > 0.6 rehabilitates questionnaire audits, |r| < 0.3 confirms Rottger and
strengthens the error budget. The one outcome to avoid is an ambiguous middle
reported as if it were a finding, so an ambiguous result is labelled as
underpowered and paired with the roster size that would resolve it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from variance_components import bootstrap_ci  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ISSUEBENCH_DIR = REPO / "data" / "issuebench"
GEN_DIR = ISSUEBENCH_DIR / "generations"
VALIDATION_CSV = ISSUEBENCH_DIR / "validation_sample.csv"
BALLOT_DIR = REPO / "data" / "ballot"
SCORES_CSV = REPO / "data" / "scores.csv"

# Which questionnaire axis each generated-stance axis is compared against, and
# the sign that puts the two on the same direction. The judge's convention is
# negative = left/libertarian; Political Compass is the same; 8Values equality
# runs the other way (high = egalitarian = left), hence the -1.
AXIS_MAP = {
    "economic": [("political_compass", "economic", +1), ("8values", "equality", -1)],
    "social": [("political_compass", "social", +1), ("8values", "liberty", -1)],
}


def load_questionnaire() -> pd.DataFrame:
    wide = pd.read_csv(SCORES_CSV)
    rows = []
    for _, r in wide.iterrows():
        for i in (1, 2, 3, 4):
            name, val = r.get(f"axis{i}_name"), r.get(f"axis{i}_value")
            if isinstance(name, str) and pd.notna(val):
                rows.append({"model": r["model"], "instrument": r["test"],
                             "axis": name, "value": float(val)})
    return pd.DataFrame(rows).groupby(["model", "instrument", "axis"])["value"].mean().reset_index()


def load_generations() -> pd.DataFrame:
    rows = []
    if not GEN_DIR.exists():
        return pd.DataFrame(rows)
    for path in sorted(GEN_DIR.glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        rows.append({"model": rec["model"], "issue": rec["issue"], "axis": rec["axis"],
                     "prompt_id": rec["prompt_id"], "lean": float(rec["lean_score"]),
                     "confidence": rec.get("judge_confidence")})
    return pd.DataFrame(rows)


def cohens_kappa(a, b, cut: float = 2.0) -> dict:
    """Cohen's kappa on stance trichotomised at +-`cut`.

    The judge emits a continuous -10..+10 score, but agreement on a continuous
    scale is a correlation, not a kappa. Trichotomising at +-2 into
    left / neutral / right is the coarsest reading that still preserves what the
    arm claims to measure, and it is the reading a reviewer will apply.
    """
    def code(v):
        return -1 if v <= -cut else (1 if v >= cut else 0)

    ca = np.array([code(v) for v in a])
    cb = np.array([code(v) for v in b])
    labels = (-1, 0, 1)
    n = len(ca)
    observed = float((ca == cb).mean())
    expected = sum((ca == l).mean() * (cb == l).mean() for l in labels)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return {"kappa": float(kappa), "observed_agreement": observed,
            "expected_agreement": float(expected), "n": int(n), "cut": cut}


def judge_validation() -> dict:
    if not VALIDATION_CSV.exists():
        return {"available": False,
                "reason": f"{VALIDATION_CSV.name} not found -- run "
                          "collect_issuebench.py --export-validation-sample and have a "
                          "human label it. Without kappa the arm is uninterpretable."}
    with open(VALIDATION_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    pairs = [(float(r["judge_lean"]), float(r["human_lean"]))
             for r in rows if r.get("human_lean", "").strip()]
    if len(pairs) < 30:
        return {"available": False,
                "reason": f"only {len(pairs)} of {len(rows)} rows have a human label; "
                          "updates/03_experiments.md asks for 200"}
    judge, human = zip(*pairs)
    k = cohens_kappa(judge, human)
    r, p = pearsonr(judge, human)
    k.update({"available": True, "pearson_r": float(r), "pearson_p": float(p),
              "n_labelled": len(pairs), "n_exported": len(rows),
              "meets_target": bool(k["kappa"] > 0.7),
              "target": "updates/03_experiments.md Task 2.1 targets kappa > 0.7"})
    return k


def issuebench_arm(gens: pd.DataFrame, questionnaire: pd.DataFrame) -> dict:
    if gens.empty:
        return {"available": False, "reason": f"no generations in {GEN_DIR}"}

    source = "unknown"
    manifest = ISSUEBENCH_DIR / "prompt_set.json"
    if manifest.exists():
        source = json.loads(manifest.read_text()).get("source", "unknown")

    by_model_axis = gens.groupby(["model", "axis"])["lean"].mean()
    out = {"available": True, "prompt_source": source,
           "n_generations": int(len(gens)), "n_models": int(gens.model.nunique()),
           "n_issues": int(gens.issue.nunique()), "correlations": []}
    if not source.startswith("issuebench"):
        out["warning"] = ("Prompts were our own construction, not IssueBench itself. "
                          "Report accordingly (updates/03_experiments.md Task 2.1 (IssueBench risk)).")

    for gen_axis, targets in AXIS_MAP.items():
        stance = by_model_axis.xs(gen_axis, level="axis") if gen_axis in \
            by_model_axis.index.get_level_values("axis") else None
        if stance is None or stance.empty:
            continue
        for instrument, axis, sign in targets:
            q = questionnaire[(questionnaire.instrument == instrument)
                              & (questionnaire.axis == axis)].set_index("model")["value"]
            common = sorted(set(stance.index) & set(q.index))
            if len(common) < 5:
                continue
            x = stance.loc[common].to_numpy()
            y = sign * q.loc[common].to_numpy()
            r, p = pearsonr(x, y)
            rho, rho_p = spearmanr(x, y)
            out["correlations"].append({
                "generated_axis": gen_axis, "instrument": instrument, "axis": axis,
                "orientation_sign": sign, "n_models": len(common),
                "pearson_r": float(r), "pearson_p": float(p),
                "spearman_rho": float(rho), "spearman_p": float(rho_p),
                "bootstrap_ci": bootstrap_ci(x, y),
                "per_model": [{"model": m, "stance": float(xi), "questionnaire": float(yi)}
                              for m, xi, yi in zip(common, x, y)],
            })

    rs = [abs(c["pearson_r"]) for c in out["correlations"]]
    out["gate_b"] = _gate_b(rs, out["n_models"])
    return out


def _gate_b(abs_rs, n_models: int) -> dict:
    if not abs_rs:
        return {"verdict": "no data"}
    best = float(np.max(abs_rs))
    if best > 0.6:
        verdict = ("PROCEED WITH CONFIDENCE: questionnaire scores predict realistic "
                   "behaviour. This rehabilitates the method and is contrarian given "
                   "the 2026 critiques.")
    elif best < 0.3:
        verdict = ("PROCEED: questionnaires do not predict realistic behaviour. This is "
                   "the stronger error-budget result and confirms Rottger and Barmettler.")
    else:
        verdict = (f"UNDERPOWERED at n = {n_models} models. Do NOT report this as a "
                   "finding (updates/04_analysis.md Gate B names this as the section 4.7 mistake). "
                   "Either expand the roster to ~30 models or report as inconclusive "
                   "with an explicit power analysis.")
    return {"max_abs_pearson_r": best, "n_models": n_models, "verdict": verdict}


def ballot_arm(questionnaire: pd.DataFrame) -> dict:
    votes_dir = BALLOT_DIR / "votes"
    csv_path = BALLOT_DIR / "propositions.csv"
    if not votes_dir.exists() or not any(votes_dir.glob("*.json")):
        return {"available": False, "reason": f"no votes in {votes_dir}"}
    if not csv_path.exists():
        return {"available": False, "reason": f"{csv_path} not found"}

    with open(csv_path, newline="", encoding="utf-8") as fh:
        props = {r["id"]: r for r in csv.DictReader(fh)}

    rows = []
    for path in sorted(votes_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        prop = props.get(rec["proposition_id"])
        if prop is None:
            continue
        yes = rec["vote"] == "YES"
        rows.append({
            "model": rec["model"], "proposition_id": rec["proposition_id"],
            "axis": rec["axis"], "trial": rec["trial"], "model_yes": yes,
            "matches_outcome": yes == (int(prop["passed"]) == 1),
            "matches_dem": None if prop["dem_endorsement"] == "none"
            else yes == (prop["dem_endorsement"] == "yes"),
            "matches_rep": None if prop["rep_endorsement"] == "none"
            else yes == (prop["rep_endorsement"] == "yes"),
            "yes_share": float(prop["yes_share"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return {"available": False, "reason": "no votes matched a known proposition"}

    per_model = []
    for model, grp in df.groupby("model"):
        dem = grp.matches_dem.dropna()
        rep = grp.matches_rep.dropna()
        per_model.append({
            "model": model, "n_votes": int(len(grp)),
            "agreement_with_outcome": float(grp.matches_outcome.mean()),
            "agreement_with_dem_endorsement": float(dem.mean()) if len(dem) else None,
            "agreement_with_rep_endorsement": float(rep.mean()) if len(rep) else None,
            "partisan_gap": float(dem.mean() - rep.mean()) if len(dem) and len(rep) else None,
        })

    out = {"available": True, "n_votes": int(len(df)),
           "n_models": int(df.model.nunique()),
           "n_propositions": int(df.proposition_id.nunique()),
           "per_model": sorted(per_model, key=lambda m: -(m["partisan_gap"] or 0)),
           "mean_agreement_with_outcome": float(df.matches_outcome.mean())}

    # Barmettler's claim, tested directly: does questionnaire lean predict how
    # a model votes on concrete measures?
    gaps = {m["model"]: m["partisan_gap"] for m in per_model if m["partisan_gap"] is not None}
    q = questionnaire[(questionnaire.instrument == "political_compass")
                      & (questionnaire.axis == "economic")].set_index("model")["value"]
    common = sorted(set(gaps) & set(q.index))
    if len(common) >= 5:
        x = np.array([gaps[m] for m in common])
        y = q.loc[common].to_numpy()
        r, p = pearsonr(x, y)
        out["partisan_gap_vs_questionnaire"] = {
            "pearson_r": float(r), "p_value": float(p), "n_models": len(common),
            "bootstrap_ci": bootstrap_ci(x, y),
            "interpretation": ("Negative r means models that score economically left on "
                               "the questionnaire also vote with Democratic endorsements "
                               "on real measures -- convergent validity against ground "
                               "truth, which no questionnaire-only design can provide."),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "results" / "w2"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    questionnaire = load_questionnaire()
    gens = load_generations()
    results = {
        "judge_validation": judge_validation(),
        "task_2_1_issuebench": issuebench_arm(gens, questionnaire),
        "task_2_2_ballot": ballot_arm(questionnaire),
    }

    print("=" * 78)
    print("W2 ECOLOGICAL VALIDITY")
    print("=" * 78)

    jv = results["judge_validation"]
    print("-- judge validation " + "-" * 57)
    if not jv["available"]:
        print(f"   NOT DONE: {jv['reason']}")
    else:
        print(f"   Cohen's kappa = {jv['kappa']:.3f} on {jv['n_labelled']} human-labelled "
              f"passages (target > 0.7: {jv['meets_target']})")

    a = results["task_2_1_issuebench"]
    print("\n-- 2.1 IssueBench " + "-" * 59)
    if not a["available"]:
        print(f"   not collected ({a['reason']})")
    else:
        print(f"   source: {a['prompt_source']}   "
              f"{a['n_generations']} generations, {a['n_models']} models")
        if a.get("warning"):
            print(f"   WARNING: {a['warning']}")
        for c in a["correlations"]:
            ci = c["bootstrap_ci"]
            print(f"   {c['generated_axis']:9s} vs {c['instrument']}:{c['axis']:10s} "
                  f"r = {c['pearson_r']:+.3f} [{ci['ci_lower']:+.2f}, {ci['ci_upper']:+.2f}] "
                  f"n = {c['n_models']}")
        print(f"   GATE B: {a['gate_b']['verdict']}")

    b = results["task_2_2_ballot"]
    print("\n-- 2.2 ballot propositions " + "-" * 50)
    if not b["available"]:
        print(f"   not collected ({b['reason']})")
    else:
        print(f"   {b['n_votes']} votes over {b['n_propositions']} propositions; "
              f"mean agreement with certified outcome "
              f"{b['mean_agreement_with_outcome']*100:.1f}%")
        for m in b["per_model"][:5]:
            print(f"     {m['model']:44s} partisan gap {m['partisan_gap']:+.3f}")

    path = out_dir / "w2_results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
