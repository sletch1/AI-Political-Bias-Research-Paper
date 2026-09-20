"""End-to-end tests of the W1 and W2 analysis pipelines on synthetic fixtures.

These pipelines run once, months from now, on data that costs several hundred
dollars to collect. Discovering then that a loader silently drops the condition
column, or that Gate A reads the wrong end of a threshold, is not recoverable.
So the fixtures below fabricate collection output with a known answer and check
that each pipeline reports that answer.

    python3 -m pytest scoring/test_analysis_pipelines.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_w1 as w1  # noqa: E402
import analyze_w2 as w2  # noqa: E402

MODELS = [f"org/model-{i}" for i in range(8)]
COMPASS_AXES = {"economic": -3.0, "social": -4.0}


def model_offsets(seed: int = 0) -> dict:
    """Per-model ideological offsets, shared between a fixture and its baseline.

    Drawing them once and passing them around is the point: an anchor-check
    test whose 'matching' baseline was drawn from a separately advanced RNG
    tests nothing except that two random samples differ.
    """
    rng = np.random.default_rng(seed)
    return {m: float(rng.normal(0, 1.0)) for m in MODELS}


def write_asker(directory: Path, condition_shift: float, seed: int = 0,
                offsets: dict = None):
    """8 models x 3 conditions x 6 trials of Political Compass scores, with a
    known per-condition shift and a known per-model offset."""
    rng = np.random.default_rng(seed + 1000)
    offsets = offsets if offsets is not None else model_offsets(seed)
    directory.mkdir(parents=True, exist_ok=True)
    shifts = {"none": 0.0, "conservative": condition_shift, "progressive": -condition_shift}
    for m in MODELS:
        offset = offsets[m]
        for condition, shift in shifts.items():
            for trial in range(1, 7):
                scores = {ax: base + offset + shift + rng.normal(0, 0.3)
                          for ax, base in COMPASS_AXES.items()}
                rec = {"model": m, "test": "political_compass", "condition": condition,
                       "trial": trial, "status": "ok", "answers": ["A"] * 62,
                       "scores": scores, "cost": 0.001}
                (directory / f"{m.replace('/', '_')}__political_compass__"
                             f"{condition}__trial{trial:02d}.json").write_text(json.dumps(rec))


def write_variants(directory: Path, inversion_shift: float, seed: int = 1):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        offset = rng.normal(0, 0.5)
        for condition in ("original", "inverted", "paraphrased"):
            shift = inversion_shift if condition == "inverted" else 0.0
            for trial in range(1, 7):
                scores = {ax: base + offset + shift + rng.normal(0, 0.2)
                          for ax, base in COMPASS_AXES.items()}
                rec = {"model": m, "test": "political_compass", "condition": condition,
                       "trial": trial, "status": "ok", "answers": ["A"] * 62,
                       "scores": scores, "cost": 0.001}
                (directory / f"{m.replace('/', '_')}__political_compass__"
                             f"{condition}__trial{trial:02d}.json").write_text(json.dumps(rec))


def write_memprobe(directory: Path, verbatim: bool):
    directory.mkdir(parents=True, exist_ok=True)
    items = [{"instrument": "8values", "item_index": i,
              "fragment": f"Statement number {i} begins",
              "full": f"Statement number {i} begins and then continues to its end."}
             for i in range(1, 6)]
    completions = [it["full"] if verbatim else f"Something else entirely for {it['item_index']}."
                   for it in items]
    rec = {"model": MODELS[0], "trial": 1, "seed": 1, "status": "ok",
           "items": items, "completions": completions, "cost": 0.0}
    (directory / f"memprobe__{MODELS[0].replace('/', '_')}__trial01.json").write_text(
        json.dumps(rec))


def write_format(directory: Path, scale_shift: float, seed: int = 2):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    configs = [{"scale": s, "item_order": i, "option_order": "published",
                "elicitation": "direct"}
               for s in ("s4_forced", "s5_neutral", "s7_likert")
               for i in ("published", "randomised")]
    for m in MODELS:
        offset = rng.normal(0, 0.5)
        for cfg in configs:
            cfg_id = "-".join(cfg[k] for k in ("scale", "item_order", "option_order",
                                               "elicitation"))
            shift = scale_shift if cfg["scale"] == "s7_likert" else 0.0
            for trial in range(1, 4):
                scores = {ax: base + offset + shift + rng.normal(0, 0.2)
                          for ax, base in COMPASS_AXES.items()}
                scores["n_neutral"] = 0
                scores["bracket_width"] = 0.0
                rec = {"model": m, "test": "political_compass", "config": cfg,
                       "config_id": cfg_id, "trial": trial, "status": "ok",
                       "answers": ["A"] * 62, "scores": scores, "cost": 0.001}
                (directory / f"{m.replace('/', '_')}__political_compass__"
                             f"{cfg_id}__trial{trial:02d}.json").write_text(json.dumps(rec))


@pytest.fixture()
def baseline():
    """A stand-in for data/raw_trials/, built from the *same* per-model offsets
    the asker fixture uses, so the anchor cell genuinely should reproduce it."""
    import pandas as pd

    rng = np.random.default_rng(2000)
    offsets = model_offsets(0)
    rows = []
    for m in MODELS:
        offset = offsets[m]
        for ax, base in COMPASS_AXES.items():
            for trial in range(1, 7):
                rows.append({"model": m, "instrument": "political_compass", "trial": trial,
                             "axis": ax, "value": base + offset + rng.normal(0, 0.3),
                             "trait": f"political_compass:{ax}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def test_loader_returns_long_form_with_condition(tmp_path):
    write_asker(tmp_path, condition_shift=1.0)
    df, failures = w1.load_trials(tmp_path, extra_keys=("condition",))
    assert set(df.columns) >= {"model", "instrument", "trial", "condition", "axis",
                               "value", "trait"}
    assert len(df) == len(MODELS) * 3 * 6 * 2
    assert failures.empty


def test_loader_counts_failures_separately_instead_of_dropping_them(tmp_path):
    write_asker(tmp_path, condition_shift=0.0)
    bad = {"model": MODELS[0], "test": "political_compass", "condition": "none",
           "trial": 99, "status": "parse_error", "cost": 0.0}
    (tmp_path / "bad__political_compass__none__trial99.json").write_text(json.dumps(bad))
    df, failures = w1.load_trials(tmp_path, extra_keys=("condition",))
    assert len(failures) == 1
    assert 99 not in set(df.trial)


def test_loader_skips_probe_files(tmp_path):
    write_asker(tmp_path, condition_shift=0.0)
    write_memprobe(tmp_path, verbatim=True)
    df, _ = w1.load_trials(tmp_path, extra_keys=("condition",))
    assert len(df) == len(MODELS) * 3 * 6 * 2


def test_loader_on_a_missing_directory_returns_empty(tmp_path):
    df, failures = w1.load_trials(tmp_path / "nope", extra_keys=("condition",))
    assert df.empty and failures.empty


# ---------------------------------------------------------------------------
# 1.1 asker identity
# ---------------------------------------------------------------------------
def test_asker_arm_detects_a_large_condition_effect(tmp_path, baseline):
    write_asker(tmp_path, condition_shift=3.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("condition",))
    out = w1.asker_arm(df, baseline)
    assert out["available"]
    assert out["gate_a"]["max_condition_eta_squared"] > 0.10
    assert out["gate_a"]["verdict"].startswith("PROCEED")


def test_asker_arm_reframes_when_there_is_no_condition_effect(tmp_path, baseline):
    write_asker(tmp_path, condition_shift=0.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("condition",))
    out = w1.asker_arm(df, baseline)
    assert out["gate_a"]["max_condition_eta_squared"] < 0.05
    assert out["gate_a"]["verdict"].startswith("REFRAME")


def test_asker_arm_reports_the_accommodation_asymmetry(tmp_path, baseline):
    write_asker(tmp_path, condition_shift=2.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("condition",))
    out = w1.asker_arm(df, baseline)
    for entry in out["per_axis"]:
        # Symmetric shifts were simulated, so the ratio must be near 1. A
        # pipeline that reported Tornberg's 8x here would be reporting a bug.
        assert entry["shift_conservative"] > 0 > entry["shift_progressive"]
        assert 0.5 < entry["asymmetry_ratio"] < 2.0


def test_anchor_check_passes_when_the_none_cell_matches_the_baseline(tmp_path, baseline):
    write_asker(tmp_path, condition_shift=1.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("condition",))
    assert w1.anchor_check(df, baseline)["anchor_holds"]


def test_anchor_check_fails_on_version_drift(tmp_path, baseline):
    write_asker(tmp_path, condition_shift=1.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("condition",))
    df.loc[df.condition == "none", "value"] += 5.0     # 25% of the Compass scale
    out = w1.anchor_check(df, baseline)
    assert not out["anchor_holds"]


def test_asker_arm_reports_absence_rather_than_crashing(tmp_path, baseline):
    import pandas as pd

    out = w1.asker_arm(pd.DataFrame(), baseline)
    assert out["available"] is False and "reason" in out


# ---------------------------------------------------------------------------
# 1.2 contamination
# ---------------------------------------------------------------------------
def test_contamination_arm_finds_a_planted_inversion_shift(tmp_path, baseline):
    write_variants(tmp_path, inversion_shift=4.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("condition",))
    out = w1.contamination_arm(df, baseline)
    assert out["available"]
    assert out["mean_contamination_index"] > 0.15      # 4 points of a 20-point scale
    assert out["n_models_flagged_after_fdr"] == len(MODELS)


def test_contamination_arm_is_near_zero_when_the_model_answers_genuinely(tmp_path, baseline):
    write_variants(tmp_path, inversion_shift=0.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("condition",))
    out = w1.contamination_arm(df, baseline)
    assert out["mean_contamination_index"] < 0.03
    assert out["n_models_flagged_after_fdr"] <= 1      # FDR-controlled false positives


def test_contamination_arm_records_which_original_it_compared_against(tmp_path, baseline):
    write_variants(tmp_path, inversion_shift=1.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("condition",))
    assert "re-collected" in w1.contamination_arm(df, baseline)["original_source"]
    df = df[df.condition != "original"]
    assert "baseline" in w1.contamination_arm(df, baseline)["original_source"]


def test_memorisation_arm_separates_verbatim_from_paraphrase(tmp_path):
    write_memprobe(tmp_path, verbatim=True)
    hit = w1.memorisation_arm(tmp_path)
    assert hit["available"] and hit["mean_exact_rate"] == 1.0

    other = tmp_path / "other"
    write_memprobe(other, verbatim=False)
    miss = w1.memorisation_arm(other)
    assert miss["mean_exact_rate"] == 0.0 and miss["mean_near_rate"] == 0.0


def test_memorisation_normalisation_ignores_punctuation_and_case():
    assert w1._normalise("It's a Test, isn't it?") == w1._normalise("ITS A TEST ISNT IT")


# ---------------------------------------------------------------------------
# 1.3 format
# ---------------------------------------------------------------------------
def test_format_arm_attributes_variance_to_the_scale_factor(tmp_path):
    write_format(tmp_path, scale_shift=4.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("config", "config_id"))
    out = w1.format_arm(df)
    assert out["available"]
    etas = [row["scale"] for row in out["partial_eta_squared_per_axis"]]
    assert max(etas) > 0.3


def test_format_arm_reports_positions_as_intervals(tmp_path):
    write_format(tmp_path, scale_shift=4.0)
    df, _ = w1.load_trials(tmp_path, extra_keys=("config", "config_id"))
    out = w1.format_arm(df)
    widest = out["design_averaged_positions"][0]
    assert widest["range_as_frac_of_scale"] > 0.1
    assert widest["min_config_mean"] < widest["design_averaged_mean"] < widest["max_config_mean"]
    assert widest["bootstrap_ci"]["ci_lower"] <= widest["bootstrap_ci"]["estimate"]


# ---------------------------------------------------------------------------
# W2
# ---------------------------------------------------------------------------
def test_cohens_kappa_is_one_on_perfect_agreement():
    v = [-8, -3, 0, 4, 9, -5, 1]
    assert w2.cohens_kappa(v, v)["kappa"] == pytest.approx(1.0)


def test_cohens_kappa_is_near_zero_on_independent_labels():
    rng = np.random.default_rng(11)
    a = rng.uniform(-10, 10, 400)
    b = rng.uniform(-10, 10, 400)
    assert abs(w2.cohens_kappa(a, b)["kappa"]) < 0.15


def test_cohens_kappa_trichotomises_at_the_stated_cut():
    # +-2 is the cut; 1.5 is neutral, 2.5 is right.
    out = w2.cohens_kappa([1.5, 2.5], [-1.5, 2.5])
    assert out["observed_agreement"] == pytest.approx(1.0)


def test_gate_b_verdicts_match_the_plans_thresholds():
    assert w2._gate_b([0.75], 19)["verdict"].startswith("PROCEED WITH CONFIDENCE")
    assert w2._gate_b([0.15], 19)["verdict"].startswith("PROCEED")
    ambiguous = w2._gate_b([0.45], 19)["verdict"]
    assert "UNDERPOWERED" in ambiguous and "section 4.7" in ambiguous


def test_gate_a_verdicts_match_the_plans_thresholds():
    assert w1._gate_a([0.2]).startswith("PROCEED")
    assert w1._gate_a([0.02]).startswith("REFRAME")
    assert w1._gate_a([0.07]).startswith("AMBIGUOUS")
    assert w1._gate_a([]) == "no data"


def test_w2_arms_report_absence_rather_than_crashing():
    import pandas as pd

    q = pd.DataFrame(columns=["model", "instrument", "axis", "value"])
    assert w2.issuebench_arm(pd.DataFrame(), q)["available"] is False
    assert w2.ballot_arm(q)["available"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# SapplyValues triangulation check (oct_fix.md F3)
# ---------------------------------------------------------------------------
def test_sapplyvalues_directional_check_matches_manual_computation():
    import analyze_sapplyvalues as asv

    sv_means = {
        "m1": {"right": -2.0, "auth": 1.0, "prog": 3.0},
        "m2": {"right": 1.0, "auth": -1.0, "prog": -2.0},
    }
    d = asv.directional_check(sv_means)
    assert d["n_models"] == 2
    assert d["n_economically_left"] == 1
    assert d["n_libertarian"] == 1
    assert d["n_progressive"] == 1


def test_sapplyvalues_convergence_check_recovers_a_perfect_correlation():
    import analyze_sapplyvalues as asv

    sv_means = {f"m{i}": {"right": float(i)} for i in range(6)}
    other_means = {f"m{i}": {"political_compass:economic": float(i)} for i in range(6)}
    out = asv.convergence_check(sv_means, other_means)
    row = next(r for r in out if r["other"] == "political_compass:economic")
    assert row["available"]
    assert row["spearman_rho"] == 1.0
    assert row["agrees_with_expected_sign"] is True


def test_sapplyvalues_convergence_check_reports_unavailable_below_four_models():
    import analyze_sapplyvalues as asv

    sv_means = {"m1": {"right": 1.0}, "m2": {"right": 2.0}}
    other_means = {"m1": {"political_compass:economic": 1.0}, "m2": {"political_compass:economic": 2.0}}
    out = asv.convergence_check(sv_means, other_means)
    row = next(r for r in out if r["other"] == "political_compass:economic")
    assert row["available"] is False
