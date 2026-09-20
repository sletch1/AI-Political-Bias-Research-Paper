"""Tests for the shared statistical layer (updates/04_analysis.md) and the IRT fit.

These run on synthetic data with a known answer, which is the only way to tell
a variance decomposition that is right from one that merely returns numbers.

    python3 -m pytest scoring/test_statistics.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import irt_analysis as irt  # noqa: E402
import variance_components as vc  # noqa: E402


def synthetic_panel(model_sd=1.0, condition_effect=0.0, noise=1.0, seed=0):
    """Panel with a known variance structure: model effect, condition effect,
    and residual noise, with everything else deliberately absent."""
    rng = np.random.default_rng(seed)
    models = [f"m{i}" for i in range(19)]
    conditions = ["none", "conservative", "progressive"]
    offsets = {m: rng.normal(0, model_sd) for m in models}
    shifts = {"none": 0.0, "conservative": condition_effect, "progressive": -condition_effect}
    rows = []
    for m in models:
        for c in conditions:
            for t in range(20):
                rows.append({"model": m, "condition": c, "trait": "i:a", "trial": t,
                             "value": offsets[m] + shifts[c] + rng.normal(0, noise)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# variance decomposition
# ---------------------------------------------------------------------------
def test_shares_sum_to_one():
    df = synthetic_panel()
    out = vc.variance_decomposition(df, "value", random_effects=["model"])
    assert sum(out["variance_share"].values()) == pytest.approx(1.0, abs=1e-6)


def test_model_variance_is_recovered_when_it_dominates():
    df = synthetic_panel(model_sd=3.0, noise=0.5)
    out = vc.variance_decomposition(df, "value", random_effects=["model"])
    assert out["variance_share"]["model"] > 0.8


def test_model_variance_is_small_when_noise_dominates():
    df = synthetic_panel(model_sd=0.2, noise=3.0)
    out = vc.variance_decomposition(df, "value", random_effects=["model"])
    assert out["variance_share"]["model"] < 0.2


def test_fixed_effects_are_removed_before_components_are_estimated():
    """A large condition effect entered as fixed must not inflate the model
    component. This is the failure mode that would turn an asker effect into a
    fake 'models differ' result."""
    df = synthetic_panel(model_sd=1.0, condition_effect=4.0, noise=0.5)
    without = vc.variance_decomposition(df, "value", random_effects=["model"])
    with_fixed = vc.variance_decomposition(df, "value", random_effects=["model"],
                                           fixed_effects=["condition"])
    assert with_fixed["variance_share"]["model"] > without["variance_share"]["model"]


def test_z_within_puts_two_scales_on_a_common_footing():
    """Without z_within, an instrument on [0, 100] swamps one on [-10, 10] and
    every component is really a statement about the bigger scale."""
    rng = np.random.default_rng(1)
    rows = []
    for m in [f"m{i}" for i in range(10)]:
        base = rng.normal()
        for trait, scale in (("8values:equality", 100.0), ("political_compass:economic", 2.0)):
            for t in range(20):
                rows.append({"model": m, "trait": trait,
                             "value": scale * (base + rng.normal(0, 0.3))})
    df = pd.DataFrame(rows)
    raw = vc.variance_decomposition(df, "value", random_effects=["model", "trait"])
    zed = vc.variance_decomposition(df, "value", random_effects=["model", "trait"],
                                    z_within="trait")
    assert raw["variance_share"]["trait"] > zed["variance_share"]["trait"]
    assert zed["variance_share"]["model"] > raw["variance_share"]["model"]


def test_interaction_random_effect_is_built_from_its_parts():
    df = synthetic_panel()
    out = vc.variance_decomposition(df, "value", random_effects=["model", "model:condition"])
    assert "model:condition" in out["variance_components"]


# ---------------------------------------------------------------------------
# effect sizes and corrections
# ---------------------------------------------------------------------------
def test_partial_eta_squared_is_large_for_a_real_effect_and_small_for_none():
    strong = vc.partial_eta_squared(synthetic_panel(condition_effect=5.0, noise=0.5),
                                    "value", ["model", "condition"])
    null = vc.partial_eta_squared(synthetic_panel(condition_effect=0.0, noise=3.0),
                                  "value", ["model", "condition"])
    assert strong["condition"]["partial_eta_squared"] > 0.5
    assert null["condition"]["partial_eta_squared"] < 0.05


def test_benjamini_hochberg_is_monotone_and_bounded():
    out = vc.benjamini_hochberg([0.001, 0.01, 0.03, 0.2, 0.9])
    adj = out["adjusted"]
    assert adj == sorted(adj)
    assert all(0 <= a <= 1 for a in adj)
    assert all(a >= p for a, p in zip(adj, [0.001, 0.01, 0.03, 0.2, 0.9]))


def test_benjamini_hochberg_rejects_fewer_than_uncorrected():
    ps = [0.01, 0.02, 0.03, 0.04, 0.045] + [0.5] * 45
    out = vc.benjamini_hochberg(ps)
    assert out["n_rejected"] < sum(p < 0.05 for p in ps)


def test_benjamini_hochberg_handles_an_empty_family():
    assert vc.benjamini_hochberg([])["n_rejected"] == 0


def test_bootstrap_ci_brackets_a_known_correlation():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    y = 0.8 * x + rng.normal(scale=0.6, size=200)
    out = vc.bootstrap_ci(x, y, n_boot=1000)
    assert out["ci_lower"] < out["estimate"] < out["ci_upper"]
    assert out["ci_lower"] > 0.5


def test_bootstrap_ci_is_wide_at_n_equals_19():
    """updates/04_analysis.md insists on intervals because n = 19 models is small.
    This is the test that keeps that honest."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=19)
    y = 0.5 * x + rng.normal(scale=1.0, size=19)
    out = vc.bootstrap_ci(x, y, n_boot=2000)
    assert out["ci_upper"] - out["ci_lower"] > 0.4


def test_bootstrap_ci_rejects_mismatched_pairs():
    with pytest.raises(ValueError):
        vc.bootstrap_ci([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# cluster bootstrap over models (oct_fix.md F4)
# ---------------------------------------------------------------------------
def test_cluster_bootstrap_brackets_a_dominant_model_share():
    df = synthetic_panel(model_sd=3.0, noise=0.5)
    out = vc.cluster_bootstrap_variance_shares(
        df, "value", random_effects=["model"], cluster="model", n_boot=200,
    )
    point = out["point_estimate"]["variance_share"]["model"]
    ci = out["variance_share_ci"]["model"]
    assert ci["ci_lower"] < point < ci["ci_upper"]
    assert point > 0.7


def test_cluster_bootstrap_resamples_whole_models_not_rows():
    """A resample that duplicated single rows instead of whole models would
    understate the interval width; this is loose enough to catch that bug
    without being a flaky width assertion."""
    df = synthetic_panel(model_sd=1.0, noise=1.0)
    row_level = vc.bootstrap_ci(df["value"].to_numpy(), n_boot=200)
    cluster_level = vc.cluster_bootstrap_variance_shares(
        df, "value", random_effects=["model"], cluster="model", n_boot=200,
    )
    ci = cluster_level["variance_share_ci"]["model"]
    assert ci["ci_upper"] - ci["ci_lower"] > 0.0
    assert row_level["n_boot"] == 200  # sanity: the row-level bootstrap still ran


def test_cluster_bootstrap_uses_every_cluster_and_records_seed():
    df = synthetic_panel(model_sd=1.0, noise=1.0)
    out = vc.cluster_bootstrap_variance_shares(
        df, "value", random_effects=["model"], cluster="model", n_boot=50, seed=7,
    )
    assert out["n_clusters"] == df["model"].nunique() == 19
    assert out["seed"] == 7
    assert out["variance_share_ci"]["model"]["n_boot"] <= 50


# ---------------------------------------------------------------------------
# MTMM
# ---------------------------------------------------------------------------
def test_mtmm_detects_convergent_validity_when_it_exists():
    rng = np.random.default_rng(5)
    rows = []
    for m in [f"m{i}" for i in range(30)]:
        econ, soc = rng.normal(), rng.normal()
        for method in ("A", "B"):
            for trait, true in (("econ", econ), ("social", soc)):
                rows.append({"model": m, "method": method, "trait": trait,
                             "value": true + rng.normal(scale=0.2)})
    out = vc.mtmm(pd.DataFrame(rows), "model", "trait", "method", "value")
    assert out["convergent_mean_r"] > 0.9
    assert out["campbell_fiske_passes"]


def test_mtmm_fails_campbell_fiske_when_methods_disagree():
    rng = np.random.default_rng(6)
    rows = []
    for m in [f"m{i}" for i in range(30)]:
        common = rng.normal()
        for method in ("A", "B"):
            for trait in ("econ", "social"):
                # A strong method factor and no trait factor: exactly the
                # pattern that should fail convergent validity.
                rows.append({"model": m, "method": method, "trait": trait,
                             "value": common + rng.normal(scale=0.1)
                             if method == "A" else rng.normal()})
    out = vc.mtmm(pd.DataFrame(rows), "model", "trait", "method", "value")
    assert not out["campbell_fiske_passes"]


# ---------------------------------------------------------------------------
# IRT
# ---------------------------------------------------------------------------
def test_grm_recovers_a_known_latent_trait():
    """Simulate from the model, fit it back, and check theta is recovered.
    Without this, a plausible-looking theta is indistinguishable from noise."""
    rng = np.random.default_rng(7)
    n_resp, n_items, n_cat = 400, 15, 5
    theta = rng.normal(size=n_resp)
    a = rng.uniform(0.8, 2.0, size=n_items)
    b = np.sort(rng.normal(scale=1.2, size=(n_items, n_cat - 1)), axis=1)
    codes = np.zeros((n_resp, n_items), dtype=int)
    for j in range(n_items):
        probs = irt._category_probs(a[j], b[j], theta)
        for i in range(n_resp):
            codes[i, j] = rng.choice(n_cat, p=probs[i] / probs[i].sum())
    fit = irt.fit_grm(codes, n_cat, max_iters=25)
    assert np.corrcoef(fit["theta"], theta)[0, 1] > 0.85
    assert np.corrcoef(fit["a"], a)[0, 1] > 0.5


def test_thresholds_are_monotone_for_any_free_vector():
    for row in (np.array([0.0, 0.0, 0.0, 0.0]), np.array([-3.0, 1.0, -2.0, 5.0]),
                np.array([2.0, 40.0, -40.0, 0.0])):
        b = irt._thresholds(row)
        assert np.all(np.diff(b) > 0)
        assert np.all(np.isfinite(b))


def test_category_probabilities_sum_to_one():
    theta = np.linspace(-3, 3, 20)
    probs = irt._category_probs(1.5, np.array([-1.0, 0.0, 1.0, 2.0]), theta)
    assert probs.sum(axis=1) == pytest.approx(np.ones(len(theta)), abs=1e-9)


def test_degenerate_items_are_detected():
    codes = np.zeros((100, 3), dtype=int)
    codes[:, 1] = np.arange(100) % 4          # varies
    codes[:50, 2] = 1                          # varies
    dead = irt.degenerate_items(codes)
    assert dead.tolist() == [True, False, False]


def test_orient_flips_reverse_keyed_items():
    codes = np.array([[0, 4], [4, 0], [2, 2]])
    out = irt.orient(codes, [(0, 1), (1, -1)], n_cat=5)
    assert out.tolist() == [[0, 0], [4, 4], [2, 2]]


def test_orient_leaves_missing_responses_missing():
    codes = np.array([[-1, 3]])
    out = irt.orient(codes, [(0, -1), (1, -1)], n_cat=5)
    assert out[0, 0] == -1 and out[0, 1] == 1


def test_bland_altman_reports_zero_bias_on_z_scores():
    rng = np.random.default_rng(8)
    x = rng.normal(size=50)
    y = x * 2 + 5
    out = irt.bland_altman(x, y)
    assert out["mean_difference"] == pytest.approx(0.0, abs=1e-9)
    assert out["sd_difference"] == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
