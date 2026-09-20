"""Tests for the Task 1.3 response-format design and its scale mappings.

updates/03_experiments.md Task 1.3 requires the mapping from non-native response scales
back to each instrument's own scale to be documented *and unit-tested*. It is
the part of the factorial most likely to be silently wrong: a bad mapping still
produces numbers, and those numbers still correlate with something.

    python3 -m pytest scoring/test_format_design.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import format_design as fd  # noqa: E402
from score_8values import load_questions as load_8v  # noqa: E402
from score_8values import score_8values  # noqa: E402


# ---------------------------------------------------------------------------
# the design
# ---------------------------------------------------------------------------
def test_full_factorial_is_24_cells():
    # updates/03_experiments.md budgets 24 configs. 3 x 2 x 2 x 2 = 24: the "full" design and the
    # plan's "fractional" design are the same size, which is why FULL is default.
    assert len(fd.full_factorial()) == 24


def test_full_factorial_cells_are_distinct():
    ids = [fd.config_id(c) for c in fd.full_factorial()]
    assert len(set(ids)) == len(ids)


def test_l16_is_16_runs_and_balanced_on_the_two_level_factors():
    rows = fd.l16_design()
    assert len(rows) == 16
    for factor, levels in (("item_order", fd.ITEM_ORDERS),
                           ("option_order", fd.OPTION_ORDERS),
                           ("elicitation", fd.ELICITATIONS)):
        counts = [sum(r[factor] == lvl for r in rows) for lvl in levels]
        assert counts == [8, 8], f"{factor} unbalanced: {counts}"


def test_l16_covers_every_scale_level():
    assert {r["scale"] for r in fd.l16_design()} == set(fd.SCALES)


def test_l16_runs_are_distinct_and_are_a_subset_of_the_full_factorial():
    rows = fd.l16_design()
    ids = [fd.config_id(r) for r in rows]
    assert len(set(ids)) == 16, "a repeated run wastes budget without adding information"
    assert set(ids) <= {fd.config_id(c) for c in fd.full_factorial()}


def test_l16_crosses_every_scale_level_with_both_levels_of_every_other_factor():
    rows = fd.l16_design()
    for scale in fd.SCALES:
        sub = [r for r in rows if r["scale"] == scale]
        for factor, levels in (("item_order", fd.ITEM_ORDERS),
                               ("option_order", fd.OPTION_ORDERS),
                               ("elicitation", fd.ELICITATIONS)):
            assert {r[factor] for r in sub} == set(levels), (
                f"{scale} never appears with both levels of {factor}; its main effect "
                "would be aliased with that factor"
            )


def test_unknown_design_rejected():
    with pytest.raises(ValueError):
        fd.design("plackett-burman")


# ---------------------------------------------------------------------------
# item order
# ---------------------------------------------------------------------------
def test_published_order_is_the_identity():
    assert fd.permutation(10, "published", 1) == list(range(10))


def test_randomised_order_is_a_permutation_and_is_seeded():
    a = fd.permutation(70, "randomised", 12345)
    b = fd.permutation(70, "randomised", 12345)
    c = fd.permutation(70, "randomised", 54321)
    assert sorted(a) == list(range(70))
    assert a == b, "same seed must reproduce the same presentation order"
    assert a != c


def test_unpermute_inverts_permutation():
    order = fd.permutation(20, "randomised", 7)
    original = [f"item{i}" for i in range(20)]
    presented = [original[i] for i in order]
    assert fd.unpermute(presented, order) == original


def test_unpermute_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        fd.unpermute(["a", "b"], [0, 1, 2])


# ---------------------------------------------------------------------------
# option order
# ---------------------------------------------------------------------------
def test_reversing_option_order_reverses_labels_without_corrupting_them():
    published = fd.scale_description("s5_neutral", "published")
    reversed_ = fd.scale_description("s5_neutral", "reversed")
    assert published.startswith("SD (Strongly Disagree)")
    assert reversed_.startswith("SA (Strongly Agree)")
    # The regression this guards: "D (" is a substring of "SD (", so slicing the
    # description string mislabels D as "Strongly Disagree".
    assert "D (Disagree)" in reversed_
    assert "D (Strongly Disagree)" not in reversed_.replace("SD (Strongly Disagree)", "")


def test_option_order_preserves_the_label_set():
    for scale in fd.SCALES:
        for order in fd.OPTION_ORDERS:
            desc = fd.scale_description(scale, order)
            for label in fd.scale_labels(scale):
                assert f"{label} (" in desc


def test_unknown_option_order_rejected():
    with pytest.raises(ValueError):
        fd.scale_description("s4_forced", "shuffled")


# ---------------------------------------------------------------------------
# scale mapping -- the part updates/03_experiments.md explicitly requires tested
# ---------------------------------------------------------------------------
def test_multipliers_are_monotone_and_span_the_full_range():
    for scale in fd.SCALES:
        values = [fd.to_multiplier(l, scale) for l in fd.scale_labels(scale)]
        assert values == sorted(values)
        assert values[0] == -1.0 and values[-1] == 1.0


def test_multipliers_are_symmetric_about_zero():
    for scale in fd.SCALES:
        values = [fd.to_multiplier(l, scale) for l in fd.scale_labels(scale)]
        assert values == pytest.approx([-v for v in reversed(values)])


def test_five_point_scale_reproduces_8values_native_scoring_exactly():
    # s5_neutral IS the 8Values scale. If it did not reproduce the site's own
    # numbers, the whole factorial would be incomparable to the baseline.
    qs = load_8v()
    for label in ("SA", "A", "N", "D", "SD"):
        native = score_8values([label] * len(qs), qs)
        mapped = fd.score_format_response([label] * len(qs), "8values", "s5_neutral")
        for axis in native:
            assert mapped[axis] == native[axis]


def test_four_and_five_point_scales_share_their_D_and_A_intensities():
    # Deliberate: it makes the s4-vs-s5 contrast isolate neutral availability
    # rather than confounding it with a change of intensity spacing.
    for label in ("SD", "D", "A", "SA"):
        assert fd.to_multiplier(label, "s4_forced") == fd.to_multiplier(label, "s5_neutral")


def test_off_scale_label_is_rejected_rather_than_coerced():
    with pytest.raises(ValueError):
        fd.to_multiplier("N", "s4_forced")       # the 4-point scale has no neutral
    with pytest.raises(ValueError):
        fd.to_multiplier("MAYBE", "s7_likert")


# ---------------------------------------------------------------------------
# forced-choice bracketing
# ---------------------------------------------------------------------------
def test_neutral_brackets_on_the_political_compass():
    assert fd.to_compass_options("N", "s5_neutral") == ["D", "A"]
    assert fd.to_compass_options("N", "s7_likert") == ["D", "A"]


def test_non_neutral_labels_map_to_exactly_one_native_option():
    for scale in fd.SCALES:
        for label in fd.scale_labels(scale):
            options = fd.to_compass_options(label, scale)
            assert len(options) == (2 if fd.to_multiplier(label, scale) == 0.0 else 1)
            assert all(o in ("SD", "D", "A", "SA") for o in options)


def test_compass_mapping_preserves_sign():
    for scale in fd.SCALES:
        for label in fd.scale_labels(scale):
            m = fd.to_multiplier(label, scale)
            if m == 0:
                continue
            options = fd.to_compass_options(label, scale)
            assert (options[0] in ("A", "SA")) == (m > 0)


def test_answer_sets_are_identical_when_nothing_was_neutral():
    low, high, n = fd.compass_answer_sets(["A", "SD", "SA", "D"], "s5_neutral")
    assert low == high
    assert n == 0


def test_answer_sets_bracket_every_neutral():
    low, high, n = fd.compass_answer_sets(["N", "A", "N"], "s5_neutral")
    assert n == 2
    assert low == ["D", "A", "D"]
    assert high == ["A", "A", "A"]


# ---------------------------------------------------------------------------
# end-to-end scoring
# ---------------------------------------------------------------------------
def test_scoring_unpermutes_before_scoring():
    qs = load_8v()
    n = len(qs)
    answers = (["SA", "D", "N", "A", "SD"] * n)[:n]
    order = fd.permutation(n, "randomised", 999)
    presented = [answers[i] for i in order]
    assert (fd.score_format_response(presented, "8values", "s5_neutral", order=order)
            == fd.score_format_response(answers, "8values", "s5_neutral"))


def test_8values_bracket_width_is_zero_because_it_has_a_native_neutral():
    n = len(load_8v())
    out = fd.score_format_response(["N"] * n, "8values", "s5_neutral")
    assert out["n_neutral"] == n
    assert out["bracket_width"] == 0.0


def test_unknown_instrument_rejected():
    with pytest.raises(ValueError):
        fd.score_format_response(["A"], "sapply_values", "s4_forced")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
