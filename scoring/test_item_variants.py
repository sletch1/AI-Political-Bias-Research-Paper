"""Tests for the Task 1.2 item-variant banks and their scoring.

The contamination control is only as good as its banks. A silently broken
inversion does not fail loudly at collection time -- it produces a plausible
number that means nothing, after the API budget has been spent. These tests are
the guard.

    python3 -m pytest scoring/test_item_variants.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_item_variants as biv  # noqa: E402
import variant_scoring as vs  # noqa: E402
from score_8values import load_questions as load_8v  # noqa: E402
from score_8values import score_8values  # noqa: E402


# ---------------------------------------------------------------------------
# bank structure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("instrument,n", [("8values", 70), ("political_compass", 62)])
@pytest.mark.parametrize("condition", ["inverted", "paraphrased"])
def test_variant_bank_has_same_length_as_original(instrument, n, condition):
    assert len(vs.load_bank(instrument, condition)) == n


@pytest.mark.parametrize("condition", ["inverted", "paraphrased"])
def test_8values_variant_banks_preserve_scoring_weights(condition):
    original = vs.load_bank("8values", "original")
    variant = vs.load_bank("8values", condition)
    for o, v in zip(original, variant):
        assert o["effect"] == v["effect"], (
            "variant banks must carry the originals' weights unchanged; the "
            "inversion is applied to the answer, never to the key"
        )


@pytest.mark.parametrize("condition", ["inverted", "paraphrased"])
def test_compass_variant_banks_preserve_form_fields(condition):
    original = vs.load_bank("political_compass", "original")
    variant = vs.load_bank("political_compass", condition)
    for o, v in zip(original, variant):
        # The live site's radio groups are keyed by these names. Change them and
        # the browser adapter cannot submit the form at all.
        assert o["name"] == v["name"]
        assert o["page"] == v["page"]


@pytest.mark.parametrize("instrument", ["8values", "political_compass"])
@pytest.mark.parametrize("condition", ["inverted", "paraphrased"])
def test_every_variant_item_differs_from_its_original(instrument, condition):
    original = vs.load_bank(instrument, "original")
    variant = vs.load_bank(instrument, condition)
    for i, (o, v) in enumerate(zip(original, variant), 1):
        assert o["question"] != v["question"], f"{instrument} item {i} was not rewritten"
        assert v["source_question"] == o["question"], f"{instrument} item {i} lost provenance"


def test_source_file_stays_in_sync_with_the_live_banks():
    # build_item_variants raises if an entry's `original` no longer matches the
    # bank, which is what catches an item bank re-scraped from a changed site.
    biv.main.__wrapped__ if hasattr(biv.main, "__wrapped__") else None
    source = biv.load_source()
    for key, filename, _ in biv.INSTRUMENTS:
        originals = biv.load_bank(filename)
        biv.build_bank(originals, source[key], "inverted")  # raises on drift


# ---------------------------------------------------------------------------
# mirroring and scoring
# ---------------------------------------------------------------------------
def test_mirror_is_an_involution():
    for instrument, table in vs.MIRROR.items():
        for label in table:
            assert vs.mirror_answers(vs.mirror_answers([label], instrument),
                                     instrument) == [label]


def test_mirror_rejects_off_scale_labels():
    with pytest.raises(ValueError):
        vs.mirror_answers(["MAYBE"], "8values")
    # The Political Compass has no neutral; a stray "N" must not pass silently.
    with pytest.raises(ValueError):
        vs.mirror_answers(["N"], "political_compass")


def test_8values_neutral_mirrors_to_itself():
    assert vs.mirror_answers(["N"], "8values") == ["N"]


@pytest.mark.parametrize("original,inverted", [("SA", "SD"), ("A", "D"), ("N", "N")])
def test_inverted_condition_reproduces_original_score(original, inverted):
    """The invariance the whole contamination test rests on.

    Answering the inverted bank in the mirror of how you answered the original
    is, semantically, the same set of positions. It must score identically --
    otherwise a non-zero Contamination Index would just be measuring our own
    scoring bug.
    """
    n = len(load_8v())
    a = vs.score_response([original] * n, "8values", "original")
    b = vs.score_response([inverted] * n, "8values", "inverted")
    assert a == b


def test_inverted_scoring_matches_hand_computed_mirror():
    n = len(load_8v())
    answers = (["SA", "D", "N"] * n)[:n]
    expected = score_8values([{"SA": "SD", "D": "A", "N": "N"}[a] for a in answers], load_8v())
    assert vs.score_response(answers, "8values", "inverted") == expected


def test_paraphrased_condition_is_scored_like_the_original():
    n = len(load_8v())
    answers = (["A", "SD", "N", "SA"] * n)[:n]
    assert (vs.score_response(answers, "8values", "paraphrased")
            == vs.score_response(answers, "8values", "original"))


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        vs.score_response(["A"] * 70, "8values", "reversed")


# ---------------------------------------------------------------------------
# contamination index
# ---------------------------------------------------------------------------
def test_contamination_index_is_zero_under_invariance():
    assert vs.contamination_index(62.5, 62.5, "8values") == 0.0
    assert vs.contamination_index(-3.4, -3.4, "political_compass") == 0.0


def test_contamination_index_is_normalised_by_instrument_range():
    # Ten points on the Political Compass ([-10, 10], range 20) and fifty on
    # 8Values ([0, 100]) are the same fraction of scale, and must compare equal.
    assert vs.contamination_index(0, 10, "political_compass") == pytest.approx(0.5)
    assert vs.contamination_index(25, 75, "8values") == pytest.approx(0.5)


def test_contamination_index_is_symmetric():
    assert (vs.contamination_index(30, 70, "8values")
            == vs.contamination_index(70, 30, "8values"))


# ---------------------------------------------------------------------------
# structural validation of the authored text
# ---------------------------------------------------------------------------
def test_validator_rejects_an_identical_inversion():
    entries = [{"index": 1, "original": "X.", "inverted": "X.", "paraphrased": "Y."}]
    with pytest.raises(ValueError):
        biv.validate("test", [{"question": "X."}], entries)


def test_validator_flags_a_paraphrase_that_kept_the_original_verbatim():
    original = "It is important that we maintain the traditions of our past always."
    entries = [{"index": 1, "original": original,
                "inverted": "We should abandon the traditions of our past.",
                "paraphrased": original + " Indeed."}]
    warnings = biv.validate("test", [{"question": original}], entries)
    assert any("verbatim match may survive" in w for w in warnings)


def test_negation_heuristic_does_not_fire_on_ordinary_words():
    # "unions", "unemployment" and "important" are not negations; an over-eager
    # pattern buries the real double-negative warnings in noise.
    assert biv.count_negations("I oppose regional unions, such as the European Union.") <= 1
    assert biv.count_negations("Controlling unemployment is more important.") == 0


def test_negation_heuristic_finds_a_real_double_negative():
    assert biv.count_negations("It is not true that nobody should be untaxed.") >= 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
