"""Golden-answer validation for both scoring adapters.

These are invariant/property checks rather than checks against a fixed
external "known answer," since the whole point of both adapters is that they
defer to an authoritative source (the real 8Values scoring algorithm; the
real Political Compass website) rather than a hand-derived formula. Run with:

    python3 -m pytest scoring/test_scoring.py -v

The Political Compass tests require network access (they drive the live
site) and are skipped automatically if it's unreachable.
"""

import socket

import pytest

from score_8values import load_questions as load_8v_questions
from score_8values import score_8values
from score_sapplyvalues import load_questions as load_sv_questions
from score_sapplyvalues import score_sapplyvalues


def _network_available(host="www.politicalcompass.org", port=443, timeout=3):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# 8Values: exact port, so we can assert exact mathematical invariants.
# ---------------------------------------------------------------------------

def test_8values_neutral_is_exactly_50_50_50_50():
    qs = load_8v_questions()
    result = score_8values(["N"] * len(qs), qs)
    assert result == {"equality": 50.0, "peace": 50.0, "liberty": 50.0, "progress": 50.0}


def test_8values_agree_and_disagree_are_complementary():
    qs = load_8v_questions()
    agree = score_8values(["SA"] * len(qs), qs)
    disagree = score_8values(["SD"] * len(qs), qs)
    for axis in ("equality", "peace", "liberty", "progress"):
        assert round(agree[axis] + disagree[axis], 1) == 100.0


def test_8values_scores_bounded_0_100():
    qs = load_8v_questions()
    for pattern in (["SA"] * len(qs), ["SD"] * len(qs), ["N"] * len(qs)):
        result = score_8values(pattern, qs)
        for v in result.values():
            assert 0.0 <= v <= 100.0


def test_8values_question_count_matches_live_source():
    # questions_8values.json was extracted (not retyped) from
    # github.com/8values/8values.github.io's questions.js.
    assert len(load_8v_questions()) == 70


def test_8values_rejects_wrong_length_answers():
    qs = load_8v_questions()
    with pytest.raises(ValueError):
        score_8values(["SA"] * (len(qs) - 1), qs)


# ---------------------------------------------------------------------------
# SapplyValues: exact port, like 8Values, so we can assert exact invariants.
# ---------------------------------------------------------------------------

def test_sapplyvalues_neutral_is_exactly_zero():
    qs = load_sv_questions()
    result = score_sapplyvalues(["N"] * len(qs), qs)
    assert result == {"right": 0.0, "auth": 0.0, "prog": 0.0}


def test_sapplyvalues_agree_and_disagree_are_mirror_images():
    qs = load_sv_questions()
    agree = score_sapplyvalues(["SA"] * len(qs), qs)
    disagree = score_sapplyvalues(["SD"] * len(qs), qs)
    # As with Political Compass, a uniform answer pattern is politically
    # incoherent and the item pool isn't symmetric per axis (e.g. 8 right-
    # coded vs. 7 left-coded items), so all-agree need not hit +/-10. The
    # invariant that must hold is that flipping every answer flips the sign.
    for axis in ("right", "auth", "prog"):
        assert agree[axis] == pytest.approx(-disagree[axis], abs=0.01)


def test_sapplyvalues_scores_bounded_minus10_to_10():
    qs = load_sv_questions()
    for pattern in (["SA"] * len(qs), ["SD"] * len(qs), ["N"] * len(qs)):
        result = score_sapplyvalues(pattern, qs)
        for v in result.values():
            assert -10.0 <= v <= 10.0


def test_sapplyvalues_question_count_matches_live_source():
    # questions_sapplyvalues.json was extracted (not retyped) from
    # github.com/SapplyValues/SapplyValues.github.io's questions.js.
    assert len(load_sv_questions()) == 46


def test_sapplyvalues_rejects_wrong_length_answers():
    qs = load_sv_questions()
    with pytest.raises(ValueError):
        score_sapplyvalues(["SA"] * (len(qs) - 1), qs)


# ---------------------------------------------------------------------------
# Political Compass: drives the live site, so only run when reachable.
# ---------------------------------------------------------------------------

pc_available = _network_available()
pc_skip_reason = "politicalcompass.org unreachable from this environment"


@pytest.mark.skipif(not pc_available, reason=pc_skip_reason)
def test_political_compass_question_count_matches_live_site():
    from score_political_compass import load_questions as load_pc_questions
    assert len(load_pc_questions()) == 62


@pytest.mark.skipif(not pc_available, reason=pc_skip_reason)
def test_political_compass_agree_and_disagree_are_mirror_images():
    from score_political_compass import load_questions as load_pc_questions
    from score_political_compass import score_political_compass

    qs = load_pc_questions()
    agree = score_political_compass(["SA"] * len(qs), qs)
    disagree = score_political_compass(["SD"] * len(qs), qs)
    # Uniform answer patterns are politically incoherent (agreeing with every
    # left- and right-coded item at once), so the only invariant we assert is
    # symmetry: flipping every answer flips the sign of both axes.
    assert agree["economic"] == pytest.approx(-disagree["economic"], abs=0.01)
    assert agree["social"] == pytest.approx(-disagree["social"], abs=0.01)


# ---------------------------------------------------------------------------
# Pew 2026 Political Typology: bank integrity (offline) and a live smoke
# test (drives the real quiz, so only run when reachable).
# ---------------------------------------------------------------------------

def test_pew_typology_bank_has_24_items_across_21_screens():
    from score_pew_typology import load_bank
    bank = load_bank()
    assert len(bank["items"]) == 24
    assert len(bank["screens"]) == 21
    # every item index appears in exactly one screen
    flat = [i for screen in bank["screens"] for i in screen]
    assert sorted(flat) == list(range(24))


def test_pew_typology_has_nine_groups_left_to_right():
    from score_pew_typology import load_bank
    bank = load_bank()
    assert len(bank["_groups_left_to_right"]) == 9
    assert bank["_groups_left_to_right"][0] == "Leftward Progressives"
    assert bank["_groups_left_to_right"][-1] == "No Apologies Right"


def test_pew_typology_match_option_is_case_and_whitespace_insensitive():
    from score_pew_typology import _norm
    assert _norm("  Extremely  important ") == _norm("extremely important")


def test_pew_typology_norm_treats_curly_and_straight_apostrophes_the_same():
    # Regression: a model's uppercased reply uses a straight apostrophe even
    # when Pew's own option text uses a curly one, which silently failed
    # every exact match on this item in the first full collection run.
    from score_pew_typology import _norm
    model_reply = "AMERICA'S OPENNESS TO PEOPLE FROM ALL OVER THE WORLD IS ESSENTIAL"
    live_option = "America’s openness to people from all over the world is essential"
    assert _norm(model_reply) == _norm(live_option)


def test_pew_typology_extract_group_finds_the_group_after_the_heading():
    from score_pew_typology import _extract_group
    body = "some text\nYOUR BEST FIT\nTuned-Out Middle\nmore text"
    groups = ["Leftward Progressives", "Tuned-Out Middle", "No Apologies Right"]
    assert _extract_group(body, groups) == "Tuned-Out Middle"


def test_pew_typology_extract_group_raises_on_unrecognised_layout():
    from score_pew_typology import _extract_group
    with pytest.raises(RuntimeError):
        _extract_group("no such heading here", ["Tuned-Out Middle"])


pew_available = _network_available(host="www.pewresearch.org")
pew_skip_reason = "pewresearch.org unreachable from this environment"


@pytest.mark.skipif(not pew_available, reason=pew_skip_reason)
def test_pew_typology_all_first_option_reaches_a_recognised_group():
    from score_pew_typology import load_bank, score_pew_typology

    bank = load_bank()
    answers = [item["options"][0] for item in bank["items"]]
    result = score_pew_typology(answers, bank)
    assert result["group"] in bank["_groups_left_to_right"]
    assert 1 <= result["ordinal_position"] <= 9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
