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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
