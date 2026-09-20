"""Scoring helpers for the Task 1.2 item-variant conditions.

The contamination control (updates/03_experiments.md Task 1.2) needs an inverted item bank whose
scores are directly comparable to the original bank's. The plan's step 4 is
emphatic that the scoring engines must not change, so that every condition
stays comparable to the published 2,280-administration baseline. This module
gets that by flipping the *answer* rather than the *key*:

    a Strongly Agree to an inverted item is, by construction, semantically
    equivalent to a Strongly Disagree to the original item

so an inverted-condition response is scored by mirroring each answer and
handing the result to the unmodified score_8values / score_political_compass
adapter together with the unmodified original bank. For Political Compass this
matters twice over: the live site's form fields belong to the original items,
so the browser adapter keeps working unchanged.

Under genuine item-by-item answering the inverted condition should reproduce
the original condition's score. It is the *gap* between them that measures
instrument recognition, which is what CONTAMINATION_INDEX quantifies.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_8values import load_questions as load_8v_questions  # noqa: E402
from score_8values import score_8values  # noqa: E402

HERE = Path(__file__).resolve().parent

# Each instrument's answer scale mirrored about its own midpoint. 8Values has
# a true neutral, which maps to itself; Political Compass is forced-choice on
# four points with no neutral.
MIRROR_8V = {"SA": "SD", "A": "D", "N": "N", "D": "A", "SD": "SA"}
MIRROR_PC = {"SA": "SD", "A": "D", "D": "A", "SD": "SA"}
MIRROR = {"8values": MIRROR_8V, "political_compass": MIRROR_PC}

BANKS = {
    ("8values", "original"): "questions_8values.json",
    ("8values", "inverted"): "questions_8values_inverted.json",
    ("8values", "paraphrased"): "questions_8values_paraphrased.json",
    ("political_compass", "original"): "questions_political_compass.json",
    ("political_compass", "inverted"): "questions_political_compass_inverted.json",
    ("political_compass", "paraphrased"): "questions_political_compass_paraphrased.json",
}

CONDITIONS = ("original", "inverted", "paraphrased")

# Full width of each instrument's axis scale, used to put the contamination
# index on a common 0-1 footing across two instruments with different units.
INSTRUMENT_RANGE = {"political_compass": 20.0, "8values": 100.0}


def load_bank(instrument: str, condition: str) -> list:
    """Load the question bank for one (instrument, condition) cell."""
    import json

    try:
        filename = BANKS[(instrument, condition)]
    except KeyError:
        raise ValueError(
            f"no bank for instrument={instrument!r} condition={condition!r}; "
            f"expected one of {sorted(BANKS)}"
        ) from None
    path = HERE / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing -- run build_item_variants.py first"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mirror_answers(answers, instrument: str) -> list:
    """Map each answer to its mirror on that instrument's scale.

    Raises on an unrecognised label rather than silently dropping it: a
    mis-parsed answer that quietly became a neutral would bias the
    contamination estimate toward zero, i.e. toward "no contamination",
    which is the direction that would flatter the result.
    """
    table = MIRROR[instrument]
    out = []
    for a in answers:
        label = str(a).strip().upper()
        if label not in table:
            raise ValueError(
                f"answer {a!r} is not on the {instrument} scale {sorted(table)}"
            )
        out.append(table[label])
    return out


def score_response(answers, instrument: str, condition: str, questions=None):
    """Score one administration, aligned to the original bank's orientation.

    `original` and `paraphrased` preserve item polarity, so they are scored
    as-is. `inverted` reverses polarity, so its answers are mirrored first.
    Either way the score returned is on the same axis, in the same direction,
    as the published baseline, and is produced by the same untouched engine.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    if condition == "inverted":
        answers = mirror_answers(answers, instrument)
    # Inverted/paraphrased banks carry the originals' scoring metadata
    # unchanged, so the original bank is the correct key in every condition.
    if questions is None:
        questions = load_bank(instrument, "original")
    if instrument == "8values":
        return score_8values(answers, questions)
    from score_political_compass import score_political_compass  # imported late: needs playwright

    return score_political_compass(answers, questions)


def contamination_index(mean_original: float, mean_inverted_aligned: float,
                        instrument: str) -> float:
    """Contamination Index for one (model, axis) cell, in [0, 1].

        CI = |mean(original) - mean(inverted, aligned)| / instrument range

    Zero under genuine item-by-item answering: mirroring the items and
    mirroring the answers should cancel, leaving the same position. A non-zero
    index means the model's answer depends on the *form* of the item, not only
    its content -- the signature of instrument recognition (updates/03_experiments.md Task 1.2,
    after Bianchi et al. 2026).

    Deviation from the original design, deliberate and reported in Methods: the plan gives
    ``|mean(original) + mean(inverted_flipped)| / 2``, which vanishes only on a
    zero-centred scale. That holds for the Political Compass ([-10, 10]) but
    not for 8Values ([0, 100], neutral at 50), where the additive form is ~50
    even under perfect invariance. Scoring the inverted condition back into the
    original's orientation and differencing gives the same quantity on the
    Compass while remaining interpretable on 8Values, and normalising by the
    instrument range makes the two comparable.
    """
    return abs(mean_original - mean_inverted_aligned) / INSTRUMENT_RANGE[instrument]


if __name__ == "__main__":
    q_orig = load_8v_questions()
    n = len(q_orig)
    # Invariance demo: an inverted-bank response that mirrors an original-bank
    # response must land on the same score.
    original = ["A"] * n
    inverted = ["D"] * n  # the mirror of "A"
    s_o = score_response(original, "8values", "original")
    s_i = score_response(inverted, "8values", "inverted")
    print("original condition :", s_o)
    print("inverted condition :", s_i)
    print("contamination index:", {
        axis: round(contamination_index(s_o[axis], s_i[axis], "8values"), 6)
        for axis in s_o
    })
