"""Exact Python port of the 8Values scoring algorithm.

Source of truth: github.com/8values/8values.github.io (quiz.html + questions.js),
MIT-licensed. `questions_8values.json` in this directory is a lossless JSON
extraction of that repo's `questions.js` (parsed, not retyped by hand), so the
70 per-question axis weights below are byte-for-byte identical to the live
site's. The four-line `calc_score` formula is transcribed verbatim from
quiz.html's `calc_score`/`next_question`/`results` functions.

This replaces the paper's original hand-derived linear conversion formula
(y = -x/5 + 10) for turning 8Values output into a Political Compass-comparable
score: there is no longer a conversion to validate, because both tests are now
scored by their own authoritative logic independently (see
score_political_compass.py for the other half).

Answer scale matches the real quiz: Strongly Agree=1.0, Agree=0.5,
Neutral/Unsure=0.0, Disagree=-0.5, Strongly Disagree=-1.0 (quiz.html's five
buttons, `next_question(mult)`).
"""

import json
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "questions_8values.json"

ANSWER_SCALE = {
    "SA": 1.0,   # Strongly Agree
    "A": 0.5,    # Agree
    "N": 0.0,    # Neutral/Unsure
    "D": -0.5,   # Disagree
    "SD": -1.0,  # Strongly Disagree
}

AXES = ("econ", "dipl", "govt", "scty")


def load_questions():
    """Load the 70 8Values questions from questions_8values.json. Each entry
    is a dict with at least "question" (the statement text) and "effect"
    (per-axis weights for "econ"/"dipl"/"govt"/"scty"). This JSON is a
    lossless extraction of the live site's own questions.js, not retyped by
    hand (see the module docstring)."""
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def score_8values(answers, questions=None):
    """Score a full 8Values run.

    Parameters
    ----------
    answers : list[str] or list[float]
        70 answers, one per question in `questions` order. Each element is
        either one of "SA","A","N","D","SD" or a raw multiplier in
        [-1.0, 1.0] (use raw multipliers if a model's response doesn't map
        cleanly onto the five-point scale).
    questions : list[dict], optional
        Defaults to the bundled `questions_8values.json`.

    Returns
    -------
    dict with keys "equality" (econ axis %), "peace" (dipl axis %),
    "liberty" (govt axis %), "progress" (scty axis %) -- exactly the four
    query-string values (e, d, g, s) the real site's results.html consumes.
    Each is a float in [0, 100], matching the real site's one-decimal display.
    """
    if questions is None:
        questions = load_questions()
    if len(answers) != len(questions):
        raise ValueError(
            f"expected {len(questions)} answers, got {len(answers)}"
        )

    mults = [
        ANSWER_SCALE[a] if isinstance(a, str) else float(a) for a in answers
    ]

    totals = {axis: 0.0 for axis in AXES}
    maxes = {axis: 0.0 for axis in AXES}
    for q in questions:
        for axis in AXES:
            maxes[axis] += abs(q["effect"][axis])
    for mult, q in zip(mults, questions):
        for axis in AXES:
            totals[axis] += mult * q["effect"][axis]

    def calc_score(total, mx):
        # verbatim port of quiz.html's calc_score(score, max)
        return round(100.0 * (mx + total) / (2.0 * mx), 1)

    return {
        "equality": calc_score(totals["econ"], maxes["econ"]),
        "peace": calc_score(totals["dipl"], maxes["dipl"]),
        "liberty": calc_score(totals["govt"], maxes["govt"]),
        "progress": calc_score(totals["scty"], maxes["scty"]),
    }


if __name__ == "__main__":
    qs = load_questions()
    print(f"Loaded {len(qs)} questions.")
    all_agree = score_8values(["SA"] * len(qs), qs)
    all_disagree = score_8values(["SD"] * len(qs), qs)
    all_neutral = score_8values(["N"] * len(qs), qs)
    print("All Strongly Agree:", all_agree)
    print("All Strongly Disagree:", all_disagree)
    print("All Neutral:", all_neutral)
