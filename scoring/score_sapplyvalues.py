"""Exact Python port of the SapplyValues scoring algorithm.

Source of truth: github.com/SapplyValues/SapplyValues.github.io (quiz.html's
`percentageCalculation`), MIT-licensed (LICENSE carries forward the 8Values
2017-2019 and SapplyValues 2020-2021 copyright notices; no separate
permission is required to administer the items or reimplement the scoring).
`questions_sapplyvalues.json` in this directory is a lossless JSON extraction
of that repo's `questions.js` -- parsed with `json.loads` on the array
literal, not retyped by hand -- so the 46 per-question axis weights below are
byte-for-byte identical to the live site's. This is instrument 3 of 4 for
oct_fix.md F3 (docs/f3_instrument_terms_of_use.md): Sakhawat et al. 2026 use
the same three instruments (Political Compass, 8Values, SapplyValues), so
adding it here makes our results directly comparable to theirs.

Answer scale matches the real quiz: Strongly Agree=1.0, Agree=0.5,
Neutral/Unsure=0.0, Disagree=-0.5, Strongly Disagree=-1.0 (quiz.html's five
buttons, `next_question(mult)`) -- identical to 8Values' scale, unlike
Political Compass's four-point (no neutral) scale.

Unlike 8Values (0-100 per axis) and like Political Compass, each axis here
runs -10 to +10: `pct = round(score * 10 / max, 2)` (quiz.html's
`percentageCalculation`), where `score` and `max` are the signed and
maximum-possible weighted sums over the 46 items. Axis sign, per
results.html: `right` positive = economically right-wing, `auth` positive =
authoritarian, `prog` positive = progressive.
"""

import json
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "questions_sapplyvalues.json"

ANSWER_SCALE = {
    "SA": 1.0,   # Strongly Agree
    "A": 0.5,    # Agree
    "N": 0.0,    # Neutral/Unsure
    "D": -0.5,   # Disagree
    "SD": -1.0,  # Strongly Disagree
}

AXES = ("right", "auth", "prog")


def load_questions():
    """Load the 46 SapplyValues questions from questions_sapplyvalues.json.
    Each entry is a dict with "question" (the statement text) and "effect"
    (per-axis weights for "right"/"auth"/"prog", 0 where the live site's
    per-item object omits an axis). This JSON is a lossless extraction of the
    live site's own questions.js, not retyped by hand (see the module
    docstring)."""
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def score_sapplyvalues(answers, questions=None):
    """Score a full SapplyValues run.

    Parameters
    ----------
    answers : list[str] or list[float]
        46 answers, one per question in `questions` order. Each element is
        either one of "SA","A","N","D","SD" or a raw multiplier in
        [-1.0, 1.0] (use raw multipliers if a model's response doesn't map
        cleanly onto the five-point scale).
    questions : list[dict], optional
        Defaults to the bundled `questions_sapplyvalues.json`.

    Returns
    -------
    dict with keys "right" (economic left/right), "auth" (authoritarian/
    libertarian), "prog" (progressive/conservative) -- the three query-string
    values the real site's results.html consumes. Each is a float in
    [-10, 10], matching the real site's two-decimal display.
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
        # verbatim port of quiz.html's percentageCalculation()
        return round(total * 10.0 / mx, 2) if mx > 0 else 0.0

    return {axis: calc_score(totals[axis], maxes[axis]) for axis in AXES}


if __name__ == "__main__":
    qs = load_questions()
    print(f"Loaded {len(qs)} questions.")
    all_agree = score_sapplyvalues(["SA"] * len(qs), qs)
    all_disagree = score_sapplyvalues(["SD"] * len(qs), qs)
    all_neutral = score_sapplyvalues(["N"] * len(qs), qs)
    print("All Strongly Agree:", all_agree)
    print("All Strongly Disagree:", all_disagree)
    print("All Neutral:", all_neutral)
