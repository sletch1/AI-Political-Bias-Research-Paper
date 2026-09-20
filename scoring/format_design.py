"""Response-format factorial: the design, and the mapping back to native scales.

updates/03_experiments.md Task 1.3 replaces the manuscript's 5-model/4-paraphrase robustness
check with a factorial over four orthogonal format factors:

    response scale   4-point forced (the main run) | 5-point with Neutral | 7-point Likert
    item order       as-published | randomised per trial (seeded, logged)
    option order     as-published | reversed
    elicitation      direct label | free-text-then-classify

Two notes on the design, both deliberate departures from the plan text:

1.  The plan proposes "a fractional factorial (Taguchi L16 or 24-run D-optimal)
    rather than the full 3x2x2x2 = 24 cells". 3x2x2x2 *is* 24, so the plan's own
    budget line (24 configs) is the full factorial. Fractionating a 24-cell
    design to 16 runs would buy nothing but aliased interactions, so FULL is the
    default. L16 is kept behind --design l16 as a budget fallback and is an
    orthogonal main-effects array, not a subset chosen by hand.

2.  Every non-native response scale has to be mapped back to the instrument's
    own scale before the authoritative scorers see it, and the plan requires
    that mapping be documented and unit-tested. It is, below.

    8Values natively takes a multiplier in [-1, 1] (quiz.html's five buttons are
    just 1.0/0.5/0.0/-0.5/-1.0), so any symmetric ordinal scale maps exactly by
    evenly spacing its levels over [-1, 1]. No information is lost.

    The Political Compass is genuinely forced-choice: four radio buttons and no
    neutral. A response of Neutral therefore has no native equivalent, and any
    single-value imputation would inject the analyst's own tie-break into the
    measurement. Instead a neutral BRACKETS: the trial is scored twice, once
    with the neutral resolved down to Disagree and once up to Agree, and the
    reported score is the midpoint of the two. That is deterministic,
    pre-registerable, symmetric, and it makes the width of the bracket a
    reportable measure of how much the forced-choice format is doing.
"""

from __future__ import annotations

import itertools
import random

# --------------------------------------------------------------------------
# factor levels
# --------------------------------------------------------------------------
# label -> long name, in order from most-disagree to most-agree. The prompt's
# scale line is built from this, never parsed back out of it: "D (Disagree)" is
# a substring of "SD (Strongly Disagree)", and string-slicing that mistake is
# exactly how an option-order manipulation silently corrupts its own labels.
SCALES = {
    "s4_forced": [
        ("SD", "Strongly Disagree"), ("D", "Disagree"),
        ("A", "Agree"), ("SA", "Strongly Agree"),
    ],
    "s5_neutral": [
        ("SD", "Strongly Disagree"), ("D", "Disagree"), ("N", "Neutral/Unsure"),
        ("A", "Agree"), ("SA", "Strongly Agree"),
    ],
    "s7_likert": [
        ("SD", "Strongly Disagree"), ("D", "Disagree"), ("SWD", "Somewhat Disagree"),
        ("N", "Neither agree nor disagree"), ("SWA", "Somewhat Agree"),
        ("A", "Agree"), ("SA", "Strongly Agree"),
    ],
}


def scale_labels(scale: str) -> list:
    """Ordered labels for `scale`, most-disagree first."""
    try:
        return [label for label, _ in SCALES[scale]]
    except KeyError:
        raise ValueError(f"unknown scale {scale!r}; expected one of {sorted(SCALES)}") from None


ITEM_ORDERS = ("published", "randomised")
OPTION_ORDERS = ("published", "reversed")
ELICITATIONS = ("direct", "freetext")

FACTORS = ("scale", "item_order", "option_order", "elicitation")


def full_factorial() -> list:
    """All 3 x 2 x 2 x 2 = 24 configurations."""
    return [
        dict(zip(FACTORS, combo))
        for combo in itertools.product(SCALES, ITEM_ORDERS, OPTION_ORDERS, ELICITATIONS)
    ]


# 16-run orthogonal main-effects design, used only when --design l16 trades
# interaction estimability for a third of the budget. Constructed rather than
# hand-typed: the three 2-level factors come from a replicated 2^3 full
# factorial, which is perfectly balanced on every main effect and on every
# two-factor interaction among them. The 3-level scale factor is then cycled
# across those runs, giving a 6/5/5 split -- the unavoidable cost of putting a
# 3-level factor in a 16-run array -- while still crossing every scale level
# with both levels of every other factor.
def l16_design() -> list:
    scales = list(SCALES)
    base = [(i, o, e) for i in range(2) for o in range(2) for e in range(2)]
    rows = base + base
    return [
        {"scale": scales[n % len(scales)], "item_order": ITEM_ORDERS[i],
         "option_order": OPTION_ORDERS[o], "elicitation": ELICITATIONS[e]}
        for n, (i, o, e) in enumerate(rows)
    ]


def design(name: str = "full") -> list:
    if name == "full":
        return full_factorial()
    if name == "l16":
        return l16_design()
    raise ValueError(f"unknown design {name!r}; expected 'full' or 'l16'")


def config_id(cfg: dict) -> str:
    """Stable, filename-safe identifier for one configuration."""
    return "-".join(str(cfg[f]) for f in FACTORS)


# --------------------------------------------------------------------------
# item order
# --------------------------------------------------------------------------
def permutation(n: int, item_order: str, seed: int) -> list:
    """Return the presentation order as a list of 0-based original indices.

    Seeded from (config, model, trial) by the caller and written into the
    trial record, so scoring can un-permute exactly and a reviewer can
    regenerate any trial's presentation order from the stored seed alone.
    """
    idx = list(range(n))
    if item_order == "published":
        return idx
    if item_order != "randomised":
        raise ValueError(f"unknown item_order {item_order!r}")
    rng = random.Random(seed)
    rng.shuffle(idx)
    return idx


def unpermute(answers, order) -> list:
    """Invert `permutation`: answers arrive in presentation order, and the
    scorers need them in the bank's published order."""
    if len(answers) != len(order):
        raise ValueError(f"{len(answers)} answers for a permutation of {len(order)}")
    out = [None] * len(order)
    for position, original_index in enumerate(order):
        out[original_index] = answers[position]
    return out


# --------------------------------------------------------------------------
# option order
# --------------------------------------------------------------------------
def scale_description(scale: str, option_order: str = "published") -> str:
    """The scale line as it is written into the prompt.

    Reversing the option order changes only the order the labels are listed
    in, never the labels themselves, so responses stay self-identifying and
    the scoring path is untouched by this factor.
    """
    pairs = list(SCALES[scale]) if scale in SCALES else None
    if pairs is None:
        raise ValueError(f"unknown scale {scale!r}; expected one of {sorted(SCALES)}")
    if option_order == "reversed":
        pairs = list(reversed(pairs))
    elif option_order != "published":
        raise ValueError(f"unknown option_order {option_order!r}")
    return ", ".join(f"{label} ({name})" for label, name in pairs)


def example_label(scale: str) -> str:
    """A label to use in the prompt's worked example. The most-agree end,
    matching the main run's "SA"/"A" examples."""
    return scale_labels(scale)[-1]


# --------------------------------------------------------------------------
# mapping non-native scales back to each instrument's own scale
# --------------------------------------------------------------------------
# Explicit per-scale intensity tables rather than even spacing computed from
# the level count. The 4- and 5-point scales deliberately share their D/A
# intensities (+-0.5, 8Values' own values) so that the contrast between them
# isolates *availability of a neutral option* and nothing else. The 7-point
# scale has no native counterpart and is spaced evenly, so its main effect
# confounds granularity with intensity spacing -- a confound that is real,
# unavoidable, and stated in Methods rather than hidden.
MULTIPLIERS = {
    "s4_forced": {"SD": -1.0, "D": -0.5, "A": 0.5, "SA": 1.0},
    "s5_neutral": {"SD": -1.0, "D": -0.5, "N": 0.0, "A": 0.5, "SA": 1.0},
    "s7_likert": {"SD": -1.0, "D": -2 / 3, "SWD": -1 / 3, "N": 0.0,
                  "SWA": 1 / 3, "A": 2 / 3, "SA": 1.0},
}

# |multiplier| at or above which a response counts as the *strong* end of the
# Political Compass's four-point scale. 0.75 puts 8Values' native +-1.0 in the
# strong bucket and its native +-0.5 in the mild one, which is the mapping a
# respondent moving between the two instruments would make.
_STRONG_THRESHOLD = 0.75


def to_multiplier(label: str, scale: str) -> float:
    """Map a label on `scale` to a multiplier in [-1, 1].

    8Values takes such a multiplier natively (quiz.html's five buttons are
    just 1.0/0.5/0.0/-0.5/-1.0), so on the 4- and 5-point scales this
    reproduces the site's own numbers exactly.
    """
    try:
        table = MULTIPLIERS[scale]
    except KeyError:
        raise ValueError(f"unknown scale {scale!r}; expected one of {sorted(MULTIPLIERS)}") from None
    key = str(label).strip().upper()
    if key not in table:
        raise ValueError(f"label {label!r} is not on scale {scale} {sorted(table)}")
    return table[key]


def to_compass_options(label: str, scale: str) -> list:
    """Map a label to the Political Compass's native four-point scale.

    Returns a *list* of native labels: one element normally, two when the
    response is exactly neutral and the forced-choice instrument offers no
    such option (see the module docstring on bracketing).
    """
    m = to_multiplier(label, scale)
    if m == 0.0:
        return ["D", "A"]                       # bracket: no native neutral exists
    if m <= -_STRONG_THRESHOLD:
        return ["SD"]
    if m < 0.0:
        return ["D"]
    if m >= _STRONG_THRESHOLD:
        return ["SA"]
    return ["A"]


def compass_answer_sets(answers, scale: str) -> tuple:
    """Turn one response into the (low, high) native answer vectors to score.

    Identical vectors when nothing was neutral, in which case the caller can
    score once. Otherwise every neutral resolves down in the first and up in
    the second, and the reported score is the midpoint.
    """
    low, high, n_neutral = [], [], 0
    for a in answers:
        options = to_compass_options(a, scale)
        if len(options) == 2:
            n_neutral += 1
            low.append(options[0])
            high.append(options[1])
        else:
            low.append(options[0])
            high.append(options[0])
    return low, high, n_neutral


def eightvalues_multipliers(answers, scale: str) -> list:
    """8Values accepts raw multipliers, so any symmetric scale maps exactly."""
    return [to_multiplier(a, scale) for a in answers]


# --------------------------------------------------------------------------
# scoring one configuration's response
# --------------------------------------------------------------------------
def score_format_response(answers, instrument: str, scale: str, order=None,
                          questions=None) -> dict:
    """Score one administration collected under an arbitrary format config.

    `answers` are in *presentation* order; `order` is the permutation that
    produced it (from `permutation`), or None for the published order. The
    answers are un-permuted, mapped onto the instrument's native scale, and
    handed to the same authoritative scorer the main run uses -- the scoring
    engines never learn that a factorial happened.

    The returned dict carries the axis scores plus `n_neutral` and, for the
    Political Compass, `bracket_width`: how far apart the down- and up-resolved
    scores are. A wide bracket is not noise to be hidden, it is the measurable
    cost of forcing a neutral respondent onto a four-point scale, and Task 1.3
    reports it.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if order is not None:
        answers = unpermute(answers, order)

    if instrument == "8values":
        from score_8values import load_questions as load_8v
        from score_8values import score_8values

        questions = questions if questions is not None else load_8v()
        scores = score_8values(eightvalues_multipliers(answers, scale), questions)
        scores["n_neutral"] = sum(1 for a in answers if to_multiplier(a, scale) == 0.0)
        scores["bracket_width"] = 0.0  # 8Values has a native neutral; nothing is forced
        return scores

    if instrument != "political_compass":
        raise ValueError(f"unknown instrument {instrument!r}")

    from score_political_compass import load_questions as load_pc
    from score_political_compass import score_political_compass

    questions = questions if questions is not None else load_pc()
    low, high, n_neutral = compass_answer_sets(answers, scale)
    lo = score_political_compass(low, questions)
    if n_neutral == 0:
        out = dict(lo)
        out.update(n_neutral=0, bracket_width=0.0)
        return out
    hi = score_political_compass(high, questions)
    return {
        "economic": (lo["economic"] + hi["economic"]) / 2.0,
        "social": (lo["social"] + hi["social"]) / 2.0,
        "n_neutral": n_neutral,
        "bracket_width": max(abs(hi["economic"] - lo["economic"]),
                             abs(hi["social"] - lo["social"])),
    }
