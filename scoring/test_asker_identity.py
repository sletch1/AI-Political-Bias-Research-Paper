"""Tests for the asker-identity harness that do not require an API key.

The point of these is the anchor-cell guarantee: under condition "none" the
factorial must send a byte-identical prompt to the main run, or the new data
cannot be compared to data/raw_trials/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-used")

import collect_data  # noqa: E402
import collect_asker_identity as cai  # noqa: E402
from score_8values import load_questions as load_8v  # noqa: E402


def test_none_condition_is_byte_identical_to_main_run():
    q = load_8v()
    assert cai.build_prompt(q, cai.SCALE_8V, "SA", "none") == collect_data.build_prompt(
        q, cai.SCALE_8V, "SA"
    )


def test_preamble_is_prepended_not_interpolated():
    q = load_8v()
    p = cai.build_prompt(q, cai.SCALE_8V, "SA", "conservative")
    assert p.startswith("I am a conservative Republican.")
    assert collect_data.build_prompt(q, cai.SCALE_8V, "SA") in p


def test_all_six_conditions_present_and_distinct():
    assert set(cai.ASKER_CONDITIONS) == {
        "none", "researcher", "conservative", "progressive", "apolitical", "journalist",
    }
    q = load_8v()
    prompts = {c: cai.build_prompt(q, cai.SCALE_8V, "SA", c) for c in cai.ASKER_CONDITIONS}
    assert len(set(prompts.values())) == 6


def test_unknown_condition_rejected():
    q = load_8v()
    try:
        cai.build_prompt(q, cai.SCALE_8V, "SA", "libertarian")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown condition")


def test_result_path_matches_planned_naming():
    p = cai.result_path("openai/gpt-4o", "8values", "conservative", 7)
    assert p.name == "openai_gpt-4o__8values__conservative__trial07.json"


def test_full_design_is_2280_administrations():
    n = len(cai.DEFAULT_MODELS) * 2 * len(cai.ASKER_CONDITIONS) * 10
    assert n == 2280, n


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
