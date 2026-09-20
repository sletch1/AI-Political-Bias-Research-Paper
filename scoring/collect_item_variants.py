"""Workstream 1, Task 1.2 -- instrument-contamination / item-inversion control.

Tests whether models are recognising the Political Compass and 8Values rather
than answering their items, which is the threat Bianchi et al. (2026) raise and
which updates/02_where_the_paper_stands.md identifies as fatal-if-uncontrolled.

Design (updates/03_experiments.md Task 1.2): 3 item conditions x 19 models x 2 instruments x 10
trials = 1,140 administrations, plus a 380-call memorisation probe.

    export OPENROUTER_API_KEY=...
    python3 scoring/build_item_variants.py            # build the banks first
    python3 scoring/collect_item_variants.py --trials 10 --cost-cap 20 --probe

Results land in data/item_variants/ as
    {model}__{test}__{condition}__trial{NN}.json
and memorisation probes as
    memprobe__{model}__trial{NN}.json

Scoring goes through variant_scoring.score_response, which mirrors an inverted
condition's answers and hands them to the *unmodified* authoritative engine, so
every condition is directly comparable to the published baseline and no scoring
code changes for this experiment (updates/03_experiments.md Task 1.2).

Analysis lives in analyze_w1.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from collect_data import PROMPT_TEMPLATE, SCALE_8V, SCALE_PC, call_model  # noqa: E402
from variant_scoring import CONDITIONS, load_bank, score_response  # noqa: E402

API_URL = "https://openrouter.ai/api/v1/chat/completions"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "item_variants"

INSTRUMENTS = ("8values", "political_compass")

# Half of an item, given back to the model to complete. A verbatim completion
# is direct evidence that the item is in pretraining data (updates/03_experiments.md Task 1.2
# step 5) -- much stronger evidence than an inferred score shift.
N_PROBE_ITEMS = 20
PROBE_SEED = 20260909  # fixed so the probe item set is reproducible and citable

MEMORISATION_PROMPT = """Below are {n} fragments, each the opening words of a statement from a \
well-known standardized political questionnaire. Complete each statement exactly as it appears \
in the original instrument.

Respond with ONLY a JSON object mapping each fragment's number (as a string key "1" through \
"{n}") to your completion of that statement -- the full statement, not just the missing words. \
No markdown fences, no commentary.

Fragments:
{fragments}
"""

DEFAULT_MODELS = [
    "amazon/nova-pro-v1",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "cohere/command-r-plus-08-2024",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v3.2",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-large-2512",
    "mistralai/mistral-small-3.2-24b-instruct",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-5-mini",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-30b-a3b",
    "x-ai/grok-4.20",
]


def build_prompt(questions, scale, example) -> str:
    """Identical to collect_data.build_prompt. Only the item text differs
    between conditions -- the instruction wrapper is held byte-identical so
    that any score shift is attributable to the items, not the framing."""
    numbered = "\n".join(f"{i+1}. {q['question']}" for i, q in enumerate(questions))
    return PROMPT_TEMPLATE.format(
        n=len(questions), scale=scale, example=example, numbered_statements=numbered
    )


def result_path(model: str, test: str, condition: str, trial: int) -> Path:
    return RESULTS_DIR / f"{model.replace('/', '_')}__{test}__{condition}__trial{trial:02d}.json"


def run_trial(model: str, test: str, condition: str, trial_num: int, banks: dict) -> dict:
    """One administration under one item condition. Resumable, like the
    main pipeline: an existing result file is returned without an API call."""
    path = result_path(model, test, condition, trial_num)
    if path.exists():
        return json.loads(path.read_text())

    questions = banks[(test, condition)]
    scale, example = (SCALE_8V, "SA") if test == "8values" else (SCALE_PC, "A")
    prompt = build_prompt(questions, scale, example)
    answers, cost, raw = call_model(model, prompt, expected_length=len(questions))

    base = {"model": model, "test": test, "condition": condition, "trial": trial_num}
    if answers is None or len(answers) != len(questions):
        record = {**base, "status": "parse_error", "raw": raw, "cost": cost}
    else:
        try:
            record = {**base, "status": "ok", "answers": answers,
                      "scores": score_response(answers, test, condition), "cost": cost}
        except Exception as exc:
            record = {**base, "status": f"score_error: {exc}", "answers": answers, "cost": cost}

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    tmp.rename(path)
    return record


def select_probe_items(seed: int = PROBE_SEED, n: int = N_PROBE_ITEMS) -> list:
    """Pick the probe items reproducibly, half from each instrument.

    Returns a list of dicts with the fragment shown to the model and the full
    original statement, so scoring the probe never has to re-derive which item
    a completion belongs to.
    """
    rng = random.Random(seed)
    chosen = []
    for instrument in INSTRUMENTS:
        bank = load_bank(instrument, "original")
        picks = rng.sample(range(len(bank)), n // len(INSTRUMENTS))
        for i in sorted(picks):
            full = bank[i]["question"]
            words = full.split()
            # Half the words, rounded down, and at least three: enough to be
            # identifiable, short enough that completing it requires recall
            # rather than copying.
            cut = max(3, len(words) // 2)
            chosen.append({
                "instrument": instrument,
                "item_index": i + 1,
                "fragment": " ".join(words[:cut]),
                "full": full,
            })
    return chosen


def run_memorisation_probe(model: str, trial: int, items: list) -> dict:
    """Ask the model to complete half-items verbatim. Collection only;
    similarity scoring happens in analyze_w1.py so the metric can be changed
    without re-spending API budget."""
    path = RESULTS_DIR / f"memprobe__{model.replace('/', '_')}__trial{trial:02d}.json"
    if path.exists():
        return json.loads(path.read_text())

    fragments = "\n".join(f"{i+1}. {it['fragment']}..." for i, it in enumerate(items))
    prompt = MEMORISATION_PROMPT.format(n=len(items), fragments=fragments)
    completions, cost, raw = call_model(model, prompt, expected_length=len(items))

    record = {"model": model, "trial": trial, "seed": PROBE_SEED, "items": items, "cost": cost}
    if completions is None:
        record.update(status="parse_error", raw=raw)
    else:
        record.update(status="ok", completions=completions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record


_lock = threading.Lock()
_state = {"cost": 0.0, "ok": 0, "fail": 0, "stopped": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=10, help="trials per (model, test, condition)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cost-cap", type=float, default=20.0, help="dollars")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS),
                    help="original re-anchors the design in the same time window; "
                         "drop it to save ~a third of the budget if you are willing "
                         "to compare against the older data/raw_trials/ baseline")
    ap.add_argument("--probe", action="store_true", help="also run the memorisation probe")
    ap.add_argument("--probe-trials", type=int, default=1)
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY in the environment before running this.")

    unknown = set(args.conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown condition(s): {sorted(unknown)}; expected {list(CONDITIONS)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    banks = {(t, c): load_bank(t, c) for t in INSTRUMENTS for c in args.conditions}

    tasks = [
        (m, t, c, i)
        for m in args.models
        for t in INSTRUMENTS
        for c in args.conditions
        for i in range(1, args.trials + 1)
    ]
    print(f"Queued {len(tasks)} administrations "
          f"({len(args.models)} models x {len(INSTRUMENTS)} instruments x "
          f"{len(args.conditions)} conditions x {args.trials} trials). "
          f"Cost cap ${args.cost_cap}")

    def worker(task):
        if _state["stopped"]:
            return None
        rec = run_trial(*task, banks=banks)
        with _lock:
            _state["cost"] += rec.get("cost") or 0.0
            _state["ok" if rec["status"] == "ok" else "fail"] += 1
            if args.cost_cap and _state["cost"] >= args.cost_cap:
                _state["stopped"] = True
                print(f"  ! cost cap ${args.cost_cap} reached; no new tasks dispatched")
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for n, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if n % 25 == 0:
                print(f"  [{n}/{len(tasks)}] ok={_state['ok']} fail={_state['fail']} "
                      f"cost=${_state['cost']:.2f}")

    print(f"done: ok={_state['ok']} fail={_state['fail']} cost=${_state['cost']:.2f}")

    if args.probe:
        items = select_probe_items()
        print(f"running memorisation probe on {len(items)} items x {len(args.models)} models...")
        for m in args.models:
            for i in range(1, args.probe_trials + 1):
                run_memorisation_probe(m, i, items)
        print("probe complete")


if __name__ == "__main__":
    main()
