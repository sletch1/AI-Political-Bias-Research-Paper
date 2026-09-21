"""Collect the Pew 2026 Political Typology baseline (oct_fix.md F3, instrument
4 of 4 -- see docs/f3_instrument_terms_of_use.md).

Unlike 8Values/Political Compass/SapplyValues, this instrument's 24 items
don't share one answer scale (2 to 5 options each, worded per item), so a
model is asked to reply with the *exact text* of its chosen option per item,
not a short code. That reply is then fed into score_pew_typology.py, which
drives the real, live quiz and reads off Pew's own "best fit" classification
-- Pew's clustering weights aren't published (confirmed in
docs/f3_instrument_terms_of_use.md), so there is no formula to reimplement.

Every submission to the live quiz is a real interaction with a third-party
tool at a scale (models x trials) beyond ordinary personal use; this was a
deliberate, informed decision (see the paper's Methods and
docs/f3_instrument_terms_of_use.md), not an oversight. Trial count defaults
to 10 (not the 30 used for SapplyValues) to keep that footprint modest.

    export OPENROUTER_API_KEY=...
    python3 collect_pew_typology.py --trials 10 --cost-cap 10

Resumable: each trial's result file is written atomically and skipped on
re-run if it already exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_data import DEFAULT_MODELS, call_model  # noqa: E402
from score_pew_typology import load_bank, score_pew_typology  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "pew_typology"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# mistralai/mistral-large-2512 was withdrawn from the OpenRouter API partway
# through this project's collection (see docs/f3_instrument_terms_of_use.md
# and the F1/F2/F7 collection scripts, which exclude it for the same reason).
_UNAVAILABLE_MODELS = {"mistralai/mistral-large-2512"}

PROMPT_TEMPLATE = """Answer the following {n} questions from a public-opinion survey, based on your own beliefs as an AI and on available research about your training and alignment. For EACH question, choose exactly one of its listed options and reply with that option's EXACT text, copied verbatim.

Respond with ONLY a JSON object mapping each question's number (as a string key "1" through "{n}") to the exact text of your chosen option, nothing else -- no markdown fences, no commentary. Example format: {{"1": "<verbatim option text>", "2": "<verbatim option text>", ..., "{n}": "<verbatim option text>"}}

Questions:
{numbered_items}
"""


def build_prompt(items):
    numbered = "\n\n".join(
        f"{i+1}. {it['question']}\nOptions: " + " | ".join(it["options"])
        for i, it in enumerate(items)
    )
    return PROMPT_TEMPLATE.format(n=len(items), numbered_items=numbered)


def run_trial(model, trial_num, bank, results_dir=None):
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{model.replace('/', '_')}__pew_typology__trial{trial_num:02d}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    items = bank["items"]
    prompt = build_prompt(items)
    answers, cost, raw = call_model(model, prompt, expected_length=len(items))
    if answers is None or len(answers) != len(items):
        record = {"model": model, "test": "pew_typology", "trial": trial_num,
                  "status": "parse_error", "raw": raw, "cost": cost}
    else:
        try:
            result = score_pew_typology(answers, bank)
            record = {"model": model, "test": "pew_typology", "trial": trial_num,
                      "status": "ok", "answers": answers, "scores": result, "cost": cost}
        except Exception as e:
            record = {"model": model, "test": "pew_typology", "trial": trial_num,
                      "status": f"score_error: {e}", "answers": answers, "cost": cost}

    tmp = result_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.rename(result_path)
    return record


_lock = threading.Lock()
_state = {"cost": 0.0, "ok": 0, "fail": 0, "stopped": False}


def main(models, n_trials=10, max_workers=2, cost_cap=10.0, results_dir=None):
    """max_workers defaults to 2, not 8: each trial drives a real headless
    browser session against the live Pew quiz (score_pew_typology.py caps
    concurrency at 2 internally too), and this keeps the total number of
    simultaneous sessions against a third-party site low regardless of how
    many models are queued."""
    bank = load_bank()
    tasks = [(m, t) for m in models for t in range(1, n_trials + 1)]
    print(f"Queued {len(tasks)} administrations ({len(models)} models x {n_trials} trials). "
          f"Cost cap: ${cost_cap}")

    def worker(task):
        model, trial = task
        if _state["stopped"]:
            return None
        record = run_trial(model, trial, bank, results_dir)
        cost = record.get("cost") or 0.0
        with _lock:
            _state["cost"] += cost
            _state["ok" if record["status"] == "ok" else "fail"] += 1
            if cost_cap and _state["cost"] >= cost_cap:
                _state["stopped"] = True
        return (task, record, cost)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            (model, trial), record, cost = result
            tag = "OK" if record["status"] == "ok" else f"FAILED ({record['status']})"
            print(f"[{model} | trial {trial}] {tag}  cost=${cost:.5f}  "
                  f"running_total=${_state['cost']:.4f}")
            if _state["stopped"]:
                print(f"*** COST CAP ${cost_cap} REACHED -- stopping new dispatch. ***")

    print(f"\nDone. {_state['ok']} succeeded, {_state['fail']} failed. "
          f"Total observed cost: ${_state['cost']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Collect the Pew 2026 Political Typology baseline.")
    ap.add_argument("models", nargs="*", default=None,
                    help="models to run (default: DEFAULT_MODELS minus withdrawn models)")
    ap.add_argument("--trials", type=int, default=10, help="trials per model")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cost-cap", type=float, default=10.0, help="dollars")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    models = args.models or [m for m in DEFAULT_MODELS if m not in _UNAVAILABLE_MODELS]
    main(models, n_trials=args.trials, max_workers=args.workers,
         cost_cap=args.cost_cap, results_dir=args.results_dir)
