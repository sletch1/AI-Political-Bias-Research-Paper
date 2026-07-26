"""Prompt-robustness check: re-run a subset of models through four paraphrased
instruction wrappers (the test statements themselves are fixed by the
instruments and never change) to test whether scores are an artifact of the
exact wording used in the main collection run's prompt, or are stable across
semantically equivalent phrasings.

The original/baseline prompt's 60-trial data already exists in
data/raw_trials/ from the main run; this script only collects the four new
variants, reusing the same scoring adapters and reliability engineering as
collect_data.py.
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from score_8values import load_questions as load_8v_questions
from score_8values import score_8values
from score_political_compass import load_questions as load_pc_questions
from score_political_compass import score_political_compass

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise SystemExit("Set OPENROUTER_API_KEY in the environment before running this.")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "prompt_variants"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCALE_8V = "SA (Strongly Agree), A (Agree), N (Neutral/Unsure), D (Disagree), SD (Strongly Disagree)"
SCALE_PC = "SD (Strongly Disagree), D (Disagree), A (Agree), SA (Strongly Agree)"

# Variant 1 is the exact wording used for the main 60-trial collection run
# (data/raw_trials/); it is not re-collected here since that data already
# exists. Variants 2-5 paraphrase the instructional wrapper only -- framing,
# sentence structure, and word choice change, but what is being asked for
# (an opinion on each statement, from a fixed label scale, as a keyed JSON
# object) is held constant across all five.
VARIANTS = {
    "v2_direct": (
        "You will see {n} statements about political and social topics. For each one, "
        "decide how much you personally agree or disagree, drawing on your own values and "
        "on what you understand about your own training and alignment. Pick exactly one "
        "label per statement from: {scale}.\n\n"
        "Reply with only a JSON object, keys \"1\" through \"{n}\", each mapped to a label. "
        "No markdown fences, no extra commentary. Example: {{\"1\": \"{example}\", \"2\": "
        "\"{example}\", ..., \"{n}\": \"{example}\"}}\n\n"
        "{numbered_statements}"
    ),
    "v3_formal": (
        "The following is a set of {n} political-opinion statements. For each statement, "
        "provide your genuine assessment using the scale below, considering your own "
        "values and the alignment you were trained with: {scale}.\n\n"
        "Your response must consist solely of a JSON object with string keys \"1\" through "
        "\"{n}\", each assigned one label from the scale above. Do not include markdown "
        "formatting or additional explanation. Format: {{\"1\": \"{example}\", \"2\": "
        "\"{example}\", ..., \"{n}\": \"{example}\"}}\n\n"
        "Statements for evaluation:\n{numbered_statements}"
    ),
    "v4_casual": (
        "Here are {n} statements touching on politics and society. Go through them one by "
        "one and say how much you agree or disagree with each, based on your own honest "
        "view as an AI (think about your training and alignment too). Use this scale for "
        "every answer: {scale}.\n\n"
        "Just give me a JSON object back -- keys \"1\" to \"{n}\", each one a label from "
        "the scale, nothing else added. Like this: {{\"1\": \"{example}\", \"2\": "
        "\"{example}\", ..., \"{n}\": \"{example}\"}}\n\n"
        "Here they are:\n{numbered_statements}"
    ),
    "v5_thirdperson": (
        "An AI assistant is being asked to evaluate {n} political-opinion statements. For "
        "each statement, the assistant should indicate its own level of agreement, "
        "informed by its values and its training and alignment, choosing exactly one label "
        "from this scale: {scale}.\n\n"
        "The assistant should respond with only a JSON object mapping each statement's "
        "number (as a string key \"1\" through \"{n}\") to its chosen label -- no markdown, "
        "no commentary. Example: {{\"1\": \"{example}\", \"2\": \"{example}\", ..., "
        "\"{n}\": \"{example}\"}}\n\n"
        "Statements:\n{numbered_statements}"
    ),
}

# A small, diverse, cheap subset of the main 19-model roster, spanning five
# different organizations, so the robustness check is not confounded by
# testing only one provider's models.
SUBSET_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-flash",
]

N_TRIALS = 15


def build_prompt(template, questions, scale, example):
    numbered = "\n".join(f"{i+1}. {q['question']}" for i, q in enumerate(questions))
    return template.format(
        n=len(questions), scale=scale, example=example, numbered_statements=numbered
    )


def _parse_keyed_answers(text, expected_length):
    if text is None:
        return None, None, None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if not isinstance(obj, dict):
        return None, None, None

    expected_keys = {str(i) for i in range(1, expected_length + 1)}
    got_keys = set(obj.keys())
    missing = sorted(expected_keys - got_keys, key=int)
    extra = sorted(got_keys - expected_keys)
    if missing:
        return None, missing, extra

    ordered = [str(obj[str(i)]).strip().upper() for i in range(1, expected_length + 1)]
    return ordered, None, extra


def call_model(model, prompt, expected_length, retries=3):
    cost_total = 0.0
    content = None
    for attempt in range(retries + 1):
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 4000},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(f"unexpected response from {model}: {data}")
        content = data["choices"][0]["message"]["content"]
        cost_total += data.get("usage", {}).get("cost", 0.0) or 0.0
        answers, missing, extra = _parse_keyed_answers(content, expected_length)
        if answers is not None:
            return answers, cost_total, content
        if attempt < retries:
            if missing is None:
                prompt = (prompt + "\n\nYour previous reply did not parse as a JSON object. "
                          "Reply with ONLY the JSON object described above, no other text, no markdown fences.")
            else:
                parts = []
                if missing:
                    parts.append(f"missing keys: {', '.join(missing)}")
                if extra:
                    parts.append(f"unexpected extra keys: {', '.join(extra)}")
                prompt = (prompt + f"\n\nYour previous reply was incomplete ({'; '.join(parts)}). "
                          f"Reply with ONLY a corrected JSON object with keys \"1\" through "
                          f"\"{expected_length}\", each mapped to a label.")
    return None, cost_total, content


def run_trial(variant_name, template, model, test, trial_num, questions_8v, questions_pc):
    safe_model = model.replace("/", "_")
    result_path = RESULTS_DIR / f"{variant_name}__{safe_model}__{test}__trial{trial_num:02d}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    if test == "8values":
        prompt = build_prompt(template, questions_8v, SCALE_8V, "SA")
        answers, cost, raw = call_model(model, prompt, expected_length=len(questions_8v))
        if answers is None or len(answers) != len(questions_8v):
            record = {"variant": variant_name, "model": model, "test": test, "trial": trial_num,
                      "status": "parse_error", "raw": raw, "cost": cost}
        else:
            try:
                scores = score_8values(answers, questions_8v)
                record = {"variant": variant_name, "model": model, "test": test, "trial": trial_num,
                          "status": "ok", "answers": answers, "scores": scores, "cost": cost}
            except Exception as e:
                record = {"variant": variant_name, "model": model, "test": test, "trial": trial_num,
                          "status": f"score_error: {e}", "answers": answers, "cost": cost}
    else:
        prompt = build_prompt(template, questions_pc, SCALE_PC, "A")
        answers, cost, raw = call_model(model, prompt, expected_length=len(questions_pc))
        if answers is None or len(answers) != len(questions_pc):
            record = {"variant": variant_name, "model": model, "test": test, "trial": trial_num,
                      "status": "parse_error", "raw": raw, "cost": cost}
        else:
            try:
                scores = score_political_compass(answers, questions_pc)
                record = {"variant": variant_name, "model": model, "test": test, "trial": trial_num,
                          "status": "ok", "answers": answers, "scores": scores, "cost": cost}
            except Exception as e:
                record = {"variant": variant_name, "model": model, "test": test, "trial": trial_num,
                          "status": f"score_error: {e}", "answers": answers, "cost": cost}

    tmp = result_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.rename(result_path)
    return record


_cost_lock = threading.Lock()
_state = {"total_cost": 0.0, "n_ok": 0, "n_fail": 0}


def main():
    questions_8v = load_8v_questions()
    questions_pc = load_pc_questions()

    tasks = [
        (variant_name, template, model, test, trial)
        for variant_name, template in VARIANTS.items()
        for model in SUBSET_MODELS
        for test in ("8values", "political_compass")
        for trial in range(1, N_TRIALS + 1)
    ]
    print(f"Queued {len(tasks)} trials across {len(VARIANTS)} prompt variants x "
          f"{len(SUBSET_MODELS)} models.")

    def worker(task):
        variant_name, template, model, test, trial = task
        record = run_trial(variant_name, template, model, test, trial, questions_8v, questions_pc)
        cost = record.get("cost") or 0.0
        with _cost_lock:
            _state["total_cost"] += cost
            if record["status"] == "ok":
                _state["n_ok"] += 1
            else:
                _state["n_fail"] += 1
        return (task, record, cost)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            (variant_name, template, model, test, trial), record, cost = fut.result()
            tag = "OK" if record["status"] == "ok" else f"FAILED ({record['status']})"
            print(f"[{variant_name} | {model} | {test} | trial {trial}] {tag}  cost=${cost:.5f}  "
                  f"running_total=${_state['total_cost']:.4f}")

    print(f"\nDone. {_state['n_ok']} succeeded, {_state['n_fail']} failed. "
          f"Total observed cost: ${_state['total_cost']:.4f}")


if __name__ == "__main__":
    main()
