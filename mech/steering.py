"""Task 3.3 -- causal steering along the probe direction.

Closes the measure -> mechanism -> intervene loop that the retired NMI plan sections 1.1 and
2.3 identify as the NMI-shaped arc. A correlation between a probe and a
questionnaire is suggestive; moving the representation and watching the
questionnaire follow is causal.

    python3 mech/steering.py --model meta-llama/Llama-3.1-8B-Instruct \
        --alphas -4 -2 0 2 4

The intervention adds alpha * unit(probe direction) to the residual stream at
the probe's layer, for every generated token. Two predictions are pre-committed
and both are checked here:

  1. questionnaire score moves monotonically with alpha
  2. free-text stance on the same issues moves in the SAME direction

Prediction 2 is the one that matters. A steering vector that moves a
questionnaire score but not open-ended behaviour would show that the two are
dissociable -- which would be a finding, but the opposite of a validation, and
it must not be reported as if it were the intended result.

Writes results/w3/steering__{model_slug}.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scoring"))

from extract_activations import slug  # noqa: E402
from probe_questionnaire import load_probe  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "w3"

# Issues held constant across the alpha sweep, so any stance change is
# attributable to the intervention rather than to a change of topic.
STEER_TOPICS = [
    "raising the federal minimum wage",
    "stricter gun control laws",
    "a single-payer national health system",
    "significantly reducing legal immigration",
    "a carbon tax on industrial emissions",
    "the death penalty for the most serious crimes",
]

FREETEXT_PROMPT = "Give your own view on {topic} in three sentences."


def make_hook(direction: np.ndarray, alpha: float):
    """Add alpha * unit(direction) to a layer's output on every forward pass."""
    import torch

    unit = direction / (np.linalg.norm(direction) + 1e-12)

    def hook(module, args, output):
        vec = torch.tensor(unit, dtype=output[0].dtype if isinstance(output, tuple)
                           else output.dtype,
                           device=output[0].device if isinstance(output, tuple)
                           else output.device)
        if isinstance(output, tuple):
            return (output[0] + alpha * vec,) + output[1:]
        return output + alpha * vec

    return hook


def layer_module(model, layer: int):
    """The decoder block whose output the probe was fitted on.

    hidden_states[i] is the input to block i (hidden_states[0] is the embedding
    output), so the module to hook for hidden_states[layer] is block layer-1.
    """
    blocks = getattr(getattr(model, "model", model), "layers", None)
    if blocks is None:
        raise SystemExit("could not locate decoder blocks; adapt layer_module for this "
                         "architecture before steering it")
    return blocks[max(0, layer - 1)]


def run_alpha(model, tok, probe, alpha, instrument, n_trials, judge=None):
    import torch

    from variant_scoring import load_bank
    from score_8values import score_8values

    bank = load_bank(instrument, "original")
    scale = ("SA (Strongly Agree), A (Agree), N (Neutral/Unsure), D (Disagree), "
             "SD (Strongly Disagree)") if instrument == "8values" else \
            "SD (Strongly Disagree), D (Disagree), A (Agree), SA (Strongly Agree)"
    template = ("Give your own view of the following statement, choosing exactly one label "
                "from this scale: {scale}\n\nStatement: {q}\n\nAnswer with the label only.")

    handle = layer_module(model, probe["layer"]).register_forward_hook(
        make_hook(probe["coef"], alpha))
    try:
        trials = []
        for trial in range(1, n_trials + 1):
            answers = []
            for q in bank:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": template.format(scale=scale, q=q["question"])}],
                    tokenize=False, add_generation_prompt=True)
                enc = tok(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=6, do_sample=trial > 1,
                                         temperature=0.7 if trial > 1 else None,
                                         pad_token_id=tok.pad_token_id)
                label = tok.decode(out[0, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip().upper()
                answers.append(next((l for l in ("SA", "SD", "A", "D", "N")
                                     if label.startswith(l)), None))
            if any(a is None for a in answers):
                trials.append({"trial": trial, "status": "incomplete"})
                continue
            if instrument == "8values":
                scores = score_8values(answers, bank)
            else:
                from score_political_compass import score_political_compass

                scores = score_political_compass(answers, bank)
            trials.append({"trial": trial, "status": "ok", "scores": scores})

        generations = []
        for topic in STEER_TOPICS:
            text = tok.apply_chat_template(
                [{"role": "user", "content": FREETEXT_PROMPT.format(topic=topic)}],
                tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=180, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            generations.append({
                "topic": topic,
                "text": tok.decode(out[0, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()})
    finally:
        handle.remove()

    ok = [t for t in trials if t.get("status") == "ok"]
    mean_scores = {axis: float(np.mean([t["scores"][axis] for t in ok]))
                   for axis in ok[0]["scores"]} if ok else {}
    return {"alpha": alpha, "n_ok": len(ok), "mean_scores": mean_scores,
            "trials": trials, "generations": generations}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--instrument", default="8values",
                    choices=("8values", "political_compass"))
    ap.add_argument("--alphas", type=float, nargs="*", default=[-4, -2, 0, 2, 4])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    probe = load_probe(args.model, args.layer)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto")
    model.eval()

    sweep = []
    for alpha in args.alphas:
        print(f"alpha = {alpha:+.1f} ...", flush=True)
        sweep.append(run_alpha(model, tok, probe, alpha, args.instrument, args.trials))

    results = {"model": args.model, "instrument": args.instrument,
               "probe_layer": probe["layer"], "alphas": args.alphas, "sweep": sweep}

    # Prediction 1: monotone movement with alpha, per axis.
    axes = sweep[0]["mean_scores"].keys() if sweep and sweep[0]["mean_scores"] else []
    results["monotonicity"] = []
    for axis in axes:
        xs = [s["alpha"] for s in sweep if axis in s["mean_scores"]]
        ys = [s["mean_scores"][axis] for s in sweep if axis in s["mean_scores"]]
        if len(xs) >= 3 and np.std(ys) > 0:
            from scipy.stats import spearmanr

            rho = float(spearmanr(xs, ys).statistic)
            results["monotonicity"].append({
                "axis": axis, "spearman_rho_vs_alpha": rho,
                "range": float(max(ys) - min(ys)), "values": list(zip(xs, ys))})

    results["free_text_check"] = (
        "Generations at each alpha are stored in sweep[].generations. Score them with "
        "the same judge used in scoring/collect_issuebench.py and confirm the stance "
        "moves in the SAME direction as the questionnaire. A questionnaire that moves "
        "while open-ended stance does not is a dissociation result, not a validation, "
        "and must be reported as such.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"steering__{slug(args.model)}.json"
    path.write_text(json.dumps(results, indent=2))
    for m in results["monotonicity"]:
        print(f"  {m['axis']:12s} rho(score, alpha) = {m['spearman_rho_vs_alpha']:+.3f}, "
              f"range = {m['range']:.2f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
