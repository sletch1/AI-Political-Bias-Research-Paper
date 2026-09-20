"""Task 3.2 -- probe estimate vs questionnaire score. The novel result.

Nobody in lit_review/ has done this. Every paper in that folder either scores a
questionnaire or probes a representation; none reads the probe out *while the
model is answering the questionnaire* and asks whether the two agree.

That is the paper's convergent-validity evidence, and it is the one result that
can rehabilitate or condemn the whole genre on non-questionnaire grounds:

  * high correlation -> the questionnaire measures what the model internally
    represents, and the audit literature is measuring something real
  * low correlation  -> questionnaire audits and internal representations are
    measuring different things, which is the error budget's strongest claim

    python3 mech/probe_questionnaire.py --model meta-llama/Llama-3.1-8B-Instruct

Requires probes from train_probes.py. Writes
results/w3/probe_vs_questionnaire__{model_slug}.json.
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

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "w3"


def load_probe(model: str, layer=None):
    """Load the fitted probe, defaulting to the best layer train_probes chose."""
    stats_path = OUT_DIR / f"probes__{slug(model)}.json"
    weights_path = OUT_DIR / f"probe_weights__{slug(model)}.npz"
    if not stats_path.exists() or not weights_path.exists():
        raise SystemExit(f"probe artefacts for {model} not found -- run train_probes.py")
    stats = json.loads(stats_path.read_text())
    layer = stats["best_layer"]["layer"] if layer is None else layer
    w = np.load(weights_path)
    return {
        "layer": int(layer),
        "coef": w[f"layer_{layer}_coef"],
        "mean": w[f"layer_{layer}_mean"],
        "scale": w[f"layer_{layer}_scale"],
        "heldout_r2": stats["best_layer"]["r2_heldout"],
    }


def probe_score(activations: np.ndarray, probe: dict) -> np.ndarray:
    """Apply the probe: standardise with the training scaler, then project."""
    z = (activations - probe["mean"]) / probe["scale"]
    return z @ probe["coef"]


def administer_and_probe(model_name: str, probe: dict, instrument: str,
                         n_trials: int = 10, dtype: str = "bfloat16"):
    """Answer the questionnaire locally, reading the probe out at each item.

    The model answers item by item rather than in one keyed-JSON batch as the
    API pipeline does. That is a deliberate difference: a per-item probe read is
    the whole point of this task, and a batched answer has one activation trace
    covering seventy different items. The cost is that this condition is not
    byte-identical to the main run, and Methods must say so.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from variant_scoring import load_bank
    from score_8values import score_8values

    bank = load_bank(instrument, "original")
    scale = ("SA (Strongly Agree), A (Agree), N (Neutral/Unsure), D (Disagree), "
             "SD (Strongly Disagree)") if instrument == "8values" else \
            "SD (Strongly Disagree), D (Disagree), A (Agree), SA (Strongly Agree)"
    template = ("Give your own view of the following statement, choosing exactly one label "
                "from this scale: {scale}\n\nStatement: {q}\n\nAnswer with the label only.")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype), device_map="auto",
        output_hidden_states=True)
    model.eval()

    trials = []
    for trial in range(1, n_trials + 1):
        answers, item_probes = [], []
        for q in bank:
            text = tok.apply_chat_template(
                [{"role": "user", "content": template.format(scale=scale, q=q["question"])}],
                tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=6, do_sample=trial > 1,
                                     temperature=0.7 if trial > 1 else None,
                                     pad_token_id=tok.pad_token_id,
                                     return_dict_in_generate=True, output_hidden_states=True)
                acts = gen.hidden_states[0][probe["layer"]][:, -1, :].float().cpu().numpy()[0]
            label = tok.decode(gen.sequences[0, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip().upper()
            label = next((l for l in ("SA", "SD", "A", "D", "N") if label.startswith(l)), None)
            answers.append(label)
            item_probes.append(float(probe_score(acts, probe)))

        usable = [(a, p) for a, p in zip(answers, item_probes) if a is not None]
        if len(usable) < len(bank):
            trials.append({"trial": trial, "status": "incomplete",
                           "n_parsed": len(usable), "n_items": len(bank)})
            continue
        entry = {"trial": trial, "status": "ok", "answers": answers,
                 "item_probe_scores": item_probes,
                 "mean_probe_score": float(np.mean(item_probes))}
        if instrument == "8values":
            entry["scores"] = score_8values(answers, bank)
        else:
            from score_political_compass import score_political_compass

            entry["scores"] = score_political_compass(answers, bank)
        trials.append(entry)
    return trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--instrument", default="8values",
                    choices=("8values", "political_compass"))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    probe = load_probe(args.model, args.layer)
    print(f"probe: layer {probe['layer']}, held-out R2 = {probe['heldout_r2']:+.3f}")
    trials = administer_and_probe(args.model, probe, args.instrument, args.trials)

    ok = [t for t in trials if t.get("status") == "ok"]
    results = {"model": args.model, "instrument": args.instrument,
               "probe_layer": probe["layer"], "n_trials": len(trials),
               "n_ok": len(ok), "trials": trials}

    if len(ok) >= 3:
        probe_means = np.array([t["mean_probe_score"] for t in ok])
        results["per_axis_correlation"] = []
        for axis in ok[0]["scores"]:
            scores = np.array([t["scores"][axis] for t in ok])
            if np.std(scores) == 0 or np.std(probe_means) == 0:
                results["per_axis_correlation"].append(
                    {"axis": axis, "r": None,
                     "reason": "no variance across trials to correlate"})
                continue
            results["per_axis_correlation"].append(
                {"axis": axis, "r": float(np.corrcoef(probe_means, scores)[0, 1]),
                 "n_trials": len(ok)})

        # Item-level convergence: does the probe move with the answer the model
        # gives on *that item*? Far better powered than the trial-level
        # correlation, which has only n_trials points.
        codes = {"SD": -1.0, "D": -0.5, "N": 0.0, "A": 0.5, "SA": 1.0}
        flat_probe, flat_answer = [], []
        for t in ok:
            for a, p in zip(t["answers"], t["item_probe_scores"]):
                if a in codes:
                    flat_answer.append(codes[a])
                    flat_probe.append(p)
        if len(flat_probe) > 30 and np.std(flat_answer) > 0:
            results["item_level_correlation"] = {
                "r": float(np.corrcoef(flat_probe, flat_answer)[0, 1]),
                "n_responses": len(flat_probe),
                "note": ("Correlation between the probe read-out at an item and the "
                         "agreement the model expressed on that item. This is the "
                         "convergent-validity statistic; the trial-level one is "
                         "reported alongside but is badly underpowered."),
            }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"probe_vs_questionnaire__{slug(args.model)}.json"
    path.write_text(json.dumps(results, indent=2))
    for c in results.get("per_axis_correlation", []):
        print(f"  {c['axis']:12s} r(probe, questionnaire) = "
              f"{c['r'] if c['r'] is None else round(c['r'], 3)}")
    if "item_level_correlation" in results:
        print(f"  item-level r = {results['item_level_correlation']['r']:+.3f} "
              f"over {results['item_level_correlation']['n_responses']} responses")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
