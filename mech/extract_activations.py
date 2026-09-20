"""Task 3.1, step 2 -- activation extraction from open-weight models.

Generates text in the voice of legislators with known DW-NOMINATE scores (and,
for the generalisation test, news outlets with known slant), and records the
model's internal state while it does so. Those states are what the probes in
train_probes.py are fitted on.

    python3 mech/extract_activations.py --model meta-llama/Llama-3.1-8B-Instruct \
        --source legislators --n 200

Writes data/activations/{model_slug}__{source}.npz containing:
    X        (n_examples, n_layers, hidden_size) float16 -- mean-pooled residual
             stream over the generated continuation, per layer
    y        (n_examples,) float32 -- DW-NOMINATE dim1 or outlet slant
    meta.json alongside, with the prompt and identity behind every row

Two decisions worth stating in Methods:

*Pool over the generated continuation, not the prompt.* Pooling over the prompt
would measure the model's representation of a name it was handed. Pooling over
what it then writes measures the representation driving the political content
it produced, which is the thing the questionnaire is being compared against.

*Keep every layer.* Which layer carries a linear political direction is an
empirical question, and reporting held-out R-squared by layer is a result in
its own right (Kim, Evans & Schein find it is far from the last layer). Storing
only a chosen layer would make that plot impossible without re-renting the GPU.

Requires: pip install torch transformers accelerate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "activations"

# Open weights only -- activation access is the whole point, and closed models
# cannot be included. The manuscript must state this exclusion plainly rather
# than implying roster-wide coverage (the retired NMI plan W3, "Feasibility").
MECH_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",          # prototype target: fits on one 24GB card
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Qwen/Qwen3-30B-A3B",
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.3-70B-Instruct",         # needs 1-2x A100 80GB
]


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def load_prompts(source: str, n: int) -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if source == "legislators":
        import legislators

        return legislators.voice_prompts(legislators.load(n))
    if source == "news":
        import news_outlets

        return news_outlets.voice_prompts()
    raise ValueError(f"unknown source {source!r}; expected 'legislators' or 'news'")


def extract(model_name: str, prompts: list, max_new_tokens: int = 120,
            batch_size: int = 8, dtype: str = "bfloat16", device: str = "auto"):
    """Generate a continuation per prompt and mean-pool hidden states over it."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype), device_map=device,
        output_hidden_states=True)
    model.eval()

    rows = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 return_dict_in_generate=True, output_hidden_states=True)
            # hidden_states from generate is (step, layer, batch, tokens, hidden).
            # Mean-pool the per-step last-token states over the continuation,
            # which is the representation active while the model was writing.
            per_layer = []
            n_layers = len(gen.hidden_states[0])
            for layer in range(n_layers):
                steps = torch.stack([s[layer][:, -1, :].float()
                                     for s in gen.hidden_states], dim=1)
                per_layer.append(steps.mean(dim=1))
            stacked = torch.stack(per_layer, dim=1)      # (batch, layer, hidden)
        text = tok.batch_decode(gen.sequences[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        for i, p in enumerate(batch):
            rows.append({"meta": {**p, "generation": text[i]},
                         "acts": stacked[i].to(torch.float16).cpu().numpy()})
        print(f"  [{min(start + batch_size, len(prompts))}/{len(prompts)}]", flush=True)

    X = np.stack([r["acts"] for r in rows])
    y = np.array([r["meta"]["target"] for r in rows], dtype="float32")
    return X, y, [r["meta"] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MECH_MODELS[0], help=f"one of {MECH_MODELS}")
    ap.add_argument("--source", default="legislators", choices=("legislators", "news"))
    ap.add_argument("--n", type=int, default=200, help="legislators to sample")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None, help="cap prompts (smoke tests)")
    args = ap.parse_args()

    import numpy as np

    prompts = load_prompts(args.source, args.n)
    if args.limit:
        prompts = prompts[:args.limit]
    print(f"{args.model}: {len(prompts)} prompts from {args.source}")

    X, y, meta = extract(args.model, prompts, args.max_new_tokens, args.batch_size,
                         args.dtype)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{slug(args.model)}__{args.source}"
    np.savez_compressed(OUT_DIR / f"{stem}.npz", X=X, y=y)
    (OUT_DIR / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_DIR / (stem + '.npz')}  X={X.shape}  y={y.shape}")


if __name__ == "__main__":
    main()
