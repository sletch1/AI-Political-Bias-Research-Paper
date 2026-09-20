# Workstream 3 — the mechanistic arm

> **Not part of the current plan.** This arm was designed for a Nature Machine Intelligence
> submission that has been dropped. The NeurIPS plan in `../updates/` does not use it (see
> `updates/03_experiments.md`, "Not planned: the mechanistic arm"). Task numbers and gates below
> refer to that retired plan, not to the task IDs in `updates/`.

the retired NMI plan §1.1 and §2.3 identify the shape problem precisely: the manuscript
measures and stops, and every NMI-tier comparator has a
measure → explain → intervene arc. This directory is that arc.

The claim it supports is the one nobody in `lit_review/` has made:
**does a questionnaire score track what the model internally represents?**

| Task | File | Question |
|---|---|---|
| 3.1 | `extract_activations.py`, `train_probes.py` | Can a linear probe on activations recover DW-NOMINATE legislator ideology, and does it generalise off-distribution to news-outlet slant? |
| 3.2 | `probe_questionnaire.py` | Read the probe out *while the model answers the questionnaire*. Correlate probe estimate against questionnaire score. **This is the novel result.** |
| 3.3 | `steering.py` | Intervene linearly on the identified directions. Do questionnaire scores and IssueBench stance move together, as predicted? |

## Scope, stated plainly

This arm covers **open-weight models only**: activation access is required.
Closed models (Claude, GPT, Gemini, Grok, Nova) are excluded, and the manuscript
must say so rather than imply roster-wide coverage.

Default subset (`MECH_MODELS` in `extract_activations.py`):

- `meta-llama/Llama-3.3-70B-Instruct`
- `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- `Qwen/Qwen3-30B-A3B`
- `google/gemma-3-27b-it` — for comparability with Debevc et al.
- `meta-llama/Llama-3.1-8B-Instruct` — **prototype here first**

## Compute

A CPU server will not do 30–70B activation extraction. Budget rented GPU: 1×
A100 80GB (2× for the 70B), ~40–60 GPU-hours, ≈$150–350. Prototype the whole
pipeline end to end on Llama-3.1-8B, where it fits on a single 24GB card, before
renting anything.

the retired NMI plan R4 rates schedule overrun on this arm as **high likelihood**. Gate C is
a hard stop at week 16: if held-out probe R² ≤ 0.4 or off-distribution
generalisation fails, cut this arm cleanly. A half-working probe analysis is
worse than none.

## Ground-truth data is not shipped

Two external datasets are required and neither is generated here, for the same
reason as `data/ballot/`: they are factual records and inventing them would
destroy the only thing they contribute.

| Dataset | Where | Loader |
|---|---|---|
| DW-NOMINATE legislator ideal points | voteview.com (public, free) | `legislators.py --download` |
| News-outlet slant ratings | AllSides / Ad Fontes Media | `news_outlets.py` (supply CSV; check licence before redistributing) |

## Dependencies

Beyond the repository's `requirements.txt`:

```bash
pip install torch transformers accelerate scikit-learn
```

## Order of work

```bash
python3 mech/legislators.py --download                  # DW-NOMINATE
python3 mech/extract_activations.py --model meta-llama/Llama-3.1-8B-Instruct
python3 mech/train_probes.py --model meta-llama/Llama-3.1-8B-Instruct
python3 mech/probe_questionnaire.py --model meta-llama/Llama-3.1-8B-Instruct
python3 mech/steering.py --model meta-llama/Llama-3.1-8B-Instruct --alpha 2.0
```
