# Dataset (2,280 administrations, 19 models)

Collected via `scoring/collect_data.py` against the OpenRouter API, scored by
the authoritative adapters in `scoring/score_8values.py` and
`scoring/score_political_compass.py`. See `main.tex`'s Methods section
("Model Roster," "Data Collection Pipeline," "Statistical Analysis") for the
full methodology.

- **`raw_trials/`** — one JSON file per trial (`<model>__<test>__trial<N>.json`)
  from the main 2,280-administration run, containing the model's raw
  structured answers, the resulting score, the exact API cost of that call,
  and status. This is the full raw dataset; nothing here is post-hoc-cleaned.
- **`scores.csv`** — flattened, analysis-ready table built from `raw_trials/`
  by `scoring/consolidate.py`: one row per successful trial with per-axis
  scores.
- **`collection_summary.md`** — completion rate and cost by model and test,
  also built by `scoring/consolidate.py`.
- **`prompt_variants/`** — raw trials from the prompt-robustness check
  (`scoring/collect_prompt_variants.py`): a 5-model subset re-run through 4
  paraphrased prompt wrappers, 15 trials each. See main.tex's
  "Prompt-Robustness and Open-Ended-Generation Checks" subsection.
- **`openended/`** — raw trials from the open-ended-generation validation arm
  (`scoring/collect_openended.py`): the same 5-model subset writing short
  opinion passages on 8 policy topics, 3 trials each, judge-scored by a
  separate model. Same main.tex subsection as above.

Regenerate `scores.csv` and `collection_summary.md` from `raw_trials/` at any
time with:

```bash
cd scoring && python3 consolidate.py
```

Re-run the full statistical analysis on the main dataset with:

```bash
cd scoring && python3 analyze_expanded.py
```

`prompt_variants/` and `openended/` are analyzed inline (not by a dedicated
script); see main.tex's Results section for the exact figures and how they
were computed from these raw files.
