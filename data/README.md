# Dataset (2,280-administration baseline, plus the validity battery)

Collected via `scoring/collect_data.py` against the OpenRouter API, scored by
the authoritative adapters in `scoring/score_8values.py` and
`scoring/score_political_compass.py`. See `pa_appendix.tex`'s Methods section
("Model Roster," "Data Collection Pipeline," "Statistical Analysis") for the
full methodology.

- **`raw_trials/`** — one JSON file per trial (`<model>__<test>__trial<N>.json`)
  from the main 2,280-administration run, containing the model's raw
  structured answers, the resulting score, the exact API cost of that call,
  and status. This is the full raw dataset; nothing here is post-hoc-cleaned.
  Also holds the SapplyValues administrations (filenames contain
  `sapplyvalues`).
- **`scores.csv`** — flattened, analysis-ready table built from `raw_trials/`
  by `scoring/consolidate.py`: one row per successful trial with per-axis
  scores.
- **`collection_summary.md`** — completion rate and cost by model and test,
  also built by `scoring/consolidate.py`.
- **`asker_identity/`**, **`item_variants/`**, **`format_factorial/`** — raw
  trials for the asker-identity, instrument-contamination, and
  response-format factorials (`scoring/collect_asker_identity.py`,
  `collect_item_variants.py`, `collect_format_factorial.py`), analyzed by
  `scoring/analyze_w1.py`.
- **`pew_typology/`** — raw trials for Pew Research Center's 2026 Political
  Typology (`scoring/collect_pew_typology.py`), scored by driving the live
  quiz; folded into the shared variance-components model by
  `scoring/analyze_f3_variance_components.py`.
- **`prompt_variants/`** — raw trials from the prompt-robustness check
  (`scoring/collect_prompt_variants.py`): a 5-model subset re-run through 4
  paraphrased prompt wrappers, 15 trials each.
- **`openended/`** — raw trials from the open-ended-generation validation arm
  (`scoring/collect_openended.py`): the same 5-model subset writing short
  opinion passages on 8 policy topics, 3 trials each, judge-scored by a
  separate model.
- **`ballot/`** — placeholder only; the ballot-proposition behavioural-validity
  arm has a collection pipeline but no data collected yet (see the paper's
  Limitations).

Regenerate `scores.csv` and `collection_summary.md` from `raw_trials/` at any
time with:

```bash
cd scoring && python3 consolidate.py
```

Re-run the statistical analyses on this data with:

```bash
cd scoring
python3 analyze_w4.py                       # error budget, MTMM, ISS repair, item entropy
python3 analyze_f3_variance_components.py   # four-instrument crossing (raw_trials/ + pew_typology/)
python3 analyze_w1.py                       # asker-identity, contamination, format factorials
python3 analyze_prompt_robustness.py        # prompt_variants/ + openended/
```
