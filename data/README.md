# Expanded dataset (1,440 administrations, 12 models)

Collected via `scoring/collect_data.py` against the OpenRouter API, scored by
the authoritative adapters in `scoring/score_8values.py` and
`scoring/score_political_compass.py`. See `main.tex` Section 3.5 for the full
methodology.

- **`raw_trials/`** — one JSON file per trial (`<model>__<test>__trial<N>.json`),
  containing the model's raw structured answers, the resulting score, the
  exact API cost of that call, and status. This is the full raw dataset;
  nothing here is post-hoc-cleaned.
- **`scores.csv`** — flattened, analysis-ready table built from `raw_trials/`
  by `scoring/consolidate.py`: one row per successful trial with per-axis
  scores.
- **`collection_summary.md`** — completion rate and cost by model and test,
  also built by `scoring/consolidate.py`.

Regenerate `scores.csv` and `collection_summary.md` at any time with:

```bash
cd scoring && python3 consolidate.py
```

Re-run the full statistical analysis on this dataset with:

```bash
cd scoring && python3 analyze_expanded.py
```
