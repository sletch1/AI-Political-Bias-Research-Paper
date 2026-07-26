# Echoes of Ideology: Political Bias & Stability in LLMs

Code, data, and analysis supporting the paper **"Echoes of Ideology: Evaluating
Political Bias and Stability in Large Language Models Through Quantitative
Analysis and Standardized Typology Testing"** (`main.tex` / `main.pdf` in this
repo). Target venue: *npj Artificial Intelligence*.

**Research question:** To what extent do large language models from a broad
set of developers exhibit ideological bias and response consistency across
political typology assessments, and does either property track a model's
developer, country of origin, or size?

## Key findings

- **19 models across 11 organizations** (OpenAI, Anthropic, DeepSeek, Google,
  Meta, Mistral, Alibaba, xAI, Cohere, Amazon, NVIDIA) were each administered
  the Political Compass and 8Values tests 60 times per instrument (2,280 total
  administrations), collected programmatically via the OpenRouter API.
- **Every model leaned the same direction on every axis**: economically left
  and socially libertarian on the Political Compass; toward equality, peace,
  liberty, and progress on 8Values. Classical ANOVA, Welch's ANOVA, and
  Kruskal-Wallis all agree at p<.0001 on every axis, with large effect sizes
  (eta-squared 0.56-0.73).
- The **magnitude** of that lean, and each model's **trial-to-trial
  consistency** (a range-normalized Ideological Stability Score), varied by
  specific model rather than by provider, country, or size.
- Every model differs significantly from a neutral center-point proxy
  motivated by a Gallup population survey, even after multiple-comparison
  correction.
- Two bounded robustness checks on a five-model subset: prompt wording shifts
  score **magnitude** (often substantially) but never flipped a model's
  **direction**; a small open-ended-generation validation arm was
  underpowered to confirm or rule out agreement with the questionnaire-based
  scores.

Both instruments are scored against **authoritative sources**, not an
approximate cross-test formula: 8Values via a lossless port of its own
open-source scoring algorithm, and Political Compass via a headless browser
reading the site's own results page.

## Repository contents

| Path | Description |
|---|---|
| `main.tex` / `main.pdf` | The paper. |
| `references.bib` | Bibliography. |
| `data/raw_trials/` | One JSON file per trial from the main 2,280-administration run: the model's structured answers, resulting score, and exact API cost. |
| `data/scores.csv` | Flattened, analysis-ready table built from `raw_trials/`. |
| `data/collection_summary.md` | Per-model completion rate and cost for the main run. |
| `data/prompt_variants/` | Raw trials from the prompt-robustness check (5 models x 4 paraphrased prompts x 15 trials). |
| `data/openended/` | Raw trials from the open-ended-generation validation arm (5 models x 8 topics x 3 trials, judge-scored). |
| `scoring/collect_data.py` | Main collection pipeline (keyed-JSON prompting, concurrency, retries, resumable). |
| `scoring/score_8values.py` | Authoritative 8Values scoring (ported line-for-line from the official site's own algorithm). |
| `scoring/score_political_compass.py` | Authoritative Political Compass scoring (Playwright browser automation against the live site). |
| `scoring/consolidate.py` | Builds `data/scores.csv` and `data/collection_summary.md` from `data/raw_trials/`. |
| `scoring/analyze_expanded.py` | Full statistical analysis: descriptive stats, omnibus tests (classical/Welch/Kruskal-Wallis + eta-squared), Games-Howell/Tukey post-hoc with Benjamini-Hochberg FDR correction, human-baseline one-sample tests, ISS consistency ranking. |
| `scoring/collect_prompt_variants.py` | Prompt-robustness check pipeline. |
| `scoring/collect_openended.py` | Open-ended-generation + judge-scoring pipeline. |

## Running the analysis

Requires Python 3.9+ and:

```bash
pip install requests numpy scipy statsmodels pandas pingouin playwright
playwright install chromium
```

Regenerate the analysis-ready tables and rerun the full statistical analysis
from already-collected data:

```bash
cd scoring
python3 consolidate.py
python3 analyze_expanded.py
```

Re-running data collection requires an `OPENROUTER_API_KEY` environment
variable (never commit this key) and will incur API costs:

```bash
export OPENROUTER_API_KEY=...
python3 collect_data.py               # main 19-model run (~$9)
python3 collect_prompt_variants.py    # prompt-robustness check (~$1)
python3 collect_openended.py          # open-ended-generation arm (~$0.15)
```

All three collection scripts are resumable: each trial is written atomically
and skipped on re-run if it already exists, so an interrupted run can simply
be restarted.

## Limitations

See the paper's Limitations section for the full discussion. In brief: no
human-subject data was collected directly (a Gallup population survey is used
as a bounded, categorical proxy); every administration uses one fixed prompt
template per test, with prompt-robustness confirmed only for a 5-of-19-model
subset; and the open-ended-generation validation arm is too small (5 models)
to be conclusive on its own.

## Citation

If you use this code or data, please cite the accompanying paper (see
`main.tex`/`main.pdf` for full reference details).
