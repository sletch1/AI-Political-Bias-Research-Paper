# Echoes of Ideology: Political Bias & Stability in LLMs

Code and analysis supporting the paper **"Echoes of Ideology: Evaluating Political
Bias and Stability in Large Language Models Through Quantitative Analysis and
Standardized Typology Testing"** (full PDF included in this repo).

**Research question:** To what extent do GPT-4o, Claude 3.7 Sonnet, and DeepSeek
V-3 exhibit ideological bias and consistency across political typology
assessments?

## Key findings

- All three models leaned **economically left** and **socially libertarian**
  across every trial and both tests — a consistent, cross-model bias.
- **DeepSeek was statistically significantly more economically right-leaning
  than Claude** (Tukey HSD, mean difference −0.7413, p < 0.05); no other
  pairwise economic comparison was significant.
- Social ideology scores were **statistically indistinguishable** across all
  three models (one-way ANOVA p = 0.9575).
- The **8Values** test produced systematically less extreme scores than the
  **Political Compass** test for the same models.
- **ChatGPT (GPT-4o-mini)** was the most internally consistent model (lowest
  Ideological Stability Score); **Claude 3.7 Sonnet** was the least consistent.
- No meaningful model drift was observed within a given test across repeated
  trials, aside from elevated variability for DeepSeek.

## Methodology

1. **Models:** GPT-4o-mini, Claude 3.7 Sonnet, DeepSeek V-3 — chosen for market
   share, growth trajectory, and (for DeepSeek) geographic/cultural diversity
   relative to Western-developed LLMs.
2. **Instruments:** [Political Compass](https://www.politicalcompass.org/) and
   [8Values](https://8values.github.io/), two standardized political typology
   tests. Each model was prompted with "Answer the following questions based on
   your beliefs as an LLM and on available research" and given every question
   from both tests.
3. **Trials:** Each (model × test) pair was run **20 times**, for 120 total
   test administrations.
4. **Standardization:** Political Compass natively outputs (economic, social)
   coordinates in [-10, 10] × [-10, 10]. 8Values outputs eight percentage-based
   values; a linear mapping (derived from the two tests' shared reference
   points) converts the left-leaning 8Values dimensions (equality, globalism,
   liberty, progress) onto the same Political Compass scale so the two
   instruments are directly comparable.
5. **Cleaning:** z-score outlier removal (|z| > 2), which excluded 2 data
   points from the full dataset.
6. **Descriptive statistics:** mean, median, standard deviation, range,
   relative IQR, and a custom **Ideological Stability Score (ISS)** — a
   weighted combination of relative standard deviation and relative IQR used
   to rank models/tests by response consistency — plus an **ideological
   distance** metric (Euclidean distance of each (economic, social) point from
   the origin) used for the model-drift analysis.
7. **Inferential statistics:** one-way **ANOVA** (`scipy.stats.f_oneway`) run
   separately on economic and social scores across the three models, followed
   by **Tukey's HSD** post-hoc test (`statsmodels.stats.multicomp.pairwise_tukeyhsd`,
   α = 0.05) to identify which specific model pairs differ.

Full derivations, figures, tables, literature review, and discussion are in the
paper PDF.

## Repository contents

| File | Description |
|---|---|
| `ECHOES OF IDEOLOGY_....pdf` | Full research paper (methodology, results, figures, discussion, references). |
| `econ_one_way_anova_test_+_turkey_hsd_test.py` | ANOVA + Tukey HSD on **economic** ideology scores. |
| `social_one_way_anova_test_+_turkey_hsd_test.py` | ANOVA + Tukey HSD on **social** ideology scores. |

Both scripts embed the cleaned, standardized score data (post z-score
filtering, post 8Values→Political Compass conversion) directly as Python lists
— GPT-4o-mini, Claude 3.7 Sonnet, and DeepSeek V-3, 36–40 trials per model —
combining results from both the Political Compass and 8Values tests. There is
no separate raw-data file; the embedded arrays *are* the analysis dataset.

## Running the analysis

Requires Python 3.9+ and:

```bash
pip install scipy statsmodels numpy
```

Then, from the repo root:

```bash
python "econ_one_way_anova_test_+_turkey_hsd_test.py"
python "social_one_way_anova_test_+_turkey_hsd_test.py"
```

Each script prints:
1. The one-way ANOVA F-statistic and p-value, with a plain-language
   significant/not-significant interpretation at α = 0.05.
2. The full Tukey HSD pairwise comparison table (mean differences, adjusted
   p-values, confidence intervals, reject/fail-to-reject flags) across
   GPT-4 vs. Claude, GPT-4 vs. DeepSeek, and Claude vs. DeepSeek.

## Limitations

As noted in the paper: the study covers only three models and two typology
tests, relies on a relatively small number of trials per condition, and
measures self-reported model outputs on typology instruments rather than
downstream behavioral bias — findings should be read as indicative of these
specific models/tests/snapshots in time, not as general claims about LLM
political bias.

## Citation

If you use this code or data, please cite the accompanying paper (see PDF for
full author and reference details).
