# What This Paper Needs to Reach Nature Human Behaviour / Nature Machine Intelligence

## Starting point

This paper scored 4/5 as an AP Research project and was desk-rejected (no peer
review) by *Big Data & Society* on 2025-05-11, with the editor citing two
specific gaps: **sample/query size too small**, and **insufficient connection to
the existing literature on bias in LLMs**. Those two issues are the floor, not
the ceiling — Nature Human Behaviour (NHB) and Nature Machine Intelligence (NMI)
sit well above Big Data & Society in selectivity, and this paper is currently far
below the bar for either. This document lays out concretely how far, and what to
do about it.

## What the actual bar looks like

Three real, comparable papers in this exact subfield, to calibrate against
(not hypothetical — these are what NHB/NMI-adjacent reviewers will mentally
benchmark this paper against):

- **"Large Language Models Reflect the Ideology of their Creators"** (*npj
  Artificial Intelligence*, a Nature-portfolio journal): 19 LLMs, ~4,000
  political figures probed with open-ended generation (not a fixed
  questionnaire), 6 languages, multiple geopolitical regions, public code+data
  repo. Even this paper was criticized in follow-up commentary for having a
  *relatively small* sample of models.
- **"Generative Language Models Exhibit Social Identity Biases"** (*Nature
  Computational Science*): 77 LLMs, 2,000 generations per model in the core
  analysis (94,000 total across the full study), plus a causal component —
  the authors manipulate training-data curation and fine-tuning to show what
  *reduces* the bias, not just that it exists.
- **"On the Conversational Persuasiveness of GPT-4"** (*Nature Human
  Behaviour*): large controlled human-subject experiment with a theory-driven
  design, pre-specified analysis, and human baselines built into the core
  comparison, not tacked onto the discussion section.

Current paper, for comparison: **3 models, 120 total test administrations, one
fixed prompt, one point in time, no human baseline in the statistical design, no
code/data release beyond two analysis scripts.** That gap is the whole story.

## Major weaknesses and what to do about each

### 1. Sample size and scale (confirmed by the BD&S rejection)
- **Problem:** 3 models, 20 trials/test/model, 120 total administrations, and
  Claude's data effectively truncates further after trial #16.
- **Fix:** Scale to 15–20+ models spanning multiple families, sizes, and both
  open-weight and closed models (comparable papers use 19–77). Increase trials
  per model into the hundreds. Run at multiple points in time with pinned model
  versions to separate "response variance" from genuine temporal drift (see
  #6).

### 2. Literature engagement (the other confirmed BD&S rejection reason)
- **Problem:** The literature review discusses three papers (Rozado;
  Rettenberger et al.; Fulay et al.). The field is much larger — political-bias
  and ideology-of-LLM papers now appear regularly in *Nature Computational
  Science*, *npj Artificial Intelligence*, *Humanities and Social Sciences
  Communications*, ACL/EMNLP, and more.
- **Fix:** Build a real related-work section organized by methodology
  (questionnaire-based vs.\ open-ended-generation vs.\ stance-classification
  approaches), explicitly position this paper's typology-test method against
  the field's move toward open-ended prompting (see #3), and engage with the
  RLHF/alignment literature that offers a *mechanistic* explanation for why
  models lean left — not just more descriptive bias-measurement papers.

### 3. Instrument choice is behind the field's frontier
- **Problem:** Static questionnaires (Political Compass, 8Values) are exactly
  the methodology the field has been moving away from, precisely because
  probing ideology via direct questions doesn't necessarily reflect model
  behavior in natural use. The npj paper above switched to open-ended
  generation for this reason.
- **Fix:** At minimum, add an open-ended-generation validation arm (e.g., ask
  models to write about policy topics or public figures, then score the
  output) and show it correlates with the typology-test scores. This directly
  answers a reviewer's first objection.

### 4. The 8Values→Political Compass conversion formula is unvalidated
- **Problem:** The linear formula ($y = -\tfrac{1}{5}x + 10$) is derived from
  two *assumed* anchor points, not from empirical data showing the two tests
  actually relate this way. This is the single most fixable-yet-serious
  technical flaw — a reviewer will ask "how do you know this crosswalk is
  correct?" and the paper currently has no answer.
- **Fix:** Either (a) have real human respondents (or a large panel of model
  outputs) take both tests and regress one onto the other to derive an
  empirical conversion, or (b) drop the conversion entirely and analyze the two
  tests as separate, non-comparable outcome measures, or (c) cite an existing
  validated crosswalk if one exists in the psychometrics literature.

### 5. The Ideological Stability Score (ISS) is an ad hoc, unvalidated metric
- **Problem:** $\mathrm{ISS} = 0.7 \times \mathrm{RSD} + 0.3 \times
  \mathrm{RIQR}$ — the 0.7/0.3 weighting has no stated justification, no
  citation to precedent, and no sensitivity analysis.
- **Fix:** Either justify the weights from a validated framework, or run a
  robustness check showing the model/test ranking doesn't change under
  alternative weightings (e.g., 0.5/0.5, 0.9/0.1). Report that check.

### 6. "Model drift" is mislabeled
- **Problem:** What's measured is trial-to-trial response variance *within a
  single sitting*, not drift over calendar time (the standard ML meaning of
  "model drift" — behavior changing as a provider silently updates a model).
  A reviewer familiar with the term will flag this immediately.
- **Fix:** Rename the construct (e.g., "response variability" or "within-session
  inconsistency"). If genuine drift is a claim worth making, that requires
  repeated measurement across weeks/months with pinned model-version
  identifiers at each time point.

### 7. Statistical toolkit is too shallow for a Nature-tier venue
- **Problem:** One-way ANOVA + Tukey HSD, reported as point estimates and
  p-values, with no effect sizes, no assumption checks (normality,
  homogeneity of variance — scores are bounded to $[-10,10]$ and plausibly
  non-normal), no correction across the *full* family of tests in the paper
  (only within each individual Tukey HSD), and no power justification for why
  N≈36–40 per cell is sufficient.
- **Fix:** Report effect sizes (Cohen's $d$ or $\eta^2$) with confidence
  intervals, run and report normality/variance diagnostics, consider a
  mixed-effects model treating trial as nested within model (more
  appropriate than flat one-way ANOVA for repeated-trial data), and include a
  power analysis or at least an honest power discussion.

### 8. No human baseline in the actual analysis
- **Problem:** The single Gallup comparison (25% liberal / 37% conservative /
  34% moderate) appears only narratively in the conclusion, not as part of the
  statistical design.
- **Fix:** Build human survey data (ANES, WVS, ESS, or Gallup microdata) into
  the core comparison from the start — test whether LLM scores differ from a
  human reference distribution, not just from each other.

### 9. Missing-data handling discards a potentially interesting finding
- **Problem:** Claude's refusal to continue after trial #16 is treated purely
  as a limitation to work around.
- **Fix:** Model the refusal itself as an outcome — refusal-rate differences
  across models on political topics is arguably a more novel, publishable
  finding than the ideology scores themselves, and NHB/NMI reviewers would
  likely find it more interesting than what's currently the paper's main
  result.

### 10. Reproducibility and open science
- **Problem:** No pinned model-version identifiers or API timestamps, no
  reported decoding parameters (temperature, top-p), no raw dataset release
  (trial data lives only as Python list literals inside the two analysis
  scripts), no preregistration.
- **Fix:** Report exact model snapshots and dates, fix and report decoding
  parameters, release the full raw trial-level dataset (not just the
  post-hoc-cleaned arrays), and preregister the confirmatory hypotheses
  (leaving the ISS/ideological-distance metrics explicitly labeled as
  exploratory if they weren't preregistered).

### 11. Single fixed prompt template
- **Problem:** All 120 administrations use one exact prompt wording. LLM
  outputs are well documented to be sensitive to prompt phrasing — this is a
  direct, well-known threat to validity that a single-prompt design cannot
  rule out.
- **Fix:** Add multiple paraphrased prompt templates (and ideally multiple
  system-prompt/persona conditions) and show results are robust across them,
  or explicitly model prompt-template as a factor.

## Priority order (highest leverage first)

1. Fix or replace the 8Values conversion formula (#4) — currently the most
   exploitable technical flaw.
2. Scale up models and trials (#1) — directly answers the confirmed BD&S
   rejection reason.
3. Deepen the literature review (#2) — the other confirmed rejection reason.
4. Add a human baseline to the core design (#8) and an open-ended-generation
   validation arm (#3).
5. Upgrade the statistics (#7) and fix the ISS justification (#5).
6. Fix terminology (#6), reframe missing data as a finding (#9), and address
   reproducibility/open science (#10) and prompt robustness (#11).

Items 1–4 are what stand between this paper and a credible peer-reviewed
submission anywhere; items 5–11 are what stand between "credible" and
"Nature-tier."

## Sources consulted

- [Large Language Models Reflect the Ideology of their Creators (npj Artificial Intelligence)](https://www.nature.com/articles/s44387-025-00048-0)
- [Large Language Models Reflect the Ideology of their Creators (arXiv preprint)](https://arxiv.org/abs/2410.18417)
- [Generative Language Models Exhibit Social Identity Biases (Nature Computational Science)](https://www.nature.com/articles/s43588-024-00741-1)
- [On the Conversational Persuasiveness of GPT-4 (Nature Human Behaviour)](https://www.nature.com/articles/s41562-025-02194-6)
- [Performance and Biases of Large Language Models in Public Opinion Simulation (Humanities and Social Sciences Communications)](https://www.nature.com/articles/s41599-024-03609-x)
