# F3: terms of use for candidate instruments (checked 19 September 2026; updated 20 September)

**Status: both instruments below are now collected.** SapplyValues (540 administrations, 18
models) is a preliminary triangulation check, not yet crossed into the shared variance-components
model. Pew's 2026 Typology (180 administrations, 18 models) is scored by driving the live quiz and
reading Pew's own classification (`scoring/score_pew_typology.py`) rather than reimplementing their
unpublished clustering weights — see the "What actually happened" section below for what changed
from the plan in this file's original text.

`oct_fix.md` F3 requires the paper to use at least four instruments before it can go to
*Political Analysis*, and names SapplyValues as the priority add (directly comparable to Sakhawat
et al. 2026, who already use it), then a validated human-survey instrument with population norms
(ANES or Pew) as the second. This is the "confirm terms of use" step from the F3 checklist, done
before writing scoring adapters.

## 1. SapplyValues — clear, recommend as instrument 3

- **Source:** `github.com/SapplyValues/SapplyValues.github.io`, 46 items across three axes
  (economic, civil-liberties, cultural), same Strongly-Agree-to-Strongly-Disagree format as
  8Values.
- **License:** MIT/Expat, carrying forward the 8Values copyright notice — the same license
  `scoring/score_8values.py` was already built against. No separate permission needed to
  administer the items or reimplement the scoring in this codebase; the license only requires
  keeping the copyright/permission notice, which belongs in the adapter file's header.
- **Scoring:** published in the repository's own JS, so a scoring adapter can be written and
  tested to the standard of `score_8values.py`, per the F3 instructions, without reverse-engineering
  anything.
- **Comparability:** Sakhawat et al. 2026 use Political Compass, 8Values, *and* SapplyValues on 26
  models — adding it here makes our results directly comparable to the closest competing study
  (already engaged in `main.tex`'s Introduction).

## 2. Pew Research 2026 Political Typology — recommend as instrument 4 over ANES

Pew published a new political typology in June 2026 ("Beyond Red vs. Blue: The 2026 Political
Typology"), based on a nationally representative American Trends Panel survey (n = 10,357 U.S.
adults, fielded Nov 2025) and sorting respondents into 9 typology groups by k-means clustering over
30 items. A public 24-item quiz subset is live at pewresearch.org and assigns a respondent (or a
model) to one of the 9 groups.

- **Terms of use:** Pew's general terms permit reproducing and citing survey questions for
  research; replicating questions without directly comparing to Pew's own findings needs no
  special permission, and comparing to Pew's published group definitions (which this project would
  do) just requires standard citation — no different from citing any published instrument. The
  *microdata* has stricter redistribution terms, but that only matters if we publish Pew's
  respondent-level data, which we would not: we need the item text and the group-assignment
  procedure, both published in the report and its methodology appendix.
- **Why this over ANES:** ANES public-use data and items are also freely reusable for research with
  citation, and would work. But the Pew 2026 quiz was published this year, is a live, unattributed,
  self-contained instrument built for direct administration (a person or a model just answers the
  24 items and gets sorted into a group), and ships with published population norms (the share of
  U.S. adults in each of the 9 groups) that a model's typology assignment can be benchmarked
  against directly. ANES items would need to be assembled into a battery and lack a single
  published classification procedure as clean as Pew's. Recommend Pew as instrument 4; keep ANES
  as a fallback if the k-means group-assignment procedure turns out to be under-specified once we
  read the methodology appendix in full.
- **What's needed before writing the adapter:** read the full item wording and the k-means
  procedure in the methodology appendix (`pewresearch.org/politics/2026/06/10/appendix-b-typology-group-creation-and-analysis/`)
  to confirm the group-assignment rule is reproducible from the 24 public items alone, since the
  original 30-item clustering used the full item set.

## Recommendation

Both candidates clear the terms-of-use check. Write adapters for SapplyValues first (comparable
scoring pattern, comparable to the closest competitor), then Pew 2026 Typology (needs the
methodology appendix read first to confirm the 24-item quiz reproduces the published groups). This
reaches the required four-instrument minimum for *Political Analysis*.

## What actually happened (20 September)

The methodology appendix (read in full) does **not** publish the cluster medoids, variable weights,
or standardization constants — only the general clustering procedure. Reimplementing an approximate
classifier from that description was therefore not viable, and the "reproducible from the 24 public
items alone" question above is answered: no, not as a formula. The alternative used instead is the
same one this codebase already uses for Political Compass: drive the *live* quiz with a headless
browser and read off Pew's own computed result, rather than reimplement anything. Every item and
option was extracted directly from the live quiz (`questions_pew_typology.json`, cross-checked
against the quiz's own results-review page) and the classification is scored as the assigned
group's 1–9 ordinal position in Pew's own published left-to-right group ordering.

This means submitting the live quiz interactively at a scale (18 models × 10 trials = 180
submissions) well beyond ordinary personal use, which plausibly exceeds normal expectations for
interactive-tool use even though the *survey items themselves* are freely reusable for research
(the terms-of-use analysis above). The author was informed of this distinction explicitly and chose
to proceed; it is not something this project's own analysis of Pew's general terms of use clears on
its own, and is documented here for that reason. ANES was not pursued as a fallback because its
official site (electionstudies.org, and a university mirror of its codebook) is behind Cloudflare
bot-protection that blocked automated verification of exact item wording — a technical blocker, not
a terms-of-use one — and the author chose to proceed with Pew instead once informed of the
trade-off.

Result: 16 of 18 models classify left of centre on Pew's own typology; two
(`deepseek/deepseek-chat`, `x-ai/grok-4.20`) classify as *Pragmatic and Polite Right*. Ordinal
position does not correlate with either existing instrument's economic-ish axis at this sample
size.

## Update (21 September): both instruments crossed into the shared model

`scoring/analyze_f3_variance_components.py` now folds all four instruments into the same
`variance_decomposition` / `cluster_bootstrap_variance_shares` model Political Compass and 8Values
already used (`variance_components.py`; results in
`results/f3_variance_components/f3_variance_components.json`), rather than leaving SapplyValues and
Pew as separate directional/convergence checks. Each instrument's axis is one more trait level
(`sapplyvalues:right`, `sapplyvalues:auth`, `sapplyvalues:prog`, `pew_typology:position`), z-scored
within trait as before so the four incompatible scales (±10, 0–100, ±10, ordinal 1–9) don't dominate
each other.

The result is not a stronger version of the two-instrument finding; it is a different one. Pooled
over all four instruments, model identity's variance share falls from 12.97% (Political Compass +
8Values only) to **1.89%**. A per-instrument breakdown shows why: adding SapplyValues alone drops it
to 0.49%, while adding Pew alone (a coarser, single-axis instrument) only drops it to 8.21% — so the
collapse is driven by SapplyValues actively disagreeing with Political Compass on which models rank
where (the ρ=−0.49 MTMM disagreement already reported), not by Pew's coarseness diluting the signal.
Between-model spread and within-model trial noise are proportionally in line with the other
instruments for SapplyValues too (ratio ≈1.9 for all three continuous instruments), so this is not a
SapplyValues data-quality artefact — it is what "a variance component estimated from two instruments
is meaningless" (the *Political Analysis* objection F3 exists to pre-empt) looks like quantitatively.
See `main.tex` Sections 3.2/Discussion for how this is written up.

## Sources
- [SapplyValues.github.io/LICENSE](https://github.com/SapplyValues/SapplyValues.github.io/blob/master/LICENSE)
- [SapplyValues Political Test overview](https://www.idrlabs.com/sapply-values-political/test.php)
- [Pew: Beyond Red vs. Blue: The 2026 Political Typology](https://www.pewresearch.org/politics/2026/06/10/beyond-red-vs-blue-the-political-typology/)
- [Pew: Appendix B, typology group creation and analysis](https://www.pewresearch.org/politics/2026/06/10/appendix-b-typology-group-creation-and-analysis/)
- [Pew: Questions used in the 2026 typology](https://www.pewresearch.org/chart/questions-used-in-the-2026-typology/)
- [Pew Research Center Terms of Use](https://www.pewresearch.org/about/terms-and-conditions/)
- [ANES: How do I cite ANES data or resources?](https://electionstudies.org/ufaqs/how-do-i-cite-anes-data-or-resources-2/)
