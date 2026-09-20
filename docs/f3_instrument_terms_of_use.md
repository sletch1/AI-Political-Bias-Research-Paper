# F3: terms of use for candidate instruments (checked 19 September 2026)

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

## Sources
- [SapplyValues.github.io/LICENSE](https://github.com/SapplyValues/SapplyValues.github.io/blob/master/LICENSE)
- [SapplyValues Political Test overview](https://www.idrlabs.com/sapply-values-political/test.php)
- [Pew: Beyond Red vs. Blue: The 2026 Political Typology](https://www.pewresearch.org/politics/2026/06/10/beyond-red-vs-blue-the-political-typology/)
- [Pew: Appendix B, typology group creation and analysis](https://www.pewresearch.org/politics/2026/06/10/appendix-b-typology-group-creation-and-analysis/)
- [Pew: Questions used in the 2026 typology](https://www.pewresearch.org/chart/questions-used-in-the-2026-typology/)
- [Pew Research Center Terms of Use](https://www.pewresearch.org/about/terms-and-conditions/)
- [ANES: How do I cite ANES data or resources?](https://electionstudies.org/ufaqs/how-do-i-cite-anes-data-or-resources-2/)
