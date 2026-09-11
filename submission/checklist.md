# Pre-submission checklist

Verified mechanically where possible. Re-run `make_w4_figures.py` and rebuild before
submitting so the PDFs match the data.

| Item | Status | Evidence |
|---|---|---|
| Main text ≤3,500 words | **PASS** — 1,878 | Intro 441 + Results 793 + Discussion 644 |
| Abstract ~150 words, unreferenced | **PASS** — 155, no citations | |
| ≤6 main display items | **PASS** — 5 figures + 1 table | Others moved to SI |
| References ~50 | **PASS** — 31 | `main.bbl` |
| Data / Code availability | **PASS** | Both sections present |
| Ethics, Competing interests, Author contributions | **PASS** | All present |
| No "first study to" claims | **PASS** | Only a Campbell–Fiske false positive |
| Zero undefined refs/citations | **PASS** | XeLaTeX build clean, both documents |
| Reporting Summary | **N/A** | No human subjects in this design |
| Live NMI author guide re-read | **NOT DONE** | Requires a browser session — plan.md Action 1.2.a |
| OSF preregistration linked | **NOT DONE** | No preregistration exists |
| GitHub repo public and current | **NOT DONE** | Local commits only; nothing pushed |
| Every Discussion claim traceable | **PASS** | Each number traces to `results/w4/w4_results.json` |
| W1 validity battery | **BLOCKED** | Needs `OPENROUTER_API_KEY`, ≈$25, ≈4h |
| W2 ecological validity | **BLOCKED** | Needs API access |
| W3 mechanistic arm | **BLOCKED** | Needs GPU + open-weight activations |

**Build:** `xelatex → bibtex → xelatex → xelatex` for both `main.tex` and
`supplementary.tex`. The manuscript loads `fontspec`, so `pdflatex` fails.
