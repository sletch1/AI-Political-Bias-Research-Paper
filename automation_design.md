# Automating Data Collection: Design Proposal

## The current bottleneck

Every one of the 120 original data points required a human to: read each test
question, paste it into a prompt, send it to the model through a chat UI, copy
the model's answer, retype it into the Political Compass or 8Values website,
and transcribe the resulting score into a spreadsheet. That's roughly 60-70
manual round-trips *per trial*, which is why scaling past 3 models and 40
trials was infeasible by hand. Everything below is designed to remove the
human from every one of those steps except reviewing results.

## Key insight: don't reinvent scoring — automate the real thing

- **8Values is fully open source** ([github.com/8values/8values.github.io](https://github.com/8values/8values.github.io)).
  Its scoring logic can be extracted and ported directly (or run headlessly),
  giving byte-for-byte identical results to the real site with zero guesswork.
- **Political Compass's per-question weights are not publicly documented**
  (confirmed — even recent academic factor-analysis papers on the PCT note the
  exact weighting isn't published). Rather than reverse-engineer or approximate
  it, automate submission to the *real* Political Compass website with a
  headless browser and scrape the authoritative result. One browser submission
  per trial (not per question), so this scales fine even at high trial counts.

This also **structurally eliminates** the paper's current biggest technical
flaw (the unvalidated linear 8Values→Political-Compass conversion formula) —
with both tests scored authoritatively and independently, there's no need to
force one onto the other's scale at all. Report them natively, or derive an
empirical crosswalk from the resulting data if one is still wanted.

## Pipeline overview

```
                         ┌────────────────────┐
                         │  Question banks    │  (built once)
                         │  PC: 62 Qs          │
                         │  8V: ~70 Qs          │
                         └─────────┬──────────┘
                                   │
     ┌─────────────────────────────▼─────────────────────────────┐
     │  Experiment grid generator                                 │
     │  models × trials × prompt templates × temperature × time   │
     └─────────────────────────────┬─────────────────────────────┘
                                   │  (one job per grid cell)
                         ┌─────────▼──────────┐
                         │  Async job queue    │  resumable, rate-limited,
                         │  + worker pool      │  per-provider backoff
                         └─────────┬──────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  Model gateway (1 API)        │  OpenRouter or similar
                    │  → 100s of models, 1 schema   │
                    └──────────────┬───────────────┘
                                   │  structured JSON answers
                    ┌──────────────▼───────────────┐
                    │  Response parser + validator  │  retry-on-malformed,
                    │  refusal detector              │  refusal = labeled outcome
                    └──────────────┬───────────────┘
                          ┌────────┴────────┐
                 ┌────────▼───────┐ ┌───────▼─────────┐
                 │ 8Values scorer  │ │ Political Compass│
                 │ (ported JS,     │ │ scorer (headless │
                 │  in-process,    │ │  browser submit,  │
                 │  unlimited scale)│ │  authoritative)   │
                 └────────┬───────┘ └───────┬─────────┘
                          └────────┬────────┘
                                   │
                         ┌─────────▼──────────┐
                         │  SQLite/DuckDB store │  raw text + parsed
                         │  (atomic per-trial   │  answers + scores,
                         │   writes, resumable) │  fully re-scoreable
                         └─────────┬──────────┘
                                   │
                         ┌─────────▼──────────┐
                         │  QA / spot-check    │  golden-answer unit tests,
                         │  + cost governor    │  random manual-UI diffing
                         └────────────────────┘
```

## Components

### 1. Question banks (one-time)
Transcribe each test's questions, answer scale, and (for 8Values) axis
mapping into structured JSON once. This replaces the "carefully transcribed
each question into a Google document" step permanently — never done by hand
again.

### 2. Model gateway
Use a multi-provider aggregator (**OpenRouter** is the natural fit) so
"a ton of different models" means adding a line to a config file, not writing
a new SDK integration per provider. One API key and one request schema reach
OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek, Qwen, and dozens more,
including open-weight models at low per-token cost. Pin exact model
snapshot/version strings in the config (fixes the reproducibility gap flagged
in `improvement.md`).

### 3. Structured responses instead of free text
Use each model's JSON-mode / structured-output / tool-calling feature to force
responses as `{question_id, answer}` pairs in the test's own answer scale.
This is the step that eliminates manual re-typing *and* fragile prose parsing
in one move. For models without structured-output support, fall back to a
strict format instruction + parser with automatic "please respond only in
this JSON schema" retry on malformed output.

### 4. Scoring adapters
- **8Values:** port the open-source scoring function directly into the
  pipeline (or shell out to it via a small headless Node/Playwright call if
  porting introduces any risk of drift from the original). Runs in-process,
  free, unlimited scale.
- **Political Compass:** headless-browser adapter (Playwright) that fills the
  real 62-question form with the model's answers, submits, and scrapes the
  resulting economic/social coordinates from the results page. Pool of
  concurrent browser contexts (e.g., 10-20) keeps this fast even at thousands
  of trials.
- **Validation:** golden-answer unit tests (all-agree, all-disagree,
  alternating patterns with known expected scores) run before every batch job,
  plus periodic random spot-checks diffing the automated scorer against a
  manual walkthrough of the real site.

### 5. Orchestration and scale
- Generate the full experimental grid up front — models × trials × prompt
  templates × temperature × (optionally) time points — the same
  `expand.grid`-style design already used in the R simulation work, extended
  to also vary prompt wording (fixes the single-fixed-prompt validity threat
  from `improvement.md`) and to support repeated measurement across calendar
  time (so "drift" can finally mean actual drift, not just within-session
  variance).
- Async worker pool with per-provider rate limiting and exponential backoff.
- **Resumable by design**, mirroring the pattern already proven in
  `genSlowData_3D_fast.R`: every completed (model, trial, template) unit
  writes its own result atomically and is skipped on restart. A crash loses
  at most the in-flight batch, never prior progress.
- Refusals are captured and stored as a labeled outcome, not discarded —
  refusal-rate-by-model-and-topic is itself a viable finding, per
  `improvement.md` item 9.
- **Cost governor:** pre-flight token/cost estimate before launching a large
  batch, running spend tracked against a configured budget cap per provider,
  with an alert/pause when approaching it.

### 6. Storage
SQLite or DuckDB, not Google Sheets. Tables: `trials` (model, version,
template, temperature, timestamp, status), `raw_responses` (full model text,
kept for auditability), `answers` (parsed per-question answers), `scores`
(per-axis results per test). Keeping raw responses means the whole dataset can
be **re-scored later for free** if a scoring bug is found, and the database
itself can be released as the paper's open dataset — directly answering the
open-science gap in `improvement.md`.

## Realistic scale and cost framing

"Millions of data points" is achievable, but it's worth being precise about
what that means. A design of **30 models × 300 trials/model × 2 tests ×
~65 questions/test** is about **1.17 million individual question-answer data
points**, generated from "only" 18,000 full test administrations — a ~150×
increase over the original 120, and comfortably beyond the 19–77 model /
thousands-of-generations scale of the comparator papers in `improvement.md`.
Rough cost estimate for that scale, using a mix of frontier and open-weight
models via OpenRouter, is on the order of a few hundred to low thousands of
dollars, achievable in days with async concurrency rather than weeks. Pushing
to tens of millions of *questions* would mean either far more trials per model
(diminishing statistical returns) or far more models (better — spend the
budget on breadth of models, which is what the field's own comparator papers
prioritize).

## Suggested phased rollout

1. Build and validate both scoring adapters against known golden answers and
   a handful of the original 40 manually-collected trials (must reproduce the
   same scores).
2. Wire up the model gateway and structured-output prompting for the original
   3 models; reproduce the original 120-trial dataset end-to-end
   automatically as a correctness check.
3. Expand the model list and trial count; add prompt-template variation.
4. Add the cost governor and resumability layer before running any
   large/expensive batch.
5. Run the full-scale collection, with periodic spot-check QA throughout.

This is a design proposal — happy to start scaffolding the actual pipeline
code (question-bank JSON, OpenRouter client, scoring adapters, orchestrator)
whenever you want to move to implementation.
