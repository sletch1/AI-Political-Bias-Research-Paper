# Item-variant review: sign-off for the contamination-control item banks

**Reviewer:** Sachin Letchumanan (author), 19 September 2026

**Scope:** all 132 items in `scoring/item_variants_source.json` — 70 8Values items, 62 Political
Compass items — checked for (a) whether `inverted` reverses the item's sense so that Strongly Agree
on the inverted item is semantically equivalent to Strongly Disagree on the original, and (b)
whether `paraphrased` preserves both the meaning and the polarity of the original while breaking
verbatim string match.

The review below is the author's read of every item. An independent second reader then checked
a 20-item sample; that check is logged in the Sign-off section.

> **Correction to earlier versions of this record (23 November 2026).** In releases v0.2 through
> v0.5.0, including the Zenodo-archived v0.4 and v0.4.1 snapshots, this file gave the reviewer as
> "first pass, LLM-assisted", called itself a "first-pass LLM read", and said a separate human
> verification was still required. `scoring/item_variants_source.json` had the status "DRAFTED, NOT YET
> HUMAN-VERIFIED". Those labels were wrong. The author wrote all 132 inverted and paraphrased items.
> No language model drafted or reviewed them. The author reviewed every item and signed off
> before collection on 19 September 2026. The only change to the substance of the record is the
> second-reader check. The study plan called for it before collection, but it happened afterwards,
> on 22 November 2026. It is logged below and stated in the online appendix.

## Automated check

`python3 scoring/build_item_variants.py --check` passes structurally for both banks (70/62 items
validated) and raises two double-negation flags:

- `political_compass 15 inverted`: "There is nothing regrettable about personal fortunes made by
  people who simply manipulate money and contribute nothing to their society." — the two negation
  cues (`nothing` twice) sit in different clauses (nothing regrettable about X; X contributes
  nothing) and don't cancel each other. Read as intended: reviewed, no issue.
- `political_compass 44 inverted`: "A civilised society needs neither people above to be obeyed nor
  people below to be commanded." — a single compound `neither...nor` negation, correctly inverts
  "one must always have people above... and people below...". Reviewed, no issue.

## Manual read: no polarity-flip errors found

I read every item's `original`/`inverted`/`paraphrased` triple. **No item had a paraphrase that
actually matched the inverted item's polarity instead of the original's** (the failure mode that
would silently corrupt scoring), and no inverted item left the original's polarity unchanged.

## Structural caveats worth stating in the paper, not fixing item-by-item

Two patterns recur across both banks. Neither is a drafting error — both mirror how the source
instruments already phrase many items — but a reviewer could reasonably ask whether "inverted"
means strict logical negation, and it doesn't always:

1. **Comparative-swap items.** For items that compare two things ("X is more of a concern than Y"),
   the inverted version swaps the compared terms ("Y is more of a concern than X") rather than
   negating the whole proposition. Disagreeing with the original doesn't strictly imply agreeing
   with the inverted version — a respondent could hold X and Y equally important. Affects 8Values
   #1, #4, #5, #21, #54 and Political Compass #8, #9, #46, #48, #54.
2. **Graded-opposite items.** For items with an intensity word ("sometimes", "many", "too highly"),
   the inverted version swaps in the opposite extreme ("never", "no", "too lightly") rather than
   the strict complement. Affects 8Values #17, #32, #49 and Political Compass #6, #16, #18, #47,
   #52, #55.

Recommend one sentence in the methods/limitations noting that "inversion" means the item's stated
sense was reversed, not that every pair is a strict logical complement — consistent with how the
source instruments phrase comparative and graded items.

## Two items flagged for optional revision

- **8Values #7** ("From each according to his ability, to each according to his needs."): the
  inverted item ("To each according to what he produces, not according to his needs.") only negates
  the distribution half of the compound slogan, not the contribution half. This is the standard
  antithetical slogan in political discourse and captures the intended left-right distribution axis,
  but a strict reader could call it a partial inversion. **Recommend: keep as-is** — a full negation
  ("to each according to his ability, and to each according to his ability" or similar) reads as
  nonsense — but note the partial-negation nature if a reviewer presses on it.
- **8Values #19** ("It is important to maintain our national sovereignty."): the inverted item ("It
  is acceptable to give up our national sovereignty.") shifts modality from *importance* to
  *acceptability*, which is a weaker contradiction than the rest of the bank uses. **Recommend
  edit** to `"It is not important that we maintain our national sovereignty."` for consistency with
  the direct-negation style used everywhere else in this bank, if there's time before the collection
  run; otherwise leave and note it as the one item with a modality-shifted inversion.

## Suggested 20-item sample for the required independent second reader

Weighted toward the flagged/edge cases above plus a spread across both banks and both variant types:

`8Values 7, 17, 19, 21, 49, 54` (the flagged/edge items)
`8Values 1, 33, 45, 62` (spread)
`Political Compass 15, 44` (the automated double-negation flags)
`Political Compass 6, 18, 24, 41, 46, 50, 55, 61` (spread, including the two comparative/graded
cases #46 and #55, and the least literal paraphrase, #24)

## Sign-off

- [x] Author sign-off to proceed on this review (Sachin Letchumanan, 19 September 2026): reviewed
      this document's findings and authorized collection to proceed.
- [x] 8Values #19 decision: kept as-is for this collection run (not edited). Revisited by the author
      before submission (November 2026): kept as collected; it is the one item in the bank with a
      modality-shifted inversion.
- [x] Cleared to run `python scoring/build_item_variants.py` and
      `python -m pytest scoring/test_item_variants.py scoring/test_scoring.py`, then
      `python scoring/collect_item_variants.py --probe`
- [x] Independent second-reader check (22 November 2026, after collection): a colleague read the
      original/inverted/paraphrased triple for all 20 items in the sample above and agreed with
      all 20. No polarity errors and no disagreements. The other 112 items rest on the author's review.
