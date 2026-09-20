"""Build the inverted and paraphrased question banks for updates/03_experiments.md Task 1.2.

Task 1.2 tests whether models recognise the Political Compass and 8Values
rather than answering their items. That needs three item conditions:

    original      verbatim items (already collected, data/raw_trials/)
    inverted      each item's *sense* negated, scoring key flipped
    paraphrased   meaning and polarity preserved, verbatim string match broken

The variant *text* is authored by hand in item_variants_source.json. This
script does only the mechanical half: it attaches each original item's scoring
metadata to its variant text, validates the result, and writes four banks that
are drop-in replacements for the originals:

    questions_8values_inverted.json
    questions_8values_paraphrased.json
    questions_political_compass_inverted.json
    questions_political_compass_paraphrased.json

Crucially the metadata is copied unchanged -- 8Values keeps its per-axis
`effect` weights, Political Compass keeps its `name`/`page` form fields. An
inverted bank is therefore scored by flipping the *answers* and handing them
to the unmodified authoritative engine (see variant_scoring.py), not by
editing the scoring key. The scoring engines are never touched, so every
condition stays comparable to the published 2,280-administration baseline.

    python3 scoring/build_item_variants.py          # write the banks
    python3 scoring/build_item_variants.py --check   # validate without writing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "item_variants_source.json"

# (instrument key, original bank filename, output stem)
INSTRUMENTS = [
    ("8values", "questions_8values.json", "questions_8values"),
    ("political_compass", "questions_political_compass.json", "questions_political_compass"),
]

VARIANTS = ("inverted", "paraphrased")

# Cheap red flags for the classes of drafting error updates/03_experiments.md Task 1.2 step 1 warns about.
# These do not replace human review; they catch the mechanical mistakes so the
# reviewer's attention goes to semantics.
_NEGATORS = (
    "not", "never", "no", "none", "nothing", "nobody", "cannot", "nor",
    "without", "rarely", "neither", "unless",
    "can't", "shouldn't", "don't", "doesn't", "isn't", "aren't", "wasn't", "won't",
    # negated-prefix forms that actually occur in these banks; a blanket
    # ``un\\w+`` pattern fires on "unions"/"unemployment" and buries the signal
    "unnecessary", "unacceptable", "unimportant", "unnatural", "unjustified",
    "unpenalised", "unremarkable", "illegitimate", "illegal", "immoral",
    "impossible", "irrelevant", "inability", "pointless",
)
_NEG = r"\b(" + "|".join(_NEGATORS) + r")\b"


def count_negations(text: str) -> int:
    return len(re.findall(_NEG, text, flags=re.IGNORECASE))


def load_source() -> dict:
    with open(SOURCE, encoding="utf-8") as fh:
        return json.load(fh)


def load_bank(filename: str) -> list:
    with open(HERE / filename, encoding="utf-8") as fh:
        return json.load(fh)


def build_bank(originals: list, entries: list, variant: str) -> list:
    """Return `originals` with each item's question text swapped for its
    `variant` text, all other fields (scoring weights, form-field names,
    page numbers) copied through untouched, plus provenance fields."""
    if len(originals) != len(entries):
        raise ValueError(f"bank has {len(originals)} items, source has {len(entries)}")
    out = []
    for orig, entry in zip(originals, entries):
        if entry["original"] != orig["question"]:
            raise ValueError(
                f"source item {entry['index']} is out of sync with the live bank:\n"
                f"  bank:   {orig['question']!r}\n  source: {entry['original']!r}"
            )
        item = dict(orig)
        item["question"] = entry[variant]
        item["variant"] = variant
        item["source_question"] = entry["original"]
        out.append(item)
    return out


def validate(instrument: str, originals: list, entries: list) -> list:
    """Structural checks over the authored text. Returns a list of warnings;
    raises on anything that would silently corrupt the experiment."""
    warnings = []
    for entry in entries:
        idx, orig = entry["index"], entry["original"]
        for variant in VARIANTS:
            text = entry[variant].strip()
            if not text:
                raise ValueError(f"{instrument} item {idx}: empty {variant} text")
            if text == orig:
                raise ValueError(f"{instrument} item {idx}: {variant} is identical to the original")
            if not text.endswith((".", "?", "!", "”", '"')):
                warnings.append(f"{instrument} {idx} {variant}: no terminal punctuation")
        # A double negative is the single most common way an inversion goes
        # wrong (updates/03_experiments.md Task 1.2 step 1); flag any inversion that stacks negations.
        inv_neg = count_negations(entry["inverted"])
        if inv_neg >= 2:
            warnings.append(
                f"{instrument} {idx} inverted: {inv_neg} negation cues -- "
                f"check for a double negative: {entry['inverted']!r}"
            )
        # A paraphrase that keeps a long verbatim run has not destroyed the
        # string match it exists to destroy.
        run = _longest_shared_run(orig.lower(), entry["paraphrased"].lower())
        if run >= 40:
            warnings.append(
                f"{instrument} {idx} paraphrased: shares a {run}-character run "
                "with the original; verbatim match may survive"
            )
    texts = [e["inverted"] for e in entries] + [e["paraphrased"] for e in entries]
    if len(set(texts)) != len(texts):
        # Duplicates are legitimate only where the instruments share an item
        # verbatim (8Values 38 and Compass 23 are both "All authority should
        # be questioned"), so this is checked within an instrument only.
        dupes = {t for t in texts if texts.count(t) > 1}
        warnings.append(f"{instrument}: duplicate variant text: {sorted(dupes)[:3]}")
    return warnings


def _longest_shared_run(a: str, b: str) -> int:
    """Length of the longest common substring, used to detect paraphrases that
    left too much of the original intact."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only; write nothing")
    args = ap.parse_args()

    source = load_source()
    all_warnings = []
    written = []

    for key, filename, stem in INSTRUMENTS:
        originals = load_bank(filename)
        entries = source[key]
        all_warnings += validate(key, originals, entries)
        for variant in VARIANTS:
            bank = build_bank(originals, entries, variant)
            path = HERE / f"{stem}_{variant}.json"
            if not args.check:
                path.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
                written.append(path)
        print(f"{key}: {len(originals)} items x {len(VARIANTS)} variants validated")

    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s) for human review:")
        for w in all_warnings:
            print(f"  ! {w}")
    else:
        print("\nno structural warnings")

    for path in written:
        print(f"wrote {path.relative_to(HERE.parent)}")

    print("\nREMINDER (updates/03_experiments.md Task 1.2 step 1): these banks are drafted, not verified. "
          "Hand-check every inversion and have a second reader check a 20-item "
          "sample before spending API budget on them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
