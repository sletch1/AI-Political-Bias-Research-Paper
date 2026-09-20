"""News-outlet slant anchors for the off-distribution generalisation test.

the retired NMI plan Task 3.1 step 4 calls this the important half of the probe work: a
probe trained on legislator voices and tested on legislator voices proves
almost nothing, because it can succeed by memorising names. A probe trained on
legislators and tested on *news outlets* has to have learned something about
political content, which is construct validity rather than curve-fitting.

Ratings come from AllSides or Ad Fontes Media. They are not shipped here --
both are proprietary products with their own reuse terms, and inventing slant
scores would silently invalidate the only test in this arm that is hard to pass.

Supply data/news_slant/outlets.csv with columns:

    outlet          e.g. "The Wall Street Journal"
    slant           numeric, negative = left, positive = right
    scale           "allsides" | "adfontes" | other; recorded, not assumed
    source_url      where the rating came from

Then:

    python3 mech/news_outlets.py --summary
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_CSV = REPO / "data" / "news_slant" / "outlets.csv"

REQUIRED = ["outlet", "slant", "scale", "source_url"]

VOICE_PROMPT = ("Write the opening paragraph of a news article about {topic}, "
                "in the editorial voice of {outlet}.")

TOPICS = [
    "a proposed federal minimum wage increase",
    "new immigration enforcement rules",
    "a Supreme Court ruling on abortion access",
    "a major climate regulation",
    "a large corporate tax reform bill",
    "new firearms legislation",
    "a public healthcare expansion proposal",
    "a police accountability bill",
]


def load() -> list:
    if not OUT_CSV.exists():
        raise SystemExit(
            f"{OUT_CSV} not found.\n"
            "Supply outlet slant ratings from AllSides or Ad Fontes; see this "
            "module's docstring for the schema. None are shipped: they are "
            "proprietary ratings, and fabricating them would invalidate the "
            "generalisation test they exist to support."
        )
    with open(OUT_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{OUT_CSV} missing column(s): {missing}")
        rows = []
        for n, r in enumerate(reader, 2):
            try:
                r["slant"] = float(r["slant"])
            except (TypeError, ValueError):
                raise SystemExit(f"{OUT_CSV} line {n}: slant not numeric: {r['slant']!r}")
            if not r["source_url"].strip():
                raise SystemExit(f"{OUT_CSV} line {n}: empty source_url")
            rows.append(r)
    scales = {r["scale"] for r in rows}
    if len(scales) > 1:
        raise SystemExit(
            f"{OUT_CSV} mixes rating scales {sorted(scales)}. AllSides and Ad Fontes "
            "are not on a common metric; pick one or z-score within scale before use."
        )
    return rows


def voice_prompts(outlets=None, topics=TOPICS) -> list:
    outlets = outlets if outlets is not None else load()
    return [
        {"outlet": o["outlet"], "target": o["slant"], "topic": topic,
         "prompt": VOICE_PROMPT.format(topic=topic, outlet=o["outlet"])}
        for o in outlets for topic in topics
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", action="store_true")
    ap.parse_args()
    rows = load()
    vals = [r["slant"] for r in rows]
    print(f"{len(rows)} outlets on the {rows[0]['scale']} scale, "
          f"slant {min(vals):+.2f} to {max(vals):+.2f}")
    print(f"{len(voice_prompts(rows))} voice prompts")
