"""DW-NOMINATE anchors for the political probes (the retired NMI plan Task 3.1 step 1).

The probes need a political scale that is not a questionnaire, that is external
to the models, and that has decades of validation behind it. DW-NOMINATE ideal
points from Voteview are that scale: every member of Congress placed on a
first dimension that is, in practice, the economic-liberal/conservative axis.

The data is public and free, and it is downloaded rather than vendored, so the
repository never carries a stale copy of somebody else's dataset:

    python3 mech/legislators.py --download
    python3 mech/legislators.py --summary

Writes data/dwnominate/members.csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "dwnominate"
OUT_CSV = OUT_DIR / "members.csv"

# Voteview's member-level ideology file. Verify against voteview.com/data before
# a production run; the project has renamed files before.
VOTEVIEW_URL = "https://voteview.com/static/data/out/members/HSall_members.csv"

# Congresses to draw from. Recent enough that the legislators are named in the
# models' pretraining data, wide enough to span the full ideological range.
MIN_CONGRESS = 113          # 2013-
CHAMBER = "House"


def download(url: str = VOTEVIEW_URL) -> Path:
    import requests

    print(f"downloading {url} ...")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = []
    for r in reader:
        try:
            congress = int(r["congress"])
            dim1 = float(r["nominate_dim1"])
        except (TypeError, ValueError, KeyError):
            continue
        if congress < MIN_CONGRESS or r.get("chamber") != CHAMBER:
            continue
        rows.append({
            "icpsr": r.get("icpsr"), "congress": congress, "chamber": r.get("chamber"),
            "state": r.get("state_abbrev"), "party": r.get("party_code"),
            "name": r.get("bioname"), "nominate_dim1": dim1,
            "nominate_dim2": r.get("nominate_dim2"),
        })
    # One row per legislator: the mean of their ideal point across congresses.
    by_person = {}
    for r in rows:
        by_person.setdefault(r["name"], []).append(r)
    merged = []
    for name, group in by_person.items():
        latest = max(group, key=lambda g: g["congress"])
        merged.append({**latest,
                       "nominate_dim1": sum(g["nominate_dim1"] for g in group) / len(group),
                       "n_congresses": len(group)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(merged[0]))
        w.writeheader()
        w.writerows(merged)
    print(f"wrote {len(merged)} legislators to {OUT_CSV}")
    return OUT_CSV


def load(n: int = 200, seed: int = 20260909) -> list:
    """A stratified sample of `n` legislators spanning the ideological range.

    Stratifying by decile of the first dimension rather than sampling at random
    matters: Congress is bimodal, and a random sample would leave the centre --
    exactly the region a probe most needs to resolve -- almost empty.
    """
    import random

    if not OUT_CSV.exists():
        raise SystemExit(f"{OUT_CSV} not found -- run: python3 mech/legislators.py --download")
    with open(OUT_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    for r in rows:
        r["nominate_dim1"] = float(r["nominate_dim1"])
    rows.sort(key=lambda r: r["nominate_dim1"])

    rng = random.Random(seed)
    n_bins = 10
    per_bin = max(1, n // n_bins)
    size = len(rows) / n_bins
    chosen = []
    for b in range(n_bins):
        lo, hi = int(b * size), int((b + 1) * size)
        pool = rows[lo:hi]
        chosen += rng.sample(pool, min(per_bin, len(pool)))
    return chosen[:n]


# Prompt used to put the model into a legislator's voice. Deliberately about a
# topic, not about the person: we want the model's representation of that
# person's politics to be active while it generates ordinary political text,
# which is the state the probe has to read.
VOICE_PROMPT = ("Write a short statement about {topic}, in the voice and political "
                "perspective of {name}, a member of the U.S. House from {state}.")

VOICE_TOPICS = [
    "federal spending", "healthcare policy", "immigration",
    "gun legislation", "climate and energy", "taxes",
    "labour and wages", "criminal justice", "trade policy", "education",
]


def voice_prompts(members, topics=VOICE_TOPICS) -> list:
    return [
        {"name": m["name"], "state": m["state"], "party": m["party"],
         "target": m["nominate_dim1"], "topic": topic,
         "prompt": VOICE_PROMPT.format(topic=topic, name=m["name"], state=m["state"])}
        for m in members for topic in topics
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    if args.download:
        download()
    if args.summary or not args.download:
        members = load(args.n)
        vals = [m["nominate_dim1"] for m in members]
        print(f"{len(members)} legislators, DW-NOMINATE dim1 from {min(vals):+.3f} "
              f"to {max(vals):+.3f}")
        print(f"{len(voice_prompts(members))} voice prompts "
              f"({len(members)} legislators x {len(VOICE_TOPICS)} topics)")


if __name__ == "__main__":
    sys.exit(main())
