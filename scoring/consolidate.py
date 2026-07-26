"""Consolidate the raw per-trial JSON files in data/raw_trials/ into:
  - data/scores.csv: one row per successful trial (model, test, trial, axis scores)
  - data/collection_summary.md: completion/refusal rates and cost by model

This is the "clean, released dataset" for improvement.md #10 -- raw answers
are preserved in data/raw_trials/*.json (auditable/re-scoreable); this script
just flattens the scored subset into an analysis-ready table.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw_trials"


def main():
    """Read every JSON file in data/raw_trials/, then write:
      - data/scores.csv: one row per successful ("ok" status) trial, with
        that trial's scores spread across the axis1..axis4_name/value
        columns (8values trials use all 4; political_compass trials use
        only axis1/axis2 and leave axis3/axis4 blank).
      - data/collection_summary.md: a per-(model, test) table of completed
        count, failed count, completion rate, and total API cost, plus a
        grand total.
    Trials with any other status (parse_error, score_error: ...) are counted
    as failed in the summary and excluded from scores.csv."""
    rows = []
    status_by_model_test = defaultdict(lambda: {"ok": 0, "fail": 0, "cost": 0.0})

    for path in sorted(RAW_DIR.glob("*.json")):
        d = json.loads(path.read_text())
        key = (d["model"], d["test"])
        status_by_model_test[key]["cost"] += d.get("cost") or 0.0
        if d["status"] == "ok":
            status_by_model_test[key]["ok"] += 1
            scores = d["scores"]
            if d["test"] == "8values":
                rows.append({
                    "model": d["model"], "test": "8values", "trial": d["trial"],
                    "axis1_name": "equality", "axis1_value": scores["equality"],
                    "axis2_name": "peace", "axis2_value": scores["peace"],
                    "axis3_name": "liberty", "axis3_value": scores["liberty"],
                    "axis4_name": "progress", "axis4_value": scores["progress"],
                })
            else:
                rows.append({
                    "model": d["model"], "test": "political_compass", "trial": d["trial"],
                    "axis1_name": "economic", "axis1_value": scores["economic"],
                    "axis2_name": "social", "axis2_value": scores["social"],
                    "axis3_name": "", "axis3_value": "",
                    "axis4_name": "", "axis4_value": "",
                })
        else:
            status_by_model_test[key]["fail"] += 1

    scores_path = DATA_DIR / "scores.csv"
    with open(scores_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "test", "trial", "axis1_name", "axis1_value",
            "axis2_name", "axis2_value", "axis3_name", "axis3_value",
            "axis4_name", "axis4_value",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} scored trials to {scores_path}")

    summary_path = DATA_DIR / "collection_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Collection summary\n\n")
        f.write("| Model | Test | Completed | Failed | Completion rate | Cost |\n")
        f.write("|---|---|---|---|---|---|\n")
        total_cost = 0.0
        for (model, test), s in sorted(status_by_model_test.items()):
            total = s["ok"] + s["fail"]
            rate = 100 * s["ok"] / total if total else 0
            total_cost += s["cost"]
            f.write(f"| {model} | {test} | {s['ok']} | {s['fail']} | {rate:.0f}% | ${s['cost']:.4f} |\n")
        f.write(f"\n**Total observed cost: ${total_cost:.4f}**\n")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
