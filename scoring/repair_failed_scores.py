"""Re-score trials that failed at the local scoring step (score_error) using
their already-saved LLM answers -- no re-billing, since the model call
already succeeded and cost money; only the local Playwright automation had
failed. Run after fixing score_political_compass.py's robustness.
"""

import json
from pathlib import Path

from score_8values import load_questions as load_8v_questions
from score_8values import score_8values
from score_political_compass import load_questions as load_pc_questions
from score_political_compass import score_political_compass

DATA_DIR = Path(__file__).parent.parent / "data" / "raw_trials"


def main():
    """Scan data/raw_trials/ for records whose status starts with
    "score_error" (the model call succeeded and its answers were saved, but
    the local scoring step raised an exception at the time), re-score them
    with the already-saved `answers` using the current scoring adapters, and
    overwrite the record in place with status "ok" and a "repaired": true
    flag if scoring now succeeds. Records with no saved answers, or whose
    status is anything else (e.g. "ok" or "parse_error"), are left
    untouched. Prints a per-file outcome and a final repaired/still-failing
    count."""
    questions_8v = load_8v_questions()
    questions_pc = load_pc_questions()

    repaired, still_failed = 0, 0
    for path in sorted(DATA_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        if not record["status"].startswith("score_error"):
            continue
        answers = record.get("answers")
        if not answers:
            continue  # nothing to repair with
        try:
            if record["test"] == "8values":
                scores = score_8values(answers, questions_8v)
            else:
                scores = score_political_compass(answers, questions_pc)
            record["scores"] = scores
            record["status"] = "ok"
            record["repaired"] = True
            path.write_text(json.dumps(record, indent=2))
            repaired += 1
            print(f"REPAIRED {path.name}: {scores}")
        except Exception as e:
            still_failed += 1
            print(f"STILL FAILING {path.name}: {e}")

    print(f"\nRepaired {repaired}, still failing {still_failed}.")


if __name__ == "__main__":
    main()
