"""Inter-judge agreement for the open-ended arm (Plan B 2f) -- NOT a human validation.

The open-ended arm's passages were scored by one judge (openai/gpt-5-mini). Human ratings are the proper
validation and were not collected. This script re-scores the same 120 stored passages with two further
judges from other model families and reports agreement with the original scores (Spearman rho, absolute
difference on the -10..10 scale, ICC(2,1)) with bootstrap CIs over passages. High agreement among LLM judges
shows the scores are not an idiosyncrasy of one judge; it does not show they match human raters.

    export OPENROUTER_API_KEY=...   # re-judging costs about $0.5
    python3 scoring/validate_judge_agreement.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Offline mode: with the cached re-judgements in results/judge_agreement/rejudged.json no API call is made,
# so a placeholder key lets collect_openended (which insists on a key at import) load.
OFFLINE = not os.environ.get("OPENROUTER_API_KEY")
if OFFLINE:
    os.environ["OPENROUTER_API_KEY"] = "offline-cached"
import collect_openended as co  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/judge_agreement"
OTHER_JUDGES = ["anthropic/claude-sonnet-4.5", "x-ai/grok-4.20"]
TOPICS = {t["id"]: t for t in co.TOPICS} if hasattr(co, "TOPICS") else None


def icc21(x):
    """ICC(2,1), two-way random, absolute agreement; x is (n_items, k_raters)."""
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def main():
    recs = [json.loads(p.read_text()) for p in sorted((REPO / "data/openended").glob("*.json"))]
    recs = [r for r in recs if r.get("status") == "ok"]
    cache = OUT / "rejudged.json"
    OUT.mkdir(parents=True, exist_ok=True)
    done = json.loads(cache.read_text()) if cache.exists() else {}
    topic_by_id = {t["id"]: t for t in co.TOPICS}
    for j in OTHER_JUDGES:
        for r in recs:
            key = f"{j}|{r['model']}|{r['topic']}|{r['trial']}"
            if key in done:
                continue
            if OFFLINE:
                raise SystemExit("cache incomplete and OPENROUTER_API_KEY is not set")
            t = topic_by_id[r["topic"]]
            prompt = co.JUDGE_PROMPT.format(prompt_topic=t["prompt_topic"], axis_desc=co.AXIS_DESC[t["axis"]], passage=r["passage"])
            val, cost = None, 0.0
            for _ in range(3):
                try:
                    raw, c = co.call_api(j, prompt, max_tokens=1500); cost += c
                except Exception:
                    continue
                p = co.parse_judge_json(raw)
                if p: val = p["lean_score"]; break
            done[key] = {"score": val, "cost": cost}
        cache.write_text(json.dumps(done))
    rng = np.random.default_rng(20260919)
    ok = [r for r in recs if all(done[f"{j}|{r['model']}|{r['topic']}|{r['trial']}"]["score"] is not None for j in OTHER_JUDGES)]
    cols = {"openai/gpt-5-mini (original)": np.array([float(r["lean_score"]) for r in ok])}
    for j in OTHER_JUDGES:
        cols[j] = np.array([done[f"{j}|{r['model']}|{r['topic']}|{r['trial']}"]["score"] for r in ok])
    names = list(cols); X = np.column_stack([cols[n] for n in names]); n = len(X)
    res = {"n_passages": n, "n_requested": len(recs), "judges": names,
           "total_cost": sum(v["cost"] for v in done.values()),
           "note": "agreement among LLM judges; NOT validation against human raters"}
    def boot(fn):
        d = [fn(X[rng.integers(0, n, n)]) for _ in range(2000)]
        return [float(v) for v in np.percentile(d, [2.5, 97.5])]
    pairs = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            f_rho = lambda M, a=a, b=b: spearmanr(M[:, a], M[:, b])[0]
            f_mad = lambda M, a=a, b=b: np.abs(M[:, a] - M[:, b]).mean()
            pairs[f"{names[a]} vs {names[b]}"] = {
                "spearman": float(f_rho(X)), "spearman_ci": boot(f_rho),
                "mean_abs_diff": float(f_mad(X)), "mean_abs_diff_ci": boot(f_mad)}
    res["pairs"] = pairs
    res["icc21_all_judges"] = {"estimate": float(icc21(X)), "ci": boot(icc21)}
    (OUT / "judge_agreement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
