"""Temperature-sensitivity check (Plan B 2b): re-administer 8Values to a cheap model subset at
temperatures 0 and 1.0, same prompt wrapper and protocol as the main run (collect_data.py).

The main study used temperature 0.7 only. This is a bounded check, not a replication: 8 models,
8Values only (Political Compass needs the live-site browser scorer), 20 trials per cell. The
temperature-0.7 comparison uses the first 20 trials of the existing main-run data. Results go to
data/temperature_sweep/T<temp>/. Needs OPENROUTER_API_KEY; costs about $0.5; resumable.

    export OPENROUTER_API_KEY=...
    python3 scoring/collect_temperature.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_data  # noqa: E402

MODELS = [
    "deepseek/deepseek-v3.2", "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-4-maverick",
    "mistralai/mistral-small-3.2-24b-instruct", "google/gemini-2.5-flash", "openai/gpt-4o-mini",
    "qwen/qwen3-30b-a3b", "amazon/nova-pro-v1",
]
TEMPS = (0.0, 1.0)
N_TRIALS = 20
ROOT = Path(__file__).resolve().parent.parent / "data" / "temperature_sweep"

if __name__ == "__main__":
    for t in TEMPS:
        collect_data.TEMPERATURE = t
        print(f"=== temperature {t} ===")
        collect_data.main(MODELS, n_trials_per_test=N_TRIALS, max_workers=6, cost_cap=1.5,
                          tests=("8values",), results_dir=ROOT / f"T{t}")
