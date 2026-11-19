"""Export the two analysis panels (direction-aligned, z-scored within trait) for the independent R/lme4 cross-check.
Writes results/lme4_check/panel_two.csv and panel_four.csv.   python export_panels_for_r.py"""
from pathlib import Path
import numpy as np
from analyze_f3_variance_components import load_pooled_long

out = Path(__file__).resolve().parent.parent / "results" / "lme4_check"; out.mkdir(parents=True, exist_ok=True)
for name, inst in (("two", ("political_compass", "8values")), ("four", None)):
    d = load_pooled_long(inst) if inst else load_pooled_long()
    d["y"] = d.groupby("trait")["value"].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    d[["model", "instrument", "trait", "trial", "y"]].to_csv(out / f"panel_{name}.csv", index=False)
    print(name, len(d), d.model.nunique())
