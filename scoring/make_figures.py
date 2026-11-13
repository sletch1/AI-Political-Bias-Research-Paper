"""Submission figures: vector PDF (+ 300 dpi TIFF), Arial text, with CIs.

Journal figure requirements: Arial or Verdana text, 300+ dpi, TIFF/EPS/PDF, multi-panel figures with
identical axes, and uncertainty on plotted numbers. This script regenerates the two main-text
figures from the result JSONs (never from hard-coded numbers):

  fig_error_budget.{pdf,tiff}   nested variance decomposition with bootstrap CIs
  fig_mtmm.{pdf,tiff}           MTMM coefficients with standard errors and 95% CIs

Run after analyze_nested_design.py and analyze_se_supplement.py:

    python3 scoring/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "figures"
plt.rcParams.update({
    "font.family": ["Arial", "Verdana", "DejaVu Sans"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Colour-blind-safe, each pair >= 3:1 against white (non-text marks) or against label colour.
COL = {"model": "#0b5cad", "model:instrument": "#b35806", "model:trait": "#e0a458",
       "residual": "#6b6b6b"}
LAB = {"model": "Model (portable)", "model:instrument": "Model × instrument",
       "model:trait": "Model × axis within instrument", "residual": "Trial noise"}
ORDER = ["model", "model:instrument", "model:trait", "residual"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tiff", dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def fig_error_budget():
    nd = json.loads((REPO / "results/nested_design/nested_design.json").read_text())
    from make_w4_figures import per_axis_error_budget  # reuse the within-axis computation
    import pandas as pd
    from analyze_f3_variance_components import load_pooled_long
    df = load_pooled_long(("political_compass", "8values"))
    budget = per_axis_error_budget(df)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True,
                               gridspec_kw={"width_ratios": [1.15, 1.6]})
    # Panel (a): nested shares with model-clustered 95% CIs, for both panels of instruments.
    panels = [("two_instruments", "2 instruments\n(n = 2,280)"),
              ("four_instruments", "4 instruments\n(n = 3,000)")]
    w = 0.16
    for pi, (key, title) in enumerate(panels):
        ent = nd[key]["nested"]
        for ki, k in enumerate(ORDER):
            v = ent["point"]["share"].get(k, 0) * 100
            ci = ent["model_bootstrap"]["ci"][k]
            x = pi + (ki - 1.5) * (w + 0.03)
            a.bar(x, v, width=w, color=COL[k], label=LAB[k] if pi == 0 else None)
            a.errorbar(x, v, yerr=[[v - ci["ci_lower"] * 100], [ci["ci_upper"] * 100 - v]],
                       color="black", capsize=2, lw=0.8)
    a.set_xticks([0, 1])
    a.set_xticklabels([t for _, t in panels])
    a.set_ylim(0, 100)
    a.set_ylabel("Share of variance (%)")
    a.set_title("(a) Nested decomposition, pooled", loc="left", fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="upper left")
    a.yaxis.grid(True, color="#dddddd", lw=0.6)
    a.set_axisbelow(True)

    # Panel (b): within-axis model share (instrument held fixed), same y-axis (0-100).
    x = np.arange(len(budget))
    mv = budget["model"].to_numpy() * 100
    b.bar(x, mv, width=0.6, color=COL["model"], label="Model identity")
    b.bar(x, 100 - mv, bottom=mv, width=0.6, color=COL["residual"], label="Trial noise")
    for xi, v in enumerate(mv):
        b.text(xi, v / 2, f"{v:.0f}", ha="center", va="center", color="white", fontsize=8)
    b.set_xticks(x)
    b.set_xticklabels(budget["label"], rotation=30, ha="right", fontsize=8)
    b.set_ylim(0, 100)
    b.set_title("(b) Within a single axis", loc="left", fontsize=9)
    b.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.30))
    b.yaxis.grid(True, color="#dddddd", lw=0.6)
    b.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_error_budget")


def fig_mtmm():
    se = json.loads((REPO / "results/se_supplement/se_supplement.json").read_text())
    pairs = se["mtmm_pairs"]["pairs"]
    nice = lambda s: (s.replace("political_compass", "PC").replace("8values", "8V")
                      .replace(":", " "))
    rows = [(f"{nice(p['a'])} – {nice(p['b'])}", p["r"], p["ci_fisher"], p["type"])
            for p in pairs]
    m = se["mtmm_means"]
    rows.append(("Mean convergent", m["mean_convergent"]["estimate"],
                 [m["mean_convergent"]["ci_lower"], m["mean_convergent"]["ci_upper"]], "convergent"))
    rows.append(("Mean discriminant", m["mean_discriminant"]["estimate"],
                 [m["mean_discriminant"]["ci_lower"], m["mean_discriminant"]["ci_upper"]], "discriminant"))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r, ci, typ) in zip(y, rows):
        c = COL["model:instrument"] if typ == "convergent" else COL["model"]
        bold = lab.startswith("Mean")
        ax.errorbar(r, yi, xerr=[[r - ci[0]], [ci[1] - r]], fmt="D" if bold else "o",
                    color=c, capsize=3, ms=6 if bold else 5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Correlation across models (95% CI)")
    ax.xaxis.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=COL["model:instrument"], ls="", label="Convergent (same construct)"),
                       Line2D([], [], marker="o", color=COL["model"], ls="", label="Discriminant (different construct)")],
              frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    save(fig, "fig_mtmm")


if __name__ == "__main__":
    fig_mtmm()
    if (REPO / "results/nested_design/nested_design.json").exists():
        fig_error_budget()
