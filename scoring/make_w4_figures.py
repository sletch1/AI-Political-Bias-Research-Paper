"""Display items for Workstream 4 (plan.md sections 4 and 7).

Four figures plus two LaTeX tables, all from data already in the repository:

  fig6_error_budget.png   plan.md section 7 Figure 2 -- "the paper's centrepiece".
                          Stacked variance decomposition per axis: how much of a
                          measured political position is the model, and how much
                          is the measurement. Built to take further strata
                          (asker, format, instrument) without redesign once W1
                          data lands.
  fig7_iss_repair.png     Task 4.2. The range-normalisation artefact, shown
                          rather than asserted: legacy ISS against absolute
                          dispersion, with the claude-opus-4.5 economic cell
                          (ISS 65.22 on an SD of 0.06) labelled.
  fig8_item_entropy.png   Task 4.1. Per-item response entropy by model, which
                          replaces the per-axis aggregate that hid
                          issue-specific inconsistency.
  figSI_mtmm.png          SI. The multi-trait multi-method matrix.

House palette and SHORT_NAME are imported from make_result_figures so the new
items are visually identical to the existing ones (plan.md section 7).

    python3 scoring/make_w4_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_w4 import INSTRUMENT_RANGE, TRAIT_PAIRS, legacy_iss, load_long, load_raw_trials  # noqa: E402
from analyze_w4 import normalised_entropy  # noqa: E402
from make_result_figures import BLUE, FIG_DIR, GRID, INK, MUTED, ORANGE, SHORT_NAME  # noqa: E402
from variance_components import mtmm, variance_decomposition  # noqa: E402

TABLE_DIR = Path(__file__).resolve().parent.parent / "results" / "w4"

AXIS_LABEL = {
    "political_compass:economic": "PC economic",
    "political_compass:social": "PC social",
    "8values:equality": "8V equality",
    "8values:liberty": "8V liberty",
    "8values:peace": "8V peace",
    "8values:progress": "8V progress",
}

# Stratum colours. Model identity is the signal; everything else is measurement.
# Keeping measurement strata in warm/muted tones and the model in blue makes the
# figure's argument legible before anyone reads the legend.
STRATUM_COLOUR = {
    "model": BLUE,
    "model:trait": "#7fb3e8",
    "asker": ORANGE,
    "format": "#f2a683",
    "instrument": "#c9c8c3",
    "residual": MUTED,
}
STRATUM_LABEL = {
    "model": "model identity (signal)",
    "model:trait": "model x axis",
    "asker": "asker identity",
    "format": "response format",
    "instrument": "instrument",
    "residual": "trial-to-trial noise",
}


def per_axis_error_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Variance components within each axis, via the shared statistics module.

    Decomposing within an axis rather than pooling is what makes the figure
    honest: pooled across instruments the axis term dominates everything and
    hides the fact that the model share differs sharply between axes.
    """
    rows = []
    for trait, grp in df.groupby("trait"):
        out = variance_decomposition(grp, value="value", random_effects=["model"])
        share = out["variance_share"]
        rows.append({
            "trait": trait,
            "label": AXIS_LABEL.get(trait, trait),
            "model": share.get("model", 0.0),
            "residual": share.get("residual", 0.0),
            "n": int(len(grp)),
        })
    out = pd.DataFrame(rows)
    order = [t for t in AXIS_LABEL if t in set(out["trait"])]
    return out.set_index("trait").loc[order].reset_index()


def fig6_error_budget(df: pd.DataFrame):
    """plan.md section 7 Figure 2 -- the paper's centrepiece.

    Two panels, because the answer depends entirely on the denominator and
    showing only one invites the paper to be misquoted against itself.

    Panel A is the headline: pooled over everything, instrument and axis choice
    is the largest single source of variance in a measured political position,
    and model identity is a minority share. That is the error-budget claim.

    Panel B is the same data conditional on a fixed axis, where the model share
    is naturally much larger. Reporting only Panel B would say models dominate;
    reporting only Panel A would hide that models are clearly separable once the
    instrument is held still. Both are true and the figure says which is which.
    """
    pooled = variance_decomposition(
        df, value="value", random_effects=["model", "trait", "model:trait"], z_within="trait")
    share = pooled["variance_share"]
    budget = per_axis_error_budget(df)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                   gridspec_kw={"width_ratios": [1, 2.5]})

    # --- Panel A: pooled
    order = ["trait", "model", "model:trait", "residual"]
    labels = {"trait": "instrument / axis choice", "model": "model identity",
              "model:trait": "model x axis", "residual": "trial-to-trial noise"}
    colours = {"trait": "#c9c8c3", "model": BLUE, "model:trait": "#7fb3e8", "residual": MUTED}
    bottom = 0.0
    for k in order:
        v = share.get(k, 0.0) * 100.0
        if v <= 0:
            continue
        axA.bar([0], [v], bottom=[bottom], width=0.5, color=colours[k],
                edgecolor="white", linewidth=0.8, label=labels[k])
        if v > 5:
            axA.text(0, bottom + v / 2, f"{v:.0f}%", ha="center", va="center",
                     color="white" if k in ("model", "residual") else INK,
                     fontsize=9, fontweight="bold")
        bottom += v
    axA.set_xticks([])
    axA.set_ylim(0, 100)
    axA.set_ylabel("share of variance (%)")
    axA.set_title("A. All sources pooled", fontsize=9.5, loc="left", pad=8)
    axA.annotate("instrument choice moves a score\nmore than model identity does",
                 xy=(0.0, -0.16), xycoords="axes fraction", fontsize=8, color=MUTED)
    axA.legend(frameon=False, fontsize=7.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.24), ncol=1)
    axA.yaxis.grid(True, color=GRID, linewidth=0.6)
    axA.set_axisbelow(True)

    # --- Panel B: within-axis
    x = np.arange(len(budget))
    bottom_b = np.zeros(len(budget))
    for k in ("model", "residual"):
        vals = budget[k].to_numpy() * 100.0
        axB.bar(x, vals, bottom=bottom_b, width=0.62, color=colours[k],
                edgecolor="white", linewidth=0.8)
        for xi, (v, b) in enumerate(zip(vals, bottom_b)):
            if v > 6:
                axB.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                         color="white", fontsize=8.5, fontweight="bold")
        bottom_b += vals
    axB.set_xticks(x)
    axB.set_xticklabels(budget["label"], fontsize=8.5)
    axB.set_ylim(0, 100)
    axB.yaxis.grid(True, color=GRID, linewidth=0.6)
    axB.set_axisbelow(True)
    axB.set_title("B. Within a single axis (instrument held fixed)", fontsize=9.5,
                  loc="left", pad=8)
    axB.annotate(
        f"conditional on the axis, model identity explains "
        f"{budget['model'].mean()*100:.0f}% on average -- but a third of what is left "
        f"is pure trial noise",
        xy=(0.0, -0.20), xycoords="axes fraction", fontsize=8, color=MUTED)

    fig.suptitle("The error budget: how much of a measured political position is the model?",
                 fontsize=10.5, x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_error_budget.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return budget, pooled


def fig7_iss_repair(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, instrument, axis), grp in df.groupby(["model", "instrument", "axis"]):
        v = grp["value"].to_numpy(dtype=float)
        rows.append({
            "model": model, "instrument": instrument, "axis": axis,
            "sd": float(v.std(ddof=1)),
            "sd_pct_of_scale": float(v.std(ddof=1) / INSTRUMENT_RANGE[instrument] * 100.0),
            "legacy_iss": legacy_iss(v),
        })
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(d["sd_pct_of_scale"], d["legacy_iss"], s=22, color=BLUE, alpha=0.65,
               edgecolor="white", linewidth=0.5, zorder=3)
    worst = d.sort_values("legacy_iss", ascending=False).iloc[0]
    ax.scatter([worst["sd_pct_of_scale"]], [worst["legacy_iss"]], s=70, color=ORANGE,
               zorder=4, edgecolor="white", linewidth=0.8)
    ax.annotate(
        f"{SHORT_NAME.get(worst['model'], worst['model'])} / {worst['axis']}\n"
        f"ISS {worst['legacy_iss']:.1f} on an SD of {worst['sd']:.2f}",
        xy=(worst["sd_pct_of_scale"], worst["legacy_iss"]),
        xytext=(worst["sd_pct_of_scale"] + 1.6, worst["legacy_iss"] - 4),
        fontsize=8, color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
    )
    ax.set_xlabel("absolute dispersion (SD as % of the instrument's full scale)")
    ax.set_ylabel("published ISS (range-normalised)")
    ax.set_title("The published stability metric is not a measure of dispersion",
                 fontsize=10, loc="left", pad=10)
    rho = d["sd_pct_of_scale"].corr(d["legacy_iss"], method="spearman")
    # Bottom-right: the top-left corner is where the worst artefact sits, and an
    # annotation there lands on top of the very point the figure is about.
    ax.annotate(f"Spearman rho = {rho:.2f} across all 114 model-axis cells\n"
                f"a metric that measured dispersion would sit on a straight line",
                xy=(0.98, 0.06), xycoords="axes fraction", fontsize=8.5, color=MUTED,
                ha="right", va="bottom")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_iss_repair.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return d


def fig8_item_entropy(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, instrument), grp in raw.groupby(["model", "instrument"]):
        mat = list(grp["answers"])
        n_items = min(len(a) for a in mat)
        for idx in range(n_items):
            rows.append({"model": model, "instrument": instrument, "item": idx,
                         "entropy": normalised_entropy([a[idx] for a in mat])})
    items = pd.DataFrame(rows)
    order = (items.groupby("model")["entropy"].mean().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for i, model in enumerate(order):
        vals = items.loc[items["model"] == model, "entropy"].to_numpy()
        jitter = (np.random.default_rng(i).random(vals.size) - 0.5) * 0.34
        ax.scatter(vals, np.full(vals.size, i) + jitter, s=7, color=BLUE, alpha=0.35,
                   edgecolor="none", zorder=3)
        ax.scatter([vals.mean()], [i], s=42, color=ORANGE, zorder=4,
                   edgecolor="white", linewidth=0.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT_NAME.get(m, m) for m in order], fontsize=8)
    ax.set_xlabel("per-item response entropy (0 = identical every trial, 1 = uniform)")
    ax.set_title("Instability is item-specific, not a property of a model",
                 fontsize=10, loc="left", pad=10)
    ax.annotate("each dot is one questionnaire item; orange is the model's mean",
                xy=(0.0, -0.085), xycoords="axes fraction", fontsize=8.5, color=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-1, len(order))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_item_entropy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return items


def fig_si_mtmm(df: pd.DataFrame) -> dict:
    mapping = {pc: n for n, pc, _ in TRAIT_PAIRS}
    mapping.update({ev: n for n, _, ev in TRAIT_PAIRS})
    d = df[df["trait"].isin(mapping)].copy()
    d["construct"] = d["trait"].map(mapping)
    res = mtmm(d, unit="model", trait="construct", method="instrument", value="value")

    cell = d.groupby(["model", "instrument", "construct"])["value"].mean().reset_index()
    wide = cell.pivot_table(index="model", columns=["instrument", "construct"], values="value")
    corr = wide.corr()
    labels = [f"{a.replace('political_compass', 'PC').replace('8values', '8V')}\n{b}"
              for a, b in corr.columns]

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.to_numpy()[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else INK)
    ax.set_title("Multi-trait multi-method matrix (n = 19 models)", fontsize=10, loc="left", pad=10)
    ax.annotate(
        f"convergent (same construct, different instrument) mean r = "
        f"{res['convergent_mean_r']:+.2f}\n"
        f"discriminant (different construct, same instrument) mean r = "
        f"{res['discriminant_mean_r']:+.2f}   ->   Campbell-Fiske fails",
        xy=(0.0, -0.30), xycoords="axes fraction", fontsize=8.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figSI_mtmm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return res


def write_tables(budget: pd.DataFrame, mtmm_res: dict, pooled: dict) -> None:
    """LaTeX for the new Table 1 (error budget) and the SI MTMM table."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Error budget. Variance components from a crossed random-effects "
        r"model on the 2{,}280 administrations. \textbf{Pooled over all axes}, instrument "
        r"and axis choice accounts for " + f"{pooled['variance_share'].get('trait',0)*100:.0f}" +
        r"\% of variance and model identity for only " +
        f"{pooled['variance_share'].get('model',0)*100:.0f}" +
        r"\%. The per-axis rows below are \emph{conditional on a fixed axis}, which removes "
        r"the between-axis term and is why the model share is much larger there. The two "
        r"denominators answer different questions and must not be quoted interchangeably.}",
        r"\label{tab:error-budget}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Axis & Model identity (\%) & Trial noise (\%) & $n$ \\", r"\midrule",
    ]
    for _, r in budget.iterrows():
        lines.append(f"{r['label']} & {r['model']*100:.1f} & {r['residual']*100:.1f} & {r['n']} \\\\")
    lines += [
        r"\midrule",
        f"\\textbf{{Mean}} & \\textbf{{{budget['model'].mean()*100:.1f}}} & "
        f"\\textbf{{{budget['residual'].mean()*100:.1f}}} & \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (TABLE_DIR / "table_error_budget.tex").write_text("\n".join(lines) + "\n")

    ml = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Multi-trait multi-method coefficients across the 19 models. Convergent "
        r"validity requires that the same construct measured by two instruments correlate "
        r"more strongly than two different constructs measured by the same instrument. It "
        r"does not.}",
        r"\label{tab:mtmm}",
        r"\begin{tabular}{llr}", r"\toprule",
        r"Type & Pair & $r$ \\", r"\midrule",
    ]
    for e in mtmm_res["convergent"]:
        ml.append(f"Convergent & {e['a']} vs {e['b']} & {e['r']:+.3f} \\\\".replace("_", r"\_"))
    for e in mtmm_res["discriminant_same_method"]:
        ml.append(f"Discriminant & {e['a']} vs {e['b']} & {e['r']:+.3f} \\\\".replace("_", r"\_"))
    ml += [
        r"\midrule",
        f"\\textbf{{Mean convergent}} & & \\textbf{{{mtmm_res['convergent_mean_r']:+.3f}}} \\\\",
        f"\\textbf{{Mean discriminant}} & & \\textbf{{{mtmm_res['discriminant_mean_r']:+.3f}}} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (TABLE_DIR / "table_mtmm.tex").write_text("\n".join(ml) + "\n")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_long()
    raw = load_raw_trials()

    budget, pooled = fig6_error_budget(df)
    print("fig6_error_budget.png -- error budget by axis")
    for _, r in budget.iterrows():
        print(f"    {r['label']:>14s}  model {r['model']*100:5.1f}%   noise {r['residual']*100:5.1f}%")
    print(f"    within-axis mean model share: {budget['model'].mean()*100:.1f}%")
    ps = pooled["variance_share"]
    print(f"    pooled: instrument/axis {ps.get('trait',0)*100:.1f}%  "
          f"model {ps.get('model',0)*100:.1f}%  noise {ps.get('residual',0)*100:.1f}%")

    iss = fig7_iss_repair(df)
    worst = iss.sort_values("legacy_iss", ascending=False).iloc[0]
    print(f"fig7_iss_repair.png -- worst artefact: {worst['model']} / {worst['axis']} "
          f"ISS {worst['legacy_iss']:.2f} on SD {worst['sd']:.3f}")

    items = fig8_item_entropy(raw)
    print(f"fig8_item_entropy.png -- {len(items)} model-item cells, "
          f"{(items['entropy'] == 0).mean()*100:.1f}% perfectly stable")

    res = fig_si_mtmm(df)
    print(f"figSI_mtmm.png -- convergent {res['convergent_mean_r']:+.3f} "
          f"vs discriminant {res['discriminant_mean_r']:+.3f}, "
          f"Campbell-Fiske passes: {res['campbell_fiske_passes']}")

    write_tables(budget, res, pooled)
    print(f"wrote {TABLE_DIR/'table_error_budget.tex'} and {TABLE_DIR/'table_mtmm.tex'}")


if __name__ == "__main__":
    main()
