"""Generate the paper's results figures directly from data/scores.csv.

Three figures, each visualizing an actual finding from the Results section
rather than example single-trial instrument output (figures/fig1 and fig2
already cover that):

  fig3_compass_scatter.png  -- all 19 models' mean (economic, social) position
                                on the Political Compass plane at once, the
                                direct visual counterpart to Table 1 and
                                Section "Directional Consistency".
  fig4_8values_dotplot.png  -- small-multiples dot plot, one panel per 8Values
                                axis, all 19 models against the 50-neutral
                                reference line.
  fig5_iss_distribution.png -- strip plot of all 114 model-axis Ideological
                                Stability Score values, grouped by axis, the
                                visual counterpart to Table 3 and the
                                consistency section.

Run after consolidate.py has built data/scores.csv:
    python3 make_result_figures.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"

# dataviz-skill categorical slots (validated colorblind-safe, all-pairs, <=3 series)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#8a8a86"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SHORT_NAME = {
    "amazon/nova-pro-v1": "nova-pro",
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
    "anthropic/claude-opus-4.5": "claude-opus-4.5",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "cohere/command-r-plus-08-2024": "command-r-plus",
    "deepseek/deepseek-chat": "deepseek-chat",
    "deepseek/deepseek-v3.2": "deepseek-v3.2",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b",
    "meta-llama/llama-4-maverick": "llama-4-maverick",
    "mistralai/mistral-large-2512": "mistral-large",
    "mistralai/mistral-small-3.2-24b-instruct": "mistral-small",
    "nvidia/nemotron-3-ultra-550b-a55b": "nemotron-3-ultra",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-5-mini": "gpt-5-mini",
    "qwen/qwen3-235b-a22b": "qwen3-235b",
    "qwen/qwen3-30b-a3b": "qwen3-30b",
    "x-ai/grok-4.20": "grok-4.20",
}


def load_scores():
    by_axis = defaultdict(lambda: defaultdict(list))
    with open(DATA_DIR / "scores.csv") as f:
        for row in csv.DictReader(f):
            test = row["test"]
            for i in (1, 2, 3, 4):
                name = row[f"axis{i}_name"]
                val = row[f"axis{i}_value"]
                if name and val:
                    by_axis[(test, name)][row["model"]].append(float(val))
    return by_axis


def relative_sd(vals):
    vals = np.asarray(vals, dtype=float)
    rng = vals.max() - vals.min()
    return 0.0 if rng == 0 else 100 * vals.std(ddof=1) / rng


def relative_iqr(vals):
    vals = np.asarray(vals, dtype=float)
    rng = vals.max() - vals.min()
    if rng == 0:
        return 0.0
    q75, q25 = np.percentile(vals, [75, 25])
    return 100 * (q75 - q25) / rng


def iss(vals):
    return 0.7 * relative_sd(vals) + 0.3 * relative_iqr(vals)


def fig3_compass_scatter(by_axis):
    econ = by_axis[("political_compass", "economic")]
    soc = by_axis[("political_compass", "social")]
    models = sorted(econ.keys())

    xs = [np.mean(econ[m]) for m in models]
    ys = [np.mean(soc[m]) for m in models]
    x_lo, x_hi = min(xs + [0]) - 1.2, max(xs + [0]) + 1.2
    y_lo, y_hi = min(ys + [0]) - 1.2, max(ys + [0]) + 1.2

    fig, (ax, ax_full) = plt.subplots(
        1, 2, figsize=(9.6, 6.2), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax.axhline(0, color=GRID, linewidth=1, zorder=1)
    ax.axvline(0, color=GRID, linewidth=1, zorder=1)
    ax.scatter(xs, ys, s=40, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)

    texts = [
        ax.text(x, y, SHORT_NAME.get(m, m), fontsize=7.4, color=INK, zorder=4)
        for m, x, y in zip(models, xs, ys)
    ]
    adjust_text(
        texts, x=xs, y=ys, ax=ax,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5),
        expand=(1.6, 2.0), force_text=(0.5, 0.8), force_static=(0.3, 0.4),
    )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Economic axis (negative = left)")
    ax.set_ylabel("Social axis (negative = libertarian)")
    ax.set_title("(a) All 19 models, zoomed to the occupied region",
                 fontsize=9.5, loc="left")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED)

    # small full -10..10 panel for scale context, with a rectangle marking
    # the zoomed region shown at left
    ax_full.axhline(0, color=GRID, linewidth=1, zorder=1)
    ax_full.axvline(0, color=GRID, linewidth=1, zorder=1)
    ax_full.scatter(xs, ys, s=14, color=BLUE, zorder=3)
    ax_full.scatter([0], [0], marker="+", s=70, color=ORANGE, linewidth=1.4, zorder=4)
    ax_full.add_patch(plt.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                                     fill=False, edgecolor=MUTED, linewidth=0.8,
                                     linestyle="--", zorder=2))
    ax_full.set_xlim(-10, 10)
    ax_full.set_ylim(-10, 10)
    ax_full.set_xticks([-10, 0, 10])
    ax_full.set_yticks([-10, 0, 10])
    ax_full.set_title("(b) Full instrument\nscale, for context", fontsize=9.5, loc="left")
    ax_full.set_xlabel("Economic")
    ax_full.set_ylabel("Social")
    for spine in ("left", "bottom"):
        ax_full.spines[spine].set_color(GRID)
    ax_full.tick_params(colors=MUTED)

    fig.suptitle("Political Compass position of all 19 models (mean of 60 trials each); "
                 "+ marks the neutral human-baseline proxy",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig3_compass_scatter.png", dpi=300)
    plt.close(fig)


def fig4_8values_dotplot(by_axis):
    axes4 = ["equality", "peace", "liberty", "progress"]
    fig, axs = plt.subplots(1, 4, figsize=(11, 6.2), sharey=True)
    models = sorted(by_axis[("8values", "equality")].keys())
    labels = [SHORT_NAME.get(m, m) for m in models]
    y = np.arange(len(models))

    for ax, axis_name in zip(axs, axes4):
        means = [np.mean(by_axis[("8values", axis_name)][m]) for m in models]
        order = np.argsort(means)
        ax.axvline(50, color=GRID, linewidth=1, linestyle="--", zorder=1)
        ax.scatter(np.array(means)[order], y, s=22, color=BLUE, edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(np.array(labels)[order], fontsize=6.6)
        ax.set_xlim(45, 90)
        ax.set_title(axis_name, fontsize=9)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED)

    axs[0].set_ylabel("")
    fig.supxlabel("8Values score (0–100; 50 = neutral, dashed reference line)", fontsize=8.5)
    fig.suptitle("8Values scores of all 19 models, four axes (mean of 60 trials each)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig4_8values_dotplot.png", dpi=300)
    plt.close(fig)


def fig5_iss_distribution(by_axis):
    axes6 = [
        ("8values", "equality"), ("8values", "peace"), ("8values", "liberty"),
        ("8values", "progress"), ("political_compass", "economic"),
        ("political_compass", "social"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    rng = np.random.default_rng(0)
    for i, key in enumerate(axes6):
        test, axis_name = key
        vals = [iss(v) for v in by_axis[key].values()]
        color = BLUE if test == "8values" else ORANGE
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=color,
                   alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter([i], [np.median(vals)], marker="_", s=340, color=INK,
                   linewidth=1.6, zorder=4)

    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{t.replace('political_compass', 'Pol. Compass').replace('8values', '8Values')}\n{a}"
                        for t, a in axes6], fontsize=7.6)
    ax.set_ylabel("Ideological Stability Score (lower = more consistent)")
    ax.set_title("ISS across all 114 model–axis combinations, by axis "
                 "(black bar = median)", fontsize=10, loc="left")
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=6, label="8Values"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=6, label="Political Compass"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_iss_distribution.png", dpi=300)
    plt.close(fig)


def main():
    by_axis = load_scores()
    fig3_compass_scatter(by_axis)
    fig4_8values_dotplot(by_axis)
    fig5_iss_distribution(by_axis)
    print(f"wrote fig3_compass_scatter.png, fig4_8values_dotplot.png, "
          f"fig5_iss_distribution.png to {FIG_DIR}")


if __name__ == "__main__":
    main()
