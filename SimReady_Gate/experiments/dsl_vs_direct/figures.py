"""The comparison's summary figure from analyze.py's JSON: pass rate, output tokens per passing
scene and time per passing scene, per model and condition (three small multiples, each with its
own scale). Colours are the first three slots of the validated categorical palette, in fixed order:
dsl_v1 blue, direct orange, dsl_v0 aqua; every bar carries its value.

    python experiments/dsl_vs_direct/figures.py summary.json docs/media/dsl_vs_direct.png
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [("dsl_v1", "program (v1)", "#2a78d6"), ("direct", "poses (direct)", "#eb6834"), ("dsl_v0", "program (v0)", "#1baf7a")]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "claude-opus-5-5": "Opus 5.5"}


def main(src, dst):
    rep = json.load(open(src))
    models = [m for m in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5") if m in rep["overall"]]
    panels = [("pass rate", lambda v: 100 * v["success_rate"], "{:.0f}%"),
              ("output tokens per passing scene", lambda v: v["per_success"]["output"], "{:,.0f}"),
              ("seconds per passing scene (model + solver)", lambda v: v["time_per_success_s"], "{:.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    width = 0.26
    for ax, (title, get, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for k, (cond, label, color) in enumerate(SERIES):
            xs, ys = [], []
            for i, m in enumerate(models):
                v = rep["overall"][m].get(cond)
                val = None if v is None else get(v)
                if val is None:
                    continue
                xs.append(i + (k - 1) * width); ys.append(val)
            ax.bar(xs, ys, width=width - 0.02, color=color, edgecolor=SURFACE, linewidth=2, label=label, zorder=3)
            for x, y in zip(xs, ys):
                ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=6.5, color=INK2, zorder=4)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([SHORT.get(m, m) for m in models], fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.tick_params(axis="y", labelsize=7, colors=INK2, length=0); ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    axes[0].set_ylim(0, 110)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(dst, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
