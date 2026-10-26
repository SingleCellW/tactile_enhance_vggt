#!/usr/bin/env python3
"""Plot completed, predeclared policy tests with their actual denominators."""
import argparse
import json
from pathlib import Path

from tactile_vggt.paths import artifact_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("frozen_zero", "frozen_pressure", "joint_zero", "joint_pressure")
LABELS = ("Frozen\nzero touch", "Frozen\npressure", "Joint\nzero touch", "Joint\npressure")


def draw(args):
    result = json.loads((args.report / "results.json").read_text())
    if not result.get("complete"):
        raise ValueError("Only plot the complete, validated experiment")
    entries = {(row["mode"], row["n_action_steps"]): row for row in result["evaluations"] if row["kind"] == "simple"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    def bars(axis, xs, stats, width, color, label=None):
        rates = np.array([s["strict_success_rate"] * 100 for s in stats])
        cis = np.array([s["strict_success_wilson_ci95"] for s in stats]) * 100
        axis.bar(xs, rates, width=width, color=color, label=label, zorder=2)
        axis.errorbar(xs, rates, yerr=np.stack((rates-cis[:, 0], cis[:, 1]-rates)),
                      fmt="none", ecolor="#344054", capsize=3, lw=1, zorder=3)
        for x, rate, s in zip(xs, rates, stats):
            axis.annotate(f"{s['strict_successes']}/{s['cases']}", (x, rate),
                          xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)

    x = np.arange(4)
    bars(axes[0], x, [entries[(m, 8)]["subsets"]["grid36"] for m in MODES], .65,
         ["#98a2b3", "#1570ef", "#98a2b3", "#1570ef"])
    axes[0].set_title("Full XY grid: n_action_steps = 8 (36 cases)", fontsize=11)
    width = .34
    bars(axes[1], x-width/2, [entries[(m, 8)]["subsets"]["within_bbox12"] for m in MODES], width, "#98a2b3", "8 steps")
    bars(axes[1], x+width/2, [entries[(m, 2)]["subsets"]["within_bbox12"] for m in MODES], width, "#12b76a", "2 steps; same checkpoint")
    axes[1].set_title("Common XY-bbox subset (12 cases)", fontsize=11)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    for axis in axes:
        axis.set_xticks(x, LABELS, fontsize=9)
        axis.set_ylim(0, 112)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_ylabel("Strict stable success (%)")
        axis.grid(axis="y", alpha=.2, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Minimal policy follow-up: frozen VGGT, direct pressure44", fontsize=14)
    fig.supxlabel("Nominal reset reported separately. One source episode / training seed; XY-bbox inclusion is not full ID.\n"
                  "Whiskers: Wilson intervals over fixed resets. Simulation pauses during inference.", fontsize=8)
    fig.savefig(args.report / "policy_comparison.png", dpi=180)
    fig.savefig(args.report / "policy_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=artifact_path('reports', 'simple_policy_followup_20261025'))
    draw(parser.parse_args())
