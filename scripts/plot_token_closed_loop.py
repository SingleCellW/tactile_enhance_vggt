#!/usr/bin/env python3
"""Plot the completed 3 x 36 paired token-ACT reset grid on CPU only.

No final figure is written for incomplete or mismatched runs. Grid index i maps
to X=i//6 and Y=i%6, in the original physical reset-offset convention.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODES = ("visual", "late", "internal")
GRID_MM = (-35., -21., -7., 7., 21., 35.)


def case_coordinates(case_id):
    """Return the actual offset axes; never transpose a row-major heatmap."""
    match = re.fullmatch(r"grid_(\d{2})", case_id)
    if not match or not 0 <= int(match.group(1)) < 36:
        raise ValueError(f"Invalid case ID: {case_id!r}")
    number = int(match.group(1))
    return GRID_MM[number // 6], GRID_MM[number % 6]


def validate_summary(summary):
    """Return canonical plot rows only for the complete, paired declared grid."""
    if summary.get("complete") is not True:
        raise ValueError("Refusing final plot: closed-loop summary is incomplete")
    models = summary.get("models", {})
    if set(models) != set(MODES):
        raise ValueError("Require exactly visual, late and internal completed modes")
    expected = {f"grid_{number:02d}" for number in range(36)}
    signatures, result, source_episodes = {}, {}, set()
    for mode in MODES:
        model = models[mode]
        rows = model.get("results", [])
        if model.get("cases") != 36 or len(rows) != 36:
            raise ValueError(f"Refusing final plot: {mode} has fewer/different than 36 cases")
        by_id = {}
        for row in rows:
            if row.get("complete") is not True or row.get("ticks") != 600:
                raise ValueError(f"{mode}: incomplete case or wrong physical horizon")
            case = row.get("case", {})
            case_id = case.get("case_id", "")
            if case_id in by_id:
                raise ValueError(f"{mode}: duplicate {case_id}")
            x_mm, y_mm = case_coordinates(case_id)
            offsets = case.get("object_xy_offset")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"{mode}/{case_id}: missing physical XY offsets")
            if any(not math.isfinite(value) for value in offsets):
                raise ValueError(f"{mode}/{case_id}: nonfinite offset")
            if not math.isclose(offsets[0] * 1000, x_mm, abs_tol=1e-5) or not math.isclose(offsets[1] * 1000, y_mm, abs_tol=1e-5):
                raise ValueError(f"{mode}/{case_id}: offset disagrees with X=i//6,Y=i%6 convention")
            if case.get("episode_id") is None or case.get("frame") != 0 or case.get("seed") is None:
                raise ValueError(f"{mode}/{case_id}: missing source, frame or seed provenance")
            signature = (case["episode_id"], case["frame"], case["seed"], tuple(offsets))
            if case_id in signatures and signatures[case_id] != signature:
                raise ValueError(f"{case_id}: physical initial conditions differ between modes")
            signatures[case_id] = signature
            source_episodes.add(case["episode_id"])
            outcome = row.get("outcome", {})
            success = outcome.get("strict_stable_goal")
            dropped = outcome.get("lift_then_drop_outside_goal")
            distance = outcome.get("final_goal_center_xy_distance_m")
            if not isinstance(success, bool) or not isinstance(dropped, bool):
                raise ValueError(f"{mode}/{case_id}: explicit success/drop Boolean required")
            if not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0:
                raise ValueError(f"{mode}/{case_id}: invalid final goal-center distance")
            by_id[case_id] = {"case_id": case_id, "x_mm": x_mm, "y_mm": y_mm,
                              "success": success, "dropped": dropped, "goal_center_distance_cm": distance * 100}
        if set(by_id) != expected:
            raise ValueError(f"{mode}: missing or unexpected grid case IDs")
        ordered = [by_id[f"grid_{number:02d}"] for number in range(36)]
        successes = sum(row["success"] for row in ordered)
        distance_cm = sum(row["goal_center_distance_cm"] for row in ordered) / 36
        if model.get("strict_stable_goal_count") != successes:
            raise ValueError(f"{mode}: success aggregate disagrees with case outcomes")
        recorded_distance = model.get("mean_final_goal_center_xy_distance_m")
        if not isinstance(recorded_distance, (int, float)) or not math.isclose(recorded_distance * 100, distance_cm, abs_tol=1e-6):
            raise ValueError(f"{mode}: distance aggregate disagrees with case outcomes")
        result[mode] = {"rows": ordered, "successes": successes, "mean_goal_center_distance_cm": distance_cm,
                        "drop_count": sum(row["dropped"] for row in ordered)}
    if len(source_episodes) != 1:
        raise ValueError("Expected one shared source episode for this declared grid")
    return result, next(iter(source_episodes))


def plot_summary(input_path, output, act_seed=43):
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value {value}")
    source_bytes = input_path.read_bytes()
    summary = json.loads(source_bytes, parse_constant=nonfinite)
    models, episode = validate_summary(summary)
    # Keep imports below validation: integrity checks do not need matplotlib or a GPU runtime.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    success_color, failure_color, drop_color = "#278a59", "#d0d5dd", "#c33f3f"
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.30, wspace=.18)
    for ax, mode in zip(axes, MODES):
        model = models[mode]
        rows = model["rows"]
        ax.scatter([row["x_mm"] for row in rows], [row["y_mm"] for row in rows], s=210,
                   c=[success_color if row["success"] else failure_color for row in rows],
                   edgecolors=[drop_color if row["dropped"] else "#ffffff" for row in rows],
                   linewidths=[2.1 if row["dropped"] else 1.0 for row in rows], zorder=3)
        ax.set_title(f"{mode}: {model['successes']}/36 successes\nMean final goal-center distance: {model['mean_goal_center_distance_cm']:.1f} cm", fontsize=10.5, pad=12)
        ax.set(xlim=(-44, 44), ylim=(-44, 44), xticks=GRID_MM, yticks=GRID_MM,
               xlabel="Object X offset (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#e6e8ec", linewidth=.8, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Object Y offset (mm)")
    legend = [Line2D([], [], marker="o", linestyle="none", markersize=10, markerfacecolor=success_color, markeredgecolor="white", label="Strict stable success"),
              Line2D([], [], marker="o", linestyle="none", markersize=10, markerfacecolor=failure_color, markeredgecolor="white", label="Failure"),
              Line2D([], [], marker="o", linestyle="none", markersize=10, markerfacecolor="none", markeredgewidth=2, markeredgecolor=drop_color, label="Drop after lift (outline)")]
    fig.legend(handles=legend, loc="center", bbox_to_anchor=(.52, .17), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Paired closed-loop ACT evaluation: identical physical XY reset grid", fontsize=13, y=.97)
    fig.text(.5, .045, f"One held-out source episode (id={episode}); one ACT training seed ({act_seed}).\nSame 36 initial conditions per method; simulation pauses during inference.",
             ha="center", va="center", fontsize=9, linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.stem + ".partial.png")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    try:
        fig.savefig(staging, dpi=170, format="png", metadata={"Source SHA256": source_hash,
            "Scope": "One source episode and ACT seed; paused inference; no cross-episode or realtime claim"})
        staging.replace(output)
    finally:
        plt.close(fig)
        if staging.exists():
            staging.unlink()
    return {"output": str(output.resolve()), "source_sha256": source_hash, "source_episode": episode,
            "act_seed_label": act_seed, "models": {mode: {key: value for key, value in model.items() if key != "rows"} for mode, model in models.items()}}


def self_test():
    """Small CPU-only fixture checks; these do not import plotting libraries."""
    assert case_coordinates("grid_00") == (-35., -35.)
    assert case_coordinates("grid_05") == (-35., 35.)
    assert case_coordinates("grid_06") == (-21., -35.)
    assert case_coordinates("grid_30") == (35., -35.)
    assert case_coordinates("grid_35") == (35., 35.)
    rows = []
    for number in range(36):
        x_mm, y_mm = case_coordinates(f"grid_{number:02d}")
        rows.append({"complete": True, "ticks": 600,
                     "case": {"case_id": f"grid_{number:02d}", "episode_id": 1, "frame": 0,
                              "seed": 43 + number, "object_xy_offset": [x_mm / 1000, y_mm / 1000]},
                     "outcome": {"strict_stable_goal": number % 2 == 0, "lift_then_drop_outside_goal": number % 5 == 0,
                                 "final_goal_center_xy_distance_m": .1}})
    summary = {"complete": True, "models": {mode: {"cases": 36, "strict_stable_goal_count": 18,
                "mean_final_goal_center_xy_distance_m": .1, "results": copy.deepcopy(rows)} for mode in MODES}}
    result, episode = validate_summary(summary)
    assert episode == 1 and result["visual"]["successes"] == 18
    assert result["visual"]["rows"][5]["x_mm"] == -35
    assert result["visual"]["rows"][5]["y_mm"] == 35
    mutations = []
    partial = copy.deepcopy(summary); partial["complete"] = False; mutations.append(partial)
    missing = copy.deepcopy(summary); missing["models"]["late"]["results"].pop(); mutations.append(missing)
    swapped = copy.deepcopy(summary); swapped["models"]["late"]["results"][5]["case"]["object_xy_offset"].reverse(); mutations.append(swapped)
    mismatch = copy.deepcopy(summary); mismatch["models"]["late"]["results"][0]["case"]["seed"] += 1; mutations.append(mismatch)
    duplicate = copy.deepcopy(summary); duplicate["models"]["internal"]["results"][-1] = copy.deepcopy(duplicate["models"]["internal"]["results"][0]); mutations.append(duplicate)
    for fixture in mutations:
        try:
            validate_summary(fixture)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid fixture was accepted")
    return {"passed": True, "checks": ["XY orientation/corners", "complete 3x36 grid", "partial-summary rejection",
        "missing-case rejection", "transposed-coordinate rejection", "paired-seed mismatch rejection", "duplicate-case rejection"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/token_pilot_20261025/closed_loop36/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/token_pilot_20261025/policy_xy_grid.png")
    parser.add_argument("--act-seed", type=int, default=43, help="ACT training seed label for this declared campaign")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return
    if args.act_seed < 0:
        parser.error("ACT seed must be nonnegative")
    try:
        report = plot_summary(args.input, args.output, args.act_seed)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
