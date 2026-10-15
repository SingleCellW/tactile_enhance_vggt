#!/usr/bin/env python3
"""Create a new cache with commands aligned to demonstrated physics transitions.

The historical converter wrote command[s+1] as label[s]. An independently
audited transition from qpos[s] to qpos[s+1] instead uses command[s]. This tool
requires the entire old-label relation to match before making a separate cache.
It preserves visual/tactile observations, their alignment, and the episode split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def commands_for_episode(root, name, frames):
    """Prefer original controls; otherwise use the generator's unchanged export."""
    episode = Path(root) / name
    raw_paths = sorted((episode / "action").glob("*.npy"))
    exported = episode / "tacsim_dataset_processed/dataset_actions.npz"
    if raw_paths and len(raw_paths) != frames:
        raise ValueError(f"Partial raw action directory for {name}: {len(raw_paths)} != {frames}")
    commands = None
    proof = {"raw_episode": name}
    if exported.exists():
        with np.load(exported, allow_pickle=False) as data:
            commands = data["data"].copy()
        proof.update(command_source_kind="generator_unchanged_action_list_export", command_export=fingerprint(exported))
    if raw_paths:
        raw = np.stack([np.load(path, allow_pickle=False) for path in raw_paths])
        if commands is not None:
            np.testing.assert_array_equal(raw, commands, err_msg=f"Raw controls and generator export differ in {name}")
        commands = raw
        proof.update(command_source_kind="original_action_npy", original_actions=[fingerprint(p) for p in raw_paths])
    if commands is None:
        raise FileNotFoundError(f"No recorded command series for {name}")
    if commands.shape != (frames, 22) or not np.isfinite(commands).all():
        raise ValueError(f"Expected finite complete [{frames},22] command series for {name}, got {commands.shape}")
    return commands, proof


def repair(source, raw_root, episode_map, destination, timing_audit):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new destination: {destination}")
    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Source cache is incomplete")
    if manifest["configuration"].get("action_label_repair"):
        raise ValueError("Action labels were already repaired; do not apply the correction twice")
    audit = json.loads(Path(timing_audit).read_text())
    if not (audit.get("complete") is True and audit.get("verified") is True
            and audit.get("action_target_offset_frames") == 0
            and audit.get("substeps_per_action") == 1
            and audit.get("velocity_dt_s") == audit.get("compiled_timestep_s") == .01):
        raise ValueError("Require verified one-step 0.01s physics/action timing audit")
    old = np.load(source / "action.npy", mmap_mode="r")
    episode = np.load(source / "episode_index.npy", mmap_mode="r")
    frame = np.load(source / "frame_index.npy", mmap_mode="r")
    if old.shape != (len(episode), 22) or frame.shape != episode.shape:
        raise ValueError("Source action/episode/frame shapes do not match")
    mapping = json.loads(Path(episode_map).read_text())["rows"]
    by_id = {}
    for row in mapping:
        key = int(row["episode"])
        if key in by_id:
            raise ValueError(f"Duplicate episode mapping: {key}")
        by_id[key] = row
    if set(by_id) != set(np.unique(episode).tolist()):
        raise ValueError("Episode mapping must exactly cover the source episodes")
    new = np.empty_like(old)
    evidence = []
    for key in sorted(by_id):
        positions = np.flatnonzero(episode == key)
        row = by_id[key]
        count = len(positions)
        if int(row["frames"]) != count or (count > 1 and not np.all(np.diff(positions) == 1)):
            raise ValueError(f"Noncontiguous or incomplete source episode {key}")
        np.testing.assert_array_equal(frame[positions], np.arange(count), err_msg=f"Invalid local frame order for episode {key}")
        commands, proof = commands_for_episode(raw_root, row["raw_episode"], count)
        # Match the historical float32 converter exactly, including final-row
        # repeat. No inferred shift or approximate matching is accepted.
        cast = commands.astype(old.dtype)
        np.testing.assert_array_equal(old[positions], cast[np.minimum(np.arange(count) + 1, count - 1)],
                                      err_msg=f"Source is not the historical next-frame label series in episode {key}")
        new[positions] = cast
        evidence.append(dict(proof, episode=key, frames=count, old_next_frame_relation_exact=True))
    train_ids = manifest["split"]["train"]
    if set(train_ids) & set(manifest["split"]["val"]):
        raise ValueError("Train and validation episodes overlap")
    train = new[np.isin(episode, train_ids)].astype(np.float64)
    if not len(train):
        raise ValueError("No training action frames")
    normalization = dict(manifest["normalization"])
    normalization.update(action_mean=train.mean(0).tolist(), action_std=np.maximum(train.std(0), 1e-3).tolist())
    staging = destination.with_name(destination.name + f".partial-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    for entry in source.iterdir():
        if entry.name in ("manifest.json", "action.npy", "normalization.json"):
            continue
        if entry.is_dir():
            shutil.copytree(entry, staging / entry.name)
        else:
            shutil.copyfile(entry, staging / entry.name)
    np.save(staging / "action.npy", new)
    if (source / "normalization.json").exists():
        (staging / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    preserved = {}
    for name in ("points.npy", "state.npy", "tactile.npy", "episode_index.npy", "frame_index.npy",
                 "obs_frame_indices.npy", "rgb_source_frame_indices.npy"):
        if (source / name).exists():
            source_hash = fingerprint(source / name)["sha256"]
            if fingerprint(staging / name)["sha256"] != source_hash:
                raise AssertionError(f"Action-label repair changed {name}")
            preserved[name] = source_hash
    report = {"complete": True, "frames": len(new), "episodes": len(evidence),
              "action_target_offset_frames": 0, "old_action_target_offset_frames": 1,
              "new_action_sha256": fingerprint(staging / "action.npy")["sha256"],
              "source_action": fingerprint(source / "action.npy"),
              "source_manifest": fingerprint(source / "manifest.json"),
              "episode_map": fingerprint(episode_map), "timing_audit": fingerprint(timing_audit),
              "implementation": fingerprint(__file__), "unchanged_arrays_sha256": preserved,
              "command_sources": evidence,
              "scope": "Only actions and train-only action normalization changed. Native tactile velocity convention is inherited from the input cache and must be audited separately."}
    (staging / "action_label_repair_report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(timing_audit, staging / "physics_transition_audit.json")
    manifest["normalization"] = normalization
    manifest["configuration"]["action_label_repair"] = {
        "source_manifest": report["source_manifest"], "timing_audit": report["timing_audit"],
        "implementation": report["implementation"], "action_target_offset_frames": 0,
        "new_action_sha256": report["new_action_sha256"],
        "report_sha256": fingerprint(staging / "action_label_repair_report.json")["sha256"]}
    manifest["configuration_fingerprint"] = hashlib.sha256(json.dumps(manifest["configuration"], sort_keys=True).encode()).hexdigest()
    manifest["action_semantics"] = (
        "At synchronized state row s, target is recorded command[s], the command that advances "
        "qpos[s] toward qpos[s+1] in one 0.01s physics step. Original 20Hz video/wallclock metadata "
        "does not specify simulated time. Action chunks advance one source row per command; "
        "padding remains confined to each episode. Historical extra next-frame shift removed.")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    staging.rename(destination)
    print(json.dumps({"complete": True, "destination": str(destination), "frames": len(new),
                      "episodes": len(evidence), "action_sha256": report["new_action_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--episode-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-audit", type=Path, required=True)
    args = parser.parse_args()
    repair(args.source, args.raw_root, args.episode_map, args.output, args.timing_audit)
