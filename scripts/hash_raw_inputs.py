#!/usr/bin/env python3
"""Hash all recorded full qpos29 inputs and their timestamps without modifying data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--timestamps-root", type=Path, required=True)
    p.add_argument("--source-readouts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attach-report", type=Path, help="Attach the profile hash to a completed merged repair report")
    p.add_argument("--rgb-timing-audit", type=Path, help="Record independently verified RGB lag when attaching a report")
    p.add_argument("--physics-timing-audit", type=Path, help="Bind independent raw-state transition evidence for a fixed physical velocity dt")
    args = p.parse_args()
    with np.load(args.source_readouts, allow_pickle=False) as source:
        state = source["state22"].copy()
        episode = source["episode_index"].copy()
        frame = source["frame_index"].copy()
    raw_episodes = sorted(x for x in args.raw_root.iterdir() if (x / "prioception").exists())
    profiles, all_hash = [], hashlib.sha256()
    for ep in np.unique(episode):
        raw = raw_episodes[int(ep)]
        csv_path = args.timestamps_root / f"teleop_log_{raw.name}.csv"
        rows = list(csv.DictReader(csv_path.open()))
        ep_hash = hashlib.sha256()
        selected = np.flatnonzero(episode == ep)
        for index in selected:
            f = int(frame[index])
            relative = rows[f]["prioception_path"]
            path = raw / relative
            qpos = np.load(path, allow_pickle=False)
            if qpos.shape != (29,) or not np.isfinite(qpos).all():
                raise ValueError(f"Invalid full qpos: {path}")
            np.testing.assert_array_equal(qpos[:22].astype(np.float32), state[index])
            record = {"episode": int(ep), "raw_episode": raw.name, "frame": f,
                      "relative_path": relative, "file_sha256": sha(path),
                      "timestamp": rows[f]["timestamp"]}
            blob = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ep_hash.update(blob)
            all_hash.update(blob)
        profiles.append({"episode": int(ep), "raw_episode": raw.name, "frames": len(selected),
                         "timestamp_file_sha256": sha(csv_path), "qpos_records_sha256": ep_hash.hexdigest()})
    profile = {"complete": True, "frames": len(state), "episodes": len(profiles),
               "raw_root": str(args.raw_root.resolve()), "timestamps_root": str(args.timestamps_root.resolve()),
               "source_readouts_sha256": sha(args.source_readouts),
               "qpos29_records_sha256": all_hash.hexdigest(),
               "hash_definition": "SHA256 of ordered UTF-8 compact sorted-key JSON records plus LF; each record binds episode, raw_episode, frame, relative_path, complete NPY file SHA256 and original timestamp string",
               "scope": "All 29 positions including 7 privileged object pose values; object pose is used for sensor synthesis only and excluded from policy state22",
               "episode_profiles": profiles}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    if args.attach_report:
        report = json.loads(args.attach_report.read_text())
        if not report["complete"] or report["frames"] != len(state) or report["source_readouts_sha256"] != profile["source_readouts_sha256"]:
            raise ValueError("Repair report does not match the complete input profile")
        report["raw_input_profile"] = {"path": str(args.output.resolve()), "sha256": sha(args.output),
                                       "qpos29_records_sha256": profile["qpos29_records_sha256"]}
        if args.rgb_timing_audit:
            audit = json.loads(args.rgb_timing_audit.read_text())
            if audit.get("verified") is not True or type(audit.get("rgb_lag_frames")) is not int:
                raise ValueError("RGB timing audit is not verified")
            lag = audit["rgb_lag_frames"]
            report["rgb_timing_audit"] = {"path": str(args.rgb_timing_audit.resolve()), "sha256": sha(args.rgb_timing_audit),
                                           "verified": True, "rgb_lag_frames": lag}
            report["rgb_status"] = f"Independent render audit confirms RGB[t] corresponds to qpos[t-{lag}]; this raw readout stays indexed by qpos. Cache assembly must apply this within-episode mapping and discard unavailable leading observations."
        if args.physics_timing_audit:
            audit = json.loads(args.physics_timing_audit.read_text())
            if (audit.get("complete") is not True or report.get("velocity_dt") is None
                    or not np.isclose(report["velocity_dt"], audit["compiled_timestep_s"], rtol=0, atol=1e-12)):
                raise ValueError("Physical timing audit does not match the fixed native velocity interval")
            report["physics_timing_audit"] = {"path": str(args.physics_timing_audit.resolve()),
                                               "sha256": sha(args.physics_timing_audit), "complete": True,
                                               "compiled_timestep_s": audit["compiled_timestep_s"],
                                               "scope": "Independent sampled raw-state transition evidence; not hidden-state-perfect replay"}
        args.attach_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in profile.items() if k != "episode_profiles"}, indent=2))


if __name__ == "__main__":
    main()
