"""Historical command indexing and observation-free replay protocol checks."""

from argparse import Namespace
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_token_replay as replay


def make_sources(tmp_path):
    cache, raw, timestamps = tmp_path / "cache", tmp_path / "raw", tmp_path / "timestamps"
    cache.mkdir()
    raw.mkdir()
    timestamps.mkdir()
    arrays = {"action": np.repeat(np.arange(5, dtype=np.float32)[:, None], 22, axis=1),
              "state": np.arange(5 * 22, dtype=np.float32).reshape(5, 22),
              "episode_index": np.array([0, 0, 1, 1, 1]), "frame_index": np.array([0, 1, 0, 1, 2])}
    metadata = {"complete": True, "native_tactile_report": {"velocity_dt": .01},
                "action_semantics": "Full-rate source labels unchanged. For corrected state row s and RGB row s+lag, the target is source action[s], historically raw command[s+1]; episode-local padding."}
    parquet = tmp_path / "original.parquet"
    parquet.write_bytes(b"original action parquet IO is replaced only in CPU tests")
    metadata["configuration"] = {"source": {"source_parquet": [
        {"path": str(parquet), "sha256": replay.digest(parquet)}]}}
    (cache / "manifest.json").write_text(json.dumps(metadata))
    for name, value in arrays.items():
        np.save(cache / f"{name}.npy", value)
    first_qpos = None
    for ep, length in ((0, 2), (1, 3)):
        directory = raw / f"episode_{ep}"
        (directory / "prioception").mkdir(parents=True)
        records = []
        for frame in range(length):
            row = int(np.flatnonzero((arrays["episode_index"] == ep) & (arrays["frame_index"] == frame))[0])
            qpos = np.r_[arrays["state"][row], [.3, .1, .04, 1., 0., 0., 0.]]
            path = directory / "prioception" / f"{frame}.npy"
            np.save(path, qpos)
            if ep == 1 and frame == 0:
                first_qpos = path
            records.append({"prioception_path": str(path.relative_to(directory))})
        with (timestamps / f"teleop_log_{directory.name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["prioception_path"])
            writer.writeheader()
            writer.writerows(records)
    source = tmp_path / "readouts.npz"
    np.savez(source, state22=arrays["state"], episode_index=arrays["episode_index"], frame_index=arrays["frame_index"],
             force44=np.zeros((5, 44), dtype=np.float32), contact44=np.zeros((5, 44), dtype=np.float32),
             anchors_world=np.zeros((5, 44, 3), dtype=np.float32))
    args = Namespace(cache=cache, output=tmp_path / "output", episode=1, seed=43, case_limit=2,
                     device="cpu", worker_python=Path("unused-python"), dynamic_root=tmp_path / "dynamic",
                     raw_root=raw, timestamps_root=timestamps, source_readouts=source,
                     trace_every=100, worker_timeout=120., resume=False)
    return args, arrays, first_qpos


def test_nominal_plus_unchanged36_grid_and_full_rate_terminal_hold(tmp_path):
    args, _, _ = make_sources(tmp_path)
    full, selected = replay.replay_cases()
    assert len(full) == len(selected) == 37
    assert full[0]["case_id"] == "nominal" and full[0]["object_xy_offset"] == [0., 0.]
    assert full[1:] == replay.make_cases()[0]
    _, _, rows, frames, commands = replay.load_replay_source(args.cache, 1)
    np.testing.assert_array_equal(rows, [2, 3, 4])
    np.testing.assert_array_equal(frames[:5], [0, 1, 2, 2, 2])
    np.testing.assert_array_equal(commands[:5, 0], [2, 3, 4, 4, 4])
    assert commands.shape == (600, 22)
    with pytest.raises(ValueError, match="case-limit"):
        replay.replay_cases(case_limit=38)


def test_rejects_unsorted_episode_frames_and_mismatched_nominal_state(tmp_path):
    args, arrays, qpos_path = make_sources(tmp_path)
    np.save(args.cache / "frame_index.npy", np.array([0, 1, 0, 2, 1]))
    with pytest.raises(ValueError, match="sorted, full-rate"):
        replay.load_replay_source(args.cache, 1)
    np.save(args.cache / "frame_index.npy", arrays["frame_index"])
    metadata, actual, rows, _, _ = replay.load_replay_source(args.cache, 1)
    qpos = np.load(qpos_path)
    qpos[0] += 1
    np.save(qpos_path, qpos)
    with pytest.raises(AssertionError, match="Nominal raw frame0"):
        replay.validate_nominal_source(args, metadata, actual, rows)


def test_action_source_uses_verified_parquet_and_preserves_historical_semantics(tmp_path, monkeypatch):
    args, arrays, _ = make_sources(tmp_path)
    metadata, actual, rows, _, _ = replay.load_replay_source(args.cache, 1)
    source = {name: arrays[name].copy() for name in ("action", "episode_index", "frame_index")}
    monkeypatch.setattr(replay, "read_parquet_actions", lambda paths: source)
    proof = replay.validate_nominal_source(args, metadata, actual, rows)
    assert set(proof["source_readout_schema"]) == {
        "force44", "contact44", "anchors_world", "state22", "episode_index", "frame_index"}
    assert proof["action_source"]["action_column"] == "action"
    assert "raw command[s+1]" in metadata["action_semantics"]
    source["action"][rows[0], 0] += 1
    with pytest.raises(AssertionError, match="parquet/cache episode mismatch: action"):
        replay.validate_nominal_source(args, metadata, actual, rows)
    entry = metadata["configuration"]["source"]["source_parquet"][0]
    Path(entry["path"]).write_bytes(b"changed parquet")
    with pytest.raises(ValueError, match="checksum mismatch"):
        replay.validate_nominal_source(args, metadata, actual, rows)


def test_fake_worker_replay_never_observes_and_runs600_ticks_per_case(tmp_path, monkeypatch):
    args, arrays, qpos_path = make_sources(tmp_path)
    monkeypatch.setattr(replay, "read_parquet_actions", lambda paths: {
        name: arrays[name].copy() for name in ("action", "episode_index", "frame_index")})
    calls, closed = [], []

    class Client:
        def __init__(self, command, *unused):
            assert command[command.index("--device") + 1] == "cpu"

        def request(self, command, **values):
            assert values["observe"] is False
            assert command in ("reset", "step")
            calls.append((command, values))
            if command == "reset":
                self.tick = 0
                self.offset = np.r_[values["object_xy_offset"], 0.]
            else:
                self.tick += 1
                expected = [2, 3, 4][min(self.tick - 1, 2)]
                np.testing.assert_array_equal(values["action"], np.full(22, expected))
            center = np.array([.3, .1, .04]) + self.offset
            inside = self.tick >= 100
            if inside:
                center = np.array([.5, -.3, .04])
            info = {"cube_center": center.tolist(), "lifted": False, "inside_goal": inside,
                    "strict_stable_goal": self.tick >= 200, "control_tick": self.tick,
                    "initialization": {"episode_id": 1, "frame": 0, "qpos_sha256": replay.digest(qpos_path)}}
            return {"ok": True, "info": info, "sim_time": self.tick * .01,
                    "provenance": {"source_readouts_sha256": replay.digest(args.source_readouts)}}

        def close(self):
            closed.append(True)

    monkeypatch.setattr(replay, "SimulationClient", Client)
    replay.run(args)
    assert sum(name == "reset" for name, _ in calls) == 2
    assert sum(name == "step" for name, _ in calls) == 1200
    assert closed == [True]
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["complete"] and summary["nominal_strict_stable_goal"]
    assert summary["grid"]["cases"] == 1 and summary["grid"]["strict_stable_goal_count"] == 1
    config = json.loads((args.output / "eval_config.json").read_text())
    assert config["held_last_command_ticks"] == 597
    assert config["array_sha256"]["action"] == replay.digest(args.cache / "action.npy")
    assert config["array_sha256"]["episode_index"] == replay.digest(args.cache / "episode_index.npy")
    assert config["action_semantics"] == json.loads((args.cache / "manifest.json").read_text())["action_semantics"]
    assert "action_label_repair" not in config
    with np.load(args.output / "nominal" / "trajectory.npz") as trajectory:
        assert trajectory["actions"].shape == (600, 22)
        np.testing.assert_array_equal(trajectory["source_global_rows"][:4], [2, 3, 4, 4])
    args.resume = True
    replay.run(args)
    assert sum(name == "step" for name, _ in calls) == 1200
    assert json.loads((args.output / "summary.json").read_text())["complete"]
