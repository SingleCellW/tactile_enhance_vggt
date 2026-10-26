"""Coordinate, sampling and episode-cluster metrics for the camera pilot.

Known camera poses are exogenous calibration. Object labels are supervision and
scoring only. Every variant is expressed using its own camera orientation but a
single nominal-training physical baseline; no fitted prediction alignment.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .token_metrics import paired_group_bootstrap

DOMAINS = {0: "nominal", 1: "near", 2: "strong"}


def camera_vectors_to_base(vectors, camera_to_base, baseline_m):
    vectors = np.asarray(vectors, dtype=np.float64)
    poses = np.asarray(camera_to_base, dtype=np.float64)
    if vectors.shape != (len(poses), 3) or poses.shape != (len(vectors), 4, 4):
        raise ValueError("Need vectors[N,3] and camera poses[N,4,4]")
    # Relative vectors rotate only: adding camera translation here is wrong.
    return np.einsum("ni,nji->nj", vectors * baseline_m, poses[:, :3, :3])


def episode_summary(values, episodes, seed=43):
    values = np.asarray(values, dtype=np.float64)
    result = paired_group_bootstrap(values, np.zeros(len(values)), episodes, seed=seed)
    return {"episode_mean_mm": result["mean_difference"], "ci95_mm": result["ci95"],
            "observations": len(values), "episodes": result["group_count"],
            "row_mean_mm": float(values.mean()) if len(values) else None,
            "median_mm": float(np.median(values)) if len(values) else None,
            "p90_mm": float(np.quantile(values, .9)) if len(values) else None,
            "per_episode": {str(int(ep)): float(values[np.asarray(episodes) == ep].mean())
                            for ep in np.unique(episodes)}}


class CameraArrays:
    """Flat camera rows with one immutable identity per original physical row."""
    NAMES = ("episode", "group", "split", "variant", "physical_sample", "camera_variant",
             "camera_domain", "pressure", "state", "anchors", "labels", "palm", "known_cameras")

    def __init__(self, data, *, check=True):
        self.root = Path(data)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if not self.manifest.get("complete"):
            raise ValueError("Camera data manifest is incomplete")
        for name in self.NAMES:
            setattr(self, name, np.load(self.root / (name + ".npy")))
        self.camera = self.known_cameras[:, 0]
        self.nominal = np.flatnonzero(self.camera_variant == 0)
        if check:
            self.validate()
        self.lookup = {(int(p), int(c)): i for i, (p, c) in
                       enumerate(zip(self.physical_sample, self.camera_variant))}
        self.physical_nominal = {int(self.physical_sample[i]): int(i) for i in self.nominal}
        train = self.nominal[self.split[self.nominal] == 0]
        baselines = np.linalg.norm(self.known_cameras[:, 1, :3, 3] - self.camera[:, :3, 3], axis=1)
        self.baseline = float(np.mean(baselines[train]))
        declared = self.manifest.get("reference_baseline_m")
        if declared is not None and abs(float(declared) - self.baseline) > 1e-5:
            raise ValueError("Declared reference baseline differs from nominal train cameras")
        if not np.isfinite(baselines).all() or self.baseline < 1e-5 or np.max(np.abs(baselines - self.baseline)) > 1e-5:
            raise ValueError("Camera augmentation changed the fixed stereo baseline")
        rotation, origin = self.camera[:, :3, :3], self.camera[:, :3, 3]
        self.target = np.einsum("ni,nij->nj", self.labels - self.palm[:, :3, 3], rotation) / self.baseline
        self.anchor_input = np.einsum("nki,nij->nkj", self.anchors - origin[:, None], rotation) / self.baseline
        self.flags = self.pressure > 5e-4
        self.state_mean = self.state[train].mean(0)
        self.state_std = np.maximum(self.state[train].std(0), 1e-3)
        self.target_mean = self.target[train].mean(0)
        self.target_std = np.maximum(self.target[train].std(0), 1e-3)
        self.pressure_scale = float(max(np.quantile(np.log1p(np.maximum(self.pressure[train], 0)), .99), 1e-3))
        self.state_input = (self.state - self.state_mean) / self.state_std
        self.pressure_input = np.log1p(np.maximum(self.pressure, 0)) / self.pressure_scale
        self.train_physical = self.physical_sample[train].astype(int)
        self.dev_ids = np.flatnonzero((self.split == 1) & (self.camera_domain < 2))
        self.test_ids = np.flatnonzero(self.split == 2)
        self.normalization = {"scope": "Only nominal-camera TRAIN physical rows; identical for all regimes/modes/seeds",
            "reference_baseline_m": self.baseline, "fit_rows": train.tolist(),
            "state_mean": self.state_mean.tolist(), "state_std": self.state_std.tolist(),
            "target_mean": self.target_mean.tolist(), "target_std": self.target_std.tolist(),
            "pressure_scale": self.pressure_scale}
        self.training_pairs = self.object_pairs(np.flatnonzero((self.split == 0) & (self.camera_variant == 0)))

    def validate(self):
        n = len(self.episode)
        if any(len(getattr(self, key)) != n for key in self.NAMES) or not n:
            raise ValueError("Camera arrays have inconsistent/empty row counts")
        shapes = {"pressure": (n, 44), "state": (n, 22), "anchors": (n, 44, 3),
                  "labels": (n, 3), "palm": (n, 4, 4), "known_cameras": (n, 2, 4, 4)}
        for key, shape in shapes.items():
            if getattr(self, key).shape != shape or not np.isfinite(getattr(self, key)).all():
                raise ValueError(f"Invalid {key} shape/nonfinite values")
        if (self.pressure < 0).any() or not set(self.split.tolist()) <= {0, 1, 2}:
            raise ValueError("Invalid pressure or split")
        for value in range(3):
            if not (self.split == value).any():
                raise ValueError("Empty train/dev/test split")
        if any(len(np.unique(self.split[self.episode == ep])) != 1 for ep in np.unique(self.episode)):
            raise ValueError("An original episode crosses splits")
        if len(set(zip(self.physical_sample.tolist(), self.camera_variant.tolist()))) != n:
            raise ValueError("Duplicate physical-sample/camera key")
        if not np.array_equal(self.camera_domain, np.where(self.camera_variant == 0, 0, np.where(self.camera_variant <= 3, 1, 2))):
            raise ValueError("Unexpected camera domain/variant mapping")
        for physical in np.unique(self.physical_sample):
            ids = np.flatnonzero(self.physical_sample == physical)
            first = ids[0]
            expected = set(range(7)) if self.split[first] == 2 else set(range(4))
            if set(self.camera_variant[ids].tolist()) != expected:
                raise ValueError("Missing/extra camera variants for a physical sample")
            for key in ("episode", "split", "group", "variant", "state", "pressure", "anchors", "labels", "palm"):
                if not np.allclose(getattr(self, key)[ids], getattr(self, key)[first], rtol=0, atol=1e-6):
                    raise ValueError(f"Camera render changed physical data: {key}")
        # Sibling object states must share exactly one camera transform for a
        # given view ID; otherwise displacement pairs confound camera motion.
        for group in np.unique(self.group):
            ids = np.flatnonzero(self.group == group)
            if len(np.unique(self.episode[ids])) != 1 or len(np.unique(self.split[ids])) != 1:
                raise ValueError("Source group crosses episode/split")
            for view in np.unique(self.camera_variant[ids]):
                selected = ids[self.camera_variant[ids] == view]
                if not np.allclose(self.known_cameras[selected], self.known_cameras[selected[0]], rtol=0, atol=1e-6):
                    raise ValueError("Object siblings do not share the same camera pose")
        rotations = self.known_cameras[:, :, :3, :3]
        if not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
            raise ValueError("Camera rotations are not proper rigid rotations")

    def sample(self, rng, view_rng, batch_size, regime):
        physical = rng.choice(self.train_physical, batch_size, replace=True)
        views = view_rng.integers(0, 4, batch_size) if regime == "augmented" else np.zeros(batch_size, dtype=int)
        if regime not in {"fixed", "augmented"}:
            raise ValueError("Unknown training regime")
        return np.asarray([self.lookup[(int(p), int(v))] for p, v in zip(physical, views)], dtype=int)

    def base_predictions(self, camera_prediction, ids):
        relative = camera_vectors_to_base(camera_prediction, self.camera[ids], self.baseline)
        return relative + self.palm[ids, :3, 3]

    def object_pairs(self, ids):
        pairs = []
        ids = np.asarray(ids)
        for group in np.unique(self.group[ids]):
            selected = ids[self.group[ids] == group]
            for view in np.unique(self.camera_variant[selected]):
                rows = selected[self.camera_variant[selected] == view]
                nominal = rows[self.variant[rows] == 0]
                if len(nominal) != 1:
                    raise ValueError("Each object source/view requires exactly one object nominal")
                for other in rows[self.variant[rows] != 0]:
                    pairs.append((int(group), int(nominal[0]), int(other)))
        return pairs


def source_equal_displacement(data, ids, predicted_base):
    ids = np.asarray(ids)
    prediction = {int(row): value for row, value in zip(ids, predicted_base)}
    sources = {}
    for group, first, second in data.object_pairs(ids):
        actual = data.labels[second] - data.labels[first]
        predicted = prediction[second] - prediction[first]
        true_length, length = np.linalg.norm(actual), np.linalg.norm(predicted)
        if true_length < 1e-8:
            raise ValueError("Object intervention has zero translation")
        along = float(predicted @ actual / true_length)
        sources.setdefault(group, []).append({
            "displacement_error_mm": float(np.linalg.norm(predicted - actual) * 1000),
            "zero_shift_error_mm": float(true_length * 1000),
            "predicted_shift_mm": float(length * 1000), "magnitude_ratio": float(length / true_length),
            "direction_cosine": float(along / length) if length > 1e-8 else None,
            "perpendicular_response_mm": float(np.linalg.norm(predicted - along * actual / true_length) * 1000)})
    rows = []
    for group, values in sources.items():
        row = {"group": int(group), "episode": int(data.episode[np.flatnonzero(data.group == group)[0]]), "pairs": len(values)}
        for key in values[0]:
            finite = [v[key] for v in values if v[key] is not None]
            row[key] = float(np.mean(finite)) if finite else None
        rows.append(row)
    metrics = {}
    for key in (rows[0] if rows else {}):
        if key in {"group", "episode", "pairs"}:
            continue
        selected = [r for r in rows if r[key] is not None]
        result = paired_group_bootstrap(np.asarray([r[key] for r in selected]), np.zeros(len(selected)), np.asarray([r["episode"] for r in selected]))
        metrics[key] = {"mean": result["mean_difference"], "ci95": result["ci95"]}
    return {"source_equal_metrics": metrics, "sources": rows,
            "pairs": sum(r["pairs"] for r in rows), "source_groups": len(rows),
            "scope": "same source + same camera; object nominal-to-variant; source then original episode equally weighted"}


def cross_view_consistency(data, ids, predictions, domain):
    lookup = {int(row): pred for row, pred in zip(ids, predictions)}
    rows = []
    for physical in np.unique(data.physical_sample[ids]):
        nominal = data.lookup[(int(physical), 0)]
        selected = [i for i in ids if data.physical_sample[i] == physical and data.camera_domain[i] == domain]
        if not selected or nominal not in lookup:
            continue
        distances = [np.linalg.norm(lookup[int(i)] - lookup[nominal]) * 1000 for i in selected]
        rows.append({"physical_sample": int(physical), "episode": int(data.episode[nominal]),
                     "distance_to_nominal_mm": float(np.mean(distances))})
    if not rows:
        return None
    return {**episode_summary([r["distance_to_nominal_mm"] for r in rows], [r["episode"] for r in rows]),
            "rows": rows, "scope": "mean base-frame prediction shift from the SAME physical sample's nominal view; accuracy must also be checked"}


def evaluate_predictions(data, ids, predicted_camera):
    ids = np.asarray(ids)
    predicted = data.base_predictions(predicted_camera, ids)
    errors = np.linalg.norm(predicted - data.labels[ids], axis=1) * 1000
    domains = {}
    for domain, name in DOMAINS.items():
        mask = data.camera_domain[ids] == domain
        if not mask.any():
            continue
        domains[name] = {"center": episode_summary(errors[mask], data.episode[ids[mask]]),
            "displacement": source_equal_displacement(data, ids[mask], predicted[mask]),
            "cross_view": cross_view_consistency(data, ids, predicted, domain) if domain else None}
    return {"domains": domains, "scope": "spatial representation only; calibrated base-frame millimetres; not surface reconstruction or policy success"}, predicted, errors
