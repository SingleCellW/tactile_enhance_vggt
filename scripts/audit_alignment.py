#!/usr/bin/env python3
"""Audit legacy replay tactile anchors against raw-state FK without changing data.

Lag comparisons stay inside each original episode. This audits anchor timing,
not RGB pixel identity; a low median does not certify force/RGB synchronization.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


def load_model(asset_root, mujoco):
    assets = {}
    for path in sorted(Path(asset_root).rglob("*")):
        if not path.is_file():
            continue
        if path.name in assets:
            raise ValueError(f"Ambiguous asset basename: {path.name}")
        if path.suffix == ".xml":
            tree = ET.fromstring(path.read_text())
            for element in tree.iter():
                if "file" in element.attrib:
                    element.set("file", Path(element.get("file")).name)
                if element.tag == "compiler":
                    element.attrib.pop("meshdir", None)
                    element.attrib.pop("texturedir", None)
            assets[path.name] = ET.tostring(tree)
        else:
            assets[path.name] = path.read_bytes()
    return mujoco.MjModel.from_xml_string(assets.pop("scene.xml").decode(), assets=assets)


def statistics(values):
    values = np.asarray(values).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "median_m": float(np.median(values)),
            "p90_m": float(np.quantile(values, .9)), "p95_m": float(np.quantile(values, .95)),
            "p99_m": float(np.quantile(values, .99)), "max_m": float(values.max()),
            "rmse_m": float(np.sqrt(np.mean(values ** 2))),
            "fraction_over_1cm": float(np.mean(values > .01)),
            "fraction_over_2cm": float(np.mean(values > .02)),
            "fraction_over_5cm": float(np.mean(values > .05))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--projector-file", type=Path, required=True)
    parser.add_argument("--tactile-points", type=Path, required=True)
    parser.add_argument("--physics-site", type=Path)
    parser.add_argument("--lag-radius", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.physics_site:
        sys.path.insert(0, str(args.physics_site))
    import mujoco
    spec = importlib.util.spec_from_file_location("audited_fixed44_projector", args.projector_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = load_model(args.asset_root, mujoco)
    projector = module.Fixed44SaliencyProjector(model, args.tactile_points)
    data = mujoco.MjData(model)
    with np.load(args.readouts, allow_pickle=False) as cache:
        state = cache["state22"].copy()
        recorded = cache["anchors_world"].copy()
        episode = cache["episode_index"].copy()
        contact = cache["contact44"].copy() > 0
    fk = np.empty_like(recorded)
    for i, q in enumerate(state):
        data.qpos[:] = model.qpos0
        data.qpos[:22] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        fk[i] = projector.anchor_world(data)
    errors = np.linalg.norm(recorded - fk, axis=-1)
    lag_reports = []
    for lag in range(-args.lag_radius, args.lag_radius + 1):
        source = np.arange(len(state))
        target = source + lag
        valid = (target >= 0) & (target < len(state))
        source, target = source[valid], target[valid]
        same_episode = episode[source] == episode[target]
        source, target = source[same_episode], target[same_episode]
        shifted = np.linalg.norm(recorded[source] - fk[target], axis=-1)
        lag_reports.append(dict(lag=lag, active=statistics(shifted[contact[source]])))
    report = {
        "status": "AUDIT_ONLY_NOT_CERTIFIED",
        "mujoco_version": mujoco.__version__, "frames": len(state),
        "episodes": len(np.unique(episode)),
        "readouts_sha256": hashlib.sha256(args.readouts.read_bytes()).hexdigest(),
        "comparison": "recorded ctrl-replay force-field anchors versus FK(raw state22)",
        "zero_lag_all_anchors": statistics(errors), "zero_lag_active_anchors": statistics(errors[contact]),
        "lag_comparisons": lag_reports,
        "best_global_lag_by_active_rmse": min(lag_reports, key=lambda x: x["active"]["rmse_m"]),
        "interpretation": "Reindexing is only valid if temporal and geometric consistency is demonstrated. Never relocate old forces to raw FK and call them synchronized. RGB capture order requires a separate source/render audit.",
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
