"""Synchronous PickLift observations in the frozen native tactile environment.

Importing this module does not import MuJoCo or Torch. The worker owns those
dependencies; policy inference stays in its separate Python environment.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

import numpy as np

CAMERAS = ("left_cam", "right_cam")
LINK_MAP = {"palm_lower": "palm", "fingertip": "if_ds", "dip": "if_md", "pip": "if_px", "mcp_joint": "if_bs",
            "fingertip_2": "mf_ds", "dip_2": "mf_md", "pip_2": "mf_px", "mcp_joint_2": "mf_bs",
            "fingertip_3": "rf_ds", "dip_3": "rf_md", "pip_3": "rf_px", "mcp_joint_3": "rf_bs",
            "thumb_temp_base": "th_mp", "thumb_pip": "th_bs", "thumb_dip": "th_px", "thumb_fingertip": "th_ds"}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_model(asset_root, mujoco):
    """Resolve the historical absolute asset paths in memory only."""
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


def training_rgb_transform(rgb):
    """Match the historical converter's PIL center crop and LANCZOS resize."""
    from PIL import Image
    image = np.asarray(rgb)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError(f"Expected native RGB uint8[480,640,3], got {image.shape}/{image.dtype}")
    return np.asarray(Image.fromarray(image).crop((80, 0, 560, 480)).resize(
        (224, 224), Image.Resampling.LANCZOS)).copy()


def body_origin_state(model, data, bid, mujoco, *, object_xyzw=False):
    """Return native xyz/quat/linear/angular state, verified at the xyz origin."""
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, velocity, 0)
    velocity[3:] += np.cross(velocity[:3], data.xpos[bid] - data.xipos[bid])
    jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[bid], bid)
    error = float(np.max(np.abs(velocity - np.r_[jacr @ data.qvel, jacp @ data.qvel])))
    if error > 1e-8:
        raise ValueError(f"Body-origin velocity/Jacobian mismatch: {error}")
    quat = data.xquat[bid][[1, 2, 3, 0]] if object_xyzw else data.xquat[bid]
    return np.r_[data.xpos[bid], quat, velocity[3:], velocity[:3]], error


class NativeTactileAdapter:
    """The same frozen readout and corrected state adapter used for training."""
    def __init__(self, model, dynamic_root, device):
        self.root, self.device, self.model = Path(dynamic_root), device, model
        sys.path[:0] = [str(self.root / "eval/vendor"), str(self.root / "eval/sim"), str(self.root)]
        import mujoco
        import torch
        from tactile_sim.models import models
        from fixed44_geometry import Fixed44SaliencyProjector
        from dynamic44 import continuous_readout, recorded_anchors
        self.mujoco, self.torch = mujoco, torch
        self.continuous_readout, self.recorded_anchors = continuous_readout, recorded_anchors
        self.geometry = np.load(self.root / "readouts/geometry.npz", allow_pickle=False)
        self.bounds = json.loads((self.root / "readouts/geometry.json").read_text())["link_bounds"]
        self.canonical = np.load(self.root / "eval/sim/tactile_sim/tac-pts.npy", allow_pickle=False)
        self.projector = Fixed44SaliencyProjector(model, self.root / "eval/sim/tactile_sim/tac-pts.npy")
        self.object_mesh = self.root / "eval/assets/rotate/meshes/0.07cube.obj"
        urdf = self.root / "eval/sim/tactile_sim/urdf"
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        self.hand = models(str(urdf / "leap_hand_right_tactile.urdf"), str(urdf), str(self.object_mesh), device)
        self.hand.set_object(str(self.object_mesh))
        self.hand.setup_models()
        for name in self.hand.hand_model.mesh:
            self.hand.hand_model.mesh[name]["surface_points"] = torch.tensor(self.geometry["surface_" + name], device=device)
        if self.hand.ind2link != self.bounds:
            raise ValueError("Native dense point order differs from frozen geometry")
        self.body_ids = {name: model.body(name).id for name in set(LINK_MAP.values()) | {"palm", "cube"}}
        self.reset()

    def reset(self):
        self.hand.init = False

    def observe(self, data):
        max_velocity_error = 0.0

        def state(name, object_xyzw=False):
            nonlocal max_velocity_error
            value, error = body_origin_state(self.model, data, self.body_ids[name], self.mujoco,
                                             object_xyzw=object_xyzw)
            max_velocity_error = max(max_velocity_error, error)
            return value

        hand_state, object_state = state("palm"), state("cube", True)
        links = np.stack([state(LINK_MAP[name]) for name in self.hand.ind2link])
        tensor = lambda value: self.torch.as_tensor(value, dtype=self.torch.float32, device=self.device).unsqueeze(0)
        qpos_before = data.qpos.copy()
        # Native SDF normals use autograd internally; inference_mode/no_grad
        # would disable that geometry calculation even though no model trains.
        self.hand.cal_tactile_info(tensor(hand_state), tensor(object_state), tensor(data.qpos[6:22]), tensor(links))
        points = self.hand.current_surface_points[0].detach().cpu().numpy()
        forces = self.hand.current_forces[0].detach().cpu().numpy()
        if not np.isfinite(forces).all():
            raise ValueError("Native force contains nonfinite values")
        tactile = self.continuous_readout(forces[:, -1], self.geometry["weights"]).astype(np.float32)
        anchors = self.projector.anchor_world(data).astype(np.float32)
        recovered, rigid_error = self.recorded_anchors(points[None], self.geometry["points_reference"], self.bounds, self.canonical)
        fk_error = float(np.linalg.norm(recovered[0] - anchors, axis=-1).max())
        if fk_error > 2e-5 or rigid_error > 2e-5:
            raise ValueError(f"Native/FK anchor mismatch: {fk_error}, {rigid_error}")
        bid = self.body_ids["cube"]
        vertices = self.hand.object_model.object_vertices_list[0].detach().cpu().numpy()
        mesh = np.asarray(self.hand.object_model.object_mesh_list[0].vertices)
        expected = mesh @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
        mesh_error = float(np.linalg.norm(vertices - expected, axis=-1).max())
        if mesh_error > 2e-6:
            raise ValueError(f"Object quaternion convention mismatch: {mesh_error}")
        np.testing.assert_array_equal(data.qpos, qpos_before)
        return tactile, anchors, {"native_fk_anchor_max_m": fk_error, "rigid_error_m": float(rigid_error),
                                  "object_vertices_max_m": mesh_error,
                                  "body_origin_velocity_jacobian_max_abs": max_velocity_error}


class PickLiftSimulation:
    """One persistent simulator; physics and sensor-proxy time are explicit."""
    def __init__(self, dynamic_root, raw_root, timestamps_root, source_readouts, device="cuda:0", substeps=1, sensor_dt=.05):
        import mujoco
        if mujoco.__version__ != "3.4.0":
            raise RuntimeError(f"Expected frozen MuJoCo 3.4.0, got {mujoco.__version__}")
        if type(substeps) is not int or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        if not np.isfinite(sensor_dt) or sensor_dt <= 0:
            raise ValueError("sensor_dt must be positive")
        self.mujoco, self.root = mujoco, Path(dynamic_root)
        self.model = load_frozen_model(self.root / "eval/assets/picklift", mujoco)
        if (self.model.nq, self.model.nv, self.model.nu) != (29, 28, 22):
            raise ValueError("Unexpected frozen PickLift state/action dimensions")
        np.testing.assert_allclose(self.model.opt.timestep, .01, rtol=0, atol=1e-12)
        self.substeps, self.control_dt = substeps, float(self.model.opt.timestep * substeps)
        self.sensor_dt = float(sensor_dt)
        self.stable_required_ticks = int(np.ceil(1.0 / self.control_dt))
        self.data, self.sensor_data = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.raw_root, self.timestamps_root = Path(raw_root), Path(timestamps_root)
        self.raw_episodes = sorted(path for path in self.raw_root.iterdir() if (path / "prioception").exists())
        with np.load(source_readouts, allow_pickle=False) as source:
            self.source_state = source["state22"].copy()
            self.source_episode = source["episode_index"].copy()
            self.source_frame = source["frame_index"].copy()
        self.tactile = NativeTactileAdapter(self.model, self.root, device)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.base_id, self.cube_id = self.model.body("link_base").id, self.model.body("cube").id
        def descendants(root):
            result = set()
            for body in range(self.model.nbody):
                ancestor = body
                while ancestor:
                    if ancestor == root:
                        result.add(body)
                        break
                    ancestor = int(self.model.body_parentid[ancestor])
            return result
        self.robot_bodies, self.cube_bodies = descendants(self.base_id), descendants(self.cube_id)
        self.camera_ids = [self.model.camera(name).id for name in CAMERAS]
        self.ready = False
        self.provenance = {"mujoco_version": mujoco.__version__, "physics_dt": float(self.model.opt.timestep),
                           "control_dt": self.control_dt, "substeps": substeps, "sensor_dt": self.sensor_dt,
                           "control_semantics": f"absolute 22 position targets, held across {substeps} physics steps; physical seconds per action={self.control_dt}",
                           "velocity": "full qpos29 causal difference between consecutive control ticks divided by explicit sensor_dt; .05s default matches training wall-time proxy, not necessarily physical elapsed time; separate sensor MjData; COM to xpos shift verified against Jacobian",
                           "force_law": "unchanged frozen native proxy, legacy cross(r,omega) retained; not calibrated Newtons",
                           "rgb_transform": "MuJoCo RGB480x640 -> center crop x80:560 -> PIL LANCZOS224; freshly forwarded current state, no recorded-data lag",
                           "geometry_sha256": file_sha(self.root / "readouts/geometry.npz"),
                           "native_models_sha256": file_sha(self.root / "eval/sim/tactile_sim/models.py"),
                           "object_mesh_sha256": file_sha(self.tactile.object_mesh),
                           "source_readouts_sha256": file_sha(source_readouts)}

    def _sensor_snapshot(self, previous=None, dt=None):
        self.sensor_data.qpos[:] = self.data.qpos
        self.sensor_data.qvel[:] = 0
        if previous is not None:
            self.mujoco.mj_differentiatePos(self.model, self.sensor_data.qvel, dt, previous, self.data.qpos)
        self.sensor_data.time = self.data.time
        self.mujoco.mj_forward(self.model, self.sensor_data)
        np.testing.assert_allclose(self.sensor_data.xpos, self.data.xpos, rtol=0, atol=1e-12)

    def reset(self, episode_id, frame=0, seed=0, object_xy_offset=None):
        if type(episode_id) is not int or not 0 <= episode_id < len(self.raw_episodes):
            raise ValueError("episode_id must index the sorted original episodes")
        raw = self.raw_episodes[episode_id]
        csv_path = self.timestamps_root / f"teleop_log_{raw.name}.csv"
        rows = list(csv.DictReader(csv_path.open()))
        if type(frame) is not int or not 0 <= frame < len(rows):
            raise ValueError("Invalid initialization frame")
        qpath = raw / rows[frame]["prioception_path"]
        qpos = np.load(qpath, allow_pickle=False)
        indices = np.flatnonzero((self.source_episode == episode_id) & (self.source_frame == frame))
        if len(indices) != 1 or qpos.shape != (29,) or not np.isfinite(qpos).all():
            raise ValueError("Raw initialization state does not match source indexing")
        np.testing.assert_array_equal(qpos[:22].astype(np.float32), self.source_state[indices[0]])
        previous, wall_dt = None, None
        if frame:
            previous = np.load(raw / rows[frame - 1]["prioception_path"], allow_pickle=False)
            wall_dt = float(rows[frame]["timestamp"]) - float(rows[frame - 1]["timestamp"])
            if not np.isfinite(wall_dt) or wall_dt <= 0:
                raise ValueError("Non-positive causal initialization dt")
        offset = np.zeros(2) if object_xy_offset is None else np.asarray(object_xy_offset, dtype=np.float64)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError("object_xy_offset must contain two finite values")
        qpos = qpos.copy()
        cube_qadr = self.model.jnt_qposadr[self.model.joint("cube_freejoint").id]
        qpos[cube_qadr:cube_qadr + 2] += offset
        if previous is not None:
            previous = previous.copy()
            previous[cube_qadr:cube_qadr + 2] += offset
        self.mujoco.mj_resetData(self.model, self.data)
        self.mujoco.mj_resetData(self.model, self.sensor_data)
        self.data.qpos[:] = qpos
        if previous is not None:
            # Consecutive recorded states advance one frozen .01s physics
            # step. Wall timestamps describe recording cadence, not dynamics.
            self.mujoco.mj_differentiatePos(self.model, self.data.qvel, float(self.model.opt.timestep), previous, qpos)
        self.data.ctrl[:] = qpos[:22]
        self.mujoco.mj_forward(self.model, self.data)
        # Sensor velocity follows its independently configured training time
        # basis, exactly as it does after every subsequent control tick.
        self._sensor_snapshot(previous, self.sensor_dt if previous is not None else None)
        self.tactile.reset()
        self.episode_id, self.control_tick = episode_id, 0
        self.max_cube_height = self.initial_cube_height = float(self.data.xpos[self.cube_id, 2])
        self.stable_goal_ticks, self.cumulative_clipped_count = 0, 0
        self.first_goal_tick = None
        self.last_clipped_dims = []
        self.initialization = {"episode_id": episode_id, "raw_episode": raw.name, "frame": frame, "seed": int(seed),
                               "qpos_path": str(qpath), "qpos_sha256": file_sha(qpath),
                               "timestamp_file_sha256": file_sha(csv_path), "object_xy_offset": offset.tolist(),
                               "recorded_wall_dt": wall_dt,
                               "physics_velocity_dt": float(self.model.opt.timestep) if previous is not None else None,
                               "sensor_velocity_dt": self.sensor_dt if previous is not None else None}
        self.ready = True
        if self.info()["inside_goal"]:
            self.first_goal_tick = 0
        return self.info()

    def info(self):
        if not self.ready:
            raise RuntimeError("Reset the simulator first")
        center = self.data.xpos[self.cube_id].copy()
        object_state, error = body_origin_state(self.model, self.data, self.cube_id, self.mujoco)
        speed, angular_speed = float(np.linalg.norm(object_state[7:10])), float(np.linalg.norm(object_state[10:13]))
        inside = bool(.4 <= center[0] <= .6 and -.39 <= center[1] <= -.21 and center[2] <= .1)
        contact_count = 0
        for contact in self.data.contact:
            body_a, body_b = (int(self.model.geom_bodyid[geom]) for geom in contact.geom)
            if ((body_a in self.robot_bodies and body_b in self.cube_bodies)
                    or (body_b in self.robot_bodies and body_a in self.cube_bodies)):
                contact_count += 1
        return {"episode_id": self.episode_id, "control_tick": self.control_tick, "sim_time": float(self.data.time),
                "cube_center": center.tolist(), "cube_quaternion_wxyz": self.data.xquat[self.cube_id].tolist(),
                "cube_linear_speed": speed, "cube_angular_speed": angular_speed,
                "linear_speed": speed, "angular_speed": angular_speed,
                "robot_cube_contact_count": contact_count, "first_goal_tick": self.first_goal_tick,
                "inside_goal": inside, "success": inside,
                "max_cube_height": self.max_cube_height, "initial_cube_height": self.initial_cube_height,
                "lifted": bool(self.max_cube_height >= self.initial_cube_height + .05),
                "stable_goal_ticks": self.stable_goal_ticks, "stable_required_ticks": self.stable_required_ticks,
                "strict_stable_goal": self.stable_goal_ticks >= self.stable_required_ticks,
                "strict_stability_rule": "inside_goal and linear_speed<.05m/s and angular_speed<.5rad/s and zero robot-cube contacts for at least one physical second",
                "success_rule": "legacy final cube-center x[.4,.6], y[-.39,-.21], z<=.1; caller evaluates at horizon, no early stop",
                "clipped_dims": self.last_clipped_dims, "clipped_count": len(self.last_clipped_dims),
                "cumulative_clipped_count": self.cumulative_clipped_count,
                "body_origin_velocity_jacobian_max_abs": error,
                "initialization": self.initialization}

    def step(self, action):
        if not self.ready:
            raise RuntimeError("Reset the simulator first")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (22,) or not np.isfinite(action).all():
            raise ValueError("action must contain 22 finite absolute position targets")
        limited = self.model.actuator_ctrllimited.astype(bool)
        clipped = action.copy()
        clipped[limited] = np.clip(action[limited], self.model.actuator_ctrlrange[limited, 0], self.model.actuator_ctrlrange[limited, 1])
        self.last_clipped_dims = np.flatnonzero(clipped != action).tolist()
        self.cumulative_clipped_count += len(self.last_clipped_dims)
        previous = self.data.qpos.copy()
        self.data.ctrl[:] = clipped
        bad_warnings = [self.mujoco.mjtWarning.mjWARN_BADQPOS, self.mujoco.mjtWarning.mjWARN_BADQVEL, self.mujoco.mjtWarning.mjWARN_BADQACC]
        before = [self.data.warning[int(kind)].number for kind in bad_warnings]
        for _ in range(self.substeps):
            self.mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("Nonfinite MuJoCo trajectory")
            if any(self.data.warning[int(kind)].number > count for kind, count in zip(bad_warnings, before)):
                raise RuntimeError("MuJoCo reported invalid physics state; rollout must be marked failed")
        # mj_step leaves some derived positions at the preceding integration
        # stage. Refresh them before rendering, FK, metrics or sensor synthesis.
        self.mujoco.mj_forward(self.model, self.data)
        self._sensor_snapshot(previous, self.sensor_dt)
        self.control_tick += 1
        self.max_cube_height = max(self.max_cube_height, float(self.data.xpos[self.cube_id, 2]))
        info = self.info()
        if info["inside_goal"] and self.first_goal_tick is None:
            self.first_goal_tick = self.control_tick
        stable = (info["inside_goal"] and info["cube_linear_speed"] < .05
                  and info["cube_angular_speed"] < .5 and info["robot_cube_contact_count"] == 0)
        self.stable_goal_ticks = self.stable_goal_ticks + 1 if stable else 0
        return self.info()

    def observe(self):
        if not self.ready:
            raise RuntimeError("Reset the simulator first")
        before = self.data.qpos.copy()
        tactile, anchors, diagnostics = self.tactile.observe(self.sensor_data)
        rgb, cameras = [], []
        for name, camera_id in zip(CAMERAS, self.camera_ids):
            self.renderer.update_scene(self.data, camera=name)
            rgb.append(training_rgb_transform(self.renderer.render()))
            transform = np.eye(4)
            transform[:3, :3] = self.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
            transform[:3, 3] = self.data.cam_xpos[camera_id]
            cameras.append(transform)
        base = np.eye(4)
        base[:3, :3] = self.data.xmat[self.base_id].reshape(3, 3)
        base[:3, 3] = self.data.xpos[self.base_id]
        state = self.data.qpos[:22].astype(np.float32)
        np.testing.assert_array_equal(self.data.qpos, before)
        observation = {"rgb": np.stack(rgb), "state": state, "state22": state,
                       "tactile": tactile, "force44": tactile, "anchors_world": anchors,
                       "known_cameras_cv": np.stack(cameras), "base_to_world": base}
        if not all(np.isfinite(value).all() for value in observation.values()):
            raise ValueError("Nonfinite observation")
        return observation, diagnostics

    def close(self):
        self.renderer.close()
        self.tactile.geometry.close()
