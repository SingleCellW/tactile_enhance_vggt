"""Single-environment deployment adapter for a saved PointACTPolicy.

This is an inference interface, not a simulator driver or a closed-loop success
evaluation. Inputs must use the training sensor order, joint/action convention,
RGB preprocessing, and robot-base metric coordinates. Geometry callbacks remain
responsible for current calibrated cameras and current FK tactile anchors.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from .policy import PointACTPolicy


class PointACTInference:
    """Normalize raw observations and return actions in original joint units.

    ``point_builder(rgb, state, tactile, geometry_context)`` optionally returns
    raw points [visual_count + tactile_dim, 8]. It runs only on a fresh policy
    query. While ``needs_observation`` is false, ``select_action()`` can be called
    without sensor inputs and consumes the existing ACT action queue. Use one
    adapter per environment and call ``reset`` at every episode boundary.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        amp: str = "float32",
        point_builder: Callable[[Any, Any, Any, Any], Any] | None = None,
    ) -> None:
        # Pin a moving "latest" symlink once so weights/config/statistics always
        # come from the same completed checkpoint during background training.
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        self.device = torch.device(device)
        if amp not in {"float32", "bfloat16"}:
            raise ValueError("amp must be float32 or bfloat16")
        if amp == "bfloat16":
            if self.device.type != "cuda":
                raise ValueError("This adapter enables bfloat16 autocast only on CUDA")
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("The selected CUDA device does not support bfloat16")
        self.amp = amp
        self.policy = PointACTPolicy.from_pretrained(self.checkpoint, map_location=self.device)
        self.point_builder = point_builder
        self.stats = json.loads((self.checkpoint / "normalization.json").read_text(encoding="utf-8"))
        self._normalization: dict[str, Tensor] = {}
        for name, dimension in (("state", self.policy.config["state_dim"]), ("action", self.policy.config["action_dim"])):
            for statistic in ("mean", "std"):
                key = f"{name}_{statistic}"
                value = torch.as_tensor(self.stats[key], dtype=torch.float32, device=self.device)
                if value.shape != (dimension,) or not torch.isfinite(value).all():
                    raise ValueError(f"{key} must contain {dimension} finite values")
                if statistic == "std" and not (value > 0).all():
                    raise ValueError(f"{key} must be strictly positive")
                self._normalization[key] = value
        self.pressure_scale = float(self.stats["pressure_scale"])
        if not torch.isfinite(torch.tensor(self.pressure_scale)) or self.pressure_scale <= 0:
            raise ValueError("pressure_scale must be finite and positive")

    @property
    def needs_observation(self) -> bool:
        """Whether the next control tick needs a current geometry observation."""
        return self.policy.config["temporal_ensemble_coeff"] is not None or not self.policy.act._action_queue

    def autocast_context(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.amp == "bfloat16" else nullcontext()
        )

    def reset(self) -> None:
        self.policy.reset()

    def _raw_tensor(self, value: Any, shape: tuple[int, ...], name: str) -> Tensor:
        result = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, received {tuple(result.shape)}")
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} contains nonfinite values")
        return result

    def prepare_observation(self, raw_state: Any, raw_tactile: Any, raw_points: Any) -> dict[str, Tensor]:
        """Apply exactly the training dataset's normalization, without mutation."""
        cfg = self.policy.config
        state = self._raw_tensor(raw_state, (cfg["state_dim"],), "raw_state")
        tactile = self._raw_tensor(raw_tactile, (cfg["tactile_dim"],), "raw_tactile")
        points = self._raw_tensor(raw_points, (cfg["n_visual_points"] + cfg["tactile_dim"], 8), "raw_points").clone()
        state = (state - self._normalization["state_mean"]) / self._normalization["state_std"]
        tactile = torch.log1p(torch.clamp_min(tactile, 0)) / self.pressure_scale
        points[:, 6] = torch.log1p(torch.clamp_min(points[:, 6], 0)) / self.pressure_scale
        return {
            "observation.state": state.unsqueeze(0),
            "observation.tactile": tactile.unsqueeze(0),
            "observation.points": points.unsqueeze(0),
        }

    def denormalize_action(self, normalized: Tensor) -> Tensor:
        """Convert one action or a chunk once, after ACT queue/ensembling."""
        return normalized.float() * self._normalization["action_std"] + self._normalization["action_mean"]

    def _build_observation(self, raw_state, raw_tactile, raw_points, raw_rgb, geometry_context):
        if raw_state is None or raw_tactile is None:
            raise ValueError("A fresh policy query requires raw_state and raw_tactile")
        if raw_points is None:
            if self.point_builder is None:
                raise ValueError("Provide raw_points or configure a point_builder")
            if raw_rgb is None:
                raise ValueError("A geometry refresh requires raw_rgb")
            raw_points = self.point_builder(raw_rgb, raw_state, raw_tactile, geometry_context)
        return self.prepare_observation(raw_state, raw_tactile, raw_points)

    @torch.inference_mode()
    def select_action(
        self, raw_state=None, raw_tactile=None, raw_points=None, *, raw_rgb=None, geometry_context=None,
    ) -> Tensor:
        """Return [action_dim] in physical units; queued actions need no new RGB."""
        batch = (
            self._build_observation(raw_state, raw_tactile, raw_points, raw_rgb, geometry_context)
            if self.needs_observation else {}
        )
        with self.autocast_context():
            normalized = self.policy.select_action(batch)
        return self.denormalize_action(normalized)[0]

    @torch.inference_mode()
    def predict_action_chunk(
        self, raw_state, raw_tactile, raw_points=None, *, raw_rgb=None, geometry_context=None,
    ) -> Tensor:
        """Return a fresh physical action chunk without filling the action queue."""
        batch = self._build_observation(raw_state, raw_tactile, raw_points, raw_rgb, geometry_context)
        with self.autocast_context():
            normalized = self.policy.predict_action_chunk(batch)
        return self.denormalize_action(normalized)[0]
