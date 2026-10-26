"""Matched ACT heads with the existing visual readout and direct pressure input.

VGGT and the source visual probe's state encoder run outside this module. The
only upstream component optionally updated here is the original 1088->256
visual readout; all four conditions instantiate the same ACT environment input.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .token_policy import TokenACTPolicy

READOUT_INPUT_DIM = 1088
VISUAL_DIM = 256
TACTILE_DIM = 44
ENVIRONMENT_DIM = VISUAL_DIM + TACTILE_DIM


def make_visual_readout(state_dict: dict[str, Tensor], *, projection: str) -> nn.Sequential:
    """Restore exactly the source probe readout, with explicit freeze semantics."""
    if projection not in {"frozen", "joint"}:
        raise ValueError("projection must be frozen or joint")
    readout = nn.Sequential(nn.LayerNorm(READOUT_INPUT_DIM), nn.Linear(READOUT_INPUT_DIM, VISUAL_DIM), nn.GELU())
    readout.load_state_dict(state_dict, strict=True)
    readout.requires_grad_(projection == "joint")
    return readout


def visual_probe_readout(checkpoint: str | Path) -> dict[str, Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kwargs", {}).get("mode") != "visual":
        raise ValueError("Readout initialization must come from a visual-mode probe checkpoint")
    result = {key.removeprefix("readout."): value for key, value in payload["model"].items()
              if key.startswith("readout.")}
    if not result:
        raise ValueError("Visual probe checkpoint has no readout parameters")
    return result


class SimpleTokenACTPolicy(TokenACTPolicy):
    """Existing ACT with ``readout(input1088) || pressure44`` as environment300.

    State, pressure and action inputs are already normalized by the caller.
    ``tactile='zero'`` zeroes the 44 slots inside the model, during training and
    inference, without changing cached data or parameter counts.
    """

    def __init__(self, config: dict[str, Any] | None = None, *, projection: str = "frozen",
                 tactile: str = "zero", readout_state_dict: dict[str, Tensor]) -> None:
        if projection not in {"frozen", "joint"} or tactile not in {"zero", "pressure"}:
            raise ValueError("Expected projection=frozen/joint and tactile=zero/pressure")
        config = copy.deepcopy(config or {})
        if config.get("encoder_dim", ENVIRONMENT_DIM) != ENVIRONMENT_DIM:
            raise ValueError("SimpleTokenACTPolicy requires encoder_dim=300 in every condition")
        config["encoder_dim"] = ENVIRONMENT_DIM
        if config.get("tactile_dim", TACTILE_DIM) != TACTILE_DIM:
            raise ValueError("SimpleTokenACTPolicy requires tactile_dim=44")
        # Keep ACT initialization and random-number consumption identical in all
        # conditions. Freezing changes optimizer membership, never architecture.
        super().__init__(config)
        self.readout = make_visual_readout(readout_state_dict, projection=projection)
        self.structure = {"projection": projection, "tactile": tactile,
                          "readout_input_dim": READOUT_INPUT_DIM, "visual_dim": VISUAL_DIM,
                          "tactile_dim": TACTILE_DIM}

    @classmethod
    def from_visual_probe(cls, config: dict[str, Any], checkpoint: str | Path, *,
                          projection: str, tactile: str) -> SimpleTokenACTPolicy:
        return cls(config, projection=projection, tactile=tactile,
                   readout_state_dict=visual_probe_readout(checkpoint))

    def _act_batch(self, batch: dict[str, Tensor], *, training: bool = False) -> dict[str, Tensor]:
        state = batch["observation.state"]
        visual_input = batch["observation.visual_readout_input"]
        tactile = batch["observation.tactile"]
        if state.ndim != 2 or visual_input.shape != (state.shape[0], READOUT_INPUT_DIM):
            raise ValueError("observation.visual_readout_input must have shape [B, 1088]")
        if tactile.shape != (state.shape[0], TACTILE_DIM):
            raise ValueError("observation.tactile must have shape [B, 44]")
        visual = self.readout(visual_input)
        pressure = torch.zeros_like(tactile) if self.structure["tactile"] == "zero" else tactile
        features = torch.cat((visual, pressure.to(dtype=visual.dtype)), dim=-1)
        return super()._act_batch({**batch, "observation.environment_state": features}, training=training)

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(self.config)
        config["device"] = "cpu"
        (directory / "config.json").write_text(json.dumps({
            "format_version": 1, "policy_type": "simple_token_act", "config": config,
            "structure": self.structure,
        }, indent=2) + "\n", encoding="utf-8")
        torch.save({key: value.detach().cpu() for key, value in self.state_dict().items()}, directory / "model.pt")

    @classmethod
    def from_pretrained(cls, directory: str | Path, *,
                        map_location: str | torch.device = "cpu") -> SimpleTokenACTPolicy:
        directory = Path(directory)
        payload = json.loads((directory / "config.json").read_text())
        if payload.get("format_version") != 1 or payload.get("policy_type") != "simple_token_act":
            raise ValueError("Unsupported SimpleTokenACTPolicy checkpoint format")
        structure = payload["structure"]
        if (structure.get("readout_input_dim"), structure.get("visual_dim"), structure.get("tactile_dim")) != (1088, 256, 44):
            raise ValueError("Unsupported simple-token dimensions")
        weights = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        readout = {key.removeprefix("readout."): value for key, value in weights.items() if key.startswith("readout.")}
        policy = cls(payload["config"], projection=structure["projection"], tactile=structure["tactile"],
                     readout_state_dict=readout)
        policy.load_state_dict(weights, strict=True)
        policy.to(map_location).eval()
        policy.reset()
        return policy


class SimpleTokenACTInference:
    """Raw state/pressure deployment adapter, preserving ACT's command queue."""

    def __init__(self, checkpoint: str | Path, *, device="cpu", amp="float32", n_action_steps=None) -> None:
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        self.device = torch.device(device)
        if amp not in {"float32", "bfloat16"}:
            raise ValueError("amp must be float32 or bfloat16")
        if amp == "bfloat16" and (self.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("bfloat16 inference requires a supported CUDA device")
        self.amp = amp
        self.policy = SimpleTokenACTPolicy.from_pretrained(self.checkpoint, map_location=self.device)
        self.checkpoint_n_action_steps = self.policy.config["n_action_steps"]
        if n_action_steps is not None:
            if not isinstance(n_action_steps, int) or isinstance(n_action_steps, bool) or n_action_steps not in {2, 8}:
                raise ValueError("n_action_steps override must be 2 or 8")
            if n_action_steps > self.policy.config["chunk_size"]:
                raise ValueError("n_action_steps cannot exceed chunk_size")
            if self.policy.config["temporal_ensemble_coeff"] is not None:
                raise ValueError("Execution-step overrides require ACT queue mode without temporal ensemble")
            self.policy.config["n_action_steps"] = n_action_steps
            self.policy.act.config.n_action_steps = n_action_steps
            self.policy.reset()
            if self.policy.act._action_queue.maxlen != n_action_steps:
                raise ValueError("ACT reset did not apply the requested action-queue length")
        self.stats = json.loads((self.checkpoint / "normalization.json").read_text())
        self.feature_manifest = json.loads((self.checkpoint / "feature_manifest.json").read_text())
        self._normalization = {}
        for name, dimension in (("state", self.policy.config["state_dim"]), ("action", self.policy.config["action_dim"])):
            for statistic in ("mean", "std"):
                key = f"{name}_{statistic}"
                value = torch.as_tensor(self.stats[key], dtype=torch.float32, device=self.device)
                if value.shape != (dimension,) or not torch.isfinite(value).all():
                    raise ValueError(f"{key} must contain {dimension} finite values")
                if statistic == "std" and not (value > 0).all():
                    raise ValueError(f"{key} must be strictly positive")
                self._normalization[key] = value
        self.pressure_scale = float(self.stats["simple_pressure_scale"])
        if not 0 < self.pressure_scale < float("inf"):
            raise ValueError("simple_pressure_scale must be finite and positive")

    @property
    def needs_observation(self) -> bool:
        return self.policy.config["temporal_ensemble_coeff"] is not None or not self.policy.act._action_queue

    def reset(self) -> None:
        self.policy.reset()

    def autocast_context(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.amp == "bfloat16" else nullcontext()

    def prepare_observation(self, raw_state, readout_input, raw_pressure) -> dict[str, Tensor]:
        if raw_state is None or readout_input is None or raw_pressure is None:
            raise ValueError("A fresh query requires raw_state, readout_input and raw_pressure")
        inputs = {}
        for name, raw, dimension in (("state", raw_state, self.policy.config["state_dim"]),
                                     ("visual_readout_input", readout_input, READOUT_INPUT_DIM),
                                     ("tactile", raw_pressure, TACTILE_DIM)):
            value = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
            if value.shape != (dimension,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain {dimension} finite values")
            inputs[f"observation.{name}"] = value.unsqueeze(0)
        inputs["observation.state"] = (inputs["observation.state"] - self._normalization["state_mean"]) / self._normalization["state_std"]
        inputs["observation.tactile"] = torch.log1p(inputs["observation.tactile"].clamp_min(0)) / self.pressure_scale
        return inputs

    def denormalize_action(self, action: Tensor) -> Tensor:
        return action.float() * self._normalization["action_std"] + self._normalization["action_mean"]

    @torch.inference_mode()
    def select_action(self, raw_state=None, readout_input=None, raw_pressure=None) -> Tensor:
        batch = self.prepare_observation(raw_state, readout_input, raw_pressure) if self.needs_observation else {}
        with self.autocast_context():
            action = self.policy.select_action(batch)
        return self.denormalize_action(action)[0]

    @torch.inference_mode()
    def predict_action_chunk(self, raw_state, readout_input, raw_pressure) -> Tensor:
        with self.autocast_context():
            action = self.policy.predict_action_chunk(self.prepare_observation(raw_state, readout_input, raw_pressure))
        return self.denormalize_action(action)[0]
