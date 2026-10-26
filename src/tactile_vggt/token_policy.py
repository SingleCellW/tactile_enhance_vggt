"""The existing ACT head driven by frozen, cached VGGT adapter features.

The token representation is supplied by the caller; this module contains no
VGGT or tactile encoder and never updates the upstream representation.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .policy import PointACTPolicy, _resolve_config


class TokenACTPolicy(PointACTPolicy):
    """ACT with a supplied ``observation.environment_state`` [B, encoder_dim].

    Accepts the same configuration as PointACTPolicy so the ACT architecture,
    loss, action queue, and optimizer settings can be matched exactly. Spatial
    point-encoder configuration entries are retained for compatibility but do
    not instantiate an encoder or affect computation. Inputs are already
    normalized by the caller, including state/action; features are used as-is.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        nn.Module.__init__(self)
        self.config = _resolve_config(config or {})
        cfg = self.config
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        act_keys = (
            "chunk_size", "n_action_steps", "dim_model", "n_heads", "dim_feedforward",
            "n_encoder_layers", "n_decoder_layers", "use_vae", "latent_dim",
            "n_vae_encoder_layers", "kl_weight", "dropout", "pre_norm",
            "feedforward_activation", "temporal_ensemble_coeff", "optimizer_lr",
            "optimizer_weight_decay", "device",
        )
        self.act = ACTPolicy(ACTConfig(
            **{key: cfg[key] for key in act_keys},
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (cfg["state_dim"],)),
                "observation.environment_state": PolicyFeature(FeatureType.ENV, (cfg["encoder_dim"],)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (cfg["action_dim"],))},
            normalization_mapping={name: NormalizationMode.IDENTITY for name in ("STATE", "ENV", "ACTION")},
            pretrained_backbone_weights=None,
            push_to_hub=False,
        ))

    def _act_batch(self, batch: dict[str, Tensor], *, training: bool = False) -> dict[str, Tensor]:
        state = batch["observation.state"]
        features = batch["observation.environment_state"]
        if state.ndim != 2 or state.shape[1] != self.config["state_dim"]:
            raise ValueError(f"observation.state must have shape [B, {self.config['state_dim']}]")
        if features.shape != (state.shape[0], self.config["encoder_dim"]):
            raise ValueError(f"observation.environment_state must have shape [B, {self.config['encoder_dim']}]")
        result = {"observation.state": state, "observation.environment_state": features}
        if training:
            expected = (state.shape[0], self.config["chunk_size"], self.config["action_dim"])
            if batch["action"].shape != expected:
                raise ValueError(f"action must have shape {expected}")
            padding = batch.get("action_is_pad")
            if padding is None:
                padding = torch.zeros(expected[:2], dtype=torch.bool, device=state.device)
            if padding.shape != expected[:2] or padding.dtype != torch.bool:
                raise ValueError("action_is_pad must be bool with shape [B, chunk_size]")
            result.update(action=batch["action"], action_is_pad=padding)
        return result

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(self.config)
        config["device"] = "cpu"
        (directory / "config.json").write_text(json.dumps({
            "format_version": 1, "policy_type": "token_act", "config": config,
        }, indent=2) + "\n", encoding="utf-8")
        torch.save({key: value.detach().cpu() for key, value in self.state_dict().items()}, directory / "model.pt")

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, *, map_location: str | torch.device = "cpu",
    ) -> TokenACTPolicy:
        directory = Path(directory)
        payload = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or payload.get("policy_type") != "token_act":
            raise ValueError("Unsupported TokenACTPolicy checkpoint format")
        policy = cls(payload["config"])
        policy.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True), strict=True)
        policy.to(map_location).eval()
        policy.reset()
        return policy


class TokenACTInference:
    """Single-environment raw-state / frozen-feature deployment adapter.

    ``select_action(raw_state, features)`` expects the exact frozen source
    representation recorded in feature_manifest.json, with no extra feature
    normalization. Queued actions can be consumed using ``select_action()``.
    Actions returned here are denormalized to the original command units.
    """

    def __init__(self, checkpoint: str | Path, *, device="cpu", amp="float32") -> None:
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        self.device = torch.device(device)
        if amp not in {"float32", "bfloat16"}:
            raise ValueError("amp must be float32 or bfloat16")
        if amp == "bfloat16" and (
            self.device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("bfloat16 inference requires a supported CUDA device")
        self.amp = amp
        self.policy = TokenACTPolicy.from_pretrained(self.checkpoint, map_location=self.device)
        self.stats = json.loads((self.checkpoint / "normalization.json").read_text())
        feature_path = self.checkpoint / "feature_manifest.json"
        self.feature_manifest = json.loads(feature_path.read_text()) if feature_path.exists() else None
        self._normalization = {}
        for name, dimension in (("state", self.policy.config["state_dim"]), ("action", self.policy.config["action_dim"])):
            for statistic in ("mean", "std"):
                key = f"{name}_{statistic}"
                values = torch.as_tensor(self.stats[key], dtype=torch.float32, device=self.device)
                if values.shape != (dimension,) or not torch.isfinite(values).all():
                    raise ValueError(f"{key} must contain {dimension} finite values")
                if statistic == "std" and not (values > 0).all():
                    raise ValueError(f"{key} must be strictly positive")
                self._normalization[key] = values

    @property
    def needs_observation(self) -> bool:
        return self.policy.config["temporal_ensemble_coeff"] is not None or not self.policy.act._action_queue

    def reset(self) -> None:
        self.policy.reset()

    def autocast_context(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.amp == "bfloat16" else nullcontext()

    def prepare_observation(self, raw_state: Any, features: Any) -> dict[str, Tensor]:
        if raw_state is None or features is None:
            raise ValueError("A fresh policy query requires raw_state and features")
        state = torch.as_tensor(raw_state, dtype=torch.float32, device=self.device)
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if state.shape != (self.policy.config["state_dim"],):
            raise ValueError("raw_state has the wrong shape")
        if features.shape != (self.policy.config["encoder_dim"],):
            raise ValueError("features has the wrong shape")
        if not torch.isfinite(state).all() or not torch.isfinite(features).all():
            raise ValueError("raw_state and features must be finite")
        state = (state - self._normalization["state_mean"]) / self._normalization["state_std"]
        return {"observation.state": state.unsqueeze(0), "observation.environment_state": features.unsqueeze(0)}

    def denormalize_action(self, action: Tensor) -> Tensor:
        return action.float() * self._normalization["action_std"] + self._normalization["action_mean"]

    @torch.inference_mode()
    def select_action(self, raw_state=None, features=None) -> Tensor:
        batch = self.prepare_observation(raw_state, features) if self.needs_observation else {}
        with self.autocast_context():
            prediction = self.policy.select_action(batch)
        return self.denormalize_action(prediction)[0]

    @torch.inference_mode()
    def predict_action_chunk(self, raw_state, features) -> Tensor:
        with self.autocast_context():
            prediction = self.policy.predict_action_chunk(self.prepare_observation(raw_state, features))
        return self.denormalize_action(prediction)[0]
