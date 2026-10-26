"""Spatial point observations connected to the current LeRobot ACT policy.

VGGT runs outside this module. Callers supply its frozen, metric point cache,
normalize state/actions/pressure, and denormalize returned actions. XYZ remains
in metres so that ``local_radius`` has a physical meaning. No vendor changes or
pretrained image downloads are required.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


_DEFAULTS: dict[str, Any] = {
    "mode": "anchored",
    "state_dim": 22,
    "action_dim": 22,
    "tactile_dim": 44,
    "n_visual_points": 512,
    "point_feature_dim": 8,
    "point_hidden_dims": [64, 128, 256],
    "encoder_dim": 256,
    "tactile_hidden_dim": 128,
    "use_local_pooling": True,
    "local_radius": 0.05,
    "chunk_size": 32,
    "n_action_steps": None,
    "dim_model": 512,
    "n_heads": 8,
    "dim_feedforward": 3200,
    "n_encoder_layers": 4,
    "n_decoder_layers": 1,
    "use_vae": True,
    "latent_dim": 32,
    "n_vae_encoder_layers": 4,
    "kl_weight": 10.0,
    "dropout": 0.1,
    "pre_norm": False,
    "feedforward_activation": "relu",
    "temporal_ensemble_coeff": None,
    "optimizer_lr": 1e-4,
    "optimizer_weight_decay": 1e-4,
    "device": "cpu",
}


def _resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    unknown = set(config) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown PointACTPolicy config keys: {sorted(unknown)}")
    resolved = copy.deepcopy(_DEFAULTS)
    resolved.update(copy.deepcopy(config))
    if resolved["mode"] not in {"anchored", "unanchored", "no_tactile"}:
        raise ValueError("mode must be anchored, unanchored, or no_tactile")
    if resolved["point_feature_dim"] != 8:
        raise ValueError("Point channels must be [x, y, z, r, g, b, pressure, is_tactile]")
    positive_keys = (
        "state_dim", "action_dim", "tactile_dim", "n_visual_points", "encoder_dim",
        "tactile_hidden_dim", "chunk_size", "dim_model", "n_heads", "dim_feedforward",
        "n_encoder_layers", "n_decoder_layers", "latent_dim", "n_vae_encoder_layers",
    )
    for key in positive_keys:
        if not isinstance(resolved[key], int) or isinstance(resolved[key], bool) or resolved[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    hidden = resolved["point_hidden_dims"]
    if not hidden or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in hidden):
        raise ValueError("point_hidden_dims must contain positive integers")
    resolved["point_hidden_dims"] = list(hidden)
    if resolved["local_radius"] <= 0:
        raise ValueError("local_radius must be positive (in metres)")
    if resolved["n_action_steps"] is None:
        resolved["n_action_steps"] = resolved["chunk_size"]
    steps = resolved["n_action_steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= resolved["chunk_size"]:
        raise ValueError("n_action_steps must be between 1 and chunk_size")
    if resolved["dim_model"] % resolved["n_heads"]:
        raise ValueError("dim_model must be divisible by n_heads")
    # The saved config must be portable JSON, including when given a torch.device.
    resolved["device"] = str(resolved["device"])
    return resolved


def _mlp(dimensions: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:], strict=True):
        layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
    return nn.Sequential(*layers)


class SpatialPointEncoder(nn.Module):
    """Shared point MLP, local visual/touch fusion, and global pressure branch.

    Points have fixed slots: visual points first, followed by one point for each
    tactile sensor in sensor-ID order. The modality flag is metadata; it never
    changes point counts. All modes instantiate exactly the same parameters.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = _resolve_config(config)
        cfg = self.config
        hidden = cfg["point_hidden_dims"][-1]
        self.point_mlp = _mlp([8, *cfg["point_hidden_dims"]])
        self.contact_fusion = _mlp([2 * hidden, hidden])
        self.tactile_mlp = _mlp([cfg["tactile_dim"], cfg["tactile_hidden_dim"]])
        self.output_mlp = nn.Sequential(
            nn.Linear(4 * hidden + cfg["tactile_hidden_dim"], cfg["encoder_dim"]),
            nn.LayerNorm(cfg["encoder_dim"]),
            nn.ReLU(),
        )

    @staticmethod
    def _pool(features: Tensor) -> Tensor:
        return torch.cat((features.amax(dim=1), features.mean(dim=1)), dim=-1)

    def _local_visual_features(self, visual_xyz: Tensor, contact_xyz: Tensor, features: Tensor) -> Tensor:
        """Gaussian neighborhood pooling, with zero output for an empty region."""
        # Float32 distances avoid underflow/overflow during mixed precision.
        delta = contact_xyz.float().unsqueeze(2) - visual_xyz.float().unsqueeze(1)
        distance_sq = delta.square().sum(dim=-1)
        radius_sq = self.config["local_radius"] ** 2
        weights = torch.exp(-distance_sq / (2 * radius_sq))
        weights = weights * (distance_sq <= 9 * radius_sq)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.bmm(weights.to(features.dtype), features)

    def forward(self, points: Tensor, tactile: Tensor | None = None) -> Tensor:
        cfg = self.config
        expected_points = cfg["n_visual_points"] + cfg["tactile_dim"]
        if points.ndim != 3 or points.shape[1:] != (expected_points, 8):
            raise ValueError(f"observation.points must have shape [B, {expected_points}, 8]")
        if tactile is not None and tactile.shape != (points.shape[0], cfg["tactile_dim"]):
            raise ValueError(f"observation.tactile must have shape [B, {cfg['tactile_dim']}]")

        n_visual = cfg["n_visual_points"]
        # Fixed visual slots and forcibly zeroed pressure/modality prevent the
        # no-touch ablation from observing tactile values or sensor-dependent
        # point counts. Never mutate the caller's cache tensors.
        visual = torch.cat((points[:, :n_visual, :6], points.new_zeros(points.shape[0], n_visual, 2)), dim=-1)
        visual_features = self.point_mlp(visual)
        visual_summary = self._pool(visual_features)

        if cfg["mode"] == "no_tactile":
            pressure = points.new_zeros(points.shape[0], cfg["tactile_dim"])
        elif tactile is not None:
            pressure = tactile
        else:
            pressure = points[:, n_visual:, 6]

        if cfg["mode"] == "anchored":
            contact_xyz = points[:, n_visual:, :3]
            contact_points = torch.cat(
                (
                    contact_xyz,
                    torch.zeros_like(contact_xyz),
                    pressure.unsqueeze(-1),
                    torch.ones_like(pressure).unsqueeze(-1),
                ),
                dim=-1,
            )
            contact_features = self.point_mlp(contact_points)
            local_features = (
                self._local_visual_features(visual[..., :3], contact_xyz, visual_features)
                if cfg["use_local_pooling"] else torch.zeros_like(contact_features)
            )
            contact_summary = self._pool(self.contact_fusion(torch.cat((contact_features, local_features), dim=-1)))
        else:
            # Do not read tactile XYZ, RGB, modality, or variable point counts.
            contact_summary = torch.zeros_like(visual_summary)

        return self.output_mlp(torch.cat((visual_summary, contact_summary, self.tactile_mlp(pressure)), dim=-1))


class PointACTPolicy(nn.Module):
    """Trainable point encoder feeding LeRobot ACT's environment-state input.

    Required observations: ``observation.state`` [B, state_dim] and
    ``observation.points`` [B, n_visual_points + tactile_dim, 8]. Optional
    ``observation.tactile`` [B, tactile_dim] overrides cached point pressure.
    Training additionally uses ``action`` [B, chunk_size, action_dim] and
    ``action_is_pad`` [B, chunk_size]. Normalization belongs to the caller.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = _resolve_config(config)
        # Keep LeRobot an instantiation-time dependency so geometry/encoder
        # tooling can be imported without the ACT optional dependencies.
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        cfg = self.config
        self.point_encoder = SpatialPointEncoder(cfg)
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
        if state.ndim != 2 or state.shape[1] != self.config["state_dim"]:
            raise ValueError(f"observation.state must have shape [B, {self.config['state_dim']}]")
        points = batch["observation.points"]
        if points.shape[0] != state.shape[0]:
            raise ValueError("State and point batch sizes must match")
        result = {
            "observation.state": state,
            "observation.environment_state": self.point_encoder(points, batch.get("observation.tactile")),
        }
        if training:
            action = batch["action"]
            expected = (state.shape[0], self.config["chunk_size"], self.config["action_dim"])
            if action.shape != expected:
                raise ValueError(f"action must have shape {expected}")
            padding = batch.get("action_is_pad")
            if padding is None:
                padding = torch.zeros(action.shape[:2], dtype=torch.bool, device=action.device)
            if padding.shape != action.shape[:2] or padding.dtype != torch.bool:
                raise ValueError("action_is_pad must be bool with shape [B, chunk_size]")
            result.update({"action": action, "action_is_pad": padding})
        return result

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        return self.act(self._act_batch(batch, training=True))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        return self.act.predict_action_chunk(self._act_batch(batch))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        # ACT owns its action queue and optional temporal ensemble. Avoid point
        # encoding when a cached action can be consumed without a fresh query.
        if self.config["temporal_ensemble_coeff"] is None and self.act._action_queue:
            return self.act.select_action({})
        return self.act.select_action(self._act_batch(batch))

    def reset(self) -> None:
        self.act.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        # ACT has no image backbone here; all trainable parameters share the LR.
        return [{"params": [parameter for parameter in self.parameters() if parameter.requires_grad]}]

    def save_pretrained(self, directory: str | Path) -> None:
        """Save a portable JSON configuration and plain tensor state dictionary."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(self.config)
        config["device"] = "cpu"
        (directory / "config.json").write_text(
            json.dumps({"format_version": 1, "policy_type": "point_act", "config": config}, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save({name: value.detach().cpu() for name, value in self.state_dict().items()}, directory / "model.pt")

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, map_location: str | torch.device = "cpu") -> PointACTPolicy:
        directory = Path(directory)
        payload = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or payload.get("policy_type") != "point_act":
            raise ValueError("Unsupported PointACTPolicy checkpoint format")
        policy = cls(payload["config"])
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        policy.load_state_dict(state, strict=True)
        policy.to(map_location)
        policy.eval()
        policy.reset()
        return policy
