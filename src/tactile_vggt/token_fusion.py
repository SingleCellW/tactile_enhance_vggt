"""A small tactile adapter around VGGT's real, frozen last global block.

Cache the *original* aggregator tokens after frame block 23 and immediately
before global block 23. The cache contains ``[B, S, P, C]`` tokens and matching
RoPE positions; the original VGGT has five special tokens per image. Pass the
already loaded ``aggregator.global_blocks[-1]`` to :class:`TokenFusionProbe`.
This module neither imports VGGT nor substitutes a new block for its weights.

Anchor XYZ must use the same first-camera coordinate frame and per-observation
scale as the visual geometry (for example, baseline units). Pressure scaling
belongs to the caller. The regression target can be either object center or
object-relative-to-palm XYZ, provided the caller uses one consistent convention.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TactileTokenAdapter(nn.Module):
    """One token per sensor, cross-attended by image patches only.

    All ablations instantiate the same parameters. ``use_xyz=False`` does not
    read anchor values; ``use_sensor_id=False`` does not read sensor IDs.
    ``gate=0`` is an exact identity, including the original special tokens.
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int = 128,
        num_sensors: int = 44,
        num_heads: int = 4,
        use_xyz: bool = True,
        use_sensor_id: bool = True,
        initial_gate: float = 0.0,
    ) -> None:
        super().__init__()
        if min(dim, hidden_dim, num_sensors, num_heads) < 1 or hidden_dim % num_heads:
            raise ValueError("Dimensions must be positive and hidden_dim divisible by num_heads")
        self.dim = dim
        self.num_sensors = num_sensors
        self.use_xyz = use_xyz
        self.use_sensor_id = use_sensor_id
        self.pressure_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.xyz_encoder = nn.Linear(3, hidden_dim)
        self.sensor_embedding = nn.Embedding(num_sensors, hidden_dim)
        self.query_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, hidden_dim)
        self.tactile_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, dim)
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(
        self,
        tokens: Tensor,
        pressure: Tensor,
        anchors: Tensor | None,
        *,
        pressure_flags: Tensor | None = None,
        sensor_ids: Tensor | None = None,
        patch_start: int = 5,
    ) -> Tensor:
        if tokens.ndim != 4 or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must have shape [B, S, P, {self.dim}]")
        batch_size, views, count, _ = tokens.shape
        if not isinstance(patch_start, int) or not 0 <= patch_start < count:
            raise ValueError("patch_start must leave at least one image patch per view")
        sensor_shape = (batch_size, self.num_sensors)
        if pressure.shape != sensor_shape:
            raise ValueError(f"pressure must have shape {sensor_shape}")
        if self.use_xyz and (anchors is None or anchors.shape != (*sensor_shape, 3)):
            raise ValueError(f"anchors must have shape {(*sensor_shape, 3)}")
        if pressure_flags is None:
            # Supply explicit flags when pressure has been standardized around
            # a nonzero no-contact baseline. Flags also support sensor dropout.
            pressure_flags = pressure > 0
        elif pressure_flags.shape != sensor_shape:
            raise ValueError(f"pressure_flags must have shape {sensor_shape}")

        weight = self.pressure_encoder[0].weight
        pressure = pressure.to(device=tokens.device, dtype=weight.dtype)
        flags = pressure_flags.to(device=tokens.device, dtype=weight.dtype)
        active = flags > 0
        tactile = self.pressure_encoder(torch.stack((pressure, flags), dim=-1))
        if self.use_xyz:
            tactile = tactile + self.xyz_encoder(anchors.to(device=tokens.device, dtype=weight.dtype))
        if self.use_sensor_id:
            if sensor_ids is None:
                sensor_ids = torch.arange(self.num_sensors, device=tokens.device)
            if sensor_ids.shape == (self.num_sensors,):
                sensor_ids = sensor_ids.unsqueeze(0).expand(batch_size, -1)
            if sensor_ids.shape != sensor_shape:
                raise ValueError(f"sensor_ids must have shape [{self.num_sensors}] or {sensor_shape}")
            if sensor_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("sensor_ids must contain integer embedding indices")
            tactile = tactile + self.sensor_embedding(sensor_ids.to(device=tokens.device))
        tactile = self.tactile_norm(tactile)

        patches = tokens[:, :, patch_start:].reshape(batch_size, -1, self.dim)
        queries = self.query_projection(self.query_norm(patches.to(weight.dtype)))
        # MHA produces NaNs for an all-masked sample. Temporarily expose one
        # sensor for such rows, then explicitly zero the complete residual.
        has_contact = active.any(dim=1)
        safe_active = active.clone()
        safe_active[:, 0] = safe_active[:, 0] | ~has_contact
        attended, _ = self.attention(
            queries, tactile, tactile, key_padding_mask=~safe_active, need_weights=False
        )
        residual = self.output_projection(attended)
        residual = residual * has_contact[:, None, None].to(residual.dtype)
        residual = (self.gate.tanh() * residual).to(tokens.dtype)
        patched = (patches + residual).reshape(batch_size, views, count - patch_start, self.dim)
        return torch.cat((tokens[:, :, :patch_start], patched), dim=2)


class TokenFusionProbe(nn.Module):
    """Train only a tactile adapter and readout around a frozen VGGT block.

    ``internal`` injects touch before the actual last global block; ``late``
    injects the same adapter after that block. ``visual`` omits touch entirely.
    ``no_xyz`` is shorthand for internal fusion with ``use_xyz=False``. ``state``
    skips the frozen block and uses proprioception alone. ``tactile`` skips the
    frozen block and applies the same adapter to zero visual queries/tokens;
    only touch and proprioception can influence its prediction. It has the same
    parameter layout but a different compute path. Every mode has the same
    state encoder and readout; normalize state using training statistics.

    ``forward`` returns ``prediction`` [B, 3], ``features`` [B, feature_dim]
    (suitable for ACT's environment-state input), and ``tokens`` [B, S, P, C].
    A zero-initialized gate first receives gradients itself; use a small nonzero
    ``initial_gate`` to test gradients into tactile values on the first step.
    """

    def __init__(
        self,
        frozen_global_block: nn.Module,
        dim: int = 1024,
        hidden_dim: int = 128,
        feature_dim: int = 256,
        num_sensors: int = 44,
        num_heads: int = 4,
        state_dim: int = 22,
        state_hidden_dim: int = 64,
        mode: str = "internal",
        use_xyz: bool = True,
        use_sensor_id: bool = True,
        initial_gate: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"internal", "late", "visual", "no_xyz", "state", "tactile"}:
            raise ValueError("mode must be internal, late, visual, no_xyz, state, or tactile")
        if min(feature_dim, state_dim, state_hidden_dim) < 1:
            raise ValueError("feature_dim and state dimensions must be positive")
        self.mode = mode
        self.dim = dim
        self.state_dim = state_dim
        self.frozen_global_block = frozen_global_block.requires_grad_(False).eval()
        self.adapter = TactileTokenAdapter(
            dim=dim,
            hidden_dim=hidden_dim,
            num_sensors=num_sensors,
            num_heads=num_heads,
            use_xyz=use_xyz and mode != "no_xyz",
            use_sensor_id=use_sensor_id,
            initial_gate=initial_gate,
        )
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, state_hidden_dim), nn.GELU())
        combined_dim = dim + state_hidden_dim
        self.readout = nn.Sequential(
            nn.LayerNorm(combined_dim), nn.Linear(combined_dim, feature_dim), nn.GELU()
        )
        self.prediction_head = nn.Linear(feature_dim, 3)

    def train(self, mode: bool = True) -> TokenFusionProbe:
        super().train(mode)
        # Keep the pretrained block deterministic while permitting gradients
        # through its operations into the upstream adapter (no torch.no_grad).
        self.frozen_global_block.eval()
        return self

    def forward(
        self,
        prefix_tokens: Tensor,
        pos: Tensor | None,
        pressure: Tensor | None = None,
        anchors: Tensor | None = None,
        *,
        pressure_flags: Tensor | None = None,
        sensor_ids: Tensor | None = None,
        patch_start: int = 5,
        state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if prefix_tokens.ndim != 4 or prefix_tokens.shape[-1] != self.dim:
            raise ValueError(f"prefix_tokens must have shape [B, S, P, {self.dim}]")
        batch_size, views, count, _ = prefix_tokens.shape
        if not isinstance(patch_start, int) or not 0 <= patch_start < count:
            raise ValueError("patch_start must leave at least one image patch per view")
        if pos is not None and pos.shape != (batch_size, views, count, 2):
            raise ValueError("pos must have shape [B, S, P, 2]")
        if self.mode not in {"visual", "state"} and pressure is None:
            raise ValueError("pressure is required for tactile fusion")
        if state is not None and state.shape != (batch_size, self.state_dim):
            raise ValueError(f"state must have shape [B, {self.state_dim}]")

        def fuse(tokens: Tensor) -> Tensor:
            return self.adapter(
                tokens, pressure, anchors, pressure_flags=pressure_flags,
                sensor_ids=sensor_ids, patch_start=patch_start,
            )

        if self.mode == "tactile":
            # Shape/device/dtype come from the cache, never its RGB values.
            # Zero queries make this a tactile+state control using the existing
            # adapter/readout parameters, rather than another visual encoder.
            tokens = fuse(torch.zeros_like(prefix_tokens))
        else:
            tokens = fuse(prefix_tokens) if self.mode in {"internal", "no_xyz"} else prefix_tokens
        if self.mode not in {"state", "tactile"}:
            flat_pos = None if pos is None else pos.reshape(batch_size, views * count, 2)
            tokens = self.frozen_global_block(tokens.reshape(batch_size, views * count, self.dim), pos=flat_pos)
            tokens = tokens.reshape(batch_size, views, count, self.dim)
        if self.mode == "late":
            tokens = fuse(tokens)
        # Pool image patches only. Camera and register slots retain their exact
        # indices for callers wishing to inspect them or reuse pretrained heads.
        pooled = (
            tokens.new_zeros(batch_size, self.dim)
            if self.mode == "state" else tokens[:, :, patch_start:].mean(dim=(1, 2))
        )
        state_weight = self.state_encoder[0].weight
        if state is None:
            state = state_weight.new_zeros(batch_size, self.state_dim)
        state_features = self.state_encoder(state.to(device=tokens.device, dtype=state_weight.dtype))
        combined = torch.cat((pooled.to(state_features.dtype), state_features), dim=-1)
        features = self.readout(combined)
        return {"prediction": self.prediction_head(features), "features": features, "tokens": tokens}
