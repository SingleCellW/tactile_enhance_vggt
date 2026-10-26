"""Shared helpers for frozen VGGT experiments; importing starts no training."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch

def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vggt(repository, weights, device):
    sys.path.insert(0, str(repository))
    from vggt.models.vggt import VGGT
    model = VGGT(enable_point=False, enable_track=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    result = model.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys if not k.startswith(("point_head.", "track_head."))]
    if result.missing_keys or unexpected:
        raise ValueError((result.missing_keys, unexpected))
    return model.eval().requires_grad_(False).to(device)
