"""The follow-up visual input is the original probe's exact pre-readout tensor."""
import sys
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from simple_token_runtime import SimpleVisualRuntime
from test_token_runtime import runtime_case


def test_simple_runtime_equals_existing_visual_readout_input_and_ignores_truth(runtime_case):
    base, _, raw = runtime_case
    runtime = SimpleVisualRuntime.__new__(SimpleVisualRuntime)
    runtime.base, runtime.device = base, "cpu"
    captured = []
    handle = base.probes["visual"].readout.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().numpy().copy()))
    expected = base.encode(raw["rgb"], raw["state"], raw["pressure"], raw["anchors"], raw["known_cameras"])["visual"]
    handle.remove()
    combined = runtime.encode(raw["rgb"], raw["state"])
    np.testing.assert_array_equal(combined, captured[0])
    actual = base.probes["visual"].readout(torch.from_numpy(combined)).detach().numpy()
    np.testing.assert_array_equal(actual, expected)
    observation = {"rgb": raw["rgb"][0], "state": raw["state"][0]}
    np.testing.assert_allclose(runtime.encode_observation(observation), combined[0], rtol=1e-5, atol=2e-7)
    observation.update(object_center=np.full(3, np.nan), tactile=np.full(44, np.nan),
                       anchors_world=np.full((44, 3), np.nan), known_cameras_cv=np.full((2, 4, 4), np.nan))
    np.testing.assert_allclose(runtime.encode_observation(observation), combined[0], rtol=1e-5, atol=2e-7)


def test_simple_runtime_hook_cleanup_on_failure(runtime_case):
    base, _, raw = runtime_case
    runtime = SimpleVisualRuntime.__new__(SimpleVisualRuntime)
    runtime.base, runtime.device = base, "cpu"
    base.backbone.aggregator.fail = True
    with pytest.raises(RuntimeError, match="intentional prefix failure"):
        runtime.encode(raw["rgb"], raw["state"])
    assert not base.backbone.aggregator.global_blocks[-1]._forward_pre_hooks
