import os

import pytest
import torch

from gradflow import (
    DvebArtifact,
    DvebDeviceContext,
    Solver,
    periodic_vortex,
)

MANIFEST = os.environ.get("GRADFLOW_TEST_DVEB_DEVICE_MANIFEST")

pytestmark = pytest.mark.skipif(
    MANIFEST is None or not torch.cuda.is_available(),
    reason="set GRADFLOW_TEST_DVEB_DEVICE_MANIFEST on a CUDA host",
)


def _expected(state: torch.Tensor, intervals: int, steps: int) -> torch.Tensor:
    solver = Solver(
        equations="euler",
        dimension=3,
        weno=("JS", 5),
        flux_split="global_lf",
        boundaries="periodic_duplicated",
        dtype=torch.float32,
        spacing=(10.0 / intervals,) * 3,
    )
    return solver.run(state, steps=steps, backend="pytorch-eager")


@pytest.mark.parametrize(("intervals", "steps"), [(6, 1), (6, 10), (32, 1)])
def test_device_context_matches_direct_pytorch(intervals: int, steps: int) -> None:
    assert MANIFEST is not None
    artifact = DvebArtifact.from_manifest(MANIFEST)
    state_cpu, _ = periodic_vortex((intervals,) * 3, dtype=torch.float32)
    state = state_cpu.cuda().contiguous()
    expected = _expected(state.clone(), intervals, steps)
    with DvebDeviceContext(artifact, intervals) as context:
        actual = context.run(state, steps=steps).state
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-5)


def test_device_context_supports_current_stream_and_exact_alias() -> None:
    assert MANIFEST is not None
    intervals = 6
    artifact = DvebArtifact.from_manifest(MANIFEST)
    state_cpu, _ = periodic_vortex((intervals,) * 3, dtype=torch.float32)
    original = state_cpu.cuda().contiguous()
    expected = _expected(original.clone(), intervals, 1)
    stream = torch.cuda.Stream()
    with DvebDeviceContext(artifact, intervals) as context, torch.cuda.stream(stream):
        result = context.run(original, steps=1, out=original)
    assert result.state.data_ptr() == original.data_ptr()
    torch.testing.assert_close(original, expected, rtol=0.0, atol=2.0e-5)


def test_device_context_refuses_hidden_conversion() -> None:
    assert MANIFEST is not None
    artifact = DvebArtifact.from_manifest(MANIFEST)
    state, _ = periodic_vortex((6,) * 3, dtype=torch.float32)
    with DvebDeviceContext(artifact, 6) as context:
        with pytest.raises(ValueError, match="CUDA float32"):
            context.run(state, steps=1)
