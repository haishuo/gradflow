import os
from pathlib import Path
import subprocess
import tempfile

import pytest
import torch

from gradflow import (
    DvebArtifact,
    DvebPortableAbi,
    Solver,
    periodic_vortex,
    synchronize_duplicate_endpoints,
)


MANIFEST = os.environ.get("GRADFLOW_TEST_DVEB_MANIFEST")
MODEL = os.environ.get("GRADFLOW_TEST_DVEB_MODEL")
MODEL_SHA256 = os.environ.get("GRADFLOW_TEST_DVEB_MODEL_SHA256")
EXECUTABLE = os.environ.get("GRADFLOW_TEST_DVEB_EXECUTABLE")


def _artifact(*, with_model: bool) -> DvebArtifact:
    assert MANIFEST is not None
    return DvebArtifact.from_manifest(
        MANIFEST,
        model=MODEL if with_model else None,
        verified_model_sha256=MODEL_SHA256 if with_model else None,
    )


def _solver(intervals: int, artifact: DvebArtifact) -> Solver:
    return Solver(
        equations="euler", dimension=3, weno=("JS", 5),
        flux_split="global_lf", boundaries="periodic_duplicated",
        dtype=torch.float32, spacing=(10.0 / intervals,) * 3,
        dveb_artifact=artifact,
    )


def _arbitrary_state(intervals: int) -> torch.Tensor:
    state, _ = periodic_vortex((intervals,) * 3)
    side = intervals + 1
    coordinate = torch.arange(side, dtype=state.dtype) * (2.0 * torch.pi / intervals)
    z, y, x = torch.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    factor = 1.0 + 0.01 * torch.sin(x) * torch.cos(y) * torch.sin(z)
    density = state[0]
    velocity = state[1:4] / density
    kinetic = 0.5 * (state[1:4].square().sum(dim=0) / density)
    pressure = 0.4 * (state[4] - kinetic)
    density = density * factor
    momentum = density.unsqueeze(0) * velocity
    energy = pressure / 0.4 + 0.5 * momentum.square().sum(dim=0) / density
    return synchronize_duplicate_endpoints(
        torch.cat((density.unsqueeze(0), momentum, energy.unsqueeze(0)))
    ).contiguous()


pytestmark = pytest.mark.skipif(
    MANIFEST is None,
    reason="set GRADFLOW_TEST_DVEB_MANIFEST to exercise portable ABI v1",
)


def test_artifact_query_is_hash_qualified() -> None:
    runtime = DvebPortableAbi(_artifact(with_model=False))
    query = runtime.query(6)
    assert query["dimensions"] == 3
    assert query["components"] == 5
    assert query["required_elements"] == 5 * 7**3


@pytest.mark.parametrize(("intervals", "steps"), [(6, 1), (6, 10), (32, 1)])
def test_arbitrary_state_matches_pytorch_cpu_and_cuda(
    intervals: int, steps: int
) -> None:
    artifact = _artifact(with_model=False)
    state = _arbitrary_state(intervals)
    solver = _solver(intervals, artifact)
    expected = solver.run(state, steps=steps, backend="pytorch-eager")
    cpu = solver.run(state, steps=steps, backend="cpu-simd")
    cuda = solver.run(state, steps=steps, backend="cuda-native")
    torch.testing.assert_close(cpu, expected, rtol=0.0, atol=2.0e-5)
    torch.testing.assert_close(cuda, expected, rtol=0.0, atol=2.0e-5)
    torch.testing.assert_close(cuda, cpu, rtol=0.0, atol=2.0e-5)
    assert not torch.equal(cpu, state)


@pytest.mark.skipif(
    MODEL is None or MODEL_SHA256 is None,
    reason="set the verified DVEB placement model and hash",
)
def test_solver_auto_is_bounded_and_falls_back_safely() -> None:
    artifact = _artifact(with_model=True)
    in_range = _arbitrary_state(7)
    solver = _solver(7, artifact)
    result = solver.run(in_range, steps=1)
    assert result.shape == in_range.shape
    assert solver.last_run is not None
    assert solver.last_run.backend.selected in {"dveb-cpu", "dveb-cuda"}

    outside = _arbitrary_state(4)
    solver = _solver(4, artifact)
    result = solver.run(outside, steps=1)
    assert result.shape == outside.shape
    assert solver.last_run is not None
    assert solver.last_run.backend.selected == "pytorch-eager"
    assert "outside bounded calibration range" in solver.last_run.backend.reason


@pytest.mark.skipif(
    EXECUTABLE is None,
    reason="set GRADFLOW_TEST_DVEB_EXECUTABLE to compare the direct runner",
)
@pytest.mark.parametrize(
    ("candidate", "backend"),
    [("cpu_simd[6]", "cpu-simd"), ("cuda", "cuda-native")],
)
def test_abi_matches_unchanged_direct_portable_runner(
    candidate: str, backend: str
) -> None:
    assert EXECUTABLE is not None
    intervals = 6
    state, _ = periodic_vortex((intervals,) * 3)
    solver = _solver(intervals, _artifact(with_model=False))
    abi = solver.run(state.contiguous(), steps=1, backend=backend)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "state.f32"
        environment = os.environ.copy()
        environment["DVEB_CALIBRATION"] = "1"
        process = subprocess.run(
            [EXECUTABLE, "--internal-calibration", "--candidate", candidate,
             "--size", str(intervals), "--steps", "1", "--output", str(output)],
            capture_output=True, text=True, env=environment,
        )
        assert process.returncode == 0, process.stderr
        direct = torch.frombuffer(bytearray(output.read_bytes()), dtype=torch.float32)
        direct = direct.reshape_as(state)
    torch.testing.assert_close(abi, direct, rtol=0.0, atol=2.0e-5)
