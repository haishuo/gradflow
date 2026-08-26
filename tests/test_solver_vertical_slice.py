import ast
import importlib.util
import inspect
from pathlib import Path
import textwrap

import pytest
import torch

import gradflow.euler3d as implementation
from gradflow import (
    BackendUnavailableError,
    Solver,
    UnsupportedProblemError,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    periodic_vortex,
)


ROOT = Path(__file__).resolve().parents[1]


def _solver(**overrides: object) -> Solver:
    arguments: dict[str, object] = {
        "equations": "euler",
        "dimension": 3,
        "weno": ("JS", 5),
        "flux_split": "global_lf",
        "boundaries": "periodic_duplicated",
        "dtype": torch.float32,
        "spacing": (2.5, 2.5, 2.5),
    }
    arguments.update(overrides)
    return Solver(**arguments)  # type: ignore[arg-type]


def _historical_module():
    path = ROOT / "experiments" / "shu_torch_ablation" / "shu_euler_torch.py"
    spec = importlib.util.spec_from_file_location("historical_shu_euler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packaged_step_agrees_with_bakeoff_authority() -> None:
    historical = _historical_module()
    state, spacing = periodic_vortex((4, 4, 4))
    historical_state, historical_spacing = historical.periodic_vortex((4, 4, 4))
    dt = euler_cfl_timestep(state, spacing, 0.1)
    historical_dt = historical.cfl_timestep(
        historical_state, historical_spacing, 0.1
    )
    result = euler_ssp_rk3_step(state, spacing, dt)
    historical_result = historical.ssp_rk3_step(
        historical_state, historical_spacing, historical_dt
    )
    assert torch.equal(state, historical_state)
    assert torch.equal(dt, historical_dt)
    torch.testing.assert_close(result, historical_result, rtol=0.0, atol=2.0e-6)


def test_solver_fixed_step_matches_scientific_function() -> None:
    state, spacing = periodic_vortex((4, 4, 4))
    solver = _solver(spacing=spacing)
    result = solver.run(state, steps=1)
    expected = euler_ssp_rk3_step(
        state, spacing, euler_cfl_timestep(state, spacing, 0.1)
    )
    assert torch.equal(result, expected)
    assert solver.last_run is not None
    assert solver.last_run.backend.selected == "pytorch-eager"
    assert solver.last_run.backend.device == "cpu"
    assert solver.last_run.steps == 1
    assert solver.last_run.hidden_device_transfers == 0
    assert solver.last_run.validation_device_synchronizations == 0


def test_cpu_final_time_lands_on_requested_time() -> None:
    state, spacing = periodic_vortex((4, 4, 4))
    solver = _solver(spacing=spacing)
    result = solver.run(state, final_time=0.01)
    assert result.shape == state.shape
    assert solver.last_run is not None
    assert solver.last_run.simulated_time == pytest.approx(0.01, abs=1.0e-9)
    assert solver.last_run.steps >= 1


def test_fixed_step_path_is_differentiable() -> None:
    state, spacing = periodic_vortex((4, 4, 4))
    state.requires_grad_()
    result = _solver(spacing=spacing).run(state, steps=1)
    loss = result.square().mean()
    loss.backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert torch.count_nonzero(state.grad) > 0


def test_euler_step_is_one_compile_graph() -> None:
    state, spacing = periodic_vortex((4, 4, 4))

    def step(value: torch.Tensor) -> torch.Tensor:
        dt = euler_cfl_timestep(value, spacing, 0.1)
        return euler_ssp_rk3_step(value, spacing, dt)

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(step)(state)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert not explanation.break_reasons


def test_native_backend_is_rejected_for_arbitrary_state() -> None:
    state, spacing = periodic_vortex((4, 4, 4))
    solver = _solver(spacing=spacing)
    with pytest.raises(BackendUnavailableError, match="no arbitrary-state input ABI"):
        solver.run(state, steps=1, backend="dveb")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"equations": "navier-stokes"}, "viscous terms"),
        ({"dimension": 2}, "3-D only"),
        ({"weno": ("JS", 11)}, "WENO-5"),
        ({"flux_split": "local_lf"}, "global_lf"),
        ({"boundaries": "periodic"}, "periodic_duplicated"),
        ({"dtype": torch.float64}, "float32"),
    ],
)
def test_unsupported_mathematics_fails_explicitly(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(UnsupportedProblemError, match=message):
        _solver(**override)


def test_run_contract_validation() -> None:
    state, spacing = periodic_vortex((4, 4, 4))
    solver = _solver(spacing=spacing)
    with pytest.raises(ValueError, match="exactly one"):
        solver.run(state)
    with pytest.raises(ValueError, match="exactly one"):
        solver.run(state, final_time=0.1, steps=1)
    with pytest.raises(TypeError, match="dtype"):
        solver.run(state.double(), steps=1)
    with pytest.raises(ValueError, match="layout"):
        solver.run(state[0], steps=1)
    with pytest.raises(TypeError, match="not a tensor"):
        solver.run(state, steps=1, cfl=torch.tensor(0.1))  # type: ignore[arg-type]
    invalid = state.clone()
    invalid[0] = -1.0
    with pytest.raises(ValueError, match="positive density and pressure"):
        solver.run(invalid, steps=1)


def test_fixed_step_numerical_path_has_no_transfer_or_scalar_extraction() -> None:
    forbidden = {"item", "cpu", "cuda", "to", "numpy"}
    functions = (
        implementation.synchronize_duplicate_endpoints,
        implementation._flux_and_roe_matrices,
        implementation._nonlinear_flux_correction,
        implementation._line_rhs,
        implementation.euler_weno5_rhs,
        implementation.euler_cfl_timestep,
        implementation.euler_ssp_rk3_step,
    )
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert calls.isdisjoint(forbidden), (function.__name__, calls & forbidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_solver_cpu_cuda_agreement_and_device_preservation() -> None:
    cpu_state, spacing = periodic_vortex((4, 4, 4))
    cpu = _solver(spacing=spacing).run(cpu_state, steps=1)
    cuda_state = cpu_state.cuda()
    cuda_solver = _solver(spacing=spacing)
    cuda = cuda_solver.run(cuda_state, steps=1)
    assert cuda.device == cuda_state.device
    torch.testing.assert_close(cuda.cpu(), cpu, rtol=0.0, atol=2.0e-6)
    assert cuda_solver.last_run is not None
    assert cuda_solver.last_run.hidden_device_transfers == 0
    assert cuda_solver.last_run.validation_device_synchronizations == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_final_time_refuses_hidden_control_transfer() -> None:
    state, spacing = periodic_vortex((4, 4, 4), device="cuda")
    with pytest.raises(UnsupportedProblemError, match="hidden host scalar transfer"):
        _solver(spacing=spacing).run(state, final_time=0.01)
