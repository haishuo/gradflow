import importlib.util
import math
from pathlib import Path

import pytest
import torch

from gradflow import (
    EULER_GAMMA,
    QUALIFIED_EULER_WENO_ORDERS,
    BackendUnavailableError,
    Solver,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    euler_weno_rhs,
    periodic_vortex,
)

ROOT = Path(__file__).resolve().parents[1]


def _historical_module():
    path = ROOT / "experiments" / "shu_torch_ablation" / "shu_euler_torch.py"
    spec = importlib.util.spec_from_file_location("historical_shu_euler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entropy_wave(
    intervals: int,
    *,
    transverse: int = 15,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, tuple[float, float, float], torch.Tensor]:
    x = torch.arange(intervals + 1, dtype=dtype, device=device) / intervals
    density_line = (
        1.0
        + 0.1 * torch.sin(4.0 * math.pi * x)
        + 0.03 * torch.cos(6.0 * math.pi * x)
    )
    density_derivative = (
        0.4 * math.pi * torch.cos(4.0 * math.pi * x)
        - 0.18 * math.pi * torch.sin(6.0 * math.pi * x)
    )
    velocity = (0.7, 0.2, -0.1)
    pressure = 1.0
    shape = (transverse + 1, transverse + 1, intervals + 1)
    density = density_line.reshape(1, 1, -1).expand(shape)
    speed_squared = sum(component**2 for component in velocity)
    state = torch.stack(
        (
            density,
            density * velocity[0],
            density * velocity[1],
            density * velocity[2],
            pressure / (EULER_GAMMA - 1.0)
            + 0.5 * density * speed_squared,
        )
    )
    density_rhs = -velocity[0] * density_derivative
    exact_line = torch.stack(
        (
            density_rhs,
            velocity[0] * density_rhs,
            velocity[1] * density_rhs,
            velocity[2] * density_rhs,
            0.5 * speed_squared * density_rhs,
        )
    )
    spacing = (1.0 / intervals, 1.0 / transverse, 1.0 / transverse)
    return state, spacing, exact_line


def _solver(
    order: int,
    spacing: tuple[float, float, float],
    *,
    dtype: torch.dtype,
    dveb_artifact: object | None = None,
) -> Solver:
    return Solver(
        equations="euler",
        dimension=3,
        weno=("JS", order),
        flux_split="global_lf",
        boundaries="periodic_duplicated",
        dtype=dtype,
        spacing=spacing,
        backend="pytorch",
        dveb_artifact=dveb_artifact,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 5.0e-6), (torch.float64, 2.0e-12)],
)
def test_generated_weno5_preserves_historical_rhs_and_step(
    dtype: torch.dtype, atol: float
) -> None:
    historical = _historical_module()
    state, spacing = periodic_vortex((5, 5, 5), dtype=dtype)
    historical_state, historical_spacing = historical.periodic_vortex(
        (5, 5, 5), dtype=dtype
    )
    expected_rhs = historical.euler_weno5_rhs(
        historical_state, historical_spacing
    )
    actual_rhs = euler_weno_rhs(state, spacing, order=5)
    torch.testing.assert_close(actual_rhs, expected_rhs, rtol=0.0, atol=atol)

    dt = euler_cfl_timestep(state, spacing, 0.1)
    historical_dt = historical.cfl_timestep(
        historical_state, historical_spacing, 0.1
    )
    expected_step = historical.ssp_rk3_step(
        historical_state, historical_spacing, historical_dt
    )
    actual_step = euler_ssp_rk3_step(state, spacing, dt, order=5)
    torch.testing.assert_close(actual_step, expected_step, rtol=0.0, atol=atol)


@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
def test_characteristic_entropy_wave_convergence(order: int) -> None:
    sizes = (24, 36, 54, 81)
    errors = []
    for intervals in sizes:
        state, spacing, exact_line = _entropy_wave(intervals)
        actual_line = euler_weno_rhs(
            state, spacing, order=order
        )[:, 0, 0, :-1]
        error = torch.sqrt(torch.mean((actual_line - exact_line[:, :-1]).square()))
        errors.append(float(error))
    rates = [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors, errors[1:], sizes, sizes[1:]
        )
    ]
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert max(rates) >= order - 2.0, f"errors={errors}, rates={rates}"


@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 2.0e-5), (torch.float64, 2.0e-12)],
)
def test_uniform_state_preservation(
    order: int, dtype: torch.dtype, atol: float
) -> None:
    intervals = 15
    density = torch.ones(
        (intervals + 1, intervals + 1, intervals + 1), dtype=dtype
    )
    velocity = (0.3, -0.2, 0.1)
    speed_squared = sum(component**2 for component in velocity)
    state = torch.stack(
        (
            density,
            density * velocity[0],
            density * velocity[1],
            density * velocity[2],
            torch.full_like(
                density,
                1.0 / (EULER_GAMMA - 1.0) + 0.5 * speed_squared,
            ),
        )
    )
    rhs = euler_weno_rhs(
        state, (1.0 / intervals,) * 3, order=order
    )
    torch.testing.assert_close(rhs, torch.zeros_like(rhs), rtol=0.0, atol=atol)


@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
def test_unique_periodic_cells_are_conservative(order: int) -> None:
    intervals = 19
    coordinates = [
        torch.arange(intervals + 1, dtype=torch.float64) / intervals
        for _ in range(3)
    ]
    z, y, x = torch.meshgrid(*coordinates, indexing="ij")
    density = (
        1.0
        + 0.05 * torch.sin(2.0 * math.pi * x)
        + 0.03 * torch.cos(2.0 * math.pi * y)
        + 0.02 * torch.sin(2.0 * math.pi * z)
    )
    velocity = (0.3, -0.2, 0.1)
    speed_squared = sum(component**2 for component in velocity)
    state = torch.stack(
        (
            density,
            density * velocity[0],
            density * velocity[1],
            density * velocity[2],
            1.0 / (EULER_GAMMA - 1.0) + 0.5 * density * speed_squared,
        )
    )
    rhs = euler_weno_rhs(
        state, (1.0 / intervals,) * 3, order=order
    )[:, :-1, :-1, :-1]
    residual = torch.abs(torch.sum(rhs, dim=(1, 2, 3)))
    scale = torch.finfo(rhs.dtype).eps * torch.sum(torch.abs(rhs), dim=(1, 2, 3))
    assert torch.all(residual <= 8.0 * scale), (residual / scale).tolist()


@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
def test_solver_accepts_each_qualified_order(order: int) -> None:
    state, spacing = periodic_vortex(
        (order, order, order), dtype=torch.float64
    )
    result = _solver(order, spacing, dtype=torch.float64).run(state, steps=0)
    assert result.dtype is torch.float64
    assert result.shape == state.shape


@pytest.mark.parametrize("order", [5, 11, 15])
def test_solver_fixed_step_has_finite_gradients(order: int) -> None:
    state, spacing = periodic_vortex(
        (order, order, order), dtype=torch.float64
    )
    state.requires_grad_()
    result = _solver(order, spacing, dtype=torch.float64).run(state, steps=1)
    result.square().mean().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert torch.count_nonzero(state.grad) > 0


def test_native_backend_refuses_unmatched_order_and_dtype() -> None:
    fake_artifact = object()
    state11, spacing11 = periodic_vortex((11, 11, 11))
    solver11 = _solver(
        11, spacing11, dtype=torch.float32, dveb_artifact=fake_artifact
    )
    with pytest.raises(BackendUnavailableError, match="only WENO-5"):
        solver11.run(state11, steps=1, backend="cuda-native")

    state5, spacing5 = periodic_vortex((5, 5, 5), dtype=torch.float64)
    solver5 = _solver(
        5, spacing5, dtype=torch.float64, dveb_artifact=fake_artifact
    )
    with pytest.raises(BackendUnavailableError, match="only float32"):
        solver5.run(state5, steps=1, backend="cuda-native")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 3.0e-4), (torch.float64, 5.0e-11)],
)
def test_characteristic_cpu_cuda_agreement(
    order: int, dtype: torch.dtype, atol: float
) -> None:
    intervals = max(17, order)
    state, spacing, _ = _entropy_wave(
        intervals, transverse=max(15, order), dtype=dtype
    )
    expected = euler_weno_rhs(state, spacing, order=order)
    actual = euler_weno_rhs(state.cuda(), spacing, order=order).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("order", [5, 11, 15])
def test_characteristic_compile_fullgraph_cpu(order: int) -> None:
    state, spacing, _ = _entropy_wave(
        max(17, order), transverse=max(15, order)
    )

    def call(values: torch.Tensor) -> torch.Tensor:
        return euler_weno_rhs(values, spacing, order=order)

    expected = call(state)
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    actual = compiled(state)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-11)
    explanation = torch._dynamo.explain(call)(state)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert not explanation.break_reasons


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("order", [5, 11, 15])
def test_characteristic_compile_fullgraph_cuda(order: int) -> None:
    state, spacing, _ = _entropy_wave(
        max(17, order), transverse=max(15, order), dtype=torch.float32,
        device="cuda",
    )

    def call(values: torch.Tensor) -> torch.Tensor:
        return euler_weno_rhs(values, spacing, order=order)

    expected = call(state)
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    actual = compiled(state)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=3.0e-4)
