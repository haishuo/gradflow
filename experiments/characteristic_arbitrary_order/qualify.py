#!/usr/bin/env python3
"""Record the frozen characteristic arbitrary-order WENO-JS qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from gradflow import (  # noqa: E402
    EULER_GAMMA,
    QUALIFIED_EULER_WENO_ORDERS,
    Solver,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    euler_weno_rhs,
    generate_weno_js_coefficients,
    periodic_vortex,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_payload(order: int) -> dict[str, object]:
    coefficients = generate_weno_js_coefficients(order)

    def vector(values) -> list[str]:
        return [str(value) for value in values]

    return {
        "order": order,
        "substencil_width": coefficients.substencil_width,
        "candidate_offsets": coefficients.candidate_offsets,
        "candidate_coefficients": [
            vector(values) for values in coefficients.candidate_coefficients
        ],
        "optimal_weights": vector(coefficients.optimal_weights),
        "full_offsets": coefficients.full_offsets,
        "full_coefficients": vector(coefficients.full_coefficients),
        "smoothness_matrices": [
            [vector(row) for row in matrix]
            for matrix in coefficients.smoothness_matrices
        ],
        "smoothness_factors": [
            [[str(weight), vector(values)] for weight, values in factors]
            for factors in coefficients.smoothness_factors
        ],
    }


def historical_module():
    path = ROOT / "experiments/shu_torch_ablation/shu_euler_torch.py"
    spec = importlib.util.spec_from_file_location("historical_shu_euler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entropy_wave(
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


def weno5_preservation(dtype: torch.dtype) -> dict[str, float]:
    historical = historical_module()
    state, spacing = periodic_vortex((5, 5, 5), dtype=dtype)
    old_state, old_spacing = historical.periodic_vortex((5, 5, 5), dtype=dtype)
    rhs = euler_weno_rhs(state, spacing, order=5)
    old_rhs = historical.euler_weno5_rhs(old_state, old_spacing)
    dt = euler_cfl_timestep(state, spacing, 0.1)
    old_dt = historical.cfl_timestep(old_state, old_spacing, 0.1)
    step = euler_ssp_rk3_step(state, spacing, dt, order=5)
    old_step = historical.ssp_rk3_step(old_state, old_spacing, old_dt)
    return {
        "rhs_maximum_absolute_difference": float(torch.max(torch.abs(rhs - old_rhs))),
        "step_maximum_absolute_difference": float(
            torch.max(torch.abs(step - old_step))
        ),
    }


def convergence(order: int) -> dict[str, object]:
    sizes = (24, 36, 54, 81)
    errors = []
    for intervals in sizes:
        state, spacing, exact_line = entropy_wave(intervals)
        actual = euler_weno_rhs(state, spacing, order=order)[:, 0, 0, :-1]
        errors.append(
            float(torch.sqrt(torch.mean((actual - exact_line[:, :-1]).square())))
        )
    rates = [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors, errors[1:], sizes, sizes[1:]
        )
    ]
    return {"sizes": sizes, "l2_errors": errors, "rates": rates}


def uniform_error(order: int, dtype: torch.dtype) -> float:
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
    rhs = euler_weno_rhs(state, (1.0 / intervals,) * 3, order=order)
    return float(torch.max(torch.abs(rhs)))


def conservation(order: int) -> dict[str, object]:
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
    return {
        "maximum_absolute_component_sum": float(torch.max(residual)),
        "maximum_roundoff_ratio": float(torch.max(residual / scale)),
    }


def device_agreement(order: int, dtype: torch.dtype) -> float | None:
    if not torch.cuda.is_available():
        return None
    intervals = max(17, order)
    state, spacing, _ = entropy_wave(
        intervals, transverse=max(15, order), dtype=dtype
    )
    expected = euler_weno_rhs(state, spacing, order=order)
    actual = euler_weno_rhs(state.cuda(), spacing, order=order).cpu()
    return float(torch.max(torch.abs(actual - expected)))


def finite_gradient(order: int) -> dict[str, object]:
    state, spacing = periodic_vortex(
        (order, order, order), dtype=torch.float64
    )
    state.requires_grad_()
    solver = Solver(
        equations="euler",
        dimension=3,
        weno=("JS", order),
        flux_split="global_lf",
        boundaries="periodic_duplicated",
        dtype=torch.float64,
        spacing=spacing,
        backend="pytorch",
    )
    result = solver.run(state, steps=1)
    result.square().mean().backward()
    assert state.grad is not None
    return {
        "finite": bool(torch.all(torch.isfinite(state.grad))),
        "nonzero_count": int(torch.count_nonzero(state.grad)),
        "maximum_absolute_gradient": float(torch.max(torch.abs(state.grad))),
    }


def compile_gate(order: int, device: str) -> dict[str, object] | None:
    if device == "cuda" and not torch.cuda.is_available():
        return None
    dtype = torch.float32 if device == "cuda" else torch.float64
    state, spacing, _ = entropy_wave(
        max(15, order), transverse=max(15, order), dtype=dtype, device=device
    )

    def call(values: torch.Tensor) -> torch.Tensor:
        return euler_weno_rhs(values, spacing, order=order)

    expected = call(state)
    explanation = torch._dynamo.explain(call)(state)
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    actual = compiled(state)
    return {
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "maximum_absolute_error": float(torch.max(torch.abs(actual - expected))),
        "finite": bool(torch.all(torch.isfinite(actual))),
    }


def git(command: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *command.split()],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")

    payloads = {
        str(order): exact_payload(order)
        for order in QUALIFIED_EULER_WENO_ORDERS
    }
    canonical = json.dumps(payloads, sort_keys=True, separators=(",", ":")).encode()
    per_order = {}
    for order in QUALIFIED_EULER_WENO_ORDERS:
        per_order[str(order)] = {
            "convergence": convergence(order),
            "uniform_max_abs_float32": uniform_error(order, torch.float32),
            "uniform_max_abs_float64": uniform_error(order, torch.float64),
            "conservation": conservation(order),
            "cpu_cuda_max_abs_float32": device_agreement(order, torch.float32),
            "cpu_cuda_max_abs_float64": device_agreement(order, torch.float64),
        }

    source_paths = [
        ROOT / "src/gradflow/euler3d.py",
        ROOT / "src/gradflow/solver.py",
        ROOT / "src/gradflow/weno_js.py",
        ROOT / "src/gradflow/weno_js_coefficients.py",
        ROOT / "tests/test_euler_arbitrary_order.py",
        Path(__file__),
    ]
    report = {
        "schema": "gradflow-characteristic-arbitrary-order-qualification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(
            (ROOT / "docs/CHARACTERISTIC_ARBITRARY_ORDER_PROTOCOL.md").resolve()
        ),
        "gradflow_commit": git("rev-parse HEAD"),
        "gradflow_dirty_before_record": bool(git("status --porcelain")),
        "qualified_orders": QUALIFIED_EULER_WENO_ORDERS,
        "exact_coefficient_payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "weno5_preservation": {
            "float32": weno5_preservation(torch.float32),
            "float64": weno5_preservation(torch.float64),
        },
        "orders": per_order,
        "autograd": {
            str(order): finite_gradient(order) for order in (5, 11, 15)
        },
        "torch_compile": {
            str(order): {
                device: compile_gate(order, device) for device in ("cpu", "cuda")
            }
            for order in (5, 11, 15)
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "claim_boundary": {
            "performance_measured": False,
            "dveb_changed": False,
            "boundaries": "duplicated periodic endpoints only",
            "equations": "three-dimensional ideal-gas compressible Euler only",
            "solver_orders_beyond_15_are_qualified": False,
            "native_backend_orders": [5],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "sha256": sha256(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
