#!/usr/bin/env python3
"""Run the frozen G3 qualification matrix for the R6Q CUDA candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gradflow import (  # noqa: E402
    EULER_GAMMA,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    euler_weno5_rhs,
    periodic_vortex,
    synchronize_duplicate_endpoints,
)


STEP_ATOL = 2.0e-5
RHS_ATOL = 5.0e-5
RHS_REL_RMS = 2.0e-5
SENSITIVITY_REL_RMS = 2.0e-2
CONVERGENCE_SIZES = (12, 18, 27, 40)
STEP_CASES = ((6, 1), (6, 10), (32, 1))
EXPECTED_CONTRACT = "r6q_arbitrary_state_rhs_unique_strict_f32_shu_face_once_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def unique_to_duplicate(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    state = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=dtype)
    for axis in range(1, state.ndim):
        state = torch.cat((state, state.narrow(axis, 0, 1)), dim=axis)
    return state.contiguous()


def duplicate_to_unique(state: torch.Tensor) -> np.ndarray:
    interior = (slice(None),) + (slice(0, -1),) * (state.ndim - 1)
    return np.ascontiguousarray(state[interior].detach().cpu().numpy())


def pressure(state: np.ndarray) -> np.ndarray:
    density = state[0].astype(np.float64)
    momentum = state[1:4].astype(np.float64)
    energy = state[4].astype(np.float64)
    kinetic = 0.5 * np.sum(momentum * momentum, axis=0) / density
    return (EULER_GAMMA - 1.0) * (energy - kinetic)


def run_native(
    executable: Path,
    state: np.ndarray,
    *,
    steps: int = 1,
    mode: str = "step",
) -> tuple[np.ndarray, dict[str, Any]]:
    state = np.ascontiguousarray(state, dtype=np.float32)
    n = state.shape[-1]
    expected_shape = (5, n, n, n)
    if state.shape != expected_shape:
        raise ValueError(f"expected {expected_shape}, got {state.shape}")
    with tempfile.TemporaryDirectory(prefix="gradflow-r6q-") as directory:
        input_path = Path(directory) / "input.f32"
        output_path = Path(directory) / "output.f32"
        state.tofile(input_path)
        completed = subprocess.run(
            [
                str(executable),
                "--size",
                str(n),
                "--steps",
                str(steps),
                "--warmups",
                "0",
                "--repetitions",
                "1",
                "--input-state",
                str(input_path),
                "--mode",
                mode,
                "--output-state",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = json.loads(completed.stdout)
        if metadata["contract"] != EXPECTED_CONTRACT:
            raise RuntimeError(f"unexpected contract: {metadata['contract']}")
        result = np.fromfile(output_path, dtype=np.float32).reshape(expected_shape)
    return result, metadata


def oracle_rhs(state: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    duplicate = unique_to_duplicate(state, dtype)
    spacing = (10.0 / state.shape[-1],) * 3
    return duplicate_to_unique(euler_weno5_rhs(duplicate, spacing))


def oracle_step(state: np.ndarray, steps: int, dtype: torch.dtype) -> np.ndarray:
    result = unique_to_duplicate(state, dtype)
    spacing = (10.0 / state.shape[-1],) * 3
    for _ in range(steps):
        dt = euler_cfl_timestep(result, spacing, 0.1)
        result = euler_ssp_rk3_step(result, spacing, dt)
    return duplicate_to_unique(result)


def vortex_state(n: int) -> np.ndarray:
    state, _ = periodic_vortex((n, n, n), dtype=torch.float32)
    return duplicate_to_unique(state)


def perturbed_vortex_state(n: int) -> np.ndarray:
    duplicate, _ = periodic_vortex((n, n, n), dtype=torch.float32)
    coordinate = torch.arange(n + 1, dtype=duplicate.dtype) * (2.0 * torch.pi / n)
    z, y, x = torch.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    factor = 1.0 + 0.01 * torch.sin(x) * torch.cos(y) * torch.sin(z)
    density0 = duplicate[0]
    velocity = duplicate[1:4] / density0
    kinetic0 = 0.5 * duplicate[1:4].square().sum(dim=0) / density0
    p = (EULER_GAMMA - 1.0) * (duplicate[4] - kinetic0)
    density = density0 * factor
    momentum = density.unsqueeze(0) * velocity
    energy = p / (EULER_GAMMA - 1.0) + 0.5 * momentum.square().sum(dim=0) / density
    result = synchronize_duplicate_endpoints(
        torch.cat((density.unsqueeze(0), momentum, energy.unsqueeze(0)))
    )
    return duplicate_to_unique(result)


def entropy_state_and_rhs(n: int, critical: bool) -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.arange(n, dtype=np.float64) * (10.0 / n)
    z, y, x = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    phase = 2.0 * math.pi * x / 10.0
    if critical:
        density = 1.0 + 0.1 * np.sin(phase) ** 3
        density_x = 0.3 * (2.0 * math.pi / 10.0) * np.sin(phase) ** 2 * np.cos(phase)
    else:
        density = 1.0 + 0.1 * np.sin(phase) + 0.03 * np.cos(3.0 * phase)
        density_x = (
            0.1 * (2.0 * math.pi / 10.0) * np.cos(phase)
            - 0.03 * (6.0 * math.pi / 10.0) * np.sin(3.0 * phase)
        )
    velocity = np.array((0.7, 0.2, -0.1), dtype=np.float64)
    p = 1.0
    speed_squared = float(np.dot(velocity, velocity))
    state = np.empty((5, n, n, n), dtype=np.float64)
    state[0] = density
    state[1] = density * velocity[0]
    state[2] = density * velocity[1]
    state[3] = density * velocity[2]
    state[4] = p / (EULER_GAMMA - 1.0) + 0.5 * density * speed_squared
    density_t = -velocity[0] * density_x
    exact = np.empty_like(state)
    exact[0] = density_t
    exact[1] = velocity[0] * density_t
    exact[2] = velocity[1] * density_t
    exact[3] = velocity[2] * density_t
    exact[4] = 0.5 * speed_squared * density_t
    return state.astype(np.float32), exact


def primitive_state(
    density: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray, p: np.ndarray
) -> np.ndarray:
    state = np.empty((5, *density.shape), dtype=np.float32)
    state[0] = density
    state[1] = density * u
    state[2] = density * v
    state[3] = density * w
    state[4] = p / (EULER_GAMMA - 1.0) + 0.5 * density * (u * u + v * v + w * w)
    return state


def dual_sod_state(n: int) -> np.ndarray:
    coordinate = np.arange(n, dtype=np.float32) * (10.0 / n)
    z, y, x = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    outer = (x < 2.5) | (x >= 7.5)
    density = np.where(outer, 1.0, 0.125).astype(np.float32)
    p = np.where(outer, 1.0, 0.1).astype(np.float32)
    zero = np.zeros_like(density)
    return primitive_state(density, zero, zero, zero, p)


def dual_shu_osher_state(n: int) -> np.ndarray:
    coordinate = np.arange(n, dtype=np.float32) * (10.0 / n)
    z, y, x = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    shocked = (x >= 4.0) & (x < 6.0)
    density = np.where(shocked, 3.857143, 1.0 + 0.2 * np.sin(5.0 * x)).astype(np.float32)
    u = np.where(shocked, 2.629369, 0.0).astype(np.float32)
    p = np.where(shocked, 10.33333, 1.0).astype(np.float32)
    zero = np.zeros_like(density)
    return primitive_state(density, u, zero, zero, p)


def comparison(candidate: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    difference = candidate.astype(np.float64) - oracle.astype(np.float64)
    oracle_scale = max(rms(oracle), np.finfo(np.float64).tiny)
    return {
        "linf": float(np.max(np.abs(difference))),
        "rms": rms(difference),
        "relative_rms": rms(difference) / oracle_scale,
    }


def admissibility(state: np.ndarray) -> dict[str, Any]:
    p = pressure(state)
    return {
        "finite": bool(np.isfinite(state).all()),
        "minimum_density": float(np.min(state[0])),
        "minimum_pressure": float(np.min(p)),
        "positive": bool(np.min(state[0]) > 0.0 and np.min(p) > 0.0),
    }


def archive_arrays(directory: Path, name: str, **arrays: np.ndarray) -> dict[str, Any]:
    path = directory / f"{name}.npz"
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def conservation(initial: np.ndarray, final: np.ndarray, steps: int) -> dict[str, Any]:
    initial64 = initial.astype(np.float64)
    final64 = final.astype(np.float64)
    drift = np.sum(final64 - initial64, axis=(1, 2, 3))
    bound = (
        128.0
        * steps
        * np.finfo(np.float32).eps
        * np.sum(np.abs(initial64), axis=(1, 2, 3))
    )
    return {
        "drift": drift.tolist(),
        "absolute_bound": bound.tolist(),
        "passed": bool(np.all(np.abs(drift) <= bound)),
    }


def evaluate_step_case(
    executable: Path,
    arrays: Path,
    family: str,
    n: int,
    steps: int,
    initial: np.ndarray,
) -> dict[str, Any]:
    candidate, metadata = run_native(executable, initial, steps=steps)
    oracle32 = oracle_step(initial, steps, torch.float32)
    oracle64 = oracle_step(initial, steps, torch.float64)
    parity = comparison(candidate, oracle32)
    diagnostic = comparison(candidate, oracle64)
    validity = admissibility(candidate)
    conserved = conservation(initial, candidate, steps)
    row = {
        "family": family,
        "n": n,
        "steps": steps,
        "native": metadata,
        "fp32_parity": parity,
        "fp64_diagnostic": diagnostic,
        "admissibility": validity,
        "conservation": conserved,
        "archive": archive_arrays(
            arrays,
            f"step_{family}_n{n}_s{steps}",
            initial=initial,
            candidate=candidate,
            oracle_fp32=oracle32,
            oracle_fp64=oracle64,
        ),
    }
    row["passed"] = bool(
        parity["linf"] <= STEP_ATOL
        and validity["finite"]
        and validity["positive"]
        and conserved["passed"]
    )
    return row


def rates(errors: list[float], sizes: tuple[int, ...]) -> list[float]:
    return [
        math.log(errors[index - 1] / errors[index])
        / math.log(sizes[index] / sizes[index - 1])
        for index in range(1, len(errors))
    ]


def evaluate_rhs_family(
    executable: Path, arrays: Path, *, critical: bool
) -> dict[str, Any]:
    family = "critical_entropy" if critical else "smooth_entropy"
    rows = []
    errors = []
    parity_passed = True
    for n in CONVERGENCE_SIZES:
        initial, exact = entropy_state_and_rhs(n, critical)
        candidate, metadata = run_native(executable, initial, mode="rhs")
        oracle32 = oracle_rhs(initial, torch.float32)
        oracle64 = oracle_rhs(initial, torch.float64)
        parity = comparison(candidate, oracle32)
        error = rms(candidate.astype(np.float64) - exact)
        errors.append(error)
        point_error = rms(candidate[:, :, :, 0].astype(np.float64) - exact[:, :, :, 0])
        row_passed = parity["linf"] <= RHS_ATOL and parity["relative_rms"] <= RHS_REL_RMS
        parity_passed = parity_passed and row_passed
        rows.append(
            {
                "n": n,
                "native": metadata,
                "fp32_parity": parity,
                "fp64_diagnostic": comparison(candidate, oracle64),
                "analytic_rms_error": error,
                "aligned_critical_plane_rms_error": point_error if critical else None,
                "archive": archive_arrays(
                    arrays,
                    f"rhs_{family}_n{n}",
                    initial=initial,
                    candidate=candidate,
                    oracle_fp32=oracle32,
                    oracle_fp64=oracle64,
                    exact=exact,
                ),
                "parity_passed": row_passed,
            }
        )
    observed_rates = rates(errors, CONVERGENCE_SIZES)
    decreasing = all(
        later < earlier
        for earlier, later in zip(errors[:-1], errors[1:], strict=True)
    )
    convergence_passed = decreasing and max(observed_rates) >= 3.0
    return {
        "family": family,
        "sizes": list(CONVERGENCE_SIZES),
        "rows": rows,
        "analytic_errors": errors,
        "observed_rates": observed_rates,
        "analytic_errors_strictly_decrease": decreasing,
        "parity_passed": parity_passed,
        "convergence_gate_applies": not critical,
        "convergence_passed": convergence_passed if not critical else None,
        "passed": parity_passed and (convergence_passed if not critical else True),
    }


def smooth_direction(n: int, initial: np.ndarray) -> np.ndarray:
    coordinate = np.arange(n, dtype=np.float64) * (2.0 * math.pi / n)
    z, y, x = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    waves = np.stack(
        (
            np.sin(x) * np.cos(y) * np.cos(z),
            np.cos(2.0 * x) * np.sin(y),
            np.sin(x + y + z),
            np.cos(x - 2.0 * z),
            np.sin(2.0 * x - y + z),
        )
    )
    scales = np.maximum(np.max(np.abs(initial), axis=(1, 2, 3)), 1.0)
    return (waves * scales[:, None, None, None]).astype(np.float32)


def evaluate_sensitivity(executable: Path, arrays: Path) -> dict[str, Any]:
    n = 6
    h = 1.0e-3
    initial = perturbed_vortex_state(n)
    direction = smooth_direction(n, initial)
    plus = np.asarray(initial + h * direction, dtype=np.float32)
    minus = np.asarray(initial - h * direction, dtype=np.float32)
    candidate_plus, metadata_plus = run_native(executable, plus)
    candidate_minus, metadata_minus = run_native(executable, minus)
    oracle_plus = oracle_step(plus, 1, torch.float32)
    oracle_minus = oracle_step(minus, 1, torch.float32)
    candidate_response = (candidate_plus.astype(np.float64) - candidate_minus) / (2.0 * h)
    oracle_response = (oracle_plus.astype(np.float64) - oracle_minus) / (2.0 * h)
    discrepancy = rms(candidate_response - oracle_response) / max(
        rms(oracle_response), np.finfo(np.float64).tiny
    )
    passed = bool(
        discrepancy <= SENSITIVITY_REL_RMS
        and admissibility(plus)["positive"]
        and admissibility(minus)["positive"]
    )
    return {
        "n": n,
        "h": h,
        "relative_rms_discrepancy": discrepancy,
        "bound": SENSITIVITY_REL_RMS,
        "native_plus": metadata_plus,
        "native_minus": metadata_minus,
        "archive": archive_arrays(
            arrays,
            "directional_sensitivity_n6",
            initial=initial,
            direction=direction,
            plus=plus,
            minus=minus,
            candidate_plus=candidate_plus,
            candidate_minus=candidate_minus,
            oracle_plus=oracle_plus,
            oracle_minus=oracle_minus,
            candidate_response=candidate_response,
            oracle_response=oracle_response,
        ),
        "passed": passed,
    }


def identity_gate(executable: Path, arrays: Path) -> dict[str, Any]:
    initial_path = ROOT / "experiments/gpu_native_reformulation/evidence/g1_u0_20260829/u0_n32_initial.f32"
    expected_path = ROOT / "experiments/gpu_native_reformulation/evidence/g3_recovery_20260829/r6_n32_s1_final.f32"
    initial = np.fromfile(initial_path, dtype=np.float32).reshape(5, 32, 32, 32)
    expected = np.fromfile(expected_path, dtype=np.float32).reshape(5, 32, 32, 32)
    actual, metadata = run_native(executable, initial)
    actual_path = arrays / "identity_r6q_n32_s1.f32"
    actual.tofile(actual_path)
    return {
        "input_path": str(initial_path.relative_to(ROOT)),
        "expected_path": str(expected_path.relative_to(ROOT)),
        "input_sha256": sha256(initial_path),
        "expected_sha256": sha256(expected_path),
        "actual_path": str(actual_path.relative_to(ROOT)),
        "actual_sha256": sha256(actual_path),
        "byte_identical": bool(actual.tobytes() == expected.tobytes()),
        "native": metadata,
        "passed": bool(actual.tobytes() == expected.tobytes()),
    }


def environment() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cpu_threads": torch.get_num_threads(),
        "pid": os.getpid(),
    }
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,pstate", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        record["nvidia_smi"] = completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        record["nvidia_smi_error"] = str(error)
    return record


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    executable = Path(arguments.executable).resolve()
    archive_directory = Path(arguments.archive_dir).resolve()
    arrays = archive_directory / "arrays"
    arrays.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(arguments.cpu_threads)

    identity = identity_gate(executable, arrays)
    if not identity["passed"]:
        raise RuntimeError("R6Q is not bit-identical to frozen R6; qualification aborted")

    step_rows = []
    for family, factory in (("vortex", vortex_state), ("perturbed_vortex", perturbed_vortex_state)):
        for n, steps in STEP_CASES:
            step_rows.append(
                evaluate_step_case(executable, arrays, family, n, steps, factory(n))
            )

    smooth = evaluate_rhs_family(executable, arrays, critical=False)
    critical = evaluate_rhs_family(executable, arrays, critical=True)

    discontinuities = []
    for family, factory in (("dual_sod", dual_sod_state), ("dual_shu_osher", dual_shu_osher_state)):
        for steps in (1, 10):
            discontinuities.append(
                evaluate_step_case(executable, arrays, family, 32, steps, factory(32))
            )

    sensitivity = evaluate_sensitivity(executable, arrays)
    numerical_sections = (
        identity["passed"],
        all(row["passed"] for row in step_rows),
        smooth["passed"],
        critical["passed"],
        all(row["passed"] for row in discontinuities),
        sensitivity["passed"],
    )
    return {
        "schema": "gradflow-g3-r6q-qualification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "G3_QUALIFICATION_PROTOCOL.md",
        "candidate": EXPECTED_CONTRACT,
        "tolerances": {
            "step_atol": STEP_ATOL,
            "step_rtol": 0.0,
            "rhs_atol": RHS_ATOL,
            "rhs_relative_rms": RHS_REL_RMS,
            "sensitivity_relative_rms": SENSITIVITY_REL_RMS,
        },
        "environment": environment(),
        "executable": {
            "path": str(executable.relative_to(ROOT)),
            "sha256": sha256(executable),
            "bytes": executable.stat().st_size,
        },
        "identity": identity,
        "full_step_parity": step_rows,
        "smooth_spatial_convergence": smooth,
        "critical_point_characterization": critical,
        "periodic_discontinuity_stress": discontinuities,
        "directional_sensitivity": sensitivity,
        "differentiability": {
            "reverse_mode_or_autograd_abi": False,
            "backend_admission_blocker": True,
            "note": "Finite-difference sensitivity is not an autograd qualification.",
        },
        "passed": all(numerical_sections),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--cpu-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = run(arguments)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "schema": report["schema"],
        "passed": report["passed"],
        "smooth_rates": report["smooth_spatial_convergence"]["observed_rates"],
        "critical_rates": report["critical_point_characterization"]["observed_rates"],
        "sensitivity_relative_rms": report["directional_sensitivity"]["relative_rms_discrepancy"],
    }, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
