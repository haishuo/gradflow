#!/usr/bin/env python3
"""Compare immutable G1 U0 states with the qualified GradFlow oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from gradflow import (  # noqa: E402
    EULER_GAMMA,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    euler_weno5_rhs,
    synchronize_duplicate_endpoints,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_unique(path: Path, size: int) -> torch.Tensor:
    expected = 5 * size**3
    values = torch.from_file(str(path), dtype=torch.float32, size=expected)
    return values.reshape(5, size, size, size).to(torch.float64)


def duplicate_periodic(unique: torch.Tensor) -> torch.Tensor:
    state = unique
    for axis in (3, 2, 1):
        state = torch.cat((state, state.narrow(axis, 0, 1)), dim=axis)
    return state


def unique_cells(duplicated: torch.Tensor) -> torch.Tensor:
    return duplicated[:, :-1, :-1, :-1]


def pressure(state: torch.Tensor) -> torch.Tensor:
    density = state[0]
    momentum = state[1:4]
    energy = state[4]
    kinetic = 0.5 * (momentum * momentum).sum(dim=0) / density
    return (EULER_GAMMA - 1.0) * (energy - kinetic)


def health(state: torch.Tensor) -> dict[str, object]:
    return {
        "finite": bool(torch.isfinite(state).all()),
        "minimum_density": float(torch.min(state[0])),
        "minimum_pressure": float(torch.min(pressure(state))),
        "maximum_absolute_state": float(torch.max(torch.abs(state))),
    }


def conservation(state: torch.Tensor, initial: torch.Tensor) -> dict[str, object]:
    cells = math.prod(initial.shape[1:])
    drift = (state - initial).sum(dim=(1, 2, 3))
    return {
        "component_drift": [float(value) for value in drift],
        "component_drift_per_cell": [float(value / cells) for value in drift],
    }


def error_metrics(
    actual: torch.Tensor, expected: torch.Tensor, initial: torch.Tensor
) -> dict[str, object]:
    error = actual - expected
    actual_update = actual - initial
    expected_update = expected - initial
    rms = torch.sqrt(torch.mean(error.square()))
    state_rms = torch.sqrt(torch.mean(expected.square()))
    actual_update_rms = torch.sqrt(torch.mean(actual_update.square()))
    update_rms = torch.sqrt(torch.mean(expected_update.square()))
    update_cosine = torch.sum(actual_update * expected_update) / (
        torch.linalg.vector_norm(actual_update)
        * torch.linalg.vector_norm(expected_update)
    )
    component_rms = torch.sqrt(torch.mean(error.square(), dim=(1, 2, 3)))
    component_max = torch.amax(torch.abs(error), dim=(1, 2, 3))
    return {
        "maximum_absolute_error": float(torch.max(torch.abs(error))),
        "mean_absolute_error": float(torch.mean(torch.abs(error))),
        "rms_error": float(rms),
        "rms_error_over_oracle_state_rms": float(rms / state_rms),
        "rms_error_over_oracle_update_rms": float(rms / update_rms),
        "u0_update_rms_over_oracle_update_rms": float(
            actual_update_rms / update_rms
        ),
        "u0_oracle_update_cosine": float(update_cosine),
        "component_maximum_absolute_error": [
            float(value) for value in component_max
        ],
        "component_rms_error": [float(value) for value in component_rms],
    }


def forward_euler(
    initial: torch.Tensor,
    spacing: tuple[float, float, float],
    steps: int,
    first_dt: float | None = None,
) -> tuple[torch.Tensor, list[float]]:
    state = initial
    timesteps: list[float] = []
    for step in range(steps):
        if step == 0 and first_dt is not None:
            dt = state.new_tensor(first_dt)
        else:
            dt = euler_cfl_timestep(state, spacing, 0.1)
        timesteps.append(float(dt))
        state = synchronize_duplicate_endpoints(
            state + dt * euler_weno5_rhs(state, spacing)
        )
    return state, timesteps


def ssp_rk3(
    initial: torch.Tensor,
    spacing: tuple[float, float, float],
    steps: int,
) -> tuple[torch.Tensor, list[float]]:
    state = initial
    timesteps: list[float] = []
    for _ in range(steps):
        dt = euler_cfl_timestep(state, spacing, 0.1)
        timesteps.append(float(dt))
        state = euler_ssp_rk3_step(state, spacing, dt, order=5)
    return state, timesteps


def comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
    initial: torch.Tensor,
    timesteps: list[float],
) -> dict[str, object]:
    expected_unique = unique_cells(expected)
    return {
        "oracle_timesteps": timesteps,
        "error": error_metrics(actual, expected_unique, initial),
        "u0_health": health(actual),
        "oracle_health": health(expected_unique),
        "u0_conservation": conservation(actual, initial),
        "oracle_conservation": conservation(expected_unique, initial),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    evidence = options.evidence.resolve()
    size = 32
    initial_path = evidence / "u0_n32_initial.f32"
    one_path = evidence / "u0_n32_s1_final.f32"
    ten_path = evidence / "u0_n32_s10_final.f32"
    timing_one = json.loads((evidence / "u0_n32_s1.json").read_text())
    timing_ten = json.loads((evidence / "u0_n32_s10.json").read_text())

    initial_unique = load_unique(initial_path, size)
    u0_one = load_unique(one_path, size)
    u0_ten = load_unique(ten_path, size)
    initial = duplicate_periodic(initial_unique)
    spacing = (10.0 / size,) * 3

    fe_one, fe_one_dt = forward_euler(
        initial, spacing, 1, first_dt=float(timing_one["final_dt"])
    )
    rk_one, rk_one_dt = ssp_rk3(initial, spacing, 1)
    fe_ten, fe_ten_dt = forward_euler(initial, spacing, 10)
    rk_ten, rk_ten_dt = ssp_rk3(initial, spacing, 10)

    record = {
        "study": "g2_frozen_u0_damage",
        "u0_contract": timing_one["contract"],
        "grid": [size, size, size],
        "dtype_u0": "float32",
        "dtype_oracle": "float64",
        "oracle": "qualified GradFlow characteristic FD JS-WENO-5",
        "input_sha256": sha256(initial_path),
        "u0_s1_sha256": sha256(one_path),
        "u0_s10_sha256": sha256(ten_path),
        "u0_final_timesteps": {
            "steps_1": timing_one["final_dt"],
            "steps_10": timing_ten["final_dt"],
        },
        "steps_1": {
            "forward_euler_same_first_dt": comparison(
                u0_one, fe_one, initial_unique, fe_one_dt
            ),
            "qualified_ssp_rk3": comparison(
                u0_one, rk_one, initial_unique, rk_one_dt
            ),
        },
        "steps_10": {
            "forward_euler_recomputed_cfl": comparison(
                u0_ten, fe_ten, initial_unique, fe_ten_dt
            ),
            "qualified_ssp_rk3": comparison(
                u0_ten, rk_ten, initial_unique, rk_ten_dt
            ),
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
