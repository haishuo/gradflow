"""Exact ideal-gas Euler Riemann solution used for the Sod oracle.

This module is deliberately independent of ``src/gradflow`` and PyTorch.  It
implements the pressure-function construction directly from the Euler shock
and rarefaction relations and samples the resulting self-similar solution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PrimitiveState:
    """One-dimensional primitive Euler state."""

    density: float
    velocity: float
    pressure: float

    def validate(self) -> None:
        values = (self.density, self.velocity, self.pressure)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("primitive state values must be finite")
        if self.density <= 0.0 or self.pressure <= 0.0:
            raise ValueError("density and pressure must be positive")


@dataclass(frozen=True)
class RiemannSolution:
    """Resolved star state and wave speeds for an Euler Riemann problem."""

    left: PrimitiveState
    right: PrimitiveState
    gamma: float
    star_pressure: float
    star_velocity: float
    left_star_density: float
    right_star_density: float
    left_head_speed: float
    left_tail_speed: float
    right_tail_speed: float
    right_head_speed: float


def sound_speed(state: PrimitiveState, gamma: float) -> float:
    return math.sqrt(gamma * state.pressure / state.density)


def _pressure_function(
    pressure: float, state: PrimitiveState, gamma: float
) -> tuple[float, float]:
    """Return f(p) and df/dp for one side of the Riemann problem."""
    if pressure > state.pressure:
        coefficient = 2.0 / ((gamma + 1.0) * state.density)
        offset = (gamma - 1.0) * state.pressure / (gamma + 1.0)
        root = math.sqrt(coefficient / (pressure + offset))
        value = (pressure - state.pressure) * root
        derivative = root * (
            1.0 - 0.5 * (pressure - state.pressure) / (pressure + offset)
        )
        return value, derivative

    exponent = (gamma - 1.0) / (2.0 * gamma)
    ratio = pressure / state.pressure
    speed = sound_speed(state, gamma)
    value = 2.0 * speed * (ratio**exponent - 1.0) / (gamma - 1.0)
    derivative = (
        ratio ** (-(gamma + 1.0) / (2.0 * gamma))
        / (state.density * speed)
    )
    return value, derivative


def _star_density(
    state: PrimitiveState, star_pressure: float, gamma: float
) -> float:
    ratio = star_pressure / state.pressure
    if star_pressure > state.pressure:
        constant = (gamma - 1.0) / (gamma + 1.0)
        return state.density * (ratio + constant) / (constant * ratio + 1.0)
    return state.density * ratio ** (1.0 / gamma)


def solve_riemann(
    left: PrimitiveState,
    right: PrimitiveState,
    *,
    gamma: float = 1.4,
) -> RiemannSolution:
    """Solve a non-vacuum ideal-gas Euler Riemann problem."""
    left.validate()
    right.validate()
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")

    left_speed = sound_speed(left, gamma)
    right_speed = sound_speed(right, gamma)
    vacuum_measure = (
        2.0 * (left_speed + right_speed) / (gamma - 1.0)
        - (right.velocity - left.velocity)
    )
    if vacuum_measure <= 0.0:
        raise ValueError("vacuum-generating Riemann data are unsupported")

    def residual(pressure: float) -> float:
        left_value, _ = _pressure_function(pressure, left, gamma)
        right_value, _ = _pressure_function(pressure, right, gamma)
        return left_value + right_value + right.velocity - left.velocity

    lower = np.finfo(np.float64).tiny
    upper = max(left.pressure, right.pressure, 1.0)
    while residual(upper) < 0.0:
        upper *= 2.0
        if not math.isfinite(upper):
            raise RuntimeError("failed to bracket the star pressure")

    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        value = residual(midpoint)
        if value > 0.0:
            upper = midpoint
        else:
            lower = midpoint
        if upper - lower <= 4.0 * np.finfo(np.float64).eps * upper:
            break
    star_pressure = 0.5 * (lower + upper)

    left_value, _ = _pressure_function(star_pressure, left, gamma)
    right_value, _ = _pressure_function(star_pressure, right, gamma)
    star_velocity = 0.5 * (
        left.velocity + right.velocity + right_value - left_value
    )
    left_star_density = _star_density(left, star_pressure, gamma)
    right_star_density = _star_density(right, star_pressure, gamma)

    if star_pressure > left.pressure:
        ratio = star_pressure / left.pressure
        speed = left.velocity - left_speed * math.sqrt(
            (gamma + 1.0) * ratio / (2.0 * gamma)
            + (gamma - 1.0) / (2.0 * gamma)
        )
        left_head = speed
        left_tail = speed
    else:
        left_head = left.velocity - left_speed
        left_star_speed = left_speed * (
            star_pressure / left.pressure
        ) ** ((gamma - 1.0) / (2.0 * gamma))
        left_tail = star_velocity - left_star_speed

    if star_pressure > right.pressure:
        ratio = star_pressure / right.pressure
        speed = right.velocity + right_speed * math.sqrt(
            (gamma + 1.0) * ratio / (2.0 * gamma)
            + (gamma - 1.0) / (2.0 * gamma)
        )
        right_tail = speed
        right_head = speed
    else:
        right_star_speed = right_speed * (
            star_pressure / right.pressure
        ) ** ((gamma - 1.0) / (2.0 * gamma))
        right_tail = star_velocity + right_star_speed
        right_head = right.velocity + right_speed

    return RiemannSolution(
        left=left,
        right=right,
        gamma=gamma,
        star_pressure=star_pressure,
        star_velocity=star_velocity,
        left_star_density=left_star_density,
        right_star_density=right_star_density,
        left_head_speed=left_head,
        left_tail_speed=left_tail,
        right_tail_speed=right_tail,
        right_head_speed=right_head,
    )


def _sample_left_fan(solution: RiemannSolution, similarity: float) -> PrimitiveState:
    left = solution.left
    gamma = solution.gamma
    speed = sound_speed(left, gamma)
    velocity = 2.0 * (
        speed + 0.5 * (gamma - 1.0) * left.velocity + similarity
    ) / (gamma + 1.0)
    local_speed = 2.0 * (
        speed + 0.5 * (gamma - 1.0) * (left.velocity - similarity)
    ) / (gamma + 1.0)
    ratio = local_speed / speed
    return PrimitiveState(
        density=left.density * ratio ** (2.0 / (gamma - 1.0)),
        velocity=velocity,
        pressure=left.pressure * ratio ** (2.0 * gamma / (gamma - 1.0)),
    )


def _sample_right_fan(
    solution: RiemannSolution, similarity: float
) -> PrimitiveState:
    right = solution.right
    gamma = solution.gamma
    speed = sound_speed(right, gamma)
    velocity = 2.0 * (
        -speed + 0.5 * (gamma - 1.0) * right.velocity + similarity
    ) / (gamma + 1.0)
    local_speed = 2.0 * (
        speed - 0.5 * (gamma - 1.0) * (right.velocity - similarity)
    ) / (gamma + 1.0)
    ratio = local_speed / speed
    return PrimitiveState(
        density=right.density * ratio ** (2.0 / (gamma - 1.0)),
        velocity=velocity,
        pressure=right.pressure * ratio ** (2.0 * gamma / (gamma - 1.0)),
    )


def sample_solution(
    solution: RiemannSolution,
    x: FloatArray,
    *,
    time: float,
    interface: float,
) -> FloatArray:
    """Sample ``(rho, u, p)`` at coordinates ``x``."""
    coordinates = np.asarray(x, dtype=np.float64)
    if coordinates.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not math.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and nonnegative")
    if not math.isfinite(interface):
        raise ValueError("interface must be finite")
    if time == 0.0:
        left_values = np.array(
            [solution.left.density, solution.left.velocity, solution.left.pressure]
        )[:, None]
        right_values = np.array(
            [
                solution.right.density,
                solution.right.velocity,
                solution.right.pressure,
            ]
        )[:, None]
        return np.where(coordinates[None, :] < interface, left_values, right_values)

    result = np.empty((3, coordinates.size), dtype=np.float64)
    for index, coordinate in enumerate(coordinates):
        similarity = (coordinate - interface) / time
        if similarity <= solution.star_velocity:
            if solution.star_pressure > solution.left.pressure:
                if similarity <= solution.left_head_speed:
                    state = solution.left
                else:
                    state = PrimitiveState(
                        solution.left_star_density,
                        solution.star_velocity,
                        solution.star_pressure,
                    )
            elif similarity <= solution.left_head_speed:
                state = solution.left
            elif similarity >= solution.left_tail_speed:
                state = PrimitiveState(
                    solution.left_star_density,
                    solution.star_velocity,
                    solution.star_pressure,
                )
            else:
                state = _sample_left_fan(solution, similarity)
        else:
            if solution.star_pressure > solution.right.pressure:
                if similarity >= solution.right_head_speed:
                    state = solution.right
                else:
                    state = PrimitiveState(
                        solution.right_star_density,
                        solution.star_velocity,
                        solution.star_pressure,
                    )
            elif similarity >= solution.right_head_speed:
                state = solution.right
            elif similarity <= solution.right_tail_speed:
                state = PrimitiveState(
                    solution.right_star_density,
                    solution.star_velocity,
                    solution.star_pressure,
                )
            else:
                state = _sample_right_fan(solution, similarity)
        result[:, index] = (state.density, state.velocity, state.pressure)
    return result


SOD_LEFT = PrimitiveState(1.0, 0.0, 1.0)
SOD_RIGHT = PrimitiveState(0.125, 0.0, 0.1)


def sod_solution() -> RiemannSolution:
    return solve_riemann(SOD_LEFT, SOD_RIGHT, gamma=1.4)


def validate_sod_oracle() -> dict[str, float | bool]:
    """Run relations independent of the sampler and return auditable metrics."""
    solution = sod_solution()
    left_value, _ = _pressure_function(
        solution.star_pressure, solution.left, solution.gamma
    )
    right_value, _ = _pressure_function(
        solution.star_pressure, solution.right, solution.gamma
    )
    pressure_residual = abs(
        left_value
        + right_value
        + solution.right.velocity
        - solution.left.velocity
    )

    right_star = PrimitiveState(
        solution.right_star_density,
        solution.star_velocity,
        solution.star_pressure,
    )
    shock_speed = solution.right_head_speed

    def conserved(state: PrimitiveState) -> np.ndarray:
        energy = state.pressure / (solution.gamma - 1.0) + 0.5 * (
            state.density * state.velocity**2
        )
        return np.array(
            [state.density, state.density * state.velocity, energy],
            dtype=np.float64,
        )

    def flux(state: PrimitiveState) -> np.ndarray:
        state_conserved = conserved(state)
        return np.array(
            [
                state_conserved[1],
                state_conserved[1] * state.velocity + state.pressure,
                state.velocity * (state_conserved[2] + state.pressure),
            ],
            dtype=np.float64,
        )

    rankine_hugoniot = flux(right_star) - flux(solution.right) - shock_speed * (
        conserved(right_star) - conserved(solution.right)
    )
    fan_similarity = 0.5 * (
        solution.left_head_speed + solution.left_tail_speed
    )
    fan_state = _sample_left_fan(solution, fan_similarity)
    fan_sound = sound_speed(fan_state, solution.gamma)
    left_invariant = solution.left.velocity + 2.0 * sound_speed(
        solution.left, solution.gamma
    ) / (solution.gamma - 1.0)
    fan_invariant = fan_state.velocity + 2.0 * fan_sound / (
        solution.gamma - 1.0
    )
    fan_isentropy_error = abs(
        fan_state.pressure / fan_state.density**solution.gamma
        - solution.left.pressure / solution.left.density**solution.gamma
    )
    fan_characteristic_error = abs(
        fan_similarity - (fan_state.velocity - fan_sound)
    )
    fan_invariant_error = abs(fan_invariant - left_invariant)
    samples = sample_solution(
        solution, np.linspace(-2.0, 3.0, 5001), time=1.0, interface=0.0
    )
    far_field_error = max(
        float(
            np.max(
                np.abs(
                    samples[:, 0]
                    - np.array(
                        [
                            solution.left.density,
                            solution.left.velocity,
                            solution.left.pressure,
                        ]
                    )
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    samples[:, -1]
                    - np.array(
                        [
                            solution.right.density,
                            solution.right.velocity,
                            solution.right.pressure,
                        ]
                    )
                )
            )
        ),
    )
    known_pressure_error = abs(solution.star_pressure - 0.30313017805064685)
    known_velocity_error = abs(solution.star_velocity - 0.9274526200489499)
    wave_ordered = (
        solution.left_head_speed
        < solution.left_tail_speed
        < solution.star_velocity
        < solution.right_head_speed
    )
    passed = bool(
        pressure_residual < 2.0e-14
        and np.max(np.abs(rankine_hugoniot)) < 2.0e-13
        and fan_isentropy_error < 2.0e-14
        and fan_characteristic_error < 2.0e-14
        and fan_invariant_error < 2.0e-14
        and far_field_error == 0.0
        and known_pressure_error < 2.0e-14
        and known_velocity_error < 2.0e-14
        and wave_ordered
        and np.all(np.isfinite(samples))
        and np.min(samples[0]) > 0.0
        and np.min(samples[2]) > 0.0
    )
    return {
        "passed": passed,
        "pressure_residual": pressure_residual,
        "rankine_hugoniot_linf": float(np.max(np.abs(rankine_hugoniot))),
        "fan_isentropy_error": fan_isentropy_error,
        "fan_characteristic_error": fan_characteristic_error,
        "fan_riemann_invariant_error": fan_invariant_error,
        "far_field_error": far_field_error,
        "known_star_pressure_error": known_pressure_error,
        "known_star_velocity_error": known_velocity_error,
        "star_pressure": solution.star_pressure,
        "star_velocity": solution.star_velocity,
        "left_star_density": solution.left_star_density,
        "right_star_density": solution.right_star_density,
        "left_head_speed": solution.left_head_speed,
        "left_tail_speed": solution.left_tail_speed,
        "right_shock_speed": solution.right_head_speed,
        "wave_ordered": wave_ordered,
    }
