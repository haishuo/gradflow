"""Independent finite-volume WENO-Z/HLLC Euler reference procedure.

This is intentionally not the GradFlow finite-difference formulation.  It
uses componentwise primitive-variable finite-volume WENO-Z reconstruction and
an HLLC interface flux.  Its Sod results are checked against the exact Riemann
solution before it is used to construct a high-resolution Shu--Osher record.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
InitialCondition = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class RunStatistics:
    steps: int
    final_time: float
    minimum_density: float
    minimum_pressure: float
    reconstruction_fallbacks: int
    maximum_conservation_residual: float


def primitive_to_conserved(primitive: FloatArray, gamma: float = 1.4) -> FloatArray:
    primitive = np.asarray(primitive, dtype=np.float64)
    density, velocity, pressure = primitive
    energy = pressure / (gamma - 1.0) + 0.5 * density * velocity**2
    return np.stack((density, density * velocity, energy))


def conserved_to_primitive(conserved: FloatArray, gamma: float = 1.4) -> FloatArray:
    conserved = np.asarray(conserved, dtype=np.float64)
    density = conserved[0]
    velocity = conserved[1] / density
    pressure = (gamma - 1.0) * (
        conserved[2] - 0.5 * density * velocity**2
    )
    return np.stack((density, velocity, pressure))


def _left_wenoz(values: FloatArray, start: int, stop: int) -> FloatArray:
    vm2 = values[:, start - 2 : stop - 2]
    vm1 = values[:, start - 1 : stop - 1]
    v0 = values[:, start:stop]
    vp1 = values[:, start + 1 : stop + 1]
    vp2 = values[:, start + 2 : stop + 2]

    candidates = np.stack(
        (
            vm2 / 3.0 - 7.0 * vm1 / 6.0 + 11.0 * v0 / 6.0,
            -vm1 / 6.0 + 5.0 * v0 / 6.0 + vp1 / 3.0,
            v0 / 3.0 + 5.0 * vp1 / 6.0 - vp2 / 6.0,
        )
    )
    indicators = np.stack(
        (
            (13.0 / 12.0) * (vm2 - 2.0 * vm1 + v0) ** 2
            + 0.25 * (vm2 - 4.0 * vm1 + 3.0 * v0) ** 2,
            (13.0 / 12.0) * (vm1 - 2.0 * v0 + vp1) ** 2
            + 0.25 * (vm1 - vp1) ** 2,
            (13.0 / 12.0) * (v0 - 2.0 * vp1 + vp2) ** 2
            + 0.25 * (3.0 * v0 - 4.0 * vp1 + vp2) ** 2,
        )
    )
    tau = np.abs(indicators[0] - indicators[2])
    linear = np.array([0.1, 0.6, 0.3], dtype=np.float64)[:, None, None]
    weights = linear * (1.0 + (tau[None, ...] / (indicators + 1.0e-40)) ** 2)
    weights /= np.sum(weights, axis=0, keepdims=True)
    return np.sum(weights * candidates, axis=0)


def _right_wenoz(values: FloatArray, start: int, stop: int) -> FloatArray:
    vm1 = values[:, start - 1 : stop - 1]
    v0 = values[:, start:stop]
    vp1 = values[:, start + 1 : stop + 1]
    vp2 = values[:, start + 2 : stop + 2]
    vp3 = values[:, start + 3 : stop + 3]

    candidates = np.stack(
        (
            vp3 / 3.0 - 7.0 * vp2 / 6.0 + 11.0 * vp1 / 6.0,
            -vp2 / 6.0 + 5.0 * vp1 / 6.0 + v0 / 3.0,
            vp1 / 3.0 + 5.0 * v0 / 6.0 - vm1 / 6.0,
        )
    )
    indicators = np.stack(
        (
            (13.0 / 12.0) * (vp3 - 2.0 * vp2 + vp1) ** 2
            + 0.25 * (vp3 - 4.0 * vp2 + 3.0 * vp1) ** 2,
            (13.0 / 12.0) * (vp2 - 2.0 * vp1 + v0) ** 2
            + 0.25 * (vp2 - v0) ** 2,
            (13.0 / 12.0) * (vp1 - 2.0 * v0 + vm1) ** 2
            + 0.25 * (3.0 * vp1 - 4.0 * v0 + vm1) ** 2,
        )
    )
    tau = np.abs(indicators[0] - indicators[2])
    linear = np.array([0.1, 0.6, 0.3], dtype=np.float64)[:, None, None]
    weights = linear * (1.0 + (tau[None, ...] / (indicators + 1.0e-40)) ** 2)
    weights /= np.sum(weights, axis=0, keepdims=True)
    return np.sum(weights * candidates, axis=0)


def reconstruct_primitive(
    ghosted_conserved: FloatArray,
    physical_size: int,
    *,
    gamma: float = 1.4,
    ghosts: int = 3,
) -> tuple[FloatArray, FloatArray, int]:
    """Reconstruct primitive left/right states at all physical faces."""
    primitive = conserved_to_primitive(ghosted_conserved, gamma)
    if np.min(primitive[0]) <= 0.0 or np.min(primitive[2]) <= 0.0:
        raise FloatingPointError("nonphysical cell state in reference procedure")
    start = ghosts - 1
    stop = start + physical_size + 1
    left = _left_wenoz(primitive, start, stop)
    right = _right_wenoz(primitive, start, stop)

    bad_left = (~np.isfinite(left).all(axis=0)) | (left[0] <= 0.0) | (left[2] <= 0.0)
    bad_right = (
        (~np.isfinite(right).all(axis=0)) | (right[0] <= 0.0) | (right[2] <= 0.0)
    )
    interface_indices = np.arange(start, stop)
    if np.any(bad_left):
        left[:, bad_left] = primitive[:, interface_indices[bad_left]]
    if np.any(bad_right):
        right[:, bad_right] = primitive[:, interface_indices[bad_right] + 1]
    fallbacks = int(np.count_nonzero(bad_left) + np.count_nonzero(bad_right))
    return left, right, fallbacks


def _physical_flux(primitive: FloatArray, gamma: float) -> tuple[FloatArray, FloatArray]:
    conserved = primitive_to_conserved(primitive, gamma)
    density, velocity, pressure = primitive
    flux = np.stack(
        (
            density * velocity,
            density * velocity**2 + pressure,
            velocity * (conserved[2] + pressure),
        )
    )
    return conserved, flux


def hllc_flux(left: FloatArray, right: FloatArray, gamma: float = 1.4) -> FloatArray:
    """Return the HLLC flux for arrays of primitive interface states."""
    left_conserved, left_flux = _physical_flux(left, gamma)
    right_conserved, right_flux = _physical_flux(right, gamma)
    left_density, left_velocity, left_pressure = left
    right_density, right_velocity, right_pressure = right
    left_sound = np.sqrt(gamma * left_pressure / left_density)
    right_sound = np.sqrt(gamma * right_pressure / right_density)
    left_wave = np.minimum(left_velocity - left_sound, right_velocity - right_sound)
    right_wave = np.maximum(left_velocity + left_sound, right_velocity + right_sound)
    denominator = (
        left_density * (left_wave - left_velocity)
        - right_density * (right_wave - right_velocity)
    )
    if np.any(np.abs(denominator) < 1.0e-14):
        raise FloatingPointError("degenerate HLLC contact-wave denominator")
    contact = (
        right_pressure
        - left_pressure
        + left_density * left_velocity * (left_wave - left_velocity)
        - right_density * right_velocity * (right_wave - right_velocity)
    ) / denominator

    def star_state(
        primitive: FloatArray,
        conserved: FloatArray,
        outer_wave: FloatArray,
    ) -> FloatArray:
        density, velocity, pressure = primitive
        star_density = density * (outer_wave - velocity) / (outer_wave - contact)
        specific_energy = conserved[2] / density
        star_energy = star_density * (
            specific_energy
            + (contact - velocity)
            * (contact + pressure / (density * (outer_wave - velocity)))
        )
        return np.stack((star_density, star_density * contact, star_energy))

    left_star = star_state(left, left_conserved, left_wave)
    right_star = star_state(right, right_conserved, right_wave)
    result = np.where(
        (left_wave >= 0.0)[None, :],
        left_flux,
        np.where(
            (contact >= 0.0)[None, :],
            left_flux + left_wave[None, :] * (left_star - left_conserved),
            np.where(
                (right_wave > 0.0)[None, :],
                right_flux + right_wave[None, :] * (right_star - right_conserved),
                right_flux,
            ),
        ),
    )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite HLLC flux")
    return result


def fill_transmissive(conserved: FloatArray, ghosts: int = 3) -> FloatArray:
    physical_size = conserved.shape[1]
    result = np.empty((3, physical_size + 2 * ghosts), dtype=np.float64)
    result[:, ghosts : ghosts + physical_size] = conserved
    result[:, :ghosts] = conserved[:, :1]
    result[:, ghosts + physical_size :] = conserved[:, -1:]
    return result


def finite_volume_rhs(
    conserved: FloatArray,
    spacing: float,
    *,
    gamma: float = 1.4,
) -> tuple[FloatArray, FloatArray, int, float]:
    ghosted = fill_transmissive(conserved)
    left, right, fallbacks = reconstruct_primitive(
        ghosted, conserved.shape[1], gamma=gamma
    )
    face_flux = hllc_flux(left, right, gamma)
    rhs = -(face_flux[:, 1:] - face_flux[:, :-1]) / spacing
    conservation_residual = spacing * np.sum(rhs, axis=1) + (
        face_flux[:, -1] - face_flux[:, 0]
    )
    scale = np.maximum(
        1.0,
        spacing * np.sum(np.abs(rhs), axis=1)
        + np.abs(face_flux[:, -1])
        + np.abs(face_flux[:, 0]),
    )
    normalized_residual = float(
        np.max(np.abs(conservation_residual) / (np.finfo(np.float64).eps * scale))
    )
    return rhs, face_flux, fallbacks, normalized_residual


def solve(
    initial_condition: InitialCondition,
    *,
    left: float,
    right: float,
    cells: int,
    final_time: float,
    cfl: float = 0.4,
    gamma: float = 1.4,
) -> tuple[FloatArray, FloatArray, RunStatistics]:
    """Advance a cell-centered finite-volume reference with SSP-RK3."""
    if cells < 8:
        raise ValueError("cells must be at least eight")
    if not left < right:
        raise ValueError("left must be less than right")
    if final_time <= 0.0 or cfl <= 0.0:
        raise ValueError("final_time and cfl must be positive")
    spacing = (right - left) / cells
    x = left + (np.arange(cells, dtype=np.float64) + 0.5) * spacing
    conserved = primitive_to_conserved(initial_condition(x), gamma)
    time = 0.0
    steps = 0
    fallbacks = 0
    maximum_conservation_residual = 0.0
    minimum_density = math.inf
    minimum_pressure = math.inf

    def rhs(state: FloatArray) -> FloatArray:
        nonlocal fallbacks, maximum_conservation_residual
        derivative, _, local_fallbacks, residual = finite_volume_rhs(
            state, spacing, gamma=gamma
        )
        fallbacks += local_fallbacks
        maximum_conservation_residual = max(maximum_conservation_residual, residual)
        return derivative

    while time < final_time:
        primitive = conserved_to_primitive(conserved, gamma)
        if not np.all(np.isfinite(primitive)):
            raise FloatingPointError(f"nonfinite physical state at step {steps}")
        local_density = float(np.min(primitive[0]))
        local_pressure = float(np.min(primitive[2]))
        minimum_density = min(minimum_density, local_density)
        minimum_pressure = min(minimum_pressure, local_pressure)
        if local_density <= 0.0 or local_pressure <= 0.0:
            raise FloatingPointError(f"nonphysical state at step {steps}")
        speed = np.max(
            np.abs(primitive[1]) + np.sqrt(gamma * primitive[2] / primitive[0])
        )
        timestep = min(cfl * spacing / speed, final_time - time)

        first = conserved + timestep * rhs(conserved)
        second = 0.75 * conserved + 0.25 * (first + timestep * rhs(first))
        conserved = (
            conserved + 2.0 * (second + timestep * rhs(second))
        ) / 3.0
        time += timestep
        steps += 1
        if steps > 10_000_000:
            raise RuntimeError("reference step guard exceeded")

    final_primitive = conserved_to_primitive(conserved, gamma)
    minimum_density = min(minimum_density, float(np.min(final_primitive[0])))
    minimum_pressure = min(minimum_pressure, float(np.min(final_primitive[2])))
    statistics = RunStatistics(
        steps=steps,
        final_time=time,
        minimum_density=minimum_density,
        minimum_pressure=minimum_pressure,
        reconstruction_fallbacks=fallbacks,
        maximum_conservation_residual=maximum_conservation_residual,
    )
    return x, final_primitive, statistics


def sod_initial(x: FloatArray) -> FloatArray:
    left = np.array([1.0, 0.0, 1.0], dtype=np.float64)[:, None]
    right = np.array([0.125, 0.0, 0.1], dtype=np.float64)[:, None]
    return np.where(x[None, :] < 0.5, left, right)


def shu_osher_initial(x: FloatArray) -> FloatArray:
    left = np.array([3.857143, 2.629369, 10.33333], dtype=np.float64)[:, None]
    right = np.stack((1.0 + 0.2 * np.sin(5.0 * x), np.zeros_like(x), np.ones_like(x)))
    return np.where(x[None, :] < -4.0, left, right)
