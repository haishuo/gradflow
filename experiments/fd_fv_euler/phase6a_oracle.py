"""Independent point/cell-average Euler projections for Phase 6A.

This module depends on NumPy and the preserved exact Sod oracle, but never on
PyTorch or the GradFlow production package.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.euler_boundary_shock.sod_exact import (
    RiemannSolution,
    sample_solution,
    sod_solution,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE_A = ROOT / "experiments/euler_boundary_shock/results/phase_a_20260827"
SHU_REFERENCE = PHASE_A / "shu_osher_fv_wenoz_hllc_t1p8_n12800.npz"
GAMMA = 1.4
ENTROPY_VELOCITY = 0.7
SMOOTH_SIZES = (24, 36, 54, 81)
SHOCK_SIZES = (200, 400, 800)
SMOOTH_TIMES = (0.0, 0.1)

FloatArray = NDArray[np.float64]


def primitive_to_conserved(primitive: FloatArray) -> FloatArray:
    primitive = np.asarray(primitive, dtype=np.float64)
    density, velocity, pressure = primitive
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity**2
    return np.stack((density, density * velocity, energy))


def conserved_to_primitive(conserved: FloatArray) -> FloatArray:
    conserved = np.asarray(conserved, dtype=np.float64)
    density = conserved[0]
    velocity = conserved[1] / density
    pressure = (GAMMA - 1.0) * (
        conserved[2] - 0.5 * conserved[1] ** 2 / density
    )
    return np.stack((density, velocity, pressure))


def cell_edges(left: float, right: float, cells: int) -> FloatArray:
    return np.linspace(left, right, cells + 1, dtype=np.float64)


def cell_centers(left: float, right: float, cells: int) -> FloatArray:
    edges = cell_edges(left, right, cells)
    return 0.5 * (edges[:-1] + edges[1:])


def entropy_point(cells: int, time: float) -> tuple[FloatArray, FloatArray]:
    x = cell_centers(0.0, 1.0, cells)
    phase = 2.0 * math.pi * (x - ENTROPY_VELOCITY * time)
    density = 1.0 + 0.1 * np.sin(phase)
    primitive = np.stack(
        (
            density,
            np.full_like(x, ENTROPY_VELOCITY),
            np.ones_like(x),
        )
    )
    density_rhs = (
        -ENTROPY_VELOCITY * 0.2 * math.pi * np.cos(phase)
    )
    factors = np.array(
        [1.0, ENTROPY_VELOCITY, 0.5 * ENTROPY_VELOCITY**2]
    )[:, None]
    return primitive_to_conserved(primitive), factors * density_rhs


def _mean_sine(left: FloatArray, right: FloatArray, shift: float) -> FloatArray:
    wave = 2.0 * math.pi
    spacing = right - left
    return (
        np.cos(wave * (left - shift)) - np.cos(wave * (right - shift))
    ) / (wave * spacing)


def entropy_average(cells: int, time: float) -> tuple[FloatArray, FloatArray]:
    edges = cell_edges(0.0, 1.0, cells)
    left = edges[:-1]
    right = edges[1:]
    spacing = 1.0 / cells
    shift = ENTROPY_VELOCITY * time
    density = 1.0 + 0.1 * _mean_sine(left, right, shift)
    velocity = ENTROPY_VELOCITY
    state = np.stack(
        (
            density,
            velocity * density,
            np.full_like(density, 1.0 / (GAMMA - 1.0))
            + 0.5 * velocity**2 * density,
        )
    )
    boundary_difference = 0.1 * (
        np.sin(2.0 * math.pi * (right - shift))
        - np.sin(2.0 * math.pi * (left - shift))
    )
    density_rhs = -velocity * boundary_difference / spacing
    factors = np.array([1.0, velocity, 0.5 * velocity**2])[:, None]
    return state, factors * density_rhs


def gauss_cell_average(
    sampler: Any,
    edges: FloatArray,
    *,
    order: int,
    split_points: tuple[float, ...] = (),
) -> FloatArray:
    """Integrate a vector sampler over cells, splitting known discontinuities."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    first = np.asarray(sampler(np.array([0.5 * (edges[0] + edges[1])])))[..., 0]
    result = np.empty((first.size, edges.size - 1), dtype=np.float64)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        points = [left]
        points.extend(value for value in split_points if left < value < right)
        points.append(right)
        integral = np.zeros(first.size, dtype=np.float64)
        for start, stop in zip(points[:-1], points[1:]):
            midpoint = 0.5 * (start + stop)
            half_width = 0.5 * (stop - start)
            coordinates = midpoint + half_width * nodes
            values = np.asarray(sampler(coordinates), dtype=np.float64)
            integral += half_width * np.sum(values * weights, axis=-1)
        result[:, index] = integral / (right - left)
    return result


def sod_wave_positions(solution: RiemannSolution, time: float) -> tuple[float, ...]:
    interface = 0.5
    return tuple(
        sorted(
            {
                interface + time * solution.left_head_speed,
                interface + time * solution.left_tail_speed,
                interface + time * solution.star_velocity,
                interface + time * solution.right_tail_speed,
                interface + time * solution.right_head_speed,
            }
        )
    )


def sod_point(cells: int, time: float = 0.2) -> tuple[FloatArray, FloatArray]:
    x = cell_centers(0.0, 1.0, cells)
    primitive = sample_solution(
        sod_solution(), x, time=time, interface=0.5
    )
    return primitive, primitive_to_conserved(primitive)


def sod_average(
    cells: int, time: float = 0.2, *, order: int = 64
) -> tuple[FloatArray, FloatArray]:
    solution = sod_solution()
    edges = cell_edges(0.0, 1.0, cells)

    def sampler(x: FloatArray) -> FloatArray:
        primitive = sample_solution(solution, x, time=time, interface=0.5)
        return primitive_to_conserved(primitive)

    conserved = gauss_cell_average(
        sampler,
        edges,
        order=order,
        split_points=sod_wave_positions(solution, time),
    )
    return conserved_to_primitive(conserved), conserved


def sod_integral_expected(time: float = 0.2) -> FloatArray:
    initial = np.array([0.5625, 0.0, 1.375], dtype=np.float64)
    left_flux = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right_flux = np.array([0.0, 0.1, 0.0], dtype=np.float64)
    return initial - time * (right_flux - left_flux)


def shu_initial(cells: int, method: str) -> FloatArray:
    edges = cell_edges(-5.0, 5.0, cells)
    x = 0.5 * (edges[:-1] + edges[1:])
    left = np.array([3.857143, 2.629369, 10.33333])[:, None]
    if method == "fd":
        right = np.stack(
            (1.0 + 0.2 * np.sin(5.0 * x), np.zeros_like(x), np.ones_like(x))
        )
        primitive = np.where(x[None, :] < -4.0, left, right)
        return primitive_to_conserved(primitive)
    if method != "fv":
        raise ValueError("method must be fd or fv")
    spacing = 10.0 / cells
    mean_sine = (
        np.cos(5.0 * edges[:-1]) - np.cos(5.0 * edges[1:])
    ) / (5.0 * spacing)
    right_density = 1.0 + 0.2 * mean_sine
    right_conserved = np.stack(
        (
            right_density,
            np.zeros_like(right_density),
            np.full_like(right_density, 1.0 / (GAMMA - 1.0)),
        )
    )
    left_conserved = primitive_to_conserved(left)
    return np.where(edges[:-1][None, :] < -4.0, left_conserved, right_conserved)


def shu_reference_projections(cells: int) -> dict[str, FloatArray]:
    with np.load(SHU_REFERENCE) as archive:
        reference_x = archive["x"].copy()
        reference_primitive = archive["primitive"].copy()
        reference_conserved = archive["conserved"].copy()
    reference_cells = reference_x.size
    if reference_cells % cells:
        raise ValueError("target cells must divide the Shu--Osher reference")
    factor = reference_cells // cells
    target_x = cell_centers(-5.0, 5.0, cells)
    fd_primitive = np.stack(
        [np.interp(target_x, reference_x, row) for row in reference_primitive]
    )
    fv_conserved = reference_conserved.reshape(3, cells, factor).mean(axis=-1)
    return {
        "x": target_x,
        "fd_primitive": fd_primitive,
        "fd_conserved": primitive_to_conserved(fd_primitive),
        "fv_conserved": fv_conserved,
        "fv_primitive": conserved_to_primitive(fv_conserved),
        "fine_conserved_integral": np.mean(reference_conserved, axis=-1),
        "restricted_conserved_integral": np.mean(fv_conserved, axis=-1),
    }


def build_projections() -> tuple[dict[str, FloatArray], dict[str, Any]]:
    arrays: dict[str, FloatArray] = {}
    diagnostics: dict[str, Any] = {"smooth": {}, "sod": {}, "shu_osher": {}}
    for cells in SMOOTH_SIZES:
        for time in SMOOTH_TIMES:
            label = f"smooth_n{cells}_t{str(time).replace('.', 'p')}"
            fd_state, fd_rhs = entropy_point(cells, time)
            fv_state, fv_rhs = entropy_average(cells, time)
            arrays[f"{label}_x"] = cell_centers(0.0, 1.0, cells)
            arrays[f"{label}_fd_state"] = fd_state
            arrays[f"{label}_fd_rhs"] = fd_rhs
            arrays[f"{label}_fv_state"] = fv_state
            arrays[f"{label}_fv_rhs"] = fv_rhs
            edges = cell_edges(0.0, 1.0, cells)

            def smooth_sampler(x: FloatArray) -> FloatArray:
                phase = 2.0 * math.pi * (x - ENTROPY_VELOCITY * time)
                primitive = np.stack(
                    (
                        1.0 + 0.1 * np.sin(phase),
                        np.full_like(x, ENTROPY_VELOCITY),
                        np.ones_like(x),
                    )
                )
                return primitive_to_conserved(primitive)

            quadrature = gauss_cell_average(smooth_sampler, edges, order=32)
            diagnostics["smooth"][label] = {
                "analytic_quadrature_maximum_absolute_difference": float(
                    np.max(np.abs(fv_state - quadrature))
                ),
                "fd_periodic_rhs_sum_maximum_absolute": float(
                    np.max(np.abs(np.sum(fd_rhs, axis=-1)))
                ),
                "fv_periodic_rhs_sum_maximum_absolute": float(
                    np.max(np.abs(np.sum(fv_rhs, axis=-1)))
                ),
                "point_cell_average_maximum_absolute_difference": float(
                    np.max(np.abs(fd_state - fv_state))
                ),
            }

    for cells in SHOCK_SIZES:
        label = f"sod_n{cells}"
        point_primitive, point_conserved = sod_point(cells)
        average_primitive, average_conserved = sod_average(cells, order=64)
        _, average_32 = sod_average(cells, order=32)
        arrays[f"{label}_x"] = cell_centers(0.0, 1.0, cells)
        arrays[f"{label}_fd_primitive"] = point_primitive
        arrays[f"{label}_fd_conserved"] = point_conserved
        arrays[f"{label}_fv_primitive"] = average_primitive
        arrays[f"{label}_fv_conserved"] = average_conserved
        expected_integral = sod_integral_expected()
        actual_integral = np.mean(average_conserved, axis=-1)
        diagnostics["sod"][label] = {
            "quadrature_32_64_maximum_absolute_difference": float(
                np.max(np.abs(average_32 - average_conserved))
            ),
            "conserved_integral": actual_integral.tolist(),
            "expected_conserved_integral": expected_integral.tolist(),
            "integral_maximum_absolute_difference": float(
                np.max(np.abs(actual_integral - expected_integral))
            ),
            "minimum_exact_average_density": float(np.min(average_conserved[0])),
            "minimum_exact_average_pressure": float(np.min(average_primitive[2])),
        }

        shu = shu_reference_projections(cells)
        arrays[f"shu_n{cells}_x"] = shu["x"]
        arrays[f"shu_n{cells}_fd_initial"] = shu_initial(cells, "fd")
        arrays[f"shu_n{cells}_fv_initial"] = shu_initial(cells, "fv")
        arrays[f"shu_n{cells}_fd_reference_primitive"] = shu["fd_primitive"]
        arrays[f"shu_n{cells}_fd_reference_conserved"] = shu["fd_conserved"]
        arrays[f"shu_n{cells}_fv_reference_primitive"] = shu["fv_primitive"]
        arrays[f"shu_n{cells}_fv_reference_conserved"] = shu["fv_conserved"]
        diagnostics["shu_osher"][f"shu_n{cells}"] = {
            "restriction_factor": 12800 // cells,
            "fine_restricted_integral_maximum_absolute_difference": float(
                np.max(
                    np.abs(
                        shu["fine_conserved_integral"]
                        - shu["restricted_conserved_integral"]
                    )
                )
            ),
            "fd_fv_initial_maximum_absolute_difference": float(
                np.max(
                    np.abs(
                        arrays[f"shu_n{cells}_fd_initial"]
                        - arrays[f"shu_n{cells}_fv_initial"]
                    )
                )
            ),
        }
    return arrays, diagnostics
