"""Standard-library exact oracle for smooth pre-shock periodic Burgers flow.

The oracle is intentionally independent of PyTorch, NumPy, GradFlow, and every
WENO implementation. Point values come from characteristic inversion. Cell
averages come from an exact conservation-law primitive in characteristic
coordinates.
"""

from __future__ import annotations

import math
from typing import Callable, Literal


BASE = 0.5
AMPLITUDE = 0.2
PHASE = 0.07
FINAL_TIME = 0.1
MINIMUM_STATE = BASE - AMPLITUDE
MAXIMUM_STATE = BASE + AMPLITUDE
LF_ALPHA = MAXIMUM_STATE
SHOCK_TIME = 1.0 / (2.0 * math.pi * AMPLITUDE)
MINIMUM_CHARACTERISTIC_JACOBIAN = 1.0 - (
    FINAL_TIME * 2.0 * math.pi * AMPLITUDE
)

Projection = Literal["fd", "fv"]


def initial_value(x: float) -> float:
    """Return the periodic initial state on any real lift of the domain."""
    return BASE + AMPLITUDE * math.sin(2.0 * math.pi * (x - PHASE))


def initial_derivative(x: float) -> float:
    return 2.0 * math.pi * AMPLITUDE * math.cos(
        2.0 * math.pi * (x - PHASE)
    )


def characteristic_map(xi: float, time: float) -> float:
    return xi + time * initial_value(xi)


def _validate_time(time: float) -> None:
    if not math.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and nonnegative")
    if time >= SHOCK_TIME:
        raise ValueError("the classical characteristic oracle stops before shock")


def characteristic_foot(x: float, time: float, *, iterations: int = 80) -> float:
    """Invert the pre-shock characteristic map on a real periodic lift."""
    if not math.isfinite(x):
        raise ValueError("x must be finite")
    _validate_time(time)
    if isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if time == 0.0:
        return x

    lower = x - time * MAXIMUM_STATE
    upper = x - time * MINIMUM_STATE
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        residual = characteristic_map(midpoint, time) - x
        if residual > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return 0.5 * (lower + upper)


def exact_point(x: float, time: float) -> float:
    """Return the exact smooth point value at an Eulerian coordinate."""
    return initial_value(characteristic_foot(x, time))


def exact_spatial_derivative(x: float, time: float) -> float:
    """Return the exact smooth derivative u_x before shock formation."""
    xi = characteristic_foot(x, time)
    derivative = initial_derivative(xi)
    return derivative / (1.0 + time * derivative)


def initial_primitive(xi: float) -> float:
    """An antiderivative of the periodic initial state on the real lift."""
    return BASE * xi - (AMPLITUDE / (2.0 * math.pi)) * math.cos(
        2.0 * math.pi * (xi - PHASE)
    )


def conserved_primitive(xi: float, time: float) -> float:
    """Primitive whose characteristic difference gives Eulerian mass."""
    _validate_time(time)
    value = initial_value(xi)
    return initial_primitive(xi) + 0.5 * time * value * value


def exact_cell_average(left: float, right: float, time: float) -> float:
    """Return the exact physical average over one lifted Eulerian interval."""
    if not math.isfinite(left) or not math.isfinite(right) or right <= left:
        raise ValueError("cell faces must be finite and strictly ordered")
    left_foot = characteristic_foot(left, time)
    right_foot = characteristic_foot(right, time)
    mass = conserved_primitive(right_foot, time) - conserved_primitive(
        left_foot, time
    )
    return mass / (right - left)


def projected_state(
    projection: Projection,
    cells: int,
    time: float,
) -> tuple[float, ...]:
    """Return the exact FD nodal or FV cell-average discrete projection."""
    if projection not in {"fd", "fv"}:
        raise ValueError("projection must be 'fd' or 'fv'")
    if isinstance(cells, bool) or cells < 1:
        raise ValueError("cells must be a positive integer")
    spacing = 1.0 / cells
    if projection == "fd":
        return tuple(exact_point(index * spacing, time) for index in range(cells))
    return tuple(
        exact_cell_average(index * spacing, (index + 1) * spacing, time)
        for index in range(cells)
    )


def composite_simpson(
    function: Callable[[float], float],
    left: float,
    right: float,
    panels: int,
) -> float:
    """Independent deterministic quadrature used only to check the primitive."""
    if isinstance(panels, bool) or panels < 2 or panels % 2:
        raise ValueError("Simpson panels must be a positive even integer")
    spacing = (right - left) / panels
    total = function(left) + function(right)
    total += 4.0 * sum(
        function(left + index * spacing) for index in range(1, panels, 2)
    )
    total += 2.0 * sum(
        function(left + index * spacing) for index in range(2, panels, 2)
    )
    return total * spacing / 3.0


def exact_cell_average_by_quadrature(
    left: float,
    right: float,
    time: float,
    *,
    panels: int,
) -> float:
    return composite_simpson(
        lambda x: exact_point(x, time), left, right, panels
    ) / (right - left)
