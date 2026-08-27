"""Independent exact-rational oracle for the Phase-2 scalar FV-WENO-JS5 contract.

This module intentionally uses only the Python standard library and does not
import GradFlow's finite-difference coefficient generator. Literal Jiang--Shu
tables and a separate moment-based derivation provide two inspectable lines of
evidence for the future PyTorch finite-volume implementation.
"""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Callable, Sequence


F = Fraction
RationalVector = tuple[Fraction, ...]
RationalMatrix = tuple[RationalVector, ...]

LEFT_OFFSETS = ((-2, -1, 0), (-1, 0, 1), (0, 1, 2))
LITERAL_CANDIDATES = (
    (F(1, 3), F(-7, 6), F(11, 6)),
    (F(-1, 6), F(5, 6), F(1, 3)),
    (F(1, 3), F(5, 6), F(-1, 6)),
)
LITERAL_OPTIMAL_WEIGHTS = (F(1, 10), F(3, 5), F(3, 10))
FULL_OFFSETS = (-2, -1, 0, 1, 2)
LITERAL_FULL = (
    F(1, 30),
    F(-13, 60),
    F(47, 60),
    F(9, 20),
    F(-1, 20),
)
LITERAL_SMOOTHNESS = (
    (
        (F(4, 3), F(-19, 6), F(11, 6)),
        (F(-19, 6), F(25, 3), F(-31, 6)),
        (F(11, 6), F(-31, 6), F(10, 3)),
    ),
    (
        (F(4, 3), F(-13, 6), F(5, 6)),
        (F(-13, 6), F(13, 3), F(-13, 6)),
        (F(5, 6), F(-13, 6), F(4, 3)),
    ),
    (
        (F(10, 3), F(-31, 6), F(11, 6)),
        (F(-31, 6), F(25, 3), F(-19, 6)),
        (F(11, 6), F(-19, 6), F(4, 3)),
    ),
)
MATCHED_EPSILON = F(1, 10**29)
SMOOTHNESS_SCALE = F(12)
NONLINEAR_POWER = 2


def solve_square(
    matrix: Sequence[Sequence[Fraction]], right: Sequence[Fraction]
) -> RationalVector:
    """Solve a non-singular square rational system by Gauss--Jordan elimination."""
    size = len(matrix)
    if size == 0 or len(right) != size:
        raise ValueError("invalid exact linear system")
    rows = [list(row) + [value] for row, value in zip(matrix, right)]
    if any(len(row) != size + 1 for row in rows):
        raise ValueError("matrix must be square")
    for column in range(size):
        pivot = next(
            (index for index in range(column, size) if rows[index][column]),
            None,
        )
        if pivot is None:
            raise ValueError("singular exact linear system")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for index in range(size):
            if index == column:
                continue
            multiplier = rows[index][column]
            rows[index] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(rows[index], rows[column])
            ]
    return tuple(row[-1] for row in rows)


def cell_average_monomial(offset: int, degree: int) -> Fraction:
    """Average x**degree over the unit cell centered at integer ``offset``."""
    left = F(2 * offset - 1, 2)
    right = F(2 * offset + 1, 2)
    exponent = degree + 1
    return (right**exponent - left**exponent) / exponent


def reconstruction_coefficients(offsets: tuple[int, ...]) -> RationalVector:
    """Derive cell-average-to-point coefficients at x=1/2 exactly."""
    moments = tuple(
        tuple(cell_average_monomial(offset, degree) for degree in range(len(offsets)))
        for offset in offsets
    )
    transpose = tuple(tuple(column) for column in zip(*moments))
    face_moments = tuple(F(1, 2) ** degree for degree in range(len(offsets)))
    return solve_square(transpose, face_moments)


def invert(matrix: RationalMatrix) -> RationalMatrix:
    size = len(matrix)
    columns = []
    for column in range(size):
        unit = tuple(F(int(row == column)) for row in range(size))
        columns.append(solve_square(matrix, unit))
    return tuple(
        tuple(columns[column][row] for column in range(size))
        for row in range(size)
    )


def falling_factorial(value: int, count: int) -> int:
    product = 1
    for index in range(count):
        product *= value - index
    return product


def centered_monomial_integral(degree: int) -> Fraction:
    if degree % 2:
        return F(0)
    return F(1, (2**degree) * (degree + 1))


def smoothness_matrix(offsets: tuple[int, ...]) -> RationalMatrix:
    """Derive the unit-cell Jiang--Shu derivative-integral quadratic form."""
    width = len(offsets)
    moments = tuple(
        tuple(cell_average_monomial(offset, degree) for degree in range(width))
        for offset in offsets
    )
    inverse = invert(moments)
    polynomial = [[F(0) for _ in range(width)] for _ in range(width)]
    for derivative in range(1, width):
        for left_degree in range(derivative, width):
            for right_degree in range(derivative, width):
                power = left_degree + right_degree - 2 * derivative
                polynomial[left_degree][right_degree] += F(
                    falling_factorial(left_degree, derivative)
                    * falling_factorial(right_degree, derivative)
                ) * centered_monomial_integral(power)
    result = []
    for row in range(width):
        result.append(
            tuple(
                sum(
                    inverse[left_degree][row]
                    * polynomial[left_degree][right_degree]
                    * inverse[right_degree][column]
                    for left_degree in range(width)
                    for right_degree in range(width)
                )
                for column in range(width)
            )
        )
    return tuple(result)


def derive_optimal_weights(
    candidates: tuple[RationalVector, ...],
) -> RationalVector:
    """Derive the three WENO-5 weights from independent full-stencil rows."""
    full = reconstruction_coefficients(FULL_OFFSETS)
    equations = []
    right = []
    for offset, target in zip(FULL_OFFSETS, full):
        equations.append(
            tuple(
                candidate[LEFT_OFFSETS[index].index(offset)]
                if offset in LEFT_OFFSETS[index]
                else F(0)
                for index, candidate in enumerate(candidates)
            )
        )
        right.append(target)
    # Three independent rows determine the weights; all five are checked later.
    weights = solve_square(tuple(equations[:3]), tuple(right[:3]))
    if any(
        sum(row[index] * weights[index] for index in range(3)) != target
        for row, target in zip(equations, right)
    ):
        raise ArithmeticError("candidate combination does not reproduce full stencil")
    return weights


def derive_all() -> dict[str, object]:
    candidates = tuple(reconstruction_coefficients(offsets) for offsets in LEFT_OFFSETS)
    full = reconstruction_coefficients(FULL_OFFSETS)
    smoothness = tuple(smoothness_matrix(offsets) for offsets in LEFT_OFFSETS)
    return {
        "candidate_coefficients": candidates,
        "optimal_weights": derive_optimal_weights(candidates),
        "full_coefficients": full,
        "smoothness_matrices": smoothness,
    }


def dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), F(0))


def quadratic_form(matrix: RationalMatrix, values: RationalVector) -> Fraction:
    return sum(
        values[row] * matrix[row][column] * values[column]
        for row in range(len(values))
        for column in range(len(values))
    )


def principal_minors(matrix: RationalMatrix) -> tuple[Fraction, ...]:
    """Return all principal minors of a symmetric 3x3 matrix."""
    one = tuple(matrix[index][index] for index in range(3))
    two = tuple(
        matrix[a][a] * matrix[b][b] - matrix[a][b] * matrix[b][a]
        for a, b in ((0, 1), (0, 2), (1, 2))
    )
    determinant = (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    return one + two + (determinant,)


def polynomial_cell_averages(
    coefficients: RationalVector, offsets: tuple[int, ...]
) -> RationalVector:
    return tuple(
        sum(
            coefficient * cell_average_monomial(offset, degree)
            for degree, coefficient in enumerate(coefficients)
        )
        for offset in offsets
    )


def polynomial_value(coefficients: RationalVector, location: Fraction) -> Fraction:
    terms = (
        coefficient * location**degree
        for degree, coefficient in enumerate(coefficients)
    )
    return sum(terms, F(0))


def periodic_sample(values: RationalVector, index: int) -> Fraction:
    return values[index % len(values)]


def aligned_stencil(
    values: RationalVector, face_cell: int, candidate: int, bias: str
) -> RationalVector:
    if bias not in {"left", "right"}:
        raise ValueError("bias must be left or right")
    offsets = LEFT_OFFSETS[candidate]
    if bias == "right":
        offsets = tuple(1 - offset for offset in offsets)
    return tuple(periodic_sample(values, face_cell + offset) for offset in offsets)


def js5_reconstruct(
    values: RationalVector,
    face_cell: int,
    *,
    bias: str,
    epsilon: Fraction = MATCHED_EPSILON,
) -> Fraction:
    candidates = []
    indicators = []
    for index in range(3):
        stencil = aligned_stencil(values, face_cell, index, bias)
        candidates.append(dot(LITERAL_CANDIDATES[index], stencil))
        indicators.append(
            SMOOTHNESS_SCALE * quadratic_form(LITERAL_SMOOTHNESS[index], stencil)
        )
    unnormalized = tuple(
        weight / (epsilon + indicator) ** NONLINEAR_POWER
        for weight, indicator in zip(LITERAL_OPTIMAL_WEIGHTS, indicators)
    )
    total = sum(unnormalized, F(0))
    weights = tuple(value / total for value in unnormalized)
    return dot(weights, candidates)


def periodic_face_states(
    values: RationalVector,
) -> tuple[RationalVector, RationalVector]:
    left = tuple(
        js5_reconstruct(values, index, bias="left")
        for index in range(len(values))
    )
    right = tuple(
        js5_reconstruct(values, index, bias="right")
        for index in range(len(values))
    )
    return left, right


def periodic_rusanov_rhs(
    values: RationalVector,
    spacing: Fraction,
    flux: Callable[[Fraction], Fraction],
    alpha: Fraction,
) -> tuple[RationalVector, RationalVector, RationalVector, RationalVector]:
    left, right = periodic_face_states(values)
    face_flux = tuple(
        (flux(left_value) + flux(right_value) - alpha * (right_value - left_value))
        / 2
        for left_value, right_value in zip(left, right)
    )
    rhs = tuple(
        -(face_flux[index] - face_flux[index - 1]) / spacing
        for index in range(len(values))
    )
    return rhs, face_flux, left, right


def fourier_cell_average(
    left: float,
    right: float,
    *,
    sine_amplitude: float,
    cosine_amplitude: float,
    wavenumber: float,
) -> float:
    width = right - left
    sine = sine_amplitude * (
        math.cos(wavenumber * left) - math.cos(wavenumber * right)
    ) / (wavenumber * width)
    cosine = cosine_amplitude * (
        math.sin(wavenumber * right) - math.sin(wavenumber * left)
    ) / (wavenumber * width)
    return sine + cosine


def composite_simpson_average(
    function: Callable[[float], float], left: float, right: float, panels: int = 4096
) -> float:
    if panels <= 0 or panels % 2:
        raise ValueError("Simpson panels must be a positive even integer")
    step = (right - left) / panels
    odd = math.fsum(function(left + index * step) for index in range(1, panels, 2))
    even = math.fsum(function(left + index * step) for index in range(2, panels, 2))
    integral = step * (
        function(left) + function(right) + 4.0 * odd + 2.0 * even
    ) / 3.0
    return integral / (right - left)
