"""Exact-rational construction of finite-difference WENO-JS coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

RationalVector = tuple[Fraction, ...]
RationalMatrix = tuple[RationalVector, ...]
SmoothnessFactor = tuple[Fraction, RationalVector]


@dataclass(frozen=True)
class WENOJSCoefficients:
    """All exact mathematical data for one odd-order WENO-JS scheme."""

    order: int
    substencil_width: int
    candidate_offsets: tuple[tuple[int, ...], ...]
    candidate_coefficients: tuple[RationalVector, ...]
    optimal_weights: RationalVector
    full_offsets: tuple[int, ...]
    full_coefficients: RationalVector
    smoothness_matrices: tuple[RationalMatrix, ...]
    smoothness_factors: tuple[tuple[SmoothnessFactor, ...], ...]


def _zero_matrix(rows: int, columns: int) -> list[list[Fraction]]:
    return [[Fraction(0) for _ in range(columns)] for _ in range(rows)]


def _solve_square(
    matrix: list[list[Fraction]], right_hand_side: list[Fraction]
) -> list[Fraction]:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("expected a nonempty square matrix")
    if len(right_hand_side) != size:
        raise ValueError("right-hand side has the wrong length")
    augmented = [row[:] + [value] for row, value in zip(matrix, right_hand_side)]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]), None
        )
        if pivot is None:
            raise ValueError("singular exact system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column or not augmented[row][column]:
                continue
            scale = augmented[row][column]
            augmented[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _solve_consistent(
    matrix: list[list[Fraction]], right_hand_side: list[Fraction]
) -> list[Fraction]:
    """Solve an exact full-column-rank, possibly overdetermined system."""
    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    if not rows or not columns or any(len(row) != columns for row in matrix):
        raise ValueError("expected a nonempty rectangular matrix")
    if len(right_hand_side) != rows:
        raise ValueError("right-hand side has the wrong length")
    augmented = [row[:] + [value] for row, value in zip(matrix, right_hand_side)]
    pivot_rows: dict[int, int] = {}
    active_row = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(active_row, rows) if augmented[row][column]),
            None,
        )
        if pivot is None:
            continue
        augmented[active_row], augmented[pivot] = (
            augmented[pivot],
            augmented[active_row],
        )
        scale = augmented[active_row][column]
        augmented[active_row] = [value / scale for value in augmented[active_row]]
        for row in range(rows):
            if row == active_row or not augmented[row][column]:
                continue
            scale = augmented[row][column]
            augmented[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[active_row])
            ]
        pivot_rows[column] = active_row
        active_row += 1
    if len(pivot_rows) != columns:
        raise ValueError("exact system does not have a unique solution")
    for row in augmented:
        if all(value == 0 for value in row[:-1]) and row[-1] != 0:
            raise ValueError("exact system is inconsistent")
    return [augmented[pivot_rows[column]][-1] for column in range(columns)]


def _inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    size = len(matrix)
    columns = []
    for column in range(size):
        unit = [Fraction(int(row == column)) for row in range(size)]
        columns.append(_solve_square(matrix, unit))
    return [[columns[column][row] for column in range(size)] for row in range(size)]


def _cell_average_moment(offset: int, degree: int) -> Fraction:
    left = Fraction(2 * offset - 1, 2)
    right = Fraction(2 * offset + 1, 2)
    exponent = degree + 1
    return (right**exponent - left**exponent) / exponent


def _cell_average_matrix(offsets: tuple[int, ...]) -> list[list[Fraction]]:
    return [
        [_cell_average_moment(offset, degree) for degree in range(len(offsets))]
        for offset in offsets
    ]


def _interface_reconstruction(offsets: tuple[int, ...]) -> RationalVector:
    matrix = _cell_average_matrix(offsets)
    evaluation = [Fraction(1, 2) ** degree for degree in range(len(offsets))]
    # c^T = evaluation^T A^-1, equivalently A^T c = evaluation.
    transpose = [list(column) for column in zip(*matrix)]
    return tuple(_solve_square(transpose, evaluation))


def _falling_factorial(value: int, count: int) -> int:
    result = 1
    for factor in range(count):
        result *= value - factor
    return result


def _centered_monomial_integral(degree: int) -> Fraction:
    if degree % 2:
        return Fraction(0)
    return Fraction(1, (2**degree) * (degree + 1))


def _smoothness_matrix(offsets: tuple[int, ...]) -> RationalMatrix:
    width = len(offsets)
    inverse = _inverse(_cell_average_matrix(offsets))
    polynomial_form = _zero_matrix(width, width)
    for derivative in range(1, width):
        for left_degree in range(derivative, width):
            left_scale = _falling_factorial(left_degree, derivative)
            for right_degree in range(derivative, width):
                right_scale = _falling_factorial(right_degree, derivative)
                power = left_degree + right_degree - 2 * derivative
                polynomial_form[left_degree][right_degree] += (
                    left_scale * right_scale * _centered_monomial_integral(power)
                )

    result = _zero_matrix(width, width)
    for row in range(width):
        for column in range(width):
            result[row][column] = sum(
                inverse[left_degree][row]
                * polynomial_form[left_degree][right_degree]
                * inverse[right_degree][column]
                for left_degree in range(width)
                for right_degree in range(width)
            )
    return tuple(tuple(row) for row in result)


def _ldlt_factors(matrix: RationalMatrix) -> tuple[SmoothnessFactor, ...]:
    """Return exact nonzero terms of ``matrix = L D L^T``."""
    size = len(matrix)
    lower = _zero_matrix(size, size)
    diagonal = [Fraction(0) for _ in range(size)]
    for column in range(size):
        diagonal[column] = matrix[column][column] - sum(
            lower[column][prior] ** 2 * diagonal[prior] for prior in range(column)
        )
        if diagonal[column] < 0:
            raise ArithmeticError("smoothness matrix is not positive semidefinite")
        lower[column][column] = Fraction(1)
        for row in range(column + 1, size):
            residual = matrix[row][column] - sum(
                lower[row][prior] * lower[column][prior] * diagonal[prior]
                for prior in range(column)
            )
            if diagonal[column]:
                lower[row][column] = residual / diagonal[column]
            elif residual:
                raise ArithmeticError("zero LDL pivot has a nonzero residual")
    return tuple(
        (diagonal[column], tuple(lower[row][column] for row in range(size)))
        for column in range(size)
        if diagonal[column]
    )


@lru_cache(maxsize=None)
def generate_weno_js_coefficients(order: int) -> WENOJSCoefficients:
    """Construct exact finite-difference WENO-JS data for an odd design order."""
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("WENO-JS order must be an integer")
    if order < 3 or order % 2 == 0:
        raise ValueError("WENO-JS order must be odd and at least three")
    width = (order + 1) // 2
    candidate_offsets = tuple(
        tuple(range(-width + 1 + candidate, candidate + 1))
        for candidate in range(width)
    )
    candidate_coefficients = tuple(
        _interface_reconstruction(offsets) for offsets in candidate_offsets
    )
    full_offsets = tuple(range(-width + 1, width))
    full_coefficients = _interface_reconstruction(full_offsets)

    expanded = _zero_matrix(len(full_offsets), width)
    offset_rows = {offset: row for row, offset in enumerate(full_offsets)}
    for candidate, (offsets, coefficients) in enumerate(
        zip(candidate_offsets, candidate_coefficients)
    ):
        for offset, coefficient in zip(offsets, coefficients):
            expanded[offset_rows[offset]][candidate] = coefficient
    optimal_weights = tuple(_solve_consistent(expanded, list(full_coefficients)))
    if any(weight <= 0 for weight in optimal_weights):
        raise ArithmeticError("optimal WENO-JS weights are not all positive")
    if sum(optimal_weights) != 1:
        raise ArithmeticError("optimal WENO-JS weights do not sum to one")

    smoothness_matrices = tuple(
        _smoothness_matrix(offsets) for offsets in candidate_offsets
    )
    smoothness_factors = tuple(_ldlt_factors(matrix) for matrix in smoothness_matrices)
    return WENOJSCoefficients(
        order=order,
        substencil_width=width,
        candidate_offsets=candidate_offsets,
        candidate_coefficients=candidate_coefficients,
        optimal_weights=optimal_weights,
        full_offsets=full_offsets,
        full_coefficients=full_coefficients,
        smoothness_matrices=smoothness_matrices,
        smoothness_factors=smoothness_factors,
    )
