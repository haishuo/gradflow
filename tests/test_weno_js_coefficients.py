from fractions import Fraction

import pytest

from gradflow import QUALIFIED_ORDERS, generate_weno_js_coefficients

F = Fraction


def _cell_average(offset: int, degree: int) -> Fraction:
    left = F(2 * offset - 1, 2)
    right = F(2 * offset + 1, 2)
    return (right ** (degree + 1) - left ** (degree + 1)) / (degree + 1)


def _quadratic_form(matrix, values) -> Fraction:
    return sum(
        values[row] * matrix[row][column] * values[column]
        for row in range(len(values))
        for column in range(len(values))
    )


def test_order_five_matches_known_jiang_shu_coefficients() -> None:
    coefficients = generate_weno_js_coefficients(5)
    assert coefficients.candidate_offsets == ((-2, -1, 0), (-1, 0, 1), (0, 1, 2))
    assert coefficients.candidate_coefficients == (
        (F(1, 3), F(-7, 6), F(11, 6)),
        (F(-1, 6), F(5, 6), F(1, 3)),
        (F(1, 3), F(5, 6), F(-1, 6)),
    )
    assert coefficients.optimal_weights == (F(1, 10), F(3, 5), F(3, 10))
    assert coefficients.full_coefficients == (
        F(1, 30),
        F(-13, 60),
        F(47, 60),
        F(9, 20),
        F(-1, 20),
    )
    assert coefficients.smoothness_matrices == (
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


@pytest.mark.parametrize("order", QUALIFIED_ORDERS)
def test_exact_polynomial_reproduction_and_optimal_weights(order: int) -> None:
    coefficients = generate_weno_js_coefficients(order)
    width = coefficients.substencil_width
    interface = F(1, 2)
    for offsets, candidate in zip(
        coefficients.candidate_offsets, coefficients.candidate_coefficients
    ):
        for degree in range(width):
            reconstructed = sum(
                weight * _cell_average(offset, degree)
                for offset, weight in zip(offsets, candidate)
            )
            assert reconstructed == interface**degree
    for degree in range(order):
        reconstructed = sum(
            weight * _cell_average(offset, degree)
            for offset, weight in zip(
                coefficients.full_offsets, coefficients.full_coefficients
            )
        )
        assert reconstructed == interface**degree
    assert sum(coefficients.optimal_weights) == 1
    assert all(weight > 0 for weight in coefficients.optimal_weights)


@pytest.mark.parametrize("order", QUALIFIED_ORDERS)
def test_smoothness_matrices_are_exact_symmetric_psd_factors(order: int) -> None:
    coefficients = generate_weno_js_coefficients(order)
    width = coefficients.substencil_width
    for matrix, factors in zip(
        coefficients.smoothness_matrices, coefficients.smoothness_factors
    ):
        assert matrix == tuple(tuple(row) for row in zip(*matrix))
        assert all(sum(row) == 0 for row in matrix)
        assert len(factors) == width - 1
        assert all(weight > 0 for weight, _ in factors)
        reconstructed = tuple(
            tuple(
                sum(weight * vector[row] * vector[column] for weight, vector in factors)
                for column in range(width)
            )
            for row in range(width)
        )
        assert reconstructed == matrix
        probes = (
            tuple(F(index + 1) for index in range(width)),
            tuple(F((-1) ** index * (index + 2)) for index in range(width)),
        )
        assert all(_quadratic_form(matrix, probe) >= 0 for probe in probes)


@pytest.mark.parametrize(
    ("value", "exception"),
    [(True, TypeError), (5.0, TypeError), (1, ValueError), (4, ValueError)],
)
def test_invalid_orders_are_refused(value: object, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        generate_weno_js_coefficients(value)  # type: ignore[arg-type]
