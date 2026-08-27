"""Generated odd-order finite-difference WENO-JS in ordinary PyTorch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

import torch

from .weno5 import DEFAULT_EPSILON
from .weno_js_coefficients import (
    WENOJSCoefficients,
    generate_weno_js_coefficients,
)

TensorFunction = Callable[[torch.Tensor], torch.Tensor]
QUALIFIED_ORDERS = (5, 7, 9, 11, 13, 15)
SMOOTHNESS_SCALE = 12.0
PRECISION_BLOCKS = (
    "flux_split",
    "candidates",
    "indicators",
    "weight_formation",
    "weight_normalization",
    "combination",
    "divergence",
)


@dataclass(frozen=True)
class WENOJSPrecisionPolicy:
    """Explicit compute precision for the mathematical WENO-JS blocks.

    ``None`` means the dtype of the persistent input state. A selected dtype
    changes computation precision only; it never changes device. The RHS is
    cast back to the persistent state dtype before it is returned.

    This policy is an experimental research surface. It makes precision
    assignments auditable and suitable for exhaustive enumeration; it does
    not assert that any mixed assignment is numerically safe.
    """

    flux_split: torch.dtype | None = None
    candidates: torch.dtype | None = None
    indicators: torch.dtype | None = None
    weight_formation: torch.dtype | None = None
    weight_normalization: torch.dtype | None = None
    combination: torch.dtype | None = None
    divergence: torch.dtype | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            dtype = getattr(self, field.name)
            if dtype not in (None, torch.float32, torch.float64):
                raise TypeError(
                    f"{field.name} precision must be float32, float64, or None"
                )

    def dtype_for(self, block: str, state_dtype: torch.dtype) -> torch.dtype:
        if block not in PRECISION_BLOCKS:
            raise ValueError(f"unknown WENO-JS precision block: {block}")
        selected = getattr(self, block)
        return state_dtype if selected is None else selected

    def as_names(self, state_dtype: torch.dtype = torch.float64) -> dict[str, str]:
        """Return a stable, JSON-ready block-to-dtype representation."""
        return {
            block: str(self.dtype_for(block, state_dtype)).removeprefix("torch.")
            for block in PRECISION_BLOCKS
        }


NATIVE_PRECISION = WENOJSPrecisionPolicy()


def _cast_dtype(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Change dtype without ever changing the tensor's device."""
    return values if values.dtype == dtype else values.to(dtype=dtype)


def _linear_combination(
    coefficients: tuple[float, ...], values: list[torch.Tensor]
) -> torch.Tensor:
    result = coefficients[0] * values[0]
    for coefficient, value in zip(coefficients[1:], values[1:]):
        result = result + coefficient * value
    return result


class WENOJS:
    """One generated scalar finite-difference Jiang--Shu WENO scheme.

    Construction performs exact rational mathematics once. Numerical calls
    contain only fixed Python loops, periodic rolls, and tensor elementwise
    operations; a fixed instance is suitable for ``torch.compile``.
    """

    def __init__(
        self,
        order: int,
        *,
        epsilon: float = DEFAULT_EPSILON,
        nonlinear_power: int = 2,
        precision: WENOJSPrecisionPolicy | None = None,
    ) -> None:
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("epsilon must be a positive Python scalar")
        if (
            isinstance(nonlinear_power, bool)
            or not isinstance(nonlinear_power, int)
            or nonlinear_power < 1
        ):
            raise ValueError("nonlinear_power must be a positive integer")
        exact = generate_weno_js_coefficients(order)
        self.order = exact.order
        self.substencil_width = exact.substencil_width
        self.epsilon = float(epsilon)
        self.nonlinear_power = nonlinear_power
        if precision is not None and not isinstance(precision, WENOJSPrecisionPolicy):
            raise TypeError("precision must be a WENOJSPrecisionPolicy or None")
        self.precision = NATIVE_PRECISION if precision is None else precision
        self.exact_coefficients: WENOJSCoefficients = exact
        self._candidate_offsets = exact.candidate_offsets
        self._candidate_coefficients = tuple(
            tuple(float(value) for value in candidate)
            for candidate in exact.candidate_coefficients
        )
        self._optimal_weights = tuple(float(value) for value in exact.optimal_weights)
        self._smoothness_factors = tuple(
            tuple(
                (
                    SMOOTHNESS_SCALE * float(weight),
                    tuple(float(value) for value in coefficients),
                )
                for weight, coefficients in factors
            )
            for factors in exact.smoothness_factors
        )

    @staticmethod
    def _axis(values: torch.Tensor, axis: int) -> int:
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must already be a torch.Tensor")
        if values.dtype not in (torch.float32, torch.float64):
            raise TypeError("WENO-JS requires float32 or float64 input")
        if values.ndim < 1:
            raise ValueError("values must have a spatial dimension")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("axis must be an integer")
        if not -values.ndim <= axis < values.ndim:
            raise ValueError("axis is outside the input rank")
        return axis % values.ndim

    @staticmethod
    def _shift(values: torch.Tensor, offset: int, axis: int) -> torch.Tensor:
        return torch.roll(values, shifts=-offset, dims=axis)

    def reconstruct(
        self,
        values: torch.Tensor,
        *,
        bias: str = "left",
        axis: int = -1,
    ) -> torch.Tensor:
        """Reconstruct the periodic interface value at ``i+1/2``.

        ``values`` are finite-difference flux samples on unique periodic nodes.
        ``bias="left"`` reconstructs from the upwind-left family;
        ``bias="right"`` uses its exact reflection about the interface.
        """
        normalized_axis = self._axis(values, axis)
        if values.shape[normalized_axis] < self.order:
            raise ValueError(
                f"WENO-JS order {self.order} requires at least {self.order} points"
            )
        if bias not in {"left", "right"}:
            raise ValueError("bias must be 'left' or 'right'")

        candidate_stencils = []
        for offsets in self._candidate_offsets:
            if bias == "left":
                stencil = [
                    self._shift(values, offset, normalized_axis) for offset in offsets
                ]
            else:
                stencil = [
                    self._shift(values, 1 - offset, normalized_axis)
                    for offset in offsets
                ]
            candidate_stencils.append(stencil)

        return self.reconstruct_stencils(candidate_stencils)

    def reconstruct_stencils(
        self,
        candidate_stencils: Sequence[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        """Reconstruct from candidate samples already aligned at each face.

        This is the system-facing form of :meth:`reconstruct`. Every sample
        tensor has the same shape, and each inner sequence follows the exact
        generated offset ordering for that candidate. It permits a caller to
        freeze a face-dependent characteristic projection before applying the
        same scalar WENO-JS algebra.
        """
        if len(candidate_stencils) != self.substencil_width:
            raise ValueError("wrong number of WENO-JS candidate stencils")

        first_stencil = candidate_stencils[0]
        if len(first_stencil) != self.substencil_width:
            raise ValueError("a WENO-JS candidate stencil has the wrong width")
        reference = first_stencil[0]
        if not isinstance(reference, torch.Tensor):
            raise TypeError("WENO-JS stencil samples must be torch.Tensor values")
        if reference.dtype not in (torch.float32, torch.float64):
            raise TypeError("WENO-JS requires float32 or float64 input")
        candidate_dtype = self.precision.dtype_for("candidates", reference.dtype)
        indicator_dtype = self.precision.dtype_for("indicators", reference.dtype)

        candidates = []
        indicators = []
        for stencil, coefficients, factors in zip(
            candidate_stencils,
            self._candidate_coefficients,
            self._smoothness_factors,
        ):
            if len(stencil) != self.substencil_width:
                raise ValueError("a WENO-JS candidate stencil has the wrong width")
            candidate_stencil = [
                _cast_dtype(value, candidate_dtype) for value in stencil
            ]
            indicator_stencil = [
                _cast_dtype(value, indicator_dtype) for value in stencil
            ]
            candidates.append(_linear_combination(coefficients, candidate_stencil))
            indicator = None
            for factor_weight, factor_coefficients in factors:
                factor = _linear_combination(
                    factor_coefficients, indicator_stencil
                )
                term = factor_weight * factor.square()
                indicator = term if indicator is None else indicator + term
            assert indicator is not None
            indicators.append(indicator)

        candidate_stack = torch.stack(candidates, dim=0)
        weight_dtype = self.precision.dtype_for(
            "weight_formation", reference.dtype
        )
        denominator = _cast_dtype(
            torch.stack(indicators, dim=0), weight_dtype
        ) + self.epsilon
        # Scaling by the smallest denominator preserves the normalized JS
        # weights while preventing float32 overflow on exactly constant data.
        scale = torch.amin(denominator, dim=0, keepdim=True)
        inverse_ratio = scale / denominator
        nonlinear = torch.stack(
            [
                weight * inverse_ratio[candidate].pow(self.nonlinear_power)
                for candidate, weight in enumerate(self._optimal_weights)
            ],
            dim=0,
        )
        normalization_dtype = self.precision.dtype_for(
            "weight_normalization", reference.dtype
        )
        nonlinear = _cast_dtype(nonlinear, normalization_dtype)
        weights = nonlinear / torch.sum(nonlinear, dim=0, keepdim=True)
        combination_dtype = self.precision.dtype_for(
            "combination", reference.dtype
        )
        weights = _cast_dtype(weights, combination_dtype)
        candidate_stack = _cast_dtype(candidate_stack, combination_dtype)
        return torch.sum(weights * candidate_stack, dim=0)

    def rhs(
        self,
        u: torch.Tensor,
        dx: float | torch.Tensor,
        flux: TensorFunction,
        flux_derivative: TensorFunction | None = None,
        *,
        alpha: float | torch.Tensor | None = None,
        axis: int = -1,
    ) -> torch.Tensor:
        """Compute a conservative periodic scalar RHS with global LF splitting."""
        normalized_axis = self._axis(u, axis)
        if u.shape[normalized_axis] < self.order:
            raise ValueError(
                f"WENO-JS order {self.order} requires at least {self.order} points"
            )
        state_dtype = u.dtype
        flux_dtype = self.precision.dtype_for("flux_split", state_dtype)
        flux_state = _cast_dtype(u, flux_dtype)
        physical_flux = flux(flux_state)
        if physical_flux.shape != u.shape:
            raise ValueError("flux(u) must have the same shape as u")
        if physical_flux.device != u.device:
            raise ValueError("flux(u) must remain on the input device")
        physical_flux = _cast_dtype(physical_flux, flux_dtype)
        if alpha is None:
            if flux_derivative is None:
                raise ValueError("provide flux_derivative when alpha is not explicit")
            alpha_value: float | torch.Tensor = torch.amax(
                torch.abs(flux_derivative(flux_state))
            )
        else:
            alpha_value = alpha
        if isinstance(alpha_value, torch.Tensor):
            if alpha_value.device != u.device:
                raise ValueError("alpha must remain on the input device")
            alpha_value = _cast_dtype(alpha_value, flux_dtype)
        positive = 0.5 * (physical_flux + alpha_value * flux_state)
        negative = 0.5 * (physical_flux - alpha_value * flux_state)
        interface_flux = self.reconstruct(
            positive, bias="left", axis=normalized_axis
        ) + self.reconstruct(negative, bias="right", axis=normalized_axis)
        divergence_dtype = self.precision.dtype_for("divergence", state_dtype)
        interface_flux = _cast_dtype(interface_flux, divergence_dtype)
        previous_interface = self._shift(interface_flux, -1, normalized_axis)
        result = (previous_interface - interface_flux) / dx
        return _cast_dtype(result, state_dtype)
