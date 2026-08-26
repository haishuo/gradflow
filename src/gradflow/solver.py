"""Narrow, evidence-bound GradFlow solver surface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor

from .euler3d import (
    EULER_GAMMA,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    synchronize_duplicate_endpoints,
)


class UnsupportedProblemError(ValueError):
    """Raised when the requested mathematics is outside the validated slice."""


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot honor the complete problem contract."""


@dataclass(frozen=True)
class BackendDecision:
    """Inspectable placement decision; ordinary runs need not display it."""

    requested: str
    selected: str
    reason: str
    device: str


@dataclass(frozen=True)
class RunDiagnostics:
    """Metadata from the most recent successful run."""

    backend: BackendDecision
    steps: int
    simulated_time: float | Tensor
    hidden_device_transfers: int
    validation_device_synchronizations: int


class Solver:
    """Validated 3-D Euler characteristic JS-WENO-5 vertical slice.

    The deliberately narrow implementation supports only:

    - three-dimensional compressible Euler;
    - characteristic Jiang--Shu finite-difference WENO-5;
    - the preserved per-line global-LF policy;
    - duplicated periodic endpoints on all three axes;
    - float32; and
    - direct eager PyTorch on the caller's existing device.

    Unsupported requests fail rather than being silently approximated.
    """

    _PYTORCH_BACKENDS = {"auto", "pytorch", "pytorch-eager"}
    _NATIVE_BACKENDS = {"dveb", "cuda-native", "cpu-simd"}

    def __init__(
        self,
        *,
        equations: str,
        dimension: int,
        weno: tuple[str, int],
        flux_split: str,
        boundaries: str,
        dtype: torch.dtype,
        spacing: tuple[float, float, float] | None = None,
        backend: str = "auto",
    ) -> None:
        self.equations = equations
        self.dimension = dimension
        self.weno = weno
        self.flux_split = flux_split
        self.boundaries = boundaries
        self.dtype = dtype
        self.spacing = (
            self._validate_spacing(spacing) if spacing is not None else None
        )
        self.backend = self._normalize_backend(backend)
        self.last_run: RunDiagnostics | None = None
        self._validate_problem()

    def _validate_problem(self) -> None:
        equations = self.equations.lower().replace("_", "-")
        if equations not in {"euler", "compressible-euler"}:
            detail = (
                "Navier--Stokes viscous terms are not implemented"
                if equations in {"navier-stokes", "compressible-navier-stokes"}
                else "only compressible Euler is implemented"
            )
            raise UnsupportedProblemError(detail)
        if self.dimension != 3:
            raise UnsupportedProblemError("the Solver vertical slice is 3-D only")
        if not isinstance(self.weno, tuple) or len(self.weno) != 2:
            raise UnsupportedProblemError("weno must be the pair ('JS', 5)")
        family, order = self.weno
        if str(family).upper() not in {"JS", "JIANG-SHU"} or order != 5:
            raise UnsupportedProblemError(
                "only characteristic Jiang--Shu WENO-5 is implemented"
            )
        if self.flux_split.lower() != "global_lf":
            raise UnsupportedProblemError(
                "only the preserved per-line global_lf policy is implemented"
            )
        if self.boundaries.lower() != "periodic_duplicated":
            raise UnsupportedProblemError(
                "only periodic_duplicated boundaries are implemented"
            )
        if self.dtype is not torch.float32:
            raise UnsupportedProblemError("the matched Euler slice requires torch.float32")

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
        return backend.lower().replace("_", "-")

    @staticmethod
    def _validate_spacing(
        spacing: tuple[float, float, float] | Any,
    ) -> tuple[float, float, float]:
        try:
            raw_values = tuple(spacing)
        except TypeError as error:
            raise TypeError("spacing must contain three positive real values") from error
        if len(raw_values) != 3 or any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in raw_values
        ):
            raise TypeError("spacing must contain three positive real values")
        values = tuple(float(value) for value in raw_values)
        if len(values) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("spacing must contain three positive finite values")
        return values[0], values[1], values[2]

    def _validate_state(self, state: Tensor) -> None:
        if not isinstance(state, Tensor):
            raise TypeError("initial_state must already be a torch.Tensor")
        if state.dtype is not self.dtype:
            raise TypeError(f"initial_state must have dtype {self.dtype}")
        if state.ndim != 4 or state.shape[0] != 5:
            raise ValueError("initial_state layout must be (5, nz+1, ny+1, nx+1)")
        if any(size < 5 for size in state.shape[1:]):
            raise ValueError("each spatial axis needs at least four intervals")
        if state.layout is not torch.strided:
            raise ValueError("initial_state must use torch.strided layout")

    @staticmethod
    def _validate_physical_state(state: Tensor) -> int:
        """Validate finite positive density/pressure once, before integration.

        A CUDA input requires one declared scalar synchronization for this
        check. It occurs before, never inside, the numerical loop.
        """
        density = state[0]
        momentum = state[1:4]
        energy = state[4]
        pressure = (EULER_GAMMA - 1.0) * (
            energy - 0.5 * (momentum * momentum).sum(dim=0) / density
        )
        valid = (
            torch.isfinite(state).all()
            & (density > 0.0).all()
            & torch.isfinite(pressure).all()
            & (pressure > 0.0).all()
        )
        if not bool(valid.detach()):
            raise ValueError(
                "initial_state must contain finite positive density and pressure"
            )
        return int(state.device.type == "cuda")

    def _select_backend(self, requested: str, state: Tensor) -> BackendDecision:
        if requested in self._PYTORCH_BACKENDS:
            reason = (
                "direct PyTorch is the only arbitrary-state qualified backend"
                if requested == "auto"
                else "explicit direct-PyTorch request"
            )
            return BackendDecision(
                requested=requested,
                selected="pytorch-eager",
                reason=reason,
                device=str(state.device),
            )
        if requested in self._NATIVE_BACKENDS:
            raise BackendUnavailableError(
                "the screened DVEB executable always constructs its benchmark vortex; "
                "it has no arbitrary-state input ABI and is therefore ineligible for "
                "Solver.run(initial_state, ...). Use backend='pytorch-eager'."
            )
        raise BackendUnavailableError(f"unknown backend: {requested}")

    def explain_backend(
        self, initial_state: Tensor, *, backend: str | None = None
    ) -> BackendDecision:
        """Return the decision without executing or moving the state."""
        self._validate_state(initial_state)
        requested = self.backend if backend is None else self._normalize_backend(backend)
        return self._select_backend(requested, initial_state)

    def run(
        self,
        initial_state: Tensor,
        final_time: float | None = None,
        *,
        steps: int | None = None,
        spacing: tuple[float, float, float] | None = None,
        cfl: float = 0.1,
        backend: str | None = None,
        max_steps: int = 1_000_000,
    ) -> Tensor:
        """Advance a caller-provided state without an implicit device transfer.

        Exactly one of ``final_time`` or ``steps`` is required. Fixed ``steps``
        works on CPU or CUDA and keeps all CFL values on-device. Adaptive
        ``final_time`` control is currently CPU-only because a CUDA loop would
        require one device-to-host scalar synchronization per step.
        """
        self._validate_state(initial_state)
        if (final_time is None) == (steps is None):
            raise ValueError("provide exactly one of final_time or steps")
        if isinstance(cfl, bool) or not isinstance(cfl, Real):
            raise TypeError("cfl must be a real scalar, not a tensor")
        cfl_value = float(cfl)
        if not math.isfinite(cfl_value) or cfl_value <= 0.0:
            raise ValueError("cfl must be positive and finite")
        run_spacing = (
            self.spacing if spacing is None else self._validate_spacing(spacing)
        )
        if run_spacing is None:
            raise ValueError("provide spacing in Solver(...) or run(...)")
        final_time_value: float | None = None
        if steps is not None:
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                raise ValueError("steps must be a nonnegative integer")
        else:
            assert final_time is not None
            if isinstance(final_time, bool) or not isinstance(final_time, Real):
                raise TypeError("final_time must be a real scalar, not a tensor")
            final_time_value = float(final_time)
            if not math.isfinite(final_time_value) or final_time_value < 0.0:
                raise ValueError("final_time must be nonnegative and finite")
            if initial_state.device.type != "cpu":
                raise UnsupportedProblemError(
                    "final_time control on CUDA would require a hidden host scalar "
                    "transfer each step; use fixed steps for this vertical slice"
                )
            if (
                isinstance(max_steps, bool)
                or not isinstance(max_steps, int)
                or max_steps < 1
            ):
                raise ValueError("max_steps must be a positive integer")
        requested = self.backend if backend is None else self._normalize_backend(backend)
        decision = self._select_backend(requested, initial_state)
        validation_synchronizations = self._validate_physical_state(initial_state)

        state = synchronize_duplicate_endpoints(initial_state)
        if steps is not None:
            elapsed: Tensor = torch.zeros((), dtype=state.dtype, device=state.device)
            for _ in range(steps):
                dt = euler_cfl_timestep(state, run_spacing, cfl_value)
                state = euler_ssp_rk3_step(state, run_spacing, dt)
                elapsed = elapsed + dt
            completed_steps = steps
            simulated_time: float | Tensor = elapsed
        else:
            assert final_time_value is not None
            elapsed_host = 0.0
            completed_steps = 0
            while elapsed_host < final_time_value:
                if completed_steps >= max_steps:
                    raise RuntimeError("max_steps reached before final_time")
                dt = euler_cfl_timestep(state, run_spacing, cfl_value)
                remaining = dt.new_tensor(final_time_value - elapsed_host)
                dt = torch.minimum(dt, remaining)
                state = euler_ssp_rk3_step(state, run_spacing, dt)
                elapsed_host += float(dt.detach())
                completed_steps += 1
            simulated_time = elapsed_host

        self.last_run = RunDiagnostics(
            backend=decision,
            steps=completed_steps,
            simulated_time=simulated_time,
            hidden_device_transfers=0,
            validation_device_synchronizations=validation_synchronizations,
        )
        return state
