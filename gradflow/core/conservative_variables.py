"""
GradFlow Conservative Variables Module

Differentiable conservative variable management for Euler equations.
"""

from typing import Dict, Tuple

import numpy as np
import torch


class ConservativeState:
    """
    Differentiable conservative variables [ρ, ρu, ρv, E]

    Manages the state vector for 2D Euler equations with full
    automatic differentiation support.
    """

    def __init__(
        self,
        density: torch.Tensor,
        momentum_x: torch.Tensor,
        momentum_y: torch.Tensor,
        energy: torch.Tensor,
        gamma: float = 1.4,
        requires_grad: bool = True,
    ):
        """
        Initialize conservative state

        Args:
            density: Density field ρ [..., ny, nx]
            momentum_x: x-momentum ρu [..., ny, nx]
            momentum_y: y-momentum ρv [..., ny, nx]
            energy: Total energy E [..., ny, nx]
            gamma: Specific heat ratio (1.4 for air)
            requires_grad: Enable gradient computation
        """
        # Ensure all tensors have gradients if requested
        if requires_grad:
            density = (
                density.requires_grad_(True) if not density.requires_grad else densit\
    y
            )
            momentum_x = (
                momentum_x.requires_grad_(True)
                if not momentum_x.requires_grad
                else momentum_x
            )
            momentum_y = (
                momentum_y.requires_grad_(True)
                if not momentum_y.requires_grad
                else momentum_y
            )
            energy = energy.requires_grad_(True) if not energy.requires_grad else ene\
    rgy

        self.rho = density
        self.rho_u = momentum_x
        self.rho_v = momentum_y
        self.E = energy
        self.gamma = gamma

        # Validate shapes
        self._validate_shapes()

        # Cache primitive variables (computed lazily)
        self._u_vel = None
        self._v_vel = None
        self._p = None

    def _validate_shapes(self):
        """Ensure all components have consistent shapes"""
        shape = self.rho.shape
        if not all(
            tensor.shape == shape for tensor in [self.rho_u, self.rho_v, self.E]
        ):
            raise ValueError("All conservative variables must have the same shape")

        # Ensure we're using the correct dtype and device
        device = self.rho.device
        dtype = self.rho.dtype

        for tensor in [self.rho_u, self.rho_v, self.E]:
            if tensor.device != device:
                raise ValueError("All tensors must be on the same device")
            if tensor.dtype != dtype:
                raise ValueError("All tensors must have the same dtype")

    @property
    def shape(self) -> torch.Size:
        """Get shape of state variables"""
        return self.rho.shape

    @property
    def device(self) -> torch.device:
        """Get device of state variables"""
        return self.rho.device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of state variables"""
        return self.rho.dtype

    @property
    def u_velocity(self) -> torch.Tensor:
        """Get x-velocity (computed from momentum)"""
        if self._u_vel is None or self._u_vel.shape != self.rho.shape:
            self._u_vel = self.rho_u / self.rho
        return self._u_vel

    @property
    def v_velocity(self) -> torch.Tensor:
        """Get y-velocity (computed from momentum)"""
        if self._v_vel is None or self._v_vel.shape != self.rho.shape:
            self._v_vel = self.rho_v / self.rho
        return self._v_vel

    @property
    def pressure(self) -> torch.Tensor:
        """Get pressure (computed from energy)"""
        if self._p is None or self._p.shape != self.rho.shape:
            kinetic_energy = 0.5 * (self.rho_u**2 + self.rho_v**2) / self.rho
            self._p = (self.gamma - 1) * (self.E - kinetic_energy)
        return self._p

    @property
    def sound_speed(self) -> torch.Tensor:
        """Get local sound speed"""
        return torch.sqrt(self.gamma * self.pressure / self.rho)

    @property
    def temperature(self) -> torch.Tensor:
        """Get temperature (non-dimensional)"""
        # For perfect gas: p = ρRT, with R=1 for non-dimensional form
        return self.pressure / self.rho

    @property
    def mach_number(self) -> torch.Tensor:
        """Get local Mach number"""
        velocity_magnitude = torch.sqrt(self.u_velocity**2 + self.v_velocity**2)
        return velocity_magnitude / self.sound_speed

    def to_tensor(self) -> torch.Tensor:
        """
        Convert to single tensor [4, ...]

        Returns:
            Stacked tensor with shape [4, ny, nx]
        """
        return torch.stack([self.rho, self.rho_u, self.rho_v, self.E], dim=0)

    @classmethod
    def from_tensor(
        cls, u: torch.Tensor, gamma: float = 1.4, requires_grad: bool = True
    ) -> "ConservativeState":
        """
        Create from single tensor

        Args:
            u: Conservative variables [4, ...]
            gamma: Specific heat ratio
            requires_grad: Enable gradients

        Returns:
            ConservativeState instance
        """
        if u.shape[0] != 4:
            raise ValueError(f"Expected 4 conservative variables, got {u.shape[0]}")

        return cls(
            density=u[0],
            momentum_x=u[1],
            momentum_y=u[2],
            energy=u[3],
            gamma=gamma,
            requires_grad=requires_grad,
        )

    @classmethod
    def from_primitive(
        cls,
        density: torch.Tensor,
        u_velocity: torch.Tensor,
        v_velocity: torch.Tensor,
        pressure: torch.Tensor,
        gamma: float = 1.4,
        requires_grad: bool = True,
    ) -> "ConservativeState":
        """
        Create from primitive variables

        Args:
            density: Density ρ
            u_velocity: x-velocity u
            v_velocity: y-velocity v
            pressure: Pressure p
            gamma: Specific heat ratio
            requires_grad: Enable gradients

        Returns:
            ConservativeState instance
        """
        # Convert to conservative variables
        momentum_x = density * u_velocity
        momentum_y = density * v_velocity
        kinetic_energy = 0.5 * density * (u_velocity**2 + v_velocity**2)
        energy = pressure / (gamma - 1) + kinetic_energy

        return cls(
            density=density,
            momentum_x=momentum_x,
            momentum_y=momentum_y,
            energy=energy,
            gamma=gamma,
            requires_grad=requires_grad,
        )

    def compute_flux_x(self) -> torch.Tensor:
        """
        Compute x-direction flux F(U)

        For Euler equations:
        F = [ρu, ρu² + p, ρuv, u(E + p)]

        Returns:
            Flux tensor [4, ...]
        """
        u = self.u_velocity
        p = self.pressure

        flux = torch.stack(
            [
                self.rho_u,  # Continuity flux
                self.rho_u * u + p,  # x-momentum flux
                self.rho_u * self.v_velocity,  # y-momentum flux
                u * (self.E + p),  # Energy flux
            ],
            dim=0,
        )

        return flux

    def compute_flux_y(self) -> torch.Tensor:
        """
        Compute y-direction flux G(U)

        For Euler equations:
        G = [ρv, ρuv, ρv² + p, v(E + p)]

        Returns:
            Flux tensor [4, ...]
        """
        v = self.v_velocity
        p = self.pressure

        flux = torch.stack(
            [
                self.rho_v,  # Continuity flux
                self.rho_v * self.u_velocity,  # x-momentum flux
                self.rho_v * v + p,  # y-momentum flux
                v * (self.E + p),  # Energy flux
            ],
            dim=0,
        )

        return flux

    def max_wave_speed(self) -> torch.Tensor:
        """
        Compute maximum characteristic wave speed

        For Euler equations: max(|u| + c, |v| + c)

        Returns:
            Maximum wave speed (scalar or field)
        """
        c = self.sound_speed
        wave_speed_x = torch.abs(self.u_velocity) + c
        wave_speed_y = torch.abs(self.v_velocity) + c

        return torch.maximum(wave_speed_x, wave_speed_y)

    def apply_limiter(self, limiter_func) -> "ConservativeState":
        """
        Apply a limiter function to the state

        Args:
            limiter_func: Function to limit conservative variables

        Returns:
            New limited state
        """
        limited_u = limiter_func(self.to_tensor())
        return self.from_tensor(limited_u, self.gamma)

    def clone(self) -> "ConservativeState":
        """Create a deep copy of the state"""
        return ConservativeState(
            density=self.rho.clone(),
            momentum_x=self.rho_u.clone(),
            momentum_y=self.rho_v.clone(),
            energy=self.E.clone(),
            gamma=self.gamma,
            requires_grad=self.rho.requires_grad,
        )

    def detach(self) -> "ConservativeState":
        """Detach from computation graph"""
        return ConservativeState(
            density=self.rho.detach(),
            momentum_x=self.rho_u.detach(),
            momentum_y=self.rho_v.detach(),
            energy=self.E.detach(),
            gamma=self.gamma,
            requires_grad=False,
        )

    def to_fortran_primitive(self) -> Dict[str, torch.Tensor]:
        """
        Convert to primitive variables matching FORTRAN output

        Returns:
            Dictionary with 'rho', 'u', 'v', 'p' fields
        """
        return {
            "rho": self.rho,
            "u": self.u_velocity,
            "v": self.v_velocity,
            "p": self.pressure,
        }

    def validate_physical(self, tolerance: float = 1e-10) -> Dict[str, bool]:
        """
        Check if state is physically valid

        Args:
            tolerance: Numerical tolerance for positivity

        Returns:
            Dictionary of validation results
        """
        results = {}

        # Density must be positive
        results["positive_density"] = torch.all(self.rho > tolerance).item()

        # Pressure must be positive
        results["positive_pressure"] = torch.all(self.pressure > tolerance).item()

        # Energy must be sufficient for positive pressure
        kinetic_energy = 0.5 * (self.rho_u**2 + self.rho_v**2) / self.rho
        internal_energy = self.E - kinetic_energy
        results["positive_internal_energy"] = torch.all(
            internal_energy > tolerance
        ).item()

        # Temperature must be positive
        results["positive_temperature"] = torch.all(self.temperature > tolerance).ite\
    m()

        # All checks passed
        results["is_valid"] = all(results.values())

        return results


# Initial condition generators for testing
def create_sod_shock_state(
    grid, gamma: float = 1.4, requires_grad: bool = True
) -> ConservativeState:
    """
    Create Sod shock tube initial condition

    Left state (x < 5.0):  ρ=1.0,   p=1.0
    Right state (x ≥ 5.0): ρ=0.125, p=0.1
    """
    device = grid.device
    dtype = grid.dtype

    # Initialize primitive variables
    rho = torch.ones(grid.ny, grid.nx, device=device, dtype=dtype)
    u = torch.zeros(grid.ny, grid.nx, device=device, dtype=dtype)
    v = torch.zeros(grid.ny, grid.nx, device=device, dtype=dtype)
    p = torch.ones(grid.ny, grid.nx, device=device, dtype=dtype)

    # Apply shock discontinuity
    x_coords, _ = grid.get_coordinates()
    right_region = x_coords >= 5.0

    rho[right_region] = 0.125
    p[right_region] = 0.1

    # Convert to conservative state
    return ConservativeState.from_primitive(
        density=rho,
        u_velocity=u,
        v_velocity=v,
        pressure=p,
        gamma=gamma,
        requires_grad=requires_grad,
    )


def create_isentropic_vortex_state(
    grid,
    vortex_strength: float = 5.0,
    vortex_center: Tuple[float, float] = (5.0, 5.0),
    gamma: float = 1.4,
    requires_grad: bool = True,
) -> ConservativeState:
    """
    Create isentropic vortex initial condition

    Smooth test case for verifying high-order accuracy
    """
    device = grid.device
    dtype = grid.dtype

    x, y = grid.get_coordinates()
    x0, y0 = vortex_center

    # Vortex parameters
    beta = vortex_strength
    r2 = (x - x0) ** 2 + (y - y0) ** 2

    # Base flow
    rho_inf = 1.0
    u_inf = 1.0
    v_inf = 1.0
    p_inf = 1.0

    # Vortex perturbation
    exp_term = torch.exp(0.5 * (1 - r2))

    # Temperature perturbation
    dT = -(gamma - 1) * beta**2 / (8 * gamma * np.pi**2) * exp_term
    T = 1 + dT

    # Velocity perturbation
    du = -beta / (2 * np.pi) * (y - y0) * exp_term
    dv = beta / (2 * np.pi) * (x - x0) * exp_term

    # Primitive variables
    rho = T ** (1 / (gamma - 1)) * rho_inf
    u = u_inf + du
    v = v_inf + dv
    p = T ** (gamma / (gamma - 1)) * p_inf

    return ConservativeState.from_primitive(
        density=rho,
        u_velocity=u,
        v_velocity=v,
        pressure=p,
        gamma=gamma,
        requires_grad=requires_grad,
    )
