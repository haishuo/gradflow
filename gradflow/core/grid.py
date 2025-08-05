"""
GradFlow Grid Module

GPU-optimized computational grid with differentiable coordinates.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class ComputationalGrid:
    """
    GPU-optimized grid with differentiable coordinates

    Manages the computational domain, coordinates, and cell metrics
    with full support for automatic differentiation.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        domain: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.0, 10.0),
            (0.0, 10.0),
        ),
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
        periodic: Tuple[bool, bool] = (False, False),
    ):
        """
        Initialize computational grid

        Args:
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction
            domain: ((x_min, x_max), (y_min, y_max))
            device: PyTorch device
            dtype: Data type (float64 for numerical precision)
            periodic: (periodic_x, periodic_y) boundary conditions
        """
        self.nx = nx
        self.ny = ny
        self.domain = domain
        self.device = torch.device(device)
        self.dtype = dtype
        self.periodic = periodic

        # Domain bounds
        self.x_min, self.x_max = domain[0]
        self.y_min, self.y_max = domain[1]

        # Grid spacing
        self.dx = (self.x_max - self.x_min) / nx
        self.dy = (self.y_max - self.y_min) / ny

        # Cell volumes (for finite volume method)
        self.dV = self.dx * self.dy

        # Generate coordinates
        self._generate_coordinates()

        # Precompute metrics for efficiency
        self._compute_metrics()

    def _generate_coordinates(self):
        """Generate grid coordinates (cell centers)"""
        # Cell-centered coordinates
        x_1d = torch.linspace(
            self.x_min + 0.5 * self.dx,
            self.x_max - 0.5 * self.dx,
            self.nx,
            device=self.device,
            dtype=self.dtype,
        )

        y_1d = torch.linspace(
            self.y_min + 0.5 * self.dy,
            self.y_max - 0.5 * self.dy,
            self.ny,
            device=self.device,
            dtype=self.dtype,
        )

        # 2D mesh
        self.x, self.y = torch.meshgrid(x_1d, y_1d, indexing="xy")

        # Store 1D coordinates for convenience
        self.x_1d = x_1d
        self.y_1d = y_1d

        # Interface coordinates (for flux computation)
        self.x_interfaces = torch.linspace(
            self.x_min, self.x_max, self.nx + 1, device=self.device, dtype=self.dtype
        )

        self.y_interfaces = torch.linspace(
            self.y_min, self.y_max, self.ny + 1, device=self.device, dtype=self.dtype
        )

    def _compute_metrics(self):
        """Precompute grid metrics for efficiency"""
        # For uniform grids, metrics are simple
        # For non-uniform/curvilinear grids, would compute Jacobians here

        # Aspect ratio
        self.aspect_ratio = self.dy / self.dx

        # CFL-related metrics
        self.min_spacing = min(self.dx, self.dy)
        self.max_spacing = max(self.dx, self.dy)

        # Grid quality metrics
        self.orthogonality = 1.0  # Perfect for Cartesian
        self.smoothness = 1.0  # Perfect for uniform

    def get_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get grid coordinates as 2D arrays"""
        return self.x, self.y

    def get_1d_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get 1D coordinate arrays"""
        return self.x_1d, self.y_1d

    def to_structured_array(self, u: torch.Tensor) -> torch.Tensor:
        """
        Convert conservative variables to structured array format

        Args:
            u: Conservative variables [..., ny, nx] or [..., ny*nx]

        Returns:
            Structured array [..., ny, nx]
        """
        if u.shape[-1] == self.nx * self.ny:
            # Reshape from flattened to 2D
            shape = list(u.shape[:-1]) + [self.ny, self.nx]
            return u.reshape(shape)
        elif u.shape[-2:] == (self.ny, self.nx):
            # Already in correct format
            return u
        else:
            raise ValueError(
                f"Invalid shape {u.shape} for grid ({self.ny}, {self.nx}\
    )"
            )

    def to_flattened_array(self, u: torch.Tensor) -> torch.Tensor:
        """
        Convert structured array to flattened format

        Args:
            u: Structured array [..., ny, nx]

        Returns:
            Flattened array [..., ny*nx]
        """
        if u.shape[-2:] == (self.ny, self.nx):
            shape = list(u.shape[:-2]) + [self.ny * self.nx]
            return u.reshape(shape)
        else:
            return u

    def apply_boundary_conditions(
        self, u: torch.Tensor, bc_type: str = "periodic"
    ) -> torch.Tensor:
        """
        Apply boundary conditions

        Args:
            u: Conservative variables [..., ny, nx]
            bc_type: 'periodic', 'reflective', 'outflow', or 'dirichlet'

        Returns:
            u with ghost cells filled
        """
        if bc_type == "periodic":
            # Periodic in both directions
            u = F.pad(u, (2, 2, 2, 2), mode="circular")
        elif bc_type == "reflective":
            # Reflective (solid wall) - flip normal velocity component
            u = self._apply_reflective_bc(u)
        elif bc_type == "outflow":
            # Zero-gradient extrapolation
            u = F.pad(u, (2, 2, 2, 2), mode="replicate")
        elif bc_type == "dirichlet":
            # Fixed values at boundaries (requires separate specification)
            u = self._apply_dirichlet_bc(u)
        else:
            raise ValueError(f"Unknown boundary condition: {bc_type}")

        return u

    def _apply_reflective_bc(self, u: torch.Tensor) -> torch.Tensor:
        """Apply reflective boundary conditions"""
        # This is problem-specific (depends on which component is velocity)
        # Placeholder implementation
        return F.pad(u, (2, 2, 2, 2), mode="replicate")

    def _apply_dirichlet_bc(self, u: torch.Tensor) -> torch.Tensor:
        """Apply Dirichlet boundary conditions"""
        # Requires boundary values to be specified
        # Placeholder implementation
        return F.pad(u, (2, 2, 2, 2), mode="constant", value=0.0)

    def compute_cfl_timestep(
        self, max_wave_speed: torch.Tensor, cfl_number: float = 0.3
    ) -> torch.Tensor:
        """
        Compute stable timestep using CFL condition

        Args:
            max_wave_speed: Maximum characteristic speed in domain
            cfl_number: CFL number (typically 0.1-0.9)

        Returns:
            Maximum stable timestep
        """
        dt = cfl_number * self.min_spacing / max_wave_speed
        return dt

    def match_fortran_grid(self) -> Dict[str, torch.Tensor]:
        """
        Generate grid matching FORTRAN reference exactly

        For validation against Chi-Wang Shu's implementation
        """
        # FORTRAN uses different indexing convention
        # Adjust coordinates to match exactly

        fortran_x = torch.zeros(
            self.nx + 1, self.ny + 1, device=self.device, dtype=self.dtype
        )
        fortran_y = torch.zeros(
            self.nx + 1, self.ny + 1, device=self.device, dtype=self.dtype
        )

        # FORTRAN grid points (vertices, not cell centers)
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                fortran_x[i, j] = self.x_min + i * self.dx
                fortran_y[i, j] = self.y_min + j * self.dy

        return {"x": fortran_x, "y": fortran_y, "dx": self.dx, "dy": self.dy}

    def interpolate_to_1d_cut(
        self, u: torch.Tensor, y_value: float, dim: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract 1D cut at specified y-value (for comparison with fort.8)

        Args:
            u: 2D field [ny, nx] or [4, ny, nx] for conservative variables
            y_value: y-coordinate for the cut
            dim: Which dimension contains variables (if multiple)

        Returns:
            (x_coords, interpolated_values)
        """
        # Find nearest y index
        y_idx = torch.argmin(torch.abs(self.y_1d - y_value))

        if len(u.shape) == 2:
            values = u[y_idx, :]
        elif len(u.shape) == 3 and dim == 0:
            values = u[:, y_idx, :]
        else:
            raise ValueError(f"Unsupported shape {u.shape}")

        return self.x_1d, values

    def to_fortran_format(self, u: torch.Tensor) -> np.ndarray:
        """
        Convert solution to FORTRAN output format

        Args:
            u: Conservative variables [4, ny, nx]

        Returns:
            Array matching FORTRAN fort.9 format
        """
        # FORTRAN outputs: x, y, rho, u, v, p
        # Need to convert from conservative to primitive variables

        rho = u[0]  # Density
        rho_u = u[1]  # x-momentum
        rho_v = u[2]  # y-momentum
        E = u[3]  # Total energy

        # Primitive variables
        u_vel = rho_u / rho
        v_vel = rho_v / rho

        # Pressure (assuming gamma = 1.4 for air)
        gamma = 1.4
        p = (gamma - 1) * (E - 0.5 * rho * (u_vel**2 + v_vel**2))

        # Flatten and stack for FORTRAN format
        nx, ny = self.nx, self.ny
        fortran_data = torch.zeros(nx * ny, 6, device=self.device, dtype=self.dtype)

        idx = 0
        for j in range(ny):
            for i in range(nx):
                fortran_data[idx, 0] = self.x[j, i]
                fortran_data[idx, 1] = self.y[j, i]
                fortran_data[idx, 2] = rho[j, i]
                fortran_data[idx, 3] = u_vel[j, i]
                fortran_data[idx, 4] = v_vel[j, i]
                fortran_data[idx, 5] = p[j, i]
                idx += 1

        return fortran_data.cpu().numpy()

    def info(self) -> Dict[str, Any]:
        """Get grid information summary"""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "domain": self.domain,
            "dx": self.dx.item(),
            "dy": self.dy.item(),
            "total_cells": self.nx * self.ny,
            "min_spacing": self.min_spacing.item(),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "periodic": self.periodic,
        }


# Convenience functions
def create_sod_shock_grid(device="cuda", dtype=torch.float64) -> ComputationalGrid:
    """Create grid matching FORTRAN Sod shock tube setup"""
    return ComputationalGrid(
        nx=20,
        ny=10,
        domain=((0.0, 10.0), (0.0, 10.0)),
        device=device,
        dtype=dtype,
        periodic=(False, False),
    )


def create_vortex_grid(
    nx=50, ny=50, device="cuda", dtype=torch.float64
) -> ComputationalGrid:
    """Create grid for isentropic vortex test"""
    return ComputationalGrid(
        nx=nx,
        ny=ny,
        domain=((0.0, 10.0), (0.0, 10.0)),
        device=device,
        dtype=dtype,
        periodic=(True, True),
    )
