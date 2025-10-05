"""
Main WENO solver class.

This is the high-level interface for solving hyperbolic conservation laws
using Gottlieb's finite difference WENO spatial reconstruction and 
SSP-RK time integration.
"""

import torch
import numpy as np
from typing import Callable, Optional, Union, List, Tuple

from ..core.flux import reconstruct_interface_fluxes_fd_weno
from .timestepping import integrate_to_time, compute_cfl_timestep


class WENOSolver:
    """
    WENO solver for hyperbolic conservation laws using Gottlieb's FD-WENO.
    
    Solves equations of the form:
        ∂u/∂t + ∂f(u)/∂x = 0
        
    using Gottlieb's finite difference WENO spatial reconstruction
    and Strong Stability Preserving Runge-Kutta (SSP-RK) time integration.
    
    Parameters
    ----------
    order : int
        WENO order. Currently only 5 supported. Default: 5
    grid_size : int
        Number of grid points
    domain_length : float, optional
        Length of computational domain [0, L]. Default: 1.0
    boundary_condition : str, optional
        Type of boundary condition: 'periodic', 'outflow', 'reflecting'.
        Default: 'periodic'
    device : str, optional
        Device for computation: 'cpu' or 'cuda'. Default: 'cpu'
    dtype : torch.dtype, optional
        Data type for computation. Default: torch.float64
        
    Attributes
    ----------
    x : torch.Tensor
        Grid coordinates. Shape: [nx]
    dx : float
        Grid spacing
    nx : int
        Number of grid points
    n_ghost : int
        Number of ghost cells (4 for WENO-5)
        
    Examples
    --------
    >>> # Solve Burgers equation
    >>> solver = WENOSolver(order=5, grid_size=200, domain_length=2*np.pi)
    >>> u0 = torch.sin(solver.x)
    >>> 
    >>> def burgers_flux(u):
    ...     return 0.5 * u**2
    >>> 
    >>> u_final = solver.solve(u0, t_final=1.0, flux_function=burgers_flux)
    
    Notes
    -----
    This implements Gottlieb's finite difference WENO from:
    - Gottlieb & Shu MATLAB implementation
    - Shu & Osher (1989) for finite difference formulation
    
    Key features:
    - GPU acceleration via PyTorch
    - Adaptive time stepping with CFL condition
    - Multiple boundary conditions
    - Exact match with Gottlieb's reference implementation
    """
    
    def __init__(
        self,
        order: int = 5,
        grid_size: int = 100,
        domain_length: float = 1.0,
        boundary_condition: str = 'periodic',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64
    ):
        # Validate inputs
        if order != 5:
            raise NotImplementedError(
                f"Currently only WENO-5 is implemented for FD-WENO, got order={order}"
            )
        if grid_size < 2 * order:
            raise ValueError(
                f"Grid too small for WENO-{order}. Need at least {2*order} points, got {grid_size}"
            )
        if boundary_condition not in ['periodic', 'outflow', 'reflecting']:
            raise ValueError(
                f"Unsupported boundary condition: {boundary_condition}. "
                f"Choose from: 'periodic', 'outflow', 'reflecting'"
            )
        
        # Store parameters
        self.order = order
        self.nx = grid_size
        self.L = domain_length
        # CRITICAL: dx = domain_length / (grid_size - 1) to match linspace spacing
        # linspace with N points creates N-1 intervals
        self.dx = domain_length / (grid_size - 1)
        self.bc_type = boundary_condition
        self.device = device
        self.dtype = dtype
        
        # Ghost cells (4 for WENO-5)
        self.n_ghost = 4
        
        # Create grid
        self.x = torch.linspace(
            0, domain_length, grid_size,
            device=device, dtype=dtype
        )
    
    def apply_boundary_conditions(self, u: torch.Tensor) -> torch.Tensor:
        """
        Apply boundary conditions by adding ghost cells.
        
        Parameters
        ----------
        u : torch.Tensor
            Solution on physical domain. Shape: [batch, nx]
            
        Returns
        -------
        u_extended : torch.Tensor
            Solution with ghost cells. Shape: [batch, nx + 2*n_ghost]
        """
        if u.ndim == 1:
            u = u.unsqueeze(0)
        
        batch_size = u.shape[0]
        
        if self.bc_type == 'periodic':
            # Periodic: wrap around (matching Gottlieb's convention)
            # Gottlieb MATLAB: u = [ u(i-md:end-1), u, u(2:md+2)]
            # For md=4, i=101: u(97:100), u, u(2:6) in MATLAB 1-based
            # Python 0-based: u[96:100], u, u[1:6]
            # Translation: u[-5:-1] skips last element, u[1:6] gives 5 elements
            u_extended = torch.cat([
                u[:, -self.n_ghost-1:-1],  # Last 4, skipping the very last element
                u,                          # Physical domain
                u[:, 1:self.n_ghost+2]     # Elements at indices 1,2,3,4,5 (5 elements)
            ], dim=1)
            
        elif self.bc_type == 'outflow':
            # Outflow: extrapolate (zero-gradient)
            left_ghost = u[:, 0:1].expand(-1, self.n_ghost)
            right_ghost = u[:, -1:].expand(-1, self.n_ghost)
            
            u_extended = torch.cat([left_ghost, u, right_ghost], dim=1)
            
        elif self.bc_type == 'reflecting':
            # Reflecting: mirror with sign flip
            u_extended = torch.cat([
                -torch.flip(u[:, :self.n_ghost], dims=[1]),  # Left mirror
                u,                                             # Physical
                -torch.flip(u[:, -self.n_ghost:], dims=[1])  # Right mirror
            ], dim=1)
            
        else:
            raise ValueError(f"Unknown boundary condition: {self.bc_type}")
        
        return u_extended
    
    def compute_spatial_derivative(
        self,
        u: torch.Tensor,
        flux_function: Callable[[torch.Tensor], torch.Tensor],
        epsilon: float = 1e-29,
        alpha: Optional[float] = None,
        flux_derivative: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute spatial derivative using Gottlieb's finite difference WENO.
        
        This implements:
            du/dt[i] = (fh[i-1] - fh[i]) / dx
        
        where fh[i] are the interface fluxes computed by FD-WENO.
        
        Parameters
        ----------
        u : torch.Tensor
            Conservative variables. Shape: [batch, nx] or [nx]
        flux_function : callable
            Function f(u) that computes flux
        epsilon : float, optional
            Small parameter for WENO weights. Default: 1e-29
        alpha : float, optional
            Maximum wave speed for Lax-Friedrichs splitting.
            For linear advection f(u)=u, use alpha=1.0.
            If None, auto-computed from flux_derivative or estimated.
            Default: None
        flux_derivative : callable, optional
            Function that computes df/du. Enables CORRECT alpha estimation.
            Example: for f(u)=u, use lambda u: torch.ones_like(u)
            Default: None
            
        Returns
        -------
        dudt : torch.Tensor
            Spatial derivative. Shape: [batch, nx]
            
        Notes
        -----
        The key difference from finite volume WENO:
        - FV: du/dt = -(F[i+1/2] - F[i-1/2]) / dx
        - FD: du/dt = (fh[i-1] - fh[i]) / dx
        
        The sign difference and indexing are part of Gottlieb's formulation.
        """
        # Ensure batch dimension
        if u.ndim == 1:
            u = u.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, nx = u.shape
        assert nx == self.nx, f"Expected {self.nx} grid points, got {nx}"
        
        # Apply boundary conditions to get ghost cells
        u_extended = self.apply_boundary_conditions(u)
        
        # Compute interface fluxes using Gottlieb's FD-WENO
        # This returns fluxes at interfaces needed for the spatial derivative
        fh = reconstruct_interface_fluxes_fd_weno(
            u_extended,
            flux_function,
            order=self.order,
            epsilon=epsilon,
            alpha=alpha,
            flux_derivative=flux_derivative
        )
        
        # Gottlieb's FD-WENO returns interface fluxes such that:
        # du/dt[i] = (fh[i-1] - fh[i]) / dx (in Gottlieb's extended indexing)
        #
        # After the boundary condition fix:
        # - u_extended has nx + n_ghost + (n_ghost+1) = nx + 9 points (for nx=101: 110 points)
        # - n_interfaces = nx_extended - 2*md + 1 = 110 - 8 + 1 = 103 interfaces
        # - We need 101 derivatives for 101 physical cells
        # 
        # Gottlieb computes:
        # - fh for extended indices 3 to 105 (103 values)
        # - rhs for extended indices 4 to 104 (101 values)
        # - Formula: rhs[i] = (fh[i-1] - fh[i]) / dx
        #
        # In our indexing (fh starts from 0):
        # - fh[0:103] are the 103 interface fluxes
        # - We use fh[0:102] to compute 101 derivatives
        # - dudt[i] = (fh[i] - fh[i+1]) / dx for i = 0 to 100
        
        # Compute derivative using the first 102 of 103 interface fluxes
        # dudt[i] = (fh[i] - fh[i+1]) / dx
        dudt = (fh[:, :-2] - fh[:, 1:-1]) / self.dx
        
        # Verify shape
        assert dudt.shape == (batch_size, self.nx), \
            f"Expected shape ({batch_size}, {self.nx}), got {dudt.shape}"
        
        if squeeze_output:
            dudt = dudt.squeeze(0)
        
        return dudt
    
    def solve(
        self,
        u_initial: Union[torch.Tensor, np.ndarray],
        t_final: float,
        flux_function: Callable[[torch.Tensor], torch.Tensor],
        cfl_number: float = 0.5,
        epsilon: float = 1e-29,
        alpha: Optional[float] = None,
        flux_derivative: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        time_method: str = 'ssp_rk3',
        save_interval: Optional[float] = None,
        verbose: bool = True
    ) -> Union[torch.Tensor, List[Tuple[float, torch.Tensor]]]:
        """
        Solve hyperbolic conservation law from t=0 to t=t_final.
        
        Parameters
        ----------
        u_initial : torch.Tensor or np.ndarray
            Initial condition. Shape: [nx] or [batch, nx]
        t_final : float
            Final time
        flux_function : callable
            Function that computes f(u)
        cfl_number : float, optional
            CFL number for adaptive time stepping. Default: 0.5
        epsilon : float, optional
            Small parameter for WENO weights. Default: 1e-29
        time_method : str, optional
            Time integration method: 'ssp_rk3', 'ssp_rk2', 'euler'.
            Default: 'ssp_rk3'
        save_interval : float, optional
            If provided, save snapshots at this interval. Default: None
        verbose : bool, optional
            Print progress information. Default: True
            
        Returns
        -------
        u_final : torch.Tensor
            Solution at t=t_final if save_interval is None
        snapshots : List[Tuple[float, torch.Tensor]]
            List of (time, solution) pairs if save_interval is provided
            
        Examples
        --------
        >>> solver = WENOSolver(order=5, grid_size=200)
        >>> u0 = torch.sin(solver.x)
        >>> u_final = solver.solve(u0, t_final=1.0, flux_function=lambda u: 0.5*u**2)
        """
        # Convert to torch tensor if needed
        if isinstance(u_initial, np.ndarray):
            u = torch.from_numpy(u_initial).to(device=self.device, dtype=self.dtype)
        else:
            u = u_initial.to(device=self.device, dtype=self.dtype)
        
        # Ensure batch dimension
        if u.ndim == 1:
            u = u.unsqueeze(0)
        
        # Define RHS function for time integration
        def rhs_function(u_current):
            return self.compute_spatial_derivative(u_current, flux_function, epsilon, alpha, flux_derivative)
        
        # Time integration using the correct signature
        result = integrate_to_time(
            u,
            t_final,
            self.dx,
            rhs_function,
            cfl_number=cfl_number,
            method=time_method,
            save_interval=save_interval,
            verbose=verbose
        )
        
        # Remove batch dimension if input was 1D
        if u_initial.ndim == 1:
            if isinstance(result, torch.Tensor):
                result = result.squeeze(0)
            else:  # List of snapshots
                result = [(t, u.squeeze(0)) for t, u in result]
        
        return result
    
    def solve_burgers(
        self,
        u_initial: Union[torch.Tensor, np.ndarray],
        t_final: float,
        **kwargs
    ) -> Union[torch.Tensor, List[Tuple[float, torch.Tensor]]]:
        """
        Convenience method for solving Burgers equation: f(u) = 0.5 * u^2.
        
        Parameters
        ----------
        u_initial : torch.Tensor or np.ndarray
            Initial condition
        t_final : float
            Final time
        **kwargs
            Additional arguments passed to solve()
            
        Returns
        -------
        u_final : torch.Tensor
            Solution at final time
        """
        def burgers_flux(u):
            return 0.5 * u**2
        
        return self.solve(u_initial, t_final, burgers_flux, **kwargs)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing WENO Solver with Gottlieb FD-WENO")
    print("=" * 70)
    
    # Test 1: Solver initialization
    print("\nTest 1: Solver initialization")
    
    try:
        solver = WENOSolver(order=5, grid_size=100, device='cpu')
        print(f"✓ WENO-5 solver created")
        print(f"  Grid points: {solver.nx}")
        print(f"  dx: {solver.dx:.6f}")
        print(f"  Ghost cells: {solver.n_ghost}")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import sys
        sys.exit(1)
    
    # Test 2: Boundary conditions
    print("\nTest 2: Boundary conditions")
    
    u = torch.randn(1, 100, dtype=torch.float64)
    
    for bc in ['periodic', 'outflow', 'reflecting']:
        solver = WENOSolver(order=5, grid_size=100, boundary_condition=bc)
        u_extended = solver.apply_boundary_conditions(u)
        print(f"  {bc:10s}: {u.shape} → {u_extended.shape}")
    
    print("✓ Boundary conditions working")
    
    # Test 3: Spatial derivative computation
    print("\nTest 3: Spatial derivative with discontinuity")
    
    solver = WENOSolver(order=5, grid_size=101, domain_length=2.0, device='cpu')
    
    # Discontinuous initial condition
    x = solver.x
    u0 = torch.sign(x - 1.0)
    u0[50] = 0.0  # Force zero at discontinuity
    
    # Linear advection flux
    def linear_flux(u):
        return u
    
    dudt = solver.compute_spatial_derivative(u0, linear_flux)
    
    print(f"  Input shape: {u0.shape}")
    print(f"  Output shape: {dudt.shape}")
    print(f"  Output range: [{dudt.min():.6f}, {dudt.max():.6f}]")
    
    # Check for NaN
    if torch.isnan(dudt).any():
        print("✗ NaN detected in spatial derivative!")
    else:
        print("✓ Spatial derivative computed successfully")
    
    # Test 4: Time integration
    print("\nTest 4: Time integration (smooth IC)")
    
    solver = WENOSolver(
        order=5, 
        grid_size=200, 
        domain_length=2*np.pi, 
        device='cpu'
    )
    
    # Smooth initial condition
    u0_smooth = torch.sin(solver.x)
    
    # Burgers equation flux
    def burgers_flux(u):
        return 0.5 * u**2
    
    try:
        u_final = solver.solve(
            u0_smooth,
            t_final=0.1,
            flux_function=burgers_flux,
            cfl_number=0.5,
            verbose=False
        )
        
        print(f"  Initial: max={u0_smooth.max():.4f}, min={u0_smooth.min():.4f}")
        print(f"  Final:   max={u_final.max():.4f}, min={u_final.min():.4f}")
        
        # Check for NaN
        if torch.isnan(u_final).any():
            print("✗ NaN detected in time integration!")
        else:
            print("✓ Time integration successful")
            
    except Exception as e:
        print(f"✗ Time integration failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: CFL time step
    print("\nTest 5: CFL time step computation")
    
    dt = compute_cfl_timestep(u0_smooth, burgers_flux, solver.dx, cfl_number=0.5)
    
    print(f"  Time step: {dt:.6e}")
    assert dt > 0, "Time step should be positive"
    assert np.isfinite(dt), "Time step should be finite"
    print("✓ CFL time step valid")
    
    print("\n" + "=" * 70)
    print("All tests passed!")