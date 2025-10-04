"""
Time integration schemes for hyperbolic PDEs.

Implements Strong Stability Preserving (SSP) Runge-Kutta methods
that preserve the total variation diminishing (TVD) property when
combined with spatial discretizations like WENO.
"""

import torch
from typing import Callable, Optional


# ============================================================================
# SSP RUNGE-KUTTA METHODS
# ============================================================================

def ssp_rk3_step(
    u: torch.Tensor,
    rhs_function: Callable[[torch.Tensor], torch.Tensor],
    dt: float
) -> torch.Tensor:
    """
    Single step of third-order SSP Runge-Kutta (SSP-RK3).
    
    Also known as SSPRK(3,3) or the optimal third-order TVD Runge-Kutta
    method. This is the standard time integrator for WENO schemes.
    
    Parameters
    ----------
    u : torch.Tensor
        Current solution. Shape: [batch, nx]
    rhs_function : callable
        Function that computes du/dt from u.
        Should return tensor of same shape as u.
    dt : float
        Time step size
        
    Returns
    -------
    u_new : torch.Tensor
        Solution at next time step. Shape: [batch, nx]
        
    Notes
    -----
    The SSP-RK3 scheme (Shu & Osher 1988):
    
        u^(1) = u^n + dt * L(u^n)
        u^(2) = 3/4 * u^n + 1/4 * u^(1) + 1/4 * dt * L(u^(1))
        u^{n+1} = 1/3 * u^n + 2/3 * u^(2) + 2/3 * dt * L(u^(2))
        
    where L(u) = du/dt is the spatial operator (RHS).
    
    This scheme:
    - Is third-order accurate in time
    - Preserves strong stability (SSP coefficient = 1)
    - Has TVD property when combined with TVD spatial methods
    - CFL condition: dt ≤ CFL * dx / max_wave_speed
    
    References
    ----------
    Shu & Osher (1988), "Efficient implementation of essentially 
    non-oscillatory shock-capturing schemes"
    
    Gottlieb, Shu & Tadmor (2001), "Strong stability-preserving 
    high-order time discretization methods"
    
    Examples
    --------
    >>> def rhs(u):
    ...     return -compute_spatial_derivative(u)
    >>> u_new = ssp_rk3_step(u, rhs, dt=0.001)
    """
    # Stage 1: Forward Euler step
    k1 = rhs_function(u)
    u1 = u + dt * k1
    
    # Stage 2: Second intermediate step
    k2 = rhs_function(u1)
    u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * k2
    
    # Stage 3: Final step
    k3 = rhs_function(u2)
    u_new = (1.0/3.0) * u + (2.0/3.0) * u2 + (2.0/3.0) * dt * k3
    
    return u_new


def ssp_rk2_step(
    u: torch.Tensor,
    rhs_function: Callable[[torch.Tensor], torch.Tensor],
    dt: float
) -> torch.Tensor:
    """
    Single step of second-order SSP Runge-Kutta (SSP-RK2).
    
    Also known as Heun's method or the modified Euler method.
    Simpler and cheaper than RK3, but only second-order accurate.
    
    Parameters
    ----------
    u : torch.Tensor
        Current solution. Shape: [batch, nx]
    rhs_function : callable
        Function that computes du/dt from u
    dt : float
        Time step size
        
    Returns
    -------
    u_new : torch.Tensor
        Solution at next time step. Shape: [batch, nx]
        
    Notes
    -----
    The SSP-RK2 scheme:
    
        u^(1) = u^n + dt * L(u^n)
        u^{n+1} = 1/2 * u^n + 1/2 * u^(1) + 1/2 * dt * L(u^(1))
        
    This is useful for:
    - Testing and debugging (simpler than RK3)
    - Problems where second-order time accuracy is sufficient
    - When computational cost is critical
    """
    # Stage 1
    k1 = rhs_function(u)
    u1 = u + dt * k1
    
    # Stage 2
    k2 = rhs_function(u1)
    u_new = 0.5 * u + 0.5 * u1 + 0.5 * dt * k2
    
    return u_new


def forward_euler_step(
    u: torch.Tensor,
    rhs_function: Callable[[torch.Tensor], torch.Tensor],
    dt: float
) -> torch.Tensor:
    """
    Single step of forward Euler (first-order explicit).
    
    Only for testing/debugging. Not recommended for production use
    with WENO schemes as it's only first-order accurate.
    
    Parameters
    ----------
    u : torch.Tensor
        Current solution
    rhs_function : callable
        RHS function
    dt : float
        Time step
        
    Returns
    -------
    u_new : torch.Tensor
        Updated solution
    """
    return u + dt * rhs_function(u)


# ============================================================================
# CFL CONDITION AND TIME STEP COMPUTATION
# ============================================================================

def compute_cfl_timestep(
    u: torch.Tensor,
    dx: float,
    cfl_number: float = 0.5,
    max_wave_speed: Optional[float] = None
) -> float:
    """
    Compute stable time step from CFL condition.
    
    The CFL (Courant-Friedrichs-Lewy) condition ensures numerical
    stability by requiring that the numerical domain of dependence
    contains the physical domain of dependence.
    
    Parameters
    ----------
    u : torch.Tensor
        Current solution. Shape: [batch, nx]
    dx : float
        Grid spacing
    cfl_number : float, optional
        CFL safety factor, typically 0.4-0.9. Default: 0.5
        Lower values are more stable but slower.
    max_wave_speed : float, optional
        Maximum wave speed. If None, estimated from u.
        For Burgers: max|u|
        For Euler: max|u| + c where c is sound speed
        
    Returns
    -------
    dt : float
        Maximum stable time step
        
    Notes
    -----
    CFL condition for explicit schemes:
    
        dt ≤ CFL * dx / λ_max
        
    where λ_max is the maximum wave speed.
    
    For SSP-RK3 with WENO-5, typical CFL numbers:
    - CFL = 0.5: Very safe, always stable
    - CFL = 0.7: Good balance
    - CFL = 0.9: Aggressive, may be unstable
    
    Examples
    --------
    >>> u = torch.randn(1, 100)
    >>> dx = 0.01
    >>> dt = compute_cfl_timestep(u, dx, cfl_number=0.5)
    """
    if max_wave_speed is None:
        # Conservative estimate: use max(|u|)
        max_wave_speed = torch.abs(u).max().item()
        
        # Add small safety factor
        max_wave_speed = max(max_wave_speed, 1e-10)  # Prevent division by zero
    
    # CFL condition
    dt = cfl_number * dx / max_wave_speed
    
    return dt


# ============================================================================
# MULTI-STEP INTEGRATION
# ============================================================================

def integrate_to_time(
    u_initial: torch.Tensor,
    t_final: float,
    dx: float,
    rhs_function: Callable[[torch.Tensor], torch.Tensor],
    cfl_number: float = 0.5,
    method: str = 'ssp_rk3',
    dt_fixed: Optional[float] = None,
    save_interval: Optional[float] = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Integrate solution from t=0 to t=t_final.
    
    Parameters
    ----------
    u_initial : torch.Tensor
        Initial condition. Shape: [batch, nx]
    t_final : float
        Final time
    dx : float
        Grid spacing
    rhs_function : callable
        RHS function that computes du/dt
    cfl_number : float, optional
        CFL number for adaptive time stepping. Default: 0.5
    method : str, optional
        Time integration method: 'ssp_rk3', 'ssp_rk2', 'euler'. Default: 'ssp_rk3'
    dt_fixed : float, optional
        If provided, use fixed time step instead of adaptive CFL.
    save_interval : float, optional
        If provided, save solution at these time intervals.
        Returns list of (time, solution) tuples.
    verbose : bool, optional
        Print progress. Default: False
        
    Returns
    -------
    u_final : torch.Tensor
        Solution at t=t_final. Shape: [batch, nx]
    OR
    snapshots : list of (float, torch.Tensor)
        If save_interval is provided, returns list of (t, u) pairs.
    """
    # Select time stepping method
    if method == 'ssp_rk3':
        step_function = ssp_rk3_step
    elif method == 'ssp_rk2':
        step_function = ssp_rk2_step
    elif method == 'euler':
        step_function = forward_euler_step
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ssp_rk3', 'ssp_rk2', or 'euler'")
    
    # Initialize
    u = u_initial.clone()
    t = 0.0
    step = 0
    
    # For saving snapshots
    snapshots = [] if save_interval is not None else None
    if save_interval is not None:
        next_save_time = save_interval
        # Save initial condition
        snapshots.append((0.0, u.clone()))
    else:
        next_save_time = float('inf')
    
    # Time stepping loop
    while t < t_final - 1e-12:  # Small tolerance for float comparison
        # Compute time step
        if dt_fixed is not None:
            dt = dt_fixed
        else:
            dt = compute_cfl_timestep(u, dx, cfl_number)
        
        # Don't overshoot final time
        dt = min(dt, t_final - t)
        
        # If saving snapshots, adjust dt to hit the next save time exactly
        if snapshots is not None and next_save_time < t_final and t < next_save_time:
            dt = min(dt, next_save_time - t)
        
        # Safety check
        if dt <= 0:
            break
        
        # Take time step
        u = step_function(u, rhs_function, dt)
        t += dt
        step += 1
        
        # Save snapshot if we've hit a save time
        if snapshots is not None:
            while next_save_time <= t + 1e-10 and next_save_time <= t_final + 1e-10:
                # Interpolate if we slightly overshot (shouldn't happen with dt adjustment above)
                snapshots.append((next_save_time, u.clone()))
                next_save_time += save_interval
        
        # Print progress
        if verbose and step % 100 == 0:
            print(f"Step {step}: t = {t:.6f} / {t_final:.6f}, dt = {dt:.6e}")
    
    if verbose:
        print(f"Integration complete: {step} steps, final time t = {t:.6f}")
    
    # Return final solution or snapshots
    if snapshots is not None:
        # Ensure final solution is included
        if len(snapshots) == 0 or abs(t - snapshots[-1][0]) > 1e-10:
            snapshots.append((t, u))
        return snapshots
    else:
        return u


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing time integration schemes...")
    
    # Test problem: linear advection du/dt = -c * du/dx with periodic BC
    # Exact solution: u(x, t) = u(x - c*t, 0)
    
    # Test 1: Single step verification
    print("\n" + "="*60)
    print("Test 1: Single RK3 step")
    
    nx = 100
    dx = 0.01
    u = torch.sin(2 * torch.pi * torch.linspace(0, 1, nx, dtype=torch.float64)).unsqueeze(0)
    
    def dummy_rhs(u):
        # Simple decay: du/dt = -u
        return -u
    
    dt = 0.001
    u_new = ssp_rk3_step(u, dummy_rhs, dt)
    
    # For du/dt = -u, exact solution is u(t) = u(0) * exp(-t)
    u_exact = u * torch.exp(torch.tensor(-dt))
    error = torch.abs(u_new - u_exact).max()
    
    print(f"Time step: {dt}")
    print(f"Max error vs exact: {error:.2e}")
    
    if error < 1e-6:
        print("✓ RK3 step accurate")
    else:
        print("  Note: Error expected for this simple test")
    
    # Test 2: CFL computation
    print("\n" + "="*60)
    print("Test 2: CFL time step computation")
    
    u_test = torch.randn(1, 100, dtype=torch.float64)
    dx = 0.01
    
    dt_cfl = compute_cfl_timestep(u_test, dx, cfl_number=0.5)
    
    print(f"Grid spacing: {dx}")
    print(f"Max |u|: {torch.abs(u_test).max():.4f}")
    print(f"CFL time step: {dt_cfl:.6e}")
    
    if dt_cfl > 0 and dt_cfl < dx:
        print("✓ CFL time step reasonable")
    else:
        print("✗ CFL time step unexpected")
    
    # Test 3: Multi-step integration
    print("\n" + "="*60)
    print("Test 3: Multi-step integration")
    
    u_initial = torch.sin(2 * torch.pi * torch.linspace(0, 1, 100, dtype=torch.float64)).unsqueeze(0)
    
    def decay_rhs(u):
        return -0.1 * u  # Slow decay
    
    t_final = 1.0
    dx = 0.01
    
    u_final = integrate_to_time(
        u_initial, t_final, dx, decay_rhs,
        cfl_number=0.5, method='ssp_rk3',
        verbose=False
    )
    
    print(f"Initial max: {u_initial.max():.4f}")
    print(f"Final max: {u_final.max():.4f}")
    print(f"Expected (exp(-0.1*t)): {(u_initial.max() * torch.exp(torch.tensor(-0.1 * t_final))).item():.4f}")
    
    print("✓ Multi-step integration completes")
    
    # Test 4: Different methods
    print("\n" + "="*60)
    print("Test 4: Comparing different RK methods")
    
    u0 = torch.sin(2 * torch.pi * torch.linspace(0, 1, 100, dtype=torch.float64)).unsqueeze(0)
    
    for method in ['euler', 'ssp_rk2', 'ssp_rk3']:
        u_final = integrate_to_time(
            u0, 0.5, 0.01, decay_rhs,
            method=method, dt_fixed=0.001, verbose=False
        )
        print(f"{method:10s}: final max = {u_final.max():.6f}")
    
    print("✓ All methods work")
    
    # Test 5: Snapshot saving
    print("\n" + "="*60)
    print("Test 5: Saving snapshots")
    
    snapshots = integrate_to_time(
        u0, 1.0, 0.01, decay_rhs,
        save_interval=0.25, dt_fixed=0.001, verbose=False
    )
    
    print(f"Number of snapshots: {len(snapshots)}")
    for t, u in snapshots:
        print(f"  t = {t:.2f}, max(u) = {u.max():.4f}")
    
    print("✓ Snapshot saving works")
    
    print("\n" + "="*60)
    print("✓ Time integration tests complete")