"""
Flux splitting and WENO reconstruction - Gottlieb's Finite Difference WENO.

This module implements Gottlieb's finite difference WENO formulation,
which uses a correction-based approach:
1. Compute base 4th-order central flux at each interface
2. Add WENO correction computed from flux differences
3. Use result in spatial derivative: du/dt[i] = (fh[i-1] - fh[i])/dx
"""

import torch
from typing import Callable, Tuple, Optional

from .stencils import generate_all_stencils, stencils_to_torch
from .smoothness import compute_smoothness_indicators_torch
from .weights import compute_nonlinear_weights


# ============================================================================
# GOTTLIEB FINITE DIFFERENCE WENO
# ============================================================================

def reconstruct_interface_fluxes_fd_weno(
    u_extended: torch.Tensor,
    flux_function: Callable,
    order: int = 5,
    epsilon: float = 1e-29,
    alpha: Optional[float] = None
) -> torch.Tensor:
    """
    Gottlieb's finite difference WENO reconstruction.
    
    This implements the exact algorithm from Gottlieb's MATLAB code,
    which computes interface fluxes fh[i] such that:
        du/dt[i] = (fh[i-1] - fh[i])/dx
    
    Parameters
    ----------
    u_extended : torch.Tensor
        Conservative variables with ghost cells. Shape: [batch, nx_extended]
    flux_function : callable
        Function that computes f(u)
    order : int, optional
        WENO order. Currently only 5 supported. Default: 5
    epsilon : float, optional
        Small parameter for WENO weights. Default: 1e-29 (Gottlieb's value)
    alpha : float, optional
        Maximum wave speed. If None, computed as max(|df/du|).
        
    Returns
    -------
    fh : torch.Tensor
        Interface fluxes. Shape: [batch, n_interfaces]
        fh[i] is the total flux at interface i
        
    Notes
    -----
    Gottlieb's algorithm:
    1. Compute flux differences dfp, dfm (flux splitting)
    2. For each interface: compute 4th-order central flux
    3. Add WENO correction from dfp and dfm
    4. Return interface fluxes
    
    The algorithm is described in:
    - Gottlieb & Shu MATLAB implementation
    - Shu & Osher (1989) for the finite difference approach
    """
    if order != 5:
        raise NotImplementedError("Currently only WENO-5 is implemented for FD-WENO")
    
    batch_size, nx_extended = u_extended.shape
    
    # Step 1: Compute flux at all points
    flux = flux_function(u_extended)
    
    # Step 2: Compute maximum wave speed if not provided
    if alpha is None:
        # Estimate from finite differences of flux
        alpha = torch.abs(flux[:, 1:] - flux[:, :-1]).max().item()
        # Make sure alpha > 0
        if alpha < 1e-10:
            alpha = 1.0
    
    # Step 3: Compute flux differences (Lax-Friedrichs splitting)
    # dfp[i] = (f[i+1] - f[i] + α*(u[i+1] - u[i])) / 2
    # dfm[i] = (f[i+1] - f[i] - α*(u[i+1] - u[i])) / 2
    flux_diff = flux[:, 1:] - flux[:, :-1]
    u_diff = u_extended[:, 1:] - u_extended[:, :-1]
    
    dfp = 0.5 * (flux_diff + alpha * u_diff)
    dfm = 0.5 * (flux_diff - alpha * u_diff)
    
    # Step 4: Determine interface range
    # We need 4 ghost cells on each side for WENO-5
    # Interfaces we can reconstruct: [4, nx_extended-4]
    md = 4
    n_interfaces = nx_extended - 2 * md + 1
    interface_start = md - 1  # Start at interface 3 (0-based)
    
    # Preallocate output
    fh = torch.zeros(
        batch_size, n_interfaces,
        dtype=u_extended.dtype,
        device=u_extended.device
    )
    
    # Step 5: Loop over interfaces and compute fluxes
    for idx in range(n_interfaces):
        i = interface_start + idx  # Global index in extended array
        
        # Compute 4th-order central flux
        # fh = (-f[i-1] + 7*(f[i] + f[i+1]) - f[i+2]) / 12
        central_flux = (
            -flux[:, i-1] + 7.0 * (flux[:, i] + flux[:, i+1]) - flux[:, i+2]
        ) / 12.0
        
        # Add WENO correction for positive flux (dfp)
        correction_p = compute_weno_correction(
            dfp, i, epsilon, batch_size, u_extended.dtype, u_extended.device
        )
        
        # Add WENO correction for negative flux (dfm)
        # Note: dfm uses negative values and reversed stencil
        correction_m = compute_weno_correction_negative(
            dfm, i, epsilon, batch_size, u_extended.dtype, u_extended.device
        )
        
        # Total interface flux
        fh[:, idx] = central_flux + correction_p + correction_m
    
    return fh


def compute_weno_correction(
    df: torch.Tensor,
    i: int,
    epsilon: float,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """
    Compute WENO correction for positive flux direction.
    
    This implements the correction term from Gottlieb's algorithm:
        correction = (s1*(t2-t1) + (0.5*s3-0.25)*(t3-t2)) / 3.0
    
    Parameters
    ----------
    df : torch.Tensor
        Flux differences (dfp or dfm). Shape: [batch, n_points]
    i : int
        Interface index in the extended array
    epsilon : float
        Small parameter for WENO weights
    batch_size : int
        Batch size
    dtype : torch.dtype
        Data type
    device : torch.device
        Device
        
    Returns
    -------
    correction : torch.Tensor
        WENO correction. Shape: [batch]
    """
    # Build stencil: hh[0..3] = df[i-2, i-1, i, i+1]
    hh0 = df[:, i-2]
    hh1 = df[:, i-1]
    hh2 = df[:, i]
    hh3 = df[:, i+1]
    
    # Compute differences of flux differences (second-order differences)
    t1 = hh0 - hh1
    t2 = hh1 - hh2
    t3 = hh2 - hh3
    
    # Compute smoothness indicators
    # IS0 = 13*t1^2 + 3*(hh0 - 3*hh1)^2
    # IS1 = 13*t2^2 + 3*(hh1 + hh2)^2
    # IS2 = 13*t3^2 + 3*(3*hh2 - hh3)^2
    IS0 = 13.0 * t1**2 + 3.0 * (hh0 - 3.0*hh1)**2
    IS1 = 13.0 * t2**2 + 3.0 * (hh1 + hh2)**2
    IS2 = 13.0 * t3**2 + 3.0 * (3.0*hh2 - hh3)**2
    
    # Compute weights using Gottlieb's formula (NOT standard WENO weights!)
    # tt1 = (epsilon + IS0)^2
    # tt2 = (epsilon + IS1)^2
    # tt3 = (epsilon + IS2)^2
    tt1 = (epsilon + IS0)**2
    tt2 = (epsilon + IS1)**2
    tt3 = (epsilon + IS2)**2
    
    # s1 = tt2 * tt3
    # s2 = 6.0 * tt1 * tt3
    # s3 = 3.0 * tt1 * tt2
    s1 = tt2 * tt3
    s2 = 6.0 * tt1 * tt3
    s3 = 3.0 * tt1 * tt2
    
    # Normalize
    t0 = 1.0 / (s1 + s2 + s3)
    s1 = s1 * t0
    s2 = s2 * t0
    s3 = s3 * t0
    
    # Compute correction
    correction = (s1 * (t2 - t1) + (0.5*s3 - 0.25) * (t3 - t2)) / 3.0
    
    return correction


def compute_weno_correction_negative(
    dfm: torch.Tensor,
    i: int,
    epsilon: float,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """
    Compute WENO correction for negative flux direction.
    
    For the negative flux, the stencil is reversed and negated:
        hh[0..3] = -dfm[i+2, i+1, i, i-1]
    
    Parameters
    ----------
    dfm : torch.Tensor
        Negative flux differences. Shape: [batch, n_points]
    i : int
        Interface index in the extended array
    epsilon : float
        Small parameter for WENO weights
    batch_size : int
        Batch size
    dtype : torch.dtype
        Data type
    device : torch.device
        Device
        
    Returns
    -------
    correction : torch.Tensor
        WENO correction. Shape: [batch]
    """
    # Build stencil with reversed and negated values
    hh0 = -dfm[:, i+2]
    hh1 = -dfm[:, i+1]
    hh2 = -dfm[:, i]
    hh3 = -dfm[:, i-1]
    
    # Compute differences
    t1 = hh0 - hh1
    t2 = hh1 - hh2
    t3 = hh2 - hh3
    
    # Compute smoothness indicators
    IS0 = 13.0 * t1**2 + 3.0 * (hh0 - 3.0*hh1)**2
    IS1 = 13.0 * t2**2 + 3.0 * (hh1 + hh2)**2
    IS2 = 13.0 * t3**2 + 3.0 * (3.0*hh2 - hh3)**2
    
    # Compute weights using Gottlieb's formula
    tt1 = (epsilon + IS0)**2
    tt2 = (epsilon + IS1)**2
    tt3 = (epsilon + IS2)**2
    
    s1 = tt2 * tt3
    s2 = 6.0 * tt1 * tt3
    s3 = 3.0 * tt1 * tt2
    
    # Normalize
    t0 = 1.0 / (s1 + s2 + s3)
    s1 = s1 * t0
    s2 = s2 * t0
    s3 = s3 * t0
    
    # Compute correction
    correction = (s1 * (t2 - t1) + (0.5*s3 - 0.25) * (t3 - t2)) / 3.0
    
    return correction


# ============================================================================
# LEGACY FUNCTIONS (Keep for compatibility but mark as deprecated)
# ============================================================================

def lax_friedrichs_splitting(
    flux: torch.Tensor,
    u: torch.Tensor,
    alpha: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DEPRECATED: Use reconstruct_interface_fluxes_fd_weno instead.
    
    This function is kept for backward compatibility only.
    """
    if alpha is None:
        alpha = torch.abs(u).max().item()
    
    flux_diff = flux[:, 1:] - flux[:, :-1]
    u_diff = u[:, 1:] - u[:, :-1]
    
    dfp = 0.5 * (flux_diff + alpha * u_diff)
    dfm = 0.5 * (flux_diff - alpha * u_diff)
    
    return dfp, dfm


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Gottlieb FD-WENO Implementation")
    print("=" * 70)
    
    # Test 1: Basic reconstruction
    print("\nTest 1: Basic reconstruction with discontinuity")
    
    nx = 101
    x = torch.linspace(-1, 1, nx, dtype=torch.float64)
    u0 = torch.sign(x)
    u0[50] = 0.0  # Explicit zero at discontinuity
    u0 = u0.unsqueeze(0)
    
    # Add ghost cells (periodic BC)
    n_ghost = 4
    u_extended = torch.cat([
        u0[:, -n_ghost:],
        u0,
        u0[:, :n_ghost]
    ], dim=1)
    
    # Define flux function (linear advection)
    def flux_fn(u):
        return u
    
    # Reconstruct
    fh = reconstruct_interface_fluxes_fd_weno(
        u_extended,
        flux_fn,
        order=5,
        epsilon=1e-29,
        alpha=1.0
    )
    
    print(f"  Input shape: {u_extended.shape}")
    print(f"  Output shape: {fh.shape}")
    print(f"  Output range: [{fh.min():.6f}, {fh.max():.6f}]")
    print(f"  ✓ Reconstruction completed")
    
    # Test 2: Smooth function
    print("\nTest 2: Smooth function")
    
    u_smooth = torch.sin(x).unsqueeze(0)
    u_smooth_ext = torch.cat([
        u_smooth[:, -n_ghost:],
        u_smooth,
        u_smooth[:, :n_ghost]
    ], dim=1)
    
    def burgers_flux(u):
        return 0.5 * u**2
    
    fh_smooth = reconstruct_interface_fluxes_fd_weno(
        u_smooth_ext,
        burgers_flux,
        order=5
    )
    
    print(f"  Output shape: {fh_smooth.shape}")
    print(f"  Output range: [{fh_smooth.min():.6f}, {fh_smooth.max():.6f}]")
    print(f"  ✓ Smooth reconstruction completed")
    
    print("\n" + "=" * 70)
    print("All tests passed!")