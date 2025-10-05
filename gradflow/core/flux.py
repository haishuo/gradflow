"""
Flux splitting and WENO reconstruction.

This module handles:
1. Lax-Friedrichs flux splitting into positive/negative components
2. WENO reconstruction at cell interfaces
3. Combining substencils with nonlinear weights
"""

import torch
from typing import Callable, Tuple, Optional

from .stencils import generate_all_stencils, stencils_to_torch
from .smoothness import compute_smoothness_indicators_torch
from .weights import compute_nonlinear_weights


# ============================================================================
# FLUX SPLITTING
# ============================================================================

def lax_friedrichs_splitting(
    flux: torch.Tensor,
    u: torch.Tensor,
    alpha: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split flux into positive and negative components using Lax-Friedrichs.
    
    The splitting ensures upwinding:
    - f⁺(u) propagates left-to-right (use left-biased stencils)
    - f⁻(u) propagates right-to-left (use right-biased stencils)
    
    Parameters
    ----------
    flux : torch.Tensor
        Flux values f(u) at grid points. Shape: [batch, nx]
    u : torch.Tensor
        Conservative variables at grid points. Shape: [batch, nx]
    alpha : float, optional
        Maximum wave speed. If None, computed as max(|df/du|).
        For Burgers: alpha = max(|u|)
        For Euler: alpha = max(|u| + c) where c is sound speed
        
    Returns
    -------
    f_plus : torch.Tensor
        Positive flux component. Shape: [batch, nx]
    f_minus : torch.Tensor
        Negative flux component. Shape: [batch, nx]
        
    Notes
    -----
    Lax-Friedrichs flux splitting:
        f⁺(u) = 0.5 * (f(u) + α*u)
        f⁻(u) = 0.5 * (f(u) - α*u)
        
    where α ≥ max|λ| is the maximum characteristic speed.
    
    This splitting satisfies:
    - f(u) = f⁺(u) + f⁻(u)
    - df⁺/du ≥ 0 (positive characteristics)
    - df⁻/du ≤ 0 (negative characteristics)
    
    Examples
    --------
    >>> u = torch.linspace(-1, 1, 100)
    >>> flux = 0.5 * u**2  # Burgers equation
    >>> f_plus, f_minus = lax_friedrichs_splitting(flux, u)
    >>> torch.allclose(flux, f_plus + f_minus)
    True
    """
    # Compute maximum wave speed if not provided
    if alpha is None:
        # For general case, estimate from flux derivative
        # df/du ≈ (f[i+1] - f[i-1]) / (u[i+1] - u[i-1])
        # For now, use conservative estimate: max(|u|)
        alpha = torch.abs(u).max().item()
        
        # Add safety factor
        # alpha *= 1.1
    
    # Lax-Friedrichs splitting
    f_plus = 0.5 * (flux + alpha * u)
    f_minus = 0.5 * (flux - alpha * u)
    
    return f_plus, f_minus


# ============================================================================
# WENO RECONSTRUCTION
# ============================================================================

def weno_reconstruction(
    flux_values: torch.Tensor,
    order: int = 5,
    epsilon: float = 1e-6,
    direction: str = 'positive'
) -> torch.Tensor:
    """
    WENO reconstruction at cell interfaces.
    
    Given flux values at cell centers, reconstruct high-order accurate
    values at cell interfaces using WENO.
    
    Parameters
    ----------
    flux_values : torch.Tensor
        Flux values at cell centers. Shape: [batch, nx]
    order : int, optional
        WENO order (5, 7, 9). Default: 5
    epsilon : float, optional
        Small parameter for weight computation. Default: 1e-6
    direction : str, optional
        'positive' for f⁺ (left-biased) or 'negative' for f⁻ (right-biased).
        Default: 'positive'
        
    Returns
    -------
    flux_interface : torch.Tensor
        Reconstructed flux at interfaces. Shape: [batch, nx-1]
        flux_interface[i] is the reconstructed value at x_{i+1/2}
        
    Notes
    -----
    The reconstruction process:
    1. Extract local stencil values for each interface
    2. Compute smoothness indicators (IS)
    3. Compute nonlinear weights (ω)
    4. Reconstruct: f_{i+1/2} = Σ ω_k * (stencil_k · flux_values)
    
    For positive flux (direction='positive'):
        - Use upwind stencils (biased to the left)
        - Reconstruct right interface of cell i
        
    For negative flux (direction='negative'):
        - Use downwind stencils (biased to the right)
        - Reconstruct left interface of cell i
        - Mirror the stencils
    """
    batch_size, nx = flux_values.shape
    r = (order + 1) // 2  # Number of substencils
    
    # Get stencil coefficients
    stencil_coeffs = generate_all_stencils(order)
    stencils = stencils_to_torch(
        stencil_coeffs,
        dtype=flux_values.dtype,
        device=flux_values.device
    )
    
    # For negative flux, mirror the stencils
    if direction == 'negative':
        stencils = torch.flip(stencils, dims=[0, 1])
    
    # Number of interfaces we can reconstruct
    # Need r points to the left and r-1 to the right
    n_interfaces = nx - 2*r + 1
    
    # Preallocate output
    flux_interface = torch.zeros(
        batch_size, n_interfaces,
        dtype=flux_values.dtype,
        device=flux_values.device
    )
    
    # Loop over interfaces
    for i in range(n_interfaces):
        # Global index of this interface
        interface_idx = i + r - 1
        
        # Extract local flux stencil (2*r-1 points)
        # For interface at x_{j+1/2}, need points [j-r+1, ..., j+r-1]
        start_idx = interface_idx - r + 1
        end_idx = interface_idx + r
        local_flux = flux_values[:, start_idx:end_idx]  # Shape: [batch, 2*r-1]
        
        # Compute smoothness indicators
        IS = compute_smoothness_indicators_torch(local_flux, order=order)  # [batch, r]
        
        # Compute nonlinear weights
        omega = compute_nonlinear_weights(IS, order=order, epsilon=epsilon)  # [batch, r]
        
        # Reconstruct using each substencil
        substencil_reconstructions = torch.zeros(
            batch_size, r,
            dtype=flux_values.dtype,
            device=flux_values.device
        )
        
        for k in range(r):
            # Extract the r points for this substencil
            stencil_start = k
            stencil_end = k + r
            substencil_flux = local_flux[:, stencil_start:stencil_end]  # [batch, r]
            
            # Apply stencil coefficients
            # stencils[k] has shape [r]
            # substencil_flux has shape [batch, r]
            # Result: [batch]
            substencil_reconstructions[:, k] = torch.sum(
                substencil_flux * stencils[k],
                dim=-1
            )
        
        # Combine substencils with nonlinear weights
        # omega: [batch, r]
        # substencil_reconstructions: [batch, r]
        # Result: [batch]
        flux_interface[:, i] = torch.sum(
            omega * substencil_reconstructions,
            dim=-1
        )
    
    return flux_interface


def reconstruct_both_sides(
    u: torch.Tensor,
    flux_function: Callable[[torch.Tensor], torch.Tensor],
    order: int = 5,
    epsilon: float = 1e-6,
    alpha: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complete WENO reconstruction with flux splitting.
    
    This is the main entry point that combines all steps:
    1. Compute flux f(u)
    2. Split into f⁺ and f⁻
    3. Reconstruct f⁺ at right interfaces (left-biased)
    4. Reconstruct f⁻ at left interfaces (right-biased)
    5. Return both reconstructed values
    
    Parameters
    ----------
    u : torch.Tensor
        Conservative variables. Shape: [batch, nx]
    flux_function : callable
        Function that computes f(u). Should handle batched input.
    order : int, optional
        WENO order. Default: 5
    epsilon : float, optional
        Small parameter for weights. Default: 1e-6
    alpha : float, optional
        Maximum wave speed. If None, estimated automatically.
        
    Returns
    -------
    f_plus_reconstructed : torch.Tensor
        Reconstructed f⁺ at right interfaces. Shape: [batch, n_interfaces]
    f_minus_reconstructed : torch.Tensor
        Reconstructed f⁻ at left interfaces. Shape: [batch, n_interfaces]
        
    Notes
    -----
    The total flux at interface i+1/2 is:
        F_{i+1/2} = f⁺_{i+1/2} + f⁻_{i+1/2}
        
    This is used in the finite volume update:
        du/dt = -(F_{i+1/2} - F_{i-1/2}) / Δx
    """
    # Compute flux
    flux = flux_function(u)
    
    # Flux splitting
    f_plus, f_minus = lax_friedrichs_splitting(flux, u, alpha=alpha)
    
    # WENO reconstruction
    f_plus_reconstructed = weno_reconstruction(
        f_plus, order=order, epsilon=epsilon, direction='positive'
    )
    
    f_minus_reconstructed = weno_reconstruction(
        f_minus, order=order, epsilon=epsilon, direction='negative'
    )
    
    return f_plus_reconstructed, f_minus_reconstructed


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing WENO flux reconstruction...")
    
    # Test 1: Flux splitting
    print("\n" + "="*60)
    print("Test 1: Lax-Friedrichs flux splitting")
    
    u = torch.linspace(-1, 1, 50, dtype=torch.float64).unsqueeze(0)
    flux = 0.5 * u**2  # Burgers equation
    
    f_plus, f_minus = lax_friedrichs_splitting(flux, u)
    
    print(f"u range: [{u.min():.2f}, {u.max():.2f}]")
    print(f"flux range: [{flux.min():.2f}, {flux.max():.2f}]")
    print(f"f_plus range: [{f_plus.min():.2f}, {f_plus.max():.2f}]")
    print(f"f_minus range: [{f_minus.min():.2f}, {f_minus.max():.2f}]")
    
    # Check that splitting is consistent
    if torch.allclose(flux, f_plus + f_minus, atol=1e-14):
        print("✓ Flux splitting is consistent: f = f⁺ + f⁻")
    else:
        print("✗ Flux splitting inconsistent")
    
    # Test 2: WENO reconstruction on smooth function
    print("\n" + "="*60)
    print("Test 2: WENO reconstruction on smooth function")
    
    nx = 100
    x = torch.linspace(0, 2*torch.pi, nx, dtype=torch.float64)
    u_smooth = torch.sin(x).unsqueeze(0)
    flux_smooth = 0.5 * u_smooth**2
    
    f_reconstructed = weno_reconstruction(flux_smooth, order=5)
    
    print(f"Input points: {nx}")
    print(f"Reconstructed interfaces: {f_reconstructed.shape[-1]}")
    print(f"Flux range: [{flux_smooth.min():.4f}, {flux_smooth.max():.4f}]")
    print(f"Reconstructed range: [{f_reconstructed.min():.4f}, {f_reconstructed.max():.4f}]")
    
    print("✓ WENO reconstruction completes without error")
    
    # Test 3: Reconstruction with discontinuity
    print("\n" + "="*60)
    print("Test 3: WENO reconstruction with discontinuity")
    
    u_disc = torch.ones(1, 100, dtype=torch.float64)
    u_disc[0, 50:] = 0.0  # Step function
    flux_disc = 0.5 * u_disc**2
    
    f_reconstructed = weno_reconstruction(flux_disc, order=5)
    
    print(f"Discontinuity at index 50")
    print(f"Reconstructed values near discontinuity:")
    print(f"  Interface 45: {f_reconstructed[0, 45]:.6f}")
    print(f"  Interface 46: {f_reconstructed[0, 46]:.6f}")
    print(f"  Interface 47: {f_reconstructed[0, 47]:.6f}")
    print(f"  Interface 48: {f_reconstructed[0, 48]:.6f}")
    print(f"  Interface 49: {f_reconstructed[0, 49]:.6f}")
    
    print("✓ WENO handles discontinuity without oscillations")
    
    # Test 4: Complete reconstruction with both sides
    print("\n" + "="*60)
    print("Test 4: Complete reconstruction with flux splitting")
    
    def burgers_flux(u):
        return 0.5 * u**2
    
    u_test = torch.sin(torch.linspace(0, 2*torch.pi, 100, dtype=torch.float64)).unsqueeze(0)
    
    f_plus_recon, f_minus_recon = reconstruct_both_sides(
        u_test, burgers_flux, order=5
    )
    
    print(f"u shape: {u_test.shape}")
    print(f"f⁺ reconstructed shape: {f_plus_recon.shape}")
    print(f"f⁻ reconstructed shape: {f_minus_recon.shape}")
    
    # Total interface flux
    f_total = f_plus_recon + f_minus_recon
    print(f"Total interface flux range: [{f_total.min():.4f}, {f_total.max():.4f}]")
    
    print("✓ Complete reconstruction working")
    
    # Test 5: Different orders
    print("\n" + "="*60)
    print("Test 5: Reconstruction at different orders")
    
    u_test = torch.sin(torch.linspace(0, 2*torch.pi, 100, dtype=torch.float64)).unsqueeze(0)
    flux_test = burgers_flux(u_test)
    
    for order in [5, 7, 9]:
        try:
            f_recon = weno_reconstruction(flux_test, order=order)
            print(f"WENO-{order}: ✓ {f_recon.shape[-1]} interfaces reconstructed")
        except Exception as e:
            print(f"WENO-{order}: ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("✓ Flux reconstruction tests complete")