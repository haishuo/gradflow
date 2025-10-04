"""
Smoothness indicator computation for WENO schemes.

Uses pre-computed coefficients from the literature rather than
symbolic derivation. This is faster, more reliable, and matches
exactly what's published in papers.
"""

import torch
from typing import Dict, List, Tuple


# ============================================================================
# SMOOTHNESS INDICATOR COEFFICIENTS FROM LITERATURE
# ============================================================================

# Each entry: (coefficient, [stencil pattern])
# Pattern [a, b, c] means: a*f[i] + b*f[i+1] + c*f[i+2]

# WENO-5 coefficients from Jiang & Shu (1996)
# "Efficient Implementation of Weighted ENO Schemes"
WENO5_SMOOTHNESS = {
    # IS_0: stencil [f_{j-2}, f_{j-1}, f_j]
    0: [
        (13.0/12.0, [1, -2, 1]),   # (f_{j-2} - 2f_{j-1} + f_j)^2
        (1.0/4.0,   [1, -4, 3]),   # (f_{j-2} - 4f_{j-1} + 3f_j)^2
    ],
    # IS_1: stencil [f_{j-1}, f_j, f_{j+1}]
    1: [
        (13.0/12.0, [1, -2, 1]),   # (f_{j-1} - 2f_j + f_{j+1})^2
        (1.0/4.0,   [1, 0, -1]),   # (f_{j-1} - f_{j+1})^2
    ],
    # IS_2: stencil [f_j, f_{j+1}, f_{j+2}]
    2: [
        (13.0/12.0, [1, -2, 1]),   # (f_j - 2f_{j+1} + f_{j+2})^2
        (1.0/4.0,   [3, -4, 1]),   # (3f_j - 4f_{j+1} + f_{j+2})^2
    ],
}

# WENO-7 coefficients from Balsara & Shu (2000)
# "Monotonicity Preserving WENO Schemes with Increasingly High Order"
# TODO: Add when implementing WENO-7
WENO7_SMOOTHNESS = {
    # Will be added when we implement WENO-7
}

# WENO-9 coefficients from Balsara & Shu (2000)
# TODO: Add when implementing WENO-9
WENO9_SMOOTHNESS = {
    # Will be added when we implement WENO-9
}


# ============================================================================
# PYTORCH COMPUTATION
# ============================================================================

def compute_smoothness_indicators_torch(
    flux_values: torch.Tensor,
    order: int = 5
) -> torch.Tensor:
    """
    Compute WENO smoothness indicators using hardcoded formulas.

    Parameters
    ----------
    flux_values : torch.Tensor
        Flux values at stencil points.
        Shape: [..., 2*r-1] where r = (order+1)//2
        For WENO-5: shape [..., 5]
    order : int, optional
        WENO order (5, 7, 9). Default: 5

    Returns
    -------
    IS : torch.Tensor
        Smoothness indicators for each substencil.
        Shape: [..., r] where r = (order+1)//2
        For WENO-5: shape [..., 3]

    Raises
    ------
    ValueError
        If order is not supported
        If flux_values has wrong shape

    Notes
    -----
    Uses the exact formulas from the literature:
    - WENO-5: Jiang & Shu (1996)
    - WENO-7: Balsara & Shu (2000)
    - WENO-9: Balsara & Shu (2000)

    These formulas measure the L2 norm of polynomial derivatives
    integrated over the cell interval. For smooth regions, IS ≈ 0.
    For discontinuities, IS is large.

    Examples
    --------
    >>> flux = torch.randn(100, 5)
    >>> IS = compute_smoothness_indicators_torch(flux, order=5)
    >>> IS.shape
    torch.Size([100, 3])
    """
    if order == 5:
        return _compute_weno5_smoothness(flux_values)
    elif order == 7:
        raise NotImplementedError("WENO-7 smoothness indicators not yet implemented")
    elif order == 9:
        raise NotImplementedError("WENO-9 smoothness indicators not yet implemented")
    else:
        raise ValueError(
            f"Unsupported WENO order: {order}. "
            f"Supported orders: 5 (more coming soon: 7, 9)"
        )


def _compute_weno5_smoothness(flux_values: torch.Tensor) -> torch.Tensor:
    """
    Compute WENO-5 smoothness indicators (optimized).

    This is a specialized, optimized version for WENO-5 that's
    faster than the generic implementation.
    """
    # Validate input shape
    if flux_values.shape[-1] != 5:
        raise ValueError(
            f"WENO-5 requires 5 flux values per point, got {flux_values.shape[-1]}. "
            f"Expected shape: [..., 5]"
        )

    # Extract flux values (using clearer naming)
    f = flux_values

    # IS_0: uses points [0, 1, 2] (indices in the 5-point stencil)
    # From Jiang & Shu (1996) equation (2.63)
    d1 = f[..., 0] - 2*f[..., 1] + f[..., 2]
    d2 = f[..., 0] - 4*f[..., 1] + 3*f[..., 2]
    IS_0 = (13.0/12.0) * d1**2 + (1.0/4.0) * d2**2

    # IS_1: uses points [1, 2, 3]
    # From Jiang & Shu (1996) equation (2.64)
    d1 = f[..., 1] - 2*f[..., 2] + f[..., 3]
    d2 = f[..., 1] - f[..., 3]
    IS_1 = (13.0/12.0) * d1**2 + (1.0/4.0) * d2**2

    # IS_2: uses points [2, 3, 4]
    # From Jiang & Shu (1996) equation (2.65)
    d1 = f[..., 2] - 2*f[..., 3] + f[..., 4]
    d2 = 3*f[..., 2] - 4*f[..., 3] + f[..., 4]
    IS_2 = (13.0/12.0) * d1**2 + (1.0/4.0) * d2**2

    # Stack into [batch, 3] tensor
    IS = torch.stack([IS_0, IS_1, IS_2], dim=-1)

    return IS


# ============================================================================
# UTILITIES
# ============================================================================

def get_smoothness_coefficients(order: int) -> Dict[int, List[Tuple[float, List[int]]]]:
    """
    Get the smoothness indicator coefficients for a given WENO order.

    Returns the exact coefficients from the literature.

    Parameters
    ----------
    order : int
        WENO order (5, 7, 9)

    Returns
    -------
    coefficients : dict
        Dictionary mapping substencil index to list of (coeff, pattern) tuples
    """
    if order == 5:
        return WENO5_SMOOTHNESS
    elif order == 7:
        return WENO7_SMOOTHNESS
    elif order == 9:
        return WENO9_SMOOTHNESS
    else:
        raise ValueError(f"Unsupported order: {order}")


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("Testing WENO-5 smoothness indicators...")

    # Test 1: Random data
    print("\n" + "="*60)
    print("Test 1: Random flux values")
    torch.manual_seed(42)
    flux = torch.randn(10, 5, dtype=torch.float64)

    IS = compute_smoothness_indicators_torch(flux, order=5)

    print(f"Input flux shape: {flux.shape}")
    print(f"Output IS shape: {IS.shape}")
    print(f"IS range: [{IS.min().item():.6f}, {IS.max().item():.6f}]")

    if (IS >= 0).all():
        print("✓ All IS values are non-negative")
    else:
        print("✗ Some IS values are negative (BUG!)")

    # Test 2: Constant function (should give IS ≈ 0)
    print("\n" + "="*60)
    print("Test 2: Constant function (should give IS ≈ 0)")
    flux_const = torch.ones(1, 5, dtype=torch.float64) * 5.0
    IS_const = compute_smoothness_indicators_torch(flux_const, order=5)

    print(f"Constant flux: {flux_const}")
    print(f"IS values: {IS_const}")

    if torch.allclose(IS_const, torch.zeros_like(IS_const), atol=1e-14):
        print("✓ IS vanishes for constant function")
    else:
        print(f"✗ IS should be ~0, got max={IS_const.max().item():.2e}")

    # Test 3: Linear function
    # NOTE: Jiang & Shu's formulas include grid spacing factors
    # For unit grid spacing (Δx=1), a linear function f(x)=x gives IS≠0
    # This is CORRECT - the formulas measure scaled derivatives
    print("\n" + "="*60)
    print("Test 3: Linear function on unit grid")
    flux_linear = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
    IS_linear = compute_smoothness_indicators_torch(flux_linear, order=5)

    print(f"Linear flux: {flux_linear}")
    print(f"IS values: {IS_linear}")

    # All three substencils should give the same IS value
    # (symmetry property for linear functions)
    if torch.allclose(IS_linear, IS_linear[..., 0:1].expand_as(IS_linear), rtol=1e-10):
        print("✓ IS is symmetric across substencils for linear function")
    else:
        print(f"✗ Expected symmetric IS values, got {IS_linear}")

    # Test 4: Quadratic function (should give IS > 0)
    print("\n" + "="*60)
    print("Test 4: Quadratic function (should give IS > 0)")
    x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
    flux_quad = x**2
    IS_quad = compute_smoothness_indicators_torch(flux_quad, order=5)

    print(f"Quadratic flux: {flux_quad}")
    print(f"IS values: {IS_quad}")

    if (IS_quad > 1e-10).all():
        print("✓ IS is positive for quadratic function")
    else:
        print(f"✗ IS should be >0 for quadratic")

    # Test 5: Discontinuous function (should give large IS)
    print("\n" + "="*60)
    print("Test 5: Discontinuous function (should give large IS)")
    flux_disc = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    IS_disc = compute_smoothness_indicators_torch(flux_disc, order=5)

    print(f"Discontinuous flux: {flux_disc}")
    print(f"IS values: {IS_disc}")
    print(f"IS range: [{IS_disc.min().item():.6f}, {IS_disc.max().item():.6f}]")

    if IS_disc.max() > 0.1:
        print("✓ IS is large near discontinuity")
    else:
        print(f"✗ IS should be large near discontinuity")

    print("\n" + "="*60)
    print("✓ All tests complete")
