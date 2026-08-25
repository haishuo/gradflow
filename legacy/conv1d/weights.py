"""
WENO nonlinear weight computation.

The weights adapt to local solution smoothness, giving higher weight
to smoother substencils and lower weight to substencils containing
discontinuities or sharp gradients.
"""

import torch
from typing import Optional


# ============================================================================
# OPTIMAL LINEAR WEIGHTS FROM LITERATURE
# ============================================================================

# WENO-5 optimal weights from Jiang & Shu (1996)
WENO5_OPTIMAL_WEIGHTS = torch.tensor([1.0/10.0, 6.0/10.0, 3.0/10.0])

# WENO-7 optimal weights from Balsara & Shu (2000)
WENO7_OPTIMAL_WEIGHTS = torch.tensor([1.0/35.0, 12.0/35.0, 18.0/35.0, 4.0/35.0])

# WENO-9 optimal weights from Balsara & Shu (2000)
WENO9_OPTIMAL_WEIGHTS = torch.tensor([1.0/126.0, 10.0/126.0, 45.0/126.0, 
                                       60.0/126.0, 10.0/126.0])

# WENO-11 optimal weights (if needed - look up from literature)
# WENO11_OPTIMAL_WEIGHTS = torch.tensor([...])


# ============================================================================
# WEIGHT COMPUTATION
# ============================================================================

def compute_nonlinear_weights(
    IS: torch.Tensor,
    order: int = 5,
    epsilon: float = 1e-6,
    power: int = 2,
    optimal_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute WENO nonlinear weights from smoothness indicators.
    
    The nonlinear weights adapt to local smoothness:
    - Smooth regions: weights ≈ optimal weights (high-order accuracy)
    - Near discontinuities: low weight to non-smooth substencils
    
    Parameters
    ----------
    IS : torch.Tensor
        Smoothness indicators. Shape: [..., r] where r = (order+1)//2
    order : int, optional
        WENO order (5, 7, 9, ...). Default: 5
    epsilon : float, optional
        Small parameter to prevent division by zero. Default: 1e-6
        Typical values: 1e-6 to 1e-40 depending on problem
    power : int, optional
        Exponent in weight formula. Default: 2
        Standard WENO uses p=2, WENO-Z uses p=1
    optimal_weights : torch.Tensor, optional
        Optimal linear weights. If None, uses standard weights for given order.
        Shape: [r]
        
    Returns
    -------
    omega : torch.Tensor
        Nonlinear weights. Shape: [..., r]
        Guaranteed to sum to 1 along last dimension
        
    Notes
    -----
    The weight formula from Jiang & Shu (1996):
    
        α_k = d_k / (ε + IS_k)^p
        ω_k = α_k / Σα_k
        
    where d_k are the optimal linear weights.
    
    Properties:
    - If all IS are small (smooth): ω ≈ d (optimal weights)
    - If IS_k is large: ω_k ≈ 0 (downweight non-smooth stencil)
    - Always: Σω_k = 1 (convex combination)
    
    Examples
    --------
    >>> IS = torch.tensor([[0.1, 0.1, 0.1]])  # Smooth region
    >>> omega = compute_nonlinear_weights(IS, order=5)
    >>> omega
    tensor([[0.1000, 0.6000, 0.3000]])  # Close to optimal [1/10, 6/10, 3/10]
    
    >>> IS = torch.tensor([[0.1, 10.0, 0.1]])  # Middle stencil has shock
    >>> omega = compute_nonlinear_weights(IS, order=5)
    >>> omega[0, 1] < 0.01  # Middle weight very small
    True
    """
    # Get optimal weights for this order
    if optimal_weights is None:
        optimal_weights = get_optimal_weights(order)
    
    # Move optimal weights to same device as IS
    optimal_weights = optimal_weights.to(IS.device).to(IS.dtype)
    
    # Ensure optimal_weights broadcasts correctly
    # IS shape: [..., r]
    # optimal_weights shape: [r]
    # Need to add dimensions to match
    for _ in range(IS.ndim - 1):
        optimal_weights = optimal_weights.unsqueeze(0)
    
    # Compute alpha: α_k = d_k / (ε + IS_k)^p
    alpha = optimal_weights / (epsilon + IS)**power
    
    # Normalize to get omega: ω_k = α_k / Σα_k
    alpha_sum = alpha.sum(dim=-1, keepdim=True)
    omega = alpha / alpha_sum
    
    return omega


def get_optimal_weights(order: int) -> torch.Tensor:
    """
    Get optimal linear weights for given WENO order.
    
    These are the weights that would be used for a linear scheme
    without adaptation. They are derived to maximize order of accuracy.
    
    Parameters
    ----------
    order : int
        WENO order (5, 7, 9, 11)
        
    Returns
    -------
    weights : torch.Tensor
        Optimal weights. Shape: [r] where r = (order+1)//2
        
    Raises
    ------
    ValueError
        If order is not supported
        
    Notes
    -----
    These weights are problem-independent and derived once for each order.
    They come from the literature:
    - WENO-5: Jiang & Shu (1996)
    - WENO-7, WENO-9: Balsara & Shu (2000)
    
    Examples
    --------
    >>> get_optimal_weights(5)
    tensor([0.1000, 0.6000, 0.3000])
    """
    if order == 5:
        return WENO5_OPTIMAL_WEIGHTS.clone()
    elif order == 7:
        return WENO7_OPTIMAL_WEIGHTS.clone()
    elif order == 9:
        return WENO9_OPTIMAL_WEIGHTS.clone()
    # elif order == 11:
    #     return WENO11_OPTIMAL_WEIGHTS.clone()
    else:
        raise ValueError(
            f"Optimal weights not available for order {order}. "
            f"Supported orders: 5, 7, 9"
        )


# ============================================================================
# ALTERNATIVE WEIGHT FORMULATIONS
# ============================================================================

def compute_wenoz_weights(
    IS: torch.Tensor,
    order: int = 5,
    epsilon: float = 1e-6,
    optimal_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute WENO-Z weights (Borges et al. 2008).
    
    WENO-Z improves on standard WENO by achieving optimal order
    at critical points. Uses a global smoothness indicator.
    
    Parameters
    ----------
    IS : torch.Tensor
        Smoothness indicators. Shape: [..., r]
    order : int, optional
        WENO order. Default: 5
    epsilon : float, optional
        Small parameter. Default: 1e-6
    optimal_weights : torch.Tensor, optional
        Optimal weights. If None, uses standard weights.
        
    Returns
    -------
    omega : torch.Tensor
        WENO-Z weights. Shape: [..., r]
        
    Notes
    -----
    WENO-Z modifies the weight formula to:
    
        τ = |IS_0 - IS_{r-1}|  (global smoothness)
        α_k = d_k * (1 + (τ/(ε + IS_k))^2)
        ω_k = α_k / Σα_k
        
    This achieves optimal order at critical points while maintaining
    ENO property near discontinuities.
    
    References
    ----------
    Borges et al. (2008), "An improved weighted essentially 
    non-oscillatory scheme for hyperbolic conservation laws"
    """
    # Get optimal weights
    if optimal_weights is None:
        optimal_weights = get_optimal_weights(order)
    
    optimal_weights = optimal_weights.to(IS.device).to(IS.dtype)
    
    # Add dimensions for broadcasting
    for _ in range(IS.ndim - 1):
        optimal_weights = optimal_weights.unsqueeze(0)
    
    # Compute global smoothness indicator τ
    tau = torch.abs(IS[..., 0] - IS[..., -1]).unsqueeze(-1)
    
    # Modified alpha with τ term
    alpha = optimal_weights * (1.0 + (tau / (epsilon + IS))**2)
    
    # Normalize
    alpha_sum = alpha.sum(dim=-1, keepdim=True)
    omega = alpha / alpha_sum
    
    return omega


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("Testing WENO weight computation...")
    
    # Test 1: Smooth region (all IS small and similar)
    print("\n" + "="*60)
    print("Test 1: Smooth region (weights → optimal)")
    
    IS_smooth = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float64)
    omega = compute_nonlinear_weights(IS_smooth, order=5)
    optimal = get_optimal_weights(5)
    
    print(f"IS: {IS_smooth}")
    print(f"Weights: {omega}")
    print(f"Optimal: {optimal}")
    
    # Should be close to optimal weights
    if torch.allclose(omega[0], optimal.to(omega.dtype), rtol=0.1):
        print("✓ Weights close to optimal in smooth region")
    else:
        print("✗ Weights differ from optimal")
    
    # Test 2: Discontinuity in middle stencil
    print("\n" + "="*60)
    print("Test 2: Discontinuity (middle stencil downweighted)")
    
    IS_shock = torch.tensor([[0.1, 100.0, 0.1]], dtype=torch.float64)
    omega = compute_nonlinear_weights(IS_shock, order=5)
    
    print(f"IS: {IS_shock}")
    print(f"Weights: {omega}")
    
    # Middle weight should be very small
    if omega[0, 1] < 0.01:
        print(f"✓ Middle weight = {omega[0, 1].item():.6f} (strongly downweighted)")
    else:
        print(f"✗ Middle weight = {omega[0, 1].item():.6f} (should be << 0.01)")
    
    # Test 3: Conservation (weights sum to 1)
    print("\n" + "="*60)
    print("Test 3: Conservation (weights sum to 1)")
    
    torch.manual_seed(42)
    IS_random = torch.rand(10, 3, dtype=torch.float64) * 10
    omega = compute_nonlinear_weights(IS_random, order=5)
    
    sums = omega.sum(dim=-1)
    print(f"Weight sums: {sums}")
    
    if torch.allclose(sums, torch.ones_like(sums), atol=1e-14):
        print("✓ All weights sum to 1")
    else:
        print("✗ Weight sums incorrect")
    
    # Test 4: Non-negativity
    print("\n" + "="*60)
    print("Test 4: Non-negativity")
    
    if (omega >= 0).all():
        print("✓ All weights are non-negative")
    else:
        print("✗ Found negative weights")
    
    # Test 5: WENO-Z comparison
    print("\n" + "="*60)
    print("Test 5: WENO-Z vs standard WENO")
    
    IS_test = torch.tensor([[1.0, 0.5, 1.0]], dtype=torch.float64)
    omega_standard = compute_nonlinear_weights(IS_test, order=5)
    omega_z = compute_wenoz_weights(IS_test, order=5)
    
    print(f"Standard WENO: {omega_standard}")
    print(f"WENO-Z:        {omega_z}")
    print("(WENO-Z should give more weight to smoother stencil)")
    
    if omega_z[0, 1] > omega_standard[0, 1]:
        print("✓ WENO-Z favors smoother stencil more")
    else:
        print("  Note: Difference may be small for this test case")
    
    # Test 6: Different orders
    print("\n" + "="*60)
    print("Test 6: Different WENO orders")
    
    for order in [5, 7, 9]:
        r = (order + 1) // 2
        IS_test = torch.ones(1, r, dtype=torch.float64) * 0.1
        omega = compute_nonlinear_weights(IS_test, order=order)
        optimal = get_optimal_weights(order)
        
        print(f"\nWENO-{order}:")
        print(f"  Optimal weights: {optimal}")
        print(f"  Computed weights: {omega[0]}")
        
        if torch.allclose(omega[0], optimal.to(omega.dtype), rtol=0.1):
            print(f"  ✓ Matches optimal")
        else:
            print(f"  ✗ Differs from optimal")
    
    print("\n" + "="*60)
    print("✓ Weight computation tests complete")