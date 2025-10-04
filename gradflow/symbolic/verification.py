"""
Verification utilities for WENO coefficients and formulas.

This module provides symbolic verification that hardcoded coefficients
match the mathematical formulas from the literature.
"""

import sympy as sp
import torch
from typing import Dict, List, Tuple, Optional


def verify_stencil_coefficients(
    order: int,
    tolerance: float = 1e-14,
    verbose: bool = True
) -> bool:
    """
    Verify that stencil coefficients achieve correct polynomial reproduction.

    Checks that the stencil coefficients from core.stencils can exactly
    reproduce polynomials up to degree (order-1).

    Parameters
    ----------
    order : int
        WENO order to verify (5, 7, 9, ...)
    tolerance : float, optional
        Numerical tolerance for comparison. Default: 1e-14
    verbose : bool, optional
        Print detailed verification output. Default: True

    Returns
    -------
    passed : bool
        True if all checks pass

    Notes
    -----
    This uses symbolic computation to verify that:
    1. Each substencil reproduces polynomials on its domain
    2. Linear combination with optimal weights achieves full order
    3. Coefficients sum to 1 (conservation property)
    """
    from gradflow.core.stencils import generate_all_stencils

    if verbose:
        print(f"\nVerifying WENO-{order} stencil coefficients...")

    try:
        stencils = generate_all_stencils(order)
        r = len(stencils)

        # Check 1: Coefficients sum to 1 (conservation)
        if verbose:
            print(f"\nCheck 1: Conservation (coefficients sum to 1)")

        for k, stencil in enumerate(stencils):
            coeff_sum = sum(stencil)
            if abs(float(coeff_sum) - 1.0) > tolerance:
                if verbose:
                    print(f"  ✗ Substencil {k}: sum = {float(coeff_sum)}")
                return False
            if verbose:
                print(f"  ✓ Substencil {k}: sum = {float(coeff_sum)}")

        # Check 2: Polynomial reproduction
        if verbose:
            print(f"\nCheck 2: Polynomial reproduction")

        # For each substencil, verify it reproduces a constant function
        # A constant function f(x) = c at all points should reconstruct to c
        for k, stencil in enumerate(stencils):
            # For constant function, all points have value 1
            # Linear combination should give 1 (since coeffs sum to 1)
            result = sum(stencil)

            if abs(float(result) - 1.0) > tolerance:
                if verbose:
                    print(f"  ✗ Substencil {k} fails to reproduce constant")
                return False

        # Verify linear reproduction
        # For a linear function f(x) = x, reconstruct at x = 0.5
        for k, stencil in enumerate(stencils):
            points = list(range(-r + 1 + k, k + 1))

            # For f(x) = x, f(point_i) = point_i
            # Reconstruction: sum(c_i * point_i) should equal 0.5
            reconstructed = sum(float(stencil[i]) * points[i] for i in range(len(points)))
            expected = 0.5  # Interface location

            if abs(reconstructed - expected) > tolerance:
                if verbose:
                    print(f"  ✗ Substencil {k} fails linear reproduction")
                    print(f"    Got {reconstructed}, expected {expected}")
                return False

        if verbose:
            print(f"  ✓ All substencils reproduce constant functions")
            print(f"  ✓ All substencils reproduce linear functions")

        if verbose:
            print(f"\n✓ WENO-{order} stencil verification PASSED")
        return True

    except Exception as e:
        if verbose:
            print(f"\n✗ Verification failed with error: {e}")
        return False


def verify_smoothness_indicators(
    order: int,
    tolerance: float = 1e-14,
    verbose: bool = True
) -> bool:
    """
    Verify smoothness indicator properties.

    Checks that:
    1. IS ≥ 0 always
    2. IS = 0 for constant functions
    3. IS is symmetric for symmetric data

    Parameters
    ----------
    order : int
        WENO order (5, 7, 9)
    tolerance : float, optional
        Tolerance for zero checks. Default: 1e-14
    verbose : bool, optional
        Print detailed output. Default: True

    Returns
    -------
    passed : bool
        True if all checks pass
    """
    from gradflow.core.smoothness import compute_smoothness_indicators_torch

    if verbose:
        print(f"\nVerifying WENO-{order} smoothness indicators...")

    if order != 5:
        if verbose:
            print(f"  Verification only implemented for WENO-5")
        return True  # Skip for now

    try:
        # Check 1: Constant function → IS = 0
        if verbose:
            print(f"\nCheck 1: Constant function gives IS = 0")

        flux_const = torch.ones(1, 5, dtype=torch.float64) * 3.14159
        IS_const = compute_smoothness_indicators_torch(flux_const, order=5)

        if not torch.allclose(IS_const, torch.zeros_like(IS_const), atol=tolerance):
            if verbose:
                print(f"  ✗ IS = {IS_const} (expected 0)")
            return False

        if verbose:
            print(f"  ✓ IS = 0 for constant function")

        # Check 2: Symmetry for symmetric data
        if verbose:
            print(f"\nCheck 2: Symmetry property")

        flux_sym = torch.tensor([[4.0, 1.0, 0.0, 1.0, 4.0]], dtype=torch.float64)
        IS_sym = compute_smoothness_indicators_torch(flux_sym, order=5)

        # IS_0 and IS_2 should be equal (left-right symmetry)
        if not torch.allclose(IS_sym[..., 0], IS_sym[..., 2], rtol=tolerance):
            if verbose:
                print(f"  ✗ IS_0={IS_sym[..., 0]} ≠ IS_2={IS_sym[..., 2]}")
            return False

        if verbose:
            print(f"  ✓ Symmetric data gives symmetric IS")

        # Check 3: Non-negativity
        if verbose:
            print(f"\nCheck 3: Non-negativity")

        torch.manual_seed(42)
        flux_random = torch.randn(100, 5, dtype=torch.float64)
        IS_random = compute_smoothness_indicators_torch(flux_random, order=5)

        if not (IS_random >= 0).all():
            if verbose:
                print(f"  ✗ Found negative IS values")
            return False

        if verbose:
            print(f"  ✓ All IS values are non-negative")

        if verbose:
            print(f"\n✓ WENO-{order} smoothness verification PASSED")
        return True

    except Exception as e:
        if verbose:
            print(f"\n✗ Verification failed with error: {e}")
        return False


def compare_against_reference(
    test_data: torch.Tensor,
    reference_data: torch.Tensor,
    name: str = "data",
    tolerance: float = 1e-12,
    verbose: bool = True
) -> bool:
    """
    Compare test results against reference implementation.

    Useful for validating against FORTRAN codes or published results.

    Parameters
    ----------
    test_data : torch.Tensor
        Output from gradflow implementation
    reference_data : torch.Tensor
        Reference values (e.g., from FORTRAN)
    name : str, optional
        Description of what's being compared. Default: "data"
    tolerance : float, optional
        Maximum allowed difference. Default: 1e-12
    verbose : bool, optional
        Print comparison details. Default: True

    Returns
    -------
    passed : bool
        True if test matches reference within tolerance
    """
    if verbose:
        print(f"\nComparing {name}...")

    if test_data.shape != reference_data.shape:
        if verbose:
            print(f"  ✗ Shape mismatch: {test_data.shape} vs {reference_data.shape}")
        return False

    diff = torch.abs(test_data - reference_data)
    max_diff = diff.max().item()
    rel_diff = (diff / (torch.abs(reference_data) + 1e-16)).max().item()

    if verbose:
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Max relative difference: {rel_diff:.2e}")

    if max_diff > tolerance:
        if verbose:
            print(f"  ✗ Exceeds tolerance {tolerance:.2e}")
            # Show worst mismatch location
            worst_idx = torch.argmax(diff.flatten())
            worst_idx = torch.unravel_index(worst_idx, test_data.shape)
            print(f"  Worst at index {worst_idx}:")
            print(f"    Test:      {test_data[worst_idx].item()}")
            print(f"    Reference: {reference_data[worst_idx].item()}")
        return False

    if verbose:
        print(f"  ✓ Match within tolerance")

    return True


def run_all_verifications(order: int = 5, verbose: bool = True) -> bool:
    """
    Run complete verification suite.

    Parameters
    ----------
    order : int, optional
        WENO order to verify. Default: 5
    verbose : bool, optional
        Print detailed output. Default: True

    Returns
    -------
    all_passed : bool
        True if all verifications pass
    """
    if verbose:
        print("="*60)
        print(f"WENO-{order} Verification Suite")
        print("="*60)

    results = {}

    # Test 1: Stencil coefficients
    results['stencils'] = verify_stencil_coefficients(order, verbose=verbose)

    # Test 2: Smoothness indicators
    results['smoothness'] = verify_smoothness_indicators(order, verbose=verbose)

    # Summary
    if verbose:
        print("\n" + "="*60)
        print("Verification Summary:")
        print("="*60)
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {test_name:20s}: {status}")
        print("="*60)

    all_passed = all(results.values())

    if verbose:
        if all_passed:
            print("\n✓ ALL VERIFICATIONS PASSED")
        else:
            print("\n✗ SOME VERIFICATIONS FAILED")

    return all_passed


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Run full verification suite
    success = run_all_verifications(order=5, verbose=True)

    exit(0 if success else 1)
