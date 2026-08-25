"""
Conversion utilities for symbolic coefficients to PyTorch tensors.

Simplified module that handles converting exact rational coefficients
from stencil generation into efficient PyTorch tensors.
"""

import torch
import sympy as sp
from typing import List, Union


def rational_to_float(value: Union[sp.Rational, float, int]) -> float:
    """
    Convert SymPy rational to Python float.

    Parameters
    ----------
    value : sp.Rational, float, or int
        Value to convert

    Returns
    -------
    float_value : float
        Floating point representation

    Examples
    --------
    >>> rational_to_float(sp.Rational(11, 6))
    1.8333333333333333
    """
    if isinstance(value, (float, int)):
        return float(value)
    elif isinstance(value, sp.Rational):
        return float(value)
    else:
        # Try to convert via SymPy
        return float(sp.sympify(value))


def stencils_to_torch(
    stencil_coeffs: List[List[sp.Rational]],
    dtype: torch.dtype = torch.float64,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Convert stencil coefficients to PyTorch tensor.

    Takes the exact rational coefficients from symbolic stencil generation
    and converts them to a PyTorch tensor ready for GPU computation.

    Parameters
    ----------
    stencil_coeffs : List[List[sp.Rational]]
        Stencil coefficients from core.stencils.generate_all_stencils()
        Shape: [r, r] where r = (order+1)//2
        Each inner list contains exact rational coefficients
    dtype : torch.dtype, optional
        PyTorch data type. Default: torch.float64
        Use float64 for research/accuracy, float32 for speed
    device : str, optional
        Device placement. Default: 'cuda'
        Use 'cuda' for GPU, 'cpu' for CPU

    Returns
    -------
    stencil_tensor : torch.Tensor
        Tensor of shape [r, r] ready for convolution operations
        Can be used with torch.nn.functional.conv1d

    Raises
    ------
    RuntimeError
        If CUDA requested but not available

    Notes
    -----
    This conversion loses exact rational arithmetic but preserves
    sufficient precision for numerical computation. Float64 provides
    ~16 decimal digits, which is more than adequate for WENO schemes.

    Examples
    --------
    >>> from gradflow.core.stencils import generate_all_stencils
    >>> stencils = generate_all_stencils(5)
    >>> tensor = stencils_to_torch(stencils)
    >>> tensor.shape
    torch.Size([3, 3])
    """
    # Validate device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but not available. "
            "Install CUDA-enabled PyTorch or use device='cpu'."
        )

    # Convert to nested Python floats
    float_coeffs = [
        [rational_to_float(c) for c in stencil]
        for stencil in stencil_coeffs
    ]

    # Create PyTorch tensor
    tensor = torch.tensor(float_coeffs, dtype=dtype, device=device)

    return tensor


def coefficients_dict_to_torch(
    coeffs_dict: dict,
    dtype: torch.dtype = torch.float64,
    device: str = 'cuda'
) -> dict:
    """
    Convert a dictionary of coefficients to PyTorch tensors.

    Useful for converting smoothness indicator coefficients or
    other structured coefficient dictionaries.

    Parameters
    ----------
    coeffs_dict : dict
        Dictionary with numeric values (may contain sp.Rational)
    dtype : torch.dtype, optional
        Target PyTorch dtype. Default: torch.float64
    device : str, optional
        Target device. Default: 'cuda'

    Returns
    -------
    tensor_dict : dict
        Same structure but with PyTorch tensors

    Examples
    --------
    >>> coeffs = {0: sp.Rational(13, 12), 1: sp.Rational(1, 4)}
    >>> torch_coeffs = coefficients_dict_to_torch(coeffs)
    """
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    result = {}
    for key, value in coeffs_dict.items():
        if isinstance(value, dict):
            # Recursively handle nested dicts
            result[key] = coefficients_dict_to_torch(value, dtype, device)
        elif isinstance(value, list):
            # Convert list of coefficients
            float_list = [rational_to_float(v) for v in value]
            result[key] = torch.tensor(float_list, dtype=dtype, device=device)
        else:
            # Convert single coefficient
            result[key] = torch.tensor(
                rational_to_float(value),
                dtype=dtype,
                device=device
            )

    return result


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing symbolic → PyTorch conversion...")

    # Test 1: Convert SymPy rationals
    print("\n" + "="*60)
    print("Test 1: SymPy Rational conversion")

    test_rationals = [
        sp.Rational(2, 6),
        sp.Rational(-7, 6),
        sp.Rational(11, 6),
    ]

    for rat in test_rationals:
        flt = rational_to_float(rat)
        print(f"  {rat} → {flt}")

    print("✓ Rational conversion working")

    # Test 2: Stencil conversion
    print("\n" + "="*60)
    print("Test 2: Stencil coefficient conversion")

    # Mock stencil coefficients (like what stencils.py produces)
    mock_stencils = [
        [sp.Rational(2, 6), sp.Rational(-7, 6), sp.Rational(11, 6)],
        [sp.Rational(-1, 6), sp.Rational(5, 6), sp.Rational(2, 6)],
        [sp.Rational(2, 6), sp.Rational(5, 6), sp.Rational(-1, 6)],
    ]

    tensor = stencils_to_torch(mock_stencils, device='cpu')

    print(f"  Input: 3 stencils with 3 coefficients each")
    print(f"  Output shape: {tensor.shape}")
    print(f"  Output dtype: {tensor.dtype}")
    print(f"  Output device: {tensor.device}")
    print(f"\n  First stencil:")
    print(f"  {tensor[0]}")

    # Verify values are correct
    expected_first = [2/6, -7/6, 11/6]
    computed_first = tensor[0].tolist()

    for exp, comp in zip(expected_first, computed_first):
        assert abs(exp - comp) < 1e-15, f"Mismatch: {exp} vs {comp}"

    print("✓ Stencil conversion accurate")

    # Test 3: Dictionary conversion
    print("\n" + "="*60)
    print("Test 3: Dictionary conversion")

    coeffs_dict = {
        0: sp.Rational(13, 12),
        1: sp.Rational(1, 4),
        2: [sp.Rational(1, 1), sp.Rational(-2, 1), sp.Rational(1, 1)]
    }

    torch_dict = coefficients_dict_to_torch(coeffs_dict, device='cpu')

    print(f"  Converted {len(torch_dict)} entries")
    print(f"  Entry 0: {torch_dict[0].item()}")
    print(f"  Entry 1: {torch_dict[1].item()}")
    print(f"  Entry 2 (list): {torch_dict[2].tolist()}")

    assert abs(torch_dict[0].item() - 13/12) < 1e-15
    assert abs(torch_dict[1].item() - 1/4) < 1e-15

    print("✓ Dictionary conversion working")

    print("\n" + "="*60)
    print("✓ All conversion tests passed")
