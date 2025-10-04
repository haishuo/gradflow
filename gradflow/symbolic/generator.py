"""
SymPy to PyTorch code generation.

This module converts symbolic expressions (SymPy) into executable
PyTorch operations that can run on GPU.
"""

import sympy as sp
import torch
from typing import List, Callable, Dict, Any
import warnings


def sympy_to_torch_function(
    expr: sp.Expr,
    variables: List[sp.Symbol],
    dtype: torch.dtype = torch.float64,
    device: str = 'cuda'
) -> Callable:
    """
    Convert a SymPy expression to a PyTorch function.

    Takes a symbolic expression and returns a function that can be called
    with PyTorch tensors as arguments.

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression to convert
    variables : List[sp.Symbol]
        Ordered list of symbolic variables in the expression.
        The returned function will expect arguments in this order.
    dtype : torch.dtype, optional
        PyTorch dtype for computations. Default: torch.float64
    device : str, optional
        Device for computation ('cuda' or 'cpu'). Default: 'cuda'

    Returns
    -------
    torch_func : Callable
        Function that takes PyTorch tensors and returns a PyTorch tensor.
        Signature: torch_func(*tensors) -> torch.Tensor

    Raises
    ------
    ValueError
        If expression contains symbols not in variables list

    Notes
    -----
    This function uses SymPy's lambdify with PyTorch as the backend.
    All operations are translated to their PyTorch equivalents:
        - Addition, multiplication, etc. → torch operations
        - Powers → torch.pow
        - Trigonometric → torch.sin, torch.cos, etc.

    The conversion happens at function creation time, so there's no
    symbolic overhead during actual computation.

    Examples
    --------
    >>> x, y = sp.symbols('x y')
    >>> expr = x**2 + 2*x*y + y**2
    >>> f = sympy_to_torch_function(expr, [x, y])
    >>> result = f(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    >>> result
    tensor([16., 36.])  # (1+3)^2, (2+4)^2
    """
    # Validate that all symbols in expression are in variables list
    expr_symbols = expr.free_symbols
    var_symbols = set(variables)

    if not expr_symbols.issubset(var_symbols):
        missing = expr_symbols - var_symbols
        raise ValueError(
            f"Expression contains symbols {missing} not in variables list. "
            f"Provided variables: {variables}"
        )

    # Create lambdified function with torch module
    # This converts SymPy operations to PyTorch operations
    try:
        # Use 'torch' as the module for lambdify
        # This makes sp.sin → torch.sin, sp.exp → torch.exp, etc.
        base_func = sp.lambdify(variables, expr, modules='torch')
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert SymPy expression to PyTorch: {e}\n"
            f"Expression: {expr}\n"
            f"Variables: {variables}"
        )

    def torch_func(*args):
        """
        Wrapper that ensures correct dtype and device.
        """
        # Validate number of arguments
        if len(args) != len(variables):
            raise ValueError(
                f"Expected {len(variables)} arguments, got {len(args)}"
            )

        # Ensure all arguments are tensors on correct device with correct dtype
        args_converted = []
        for i, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                # Convert to tensor if needed
                arg = torch.tensor(arg, dtype=dtype, device=device)
            else:
                # Ensure correct dtype and device
                arg = arg.to(dtype=dtype, device=device)
            args_converted.append(arg)

        # Call the lambdified function
        result = base_func(*args_converted)

        # Ensure result is a tensor (sometimes scalars are returned)
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, dtype=dtype, device=device)

        return result

    return torch_func


def batch_sympy_to_torch(
    expressions: List[sp.Expr],
    variables: List[sp.Symbol],
    dtype: torch.dtype = torch.float64,
    device: str = 'cuda'
) -> List[Callable]:
    """
    Convert multiple SymPy expressions to PyTorch functions.

    Convenience function for converting a list of expressions that
    share the same variables.

    Parameters
    ----------
    expressions : List[sp.Expr]
        List of SymPy expressions to convert
    variables : List[sp.Symbol]
        Ordered list of variables (same for all expressions)
    dtype : torch.dtype, optional
        PyTorch dtype. Default: torch.float64
    device : str, optional
        Device ('cuda' or 'cpu'). Default: 'cuda'

    Returns
    -------
    torch_funcs : List[Callable]
        List of PyTorch functions, one per expression

    Examples
    --------
    >>> x = sp.Symbol('x')
    >>> exprs = [x**2, x**3, x**4]
    >>> funcs = batch_sympy_to_torch(exprs, [x])
    >>> [f(torch.tensor(2.0)) for f in funcs]
    [tensor(4.), tensor(8.), tensor(16.)]
    """
    return [
        sympy_to_torch_function(expr, variables, dtype, device)
        for expr in expressions
    ]


def compile_stencil_operations(
    stencil_coeffs: List[List[sp.Rational]],
    dtype: torch.dtype = torch.float64,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Convert stencil coefficients to a compiled PyTorch kernel.

    This is a specialized version for stencil coefficients that
    creates a tensor ready for convolution operations.

    Parameters
    ----------
    stencil_coeffs : List[List[sp.Rational]]
        Stencil coefficients from core.stencils
    dtype : torch.dtype
        PyTorch dtype
    device : str
        Device for tensor

    Returns
    -------
    kernel_tensor : torch.Tensor
        Tensor of shape [num_stencils, stencil_width]
        Ready for use with F.conv1d

    Notes
    -----
    This is essentially a wrapper around direct tensor creation,
    but it's here in the symbolic module for consistency and
    future extensibility (e.g., JIT compilation).
    """
    # Convert symbolic rationals to floats
    float_coeffs = [[float(c) for c in stencil] for stencil in stencil_coeffs]

    # Create tensor
    kernel = torch.tensor(float_coeffs, dtype=dtype, device=device)

    return kernel


def verify_conversion(
    sympy_expr: sp.Expr,
    torch_func: Callable,
    variables: List[sp.Symbol],
    test_points: int = 100,
    tolerance: float = 1e-10
) -> bool:
    """
    Verify that PyTorch function produces same results as SymPy expression.

    Tests the converted function against the original symbolic expression
    at random points to ensure correctness.

    Parameters
    ----------
    sympy_expr : sp.Expr
        Original SymPy expression
    torch_func : Callable
        Converted PyTorch function
    variables : List[sp.Symbol]
        Variables in the expression
    test_points : int, optional
        Number of random test points. Default: 100
    tolerance : float, optional
        Maximum allowed difference. Default: 1e-10

    Returns
    -------
    verified : bool
        True if all test points match within tolerance

    Raises
    ------
    AssertionError
        If any test point fails (with details about the failure)

    Notes
    -----
    This is primarily for testing during development. In production,
    you should trust that lambdify works correctly.
    """
    # Generate random test points
    test_values = torch.randn(test_points, len(variables), dtype=torch.float64)

    max_error = 0.0

    for i in range(test_points):
        point = test_values[i]

        # Evaluate SymPy expression (slow, exact)
        subs_dict = {var: float(point[j]) for j, var in enumerate(variables)}
        sympy_result = float(sympy_expr.subs(subs_dict))

        # Evaluate PyTorch function (fast, numerical)
        torch_args = [point[j:j+1] for j in range(len(variables))]
        torch_result = float(torch_func(*torch_args))

        # Check error
        error = abs(sympy_result - torch_result)
        max_error = max(max_error, error)

        if error > tolerance:
            raise AssertionError(
                f"Conversion verification failed at test point {i}\n"
                f"Point: {[float(p) for p in point]}\n"
                f"SymPy result: {sympy_result}\n"
                f"PyTorch result: {torch_result}\n"
                f"Error: {error} > tolerance {tolerance}"
            )

    if max_error > tolerance / 10:
        warnings.warn(
            f"Conversion verification passed, but maximum error {max_error} "
            f"is close to tolerance {tolerance}. Consider using higher precision."
        )

    return True


if __name__ == "__main__":
    # Self-test: convert simple expressions
    print("Testing SymPy → PyTorch conversion...")

    # Test 1: Polynomial
    x, y = sp.symbols('x y')
    expr = x**2 + 2*x*y + y**2
    f = sympy_to_torch_function(expr, [x, y], device='cpu')

    result = f(torch.tensor(2.0), torch.tensor(3.0))
    expected = (2 + 3)**2
    print(f"\nTest 1: (x+y)² at x=2, y=3")
    print(f"Result: {result.item()}, Expected: {expected}")
    assert abs(result.item() - expected) < 1e-10, "Polynomial test failed"

    # Test 2: Batch conversion
    exprs = [x**2, x**3, x**4]
    funcs = batch_sympy_to_torch(exprs, [x], device='cpu')

    x_val = torch.tensor(2.0)
    results = [f(x_val).item() for f in funcs]
    expected = [4.0, 8.0, 16.0]
    print(f"\nTest 2: Powers of 2")
    print(f"Results: {results}, Expected: {expected}")
    assert results == expected, "Batch conversion test failed"

    # Test 3: Verify conversion
    expr = x**3 - 3*x**2 + 2*x - 1
    f = sympy_to_torch_function(expr, [x], device='cpu')
    verified = verify_conversion(expr, f, [x], test_points=50)
    print(f"\nTest 3: Verification with 50 random points")
    print(f"Verified: {verified}")

    print("\n✓ All SymPy → PyTorch conversion tests passed")
