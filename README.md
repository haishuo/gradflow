# GradFlow

Modern WENO (Weighted Essentially Non-Oscillatory) framework for solving hyperbolic partial differential equations, built from mathematical principles using symbolic computation.

## Motivation

Existing WENO implementations on GPU are limited to 5th order due to the complexity of hand-coding CUDA kernels. This project uses SymPy for symbolic derivation and PyTorch for automatic GPU acceleration, enabling easy experimentation with higher-order schemes (WENO-7, WENO-9, WENO-11+).

**Key differences from traditional implementations:**
- Mathematical first principles (no FORTRAN archaeology)
- Symbolic generation of stencil coefficients and smoothness indicators
- Automatic GPU parallelization via PyTorch
- Clean, modular design following UNIX philosophy
- Order-agnostic framework: change one parameter to get WENO-7, WENO-9, etc.

## Project Structure

```
gradflow/
├── gradflow/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── stencils.py          # Symbolic stencil generation
│   │   ├── smoothness.py        # Smoothness indicator generation
│   │   ├── weights.py           # WENO weight computation
│   │   └── flux.py              # Flux splitting and reconstruction
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── weno.py              # Main WENO solver class
│   │   └── timestepping.py      # SSP-RK and other time integrators
│   ├── symbolic/
│   │   ├── __init__.py
│   │   ├── generator.py         # SymPy → PyTorch code generation
│   │   └── verification.py      # Symbolic verification of correctness
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── boundary.py          # Boundary condition handling
│   │   ├── profiling.py         # Performance profiling utilities
│   │   └── validation.py        # Convergence testing, order verification
│   └── examples/
│       ├── __init__.py
│       ├── burgers.py           # Burgers equation example
│       ├── euler.py             # Euler equations example
│       └── benchmarks.py        # Performance benchmarks
├── tests/
│   ├── test_stencils.py
│   ├── test_smoothness.py
│   ├── test_convergence.py
│   └── test_symbolic.py
├── docs/
│   ├── mathematical_derivation.md
│   ├── api_reference.md
│   └── examples.md
├── setup.py
├── requirements.txt
└── README.md
```

## Design Principles

1. **UNIX Philosophy**: Each module does one thing well (< 400 lines)
2. **Math First**: Correctness before performance
3. **No Defaults**: Explicit parameters everywhere (except top-level API)
4. **Fail Fast**: Validate inputs immediately with clear error messages
5. **Profile-Driven Optimization**: Build with profiling in mind, optimize after verification

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from gradflow.solvers import WENOSolver

# Create WENO-5 solver
solver = WENOSolver(order=5, grid_size=200, domain_length=2*np.pi)

# Initial condition
u0 = np.sin(solver.x)

# Solve Burgers' equation
def burgers_flux(u):
    return 0.5 * u**2

u_final = solver.solve(u0, t_final=1.0, flux_function=burgers_flux)
```

Change `order=5` to `order=7` or `order=11` to use higher-order schemes - everything else is automatic.

## Mathematical Foundation

WENO schemes adaptively combine polynomial reconstructions from multiple stencils:

1. **Stencil coefficients** from Lagrange interpolation
2. **Smoothness indicators** measuring L² norm of derivatives
3. **Nonlinear weights** that penalize non-smooth stencils
4. **Flux reconstruction** via weighted combination

See `docs/mathematical_derivation.md` for detailed derivations.

## Testing

```bash
pytest tests/
```

Convergence tests verify order of accuracy. Symbolic tests validate that generated formulas match theoretical results.

## Performance

PyTorch handles GPU parallelization automatically. For optimal performance:
- Use `dtype=torch.float64` for most applications
- Mixed precision available for specialized cases
- Profiling tools in `utils/profiling.py`

## Applications

Higher-order WENO on GPU enables:
- MHD turbulence simulations with shock capturing
- Relativistic jet propagation
- Shock-turbulence interaction studies
- Combustion detonation modeling

See `examples/` for complete working examples.

## References

- Jiang & Shu (1996): "Efficient Implementation of Weighted ENO Schemes"
- Field et al. (2020): "A GPU-accelerated mixed-precision WENO method"
- Shu & Osher (1988): "Efficient implementation of essentially non-oscillatory shock-capturing schemes"

## License

MIT

## Contributing

This is a research project. Contributions welcome, especially:
- Additional test cases
- Higher-order formulas (WENO-9+)
- Performance optimizations
- Application examples
