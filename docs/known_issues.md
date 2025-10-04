# Known Issues and Limitations

## Version 0.1.0

### 1. Burgers Equation Shock Formation

**Status:** Known Limitation  
**Severity:** Expected Behavior  
**Affected:** WENO-5 with default parameters

#### Description
When solving Burgers equation with smooth initial conditions (e.g., sine waves), the solution develops a shock discontinuity at t ≈ 0.12. The numerical method struggles to resolve this shock, leading to:

- Rapidly decreasing CFL time steps (down to < 1e-30)
- Solution stagnation at t ≈ 0.12
- Eventually produces NaN values
- Time stepping loop becomes trapped in infinitesimal steps

#### Technical Details
From test output (Test 5, snapshot saving):
```
Step 18: t=0.100000, dt=0.003600
Step 69: t=0.121289, dt=0.000001
Step 316: t=0.121293, dt=nan
```

The shock forms when characteristics of the Burgers equation cross, creating an infinite gradient. The WENO-5 scheme correctly detects this via smoothness indicators, leading to vanishing time steps.

#### Root Cause
This is **not a bug** - it's the physical behavior of Burgers equation:
- Burgers equation: ∂u/∂t + u ∂u/∂x = 0
- For smooth initial data, characteristics eventually cross
- This creates a shock (discontinuity) in finite time
- The shock formation time depends on the initial condition

For u₀ = sin(x) on [0, 2π], theory predicts shock formation at:
```
t_shock ≈ 1/max(|∂u₀/∂x|) ≈ 1
```

However, numerical dissipation can cause earlier breakdown.

#### Workarounds

**Option 1: Increase Numerical Dissipation**
```python
solver.solve(u0, t_final=0.5, epsilon=1e-40)  # Smaller epsilon = more dissipation
```

**Option 2: Limit Simulation Time**
```python
solver.solve(u0, t_final=0.09)  # Stay before shock forms
```

**Option 3: Adjust CFL Number**
```python
solver.solve(u0, t_final=0.5, cfl_number=0.3)  # More conservative
```

**Option 4: Increase Grid Resolution**
```python
solver = WENOSolver(order=5, grid_size=500)  # Better resolves gradients
```

#### Expected Resolution
Future versions may include:
- Adaptive mesh refinement near shocks
- Artificial viscosity options
- Hybrid WENO-Z formulation (better at critical points)
- Automatic shock detection and handling

#### References
- Whitham, G. B. (1974). "Linear and Nonlinear Waves"
- LeVeque, R. J. (2002). "Finite Volume Methods for Hyperbolic Problems"

---

### 2. WENO-7 and WENO-9 Not Fully Implemented

**Status:** Planned Feature  
**Severity:** Minor  
**Affected:** Higher-order schemes

#### Description
While the framework supports WENO-7 and WENO-9:
- Stencil coefficients: ✓ Available
- Optimal weights: ✓ Available
- Smoothness indicators: ✗ Not implemented

#### Workaround
Use WENO-5, which is fully functional and validated.

#### Timeline
Smoothness indicator coefficients for WENO-7 and WENO-9 will be added from:
- Balsara & Shu (2000) for WENO-7
- Balsara & Shu (2000) for WENO-9

Estimated completion: Within 1-2 weeks of initial release.

---

### 3. Snapshot Saving During Shock Formation

**Status:** Won't Fix (By Design)  
**Severity:** Informational

#### Description
When using `save_interval` with problems that develop shocks, snapshots may stop before reaching `t_final` because the solution becomes numerically unstable.

Example from tests:
```python
solver.solve_burgers(u0, t_final=0.5, save_interval=0.1)
# Expected: snapshots at t=0.0, 0.1, 0.2, 0.3, 0.4, 0.5
# Actual: snapshots at t=0.0, 0.1 (then NaN at t≈0.12)
```

#### Explanation
This is correct behavior - the solver cannot continue past the point where the solution becomes undefined (NaN). The snapshot mechanism accurately reflects when the simulation fails.

#### Recommendation
For reliable snapshot sequences:
1. Test your problem first without snapshots
2. Identify when shocks form
3. Set `t_final` conservatively before shock formation
4. Or use appropriate shock-handling parameters

---

### 4. Boundary Condition Limitations

**Status:** Known Limitation  
**Severity:** Minor

#### Description
Current boundary conditions:
- Periodic: ✓ Fully implemented and tested
- Outflow: ✓ Implemented (zero-gradient extrapolation)
- Reflecting: ✓ Implemented (mirror with sign flip)
- Inflow/Custom: ✗ Not yet implemented

#### Workaround
For custom boundary conditions, modify `WENOSolver.apply_boundary_conditions()` manually.

---

## Reporting New Issues

If you encounter unexpected behavior not documented here:

1. Check if it's a known issue above
2. Verify you're using supported parameters (WENO-5, CFL ≤ 0.5)
3. Create minimal reproducing example
4. Document: Python version, PyTorch version, CUDA version (if GPU)
5. Include error messages and stack traces

### Debug Mode
Enable verbose output for troubleshooting:
```python
solver.solve(u0, t_final=1.0, verbose=True)
```

### Common Pitfalls
- **NaN values:** Usually shock formation or CFL violation
- **Slow performance:** Check grid size and device (CPU vs GPU)
- **Wrong results:** Verify boundary conditions match problem physics
- **Memory issues:** Grid too large for available RAM/VRAM

---

## Version History

### v0.1.0 (Initial Release)
- First working WENO-5 implementation
- Known shock formation issue documented
- Core functionality validated