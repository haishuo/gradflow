# Step 7: Modular Refactoring Plan

## Status: Ready to Begin

Steps 1-6 are complete and validated:
- ✅ NumPy reference matches MATLAB to machine precision
- ✅ PyTorch reference matches both NumPy and MATLAB to machine precision
- ✅ Critical fix documented: `torch.linspace` can produce near-zero values that `torch.sign` treats as -1

**Golden Rule:** Every change must be validated against `gottlieb_weno5_pytorch.py` to ensure machine precision is maintained.

---

## Architecture: What to Refactor

The monolithic `gottlieb_weno5_pytorch.py` contains:
1. **Core WENO algorithm** (150 lines) - flux reconstruction, smoothness indicators, weights
2. **Time integration** (15 lines) - SSP-RK3 loop
3. **Test harness** (30 lines) - Burgers test setup

Current GradFlow structure:
```
gradflow/
├── core/
│   ├── flux.py          ← Update with validated algorithm
│   ├── smoothness.py    ← Currently unused, may delete
│   ├── stencils.py      ← Keep for symbolic generation
│   └── weights.py       ← Currently unused, may delete
├── solvers/
│   ├── weno.py          ← Update with validated solver
│   └── timestepping.py  ← Keep, already works
└── utils/
    └── ...
```

---

## Refactoring Strategy

### Phase 1: Extract Core Algorithm (HIGH PRIORITY)

**File:** `gradflow/core/flux.py`

**Goal:** Replace current implementation with Gottlieb's validated algorithm

**What to extract from `gottlieb_weno5_pytorch.py`:**
```python
def reconstruct_interface_fluxes_fd_weno(
    u_extended: torch.Tensor,
    flux_function: Callable,
    dx: float,
    epsilon: float = 1e-29,
    flux_derivative: Optional[Callable] = None
) -> torch.Tensor:
    """
    Gottlieb's finite difference WENO reconstruction.
    
    This is the CORE algorithm - extracted line-by-line from 
    gottlieb_weno5_pytorch.py with NO algorithmic changes.
    """
    # Lines 48-205 from gottlieb_weno5_pytorch.py
    # (the entire loop-based reconstruction)
```

**Validation test:**
```python
def test_flux_reconstruction():
    # Setup identical to gottlieb_weno5_pytorch
    from gottlieb_weno5_pytorch import weno5_gottlieb_pytorch
    from gradflow.core.flux import reconstruct_interface_fluxes_fd_weno
    
    # Get fh from reference
    # Get fh from new module
    # Assert: max_error < 1e-14
```

**Success criteria:** Interface fluxes match reference to machine precision

---

### Phase 2: Update Solver (MEDIUM PRIORITY)

**File:** `gradflow/solvers/weno.py`

**Goal:** Use the validated flux reconstruction, keep existing time integration

**Changes:**
1. Update `compute_spatial_derivative()` to call new `reconstruct_interface_fluxes_fd_weno()`
2. Fix ghost cell construction to match Gottlieb's convention
3. Add the `linspace` cleaning fix for discontinuous ICs

**Key insight:** The current `WENOSolver` class structure is fine - just the algorithm inside needs updating.

**Validation test:**
```python
def test_solver_vs_reference():
    from gottlieb_weno5_pytorch import burgers_test_gottlieb_pytorch
    from gradflow.solvers.weno import WENOSolver
    
    # Run both
    # Assert: solutions match to machine precision
```

**Success criteria:** Full solver matches `burgers_test_gottlieb_pytorch()` to < 1e-12

---

### Phase 3: Cleanup (LOW PRIORITY)

**Optional improvements after validation passes:**

1. **Remove unused files:**
   - `gradflow/core/smoothness.py` - not needed with loop-based approach
   - `gradflow/core/weights.py` - weights computed inline

2. **Keep symbolic stencils:**
   - `gradflow/core/stencils.py` - useful for documentation/verification

3. **Optimize later:**
   - Current goal: correctness
   - Future: vectorize loops, use conv1d, GPU optimization
   - But ONLY after all tests pass!

---

## Critical Don'ts

**DO NOT:**
- ❌ "Improve" the algorithm while refactoring
- ❌ Vectorize or optimize before validation passes
- ❌ Change variable names "for clarity"
- ❌ Skip validation after any change
- ❌ Merge multiple changes without testing each

**DO:**
- ✅ Copy code exactly from `gottlieb_weno5_pytorch.py`
- ✅ Test after every single change
- ✅ Keep comments explaining the algorithm
- ✅ Maintain the loop structure initially
- ✅ Add unit tests comparing to reference

---

## Testing Strategy

### Test 1: Component-level
```python
# Test each extracted function independently
test_flux_reconstruction_matches_reference()
test_ghost_cells_match_reference()
test_spatial_derivative_matches_reference()
```

### Test 2: Integration-level
```python
# Test the full solver
test_solver_matches_reference()
test_burgers_equation()
test_discontinuous_ic()
```

### Test 3: Edge cases
```python
# Test known problem cases
test_linspace_zero_handling()
test_discontinuity_at_boundary()
test_smooth_solution()
```

---

## Step-by-Step Execution

### Step 7.1: Extract flux reconstruction (TODAY)
1. Create new `reconstruct_interface_fluxes_fd_weno()` in `flux.py`
2. Copy lines 48-205 from `gottlieb_weno5_pytorch.py`
3. Create validation test
4. Run test - must pass with < 1e-14 error

### Step 7.2: Update WENOSolver (TODAY)
1. Update `apply_boundary_conditions()` to match Gottlieb
2. Update `compute_spatial_derivative()` to use new flux function
3. Add `linspace` cleaning in `solve()`
4. Create validation test
5. Run test - must pass with < 1e-12 error

### Step 7.3: Full validation (TODAY)
1. Run all existing GradFlow tests
2. Run new validation tests
3. Compare against `gottlieb_weno5_pytorch.py`
4. Document any differences

### Step 7.4: Cleanup (OPTIONAL)
1. Remove unused files
2. Update documentation
3. Add inline comments explaining Gottlieb's algorithm

---

## Success Criteria for Step 7

Step 7 is complete when:
1. ✅ `flux.py` contains validated flux reconstruction
2. ✅ `weno.py` uses the new flux reconstruction
3. ✅ All tests pass with < 1e-12 error
4. ✅ GradFlow solver matches `gottlieb_weno5_pytorch.py`
5. ✅ Code is clean and well-documented

---

## Next Steps After Step 7

Once modular refactoring is complete:
1. **Optimize:** Vectorize loops, use conv1d
2. **GPU:** Test and optimize CUDA performance
3. **Extend:** Add WENO-7, WENO-9
4. **Applications:** Add more example problems

But for now: **Focus on correctness, not performance.**

---

## Questions Before Starting?

Before proceeding:
- Review this plan
- Identify any concerns
- Clarify any steps
- Then we'll start with Step 7.1
