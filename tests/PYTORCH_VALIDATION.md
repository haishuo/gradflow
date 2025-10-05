# PyTorch Translation Validation

## Status: Step 5 & 6 of Rebuild Plan

### Completed
✅ Step 1: Downloaded Professor Gottlieb's MATLAB code  
✅ Step 2: Generated .h5 gold standard output  
✅ Step 3: Created NumPy line-by-line translation (`gottlieb_weno5_reference.py`)  
✅ Step 4: Validated NumPy vs MATLAB (machine precision match)  
✅ Step 5: Created PyTorch line-by-line translation (`gottlieb_weno5_pytorch.py`)  
⏳ Step 6: Validate PyTorch vs gold standard (RUNNING NOW)

### Next: Step 7
After validation passes, refactor into modular components:
- Update `gradflow/solvers/weno.py`
- Update `gradflow/core/flux.py`
- Update other components
- Test each module as you go

---

## Running Validation (Step 6)

### Quick Test
```bash
cd tests
python run_pytorch_validation.py
```

### What It Checks

1. **PyTorch vs NumPy Reference**
   - Ensures PyTorch translation is correct
   - Should match to ~1e-14 (machine precision for float64)

2. **PyTorch vs MATLAB Gold Standard**
   - Compares against the .h5 file
   - This is the ultimate validation
   - Should match to ~1e-12 or better

### Expected Output

If successful:
```
✓✓✓ ALL VALIDATIONS PASSED ✓✓✓

Results:
  ✓ PyTorch matches NumPy reference to machine precision
  ✓ PyTorch matches MATLAB gold standard to machine precision

STATUS: Step 5 & 6 COMPLETE

NEXT STEP: Proceed to step 7
```

---

## Files Created

### Main Implementation
- `gottlieb_weno5_pytorch.py` - Line-by-line PyTorch translation
  - Keeps same structure as NumPy version
  - Uses torch.Tensor instead of np.ndarray
  - Same loops, same indexing, same MATLAB-isms
  - Ready for GPU but runs on CPU for validation

### Validation Script
- `run_pytorch_validation.py` - Automated validation runner
  - Tests against NumPy reference
  - Tests against MATLAB .h5 gold standard
  - Clear pass/fail reporting

---

## Key Differences from NumPy Version

The PyTorch version:
- Uses `torch.Tensor` instead of `np.ndarray`
- Uses `torch.cat()` instead of `np.concatenate()`
- Uses `.item()` to extract scalar values
- Uses `.clone()` instead of `.copy()`
- Has `device` and `dtype` parameters for GPU readiness

Everything else is IDENTICAL to ensure correctness.

---

## Debugging Tips

If validation fails:

1. **Check the error magnitude**
   - < 1e-12: Likely just floating-point differences (OK)
   - > 1e-10: Real algorithmic difference (needs fixing)

2. **Look at error location**
   - Errors at boundaries: Ghost cell issue
   - Errors at discontinuities: Flux splitting issue
   - Uniform errors: Likely a systematic bug

3. **Compare intermediate values**
   - Enable `debug=True` in function calls
   - Check flux splitting values
   - Check smoothness indicators
   - Check WENO weights

4. **Verify flux functions**
   - Make sure f(u) and fp(u) work with tensors
   - Check scalar vs tensor handling

---

## Next Steps After Validation

Once validation passes (Step 6 complete), proceed to Step 7:

### Step 7: Modular Refactoring

1. **Start with flux.py**
   - Extract the flux reconstruction loop
   - Keep validation against PyTorch reference
   - Test after each change

2. **Then update weno.py**
   - Use the validated flux.py
   - Add time stepping
   - Test against PyTorch reference

3. **Continue with other modules**
   - One module at a time
   - Always validate against PyTorch reference
   - Never lose correctness

### Validation Strategy for Step 7

After updating each module:
```python
# Test the modular version against PyTorch reference
from gottlieb_weno5_pytorch import burgers_test_gottlieb_pytorch
from gradflow.solvers.weno import WENOSolver

# Get reference
x_ref, u_ref = burgers_test_gottlieb_pytorch()

# Test modular version
solver = WENOSolver(...)
u_mod = solver.solve(...)

# Must match to machine precision
assert max_error < 1e-12
```

---

## Design Principles for Step 7

When refactoring to modular components:

1. **Preserve the algorithm**
   - Don't "improve" or "optimize" yet
   - Just reorganize the SAME code
   - Validate after each change

2. **Test continuously**
   - After every function extracted
   - Against PyTorch reference
   - Fail fast if accuracy degrades

3. **Document departures**
   - If you MUST change something, document why
   - Explain the change in comments
   - Test more thoroughly

4. **GPU readiness**
   - Keep using torch.Tensor throughout
   - Keep device/dtype parameters
   - But don't actually test GPU until CPU works

---

## Success Criteria

**Step 6 passes when:**
- PyTorch vs NumPy: error < 1e-12
- PyTorch vs MATLAB: error < 1e-12

**Step 7 passes when:**
- All modular components validated individually
- Full solver matches PyTorch reference < 1e-12
- Code is clean and well-documented
- Ready for GPU optimization

---

## Questions?

If something fails or is unclear:
1. Check the error magnitude and location
2. Enable debug output
3. Compare with NumPy version
4. Review this guide's debugging tips
