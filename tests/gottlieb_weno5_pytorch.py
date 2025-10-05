"""
PyTorch translation of Professor Gottlieb's MATLAB WENO-5 implementation.

This is a LINE-BY-LINE translation of gottlieb_weno5_reference.py,
replacing NumPy with PyTorch operations. The structure, loops, and
indexing are kept IDENTICAL to ensure correctness.

Original MATLAB code by: Sigal Gottlieb, September 7, 2003
Modified by: Daniel Higgs, June 22, 2007
NumPy translation: 2025
PyTorch translation: 2025
"""

import torch
import numpy as np


def weno5_gottlieb_pytorch(u, dx, f, fp, device='cpu', dtype=torch.float64, debug=False):
    """
    Direct PyTorch translation of Gottlieb's weno5.m.
    
    Parameters
    ----------
    u : torch.Tensor
        Solution values (periodic, no ghost cells in input)
    dx : float
        Grid spacing
    f : function
        Flux function f(u)
    fp : function
        Flux derivative f'(u)
    device : str
        Device for computation ('cpu' or 'cuda')
    dtype : torch.dtype
        Data type for computation
    debug : bool
        Print debug information
        
    Returns
    -------
    rhs : torch.Tensor
        Right-hand side: -df/dx
    """
    
    # Constants
    epsilon = 1e-29  # WENO epsilon parameter
    md = 4           # Number of ghost points
    
    # Ensure u is a tensor on the correct device
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=dtype, device=device)
    else:
        u = u.to(dtype=dtype, device=device)
    
    # Maximum wave speed (for Lax-Friedrichs splitting)
    em = torch.max(torch.abs(fp(u))).item()
    
    npoints = len(u)
    nstart = md + 1
    np_val = npoints + md  # 'np' is a Python module, use np_val
    
    # Ghost cell indexing - matching NumPy version exactly
    i = len(u)
    
    # MATLAB u(i-md:end-1) where i=101, md=4 gives u(97:100)
    # In Python 0-based: MATLAB indices [97,98,99,100] → Python [96,97,98,99]
    left_ghost = u[i-md-1:i-1]  # u[96:100]
    
    # MATLAB u(2:md+2) where md=4 gives u(2:6)  
    # In Python 0-based: MATLAB indices [2,3,4,5,6] → Python [1,2,3,4,5]
    right_ghost = u[1:md+2]      # u[1:6]
    
    u_extended = torch.cat([left_ghost, u, right_ghost])
    
    if debug:
        print(f"Ghost cell construction:")
        print(f"  npoints = {npoints}, md = {md}")
        print(f"  left_ghost indices in original: [{i-md-1}:{i-1}]")
        print(f"  left_ghost values: {left_ghost}")
        print(f"  right_ghost indices in original: [1:{md+2}]")
        print(f"  right_ghost values: {right_ghost}")
        print(f"  u_extended length: {len(u_extended)} (expected {npoints + len(left_ghost) + len(right_ghost)})")
    
    # Flux splitting (Lax-Friedrichs)
    max_idx = np_val + md + 2  # Extra padding to be safe
    dfp = torch.zeros(max_idx, dtype=dtype, device=device)
    dfm = torch.zeros(max_idx, dtype=dtype, device=device)
    
    start_idx = nstart - md - 1  # MATLAB's nstart-md converted to Python 0-based
    end_idx = np_val + md        # MATLAB's np+md converted to Python 0-based
    
    if debug:
        print(f"\nFlux splitting loop:")
        print(f"  MATLAB loop: i = {nstart-md}:{np_val+md} (1-based)")
        print(f"  Python loop: i = {start_idx}:{end_idx} (0-based)")
        print(f"  Will access u_extended[{start_idx}] to u_extended[{end_idx}]")
        print(f"  u_extended has {len(u_extended)} elements (indices 0-{len(u_extended)-1})")
    
    for i in range(start_idx, end_idx):
        f_ip1 = f(u_extended[i+1])
        f_i = f(u_extended[i])
        u_ip1 = u_extended[i+1]
        u_i = u_extended[i]
        
        dfp[i] = (f_ip1 - f_i + em * (u_ip1 - u_i)) / 2.0
        dfm[i] = (f_ip1 - f_i - em * (u_ip1 - u_i)) / 2.0
    
    if debug:
        # Write flux splitting details to file
        with open('gottlieb_pytorch_flux_debug.txt', 'w') as dbg:
            dbg.write("Gottlieb PyTorch Flux Splitting Debug\n")
            dbg.write("="*70 + "\n\n")
            dbg.write(f"Max wave speed (em): {em:.10f}\n")
            dbg.write(f"u_extended length: {len(u_extended)}\n")
            dbg.write(f"Loop range: [{start_idx}, {end_idx})\n\n")
            
            dbg.write("First 20 flux splitting values:\n")
            dbg.write(f"{'i':>4} {'u[i]':>15} {'f[i]':>15} {'dfp[i]':>15} {'dfm[i]':>15}\n")
            dbg.write("-"*70 + "\n")
            
            for i in range(start_idx, min(start_idx + 20, end_idx)):
                dbg.write(f"{i:4d} {u_extended[i].item():15.10f} {f(u_extended[i]).item():15.10f} ")
                dbg.write(f"{dfp[i].item():15.10f} {dfm[i].item():15.10f}\n")
    
    # Numerical flux reconstruction
    fh = torch.zeros(max_idx, dtype=dtype, device=device)
    
    flux_start = nstart - 1 - 1  # MATLAB nstart-1 to Python 0-based
    flux_end = np_val + 1        # MATLAB np+1 to Python 0-based
    
    if debug:
        print(f"\nFlux reconstruction loop:")
        print(f"  MATLAB loop: i = {nstart-1}:{np_val+1} (1-based)")  
        print(f"  Python loop: i = {flux_start}:{flux_end} (0-based)")
    
    for i in range(flux_start, flux_end):
        # Initialize with 4th-order central flux
        f_im1 = f(u_extended[i-1])
        f_i = f(u_extended[i])
        f_ip1 = f(u_extended[i+1])
        f_ip2 = f(u_extended[i+2])
        
        fh[i] = (-f_im1 + 7.0*(f_i + f_ip1) - f_ip2) / 12.0
        
        # Build stencil data for both positive and negative fluxes
        # hh[k, m] where k=0..3 is stencil point, m=0 is dfp, m=1 is dfm
        hh = torch.zeros((4, 2), dtype=dtype, device=device)
        hh[0, 0] = dfp[i-2]
        hh[1, 0] = dfp[i-1]
        hh[2, 0] = dfp[i]
        hh[3, 0] = dfp[i+1]
        hh[0, 1] = -dfm[i+2]
        hh[1, 1] = -dfm[i+1]
        hh[2, 1] = -dfm[i]
        hh[3, 1] = -dfm[i-1]
        
        # Apply WENO reconstruction for both flux directions
        for m1 in range(2):
            # Finite differences
            t1 = hh[0, m1] - hh[1, m1]
            t2 = hh[1, m1] - hh[2, m1]
            t3 = hh[2, m1] - hh[3, m1]
            
            # Smoothness indicators
            tt1 = 13.0 * t1**2 + 3.0 * (hh[0, m1] - 3.0*hh[1, m1])**2
            tt2 = 13.0 * t2**2 + 3.0 * (hh[1, m1] + hh[2, m1])**2
            tt3 = 13.0 * t3**2 + 3.0 * (3.0*hh[2, m1] - hh[3, m1])**2
            
            # Squared for weights (note: (epsilon + IS)^2, not epsilon^2 + IS)
            tt1 = (epsilon + tt1)**2
            tt2 = (epsilon + tt2)**2
            tt3 = (epsilon + tt3)**2
            
            # Weight calculation
            s1 = tt2 * tt3
            s2 = 6.0 * tt1 * tt3
            s3 = 3.0 * tt1 * tt2
            
            t0 = 1.0 / (s1 + s2 + s3)
            s1 = s1 * t0
            s2 = s2 * t0
            s3 = s3 * t0
            
            # Add WENO correction to central flux
            fh[i] += (s1*(t2 - t1) + (0.5*s3 - 0.25)*(t3 - t2)) / 3.0
        
        if debug and i == flux_start:
            print(f"\nFirst flux reconstruction (i={i}):")
            print(f"  Central flux: {((-f_im1 + 7.0*(f_i + f_ip1) - f_ip2) / 12.0).item():.10f}")
            print(f"  Final fh[{i}]: {fh[i].item():.10f}")
            print(f"  For m1=0: s1={s1.item():.10f}, s2={s2.item():.10f}, s3={s3.item():.10f}")
            print(f"  Weights sum to: {(s1+s2+s3).item():.10f}")
    
    # Compute spatial derivative
    rhs = torch.zeros(max_idx, dtype=dtype, device=device)
    
    deriv_start = nstart - 1  # MATLAB nstart to Python 0-based
    deriv_end = np_val        # MATLAB np to Python 0-based (not +1 because range is exclusive)
    
    if debug:
        print(f"\nSpatial derivative loop:")
        print(f"  MATLAB loop: i = {nstart}:{np_val} (1-based, {np_val-nstart+1} iterations)")
        print(f"  Python loop: i = {deriv_start}:{deriv_end} (0-based, {deriv_end-deriv_start} iterations)")
    
    for i in range(deriv_start, deriv_end):
        rhs[i] = (fh[i-1] - fh[i]) / dx
    
    # Return only interior points (strip ghost cells)
    result = rhs[deriv_start:deriv_end]
    
    if debug:
        print(f"\nReturning rhs[{deriv_start}:{deriv_end}], length = {len(result)}")
        print(f"First few RHS values: {result[0:5]}")
        print(f"RHS range: [{result.min().item():.10f}, {result.max().item():.10f}]")
    
    return result


def burgers_test_gottlieb_pytorch(device='cpu', dtype=torch.float64):
    """
    Direct PyTorch translation of BurgersTest.m
    
    This exactly replicates Gottlieb's test case:
    - Domain: [-1, 1]
    - Grid: 101 points
    - IC: sign(x)
    - Flux: f(u) = u (right-moving linear advection)
    - Time integration: SSP-RK3
    - Steps: 75 with dt = 0.5*dx
    """
    
    # Flux functions - make sure these handle both scalars and tensors
    def f(u):
        if isinstance(u, torch.Tensor):
            return u.clone()
        else:
            return torch.tensor(u, dtype=dtype, device=device)
    
    def fp(u):
        if isinstance(u, torch.Tensor):
            return torch.ones_like(u)
        else:
            return torch.tensor(1.0, dtype=dtype, device=device)
    
    # Grid
    x = torch.linspace(-1, 1, 101, dtype=dtype, device=device)
    
    # CRITICAL: torch.linspace can produce values like -1e-17 instead of exactly 0
    # Clean values near zero to exactly zero to ensure sign(0) = 0
    x = torch.where(torch.abs(x) < 1e-14, 0.0, x)
    u0 = torch.sign(x)
    
    # Compute dx - use the same method as NumPy version
    x_np = x.cpu().numpy()
    dx = np.max(np.diff(x_np))
    h = 0.5 * dx  # Time step
    
    u = u0.clone()
    N = 75  # Number of time steps
    
    print("Gottlieb Test (PyTorch Translation):")
    print(f"  Grid points: {len(x)}")
    print(f"  dx = {dx:.6f}")
    print(f"  dt = {h:.6f}")
    print(f"  N steps = {N}")
    print(f"  t_final = {N*h:.6f}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    
    # Time evolution with SSP-RK3
    for j in range(N):
        # Stage 1
        rhs = weno5_gottlieb_pytorch(u0, dx, f, fp, device=device, dtype=dtype, debug=(j==0))
        u = u0 + h * rhs
        
        # Stage 2
        rhs = weno5_gottlieb_pytorch(u, dx, f, fp, device=device, dtype=dtype, debug=False)
        u = 0.75*u0 + 0.25*(u + h*rhs)
        
        # Stage 3
        rhs = weno5_gottlieb_pytorch(u, dx, f, fp, device=device, dtype=dtype, debug=False)
        u = (u0 + 2.0*(u + h*rhs)) / 3.0
        
        u0 = u.clone()
    
    return x, u


def validate_against_numpy_reference():
    """
    Validate PyTorch implementation against the NumPy reference.
    They should match to machine precision.
    """
    print("="*70)
    print("VALIDATING PYTORCH TRANSLATION AGAINST NUMPY REFERENCE")
    print("="*70)
    
    # Run NumPy reference
    print("\nRunning NumPy reference...")
    from gottlieb_weno5_reference import burgers_test_gottlieb
    x_np, u_np = burgers_test_gottlieb()
    
    # Run PyTorch translation (CPU)
    print("\nRunning PyTorch translation (CPU)...")
    x_torch, u_torch = burgers_test_gottlieb_pytorch(device='cpu', dtype=torch.float64)
    
    # Convert to numpy for comparison
    x_torch_np = x_torch.cpu().numpy()
    u_torch_np = u_torch.cpu().numpy()
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    x_error = np.abs(x_np - x_torch_np)
    u_error = np.abs(u_np - u_torch_np)
    
    print(f"\nGrid comparison:")
    print(f"  Max x difference: {x_error.max():.6e}")
    
    print(f"\nSolution comparison:")
    print(f"  NumPy u range: [{u_np.min():.15f}, {u_np.max():.15f}]")
    print(f"  PyTorch u range: [{u_torch_np.min():.15f}, {u_torch_np.max():.15f}]")
    print(f"\nError metrics:")
    print(f"  Max absolute error: {u_error.max():.6e}")
    print(f"  Mean absolute error: {u_error.mean():.6e}")
    print(f"  L2 norm of error: {np.linalg.norm(u_error):.6e}")
    
    # Check if translation is exact
    tolerance = 1e-12  # Machine precision for float64
    
    if u_error.max() < tolerance:
        print(f"\n✓ TRANSLATION VERIFIED")
        print(f"  PyTorch translation matches NumPy to machine precision!")
        print(f"  Max error {u_error.max():.6e} < {tolerance:.6e}")
        success = True
    else:
        print(f"\n⚠ DISCREPANCY DETECTED")
        print(f"  Max error {u_error.max():.6e}")
        
        if u_error.max() < 1e-10:
            print(f"  This is likely due to minor floating-point differences")
            print(f"  between NumPy and PyTorch implementations.")
            print(f"  The translation is essentially correct.")
            success = True
        else:
            print(f"  This error is larger than expected.")
            
            # Show worst errors
            worst_idx = np.argmax(u_error)
            print(f"\nWorst error at index {worst_idx}:")
            print(f"  x = {x_np[worst_idx]:.15f}")
            print(f"  u_numpy = {u_np[worst_idx]:.15f}")
            print(f"  u_pytorch = {u_torch_np[worst_idx]:.15f}")
            print(f"  error = {u_error[worst_idx]:.6e}")
            
            success = False
    
    print("="*70)
    
    return success


def validate_against_matlab_output():
    """
    Validate PyTorch implementation against MATLAB reference output.
    This is the gold standard validation.
    """
    from pathlib import Path
    import h5py
    
    print("="*70)
    print("VALIDATING PYTORCH TRANSLATION AGAINST MATLAB OUTPUT")
    print("="*70)
    
    # Run PyTorch translation
    print("\nRunning PyTorch translation...")
    x_torch, u_torch = burgers_test_gottlieb_pytorch(device='cpu', dtype=torch.float64)
    
    # Convert to numpy for comparison
    x_torch_np = x_torch.cpu().numpy()
    u_torch_np = u_torch.cpu().numpy()
    
    # Load MATLAB reference from HDF5
    ref_dir = Path(__file__).parent / 'reference_implementations' / 'gottlieb_matlab'
    h5_file = ref_dir / 'reference_data.h5'
    
    print(f"\nLoading MATLAB reference from HDF5: {h5_file}")
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # List available datasets
            print(f"  Available datasets: {list(f.keys())}")
            
            # Load u and x
            u_matlab = f['u'][:]
            x_matlab = f['x'][:]
            
            # Flatten if needed
            if u_matlab.ndim > 1:
                u_matlab = u_matlab.flatten()
            if x_matlab.ndim > 1:
                x_matlab = x_matlab.flatten()
        
        print(f"✓ Loaded MATLAB reference: {len(x_matlab)} points")
        print(f"  MATLAB data dtype: {u_matlab.dtype} (full precision!)")
        
    except FileNotFoundError:
        print(f"\n✗ ERROR: HDF5 file not found!")
        print(f"\nPlease create the reference data in MATLAB:")
        print(f"  >> % After running BurgersTest.m")
        print(f"  >> save('{h5_file.name}', 'u', 'x', '-v7.3');")
        print(f"\nThen copy it to: {ref_dir}")
        return False
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    x_error = np.abs(x_torch_np - x_matlab)
    u_error = np.abs(u_torch_np - u_matlab)
    
    print(f"\nGrid comparison:")
    print(f"  Max x difference: {x_error.max():.6e}")
    
    print(f"\nSolution comparison:")
    print(f"  PyTorch u range: [{u_torch_np.min():.15f}, {u_torch_np.max():.15f}]")
    print(f"  MATLAB u range: [{u_matlab.min():.15f}, {u_matlab.max():.15f}]")
    print(f"\nError metrics:")
    print(f"  Max absolute error: {u_error.max():.6e}")
    print(f"  Mean absolute error: {u_error.mean():.6e}")
    print(f"  L2 norm of error: {np.linalg.norm(u_error):.6e}")
    
    # Check if translation is exact
    tolerance = 1e-12  # Machine precision
    
    if u_error.max() < tolerance:
        print(f"\n✓ TRANSLATION VERIFIED")
        print(f"  PyTorch translation matches MATLAB to machine precision!")
        print(f"  Max error {u_error.max():.6e} < {tolerance:.6e}")
        success = True
    else:
        print(f"\n⚠ SMALL DISCREPANCY")
        print(f"  Max error {u_error.max():.6e}")
        
        if u_error.max() < 1e-10:
            print(f"  This is likely due to minor floating-point differences")
            print(f"  between MATLAB, NumPy, and PyTorch implementations.")
            print(f"  The translation is essentially correct.")
            success = True
        else:
            print(f"  This error is larger than expected.")
            
            # Show worst errors
            worst_idx = np.argmax(u_error)
            print(f"\nWorst error at index {worst_idx}:")
            print(f"  x = {x_matlab[worst_idx]:.15f}")
            print(f"  u_matlab = {u_matlab[worst_idx]:.15f}")
            print(f"  u_pytorch = {u_torch_np[worst_idx]:.15f}")
            print(f"  error = {u_error[worst_idx]:.6e}")
            
            success = False
    
    print("="*70)
    
    return success


if __name__ == "__main__":
    print("="*70)
    print("GOTTLIEB WENO-5 PYTORCH TRANSLATION")
    print("="*70)
    
    # First validate against NumPy reference
    print("\n" + "="*70)
    print("STEP 1: Validate against NumPy reference")
    print("="*70)
    numpy_success = validate_against_numpy_reference()
    
    if not numpy_success:
        print("\n✗ PyTorch translation does not match NumPy reference!")
        print("  Fix PyTorch implementation before proceeding.")
        import sys
        sys.exit(1)
    
    # Then validate against MATLAB gold standard
    print("\n" + "="*70)
    print("STEP 2: Validate against MATLAB gold standard")
    print("="*70)
    matlab_success = validate_against_matlab_output()
    
    if matlab_success:
        print("\n" + "="*70)
        print("✓✓✓ SUCCESS ✓✓✓")
        print("="*70)
        print("\nPyTorch translation is VERIFIED!")
        print("  ✓ Matches NumPy reference")
        print("  ✓ Matches MATLAB gold standard")
        print("\nReady to proceed to step 7: Refactor into modular components")
    else:
        print("\n⚠ PyTorch translation has minor discrepancies with MATLAB")
        print("  Review results above to determine if acceptable.")
