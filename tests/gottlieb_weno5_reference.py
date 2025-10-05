"""
Direct Python translation of Professor Gottlieb's MATLAB WENO-5 implementation.

This is a LINE-BY-LINE translation to ensure exact numerical matching.
The goal is to understand what Gottlieb's code does, then compare to GradFlow.

Original MATLAB code by: Sigal Gottlieb, September 7, 2003
Modified by: Daniel Higgs, June 22, 2007
Python translation: 2025
"""

import numpy as np


def weno5_gottlieb(u, dx, f, fp, debug=False):
    """
    Direct translation of Gottlieb's weno5.m (or weno.m - they're identical).
    
    Parameters
    ----------
    u : array
        Solution values (periodic, no ghost cells in input)
    dx : float
        Grid spacing
    f : function
        Flux function f(u)
    fp : function
        Flux derivative f'(u)
    debug : bool
        Print debug information
        
    Returns
    -------
    rhs : array
        Right-hand side: -df/dx
    """
    
    # Constants
    epsilon = 1e-29  # WENO epsilon parameter
    md = 4           # Number of ghost points
    
    # Maximum wave speed (for Lax-Friedrichs splitting)
    em = np.max(np.abs(fp(u)))
    
    npoints = len(u)
    nstart = md + 1
    np_val = npoints + md  # 'np' is a Python module, use np_val
    
    # CRITICAL: Ghost cell indexing - CHECKING ALTERNATIVE
    # MATLAB code: u = [ u(i-md:end-1), u, u(2:md+2)]
    #
    # Theory 1: Direct translation (current)
    # MATLAB u(97:100) → Python u[96:100]
    # MATLAB u(2:6) → Python u[1:6]
    #
    # Theory 2: Maybe for periodic BC, should be:
    # Left: last md elements = u[-md:] = u[97:101] in Python
    # Right: first md+1 elements = u[0:md+1] = u[0:5] in Python
    #
    # Let's try Theory 1 first (MATLAB's actual code):
    i = len(u)
    
    # MATLAB u(i-md:end-1) where i=101, md=4 gives u(97:100)
    # In Python 0-based: MATLAB indices [97,98,99,100] → Python [96,97,98,99]
    left_ghost = u[i-md-1:i-1]  # u[96:100]
    
    # MATLAB u(2:md+2) where md=4 gives u(2:6)  
    # In Python 0-based: MATLAB indices [2,3,4,5,6] → Python [1,2,3,4,5]
    right_ghost = u[1:md+2]      # u[1:6]
    
    u_extended = np.concatenate([left_ghost, u, right_ghost])
    
    if debug:
        print(f"Ghost cell construction:")
        print(f"  npoints = {npoints}, md = {md}")
        print(f"  left_ghost indices in original: [{i-md-1}:{i-1}]")
        print(f"  left_ghost values: {left_ghost}")
        print(f"  right_ghost indices in original: [1:{md+2}]")
        print(f"  right_ghost values: {right_ghost}")
        print(f"  u_extended length: {len(u_extended)} (expected {npoints + len(left_ghost) + len(right_ghost)})")
    
    # Flux splitting (Lax-Friedrichs)
    # MATLAB: for i=nstart-md:np+md
    # With nstart=5, md=4, np=105: for i=1:109 (MATLAB 1-based)
    # Python: for i in range(0, 109) - but that means i goes 0...108
    # We access u_extended[i+1], so max index accessed is u_extended[109]
    # u_extended has 110 elements (indices 0-109), so this is OK
    #
    # Allocate arrays with enough space for all indices we'll access
    max_idx = np_val + md + 2  # Extra padding to be safe
    dfp = np.zeros(max_idx)
    dfm = np.zeros(max_idx)
    
    # MATLAB loop bounds: nstart-md : np+md
    # nstart=5, md=4, np=105 gives 1:109 in MATLAB (inclusive on both ends)
    # Python: range(0, 109) gives 0,1,...,108 (NOT including 109)
    # So we need range(nstart-md-1, np_val+md) to match MATLAB's inclusive range
    # Wait, let me recalculate:
    # MATLAB nstart-md = 5-4 = 1
    # MATLAB np+md = 105+4 = 109  
    # MATLAB for i=1:109 means i takes values 1,2,3,...,109 (109 iterations)
    # Python range(0, 109) means i takes values 0,1,2,...,108 (109 iterations)
    # So: range(nstart-md-1, np_val+md) = range(0, 109)
    #
    # Let me use explicit Python 0-based indexing:
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
        with open('tests/gottlieb_flux_debug.txt', 'w') as dbg:
            dbg.write("Gottlieb Flux Splitting Debug\n")
            dbg.write("="*70 + "\n\n")
            dbg.write(f"Max wave speed (em): {em:.10f}\n")
            dbg.write(f"u_extended length: {len(u_extended)}\n")
            dbg.write(f"Loop range: [{start_idx}, {end_idx})\n\n")
            
            dbg.write("First 20 flux splitting values:\n")
            dbg.write(f"{'i':>4} {'u[i]':>15} {'f[i]':>15} {'dfp[i]':>15} {'dfm[i]':>15}\n")
            dbg.write("-"*70 + "\n")
            
            for i in range(start_idx, min(start_idx + 20, end_idx)):
                dbg.write(f"{i:4d} {u_extended[i]:15.10f} {f(u_extended[i]):15.10f} ")
                dbg.write(f"{dfp[i]:15.10f} {dfm[i]:15.10f}\n")
    
    # Numerical flux reconstruction
    # MATLAB: for i = nstart-1:np+1
    # nstart=5, np=105 gives MATLAB i=4:106
    # Python 0-based: range(3, 106)
    fh = np.zeros(max_idx)
    
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
        
        fh[i] = (-f_im1 + 7*(f_i + f_ip1) - f_ip2) / 12.0
        
        # Build stencil data for both positive and negative fluxes
        # hh[k, m] where k=0..3 is stencil point, m=0 is dfp, m=1 is dfm
        hh = np.zeros((4, 2))
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
            print(f"  Central flux: {(-f_im1 + 7*(f_i + f_ip1) - f_ip2) / 12.0:.10f}")
            print(f"  Final fh[{i}]: {fh[i]:.10f}")
            print(f"  For m1=0: s1={s1:.10f}, s2={s2:.10f}, s3={s3:.10f}")
            print(f"  Weights sum to: {s1+s2+s3:.10f}")
    
    # Compute spatial derivative
    # MATLAB: for i = nstart:np
    # nstart=5, np=105 gives MATLAB i=5:105 (101 values)
    # Python 0-based: range(4, 105) gives indices 4,5,...,104 (101 values)
    rhs = np.zeros(max_idx)
    
    deriv_start = nstart - 1  # MATLAB nstart to Python 0-based
    deriv_end = np_val        # MATLAB np to Python 0-based (not +1 because range is exclusive)
    
    if debug:
        print(f"\nSpatial derivative loop:")
        print(f"  MATLAB loop: i = {nstart}:{np_val} (1-based, {np_val-nstart+1} iterations)")
        print(f"  Python loop: i = {deriv_start}:{deriv_end} (0-based, {deriv_end-deriv_start} iterations)")
    
    for i in range(deriv_start, deriv_end):
        rhs[i] = (fh[i-1] - fh[i]) / dx
    
    # Return only interior points (strip ghost cells)
    # MATLAB returns rhs(nstart:np) which is 101 values at indices 5:105 (MATLAB 1-based)
    # Python: return rhs[4:105] which is 101 values at indices 4,5,...,104
    result = rhs[deriv_start:deriv_end]
    
    if debug:
        print(f"\nReturning rhs[{deriv_start}:{deriv_end}], length = {len(result)}")
        print(f"First few RHS values: {result[0:5]}")
        print(f"RHS range: [{result.min():.10f}, {result.max():.10f}]")
    
    return result


def burgers_test_gottlieb():
    """
    Direct translation of BurgersTest.m
    
    This exactly replicates Gottlieb's test case:
    - Domain: [-1, 1]
    - Grid: 101 points
    - IC: sign(x)
    - Flux: f(u) = u (right-moving linear advection)
    - Time integration: SSP-RK3
    - Steps: 75 with dt = 0.5*dx
    """
    
    # Flux functions - make sure these handle both scalars and arrays
    def f(u):
        return np.asarray(u)  # f(u) = u
    
    def fp(u):
        if np.isscalar(u):
            return 1.0
        else:
            return np.ones_like(u)
    
    # Grid
    x = np.linspace(-1, 1, 101)
    u0 = np.sign(x)
    
    dx = np.max(np.diff(x))
    h = 0.5 * dx  # Time step
    
    u = u0.copy()
    N = 75  # Number of time steps
    
    print("Gottlieb Test (Python Translation):")
    print(f"  Grid points: {len(x)}")
    print(f"  dx = {dx:.6f}")
    print(f"  dt = {h:.6f}")
    print(f"  N steps = {N}")
    print(f"  t_final = {N*h:.6f}")
    
    # Time evolution with SSP-RK3
    for j in range(N):
        # Stage 1
        rhs = weno5_gottlieb(u0, dx, f, fp, debug=(j==0))
        u = u0 + h * rhs
        
        # Stage 2
        rhs = weno5_gottlieb(u, dx, f, fp, debug=False)
        u = 0.75*u0 + 0.25*(u + h*rhs)
        
        # Stage 3
        rhs = weno5_gottlieb(u, dx, f, fp, debug=False)
        u = (u0 + 2.0*(u + h*rhs)) / 3.0
        
        u0 = u.copy()
    
    return x, u


def validate_against_matlab_output():
    """
    Run the Python translation and compare to MATLAB reference output.
    """
    from pathlib import Path
    import h5py
    
    print("="*70)
    print("VALIDATING PYTHON TRANSLATION AGAINST MATLAB OUTPUT")
    print("="*70)
    
    # Run Python translation
    print("\nRunning Python translation...")
    x_py, u_py = burgers_test_gottlieb()
    
    # Load MATLAB reference from HDF5
    ref_dir = Path(__file__).parent / 'reference_implementations' / 'gottlieb_matlab'
    h5_file = ref_dir / 'reference_data.h5'
    
    print(f"\nLoading MATLAB reference from HDF5: {h5_file}")
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # List available datasets
            print(f"  Available datasets: {list(f.keys())}")
            
            # Load u and x
            # Note: MATLAB saves arrays as column vectors, may need to flatten/transpose
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
    
    x_error = np.abs(x_py - x_matlab)
    u_error = np.abs(u_py - u_matlab)
    
    print(f"\nGrid comparison:")
    print(f"  Max x difference: {x_error.max():.6e}")
    
    print(f"\nSolution comparison:")
    print(f"  Python u range: [{u_py.min():.15f}, {u_py.max():.15f}]")
    print(f"  MATLAB u range: [{u_matlab.min():.15f}, {u_matlab.max():.15f}]")
    print(f"\nError metrics:")
    print(f"  Max absolute error: {u_error.max():.6e}")
    print(f"  Mean absolute error: {u_error.mean():.6e}")
    print(f"  L2 norm of error: {np.linalg.norm(u_error):.6e}")
    
    # Check if translation is exact
    tolerance = 1e-14  # Machine precision
    
    if u_error.max() < tolerance:
        print(f"\n✓ TRANSLATION VERIFIED")
        print(f"  Python translation matches MATLAB to machine precision!")
        print(f"  Max error {u_error.max():.6e} < {tolerance:.6e}")
        success = True
    else:
        print(f"\n⚠ SMALL DISCREPANCY")
        print(f"  Max error {u_error.max():.6e}")
        
        if u_error.max() < 1e-12:
            print(f"  This is likely due to minor floating-point differences")
            print(f"  between MATLAB and Python/NumPy implementations.")
            print(f"  The translation is essentially correct.")
            success = True
        else:
            print(f"  This error is larger than expected.")
            
            # Show worst errors
            worst_idx = np.argmax(u_error)
            print(f"\nWorst error at index {worst_idx}:")
            print(f"  x = {x_matlab[worst_idx]:.15f}")
            print(f"  u_matlab = {u_matlab[worst_idx]:.15f}")
            print(f"  u_python = {u_py[worst_idx]:.15f}")
            print(f"  error = {u_error[worst_idx]:.6e}")
            
            success = False
    
    print("="*70)
    
    return success


if __name__ == "__main__":
    # First validate the translation
    validate_against_matlab_output()