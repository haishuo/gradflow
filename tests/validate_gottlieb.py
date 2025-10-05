"""
Validation against Professor Gottlieb's MATLAB WENO-5 reference implementation.

This script compares GradFlow's WENO-5 solver against the trusted reference
code from Professor Sigal Gottlieb (UMass Dartmouth).

Updated to use HDF5 format for full floating-point precision.

Test case: Right-moving linear advection with step function initial condition
- Domain: [-1, 1]
- Grid: 101 points
- IC: u0 = sign(x)
- Flux: f(u) = u (right-moving wave)
- Time integration: SSP-RK3, 75 steps with dt = 0.5*dx
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

from gradflow.solvers.weno import WENOSolver


def load_gottlieb_reference():
    """Load Gottlieb's MATLAB reference output from HDF5."""
    ref_dir = Path(__file__).parent / 'reference_implementations' / 'gottlieb_matlab'
    h5_file = ref_dir / 'reference_data.h5'
    
    if not h5_file.exists():
        raise FileNotFoundError(
            f"\n\nHDF5 reference file not found: {h5_file}\n\n"
            f"Please create it in MATLAB:\n"
            f"  >> % After running BurgersTest.m:\n"
            f"  >> save('reference_data.h5', 'u', 'x', '-v7.3');\n\n"
            f"Then copy the file to: {ref_dir}\n"
        )
    
    with h5py.File(h5_file, 'r') as f:
        # MATLAB saves variables with their names as keys
        # Arrays may be transposed, so we'll flatten them
        u_ref = f['u'][:].flatten()
        x_ref = f['x'][:].flatten()
    
    return x_ref, u_ref


def run_gradflow_equivalent():
    """
    Run GradFlow with parameters matching Gottlieb's test case.
    
    Her parameters from BurgersTest.m:
    - x = linspace(-1, 1, 101)
    - u0 = sign(x)
    - dx = max(diff(x)) = 2/100 = 0.02
    - h = 0.5*dx = 0.01
    - N = 75 steps
    - f(u) = u (right-moving linear advection)
    - fp(u) = 1
    """
    # Grid parameters
    nx = 101
    domain_length = 2.0  # [-1, 1] has length 2
    dx = domain_length / (nx - 1)  # 2/100 = 0.02
    dt = 0.5 * dx  # h = 0.01
    N = 75
    t_final = N * dt  # 0.75
    
    print("Gottlieb Test Parameters:")
    print(f"  Grid points: {nx}")
    print(f"  Domain: [-1, 1], length = {domain_length}")
    print(f"  dx = {dx:.6f}")
    print(f"  dt = {dt:.6f}")
    print(f"  N steps = {N}")
    print(f"  t_final = {t_final:.6f}")
    
    # Create solver
    solver = WENOSolver(
        order=5,
        grid_size=nx,
        domain_length=domain_length,
        boundary_condition='periodic',
        device='cpu',
        dtype=torch.float64
    )
    
    # Shift x to match [-1, 1]
    x = torch.linspace(-1, 1, nx, dtype=torch.float64)
    
    # Initial condition: step function
    u0 = torch.sign(x)
    
    print(f"\nInitial condition: step function")
    print(f"  u0 range: [{u0.min():.6f}, {u0.max():.6f}]")
    
    # Flux function: f(u) = u (right-moving linear advection)
    def linear_advection_flux(u):
        return u
    
    # CRITICAL: Match Gottlieb's epsilon
    epsilon = 1e-29
    
    print(f"\nRunning GradFlow...")
    print(f"  epsilon = {epsilon:.2e} (matching Gottlieb)")
    
    u_final = solver.solve(
        u0,
        t_final=t_final,
        flux_function=linear_advection_flux,
        cfl_number=0.5,
        epsilon=epsilon,
        time_method='ssp_rk3',
        verbose=True
    )
    
    return x.numpy(), u_final.squeeze().numpy()


def compare_results(x_ref, u_ref, x_grad, u_grad):
    """Compare GradFlow results against Gottlieb reference."""
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    # Check grid agreement
    if len(x_ref) != len(x_grad):
        print(f"WARNING: Grid size mismatch!")
        print(f"  Reference: {len(x_ref)} points")
        print(f"  GradFlow:  {len(x_grad)} points")
        return False
    
    x_error = np.abs(x_ref - x_grad)
    print(f"\nGrid comparison:")
    print(f"  Max x difference: {x_error.max():.2e}")
    
    # Compare solutions
    error = np.abs(u_ref - u_grad)
    rel_error = error / (np.abs(u_ref) + 1e-16)
    
    print(f"\nSolution comparison:")
    print(f"  Reference u range: [{u_ref.min():.15f}, {u_ref.max():.15f}]")
    print(f"  GradFlow u range:  [{u_grad.min():.15f}, {u_grad.max():.15f}]")
    print(f"\nError metrics:")
    print(f"  Max absolute error: {error.max():.6e}")
    print(f"  Mean absolute error: {error.mean():.6e}")
    print(f"  Max relative error: {rel_error.max():.6e}")
    print(f"  L2 norm of error: {np.linalg.norm(error):.6e}")
    
    # Validation threshold
    tolerance = 1e-12
    
    if error.max() < tolerance:
        print(f"\n✓ VALIDATION PASSED")
        print(f"  Maximum error {error.max():.6e} < tolerance {tolerance:.6e}")
        passed = True
    else:
        print(f"\n✗ VALIDATION FAILED")
        print(f"  Maximum error {error.max():.6e} exceeds tolerance {tolerance:.6e}")
        
        # Show where worst errors are
        worst_idx = np.argmax(error)
        print(f"\nWorst error at index {worst_idx}:")
        print(f"  x = {x_ref[worst_idx]:.6f}")
        print(f"  u_ref = {u_ref[worst_idx]:.15f}")
        print(f"  u_grad = {u_grad[worst_idx]:.15f}")
        print(f"  error = {error[worst_idx]:.6e}")
        
        passed = False
    
    return passed


def plot_comparison(x_ref, u_ref, x_grad, u_grad):
    """Plot reference vs GradFlow solution."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Solution comparison
    ax1.plot(x_ref, u_ref, 'b-', label='Gottlieb Reference', linewidth=2)
    ax1.plot(x_grad, u_grad, 'r--', label='GradFlow', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('Solution Comparison: Gottlieb MATLAB vs GradFlow')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error plot
    error = np.abs(u_ref - u_grad)
    ax2.semilogy(x_ref, error + 1e-17, 'k-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|u_ref - u_grad|')
    ax2.set_title('Absolute Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tests/gottlieb_validation.png', dpi=150)
    print(f"\nPlot saved to tests/gottlieb_validation.png")
    
    plt.show()


def main():
    """Run complete validation."""
    
    print("="*70)
    print("GOTTLIEB MATLAB REFERENCE VALIDATION (HDF5)")
    print("="*70)
    
    # Load reference
    print("\nLoading Gottlieb reference data from HDF5...")
    try:
        x_ref, u_ref = load_gottlieb_reference()
        print(f"✓ Reference loaded: {len(x_ref)} points (full precision)")
    except FileNotFoundError as e:
        print(str(e))
        return 1
    
    # Run GradFlow
    print("\n" + "-"*70)
    x_grad, u_grad = run_gradflow_equivalent()
    print(f"✓ GradFlow complete")
    
    # Compare
    print("\n" + "-"*70)
    passed = compare_results(x_ref, u_ref, x_grad, u_grad)
    
    # Plot
    print("\n" + "-"*70)
    plot_comparison(x_ref, u_ref, x_grad, u_grad)
    
    # Summary
    print("\n" + "="*70)
    if passed:
        print("SUCCESS: GradFlow matches Gottlieb reference!")
        print("Your WENO-5 implementation is validated.")
    else:
        print("FAILURE: GradFlow does not match reference.")
        print("Debug needed - check:")
        print("  1. epsilon parameter (should be 1e-29)")
        print("  2. Time step calculation")
        print("  3. Boundary conditions")
        print("  4. Flux splitting")
    print("="*70)
    
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())