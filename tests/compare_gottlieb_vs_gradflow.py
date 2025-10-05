"""
Compare Gottlieb's reference implementation against GradFlow.

This identifies the algorithmic differences causing the 0.52 error.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import both implementations
from gottlieb_weno5_reference import weno5_gottlieb, burgers_test_gottlieb
from gradflow.solvers.weno import WENOSolver


def compare_single_rhs_call():
    """
    Compare a single RHS evaluation between Gottlieb and GradFlow.
    This will show us exactly where they differ.
    """
    print("="*70)
    print("COMPARING SINGLE RHS EVALUATION")
    print("="*70)
    
    # Setup identical initial condition
    x = np.linspace(-1, 1, 101)
    u0_np = np.sign(x)
    u0_torch = torch.from_numpy(u0_np).unsqueeze(0)  # Add batch dimension
    
    dx = np.max(np.diff(x))
    
    # Flux functions
    f_np = lambda u: np.asarray(u)
    fp_np = lambda u: np.ones_like(u) if not np.isscalar(u) else 1.0
    
    def f_torch(u):
        return u
    
    print(f"\nInitial condition:")
    print(f"  Grid: 101 points, dx = {dx:.6f}")
    print(f"  u0 range: [{u0_np.min():.6f}, {u0_np.max():.6f}]")
    
    # Gottlieb RHS
    print(f"\nComputing Gottlieb RHS...")
    rhs_gottlieb = weno5_gottlieb(u0_np, dx, f_np, fp_np)
    print(f"  RHS range: [{rhs_gottlieb.min():.6f}, {rhs_gottlieb.max():.6f}]")
    print(f"  RHS mean: {rhs_gottlieb.mean():.6f}")
    print(f"  RHS norm: {np.linalg.norm(rhs_gottlieb):.6f}")
    
    # GradFlow RHS
    print(f"\nComputing GradFlow RHS...")
    solver = WENOSolver(
        order=5,
        grid_size=101,
        domain_length=2.0,
        boundary_condition='periodic',
        device='cpu',
        dtype=torch.float64
    )
    
    # GradFlow's RHS computation (negative of spatial derivative)
    # Note: GradFlow computes -df/dx directly in compute_spatial_derivative
    rhs_gradflow_torch = solver.compute_spatial_derivative(u0_torch, f_torch, epsilon=1e-29)
    rhs_gradflow = rhs_gradflow_torch.squeeze().numpy()
    
    print(f"  RHS range: [{rhs_gradflow.min():.6f}, {rhs_gradflow.max():.6f}]")
    print(f"  RHS mean: {rhs_gradflow.mean():.6f}")
    print(f"  RHS norm: {np.linalg.norm(rhs_gradflow):.6f}")
    
    # Compare
    print(f"\n" + "="*70)
    print("RHS COMPARISON")
    print("="*70)
    
    rhs_diff = np.abs(rhs_gottlieb - rhs_gradflow)
    
    print(f"\nDifference statistics:")
    print(f"  Max difference: {rhs_diff.max():.6e}")
    print(f"  Mean difference: {rhs_diff.mean():.6e}")
    print(f"  L2 norm of difference: {np.linalg.norm(rhs_diff):.6e}")
    
    # Write detailed comparison to file
    print(f"\nWriting detailed comparison to tests/rhs_debug.txt...")
    with open('tests/rhs_debug.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("DETAILED RHS COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        f.write("Grid and Parameters:\n")
        f.write(f"  Grid points: 101\n")
        f.write(f"  dx: {dx:.10f}\n")
        f.write(f"  Initial condition: sign(x) on [-1, 1]\n\n")
        
        f.write("RHS Statistics:\n")
        f.write(f"  Gottlieb RHS range: [{rhs_gottlieb.min():.10f}, {rhs_gottlieb.max():.10f}]\n")
        f.write(f"  GradFlow RHS range: [{rhs_gradflow.min():.10f}, {rhs_gradflow.max():.10f}]\n")
        f.write(f"  Max difference: {rhs_diff.max():.10e}\n")
        f.write(f"  Mean difference: {rhs_diff.mean():.10e}\n\n")
        
        f.write("Point-by-point comparison (showing all 101 points):\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Index':>6} {'x':>12} {'Gottlieb':>15} {'GradFlow':>15} {'Difference':>15}\n")
        f.write("-"*70 + "\n")
        
        for i in range(len(x)):
            f.write(f"{i:6d} {x[i]:12.6f} {rhs_gottlieb[i]:15.10f} {rhs_gradflow[i]:15.10f} {rhs_diff[i]:15.10e}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        # Find largest errors
        sorted_indices = np.argsort(rhs_diff)[::-1]
        f.write("Top 10 largest errors:\n")
        for idx in sorted_indices[:10]:
            f.write(f"  Index {idx:3d} (x={x[idx]:7.3f}): ")
            f.write(f"Gottlieb={rhs_gottlieb[idx]:12.6f}, ")
            f.write(f"GradFlow={rhs_gradflow[idx]:12.6f}, ")
            f.write(f"Error={rhs_diff[idx]:12.6e}\n")
    
    print(f"✓ Detailed comparison saved")
    
    if rhs_diff.max() > 1e-10:
        print(f"\n⚠ SIGNIFICANT DIFFERENCE DETECTED")
        
        # Show worst errors
        worst_indices = np.argsort(rhs_diff)[-5:][::-1]
        print(f"\nTop 5 worst differences:")
        for idx in worst_indices:
            print(f"  x[{idx}] = {x[idx]:.6f}:")
            print(f"    Gottlieb: {rhs_gottlieb[idx]:12.6e}")
            print(f"    GradFlow: {rhs_gradflow[idx]:12.6e}")
            print(f"    Diff:     {rhs_diff[idx]:12.6e}")
    else:
        print(f"\n✓ RHS computations match!")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(x, rhs_gottlieb, 'b-', label='Gottlieb', linewidth=2)
    ax1.plot(x, rhs_gradflow, 'r--', label='GradFlow', linewidth=2, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('RHS')
    ax1.set_title('RHS Comparison (Single Evaluation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(x, rhs_diff + 1e-16, 'k-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|RHS_Gottlieb - RHS_GradFlow|')
    ax2.set_title('Absolute Difference')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tests/rhs_comparison.png', dpi=150)
    print(f"\nPlot saved to tests/rhs_comparison.png")
    
    return rhs_diff.max() < 1e-10


def compare_full_evolution():
    """
    Compare the full time evolution between Gottlieb and GradFlow.
    """
    print("\n" + "="*70)
    print("COMPARING FULL TIME EVOLUTION")
    print("="*70)
    
    # Run Gottlieb reference
    print("\nRunning Gottlieb reference...")
    x_gottlieb, u_gottlieb = burgers_test_gottlieb()
    
    # Run GradFlow
    print("\nRunning GradFlow...")
    solver = WENOSolver(
        order=5,
        grid_size=101,
        domain_length=2.0,
        boundary_condition='periodic',
        device='cpu',
        dtype=torch.float64
    )
    
    x_torch = torch.linspace(-1, 1, 101, dtype=torch.float64)
    u0_torch = torch.sign(x_torch)
    
    def flux_torch(u):
        return u
    
    u_gradflow_torch = solver.solve(
        u0_torch,
        t_final=0.75,
        flux_function=flux_torch,
        cfl_number=0.5,
        epsilon=1e-29,
        time_method='ssp_rk3',
        verbose=False
    )
    
    u_gradflow = u_gradflow_torch.squeeze().numpy()
    
    # Compare
    print("\n" + "="*70)
    print("FINAL SOLUTION COMPARISON")
    print("="*70)
    
    diff = np.abs(u_gottlieb - u_gradflow)
    
    print(f"\nSolution statistics:")
    print(f"  Gottlieb range: [{u_gottlieb.min():.6f}, {u_gottlieb.max():.6f}]")
    print(f"  GradFlow range: [{u_gradflow.min():.6f}, {u_gradflow.max():.6f}]")
    
    print(f"\nDifference statistics:")
    print(f"  Max difference: {diff.max():.6e}")
    print(f"  Mean difference: {diff.mean():.6e}")
    print(f"  L2 norm: {np.linalg.norm(diff):.6e}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(x_gottlieb, u_gottlieb, 'b-', label='Gottlieb', linewidth=2)
    ax1.plot(x_gottlieb, u_gradflow, 'r--', label='GradFlow', linewidth=2, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('Final Solution Comparison (t=0.75)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(x_gottlieb, diff + 1e-16, 'k-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|u_Gottlieb - u_GradFlow|')
    ax2.set_title('Absolute Difference')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tests/solution_comparison.png', dpi=150)
    print(f"\nPlot saved to tests/solution_comparison.png")
    
    return diff.max()


def main():
    """
    Run complete comparison to identify differences.
    """
    print("="*70)
    print("GOTTLIEB VS GRADFLOW COMPARISON")
    print("="*70)
    
    # First compare a single RHS call
    rhs_matches = compare_single_rhs_call()
    
    if not rhs_matches:
        print("\n" + "="*70)
        print("DIAGNOSIS")
        print("="*70)
        print("\n⚠ RHS computations differ significantly!")
        print("\nThis means there's a fundamental algorithmic difference.")
        print("Possible causes:")
        print("  1. Different flux splitting approach")
        print("  2. Different smoothness indicator formulas")
        print("  3. Different stencil construction")
        print("  4. Different weight calculations")
        print("  5. Different boundary condition handling")
        print("\nRecommendation: Debug GradFlow's compute_rhs() method")
        print("by comparing intermediate values with Gottlieb's implementation.")
    else:
        print("\n✓ RHS computations match!")
        print("\nThe error must accumulate during time integration.")
    
    # Then compare full evolution
    max_error = compare_full_evolution()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if max_error < 1e-10:
        print("\n✓ SUCCESS: GradFlow matches Gottlieb reference!")
    elif max_error < 1e-6:
        print(f"\n⚠ CLOSE: Max error {max_error:.6e} is small but not machine precision")
        print("This could be due to:")
        print("  - Floating point differences")
        print("  - Slightly different algorithm details")
    else:
        print(f"\n✗ FAILURE: Max error {max_error:.6e} is too large")
        print("\nGradFlow needs debugging. Check the RHS comparison plot")
        print("to see where the implementations diverge.")
    
    print("="*70)


if __name__ == "__main__":
    main()