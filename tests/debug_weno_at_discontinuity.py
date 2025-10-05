"""
Deep debugging of WENO reconstruction at the discontinuity.

This script traces through every step of the WENO reconstruction
for both Gottlieb and GradFlow at the discontinuity to find where
they diverge.
"""

import numpy as np
import torch
from pathlib import Path

# Import implementations
from tests.gottlieb_weno5_reference import weno5_gottlieb
from gradflow.solvers.weno import WENOSolver
from gradflow.core.flux import lax_friedrichs_splitting
from gradflow.core.smoothness import compute_smoothness_indicators_torch
from gradflow.core.weights import compute_nonlinear_weights


def debug_gottlieb_reconstruction():
    """
    Manually trace through Gottlieb's reconstruction at the discontinuity.
    """
    print("="*70)
    print("GOTTLIEB RECONSTRUCTION DEBUG")
    print("="*70)
    
    # Setup
    x = np.linspace(-1, 1, 101)
    u0 = np.sign(x)
    dx = 0.02
    
    # Flux function
    f = lambda u: np.asarray(u)
    fp = lambda u: np.ones_like(u) if not np.isscalar(u) else 1.0
    
    # Extend with ghost cells (matching Gottlieb's approach)
    md = 4
    i = len(u0)
    u_extended = np.concatenate([u0[i-md-1:i-1], u0, u0[1:md+2]])
    
    print(f"\nGrid setup:")
    print(f"  Physical grid: {len(u0)} points")
    print(f"  Extended grid: {len(u_extended)} points")
    print(f"  Ghost cells: {md} on each side")
    
    # Flux splitting
    em = 1.0  # max wave speed
    
    # Focus on the discontinuity region
    # The discontinuity is at physical index 50 (x=0)
    # After adding 4 left ghost cells, this becomes extended index 54
    
    disc_idx = 54  # Discontinuity in extended array
    
    print(f"\nDiscontinuity location:")
    print(f"  Physical index: 50 (x = 0.0)")
    print(f"  Extended index: {disc_idx}")
    print(f"  u values around discontinuity:")
    for i in range(disc_idx - 3, disc_idx + 4):
        print(f"    u_extended[{i}] = {u_extended[i]:.1f}")
    
    # Compute flux splitting
    dfp = np.zeros(len(u_extended))
    dfm = np.zeros(len(u_extended))
    
    for i in range(len(u_extended) - 1):
        f_ip1 = f(u_extended[i+1])
        f_i = f(u_extended[i])
        u_ip1 = u_extended[i+1]
        u_i = u_extended[i]
        
        dfp[i] = (f_ip1 - f_i + em * (u_ip1 - u_i)) / 2.0
        dfm[i] = (f_ip1 - f_i - em * (u_ip1 - u_i)) / 2.0
    
    print(f"\nFlux splitting at discontinuity:")
    print(f"  dfp values:")
    for i in range(disc_idx - 2, disc_idx + 3):
        print(f"    dfp[{i}] = {dfp[i]:.6f}")
    print(f"  dfm values:")
    for i in range(disc_idx - 2, disc_idx + 3):
        print(f"    dfm[{i}] = {dfm[i]:.6f}")
    
    # Now manually compute WENO reconstruction for f_plus at interface disc_idx + 1/2
    # This uses stencils centered at disc_idx-2, disc_idx-1, disc_idx
    
    print(f"\nWENO reconstruction for f_plus at interface {disc_idx}+1/2:")
    
    # Stencil 0: points [disc_idx-2, disc_idx-1, disc_idx]
    f0_vals = [dfp[disc_idx-2], dfp[disc_idx-1], dfp[disc_idx]]
    print(f"  Stencil 0 (indices [{disc_idx-2}, {disc_idx-1}, {disc_idx}]):")
    print(f"    Values: {f0_vals}")
    
    # Compute IS0
    IS0 = (13.0/12.0) * (f0_vals[0] - 2*f0_vals[1] + f0_vals[2])**2 + \
          (1.0/4.0) * (f0_vals[0] - 4*f0_vals[1] + 3*f0_vals[2])**2
    print(f"    IS0 = {IS0:.10e}")
    
    # Stencil 1: points [disc_idx-1, disc_idx, disc_idx+1]
    f1_vals = [dfp[disc_idx-1], dfp[disc_idx], dfp[disc_idx+1]]
    print(f"  Stencil 1 (indices [{disc_idx-1}, {disc_idx}, {disc_idx+1}]):")
    print(f"    Values: {f1_vals}")
    
    IS1 = (13.0/12.0) * (f1_vals[0] - 2*f1_vals[1] + f1_vals[2])**2 + \
          (1.0/4.0) * (f1_vals[0] - f1_vals[2])**2
    print(f"    IS1 = {IS1:.10e}")
    
    # Stencil 2: points [disc_idx, disc_idx+1, disc_idx+2]
    f2_vals = [dfp[disc_idx], dfp[disc_idx+1], dfp[disc_idx+2]]
    print(f"  Stencil 2 (indices [{disc_idx}, {disc_idx+1}, {disc_idx+2}]):")
    print(f"    Values: {f2_vals}")
    
    IS2 = (13.0/12.0) * (f2_vals[0] - 2*f2_vals[1] + f2_vals[2])**2 + \
          (1.0/4.0) * (3*f2_vals[0] - 4*f2_vals[1] + f2_vals[2])**2
    print(f"    IS2 = {IS2:.10e}")
    
    # Compute weights
    epsilon = 1e-29
    alpha0 = 0.1 / (epsilon + IS0)**2
    alpha1 = 0.6 / (epsilon + IS1)**2
    alpha2 = 0.3 / (epsilon + IS2)**2
    alpha_sum = alpha0 + alpha1 + alpha2
    
    omega0 = alpha0 / alpha_sum
    omega1 = alpha1 / alpha_sum
    omega2 = alpha2 / alpha_sum
    
    print(f"\n  Nonlinear weights:")
    print(f"    ω0 = {omega0:.10f}")
    print(f"    ω1 = {omega1:.10f}")
    print(f"    ω2 = {omega2:.10f}")
    print(f"    Sum = {omega0 + omega1 + omega2:.10f}")
    
    # Compute stencil reconstructions
    flux0 = (2.0/6.0)*f0_vals[0] - (7.0/6.0)*f0_vals[1] + (11.0/6.0)*f0_vals[2]
    flux1 = -(1.0/6.0)*f1_vals[0] + (5.0/6.0)*f1_vals[1] + (2.0/6.0)*f1_vals[2]
    flux2 = (2.0/6.0)*f2_vals[0] + (5.0/6.0)*f2_vals[1] - (1.0/6.0)*f2_vals[2]
    
    print(f"\n  Stencil reconstructions:")
    print(f"    flux0 = {flux0:.10f}")
    print(f"    flux1 = {flux1:.10f}")
    print(f"    flux2 = {flux2:.10f}")
    
    # Final reconstruction
    f_plus_reconstructed = omega0*flux0 + omega1*flux1 + omega2*flux2
    print(f"\n  Final f_plus reconstruction: {f_plus_reconstructed:.10f}")
    
    return {
        'IS': [IS0, IS1, IS2],
        'omega': [omega0, omega1, omega2],
        'flux_stencils': [flux0, flux1, flux2],
        'f_plus_reconstructed': f_plus_reconstructed
    }


def debug_gradflow_reconstruction():
    """
    Manually trace through GradFlow's reconstruction at the discontinuity.
    """
    print("\n" + "="*70)
    print("GRADFLOW RECONSTRUCTION DEBUG")
    print("="*70)
    
    # Setup - match Gottlieb exactly
    x = torch.linspace(-1, 1, 101, dtype=torch.float64)
    u0 = torch.sign(x).unsqueeze(0)  # Add batch dimension
    
    print(f"\nGrid setup:")
    print(f"  Physical grid: {u0.shape[1]} points")
    print(f"  u0 range: [{u0.min():.1f}, {u0.max():.1f}]")
    
    # Apply periodic BCs to add ghost cells
    n_ghost = 3  # For WENO-5
    u_extended = torch.cat([
        u0[:, -n_ghost:],
        u0,
        u0[:, :n_ghost]
    ], dim=1)
    
    print(f"  Extended grid: {u_extended.shape[1]} points")
    print(f"  Ghost cells: {n_ghost} on each side")
    
    # The discontinuity is at physical index 50
    # After adding 3 left ghost cells, this becomes extended index 53
    disc_idx = 53
    
    print(f"\nDiscontinuity location:")
    print(f"  Physical index: 50 (x = 0.0)")
    print(f"  Extended index: {disc_idx}")
    print(f"  u values around discontinuity:")
    for i in range(disc_idx - 3, disc_idx + 4):
        print(f"    u_extended[0,{i}] = {u_extended[0,i].item():.1f}")
    
    # Compute flux
    flux = u_extended  # f(u) = u for linear advection
    
    # Flux splitting
    alpha = torch.abs(u_extended).max().item()
    f_plus = 0.5 * (flux + alpha * u_extended)
    f_minus = 0.5 * (flux - alpha * u_extended)
    
    print(f"\nFlux splitting:")
    print(f"  alpha (max wave speed): {alpha:.6f}")
    print(f"  f_plus at discontinuity:")
    for i in range(disc_idx - 2, disc_idx + 3):
        print(f"    f_plus[0,{i}] = {f_plus[0,i].item():.6f}")
    
    # Now manually compute WENO reconstruction using GradFlow's functions
    # Extract the stencil values needed for reconstruction at disc_idx + 1/2
    
    print(f"\nWENO reconstruction for f_plus at interface {disc_idx}+1/2:")
    
    # For WENO-5 reconstruction at point j, we need values at j-2, j-1, j, j+1, j+2
    stencil_indices = [disc_idx-2, disc_idx-1, disc_idx, disc_idx+1, disc_idx+2]
    stencil_vals = f_plus[0, stencil_indices]
    
    print(f"  Stencil values (indices {stencil_indices}):")
    for idx, val in zip(stencil_indices, stencil_vals):
        print(f"    f_plus[{idx}] = {val.item():.6f}")
    
    # Compute smoothness indicators manually
    # IS0 uses points [j-2, j-1, j]
    v0 = stencil_vals[0:3]
    IS0 = (13.0/12.0) * (v0[0] - 2*v0[1] + v0[2])**2 + \
          (1.0/4.0) * (v0[0] - 4*v0[1] + 3*v0[2])**2
    
    # IS1 uses points [j-1, j, j+1]
    v1 = stencil_vals[1:4]
    IS1 = (13.0/12.0) * (v1[0] - 2*v1[1] + v1[2])**2 + \
          (1.0/4.0) * (v1[0] - v1[2])**2
    
    # IS2 uses points [j, j+1, j+2]
    v2 = stencil_vals[2:5]
    IS2 = (13.0/12.0) * (v2[0] - 2*v2[1] + v2[2])**2 + \
          (1.0/4.0) * (3*v2[0] - 4*v2[1] + v2[2])**2
    
    print(f"\n  Smoothness indicators:")
    print(f"    IS0 = {IS0.item():.10e}")
    print(f"    IS1 = {IS1.item():.10e}")
    print(f"    IS2 = {IS2.item():.10e}")
    
    # Compute weights
    epsilon = 1e-29
    IS_tensor = torch.stack([IS0, IS1, IS2]).unsqueeze(0)
    
    # Ideal weights for WENO-5
    C = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64).unsqueeze(0)
    
    # Nonlinear weights
    alpha = C / (epsilon + IS_tensor)**2
    omega = alpha / alpha.sum(dim=-1, keepdim=True)
    
    print(f"\n  Nonlinear weights:")
    print(f"    ω0 = {omega[0,0].item():.10f}")
    print(f"    ω1 = {omega[0,1].item():.10f}")
    print(f"    ω2 = {omega[0,2].item():.10f}")
    print(f"    Sum = {omega.sum().item():.10f}")
    
    # Compute stencil reconstructions
    flux0 = (2.0/6.0)*v0[0] - (7.0/6.0)*v0[1] + (11.0/6.0)*v0[2]
    flux1 = -(1.0/6.0)*v1[0] + (5.0/6.0)*v1[1] + (2.0/6.0)*v1[2]
    flux2 = (2.0/6.0)*v2[0] + (5.0/6.0)*v2[1] - (1.0/6.0)*v2[2]
    
    print(f"\n  Stencil reconstructions:")
    print(f"    flux0 = {flux0.item():.10f}")
    print(f"    flux1 = {flux1.item():.10f}")
    print(f"    flux2 = {flux2.item():.10f}")
    
    # Final reconstruction
    f_plus_reconstructed = omega[0,0]*flux0 + omega[0,1]*flux1 + omega[0,2]*flux2
    print(f"\n  Final f_plus reconstruction: {f_plus_reconstructed.item():.10f}")
    
    return {
        'IS': [IS0.item(), IS1.item(), IS2.item()],
        'omega': [omega[0,0].item(), omega[0,1].item(), omega[0,2].item()],
        'flux_stencils': [flux0.item(), flux1.item(), flux2.item()],
        'f_plus_reconstructed': f_plus_reconstructed.item()
    }


def compare_reconstructions():
    """
    Compare Gottlieb and GradFlow reconstruction step by step.
    """
    print("\n" + "="*70)
    print("COMPARING RECONSTRUCTIONS")
    print("="*70)
    
    gottlieb = debug_gottlieb_reconstruction()
    gradflow = debug_gradflow_reconstruction()
    
    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*70)
    
    print(f"\nSmoothn indicators:")
    print(f"  {'':20s} {'Gottlieb':>15s} {'GradFlow':>15s} {'Difference':>15s}")
    print(f"  {'-'*65}")
    for i in range(3):
        diff = abs(gottlieb['IS'][i] - gradflow['IS'][i])
        print(f"  {'IS' + str(i):20s} {gottlieb['IS'][i]:15.10e} {gradflow['IS'][i]:15.10e} {diff:15.10e}")
    
    print(f"\nNonlinear weights:")
    print(f"  {'':20s} {'Gottlieb':>15s} {'GradFlow':>15s} {'Difference':>15s}")
    print(f"  {'-'*65}")
    for i in range(3):
        diff = abs(gottlieb['omega'][i] - gradflow['omega'][i])
        print(f"  {'ω' + str(i):20s} {gottlieb['omega'][i]:15.10f} {gradflow['omega'][i]:15.10f} {diff:15.10e}")
    
    print(f"\nStencil reconstructions:")
    print(f"  {'':20s} {'Gottlieb':>15s} {'GradFlow':>15s} {'Difference':>15s}")
    print(f"  {'-'*65}")
    for i in range(3):
        diff = abs(gottlieb['flux_stencils'][i] - gradflow['flux_stencils'][i])
        print(f"  {'flux' + str(i):20s} {gottlieb['flux_stencils'][i]:15.10f} {gradflow['flux_stencils'][i]:15.10f} {diff:15.10e}")
    
    print(f"\nFinal reconstruction:")
    diff = abs(gottlieb['f_plus_reconstructed'] - gradflow['f_plus_reconstructed'])
    print(f"  {'Gottlieb':>20s}: {gottlieb['f_plus_reconstructed']:15.10f}")
    print(f"  {'GradFlow':>20s}: {gradflow['f_plus_reconstructed']:15.10f}")
    print(f"  {'Difference':>20s}: {diff:15.10e}")
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    # Check where the first difference appears
    is_diff = max(abs(gottlieb['IS'][i] - gradflow['IS'][i]) for i in range(3))
    omega_diff = max(abs(gottlieb['omega'][i] - gradflow['omega'][i]) for i in range(3))
    flux_diff = max(abs(gottlieb['flux_stencils'][i] - gradflow['flux_stencils'][i]) for i in range(3))
    
    if is_diff > 1e-10:
        print("\n⚠ Smoothness indicators differ!")
        print(f"  Max IS difference: {is_diff:.10e}")
        print("  → Check smoothness indicator formulas")
    elif omega_diff > 1e-10:
        print("\n⚠ Weights differ!")
        print(f"  Max weight difference: {omega_diff:.10e}")
        print("  → Check weight computation formula")
    elif flux_diff > 1e-10:
        print("\n⚠ Stencil reconstructions differ!")
        print(f"  Max stencil difference: {flux_diff:.10e}")
        print("  → Check reconstruction coefficients")
    else:
        print("\n⚠ All intermediate values match but final result differs")
        print("  → Check how results are combined or indexed")


if __name__ == "__main__":
    compare_reconstructions()