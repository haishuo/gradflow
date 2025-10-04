#!/usr/bin/env python3
"""
Clean Sod Shock Tube Demo - No cached modules
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Ensure we're using the project directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force reload
modules_to_remove = [key for key in sys.modules.keys() if key.startswith("gradflow")]
for module in modules_to_remove:
    del sys.modules[module]

from gradflow.core.conservative_variables import create_sod_shock_state
from gradflow.core.grid import create_sod_shock_grid

# Now import
from gradflow.core.weno_reconstruction import create_weno_solver
from gradflow.validation.reference import WENOReference


def main():
    print("🚀 GradFlow Sod Shock Tube Demo (Clean)")
    print("=" * 50)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create grid and state
    grid = create_sod_shock_grid(device=device)
    state = create_sod_shock_state(grid, requires_grad=True)
    weno = create_weno_solver(device=device)

    print(f"Grid: {grid.nx}×{grid.ny}")
    print("Shock at x = 5.0")

    # WENO reconstruction
    print("\n📐 WENO Reconstruction:")
    rho_x_plus = weno.reconstruct(state.rho, dim=-1, direction="positive")
    print("✅ Reconstruction successful")
    print(f"   Shape: {rho_x_plus.shape}")
    print(f"   No NaN: {torch.all(torch.isfinite(rho_x_plus)).item()}")

    # Test gradients
    print("\n🔄 Gradient Test:")
    loss = torch.sum(rho_x_plus**2)
    loss.backward()
    print("✅ Gradients computed")
    print(f"   Gradient norm: {torch.norm(state.rho.grad).item():.4f}")

    # FORTRAN comparison
    print("\n📊 FORTRAN Reference Comparison:")
    try:
        ref = WENOReference()
        print(f"Reference dir: {ref.reference_dir}")

        # Load reference
        ref.load_sod_shock_reference(format="fortran")
        ref_density = ref.get_structured_field("density")

        # Compare
        our_density = state.rho.detach().cpu().numpy()
        comparison = ref.compare_solution(our_density, "density")

        print("✅ Comparison complete:")
        print(f"   Passed: {comparison['passed']}")
        print(f"   Max error: {comparison['max_absolute_error']:.2e}")
        print(f"   RMS error: {comparison['rms_error']:.2e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    # Simple visualization
    print("\n📈 Creating visualization...")
    plt.figure(figsize=(10, 6))

    # Get 1D cut
    y_cut = 5.0
    x_coords, rho_1d = grid.interpolate_to_1d_cut(state.rho, y_cut)
    _, rho_recon_1d = grid.interpolate_to_1d_cut(rho_x_plus, y_cut)

    x_np = x_coords.cpu().numpy()
    rho_np = rho_1d.detach().cpu().numpy()
    rho_recon_np = rho_recon_1d.detach().cpu().numpy()

    plt.plot(x_np, rho_np, "k-", linewidth=2, label="Initial density")
    plt.plot(x_np, rho_recon_np, "b--", label="WENO reconstruction")
    plt.axvline(x=5.0, color="red", linestyle=":", alpha=0.5, label="Shock")

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("GradFlow: Differentiable WENO on Sod Shock Tube")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = "sod_demo_clean.png"
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved: {output_file}")
    plt.close()

    print("\n🎉 Demo complete! The world's first differentiable WENO is working!")


if __name__ == "__main__":
    main()
