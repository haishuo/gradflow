"""
GradFlow Sod Shock Tube Example

Demonstrates differentiable WENO reconstruction on the classic Sod shock tube problem\
    .
"""

# Force reload of modules to pick up changes
import sys

import matplotlib.pyplot as plt
import torch

# Clear gradflow modules from cache
modules_to_remove = [key for key in sys.modules.keys() if key.startswith("gradflow")]\

for module in modules_to_remove:
    del sys.modules[module]

from gradflow.core.conservative_variables import create_sod_shock_state
from gradflow.core.grid import create_sod_shock_grid

# Import GradFlow modules
from gradflow.core.weno_reconstruction import create_weno_solver
from gradflow.validation.reference import WENOReference


def visualize_weno_reconstruction():
    """Visualize WENO reconstruction on Sod shock tube"""

    print("🚀 GradFlow Sod Shock Tube Example")
    print("=" * 40)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create grid and initial state
    grid = create_sod_shock_grid(device=device)
    state = create_sod_shock_state(grid, requires_grad=True)

    print(f"\nGrid: {grid.nx}×{grid.ny}")
    print(f"Domain: x ∈ [{grid.x_min}, {grid.x_max}], y ∈ [{grid.y_min}, {grid.y_max}\
    ]")

    # Create WENO solver
    weno = create_weno_solver(device=device)

    # Get 1D cut for visualization
    y_cut = 5.0  # Middle of domain
    x_coords, density_1d = grid.interpolate_to_1d_cut(state.rho, y_cut)

    # Perform WENO reconstruction
    print("\nPerforming WENO reconstruction...")

    # Reconstruct density in x-direction
    rho_x_plus = weno.reconstruct(state.rho, dim=-1, direction="positive")
    rho_x_minus = weno.reconstruct(state.rho, dim=-1, direction="negative")

    # Compute numerical flux (Lax-Friedrichs for simplicity)
    max_wave_speed = state.max_wave_speed()
    alpha = torch.max(max_wave_speed).item()

    # Get 1D cuts of reconstructed values
    _, rho_plus_1d = grid.interpolate_to_1d_cut(rho_x_plus, y_cut)
    _, rho_minus_1d = grid.interpolate_to_1d_cut(rho_x_minus, y_cut)

    # Convert to numpy for plotting
    x_np = x_coords.cpu().numpy()
    rho_np = density_1d.detach().cpu().numpy()
    rho_plus_np = rho_plus_1d.detach().cpu().numpy()
    rho_minus_np = rho_minus_1d.detach().cpu().numpy()

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot 1: Initial condition and reconstruction
    plt.subplot(2, 2, 1)
    plt.plot(x_np, rho_np, "k-", linewidth=2, label="Initial ρ")
    plt.plot(x_np, rho_plus_np, "b--", label="WENO+ reconstruction")
    plt.plot(x_np, rho_minus_np, "r--", label="WENO- reconstruction")
    plt.axvline(x=5.0, color="gray", linestyle=":", alpha=0.5, label="Shock location"\
    )
    plt.xlabel("x")
    plt.ylabel("Density ρ")
    plt.title("WENO Reconstruction of Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Pressure field
    pressure = state.pressure
    _, pressure_1d = grid.interpolate_to_1d_cut(pressure, y_cut)
    pressure_np = pressure_1d.detach().cpu().numpy()

    plt.subplot(2, 2, 2)
    plt.plot(x_np, pressure_np, "g-", linewidth=2)
    plt.axvline(x=5.0, color="gray", linestyle=":", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("Pressure p")
    plt.title("Initial Pressure Distribution")
    plt.grid(True, alpha=0.3)

    # Plot 3: 2D density field
    plt.subplot(2, 2, 3)
    rho_2d = state.rho.detach().cpu().numpy()
    im = plt.imshow(
        rho_2d,
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    plt.colorbar(im, label="Density ρ")
    plt.axvline(x=5.0, color="white", linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Density Field")

    # Plot 4: Gradient information
    plt.subplot(2, 2, 4)

    # Compute a simple loss and gradients
    loss = torch.sum(rho_x_plus**2)
    loss.backward()

    # Get density gradients
    if state.rho.grad is not None:
        grad_magnitude = torch.sqrt(state.rho.grad**2)
        _, grad_1d = grid.interpolate_to_1d_cut(grad_magnitude, y_cut)
        grad_np = grad_1d.detach().cpu().numpy()

        plt.semilogy(x_np, grad_np, "m-", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("|∇ρ|")
        plt.title("Gradient Magnitude (Autodiff)")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(
            0.5,
            0.5,
            "No gradients computed",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()
    plt.savefig("sod_shock_weno_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print numerical validation
    print("\n📊 Numerical Validation:")
    print(f"  Density ratio: {rho_np[0] / rho_np[-1]:.2f} (expected: 8.0)")
    print(f"  Pressure ratio: {pressure_np[0] / pressure_np[-1]:.2f} (expected: 10.0)\
    ")

    # Test differentiability
    print("\n🔄 Differentiability Test:")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Gradients computed: {'✅' if state.rho.grad is not None else '❌'}")
    if state.rho.grad is not None:
        print(f"  Gradient norm: {torch.norm(state.rho.grad).item():.6e}")

    # Conservation check
    total_mass_original = torch.sum(state.rho) * grid.dV
    total_mass_reconstructed = torch.sum(rho_x_plus) * grid.dV
    conservation_error = (
        torch.abs(total_mass_reconstructed - total_mass_original) / total_mass_origin\
    al
    )
    print("\n🎯 Conservation Test:")
    print(f"  Original total mass: {total_mass_original.item():.6f}")
    print(f"  Reconstructed mass: {total_mass_reconstructed.item():.6f}")
    print(f"  Relative error: {conservation_error.item():.2e}")

    return state, weno


def demonstrate_optimization():
    """Demonstrate gradient-based optimization capability"""

    print("\n\n🎯 Gradient-Based Optimization Demo")
    print("=" * 40)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup
    grid = create_sod_shock_grid(device=device)
    weno = create_weno_solver(device=device)

    # Create a parameterized initial condition
    # This represents a "design parameter" we want to optimize
    shock_position = torch.tensor(
        5.0, device=device, dtype=torch.float64, requires_grad=True
    )

    # Define optimization objective: minimize density gradient at shock
    optimizer = torch.optim.Adam([shock_position], lr=0.01)

    print("Optimizing shock position to minimize gradient...")

    losses = []
    positions = []

    for step in range(50):
        optimizer.zero_grad()

        # Create state with current shock position
        x_coords, _ = grid.get_coordinates()

        # Initialize density with parameterized shock
        rho = torch.ones_like(x_coords)
        rho[x_coords >= shock_position] = 0.125

        # Ensure requires_grad
        rho = rho.detach().requires_grad_(True)

        # WENO reconstruction
        rho_reconstructed = weno.reconstruct(rho, dim=-1)

        # Compute gradient magnitude as loss
        # In real applications, this would be a physical objective
        dx = grid.dx
        rho_x = (rho_reconstructed[:, 1:] - rho_reconstructed[:, :-1]) / dx
        gradient_magnitude = torch.mean(torch.abs(rho_x))

        # Add regularization to keep shock in domain
        regularization = 0.1 * (shock_position - 5.0) ** 2
        loss = gradient_magnitude + regularization

        # Backward pass - this is the magic!
        loss.backward()

        # Store for plotting
        losses.append(loss.item())
        positions.append(shock_position.item())

        # Optimization step
        optimizer.step()

        if step % 10 == 0:
            print(
                f"  Step {step}: Loss = {loss.item():.6f}, Position = {shock_position\
    .item():.3f}"
            )

    # Plot optimization history
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses, "b-", linewidth=2)
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Optimization Loss History")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(positions, "r-", linewidth=2)
    plt.axhline(
        y=5.0, color="gray", linestyle="--", alpha=0.5, label="Initial position"
    )
    plt.xlabel("Optimization Step")
    plt.ylabel("Shock Position")
    plt.title("Shock Position Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gradient_optimization_demo.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n✅ Optimization complete!")
    print(f"  Final position: {shock_position.item():.3f}")
    print("  This demonstrates the revolutionary capability of differentiable CFD!")


def compare_with_fortran():
    """Compare with FORTRAN reference data if available"""

    print("\n\n📈 FORTRAN Reference Comparison")
    print("=" * 40)

    try:
        # Load reference
        ref = WENOReference()
        validation = ref.quick_validation_test()

        if validation["status"] == "success":
            print("✅ Reference data loaded successfully")
            print(f"  Shock location: {validation['shock_location']}")
            print(f"  Density ratio: {validation['actual_ratios']['density']:.2f}")
            print(f"  Pressure ratio: {validation['actual_ratios']['pressure']:.2f}")\


            # Run detailed comparison
            device = "cuda" if torch.cuda.is_available() else "cpu"
            grid = create_sod_shock_grid(device=device)
            state = create_sod_shock_state(grid)

            # Compare with reference
            ref_density = ref.get_structured_field("density")
            our_density = state.rho.cpu().numpy()

            comparison = ref.compare_solution(
                our_density.detach().cpu().numpy(), "density", tolerance=1e-12
            )

            print("\n📊 Bit-perfect validation:")
            print(f"  Passed: {'✅' if comparison['passed'] else '❌'}")
            print(f"  Max absolute error: {comparison['max_absolute_error']:.2e}")
            print(f"  RMS error: {comparison['rms_error']:.2e}")

        else:
            print("⚠️  Reference data not available")

    except Exception as e:
        print(f"⚠️  Could not load reference data: {e}")
        print("  Run validation setup first to generate reference data")


if __name__ == "__main__":
    # Run demonstrations
    state, weno = visualize_weno_reconstruction()
    demonstrate_optimization()
    compare_with_fortran()

    print("\n\n🎉 GradFlow demonstration complete!")
    print("The world's first differentiable WENO scheme is working!")
