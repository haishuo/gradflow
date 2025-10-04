#!/usr/bin/env python3
"""
Test script to debug reference data loading
"""

# Add gradflow to path
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from gradflow.core.conservative_variables import create_sod_shock_state
from gradflow.core.grid import create_sod_shock_grid
from gradflow.validation.reference import WENOReference


def test_reference_loading():
    print("🔍 Testing Reference Data Loading")
    print("=" * 50)

    # Step 1: Check reference directory
    ref = WENOReference()
    print(f"Reference directory: {ref.reference_dir}")
    print(f"Directory exists: {ref.reference_dir.exists()}")

    if ref.reference_dir.exists():
        print("\nFiles in reference directory:")
        for f in ref.reference_dir.glob("*"):
            print(f"  - {f.name}")

    # Step 2: Test quick validation
    print("\n📊 Quick validation test:")
    validation = ref.quick_validation_test()
    print(f"Status: {validation['status']}")

    if validation["status"] == "success":
        print(f"Shock location: {validation['shock_location']}")
        print(f"Density ratio: {validation['actual_ratios']['density']:.2f}")
        print(f"Pressure ratio: {validation['actual_ratios']['pressure']:.2f}")

    # Step 3: Load reference data
    print("\n📁 Loading reference data:")
    try:
        ref.load_sod_shock_reference(format="fortran")
        print("✅ Reference data loaded successfully")

        # Get density field
        density = ref.get_structured_field("density")
        print(f"Density shape: {density.shape}")
        print(f"Density type: {type(density)}")
        print(f"Density range: [{np.min(density):.3f}, {np.max(density):.3f}]")

    except Exception as e:
        print(f"❌ Error loading reference: {e}")
        import traceback

        traceback.print_exc()

    # Step 4: Create our state and compare
    print("\n🔬 Creating GradFlow state:")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = create_sod_shock_grid(device=device)
    state = create_sod_shock_state(grid, requires_grad=True)

    print(f"State density shape: {state.rho.shape}")
    print(f"State density device: {state.rho.device}")
    print(f"State density requires_grad: {state.rho.requires_grad}")

    # Step 5: Test comparison
    print("\n🔀 Testing comparison:")

    # Method 1: Detach before numpy
    print("Method 1: Detach before numpy")
    our_density_np = state.rho.detach().cpu().numpy()
    print(f"  Numpy array shape: {our_density_np.shape}")

    try:
        comparison = ref.compare_solution(our_density_np, "density", tolerance=1e-12)
        print(f"  ✅ Comparison successful: {comparison['passed']}")
        print(f"  Max error: {comparison['max_absolute_error']:.2e}")
    except Exception as e:
        print(f"  ❌ Comparison failed: {e}")

    # Method 2: Pass tensor directly (should be handled by compare_solution)
    print("\nMethod 2: Pass tensor directly")
    try:
        comparison = ref.compare_solution(state.rho, "density", tolerance=1e-12)
        print(f"  ✅ Comparison successful: {comparison['passed']}")
        print(f"  Max error: {comparison['max_absolute_error']:.2e}")
    except Exception as e:
        print(f"  ❌ Comparison failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_reference_loading()
