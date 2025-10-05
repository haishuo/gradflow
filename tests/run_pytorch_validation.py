#!/usr/bin/env python3
"""
Run validation for PyTorch translation of Gottlieb's WENO-5.

This script validates step 5 & 6 of the rebuild plan:
- Step 5: PyTorch line-by-line translation ✓
- Step 6: Validate against gold standard ⏳
"""

import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gottlieb_weno5_pytorch import (
    validate_against_numpy_reference,
    validate_against_matlab_output
)

def main():
    print("="*70)
    print("VALIDATION: PyTorch Translation of Gottlieb WENO-5")
    print("="*70)
    
    # Step 6a: Validate against NumPy reference
    print("\n" + "="*70)
    print("Validation 1: PyTorch vs NumPy Reference")
    print("="*70)
    numpy_ok = validate_against_numpy_reference()
    
    if not numpy_ok:
        print("\n✗ FAILED: PyTorch does not match NumPy reference")
        print("  Debug the PyTorch implementation before proceeding.")
        return False
    
    # Step 6b: Validate against MATLAB gold standard
    print("\n" + "="*70)
    print("Validation 2: PyTorch vs MATLAB Gold Standard (.h5)")
    print("="*70)
    matlab_ok = validate_against_matlab_output()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if numpy_ok and matlab_ok:
        print("\n✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
        print("\nResults:")
        print("  ✓ PyTorch matches NumPy reference to machine precision")
        print("  ✓ PyTorch matches MATLAB gold standard to machine precision")
        print("\nSTATUS: Step 5 & 6 COMPLETE")
        print("\nNEXT STEP: Proceed to step 7")
        print("  - Refactor monolithic PyTorch code into modular components")
        print("  - Update weno.py, flux.py, etc. to match validated algorithm")
        print("  - Test each module against PyTorch reference as you go")
        return True
    elif numpy_ok and not matlab_ok:
        print("\n⚠ PARTIAL SUCCESS")
        print("  ✓ PyTorch matches NumPy")
        print("  ✗ PyTorch differs from MATLAB")
        print("\nThis suggests the NumPy reference might have issues.")
        print("Review the error magnitudes to determine if acceptable.")
        return False
    else:
        print("\n✗ VALIDATION FAILED")
        print("  PyTorch implementation needs debugging.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
