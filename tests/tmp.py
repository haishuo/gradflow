"""
Quick test to verify the weight formula matches Gottlieb.
"""

import numpy as np

# Test values
epsilon = 1e-29
IS0 = 0.1
IS1 = 0.2
IS2 = 0.3

print("Testing WENO weight formulas")
print("="*70)
print(f"epsilon = {epsilon:.2e}")
print(f"IS0 = {IS0}, IS1 = {IS1}, IS2 = {IS2}")

# Standard WENO weights (WRONG for Gottlieb)
print("\n--- Standard WENO weights (WRONG) ---")
alpha0 = 0.1 / (epsilon + IS0)**2
alpha1 = 0.6 / (epsilon + IS1)**2
alpha2 = 0.3 / (epsilon + IS2)**2
alpha_sum = alpha0 + alpha1 + alpha2

s1_wrong = alpha0 / alpha_sum
s2_wrong = alpha1 / alpha_sum
s3_wrong = alpha2 / alpha_sum

print(f"s1 = {s1_wrong:.15f}")
print(f"s2 = {s2_wrong:.15f}")
print(f"s3 = {s3_wrong:.15f}")
print(f"sum = {s1_wrong + s2_wrong + s3_wrong:.15f}")

# Gottlieb's weight formula (CORRECT)
print("\n--- Gottlieb's weight formula (CORRECT) ---")
tt1 = (epsilon + IS0)**2
tt2 = (epsilon + IS1)**2
tt3 = (epsilon + IS2)**2

s1_correct = tt2 * tt3
s2_correct = 6.0 * tt1 * tt3
s3_correct = 3.0 * tt1 * tt2

t0 = 1.0 / (s1_correct + s2_correct + s3_correct)
s1_correct = s1_correct * t0
s2_correct = s2_correct * t0
s3_correct = s3_correct * t0

print(f"s1 = {s1_correct:.15f}")
print(f"s2 = {s2_correct:.15f}")
print(f"s3 = {s3_correct:.15f}")
print(f"sum = {s1_correct + s2_correct + s3_correct:.15f}")

print("\n--- Difference ---")
print(f"Δs1 = {abs(s1_correct - s1_wrong):.15e}")
print(f"Δs2 = {abs(s2_correct - s2_wrong):.15e}")
print(f"Δs3 = {abs(s3_correct - s3_wrong):.15e}")

print("\n" + "="*70)
print("The two formulas give VERY different weights!")
print("This is why the solution doesn't match.")