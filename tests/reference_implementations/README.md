# Reference Implementations for Validation

## Primary Reference (Use This)

### gottlieb_matlab/
Professor Sigal Gottlieb's MATLAB WENO-5 implementation.
- **Status:** Trusted, expert-verified
- **Use for:** Primary validation of GradFlow WENO-5

## External References (Separate Repositories)

### Modified FORTRAN Code
**Location:** `/mnt/projects/weno-reference/` (separate git repository)

**WARNING: DO NOT USE FOR VALIDATION**

This code has been modified from the original Jiang & Shu reference and is
not suitable for validation. It exists as a separate project.

For validation, use gottlieb_matlab instead.

### Original Jiang & Shu FORTRAN
**Status:** Not yet located

Sources to check:
1. Chi-Wang Shu's website: https://www.dam.brown.edu/people/shu/
2. JCP 1996 paper supplementary materials
3. Contact Professor Gottlieb

## Validation Protocol

1. Port Gottlieb MATLAB → Python
2. Run identical test cases  
3. Compare GradFlow vs Gottlieb (target: error < 1e-12)
4. Document in docs/validation.md

