# Convolution-oriented experiments (noncanonical)

These exact files came from GradFlow commit
`4c861fdf4ec31932a8dd815ae9884be8ceba3a37`. They generate or store stencil,
smoothness, and weight coefficients intended to be usable with `conv1d`.
They do not constitute a validated convolution WENO solver; the historical
active solver still used explicit loops and slicing.

The old `conv1d` idea is now one unproven representation hypothesis among
several. It may later be compared with shifts, slicing/indexing, and generated
ordinary-PyTorch expressions, but it is not imported by `src/gradflow` and no
performance conclusion is carried into the restarted project.

| File | SHA-256 |
|---|---|
| `generator.py` | `c3908ac9d268b81c92f4deb6da48f112130349b9e6bf6cff71d392e5deae8a08` |
| `smoothness.py` | `2223a8ff87cd1e59492864df663235b4c2e61a044bda1b96bab1b23e900a3df0` |
| `stencils.py` | `db73d80bc129420992a4140a94944ef373b7f3d0d54a06cbce77b03a9f6eaeb4` |
| `weights.py` | `f66945f83252cf2e243b1fadb8bd534d4e02ec94fd75c40c21fa51b47c563739` |
