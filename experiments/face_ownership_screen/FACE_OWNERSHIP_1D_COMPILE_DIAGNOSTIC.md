# Prospective 1-D compilation diagnostic

Status: **frozen after the primary screen and before this diagnostic was run**.

Date: 2026-08-30 (UTC)

## Trigger

The frozen face-ownership screen found that all 3-D correctness gates passed,
but all four large 1-D configurations failed the compiled-versus-eager gate at
`N=1,048,576`. The failures occurred with one captured graph and no graph
break. The original results remain immutable and excluded from timing.

## Question

Is the 1-D failure associated with the global Lax--Friedrichs reduction, a
particular grid-size range, reconstruction order, precision, or face-ownership
representation?

This is a timing-free compiler diagnostic. It cannot replace a failed primary
screen point or support a speed claim.

## Frozen diagnostic

Use the same deterministic input, Burgers flux, exact `WENOJS` implementation,
error metric, and compiled-versus-eager thresholds as the primary protocol.
For each case, record eager and compiled global `amax(abs(u))`, both RHS
representations, graph counts, pointwise error, finiteness, and conservation.

The matrix is:

```text
orders = {5, 15}
dtypes = {float32, float64}
N      = {65,536; 262,144; 524,288; 786,432; 884,736; 1,048,576}
```

`884,736` matches the cell count of the passing `96^3` case, while the
neighboring values help separate rank-specific behavior from total work.
No execution is timed. Compilation wall time is metadata only.

The following focused probes are also recorded at every size:

- compiled scalar `amax(abs(u))` versus eager;
- `face_once` compiled versus its eager output;
- `cell_recompute` compiled versus its eager output; and
- compiled representation parity.

No thresholds, compiler options, source mathematics, or shapes may be changed
after observation. A follow-up compiler workaround or new timing protocol must
be proposed separately.

## Interpretation boundary

This diagnostic can locate and classify the observed failure on the installed
PyTorch nightly and RTX 5070 Ti. It cannot establish whether the cause belongs
to PyTorch, TorchInductor, CUDA, the Blackwell backend, or GradFlow without
additional evidence, and it cannot generalize to another software build.
