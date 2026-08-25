# GradFlow research direction

## Research question

GradFlow's new research question is:

> Can a direct, maintainable PyTorch system construct, verify, differentiate, and efficiently execute arbitrary-order finite-difference WENO schemes—including a realistic WENO-15 case—without bespoke CUDA or Triton engineering?

The repository currently establishes only a scalar finite-difference WENO-5
seed. It does not yet answer the research question.

## Conclusions and claim boundaries

1. A fixed WENO-5 implementation expressed in PyTorch is not, by itself, a
   novelty claim.

2. Published GPU finite-difference WENO implementations exist. They commonly
   rely on specialized CUDA or other low-level GPU engineering, so GradFlow
   must compare against the relevant literature rather than imply a blank
   field.

3. PyTorch WENO examples and application-specific solvers also exist.
   GradFlow must not claim that WENO has never been attempted in PyTorch.

4. A preliminary search did not identify a prominent, maintained system for
   general arbitrary-order finite-difference WENO construction in ordinary
   PyTorch. This is a literature-review finding, not proof of absence. Search
   terms, databases, inclusion criteria, and close prior art must be recorded
   before any novelty or publication claim.

5. JAX-Fluids is relevant prior art, but it is primarily a finite-volume
   framework. Finite-volume reconstruction and finite-difference flux
   reconstruction are related but not interchangeable, and GradFlow will not
   conflate them.

6. The former foundational `conv1d` idea is now one candidate representation.
   It must eventually be compared empirically with direct shifts,
   slicing/indexing, generated expressions, and other maintainable
   ordinary-PyTorch formulations under identical mathematics.

7. DVEB trunk-001 found that TorchInductor fused the direct WENO-5 formulation
   successfully and made the best ordinary-PyTorch variant substantially
   faster than the DVEB implementation on the screened workload. The negative
   DVEB decision is positive evidence for continuing the PyTorch research
   direction. It is evidence for that workload and environment, not a
   universal performance claim.

8. The PyTorch comparator used no handwritten Triton or CUDA. TorchInductor
   generated backend kernels automatically from ordinary PyTorch source. The
   generated Triton code was compiler output, not comparator source or bespoke
   engineering.

9. WENO-15 is a proposed stress case, not “WENO-5 with a longer stencil.” It
   raises serious questions about:

   - automatic candidate-polynomial generation;
   - optimal linear weights;
   - smoothness-indicator matrices;
   - coefficient conditioning;
   - critical-point accuracy;
   - boundary closures;
   - expression growth;
   - compiler behavior;
   - register pressure and spilling;
   - floating-point stability; and
   - validation against independent mathematics.

10. Potential research contributions may include:

    - arbitrary odd-order finite-difference WENO construction;
    - symbolic or exact-rational coefficient generation;
    - componentwise and characteristic reconstruction;
    - multiple flux-splitting policies;
    - boundary treatments;
    - differentiability for inverse problems and scientific machine learning;
    - CPU, CUDA, and Apple-GPU portability through PyTorch;
    - automatic selection among equivalent PyTorch representations; and
    - systematic accuracy and compiler-performance characterization as order
      increases.

11. Novelty and publishability remain unclaimed until a systematic literature
    review is complete.

## Evidence inherited from DVEB trunk-001

The screened comparator expressed a batched, unique-periodic-node,
right-moving linear-advection WENO-5 RHS and SSP-RK3 step with `torch.roll`
and elementwise tensor operations. Its fullgraph dynamic compile was captured
as one graph with zero breaks and no recompiles across the screened shapes.
The permanent DVEB report records two generated kernels per RHS and six per RK
step, zero transfers in the timed region, and a 1.9--2.1x advantage over DVEB
at the six primary points. Those measurements are preserved in DVEB and are
not being rerun or extended during this refoundation.

The refoundation comparison also found that the baseline's negative split
sign is correct only vacuously for its screened `a = alpha > 0` workload,
where that split is zero. This does not invalidate the trunk decision; it
limits what the screen established. The exact file remains under `baselines/`,
while the canonical scalar implementation follows Gottlieb's sign for both
split families.

## Current representation policy

The canonical source must read as scientific WENO code. For the WENO-5 seed,
the implementation uses only ordinary PyTorch periodic shifts, slicing, and
elementwise operations. It does not select or optimize between representations.

Future representation comparisons must freeze the formulation, correctness
gate, shapes, dtype, devices, compiler version, and measurement protocol
before timing. A compiler-generated low-level kernel remains evidence about
ordinary PyTorch compilation; it does not turn the source into handwritten
Triton.

## Seed gate before arbitrary order

Work on arbitrary order is gated on the bounded checks in
`docs/WENO5_SEED_GATE.md`: Gottlieb oracle agreement, fifth-order smooth
spatial convergence, conservation, float64 CPU/CUDA agreement, eager and
`torch.compile` execution with graph behavior recorded, and absence of hidden
host/device transfers. MPS is recorded as untested when Apple Silicon is not
available; it is never simulated.

The next phase, if separately authorized after this gate, is mathematical and
literature design—not WENO-15 benchmarking or broad optimization.
