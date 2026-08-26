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

## Matched 3-D deployment evidence

The later matched Shu Euler WENO-5 campaign asks a different question from
DVEB trunk-001. It compares complete fresh-process latency for the same
float32 3-D characteristic finite-difference formulation, including pageable
host initialization, CFL recomputation, all SSP-RK3 stages, transfers, final
host materialization, validation, and process exit.

On the Ryzen 7600X / RTX 5070 Ti machine, a hash-frozen automatic-placement
DVEB artifact won 8 of 9 declared points over Fortran, direct eager PyTorch,
and fixed-shape AOTInductor PyTorch. It selected a six-thread CPU schedule for
one-step N=8--32 work and CUDA for one-step N=64--128 work; for ten steps it
selected CUDA beginning at N=32. Fortran retained the N=8 / one-step win. At
N=128 / ten steps, median complete latency was 0.343 s for DVEB, 3.377 s for
AOT PyTorch, 5.432 s for eager PyTorch, and 30.582 s for Fortran.

This establishes a bounded engineering result: application-calibrated native
CPU/CUDA dispatch can be substantially more effective than forcing every job
through Python or one device, and DVEB's emitted WENO CUDA is a strong backend
candidate. It does not reverse trunk-001's scalar result, prove a general
language claim, or establish novelty. DVEB's later generic disjoint-point
campaign at commit `2f1f3ab` recorded NO-GO for the initial automatic selector
because fresh-process maximum regret and CPU-schedule proximity missed their
frozen bands. The WENO campaign trained on its evaluated points, so it does not
override that held-out decision. The campaign covers one 3-D
formulation, one initial condition, float32, and one machine. The DVEB artifact
was produced from an uncommitted compiler worktree state and is hash-frozen;
DVEB subsequently committed and qualified a different final artifact. See
`experiments/shu_torch_ablation/DVEB_BAKEOFF_RESULTS.md`.

## Seed gate before arbitrary order

Work on arbitrary order is gated on the bounded checks in
`docs/WENO5_SEED_GATE.md`: Gottlieb oracle agreement, fifth-order smooth
spatial convergence, conservation, float64 CPU/CUDA agreement, eager and
`torch.compile` execution with graph behavior recorded, and absence of hidden
host/device transfers. MPS is recorded as untested when Apple Silicon is not
available; it is never simulated.

The next phase, if separately authorized after this gate, is mathematical and
literature design—not WENO-15 benchmarking or broad optimization.

## Final DVEB WENO requalification

GradFlow subsequently requalified DVEB's final reproducible Shu Euler artifact
at committed DVEB revision `2f1f3ab` using disjoint calibration and evaluation
sizes. Full-state error remained at most `7.153e-7`. At ten held-out N=8--64,
one/ten-step points, all automatic selections were stable; median
fresh-process regret was 1.0014 and maximum regret was 1.2255 under the frozen
acceptance bands. The selector moved the one-step workload to CUDA at N=64 and
the ten-step workload at N=32.

At N=96/128, automatic placement safely refused because the model was bounded
by its N=7--72 calibration range. Forced generated CUDA nevertheless remained
within 1.65% of the independent native ceiling over the four declared
large-grid points. This yields a deliberately split conclusion: DVEB has a
validated role as a native backend for this WENO formulation, and its
WENO-specific automatic placement is qualified only within the tested
machine-specific envelope. Its generic selector remains NO-GO. See
`experiments/shu_torch_ablation/DVEB_FINAL_REQUALIFICATION_RESULTS.md` and
`docs/BACKEND_SELECTION_CONTRACT.md`.
