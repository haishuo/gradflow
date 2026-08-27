# GradFlow research direction

## Governing engineering law

All work in this research program is subordinate to the strict precedence

> **Correctness > performance > convenience.**

Correctness qualification precedes performance conclusions. Performance
qualification precedes promotion as a convenient product capability. Neither
an optimization nor an interface simplification may silently change the
requested mathematics. The normative gates, promotion rules, and
technical-debt policy are defined in `ENGINEERING_CHARTER.md`.

## Research question

GradFlow's new research question is:

> Can a direct, maintainable PyTorch system construct, verify, differentiate, and efficiently execute arbitrary-order finite-difference WENO schemes—including a realistic WENO-15 case—without bespoke CUDA or Triton engineering?

The repository now establishes an exact-rational finite-difference WENO-JS
constructor qualified for orders 5--15 in both a scalar periodic path and one
3-D Roe-characteristic Euler path. It does not yet answer the full research
question because general equations, boundaries, geometry, representation
performance, native lowering, and the broader `Solver` surface remain open.

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

9. Scalar periodic and 3-D characteristic Euler WENO-15 have now served as
   construction and compiler stress cases; neither was implemented as
   “WENO-5 with a longer stencil.” The completed seeds address automatic
   candidate polynomials, optimal weights, exact smoothness matrices,
   face-frozen characteristic projection, expression growth through
   full-graph compilation, critical-point characterization, and independent
   validation. They leave serious questions about:

   - coefficient conditioning;
   - boundary closures;
   - register pressure and spilling;
   - floating-point stability;
   - characteristic boundary treatment and other systems; and
   - compiler performance as order and dimension increase.

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

The longer-term product architecture is documented separately in
`PRODUCT_VISION.md`, `PROBLEM_MODEL.md`, `EQUATION_EXTENSION_CONTRACT.md`,
`RESULT_AND_PROVENANCE_MODEL.md`, and `UI_WORKFLOW_CONCEPT.md`. Those documents
state design targets rather than claims that the current narrow solver already
implements a general CFD product.

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

That gate passed before the separately frozen arbitrary-order trunk began.

## Arbitrary-order scalar seed result

One exact-rational constructor now generates candidate reconstructions,
positive optimal weights, Jiang--Shu smoothness matrices, and stable exact
LDLT factors for any odd order of at least three. Orders 5, 7, 9, 11, 13, and
15 passed exact polynomial, smooth convergence, conservation, differentiable,
CPU/CUDA, and fixed-shape full-graph compiler gates. Generated order five
agrees with the canonical seed within `1.715e-13` on the frozen probes.

The critical family `sin(2*pi*x)^3` records material JS order loss—including
approximately second-order point behavior for WENO-5—without post-hoc epsilon
changes or substitution of WENO-Z. This result is part of the mathematical
characterization, not a failed compiler gate. No performance timing was
collected. See `ARBITRARY_ORDER_WENO_JS_RESULTS.md`.

## Characteristic arbitrary-order result

The same generated reconstruction data now drives a face-frozen Roe
characteristic split-flux path for 3-D ideal-gas Euler. Orders 5--15 pass a
smooth entropy-wave convergence gate, exact uniform-state preservation,
conservation, CPU/CUDA agreement, fixed-step differentiation, and fixed-shape
full-graph execution. Generated order five preserves the historical Shu
bakeoff RHS within `2.534e-7` in float32 and `1.222e-15` in float64.

This establishes one realistic system WENO-15 path through the public
`Solver`; it does not establish Navier--Stokes, general equations, boundary
closures, geometric complexity, or application performance. Fresh-cache
compiler preparation was operationally substantial, but no timing protocol
was run and no performance conclusion is claimed. See
`CHARACTERISTIC_ARBITRARY_ORDER_RESULTS.md`.

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

## First system vertical slice

The package now exposes an intentionally restricted 3-D Euler characteristic
WENO-JS `Solver` accepting caller-provided states at qualified orders 5, 7, 9,
11, 13, and 15. It provides direct eager PyTorch on the caller's CPU or CUDA
device, fixed-step differentiation with a finite-gradient gate, physical-state
validation, explicit backend diagnostics, and exact rejection of unsupported
mathematics.

This is not the general API target achieved. Navier--Stokes, general
boundaries, non-Cartesian geometry, and other systems remain unsupported. A
subsequent DVEB portable ABI v1 gate enabled hash-qualified arbitrary-state
forward execution only for the original float32 WENO-5 formulation on CPU or
CUDA. Higher-order or float64 native requests refuse or fall back to PyTorch;
they never substitute different mathematics. The ABI gate's worst
CPU/CUDA/PyTorch difference was `8.345e-7` against a `2e-5` bound. It
establishes an honest integration surface, not arbitrary equations, gradients
through native code, or a new performance claim. The existing fresh-process
selector record is not a substitute for future ABI-endpoint calibration.

Portable device ABI v2 then exposed the same fixed generated CUDA solver to
caller-owned resident tensors and reusable workspace. In its separately frozen
E4 addendum, it passed the full-array gate and won all ten points and all 60
randomized blocks, running 2.53--7.36 times faster than packaged AOTInductor.
This is strong evidence that DVEB deserves a fixed-program native CUDA role.
It remains a result for one float32 formulation, GPU, and endpoint—not evidence
for general automatic placement, arbitrary-order WENO, or novelty.
