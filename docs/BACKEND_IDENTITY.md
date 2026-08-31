# GradFlow system and backend identity

Status: **normative terminology and interpretation policy**.

Date: 2026-08-31 (UTC)

## GradFlow is the system, not one backend

GradFlow is the system that constructs or selects a numerical formulation,
qualifies it against the requested correctness contract, and then chooses the
fastest admitted execution path for the submitted problem and machine. It is
not synonymous with PyTorch, TorchInductor, DVEB, native C++, CUDA, finite
difference, or finite volume.

The governing order remains:

> **Correctness > performance > convenience.**

GradFlow may choose only among paths that preserve the user's declared
mathematics and pass the applicable accuracy, conservation, stability, and
state-semantics gates. Once multiple paths are admitted, measured performance
decides; familiarity or implementation history does not grant a backend
priority.

The conceptual layers are separate:

1. **problem:** equation, dimension, state, parameters, initial and boundary
   conditions;
2. **numerical contract:** finite difference or finite volume, WENO family and
   order, flux split, reconstruction, precision, boundary closure, and time
   integration;
3. **execution backend:** native CPU, PyTorch eager, TorchInductor JIT,
   AOTInductor, DVEB CPU/CUDA, or another qualified implementation; and
4. **selection policy:** automatic or an explicit user override.

Finite difference versus finite volume can change the numerical contract and
therefore requires stronger equivalence or accuracy-to-time reasoning than a
CPU-versus-CUDA implementation choice. GradFlow must never exchange one for
the other merely because it is faster without disclosing and qualifying that
mathematical change.

## PyTorch was an implementation hypothesis

The 2025--2026 refoundation asked whether direct, maintainable ordinary
PyTorch could construct, differentiate, and efficiently execute arbitrary-
order FD-WENO while TorchInductor generated low-level kernels automatically.
That was and remains a valuable research hypothesis. It was never a
definition of GradFlow's identity.

The evidence is regime-dependent:

- DVEB trunk-001 showed that TorchInductor could fuse direct WENO-5 and beat
  the then-current DVEB implementation on its screened workload.
- The arbitrary-order work showed one full graph with zero breaks through
  WENO-15 and established that ordinary PyTorch can be a readable,
  differentiable construction and verification surface.
- U4-E rejected PyTorch/TorchInductor as the fastest admitted execution path
  for scalar one-dimensional float64 WENO-JS5 at `N=8192` on Forge. DVEB was
  `1.60x` faster on one-thread CPU and `3.58x` faster on resident CUDA; the
  independent OpenSBLI/OPS lane was also faster than PyTorch on both devices.

For that frozen regime, the competitive-performance form of the PyTorch
hypothesis is disproven. GradFlow should not select PyTorch there.

This does not prove that Python or PyTorch must lose universally. A larger,
higher-order, multidimensional, batched, differentiated, or more deeply fused
workload may amortize framework and launch overhead and may favor
TorchInductor's fusion. Such a regime must be demonstrated prospectively; it
cannot be presumed as a rescue argument.

U4-E also does not isolate all of its resident gap as Python-interpreter
overhead. The CUDA decision endpoint excluded compilation and transfer and
used device events around the scheduled numerical work. Its gap therefore
includes the efficiency and launch topology of the generated device program,
not merely Python startup. The much larger prepared launch-to-answer gap does
directly demonstrate additional runtime/framework weight. Both observations
matter, but they answer different questions.

PyTorch may remain the clearest mathematical seed, differentiable oracle, or
fallback for unsupported schemes even when it is not the selected production
backend. Correctness authority and execution preference are distinct roles.

## Historical evidence-key clarification

Frozen U4-C through U4-E machine-readable records use these historical keys:

| frozen key | actual implementation |
|---|---|
| `gradflow` | repository-native PyTorch with `torch.compile`/TorchInductor |
| `gradflow_cpu` | that PyTorch/TorchInductor implementation on CPU |
| `gradflow_cuda` | that PyTorch/TorchInductor implementation on CUDA |
| `gradflow_aot` | the packaged PyTorch AOTInductor implementation |

Those keys are immutable provenance and must not be rewritten, because doing
so would invalidate retained evidence hashes. They are legacy aliases, not a
claim that PyTorch alone is GradFlow. Human-facing text must call these lanes
“PyTorch/TorchInductor” or “GradFlow's PyTorch/TorchInductor backend.”

New experiments must use backend-specific identifiers such as
`pytorch_inductor`, `pytorch_aotinductor`, `dveb_native`, and
`opensbli_ops`. The unqualified identifier `gradflow` is reserved for the
encompassing system or a complete automatic-policy result, never an individual
backend.

## Consequence for U4-E

The correct U4-E statement is:

> For the frozen scalar float64 WENO-JS5 `N=8192` contract, the automatically
> scheduled DVEB implementation was the resolved resident winner over the
> OpenSBLI/OPS-generated implementation and the PyTorch/TorchInductor
> implementation on both one-thread CPU and CUDA.

It is incorrect to say that DVEB “beat GradFlow.” DVEB is a candidate GradFlow
backend. A future GradFlow automatic policy that selects DVEB for this cell
would be using the U4-E evidence exactly as intended.
