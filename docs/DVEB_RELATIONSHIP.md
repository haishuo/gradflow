# GradFlow and DVEB relationship

## Independence rule

DVEB is a separate language/compiler project, not GradFlow's private code
generator. GradFlow may motivate or test a DVEB capability only when the
answer to the following counterfactual is yes:

> If GradFlow did not exist, would this capability still make sense for DVEB
> as a general scientific language and compiler?

If the answer is no, the feature does not belong in DVEB. It may belong in
GradFlow, in a GradFlow-owned generator, or only in an experiment.

## Project priority

For this project, GradFlow is primary and DVEB is a secondary development arm.
DVEB may develop in parallel, but GradFlow's academic progress must not depend
on speculative DVEB work. The canonical ordinary-PyTorch implementation,
mathematical qualification, and paper evidence must stand without it.

DVEB is eligible to participate as:

- a separately versioned comparator;
- an optional native execution backend with exact semantic parity;
- evidence about generated CPU/CUDA performance and automatic placement; and
- a general-language research result in its own right.

GradFlow does not contort its mathematics or public API to manufacture a DVEB
win. A negative DVEB result is preserved and does not block GradFlow.

## Capabilities that can be general DVEB work

The independence rule can admit capabilities such as:

- versioned host- and device-resident array ABIs;
- caller-owned buffers, streams, and reusable workspace;
- general stencil/indexing and reduction lowering;
- CPU SIMD and multithreaded scheduling;
- CUDA lowering for general parallel loops and array expressions;
- evidence-based target selection and calibrated refusal outside a model's
  envelope;
- ahead-of-time artifact packaging, hashing, and compatibility checks;
- explicit target overrides; and
- diagnostics for transfers, memory, compilation, and execution.

These capabilities remain useful to many scientific programs even if WENO is
removed from the motivating examples.

## Capabilities that fail the rule

The following do not belong in DVEB merely to help GradFlow:

- a WENO-5 opcode or syntax with no general language semantics;
- a hard-coded five-component Euler state or Shu-vortex program;
- special compiler recognition of GradFlow function names;
- a placement table containing only hand-entered GradFlow benchmark cases;
- a one-off ABI that cannot represent other array programs; or
- semantic concessions made solely to beat another GradFlow backend.

A fixed WENO artifact may still be a valid generated test artifact. It must not
be confused with a general DVEB language capability.

## Change-admission test

Before GradFlow requests DVEB development, record:

1. the missing general capability;
2. at least one plausible non-GradFlow use;
3. the language/runtime contract rather than the WENO-specific symptom;
4. independent DVEB tests that do not import GradFlow;
5. the GradFlow experiment that will consume it; and
6. the stop condition if it is not competitive or maintainable.

Passing this test permits investigation; it does not predetermine adoption.
The emitted result still passes GradFlow correctness and performance gates.

## Ownership and integration

DVEB owns its language semantics, compiler implementation, generic tests, and
release artifacts. GradFlow owns its equation/discretization contract,
backend adapter, parity tests, benchmark protocol, and decision about whether
the backend is eligible.

Integration uses versioned public artifacts or ABIs with hashes and explicit
capability declarations. Neither repository silently edits or vendors an
unreleased copy of the other's canonical source.

## Academic claim boundary

The main GradFlow paper may report DVEB observations when the comparison is
mathematically matched and reproducible. General claims about the DVEB
language belong to DVEB's own research record. GradFlow's central academic
claim must not collapse into “a bespoke compiler can run this one solver.”
