# Result and provenance model

## Purpose

A GradFlow result is more than a tensor. It must carry enough information to
interpret, validate, reproduce, compare, and audit a computation. The same
model should serve Python, experiments, stored studies, and a future UI.

This is a target contract. The current `Solver.run` returns a tensor and
exposes limited metadata through `solver.last_run`.

## Result contents

A mature result contains or references:

- requested and resolved scientific problem;
- final state and coordinates or mesh identity;
- simulated time, step count, and stopping reason;
- requested snapshots, histories, and derived quantities;
- units and variable definitions;
- validation outcomes and warnings;
- execution diagnostics; and
- a reproducibility/provenance record.

Large arrays may remain external or device-resident, but their location,
format, dtype, shape, and ownership are explicit. Reading metadata must not
trigger an undeclared device transfer.

## Numerical diagnostics

Relevant diagnostics include conservation residuals, admissibility minima,
CFL/step history, convergence history, boundary flux balance, nonfinite
detection, gradient availability, and application-specific checks.

A warning cannot convert a known-wrong result into success. Violation of a
required correctness condition stops execution or marks the result failed
according to a declared policy.

## Execution diagnostics

The record distinguishes requested and selected backends and explains any
fallback. It includes backend/artifact identity, device and hardware, dtype
and residency, compiler/runtime versions, preparation/transfer/execution
behavior, hidden-transfer count, cache state, material configuration, selector
model identity, and whether the request lies in its calibrated envelope.

Ordinary users need not see every field during a successful run, but all
fields remain inspectable. A legal expert override outside performance
qualification is identified as such.

## Provenance identity

Reproduction requires identities for:

- GradFlow revision and dirty/clean state;
- resolved problem schema and configuration;
- equation and discretization formulation versions;
- generated coefficient payloads;
- native/AOT artifacts and hashes;
- references and oracle hashes;
- dependency and compiler versions;
- machine, accelerator, driver, and runtime;
- seeds and deterministic settings; and
- input state or dataset.

If input cannot be embedded, record a stable hash and useful source reference.
Portable records must not expose secrets or unnecessary personal paths.

## Resolved configuration and Show code

Presets and UI choices alone are insufficient provenance. Results store the
resolved fields. **Show code** generates public Python from that same resolved
configuration and discloses automatic inferences that affect mathematics.

Generated code is a presentation of the problem model, never a separate path
with different defaults or validation.

## Qualification status and serialization

Results distinguish a fully qualified request, a legal expert override outside
performance qualification, an experimental bounded capability, and an
unsupported request (which does not run with substituted semantics).

Stored problems and results have explicit schema versions. Metadata may be
migrated only when semantics are preserved; a numerical change creates a new
problem identity. Scientific array formats should be chosen from real workflow
requirements, but provenance must not be postponed until after results exist.
