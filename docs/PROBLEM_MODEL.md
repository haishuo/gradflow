# Scientific problem model

## Purpose

The problem model is the backend-neutral contract between user intent and
numerical execution. Python, a future UI, command-line tools, and stored
studies must all resolve to the same model.

This document specifies direction, not an already implemented general API.
The current `Solver` remains the narrow slice in `SOLVER_VERTICAL_SLICE.md`.

## Separation of concerns

A complete request has distinct parts:

```text
Problem
|- equation specification
|- domain and geometry
|- physical parameters
|- initial state or initial-condition construction
|- boundary regions and conditions
|- spatial discretization policy
|- temporal policy and stopping condition
|- requested outputs
`- execution policy
```

These parts must not collapse into a benchmark name. Compressible Euler is an
equation family; the Sod shock tube is a scenario; JS-5 is a discretization;
CPU SIMD is an execution target; and a shock position is a requested result.

## Equation specification

An equation entry identifies:

- a stable family name and version;
- supported dimensions and coordinate systems;
- conserved, primitive, and auxiliary variables with units and layout;
- physical parameters and admissible ranges;
- inviscid fluxes, diffusive operators, and sources;
- closure relations and thermodynamic assumptions;
- state-validation rules and required characteristic information;
- timestep restrictions; and
- derived quantities available for output.

It describes mathematics and never selects CPU, CUDA, compilation, or DVEB.

## Domain, initial state, and boundaries

The domain records coordinates, extent, mesh, topology, endpoint convention,
and named boundary regions. Geometry ingestion and mesh generation require
their own qualification. Accepting a geometry file is not evidence that its
mesh is numerically supported.

An initial condition may be caller-owned state, a qualified scenario, or
eventually a callable satisfying the equation's tensor and admissibility
contract. Boundaries are assigned by region and declare their required data,
supported equations, numerical closure, and compatibility restrictions.
Periodic, wall, inflow, and outflow are mathematical contracts, not labels.

The engine rejects missing, over-specified, incompatible, or unqualified
boundary combinations before execution.

## Discretization and time policy

The resolved spatial policy includes finite-difference/finite-volume family,
WENO family and order, characteristic/componentwise policy, flux splitting,
epsilon and smoothness scaling, boundary closure, endpoint convention, dtype,
and any stabilization or positivity policy.

The temporal policy includes integrator, CFL or explicit step, stopping
condition, maximum-step guard, and output schedule. Friendly presets such as
`accuracy="high"` may resolve to explicit fields only through a qualified,
versioned mapping. Results record the resolved values, not only the preset.

## Outputs and execution

The caller specifies final state, snapshots, histories, derived fields,
gradients, sensitivities, or application metrics. Output residency and
materialization are part of backend eligibility and performance timing.

Execution policy contains preferences rather than mathematics: `auto` or an
explicit backend, device/resource constraints, preparation permission,
differentiability, reproducibility, and residency. Automatic selection may
choose only among backends implementing the fully resolved scientific
signature. See `BACKEND_SELECTION_CONTRACT.md`.

## Resolution lifecycle

1. Parse user-facing names and values.
2. Resolve presets, units, defaults, and equation-specific branches.
3. Validate mathematics and physical admissibility.
4. Construct coefficients, closures, and operators reproducibly.
5. Determine eligible backends from the complete signature.
6. Select and prepare a plan using qualified evidence.
7. Execute without silent semantic changes or hidden transfers.
8. Validate outputs and return diagnostics and provenance.

The resolved problem is immutable for an execution. Cached plans and artifacts
reference the problem identity from which they were created.

## Current slice mapping

The current `gradflow.Solver` implements one resolved branch: ideal-gas 3-D
compressible Euler, Cartesian caller state, duplicated periodic endpoints,
Roe-characteristic finite-difference WENO-JS orders 5--15, the preserved
global-LF/SSP-RK3 policies, eager PyTorch, and tightly bounded optional DVEB.

This shows that a vertical dependency chain can work. It does not prove that
the proposed general model is correct. New Euler boundaries and compressible
Navier--Stokes should provide the concrete uses from which stable shared
interfaces are extracted.
