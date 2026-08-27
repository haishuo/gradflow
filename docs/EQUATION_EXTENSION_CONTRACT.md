# Equation extension contract

## Intent

GradFlow's equation library must be extensible without turning every equation
or scenario into a separate solver. This is a design and promotion contract,
not a claim that a public equation plugin API exists today. Shared interfaces
should be extracted from real Euler and compressible Navier--Stokes work.

## Required scientific definition

An equation family must declare:

1. equations and modeling assumptions;
2. supported dimensions and coordinates;
3. conserved, primitive, and auxiliary variables;
4. parameter types, units, defaults, and admissible ranges;
5. state transformations;
6. inviscid fluxes, diffusive terms, and sources;
7. closures and equations of state;
8. admissibility conditions and failure behavior;
9. characteristic structure required by supported WENO policies;
10. timestep and stability restrictions; and
11. derived quantities available to results.

Every convention that can change a number must be explicit and testable, with
provenance identifying the formulation behind the implementation.

## Parameters and schemas

Parameters are equation-specific. Common concepts must not force physically
different systems through a lowest-common-denominator dictionary. The eventual
schema supports Python/UI validation, descriptions and units, conditional
fields, cross-field constraints, versioned serialization, and resolution from
friendly inputs to explicit numerical values.

A loose mapping may enter at the user boundary, but the canonical internal
representation is typed, resolved, and validated before numerical execution.

## Scenarios, boundaries, and user data

Named scenarios bind an equation to qualified parameters, domain, initial
data, and boundaries. They are workflows, not equations.

An extension identifies compatible boundary families and the state or flux
operations they require. A boundary is qualified only after closure,
ghost/endpoint, conservation, and convergence tests. Future user callables
must obey tensor, dtype, device, differentiability, and admissibility
contracts; convenience must not introduce Python calls or transfers inside a
qualified numerical loop.

## Discretization and backend compatibility

Compatibility records state componentwise/characteristic reconstruction,
eigensystem policy, flux splitting, qualified WENO families/orders/precision,
boundaries, and stabilization. Unsupported combinations are rejected; the
registry never silently substitutes another order, split, reconstruction, or
equation specialization.

The equation definition is backend-independent. A native artifact may
specialize it for speed, but that specialization belongs to backend eligibility
and artifact provenance. Native paths establish correctness parity before
becoming eligible.

## Promotion gate

A canonical equation family requires:

- reviewed formulation and provenance;
- tests for transforms, fluxes, closures, and admissibility;
- exact, manufactured, or independently generated references;
- smooth convergence and relevant nonsmooth stress cases;
- conservation and boundary tests;
- CPU/device/backend agreement at declared tolerances;
- differentiation evidence for each claimed gradient path;
- a bounded performance comparison for intended workloads;
- precise unsupported-combination errors; and
- examples using the same schema intended for product use.

Correctness completes before timing chooses implementations. Performance
qualification completes before convenience promotes a supported workflow.

## First architectural test: compressible Navier--Stokes

Compressible Navier--Stokes is the first major test because it reuses Euler's
convective structure while adding viscous stress and heat flux, transport
models, Reynolds/Prandtl parameters, additional derivatives and stability
limits, wall thermal/no-slip policies, and new reference problems.

It must not be forced through an Euler-only interface. Where it exposes a
false abstraction, the shared model is revised before more equations are
added.
