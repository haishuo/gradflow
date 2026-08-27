# GradFlow product vision

## One engine, not a family of benchmark programs

The product target is one GradFlow scientific engine. It must not become an
Euler GradFlow, a Navier--Stokes GradFlow, a Sod-shock GradFlow, and a separate
program for every new scenario.

A user selects from a qualified library of equation families and then supplies
the physical and numerical parameters meaningful for that family. Initial and
boundary conditions, domain or geometry, requested outputs, precision, and
accuracy policy form one validated problem description. The engine constructs
the numerical problem and selects an eligible execution path.

“Arbitrary PDE selection” means selection from an extensible catalog of
implemented and independently qualified equations. It does not currently mean
that an arbitrary symbolic PDE entered as text can be solved correctly. A
future developer extension interface may support new equations, but it carries
the same correctness obligations as built-in equations.

## Intended users

GradFlow should serve three levels without creating three engines:

- A guided user understands the physical application but need not know WENO
  implementation or GPU compiler details.
- An advanced scientific user chooses numerical policies, outputs, and
  accuracy controls.
- An expert or researcher can pin a backend, dtype, WENO family and order, or
  inspect the resolved execution plan.

An aerospace engineer should be able to describe an external-flow problem in
the language of the application. GradFlow should translate qualified choices
into numerical configuration, explain required physical inputs, and reject
incompatible combinations. It should not require the user to understand
kernel fusion, PCIe transfer costs, or backend saturation.

## Target Python experience

The following is a design target, not the current implemented API:

```python
import gradflow

problem = gradflow.Problem(
    equations="compressible_navier_stokes",
    dimension=3,
    geometry="wing.step",
    parameters={
        "mach": 0.82,
        "altitude": "35000 ft",
        "angle_of_attack": "3 deg",
    },
    initial_condition="freestream",
    boundaries="external_aerodynamics",
)

result = gradflow.solve(
    problem,
    final_time=...,
    accuracy="high",
    backend="auto",
)
```

Advanced controls may refine that request:

```python
result = gradflow.solve(
    problem,
    final_time=...,
    weno=("JS", 11),
    flux_split="global_lf",
    dtype=...,
    backend="cuda",
)
```

An explicit target is permission to use a legal target, not permission to
change the mathematics or skip validation. Automatic execution should be the
ordinary path; explicit controls remain available for reproducibility and
research.

## Guided application

A future application should be generated from the same equation and problem
schemas consumed by the Python API. It should offer:

1. **Guided mode** — equation-appropriate forms, units, qualified presets, and
   explanations in application language;
2. **Advanced mode** — numerical policies, mesh and stopping controls,
   precision, outputs, and backend preferences; and
3. **Show code** — valid Python that reproduces the resolved request.

The primary action may be presented as **Analyze**. Before execution, the
application validates physical admissibility, units, boundary data, resource
feasibility, and whether the request lies inside a qualified capability
envelope. After execution, it presents scientific results together with
accessible warnings and an inspectable provenance record.

The UI must not implement separate numerical logic, invent unsupported
defaults, or suppress a material warning to make a workflow appear simple.

## Product success criteria

A product capability is complete only when:

- its numbers pass the applicable correctness gate;
- at least one appropriate execution path is competitively qualified;
- the same problem can be expressed through the public Python model;
- automatic choices and fallbacks are inspectable and reproducible;
- guided input cannot create silently inconsistent configurations;
- errors state how to correct the request; and
- results contain enough provenance for a later audit or rerun.

The engine need not choose the GPU. Small problems may be faster on generated
or vectorized CPU code. Large resident three-dimensional problems may favor
CUDA. The product value is a correct decision based on evidence, not a promise
that every problem uses a particular device.

## Relationship to research

The research program determines how WENO constructions, representations,
compilers, native backends, precision, and device placement behave. The
product consumes only qualified outcomes.

The product must not force a favorable academic conclusion. Conversely, a
research result does not enter the product solely because it is interesting.
Promotion follows `ENGINEERING_CHARTER.md`: correctness, then performance,
then convenience.
