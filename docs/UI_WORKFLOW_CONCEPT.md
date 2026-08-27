# Guided UI workflow concept

## Product role

The future GradFlow application presents the same validated engine as the
Python API. It is not a separate solver and does not reproduce numerical logic
in a front end. It lets an application-domain user configure, run, understand,
and reproduce a qualified analysis without expertise in WENO implementation,
GPU placement, or compiler behavior.

## Three views of one problem

### Guided

Guided mode starts from the physical task:

1. choose an equation/application family;
2. choose a qualified scenario or geometry workflow;
3. enter equation-appropriate physical parameters with units;
4. define compatible initial and boundary conditions;
5. select desired accuracy and outputs; and
6. review validation and resource estimates before **Analyze**.

Forms are generated from equation, boundary, and scenario schemas. They do not
present irrelevant fields or accept incompatible combinations.

### Advanced

Advanced mode exposes qualified mesh, WENO, flux, time, CFL, dtype, backend,
and result controls. It identifies defaults and warns when a legal override is
outside performance qualification. It never bypasses correctness or physical
validation.

### Show code

Show code displays Python for the fully resolved problem and solve request,
including defaults and inferences that materially affect the mathematics.
Running it in a compatible environment constructs the same problem identity.

## Validation and execution

Before expensive work, the application explains what is invalid, why, which
qualified alternatives exist, and whether the limitation belongs to physics,
boundaries, numerics, hardware, or the current implementation. Unsupported
capability and invalid physics are distinct; neither is silently repaired by
changing the selected method.

The ordinary backend is `auto`; detailed placement remains inspectable.
Progress uses physical time, steps, output events, and meaningful resource
state. Preparation is not displayed as simulation progress, and runtime
preparation counts toward end-to-end elapsed time. Cancellation, invalid-state
termination, memory exhaustion, and partial results have explicit policies.

## Results

The initial view answers the application question with qualified fields,
plots, and metrics. It also exposes warnings, units, resolved settings,
backend/environment diagnostics, downloadable state and provenance,
rerun/modify controls, and generated Python.

Visualizations present engine results. Derived display values use registered,
tested transforms rather than ad hoc front-end formulas.

## Safety, honesty, and sequencing

The interface cannot make an unsupported aerospace analysis valid. Geometry,
turbulence, boundary, material, and model assumptions remain visible where
needed for responsible interpretation.

The intended experience is **physics-aware and numerics-agnostic**: the user
understands the physical question, while GradFlow exposes, validates, executes,
and records the numerical choices.

UI implementation begins after the shared problem schema has at least two real
equation uses and a stable result model. Early prototypes may test terminology
but cannot create a competing schema or be advertised as qualified analysis.
This follows the charter: usability is designed early, while convenience is
promoted only after correctness and performance pass.
