# G2 Frozen-U0 Damage Protocol

Status: frozen after G1 hashing and before the first U0/oracle comparison.

## Question

How much numerical damage did the deliberately reckless U0 formulation incur
on the preserved three-dimensional periodic isentropic vortex?

G2 diagnoses the frozen U0 artifact. It cannot modify, rebuild, retune, or
replace U0.

## Input and oracle

- Input: exact `N=32` component-major FP32 state frozen by the U0 executable.
- Oracle spatial operator: qualified GradFlow characteristic finite-difference
  JS-WENO-5 in float64, derived from and previously checked against the
  preserved Shu formulation.
- Oracle boundary convention: duplicated periodic endpoints, reduced back to
  unique cells for comparison.
- CFL: Shu sum-of-directional-speeds policy at 0.1.

The frozen FP32 input is promoted exactly to float64 before oracle execution;
the oracle does not regenerate the initial condition.

## Comparisons

For one step, compare U0 to both:

1. characteristic WENO-5 plus Forward Euler using U0's frozen timestep, which
   isolates the aggregate spatial-formulation difference as closely as this
   experiment permits;
2. the fully qualified characteristic WENO-5 plus SSP-RK3 method using the
   qualified CFL timestep, which measures end-method displacement.

For ten steps, compare U0 against independently CFL-recomputed Forward Euler
and SSP-RK3 trajectories. This is an accumulated aggregate comparison; it does
not isolate timestep-rounding effects because U0's complete timestep sequence
was not recorded.

## Metrics fixed in advance

For every comparison record:

- maximum, mean absolute, and RMS state error;
- RMS error relative to RMS state magnitude;
- RMS error relative to the oracle state change from the shared initial state;
- componentwise maximum and RMS errors;
- U0 and oracle minimum density and pressure;
- finiteness;
- componentwise conservation drift, normalized per cell;
- oracle timestep(s) and U0's frozen final timestep.

No pass threshold is imposed: U0 was designed to be unsafe. G2 reports damage
rather than qualifying or rejecting a backend.

## Interpretation boundary

Because U0 changed several numerical policies simultaneously, G2 may establish
the total cost of those changes but cannot assign causality to one change.
That requires the subsequent one-change-at-a-time recovery ladder.
