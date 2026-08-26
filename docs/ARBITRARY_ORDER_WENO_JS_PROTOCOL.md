# Arbitrary-order finite-difference WENO-JS seed protocol

Status: **frozen before implementation or qualification**.

Date: 2026-08-26

## Question

Can one readable, differentiable ordinary-PyTorch implementation construct and
execute odd-order finite-difference Jiang--Shu WENO without order-specific
scientific code, while reproducing GradFlow's validated WENO-5 seed and
remaining compilable through TorchInductor?

This trunk establishes the scalar periodic mathematical constructor. It does
not add Navier--Stokes, general boundaries, characteristic reconstruction,
automatic backend placement, DVEB lowering, or a performance campaign.

## Mathematical contract

For design order `p = 2r - 1`, the constructor shall use exact rational
arithmetic to generate:

- the `r` candidate reconstructions at `x_{i+1/2}` from `r` consecutive
  finite-difference flux values;
- the `2r-1` point optimal reconstruction;
- the unique optimal linear weights expressing the latter as the former;
- the Jiang--Shu derivative-integral smoothness matrix for every candidate;
  and
- an exact positive-semidefinite factorization used for stable floating-point
  evaluation.

The reconstruction coefficients are obtained by treating nodal physical-flux
values as cell averages of the auxiliary finite-difference numerical flux,
interpolating that auxiliary polynomial, and evaluating it at the interface.
The smoothness indicator is

```text
sum(l=1..r-1) integral(-1/2..1/2) (d^l p/dx^l)^2 dx
```

on a unit grid. GradFlow multiplies every candidate indicator by 12 and uses
nonlinear power 2 so order five retains the historical Gottlieb/Jiang--Shu
scaling. The default epsilon remains `1e-29`. Weight normalization must avoid
overflow in float32 when all indicators vanish.

Left reconstruction uses candidate offsets
`[-r+1+k, ..., k]`, `k=0,...,r-1`. Right reconstruction is its exact mirror
about `x_{i+1/2}`. The scalar RHS uses global Lax--Friedrichs splitting and a
conservative interface-flux difference. Periodic unique nodes are the only
boundary convention qualified here.

## Public surface

The bounded initial surface is:

```python
scheme = gradflow.WENOJS(order=11)
left = scheme.reconstruct(values, bias="left", axis=-1)
rhs = scheme.rhs(u, dx, flux, flux_derivative, alpha=...)
```

Any leading tensor dimensions are batch dimensions. `axis` selects the
periodic spatial dimension. Inputs remain on their existing device. Float32
and float64 are legal; no implicit conversion, scalar extraction, host/device
transfer, custom operator, handwritten CUDA, or Triton is permitted.

The constructor may mathematically construct any odd order of at least three.
The first qualified set is `{5,7,9,11,13,15}`. Constructible is not synonymous
with qualified beyond that set.

## Frozen acceptance gate

1. Exact order-five candidate coefficients, optimal weights, full-stencil
   coefficients, and smoothness matrices equal independently written known
   Jiang--Shu values.
2. For every qualified order, exact candidate reconstruction reproduces
   polynomials through degree `r-1`; the optimal reconstruction reproduces
   polynomials through degree `2r-2`; optimal weights are positive and sum to
   one; smoothness matrices are symmetric positive semidefinite with constants
   in their nullspace.
3. Generated order-five scalar RHS agrees with the existing canonical direct
   WENO-5 seed within `5e-13` absolute in float64 for positive linear,
   negative linear, and Burgers fluxes on deterministic nontrivial states.
   The committed Gottlieb MATLAB oracle remains covered by the unchanged seed.
4. Smooth periodic linear-advection refinement is measured for every qualified
   order. Before the float64 error floor, at least one consecutive refinement
   pair must exhibit rate `p-1` or better, and refinement may not materially
   reverse. Exact polynomial reproduction remains the primary design-order
   proof when floating-point conditioning reaches the floor.
5. A smooth critical-point family is recorded separately. WENO-JS accuracy
   loss at critical points is a mathematical result to report, not conceal or
   retune after observation. Results must remain finite and converge under
   refinement.
6. Periodic conservation holds to a declared roundoff-scaled bound for all
   qualified orders.
7. Float64 autograd passes finite-gradient and double-precision `gradcheck`
   gates for representative order 5, 11, and 15 reconstructions/RHS calls.
8. Eager CPU execution passes for every qualified order. CUDA float32/float64
   agrees with CPU under declared dtype-specific tolerances when CUDA is
   available. Consumer-GPU float64 timing is not measured or interpreted.
9. `torch.compile(fullgraph=True, dynamic=False)` executes order 5, 11, and 15
   on CPU and CUDA where available, with graph behavior recorded. Compilation
   time and runtime are not benchmarked.
10. Static inspection and profiler evidence show no hidden `.cpu()`, `.cuda()`,
    `.to()`, `.item()`, NumPy conversion, or device transfer in the numerical
    path.

## Stop and claim boundary

Stop after the generator, PyTorch scalar implementation, frozen gate, result
record, coherent local commits, and clean working tree exist. Do not start a
representation bakeoff, DVEB extension, automatic selector, general boundary
API, Euler migration, or WENO-15 performance campaign in this trunk. Do not
push without explicit authorization.

Passing this gate would establish a reusable arbitrary-order scalar WENO-JS
foundation. It would not establish general CFD, aerospace readiness,
arbitrary-order characteristic systems, performance superiority, novelty, or
publishability.
