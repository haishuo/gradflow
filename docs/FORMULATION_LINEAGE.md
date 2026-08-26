# WENO formulation lineage

## Relationship at a glance

| Material | Role | Formulation and scope | Oracle status |
|---|---|---|---|
| Chi-Wang Shu original Fortran | Authentic ancestral Jiang--Shu FD-WENO implementation | Two-dimensional Euler system, characteristic reconstruction, `epsilon=1e-6`, per-family global LF speeds with a 10% enlargement | Lineage reference; not the scalar pointwise oracle |
| Sigal Gottlieb MATLAB | Scalar specialization selected for the seed | One-dimensional scalar correction-form WENO-5, global `max(abs(f'(u)))`, `epsilon=1e-29`, duplicated periodic endpoint | Authoritative scalar formulation and pointwise oracle fixture |
| Upstream GradFlow PyTorch translation at `4c861fd` | Readable literal translation | Scalar Gottlieb loops, extended arrays, indexed assignment, host scalar extraction | Correct oracle bridge; archived, not canonical |
| DVEB screened comparator | Compiler/performance evidence | Batched unique-node right-moving linear specialization using rolls; optional `conv1d` variant | Correct for the screened workload; not a general oracle |
| Canonical `src/gradflow/weno5.py` | Restarted WENO-5 seed | Direct shifts/slices and elementwise operations, general scalar flux, corrected two-sided LF split, explicit grid conventions | Must pass the bounded Gottlieb gate |
| Generated `src/gradflow/weno_js.py` | Arbitrary-order scalar seed | Exact-rational auxiliary-flux reconstruction, generated JS matrices, stable LDLT indicators, orders 5--15 qualified | Reproduces the canonical WENO-5 seed; higher orders use independent algebraic and convergence gates |
| Packaged 3-D Euler slice | Generated characteristic system seed | Shu Roe/global-LF/epsilon policies with exact generated WENO-JS reconstruction, duplicated endpoints, orders 5--15 qualified | Order five is forward-gated against the frozen bakeoff source; higher orders use an exact entropy-wave oracle |
| Old GradFlow package | Historical experiments | Premature symbolic/order-general surface and convolution-oriented coefficient ideas, with a loop-based active solver | Noncanonical; recoverable from history/archives |

## Shared scalar reconstruction algebra

For point values `u_j`, flux values `f_j=f(u_j)`, and a global
Lax--Friedrichs speed `alpha`, the Gottlieb specialization forms

```text
delta_u_j = u_{j+1} - u_j
delta_f_j = f_{j+1} - f_j
delta_plus_j  = (delta_f_j + alpha*delta_u_j)/2
delta_minus_j = (delta_f_j - alpha*delta_u_j)/2
```

At face `j+1/2`, a fourth-order central flux is corrected once from each split
difference family. For four directional values `h1,...,h4`, it sets
`t1=h1-h2`, `t2=h2-h3`, and `t3=h3-h4`. Its indicators are

```text
beta1 = 13*t1^2 + 3*(h1 - 3*h2)^2
beta2 = 13*t2^2 + 3*(h2 + h3)^2
beta3 = 13*t3^2 + 3*(3*h3 - h4)^2
```

These are 12 times the commonly printed Jiang--Shu indicators. The source
does not rescale epsilon to compensate: it uses `epsilon=1e-29` directly and
squares each `(epsilon + beta)`. The product-form weight algebra has linear
weight ratio `(1,6,3)`. The semidiscrete sign is
`rhs_j=(fh_{j-1}-fh_j)/dx`.

The Shu Fortran contains the same central-plus-correction structure and scaled
indicators, showing the direct ancestry. It differs materially in problem
class and policies: Euler characteristic projection, `epweno=1e-6`, and
per-characteristic global speeds enlarged by 10 percent. Those choices make
it unsuitable as a pointwise oracle for the scalar specialization despite the
shared reconstruction core.

## Full comparison of the two inherited PyTorch implementations

The upstream file is recoverable at
`archive/pre-refoundation-2026-08-25:tests/gottlieb_weno5_pytorch.py`. The
screened file is preserved exactly under `baselines/dveb_screened_pytorch/`.

| Dimension | Upstream GradFlow translation | DVEB screened comparator |
|---|---|---|
| Exact formulation | Implements the general scalar Gottlieb algebra, including both split directions | Matches it for the screened right-moving linear case; not exactly general because the negative-family sign is reversed after defining `gm=gp-df` |
| Grid | One 1-D vector including both periodic endpoints; the endpoints may hold distinct discontinuity traces | Tensor shape `(nb,n)` with unique samples `x_j=j*dx`, `j=0,...,n-1` |
| Endpoint/boundary convention | Concatenates four samples before the left endpoint and five after the right, excluding the endpoint values from the wrap copies exactly as MATLAB does | `torch.roll` circular wrap; optional convolution variant uses circular padding |
| Flux | Callable scalar `f(u)` | Fixed linear `f=a*u` |
| LF alpha | Computes global `max(abs(fp(u)))` and extracts it with `.item()` | Receives `em` as an argument; the screen passes `abs(a)` |
| Epsilon/scaling | `1e-29`; 12-times-scaled indicators; `(epsilon+beta)^2` | Same |
| Reconstruction | Central flux plus positive and reversed/negated negative corrections | Same central and positive correction; negative correction has the scope limitation above |
| Time integration | Driver uses SSP-RK3, 75 steps, `dt=0.5*dx` | SSP-RK3 helper; screen uses `CFL=0.4` and several fixed-step protocols |
| Dtype/device | Defaults to float64 but coerces/copies input with `.to(dtype,device)`; alpha `.item()` synchronizes to host; test-grid `dx` goes through CPU NumPy | Screen contract is resident float64 CUDA, with batched tensors and no in-loop host access; scalar `a`, `em`, `dx`, and `dt` are arguments |
| Tensor representation | Python loops, scalar tensor indexing, indexed assignments, `torch.cat`, and temporary zero tensors; no `conv1d` in the RHS | Canonical screen path uses rolls, slicing, and elementwise operations; a separate comparison variant uses `conv1d` for two linear stencil pieces |
| `torch.compile` behavior | On PyTorch 2.13 with default scalar capture disabled, empirical `torch._dynamo.explain` at `n=8` produced 2 graphs and 1 break at `.item()`; fixed-size loops can be unrolled, but create a large shape-specialized graph. The debug file path is outside the numerical contract | Screened fullgraph dynamic path: one graph, zero breaks, zero recompiles across 12 shapes. Eager and `conv1d` variants are valid but were not the selected compiled result |
| Correctness evidence | Prior GradFlow records say the 75-step result matches the committed MATLAB HDF5 fixture within `1e-12`; direct refoundation comparisons agree to about `1e-14` per RHS | DVEB correctness record passes its `n=400` oracle, analytic bounds, cross-implementation RHS checks, 50-step states, and selected full-period checks for the positive linear workload |
| Generalizability | Mathematical scope is wider, but loop/index bookkeeping, allocations, coercions, and debug/test concerns obscure the reconstruction and compiler path | Direct tensor structure is much easier to extend and compile, provided benchmark-specific constants and the negative-split sign are not mistaken for a general API |

### Sign-scope finding

DVEB names `gm = gp - df`, which is `-delta_minus` under Gottlieb's notation.
Its negative correction then supplies `-gm`, introducing the opposite sign.
For `a=alpha>0`, `gm` is exactly zero, so all screened checks pass and cannot
observe the issue. Refoundation probes found machine-scale agreement for
`a=1`, but large disagreement for `a=-1` and nonlinear Burgers flux. Replacing
that path with an explicit `delta_minus=(delta_f-alpha*delta_u)/2` and the
Gottlieb reversed/negated stencil restores approximately `1e-14` agreement in
both probes.

## Canonical decision

The canonical seed takes the DVEB comparator's direct tensor organization,
because it is concise, device-resident, differentiable, empirically captured
by TorchInductor, and easier to generalize. It takes the actual split algebra,
epsilon, endpoint oracle convention, and SSP-RK3 coefficients from Gottlieb.
The public `weno5_rhs` uses unique periodic nodes; the separately named
`weno5_rhs_gottlieb_periodic` exists only for the duplicated-endpoint oracle.

No input is silently converted to a device or dtype. The validated seed
requires float64, computes alpha on-device when a derivative is supplied, and
accepts an already compatible explicit alpha otherwise. The exact DVEB file
is preserved so this cleanup cannot rewrite what the screen actually tested.

## Packaged Euler solver relationship

`src/gradflow/euler3d.py` retains the direct 2-D/3-D characteristic Euler
formulation frozen in `experiments/shu_torch_ablation/shu_euler_torch.py`; the
public `Solver` admits its validated 3-D case at orders 5--15. The physical
flux, Roe projection, epsilon, LF enlargement, duplicated endpoints, CFL, and
SSP-RK3 policies are unchanged.

Order five is deliberately written in an algebraically refactored form. The
frozen source normalizes product-form weights
`q2*q3 : 6*q1*q3 : 3*q1*q2`. Dividing all three by the common product gives
the packaged inverse form `1/q1 : 6/q2 : 3/q3`. In perfectly smooth float32
regions, autograd through the product form first differentiates a reciprocal
near `1e-24`, overflows, and produces NaNs despite finite final weights. The
normalized inverse form yields finite gradients. The benchmark source remains
untouched, and generated order five is forward-gated against it rather than
declared bitwise identical.

For every face and order, the package freezes the Roe matrices, projects the
required positive and negative split-flux samples, applies the same exact
candidate and smoothness algebra qualified in the scalar trunk, and transforms
the reconstructed characteristic flux back to conserved variables. Orders
7--15 are independently gated on an analytic 3-D Euler entropy wave,
conservation, uniform-state preservation, differentiation, devices, and
compiler behavior. They are descendants of Shu's system policy, not scalar
Gottlieb oracles.

## Old GradFlow and generated arbitrary order

The old package exposed symbolic generation and multiple orders before those
claims had an adequate numerical gate. Its limited convolution-oriented
coefficient material remains under `legacy/conv1d/`; the full implementation
is in the pre-refoundation archive and Git history.

The restarted generator does not lengthen the WENO-5 correction stencil. For
every order `p=2r-1`, it independently reconstructs the auxiliary flux
polynomials from cell averages, solves for optimal weights, integrates the
Jiang--Shu derivative indicators, and factors their exact matrices. Generated
order five agrees with the canonical Gottlieb seed, while orders 7--15 are
qualified by exact polynomial reproduction and independent convergence gates.

The scalar generator retains Gottlieb's 12-times indicator scaling, epsilon
`1e-29`, and nonlinear power two. Its critical-point record demonstrates
expected WENO-JS accuracy loss rather than conflating a generated higher-order
stencil with uniform high-order behavior.

The characteristic migration reuses the exact candidates, weights, and
smoothness factors but deliberately supplies Shu's `1e-6` epsilon and system
flux policy. Generated order five matches the preserved characteristic
implementation within floating-point roundoff-scale bounds; the higher orders
extend that lineage under their separately frozen gate. The two epsilon and
problem policies therefore remain explicit rather than being conflated.
