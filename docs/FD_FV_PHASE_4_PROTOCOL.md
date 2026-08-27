# FD/FV Phase-4 scalar matched bakeoff protocol

Status: frozen before Phase-4 multidimensional qualification or timing.

Freeze date: 2026-08-27 UTC.

## Purpose and bounded claim

Phase 4 measures the first accuracy-to-time and accuracy-to-memory comparison
between GradFlow's qualified classical finite-difference WENO-JS5 path and its
qualified dimension-by-dimension finite-volume WENO-JS5 seed. It is a
structured, periodic, scalar linear-advection experiment. It cannot establish
a universal FD/FV winner, nonlinear-shock performance, Euler behavior,
genuinely multidimensional FV performance, production CFD capability, or GPU
performance when CUDA is unavailable.

Correctness > performance > convenience is binding. Phase 4 has two ordered
stages:

1. **Phase 4A admission:** multidimensional correctness, convergence,
   conservation, and compiler parity, with no timing.
2. **Phase 4B measurement:** timing and memory only if the immutable Phase-4A
   record passes and verifies.

If Phase 4A fails, Phase 4 stops before any performance measurement.

## Registered formulations

The matched-component lane compares:

- `fd_classical_js5_global_lf_periodic_v1`: persistent point values,
  classical conservative split-physical-flux reconstruction through
  `WENOJS(5).rhs`, one axis at a time;
- `fv_dimensional_js5_global_lf_periodic_v1`: persistent physical cell
  averages, left/right state reconstruction, global LF/Rusanov face flux, and
  conservative face-flux difference through `fv_weno5_rhs`, one axis at a
  time.

Both use the same exact-generated WENO-JS5 coefficient and smoothness tables,
`epsilon=1e-29`, smoothness scaling `12`, nonlinear power `2`, explicit
directional `alpha=abs(c_k)`, float64, periodic unique grids, and SSP-RK3.
Their discrete states and exact projections remain different.

No AFD, genuinely multidimensional FV, conv1d, DVEB, Fortran, custom kernel,
mixed precision, or alternative epsilon policy participates. None is currently
a mathematically matched, qualified scalar implementation of both registered
formulations.

## Continuous problem and projections

For dimension `d in {1,2,3}`, solve on `[0,1)^d`

```text
u_t + sum_k c_k u_xk = 0,       c_k = 1/d,
u(x,0) = 0.5 + (0.2/d) * sum_k sin(2*pi*(x_k-phi_k)),
phi = (0.07, 0.19, 0.31).
```

The exact solution translates coordinate `k` by `c_k*t`. FD is initialized
and evaluated at `x_j=j/N`. FV is initialized and evaluated with the analytic
cell average over `[j/N,(j+1)/N]` in every direction. No point/average
conversion is hidden in either solve or its timing.

All grids are isotropic and Cartesian with `dx=1/N`. Use

```text
final_time = 0.01
nominal_dt = 0.2 * dx**(5/3) / sum_k(abs(c_k))
steps = ceil(final_time/nominal_dt)
dt = final_time/steps
```

The fixed `dt` avoids a different final-step branch and makes third-order
temporal error asymptotically commensurate with fifth-order spatial error.

## Phase-4A admission matrix

The frozen sizes are:

| Dimension | N values |
|---:|---|
| 1 | 24, 36, 54, 81 |
| 2 | 12, 18, 27, 40 |
| 3 | 8, 12, 18, 27 |

For every method, dimension, and size, eager float64 execution must produce a
finite result. L1 and L2 errors against the method-appropriate exact projection
must decrease on every refinement, and at least one consecutive L2 rate in
each method/dimension sequence must be at least `4.0`. Mass must satisfy

```text
abs(dx**d * sum(final-initial))
<= 64*eps*dx**d*sum(abs(initial)) + 2e-15.
```

At the largest size in each dimension, one SSP-RK3 step compiled with
`torch.compile(fullgraph=True, dynamic=False)` must have one graph, zero graph
breaks, and eager agreement at `rtol=0, atol=2e-11`. The compiled result must
preserve shape, CPU device, and float64 dtype. Source inspection must retain
the Phase-3R no-device-transfer finding.

CUDA is a separate admission stratum. If visible, the same largest-size step
must pass CPU/CUDA and eager/compiled parity at the existing float64 tolerance
before any CUDA timing. If unavailable, CUDA Phase 4 is
`untested_unavailable`; CPU admission may still pass. MPS is recorded but is
not simulated.

The immutable Phase-4A record contains source/protocol/Phase-3R hashes,
environment identity, errors, rates, conservation, graph evidence, and gate
decisions. It records `performance_measurements_collected=false` and must be
committed and independently verified before Phase 4B begins.

## Phase-4B measurement matrix

Phase 4B uses exactly the admitted Phase-4A method/dimension/size cells. A
timing cell is eligible only when its Phase-4A sequence and relevant device
stratum passed. Numerical output is checked again in the measurement worker;
failed parity makes its timings ineligible rather than silently usable.

### CPU controls

The local CPU stratum uses six PyTorch intra-operation threads (the six
physical cores of the recorded Ryzen 5 7600X) and one inter-operation thread.
The worker records visible logical CPUs and process affinity. No affinity or
frequency privilege is assumed. Each method/dimension/size runs in an isolated
process with its own temporary TorchInductor cache.

For eager and compiled execution separately:

- complete solve: one untimed warmup, then five repetitions;
- one SSP-RK3 step: five untimed warmups, then thirty repetitions;
- report all samples, median, mean, minimum, maximum, Q1, and Q3.

CPU calls are synchronous. Timing uses `time.perf_counter_ns`. The compiled
first complete solve is recorded separately and includes that worker's first
Inductor compilation, but excludes interpreter/package import and is named
`first_compiled_solve`, not full cold latency.

At the largest size in each dimension, a separate cold worker is timed by the
orchestrator from subprocess launch through a host-visible error/conservation
answer and process exit. It includes Python/PyTorch import, state construction,
compilation, solve, and output serialization. There are six such cold points,
one per method and dimension, each run once as frozen pilot evidence.

No AOT artifact exists for this new formulation pair. The prepared/AOT
endpoint is recorded as `not_implemented`, not approximated by a warm cache.

### CUDA controls

If CUDA passed Phase 4A, the same matrix runs in isolated processes. Inputs are
created before device-resident timing, CUDA events bracket warm solve/step
samples, and synchronization precedes and follows each measured group. CUDA
uses five warmups and thirty repetitions for steps, one warmup and five for
complete solves. Peak allocated and reserved memory are reset and recorded.
Transfer-inclusive cold workers start from CPU state and return the final
answer to CPU. Device name, capability, driver/runtime, total memory, and FP64
hardware context are recorded.

CUDA unavailable in the execution environment means no CUDA timing record and
no GPU conclusion.

## Memory and outcomes

Every record includes logical cells, persistent degrees of freedom, persistent
state bytes, final L1/L2 error, conservation, and solve/step timing. Isolated
CPU workers record absolute process peak RSS from `resource.getrusage`; because
this includes Python, PyTorch, and compiler state, it is reported alongside,
not substituted for, persistent tensor bytes. CUDA records PyTorch peak
allocated/reserved bytes when available. Compiler-cache bytes are recorded for
compiled workers.

Primary reported views are:

- achieved L1/L2 error versus warm complete-solve median;
- achieved L1/L2 error versus cold complete-solve wall time at representative
  largest sizes;
- achieved error versus peak process/device memory; and
- the fastest qualified eager-or-compiled warm endpoint for each method.

Equal-grid solve/step ratios are secondary causal diagnostics. A ratio within
5% is `unresolved_within_5_percent`; otherwise the faster observed endpoint is
named for that cell. No interpolation is used to manufacture an equal-error
crossover from four points; Pareto dominance and bracketing are reported only
when directly supported.

## Matched and best-practical lanes

The matched-component lane reports eager-versus-eager and compiled-versus-
compiled at each equal grid, while keeping the different point/cell-average
projections explicit.

The bounded best-practical lane selects the lower qualified warm median of
eager or compiled independently for FD and FV at each cell. This is an
execution-policy comparison of the same ordinary-PyTorch scientific sources,
not a claim to represent the best possible FD or FV implementation in the
literature. The maturity audit must state that the FV wrapper is newer, both
share the reconstruction core, neither has received formulation-specific
low-level optimization, and external native solver ceilings are absent.

## Record, verifier, and stop condition

Raw isolated-worker records are preserved. The aggregate contains commands,
source and qualification identities, failures, environment, complete samples,
derived ratios, Pareto summaries, and explicit claim boundaries. SHA-256
manifests cover the aggregate and raw files. A verifier recomputes eligibility,
statistics, ratios, hashes, and the no-performance-before-admission ordering.

Stop after the CPU matrix and any available qualified CUDA matrix, cold pilot,
interpretation, tests, coherent local commits, and clean working tree. Do not
add nonlinear scalar problems, Euler, arbitrary-order FV, automatic selection,
new DVEB work, representation optimization, or publication claims. Do not push
Phase-4 commits without new explicit authorization.
