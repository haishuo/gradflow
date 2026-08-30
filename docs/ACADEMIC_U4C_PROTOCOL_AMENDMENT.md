# Academic U4-C protocol amendment: byte-identical performance input

Status: **frozen before any U4-C C2 result is admitted**.

Date: 2026-08-30 (UTC)

## Trigger

The first attempted C2 campaign passed at `N=8192` but excluded the OpenSBLI
lanes at larger sizes. Inspection showed that the two implementations had
independently evaluated the same analytic initial-condition expression. Their
input values therefore differed at floating-point roundoff level. A spatial
derivative scales that input discrepancy by `1/dx`, making the comparison
increasingly measure transcendental-library and coordinate-evaluation
differences rather than the WENO operators named by U4-C.

The attempted campaign is invalid and is not retained as performance evidence.
No result from it may be quoted. The discovery is retained in this amendment.

## Correction

For every frozen size, the parent harness now creates one float64 CPU byte
array for the already-frozen analytic state. That exact array is:

1. the input to the canonical GradFlow CPU residual;
2. loaded by each GradFlow CPU and CUDA worker; and
3. loaded into the exact interior slab `[0,N)` through the public
   `ops_dat_set_data_slab_memspace` interface by each OpenSBLI OPS sequential
   and CUDA process. The existing generated periodic halo exchange then fills
   halo storage from that byte-identical interior before every admitted
   evaluation.

The interior-slab API is intentional. OPS's whole-dataset setter includes
halo storage for this generated data set and therefore does not represent the
frozen `N`-value input buffer.

Each state is retained and hashed. All qualifications and all timing cells are
restarted from empty evidence. Sizes, mathematics, lanes, warmups, sample
counts, worker counts, randomization seed, thresholds, and interpretation
rules remain unchanged.

This amendment removes an unintended input mismatch; it does not tune either
operator or alter the frozen performance state.

## Sequential OPS thread pin

The first admitted CPU worker exposed that the OPS target named `seq` is linked
through an OpenMP-capable generated translation and otherwise inherits the
machine-wide OpenMP thread default. That is not the frozen sequential CPU lane
and produced discontinuous thread-launch overhead. Every restarted process now
sets `OMP_NUM_THREADS=1` and `OMP_DYNAMIC=FALSE`. This matches the protocol's
sequential-OPS lane and GradFlow's already-frozen one-thread CPU policy; it is
not a post-hoc thread-count search. No observation made without this pin is
retained.
