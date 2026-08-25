# Original Jiang--Shu Fortran scaling experiment

This experiment answers a narrow engineering question: how far can the
authentic two-dimensional Euler WENO program be scaled after removing its
compile-time 200-by-200 array bounds? It does not make the modified program a
canonical reference and does not replace the scalar Gottlieb oracle used by
GradFlow.

## Frozen source and modified descendant

`original_snapshot/` is an exact, read-only-intent copy of the authoritative
files in `references/jiang_shu_fortran/`. The copies were verified on
2026-08-25:

| File | SHA-256 |
|---|---|
| `original_snapshot/weno.f` | `9f1231516ef92b496333475ef29bfbba23afe77423163e7797bc8775a50186c5` |
| `original_snapshot/comm.inc` | `efc977da6582767cfa20ef76b0c3a0ace83e64083ca78f161668124e4cdbe3a7` |

Do not edit those two files. `dynamic/` is explicitly a modified experimental
descendant. The authoritative copy and its redistribution caveat remain in
`references/README.md`.

The original limit comes from `comm.inc`:

```fortran
      parameter(md=3,mnm=4)
      parameter(nxd=200+md,nyd=200+md,nsd=200+md)
```

All state arrays were static members of COMMON blocks dimensioned with those
parameters. The descendant replaces the COMMON storage with a Fortran module
whose arrays are allocated from the input `nx` and `ny`. It also changes local
line-work arrays to runtime bounds.

The frozen file contains a separate fixed-form transcription problem. Seventeen
intended statements begin with `c` in column 1, so a fixed-form compiler treats
them as comments. They include initialization assignments and the calls to the
boundary condition, x/y flux sweeps, WENO reconstruction, CFL calculation, and
Runge--Kutta update. The unmodified file therefore compiles but does not execute
its intended solver. `dynamic/fixed_form_repairs.patch` moves only those
unambiguous statements into the fixed-form statement field. It is applied at
build time to both a static-array copy and the dynamic-array copy. It never
modifies `original_snapshot/`. The dynamic state module also initializes the
two CPU-timer placeholders whose Cray-specific `second()` calls are disabled,
so its diagnostic output does not read indeterminate values.

This produces four deliberately distinct binaries:

- `weno_original`: byte-for-byte source as received, including the column-1
  behavior and 200-point limit;
- `weno_storage_only`: dynamic storage but the same column-1 behavior, used to
  isolate and test the allocation change;
- `weno_original_repaired`: fixed-form repairs with the original static arrays;
- `weno_dynamic`: fixed-form repairs plus runtime-sized arrays, used for real
  WENO scaling runs.

Two opt-in environment controls support experiments without changing normal
program behavior:

- `WENO_WRITE_SOLUTION=0` suppresses the enormous `fort.8`, `fort.9`, and
  `restart` output files.
- `WENO_TOUCH_ALL=1` initializes every persistent allocation, forcing Linux to
  back every page with physical memory. This is a memory-capacity test, not a
  WENO time step.

## Build and regression

The tested compiler was GNU Fortran 13.3.0. Build both programs and compare a
one-step 10-by-10 run with:

```text
make
make regression
```

After building, reproduce a square-grid run with `./run_case.sh N`. Pass `0`
as the second argument and `1` as the third to run the full-residency capacity
test without a time step; for example, `./run_case.sh 12000 0 1`.

The storage-only dynamic output is byte-identical to the frozen original for
both `fort.8` and `fort.9`. Their respective SHA-256 hashes are:

```text
a858a117fa4017fcab0f8c488a865de7488b745ba51696d0198c6538eac88633  fort.8
9a650225d16615f1b66e15711339e665fe5786a4eaa97ec0a46b5ffdd4e761fc  fort.9
```

The repaired dynamic output is separately byte-identical to the repaired
static program. Its output hashes are:

```text
f0e087c6c1a68214669df5b44a647ff5119901a175d7408de250087f0247f2a8  fort.8
684fc654bbe835b5890c6761594b0569a15944f6477125fede4901175e5b88e7  fort.9
```

This isolates the storage refactor from the fixed-form repair and proves
preservation for the checked RK3 path. It is not a proof that every problem
option and RK4 path are equivalent.

## Memory model

For a square `N`-by-`N` grid, the persistent allocation contains approximately
20 single-precision reals per cell, or **80 N-squared bytes**, plus line-work
and ghost-cell terms of order N. The program prints the exact estimate used by
the experiment after allocation. This is substantially larger than the cells
actually made resident by a single step because Linux allocates pages lazily.

On the test host (AMD Ryzen 5 7600X, six physical cores, 64,974,184 KiB
installed RAM, Linux 6.8.0), the projected full-residency sizes are:

| Square grid | Cells | Persistent estimate |
|---:|---:|---:|
| 12,000 | 144,000,000 | 10.743 GiB |
| 16,000 | 256,000,000 | 19.093 GiB |
| 20,000 | 400,000,000 | 29.827 GiB |
| 24,000 | 576,000,000 | 42.944 GiB |
| 26,000 | 676,000,000 | about 50.4 GiB |
| 27,000 | 729,000,000 | about 54.3 GiB |

The simple all-RAM ceiling is roughly 28,000 per side, but the operating
system and other processes require memory too. It is not a safe usable limit.

## Results on 2026-08-25

All runs used `-O3 -march=native`, one process, and the original
single-threaded code. Solution files were disabled for scaling runs.

One complete three-stage SSP-RK3/WENO step of problem 1, using the repaired
dynamic binary, scaled as follows:

| Grid | Wall time | Persistent estimate |
|---:|---:|---:|
| 400 x 400 | 0.03 s | 0.012 GiB |
| 800 x 800 | 0.12 s | 0.049 GiB |
| 1,600 x 1,600 | 0.86 s | 0.193 GiB |
| 3,200 x 3,200 | 3.87 s | 0.767 GiB |
| 6,400 x 6,400 | 16.85 s | 3.060 GiB |
| 8,000 x 8,000 | 26.65 s | 4.778 GiB |
| 12,000 x 12,000 | 49.15 s | 10.743 GiB |
| 24,000 x 24,000 | 235.17 s | 42.944 GiB |

The 24,000 case completed one actual RK3/WENO step with a peak resident set of
36,029,032 KiB, no swap, and an accepted CFL step of
`1.22816173e-5`. A separate full-touch capacity test made all 42.944 GiB
resident, reached 45,033,216 KiB peak RSS, used no swap, and exited normally
in 34.49 seconds. Earlier full-touch points were:

| Grid | Peak RSS | Wall time |
|---:|---:|---:|
| 8,000 x 8,000 | 5,013,120 KiB | 3.79 s |
| 12,000 x 12,000 | 11,268,224 KiB | 8.51 s |
| 16,000 x 16,000 | 20,023,040 KiB | 15.21 s |
| 20,000 x 20,000 | 31,278,208 KiB | 23.67 s |
| 24,000 x 24,000 | 45,033,216 KiB | 34.49 s |

Therefore **24,000 x 24,000 is the largest safely demonstrated grid on this
host**, both for a real WENO step and for full physical commitment. This is
576 million cells: 120 times the original limit in each dimension and 14,400
times its 200-by-200 cell count.
The source-imposed ceiling is gone. The next constraint is RAM, followed by
quadratic elapsed time. Testing 26,000--28,000 by forcing every page resident
was deliberately avoided because it would leave too little headroom and could
invoke the system-wide out-of-memory killer.

The frozen and storage-only binaries report divide-by-zero because their CFL
call is one of the column-1 statements treated as a comment. The repaired
binaries remove that serious exception. GNU Fortran still reports underflow
and denormal flags in the WENO weight algebra on this smooth-vortex problem;
the outputs contain neither NaN nor infinity. That inherited behavior should
be characterized before treating the modified program as a production solver.
The scaling result claims only that the allocation limit and fixed-form defect
were isolated, the repaired static and dynamic paths agree for the checked
case, and a genuine 24,000-square-grid step completed.
