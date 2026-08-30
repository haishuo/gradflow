# Academic U4-B OpenSBLI qualification

U4-B is the correctness-only attempt governed by
`docs/ACADEMIC_U4B_PROTOCOL.md`. It may adapt OpenSBLI only within the frozen
boundary and may not report performance.

The frozen run qualified the adapted OpenSBLI operator. See
`docs/ACADEMIC_U4B_RESULTS.md` and
`evidence/u4b_20260830/qualification.json`.

The evidence can be checked offline from an installed GradFlow environment:

```bash
python experiments/academic_u4b/verify_u4b.py
```

Reproduction additionally requires clean checkouts of the pinned OpenSBLI and
OPS revisions, OpenSBLI's declared SymPy 1.1 dependency installed into an
isolated target directory, GNU C++, and HDF5. The exact generation, build, and
execution commands from the frozen run are in `COMMANDS.txt`. A representative
top-level invocation is:

```bash
PYTHONPATH=src python experiments/academic_u4b/run_u4b.py \
  --opensbli-root /path/to/opensbli \
  --ops-root /path/to/OPS \
  --sympy-root /path/to/isolated/sympy-1.1 \
  --work-root /tmp/gradflow-u4b-work \
  --evidence-dir experiments/academic_u4b/evidence/u4b_YYYYMMDD \
  --hdf5-root /path/to/hdf5/prefix
```

The adapter defines only the scalar eigensystem and experiment. OpenSBLI still
generates the smoothness indicators, nonlinear weights, reconstruction, native
LLF split, flux divergence, periodic halo exchange, and OPS execution. The
small retained patch generalizes existing Euler assumptions and exposes the JS
epsilon. A separately hashed instrumentation script writes the first native
residual and exits before an RK update.
