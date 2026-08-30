# Reproduction environments

`academic-a4-forge.json` records the machine and software that produced the
first academic release-candidate evidence. `academic-a4-core.txt` is the
minimal exact package inventory needed to interpret that evidence; it is not
presented as a universally installable lock file because the PyTorch build is
a dated development CUDA wheel.

The portable project requirements remain in `pyproject.toml`. Reproducing the
reported performance requires a compatible PyTorch/CUDA/compiler stack and
must record any substitution. Numerical reproduction on a newer supported
PyTorch version is welcome, but it is a new environment stratum rather than
the original Forge environment.

