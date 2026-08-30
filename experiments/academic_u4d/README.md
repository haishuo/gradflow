# Academic U4-D DVEB three-way study

U4-D is governed by `docs/ACADEMIC_U4D_PROTOCOL.md`. It adds the pinned DVEB
compiler as an internal implementation control to the exact scalar WENO-JS5
contract that OpenSBLI and GradFlow qualified in U4-C.

The DVEB repository remains unmodified. U4-D builds a pinned detached copy and
wraps compiler-generated CPU/CUDA launchers with a retained benchmark adapter.
No U4-D comparative timing existed when the protocol was frozen.
