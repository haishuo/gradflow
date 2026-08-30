# Academic U4-D DVEB three-way study

U4-D is governed by `docs/ACADEMIC_U4D_PROTOCOL.md`. It adds the pinned DVEB
compiler as an internal implementation control to the exact scalar WENO-JS5
contract that OpenSBLI and GradFlow qualified in U4-C.

The DVEB repository remains unmodified. U4-D builds a pinned detached copy and
wraps compiler-generated CPU/CUDA launchers with a retained benchmark adapter.
No U4-D comparative timing existed when the protocol was frozen.

D1 correctness qualification is complete. All six DVEB, OpenSBLI, and GradFlow
CPU/CUDA lanes passed at the sole U4-C-admitted size, `N=8192`. See
`docs/ACADEMIC_U4D_D1_RESULTS.md` and run
`python experiments/academic_u4d/verify_qualification.py` to verify its frozen
evidence offline. D1 contains no comparative performance claim.

D2/D3 are also complete at `N=8192`. OpenSBLI is the resolved resident winner
on both CPU and CUDA; DVEB beats GradFlow on resident CUDA and is descriptively
fastest in the one-worker transfer and three-process prepared-launch endpoints.
See `docs/ACADEMIC_U4D_RESULTS.md` and run
`python experiments/academic_u4d/verify_campaign.py` for the complete results
and offline evidence verification.
