# Academic U4-E prospective DVEB requalification

U4-E is governed by `docs/ACADEMIC_U4E_PROTOCOL.md`.  It prospectively replaces
only U4-D's original DVEB artifact with the immutable Trunk 005 scheduling
handoff, then requalifies and reruns DVEB, OpenSBLI/OPS, and the repository's
PyTorch/TorchInductor backend on the unchanged scalar float64 WENO-JS5
`N=8192` contract.

The frozen evidence key `gradflow` is a legacy alias for that PyTorch backend;
it does not identify the GradFlow system as a whole. See
`docs/BACKEND_IDENTITY.md`.

No U4-E comparative timing existed when the protocol was frozen.  The DVEB
repository and handoff are read-only; the external decision endpoint uses the
artifact's automatic policies exactly as selected.

E1 correctness qualification is complete: all six lanes passed, and the DVEB
artifact selected the predicted automatic schedules without an override.  See
`docs/ACADEMIC_U4E_E1_RESULTS.md` and run
`python3 experiments/academic_u4e/verify_qualification.py` for offline
verification.  E1 contains no comparative performance claim.

E2/E3 are complete.  Under the frozen resident decision endpoint, DVEB is the
resolved winner on both one-thread CPU and CUDA.  The separately reported
transfer-inclusive and prepared-launch endpoints are descriptive only.  See
`docs/ACADEMIC_U4E_RESULTS.md` and run
`python3 experiments/academic_u4e/verify_campaign.py` for offline verification.
This result tells GradFlow to prefer its DVEB backend for the admitted cell; it
does not mean that DVEB “beat GradFlow.”
The final CUDA-visible GradFlow regression passed with 355 tests and 12
expected external-artifact skips.
