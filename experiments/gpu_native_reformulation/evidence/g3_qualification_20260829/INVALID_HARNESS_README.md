# Invalid CFL 0.6 harness record

`qualification_invalid_harness_cfl_0p6.json` and
`arrays_invalid_harness_cfl_0p6/` preserve the first qualification execution,
which used CFL 0.6 in the Python step oracle instead of the pre-existing Shu/R6
CFL 0.1 policy. They are retained for auditability and are not candidate
evidence.

The JSON is the unmodified raw record. Its embedded archive paths therefore
retain the original `arrays/` names that existed at execution time; those exact
files were subsequently moved, without rewriting them, into
`arrays_invalid_harness_cfl_0p6/` before the corrected run created the valid
`arrays/` directory.
