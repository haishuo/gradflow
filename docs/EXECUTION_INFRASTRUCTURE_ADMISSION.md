# Execution-infrastructure admission contract

Status: active for all GradFlow experiments frozen on or after 2026-08-28.

## Why this correction exists

Forge has a working NVIDIA GeForce RTX 5070 Ti, but the default Codex process
sandbox does not expose the NVIDIA device files. A process-local
`torch.cuda.is_available() == False` observation was therefore previously easy
to misread as a statement about the host. It is not one.

Historical records remain immutable and accurately describe their original
processes. Linked CUDA supplements contain the later host-visible evidence.
This contract changes future infrastructure language and admission records; it
does not change any previously recorded numerical result.

## Two observations, never one overloaded flag

Every accelerator stratum records these separately:

1. **Host inventory:** whether the physical host is independently confirmed to
   contain the requested accelerator and driver stack.
2. **Execution-context visibility:** whether the process collecting the result
   can enumerate and use that accelerator.

Process-local library discovery is evidence only for item 2. Host inventory
must come from a host-level probe or a linked, hashed result collected in a
context that had device access. A host is not declared accelerator-free merely
because a container, scheduler allocation, remote worker, or sandbox hides it.

## Required status vocabulary

Future machine-readable records use one of these statuses:

- `admitted`: visible in the measurement process and all phase-specific
  correctness gates passed;
- `visible_unqualified`: visible, but admission has not yet run;
- `visible_admission_failed`: visible, but a correctness, compiler, residency,
  or environment gate failed;
- `process_hidden_host_present`: not visible to this process, while independent
  host inventory confirms the device exists;
- `host_confirmed_absent`: a host-level inventory confirms that the device does
  not exist on the selected host;
- `not_visible_host_unknown`: not visible to this process and no authoritative
  host inventory is available; or
- `probe_failed`: visibility or inventory could not be determined because the
  probe itself failed.

The bare labels `CUDA unavailable`, `GPU unavailable`, and
`untested_unavailable` are prohibited in new records because they conflate at
least three materially different conditions. Historical schemas that used
those labels are not rewritten.

## Admission sequence

Before a CUDA result is skipped or measured:

1. record the selected host and execution-context identity;
2. obtain or link host inventory evidence;
3. probe CUDA visibility in the exact process class that would run the study;
4. if the host is present but the process is isolated, use an explicitly
   authorized device-visible context rather than concluding the hardware is
   absent;
5. run the phase-specific CPU/CUDA, eager/compiled, dtype, graph, and residency
   gates; and
6. collect timing only after status becomes `admitted`.

If policy or permissions prevent the device-visible run, record
`process_hidden_host_present` and stop that stratum. Do not simulate CUDA and do
not substitute a different host silently.

## Current Forge evidence

The host inventory for Forge is established by the immutable Phase-4R CUDA
supplement and deferred CUDA qualification:

- NVIDIA GeForce RTX 5070 Ti;
- driver 580.173.02;
- PyTorch 2.13.0+cu130 and CUDA runtime 13.0 in the admitted environment; and
- successful resident CUDA correctness and compiler gates.

The linked records are `docs/FD_FV_PHASE_4_CUDA_RESULTS.md` and
`docs/DEFERRED_CUDA_GATES_RESULTS.md`. A future study must still perform its
own formulation-specific admission; host presence is not numerical admission.

Apple MPS on Forge is `host_confirmed_absent`, because Forge is not Apple
hardware. MPS is never simulated.
