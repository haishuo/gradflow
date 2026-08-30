# G6 compiler-resource record correction

The immutable `gate/forward_gate.json` reports 24-byte stack, spill-store, and
spill-load values for the three `r96` face candidates. Those values belong to
the compiled but unused `pencil_kernel`. The original parser allowed the next
kernel's name to match the preceding function-properties section.

The preserved compiler logs and CUDA runtime attributes give the correct
`face_kernel` record for all three `r96` candidates:

```text
40 bytes stack frame, 80 bytes spill stores, 88 bytes spill loads
Used 96 registers
```

The live and archived gate runner now anchor its regular expression on the
`face_kernel` function-properties header. The historical JSON is retained
unchanged because its SHA-256 is embedded in the completed timing campaign.
This metadata correction does not affect compilation, numerical output,
candidate eligibility, timing, or any G6 decision.
