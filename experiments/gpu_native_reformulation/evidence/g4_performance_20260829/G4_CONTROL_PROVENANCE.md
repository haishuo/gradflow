# G4 cell-recompute control provenance

The G4 control derives from the separately authored native CUDA ceiling in the
DVEB development repository. DVEB is a read-only source for this GradFlow
experiment; no DVEB file is modified.

Source repository state:

```text
repository: /mnt/projects/dveb
commit: bd4bc791b6e8f4a2ba2b0b28ecdb3086a4d3d97c
branch: codex/trunk-004-device-resident-abi
source directory: tools/shu_euler3d_ceiling
```

Upstream identities observed before the G4 copy:

| File | SHA-256 |
|---|---|
| `main.cpp` | `c30918f2ec4bb80eb7961c1b86f8149277871ad2d47ddecb16593b417f9deda6` |
| `cuda.cu` | `c3964d31399bb4d2b68bdd2c33a70aa5263ea3b370a3d94e2dde2f169dfcfb6d` |
| `cpu.cpp` | `d7282983ac5861b17a75daed3cc9457f9cefb75414a58309ac5112a535e7041c` |
| `shu_math.h` | `125dd8ec0d60cc4c965e1a8f804b12ae471cf73850e3484520cc400ae0db9009` |
| `runner.h` | `56b07cad0b63ba8425d1e8b8b745c94e3a98e918c6fd32832e86cc9ea2252aaa` |
| `build.sh` | `811d840500dbcd82ea48e0718d7d45d3bfbfb0225405c9a27768b2542c8b842f` |
| observed upstream binary | `873a9227196664398012e7d42a27e29ec9cd3610c45a4c61ab40a0688aed3caa` |

The G4 local copy is not represented as DVEB output and does not supersede the
upstream ceiling. Its only permitted changes are the input/layout adapter,
contract metadata, and CUDA-event timing described in the frozen protocol.
The mathematical CUDA kernels and launch geometry remain an attributed control
derived from the source above.

The face-once candidate is the G3 R6Q binary with SHA-256
`74019071fa3fae6a842464d8ffcfc5fad6fa2fa2d805bf22b8616814c8a5e438`.
