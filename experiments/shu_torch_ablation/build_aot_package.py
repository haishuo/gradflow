#!/usr/bin/env python3
"""Build a fixed-shape AOTInductor package for the matched 3-D Shu step."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch._inductor

from bakeoff_worker import DirectWenoAdvance
from shu_euler_torch import periodic_vortex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    state, spacing = periodic_vortex(
        (arguments.size,) * 3, device="cuda", dtype=torch.float32
    )
    model = DirectWenoAdvance(spacing).eval().cuda()

    torch.cuda.synchronize()
    export_start = time.perf_counter()
    exported = torch.export.export(model, (state,), strict=False)
    export_seconds = time.perf_counter() - export_start

    compile_start = time.perf_counter()
    torch._inductor.aoti_compile_and_package(
        exported, package_path=str(arguments.output)
    )
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start
    print(
        json.dumps(
            {
                "size": arguments.size,
                "export_seconds": export_seconds,
                "compile_package_seconds": compile_seconds,
                "package_bytes": arguments.output.stat().st_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
