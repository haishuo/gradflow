#!/usr/bin/env python3
"""Build one fixed-shape package from GradFlow's canonical Euler source."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch._inductor  # noqa: E402

from gradflow import periodic_vortex  # noqa: E402
from abi_bakeoff_worker import CanonicalAdvance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state, spacing = periodic_vortex((args.size,) * 3, device="cuda", dtype=torch.float32)
    module = CanonicalAdvance(tuple(float(value) for value in spacing)).eval()
    torch.cuda.synchronize()
    export_started = time.perf_counter()
    exported = torch.export.export(module, (state,), strict=False)
    export_seconds = time.perf_counter() - export_started
    compile_started = time.perf_counter()
    torch._inductor.aoti_compile_and_package(exported, package_path=str(args.output))
    torch.cuda.synchronize()
    record = {
        "size": args.size,
        "export_seconds": export_seconds,
        "compile_package_seconds": time.perf_counter() - compile_started,
        "package_bytes": args.output.stat().st_size,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
