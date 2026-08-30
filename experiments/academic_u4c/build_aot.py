#!/usr/bin/env python3
"""Build the fixed-shape float64 U4-C AOTInductor CUDA package."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch._inductor

from gradflow import weno5_rhs


class ScalarWENO5(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.spacing = 1.0 / size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return weno5_rhs(
            values, self.spacing, lambda value: value, alpha=1.0
        )


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.package.exists() or args.record.exists():
        raise SystemExit("refusing to overwrite U4-C AOT output")
    args.package.parent.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        values = np.fromfile(args.input, dtype=np.float64)
        if values.shape != (args.size,):
            raise RuntimeError("frozen input shape mismatch")
        state = torch.from_numpy(values.copy()).cuda()
        module = ScalarWENO5(args.size).eval().cuda()
        torch.cuda.synchronize()
        export_started = time.perf_counter()
        exported = torch.export.export(module, (state,), strict=False)
        torch.cuda.synchronize()
        export_seconds = time.perf_counter() - export_started
        compile_started = time.perf_counter()
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(args.package)
        )
        torch.cuda.synchronize()
        record = {
            "schema": "gradflow.academic_u4c.aot_build.v1",
            "status": "complete",
            "size": args.size,
            "dtype": "float64",
            "device": "cuda",
            "export_seconds": export_seconds,
            "compile_package_seconds": time.perf_counter() - compile_started,
            "total_build_seconds": time.perf_counter() - started,
            "package_sha256": digest(args.package),
            "package_bytes": args.package.stat().st_size,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
    except Exception as error:  # noqa: BLE001 - retained experimental result
        record = {
            "schema": "gradflow.academic_u4c.aot_build.v1",
            "status": "failed",
            "size": args.size,
            "dtype": "float64",
            "device": "cuda",
            "total_build_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
