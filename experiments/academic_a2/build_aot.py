#!/usr/bin/env python3
"""Build one fixed-shape scalar A2 CUDA AOTInductor package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
import torch._inductor

from gradflow import WENOJS

from benchmark_worker import smooth_input


class ScalarRHSModule(torch.nn.Module):
    def __init__(self, order: int, n: int) -> None:
        super().__init__()
        self.scheme = WENOJS(order)
        self.dx = 2.0 * math.pi / n

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        result = self.scheme.rhs(
            values,
            self.dx,
            lambda q: 0.5 * q.square(),
            lambda q: q,
            axis=0,
        )
        for axis in (1, 2):
            result = result + self.scheme.rhs(
                values,
                self.dx,
                lambda q: 0.5 * q.square(),
                lambda q: q,
                axis=axis,
            )
        return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=(5, 11, 15), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.record.exists():
        raise SystemExit("refusing to overwrite A2 AOT artifact")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.record.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record: dict[str, object]
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the A2 AOT builder")
        state = smooth_input(64, 3, torch.float32).cuda()
        module = ScalarRHSModule(arguments.order, 64).eval().cuda()
        torch.cuda.synchronize()
        export_started = time.perf_counter()
        exported = torch.export.export(module, (state,), strict=False)
        torch.cuda.synchronize()
        export_seconds = time.perf_counter() - export_started
        compile_started = time.perf_counter()
        torch._inductor.aoti_compile_and_package(exported, package_path=str(arguments.output))
        torch.cuda.synchronize()
        record = {
            "status": "complete",
            "order": arguments.order,
            "dtype": "float32",
            "dimensions": 3,
            "n": 64,
            "export_seconds": export_seconds,
            "compile_package_seconds": time.perf_counter() - compile_started,
            "total_build_seconds": time.perf_counter() - started,
            "package_path": str(arguments.output),
            "package_sha256": sha256(arguments.output),
            "package_bytes": arguments.output.stat().st_size,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
        }
    except Exception as error:  # noqa: BLE001 - failure is A2 evidence
        record = {
            "status": "failed",
            "order": arguments.order,
            "dtype": "float32",
            "dimensions": 3,
            "n": 64,
            "total_build_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    arguments.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
