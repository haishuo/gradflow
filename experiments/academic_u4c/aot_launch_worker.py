#!/usr/bin/env python3
"""Fresh-process U4-C AOT launch-to-answer worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def tensor_output(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        if isinstance(value[0], torch.Tensor):
            return value[0]
    raise TypeError(f"unexpected AOT output: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    values = np.fromfile(args.input, dtype=np.float64)
    if values.shape != (args.size,):
        raise RuntimeError("frozen input shape mismatch")
    state = torch.from_numpy(values.copy()).cuda()
    call = torch._inductor.aoti_load_package(str(args.package))
    with torch.inference_mode():
        output = tensor_output(call(state))
    torch.cuda.synchronize()
    host = output.cpu().numpy()
    if args.output is not None:
        host.tofile(args.output)
    print(
        json.dumps(
            {
                "schema": "gradflow.academic_u4c.aot_launch.v1",
                "status": "complete",
                "size": args.size,
                "finite": bool(np.all(np.isfinite(host))),
                "checksum_float64": float(np.sum(host, dtype=np.float64)),
                "maximum_absolute": float(np.max(np.abs(host))),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
