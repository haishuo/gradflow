#!/usr/bin/env python3
"""Fresh-process worker for one matched Shu WENO bake-off lane."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from shu_euler_torch import cfl_timestep, periodic_vortex, ssp_rk3_step
from shu_euler_torch_conv import ConvFeatureWenoStep


class DirectWenoAdvance(torch.nn.Module):
    def __init__(self, spacing: tuple[float, ...]) -> None:
        super().__init__()
        self.spacing = spacing

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        dt = cfl_timestep(state, self.spacing, 0.1)
        return ssp_rk3_step(state, self.spacing, dt)


class ConvWenoAdvance(torch.nn.Module):
    def __init__(self, spacing: tuple[float, ...]) -> None:
        super().__init__()
        self.spacing = spacing
        self.step = ConvFeatureWenoStep(3, spacing)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        dt = cfl_timestep(state, self.spacing, 0.1)
        return self.step(state, dt)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lane",
        choices=("direct-eager", "conv-eager", "compile", "aot"),
        required=True,
    )
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--package", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if arguments.steps < 1:
        raise SystemExit("steps must be positive")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()

    timed_start = time.perf_counter()
    state_cpu, spacing = periodic_vortex(
        (arguments.size,) * 3, device="cpu", dtype=torch.float32
    )
    state = state_cpu.to("cuda")
    if arguments.lane == "direct-eager":
        advance = DirectWenoAdvance(spacing).cuda()
    elif arguments.lane == "conv-eager":
        advance = ConvWenoAdvance(spacing).cuda()
    elif arguments.lane == "compile":
        advance = torch.compile(
            DirectWenoAdvance(spacing).cuda(), fullgraph=True, dynamic=False
        )
    else:
        if arguments.package is None:
            raise SystemExit("the AOT lane requires --package")
        advance = torch._inductor.aoti_load_package(str(arguments.package))

    torch.cuda.synchronize()
    execution_start = time.perf_counter()
    result = state
    with torch.inference_mode():
        for _ in range(arguments.steps):
            result = advance(result)
    torch.cuda.synchronize()
    execution_seconds = time.perf_counter() - execution_start

    result_cpu = result.cpu()
    torch.cuda.synchronize()
    end_to_host_seconds = time.perf_counter() - timed_start
    checksum = float(result_cpu.to(torch.float64).sum())
    finite = bool(torch.isfinite(result_cpu).all())

    print(
        json.dumps(
            {
                "lane": arguments.lane,
                "size": arguments.size,
                "steps": arguments.steps,
                "cfl": 0.1,
                "execution_seconds": execution_seconds,
                "end_to_host_after_import_seconds": end_to_host_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "checksum": checksum,
                "finite": finite,
                "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
