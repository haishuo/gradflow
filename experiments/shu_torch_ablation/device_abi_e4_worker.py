#!/usr/bin/env python3
"""One isolated worker for the frozen DVEB device-ABI E4 addendum."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from gradflow import (  # noqa: E402
    DvebArtifact,
    DvebDeviceContext,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    periodic_vortex,
)


LANES = ("dveb-device", "direct-eager", "persistent-compile", "aot-inductor")


class CanonicalAdvance(torch.nn.Module):
    def __init__(self, spacing: tuple[float, float, float]) -> None:
        super().__init__()
        self.spacing = spacing

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        dt = euler_cfl_timestep(state, self.spacing, 0.1)
        return euler_ssp_rk3_step(state, self.spacing, dt)


def normalize_output(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1:
        item = value[0]
        if isinstance(item, torch.Tensor):
            return item
    raise TypeError(f"unsupported numerical output {type(value)!r}")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-state", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.size < 4 or args.steps < 1:
        raise SystemExit("invalid grid or step count")
    if args.warmups < 0 or args.repetitions < 1:
        raise SystemExit("invalid warmup or repetition count")
    if args.lane == "dveb-device" and args.artifact_manifest is None:
        raise SystemExit("dveb-device requires --artifact-manifest")
    if args.lane == "aot-inductor" and args.package is None:
        raise SystemExit("aot-inductor requires --package")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    process_started = time.perf_counter()
    state_cpu, spacing_raw = periodic_vortex(
        (args.size,) * 3, device="cpu", dtype=torch.float32
    )
    state = state_cpu.cuda().contiguous()
    spacing = tuple(float(value) for value in spacing_raw)

    setup_started = time.perf_counter()
    context: DvebDeviceContext | None = None
    last_native: dict[str, object] = {}
    if args.lane == "dveb-device":
        assert args.artifact_manifest is not None
        artifact = DvebArtifact.from_manifest(args.artifact_manifest)
        context = DvebDeviceContext(artifact, args.size, device=state.device)
        output = torch.empty_like(state)

        def run() -> torch.Tensor:
            result = context.run(state, steps=args.steps, out=output)
            last_native.update(
                execution_seconds=result.execution_seconds,
                total_seconds=result.total_seconds,
                workspace_bytes=result.workspace_bytes,
            )
            return result.state
    else:
        direct = CanonicalAdvance(spacing).eval()
        if args.lane == "direct-eager":
            module: Callable[[torch.Tensor], object] = direct
        elif args.lane == "persistent-compile":
            module = torch.compile(direct, fullgraph=True, dynamic=False)
        else:
            assert args.package is not None
            module = torch._inductor.aoti_load_package(str(args.package))

        def run() -> torch.Tensor:
            result = state
            with torch.inference_mode():
                for _ in range(args.steps):
                    result = normalize_output(module(result))
            torch.cuda.synchronize()
            return result

    setup_seconds = time.perf_counter() - setup_started
    last_state: torch.Tensor | None = None
    for _ in range(args.warmups):
        last_state = run()
    torch.cuda.reset_peak_memory_stats()

    observations = []
    for repetition in range(args.repetitions):
        started = time.perf_counter()
        last_state = run()
        observations.append({
            "repetition_in_worker": repetition,
            "call_seconds": time.perf_counter() - started,
        })
    assert last_state is not None
    validation_state = last_state.detach().cpu()
    if args.output_state is not None:
        args.output_state.parent.mkdir(parents=True, exist_ok=True)
        validation_state.contiguous().numpy().tofile(args.output_state)
    record = {
        "schema": "gradflow-dveb-device-e4-worker-v1",
        "lane": args.lane,
        "size": args.size,
        "steps": args.steps,
        "setup_seconds": setup_seconds,
        "process_seconds_after_main": time.perf_counter() - process_started,
        "warmups": args.warmups,
        "observations": observations,
        "finite": bool(torch.isfinite(validation_state).all()),
        "checksum_float64": float(validation_state.to(torch.float64).sum()),
        "output_elements": validation_state.numel(),
        "diagnostics": {
            **last_native,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(state.device),
    }
    if context is not None:
        context.close()
    print(json.dumps(record, sort_keys=True), flush=True)
    if not record["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
