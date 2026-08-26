#!/usr/bin/env python3
"""One isolated worker for the frozen forced-target ABI bakeoff."""

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
    Solver,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    periodic_vortex,
)


PYTORCH_LANES = {"direct-eager", "persistent-compile", "aot-inductor"}
DVEB_LANES = {"dveb-cpu6", "dveb-cpu12", "dveb-cuda"}


class CanonicalAdvance(torch.nn.Module):
    """One canonical CFL-plus-SSP-RK3 step, suitable for export/compile."""

    def __init__(self, spacing: tuple[float, float, float]) -> None:
        super().__init__()
        self.spacing = spacing

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        dt = euler_cfl_timestep(state, self.spacing, 0.1)
        return euler_ssp_rk3_step(state, self.spacing, dt)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=sorted(PYTORCH_LANES | DVEB_LANES), required=True)
    parser.add_argument(
        "--endpoint",
        choices=("single", "warm", "resident", "prepare", "correctness"),
        required=True,
    )
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-state", type=Path)
    return parser.parse_args()


def normalize_output(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1:
        item = value[0]
        if isinstance(item, torch.Tensor):
            return item
    raise TypeError(f"numerical callable returned unsupported value {type(value)!r}")


def make_dveb_runner(
    lane: str,
    state: torch.Tensor,
    spacing: tuple[float, float, float],
    steps: int,
    manifest: Path,
) -> tuple[Callable[[], torch.Tensor], Callable[[], dict[str, object]]]:
    workers = 6 if lane == "dveb-cpu6" else 12
    backend = "cuda-native" if lane == "dveb-cuda" else "cpu-simd"
    artifact = DvebArtifact.from_manifest(manifest)
    solver = Solver(
        equations="euler",
        dimension=3,
        weno=("JS", 5),
        flux_split="global_lf",
        boundaries="periodic_duplicated",
        dtype=torch.float32,
        spacing=spacing,
        dveb_artifact=artifact,
        cpu_workers=workers,
    )

    def run() -> torch.Tensor:
        return solver.run(state, steps=steps, backend=backend)

    def diagnostics() -> dict[str, object]:
        item = solver.last_run
        if item is None:
            return {}
        return {
            "selected": item.backend.selected,
            "native_execution_seconds": item.native_execution_seconds,
            "native_total_seconds": item.native_total_seconds,
            "native_peak_bytes": item.native_peak_bytes,
            "hidden_device_transfers": item.hidden_device_transfers,
        }

    return run, diagnostics


def make_pytorch_runner(
    lane: str,
    state_cpu: torch.Tensor,
    spacing: tuple[float, float, float],
    steps: int,
    package: Path | None,
) -> tuple[
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    Callable[[], dict[str, object]],
]:
    module: Callable[[torch.Tensor], object]
    direct = CanonicalAdvance(spacing).eval()
    if lane == "direct-eager":
        module = direct
    elif lane == "persistent-compile":
        module = torch.compile(direct, fullgraph=True, dynamic=False)
    else:
        if package is None:
            raise ValueError("AOTInductor requires --package")
        module = torch._inductor.aoti_load_package(str(package))

    def advance(state: torch.Tensor) -> torch.Tensor:
        result = state
        with torch.inference_mode():
            for _ in range(steps):
                result = normalize_output(module(result))
        return result

    def cpu_roundtrip() -> torch.Tensor:
        device_state = state_cpu.to("cuda")
        result = advance(device_state)
        torch.cuda.synchronize()
        result_cpu = result.to("cpu")
        torch.cuda.synchronize()
        return result_cpu

    resident_state: torch.Tensor | None = None

    def resident() -> torch.Tensor:
        nonlocal resident_state
        if resident_state is None:
            resident_state = state_cpu.to("cuda")
            torch.cuda.synchronize()
        result = advance(resident_state)
        torch.cuda.synchronize()
        return result

    def diagnostics() -> dict[str, object]:
        return {
            "selected": lane,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    return cpu_roundtrip, resident, diagnostics


def validate_result(state: torch.Tensor) -> dict[str, object]:
    state_cpu = state.detach().to("cpu")
    return {
        "finite": bool(torch.isfinite(state_cpu).all()),
        "checksum_float64": float(state_cpu.to(torch.float64).sum()),
        "output_elements": state_cpu.numel(),
    }


def main() -> None:
    args = arguments()
    if args.size < 4 or args.steps < 1:
        raise SystemExit("size must be at least four and steps must be positive")
    if args.repetitions < 1 or args.warmups < 0:
        raise SystemExit("invalid repetition count")
    if args.lane in DVEB_LANES and args.artifact_manifest is None:
        raise SystemExit("DVEB lanes require --artifact-manifest")
    if args.endpoint == "resident" and args.lane not in PYTORCH_LANES:
        raise SystemExit("DVEB ABI v1 does not expose resident device state")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    process_main_started = time.perf_counter()
    state_cpu, spacing_raw = periodic_vortex(
        (args.size,) * 3, device="cpu", dtype=torch.float32
    )
    state_cpu = state_cpu.contiguous()
    spacing = (float(spacing_raw[0]), float(spacing_raw[1]), float(spacing_raw[2]))

    setup_started = time.perf_counter()
    if args.lane in DVEB_LANES:
        assert args.artifact_manifest is not None
        run_cpu, diagnostic = make_dveb_runner(
            args.lane, state_cpu, spacing, args.steps, args.artifact_manifest
        )
        run_resident = None
    else:
        run_cpu, run_resident, diagnostic = make_pytorch_runner(
            args.lane, state_cpu, spacing, args.steps, args.package
        )
    setup_seconds = time.perf_counter() - setup_started

    call = run_resident if args.endpoint == "resident" else run_cpu
    assert call is not None
    observations: list[dict[str, object]] = []
    last_state: torch.Tensor | None = None
    for _ in range(args.warmups):
        last_state = call()
    if args.warmups and args.lane in PYTORCH_LANES:
        torch.cuda.reset_peak_memory_stats()

    for repetition in range(args.repetitions):
        started = time.perf_counter()
        last_state = call()
        call_seconds = time.perf_counter() - started
        observations.append({"repetition_in_worker": repetition, "call_seconds": call_seconds})

    assert last_state is not None
    validation = validate_result(last_state)
    if args.output_state is not None:
        output_cpu = last_state.detach().to("cpu").contiguous()
        args.output_state.parent.mkdir(parents=True, exist_ok=True)
        output_cpu.numpy().tofile(args.output_state)

    record = {
        "schema": "gradflow-dveb-abi-bakeoff-worker-v1",
        "lane": args.lane,
        "endpoint": args.endpoint,
        "size": args.size,
        "steps": args.steps,
        "setup_seconds": setup_seconds,
        "process_seconds_after_main": time.perf_counter() - process_main_started,
        "warmups": args.warmups,
        "observations": observations,
        "diagnostics": diagnostic(),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **validation,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if args.lane in PYTORCH_LANES:
        record["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(record, sort_keys=True), flush=True)
    if not validation["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
