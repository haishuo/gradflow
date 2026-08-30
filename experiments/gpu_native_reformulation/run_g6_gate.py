#!/usr/bin/env python3
"""Run the frozen G6 exact-math occupancy pre-timing gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
G3_ARRAYS = (
    ROOT
    / "experiments/gpu_native_reformulation/evidence/"
    / "g3_qualification_20260829/arrays"
)
CANDIDATES = tuple(
    f"b{block}_{policy}"
    for block in (64, 128, 256)
    for policy in ("u", "r112", "r96")
)
STEP_BOUND = 2.0e-5
RHS_BOUND = 5.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pressure(state: np.ndarray) -> np.ndarray:
    density = state[0].astype(np.float64)
    momentum = state[1:4].astype(np.float64)
    energy = state[4].astype(np.float64)
    return 0.4 * (energy - 0.5 * np.sum(momentum * momentum, axis=0) / density)


def health(state: np.ndarray) -> dict[str, Any]:
    p = pressure(state)
    return {
        "finite": bool(np.isfinite(state).all()),
        "minimum_density": float(np.min(state[0])),
        "minimum_pressure": float(np.min(p)),
        "positive": bool(np.min(state[0]) > 0.0 and np.min(p) > 0.0),
    }


def compare(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "bitwise_identical": bool(np.array_equal(candidate, reference)),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
    }


def run_native(
    executable: Path,
    expected_contract: str,
    input_path: Path,
    shape: tuple[int, ...],
    *,
    steps: int,
    mode: str,
    output_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = shape[-1]
    completed = subprocess.run(
        [
            str(executable),
            "--size", str(n),
            "--steps", str(steps),
            "--warmups", "0",
            "--repetitions", "1",
            "--input-state", str(input_path),
            "--mode", mode,
            "--output-state", str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(completed.stdout)
    if metadata.get("contract") != expected_contract:
        raise RuntimeError(f"unexpected contract: {metadata.get('contract')}")
    result = np.fromfile(output_path, dtype=np.float32).reshape(shape)
    return result, metadata


def parse_spills(log: Path) -> dict[str, int]:
    text = log.read_text()
    match = re.search(
        r"Function properties for [^\n]*face_kernel[^\n]*\n"
        r"\s*(\d+) bytes stack frame, (\d+) bytes spill stores, "
        r"(\d+) bytes spill loads",
        text,
    )
    if match is None:
        raise RuntimeError(f"could not find face-kernel resource record in {log}")
    stack, stores, loads = (int(value) for value in match.groups())
    return {
        "stack_frame_bytes": stack,
        "spill_store_bytes": stores,
        "spill_load_bytes": loads,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    build = Path(arguments.build_dir).resolve()
    logs = Path(arguments.compiler_log_dir).resolve()
    frozen = Path(arguments.frozen_r6q).resolve()
    output = Path(arguments.output_dir).resolve()
    if output.exists():
        raise RuntimeError("G6 gate output already exists; refusing to overwrite")
    output.mkdir(parents=True)
    cases = (
        ("vortex_n32_s1", "step_vortex_n32_s1.npz", "step", 1),
        ("vortex_n32_s10", "step_vortex_n32_s1.npz", "step", 10),
        ("perturbed_vortex_n32_s1", "step_perturbed_vortex_n32_s1.npz", "step", 1),
        ("dual_shu_osher_n32_s10", "step_dual_shu_osher_n32_s10.npz", "step", 10),
        ("smooth_entropy_rhs_n40", "rhs_smooth_entropy_n40.npz", "rhs", 1),
    )
    artifacts = {}
    for candidate in CANDIDATES:
        executable = build / f"gradflow_g6_{candidate}"
        log = logs / f"{candidate}.log"
        artifacts[candidate] = {
            "executable": {"path": str(executable), "sha256": sha256(executable)},
            "compiler_log": {"path": str(log), "sha256": sha256(log)},
            "face_compiler_resources": parse_spills(log),
        }
    artifacts["frozen_r6q"] = {
        "executable": {"path": str(frozen), "sha256": sha256(frozen)}
    }

    rows = []
    resource_metadata = {}
    with tempfile.TemporaryDirectory(prefix="gradflow-g6-gate-") as directory:
        scratch = Path(directory)
        for case_name, source_name, mode, steps in cases:
            source = G3_ARRAYS / source_name
            with np.load(source) as archive:
                initial = np.ascontiguousarray(archive["initial"], dtype=np.float32)
            input_path = scratch / f"{case_name}.f32"
            initial.tofile(input_path)
            frozen_path = output / f"{case_name}_frozen_r6q.f32"
            reference, frozen_metadata = run_native(
                frozen,
                "r6q_arbitrary_state_rhs_unique_strict_f32_shu_face_once_v1",
                input_path,
                initial.shape,
                steps=steps,
                mode=mode,
                output_path=frozen_path,
            )
            candidates = []
            for candidate in CANDIDATES:
                candidate_path = output / f"{case_name}_{candidate}.f32"
                state, metadata = run_native(
                    build / f"gradflow_g6_{candidate}",
                    f"g6_r6q_{candidate}_v1",
                    input_path,
                    initial.shape,
                    steps=steps,
                    mode=mode,
                    output_path=candidate_path,
                )
                parity = compare(state, reference)
                bound = RHS_BOUND if mode == "rhs" else STEP_BOUND
                row: dict[str, Any] = {
                    "candidate": candidate,
                    "metadata": metadata,
                    "comparison": parity,
                    "bound": bound,
                    "output": {
                        "path": candidate_path.name,
                        "sha256": sha256(candidate_path),
                    },
                }
                if mode == "step":
                    row["health"] = health(state)
                    row["passed"] = bool(
                        parity["maximum_absolute_difference"] <= bound
                        and row["health"]["finite"]
                        and row["health"]["positive"]
                    )
                else:
                    row["finite"] = bool(np.isfinite(state).all())
                    row["passed"] = bool(
                        parity["maximum_absolute_difference"] <= bound
                        and row["finite"]
                    )
                candidates.append(row)
                resource_metadata.setdefault(candidate, {
                    key: metadata[key]
                    for key in (
                        "face_threads",
                        "declared_register_limit",
                        "compiled_face_registers_per_thread",
                        "compiled_face_local_bytes_per_thread",
                        "compiled_face_static_shared_bytes",
                        "face_active_blocks_per_sm",
                        "face_theoretical_occupancy_percent",
                        "peak_allocated_bytes",
                    )
                })
            rows.append({
                "case": case_name,
                "mode": mode,
                "steps": steps,
                "n": int(initial.shape[-1]),
                "source": {"path": str(source.relative_to(ROOT)), "sha256": sha256(source)},
                "input_sha256": sha256(input_path),
                "frozen_r6q": {
                    "metadata": frozen_metadata,
                    "output": {"path": frozen_path.name, "sha256": sha256(frozen_path)},
                },
                "candidates": candidates,
            })

    passing = [
        candidate
        for candidate in CANDIDATES
        if all(
            next(item for item in case["candidates"] if item["candidate"] == candidate)[
                "passed"
            ]
            for case in rows
        )
    ]
    result = {
        "schema": "gradflow-g6-occupancy-forward-gate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_backend_admitted": False,
        "candidate_order": list(CANDIDATES),
        "passing_candidates": passing,
        "artifacts": artifacts,
        "resource_metadata": resource_metadata,
        "cases": rows,
        "passed": len(passing) == len(CANDIDATES),
    }
    (output / "forward_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if not result["passed"]:
        raise RuntimeError("one or more G6 candidates failed the pre-timing gate")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--compiler-log-dir", required=True)
    parser.add_argument("--frozen-r6q", required=True)
    parser.add_argument("--output-dir", required=True)
    result = run(parser.parse_args())
    print(json.dumps({
        "schema": result["schema"],
        "passed": result["passed"],
        "passing_candidates": result["passing_candidates"],
        "resources": result["resource_metadata"],
        "all_bitwise": all(
            item["comparison"]["bitwise_identical"]
            for case in result["cases"]
            for item in case["candidates"]
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
