#!/usr/bin/env python3
"""Run the frozen G5 shared-pencil pre-timing forward gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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
P1_CONTRACT = "p1_shared_pencil_unique_strict_f32_shu_fused_update_v1"
R6Q_CONTRACT = "r6q_arbitrary_state_rhs_unique_strict_f32_shu_face_once_v1"
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


def comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "bitwise_identical": bool(np.array_equal(candidate, reference)),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
    }


def run_native(
    executable: Path,
    contract: str,
    initial: np.ndarray,
    *,
    steps: int,
    mode: str,
    output_path: Path,
    scratch: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = initial.shape[-1]
    input_path = scratch / f"input_{mode}_n{n}_s{steps}.f32"
    np.ascontiguousarray(initial, dtype=np.float32).tofile(input_path)
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
    if metadata.get("contract") != contract:
        raise RuntimeError(f"unexpected contract: {metadata.get('contract')}")
    result = np.fromfile(output_path, dtype=np.float32).reshape(initial.shape)
    return result, metadata


def load_initial(name: str) -> tuple[np.ndarray, Path]:
    source = G3_ARRAYS / name
    with np.load(source) as archive:
        initial = np.ascontiguousarray(archive["initial"], dtype=np.float32)
    return initial, source


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    p1 = Path(arguments.p1).resolve()
    r6q = Path(arguments.r6q).resolve()
    output = Path(arguments.output_dir).resolve()
    if output.exists():
        raise RuntimeError("G5 gate output already exists; refusing to overwrite")
    output.mkdir(parents=True)
    cases = (
        ("vortex_n32_s1", "step_vortex_n32_s1.npz", "step", 1),
        ("vortex_n32_s10", "step_vortex_n32_s1.npz", "step", 10),
        ("perturbed_vortex_n32_s1", "step_perturbed_vortex_n32_s1.npz", "step", 1),
        ("dual_shu_osher_n32_s10", "step_dual_shu_osher_n32_s10.npz", "step", 10),
        ("smooth_entropy_rhs_n40", "rhs_smooth_entropy_n40.npz", "rhs", 1),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="gradflow-g5-gate-") as directory:
        scratch = Path(directory)
        for case_name, source_name, mode, steps in cases:
            initial, source = load_initial(source_name)
            p1_path = output / f"{case_name}_p1.f32"
            r6q_path = output / f"{case_name}_r6q.f32"
            p1_state, p1_record = run_native(
                p1, P1_CONTRACT, initial, steps=steps, mode=mode,
                output_path=p1_path, scratch=scratch,
            )
            r6q_state, r6q_record = run_native(
                r6q, R6Q_CONTRACT, initial, steps=steps, mode=mode,
                output_path=r6q_path, scratch=scratch,
            )
            parity = comparison(p1_state, r6q_state)
            bound = RHS_BOUND if mode == "rhs" else STEP_BOUND
            row: dict[str, Any] = {
                "case": case_name,
                "mode": mode,
                "n": int(initial.shape[-1]),
                "steps": steps,
                "source": {
                    "path": str(source.relative_to(ROOT)),
                    "sha256": sha256(source),
                },
                "input_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                "p1": p1_record,
                "r6q": r6q_record,
                "comparison": parity,
                "bound": bound,
                "outputs": {
                    "p1": {"path": p1_path.name, "sha256": sha256(p1_path)},
                    "r6q": {"path": r6q_path.name, "sha256": sha256(r6q_path)},
                },
            }
            if mode == "step":
                row["p1_health"] = health(p1_state)
                row["r6q_health"] = health(r6q_state)
                row["passed"] = bool(
                    parity["maximum_absolute_difference"] <= bound
                    and row["p1_health"]["finite"]
                    and row["p1_health"]["positive"]
                    and row["r6q_health"]["finite"]
                    and row["r6q_health"]["positive"]
                )
            else:
                row["finite"] = bool(
                    np.isfinite(p1_state).all() and np.isfinite(r6q_state).all()
                )
                row["passed"] = bool(
                    parity["maximum_absolute_difference"] <= bound and row["finite"]
                )
            rows.append(row)

    p1_peak = int(rows[0]["p1"]["peak_allocated_bytes"])
    r6q_peak = int(rows[0]["r6q"]["peak_allocated_bytes"])
    memory = {
        "p1_peak_allocated_bytes": p1_peak,
        "r6q_peak_allocated_bytes": r6q_peak,
        "p1_over_r6q": p1_peak / r6q_peak,
        "maximum_ratio": 0.70,
        "passed": p1_peak / r6q_peak <= 0.70,
    }
    record = {
        "schema": "gradflow-g5-shared-pencil-forward-gate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_backend_admitted": False,
        "contracts": {"p1": P1_CONTRACT, "r6q": R6Q_CONTRACT},
        "artifacts": {
            "p1": {"path": str(p1), "sha256": sha256(p1)},
            "r6q": {"path": str(r6q), "sha256": sha256(r6q)},
        },
        "cases": rows,
        "memory": memory,
        "passed": bool(all(row["passed"] for row in rows) and memory["passed"]),
    }
    (output / "forward_gate.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    if not record["passed"]:
        raise RuntimeError("G5 pre-timing forward gate failed")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", required=True)
    parser.add_argument("--r6q", required=True)
    parser.add_argument("--output-dir", required=True)
    result = run(parser.parse_args())
    print(json.dumps({
        "schema": result["schema"],
        "passed": result["passed"],
        "memory": result["memory"],
        "cases": [
            {
                "case": row["case"],
                "passed": row["passed"],
                **row["comparison"],
            }
            for row in result["cases"]
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
