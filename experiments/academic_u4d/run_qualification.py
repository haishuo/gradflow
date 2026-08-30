#!/usr/bin/env python3
"""Build and correctness-qualify all six frozen U4-D lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
U4C = ROOT / "experiments" / "academic_u4c"
sys.path.insert(0, str(U4C))

from run_performance import (  # noqa: E402
    ADAPTER as OPENSBLI_ADAPTER,
    INSTRUMENTER as OPENSBLI_INSTRUMENTER,
    MAKEFILE as OPENSBLI_MAKEFILE,
    MAXIMUM_LIMIT,
    OPENSBLI_COMMIT,
    OPENSBLI_TREE,
    OPS_COMMIT,
    OPS_TREE,
    RMS_LIMIT,
    comparison,
    digest,
    ensure_patch,
    execute,
    git_value,
    parse_json,
    require,
    write_checksums,
)


DRIVER = HERE / "adapter" / "dveb_u4d_driver.cpp"
GRADFLOW_WORKER = U4C / "gradflow_worker.py"
DVEB_COMMIT = "bd4bc791b6e8f4a2ba2b0b28ecdb3086a4d3d97c"
DVEB_TREE = "ca0f146b1951e8f02b79c5a7dd37d1dba3bbc44d"
DVEB_SOURCE_SHA256 = "b4236d640c8429400f44792fae0198b7eed013676444660eb036d99937584ab8"
SIZE = 8192


def conservation(values: np.ndarray) -> dict[str, float | bool]:
    total = float(np.sum(values, dtype=np.float64))
    absolute = float(np.sum(np.abs(values), dtype=np.float64))
    bound = float(32.0 * np.finfo(np.float64).eps * absolute)
    return {
        "sum": total,
        "sum_abs": absolute,
        "bound": bound,
        "passed": bool(abs(total) <= bound),
    }


def source_hashes(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: digest(directory / name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dveb-root", type=Path, required=True)
    parser.add_argument("--opensbli-root", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--sympy-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    source_dveb = args.dveb_root.resolve()
    opensbli = args.opensbli_root.resolve()
    ops = args.ops_root.resolve()
    cuda = args.cuda_root.resolve()
    hdf5 = args.hdf5_root.resolve()
    work = args.work_root.resolve()
    evidence = args.evidence_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    raw = evidence / "raw"
    arrays = evidence / "qualification_arrays"
    raw.mkdir()
    arrays.mkdir()
    commands = [
        f"(cd {shlex.quote(str(Path.cwd()))} && "
        f"{shlex.join([sys.executable, *sys.argv])})"
    ]

    if git_value(source_dveb, "rev-parse", "HEAD") != DVEB_COMMIT:
        raise RuntimeError("DVEB checkout differs from frozen commit")
    if git_value(source_dveb, "rev-parse", "HEAD^{tree}") != DVEB_TREE:
        raise RuntimeError("DVEB checkout differs from frozen tree")
    if subprocess.check_output(
        ["git", "-C", str(source_dveb), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("DVEB source checkout is not clean")
    if git_value(opensbli, "rev-parse", "HEAD") != OPENSBLI_COMMIT:
        raise RuntimeError("OpenSBLI checkout differs from frozen commit")
    if git_value(opensbli, "rev-parse", "HEAD^{tree}") != OPENSBLI_TREE:
        raise RuntimeError("OpenSBLI checkout differs from frozen tree")
    if git_value(ops, "rev-parse", "HEAD") != OPS_COMMIT:
        raise RuntimeError("OPS checkout differs from frozen commit")
    if git_value(ops, "rev-parse", "HEAD^{tree}") != OPS_TREE:
        raise RuntimeError("OPS checkout differs from frozen tree")

    native_flags = (
        "-Xcompiler=-fPIC -O3 -g -std=c++11 "
        "-gencode arch=compute_120,code=sm_120"
    )
    opensbli_env = os.environ.copy()
    opensbli_env.update(
        {
            "OPS_INSTALL_PATH": str(ops / "ops"),
            "OPS_COMPILER": "gnu",
            "CUDA_INSTALL_PATH": str(cuda),
            "HDF5_INSTALL_PATH": str(hdf5),
            "MPICXX": "g++",
            "NVCCFLAGS": native_flags,
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "PYTHONPATH": os.pathsep.join(
                [str(args.sympy_root.resolve()), str(opensbli)]
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(
                [str(cuda / "lib64"), str(hdf5 / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
            ),
        }
    )
    hdf5_link = " ".join(
        [
            f"-L{cuda / 'lib64'}",
            "-lops_hdf5_seq",
            f"-L{hdf5 / 'lib'}",
            "-lhdf5_hl -lhdf5 -lz",
        ]
    )
    ensure_patch(opensbli, commands, opensbli_env)

    c2_arrays = U4C / "evidence" / "u4c_c2_20260830" / "qualification_arrays"
    state_path = c2_arrays / f"n{SIZE}_state.bin"
    canonical_path = c2_arrays / f"n{SIZE}_canonical.bin"
    state = np.fromfile(state_path, dtype=np.float64)
    canonical = np.fromfile(canonical_path, dtype=np.float64)
    if state.shape != (SIZE,) or canonical.shape != (SIZE,):
        raise RuntimeError("frozen U4-C input or canonical array is unavailable")

    preparation: dict[str, float] = {}

    # Build DVEB from a detached local copy; the source repository stays untouched.
    dveb_copy = work / "dveb"
    cloned, elapsed = execute(
        ["git", "clone", "--no-hardlinks", "--no-checkout", str(source_dveb), str(dveb_copy)],
        cwd=work,
        env=os.environ.copy(),
        commands=commands,
    )
    require(cloned, "DVEB local clone")
    preparation["dveb_local_clone_seconds"] = elapsed
    checked_out, elapsed = execute(
        ["git", "checkout", "--detach", DVEB_COMMIT],
        cwd=dveb_copy,
        env=os.environ.copy(),
        commands=commands,
    )
    require(checked_out, "DVEB detached checkout")
    preparation["dveb_checkout_seconds"] = elapsed
    dveb_env = os.environ.copy()
    dveb_env.update(
        {
            "DVEB_CONTRACT": "fma",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
        }
    )
    dveb_build, elapsed = execute(
        ["./dveb", "build", "examples/weno5/weno5.dveb"],
        cwd=dveb_copy,
        env=dveb_env,
        commands=commands,
    )
    require(dveb_build, "DVEB compiler build")
    preparation["dveb_parse_codegen_and_standard_build_seconds"] = elapsed
    (raw / "dveb_build.stdout").write_text(dveb_build.stdout)
    (raw / "dveb_build.stderr").write_text(dveb_build.stderr)
    generated = dveb_copy / "build" / "weno5"
    if digest(dveb_copy / "examples" / "weno5" / "weno5.dveb") != DVEB_SOURCE_SHA256:
        raise RuntimeError("DVEB scalar source hash mismatch")
    driver_object = generated / "u4d_driver.o"
    driver_compile, elapsed = execute(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-fopenmp",
            "-ffp-contract=fast",
            "-I",
            str(dveb_copy / "dvebrt"),
            "-I",
            str(generated),
            "-I",
            str(cuda / "include"),
            "-c",
            str(DRIVER),
            "-o",
            str(driver_object),
        ],
        cwd=dveb_copy,
        env=dveb_env,
        commands=commands,
    )
    require(driver_compile, "DVEB U4-D adapter compile")
    dveb_executable = generated / "u4d_dveb"
    driver_link, link_elapsed = execute(
        [
            "g++",
            "-fopenmp",
            str(driver_object),
            str(generated / "weno5_kernels.o"),
            str(generated / "weno5_kernels_cu.o"),
            str(generated / "dveb_rt.o"),
            str(generated / "dveb_rt_cuda_cu.o"),
            "-o",
            str(dveb_executable),
            "-L",
            str(cuda / "lib64"),
            "-lcudart",
            f"-Wl,-rpath,{cuda / 'lib64'}",
        ],
        cwd=dveb_copy,
        env=dveb_env,
        commands=commands,
    )
    require(driver_link, "DVEB U4-D adapter link")
    preparation["dveb_adapter_compile_and_link_seconds"] = elapsed + link_elapsed
    (raw / "dveb_adapter_compile.stdout").write_text(driver_compile.stdout)
    (raw / "dveb_adapter_compile.stderr").write_text(driver_compile.stderr)
    (raw / "dveb_adapter_link.stdout").write_text(driver_link.stdout)
    (raw / "dveb_adapter_link.stderr").write_text(driver_link.stderr)

    # Build the same U4-C OpenSBLI adapter afresh for U4-D.
    opensbli_app = work / "opensbli"
    opensbli_app.mkdir()
    shutil.copy2(OPENSBLI_MAKEFILE, opensbli_app / "Makefile")
    created, elapsed = execute(
        [sys.executable, str(OPENSBLI_ADAPTER), "--case", "state_a", "--size", str(SIZE)],
        cwd=opensbli_app,
        env=opensbli_env,
        commands=commands,
    )
    require(created, "U4-D OpenSBLI generation")
    preparation["opensbli_symbolic_generation_seconds"] = elapsed
    instrumented, elapsed = execute(
        [sys.executable, str(OPENSBLI_INSTRUMENTER), "opensbli.cpp", "opensbliblock00_kernels.h"],
        cwd=opensbli_app,
        env=opensbli_env,
        commands=commands,
    )
    require(instrumented, "U4-D OpenSBLI instrumentation")
    preparation["opensbli_instrumentation_seconds"] = elapsed
    seq_build, elapsed = execute(
        ["make", "opensbli_seq", f"HDF5_LIB_SEQ={hdf5_link}"],
        cwd=opensbli_app,
        env=opensbli_env,
        commands=commands,
    )
    require(seq_build, "U4-D OpenSBLI CPU build")
    preparation["opensbli_translation_and_cpu_build_seconds"] = elapsed
    cuda_build, elapsed = execute(
        [
            "make",
            "opensbli_cuda",
            "NVCC_FLAG_SET=1",
            f"NVCCFLAGS={native_flags}",
            "CXXFLAGS=-O3 -fPIC -Wall -g -std=c++11 -DU4C_CUDA",
            f"HDF5_LIB_SEQ={hdf5_link}",
        ],
        cwd=opensbli_app,
        env=opensbli_env,
        commands=commands,
    )
    require(cuda_build, "U4-D OpenSBLI CUDA build")
    preparation["opensbli_cuda_build_seconds"] = elapsed
    (raw / "opensbli_cpu_build.stdout").write_text(seq_build.stdout)
    (raw / "opensbli_cpu_build.stderr").write_text(seq_build.stderr)
    (raw / "opensbli_cuda_build.stdout").write_text(cuda_build.stdout)
    (raw / "opensbli_cuda_build.stderr").write_text(cuda_build.stderr)

    grad_env = os.environ.copy()
    grad_env["PYTHONPATH"] = str(ROOT / "src")
    grad_env["TORCHINDUCTOR_CACHE_DIR"] = str(work / "torchinductor_cache")

    candidates: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for implementation in ("dveb", "opensbli", "gradflow"):
        for device in ("cpu", "cuda"):
            lane = f"{implementation}_{device}"
            output = arrays / f"{lane}.bin"
            if implementation == "dveb":
                completed, elapsed = execute(
                    [
                        str(dveb_executable),
                        "--size",
                        str(SIZE),
                        "--backend",
                        device,
                        "--mode",
                        "qualify",
                        "--input",
                        str(state_path),
                        "--output",
                        str(output),
                    ],
                    cwd=dveb_copy,
                    env=dveb_env,
                    commands=commands,
                )
            elif implementation == "opensbli":
                binary = "opensbli_seq" if device == "cpu" else "opensbli_cuda"
                lane_env = opensbli_env | {
                    "U4C_MODE": "qualify",
                    "U4C_STATE_PATH": str(state_path),
                    "U4C_RHS_PATH": str(output),
                    "OPS_BLOCK_SIZE_X": "256",
                }
                completed, elapsed = execute(
                    [f"./{binary}"],
                    cwd=opensbli_app,
                    env=lane_env,
                    commands=commands,
                )
            else:
                completed, elapsed = execute(
                    [
                        sys.executable,
                        str(GRADFLOW_WORKER),
                        "--size",
                        str(SIZE),
                        "--device",
                        device,
                        "--mode",
                        "qualify",
                        "--input",
                        str(state_path),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    env=grad_env,
                    commands=commands,
                )
            require(completed, f"U4-D {lane} qualification")
            (raw / f"{lane}.stdout").write_text(completed.stdout)
            (raw / f"{lane}.stderr").write_text(completed.stderr)
            values = np.fromfile(output, dtype=np.float64)
            if values.shape != canonical.shape:
                raise RuntimeError(f"U4-D {lane} output shape mismatch")
            candidates[lane] = values
            metadata[lane] = {"process_seconds": elapsed}
            if implementation == "gradflow":
                worker = parse_json(completed.stdout)
                metadata[lane]["worker"] = worker
                if worker["graph"] != {"unique_graphs": 1, "graph_break_count": 0}:
                    raise RuntimeError(f"U4-D {lane} graph gate failed")
                preparation[f"{lane}_first_call_seconds"] = worker["first_call_seconds"]

    qualification = {
        lane: {
            **comparison(values, canonical),
            "sha256": digest(arrays / f"{lane}.bin"),
            "metadata": metadata[lane],
        }
        for lane, values in candidates.items()
    }
    for device in ("cpu", "cuda"):
        lhs = candidates[f"dveb_{device}"]
        rhs = candidates[f"gradflow_{device}"]
        qualification[f"dveb_{device}"]["versus_gradflow_same_device"] = comparison(lhs, rhs)
    qualification["dveb_cpu_cuda"] = comparison(candidates["dveb_cpu"], candidates["dveb_cuda"])
    all_admitted = bool(
        all(qualification[lane]["passed"] for lane in candidates)
        and qualification["dveb_cpu_cuda"]["passed"]
    )
    decision = "all_six_lanes_qualified" if all_admitted else "correctness_excluded"

    record = {
        "schema": "gradflow.academic_u4d.qualification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "timing_interpretation_prohibited": True,
        "size": SIZE,
        "bounds": {
            "maximum_normalized": MAXIMUM_LIMIT,
            "rms_normalized": RMS_LIMIT,
        },
        "canonical": {
            "state_sha256": digest(state_path),
            "rhs_sha256": digest(canonical_path),
            "finite": bool(np.all(np.isfinite(canonical))),
            "conservation": conservation(canonical),
        },
        "qualification": qualification,
        "preparation": preparation,
        "sources": {
            "dveb": {"commit": DVEB_COMMIT, "tree": DVEB_TREE, "source_sha256": DVEB_SOURCE_SHA256},
            "opensbli": {"commit": OPENSBLI_COMMIT, "tree": OPENSBLI_TREE},
            "ops": {"commit": OPS_COMMIT, "tree": OPS_TREE},
            "driver_sha256": digest(DRIVER),
        },
        "generated": {
            "dveb": source_hashes(
                generated,
                ("weno5_gen.h", "weno5_math.inc", "weno5_kernels.cpp", "weno5_kernels.cu"),
            ),
            "opensbli": source_hashes(
                opensbli_app,
                ("opensbli.cpp", "opensbli_ops.cpp", "opensbliblock00_kernels.h"),
            ),
        },
        "artifacts": {
            "work_root": str(work),
            "dveb_root": str(dveb_copy),
            "dveb_executable": str(dveb_executable),
            "dveb_executable_sha256": digest(dveb_executable),
            "opensbli_root": str(opensbli_app),
            "opensbli_cpu_executable": str(opensbli_app / "opensbli_seq"),
            "opensbli_cpu_sha256": digest(opensbli_app / "opensbli_seq"),
            "opensbli_cuda_executable": str(opensbli_app / "opensbli_cuda"),
            "opensbli_cuda_sha256": digest(opensbli_app / "opensbli_cuda"),
            "torchinductor_cache": str(work / "torchinductor_cache"),
        },
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dveb_contract": "fma",
            "cpu_threads": 1,
        },
    }
    (evidence / "qualification.json").write_text(json.dumps(record, indent=2) + "\n")
    (evidence / "COMMANDS.txt").write_text("\n".join(commands) + "\n")
    write_checksums(evidence)
    print(json.dumps({"decision": decision, "qualification": qualification}, indent=2))
    if not all_admitted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
