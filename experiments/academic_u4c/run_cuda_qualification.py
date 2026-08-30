#!/usr/bin/env python3
"""Execute the C1 OpenSBLI CPU/CUDA correctness qualification."""

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
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
U4B_ADAPTER = ROOT / "experiments" / "academic_u4b" / "adapter" / "opensbli_scalar_u4b.py"
U4B_PATCH = ROOT / "experiments" / "academic_u4b" / "adapter" / "opensbli-u4b.patch"
EXPOSER = HERE / "adapter" / "expose_first_residual_binary.py"
MAKEFILE = HERE / "adapter" / "Makefile.ops"
OPENSBLI_COMMIT = "e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d"
OPENSBLI_TREE = "0ff053443f6b243b2bd42475f98122306151427d"
OPS_COMMIT = "c0af0f124469e5fd856b594a23ff1206c3e9c7a8"
OPS_TREE = "82c3fd0c0b4724c6e8474e16f730e7560845235f"
ATOL = 2.0e-12


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    commands: list[str],
    log: Path | None = None,
) -> str:
    commands.append(f"(cd {shlex.quote(str(cwd))} && {shlex.join(command)})")
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log is not None:
        log.write_text(completed.stdout)
    return completed.stdout


def git_value(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def ensure_patch(root: Path, commands: list[str], env: dict[str, str]) -> None:
    reverse = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(U4B_PATCH)],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if reverse.returncode == 0:
        commands.append(f"git -C {shlex.quote(str(root))} apply # already applied")
        return
    run(
        ["git", "apply", "--check", str(U4B_PATCH)],
        cwd=root,
        env=env,
        commands=commands,
    )
    run(
        ["git", "apply", str(U4B_PATCH)],
        cwd=root,
        env=env,
        commands=commands,
    )


def conservation(rhs: np.ndarray) -> dict[str, float | bool]:
    total = float(np.sum(rhs, dtype=np.float64))
    absolute = float(np.sum(np.abs(rhs), dtype=np.float64))
    bound = float(32.0 * np.finfo(np.float64).eps * absolute)
    return {
        "sum": total,
        "sum_abs": absolute,
        "bound": bound,
        "passed": bool(abs(total) <= bound),
    }


def write_checksums(directory: Path) -> None:
    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (directory / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.relative_to(directory)}\n" for path in files)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensbli-root", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--sympy-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    opensbli = args.opensbli_root.resolve()
    ops = args.ops_root.resolve()
    cuda = args.cuda_root.resolve()
    hdf5 = args.hdf5_root.resolve()
    work = args.work_root.resolve()
    evidence = args.evidence_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    commands = [
        f"(cd {shlex.quote(str(Path.cwd()))} && {shlex.join([sys.executable, *sys.argv])})"
    ]

    assert git_value(opensbli, "rev-parse", "HEAD") == OPENSBLI_COMMIT
    assert git_value(opensbli, "rev-parse", "HEAD^{tree}") == OPENSBLI_TREE
    assert git_value(ops, "rev-parse", "HEAD") == OPS_COMMIT
    assert git_value(ops, "rev-parse", "HEAD^{tree}") == OPS_TREE

    environment = os.environ.copy()
    environment.update(
        {
            "OPS_INSTALL_PATH": str(ops / "ops"),
            "OPS_COMPILER": "gnu",
            "CUDA_INSTALL_PATH": str(cuda),
            "HDF5_INSTALL_PATH": str(hdf5),
            "MPICXX": "g++",
            "NVCCFLAGS": (
                "-Xcompiler=-fPIC -O3 -g -std=c++11 "
                "-gencode arch=compute_120,code=sm_120"
            ),
        }
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(args.sympy_root.resolve()), str(opensbli)]
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(cuda / "lib64"), str(hdf5 / "lib"), environment.get("LD_LIBRARY_PATH", "")]
    )
    # CUDA 13 must precede the conda CUDA runtime in the final link search.
    environment["HDF5_LIB_SEQ"] = " ".join(
        [
            f"-L{cuda / 'lib64'}",
            "-lops_hdf5_seq",
            f"-L{hdf5 / 'lib'}",
            "-lhdf5_hl -lhdf5 -lz",
        ]
    )
    ensure_patch(opensbli, commands, environment)

    native_nvcc_flags = environment["NVCCFLAGS"]
    run(
        [
            "make",
            "cuda",
            "NVCC_FLAG_SET=1",
            f"NVCCFLAGS={native_nvcc_flags}",
        ],
        cwd=ops / "ops" / "c",
        env=environment,
        commands=commands,
        log=evidence / "ops_cuda_build.log",
    )

    u4b_arrays = np.load(
        ROOT / "experiments" / "academic_u4b" / "evidence" / "u4b_20260830" / "qualification_arrays.npz"
    )
    payload: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    for case in ("state_a", "state_b", "constant"):
        case_dir = work / case
        case_dir.mkdir()
        shutil.copy2(MAKEFILE, case_dir / "Makefile")
        run(
            [sys.executable, str(U4B_ADAPTER), "--case", case, "--size", "64"],
            cwd=case_dir,
            env=environment,
            commands=commands,
            log=evidence / f"{case}_generation.log",
        )
        run(
            [sys.executable, str(EXPOSER), "opensbli.cpp"],
            cwd=case_dir,
            env=environment,
            commands=commands,
        )
        hdf5_link = environment["HDF5_LIB_SEQ"]
        run(
            ["make", "opensbli_seq", f"HDF5_LIB_SEQ={hdf5_link}"],
            cwd=case_dir,
            env=environment,
            commands=commands,
            log=evidence / f"{case}_seq_build.log",
        )
        run(
            [
                "make",
                "opensbli_cuda",
                "NVCC_FLAG_SET=1",
                f"NVCCFLAGS={native_nvcc_flags}",
                f"HDF5_LIB_SEQ={hdf5_link}",
            ],
            cwd=case_dir,
            env=environment,
            commands=commands,
            log=evidence / f"{case}_cuda_build.log",
        )
        seq_path = case_dir / "seq_rhs.bin"
        cuda_path = case_dir / "cuda_rhs.bin"
        seq_env = environment | {"U4C_RHS_PATH": str(seq_path)}
        cuda_env = environment | {
            "U4C_RHS_PATH": str(cuda_path),
            "OPS_BLOCK_SIZE_X": "256",
        }
        run(
            ["./opensbli_seq"],
            cwd=case_dir,
            env=seq_env,
            commands=commands,
            log=evidence / f"{case}_seq_execution.log",
        )
        run(
            ["./opensbli_cuda"],
            cwd=case_dir,
            env=cuda_env,
            commands=commands,
            log=evidence / f"{case}_cuda_execution.log",
        )
        seq_rhs = np.fromfile(seq_path, dtype=np.float64)
        cuda_rhs = np.fromfile(cuda_path, dtype=np.float64)
        canonical = u4b_arrays[f"{case}_n64_canonical_rhs"]
        if seq_rhs.shape != (64,) or cuda_rhs.shape != (64,):
            raise RuntimeError("unexpected U4-C residual shape")
        seq_cuda = float(np.max(np.abs(seq_rhs - cuda_rhs)))
        cuda_canonical = float(np.max(np.abs(cuda_rhs - canonical)))
        record = {
            "case": case,
            "size": 64,
            "seq_cuda_max_abs_difference": seq_cuda,
            "cuda_canonical_max_abs_difference": cuda_canonical,
            "seq_finite": bool(np.all(np.isfinite(seq_rhs))),
            "cuda_finite": bool(np.all(np.isfinite(cuda_rhs))),
            "seq_conservation": conservation(seq_rhs),
            "cuda_conservation": conservation(cuda_rhs),
            "generated_source_sha256": {
                name: digest(case_dir / name)
                for name in ("opensbli.cpp", "opensbli_ops.cpp", "opensbliblock00_kernels.h")
            },
            "passed": bool(
                np.all(np.isfinite(seq_rhs))
                and np.all(np.isfinite(cuda_rhs))
                and seq_cuda <= ATOL
                and cuda_canonical <= ATOL
                and conservation(seq_rhs)["passed"]
                and conservation(cuda_rhs)["passed"]
                and (case != "constant" or np.max(np.abs(cuda_rhs)) <= ATOL)
            ),
        }
        payload[f"{case}_seq_rhs"] = seq_rhs
        payload[f"{case}_cuda_rhs"] = cuda_rhs
        payload[f"{case}_canonical_rhs"] = canonical
        records.append(record)

    decision = "cuda_correctness_qualified" if all(r["passed"] for r in records) else "cuda_correctness_excluded"
    result = {
        "schema": "gradflow.academic_u4c.cuda_qualification.v1",
        "decision": decision,
        "timing_interpretation_prohibited": True,
        "atol": ATOL,
        "sources": {
            "opensbli": {"commit": OPENSBLI_COMMIT, "tree": OPENSBLI_TREE},
            "ops": {"commit": OPS_COMMIT, "tree": OPS_TREE},
            "u4b_adapter_sha256": digest(U4B_ADAPTER),
            "u4b_patch_sha256": digest(U4B_PATCH),
            "residual_exposer_sha256": digest(EXPOSER),
        },
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "cuda_root": str(cuda),
            "nvcc": run(
                [str(cuda / "bin" / "nvcc"), "--version"],
                cwd=work,
                env=environment,
                commands=commands,
            ).splitlines()[-1],
            "cuda_target": "sm_120",
            "ops_backend": ["seq", "cuda"],
        },
        "cases": records,
    }
    np.savez(evidence / "qualification_arrays.npz", **payload)
    (evidence / "qualification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (evidence / "COMMANDS.txt").write_text("\n".join(commands) + "\n")
    write_checksums(evidence)
    print(json.dumps({"decision": decision, "cases": records}, indent=2))


if __name__ == "__main__":
    main()
