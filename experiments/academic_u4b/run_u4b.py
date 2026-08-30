#!/usr/bin/env python3
"""Build, execute, and freeze the U4-B OpenSBLI correctness qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from gradflow import weno5_rhs


HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "adapter" / "opensbli_scalar_u4b.py"
PATCH = HERE / "adapter" / "opensbli-u4b.patch"
MAKEFILE = HERE / "adapter" / "Makefile.ops"
RESIDUAL_EXPOSER = HERE / "adapter" / "expose_first_residual.py"
OPENSBLI_COMMIT = "e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d"
OPENSBLI_TREE = "0ff053443f6b243b2bd42475f98122306151427d"
OPS_COMMIT = "c0af0f124469e5fd856b594a23ff1206c3e9c7a8"
OPS_TREE = "82c3fd0c0b4724c6e8474e16f730e7560845235f"
POINTWISE_TOLERANCE = 2.0e-12
CONVERGENCE_RATE_FLOOR = 4.8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    commands: list[str],
    log_path: Path | None = None,
) -> str:
    rendered = f"(cd {shlex.quote(str(cwd))} && {shlex.join(command)})"
    commands.append(rendered)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log_path is not None:
        log_path.write_text(completed.stdout)
    return completed.stdout


def git_value(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def ensure_patch(opensbli_root: Path, commands: list[str]) -> None:
    reverse = subprocess.run(
        ["git", "-C", str(opensbli_root), "apply", "--check", "--reverse", str(PATCH)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if reverse.returncode == 0:
        commands.append(f"git -C {shlex.quote(str(opensbli_root))} apply # already applied")
        return
    run(
        ["git", "apply", "--check", str(PATCH)],
        cwd=opensbli_root,
        env=None,
        commands=commands,
    )
    run(
        ["git", "apply", str(PATCH)],
        cwd=opensbli_root,
        env=None,
        commands=commands,
    )


def state(case: str, size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64) / size
    if case == "state_a":
        return 0.4 + np.sin(2.0 * np.pi * x) + 0.1 * np.cos(6.0 * np.pi * x)
    if case == "state_b":
        return np.sin(6.0 * np.pi * x) + 0.15 * np.cos(8.0 * np.pi * x)
    if case == "constant":
        return np.full(size, 0.37, dtype=np.float64)
    if case == "sine":
        return np.sin(2.0 * np.pi * x)
    raise ValueError(case)


def conservation(rhs: np.ndarray) -> dict[str, float | bool]:
    residual_sum = float(np.sum(rhs, dtype=np.float64))
    absolute_sum = float(np.sum(np.abs(rhs), dtype=np.float64))
    bound = float(32.0 * np.finfo(np.float64).eps * absolute_sum)
    return {
        "sum": residual_sum,
        "sum_abs": absolute_sum,
        "bound": bound,
        "passed": bool(abs(residual_sum) <= bound),
    }


def read_ops_array(path: Path, name: str, size: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dataset = handle[f"opensbliblock00/{name}_B0"]
        lower_halo = -int(dataset.attrs["d_m"][0])
        return np.asarray(dataset[lower_halo : lower_halo + size], dtype=np.float64)


def sha256sums(directory: Path) -> None:
    paths = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    content = "".join(
        f"{digest(path)}  {path.relative_to(directory)}\n" for path in paths
    )
    (directory / "SHA256SUMS").write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensbli-root", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--sympy-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument(
        "--hdf5-root",
        type=Path,
        default=Path(sys.prefix),
    )
    args = parser.parse_args()

    opensbli_root = args.opensbli_root.resolve()
    ops_root = args.ops_root.resolve()
    sympy_root = args.sympy_root.resolve()
    work_root = args.work_root.resolve()
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    work_root.mkdir(parents=True, exist_ok=True)
    commands: list[str] = [
        f"(cd {shlex.quote(str(Path.cwd()))} && {shlex.join([sys.executable, *sys.argv])})"
    ]

    if git_value(opensbli_root, "rev-parse", "HEAD") != OPENSBLI_COMMIT:
        raise RuntimeError("OpenSBLI commit does not match the frozen protocol")
    if git_value(opensbli_root, "rev-parse", "HEAD^{tree}") != OPENSBLI_TREE:
        raise RuntimeError("OpenSBLI tree does not match the frozen protocol")
    if git_value(ops_root, "rev-parse", "HEAD") != OPS_COMMIT:
        raise RuntimeError("OPS commit does not match the frozen protocol")
    if git_value(ops_root, "rev-parse", "HEAD^{tree}") != OPS_TREE:
        raise RuntimeError("OPS tree does not match the frozen protocol")

    ensure_patch(opensbli_root, commands)

    ops_install = ops_root / "ops"
    build_env = os.environ.copy()
    build_env.update(
        {
            "OPS_INSTALL_PATH": str(ops_install),
            "OPS_COMPILER": "gnu",
            "HDF5_INSTALL_PATH": str(args.hdf5_root.resolve()),
            "MPICXX": "g++",
        }
    )
    runtime_library_path = str(args.hdf5_root.resolve() / "lib")
    if build_env.get("LD_LIBRARY_PATH"):
        runtime_library_path += os.pathsep + build_env["LD_LIBRARY_PATH"]
    build_env["LD_LIBRARY_PATH"] = runtime_library_path

    run(
        ["make", "seq", "hdf5_seq"],
        cwd=ops_install / "c",
        env=build_env,
        commands=commands,
        log_path=evidence / "ops_build.log",
    )

    generator_env = build_env.copy()
    generator_env["PYTHONPATH"] = os.pathsep.join(
        [str(sympy_root), str(opensbli_root)]
    )

    executions = [
        ("state_a", 64),
        ("state_b", 64),
        ("constant", 64),
        *(("sine", size) for size in (40, 80, 160, 320)),
    ]
    array_payload: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []

    for case, size in executions:
        label = f"{case}_n{size}"
        build = work_root / label
        build.mkdir(parents=True, exist_ok=False)
        shutil.copy2(MAKEFILE, build / "Makefile")
        run(
            [sys.executable, str(ADAPTER), "--case", case, "--size", str(size)],
            cwd=build,
            env=generator_env,
            commands=commands,
            log_path=evidence / f"{label}_generation.log",
        )
        run(
            [sys.executable, str(RESIDUAL_EXPOSER), "opensbli.cpp"],
            cwd=build,
            env=generator_env,
            commands=commands,
        )
        run(
            ["make", "opensbli_seq"],
            cwd=build,
            env=build_env,
            commands=commands,
            log_path=evidence / f"{label}_build.log",
        )
        run(
            ["./opensbli_seq"],
            cwd=build,
            env=build_env,
            commands=commands,
            log_path=evidence / f"{label}_execution.log",
        )

        external_state = read_ops_array(build / "opensbli_output.h5", "phi", size)
        external_rhs = read_ops_array(build / "opensbli_output.h5", "Residual0", size)
        expected_state = state(case, size)
        canonical_rhs = (
            weno5_rhs(
                torch.from_numpy(expected_state),
                1.0 / size,
                lambda value: value,
                alpha=1.0,
            )
            .detach()
            .numpy()
        )
        pointwise_error = float(np.max(np.abs(external_rhs - canonical_rhs)))
        record: dict[str, object] = {
            "case": case,
            "size": size,
            "external_finite": bool(np.all(np.isfinite(external_rhs))),
            "canonical_finite": bool(np.all(np.isfinite(canonical_rhs))),
            "state_max_abs_error": float(np.max(np.abs(external_state - expected_state))),
            "rhs_max_abs_difference": pointwise_error,
            "external_conservation": conservation(external_rhs),
            "canonical_conservation": conservation(canonical_rhs),
            "generated_source_sha256": {
                path.name: digest(path)
                for path in (
                    build / "opensbli.cpp",
                    build / "opensbliblock00_kernels.h",
                    build / "opensbli_ops.cpp",
                    build / "mpi_openmp" / "mpi_openmp_kernels.cpp",
                )
            },
        }
        if case == "sine":
            x = np.arange(size, dtype=np.float64) / size
            analytic = -2.0 * np.pi * np.cos(2.0 * np.pi * x)
            record["external_l2_error"] = float(
                np.sqrt(np.mean((external_rhs - analytic) ** 2))
            )
            record["canonical_l2_error"] = float(
                np.sqrt(np.mean((canonical_rhs - analytic) ** 2))
            )
            array_payload[f"{label}_analytic_rhs"] = analytic
        array_payload[f"{label}_state"] = expected_state
        array_payload[f"{label}_external_rhs"] = external_rhs
        array_payload[f"{label}_canonical_rhs"] = canonical_rhs
        records.append(record)

    sine_records = [record for record in records if record["case"] == "sine"]
    for previous, current in zip(sine_records, sine_records[1:]):
        current["external_rate_from_previous"] = math.log2(
            float(previous["external_l2_error"]) / float(current["external_l2_error"])
        )
        current["canonical_rate_from_previous"] = math.log2(
            float(previous["canonical_l2_error"]) / float(current["canonical_l2_error"])
        )

    pointwise = [record for record in records if record["case"] in ("state_a", "state_b")]
    all_finite = all(
        bool(record["external_finite"]) and bool(record["canonical_finite"])
        for record in records
    )
    all_conservative = all(
        bool(record[side]["passed"])
        for record in records
        for side in ("external_conservation", "canonical_conservation")
    )
    gates = {
        "q1_pointwise": all(
            float(record["rhs_max_abs_difference"]) <= POINTWISE_TOLERANCE
            and float(record["state_max_abs_error"]) <= POINTWISE_TOLERANCE
            for record in pointwise
        ),
        "q2_constant": bool(
            float(np.max(np.abs(array_payload["constant_n64_external_rhs"])))
            <= POINTWISE_TOLERANCE
        ),
        "q3_conservation": all_conservative,
        "q4_convergence": all_finite
        and all(
            float(record["external_rate_from_previous"]) > CONVERGENCE_RATE_FLOOR
            for record in sine_records[1:]
        ),
    }
    decision = (
        "matched_operator_adapted_qualified"
        if all(gates.values())
        else "correctness_excluded"
    )

    sympy_version = next(
        distribution.version
        for distribution in importlib.metadata.distributions(path=[str(sympy_root)])
        if distribution.metadata["Name"].lower() == "sympy"
    )
    compiler = run(
        ["g++", "--version"], cwd=work_root, env=build_env, commands=commands
    ).splitlines()[0]

    qualification = {
        "schema": "gradflow.academic_u4b.qualification.v1",
        "decision": decision,
        "performance_interpretation_prohibited": True,
        "protocol": {
            "pointwise_atol": POINTWISE_TOLERANCE,
            "convergence_rate_floor": CONVERGENCE_RATE_FLOOR,
            "dtype": "float64",
            "flux": "f(u)=u",
            "lf_path": "OpenSBLI native LLF; constant scalar eigenvalue gives alpha=1 at every face",
            "epsilon": {
                "gradflow_12_scaled": 1.0e-29,
                "opensbli_standard": 1.0e-29 / 12.0,
            },
            "residual_exposure": "generated native residual exported immediately after its first evaluation; process exits before any RK update",
        },
        "upstream": {
            "opensbli": {"commit": OPENSBLI_COMMIT, "tree": OPENSBLI_TREE},
            "ops": {"commit": OPS_COMMIT, "tree": OPS_TREE},
        },
        "source_hashes": {
            "adapter": digest(ADAPTER),
            "adapter_patch": digest(PATCH),
            "residual_exposer": digest(RESIDUAL_EXPOSER),
        },
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "sympy": sympy_version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "h5py": h5py.__version__,
            "compiler": compiler,
            "ops_backend": "seq",
        },
        "gates": gates,
        "cases": records,
    }
    weno_preimage = subprocess.check_output(
        [
            "git",
            "-C",
            str(opensbli_root),
            "show",
            f"{OPENSBLI_COMMIT}:opensbli/schemes/spatial/weno.py",
        ]
    )
    qualification["source_hashes"]["opensbli_weno_preimage"] = hashlib.sha256(
        weno_preimage
    ).hexdigest()
    shock_preimage = subprocess.check_output(
        [
            "git",
            "-C",
            str(opensbli_root),
            "show",
            f"{OPENSBLI_COMMIT}:opensbli/schemes/spatial/shock_capturing.py",
        ]
    )
    qualification["source_hashes"]["opensbli_shock_capturing_preimage"] = (
        hashlib.sha256(shock_preimage).hexdigest()
    )

    np.savez(evidence / "qualification_arrays.npz", **array_payload)
    (evidence / "qualification.json").write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n"
    )
    (evidence / "COMMANDS.txt").write_text("\n".join(commands) + "\n")
    sha256sums(evidence)
    print(json.dumps({"decision": decision, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
