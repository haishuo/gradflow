#!/usr/bin/env python3
"""Verify the Trunk 005 handoff and correctness-qualify all U4-E lanes."""

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
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
U4C = ROOT / "experiments" / "academic_u4c"
U4D = ROOT / "experiments" / "academic_u4d"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(U4C))

from run_performance import (  # noqa: E402
    MAXIMUM_LIMIT,
    OPENSBLI_COMMIT,
    OPENSBLI_TREE,
    OPS_COMMIT,
    OPS_TREE,
    RMS_LIMIT,
    comparison,
    digest,
    execute,
    parse_json,
    require,
    write_checksums,
)


PROTOCOL_COMMIT = "4788585cecd3a765e452406e08e4de5788ae7f0b"
DVEB_CLOSURE = "39bd1c323daa3dbce6421a09dc34dc0cd2109d88"
DVEB_CLOSURE_TREE = "3711d334ee48f24717900456f17c6518a1f0bada"
BUNDLE_SHA256 = "2342f66416b1b120efd42e0e4ca8838f32cef4c62a13bf43042fb12ef7354ae0"
LIBRARY_SHA256 = "9ff9172b1ac712b8bc97ca9523fd114b2637e5d7825259371ba9850459168443"
SIZE = 8192
DRIVER = HERE / "adapter" / "dveb_u4e_abi_driver.cpp"
GRADFLOW_WORKER = U4C / "gradflow_worker.py"
U4D_QUALIFICATION = U4D / "evidence" / "u4d_d1_20260830" / "qualification.json"


def git_value(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


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


def parse_policy(stdout: str, kind: str) -> dict[str, int | str]:
    prefix = f"U4E_{kind} "
    rows = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise RuntimeError(f"expected one {prefix.strip()} row, found {len(rows)}")
    fields: dict[str, int | str] = {}
    for item in rows[0].split()[1:]:
        key, value = item.split("=", 1)
        fields[key] = value if key == "target" else int(value)
    return fields


def verify_policy(query: dict[str, Any], run: dict[str, Any], device: str) -> None:
    expected = {
        "cpu": {
            "target": "cpu", "cpu_loop": 2, "cuda_block": 0, "reuse": 2,
            "launches": 2, "scratch_bytes": 65584, "elements": SIZE,
        },
        "cuda": {
            "target": "cuda", "cpu_loop": 0, "cuda_block": 32, "reuse": 2,
            "launches": 2, "scratch_bytes": 65584, "elements": SIZE,
        },
    }[device]
    for key, value in expected.items():
        if query.get(key) != value or run.get(key) != value:
            raise RuntimeError(f"unexpected DVEB auto policy for {device}: {key}")
    if query["synchronized"] != 0:
        raise RuntimeError("DVEB query unexpectedly reported synchronization")
    wanted_run_sync = 1 if device == "cpu" else 0
    if run["synchronized"] != wanted_run_sync:
        raise RuntimeError(f"unexpected DVEB {device} synchronization field")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dveb-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    dveb = args.dveb_root.resolve()
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

    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("U4-E protocol commit is not an ancestor of HEAD")
    if git_value(dveb, "rev-parse", "HEAD") != DVEB_CLOSURE:
        raise RuntimeError("DVEB checkout differs from frozen Trunk 005 closure")
    if git_value(dveb, "rev-parse", "HEAD^{tree}") != DVEB_CLOSURE_TREE:
        raise RuntimeError("DVEB closure tree mismatch")
    if git_value(dveb, "status", "--porcelain"):
        raise RuntimeError("DVEB checkout is not clean")

    manifest_path = dveb / "evidence" / "trunk005" / "handoff" / "manifest.json"
    bundle_path = dveb / "evidence" / "trunk005" / "handoff" / "weno5-schedule-v1.tar.gz"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "dveb-gradflow-scalar-handoff-v1":
        raise RuntimeError("unexpected DVEB handoff schema")
    if digest(bundle_path) != BUNDLE_SHA256 or manifest["bundle"]["sha256"] != BUNDLE_SHA256:
        raise RuntimeError("DVEB handoff bundle hash mismatch")
    if manifest["artifacts"]["library"]["sha256"] != LIBRARY_SHA256:
        raise RuntimeError("DVEB handoff library manifest mismatch")
    language_source = dveb / manifest["implementation"]["language_source"]
    if digest(language_source) != manifest["implementation"]["language_source_sha256"]:
        raise RuntimeError("DVEB language source hash mismatch")

    handoff = work / "handoff"
    handoff.mkdir()
    copied_bundle = handoff / bundle_path.name
    preparation: dict[str, float] = {}
    started = time.perf_counter()
    shutil.copy2(bundle_path, copied_bundle)
    preparation["handoff_copy_seconds"] = time.perf_counter() - started
    if digest(copied_bundle) != BUNDLE_SHA256:
        raise RuntimeError("copied handoff bundle hash mismatch")

    expected_members = {
        manifest["artifacts"]["library"]["name"]: LIBRARY_SHA256,
        manifest["artifacts"]["header"]["name"]: manifest["artifacts"]["header"]["sha256"],
        **manifest["artifacts"]["generated_sources"],
    }
    extracted = handoff / "extracted"
    extracted.mkdir()
    started = time.perf_counter()
    with tarfile.open(copied_bundle, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        if set(members) != set(expected_members):
            raise RuntimeError("DVEB handoff member set mismatch")
        for name, wanted in expected_members.items():
            if Path(name).name != name:
                raise RuntimeError("unsafe handoff member name")
            source = archive.extractfile(members[name])
            if source is None:
                raise RuntimeError(f"unable to read handoff member {name}")
            destination = extracted / name
            destination.write_bytes(source.read())
            if digest(destination) != wanted:
                raise RuntimeError(f"handoff member hash mismatch: {name}")
    preparation["handoff_verify_and_extract_seconds"] = time.perf_counter() - started

    build_env = os.environ.copy()
    for language, compiler, standard in (("c", "gcc", "c11"), ("c++", "g++", "c++17")):
        completed, elapsed = execute(
            [compiler, f"-std={standard}", "-I", str(extracted), "-include",
             "weno5_schedule_abi_v1.h", "-x", language, "-fsyntax-only", "/dev/null"],
            cwd=work, env=build_env, commands=commands,
        )
        require(completed, f"DVEB public header {standard} compile")
        preparation[f"header_{standard}_compile_seconds"] = elapsed
        (raw / f"header_{standard}.stdout").write_text(completed.stdout)
        (raw / f"header_{standard}.stderr").write_text(completed.stderr)

    executable = handoff / "u4e_dveb"
    completed, elapsed = execute(
        [
            "g++", "-O3", "-std=c++17", "-fopenmp", "-I", str(extracted),
            str(DRIVER), str(extracted / "weno5_schedule_abi_v1.so"),
            "-L", str(cuda / "lib64"), "-lcudart",
            f"-Wl,-rpath,{extracted}", f"-Wl,-rpath,{cuda / 'lib64'}",
            "-o", str(executable),
        ],
        cwd=work, env=build_env, commands=commands,
    )
    require(completed, "U4-E DVEB ABI adapter build")
    preparation["abi_adapter_build_seconds"] = elapsed
    (raw / "adapter_build.stdout").write_text(completed.stdout)
    (raw / "adapter_build.stderr").write_text(completed.stderr)

    u4d = json.loads(U4D_QUALIFICATION.read_text())
    artifacts = u4d["artifacts"]
    opensbli_root = Path(artifacts["opensbli_root"])
    opensbli_executables = {
        "cpu": Path(artifacts["opensbli_cpu_executable"]),
        "cuda": Path(artifacts["opensbli_cuda_executable"]),
    }
    for device, path in opensbli_executables.items():
        wanted = artifacts[f"opensbli_{device}_sha256"]
        if not path.is_file() or digest(path) != wanted:
            raise RuntimeError(f"qualified U4-D OpenSBLI {device} artifact unavailable")

    c2_arrays = U4C / "evidence" / "u4c_c2_20260830" / "qualification_arrays"
    state_path = c2_arrays / f"n{SIZE}_state.bin"
    canonical_path = c2_arrays / f"n{SIZE}_canonical.bin"
    state = np.fromfile(state_path, dtype=np.float64)
    canonical = np.fromfile(canonical_path, dtype=np.float64)
    if state.shape != (SIZE,) or canonical.shape != (SIZE,):
        raise RuntimeError("frozen U4-C arrays unavailable")
    if digest(state_path) != "7def0f1a410959390af68416a01f92d0ec917a23aaf022f5b90d52c366bb5530":
        raise RuntimeError("frozen state hash mismatch")
    if digest(canonical_path) != "d92a1dd5f20cba9533dd25682fd19ca2d39f584b883b9fee3c994f1dd46b3621":
        raise RuntimeError("canonical RHS hash mismatch")

    native_env = os.environ.copy()
    native_env.update({
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "LD_LIBRARY_PATH": os.pathsep.join(
            [str(cuda / "lib64"), str(hdf5 / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
        ),
    })
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
                command = [
                    str(executable), "--size", str(SIZE), "--backend", device,
                    "--mode", "qualify", "--input", str(state_path), "--output", str(output),
                ]
                completed, elapsed = execute(
                    command, cwd=handoff, env=native_env, commands=commands
                )
            elif implementation == "opensbli":
                lane_env = native_env | {
                    "U4C_MODE": "qualify", "U4C_STATE_PATH": str(state_path),
                    "U4C_RHS_PATH": str(output), "OPS_BLOCK_SIZE_X": "256",
                }
                completed, elapsed = execute(
                    [str(opensbli_executables[device])], cwd=opensbli_root,
                    env=lane_env, commands=commands,
                )
            else:
                completed, elapsed = execute(
                    [sys.executable, str(GRADFLOW_WORKER), "--size", str(SIZE),
                     "--device", device, "--mode", "qualify", "--input", str(state_path),
                     "--output", str(output)],
                    cwd=ROOT, env=grad_env, commands=commands,
                )
            require(completed, f"U4-E {lane} qualification")
            (raw / f"{lane}.stdout").write_text(completed.stdout)
            (raw / f"{lane}.stderr").write_text(completed.stderr)
            values = np.fromfile(output, dtype=np.float64)
            if values.shape != canonical.shape:
                raise RuntimeError(f"U4-E {lane} output shape mismatch")
            candidates[lane] = values
            metadata[lane] = {"process_seconds": elapsed}
            if implementation == "dveb":
                query = parse_policy(completed.stdout, "QUERY")
                run = parse_policy(completed.stdout, "RUN")
                verify_policy(query, run, device)
                metadata[lane].update({"query": query, "run": run})
            elif implementation == "gradflow":
                worker = parse_json(completed.stdout)
                if worker["graph"] != {"unique_graphs": 1, "graph_break_count": 0}:
                    raise RuntimeError(f"U4-E {lane} graph gate failed")
                metadata[lane]["worker"] = worker
                preparation[f"{lane}_first_call_seconds"] = worker["first_call_seconds"]

    qualification = {
        lane: {**comparison(values, canonical), "sha256": digest(arrays / f"{lane}.bin"),
               "metadata": metadata[lane]}
        for lane, values in candidates.items()
    }
    for device in ("cpu", "cuda"):
        qualification[f"dveb_{device}"]["versus_gradflow_same_device"] = comparison(
            candidates[f"dveb_{device}"], candidates[f"gradflow_{device}"]
        )
    qualification["dveb_cpu_cuda"] = comparison(
        candidates["dveb_cpu"], candidates["dveb_cuda"]
    )
    all_admitted = bool(
        all(qualification[lane]["passed"] for lane in candidates)
        and qualification["dveb_cpu_cuda"]["passed"]
    )
    decision = "all_six_lanes_qualified" if all_admitted else "correctness_excluded"

    record = {
        "schema": "gradflow.academic_u4e.qualification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "timing_interpretation_prohibited": True,
        "protocol_commit": PROTOCOL_COMMIT,
        "size": SIZE,
        "bounds": {"maximum_normalized": MAXIMUM_LIMIT, "rms_normalized": RMS_LIMIT},
        "canonical": {
            "state_sha256": digest(state_path), "rhs_sha256": digest(canonical_path),
            "finite": bool(np.all(np.isfinite(canonical))),
            "conservation": conservation(canonical),
        },
        "qualification": qualification,
        "preparation": preparation,
        "sources": {
            "dveb_closure": {"commit": DVEB_CLOSURE, "tree": DVEB_CLOSURE_TREE},
            "dveb_handoff_manifest": manifest,
            "opensbli": {"commit": OPENSBLI_COMMIT, "tree": OPENSBLI_TREE},
            "ops": {"commit": OPS_COMMIT, "tree": OPS_TREE},
            "driver_sha256": digest(DRIVER),
        },
        "handoff": {
            "bundle_sha256": digest(copied_bundle),
            "members": {name: digest(extracted / name) for name in sorted(expected_members)},
            "abi_version": 1,
        },
        "artifacts": {
            "work_root": str(work), "dveb_executable": str(executable),
            "dveb_executable_sha256": digest(executable),
            "dveb_library": str(extracted / "weno5_schedule_abi_v1.so"),
            "dveb_library_sha256": digest(extracted / "weno5_schedule_abi_v1.so"),
            "opensbli_root": str(opensbli_root),
            "opensbli_cpu_executable": str(opensbli_executables["cpu"]),
            "opensbli_cpu_sha256": digest(opensbli_executables["cpu"]),
            "opensbli_cuda_executable": str(opensbli_executables["cuda"]),
            "opensbli_cuda_sha256": digest(opensbli_executables["cuda"]),
            "torchinductor_cache": str(work / "torchinductor_cache"),
        },
        "environment": {
            "host": platform.node(), "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "cpu_threads": 1,
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
