#!/usr/bin/env python3
"""Resolve Phase-5C accumulated-roundoff eligibility without new timing."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from gradflow import (
    burgers_fd_weno5_rhs,
    burgers_fv_weno5_rhs,
    ssp_rk3_step,
)
from experiments.fd_fv_nonlinear.burgers_oracle import FINAL_TIME, LF_ALPHA
from experiments.fd_fv_nonlinear.performance_problem import (
    METHOD_IDS,
    errors,
    solve,
    state,
    step_function,
    timestep,
)
from experiments.fd_fv_nonlinear.run_phase5c import (
    COLD_SIZES,
    ERROR_TARGETS,
    METHODS,
    MODES,
    aggregate_complete,
    classification,
    target_selections,
)


PROTOCOL_COMMIT = "5ab6950"
PROTOCOL = ROOT / "docs/FD_FV_PHASE_5CR_PROTOCOL.md"
INITIAL_RESULTS = (
    ROOT / "experiments/fd_fv_nonlinear/results/phase_5c_20260828"
)
INITIAL_RECORD = INITIAL_RESULTS / "benchmark.json"
INITIAL_VERIFY = ROOT / "experiments/fd_fv_nonlinear/verify_phase5c_initial.py"
NUMERICAL_SOURCES = (
    ROOT / "src/gradflow/burgers.py",
    ROOT / "src/gradflow/weno_js.py",
    ROOT / "src/gradflow/fv_weno5.py",
    ROOT / "src/gradflow/weno5.py",
)
DIAGNOSTIC_SIZES = (81, 162)
METHOD_RHS = {
    "fd": burgers_fd_weno5_rhs,
    "fv": burgers_fv_weno5_rhs,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def single_bound(initial: torch.Tensor, cells: int) -> float:
    dx = 1.0 / cells
    return float(
        64.0
        * torch.finfo(torch.float64).eps
        * dx
        * torch.sum(torch.abs(initial))
        + 2.0e-15
    )


def accumulated_bound(bound: float, steps: int) -> float:
    return steps * (bound - 2.0e-15) + 2.0e-15


def mass_metrics(
    initial: torch.Tensor, final: torch.Tensor, cells: int
) -> dict[str, Any]:
    dx = 1.0 / cells
    device_reduction = float(torch.abs(dx * torch.sum(final - initial)))
    host_initial = initial.detach().cpu()
    host_final = final.detach().cpu()
    host_tensor = float(torch.abs(dx * torch.sum(host_final - host_initial)))
    host_fsum = abs(dx * math.fsum((host_final - host_initial).tolist()))
    separate_fsum = abs(
        dx
        * (
            math.fsum(host_final.tolist())
            - math.fsum(host_initial.tolist())
        )
    )
    values = (device_reduction, host_tensor, host_fsum, separate_fsum)
    agreement = max(values) - min(values)
    return {
        "device_reduction": device_reduction,
        "host_tensor_reduction": host_tensor,
        "host_fsum_difference": host_fsum,
        "host_separate_fsum_difference": separate_fsum,
        "maximum_reduction_disagreement": agreement,
        "reduction_tolerance": 2.0e-16,
        "passed": agreement <= 2.0e-16,
    }


def method_size_diagnostic(method: str, cells: int) -> dict[str, Any]:
    rhs = METHOD_RHS[method]
    dx = 1.0 / cells
    dt, steps = timestep(cells)
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    records: dict[str, Any] = {}
    for device in ("cpu", "cuda"):
        initial = state(method, cells).to(device)
        expected = state(method, cells, FINAL_TIME).to(device)
        rhs_value = rhs(initial, dx, LF_ALPHA)
        rhs_residual = float(torch.abs(dx * torch.sum(rhs_value)))
        rhs_bound = float(
            64.0
            * torch.finfo(torch.float64).eps
            * dx
            * torch.sum(torch.abs(rhs_value))
            + 2.0e-15
        )
        eager_step = step_function(method, cells)
        torch._dynamo.reset()
        compiled_step = torch.compile(eager_step, fullgraph=True, dynamic=False)
        device_outputs = {}
        device_records = {}
        for mode, step in (("eager", eager_step), ("compiled", compiled_step)):
            one = step(initial)
            final = solve(initial, step, steps)
            if device == "cuda":
                torch.cuda.synchronize()
            one_mass = mass_metrics(initial, one, cells)
            full_mass = mass_metrics(initial, final, cells)
            bound = single_bound(initial, cells)
            accumulated = accumulated_bound(bound, steps)
            l1, l2 = errors(final, expected)
            device_outputs[mode] = final
            device_records[mode] = {
                "one_step_mass": one_mass,
                "one_step_bound": bound,
                "one_step_conservation_passed": (
                    one_mass["host_fsum_difference"] <= bound
                ),
                "full_solve_mass": full_mass,
                "steps": steps,
                "single_bound": bound,
                "accumulated_bound": accumulated,
                "mass_per_step": full_mass["host_fsum_difference"] / steps,
                "accumulated_bound_utilization": (
                    full_mass["host_fsum_difference"] / accumulated
                ),
                "per_step_bound_utilization": (
                    full_mass["host_fsum_difference"] / steps / bound
                ),
                "l1_error": l1,
                "l2_error": l2,
                "finite": bool(torch.isfinite(final).all()),
                "passed": one_mass["passed"]
                and full_mass["passed"]
                and one_mass["host_fsum_difference"] <= bound
                and full_mass["host_fsum_difference"] <= accumulated
                and full_mass["host_fsum_difference"] / steps <= bound
                and math.isfinite(l1)
                and math.isfinite(l2)
                and bool(torch.isfinite(final).all()),
            }
        parity = float(
            torch.max(
                torch.abs(device_outputs["compiled"] - device_outputs["eager"])
            )
        )
        device_records["rhs_mass_residual"] = rhs_residual
        device_records["rhs_mass_bound"] = rhs_bound
        device_records["rhs_conservation_passed"] = rhs_residual <= rhs_bound
        device_records["compiled_eager_maximum_absolute_difference"] = parity
        device_records["passed"] = (
            rhs_residual <= rhs_bound
            and parity <= 2.0e-11
            and all(device_records[mode]["passed"] for mode in MODES)
        )
        outputs[device] = device_outputs
        records[device] = device_records

    cpu_cuda = {}
    for mode in MODES:
        difference = float(
            torch.max(
                torch.abs(outputs["cuda"][mode].cpu() - outputs["cpu"][mode])
            )
        )
        cpu_cuda[mode] = {
            "maximum_absolute_difference": difference,
            "tolerance": 2.0e-11,
            "passed": difference <= 2.0e-11,
        }
    return {
        "method": method,
        "cells": cells,
        "dt_hex": dt.hex(),
        "steps": steps,
        "devices": records,
        "cpu_cuda": cpu_cuda,
        "passed": all(record["passed"] for record in records.values())
        and all(case["passed"] for case in cpu_cuda.values()),
    }


def original_complete_nonconservation_gates(record: dict[str, Any]) -> bool:
    eager = record["accuracy"]["eager"]
    compiled = record["accuracy"]["compiled"]
    expected_device = record["device"]
    return (
        record["status"] == "completed"
        and record["worker_returncode"] == 0
        and record["kind"] == "complete"
        and record["formulation_id"] == METHOD_IDS[record["method"]]
        and eager["finite"]
        and compiled["finite"]
        and math.isfinite(eager["l1_error"])
        and math.isfinite(eager["l2_error"])
        and math.isfinite(compiled["l1_error"])
        and math.isfinite(compiled["l2_error"])
        and record["accuracy"]["compiled_eager_maximum_absolute_difference"]
        <= 2.0e-11
        and record["accuracy"]["compiled_repeat_maximum_absolute_difference"]
        == 0.0
        and eager["dtype"] == "float64"
        and compiled["dtype"] == "float64"
        and eager["shape"] == [record["cells"]]
        and compiled["shape"] == [record["cells"]]
        and eager["device"].split(":")[0] == expected_device
        and compiled["device"].split(":")[0] == expected_device
    )


def original_cold_nonconservation_gates(record: dict[str, Any]) -> bool:
    return (
        record["status"] == "completed"
        and record["worker_returncode"] == 0
        and record["kind"] == "cold"
        and record["formulation_id"] == METHOD_IDS[record["method"]]
        and record["finite"]
        and math.isfinite(record["l1_error"])
        and math.isfinite(record["l2_error"])
        and record["host_visible_answer"]
    )


def resolve_complete(
    records: list[dict[str, Any]], diagnostics: dict[tuple[str, int], dict]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    copies = deepcopy(records)
    for original, copy in zip(records, copies):
        bound = original["accuracy"]["eager"]["mass_bound"]
        steps = original["steps"]
        accumulated = accumulated_bound(bound, steps)
        modes = {}
        for mode in MODES:
            change = original["accuracy"][mode]["mass_change"]
            modes[mode] = {
                "mass_change": change,
                "single_bound": bound,
                "accumulated_bound": accumulated,
                "mass_per_step": change / steps,
                "accumulated_bound_utilization": change / accumulated,
                "per_step_bound_utilization": change / steps / bound,
                "passed": change <= accumulated and change / steps <= bound,
            }
        diagnostic_required = (
            original["device"] == "cuda" and original["cells"] in DIAGNOSTIC_SIZES
        )
        diagnostic_passed = (
            diagnostics[(original["method"], original["cells"])]["passed"]
            if diagnostic_required
            else True
        )
        eligible = (
            original_complete_nonconservation_gates(original)
            and all(item["passed"] for item in modes.values())
            and diagnostic_passed
        )
        copy["eligible"] = eligible
        copy["eligible_under_phase_5cr"] = eligible
        summaries.append(
            {
                "method": original["method"],
                "device": original["device"],
                "cells": original["cells"],
                "replicate": original["replicate"],
                "original_eligible": original["eligible"],
                "modes": modes,
                "diagnostic_required": diagnostic_required,
                "diagnostic_passed": diagnostic_passed,
                "eligible_under_phase_5cr": eligible,
            }
        )
    return summaries, copies


def resolve_cold(
    records: list[dict[str, Any]], diagnostics: dict[tuple[str, int], dict]
) -> list[dict[str, Any]]:
    summaries = []
    for record in records:
        bound = record["mass_bound"]
        steps = record["steps"]
        accumulated = accumulated_bound(bound, steps)
        change = record["mass_change"]
        diagnostic_required = (
            record["device"] == "cuda" and record["cells"] in DIAGNOSTIC_SIZES
        )
        diagnostic_passed = (
            diagnostics[(record["method"], record["cells"])]["passed"]
            if diagnostic_required
            else True
        )
        eligible = (
            original_cold_nonconservation_gates(record)
            and change <= accumulated
            and change / steps <= bound
            and diagnostic_passed
        )
        summaries.append(
            {
                "method": record["method"],
                "device": record["device"],
                "mode": record["mode"],
                "cells": record["cells"],
                "original_eligible": record["eligible"],
                "mass_change": change,
                "single_bound": bound,
                "accumulated_bound": accumulated,
                "mass_per_step": change / steps,
                "accumulated_bound_utilization": change / accumulated,
                "per_step_bound_utilization": change / steps / bound,
                "diagnostic_required": diagnostic_required,
                "diagnostic_passed": diagnostic_passed,
                "eligible_under_phase_5cr": eligible,
            }
        )
    return summaries


def cold_target_selections(
    records: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    eligibility = {
        (item["method"], item["device"], item["mode"], item["cells"]): item[
            "eligible_under_phase_5cr"
        ]
        for item in summaries
    }
    result: dict[str, Any] = {}
    for device in ("cpu", "cuda"):
        boundary = {}
        for target in ERROR_TARGETS:
            entry: dict[str, Any] = {}
            for method in METHODS:
                candidates = [
                    record
                    for record in records
                    if record["method"] == method
                    and record["device"] == device
                    and record["l2_error"] <= target
                    and eligibility[
                        (
                            record["method"],
                            record["device"],
                            record["mode"],
                            record["cells"],
                        )
                    ]
                ]
                if candidates:
                    selected = min(
                        candidates,
                        key=lambda item: item["process_launch_to_exit_seconds"],
                    )
                    entry[method] = {
                        "status": "reached",
                        "cells": selected["cells"],
                        "mode": selected["mode"],
                        "l2_error": selected["l2_error"],
                        "seconds": selected["process_launch_to_exit_seconds"],
                    }
                else:
                    entry[method] = {"status": "not_reached"}
            if all(entry[method]["status"] == "reached" for method in METHODS):
                ratio = entry["fv"]["seconds"] / entry["fd"]["seconds"]
                entry["fv_over_fd_ratio"] = ratio
                entry["classification"] = (
                    "unresolved_cold_pilot"
                    if 1.0 / 1.10 <= ratio <= 1.10
                    else classification(ratio)
                )
            boundary[str(target)] = entry
        result[f"cold_{device}"] = boundary
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    if source_dirty:
        raise RuntimeError("Phase 5CR requires a clean committed source tree")
    verification = subprocess.run(
        (sys.executable, str(INITIAL_VERIFY)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.returncode:
        raise RuntimeError("initial Phase 5C record failed verification")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 5CR requires Forge CUDA visibility")

    initial = json.loads(INITIAL_RECORD.read_text())
    diagnostic_records = [
        method_size_diagnostic(method, cells)
        for method in METHODS
        for cells in DIAGNOSTIC_SIZES
    ]
    diagnostics = {
        (record["method"], record["cells"]): record
        for record in diagnostic_records
    }
    complete_summaries, resolved_copies = resolve_complete(
        initial["complete_records"], diagnostics
    )
    resolved_aggregates = aggregate_complete(resolved_copies)
    cold_summaries = resolve_cold(initial["cold_records"], diagnostics)
    gates = {
        "initial_record_verified": True,
        "fresh_mechanistic_diagnostics": all(
            record["passed"] for record in diagnostic_records
        ),
        "all_complete_cells_resolved": all(
            item["eligible_under_phase_5cr"] for item in complete_summaries
        ),
        "all_cold_cells_resolved": all(
            item["eligible_under_phase_5cr"] for item in cold_summaries
        ),
        "original_step_cells_eligible": initial["all_step_cells_eligible"],
    }
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_nonlinear_phase_5cr",
        "resolution_date": "2026-08-28",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "initial_phase_5c": {
            "record": str(INITIAL_RECORD.relative_to(ROOT)),
            "record_sha256": sha256(INITIAL_RECORD),
            "manifest_sha256": sha256(INITIAL_RESULTS / "SHA256SUMS"),
            "verification_stdout": verification.stdout.strip(),
            "original_complete_gate_passed": initial[
                "all_complete_cells_eligible"
            ],
            "original_cold_gate_passed": initial["all_cold_cells_eligible"],
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PROTOCOL, INITIAL_VERIFY, *NUMERICAL_SOURCES, Path(__file__))
        },
        "bound": {
            "formula": "steps*(single_bound-2e-15)+2e-15",
            "per_step_requirement": "mass_change/steps <= single_bound",
            "fitted_to_observation": False,
        },
        "fresh_diagnostics": diagnostic_records,
        "complete_reclassification": complete_summaries,
        "resolved_complete_aggregates": resolved_aggregates,
        "resolved_target_selections": target_selections(resolved_aggregates),
        "cold_reclassification": cold_summaries,
        "resolved_cold_target_selections": cold_target_selections(
            initial["cold_records"], cold_summaries
        ),
        "preserved_step_device_crossovers": initial[
            "step_device_crossovers"
        ],
        "gate_decisions": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "passed": all(gates.values()),
        "performance_measurements_collected": False,
        "performance_samples_reused_unchanged": True,
        "implementation_changed": False,
    }
    output.mkdir(parents=True)
    record_path = output / "resolution.json"
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "SHA256SUMS").write_text(
        f"{sha256(record_path)}  resolution.json\n"
    )
    print(f"wrote Phase 5CR resolution to {record_path}")
    print(f"passed={payload['passed']} failed_gates={payload['failed_gates']}")


if __name__ == "__main__":
    main()
