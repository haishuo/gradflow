#!/usr/bin/env python3
"""Create compact, reproducible tables from the DVEB-inclusive bakeoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LANES = ("fortran", "dveb-auto", "direct-eager", "aot-inductor")
EXPERIMENT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def milliseconds(value: float) -> str:
    return f"{1000.0 * value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args()
    paths = sorted(arguments.input_dir.glob("counted_n*_s*.json"))
    if len(paths) != 9:
        raise SystemExit(f"expected nine counted result files, found {len(paths)}")

    rows: list[dict[str, object]] = []
    for path in paths:
        result = json.loads(path.read_text())
        medians = {
            lane: float(result["summary"][lane]["fresh_process_seconds"]["median"])
            for lane in LANES
        }
        winner = min(medians, key=medians.get)
        competitors = {key: value for key, value in medians.items() if key != "dveb-auto"}
        next_best = min(competitors, key=competitors.get)
        selections = sorted({
            record["selected"] for record in result["records"]
            if record["lane"] == "dveb-auto" and record["success"]
        })
        paired = []
        for repetition in range(result["repetitions"]):
            block = {
                record["lane"]: float(record["fresh_process_seconds"])
                for record in result["records"] if record["repetition"] == repetition
            }
            paired.append(block["dveb-auto"] - block[next_best])
        row = {
            "size": result["size"],
            "steps": result["steps"],
            "cells": (result["size"] + 1) ** 3,
            "fresh_process_median_seconds": medians,
            "winner": winner,
            "dveb_selected": selections,
            "dveb_speedup_over_next_best": competitors[next_best] / medians["dveb-auto"],
            "next_best_non_dveb": next_best,
            "dveb_paired_wins": sum(value < 0.0 for value in paired),
            "dveb_paired_losses": sum(value > 0.0 for value in paired),
            "raw_file": str(path.resolve()),
            "raw_sha256": sha256(path),
        }
        rows.append(row)
    rows.sort(key=lambda row: (int(row["steps"]), int(row["size"])))

    manifest_path = arguments.input_dir / "manifest.json"
    calibration_path = arguments.input_dir / "calibration.json"
    correctness_path = arguments.input_dir / "correctness.json"
    manifest = json.loads(manifest_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    correctness = json.loads(correctness_path.read_text())
    report = {
        "schema_version": 1,
        "endpoint": "external fresh process through final pageable-host materialization",
        "repetitions_per_lane_per_point": 30,
        "rows": rows,
        "correctness_maximum_error": correctness["maximum_error"],
        "correctness_bound": correctness["bound"],
        "dveb_artifact_sha256": manifest["native"]["dveb"]["sha256"],
        "dveb_model_sha256": calibration["model_sha256"],
        "input_hashes": {
            "manifest": sha256(manifest_path),
            "calibration": sha256(calibration_path),
            "correctness": sha256(correctness_path),
        },
        "gradflow_source_hashes": {
            name: sha256(EXPERIMENT / name)
            for name in (
                "shu_euler_torch.py",
                "bakeoff_worker.py",
                "build_aot_package.py",
                "fortran/shu_euler_3d.f90",
                "build/shu_euler_3d",
                "DVEB_BAKEOFF_PROTOCOL.md",
                "run_dveb_bakeoff.py",
            )
        },
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Automatic DVEB matched bakeoff results",
        "",
        "All values below are medians of 30 randomized-order fresh processes, in",
        "milliseconds. The timed endpoint ends after complete state materialization",
        "in pageable host memory. AOT build/calibration are excluded preparation",
        "costs and are reported separately in the artifact manifest.",
        "",
        "| N | Steps | Cells | Fortran | DVEB auto | PyTorch eager | PyTorch AOT | DVEB target | Winner | DVEB vs next best |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|",
    ]
    for row in rows:
        values = row["fresh_process_median_seconds"]
        lines.append(
            f"| {row['size']} | {row['steps']} | {row['cells']:,} | "
            f"{milliseconds(values['fortran'])} | {milliseconds(values['dveb-auto'])} | "
            f"{milliseconds(values['direct-eager'])} | "
            f"{milliseconds(values['aot-inductor'])} | "
            f"{', '.join(row['dveb_selected'])} | {row['winner']} | "
            f"{row['dveb_speedup_over_next_best']:.2f}x |"
        )
    build_times = [
        float(package["build"]["external_seconds"])
        for package in manifest["aot_packages"].values()
    ]
    extraction_times = [
        float(package["extraction_cache_preparation"]["external_seconds"])
        for package in manifest["aot_packages"].values()
    ]
    largest = next(row for row in rows if row["size"] == 128 and row["steps"] == 10)
    largest_times = largest["fresh_process_median_seconds"]
    lines += [
        "",
        "## Decision",
        "",
        "DVEB wins 8 of the 9 declared regions. For one-step work it selects",
        "`cpu_simd[6]` at N=8, 16, and 32, then CUDA at N=64, 96, and 128.",
        "For ten-step work it selects CUDA at every tested size, beginning at",
        "N=32. Thus placement depends on both grid size and timestep count.",
        "",
        "Fortran wins only at N=8 / one step: 2.463 ms versus DVEB's 2.777 ms,",
        "with Fortran ahead in 27 of 30 paired repetitions. DVEB wins at N=16",
        "in 26 of 30 pairs and at every larger or longer point in 30 of 30.",
        "The N=32 / one-step medians differ by 8.7%, so both qualify as",
        "competitive under the frozen 10% rule even though DVEB wins all pairs.",
        "",
        f"At N=128 / ten steps, DVEB's {largest_times['dveb-auto']:.3f} s median "
        f"is {largest_times['aot-inductor'] / largest_times['dveb-auto']:.2f}x "
        f"faster than AOT PyTorch, {largest_times['direct-eager'] / largest_times['dveb-auto']:.2f}x "
        f"faster than eager PyTorch, and {largest_times['fortran'] / largest_times['dveb-auto']:.2f}x "
        "faster than Fortran at the complete start-to-finish endpoint.",
        "",
        "This validates a bounded reason for DVEB to exist in WENO: one native",
        "artifact spans the near-Fortran small CPU region and the high-throughput",
        "CUDA region without exposing placement machinery to the caller. It is",
        "not evidence that DVEB wins other formulations or machines.",
        "",
        "## Preparation and limitations",
        "",
        f"The fixed-shape AOT packages took {min(build_times):.2f}–{max(build_times):.2f} s "
        f"each to build and {min(extraction_times):.2f}–{max(extraction_times):.2f} s "
        "for the recorded first extraction/preparation runs. DVEB calibration",
        "also ran before timing; its per-observation raw records are committed,",
        "but this first harness did not record one enclosing wall-clock duration",
        "for calibration including warmups.",
        "",
        "The selector was calibrated at the same grid/step points with separate",
        "observations. That is valid profile-guided deployment evidence, not a",
        "held-out generalization test. The campaign covers one float32 3-D Shu",
        "Euler WENO-5 workload, one vortex, and one Ryzen 7600X / RTX 5070 Ti",
        "machine. The frozen DVEB executable came from an uncommitted compiler",
        "worktree state, so its hash makes this run auditable but a clean-source",
        "rebuild remains unresolved until DVEB commits and requalifies it.",
        "",
        f"Full-array correctness maximum: `{correctness['maximum_error']:.9e}` "
        f"(bound `{correctness['bound']:.1e}`).",
        "",
        "The paired-win counts and SHA-256 identities for every raw result are in",
        f"`{arguments.output_json.name}`. That record also hashes the exact",
        "GradFlow PyTorch source, Fortran source/executable, protocol, harness,",
        "DVEB executable, and placement model used by the campaign.",
    ]
    arguments.output_markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
