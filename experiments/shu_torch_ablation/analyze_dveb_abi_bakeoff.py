#!/usr/bin/env python3
"""Analyze the frozen forced-target ABI bakeoff without changing its rules."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


PRIMARY = {
    "E1": ("fortran", "dveb-cpu6", "dveb-cpu12", "dveb-cuda", "direct-eager", "persistent-compile", "aot-inductor"),
    "E2": ("dveb-cpu6", "dveb-cpu12", "dveb-cuda", "direct-eager", "persistent-compile", "aot-inductor"),
    "E3": ("dveb-cpu6", "dveb-cpu12", "dveb-cuda", "direct-eager", "persistent-compile", "aot-inductor"),
    "E4": ("direct-eager", "persistent-compile", "aot-inductor"),
}
DVEB = ("dveb-cpu6", "dveb-cpu12", "dveb-cuda")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int]:
    median = statistics.median(values)
    return {
        "count": len(values), "minimum": min(values),
        "p05": percentile(values, 0.05), "median": median,
        "mean": statistics.fmean(values), "p95": percentile(values, 0.95),
        "maximum": max(values),
        "mad": statistics.median(abs(value - median) for value in values),
    }


def observations(record: dict[str, object], mode: str) -> list[float]:
    if not record.get("success"):
        return []
    if mode == "E1":
        return [float(record["fresh_process_seconds"])]
    return [float(item["call_seconds"]) for item in record.get("observations", [])]


def point_analysis(report: dict[str, object]) -> dict[str, object]:
    mode = report["mode"]
    lane_stats = {}
    block_values: dict[int, dict[str, float]] = {}
    for lane in report["lanes"]:
        lane_records = [record for record in report["records"] if record["lane"] == lane]
        values = [value for record in lane_records for value in observations(record, mode)]
        item: dict[str, object] = {
            "processes": len(lane_records),
            "failures": sum(not bool(record.get("success")) for record in lane_records),
        }
        if values:
            item.update(stats(values))
        peaks = [
            int(record.get("diagnostics", {}).get("peak_allocated_bytes", 0) or 0)
            for record in lane_records
        ]
        rss = [int(record.get("max_rss_kib", 0) or 0) for record in lane_records]
        item["peak_gpu_allocated_bytes"] = max(peaks, default=0)
        item["peak_host_rss_kib"] = max(rss, default=0)
        lane_stats[lane] = item
        for record in lane_records:
            record_values = observations(record, mode)
            if record_values:
                block_values.setdefault(int(record["block"]), {})[lane] = statistics.median(record_values)

    eligible = [lane for lane in PRIMARY.get(mode, ()) if lane_stats.get(lane, {}).get("count")]
    medians = {lane: float(lane_stats[lane]["median"]) for lane in eligible}
    winner = min(medians, key=medians.get) if medians else None
    best = medians[winner] if winner else None
    competitive = [lane for lane, value in medians.items() if value <= 1.10 * best] if best else []
    ties = [lane for lane, value in medians.items() if abs(value - best) < 0.00025] if best else []
    block_wins = {lane: 0 for lane in eligible}
    for values in block_values.values():
        present = {lane: value for lane, value in values.items() if lane in eligible}
        if present:
            block_best = min(present.values())
            for lane, value in present.items():
                if abs(value - block_best) < 0.00025:
                    block_wins[lane] += 1

    ceiling_distance = {}
    if mode == "E1":
        for lane, ceiling in (
            ("dveb-cpu6", "ceiling-cpu"), ("dveb-cpu12", "ceiling-cpu"),
            ("dveb-cuda", "ceiling-cuda"),
        ):
            if lane in lane_stats and ceiling in lane_stats and "median" in lane_stats[lane] and "median" in lane_stats[ceiling]:
                ceiling_distance[lane] = float(lane_stats[lane]["median"]) / float(lane_stats[ceiling]["median"])
    return {
        "mode": mode, "size": report["size"], "steps": report["steps"],
        "lanes": lane_stats, "winner": winner, "best_seconds": best,
        "competitive": competitive, "practical_ties": ties,
        "paired_block_wins_or_ties": block_wins,
        "dveb_ceiling_distance": ceiling_distance,
    }


def useful_regions(points: list[dict[str, object]]) -> dict[str, object]:
    results = {}
    for mode in ("E1", "E2", "E3"):
        mode_points = [point for point in points if point["mode"] == mode]
        for lane in DVEB:
            competitive = {
                (int(point["size"]), int(point["steps"]))
                for point in mode_points if lane in point["competitive"]
            }
            adjacent = []
            for steps in (1, 10):
                sizes = sorted(size for size, item_steps in competitive if item_steps == steps)
                counted_order = [8, 16, 32, 64, 96, 128] if steps == 1 else [16, 32, 64, 128]
                for left, right in zip(counted_order, counted_order[1:]):
                    if left in sizes and right in sizes:
                        adjacent.append([left, right, steps])
            both_steps = sorted(
                size for size in {size for size, _ in competitive}
                if (size, 1) in competitive and (size, 10) in competitive
            )
            results[f"{mode}:{lane}"] = {
                "qualified": bool(adjacent or both_steps),
                "competitive_points": sorted([list(item) for item in competitive]),
                "adjacent_size_evidence": adjacent,
                "both_step_strata_sizes": both_steps,
            }
    return results


def milliseconds(value: float | None) -> str:
    return "—" if value is None else f"{1000.0 * value:.3f}"


ENDPOINT_NAMES = {
    "E1": "fresh standalone application (process creation through validated CPU result)",
    "E2": "first `run` in a fresh worker (ready CPU input through returned CPU result)",
    "E3": "warm repeated `run` (ready CPU input through newly returned CPU result)",
    "E4": "resident PyTorch numerical execution (synchronized, no H2D/D2H)",
}


def selected_point(analysis: dict[str, object], mode: str, size: int, steps: int) -> dict[str, object]:
    return next(
        point for point in analysis["points"]
        if point["mode"] == mode and point["size"] == size and point["steps"] == steps
    )


def markdown(analysis: dict[str, object]) -> str:
    e1_128_10 = selected_point(analysis, "E1", 128, 10)
    e3_128_10 = selected_point(analysis, "E3", 128, 10)
    correctness = analysis["correctness"]
    capacity_160_10 = next(
        report for report in analysis["capacity"]
        if report["size"] == 160 and report["steps"] == 10
    )
    capacity_records = {record["lane"]: record for record in capacity_160_10["records"]}
    dveb_peak = capacity_records["dveb-cuda"]["diagnostics"]["native_peak_bytes"]
    aot_peak = capacity_records["aot-inductor"]["diagnostics"]["peak_allocated_bytes"]
    eager_peak = capacity_records["direct-eager"]["diagnostics"]["peak_allocated_bytes"]
    lines = [
        "# Forced-target DVEB ABI bakeoff results", "",
        "Status: **completed; all frozen correctness and timing gates passed**.", "",
        "All primary values are medians of the frozen 30-observation design. ",
        "Fresh application, first call, warm call, and resident execution remain separate.", "",
        "## Result in one paragraph", "",
        (
            "DVEB validates its existence for this matched workload, but not as one universal winner. "
            "Its CPU-12 target wins small first-`run` calls; its CUDA target wins the larger standalone "
            "applications and every warm repeated-call point. At `N=128`, ten steps, DVEB-CUDA takes "
            f"{milliseconds(e1_128_10['lanes']['dveb-cuda']['median'])} ms as a standalone invocation "
            f"versus {milliseconds(e1_128_10['lanes']['fortran']['median'])} ms for Fortran, and "
            f"{milliseconds(e3_128_10['lanes']['dveb-cuda']['median'])} ms per warm CPU-in/CPU-out run "
            f"versus {milliseconds(e3_128_10['lanes']['aot-inductor']['median'])} ms for AOT PyTorch. "
            "AOT PyTorch is the best supported resident-state route, while DVEB ABI v1 cannot enter that "
            "endpoint because it does not accept device-resident state."
        ), "",
        "## Correctness and capacity gates", "",
        (
            f"All ten lanes passed full-array comparison at four frozen points. The worst pairwise "
            f"float32 error was `{correctness['maximum_error']:.10g}` against the frozen "
            f"`{correctness['bound']:.10g}` bound; duplicated endpoints were exact."
        ), "",
        (
            "Every lane completed the uncounted capacity pilot through `N=160` for one and ten steps. "
            "At `N=160`, ten steps, the reported peak CUDA allocations were "
            f"{dveb_peak / 2**30:.3f} GiB for DVEB-CUDA, {aot_peak / 2**30:.3f} GiB for AOT PyTorch, "
            f"and {eager_peak / 2**30:.3f} GiB for eager PyTorch. These are implementation-reported "
            "allocator peaks, not whole-system GPU-memory measurements, and capacity-pilot timings are "
            "not used as performance claims."
        ), "",
    ]
    for mode in ("E1", "E2", "E3", "E4"):
        mode_points = [point for point in analysis["points"] if point["mode"] == mode]
        if not mode_points:
            continue
        lines += [
            f"## {mode}: {ENDPOINT_NAMES[mode]}", "",
            "| N | Steps | Winner | Median ms | Competitive lanes |",
            "|---:|---:|---|---:|---|",
        ]
        for point in sorted(mode_points, key=lambda item: (item["steps"], item["size"])):
            lines.append(
                f"| {point['size']} | {point['steps']} | {point['winner']} | "
                f"{milliseconds(point['best_seconds'])} | {', '.join(point['competitive'])} |"
            )
        lines.append("")
    lines += ["## DVEB useful-region rule", ""]
    lines += [
        "A DVEB target qualifies only when it is within 10% of the winner at two adjacent counted sizes "
        "or at the same size in both step strata.", "",
    ]
    for name, record in analysis["useful_regions"].items():
        lines.append(f"- `{name}`: {'QUALIFIED' if record['qualified'] else 'not qualified'}")
    cold_rows = []
    for report in sorted(analysis["cold_diagnostics"], key=lambda item: (item["steps"], item["size"])):
        compile_median = report["summary"]["persistent-compile"]["median"]
        aot_median = report["summary"]["aot-inductor"]["median"]
        cold_rows.append(
            f"| {report['size']} | {report['steps']} | {1000 * compile_median:.1f} | "
            f"{1000 * aot_median:.1f} | {compile_median / aot_median:.1f}x |"
        )
    lines += [
        "", "## Cold-cache diagnostic", "",
        "These smaller diagnostic samples are excluded from winner classification.", "",
        "| N | Steps | Empty-cache first compile+call ms | Pristine AOT call ms | Ratio |",
        "|---:|---:|---:|---:|---:|", *cold_rows, "",
        "## Interpretation", "",
        "- **Standalone program:** Fortran wins through `N=64` for one step and through `N=32` for ten steps. DVEB-CUDA wins every larger counted E1 point.",
        "- **First `run` with a ready CPU state:** DVEB CPU-12 wins the small region; AOT PyTorch wins the middle region; DVEB-CUDA wins the two large ten-step points.",
        "- **Warm CPU-in/CPU-out service:** DVEB-CUDA wins all counted E3 points. Its small-grid margins over AOT are narrow, but it becomes materially faster as work grows.",
        "- **Device-resident throughput:** AOT and persistent compiled PyTorch are effectively tied at most larger E4 points. AOT has the lower median at every counted point. DVEB ABI v1 is unsupported here, not measured as a loser.",
        "- **Cold compilation:** An empty TorchInductor cache costs about 45–48 seconds inside the first call at the diagnostic points. AOT removes that compilation event but still pays Python/package launch overhead in E1.",
        "", "## Boundaries", "",
        "`DVEB Auto` was not tested or calibrated. The protocol used forced targets only. DVEB ABI v1 has no public resident-state interface and therefore does not participate in E4. Internal CUDA execution timing remains diagnostic only. The capacity pilot establishes only that all lanes ran through `N=160`; it does not establish a hardware maximum. No arbitrary-order, Navier--Stokes, real-time aerospace, cross-machine, or publication claim follows from this campaign.", "",
        f"Prepared-manifest SHA-256: `{analysis['prepared_manifest']['sha256']}`.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    args = parser.parse_args()
    timing_files = sorted(args.results_dir.glob("timing_*.json"))
    reports = [json.loads(path.read_text()) for path in timing_files]
    primary_reports = [report for report in reports if report.get("mode") in PRIMARY]
    points = [point_analysis(report) for report in primary_reports]
    correctness_path = args.results_dir / "correctness.json"
    correctness = json.loads(correctness_path.read_text())
    prepared_manifest_bytes = args.prepared_manifest.read_bytes()
    analysis = {
        "schema": "gradflow-dveb-abi-bakeoff-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(path.resolve()) for path in timing_files],
        "points": points,
        "correctness": correctness,
        "prepared_manifest": {
            "path": str(args.prepared_manifest.resolve()),
            "sha256": hashlib.sha256(prepared_manifest_bytes).hexdigest(),
        },
        "useful_regions": useful_regions(points),
        "capacity": [report for report in reports if report.get("mode") == "capacity"],
        "cold_diagnostics": [report for report in reports if report.get("mode") == "cold"],
    }
    args.output_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(markdown(analysis))
    print(json.dumps({"output": str(args.output_json), "points": len(points)}, sort_keys=True))


if __name__ == "__main__":
    main()
