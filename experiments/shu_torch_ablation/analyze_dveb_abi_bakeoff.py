#!/usr/bin/env python3
"""Analyze the frozen forced-target ABI bakeoff without changing its rules."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Forced-target DVEB ABI bakeoff results", "",
        "All primary values are medians of the frozen 30-observation design. ",
        "Fresh application, first call, warm call, and resident execution remain separate.", "",
    ]
    for mode in ("E1", "E2", "E3", "E4"):
        mode_points = [point for point in analysis["points"] if point["mode"] == mode]
        if not mode_points:
            continue
        lines += [f"## {mode}", "", "| N | Steps | Winner | Median ms | Competitive lanes |", "|---:|---:|---|---:|---|"]
        for point in sorted(mode_points, key=lambda item: (item["steps"], item["size"])):
            lines.append(
                f"| {point['size']} | {point['steps']} | {point['winner']} | "
                f"{milliseconds(point['best_seconds'])} | {', '.join(point['competitive'])} |"
            )
        lines.append("")
    lines += ["## DVEB useful-region rule", ""]
    for name, record in analysis["useful_regions"].items():
        lines.append(f"- `{name}`: {'QUALIFIED' if record['qualified'] else 'not qualified'}")
    lines += [
        "", "## Boundaries", "",
        "`DVEB Auto` was not tested or calibrated. DVEB ABI v1 has no public resident-state interface and therefore does not participate in E4. Internal CUDA execution timing remains diagnostic only. No arbitrary-order or publication claim follows from this campaign.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    timing_files = sorted(args.results_dir.glob("timing_*.json"))
    reports = [json.loads(path.read_text()) for path in timing_files]
    primary_reports = [report for report in reports if report.get("mode") in PRIMARY]
    points = [point_analysis(report) for report in primary_reports]
    analysis = {
        "schema": "gradflow-dveb-abi-bakeoff-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(path.resolve()) for path in timing_files],
        "points": points,
        "useful_regions": useful_regions(points),
        "capacity": [report for report in reports if report.get("mode") == "capacity"],
        "cold_diagnostics": [report for report in reports if report.get("mode") == "cold"],
    }
    args.output_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(markdown(analysis))
    print(json.dumps({"output": str(args.output_json), "points": len(points)}, sort_keys=True))


if __name__ == "__main__":
    main()
