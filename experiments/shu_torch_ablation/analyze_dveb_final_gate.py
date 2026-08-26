#!/usr/bin/env python3
"""Analyze the frozen held-out DVEB WENO selector gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args()
    selector_paths = sorted(arguments.input_dir.glob("selector_n*_s*.json"))
    large_paths = sorted(arguments.input_dir.glob("large_n*_s*.json"))
    if len(selector_paths) != 10 or len(large_paths) != 4:
        raise SystemExit(
            f"expected 10 selector and 4 large files, got {len(selector_paths)} and {len(large_paths)}"
        )

    selector_rows: list[dict[str, object]] = []
    for path in selector_paths:
        result = json.loads(path.read_text())
        if not result.get("complete"):
            raise SystemExit(f"incomplete result: {path}")
        medians = {lane: float(summary["fresh_process_seconds"]["median"])
                   for lane, summary in result["summary"].items()}
        forced = {lane: value for lane, value in medians.items() if lane != "auto"}
        best = min(forced, key=forced.get)
        selections = sorted({str(record.get("selected")) for record in result["records"]
                             if record["lane"] == "auto" and record["success"]})
        cpu_forced = {lane: value for lane, value in forced.items() if lane.startswith("cpu")}
        selected = selections[0] if len(selections) == 1 else None
        selector_rows.append({
            "size": result["size"], "steps": result["steps"], "medians_seconds": medians,
            "selected": selections, "best_candidate": best,
            "regret": medians["auto"] / forced[best],
            "absolute_loss_seconds": medians["auto"] - forced[best],
            "stable_and_complete": len(selections) == 1 and
                result["summary"]["auto"]["successes"] == 30 and
                all(summary["failures"] == 0 for summary in result["summary"].values()),
            "cpu_proximity": (
                medians[selected] / min(cpu_forced.values())
                if selected in cpu_forced else None
            ),
            "raw": str(path.resolve()), "raw_sha256": sha256(path),
        })
    selector_rows.sort(key=lambda row: (int(row["steps"]), int(row["size"])))
    regrets = [float(row["regret"]) for row in selector_rows]
    pass_conditions = {
        "stable_and_complete": all(bool(row["stable_and_complete"]) for row in selector_rows),
        "median_regret_le_1_10": sorted(regrets)[len(regrets) // 2 - 1:len(regrets) // 2 + 1]
            and (sorted(regrets)[4] + sorted(regrets)[5]) / 2 <= 1.10,
        "at_least_80pct_le_1_15": sum(value <= 1.15 for value in regrets) >= 8,
        "no_large_regret_without_tie": all(
            float(row["regret"]) <= 1.30 or abs(float(row["absolute_loss_seconds"])) < 0.00025
            for row in selector_rows
        ),
        "cpu_proximity": all(
            row["cpu_proximity"] is None or float(row["cpu_proximity"]) <= 1.10
            for row in selector_rows
        ),
    }

    large_rows: list[dict[str, object]] = []
    for path in large_paths:
        result = json.loads(path.read_text())
        medians = {lane: float(summary["fresh_process_seconds"]["median"])
                   for lane, summary in result["summary"].items()}
        selections = sorted({str(record.get("selected")) for record in result["records"]
                             if record["lane"] == "auto" and record["success"]})
        large_rows.append({
            "size": result["size"], "steps": result["steps"],
            "medians_seconds": medians, "selected": selections,
            "auto_vs_forced_cuda": medians["auto"] / medians["cuda"],
            "forced_cuda_vs_ceiling": medians["cuda"] / medians["ceiling-cuda"],
            "raw": str(path.resolve()), "raw_sha256": sha256(path),
        })
    large_rows.sort(key=lambda row: (int(row["steps"]), int(row["size"])))

    report = {
        "schema_version": 1, "selector_rows": selector_rows,
        "selector_median_regret": (sorted(regrets)[4] + sorted(regrets)[5]) / 2,
        "selector_maximum_regret": max(regrets), "pass_conditions": pass_conditions,
        "selector_passed": all(pass_conditions.values()), "large_rows": large_rows,
    }
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Final DVEB WENO requalification results", "",
        "## Held-out selector gate", "",
        "Fresh-process medians from 30 randomized blocks; calibration sizes are disjoint.", "",
        "| N | Steps | Selected | Best forced | Auto ms | Best ms | Regret | Loss ms |",
        "|---:|---:|:---|:---|---:|---:|---:|---:|",
    ]
    for row in selector_rows:
        medians = row["medians_seconds"]
        best = row["best_candidate"]
        lines.append(
            f"| {row['size']} | {row['steps']} | {', '.join(row['selected'])} | {best} | "
            f"{1000 * medians['auto']:.3f} | {1000 * medians[best]:.3f} | "
            f"{row['regret']:.4f} | {1000 * row['absolute_loss_seconds']:.3f} |"
        )
    lines += ["", f"Decision: **{'PASS' if report['selector_passed'] else 'NO-GO'}**.",
              f"Median regret: `{report['selector_median_regret']:.4f}`; maximum: "
              f"`{report['selector_maximum_regret']:.4f}`.", "", "## Large-grid confirmation", "",
              "| N | Steps | Selected | Auto ms | Forced CUDA ms | Ceiling ms | Generated/ceiling |",
              "|---:|---:|:---|---:|---:|---:|---:|"]
    for row in large_rows:
        medians = row["medians_seconds"]
        lines.append(
            f"| {row['size']} | {row['steps']} | {', '.join(row['selected'])} | "
            f"{1000 * medians['auto']:.3f} | {1000 * medians['cuda']:.3f} | "
            f"{1000 * medians['ceiling-cuda']:.3f} | {row['forced_cuda_vs_ceiling']:.4f} |"
        )
    lines += ["", "This is a WENO-specific, machine-specific qualification. DVEB's generic",
              "automatic selector remains NO-GO at commit `2f1f3ab`."]
    arguments.output_markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
