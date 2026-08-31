#!/usr/bin/env python3
"""Export the rc2 paper dataset with stable U5 and batched U4-F evidence."""

from __future__ import annotations

import json
from pathlib import Path

from export_paper_data import ROOT, load, sha256


EXPORT_ID = "academic-v0.1.0-rc2-paper-v4"
EXPORT = ROOT / "experiments/academic_a4/exports" / EXPORT_ID
SOURCE_RELEASE_TAG = "academic-v0.1.0-rc2"
SOURCE_REVISION = "c5e8ab81ef5b33a2138b2db33afc538398b6f57f"
INPUTS = {
    "paper_v3": ROOT / "experiments/academic_a4/exports/academic-692f822-paper-v3/paper_data.json",
    "u5_comparison": ROOT / "experiments/academic_u5/evidence/u5_20260831/comparison.json",
    "u5_environment": ROOT / "experiments/academic_u5/evidence/u5_20260831/environment.json",
    "u5_a1": ROOT / "experiments/academic_u5/evidence/u5_20260831/a1/numerical_limits.json",
    "u5_a2": ROOT / "experiments/academic_u5/evidence/u5_20260831/a2/analysis.json",
    "u5_a3": ROOT / "experiments/academic_u5/evidence/u5_20260831/a3/campaign.json",
    "u5_u4f": ROOT / "experiments/academic_u5/evidence/u5_20260831/u4f/campaign.json",
    "rc2_index": ROOT / "experiments/academic_a4/evidence/a4_rc2_20260831/artifact_index.json",
    "rc2_cleanroom": ROOT / "experiments/academic_a4/evidence/a4_rc2_20260831/cleanroom.json",
}


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    data = load(INPUTS["paper_v3"])
    comparison = load(INPUTS["u5_comparison"])
    environment = load(INPUTS["u5_environment"])
    a1 = load(INPUTS["u5_a1"])
    a2 = load(INPUTS["u5_a2"])
    a3 = load(INPUTS["u5_a3"])
    u4f = load(INPUTS["u5_u4f"])
    index = load(INPUTS["rc2_index"])
    cleanroom = load(INPUTS["rc2_cleanroom"])

    development_snapshot = {
        "environment": data["environment"],
        "performance_64cube": data["performance_64cube"],
        "crossover_3d": data["crossover_3d"],
        "isolated_cache_deployment": data["isolated_cache_deployment"],
        "aot_packages": data["aot_packages"],
        "differentiation_benchmarks": data["differentiation_benchmarks"],
        "source_export": data["export_id"],
    }
    stable = {
        "schema": "gradflow-academic-stable-toolchain-paper-v1",
        "environment": {
            "python": environment["python"],
            "torch": environment["torch"],
            "torch_git_version": environment["torch_git_version"],
            "torch_cuda": environment["torch_cuda"],
            "numpy": environment["numpy"],
            "gpu": environment["gpu"],
            "compute_capability": environment["compute_capability"],
        },
        "numerical_limits": {
            "qualified_orders": a1["qualified_orders"],
            "coefficient_diagnostics": a1["coefficient_diagnostics"],
            "roundoff_sweeps": a1["roundoff_sweeps"],
            "epsilon_sweeps": a1["epsilon_sweeps"],
            "comparison": comparison["a1"],
        },
        "performance_64cube": comparison["a2"]["cross_order_64cube"],
        "speedup_ranges": comparison["a2"]["stable_speedup_ranges"],
        "correctness_exclusions": {
            "stable_count": comparison["a2"]["stable_exclusion_count"],
            "development_count": comparison["a2"]["development_exclusion_count"],
            "removed": comparison["a2"]["exclusions_removed"],
            "added": comparison["a2"]["exclusions_added"],
            "stable_records": a2["correctness_exclusions"],
        },
        "scale": a2["scale"],
        "aot": comparison["a2"]["aot"],
        "deployment": comparison["a2"]["deployment"],
        "differentiation": comparison["a3"],
        "inverse_problem": {
            "derivative_gate": a3["derivative_gate"],
            "inverse_gate": a3["inverse_gate"],
            "resolution_study": a3["resolution_study"],
        },
        "u4f": comparison["u4f"],
        "u4f_protocol": {
            "size": u4f["size"],
            "batches": u4f["batches"],
            "bounds": u4f["bounds"],
        },
        "cross_version_timing_is_descriptive": True,
    }
    data.update(
        {
            "schema": "gradflow-academic-paper-data-v4",
            "export_id": EXPORT_ID,
            "release_candidate": SOURCE_RELEASE_TAG,
            "source_release_candidate": SOURCE_RELEASE_TAG,
            "source_revision": SOURCE_REVISION,
            "primary_toolchain": "stable_pytorch_2_13",
            "stable_toolchain": stable,
            "development_toolchain": development_snapshot,
            "artifact": {
                "tag": SOURCE_RELEASE_TAG,
                "tagged_commit": SOURCE_REVISION,
                "indexed_source_commit": index["source_commit"],
                "file_count": index["file_count"],
                "total_bytes": index["total_bytes"],
                "artifact_index_sha256": sha256(INPUTS["rc2_index"]),
                "cleanroom_all_passed": cleanroom["all_passed"],
                "cleanroom_command_count": len(cleanroom["commands"]),
                "cleanroom_tested_commit": cleanroom["tested_commit"],
                "cleanroom_network_used": cleanroom["network_used"],
            },
        }
    )
    data["input_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()
    }
    dataset = EXPORT / "paper_data.json"
    dataset.write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "schema": "gradflow-academic-export-v4",
        "export_id": EXPORT_ID,
        "source_revision": SOURCE_REVISION,
        "source_release_tag": SOURCE_RELEASE_TAG,
        "generation_date_utc": "2026-08-31",
        "generator": "experiments/academic_a4/export_paper_data_v4.py",
        "generator_sha256": sha256(Path(__file__)),
        "input_sha256": data["input_sha256"],
        "outputs": {
            "paper_data.json": {
                "bytes": dataset.stat().st_size,
                "sha256": sha256(dataset),
            }
        },
    }
    manifest_path = EXPORT / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (EXPORT / "README.md").write_text(
        f"# GradFlow paper export `{EXPORT_ID}`\n\n"
        f"Source tag: `{SOURCE_RELEASE_TAG}`  \n"
        f"Tagged commit: `{SOURCE_REVISION}`\n\n"
        "This reporting-complete export makes stable PyTorch 2.13 the primary "
        "toolchain evidence, retains the development build as a version-sensitivity "
        "stratum, and adds the U4-F backend regime and rc2 clean-room record. "
        "It does not close second-machine, external-review, rights, or licensing gates.\n"
    )
    print(f"exported {EXPORT_ID}; manifest sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()

