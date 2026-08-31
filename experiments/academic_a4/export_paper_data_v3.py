#!/usr/bin/env python3
"""Export the paper dataset including the frozen U4-C/U4-D/U4-E controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from export_paper_data import ROOT, load, sha256


EXPORT_ID = "academic-692f822-paper-v3"
EXPORT = ROOT / "experiments/academic_a4/exports" / EXPORT_ID
SOURCE_REVISION = "692f822ef7fef9770247ac56e3526b0f3ac2436c"
BASE_RELEASE_TAG = "academic-v0.1.0-rc1"
INPUTS = {
    "paper_v2": ROOT
    / "experiments/academic_a4/exports/academic-v0.1.0-rc1-paper-v2/paper_data.json",
    "u4c_c2": ROOT
    / "experiments/academic_u4c/evidence/u4c_c2_20260830/campaign.json",
    "u4c_c3": ROOT
    / "experiments/academic_u4c/evidence/u4c_c3_20260830/endpoints.json",
    "u4d": ROOT
    / "experiments/academic_u4d/evidence/u4d_campaign_20260830/campaign.json",
    "u4e_e1": ROOT
    / "experiments/academic_u4e/evidence/u4e_e1_20260831/qualification.json",
    "u4e": ROOT
    / "experiments/academic_u4e/evidence/u4e_campaign_20260831/campaign.json",
    "backend_identity": ROOT / "docs/BACKEND_IDENTITY.md",
}
SUMMARY_KEYS = (
    "count",
    "values",
    "minimum",
    "q1",
    "median",
    "q3",
    "maximum",
    "mean",
    "median_absolute_deviation",
    "sample_standard_deviation",
)


def summary(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in SUMMARY_KEYS if key in record}


def resident(campaign: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for device in ("cpu", "cuda"):
        analysis = campaign["resident"][device]["analysis"]
        result[device] = {
            "workers_per_lane": campaign["resident"][device]["workers_per_lane"],
            "warmups_per_worker": campaign["resident"][device][
                "warmups_per_worker"
            ],
            "samples_per_worker": campaign["resident"][device][
                "samples_per_worker"
            ],
            "lanes": {
                ("pytorch_inductor" if lane == "gradflow" else lane): summary(
                    value["worker_medians"]
                )
                for lane, value in analysis["lanes"].items()
            },
            "paired_worker_median_ratios": {
                key.replace("gradflow", "pytorch_inductor"): value
                for key, value in analysis["paired_worker_median_ratios"].items()
            },
            "overall_winner": (
                "pytorch_inductor"
                if analysis["overall_winner"] == "gradflow"
                else analysis["overall_winner"]
            ),
        }
    return result


def endpoint(campaign: dict[str, Any], key: str) -> dict[str, Any]:
    analysis = campaign[key]["analysis"]
    renamed = {}
    for lane, value in analysis.items():
        renamed[lane.replace("gradflow_aot", "pytorch_aotinductor").replace(
            "gradflow", "pytorch_inductor"
        )] = value if not isinstance(value, dict) else summary(value)
    return renamed


def u4c_admission(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for n_text, record in campaign["sizes"].items():
        qualification = record["qualification"]
        opensbli = [qualification["opensbli_cpu"], qualification["opensbli_cuda"]]
        rows.append(
            {
                "n": int(n_text),
                "status": record["status"],
                "all_lanes_admitted": qualification["all_lanes_admitted"],
                "largest_opensbli_maximum_normalized": max(
                    value["maximum_normalized"] for value in opensbli
                ),
                "largest_opensbli_rms_normalized": max(
                    value["rms_normalized"] for value in opensbli
                ),
            }
        )
    return sorted(rows, key=lambda row: row["n"])


def build_data_v3() -> dict[str, Any]:
    data = load(INPUTS["paper_v2"])
    u4c_c2 = load(INPUTS["u4c_c2"])
    u4c_c3 = load(INPUTS["u4c_c3"])
    u4d = load(INPUTS["u4d"])
    u4e_e1 = load(INPUTS["u4e_e1"])
    u4e = load(INPUTS["u4e"])
    data.update(
        {
            "schema": "gradflow-academic-paper-data-v3",
            "export_id": EXPORT_ID,
            "source_revision": SOURCE_REVISION,
            "base_release_candidate": BASE_RELEASE_TAG,
            "backend_identity": {
                "system": "gradflow",
                "tested_backends": (
                    "dveb_native",
                    "opensbli_ops",
                    "pytorch_inductor",
                    "pytorch_aotinductor",
                ),
                "legacy_evidence_alias": {
                    "gradflow": "pytorch_inductor",
                    "gradflow_aot": "pytorch_aotinductor",
                },
                "policy": (
                    "GradFlow is the encompassing system; PyTorch was an "
                    "implementation and performance hypothesis, not the system identity."
                ),
            },
            "external_baseline": {
                "schema": "gradflow-academic-external-baseline-paper-v1",
                "contract": {
                    "operator": "scalar finite-difference WENO-JS5 RHS",
                    "dimension": 1,
                    "dtype": "float64",
                    "n": u4e["size"],
                    "boundary": "unique periodic points",
                    "cpu_threads": 1,
                    "gpu": u4e["environment"]["gpu"],
                    "reason_single_size": u4e["reason_single_size"],
                },
                "u4c_admission_surface": u4c_admission(u4c_c2),
                "u4c_transfer": {
                    "pytorch_inductor": summary(
                        u4c_c3["transfer"]["analysis"]["gradflow"]
                    ),
                    "opensbli": summary(
                        u4c_c3["transfer"]["analysis"]["opensbli"]
                    ),
                },
                "u4d_historical_resident": resident(u4d),
                "u4e_qualification": {
                    "decision": u4e_e1["decision"],
                    "bounds": u4e_e1["bounds"],
                    "canonical": u4e_e1["canonical"],
                    "lanes": {
                        key.replace("gradflow", "pytorch_inductor"): value
                        for key, value in u4e_e1["qualification"].items()
                    },
                },
                "u4e_resident": resident(u4e),
                "u4e_transfer": endpoint(u4e, "transfer"),
                "u4e_prepared_launch": endpoint(u4e, "prepared_launch"),
                "u4e_historical_resident_medians_milliseconds": {
                    device: {
                        lane.replace("gradflow", "pytorch_inductor"): value
                        for lane, value in lanes.items()
                    }
                    for device, lanes in u4e[
                        "u4d_historical_resident_medians_milliseconds"
                    ].items()
                },
                "u4e_schedule": {
                    "cpu": u4e["resident"]["cpu"]["worker_records"]["dveb"][0][
                        "query"
                    ],
                    "cuda": u4e["resident"]["cuda"]["worker_records"]["dveb"][0][
                        "query"
                    ],
                },
                "cross_campaign_comparison_is_descriptive": u4e[
                    "u4d_to_u4e_cross_campaign_comparison_is_descriptive"
                ],
            },
        }
    )
    data["input_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()
    }
    return data


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    data = build_data_v3()
    dataset = EXPORT / "paper_data.json"
    dataset.write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "schema": "gradflow-academic-export-v3",
        "export_id": EXPORT_ID,
        "source_revision": SOURCE_REVISION,
        "base_release_tag": BASE_RELEASE_TAG,
        "source_release_tag": None,
        "generation_date_utc": "2026-08-31",
        "generator": "experiments/academic_a4/export_paper_data_v3.py",
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
    print(f"exported {EXPORT_ID}; manifest sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
