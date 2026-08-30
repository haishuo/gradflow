#!/usr/bin/env python3
"""Build the A1 source index, claim matrix, and derived summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ORDERS = (5, 7, 9, 11, 13, 15)
SOURCE_RECORDS = {
    "scalar_qualification": "experiments/weno_js_arbitrary_order/results/qualification_20260826.json",
    "characteristic_qualification": "experiments/characteristic_arbitrary_order/results/qualification_20260826.json",
    "scalar_mixed_precision": "experiments/mixed_precision/results/phase_d_tier1b_20260827/search.json",
    "scalar_mixed_performance": "experiments/mixed_precision/results/phase_d_performance_20260827/benchmark.json",
    "characteristic_mixed_precision": "experiments/mixed_precision/results/phase_d_tier2_20260827/qualification.json",
    "boundary_shock_qualification": "experiments/euler_boundary_shock/results/phase_b_20260827/qualification.json",
    "literature_studies": "experiments/literature_review/results/phase_c_20260827/studies.json",
    "literature_claims": "experiments/literature_review/results/phase_c_20260827/claim_matrix.json",
    "face_ownership_screen": "experiments/face_ownership_screen/evidence/face_ownership_20260830/screen.json",
    "native_face_ownership": "experiments/gpu_native_reformulation/evidence/g4_performance_20260829/campaign.json",
    "fd_fv_prepared_aot": "experiments/fd_fv_euler/results/phase_6g_performance_20260829/benchmark.json",
    "a1_numerical_limits": "experiments/academic_a1/evidence/a1_20260830/numerical_limits.json",
}
PRIOR_ART_IDS = ("opensbli", "pyweno", "pyclaw_2012", "hope", "jax_fluids", "jax_shock")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text())


def source_index() -> dict[str, Any]:
    return {
        name: {"path": relative, "sha256": sha256(ROOT / relative)}
        for name, relative in SOURCE_RECORDS.items()
    }


def claim_matrix() -> list[dict[str, Any]]:
    return [
        {
            "id": "M1",
            "status": "established",
            "statement": "The exact generator reproduces polynomials through each declared candidate and optimal-stencil degree for orders 5--15, with positive optimal weights summing exactly to one.",
            "sources": ["scalar_qualification"],
            "paper_role": "core mathematics",
        },
        {
            "id": "M2",
            "status": "established",
            "statement": "Generated order five reproduces the known Jiang--Shu coefficients and agrees with the independently written GradFlow WENO-5 scalar seed within its frozen oracle bound.",
            "sources": ["scalar_qualification"],
            "paper_role": "lineage/oracle",
        },
        {
            "id": "O1",
            "status": "observed",
            "statement": "The qualified scalar periodic implementation converges on the smooth family and remains conservative, differentiable, device-consistent, and full-graph compilable for the tested orders and representative gates.",
            "sources": ["scalar_qualification"],
            "paper_role": "core numerical result",
        },
        {
            "id": "O2",
            "status": "observed",
            "statement": "Classical WENO-JS loses design order on the frozen higher-order critical-point family; the loss is formulation behavior, not hidden by retuning epsilon.",
            "sources": ["scalar_qualification", "a1_numerical_limits"],
            "paper_role": "numerical limitation",
        },
        {
            "id": "O3",
            "status": "observed",
            "statement": "The generated reconstruction transfers through the qualified face-frozen Roe-characteristic 3-D Euler path for orders 5--15 under its distinct Shu epsilon, LF, grid, and time-integration contract.",
            "sources": ["characteristic_qualification", "boundary_shock_qualification"],
            "paper_role": "system qualification",
        },
        {
            "id": "O4",
            "status": "observed",
            "statement": "A scalar binary32 indicator/weight-formation seam can preserve frozen accuracy and accelerate compiled execution, but the same policy does not satisfy the strict higher-order characteristic-Euler contract.",
            "sources": ["scalar_mixed_precision", "scalar_mixed_performance", "characteristic_mixed_precision"],
            "paper_role": "precision boundary",
        },
        {
            "id": "O5",
            "status": "observed",
            "statement": "Logical single-owner face reconstruction reduces ordinary-PyTorch 3-D RHS time at every valid screened endpoint and reduces compiler temporary allocation at moderate/large sizes.",
            "sources": ["face_ownership_screen"],
            "paper_role": "representation result",
        },
        {
            "id": "O6",
            "status": "observed",
            "statement": "The matched native-CUDA WENO-5 face-once schedule exhibits an approximately twofold large-grid resident speedup but pays approximately twofold global workspace.",
            "sources": ["native_face_ownership"],
            "paper_role": "fixed native control",
        },
        {
            "id": "O7",
            "status": "observed",
            "statement": "Coefficient-basis diagnostics and exact rational complexity grow strongly with order, while sampled roundoff onset moves to progressively coarser grids under the A1 scalar contract.",
            "sources": ["a1_numerical_limits"],
            "paper_role": "order-dependent numerical limit",
        },
        {
            "id": "O8",
            "status": "observed",
            "statement": "Scalar epsilon 1e-29 is numerically indistinguishable from the 1e-40 comparison lane in the frozen N=128 amplitude range; larger epsilons can materially change scale-dependent smooth and critical cases.",
            "sources": ["a1_numerical_limits"],
            "paper_role": "epsilon boundary",
        },
        {
            "id": "I1",
            "status": "inferred",
            "statement": "Earlier sampled roundoff onset at higher order is consistent with growing coefficient/expression sensitivity, but A1 does not causally apportion the floor among coefficient conversion, evaluation order, cancellation, and input rounding.",
            "sources": ["a1_numerical_limits"],
            "paper_role": "bounded interpretation",
        },
        {
            "id": "U1",
            "status": "untested",
            "statement": "The complete matched order-5--15 cold/warm/AOT/CPU/GPU performance surface is not yet measured.",
            "sources": [],
            "paper_role": "A2 gate",
        },
        {
            "id": "U2",
            "status": "untested",
            "statement": "No independently gradient-checked inverse or sensitivity application has yet demonstrated differentiable utility.",
            "sources": [],
            "paper_role": "A3 gate",
        },
        {
            "id": "U3",
            "status": "untested",
            "statement": "MPS, orders above 15, general boundaries in multiple dimensions, Navier--Stokes, curvilinear geometry, and production aerospace workflows remain unqualified.",
            "sources": [],
            "paper_role": "explicit exclusion",
        },
        {
            "id": "P1",
            "status": "prohibited",
            "statement": "GradFlow is the first arbitrary-order WENO generator, first PyTorch WENO, first differentiable WENO, or first GPU WENO.",
            "sources": ["literature_claims", "literature_studies"],
            "paper_role": "prohibited novelty wording",
        },
        {
            "id": "P2",
            "status": "prohibited",
            "statement": "Finite-difference WENO, PyTorch, GPUs, or GradFlow are universally faster or superior across formulations and endpoints.",
            "sources": ["literature_claims", "fd_fv_prepared_aot", "face_ownership_screen"],
            "paper_role": "prohibited universal wording",
        },
        {
            "id": "P3",
            "status": "prohibited",
            "statement": "Current GradFlow is a general, real-time, production-ready aerospace CFD solver.",
            "sources": ["characteristic_qualification"],
            "paper_role": "prohibited product wording",
        },
    ]


def numerical_summary(limits: dict[str, Any]) -> list[dict[str, Any]]:
    coefficients = {item["order"]: item for item in limits["coefficient_diagnostics"]}
    roundoff = {
        (item["order"], item["dtype"]): item for item in limits["roundoff_sweeps"]
    }
    epsilon = {item["order"]: item for item in limits["epsilon_sweeps"]}
    return [
        {
            "order": order,
            "minimum_optimal_weight": coefficients[order]["minimum_optimal_weight"],
            "weight_dynamic_range": coefficients[order]["optimal_weight_dynamic_range"],
            "maximum_candidate_l1": coefficients[order]["maximum_candidate_coefficient_l1"],
            "full_moment_condition_2": coefficients[order]["full_moment_condition_2"],
            "maximum_smoothness_condition_2": coefficients[order]["maximum_smoothness_restricted_condition_2"],
            "float32_floor_l2": roundoff[(order, "float32")]["sampled_minimum_l2"],
            "float32_floor_n": roundoff[(order, "float32")]["sampled_minimum_n"],
            "float32_onset_n": roundoff[(order, "float32")]["first_sampled_roundoff_onset_n"],
            "float64_floor_l2": roundoff[(order, "float64")]["sampled_minimum_l2"],
            "float64_floor_n": roundoff[(order, "float64")]["sampled_minimum_n"],
            "float64_onset_n": roundoff[(order, "float64")]["first_sampled_roundoff_onset_n"],
            "epsilon_material_change_count": epsilon[order]["material_change_count"],
        }
        for order in ORDERS
    ]


def prior_art(studies: dict[str, Any]) -> list[dict[str, Any]]:
    selected = {item["id"]: item for item in studies["studies"]}
    fields = (
        "id",
        "title",
        "spatial_formulation",
        "weno_family_orders",
        "coefficient_policy",
        "reconstruction",
        "implementation",
        "hardware",
        "autodiff",
        "correctness_performance",
        "maintenance_license",
        "evidence",
    )
    return [{field: selected[item][field] for field in fields} for item in PRIOR_ART_IDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing existing output: {arguments.output}")
    limits = load(SOURCE_RECORDS["a1_numerical_limits"])
    studies = load(SOURCE_RECORDS["literature_studies"])
    document = {
        "schema": "gradflow-academic-a1-consolidation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "source_records": source_index(),
        "claims": claim_matrix(),
        "numerical_summary": numerical_summary(limits),
        "prior_art": prior_art(studies),
        "first_paper_scope": {
            "headline": "exact-generated ordinary-PyTorch FD-WENO-JS orders 5--15",
            "fd_fv_role": "supporting study, not a headline superiority claim",
            "native_cuda_role": "fixed WENO-5 schedule control",
            "dveb_role": "optional fixed comparator; cannot block the paper",
            "remaining_gates": ["A2", "A3", "A4"],
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "sha256": sha256(arguments.output)}))


if __name__ == "__main__":
    main()
