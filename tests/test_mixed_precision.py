import json
import subprocess
import sys
from pathlib import Path

import torch

from gradflow import PRECISION_BLOCKS, WENOJS, WENOJSPrecisionPolicy
from experiments.mixed_precision.benchmark_worker import (
    POLICY_MASKS,
    policy_for_name,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_128_precision_assignments_are_distinct() -> None:
    assignments = set()
    for mask in range(128):
        values = tuple(
            torch.float32 if mask & (1 << index) else torch.float64
            for index, _ in enumerate(PRECISION_BLOCKS)
        )
        policy = WENOJSPrecisionPolicy(**dict(zip(PRECISION_BLOCKS, values)))
        assignments.add(tuple(policy.as_names().items()))
    assert len(assignments) == 128


def test_all_float32_internal_policy_returns_float64_state_dtype() -> None:
    state = torch.linspace(-1.0, 1.0, 65, dtype=torch.float64)
    policy = WENOJSPrecisionPolicy(
        **{block: torch.float32 for block in PRECISION_BLOCKS}
    )
    rhs = WENOJS(5, precision=policy).rhs(
        state, 1.0 / state.shape[-1], lambda q: q, alpha=1.0
    )
    assert rhs.dtype is torch.float64
    assert rhs.device == state.device
    assert torch.isfinite(rhs).all()


def test_frozen_benchmark_policies_select_intended_blocks() -> None:
    assert set(POLICY_MASKS) == {
        "all_f64",
        "indicators_f32",
        "weight_formation_f32",
        "indicators_and_weight_formation_f32",
        "all_internal_f32",
    }
    combined = policy_for_name("indicators_and_weight_formation_f32").as_names()
    assert {name for name, dtype in combined.items() if dtype == "float32"} == {
        "indicators",
        "weight_formation",
    }


def test_restricted_search_smoke_record(tmp_path: Path) -> None:
    output = tmp_path / "partial"
    completed = subprocess.run(
        (
            sys.executable,
            str(ROOT / "experiments/mixed_precision/search.py"),
            "--output",
            str(output),
            "--orders",
            "5",
            "--masks",
            "0",
            "127",
        ),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout
    payload = json.loads((output / "search.json").read_text())
    assert not payload["complete_frozen_matrix"]
    assert len(payload["records"]) == 2
    assert {record["mask"] for record in payload["records"]} == {0, 127}
    assert (output / "SHA256SUMS").is_file()
