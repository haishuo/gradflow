#!/usr/bin/env python3
"""Offline checksum and semantic verifier for frozen U4-B evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXPECTED_OPENSBLI_COMMIT = "e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d"
EXPECTED_OPENSBLI_TREE = "0ff053443f6b243b2bd42475f98122306151427d"
EXPECTED_OPS_COMMIT = "c0af0f124469e5fd856b594a23ff1206c3e9c7a8"
EXPECTED_OPS_TREE = "82c3fd0c0b4724c6e8474e16f730e7560845235f"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(evidence: Path) -> None:
    entries = (evidence / "SHA256SUMS").read_text().splitlines()
    for entry in entries:
        expected, relative = entry.split("  ", maxsplit=1)
        actual = digest(evidence / relative)
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {relative}")


def conservation(rhs: np.ndarray) -> bool:
    total = float(np.sum(rhs, dtype=np.float64))
    absolute = float(np.sum(np.abs(rhs), dtype=np.float64))
    bound = float(32.0 * np.finfo(np.float64).eps * absolute)
    return abs(total) <= bound


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "evidence",
        type=Path,
        nargs="?",
        default=HERE / "evidence" / "u4b_20260830",
    )
    args = parser.parse_args()
    evidence = args.evidence.resolve()

    verify_checksums(evidence)
    qualification = json.loads((evidence / "qualification.json").read_text())
    arrays = np.load(evidence / "qualification_arrays.npz")

    assert qualification["schema"] == "gradflow.academic_u4b.qualification.v1"
    assert qualification["decision"] == "matched_operator_adapted_qualified"
    assert qualification["performance_interpretation_prohibited"] is True
    assert all(qualification["gates"].values())
    assert qualification["upstream"]["opensbli"] == {
        "commit": EXPECTED_OPENSBLI_COMMIT,
        "tree": EXPECTED_OPENSBLI_TREE,
    }
    assert qualification["upstream"]["ops"] == {
        "commit": EXPECTED_OPS_COMMIT,
        "tree": EXPECTED_OPS_TREE,
    }

    sources = qualification["source_hashes"]
    assert sources["adapter"] == digest(HERE / "adapter" / "opensbli_scalar_u4b.py")
    assert sources["adapter_patch"] == digest(HERE / "adapter" / "opensbli-u4b.patch")
    assert sources["residual_exposer"] == digest(
        HERE / "adapter" / "expose_first_residual.py"
    )

    patch = (HERE / "adapter" / "opensbli-u4b.patch").read_text()
    assert "generate_smoothness_coefficients" not in patch
    assert "generate_reconstruction" not in patch
    assert "characteristic_flux_splitting" not in patch
    assert "_discretise_derivative" not in patch

    pointwise_tolerance = float(qualification["protocol"]["pointwise_atol"])
    sine_errors: list[float] = []
    for record in qualification["cases"]:
        label = f"{record['case']}_n{record['size']}"
        external = arrays[f"{label}_external_rhs"]
        canonical = arrays[f"{label}_canonical_rhs"]
        state = arrays[f"{label}_state"]
        assert external.shape == canonical.shape == state.shape == (record["size"],)
        assert external.dtype == canonical.dtype == state.dtype == np.float64
        assert np.all(np.isfinite(external))
        assert np.all(np.isfinite(canonical))
        assert conservation(external)
        assert conservation(canonical)
        difference = float(np.max(np.abs(external - canonical)))
        assert math.isclose(
            difference,
            float(record["rhs_max_abs_difference"]),
            rel_tol=0.0,
            abs_tol=1.0e-18,
        )
        if record["case"] in ("state_a", "state_b"):
            assert difference <= pointwise_tolerance
        elif record["case"] == "constant":
            assert float(np.max(np.abs(external))) <= pointwise_tolerance
        elif record["case"] == "sine":
            analytic = arrays[f"{label}_analytic_rhs"]
            error = float(np.sqrt(np.mean((external - analytic) ** 2)))
            assert math.isclose(
                error,
                float(record["external_l2_error"]),
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            sine_errors.append(error)

    rates = [math.log2(coarse / fine) for coarse, fine in zip(sine_errors, sine_errors[1:])]
    assert all(rate > 4.8 for rate in rates)
    print(
        json.dumps(
            {
                "decision": qualification["decision"],
                "checksums": "passed",
                "semantic_gates": "passed",
                "external_convergence_rates": rates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
