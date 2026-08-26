#!/usr/bin/env python3
"""Produce the machine-readable arbitrary-order WENO-JS qualification record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from gradflow import (  # noqa: E402
    QUALIFIED_ORDERS,
    WENOJS,
    generate_weno_js_coefficients,
    weno5_rhs,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_payload(order: int) -> dict[str, object]:
    coefficients = generate_weno_js_coefficients(order)

    def vector(values) -> list[str]:
        return [str(value) for value in values]

    return {
        "order": order,
        "substencil_width": coefficients.substencil_width,
        "candidate_offsets": coefficients.candidate_offsets,
        "candidate_coefficients": [
            vector(values) for values in coefficients.candidate_coefficients
        ],
        "optimal_weights": vector(coefficients.optimal_weights),
        "full_offsets": coefficients.full_offsets,
        "full_coefficients": vector(coefficients.full_coefficients),
        "smoothness_matrices": [
            [vector(row) for row in matrix]
            for matrix in coefficients.smoothness_matrices
        ],
        "smoothness_factors": [
            [[str(weight), vector(values)] for weight, values in factors]
            for factors in coefficients.smoothness_factors
        ],
    }


def weno5_equivalence() -> dict[str, float]:
    n = 257
    x = torch.arange(n, dtype=torch.float64) / n
    state = 0.4 + torch.sin(2.0 * math.pi * x) + 0.1 * torch.cos(6.0 * math.pi * x)
    cases = {
        "positive_linear": (lambda q: q, lambda q: torch.ones_like(q), 1.0),
        "negative_linear": (lambda q: -q, lambda q: -torch.ones_like(q), 1.0),
        "burgers": (lambda q: 0.5 * q.square(), lambda q: q, None),
    }
    result = {}
    for name, (flux, derivative, alpha) in cases.items():
        expected = weno5_rhs(state, 1.0 / n, flux, derivative, alpha=alpha)
        actual = WENOJS(5).rhs(state, 1.0 / n, flux, derivative, alpha=alpha)
        result[name] = float(torch.max(torch.abs(actual - expected)))
    return result


def convergence(order: int) -> dict[str, object]:
    sizes = (24, 36, 54, 81)
    errors = []
    scheme = WENOJS(order)
    for n in sizes:
        x = torch.arange(n, dtype=torch.float64) / n
        state = torch.sin(6.0 * math.pi * x) + 0.15 * torch.cos(8.0 * math.pi * x)
        exact = -(
            6.0 * math.pi * torch.cos(6.0 * math.pi * x)
            - 1.2 * math.pi * torch.sin(8.0 * math.pi * x)
        )
        actual = scheme.rhs(state, 1.0 / n, lambda q: q, alpha=1.0)
        errors.append(float(torch.sqrt(torch.mean((actual - exact).square()))))
    rates = [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(errors, errors[1:], sizes, sizes[1:])
    ]
    return {"sizes": sizes, "l2_errors": errors, "rates": rates}


def critical_point(order: int) -> dict[str, object]:
    sizes = (32, 64, 128, 256)
    errors = []
    scheme = WENOJS(order)
    for n in sizes:
        x = torch.arange(n, dtype=torch.float64) / n
        state = torch.sin(2.0 * math.pi * x).pow(3)
        actual = scheme.rhs(state, 1.0 / n, lambda q: q, alpha=1.0)
        errors.append(abs(float(actual[0])))
    rates = [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:])]
    return {
        "family": "sin(2*pi*x)^3 at x=0",
        "sizes": sizes,
        "point_errors": errors,
        "rates": rates,
    }


def conservation(order: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(20260826 + order)
    state = torch.randn(3, 257, generator=generator, dtype=torch.float64)
    rhs = WENOJS(order).rhs(state, 1.0 / 257, lambda q: 0.5 * q.square(), lambda q: q)
    residual = torch.abs(torch.sum(rhs, dim=-1))
    scale = torch.finfo(torch.float64).eps * torch.sum(torch.abs(rhs), dim=-1)
    return {
        "maximum_absolute_sum": float(torch.max(residual)),
        "maximum_roundoff_ratio": float(torch.max(residual / scale)),
    }


def device_agreement(order: int, dtype: torch.dtype) -> float | None:
    if not torch.cuda.is_available():
        return None
    n = 37
    x = torch.arange(n, dtype=dtype) / n
    state = 0.3 + torch.sin(2.0 * math.pi * x) + 0.1 * torch.cos(6.0 * math.pi * x)
    scheme = WENOJS(order)
    cpu = scheme.rhs(state, 1.0 / n, lambda q: 0.5 * q.square(), alpha=1.5)
    cuda = scheme.rhs(
        state.cuda(), 1.0 / n, lambda q: 0.5 * q.square(), alpha=1.5
    ).cpu()
    return float(torch.max(torch.abs(cpu - cuda)))


def gradcheck(order: int) -> bool:
    generator = torch.Generator().manual_seed(3000 + order)
    state = 0.2 * torch.randn(
        max(19, order + 2), generator=generator, dtype=torch.float64
    )
    state.requires_grad_()
    scheme = WENOJS(order)

    def result(values: torch.Tensor) -> torch.Tensor:
        rhs = scheme.rhs(
            values,
            1.0 / values.shape[-1],
            lambda q: 0.5 * q.square(),
            lambda q: q,
            alpha=1.0,
        )
        return rhs.square().mean()

    return bool(
        torch.autograd.gradcheck(result, (state,), eps=1.0e-6, atol=2.0e-5, rtol=2.0e-4)
    )


def compile_gate(order: int, device: str) -> dict[str, object] | None:
    if device == "cuda" and not torch.cuda.is_available():
        return None
    dtype = torch.float32 if device == "cuda" else torch.float64
    n = max(32, order + 2)
    state = torch.linspace(-0.7, 0.8, n, dtype=dtype, device=device)
    scheme = WENOJS(order)

    def call(values: torch.Tensor) -> torch.Tensor:
        return scheme.rhs(values, 1.0 / n, lambda q: q, alpha=1.0)

    expected = call(state)
    explanation = torch._dynamo.explain(call)(state)
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    actual = compiled(state)
    return {
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "maximum_absolute_error": float(torch.max(torch.abs(actual - expected))),
        "finite": bool(torch.all(torch.isfinite(actual))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")

    payloads = {str(order): exact_payload(order) for order in QUALIFIED_ORDERS}
    canonical = json.dumps(payloads, sort_keys=True, separators=(",", ":")).encode()
    per_order = {}
    for order in QUALIFIED_ORDERS:
        per_order[str(order)] = {
            "optimal_weights": payloads[str(order)]["optimal_weights"],
            "convergence": convergence(order),
            "critical_point": critical_point(order),
            "conservation": conservation(order),
            "cpu_cuda_max_abs_float32": device_agreement(order, torch.float32),
            "cpu_cuda_max_abs_float64": device_agreement(order, torch.float64),
        }
    compile_results = {
        str(order): {device: compile_gate(order, device) for device in ("cpu", "cuda")}
        for order in (5, 11, 15)
    }
    source_paths = [
        ROOT / "src/gradflow/weno_js.py",
        ROOT / "src/gradflow/weno_js_coefficients.py",
        ROOT / "tests/test_weno_js.py",
        ROOT / "tests/test_weno_js_coefficients.py",
        Path(__file__),
    ]
    report = {
        "schema": "gradflow-arbitrary-order-weno-js-qualification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str((ROOT / "docs/ARBITRARY_ORDER_WENO_JS_PROTOCOL.md").resolve()),
        "gradflow_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "qualified_orders": QUALIFIED_ORDERS,
        "exact_coefficient_payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "weno5_maximum_absolute_difference": weno5_equivalence(),
        "orders": per_order,
        "gradcheck": {str(order): gradcheck(order) for order in (5, 11, 15)},
        "torch_compile": compile_results,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "claim_boundary": {
            "performance_measured": False,
            "dveb_changed": False,
            "boundaries": "unique periodic nodes only",
            "equations": "scalar conservation laws only",
            "constructible_beyond_15_is_qualified": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
