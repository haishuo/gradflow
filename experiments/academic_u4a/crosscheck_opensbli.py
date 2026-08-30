#!/usr/bin/env python3
"""Cross-check OpenSBLI's exact WENO data without importing PyTorch."""

from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import warnings


ROOT = Path(__file__).resolve().parents[2]
ORDERS = (5, 7, 9, 11, 13, 15)
SMOOTHNESS_ORDERS = (5, 7, 9)


def load_gradflow_coefficients():
    path = ROOT / "src" / "gradflow" / "weno_js_coefficients.py"
    spec = importlib.util.spec_from_file_location("u4a_gradflow_coefficients", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensbli-root", type=Path, required=True)
    arguments = parser.parse_args()
    sys.path.insert(0, str(arguments.opensbli_root.resolve()))

    # OpenSBLI's pinned revision imports a SymPy module path removed by newer
    # SymPy. The alias changes only import routing for this source audit.
    import sympy
    import sympy.printing.c as sympy_c

    sys.modules["sympy.printing.ccode"] = sympy_c
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from opensbli.schemes.spatial.weno import ConfigureWeno, JS_smoothness

    gradflow = load_gradflow_coefficients()
    candidate_records = []
    for order in ORDERS:
        width = (order + 1) // 2
        expected = gradflow.generate_weno_js_coefficients(order)
        actual = ConfigureWeno(width, 1)
        candidate_indices = reversed(range(width))
        candidate_indices = tuple(candidate_indices)
        offsets = tuple(
            tuple(actual.func_points[index * width + item] for item in range(width))
            for index in candidate_indices
        )
        coefficients = tuple(
            tuple(Fraction(actual.c_rj[(index, item)]) for item in range(width))
            for index in candidate_indices
        )
        weights = tuple(
            Fraction(actual.opt_weights[(0, index)]) for index in candidate_indices
        )
        candidate_records.append(
            {
                "order": order,
                "offsets": offsets == expected.candidate_offsets,
                "coefficients": coefficients == expected.candidate_coefficients,
                "optimal_weights": weights == expected.optimal_weights,
                "exact_match": (
                    offsets == expected.candidate_offsets
                    and coefficients == expected.candidate_coefficients
                    and weights == expected.optimal_weights
                ),
            }
        )

    smoothness_records = []
    for order in SMOOTHNESS_ORDERS:
        width = (order + 1) // 2
        expected = gradflow.generate_weno_js_coefficients(order)
        actual = JS_smoothness(width)
        matrices = []
        for candidate in reversed(range(width)):
            matrix = []
            for row in range(width):
                values = []
                for column in range(width):
                    left, right = sorted((row, column))
                    expanded = Fraction(
                        actual.smooth_coeffs.get((candidate, left, right), 0)
                    )
                    values.append(expanded if row == column else expanded / 2)
                matrix.append(tuple(values))
            matrices.append(tuple(matrix))
        smoothness_records.append(
            {
                "order": order,
                "exact_match": tuple(matrices) == expected.smoothness_matrices,
            }
        )

    print(
        json.dumps(
            {
                "schema": "gradflow-academic-u4a-symbolic-crosscheck-v1",
                "python": ".".join(map(str, sys.version_info[:3])),
                "sympy": sympy.__version__,
                "candidate_mapping": "reverse_OpenSBLI_right_candidate_index",
                "candidate_checks": candidate_records,
                "smoothness_checks": smoothness_records,
                "performance_result": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
