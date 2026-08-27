from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path

import pytest
import torch

import gradflow.fv_weno5 as implementation
from gradflow import (
    FV_WENO5_FORMULATION_ID,
    fv_global_lax_friedrichs_flux,
    fv_weno5_face_states,
    fv_weno5_rhs,
)
from experiments.fd_fv_qualification.verify_phase_3 import (
    main as verify_phase_3,
)
from experiments.fd_fv_qualification.verify_phase_3r import (
    main as verify_phase_3r,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = (
    ROOT
    / "experiments/fd_fv_contract/results/phase_2_20260827/oracle_cases.json"
)


def _fractions(values: list[str]) -> torch.Tensor:
    return torch.tensor(
        [float(Fraction(value)) for value in values],
        dtype=torch.float64,
    )


def _oracle() -> dict:
    return json.loads(ORACLE_PATH.read_text())


def test_formulation_identity_is_frozen() -> None:
    assert FV_WENO5_FORMULATION_ID == "fv_dimensional_js5_global_lf_periodic_v1"


@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_float64_matches_exact_fraction_oracle(direction: str) -> None:
    record = _oracle()["semidiscrete"]
    values = _fractions(record["deterministic_cell_averages"])
    case = record["linear_advection"][direction]
    speed = float(Fraction(case["speed"]))
    left, right = fv_weno5_face_states(values)
    fluxes = fv_global_lax_friedrichs_flux(
        left,
        right,
        lambda state: speed * state,
        abs(speed),
    )
    rhs = fv_weno5_rhs(
        values,
        float(Fraction(record["spacing"])),
        lambda state: speed * state,
        abs(speed),
    )
    torch.testing.assert_close(
        left, _fractions(case["left_face_states"]), rtol=0.0, atol=2.0e-13
    )
    torch.testing.assert_close(
        right, _fractions(case["right_face_states"]), rtol=0.0, atol=2.0e-13
    )
    torch.testing.assert_close(
        fluxes, _fractions(case["face_fluxes"]), rtol=0.0, atol=2.0e-13
    )
    torch.testing.assert_close(
        rhs, _fractions(case["rhs"]), rtol=0.0, atol=2.0e-13
    )


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 2.0e-6), (torch.float64, 5.0e-15)],
)
def test_constant_state_and_conservation(dtype: torch.dtype, atol: float) -> None:
    state = torch.full((3, 37), 7.0 / 3.0, dtype=dtype)
    left, right = fv_weno5_face_states(state)
    rhs = fv_weno5_rhs(state, 1.0 / 37.0, lambda value: 2.0 * value, 2.0)
    torch.testing.assert_close(left, state, rtol=0.0, atol=atol)
    torch.testing.assert_close(right, state, rtol=0.0, atol=atol)
    torch.testing.assert_close(rhs, torch.zeros_like(rhs), rtol=0.0, atol=atol)

    generator = torch.Generator().manual_seed(20260827)
    random_state = torch.randn((3, 37), generator=generator, dtype=dtype)
    random_rhs = fv_weno5_rhs(
        random_state,
        1.0 / 37.0,
        lambda value: 0.5 * value.square(),
        2.0,
    )
    residual = torch.abs(torch.sum(random_rhs, dim=-1))
    bound = 8.0 * torch.finfo(dtype).eps * torch.sum(torch.abs(random_rhs), dim=-1)
    assert torch.all(residual <= bound)


@pytest.mark.parametrize("speed", [1.0, -1.0])
def test_smooth_spatial_errors_decrease(speed: float) -> None:
    errors = []
    for cells in (32, 48, 72, 108):
        dx = 1.0 / cells
        faces = torch.arange(cells + 1, dtype=torch.float64) * dx
        cell_averages = (
            (torch.cos(2.0 * math.pi * faces[:-1])
             - torch.cos(2.0 * math.pi * faces[1:]))
            / (2.0 * math.pi * dx)
            + 0.15
            * (torch.sin(6.0 * math.pi * faces[1:])
               - torch.sin(6.0 * math.pi * faces[:-1]))
            / (6.0 * math.pi * dx)
        )
        exact = -speed * (
            torch.sin(2.0 * math.pi * faces[1:])
            + 0.15 * torch.cos(6.0 * math.pi * faces[1:])
            - torch.sin(2.0 * math.pi * faces[:-1])
            - 0.15 * torch.cos(6.0 * math.pi * faces[:-1])
        ) / dx
        actual = fv_weno5_rhs(
            cell_averages,
            dx,
            lambda value, speed=speed: speed * value,
            abs(speed),
        )
        errors.append(torch.sqrt(torch.mean((actual - exact).square())).item())
    rates = [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors, errors[1:], (32, 48, 72), (48, 72, 108)
        )
    ]
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert all(math.isfinite(rate) for rate in rates)


def test_axis_and_refusal_contract() -> None:
    state = torch.randn(2, 11, 3, dtype=torch.float64)
    left, right = fv_weno5_face_states(state, axis=1)
    moved_left, moved_right = fv_weno5_face_states(state.movedim(1, -1))
    torch.testing.assert_close(left, moved_left.movedim(-1, 1))
    torch.testing.assert_close(right, moved_right.movedim(-1, 1))

    with pytest.raises(TypeError, match="float32 or float64"):
        fv_weno5_face_states(torch.ones(8, dtype=torch.int64))
    with pytest.raises(ValueError, match="at least five"):
        fv_weno5_face_states(torch.ones(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="outside"):
        fv_weno5_face_states(torch.ones(8, dtype=torch.float64), axis=2)
    with pytest.raises(ValueError, match="bias"):
        implementation._fv_weno5_reconstruct(
            torch.ones(8, dtype=torch.float64), bias="center", axis=-1
        )
    with pytest.raises(ValueError, match="finite and positive"):
        fv_weno5_rhs(
            torch.ones(8, dtype=torch.float64), 0.0, lambda value: value, 1.0
        )
    with pytest.raises(ValueError, match="finite and positive"):
        fv_weno5_rhs(
            torch.ones(8, dtype=torch.float64), 1.0, lambda value: value, -1.0
        )
    with pytest.raises(ValueError, match="scalar"):
        fv_weno5_rhs(
            torch.ones(8, dtype=torch.float64),
            1.0,
            lambda value: value,
            torch.ones(2, dtype=torch.float64),
        )
    with pytest.raises(TypeError, match="state dtype"):
        fv_weno5_rhs(
            torch.ones(8, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float32),
            lambda value: value,
            1.0,
        )
    with pytest.raises(ValueError, match="preserve shape"):
        fv_weno5_rhs(
            torch.ones(8, dtype=torch.float64),
            1.0,
            lambda value: value[:-1],
            1.0,
        )
    with pytest.raises(TypeError, match="preserve the state dtype"):
        fv_weno5_rhs(
            torch.ones(8, dtype=torch.float64),
            1.0,
            lambda value: value.float(),
            1.0,
        )


def test_float64_gradcheck() -> None:
    state = torch.linspace(-0.4, 0.7, 19, dtype=torch.float64, requires_grad=True)

    def objective(values: torch.Tensor) -> torch.Tensor:
        rhs = fv_weno5_rhs(
            values,
            1.0 / 19.0,
            lambda value: 0.5 * value.square(),
            1.0,
        )
        return rhs.square().mean()

    assert torch.autograd.gradcheck(
        objective,
        (state,),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
    )


def test_torch_compile_fullgraph_cpu() -> None:
    state = torch.linspace(-0.4, 0.7, 37, dtype=torch.float64)

    def rhs(values: torch.Tensor) -> torch.Tensor:
        return fv_weno5_rhs(values, 1.0 / 37.0, lambda value: value, 1.0)

    expected = rhs(state)
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(rhs)(state)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    torch._dynamo.reset()
    compiled = torch.compile(rhs, fullgraph=True, dynamic=False)
    torch.testing.assert_close(compiled(state), expected, rtol=0.0, atol=2.0e-12)


def test_source_has_no_transfer_or_scalar_extraction_calls() -> None:
    source = (ROOT / "src/gradflow/fv_weno5.py").read_text()
    for forbidden in (".cpu(", ".cuda(", ".to(", ".item(", ".numpy("):
        assert forbidden not in source


def test_frozen_phase_3_record_verifies() -> None:
    verify_phase_3()


def test_frozen_phase_3r_record_verifies() -> None:
    verify_phase_3r()
