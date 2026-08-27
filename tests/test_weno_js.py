import math
from pathlib import Path

import pytest
import torch

from gradflow import (
    PRECISION_BLOCKS,
    QUALIFIED_ORDERS,
    WENOJS,
    WENOJSPrecisionPolicy,
    weno5_rhs,
)


def _identity(q: torch.Tensor) -> torch.Tensor:
    return q


def _positive_derivative(q: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(q)


def _negative_flux(q: torch.Tensor) -> torch.Tensor:
    return -q


def _negative_derivative(q: torch.Tensor) -> torch.Tensor:
    return -torch.ones_like(q)


def _burgers_flux(q: torch.Tensor) -> torch.Tensor:
    return 0.5 * q.square()


@pytest.mark.parametrize("flux_name", ["positive", "negative", "burgers"])
def test_generated_order_five_matches_canonical_seed(flux_name: str) -> None:
    n = 257
    x = torch.arange(n, dtype=torch.float64) / n
    u = 0.4 + torch.sin(2.0 * math.pi * x) + 0.1 * torch.cos(6.0 * math.pi * x)
    if flux_name == "positive":
        flux = _identity
        derivative = _positive_derivative
        alpha = 1.0
    elif flux_name == "negative":
        flux = _negative_flux
        derivative = _negative_derivative
        alpha = 1.0
    else:
        flux = _burgers_flux
        derivative = _identity
        alpha = None
    expected = weno5_rhs(u, 1.0 / n, flux, derivative, alpha=alpha)
    actual = WENOJS(5).rhs(u, 1.0 / n, flux, derivative, alpha=alpha)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=5.0e-13)


@pytest.mark.parametrize("order", QUALIFIED_ORDERS)
def test_smooth_periodic_spatial_convergence(order: int) -> None:
    sizes = (24, 36, 54, 81)
    errors = []
    scheme = WENOJS(order)
    for n in sizes:
        x = torch.arange(n, dtype=torch.float64) / n
        u = torch.sin(6.0 * math.pi * x) + 0.15 * torch.cos(8.0 * math.pi * x)
        exact = -(
            6.0 * math.pi * torch.cos(6.0 * math.pi * x)
            - 1.2 * math.pi * torch.sin(8.0 * math.pi * x)
        )
        actual = scheme.rhs(u, 1.0 / n, lambda q: q, alpha=1.0)
        errors.append(torch.sqrt(torch.mean((actual - exact).square())).item())
    rates = [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(errors, errors[1:], sizes, sizes[1:])
    ]
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert max(rates) >= order - 1.0, f"errors={errors}, rates={rates}"


@pytest.mark.parametrize("order", QUALIFIED_ORDERS)
def test_high_order_critical_point_family_converges(order: int) -> None:
    errors = []
    scheme = WENOJS(order)
    for n in (32, 64, 128, 256):
        x = torch.arange(n, dtype=torch.float64) / n
        u = torch.sin(2.0 * math.pi * x).pow(3)
        actual = scheme.rhs(u, 1.0 / n, lambda q: q, alpha=1.0)
        errors.append(abs(actual[0].item()))
    assert all(math.isfinite(error) for error in errors)
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))


@pytest.mark.parametrize("order", QUALIFIED_ORDERS)
def test_periodic_conservation_and_float32_constant_stability(order: int) -> None:
    generator = torch.Generator().manual_seed(20260826 + order)
    u = torch.randn(3, 257, generator=generator, dtype=torch.float64)
    rhs = WENOJS(order).rhs(u, 1.0 / 257, lambda q: 0.5 * q.square(), lambda q: q)
    residual = torch.abs(torch.sum(rhs, dim=-1))
    bound = 8.0 * torch.finfo(torch.float64).eps * torch.sum(torch.abs(rhs), dim=-1)
    assert torch.all(residual <= bound)

    constant = torch.full((2, max(32, order)), 1.25, dtype=torch.float32)
    reconstruction = WENOJS(order).reconstruct(constant)
    torch.testing.assert_close(reconstruction, constant, rtol=0.0, atol=2.0e-6)
    assert torch.all(torch.isfinite(reconstruction))


def test_axis_and_refusal_contract() -> None:
    values = torch.randn(3, 32, 4, dtype=torch.float64)
    scheme = WENOJS(7)
    axis_result = scheme.reconstruct(values, axis=1)
    transposed = scheme.reconstruct(values.movedim(1, -1)).movedim(-1, 1)
    torch.testing.assert_close(axis_result, transposed, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="requires at least"):
        scheme.reconstruct(torch.ones(6, dtype=torch.float64))
    with pytest.raises(TypeError, match="float32 or float64"):
        scheme.reconstruct(torch.ones(8, dtype=torch.int64))
    with pytest.raises(ValueError, match="bias"):
        scheme.reconstruct(torch.ones(8, dtype=torch.float64), bias="center")


def test_precision_policy_is_explicit_and_default_preserving() -> None:
    state = torch.linspace(-0.7, 0.8, 37, dtype=torch.float64)
    default = WENOJS(7).rhs(state, 1.0 / 37, lambda q: q, alpha=1.0)
    explicit = WENOJS(
        7,
        precision=WENOJSPrecisionPolicy(
            **{block: torch.float64 for block in PRECISION_BLOCKS}
        ),
    ).rhs(state, 1.0 / 37, lambda q: q, alpha=1.0)
    torch.testing.assert_close(explicit, default, rtol=0.0, atol=0.0)

    mixed = WENOJS(
        7,
        precision=WENOJSPrecisionPolicy(
            indicators=torch.float32,
            weights=torch.float32,
        ),
    ).rhs(state, 1.0 / 37, lambda q: q, alpha=1.0)
    assert mixed.dtype is state.dtype
    assert mixed.device == state.device
    assert torch.isfinite(mixed).all()


def test_precision_policy_refuses_unsupported_dtypes() -> None:
    with pytest.raises(TypeError, match="float32, float64, or None"):
        WENOJSPrecisionPolicy(weights=torch.float16)
    with pytest.raises(ValueError, match="unknown WENO-JS precision block"):
        WENOJSPrecisionPolicy().dtype_for("made_up", torch.float64)


@pytest.mark.parametrize("order", [5, 11, 15])
def test_float64_gradcheck(order: int) -> None:
    generator = torch.Generator().manual_seed(3000 + order)
    state = torch.randn(max(19, order + 2), generator=generator, dtype=torch.float64)
    state = (0.2 * state).requires_grad_()
    scheme = WENOJS(order)

    def scalar_result(values: torch.Tensor) -> torch.Tensor:
        rhs = scheme.rhs(
            values,
            1.0 / values.shape[-1],
            lambda q: 0.5 * q.square(),
            lambda q: q,
            alpha=1.0,
        )
        return rhs.square().mean()

    assert torch.autograd.gradcheck(
        scalar_result, (state,), eps=1.0e-6, atol=2.0e-5, rtol=2.0e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("order", QUALIFIED_ORDERS)
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 2.0e-4), (torch.float64, 2.0e-11)],
)
def test_cpu_cuda_agreement(order: int, dtype: torch.dtype, atol: float) -> None:
    n = 37
    x = torch.arange(n, dtype=dtype) / n
    cpu = 0.3 + torch.sin(2.0 * math.pi * x) + 0.1 * torch.cos(6.0 * math.pi * x)
    scheme = WENOJS(order)
    expected = scheme.rhs(cpu, 1.0 / n, lambda q: 0.5 * q.square(), alpha=1.5)
    actual = scheme.rhs(
        cpu.cuda(), 1.0 / n, lambda q: 0.5 * q.square(), alpha=1.5
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("order", [5, 11, 15])
def test_torch_compile_fullgraph_cpu(order: int) -> None:
    n = max(32, order + 2)
    state = torch.linspace(-0.7, 0.8, n, dtype=torch.float64)
    scheme = WENOJS(order)

    def call(values: torch.Tensor) -> torch.Tensor:
        return scheme.rhs(values, 1.0 / n, lambda q: q, alpha=1.0)

    expected = call(state)
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    actual = compiled(state)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-12)
    explanation = torch._dynamo.explain(call)(state)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("order", [5, 11, 15])
def test_torch_compile_fullgraph_cuda(order: int) -> None:
    n = max(32, order + 2)
    state = torch.linspace(-0.7, 0.8, n, dtype=torch.float32, device="cuda")
    scheme = WENOJS(order)

    def call(values: torch.Tensor) -> torch.Tensor:
        return scheme.rhs(values, 1.0 / n, lambda q: q, alpha=1.0)

    expected = call(state)
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    actual = compiled(state)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=5.0e-5)


def test_numerical_source_has_no_hidden_transfer_or_numpy() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src/gradflow/weno_js.py"
    ).read_text()
    for forbidden in (".cpu(", ".cuda(", ".item(", ".numpy("):
        assert forbidden not in source
    # Phase D requires explicit dtype conversions. The sole conversion helper
    # supplies only ``dtype`` and therefore cannot change device.
    assert source.count(".to(dtype=dtype)") == 1
