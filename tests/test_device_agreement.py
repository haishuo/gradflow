import ast
import inspect
import math
import textwrap

import pytest
import torch

import gradflow.weno5 as implementation
from gradflow import ssp_rk3_step, weno5_rhs


def _initial_state(device: str) -> torch.Tensor:
    n = 257
    x = torch.arange(n, dtype=torch.float64, device=device) / n
    return torch.sin(2.0 * math.pi * x) + 0.1 * torch.cos(6.0 * math.pi * x)


def _rhs(state: torch.Tensor) -> torch.Tensor:
    return weno5_rhs(state, 1.0 / 257, lambda q: 0.5 * q * q, lambda q: q)


def _step(state: torch.Tensor) -> torch.Tensor:
    return ssp_rk3_step(state, 0.1 / 257, _rhs)


def test_eager_preserves_dtype_and_device() -> None:
    u = _initial_state("cpu")
    result = _step(u)
    assert result.dtype == torch.float64
    assert result.device == u.device


def test_torch_compile_fullgraph() -> None:
    u = _initial_state("cpu")
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(_step)(u)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert not explanation.break_reasons

    torch._dynamo.reset()
    compiled = torch.compile(_step, fullgraph=True)
    torch.testing.assert_close(compiled(u), _step(u), rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_float64_agreement() -> None:
    cpu = _step(_initial_state("cpu"))
    cuda = _step(_initial_state("cuda")).cpu()
    tolerance = max(1.0e-11, 8.0 * torch.finfo(torch.float64).eps * 257)
    torch.testing.assert_close(cuda, cpu, rtol=0.0, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_torch_compile_cuda_fullgraph() -> None:
    u = _initial_state("cuda")
    torch._dynamo.reset()
    compiled = torch.compile(_step, fullgraph=True)
    torch.testing.assert_close(compiled(u), _step(u), rtol=1.0e-13, atol=1.0e-13)


def test_numerical_path_has_no_transfer_or_scalar_extraction_calls() -> None:
    forbidden = {"item", "cpu", "cuda", "to", "numpy"}
    functions = (
        implementation._global_lax_friedrichs_speed,
        implementation._shift,
        implementation._weno_correction,
        implementation.weno5_rhs,
        implementation.weno5_rhs_gottlieb_periodic,
        implementation.ssp_rk3_step,
    )
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert calls.isdisjoint(forbidden), (function.__name__, calls & forbidden)
