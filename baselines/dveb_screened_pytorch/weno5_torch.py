"""Trunk 001 ordinary-PyTorch comparator: the pinned Gottlieb WENO-5.

Implements exactly the formulation declared in docs/trunks/001-weno5.md
(*Declared formulation*): IEEE f64 point samples at x_j = j*dx, periodic;
Lax-Friedrichs splitting of flux differences with global em = max|f'(u)|;
4th-order central flux plus one WENO correction per split-difference
family; 12x-scaled smoothness indicators with eps = 1e-29 inside the
square; linear weights (1, 6, 3); rhs_j = (fh_{j-1} - fh_j)/dx; SSP-RK3.

Ordinary PyTorch only: tensors, rolls/slices, elementwise math, conv1d,
torch.compile. No custom operators, no handwritten Triton, no CuPy/Cython.
Fields are (nb, n) f64 CUDA tensors, device-resident; nothing in the
numerical path touches the host.
"""

import torch

EPS = 1e-29
CFL = 0.4


def weno_correction(h1, h2, h3, h4):
    """WENO correction for one split-difference family (four neighbours)."""
    t1 = h1 - h2
    t2 = h2 - h3
    t3 = h3 - h4
    q1 = EPS + 13.0 * t1 * t1 + 3.0 * (h1 - 3.0 * h2) ** 2
    q2 = EPS + 13.0 * t2 * t2 + 3.0 * (h2 + h3) ** 2
    q3 = EPS + 13.0 * t3 * t3 + 3.0 * (3.0 * h3 - h4) ** 2
    q1, q2, q3 = q1 * q1, q2 * q2, q3 * q3
    s1 = q2 * q3
    s3 = 3.0 * q1 * q2
    t0 = 1.0 / (s1 + 6.0 * q1 * q3 + s3)
    return (s1 * t0 * (t2 - t1) + (0.5 * s3 * t0 - 0.25) * (t3 - t2)) / 3.0


def shift(v, k):
    """v_{j+k} with periodic wrap (torch.roll shifts right for positive)."""
    return torch.roll(v, shifts=-k, dims=-1)


def weno5_rhs(u, dx, a, em):
    """Fused-in-source WENO-5 spatial RHS; PyTorch decides the execution."""
    f = a * u
    du = shift(u, 1) - u              # du_j between samples j and j+1
    df = a * du
    gp = 0.5 * (df + em * du)         # positive split difference
    gm = gp - df                      # negative split difference

    central = (-shift(f, -1) + 7.0 * (f + shift(f, 1)) - shift(f, 2)) / 12.0
    plus = weno_correction(shift(gp, -2), shift(gp, -1), gp, shift(gp, 1))
    minus = weno_correction(-shift(gm, 2), -shift(gm, 1), -gm, -shift(gm, -1))
    fh = central + plus + minus       # numerical flux at face j+1/2
    return (shift(fh, -1) - fh) / dx  # (fh_{j-1} - fh_j)/dx


def _conv(v, kernel):
    """Circular 1-D convolution of (nb, n) with a short kernel, exact f64."""
    pad = (kernel.numel() - 1) // 2
    padded = torch.nn.functional.pad(v.unsqueeze(1), (pad, pad),
                                     mode="circular")
    return torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1)).squeeze(1)


_CONV_KERNELS = {}


def _stencil_kernels(dtype, device):
    """Constant conv kernels, created once per (dtype, device).

    Recorded incident: an earlier revision built these tensors inside the
    RHS, which silently performed two host-to-device PCIe copies on every
    evaluation (torch.tensor(list, device='cuda') uploads each call). That
    violated the no-in-loop-transfers constraint and is preserved here as
    ergonomic evidence; hoisting the constants is the ordinary fix."""
    key = (dtype, device)
    if key not in _CONV_KERNELS:
        dev = dict(dtype=dtype, device=device)
        _CONV_KERNELS[key] = (
            torch.tensor([0.0, -1.0, 1.0], **dev),                    # du
            torch.tensor([0.0, -1.0, 7.0, 7.0, -1.0], **dev) / 12.0,  # central
        )
    return _CONV_KERNELS[key]


def weno5_rhs_conv(u, dx, a, em):
    """Convolution-assisted formulation: identical mathematics, with the
    linear stencil pieces (neighbour difference, central flux) expressed as
    conv1d; the nonlinear corrections still use shifted views."""
    # conv1d cross-correlates, so kernels are written left-to-right in j.
    k_du, k_central = _stencil_kernels(u.dtype, u.device)
    du = _conv(u, k_du)
    df = a * du
    gp = 0.5 * (df + em * du)
    gm = gp - df
    central = _conv(a * u, k_central)
    plus = weno_correction(shift(gp, -2), shift(gp, -1), gp, shift(gp, 1))
    minus = weno_correction(-shift(gm, 2), -shift(gm, 1), -gm, -shift(gm, -1))
    fh = central + plus + minus
    return (shift(fh, -1) - fh) / dx


def rk3_step(u, dx, dt, a, em, rhs):
    """One SSP-RK3 step (identical coefficients to the DVEB program)."""
    u1 = u + dt * rhs(u, dx, a, em)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1, dx, a, em))
    return (u + 2.0 * (u2 + dt * rhs(u2, dx, a, em))) / 3.0


# ---------------------------------------------------------------------------
# The five declared comparator variants. Each entry: (rhs function, compile
# wrapper or None). Compilation is applied to the whole RK step so the
# compiler sees the full composition.
# ---------------------------------------------------------------------------

VARIANTS = {
    "eager": (weno5_rhs, None),
    "compile": (weno5_rhs, dict(fullgraph=True, dynamic=True)),
    "compile-ro": (weno5_rhs, dict(fullgraph=True, dynamic=True,
                                   mode="reduce-overhead")),
    "compile-ta": (weno5_rhs, dict(fullgraph=True, dynamic=True,
                                   mode="max-autotune")),
    "conv": (weno5_rhs_conv, None),
}


class Solver:
    def __init__(self, variant: str):
        self.name = variant
        rhs, compile_kwargs = VARIANTS[variant]
        self._rhs_raw = rhs
        if compile_kwargs is None:
            self.rhs = rhs
            self.step = lambda u, dx, dt, a, em: rk3_step(u, dx, dt, a, em,
                                                          rhs)
        else:
            self.rhs = torch.compile(rhs, **compile_kwargs)
            step = lambda u, dx, dt, a, em: rk3_step(u, dx, dt, a, em, rhs)
            self.step = torch.compile(step, **compile_kwargs)

    def solve(self, u, dx, dt, steps, a, em):
        # CUDA-Graphs modes (reduce-overhead / max-autotune) overwrite a
        # replay's output buffers on the next replay. Feeding each step's
        # output back as the next input therefore requires BOTH documented
        # remedies: mark the iteration boundary AND clone the output out of
        # the graph pool every step. The naive loop crashes without them,
        # and the clone is one extra full-field copy per step — ordinary,
        # documented usage, recorded as ergonomic/performance-relevant
        # evidence for the screen.
        uses_cudagraphs = self.name in ("compile-ro", "compile-ta")
        for _ in range(steps):
            if uses_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
                u = self.step(u, dx, dt, a, em).clone()
            else:
                u = self.step(u, dx, dt, a, em)
        return u


def steps_for_full_period(n: int, a: float = 1.0) -> int:
    import math
    dx = 1.0 / n
    return int(math.ceil(1.0 / (CFL * dx / abs(a))))
