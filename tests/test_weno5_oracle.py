from pathlib import Path

import h5py
import torch

from gradflow import ssp_rk3_step, weno5_rhs_gottlieb_periodic


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references" / "gottlieb_matlab" / "reference_data.h5"


def _linear_flux(u: torch.Tensor) -> torch.Tensor:
    return u


def _gottlieb_solution() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    x = torch.where(torch.abs(x) < 1.0e-14, 0.0, x)
    u = torch.sign(x)
    # Match MATLAB/NumPy exactly: max(diff(linspace)), not the decimal 0.02.
    dx = torch.max(torch.diff(x)).item()
    dt = 0.5 * dx

    def rhs(state: torch.Tensor) -> torch.Tensor:
        return weno5_rhs_gottlieb_periodic(
            state, dx, _linear_flux, alpha=1.0
        )

    for _ in range(75):
        u = ssp_rk3_step(u, dt, rhs)
    return x, u


def test_gottlieb_matlab_oracle() -> None:
    x, u = _gottlieb_solution()
    with h5py.File(REFERENCE, "r") as reference:
        x_reference = torch.from_numpy(reference["x"][:].reshape(-1))
        u_reference = torch.from_numpy(reference["u"][:].reshape(-1))

    torch.testing.assert_close(x, x_reference, rtol=0.0, atol=2.0e-15)
    torch.testing.assert_close(u, u_reference, rtol=0.0, atol=1.0e-12)


def test_reference_hashes() -> None:
    import hashlib

    expected = {
        ROOT / "references/gottlieb_matlab/weno5.m":
            "fd555073570885197b8f46d9029ec5ee751c0c104a62277a17137f83c8ad09f6",
        ROOT / "references/jiang_shu_fortran/weno.f":
            "9f1231516ef92b496333475ef29bfbba23afe77423163e7797bc8775a50186c5",
        ROOT / "references/jiang_shu_fortran/comm.inc":
            "efc977da6582767cfa20ef76b0c3a0ace83e64083ca78f161668124e4cdbe3a7",
        ROOT / "baselines/dveb_screened_pytorch/weno5_torch.py":
            "2cff04949eb4c56ada030975ce5b0ce641abf702fcfc886cda0897578aff23ed",
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
