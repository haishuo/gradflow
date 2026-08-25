"""Numerical parity checks for the convolutional feature-bank experiment."""

from __future__ import annotations

import unittest

import torch

from shu_euler_torch import cfl_timestep, periodic_vortex, ssp_rk3_step
from shu_euler_torch_conv import ConvFeatureWenoStep


class ConvFeatureParityTests(unittest.TestCase):
    def test_float64_cpu_parity(self) -> None:
        for dimension, size in ((2, 8), (3, 6)):
            with self.subTest(dimension=dimension):
                state, spacing = periodic_vortex(
                    (size,) * dimension, dtype=torch.float64
                )
                dt = torch.tensor(0.001, dtype=torch.float64)
                direct = ssp_rk3_step(state, spacing, dt)
                convolutional = ConvFeatureWenoStep(
                    dimension, spacing, dtype=torch.float64
                )(state, dt)
                torch.testing.assert_close(
                    convolutional, direct, rtol=2.0e-14, atol=1.0e-17
                )

    def test_float64_cpu_multistep_parity(self) -> None:
        state, spacing = periodic_vortex((6, 6, 6), dtype=torch.float64)
        direct = state
        convolutional = state
        convolutional_step = ConvFeatureWenoStep(
            3, spacing, dtype=torch.float64
        )
        for _ in range(3):
            direct_dt = cfl_timestep(direct, spacing, 0.1)
            convolutional_dt = cfl_timestep(convolutional, spacing, 0.1)
            direct = ssp_rk3_step(direct, spacing, direct_dt)
            convolutional = convolutional_step(
                convolutional, convolutional_dt
            )
        torch.testing.assert_close(
            convolutional, direct, rtol=6.0e-14, atol=3.0e-17
        )


if __name__ == "__main__":
    unittest.main()
