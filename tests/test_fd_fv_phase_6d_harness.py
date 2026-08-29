from __future__ import annotations

from experiments.fd_fv_euler.run_phase6d import (
    INTERACTION_SIZES,
    PRIMARY_SIZES,
    aggregate_shocks,
    expected_replicates,
)


def test_phase6d_frozen_cpu_matrix() -> None:
    assert PRIMARY_SIZES == (
        2048,
        4096,
        6144,
        8192,
        12288,
        16384,
        24576,
        32768,
    )
    assert INTERACTION_SIZES == (4096, 8192, 32768)
    assert expected_replicates(4096, 1) == 3
    assert expected_replicates(4096, 6) == 3
    assert expected_replicates(4096, 2) == 1
    assert expected_replicates(6144, 6) == 1


def test_phase6d_shock_confirmation_requires_every_pair() -> None:
    records = []
    for problem, cpu_mode in (("sod", "eager"), ("shu_osher", "compiled")):
        for method in ("fd", "fv"):
            for replicate, ratio in enumerate((0.8, 0.85, 0.9)):
                records.extend(
                    (
                        {
                            "problem": problem,
                            "method": method,
                            "device": "cpu",
                            "mode": cpu_mode,
                            "replicate": replicate,
                            "process_launch_to_exit_seconds": 10.0,
                            "eligible": True,
                        },
                        {
                            "problem": problem,
                            "method": method,
                            "device": "cuda",
                            "mode": "compiled",
                            "replicate": replicate,
                            "process_launch_to_exit_seconds": 10.0 * ratio,
                            "eligible": True,
                        },
                    )
                )
    result = aggregate_shocks(records)
    assert all(
        result[problem][method]["confirmed"]
        for problem in ("sod", "shu_osher")
        for method in ("fd", "fv")
    )
    records[-1]["process_launch_to_exit_seconds"] = 10.0
    result = aggregate_shocks(records)
    assert not result["shu_osher"]["fv"]["confirmed"]
