from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "experiments/academic_a4/unity/run_second_machine.py"
    specification = importlib.util.spec_from_file_location("unity_runner", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def worker_record(device: str) -> dict:
    correctness = {
        "eager": {"admitted": True},
        "compiled": {
            "admitted": True,
            "graph": {"unique_graphs": 1, "graph_break_count": 0},
        },
    }
    record = {"device": device, "correctness": correctness}
    if device == "cpu":
        record["cpu"] = {
            threads: {
                "correctness": {
                    "eager": {"admitted": True},
                    "compiled": {"admitted": True},
                },
                "resident_timing": {
                    "lanes": {
                        "eager": {"median": 3.0},
                        "compiled": {"median": 2.0},
                    }
                },
            }
            for threads in ("1", "6")
        }
    else:
        record["cuda"] = {
            "resident_timing": {
                "lanes": {
                    "eager": {"median": 1.0},
                    "compiled": {"median": 0.5},
                }
            }
        }
    return record


def test_unity_analyzer_accepts_complete_qualitative_replication() -> None:
    runner = load_runner()
    workers = []
    for order in runner.ORDERS:
        for dtype in runner.DTYPES:
            for device in runner.DEVICES:
                for _ in range(runner.WORKERS):
                    workers.append(
                        {
                            "order": order,
                            "dtype": dtype,
                            "device": device,
                            "record": worker_record(device),
                        }
                    )
    analysis = runner.analyze_a2(workers)
    assert analysis["all_expected_workers_parsed"]
    assert analysis["all_compiled_graphs_one_with_zero_breaks"]
    assert analysis["materially_useful_binary32_cuda_observed"]
    assert analysis["admission_failures"] == []


def test_unity_protocol_keeps_execution_off_login_and_under_work() -> None:
    protocol = (ROOT / "docs/ACADEMIC_A4_UNITY_PROTOCOL.md").read_text()
    setup = (
        ROOT / "experiments/academic_a4/unity/setup_environment.sbatch"
    ).read_text()
    replication = (ROOT / "experiments/academic_a4/unity/replicate.sbatch").read_text()
    for text in (protocol, setup, replication):
        assert "/work/pi_zchen2_umassd_edu/hshu" in text
    assert "login directory is used only" in protocol
    assert "#SBATCH --partition=cpu" in setup
    assert "#SBATCH --partition=gpu" in replication
    assert "#SBATCH --gpus=1" in replication

