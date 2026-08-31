from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "experiments/academic_a4/unity/run_second_machine.py"
    specification = importlib.util.spec_from_file_location("second_machine_runner", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_capture_text_records_missing_optional_command(tmp_path: Path) -> None:
    runner = load_runner()
    record = runner.capture_text(["gradflow-command-that-does-not-exist"], tmp_path)
    assert record["returncode"] == 127
    assert record["stdout"] == ""
    assert record["stderr"]


def test_moody_protocol_preserves_frozen_scientific_surface() -> None:
    protocol = (ROOT / "docs/ACADEMIC_A4_MOODY_PROTOCOL.md").read_text()
    setup = (
        ROOT / "experiments/academic_a4/moody/setup_environment.sh"
    ).read_text()
    run = (ROOT / "experiments/academic_a4/moody/run_moody.sh").read_text()
    stage = (ROOT / "experiments/academic_a4/moody/stage_moody.sh").read_text()
    for text in (protocol, setup, run, stage):
        assert "/mnt/projects" in text
    assert "orders" not in run.lower()
    assert "torch==2.13.0" in setup
    assert "--execution-context standalone" in run
    assert "ACADEMIC_A4_MOODY_PROTOCOL.md" in run
    assert "academic-v0.1.0-rc2" in stage
    assert "three fresh A2 workers" in protocol
