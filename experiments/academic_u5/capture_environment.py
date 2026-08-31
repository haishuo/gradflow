#!/usr/bin/env python3
"""Capture the selected stable-PyTorch U5 execution environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*arguments: str) -> str:
    return subprocess.run(
        arguments, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    executable = Path(sys.executable).resolve()
    document = {
        "schema": "gradflow.academic_u5.environment.v1",
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": str(executable),
        "python_executable_sha256": sha256(executable),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "torch_config": torch.__config__.show(),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
        "nvidia_smi": command("nvidia-smi", "-q"),
        "pip_freeze": command(sys.executable, "-m", "pip", "freeze").splitlines(),
        "environment_keys": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith(("CUDA", "TORCH", "OMP", "MKL"))
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()

