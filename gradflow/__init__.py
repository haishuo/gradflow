"""
GradFlow - Revolutionary Differentiable CFD

World's first differentiable WENO scheme for computational fluid dynamics.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Auto-setup directories on import
from pathlib import Path

_project_root = Path(__file__).parent.parent

# Smart path detection
_data_candidates = [
    _project_root / "data",
    Path("/mnt/data/gradflow"),
    _project_root / "local_data",
]

_artifacts_candidates = [
    _project_root / "artifacts",
    Path("/mnt/artifacts/gradflow"),
    _project_root / "local_artifacts",
]

# Find existing directories
DATA_ROOT = next((p for p in _data_candidates if p.exists()), _data_candidates[0])
ARTIFACTS_ROOT = next(
    (p for p in _artifacts_candidates if p.exists()), _artifacts_candidates[0]
)

# Import main modules
from . import applications, core, validation

__all__ = ["core", "validation", "applications", "DATA_ROOT", "ARTIFACTS_ROOT"]
