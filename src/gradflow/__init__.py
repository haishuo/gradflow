"""GradFlow: research code for differentiable finite-difference WENO."""

from .weno5 import (
    DEFAULT_EPSILON,
    ssp_rk3_step,
    weno5_rhs,
    weno5_rhs_gottlieb_periodic,
)
from .euler3d import (
    EULER_GAMMA,
    EULER_LF_ENLARGEMENT,
    EULER_WENO_EPSILON,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    euler_weno5_rhs,
    periodic_vortex,
    synchronize_duplicate_endpoints,
)
from .solver import (
    BackendDecision,
    BackendUnavailableError,
    RunDiagnostics,
    Solver,
    UnsupportedProblemError,
)

__all__ = [
    "DEFAULT_EPSILON",
    "ssp_rk3_step",
    "weno5_rhs",
    "weno5_rhs_gottlieb_periodic",
    "BackendDecision",
    "BackendUnavailableError",
    "EULER_GAMMA",
    "EULER_LF_ENLARGEMENT",
    "EULER_WENO_EPSILON",
    "RunDiagnostics",
    "Solver",
    "UnsupportedProblemError",
    "euler_cfl_timestep",
    "euler_ssp_rk3_step",
    "euler_weno5_rhs",
    "periodic_vortex",
    "synchronize_duplicate_endpoints",
]
