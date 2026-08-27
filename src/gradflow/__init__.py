"""GradFlow: research code for differentiable finite-difference WENO."""

from .dveb_abi import (
    DvebAbiError,
    DvebArtifact,
    DvebDeviceContext,
    DvebDeviceRunResult,
    DvebPortableAbi,
    DvebRunResult,
)
from .euler3d import (
    EULER_GAMMA,
    EULER_LF_ENLARGEMENT,
    EULER_WENO_EPSILON,
    QUALIFIED_EULER_WENO_ORDERS,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    euler_weno5_rhs,
    euler_weno_rhs,
    periodic_vortex,
    synchronize_duplicate_endpoints,
)
from .euler1d import (
    EULER1D_BOUNDARIES,
    euler1d_cfl_timestep,
    euler1d_rhs,
    euler1d_rhs_with_boundary_fluxes,
    euler1d_ssp_rk3_step,
)
from .solver import (
    BackendDecision,
    BackendUnavailableError,
    RunDiagnostics,
    Solver,
    UnsupportedProblemError,
)
from .weno5 import (
    DEFAULT_EPSILON,
    ssp_rk3_step,
    weno5_rhs,
    weno5_rhs_gottlieb_periodic,
)
from .weno_js import QUALIFIED_ORDERS, SMOOTHNESS_SCALE, WENOJS
from .weno_js_coefficients import (
    WENOJSCoefficients,
    generate_weno_js_coefficients,
)

__all__ = [
    "DEFAULT_EPSILON",
    "QUALIFIED_ORDERS",
    "SMOOTHNESS_SCALE",
    "WENOJS",
    "WENOJSCoefficients",
    "generate_weno_js_coefficients",
    "DvebAbiError",
    "DvebArtifact",
    "DvebDeviceContext",
    "DvebDeviceRunResult",
    "DvebPortableAbi",
    "DvebRunResult",
    "ssp_rk3_step",
    "weno5_rhs",
    "weno5_rhs_gottlieb_periodic",
    "BackendDecision",
    "BackendUnavailableError",
    "EULER_GAMMA",
    "EULER_LF_ENLARGEMENT",
    "EULER_WENO_EPSILON",
    "QUALIFIED_EULER_WENO_ORDERS",
    "RunDiagnostics",
    "Solver",
    "UnsupportedProblemError",
    "euler_cfl_timestep",
    "EULER1D_BOUNDARIES",
    "euler1d_cfl_timestep",
    "euler1d_rhs",
    "euler1d_rhs_with_boundary_fluxes",
    "euler1d_ssp_rk3_step",
    "euler_ssp_rk3_step",
    "euler_weno_rhs",
    "euler_weno5_rhs",
    "periodic_vortex",
    "synchronize_duplicate_endpoints",
]
