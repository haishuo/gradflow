"""GradFlow: research code for differentiable finite-difference WENO."""

from .burgers import (
    BURGERS_FD_WENO5_FORMULATION_ID,
    BURGERS_FV_WENO5_FORMULATION_ID,
    burgers_fd_weno5_rhs,
    burgers_flux,
    burgers_fv_weno5_rhs,
)

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
from .fv_weno5 import (
    FV_WENO5_FORMULATION_ID,
    fv_global_lax_friedrichs_flux,
    fv_weno5_face_states,
    fv_weno5_rhs,
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
from .weno_js import (
    NATIVE_PRECISION,
    PRECISION_BLOCKS,
    QUALIFIED_ORDERS,
    SMOOTHNESS_SCALE,
    WENOJS,
    WENOJSPrecisionPolicy,
)
from .weno_js_coefficients import (
    WENOJSCoefficients,
    generate_weno_js_coefficients,
)

__all__ = [
    "BURGERS_FD_WENO5_FORMULATION_ID",
    "BURGERS_FV_WENO5_FORMULATION_ID",
    "DEFAULT_EPSILON",
    "QUALIFIED_ORDERS",
    "PRECISION_BLOCKS",
    "SMOOTHNESS_SCALE",
    "WENOJS",
    "WENOJSPrecisionPolicy",
    "NATIVE_PRECISION",
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
    "burgers_fd_weno5_rhs",
    "burgers_flux",
    "burgers_fv_weno5_rhs",
    "EULER_GAMMA",
    "EULER_LF_ENLARGEMENT",
    "EULER_WENO_EPSILON",
    "QUALIFIED_EULER_WENO_ORDERS",
    "RunDiagnostics",
    "Solver",
    "UnsupportedProblemError",
    "euler_cfl_timestep",
    "EULER1D_BOUNDARIES",
    "FV_WENO5_FORMULATION_ID",
    "euler1d_cfl_timestep",
    "euler1d_rhs",
    "euler1d_rhs_with_boundary_fluxes",
    "euler1d_ssp_rk3_step",
    "fv_global_lax_friedrichs_flux",
    "fv_weno5_face_states",
    "fv_weno5_rhs",
    "euler_ssp_rk3_step",
    "euler_weno_rhs",
    "euler_weno5_rhs",
    "periodic_vortex",
    "synchronize_duplicate_endpoints",
]
