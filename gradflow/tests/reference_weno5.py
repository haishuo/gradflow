import numpy as np
import matplotlib.pyplot as plt

"""
WENO (Weighted Essentially Non-Oscillatory) Solver
Based on Jiang & Shu's fifth-order WENO method
Educational implementation following Field et al. 2020

This solver implements WENO for problems of the form:
    u_t + f(u)_x = 0
"""

class WENOSolver:
    """
    Fifth-order WENO solver for hyperbolic conservation laws.

    The WENO method adaptively chooses between different stencils
    based on smoothness indicators to avoid spurious oscillations
    near discontinuities while maintaining high-order accuracy
    in smooth regions.
    """

    def __init__(self, grid_size, domain_length=1.0, epsilon=1e-6):
        """
        Initialize the WENO solver.

        Parameters:
        -----------
        grid_size : int
            Number of spatial grid points
        domain_length : float
            Physical length of the domain
        epsilon : float
            Small parameter to prevent division by zero in weight calculation
            (denoted as ε in the paper)
        """
        self.nx = grid_size
        self.L = domain_length
        self.dx = domain_length / grid_size  # Grid spacing (Δρ in the paper)
        self.epsilon = epsilon  # The ε parameter from equations

        # Create spatial grid
        self.x = np.linspace(0, domain_length, grid_size)

    def flux_split(self, f, u):
        """
        Split flux into positive and negative parts: f(u) = f⁺(u) + f⁻(u)

        From the paper: "we split the flux into its positive and negative parts
        such that df⁺(u)/du ≥ 0, and df⁻(u)/du ≤ 0"

        Using Lax-Friedrichs flux splitting:
        f⁺(u) = 0.5 * (f(u) + α*u)
        f⁻(u) = 0.5 * (f(u) - α*u)

        where α is chosen as the maximum wave speed
        """
        # Calculate maximum characteristic speed (wave speed)
        alpha = np.max(np.abs(self.flux_derivative(u)))

        # Split the flux
        f_values = f(u)
        f_plus = 0.5 * (f_values + alpha * u)
        f_minus = 0.5 * (f_values - alpha * u)

        return f_plus, f_minus

    def flux_derivative(self, u):
        """
        Compute df/du for the current state.
        For linear advection with speed a: df/du = a
        For Burgers equation f(u) = u²/2: df/du = u

        This is needed for flux splitting to determine characteristic speed.
        """
        # Default: assume Burgers' equation f(u) = u²/2, so df/du = u
        return u

    def smoothness_indicators_fifth_order(self, f, j):
        """
        Calculate smoothness indicators (IS) for the three candidate stencils.

        From the paper, for the positive flux (f⁺):

        IS₀⁺ = (13/12)(f⁺ⱼ₋₂ - 2f⁺ⱼ₋₁ + f⁺ⱼ)² + (1/4)(f⁺ⱼ₋₂ - 4f⁺ⱼ₋₁ + 3f⁺ⱼ)²
        IS₁⁺ = (13/12)(f⁺ⱼ₋₁ - 2f⁺ⱼ + f⁺ⱼ₊₁)² + (1/4)(f⁺ⱼ₋₁ - f⁺ⱼ₊₁)²
        IS₂⁺ = (13/12)(f⁺ⱼ - 2f⁺ⱼ₊₁ + f⁺ⱼ₊₂)² + (1/4)(3f⁺ⱼ - 4f⁺ⱼ₊₁ + f⁺ⱼ₊₂)²

        These measure the smoothness of the solution on each stencil.
        Larger IS means less smooth (possibly near a discontinuity).
        """
        # Stencil 0: uses points {j-2, j-1, j}
        IS0 = (13.0/12.0) * (f[j-2] - 2*f[j-1] + f[j])**2 + \
              (1.0/4.0) * (f[j-2] - 4*f[j-1] + 3*f[j])**2

        # Stencil 1: uses points {j-1, j, j+1}
        IS1 = (13.0/12.0) * (f[j-1] - 2*f[j] + f[j+1])**2 + \
              (1.0/4.0) * (f[j-1] - f[j+1])**2

        # Stencil 2: uses points {j, j+1, j+2}
        IS2 = (13.0/12.0) * (f[j] - 2*f[j+1] + f[j+2])**2 + \
              (1.0/4.0) * (3*f[j] - 4*f[j+1] + f[j+2])**2

        return IS0, IS1, IS2

    def compute_weights_fifth_order(self, IS0, IS1, IS2, direction='plus'):
        """
        Compute the nonlinear weights ω for combining stencils.

        From the paper:
        α₀ = (1/10) / (ε + IS₀)²
        α₁ = (6/10) / (ε + IS₁)²
        α₂ = (3/10) / (ε + IS₂)²

        Then normalize:
        ω₀ = α₀/(α₀ + α₁ + α₂)
        ω₁ = α₁/(α₀ + α₁ + α₂)
        ω₂ = α₂/(α₀ + α₁ + α₂)

        The coefficients 1/10, 6/10, 3/10 are the "optimal weights" that
        would give fifth-order accuracy in smooth regions. WENO adaptively
        deviates from these near discontinuities.
        """
        # Optimal linear weights (for fifth-order accuracy in smooth regions)
        d0, d1, d2 = 1.0/10.0, 6.0/10.0, 3.0/10.0

        # Calculate alpha values (unnormalized weights)
        # The (ε + IS)² in denominator prevents division by zero and
        # ensures smooth stencils get higher weight
        alpha0 = d0 / (self.epsilon + IS0)**2
        alpha1 = d1 / (self.epsilon + IS1)**2
        alpha2 = d2 / (self.epsilon + IS2)**2

        # Normalize to get weights that sum to 1
        alpha_sum = alpha0 + alpha1 + alpha2
        omega0 = alpha0 / alpha_sum
        omega1 = alpha1 / alpha_sum
        omega2 = alpha2 / alpha_sum

        return omega0, omega1, omega2

    def reconstruct_flux_fifth_order_plus(self, f_plus, j):
        """
        Reconstruct the flux at the interface j+1/2 for positive flux.

        From the paper, the flux at j+1/2 is:
        f̂⁺ⱼ₊₁/₂ = ω₀⁺(2/6·f⁺ⱼ₋₂ - 7/6·f⁺ⱼ₋₁ + 11/6·f⁺ⱼ) +
                  ω₁⁺(-1/6·f⁺ⱼ₋₁ + 5/6·f⁺ⱼ + 2/6·f⁺ⱼ₊₁) +
                  ω₂⁺(2/6·f⁺ⱼ + 5/6·f⁺ⱼ₊₁ - 1/6·f⁺ⱼ₊₂)

        Each term is a polynomial reconstruction on a different 3-point stencil.
        """
        # Get smoothness indicators for this point
        IS0, IS1, IS2 = self.smoothness_indicators_fifth_order(f_plus, j)

        # Get nonlinear weights
        omega0, omega1, omega2 = self.compute_weights_fifth_order(IS0, IS1, IS2, 'plus')

        # Reconstruct using the three stencils with their weights
        # Stencil 0: biased to the left (upwind for positive flux)
        flux0 = (2.0/6.0)*f_plus[j-2] - (7.0/6.0)*f_plus[j-1] + (11.0/6.0)*f_plus[j]

        # Stencil 1: centered
        flux1 = -(1.0/6.0)*f_plus[j-1] + (5.0/6.0)*f_plus[j] + (2.0/6.0)*f_plus[j+1]

        # Stencil 2: biased to the right
        flux2 = (2.0/6.0)*f_plus[j] + (5.0/6.0)*f_plus[j+1] - (1.0/6.0)*f_plus[j+2]

        # Weighted combination
        f_hat_plus = omega0*flux0 + omega1*flux1 + omega2*flux2

        return f_hat_plus

    def reconstruct_flux_fifth_order_minus(self, f_minus, j):
        """
        Reconstruct the flux at the interface j+1/2 for negative flux.

        From the paper, for negative flux (which travels right to left),
        we use the "mirror" stencils:

        f̂⁻ⱼ₊₁/₂ = ω₂⁻(-1/6·f⁻ⱼ₋₁ + 5/6·f⁻ⱼ + 2/6·f⁻ⱼ₊₁) +
                  ω₁⁻(2/6·f⁻ⱼ + 5/6·f⁻ⱼ₊₁ - 1/6·f⁻ⱼ₊₂) +
                  ω₀⁻(11/6·f⁻ⱼ₊₁ - 7/6·f⁻ⱼ₊₂ + 2/6·f⁻ⱼ₊₃)

        Note the indices are shifted because negative flux propagates
        in the opposite direction.
        """
        # Get smoothness indicators (but for negative flux, shift indices)
        # For negative flux at j+1/2, we look at points {j+1, j+2, j+3}
        IS0_minus = (13.0/12.0) * (f_minus[j+1] - 2*f_minus[j+2] + f_minus[j+3])**2 + \
                    (1.0/4.0) * (3*f_minus[j+1] - 4*f_minus[j+2] + f_minus[j+3])**2

        IS1_minus = (13.0/12.0) * (f_minus[j] - 2*f_minus[j+1] + f_minus[j+2])**2 + \
                    (1.0/4.0) * (f_minus[j] - f_minus[j+2])**2

        IS2_minus = (13.0/12.0) * (f_minus[j-1] - 2*f_minus[j] + f_minus[j+1])**2 + \
                    (1.0/4.0) * (f_minus[j-1] - 4*f_minus[j] + 3*f_minus[j+1])**2

        # Get weights (note: order is reversed for minus flux)
        omega0, omega1, omega2 = self.compute_weights_fifth_order(IS0_minus, IS1_minus, IS2_minus, 'minus')

        # Reconstruct (stencils are mirrored compared to plus flux)
        flux0 = (11.0/6.0)*f_minus[j+1] - (7.0/6.0)*f_minus[j+2] + (2.0/6.0)*f_minus[j+3]
        flux1 = (2.0/6.0)*f_minus[j] + (5.0/6.0)*f_minus[j+1] - (1.0/6.0)*f_minus[j+2]
        flux2 = -(1.0/6.0)*f_minus[j-1] + (5.0/6.0)*f_minus[j] + (2.0/6.0)*f_minus[j+1]

        f_hat_minus = omega0*flux0 + omega1*flux1 + omega2*flux2

        return f_hat_minus

    def spatial_derivative_weno5(self, u, flux_function):
        """
        Compute the spatial derivative using fifth-order WENO.

        This implements: ∂f(u)/∂x ≈ (f̂ⱼ₊₁/₂ - f̂ⱼ₋₁/₂)/Δx

        where f̂ are the reconstructed fluxes at cell interfaces.
        """
        # Split flux into positive and negative parts
        f_plus, f_minus = self.flux_split(flux_function, u)

        # Initialize derivative array
        du_dx = np.zeros_like(u)

        # Loop over interior points (need 3 ghost cells on each side for 5th order)
        for j in range(3, self.nx - 3):
            # Reconstruct positive flux at j+1/2 and j-1/2
            f_hat_plus_j_half = self.reconstruct_flux_fifth_order_plus(f_plus, j)
            f_hat_plus_jm_half = self.reconstruct_flux_fifth_order_plus(f_plus, j-1)

            # Reconstruct negative flux at j+1/2 and j-1/2
            f_hat_minus_j_half = self.reconstruct_flux_fifth_order_minus(f_minus, j)
            f_hat_minus_jm_half = self.reconstruct_flux_fifth_order_minus(f_minus, j-1)

            # Total flux at interfaces
            f_hat_j_plus_half = f_hat_plus_j_half + f_hat_minus_j_half
            f_hat_j_minus_half = f_hat_plus_jm_half + f_hat_minus_jm_half

            # Compute derivative as per paper: (f̂ⱼ₊₁/₂ - f̂ⱼ₋₁/₂)/Δx
            du_dx[j] = (f_hat_j_plus_half - f_hat_j_minus_half) / self.dx

        return du_dx

    def rk3_ssp_step(self, u, dt, flux_function):
        """
        Third-order Strong Stability Preserving (SSP) Runge-Kutta time stepping.

        From the paper (SSP-RK(3,3)):
        u⁽¹⁾ = uⁿ + Δt·F(uⁿ)
        u⁽²⁾ = (3/4)uⁿ + (1/4)[u⁽¹⁾ + Δt·F(u⁽¹⁾)]
        uⁿ⁺¹ = (1/3)uⁿ + (2/3)[u⁽²⁾ + Δt·F(u⁽²⁾)]

        where F(u) = -∂f(u)/∂x is the right-hand side of u_t + f(u)_x = 0
        """
        # Stage 1
        F_u = -self.spatial_derivative_weno5(u, flux_function)
        u1 = u + dt * F_u

        # Stage 2
        F_u1 = -self.spatial_derivative_weno5(u1, flux_function)
        u2 = 0.75*u + 0.25*(u1 + dt*F_u1)

        # Stage 3
        F_u2 = -self.spatial_derivative_weno5(u2, flux_function)
        u_new = (1.0/3.0)*u + (2.0/3.0)*(u2 + dt*F_u2)

        return u_new

    def solve(self, u0, t_final, flux_function, cfl=0.4):
        """
        Solve the PDE u_t + f(u)_x = 0 from t=0 to t=t_final.

        Parameters:
        -----------
        u0 : array
            Initial condition (without ghost cells)
        t_final : float
            Final time
        flux_function : function
            The flux function f(u)
        cfl : float
            CFL number for time step (must be < 1 for stability)

        Returns:
        --------
        u : array
            Solution at final time (with ghost cells)
        """
        # Initialize array WITH ghost cells from the start
        u = np.zeros(len(u0) + 6)
        u[3:-3] = u0.copy()  # Put initial condition in interior

        t = 0.0

        # Apply periodic boundary conditions initially
        u = self.apply_periodic_bc(u)

        while t < t_final:
            # Calculate time step based on CFL condition
            # Only use interior points for max speed calculation
            u_interior = u[3:-3]
            max_speed = np.max(np.abs(self.flux_derivative(u_interior)))
            dt = cfl * self.dx / max_speed if max_speed > 0 else cfl * self.dx

            # Don't overshoot final time
            if t + dt > t_final:
                dt = t_final - t

            # Take one RK3 step
            u = self.rk3_ssp_step(u, dt, flux_function)

            # Apply boundary conditions
            u = self.apply_periodic_bc(u)

            t += dt

        return u

    def apply_periodic_bc(self, u):
        """
        Apply periodic boundary conditions.
        For WENO5 we need 3 ghost cells on each side.

        This function assumes u already has ghost cells (length nx + 6).
        It fills the ghost cells with periodic values.
        """
        # Left boundary (ghost cells) - copy from right interior
        u[0:3] = u[-6:-3]

        # Right boundary (ghost cells) - copy from left interior
        u[-3:] = u[3:6]

        return u


# ============================================================================
# DEMONSTRATION: Solving Burgers' Equation
# ============================================================================

def burgers_flux(u):
    """
    Flux function for Burgers' equation: f(u) = u²/2
    This is a classic test case for WENO methods.
    """
    return 0.5 * u**2

def test_weno_burgers():
    """
    Test the WENO solver on Burgers' equation with a sine wave initial condition.
    Burgers' equation: u_t + (u²/2)_x = 0
    """
    # Setup
    nx = 200
    L = 2.0 * np.pi
    solver = WENOSolver(nx, L)

    # Initial condition: smooth sine wave
    u0 = np.sin(solver.x)

    # Solve to time t = 0.5 (before shock formation)
    t_final = 0.5
    u_final = solver.solve(u0, t_final, burgers_flux, cfl=0.4)

    # Remove ghost cells for plotting
    u_final_interior = u_final[3:-3]

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(solver.x, u0, 'b-', label='Initial condition', linewidth=2)
    plt.plot(solver.x, u_final_interior, 'r-', label=f'WENO solution at t={t_final}', linewidth=2)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u', fontsize=12)
    plt.title("WENO5 Solution of Burgers' Equation", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Simulation complete:")
    print(f"  Grid points: {nx}")
    print(f"  Domain: [0, {L:.2f}]")
    print(f"  Final time: {t_final}")
    print(f"  Grid spacing: {solver.dx:.6f}")

if __name__ == "__main__":
    test_weno_burgers()
