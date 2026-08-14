"""
Spectral Damping for the Two-Dimensional Chafee--Infante Equation

This script illustrates the preferential damping of high-frequency spatial
modes under the forward evolution of the two-dimensional Chafee--Infante
equation

    u_t - gamma * Delta u + kappa * (u^3 - u) = 0

on the square domain

    Omega = [-L, L] x [-L, L],

subject to homogeneous Dirichlet boundary conditions

    u = 0 on boundary(Omega).

The initial condition is constructed from selected Dirichlet sine modes,

    phi_{m,n}(x,y)
        = sin(m*pi*xi) sin(n*pi*eta),

where

    xi  = (x + L)/(2L),
    eta = (y + L)/(2L).

One low-frequency mode is combined with several high-frequency modes.
The resulting evolution demonstrates the rapid suppression of fine-scale
spectral content by the diffusive dynamics.

Time integration uses an Eyre-type convex--concave splitting. The diffusion
and cubic terms are treated implicitly, while the destabilizing linear term
is treated explicitly. The nonlinear system at each time step is solved by
Newton's method.

Requirements
------------
numpy
scipy
matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


# =============================================================================
# Parameters
# =============================================================================

N_STEPS = 2000
DELTA_T = 0.001

L = 1.0
M = 128

GAMMA = 0.005
KAPPA = 4.7

SNAPSHOT_STRIDE = 100

LOW_MODE_AMPLITUDE = 0.02
HIGH_MODE_AMPLITUDE = 0.02

LOW_MODE = (2, 2)

HIGH_MODES = (
    (18, 18),
    (24, 20),
    (20, 24),
)

NEWTON_MAX_ITERS = 12
NEWTON_TOL = 1.0e-10

LYAPUNOV_ATOL = 1.0e-10
LYAPUNOV_RTOL = 1.0e-10

IMAGE_DIR = Path("images")
DATA_DIR = Path("data")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Grid
# =============================================================================

X = np.linspace(-L, L, M)
Y = np.linspace(-L, L, M)

HX = X[1] - X[0]
HY = Y[1] - Y[0]

CELL_AREA = HX * HY

MI = M - 2
N_INTERIOR = MI * MI


# =============================================================================
# Discrete operators
# =============================================================================

def build_dirichlet_laplacian_1d(n_points, mesh_size):
    """
    Construct the one-dimensional second-difference Laplacian.

    Homogeneous Dirichlet boundary values are imposed externally,
    so the matrix acts only on the interior unknowns.
    """

    main_diagonal = -2.0 * np.ones(n_points)
    off_diagonal = np.ones(n_points - 1)

    laplacian = diags(
        [off_diagonal, main_diagonal, off_diagonal],
        offsets=[-1, 0, 1],
        shape=(n_points, n_points),
        format="csr",
    )

    return laplacian / mesh_size**2


def build_dirichlet_laplacian_2d(total_grid_points, mesh_size):
    """
    Construct the two-dimensional Dirichlet Laplacian.

    The discrete operator is the Kronecker sum

        Delta_h = I kron T + T kron I,

    where T is the one-dimensional second-difference matrix.
    """

    n_interior = total_grid_points - 2

    laplacian_1d = build_dirichlet_laplacian_1d(
        n_interior,
        mesh_size,
    )

    identity = eye(
        n_interior,
        format="csr",
    )

    return (
        kron(identity, laplacian_1d, format="csr")
        + kron(laplacian_1d, identity, format="csr")
    )


LAPLACIAN = build_dirichlet_laplacian_2d(
    M,
    HX,
)

IDENTITY = eye(
    N_INTERIOR,
    format="csr",
)


# =============================================================================
# Spectrally engineered initial condition
# =============================================================================

def dirichlet_sine_mode(m, n):
    r"""
    Construct the Dirichlet sine mode

        phi_{m,n}(x,y)
            = sin(m*pi*xi) sin(n*pi*eta),

    where

        xi  = (x + L)/(2L),
        eta = (y + L)/(2L).

    The mode vanishes exactly on the boundary of the square.
    """

    x_grid, y_grid = np.meshgrid(
        X,
        Y,
        indexing="ij",
    )

    xi = (x_grid + L) / (2.0 * L)
    eta = (y_grid + L) / (2.0 * L)

    return (
        np.sin(m * np.pi * xi)
        * np.sin(n * np.pi * eta)
    )


def engineered_sine_initial_condition():
    """
    Construct the spectrally engineered initial condition.

    The field contains one low-frequency mode and several high-frequency
    modes. This permits direct visualization of the preferential damping
    of high-frequency information under forward parabolic evolution.
    """

    m_low, n_low = LOW_MODE

    u0 = (
        LOW_MODE_AMPLITUDE
        * dirichlet_sine_mode(
            m_low,
            n_low,
        )
    )

    for m_high, n_high in HIGH_MODES:

        u0 += (
            HIGH_MODE_AMPLITUDE
            * dirichlet_sine_mode(
                m_high,
                n_high,
            )
        )

    # Enforce the Dirichlet boundary condition exactly.
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    u0[:, 0] = 0.0
    u0[:, -1] = 0.0

    return np.asarray(
        u0,
        dtype=np.float64,
    )


# =============================================================================
# Energies and diagnostics
# =============================================================================

def quadratic_energy(u):
    r"""
    Compute the discrete quadratic energy

        E_2(u) = (1/2) integral_Omega u^2 dx.
    """

    return (
        0.5
        * CELL_AREA
        * np.sum(u**2)
    )


def lyapunov_energy(u):
    r"""
    Compute the discrete Chafee--Infante Lyapunov energy

        E(u)
        =
        integral_Omega [
            (gamma/2) |grad u|^2
            + kappa (u^4/4 - u^2/2)
        ] dx.

    Forward differences are used to approximate the gradient.
    """

    ux = np.diff(
        u,
        axis=0,
    ) / HX

    uy = np.diff(
        u,
        axis=1,
    ) / HY

    gradient_term = (
        np.sum(ux**2)
        + np.sum(uy**2)
    )

    potential = (
        0.25 * u**4
        - 0.5 * u**2
    )

    return (
        0.5 * GAMMA * gradient_term
        + KAPPA * np.sum(potential)
    ) * CELL_AREA


def solution_diagnostics(u):
    """
    Return the minimum and maximum values of the numerical solution.
    """

    return (
        np.min(u),
        np.max(u),
    )


# =============================================================================
# Eyre-type semi-implicit time step
# =============================================================================

def eyre_step(u_previous):
    r"""
    Advance the numerical solution by one Eyre-type time step.

    The scheme is

        u^{n+1}
        - delta_t gamma Delta_h u^{n+1}
        + delta_t kappa (u^{n+1})^3

        =
        (1 + delta_t kappa) u^n.

    Newton's method is used to solve the resulting nonlinear system.
    """

    previous_interior = u_previous[
        1:-1,
        1:-1,
    ]

    rhs = (
        1.0
        + DELTA_T * KAPPA
    ) * previous_interior

    u_interior = previous_interior.copy()

    converged = False
    residual_norm = np.inf

    for iteration in range(
        1,
        NEWTON_MAX_ITERS + 1,
    ):

        u_flat = u_interior.ravel()

        residual = (
            u_flat
            - DELTA_T
            * GAMMA
            * (LAPLACIAN @ u_flat)
            + DELTA_T
            * KAPPA
            * u_flat**3
            - rhs.ravel()
        )

        residual_norm = np.linalg.norm(
            residual,
            ord=2,
        )

        if residual_norm < NEWTON_TOL:
            converged = True
            break

        jacobian_diagonal = (
            3.0
            * DELTA_T
            * KAPPA
            * u_flat**2
        )

        jacobian = (
            IDENTITY
            - DELTA_T
            * GAMMA
            * LAPLACIAN
            + diags(
                jacobian_diagonal,
                offsets=0,
                format="csr",
            )
        )

        correction = spsolve(
            jacobian,
            -residual,
        )

        u_interior = (
            u_flat + correction
        ).reshape(
            MI,
            MI,
        )

    if not converged:

        raise RuntimeError(
            "Newton iteration failed to converge after "
            f"{NEWTON_MAX_ITERS} iterations; "
            f"final residual norm = {residual_norm:.3e}."
        )

    u = np.zeros(
        (M, M),
        dtype=np.float64,
    )

    u[1:-1, 1:-1] = u_interior

    return (
        u,
        iteration,
        residual_norm,
    )


# =============================================================================
# Output
# =============================================================================

def save_snapshot(u, step):
    """
    Save a solution snapshot.
    """

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        u,
        origin="lower",
        extent=[-L, L, -L, L],
        cmap="magma",
        interpolation="none",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="$u(x,y,t)$",
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.set_title(
        f"Chafee--Infante Solution, Step {step}"
    )

    fig.tight_layout()

    fig.savefig(
        IMAGE_DIR
        / f"spectral_phi{step:05d}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_diagnostic_plots(
    times,
    quadratic_values,
    lyapunov_values,
    minimum_values,
    maximum_values,
):
    """
    Save energy and solution-extrema diagnostics.
    """

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.plot(
        times,
        quadratic_values,
        linewidth=1.5,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$E_2(u)$")
    ax.set_title("Quadratic Energy")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    fig.savefig(
        DATA_DIR / "spectral_quadratic_energy.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.plot(
        times,
        lyapunov_values,
        linewidth=1.5,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$E(u)$")
    ax.set_title("Discrete Lyapunov Energy")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    fig.savefig(
        DATA_DIR / "spectral_lyapunov_energy.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.plot(
        times,
        maximum_values,
        linestyle="--",
        linewidth=1.5,
        label="Maximum",
    )

    ax.plot(
        times,
        minimum_values,
        linestyle=":",
        linewidth=1.5,
        label="Minimum",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Solution value")
    ax.set_title("Minimum and Maximum Solution Values")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    fig.savefig(
        DATA_DIR / "spectral_solution_extrema.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# Main computation
# =============================================================================

def main():
    """
    Run the spectral-damping Chafee--Infante experiment.
    """

    print(
        "Chafee--Infante spectral-damping experiment"
    )

    print(
        "-------------------------------------------"
    )

    print(f"Grid:          {M} x {M}")
    print(f"Domain:        [-{L}, {L}] x [-{L}, {L}]")
    print(f"Mesh size:     {HX:.6e}")
    print(f"Time steps:    {N_STEPS}")
    print(f"Time step:     {DELTA_T:.6e}")
    print(f"Final time:    {N_STEPS * DELTA_T:.6f}")
    print(f"gamma:         {GAMMA:.6e}")
    print(f"kappa:         {KAPPA:.6f}")

    print(f"Low mode:      {LOW_MODE}")
    print(f"High modes:    {HIGH_MODES}")

    print()

    # -------------------------------------------------------------------------
    # Initial condition
    # -------------------------------------------------------------------------

    u = engineered_sine_initial_condition()

    initial_data_path = (
        DATA_DIR
        / "spectral_initial_data.csv"
    )

    np.savetxt(
        initial_data_path,
        u,
        delimiter=",",
    )

    boundary_max = max(
        np.max(np.abs(u[0, :])),
        np.max(np.abs(u[-1, :])),
        np.max(np.abs(u[:, 0])),
        np.max(np.abs(u[:, -1])),
    )

    if boundary_max > 1.0e-14:

        raise ValueError(
            "Initial condition does not satisfy the "
            "homogeneous Dirichlet boundary condition."
        )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    times = np.empty(
        N_STEPS + 1,
        dtype=np.float64,
    )

    quadratic_values = np.empty(
        N_STEPS + 1,
        dtype=np.float64,
    )

    lyapunov_values = np.empty(
        N_STEPS + 1,
        dtype=np.float64,
    )

    minimum_values = np.empty(
        N_STEPS + 1,
        dtype=np.float64,
    )

    maximum_values = np.empty(
        N_STEPS + 1,
        dtype=np.float64,
    )

    times[0] = 0.0

    quadratic_values[0] = quadratic_energy(u)
    lyapunov_values[0] = lyapunov_energy(u)

    (
        minimum_values[0],
        maximum_values[0],
    ) = solution_diagnostics(u)

    save_snapshot(
        u,
        step=0,
    )

    print(
        "Initial minimum / maximum: "
        f"{minimum_values[0]:.6f}, "
        f"{maximum_values[0]:.6f}"
    )

    print(
        "Initial energies: "
        f"E2 = {quadratic_values[0]:.10e}, "
        f"Lyapunov = {lyapunov_values[0]:.10e}"
    )

    # -------------------------------------------------------------------------
    # Time stepping
    # -------------------------------------------------------------------------

    completed_steps = N_STEPS

    for step in range(
        1,
        N_STEPS + 1,
    ):

        (
            u,
            newton_iterations,
            residual_norm,
        ) = eyre_step(u)

        times[step] = (
            step * DELTA_T
        )

        quadratic_values[step] = (
            quadratic_energy(u)
        )

        lyapunov_values[step] = (
            lyapunov_energy(u)
        )

        (
            minimum_values[step],
            maximum_values[step],
        ) = solution_diagnostics(u)

        # ---------------------------------------------------------------------
        # Discrete energy check
        # ---------------------------------------------------------------------

        energy_increase = (
            lyapunov_values[step]
            - lyapunov_values[step - 1]
        )

        energy_tolerance = (
            LYAPUNOV_ATOL
            + LYAPUNOV_RTOL
            * abs(
                lyapunov_values[step - 1]
            )
        )

        if energy_increase > energy_tolerance:

            print()

            print(
                "Discrete Lyapunov energy increased "
                "beyond numerical tolerance."
            )

            print(
                f"Previous energy: "
                f"{lyapunov_values[step - 1]:.12e}"
            )

            print(
                f"Current energy:  "
                f"{lyapunov_values[step]:.12e}"
            )

            print(
                f"Increase:        "
                f"{energy_increase:.12e}"
            )

            print(
                f"Tolerance:       "
                f"{energy_tolerance:.12e}"
            )

            completed_steps = step
            break

        # ---------------------------------------------------------------------
        # Snapshot
        # ---------------------------------------------------------------------

        if step % SNAPSHOT_STRIDE == 0:

            save_snapshot(
                u,
                step,
            )

            print(
                f"Step {step:5d}/{N_STEPS} | "
                f"min = {minimum_values[step]: .6f}, "
                f"max = {maximum_values[step]: .6f} | "
                f"E2 = {quadratic_values[step]:.10e}, "
                f"Ly = {lyapunov_values[step]:.10e} | "
                f"Newton = {newton_iterations:2d} | "
                f"residual = {residual_norm:.3e}"
            )

    # -------------------------------------------------------------------------
    # Trim arrays if terminated early
    # -------------------------------------------------------------------------

    times = times[
        : completed_steps + 1
    ]

    quadratic_values = quadratic_values[
        : completed_steps + 1
    ]

    lyapunov_values = lyapunov_values[
        : completed_steps + 1
    ]

    minimum_values = minimum_values[
        : completed_steps + 1
    ]

    maximum_values = maximum_values[
        : completed_steps + 1
    ]

    # -------------------------------------------------------------------------
    # Save final state and diagnostics
    # -------------------------------------------------------------------------

    np.savetxt(
        DATA_DIR
        / "spectral_final_solution.csv",
        u,
        delimiter=",",
    )

    save_diagnostic_plots(
        times,
        quadratic_values,
        lyapunov_values,
        minimum_values,
        maximum_values,
    )

    print()

    print("Computation complete.")

    print(
        f"Final time reached: "
        f"{times[-1]:.6f}"
    )

    print(
        f"Final minimum / maximum: "
        f"{minimum_values[-1]:.6f}, "
        f"{maximum_values[-1]:.6f}"
    )

    print(
        f"Final Lyapunov energy: "
        f"{lyapunov_values[-1]:.10e}"
    )


if __name__ == "__main__":
    main()