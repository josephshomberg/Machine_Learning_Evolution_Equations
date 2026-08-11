"""
Two-dimensional Chafee--Infante equation: Eyre time stepping
==============================================================

This script computes numerical solutions of the two-dimensional
Chafee--Infante equation

    u_t - gamma * Delta u + kappa * (u^3 - u) = 0

on the square domain [-L, L] x [-L, L], subject to homogeneous
Dirichlet boundary conditions.

The time discretization uses an Eyre-style convex--concave splitting.
The diffusion term and cubic term are treated implicitly, while the
expansive linear term is treated explicitly. At each time step, the
resulting nonlinear system is solved by Newton's method on the
interior grid.

The computation records selected solution snapshots, the initial
condition, the quadratic energy, the discrete Lyapunov energy, and
the minimum and maximum solution values.

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


# ============================================================================
# Parameters
# ============================================================================

N_STEPS = 30000
DELTA_T = 1.0e-4

L = 1.0
M = 128

GAMMA = 5.0e-4
KAPPA = 4.7

SNAPSHOT_STRIDE = 1000

INITIAL_AMPLITUDE = 0.02
POISSON_LENGTH = 0.06
RANDOM_SEED = 0

NEWTON_MAX_ITERS = 12
NEWTON_TOL = 1.0e-10

# Terminate if the discrete Lyapunov energy increases by more than this
# tolerance. Set to zero to require monotone decrease up to roundoff.
LYAPUNOV_TOL = 0.0

IMAGE_DIR = Path("images")
DATA_DIR = Path("data")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Grid
# ============================================================================

# M grid points include both endpoints -L and L.
X = np.linspace(-L, L, M)
Y = np.linspace(-L, L, M)

HX = X[1] - X[0]
HY = Y[1] - Y[0]
CELL_AREA = HX * HY

# Interior grid size for homogeneous Dirichlet boundary conditions.
MI = M - 2
N_INTERIOR = MI * MI


# ============================================================================
# Discrete operators
# ============================================================================

def build_dirichlet_laplacian_1d(n_points, mesh_size):
    """
    Construct the one-dimensional Laplacian on the interior grid.

    Homogeneous Dirichlet values are imposed at the two boundary points,
    so the matrix acts only on the interior unknowns.
    """
    main = -2.0 * np.ones(n_points)
    off = np.ones(n_points - 1)

    laplacian = diags(
        [off, main, off],
        offsets=[-1, 0, 1],
        shape=(n_points, n_points),
        format="csr",
    )

    return laplacian / mesh_size**2


def build_dirichlet_laplacian_2d(n_points, mesh_size):
    """
    Construct the two-dimensional Dirichlet Laplacian.

    The operator is assembled from the one-dimensional operator by
    Kronecker products and acts only on the interior grid.
    """
    n_interior = n_points - 2
    laplacian_1d = build_dirichlet_laplacian_1d(
        n_interior, mesh_size
    )
    identity = eye(n_interior, format="csr")

    return (
        kron(identity, laplacian_1d)
        + kron(laplacian_1d, identity)
    ).tocsr()


LAPLACIAN = build_dirichlet_laplacian_2d(M, HX)
IDENTITY = eye(N_INTERIOR, format="csr")


# ============================================================================
# Initial condition
# ============================================================================

def smooth_poisson_initial_condition(
    laplacian,
    length_scale=POISSON_LENGTH,
    amplitude=INITIAL_AMPLITUDE,
    seed=RANDOM_SEED,
):
    r"""
    Generate a smooth initial condition satisfying homogeneous
    Dirichlet boundary conditions.

    A white-noise field xi is smoothed by the screened Poisson problem

        (I - ell^2 Delta) u = xi

    on the interior grid. The result is rescaled so that its maximum
    absolute value is ``amplitude``.
    """
    rng = np.random.default_rng(seed)

    noise = rng.standard_normal((MI, MI))
    rhs = noise.ravel()

    screened_operator = IDENTITY - length_scale**2 * laplacian
    u_interior = spsolve(
        screened_operator, rhs
    ).reshape(MI, MI)

    maximum = np.max(np.abs(u_interior))

    if maximum > 0.0:
        u_interior *= amplitude / maximum

    u = np.zeros((M, M), dtype=np.float64)
    u[1:-1, 1:-1] = u_interior

    return u.astype(np.float32)


# ============================================================================
# Energies and diagnostics
# ============================================================================

def quadratic_energy(u):
    r"""
    Compute the quadratic energy

        E_2(u) = (1/2) integral_Omega u^2 dx

    using rectangular quadrature.
    """
    return 0.5 * CELL_AREA * np.sum(u**2)


def lyapunov_energy(u):
    r"""
    Compute the discrete Lyapunov energy

        E(u) = integral_Omega [
            (gamma/2) |grad u|^2
            + kappa (u^4/4 - u^2/2)
        ] dx.

    Forward differences are used for the gradient.
    """
    ux = np.diff(u, axis=0) / HX
    uy = np.diff(u, axis=1) / HY

    gradient_squared = np.sum(ux**2) + np.sum(uy**2)
    potential = 0.25 * u**4 - 0.5 * u**2

    return (
        0.5 * GAMMA * gradient_squared
        + KAPPA * np.sum(potential)
    ) * CELL_AREA


def solution_diagnostics(u):
    """Return the minimum and maximum values of the solution."""
    return np.min(u), np.max(u)


# ============================================================================
# Eyre time step
# ============================================================================

def eyre_step(u_previous):
    r"""
    Advance the solution by one Eyre time step.

    For

        u_t - gamma Delta u + kappa (u^3 - u) = 0,

    the cubic term and diffusion are treated implicitly, while the
    expansive linear term is treated explicitly:

        u^{n+1}
        - dt gamma Delta u^{n+1}
        + dt kappa (u^{n+1})^3
        =
        u^n + dt kappa u^n.

    Define

        F(v) =
            v - dt gamma L v
            + dt kappa v^3
            - (u^n + dt kappa u^n).

    Newton's method solves F(v) = 0 on the interior grid.
    """
    previous_interior = u_previous[1:-1, 1:-1]
    rhs = (1.0 + DELTA_T * KAPPA) * previous_interior
    u_interior = previous_interior.copy()

    converged = False
    residual_norm = np.inf

    for iteration in range(1, NEWTON_MAX_ITERS + 1):
        u_flat = u_interior.ravel()

        residual = (
            u_flat
            - DELTA_T * GAMMA * (LAPLACIAN @ u_flat)
            + DELTA_T * KAPPA * u_flat**3
            - rhs.ravel()
        )

        residual_norm = np.linalg.norm(residual, ord=2)

        if residual_norm < NEWTON_TOL:
            converged = True
            break

        jacobian_diagonal = (
            3.0 * DELTA_T * KAPPA * u_flat**2
        )

        jacobian = (
            IDENTITY
            - DELTA_T * GAMMA * LAPLACIAN
            + diags(jacobian_diagonal, 0, format="csr")
        )

        correction = spsolve(jacobian, -residual)
        u_interior = (u_flat + correction).reshape(MI, MI)

    if not converged:
        raise RuntimeError(
            "Newton iteration failed to converge after "
            f"{NEWTON_MAX_ITERS} iterations; "
            f"final residual norm = {residual_norm:.3e}."
        )

    u = np.zeros((M, M), dtype=np.float32)
    u[1:-1, 1:-1] = u_interior.astype(np.float32)

    return u, iteration, residual_norm


# ============================================================================
# Output
# ============================================================================

def save_snapshot(u, step):
    """Save a solution snapshot as a PNG image."""
    figure = plt.figure()
    plt.axis("off")
    plt.imshow(u, cmap="magma", interpolation="none")
    plt.colorbar()
    plt.savefig(
        IMAGE_DIR / f"phi{step:05d}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close(figure)


def save_diagnostic_plots(
    times,
    quadratic_values,
    lyapunov_values,
    minimum_values,
    maximum_values,
):
    """Save the energy and solution-extrema plots."""
    figure = plt.figure()
    plt.plot(times, quadratic_values, color="k")
    plt.xlabel("Time")
    plt.ylabel(r"$E_2(u)$")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "energy.png", dpi=300)
    plt.close(figure)

    figure = plt.figure()
    plt.plot(times, lyapunov_values, color="k")
    plt.xlabel("Time")
    plt.ylabel(r"$E(u)$")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "Lyapunov_energy.png", dpi=300)
    plt.close(figure)

    figure = plt.figure()
    plt.plot(times, maximum_values, "k--", label="maximum")
    plt.plot(times, minimum_values, "k:", label="minimum")
    plt.xlabel("Time")
    plt.ylabel("Extremal values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DATA_DIR / "max-min.png", dpi=300)
    plt.close(figure)


# ============================================================================
# Main computation
# ============================================================================

def main():
    """Run the Chafee--Infante simulation."""
    print("Two-dimensional Chafee--Infante equation")
    print("------------------------------------------")
    print(f"Grid:          {M} x {M}")
    print(f"Domain:        [-{L}, {L}] x [-{L}, {L}]")
    print(f"Mesh size:     {HX:.6e}")
    print(f"Time steps:    {N_STEPS}")
    print(f"Time step:     {DELTA_T:.6e}")
    print(f"Final time:    {N_STEPS*DELTA_T:.6f}")
    print(f"gamma:         {GAMMA:.6e}")
    print(f"kappa:         {KAPPA:.6f}")
    print()

    initial_data_path = DATA_DIR / "initial_data.csv"

    if initial_data_path.exists():
        print("Loading existing initial condition...")
        u = np.loadtxt(initial_data_path, delimiter=",").astype(
            np.float32
        )
    else:
        print("Generating smooth Poisson initial condition...")
        u = smooth_poisson_initial_condition(LAPLACIAN)
        np.savetxt(initial_data_path, u, delimiter=",")
        print(f"Initial condition saved to {initial_data_path}")

    if u.shape != (M, M):
        raise ValueError(
            f"Initial condition has shape {u.shape}; "
            f"expected {(M, M)}."
        )

    # Store only scalar diagnostics. The complete solution history is
    # intentionally not retained in memory.
    times = np.empty(N_STEPS + 1)
    quadratic_values = np.empty(N_STEPS + 1)
    lyapunov_values = np.empty(N_STEPS + 1)
    minimum_values = np.empty(N_STEPS + 1)
    maximum_values = np.empty(N_STEPS + 1)

    times[0] = 0.0
    quadratic_values[0] = quadratic_energy(u)
    lyapunov_values[0] = lyapunov_energy(u)
    minimum_values[0], maximum_values[0] = solution_diagnostics(u)

    save_snapshot(u, 0)

    print(
        f"Initial minimum / maximum: "
        f"{minimum_values[0]:.6f}, {maximum_values[0]:.6f}"
    )
    print(
        f"Initial energies: "
        f"E2 = {quadratic_values[0]:.10e}, "
        f"Lyapunov = {lyapunov_values[0]:.10e}"
    )

    completed_steps = N_STEPS

    for step in range(1, N_STEPS + 1):
        u, newton_iterations, residual_norm = eyre_step(u)

        times[step] = step * DELTA_T
        quadratic_values[step] = quadratic_energy(u)
        lyapunov_values[step] = lyapunov_energy(u)
        minimum_values[step], maximum_values[step] = (
            solution_diagnostics(u)
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

        if step % SNAPSHOT_STRIDE == 0:
            save_snapshot(u, step)

        energy_increase = (
            lyapunov_values[step] - lyapunov_values[step - 1]
        )

        if energy_increase > LYAPUNOV_TOL:
            print(
                "Discrete Lyapunov energy increased; "
                "terminating early."
            )
            completed_steps = step
            break

    times = times[:completed_steps + 1]
    quadratic_values = quadratic_values[:completed_steps + 1]
    lyapunov_values = lyapunov_values[:completed_steps + 1]
    minimum_values = minimum_values[:completed_steps + 1]
    maximum_values = maximum_values[:completed_steps + 1]

    save_diagnostic_plots(
        times,
        quadratic_values,
        lyapunov_values,
        minimum_values,
        maximum_values,
    )

    print()
    print("Computation complete.")
    print(f"Final time reached: {times[-1]:.6f}")
    print(f"Final Lyapunov energy: {lyapunov_values[-1]:.10e}")


if __name__ == "__main__":
    main()
