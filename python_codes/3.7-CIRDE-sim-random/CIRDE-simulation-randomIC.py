#"""
#2D Chafee--Infante equation with uniformly random initial data in [-1,1]
#and homogeneous Dirichlet boundary conditions.
#
#This script computes numerical solutions of the two-dimensional
#Chafee--Infante equation
#    u_t - gamma * Delta u + kappa * (u^3 - u) = 0
#on the square domain [-L, L] x [-L, L], subject to homogeneous
#Dirichlet boundary conditions
#    u = 0 on the boundary.
#
#Time discretization is performed by a semi-implicit convex--concave
#splitting scheme. The diffusion term and the cubic nonlinearity are
#treated implicitly, while the expansive linear term is treated
#explicitly. At each time step, the resulting nonlinear system is
#solved by Newton's method on the interior grid.
#
#The script records:
#    - solution snapshots,
#    - the initial condition,
#    - the quadratic energy,
#    - the discrete Lyapunov energy,
#    - maximum and minimum solution values.
#
#Outputs
#-------
#images/phiXXXXX.png
#    Solution snapshots at prescribed time intervals.
#
#data/initial_data.csv
#    Initial condition on the full grid.
#
#data/energy.png
#    Plot of the quadratic energy
#        E(u) = (1/2) ∫_Omega u^2 dx.
#
#data/Lyapunov_energy.png
#    Plot of the discrete Lyapunov energy associated with the
#    gradient-flow structure.
#
#data/max-min.png
#    Plot of the maximum and minimum values of the solution
#    as functions of time.
#"""
#
#import os
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.sparse import diags, kron, eye
#from scipy.sparse.linalg import spsolve
#
#
## ============================================================
## Parameters
## ============================================================
#
#T = 1.0
#N = 30000                  # number of time steps
#snapshot_stride = 100      # save a snapshot every snapshot_stride steps
#
#delta_t = 1.0e-4           # time step
#L = 1.0                    # spatial domain [-L, L] x [-L, L]
#M = 128                    # number of grid points in each spatial direction
#
#hx = 2.0 * L / M
#hy = hx
#h = hx * hy
#
#gamma = 5.0e-4             # diffusion coefficient
#kappa = 1.0                # reaction coefficient (e.g. 1.0 or 4.7)
#
#lyapunov_tol = 0.0         # optional stopping threshold if Lyapunov energy increases
#
#NEWTON_MAX_ITERS = 12
#NEWTON_TOL = 1.0e-10
#
#IC_SEED = 0                # used only if a new IC is generated
#
#os.makedirs("images", exist_ok=True)
#os.makedirs("data", exist_ok=True)
#
#
## ============================================================
## Interior grid size for homogeneous Dirichlet boundary data
## ============================================================
#
#Mi = M - 2                 # number of interior points in each direction
#J = Mi * Mi                # total number of interior unknowns
#
#
## ============================================================
## Utility functions
## ============================================================
#
#def build_dirichlet_laplacian_1d(Mi, hx):
#    """
#    Construct the one-dimensional discrete Laplacian on the interior grid
#    with homogeneous Dirichlet boundary conditions.
#    """
#    main = -2.0 * np.ones(Mi)
#    off = 1.0 * np.ones(Mi - 1)
#    T = diags([off, main, off], [-1, 0, 1], shape=(Mi, Mi), format="csr")
#    return (1.0 / hx**2) * T
#
#
#def build_dirichlet_laplacian_2d(M, hx):
#    """
#    Construct the two-dimensional discrete Laplacian on the interior grid
#    with homogeneous Dirichlet boundary conditions.
#    """
#    Mi = M - 2
#    T = build_dirichlet_laplacian_1d(Mi, hx)
#    I = eye(Mi, format="csr")
#    L2D = kron(I, T) + kron(T, I)
#    return L2D.tocsr()
#
#
#def flatten_interior(u_int):
#    """Flatten an interior (Mi x Mi) array to a vector of length J."""
#    return u_int.ravel()
#
#
#def unflatten_interior(v):
#    """Reshape a vector of length J into an interior (Mi x Mi) array."""
#    return v.reshape(Mi, Mi)
#
#
#def random_dbc_ic_uniform(M, low=-1.0, high=1.0, seed=None):
#    """
#    Generate uniformly random initial data on the interior grid, with
#    homogeneous Dirichlet boundary values fixed to zero.
#
#    The interior values are sampled independently from Uniform[low, high].
#
#    Returns
#    -------
#    np.ndarray
#        Full (M x M) array with zero boundary values.
#    """
#    rng = np.random.default_rng(seed)
#
#    Mi = M - 2
#    u_full = np.zeros((M, M), dtype=np.float64)
#    u_full[1:-1, 1:-1] = rng.uniform(low, high, size=(Mi, Mi))
#    return u_full.astype(np.float32)
#
#
#def quadratic_energy(u):
#    """
#    Compute the quadratic energy
#        E(u) = (1/2) ∫_Omega u^2 dx
#    using a rectangular quadrature rule.
#    """
#    return 0.5 * h * np.sum(u**2)
#
#
#def lyapunov_energy_discrete(u):
#    """
#    Compute the discrete Lyapunov energy
#        E_h(u) = ∫_Omega [ (gamma/2) |grad_h u|^2
#                           + kappa (u^4/4 - u^2/2) ] dx,
#    using forward differences on the full grid.
#    """
#    ux_f = (u[1:, :] - u[:-1, :]) / hx
#    uy_f = (u[:, 1:] - u[:, :-1]) / hy
#
#    grad_sq = np.sum(ux_f**2) + np.sum(uy_f**2)
#    potential = 0.25 * u**4 - 0.5 * u**2
#
#    return (0.5 * gamma * grad_sq + kappa * np.sum(potential)) * h
#
#
## ============================================================
## Discrete Laplacian and identity on the interior grid
## ============================================================
#
#L2D = build_dirichlet_laplacian_2d(M, hx)
#I_big = eye(J, format="csr")
#
#
## ============================================================
## Initialization
## ============================================================
#
#phi = np.zeros((N + 1, M, M), dtype=np.float32)
#
#initial_data_path = "data/initial_data.csv"
#
#if os.path.exists(initial_data_path):
#    print("Loading existing initial condition...")
#    phi[0] = np.loadtxt(initial_data_path, delimiter=",").astype(np.float32)
#else:
#    print("Generating new uniformly random initial condition in [-1,1]...")
#    phi[0] = random_dbc_ic_uniform(
#        M,
#        low=-1.0,
#        high=1.0,
#        seed=IC_SEED
#    )
#    np.savetxt(initial_data_path, phi[0], delimiter=",", fmt="%.12e")
#    print("Initial condition saved to:", initial_data_path)
#
#energy_vals = np.zeros(N + 1)
#lyapunov_vals = np.zeros(N + 1)
#u_min_vals = np.zeros(N + 1)
#u_max_vals = np.zeros(N + 1)
#
#energy_vals[0] = quadratic_energy(phi[0])
#lyapunov_vals[0] = lyapunov_energy_discrete(phi[0])
#u_min_vals[0] = np.min(phi[0])
#u_max_vals[0] = np.max(phi[0])
#
#plt.figure()
#plt.axis("off")
#plt.imshow(phi[0], cmap="magma", interpolation="none")
#plt.colorbar()
#plt.savefig("images/phi00000.png", bbox_inches="tight", pad_inches=0)
#plt.close()
#
#print(f"Initial minimum / maximum: {u_min_vals[0]:.6f}, {u_max_vals[0]:.6f}")
#print(f"Initial variance: {np.var(phi[0]):.6f}")
#print(f"Initial energies: E = {energy_vals[0]:.10f}, Lyapunov = {lyapunov_vals[0]:.10f}")
#
#
## ============================================================
## Time stepping:
## semi-implicit convex--concave splitting with Newton iteration
## ============================================================
##
## We solve, on the interior grid,
##
##   u^{n+1} - delta_t * gamma * L u^{n+1}
##           + delta_t * kappa * (u^{n+1})^3
##   = u^n + delta_t * kappa * u^n.
##
## Equivalently,
##
##   F(u) = u - delta_t * gamma * L u + delta_t * kappa * u^3
##          - (u^n + delta_t * kappa * u^n),
##
## and solve F(u) = 0 by Newton's method.
## ============================================================
#
#for n in range(1, N + 1):
#    u_prev = phi[n - 1].copy()
#    u_prev_int = u_prev[1:-1, 1:-1]
#
#    rhs_int = u_prev_int + delta_t * kappa * u_prev_int
#    u_int = u_prev_int.copy()
#
#    for newton_iter in range(NEWTON_MAX_ITERS):
#        u_flat = flatten_interior(u_int)
#
#        residual = (
#            u_flat
#            - delta_t * gamma * (L2D @ u_flat)
#            + delta_t * kappa * (u_flat**3)
#            - rhs_int.ravel()
#        )
#
#        residual_norm = np.linalg.norm(residual, ord=2)
#        if residual_norm < NEWTON_TOL:
#            break
#
#        jacobian_diag = delta_t * kappa * 3.0 * (u_flat**2)
#        jacobian = (I_big - delta_t * gamma * L2D) + diags(jacobian_diag, 0, format="csr")
#
#        delta = spsolve(jacobian, -residual)
#        u_flat = u_flat + delta
#        u_int = unflatten_interior(u_flat)
#
#    u_full = np.zeros((M, M), dtype=np.float64)
#    u_full[1:-1, 1:-1] = u_int
#
#    # enforce homogeneous Dirichlet boundary data explicitly
#    u_full[0, :] = 0.0
#    u_full[-1, :] = 0.0
#    u_full[:, 0] = 0.0
#    u_full[:, -1] = 0.0
#
#    phi[n] = u_full
#
#    energy_vals[n] = quadratic_energy(phi[n])
#    lyapunov_vals[n] = lyapunov_energy_discrete(phi[n])
#    u_min_vals[n] = np.min(phi[n])
#    u_max_vals[n] = np.max(phi[n])
#
#    print(
#        f"Step {n:5d}/{N} | "
#        f"min = {u_min_vals[n]: .6f}, max = {u_max_vals[n]: .6f} | "
#        f"E = {energy_vals[n]:.10f}, Ly = {lyapunov_vals[n]:.10f} | "
#        f"Newton iterations = {newton_iter + 1}"
#    )
#
#    if n % snapshot_stride == 0:
#        plt.figure()
#        plt.axis("off")
#        plt.imshow(phi[n], cmap="magma", interpolation="none")
#        plt.colorbar()
#        plt.savefig(f"images/phi{str(n).zfill(5)}.png", bbox_inches="tight", pad_inches=0)
#        plt.close()
#
#        plt.figure()
#        x = np.arange(1, n + 1)
#        plt.plot(x, energy_vals[1:n + 1], color="k")
#        plt.xlabel("Time step")
#        plt.ylabel("Quadratic energy")
#        plt.savefig("data/energy.png")
#        plt.close()
#
#        plt.figure()
#        x = np.arange(1, n + 1)
#        plt.plot(x, lyapunov_vals[1:n + 1], color="k")
#        plt.xlabel("Time step")
#        plt.ylabel("Discrete Lyapunov energy")
#        plt.savefig("data/Lyapunov_energy.png")
#        plt.close()
#
#        plt.figure()
#        x = np.arange(0, n + 1)
#        plt.plot(x, u_max_vals[0:n + 1], "k--", label="max")
#        plt.plot(x, u_min_vals[0:n + 1], "k:", label="min")
#        plt.xlabel("Time step")
#        plt.ylabel("Extremal values")
#        plt.legend()
#        plt.savefig("data/max-min.png")
#        plt.close()
#
#    if lyapunov_vals[n] - lyapunov_vals[n - 1] > lyapunov_tol:
#        print("Discrete Lyapunov energy increased; terminating early.")
#        break
#
#print("Computation complete.")
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

initial_data_path = "data/initial_data.csv"

if os.path.exists(initial_data_path):
    print("Loading existing initial condition...")
    phi[0] = np.loadtxt(initial_data_path, delimiter=",").astype(np.float32)
else:
    print("Generating new uniformly random initial condition in [-1,1]...")
    phi[0] = random_dbc_ic_uniform(
        M,
        low=-1.0,
        high=1.0,
        seed=IC_SEED
    )
    np.savetxt(initial_data_path, phi[0], delimiter=",", fmt="%.12e")
    print("Initial condition saved to:", initial_data_path)


def random_initial_condition(
    laplacian,
    length_scale=POISSON_LENGTH,
    amplitude=INITIAL_AMPLITUDE,
    seed=RANDOM_SEED,
):
    """
    Generate a smooth initial condition satisfying homogeneous
    Dirichlet boundary conditions.

    A white-noise field xi is smoothed by the screened Poisson problem

        (I - ell^2 Delta) u = xi

    on the interior grid. The result is rescaled so that its maximum
    absolute value is ``amplitude``.
    """
    initial_data_path = "data/initial_data.csv"

    if os.path.exists(initial_data_path):
        print("Loading existing initial condition...")
        phi[0] = np.loadtxt(initial_data_path, delimiter=",").astype(np.float32)
    else:
        print("Generating new uniformly random initial condition in [-1,1]...")
        phi[0] = random_dbc_ic_uniform(
            M,
            low=-1.0,
            high=1.0,
            seed=IC_SEED
        )
        np.savetxt(initial_data_path, phi[0], delimiter=",", fmt="%.12e")
        print("Initial condition saved to:", initial_data_path)

    
    
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
        u = random_initial_condition(LAPLACIAN)
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
