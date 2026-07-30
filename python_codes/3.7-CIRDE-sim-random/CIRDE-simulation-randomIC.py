"""
2D Chafee--Infante equation with uniformly random initial data in [-1,1]
and homogeneous Dirichlet boundary conditions.

This script computes numerical solutions of the two-dimensional
Chafee--Infante equation
    u_t - gamma * Delta u + kappa * (u^3 - u) = 0
on the square domain [-L, L] x [-L, L], subject to homogeneous
Dirichlet boundary conditions
    u = 0 on the boundary.

Time discretization is performed by a semi-implicit convex--concave
splitting scheme. The diffusion term and the cubic nonlinearity are
treated implicitly, while the expansive linear term is treated
explicitly. At each time step, the resulting nonlinear system is
solved by Newton's method on the interior grid.

The script records:
    - solution snapshots,
    - the initial condition,
    - the quadratic energy,
    - the discrete Lyapunov energy,
    - maximum and minimum solution values.

Outputs
-------
images/phiXXXXX.png
    Solution snapshots at prescribed time intervals.

data/initial_data.csv
    Initial condition on the full grid.

data/energy.png
    Plot of the quadratic energy
        E(u) = (1/2) ∫_Omega u^2 dx.

data/Lyapunov_energy.png
    Plot of the discrete Lyapunov energy associated with the
    gradient-flow structure.

data/max-min.png
    Plot of the maximum and minimum values of the solution
    as functions of time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve


# ============================================================
# Parameters
# ============================================================

T = 1.0
N = 30000                  # number of time steps
snapshot_stride = 100      # save a snapshot every snapshot_stride steps

delta_t = 1.0e-4           # time step
L = 1.0                    # spatial domain [-L, L] x [-L, L]
M = 128                    # number of grid points in each spatial direction

hx = 2.0 * L / M
hy = hx
h = hx * hy

gamma = 5.0e-4             # diffusion coefficient
kappa = 1.0                # reaction coefficient (e.g. 1.0 or 4.7)

lyapunov_tol = 0.0         # optional stopping threshold if Lyapunov energy increases

NEWTON_MAX_ITERS = 12
NEWTON_TOL = 1.0e-10

IC_SEED = 0                # used only if a new IC is generated

os.makedirs("images", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ============================================================
# Interior grid size for homogeneous Dirichlet boundary data
# ============================================================

Mi = M - 2                 # number of interior points in each direction
J = Mi * Mi                # total number of interior unknowns


# ============================================================
# Utility functions
# ============================================================

def build_dirichlet_laplacian_1d(Mi, hx):
    """
    Construct the one-dimensional discrete Laplacian on the interior grid
    with homogeneous Dirichlet boundary conditions.
    """
    main = -2.0 * np.ones(Mi)
    off = 1.0 * np.ones(Mi - 1)
    T = diags([off, main, off], [-1, 0, 1], shape=(Mi, Mi), format="csr")
    return (1.0 / hx**2) * T


def build_dirichlet_laplacian_2d(M, hx):
    """
    Construct the two-dimensional discrete Laplacian on the interior grid
    with homogeneous Dirichlet boundary conditions.
    """
    Mi = M - 2
    T = build_dirichlet_laplacian_1d(Mi, hx)
    I = eye(Mi, format="csr")
    L2D = kron(I, T) + kron(T, I)
    return L2D.tocsr()


def flatten_interior(u_int):
    """Flatten an interior (Mi x Mi) array to a vector of length J."""
    return u_int.ravel()


def unflatten_interior(v):
    """Reshape a vector of length J into an interior (Mi x Mi) array."""
    return v.reshape(Mi, Mi)


def random_dbc_ic_uniform(M, low=-1.0, high=1.0, seed=None):
    """
    Generate uniformly random initial data on the interior grid, with
    homogeneous Dirichlet boundary values fixed to zero.

    The interior values are sampled independently from Uniform[low, high].

    Returns
    -------
    np.ndarray
        Full (M x M) array with zero boundary values.
    """
    rng = np.random.default_rng(seed)

    Mi = M - 2
    u_full = np.zeros((M, M), dtype=np.float64)
    u_full[1:-1, 1:-1] = rng.uniform(low, high, size=(Mi, Mi))
    return u_full.astype(np.float32)


def quadratic_energy(u):
    """
    Compute the quadratic energy
        E(u) = (1/2) ∫_Omega u^2 dx
    using a rectangular quadrature rule.
    """
    return 0.5 * h * np.sum(u**2)


def lyapunov_energy_discrete(u):
    """
    Compute the discrete Lyapunov energy
        E_h(u) = ∫_Omega [ (gamma/2) |grad_h u|^2
                           + kappa (u^4/4 - u^2/2) ] dx,
    using forward differences on the full grid.
    """
    ux_f = (u[1:, :] - u[:-1, :]) / hx
    uy_f = (u[:, 1:] - u[:, :-1]) / hy

    grad_sq = np.sum(ux_f**2) + np.sum(uy_f**2)
    potential = 0.25 * u**4 - 0.5 * u**2

    return (0.5 * gamma * grad_sq + kappa * np.sum(potential)) * h


# ============================================================
# Discrete Laplacian and identity on the interior grid
# ============================================================

L2D = build_dirichlet_laplacian_2d(M, hx)
I_big = eye(J, format="csr")


# ============================================================
# Initialization
# ============================================================

phi = np.zeros((N + 1, M, M), dtype=np.float32)

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

energy_vals = np.zeros(N + 1)
lyapunov_vals = np.zeros(N + 1)
u_min_vals = np.zeros(N + 1)
u_max_vals = np.zeros(N + 1)

energy_vals[0] = quadratic_energy(phi[0])
lyapunov_vals[0] = lyapunov_energy_discrete(phi[0])
u_min_vals[0] = np.min(phi[0])
u_max_vals[0] = np.max(phi[0])

plt.figure()
plt.axis("off")
plt.imshow(phi[0], cmap="magma", interpolation="none")
plt.colorbar()
plt.savefig("images/phi00000.png", bbox_inches="tight", pad_inches=0)
plt.close()

print(f"Initial minimum / maximum: {u_min_vals[0]:.6f}, {u_max_vals[0]:.6f}")
print(f"Initial variance: {np.var(phi[0]):.6f}")
print(f"Initial energies: E = {energy_vals[0]:.10f}, Lyapunov = {lyapunov_vals[0]:.10f}")


# ============================================================
# Time stepping:
# semi-implicit convex--concave splitting with Newton iteration
# ============================================================
#
# We solve, on the interior grid,
#
#   u^{n+1} - delta_t * gamma * L u^{n+1}
#           + delta_t * kappa * (u^{n+1})^3
#   = u^n + delta_t * kappa * u^n.
#
# Equivalently,
#
#   F(u) = u - delta_t * gamma * L u + delta_t * kappa * u^3
#          - (u^n + delta_t * kappa * u^n),
#
# and solve F(u) = 0 by Newton's method.
# ============================================================

for n in range(1, N + 1):
    u_prev = phi[n - 1].copy()
    u_prev_int = u_prev[1:-1, 1:-1]

    rhs_int = u_prev_int + delta_t * kappa * u_prev_int
    u_int = u_prev_int.copy()

    for newton_iter in range(NEWTON_MAX_ITERS):
        u_flat = flatten_interior(u_int)

        residual = (
            u_flat
            - delta_t * gamma * (L2D @ u_flat)
            + delta_t * kappa * (u_flat**3)
            - rhs_int.ravel()
        )

        residual_norm = np.linalg.norm(residual, ord=2)
        if residual_norm < NEWTON_TOL:
            break

        jacobian_diag = delta_t * kappa * 3.0 * (u_flat**2)
        jacobian = (I_big - delta_t * gamma * L2D) + diags(jacobian_diag, 0, format="csr")

        delta = spsolve(jacobian, -residual)
        u_flat = u_flat + delta
        u_int = unflatten_interior(u_flat)

    u_full = np.zeros((M, M), dtype=np.float64)
    u_full[1:-1, 1:-1] = u_int

    # enforce homogeneous Dirichlet boundary data explicitly
    u_full[0, :] = 0.0
    u_full[-1, :] = 0.0
    u_full[:, 0] = 0.0
    u_full[:, -1] = 0.0

    phi[n] = u_full

    energy_vals[n] = quadratic_energy(phi[n])
    lyapunov_vals[n] = lyapunov_energy_discrete(phi[n])
    u_min_vals[n] = np.min(phi[n])
    u_max_vals[n] = np.max(phi[n])

    print(
        f"Step {n:5d}/{N} | "
        f"min = {u_min_vals[n]: .6f}, max = {u_max_vals[n]: .6f} | "
        f"E = {energy_vals[n]:.10f}, Ly = {lyapunov_vals[n]:.10f} | "
        f"Newton iterations = {newton_iter + 1}"
    )

    if n % snapshot_stride == 0:
        plt.figure()
        plt.axis("off")
        plt.imshow(phi[n], cmap="magma", interpolation="none")
        plt.colorbar()
        plt.savefig(f"images/phi{str(n).zfill(5)}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        plt.figure()
        x = np.arange(1, n + 1)
        plt.plot(x, energy_vals[1:n + 1], color="k")
        plt.xlabel("Time step")
        plt.ylabel("Quadratic energy")
        plt.savefig("data/energy.png")
        plt.close()

        plt.figure()
        x = np.arange(1, n + 1)
        plt.plot(x, lyapunov_vals[1:n + 1], color="k")
        plt.xlabel("Time step")
        plt.ylabel("Discrete Lyapunov energy")
        plt.savefig("data/Lyapunov_energy.png")
        plt.close()

        plt.figure()
        x = np.arange(0, n + 1)
        plt.plot(x, u_max_vals[0:n + 1], "k--", label="max")
        plt.plot(x, u_min_vals[0:n + 1], "k:", label="min")
        plt.xlabel("Time step")
        plt.ylabel("Extremal values")
        plt.legend()
        plt.savefig("data/max-min.png")
        plt.close()

    if lyapunov_vals[n] - lyapunov_vals[n - 1] > lyapunov_tol:
        print("Discrete Lyapunov energy increased; terminating early.")
        break

print("Computation complete.")
