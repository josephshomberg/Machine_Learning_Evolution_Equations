"""
2D Dirichlet Poisson Solver on a Cartesian Grid

This example generates smooth initial data by solving the Poisson problem

    -Delta u = f    in Omega,
           u = 0    on boundary(Omega),

where

    Omega = [-L, L]^2,

and f is a random source term.

The interior solution is normalized and scaled to produce smooth
initial data suitable for numerical experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


# ============================================================
# Parameters
# ============================================================

M = 128
L = 1.0
AMP = 0.02
SEED = 42


# ============================================================
# Generate smooth Poisson initial data
# ============================================================

def generate_poisson_initial_data(
    M=128,
    L=1.0,
    amplitude=0.02,
    seed=42,
):
    """
    Generate smooth initial data by solving a discrete Poisson problem.

    The continuous problem is

        -Delta u = f    in Omega,
               u = 0    on boundary(Omega),

    where Omega = [-L, L]^2 and f is a random source.

    Parameters
    ----------
    M : int
        Total number of grid points in each spatial direction,
        including boundary points.
    L : float
        Half-width of the square domain.
    amplitude : float
        Maximum absolute amplitude of the returned field.
    seed : int
        Random seed used to generate the source term.

    Returns
    -------
    u0 : ndarray
        Array of shape (M, M) containing the initial data.
    f_source : ndarray
        Interior random source term.
    hx : float
        Spatial grid spacing.
    """

    if M < 3:
        raise ValueError("M must be at least 3.")

    if L <= 0.0:
        raise ValueError("L must be positive.")

    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive.")

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    hx = 2.0 * L / (M - 1)

    mi = M - 2
    n_int = mi * mi

    # --------------------------------------------------------
    # One-dimensional second-difference operator
    # --------------------------------------------------------

    main_diag = -2.0 * np.ones(mi)
    off_diag = np.ones(mi - 1)

    T = diags(
        [off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        shape=(mi, mi),
        format="csr",
    )

    T /= hx**2

    # --------------------------------------------------------
    # Two-dimensional Dirichlet Laplacian
    #
    # Delta_h = I kron T + T kron I
    # --------------------------------------------------------

    I = eye(mi, format="csr")

    laplacian = (
        kron(I, T, format="csr")
        + kron(T, I, format="csr")
    )

    # --------------------------------------------------------
    # Random source and discrete Poisson solve
    #
    # -Delta_h u = f
    # --------------------------------------------------------

    f_source = rng.standard_normal(n_int)

    u_interior = spsolve(
        -laplacian,
        f_source,
    )

    u_interior = u_interior.reshape((mi, mi))

    # --------------------------------------------------------
    # Normalize and scale
    # --------------------------------------------------------

    max_abs = np.max(np.abs(u_interior))

    if max_abs == 0.0:
        raise RuntimeError(
            "The Poisson solution has zero magnitude and cannot be normalized."
        )

    u_interior /= max_abs

    # --------------------------------------------------------
    # Embed into the full grid
    #
    # Homogeneous Dirichlet boundary conditions are imposed
    # by setting all boundary values equal to zero.
    # --------------------------------------------------------

    u0 = np.zeros((M, M), dtype=float)

    u0[1:-1, 1:-1] = amplitude * u_interior

    return u0, f_source, hx


# ============================================================
# Main program
# ============================================================

def main():
    """Generate and display smooth Poisson initial data."""

    u0, _, hx = generate_poisson_initial_data(
        M=M,
        L=L,
        amplitude=AMP,
        seed=SEED,
    )

    print(f"Grid size: {u0.shape}")
    print(f"Grid spacing: h = {hx:.8f}")
    print(f"Minimum value: {u0.min():.6f}")
    print(f"Maximum value: {u0.max():.6f}")

    boundary_max = max(
        np.max(np.abs(u0[0, :])),
        np.max(np.abs(u0[-1, :])),
        np.max(np.abs(u0[:, 0])),
        np.max(np.abs(u0[:, -1])),
    )

    print(
        "Maximum absolute boundary value: "
        f"{boundary_max:.6e}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        u0,
        origin="lower",
        extent=[-L, L, -L, L],
        cmap="coolwarm",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Amplitude",
    )

    ax.set_title("Smooth Poisson Initial Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
