"""
Two-dimensional Allen--Cahn equation on a periodic square.

Equation
--------
    u_t = Delta u - (u^3 - u)
        = Delta u + u - u^3,

on
    Omega = [0, L]^2,

with periodic boundary conditions.

Spatial discretization
----------------------
A uniform M x M periodic grid is used with

    dx = dy = L / M.

The Laplacian is approximated by the standard second-order five-point
periodic finite-difference stencil.

Time discretization
-------------------
The diffusion term is treated implicitly and the reaction term explicitly:

    (u^{n+1} - u^n) / dt
        = Delta_h u^{n+1} + u^n - (u^n)^3,

or equivalently

    (I - dt Delta_h) u^{n+1}
        = (1 + dt) u^n - dt (u^n)^3.

This is a first-order semi-implicit (IMEX Euler) scheme.  It should not be
called a convex-splitting scheme: the nonlinear cubic term is evaluated
explicitly.  The matrix I - dt Delta_h is constant in time and is therefore
factorized once before the time-stepping loop.

Diagnostics
-----------
The code records

    * discrete Ginzburg--Landau energy,
    * discrete interfacial density,
    * solution minimum and maximum,
    * phase-field snapshots.

The discrete interfacial density is

    rho_int,h(u)
      = (dx dy / |Omega|) sum_ij |grad_h u_ij|,

which is the discrete quantity used in the book.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import factorized


# ============================================================
# Parameters
# ============================================================

L = 1.0
M = 128
DT = 5.0e-4
T_FINAL = 5.0
N_STEPS = int(round(T_FINAL / DT))

SNAPSHOT_STRIDE = 100
DIAGNOSTIC_STEPS = 20
SEED = 0

IMAGE_DIR = Path("images")
DATA_DIR = Path("data")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Periodic grid on Omega = [0,L]^2
# ============================================================

DX = L / M
DY = L / M
CELL_AREA = DX * DY
DOMAIN_AREA = L * L
NUM_DOF = M * M

x = np.linspace(0.0, L, M, endpoint=False)
y = np.linspace(0.0, L, M, endpoint=False)


# ============================================================
# Periodic finite-difference Laplacian
# ============================================================

def build_periodic_laplacian_1d(num_points: int, spacing: float):
    """Return the second-order 1D periodic finite-difference Laplacian."""
    e = np.ones(num_points)
    lap = diags(
        diagonals=[e, -2.0 * e, e],
        offsets=[-1, 0, 1],
        shape=(num_points, num_points),
        format="lil",
    )
    lap[0, -1] = 1.0
    lap[-1, 0] = 1.0
    return (lap.tocsr()) / spacing**2


def build_periodic_laplacian_2d(num_points: int, spacing: float):
    """Return the five-point periodic Laplacian on an M x M square grid."""
    lap_1d = build_periodic_laplacian_1d(num_points, spacing)
    identity_1d = eye(num_points, format="csr")
    return (
        kron(identity_1d, lap_1d, format="csr")
        + kron(lap_1d, identity_1d, format="csr")
    )


LAPLACIAN_2D = build_periodic_laplacian_2d(M, DX)
IDENTITY = eye(NUM_DOF, format="csr")


# ============================================================
# Discrete diagnostics
# ============================================================

def periodic_forward_gradient(u: np.ndarray):
    """Return first-order forward periodic differences in x and y."""
    ux = (np.roll(u, -1, axis=0) - u) / DX
    uy = (np.roll(u, -1, axis=1) - u) / DY
    return ux, uy


def discrete_energy(u: np.ndarray) -> float:
    r"""Compute the discrete Ginzburg--Landau energy.

    E_h(u) = dx dy sum_ij [ 1/2 |grad_h u_ij|^2
                           + 1/4 (u_ij^2 - 1)^2 ].
    """
    ux, uy = periodic_forward_gradient(u)
    gradient_density = 0.5 * (ux**2 + uy**2)
    potential_density = 0.25 * (u**2 - 1.0) ** 2
    return CELL_AREA * np.sum(gradient_density + potential_density)


def discrete_interfacial_density(u: np.ndarray) -> float:
    r"""Compute the discrete interfacial density.

    rho_int,h(u) = (dx dy / |Omega|) sum_ij |grad_h u_ij|.
    """
    ux, uy = periodic_forward_gradient(u)
    grad_magnitude = np.sqrt(ux**2 + uy**2)
    return (CELL_AREA / DOMAIN_AREA) * np.sum(grad_magnitude)


# ============================================================
# Initial data
# ============================================================

def highly_oscillatory_initial_condition(
    num_points: int,
    seed: int = 0,
    bias: float = 0.5,
) -> np.ndarray:
    """Return the randomized highly oscillatory initial state used in the book.

    u_ij^0 = tanh(2 (xi_ij + bias)),
    where xi_ij are independent N(0,1) random variables.

    The positive bias favors eventual selection of the +1 phase while the
    random field supplies substantial fine-scale structure.
    """
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((num_points, num_points))
    return np.tanh(2.0 * (xi + bias))


# ============================================================
# Output helpers
# ============================================================

def save_snapshot(u: np.ndarray, step: int) -> None:
    """Save a phase-field snapshot using a fixed physical color scale."""
    time = step * DT

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(
        u,
        origin="lower",
        extent=(0.0, L, 0.0, L),
        cmap="twilight_shifted",
        interpolation="none",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"Allen--Cahn: step {step}, t={time:.4f}")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / f"5-allen_cahn-phi{step:05d}.png", dpi=300)
    plt.close(fig)


def print_diagnostics(step: int, u: np.ndarray, energy: float, rho_int: float) -> None:
    """Print one compact diagnostic line."""
    time = step * DT
    print(
        f"Step {step:5d}/{N_STEPS} | "
        f"t={time:8.4f} | "
        f"min={np.min(u): .6f} | "
        f"max={np.max(u): .6f} | "
        f"E_h={energy:.8e} | "
        f"rho_int,h={rho_int:.8e}"
    )


# ============================================================
# Main simulation
# ============================================================

def main() -> None:
    phi = highly_oscillatory_initial_condition(M, seed=SEED)

    energy_values = np.empty(N_STEPS + 1)
    interfacial_density_values = np.empty(N_STEPS + 1)
    minimum_values = np.empty(N_STEPS + 1)
    maximum_values = np.empty(N_STEPS + 1)

    energy_values[0] = discrete_energy(phi)
    interfacial_density_values[0] = discrete_interfacial_density(phi)
    minimum_values[0] = np.min(phi)
    maximum_values[0] = np.max(phi)

    print_diagnostics(
        0,
        phi,
        energy_values[0],
        interfacial_density_values[0],
    )
    save_snapshot(phi, 0)

    # Semi-implicit (IMEX Euler) step:
    #
    #   (I - dt Delta_h) u^{n+1}
    #       = (1 + dt) u^n - dt (u^n)^3.
    #
    # The left-hand matrix is constant and is factorized once.
    system_matrix = (IDENTITY - DT * LAPLACIAN_2D).tocsc()
    solve_linear_system = factorized(system_matrix)

    for step in range(1, N_STEPS + 1):
        rhs = ((1.0 + DT) * phi - DT * phi**3).ravel(order="C")
        phi = solve_linear_system(rhs).reshape((M, M), order="C")

        energy_values[step] = discrete_energy(phi)
        interfacial_density_values[step] = discrete_interfacial_density(phi)
        minimum_values[step] = np.min(phi)
        maximum_values[step] = np.max(phi)

        if step % 100 == 0:
            print_diagnostics(
                step,
                phi,
                energy_values[step],
                interfacial_density_values[step],
            )

        if step % SNAPSHOT_STRIDE == 0:
            save_snapshot(phi, step)

    # --------------------------------------------------------
    # Save numerical diagnostics
    # --------------------------------------------------------

    times = DT * np.arange(N_STEPS + 1)

    np.savez_compressed(
        DATA_DIR / "5-allen_cahn-diagnostics.npz",
        time=times,
        energy=energy_values,
        interfacial_density=interfacial_density_values,
        minimum=minimum_values,
        maximum=maximum_values,
    )

    # --------------------------------------------------------
    # Plot: maximum and minimum over the full simulation
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(times, maximum_values, label="max")
    ax.plot(times, minimum_values, label="min")
    ax.set_xlabel("Time")
    ax.set_ylabel("Solution value")
    ax.set_title("Allen--Cahn maximum and minimum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA_DIR / "5-allen_cahn-maxmin.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------
    # Plot: early-time energy decay
    # --------------------------------------------------------

    last = min(DIAGNOSTIC_STEPS, N_STEPS)
    early_times = times[: last + 1]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(early_times, energy_values[: last + 1])
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$E_h(u^n)$")
    ax.set_title(f"Discrete energy: first {last} time steps")
    fig.tight_layout()
    fig.savefig(DATA_DIR / "5-allen_cahn-energy_first20.png", dpi=300)
    plt.close(fig)

    # --------------------------------------------------------
    # Plot: early-time discrete interfacial density
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(early_times, interfacial_density_values[: last + 1])
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\rho_{\mathrm{int},h}(u^n)$")
    ax.set_title(f"Discrete interfacial density: first {last} time steps")
    fig.tight_layout()
    fig.savefig(DATA_DIR / "5-allen_cahn-interface_first20.png", dpi=300)
    plt.close(fig)

    print("Finished.")


if __name__ == "__main__":
    main()
