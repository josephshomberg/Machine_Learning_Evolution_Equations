"""
Forward Euler Simulation of a Cancer-Immune Interaction Model

This example is based on Problem 3 of Chapter 13 of

    C.-S. Chou and A. Friedman,
    Introduction to Mathematical Biology:
    Modeling, Analysis, and Simulations,
    Springer, 2010.

The cancer-immune interaction model is approximated using the
forward Euler method.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Cancer-immune model
# ============================================================

def cancer_immune_rhs(
    C,
    M1,
    M2,
    T,
    lambda_c,
    mu_c,
    C_0,
    mu,
    k_1,
    gamma,
    mu_T,
    k_T,
    K_1,
    K_2,
):
    """
    Evaluate the right-hand side of the cancer-immune system.

    The model is

        dC/dt  = lambda_c C (1 - C/C_0) - mu_c T C,

        dM1/dt = k_1
                 - gamma M1 C / (K_1 + C)
                 - mu M1,

        dM2/dt = gamma M1 C / (K_1 + C)
                 - mu M2,

        dT/dt  = k_T M1 / (K_2 + M2)
                 - mu_T T.

    Returns
    -------
    dC, dM1, dM2, dT : float
        Time derivatives of the four state variables.
    """

    m1_to_m2 = gamma * M1 * C / (K_1 + C)

    dC = lambda_c * C * (1.0 - C / C_0) - mu_c * T * C

    dM1 = k_1 - m1_to_m2 - mu * M1

    dM2 = m1_to_m2 - mu * M2

    dT = k_T * M1 / (K_2 + M2) - mu_T * T

    return dC, dM1, dM2, dT


# ============================================================
# Forward Euler simulation
# ============================================================

def simulate_cancer_immune_system(
    t0=0.0,
    tend=60.0,
    N=2**15,
    lambda_c=1.0e-2,
    mu_c=1.0e-5,
    C_0=1.0e6,
    mu=0.3,
    k_1=3000.0,
    gamma=200.0,
    mu_T=0.2,
    k_T=3300.0,
    K_1=None,
    K_2=1.0e5,
    C_init=1.0e2,
    M1_init=5.0e4,
    M2_init=0.0,
    T_init=0.0,
):
    """
    Simulate the cancer-immune model using the forward Euler method.

    For a system

        Y' = F(Y),

    the forward Euler update is

        Y^{n+1} = Y^n + delta_t F(Y^n).

    Parameters
    ----------
    t0, tend : float
        Initial and final times.
    N : int
        Number of time steps.

    Returns
    -------
    t : ndarray
        Time grid.
    C, M1, M2, T : ndarray
        Forward Euler approximations of the state variables.
    delta_t : float
        Time-step size.
    """

    if N <= 0:
        raise ValueError("N must be a positive integer.")

    if tend <= t0:
        raise ValueError("tend must be greater than t0.")

    if K_1 is None:
        K_1 = 0.05 * C_0

    delta_t = (tend - t0) / N
    t = np.linspace(t0, tend, N + 1)

    C = np.zeros(N + 1, dtype=float)
    M1 = np.zeros(N + 1, dtype=float)
    M2 = np.zeros(N + 1, dtype=float)
    T = np.zeros(N + 1, dtype=float)

    C[0] = C_init
    M1[0] = M1_init
    M2[0] = M2_init
    T[0] = T_init

    for n in range(N):

        dC, dM1, dM2, dT = cancer_immune_rhs(
            C[n],
            M1[n],
            M2[n],
            T[n],
            lambda_c,
            mu_c,
            C_0,
            mu,
            k_1,
            gamma,
            mu_T,
            k_T,
            K_1,
            K_2,
        )

        C[n + 1] = C[n] + delta_t * dC
        M1[n + 1] = M1[n] + delta_t * dM1
        M2[n + 1] = M2[n] + delta_t * dM2
        T[n + 1] = T[n] + delta_t * dT

    return t, C, M1, M2, T, delta_t


# ============================================================
# Individual figures
# ============================================================

def save_individual_plots(t, C, M1, M2, T, outdir="images"):
    """Save one figure for each state variable."""

    os.makedirs(outdir, exist_ok=True)

    series = [
        ("C", C, "Cancer cells $C(t)$"),
        ("M1", M1, "Pro-inflammatory macrophages $M_1(t)$"),
        ("M2", M2, "Anti-inflammatory macrophages $M_2(t)$"),
        ("T", T, "T-cells $T(t)$"),
    ]

    for filename, values, title in series:

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(t, values, linewidth=1.5)

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Density (cells/cm$^3$)")
        ax.set_title(title)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()

        fig.savefig(
            os.path.join(outdir, f"{filename}.png"),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# Four-panel figure
# ============================================================

def save_quad_plot(
    t,
    C,
    M1,
    M2,
    T,
    outdir="images",
    filename="cancer_immune_quad.png",
):
    """Save a 2-by-2 panel figure of the four state variables."""

    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    panels = [
        (axes[0, 0], C, r"(a) Cancer cells $C(t)$"),
        (axes[0, 1], M1, r"(b) Pro-inflammatory macrophages $M_1(t)$"),
        (axes[1, 0], M2, r"(c) Anti-inflammatory macrophages $M_2(t)$"),
        (axes[1, 1], T, r"(d) T-cells $T(t)$"),
    ]

    for ax, values, title in panels:

        ax.plot(t, values, linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Density (cells/cm$^3$)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Forward Euler Simulation of the Cancer-Immune System",
        fontsize=14,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.91)

    fig.savefig(
        os.path.join(outdir, filename),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main program
# ============================================================

def main():
    """Run the simulation and save the resulting figures."""

    t, C, M1, M2, T, delta_t = simulate_cancer_immune_system()

    print(f"Step size delta_t = {delta_t:.8f}")

    outdir = "images"

    save_individual_plots(
        t,
        C,
        M1,
        M2,
        T,
        outdir=outdir,
    )

    save_quad_plot(
        t,
        C,
        M1,
        M2,
        T,
        outdir=outdir,
    )

    print("Saved figures:")

    filenames = [
        "C.png",
        "M1.png",
        "M2.png",
        "T.png",
        "cancer_immune_quad.png",
    ]

    for filename in filenames:
        print(f"  {os.path.join(outdir, filename)}")


if __name__ == "__main__":
    main()
