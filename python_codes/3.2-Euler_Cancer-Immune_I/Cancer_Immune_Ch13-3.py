"""
Simulation of Problem 3 Chapter 13 of Introduction to Mathematical Biology --
Modeling, Analysis, and Simulations by Ching-Shan Chou and Avner Friedman,
Springer, 2010.

This script simulates the cancer-immune interaction model using the forward
Euler method.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def simulate_cancer_immune_system(
    t0=0.0,
    tend=60.0,
    N=2**15,
    lambda_c=1.0e-2,  # Cancer growth rate (/day)
    mu_c=1.0e-5,  # Cancer death rate due to T-cells (/cell/day)
    C_0=1.0e6,  # Cancer carrying capacity (cells/cm^3)
    mu=0.3,  # Macrophage natural death rate (/day)
    k_1=3000.0,  # M1 macrophage production rate (cells/cm^3/day)
    gamma=200.0,  # M1 to M2 differentiation rate (/day)
    mu_T=0.2,  # T-cell natural death rate (/day)
    k_T=3300.0,  # T-cell recruitment rate by M1 macrophages (/cell/day)
    K_1=None,  # Michaelis-Menten constant for macrophages (cells/cm^3)
    K_2=1.0e5,  # Michaelis-Menten constant for T-cells (cells/cm^3)
    C_init=1.0e2,  # Initial cancer cell density (cells/cm^3)
    M1_init=5.0e4,  # Initial M1 macrophage density (cells/cm^3)
    M2_init=0.0,  # Initial M2 macrophage density (cells/cm^3)
    T_init=0.0,  # Initial T-cell density (cells/cm^3)
):
    """Simulate the cancer-immune interaction model by the forward Euler method.

    The model equations are:
        dC/dt  = lambda_c * C * (1 - C/C_0) - mu_c * T * C
        dM1/dt = k_1 - gamma * M1 * C / (K_1 + C) - mu * M1
        dM2/dt = gamma * M1 * C / (K_1 + C) - mu * M2
        dT/dt  = k_T * M1 / (K_2 + M2) - mu_T * T

    Returns:
        t : ndarray
            Time grid.
        C, M1, M2, T : ndarray
            Numerical approximations of the state variables.
        delta_t : float
            Time-step size.
    """
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
        # Calculate shared Michaelis-Menten activation factor
        mac_switch = gamma * M1[n] * C[n] / (K_1 + C[n])

        C[n + 1] = C[n] + delta_t * (
            lambda_c * C[n] * (1.0 - C[n] / C_0) - mu_c * T[n] * C[n]
        )
        M1[n + 1] = M1[n] + delta_t * (k_1 - mac_switch - mu * M1[n])
        M2[n + 1] = M2[n] + delta_t * (mac_switch - mu * M2[n])
        T[n + 1] = T[n] + delta_t * (
            k_T * M1[n] / (K_2 + M2[n]) - mu_T * T[n]
        )

    return t, C, M1, M2, T, delta_t


def save_individual_plots(t, C, M1, M2, T, outdir="images"):
    """Save one high-resolution figure for each state variable."""
    os.makedirs(outdir, exist_ok=True)
    series = [
        ("C", C, "Cancer cells $C(t)$"),
        ("M1", M1, "Pro-inflammatory macrophages $M_1(t)$"),
        ("M2", M2, "Anti-inflammatory macrophages $M_2(t)$"),
        ("T", T, "T-cells $T(t)$"),
    ]

    for filename, values, title in series:
        plt.figure(figsize=(6, 4))
        plt.plot(t, values, linewidth=1.5, color="black")  # Book style clear line
        plt.xlabel("Time (days)")
        plt.ylabel("Density (cells/cm$^3$)")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"{filename}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def save_quad_plot(
    t, C, M1, M2, T, outdir="images", filename="cancer_immune_quad.png"
):
    """Save a 2x2 panel figure with separate scales for each variable."""
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
        "Forward Euler Simulation of the Cancer-Immune System", fontsize=14
    )
    fig.tight_layout()
    # Adjust top spacing safely to prevent overlap with suptitle
    fig.subplots_adjust(top=0.91)

    fig.savefig(
        os.path.join(outdir, filename),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    t, C, M1, M2, T, delta_t = simulate_cancer_immune_system()
    print(f"Step size delta_t = {delta_t:.8f}")

    out_folder = "images"
    save_individual_plots(t, C, M1, M2, T, outdir=out_folder)
    save_quad_plot(
        t, C, M1, M2, T, outdir=out_folder, filename="cancer_immune_quad.png"
    )

    print("Saved figures:")
    for name in ["C.png", "M1.png", "M2.png", "T.png", "cancer_immune_quad.png"]:
        print(f"  {os.path.join(out_folder, name)}")


if __name__ == "__main__":
    main()
