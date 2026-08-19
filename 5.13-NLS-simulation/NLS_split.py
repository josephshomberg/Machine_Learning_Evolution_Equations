#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-dimensional cubic nonlinear Schrodinger equation on a periodic domain.

Equation
--------
    i u_t + u_xx + kappa |u|^2 u = 0,
    x in [-L/2, L/2),   t in [0,T],

with periodic boundary conditions.

Equivalently,

    u_t = i u_xx + i kappa |u|^2 u.

Numerical method
----------------
A Fourier spectral discretization is used in space together with
second-order Strang splitting in time.

Each full time step consists of

    1. a half linear dispersive step,
    2. a full nonlinear phase step,
    3. a second half linear dispersive step.

For the linear subproblem

    i u_t + u_xx = 0,

the Fourier coefficients satisfy

    u_hat(t + tau, k)
        = exp(-i k^2 tau) u_hat(t,k).

For the nonlinear subproblem

    i u_t + kappa |u|^2 u = 0,

the amplitude is constant and the exact phase update is

    u(t + tau,x)
        = exp(i kappa |u(t,x)|^2 tau) u(t,x).

Both substeps are therefore solved exactly.

Diagnostics
-----------
The code tracks the discrete mass

    M_h(t) = dx sum_j |u_j|^2,

and the discrete Hamiltonian

    H_h(t)
        = dx sum_j [ |u_x|^2 - (kappa/2)|u|^4 ].

For the cubic periodic NLS, both are conserved by the continuum flow.
Strang splitting preserves mass to roundoff because each subflow is unitary;
the Hamiltonian is not preserved exactly by the splitting but should remain
nearly constant for a sufficiently small time step.

Outputs
-------
    5-NLS_t0.png
    5-NLS_t1.png
    ...
    5-NLS_t8.png
    5-NLS_timelapse.png
    5-NLS_mass.png
    5-NLS_hamiltonian.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters
# ============================================================

N = 1024
L = 40.0

KAPPA = 1.0

DT = 0.002
T = 8.0

SNAPSHOT_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
DIAGNOSTIC_STRIDE = 20


# ============================================================
# Spatial grid and Fourier wave numbers
# ============================================================

dx = L / N

x = np.linspace(-L / 2.0, L / 2.0, N, endpoint=False)

k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)


# ============================================================
# Initial condition
# ============================================================

def initial_condition(x_grid: np.ndarray) -> np.ndarray:
    """
    Two localized complex wave packets with different carrier wave numbers.
    """
    A1 = 1.0
    A2 = 0.9

    x0 = 7.0
    sigma = 1.4

    c1 = 1.5
    c2 = -1.2

    return (
        A1
        * np.exp(-((x_grid + x0) ** 2) / (2.0 * sigma**2))
        * np.exp(1j * c1 * x_grid)
        +
        A2
        * np.exp(-((x_grid - x0) ** 2) / (2.0 * sigma**2))
        * np.exp(1j * c2 * x_grid)
    )


# ============================================================
# Exact split subflows
# ============================================================

LINEAR_HALF = np.exp(-0.5j * k**2 * DT)


def linear_half_step(u: np.ndarray) -> np.ndarray:
    """
    Exact half-step for the linear dispersive subproblem

        i u_t + u_xx = 0.
    """
    u_hat = np.fft.fft(u)
    u_hat *= LINEAR_HALF
    return np.fft.ifft(u_hat)


def nonlinear_full_step(u: np.ndarray) -> np.ndarray:
    """
    Exact full step for the nonlinear phase subproblem

        i u_t + kappa |u|^2 u = 0.
    """
    return np.exp(1j * KAPPA * np.abs(u) ** 2 * DT) * u


def strang_step(u: np.ndarray) -> np.ndarray:
    """
    Advance one second-order Strang-splitting step.
    """
    u = linear_half_step(u)
    u = nonlinear_full_step(u)
    u = linear_half_step(u)
    return u


# ============================================================
# Diagnostics
# ============================================================

def mass(u: np.ndarray) -> float:
    """
    Discrete mass

        M_h = dx sum_j |u_j|^2.
    """
    return float(dx * np.sum(np.abs(u) ** 2))


def hamiltonian(u: np.ndarray) -> float:
    """
    Discrete Hamiltonian

        H_h
        = dx sum_j [ |u_x|^2 - (kappa/2)|u|^4 ].
    """
    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat)

    density = np.abs(ux) ** 2 - 0.5 * KAPPA * np.abs(u) ** 4

    return float(dx * np.sum(density))


# ============================================================
# Time integration
# ============================================================

def solve_nls():
    """
    Integrate the nonlinear Schrodinger equation from t=0 to t=T.
    """
    num_steps = int(round(T / DT))

    if not np.isclose(num_steps * DT, T):
        raise ValueError("T must be an integer multiple of DT.")

    snapshot_steps = {
        int(round(t / DT)): t for t in SNAPSHOT_TIMES
    }

    for step, t in snapshot_steps.items():
        if not np.isclose(step * DT, t):
            raise ValueError(
                f"Snapshot time {t} is not an integer multiple of DT."
            )

    u = initial_condition(x)

    snapshots = {0: np.abs(u) ** 2}

    diagnostic_times = [0.0]
    mass_values = [mass(u)]
    hamiltonian_values = [hamiltonian(u)]

    for n in range(1, num_steps + 1):
        u = strang_step(u)

        if not np.all(np.isfinite(u)):
            raise FloatingPointError(
                f"Non-finite solution values at step {n}."
            )

        if n in snapshot_steps:
            snapshots[n] = np.abs(u) ** 2

        if n % DIAGNOSTIC_STRIDE == 0:
            diagnostic_times.append(n * DT)
            mass_values.append(mass(u))
            hamiltonian_values.append(hamiltonian(u))

    mass_values = np.asarray(mass_values)
    hamiltonian_values = np.asarray(hamiltonian_values)
    diagnostic_times = np.asarray(diagnostic_times)

    mass_drift = np.max(np.abs(mass_values - mass_values[0]))
    hamiltonian_drift = np.max(
        np.abs(hamiltonian_values - hamiltonian_values[0])
    )

    print("Nonlinear Schrodinger simulation complete")
    print(f"  grid points              : {N}")
    print(f"  dx                       : {dx:.8e}")
    print(f"  dt                       : {DT:.8e}")
    print(f"  kappa                    : {KAPPA:.8e}")
    print(f"  final time               : {T:.8e}")
    print(
        f"  initial/final mass       : "
        f"{mass_values[0]:.12e} / {mass_values[-1]:.12e}"
    )
    print(
        f"  maximum mass drift       : "
        f"{mass_drift:.12e}"
    )
    print(
        f"  initial/final Hamiltonian: "
        f"{hamiltonian_values[0]:.12e} / "
        f"{hamiltonian_values[-1]:.12e}"
    )
    print(
        f"  maximum Hamiltonian drift: "
        f"{hamiltonian_drift:.12e}"
    )

    return (
        snapshots,
        diagnostic_times,
        mass_values,
        hamiltonian_values,
    )


# ============================================================
# Plotting
# ============================================================

def save_individual_snapshots(
    snapshots: dict[int, np.ndarray],
) -> None:
    """
    Save individual intensity snapshots with one common vertical scale.
    """
    snapshot_steps = [
        int(round(t / DT)) for t in SNAPSHOT_TIMES
    ]

    rho_max = max(
        np.max(snapshots[n]) for n in snapshot_steps
    )

    for t, n in zip(SNAPSHOT_TIMES, snapshot_steps):
        fig, ax = plt.subplots(figsize=(5, 3))

        ax.plot(x, snapshots[n], linewidth=2.0)

        ax.set_xlim(-L / 2.0, L / 2.0)
        ax.set_ylim(0.0, 1.05 * rho_max)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|u(x,t)|^2$")
        ax.set_title(rf"NLS intensity, $t={t:g}$")

        fig.tight_layout()
        fig.savefig(
            f"5-NLS_t{int(t)}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_timelapse(
    snapshots: dict[int, np.ndarray],
) -> None:
    """
    Save a 2 x 4 intensity montage.
    """
    snapshot_steps = [
        int(round(t / DT)) for t in SNAPSHOT_TIMES
    ]

    rho_max = max(
        np.max(snapshots[n]) for n in snapshot_steps
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 6),
        constrained_layout=True,
    )

    for ax, t, n in zip(
        axes.ravel(),
        SNAPSHOT_TIMES,
        snapshot_steps,
    ):
        ax.plot(x, snapshots[n], linewidth=1.8)

        ax.set_xlim(-L / 2.0, L / 2.0)
        ax.set_ylim(0.0, 1.05 * rho_max)

        ax.set_title(rf"$t={t:g}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|u|^2$")

    fig.suptitle("Nonlinear Schrodinger equation: intensity time-lapse")

    fig.savefig(
        "5-NLS_timelapse.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_diagnostic(
    times: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """
    Save a scalar time-series diagnostic.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(times, values, linewidth=2.0)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    (
        snapshots,
        diagnostic_times,
        mass_values,
        hamiltonian_values,
    ) = solve_nls()

    save_individual_snapshots(snapshots)
    plot_timelapse(snapshots)

    plot_diagnostic(
        diagnostic_times,
        mass_values,
        r"$M_h(t)$",
        "Nonlinear Schrodinger equation: mass conservation",
        "5-NLS_mass.png",
    )

    plot_diagnostic(
        diagnostic_times,
        hamiltonian_values,
        r"$H_h(t)$",
        "Nonlinear Schrodinger equation: Hamiltonian diagnostic",
        "5-NLS_hamiltonian.png",
    )


if __name__ == "__main__":
    main()