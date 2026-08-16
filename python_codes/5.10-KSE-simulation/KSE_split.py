#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kuramoto--Sivashinsky equation on a one-dimensional periodic domain.

Equation
--------
    u_t + u u_x + u_xx + u_xxxx = 0,
    x in [0,L),   t in [0,T],

with periodic boundary conditions.

Equivalently,

    u_t = -u u_x - u_xx - u_xxxx.

Numerical method
----------------
A Fourier pseudo-spectral discretization is used in space.

The linear operator

    L(k) = k^2 - k^4

is treated implicitly, while the quadratic nonlinearity

    N(u) = -u u_x

is treated explicitly.  Thus,

    (1 - dt L(k)) u_hat^{n+1}
        = u_hat^n + dt N_hat(u^n).

This is a first-order semi-implicit Euler method.

The nonlinear term is evaluated in physical space and filtered with
the 2/3 de-aliasing rule before the Fourier update.

Diagnostics
-----------
The spatial mean is conserved by the periodic equation and is monitored
numerically.  The L2 norm is also tracked as a boundedness diagnostic;
unlike gradient-flow problems, it need not decay monotonically.

Outputs
-------
    5-KS_snapshots.png
    5-KS_spacetime.png
    5-KS_mean.png
    5-KS_l2.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters
# ============================================================

N = 256
L = 32.0 * np.pi

DT = 0.25
T = 300.0

SEED = 4
SPACETIME_STRIDE = 4
SNAPSHOT_TIMES = (0.0, 50.0, 150.0, 300.0)


# ============================================================
# Spatial grid and Fourier wave numbers
# ============================================================

dx = L / N

x = np.linspace(0.0, L, N, endpoint=False)
k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

linear_multiplier = k**2 - k**4

# 2/3 de-aliasing mask for the quadratic nonlinearity.
mode_numbers = np.fft.fftfreq(N) * N
DEALIAS_MASK = np.abs(mode_numbers) <= N / 3.0


# ============================================================
# Initial condition
# ============================================================

def initial_condition(
    x_grid: np.ndarray,
    seed: int = SEED,
) -> np.ndarray:
    """
    Smooth random periodic initial condition.

    The low Fourier modes are perturbed reproducibly with amplitudes
    decaying like m^{-2}.
    """
    rng = np.random.default_rng(seed)

    u0 = 0.2 * np.cos(2.0 * np.pi * x_grid / L)

    for m in range(1, 9):
        a = rng.standard_normal()
        b = rng.standard_normal()

        u0 += (
            0.15
            * (
                a * np.cos(2.0 * np.pi * m * x_grid / L)
                + b * np.sin(2.0 * np.pi * m * x_grid / L)
            )
            / m**2
        )

    return u0


# ============================================================
# Nonlinear term
# ============================================================

def nonlinear_hat(uhat: np.ndarray) -> np.ndarray:
    """
    Compute the nonlinear term in conservative form,

        N(u) = -1/2 (u^2)_x.

    For smooth solutions this is equivalent to -u u_x, but the
    conservative Fourier form preserves the zero Fourier mode exactly.

    The quadratic product is evaluated pseudo-spectrally and filtered
    with the 2/3 de-aliasing rule.
    """
    u = np.fft.ifft(uhat).real

    u2_hat = np.fft.fft(u**2)
    u2_hat *= DEALIAS_MASK

    nhat = -0.5j * k * u2_hat
    nhat[0] = 0.0

    return nhat


# ============================================================
# Diagnostics
# ============================================================

def discrete_mean(u: np.ndarray) -> float:
    """
    Discrete spatial mean.
    """
    return float(np.mean(u))


def discrete_l2(u: np.ndarray) -> float:
    """
    Discrete L2 norm

        ||u||_{L2,h}
        = sqrt(dx sum_j u_j^2).
    """
    return float(np.sqrt(dx * np.sum(u**2)))


# ============================================================
# One semi-implicit Euler step
# ============================================================

def time_step(uhat: np.ndarray) -> np.ndarray:
    """
    Advance one semi-implicit Euler step:

        (1 - dt L) u_hat^{n+1}
            = u_hat^n + dt N_hat(u^n),

    where

        L(k) = k^2 - k^4.
    """
    nhat = nonlinear_hat(uhat)

    denominator = 1.0 - DT * linear_multiplier

    if np.any(np.abs(denominator) < 1.0e-12):
        raise RuntimeError(
            "Semi-implicit denominator is too close to zero."
        )

    return (uhat + DT * nhat) / denominator


# ============================================================
# Time integration
# ============================================================

def solve_ks():
    """
    Integrate the Kuramoto--Sivashinsky equation from t=0 to t=T.
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

    u0 = initial_condition(x)
    uhat = np.fft.fft(u0)

    snapshots = {0: u0.copy()}

    space_time = [u0.copy()]
    time_values = [0.0]

    times = np.empty(num_steps + 1)
    means = np.empty(num_steps + 1)
    l2_values = np.empty(num_steps + 1)

    times[0] = 0.0
    means[0] = discrete_mean(u0)
    l2_values[0] = discrete_l2(u0)

    for n in range(1, num_steps + 1):
        uhat = time_step(uhat)

        if not np.all(np.isfinite(uhat)):
            raise FloatingPointError(
                f"Non-finite Fourier coefficients at step {n}."
            )

        u = np.fft.ifft(uhat).real
        t = n * DT

        times[n] = t
        means[n] = discrete_mean(u)
        l2_values[n] = discrete_l2(u)

        if n in snapshot_steps:
            snapshots[n] = u.copy()

        if n % SPACETIME_STRIDE == 0:
            space_time.append(u.copy())
            time_values.append(t)

    print("Kuramoto--Sivashinsky simulation complete")
    print(f"  grid points              : {N}")
    print(f"  dx                       : {dx:.8e}")
    print(f"  dt                       : {DT:.8e}")
    print(f"  final time               : {T:.8e}")
    print(f"  initial mean             : {means[0]:.12e}")
    print(f"  final mean               : {means[-1]:.12e}")
    print(
        f"  absolute mean drift      : "
        f"{abs(means[-1] - means[0]):.12e}"
    )
    print(
        f"  initial/final L2 norm    : "
        f"{l2_values[0]:.12e} / {l2_values[-1]:.12e}"
    )
    print(
        f"  maximum L2 norm          : "
        f"{np.max(l2_values):.12e}"
    )

    return (
        snapshots,
        np.asarray(space_time),
        np.asarray(time_values),
        times,
        means,
        l2_values,
    )


# ============================================================
# Plotting
# ============================================================

def plot_snapshots(
    snapshots: dict[int, np.ndarray],
) -> None:
    """
    Plot selected solution snapshots.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for t in SNAPSHOT_TIMES:
        n = int(round(t / DT))
        ax.plot(x, snapshots[n], label=rf"$t={t:g}$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")
    ax.set_title("Kuramoto--Sivashinsky equation: solution snapshots")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        "5-KS_snapshots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_spacetime(
    space_time: np.ndarray,
    time_values: np.ndarray,
) -> None:
    """
    Plot the spatiotemporal dynamics u(x,t).
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    image = ax.imshow(
        space_time,
        extent=[
            0.0,
            L,
            time_values[0],
            time_values[-1],
        ],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(
        "Kuramoto--Sivashinsky equation: spatiotemporal dynamics"
    )

    fig.colorbar(image, ax=ax, label=r"$u(x,t)$")

    fig.tight_layout()
    fig.savefig(
        "5-KS_spacetime.png",
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
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, values)
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
        space_time,
        time_values,
        times,
        means,
        l2_values,
    ) = solve_ks()

    plot_snapshots(snapshots)
    plot_spacetime(space_time, time_values)

    plot_diagnostic(
        times,
        means,
        r"$\overline{u}_h(t)$",
        "Kuramoto--Sivashinsky equation: mean conservation",
        "5-KS_mean.png",
    )

    plot_diagnostic(
        times,
        l2_values,
        r"$\|u_h(t)\|_{L^2}$",
        "Kuramoto--Sivashinsky equation: L2 diagnostic",
        "5-KS_l2.png",
    )


if __name__ == "__main__":
    main()