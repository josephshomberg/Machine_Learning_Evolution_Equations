#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Viscous Burgers equation on a periodic one-dimensional domain.

Equation
--------
    u_t + u u_x = nu u_xx,
    x in [0, L),   t in [0, T],

with periodic boundary conditions.

Numerical method
----------------
The equation is written in conservative form,

    u_t + (u^2/2)_x = nu u_xx.

A first-order Lie splitting is used at each time step:

    1. nonlinear transport:
           u_t + (u^2/2)_x = 0,

       discretized by a conservative local Lax--Friedrichs
       (Rusanov) finite-volume flux;

    2. diffusion:
           u_t = nu u_xx,

       advanced exactly in Fourier space.

The transport step satisfies the usual explicit CFL restriction

    dt * max_j |u_j| / dx <= CFL_MAX.

The Fourier diffusion step is unconditionally stable.

Diagnostics
-----------
For periodic viscous Burgers flow,

    mean(u(t)) = mean(u(0))

is conserved, while the quadratic energy

    E(t) = 1/2 int_0^L u(x,t)^2 dx

is nonincreasing in the continuum problem.  Both quantities are
tracked numerically.

Outputs
-------
    5-VBE_snapshots.png
    5-VBE_spacetime.png
    5-VBE_energy.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters
# ============================================================

M = 512
L = 2.0 * np.pi
NU = 0.03

T = 2.0
DT = 1.0e-3

SEED = 3
CFL_MAX = 0.90
SPACETIME_STRIDE = 5

SNAPSHOT_TIMES = (0.0, 0.4, 1.0, 2.0)


# ============================================================
# Spatial grid and Fourier wave numbers
# ============================================================

x = np.linspace(0.0, L, M, endpoint=False)
dx = L / M

k = 2.0 * np.pi * np.fft.fftfreq(M, d=dx)


# ============================================================
# Initial condition
# ============================================================

def initial_condition(x_grid: np.ndarray, seed: int = SEED) -> np.ndarray:
    """
    Smooth random periodic initial condition.

    The leading modes are deterministic, while higher modes receive
    small reproducible random perturbations.
    """
    rng = np.random.default_rng(seed)

    u0 = np.sin(x_grid) + 0.35 * np.sin(2.0 * x_grid)

    for m in range(3, 8):
        a = rng.standard_normal()
        b = rng.standard_normal()

        u0 += (
            0.12
            * (a * np.cos(m * x_grid) + b * np.sin(m * x_grid))
            / m**2
        )

    return u0


# ============================================================
# Burgers flux and transport step
# ============================================================

def flux(v: np.ndarray) -> np.ndarray:
    """
    Burgers flux

        f(u) = u^2 / 2.
    """
    return 0.5 * v**2


def transport_step(u: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """
    Conservative local Lax--Friedrichs (Rusanov) update for

        u_t + (u^2/2)_x = 0.

    The numerical flux at x_{j+1/2} is

        F_{j+1/2}
        = 1/2 [f(u_j) + f(u_{j+1})]
          - 1/2 alpha_{j+1/2} (u_{j+1} - u_j),

    where

        alpha_{j+1/2} = max(|u_j|, |u_{j+1}|).

    Periodicity is imposed with np.roll.
    """
    u_right = np.roll(u, -1)
    u_left = np.roll(u, 1)

    alpha_right = np.maximum(np.abs(u), np.abs(u_right))
    alpha_left = np.maximum(np.abs(u_left), np.abs(u))

    flux_right = (
        0.5 * (flux(u) + flux(u_right))
        - 0.5 * alpha_right * (u_right - u)
    )

    flux_left = (
        0.5 * (flux(u_left) + flux(u))
        - 0.5 * alpha_left * (u - u_left)
    )

    return u - (dt / dx) * (flux_right - flux_left)


# ============================================================
# Exact Fourier diffusion step
# ============================================================

def diffusion_step(
    u: np.ndarray,
    dt: float,
    nu: float,
    wave_numbers: np.ndarray,
) -> np.ndarray:
    """
    Exact Fourier update for the heat equation

        u_t = nu u_xx.

    If u_hat(k,t) denotes the Fourier coefficient, then

        u_hat(k,t+dt)
        = exp(-nu k^2 dt) u_hat(k,t).
    """
    u_hat = np.fft.fft(u)
    heat_factor = np.exp(-nu * wave_numbers**2 * dt)
    u_hat *= heat_factor

    return np.fft.ifft(u_hat).real


# ============================================================
# Diagnostics
# ============================================================

def discrete_mean(u: np.ndarray) -> float:
    """
    Discrete spatial mean.
    """
    return float(np.mean(u))


def discrete_energy(u: np.ndarray, dx: float) -> float:
    """
    Discrete quadratic energy

        E_h(u) = (dx/2) sum_j u_j^2.
    """
    return float(0.5 * dx * np.sum(u**2))


def cfl_number(u: np.ndarray, dt: float, dx: float) -> float:
    """
    Explicit transport CFL number

        CFL = dt * max_j |u_j| / dx.
    """
    return float(dt * np.max(np.abs(u)) / dx)


# ============================================================
# Time integration
# ============================================================

def solve_burgers() -> tuple[
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Advance the viscous Burgers equation to time T.

    Returns
    -------
    snapshots
        Dictionary keyed by time-step index.
    space_time
        Stored solution rows for the spatiotemporal plot.
    time_values
        Times corresponding to space_time.
    diagnostic_times
        Times at every numerical step.
    energy_values
        Discrete quadratic energy at every numerical step.
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

    snapshots: dict[int, np.ndarray] = {0: u.copy()}

    space_time = [u.copy()]
    time_values = [0.0]

    diagnostic_times = [0.0]
    energy_values = [discrete_energy(u, dx)]

    initial_mean = discrete_mean(u)

    for n in range(1, num_steps + 1):
        cfl = cfl_number(u, DT, dx)

        if cfl > CFL_MAX:
            raise RuntimeError(
                f"CFL condition violated at step {n}: "
                f"CFL={cfl:.6f} > CFL_MAX={CFL_MAX:.6f}. "
                "Reduce DT or increase spatial resolution."
            )

        # First-order Lie splitting:
        # nonlinear transport followed by viscous diffusion.
        u = transport_step(u, DT, dx)
        u = diffusion_step(u, DT, NU, k)

        t = n * DT

        if n in snapshot_steps:
            snapshots[n] = u.copy()

        if n % SPACETIME_STRIDE == 0:
            space_time.append(u.copy())
            time_values.append(t)

        diagnostic_times.append(t)
        energy_values.append(discrete_energy(u, dx))

    final_mean = discrete_mean(u)

    print("Viscous Burgers simulation complete")
    print(f"  grid points              : {M}")
    print(f"  dx                       : {dx:.8e}")
    print(f"  dt                       : {DT:.8e}")
    print(f"  viscosity                : {NU:.8e}")
    print(f"  final time               : {T:.8e}")
    print(f"  initial mean             : {initial_mean:.12e}")
    print(f"  final mean               : {final_mean:.12e}")
    print(
        f"  absolute mean drift      : "
        f"{abs(final_mean - initial_mean):.12e}"
    )
    print(
        f"  initial/final energy     : "
        f"{energy_values[0]:.12e} / {energy_values[-1]:.12e}"
    )

    return (
        snapshots,
        np.asarray(space_time),
        np.asarray(time_values),
        np.asarray(diagnostic_times),
        np.asarray(energy_values),
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
    plt.figure(figsize=(11, 6))

    for t in SNAPSHOT_TIMES:
        step = int(round(t / DT))
        plt.plot(x, snapshots[step], label=rf"$t={t:g}$")

    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$")
    plt.title("Viscous Burgers equation: solution snapshots")
    plt.legend()
    plt.tight_layout()
    plt.savefig("5-VBE_snapshots.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_spacetime(
    space_time: np.ndarray,
    time_values: np.ndarray,
) -> None:
    """
    Plot the spatiotemporal evolution u(x,t).
    """
    plt.figure(figsize=(11, 6))

    image = plt.imshow(
        space_time,
        extent=[0.0, L, time_values[0], time_values[-1]],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.title("Viscous Burgers equation: spatiotemporal evolution")
    plt.colorbar(image, label=r"$u(x,t)$")
    plt.tight_layout()
    plt.savefig("5-VBE_spacetime.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_energy(
    diagnostic_times: np.ndarray,
    energy_values: np.ndarray,
) -> None:
    """
    Plot the discrete quadratic energy.
    """
    plt.figure(figsize=(11, 6))

    plt.plot(diagnostic_times, energy_values)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E_h(t)$")
    plt.title("Viscous Burgers equation: discrete energy decay")
    plt.tight_layout()
    plt.savefig("5-VBE_energy.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    (
        snapshots,
        space_time,
        time_values,
        diagnostic_times,
        energy_values,
    ) = solve_burgers()

    plot_snapshots(snapshots)
    plot_spacetime(space_time, time_values)
    plot_energy(diagnostic_times, energy_values)


if __name__ == "__main__":
    main()