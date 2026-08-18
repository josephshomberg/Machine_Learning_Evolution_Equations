#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-dimensional thin-film / dewetting equation with degenerate mobility.

Equation
--------
    u_t = div( M(u) grad mu ),

    mu = -eps^2 Delta u + f(u),

    f(u) = u^3 - u,

    M(u) = max(u, u_mob)^m,

on the periodic square

    Omega = [0,L) x [0,L).

The mobility floor u_mob > 0 is used only in the mobility evaluation to
avoid numerical degeneracy when u becomes very small.  The solution itself
is not clipped after each time step, so the conservative structure is not
artificially altered by a positivity projection.

Numerical method
----------------
A Fourier pseudo-spectral method is used in space.  The nonlinear fluxes are
formed in physical space and filtered with the 2/3 de-aliasing rule.

To obtain a practical semi-implicit scheme, write

    M(u^n) = Mbar^n + (M(u^n) - Mbar^n),

where Mbar^n is the spatial mean mobility.  Then

    u_t
      = -Mbar^n eps^2 Delta^2 u
        -eps^2 div[(M-Mbar^n) grad(Delta u)]
        +div[M grad f(u)].

The constant-coefficient fourth-order part is treated implicitly and the
remaining variable-mobility terms are treated explicitly:

    (1 + dt Mbar^n eps^2 |k|^4) u_hat^{n+1}
      =
      u_hat^n + dt R_hat^n.

This is a first-order semi-implicit Fourier scheme.

Diagnostics
-----------
The periodic divergence form conserves the spatial mean (mass),

    M_h(t) = dx^2 sum_{i,j} u_ij,

up to roundoff for the unprojected spectral update.

The discrete free energy

    E_h(u)
      = int_Omega [
            eps^2/2 |grad u|^2
            + 1/4 (u^2 - 1)^2
        ] dx

is monitored as a qualitative dissipation diagnostic.  Because the
variable-mobility remainder is treated explicitly, strict step-by-step
energy monotonicity is not guaranteed by this first-order splitting.

Outputs
-------
    5-TF_t0.png
    5-TF_t1.png
    ...
    5-TF_t8.png
    5-TF_timelapse.png
    5-TF_mass.png
    5-TF_energy.png
    5-TF_minmax.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters
# ============================================================

N = 128
L = 1.0

EPS = 0.018
MOBILITY_EXPONENT = 3.0
MOBILITY_FLOOR = 0.02

DT = 5.0e-5
T = 0.25

SEED = 7
DIAGNOSTIC_STRIDE = 20

SNAPSHOT_TIMES = (0.0, 0.03, 0.06, 0.09, 0.13, 0.17, 0.21, 0.25)
SNAPSHOT_LABELS = (0, 1, 2, 3, 4, 5, 6, 8)


# ============================================================
# Grid and Fourier wave numbers
# ============================================================

dx = L / N

x = np.linspace(0.0, L, N, endpoint=False)
y = np.linspace(0.0, L, N, endpoint=False)

X, Y = np.meshgrid(x, y, indexing="ij")

kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

KX, KY = np.meshgrid(kx, ky, indexing="ij")

K2 = KX**2 + KY**2
K4 = K2**2

# 2/3 de-aliasing mask.
mode_numbers = np.fft.fftfreq(N) * N
MX, MY = np.meshgrid(mode_numbers, mode_numbers, indexing="ij")
DEALIAS_MASK = (
    (np.abs(MX) <= N / 3.0)
    & (np.abs(MY) <= N / 3.0)
)


# ============================================================
# Initial condition
# ============================================================

def initial_condition() -> np.ndarray:
    """
    Smooth random perturbation of a nearly flat film.
    """
    rng = np.random.default_rng(SEED)
    eta = rng.standard_normal((N, N))

    # Periodic nearest-neighbor smoothing.
    for _ in range(12):
        eta = (
            eta
            + np.roll(eta, 1, axis=0)
            + np.roll(eta, -1, axis=0)
            + np.roll(eta, 1, axis=1)
            + np.roll(eta, -1, axis=1)
        ) / 5.0

    eta /= np.max(np.abs(eta))

    u0 = 0.55 + 0.12 * eta

    return u0


# ============================================================
# Fourier differential operators
# ============================================================

def ddx(f: np.ndarray) -> np.ndarray:
    """
    Spectral derivative with respect to x.
    """
    return np.fft.ifft2(1j * KX * np.fft.fft2(f)).real


def ddy(f: np.ndarray) -> np.ndarray:
    """
    Spectral derivative with respect to y.
    """
    return np.fft.ifft2(1j * KY * np.fft.fft2(f)).real


def laplacian(f: np.ndarray) -> np.ndarray:
    """
    Spectral Laplacian.
    """
    return np.fft.ifft2(-K2 * np.fft.fft2(f)).real


def divergence(fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """
    Spectral divergence of a vector field.
    """
    return ddx(fx) + ddy(fy)


def dealias(f: np.ndarray) -> np.ndarray:
    """
    Apply the 2/3 Fourier de-aliasing rule.
    """
    fhat = np.fft.fft2(f)
    fhat *= DEALIAS_MASK
    return np.fft.ifft2(fhat).real


# ============================================================
# Model functions
# ============================================================

def bulk_derivative(u: np.ndarray) -> np.ndarray:
    """
    Derivative of the double-well potential

        F(u) = 1/4 (u^2 - 1)^2,

    namely

        F'(u) = u^3 - u.
    """
    return u**3 - u


def mobility(u: np.ndarray) -> np.ndarray:
    """
    Regularized degenerate mobility

        M(u) = max(u, MOBILITY_FLOOR)^m.
    """
    return np.maximum(u, MOBILITY_FLOOR) ** MOBILITY_EXPONENT


# ============================================================
# Semi-implicit variable-mobility step
# ============================================================

def explicit_remainder(u: np.ndarray, mobility_field: np.ndarray) -> np.ndarray:
    """
    Explicit variable-mobility remainder

        R(u)
          =
          -eps^2 div[(M-Mbar) grad(Delta u)]
          +div[M grad f(u)].

    Nonlinear flux products are de-aliased before differentiation.
    """
    mbar = float(np.mean(mobility_field))
    mdev = mobility_field - mbar

    lap_u = laplacian(u)

    lap_ux = ddx(lap_u)
    lap_uy = ddy(lap_u)

    f = bulk_derivative(u)
    fx = ddx(f)
    fy = ddy(f)

    flux4_x = dealias(mdev * lap_ux)
    flux4_y = dealias(mdev * lap_uy)

    flux2_x = dealias(mobility_field * fx)
    flux2_y = dealias(mobility_field * fy)

    return (
        -EPS**2 * divergence(flux4_x, flux4_y)
        + divergence(flux2_x, flux2_y)
    )


def time_step(u: np.ndarray) -> np.ndarray:
    """
    Advance one first-order semi-implicit Fourier step.
    """
    m = mobility(u)
    mbar = float(np.mean(m))

    remainder = explicit_remainder(u, m)
    remainder_hat = np.fft.fft2(remainder)

    uhat = np.fft.fft2(u)

    denominator = 1.0 + DT * mbar * EPS**2 * K4

    uhat_new = (uhat + DT * remainder_hat) / denominator

    # The zero Fourier mode is preserved exactly by the divergence-form RHS.
    uhat_new[0, 0] = uhat[0, 0]

    return np.fft.ifft2(uhat_new).real


# ============================================================
# Diagnostics
# ============================================================

def total_mass(u: np.ndarray) -> float:
    """
    Discrete total mass

        M_h = dx^2 sum_{i,j} u_ij.
    """
    return float(dx**2 * np.sum(u))


def free_energy(u: np.ndarray) -> float:
    """
    Discrete free energy

        E_h
          = dx^2 sum [
                eps^2/2 |grad u|^2
                + 1/4 (u^2 - 1)^2
            ].
    """
    ux = ddx(u)
    uy = ddy(u)

    density = (
        0.5 * EPS**2 * (ux**2 + uy**2)
        + 0.25 * (u**2 - 1.0) ** 2
    )

    return float(dx**2 * np.sum(density))


# ============================================================
# Time integration
# ============================================================

def solve_thin_film():
    """
    Integrate the thin-film / dewetting equation from t=0 to t=T.
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

    u = initial_condition()

    snapshots = {0: u.copy()}

    diagnostic_times = [0.0]
    mass_values = [total_mass(u)]
    energy_values = [free_energy(u)]
    min_values = [float(np.min(u))]
    max_values = [float(np.max(u))]

    for step in range(1, num_steps + 1):
        u = time_step(u)

        if not np.all(np.isfinite(u)):
            raise FloatingPointError(
                f"Non-finite solution values at step {step}."
            )

        if step in snapshot_steps:
            snapshots[step] = u.copy()

        if step % DIAGNOSTIC_STRIDE == 0:
            diagnostic_times.append(step * DT)
            mass_values.append(total_mass(u))
            energy_values.append(free_energy(u))
            min_values.append(float(np.min(u)))
            max_values.append(float(np.max(u)))

    diagnostic_times = np.asarray(diagnostic_times)
    mass_values = np.asarray(mass_values)
    energy_values = np.asarray(energy_values)
    min_values = np.asarray(min_values)
    max_values = np.asarray(max_values)

    mass_drift = np.max(np.abs(mass_values - mass_values[0]))

    print("Thin-film / dewetting simulation complete")
    print(f"  grid                     : {N} x {N}")
    print(f"  dx                       : {dx:.8e}")
    print(f"  dt                       : {DT:.8e}")
    print(f"  epsilon                  : {EPS:.8e}")
    print(f"  mobility exponent        : {MOBILITY_EXPONENT:.8e}")
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
        f"  initial/final energy     : "
        f"{energy_values[0]:.12e} / {energy_values[-1]:.12e}"
    )
    print(
        f"  minimum film thickness   : "
        f"{np.min(min_values):.12e}"
    )
    print(
        f"  maximum film thickness   : "
        f"{np.max(max_values):.12e}"
    )

    return (
        snapshots,
        diagnostic_times,
        mass_values,
        energy_values,
        min_values,
        max_values,
    )


# ============================================================
# Plotting
# ============================================================

def snapshot_color_limits(
    snapshots: dict[int, np.ndarray],
) -> tuple[float, float]:
    """
    Common color limits for all saved snapshots.
    """
    snapshot_steps = [
        int(round(t / DT)) for t in SNAPSHOT_TIMES
    ]

    umin = min(np.min(snapshots[n]) for n in snapshot_steps)
    umax = max(np.max(snapshots[n]) for n in snapshot_steps)

    return float(umin), float(umax)


def save_individual_snapshots(
    snapshots: dict[int, np.ndarray],
) -> None:
    """
    Save individual film-thickness images for LaTeX.
    """
    snapshot_steps = [
        int(round(t / DT)) for t in SNAPSHOT_TIMES
    ]

    umin, umax = snapshot_color_limits(snapshots)

    for label, step in zip(
        SNAPSHOT_LABELS,
        snapshot_steps,
    ):
        fig, ax = plt.subplots(figsize=(4, 4))

        ax.imshow(
            snapshots[step].T,
            extent=[0.0, L, 0.0, L],
            origin="lower",
            cmap="viridis",
            vmin=umin,
            vmax=umax,
            interpolation="nearest",
        )

        ax.axis("off")

        fig.savefig(
            f"5-TF_t{label}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)


def plot_timelapse(
    snapshots: dict[int, np.ndarray],
) -> None:
    """
    Save the 2 x 4 thin-film time-lapse montage.
    """
    snapshot_steps = [
        int(round(t / DT)) for t in SNAPSHOT_TIMES
    ]

    umin, umax = snapshot_color_limits(snapshots)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6),
        constrained_layout=True,
    )

    for ax, t, step in zip(
        axes.ravel(),
        SNAPSHOT_TIMES,
        snapshot_steps,
    ):
        image = ax.imshow(
            snapshots[step].T,
            extent=[0.0, L, 0.0, L],
            origin="lower",
            cmap="viridis",
            vmin=umin,
            vmax=umax,
            interpolation="nearest",
        )

        ax.set_title(rf"$t={t:.2f}$")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Thin-film dewetting and coarsening")

    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        label="film thickness",
    )

    fig.savefig(
        "5-TF_timelapse.png",
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
    Save a scalar diagnostic.
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


def plot_minmax(
    times: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
) -> None:
    """
    Plot minimum and maximum film thickness.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(times, min_values, label="minimum")
    ax.plot(times, max_values, label="maximum")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u$")
    ax.set_title("Thin-film equation: minimum/maximum thickness")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        "5-TF_minmax.png",
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
        energy_values,
        min_values,
        max_values,
    ) = solve_thin_film()

    save_individual_snapshots(snapshots)
    plot_timelapse(snapshots)

    plot_diagnostic(
        diagnostic_times,
        mass_values,
        r"$M_h(t)$",
        "Thin-film equation: mass conservation",
        "5-TF_mass.png",
    )

    plot_diagnostic(
        diagnostic_times,
        energy_values,
        r"$E_h(t)$",
        "Thin-film equation: free-energy diagnostic",
        "5-TF_energy.png",
    )

    plot_minmax(
        diagnostic_times,
        min_values,
        max_values,
    )


if __name__ == "__main__":
    main()