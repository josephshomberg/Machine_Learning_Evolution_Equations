#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-resolution two-dimensional thin-film / dewetting simulation
with degenerate mobility on a periodic square.

Equation
--------
    u_t = div( M(u) grad(mu) ),

    mu = -eps^2 Delta u + f(u),

    f(u) = u^3 - u,

    M(u) = max(u, u_mob)^m,

on

    Omega = [0,L) x [0,L),

with periodic boundary conditions.

Numerical method
----------------
A Fourier pseudo-spectral method is used in space, with nonlinear
fluxes formed in physical space and filtered by the 2/3 de-aliasing rule.

Write

    M(u^n) = Mbar^n + (M(u^n) - Mbar^n).

The constant-coefficient fourth-order part is treated implicitly.
The variable-mobility remainder is treated explicitly.  A linear
stabilization term is added and subtracted so that the method remains
consistent while permitting a practical time step.

In Fourier variables, one step has the form

    [1 + dt Mbar (eps^2 |k|^4 + S |k|^2)] u_hat^{n+1}
        =
    u_hat^n
    + dt [R_hat^n + Mbar S |k|^2 u_hat^n],

where R_hat^n contains the de-aliased variable-mobility fluxes.

The zero Fourier mode is preserved exactly, so the discrete mass is
conserved up to roundoff.

This version is designed for the Chapter 5 figures:
    * 256 x 256 spatial grid,
    * smooth band-limited random initial film,
    * stronger spinodal/dewetting development,
    * common publication-quality color scale,
    * individual images and a 2 x 4 time-lapse panel,
    * mass, energy, and minimum/maximum diagnostics.

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

M = 256
L = 1.0

EPS = 0.014
MOBILITY_EXPONENT = 3.0
MOBILITY_FLOOR = 0.02

DT = 1.0e-4
T = 0.50

# Linear stabilization added and subtracted in the semi-implicit split.
STABILIZATION = 1.0

MEAN_FILM = 0.48
INITIAL_AMPLITUDE = 0.07
SEED = 7

# Spectral center and width of the smooth random perturbation.
NOISE_MODE_CENTER = 10.0
NOISE_MODE_WIDTH = 4.0

DIAGNOSTIC_STRIDE = 10

SNAPSHOT_TIMES = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
SNAPSHOT_LABELS = (0, 1, 2, 3, 4, 5, 6, 8)

CMAP = "magma"
FIG_DPI = 400


# ============================================================
# Grid and Fourier wave numbers
# ============================================================

dx = L / M

x = np.linspace(0.0, L, M, endpoint=False)
y = np.linspace(0.0, L, M, endpoint=False)

X, Y = np.meshgrid(x, y, indexing="ij")

kx = 2.0 * np.pi * np.fft.fftfreq(M, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(M, d=dx)

KX, KY = np.meshgrid(kx, ky, indexing="ij")

K2 = KX**2 + KY**2
K4 = K2**2

# Integer Fourier-mode numbers, used for de-aliasing and initial-data filtering.
mode_numbers = np.fft.fftfreq(M) * M
MX, MY = np.meshgrid(mode_numbers, mode_numbers, indexing="ij")
MODE_RADIUS = np.sqrt(MX**2 + MY**2)

DEALIAS_MASK = (
    (np.abs(MX) <= M / 3.0)
    & (np.abs(MY) <= M / 3.0)
)


# ============================================================
# Initial condition
# ============================================================

def initial_condition() -> np.ndarray:
    """
    Smooth band-limited random perturbation of a nearly flat film.

    A real Gaussian random field is filtered in Fourier space so that the
    initial perturbation is concentrated around a prescribed band of spatial
    frequencies.  This avoids grid-scale noise and produces a much cleaner
    dewetting/coarsening experiment.
    """

    rng = np.random.default_rng(SEED)
    eta = rng.standard_normal((M, M))

    eta_hat = np.fft.fft2(eta)

    spectral_filter = np.exp(
        -0.5
        * ((MODE_RADIUS - NOISE_MODE_CENTER) / NOISE_MODE_WIDTH) ** 2
    )
    spectral_filter[0, 0] = 0.0

    eta = np.fft.ifft2(eta_hat * spectral_filter).real
    eta /= np.max(np.abs(eta))

    u0 = MEAN_FILM + INITIAL_AMPLITUDE * eta

    if np.min(u0) <= 0.0:
        raise ValueError(
            "Initial film thickness is nonpositive. "
            "Reduce INITIAL_AMPLITUDE or increase MEAN_FILM."
        )

    return u0


# ============================================================
# Fourier differential utilities
# ============================================================

def gradient_from_hat(
    f_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spectral gradient of a field already represented in Fourier space.
    """

    fx = np.fft.ifft2(1j * KX * f_hat).real
    fy = np.fft.ifft2(1j * KY * f_hat).real

    return fx, fy


def divergence_of_dealiased_flux_hat(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
) -> np.ndarray:
    """
    Fourier transform of div(flux), after 2/3 de-aliasing of both
    nonlinear flux components.
    """

    flux_x_hat = np.fft.fft2(flux_x)
    flux_y_hat = np.fft.fft2(flux_y)

    flux_x_hat *= DEALIAS_MASK
    flux_y_hat *= DEALIAS_MASK

    return (
        1j * KX * flux_x_hat
        + 1j * KY * flux_y_hat
    )


# ============================================================
# Model functions
# ============================================================

def bulk_derivative(u: np.ndarray) -> np.ndarray:
    """
    Derivative of

        F(u) = 1/4 (u^2 - 1)^2,

    namely

        F'(u) = u^3 - u.
    """

    return u**3 - u


def mobility(u: np.ndarray) -> np.ndarray:
    """
    Regularized degenerate mobility

        M(u) = max(u, MOBILITY_FLOOR)^m.

    The floor is used only in the mobility evaluation; the solution itself
    is not clipped.
    """

    return np.maximum(
        u,
        MOBILITY_FLOOR,
    ) ** MOBILITY_EXPONENT


# ============================================================
# Semi-implicit variable-mobility step
# ============================================================

def time_step(u: np.ndarray) -> np.ndarray:
    """
    Advance one first-order stabilized semi-implicit Fourier step.

    The fourth-order constant-mobility contribution and the linear
    stabilization are implicit.  Variable-mobility and nonlinear terms are
    explicit and de-aliased.
    """

    u_hat = np.fft.fft2(u)

    mobility_field = mobility(u)
    mbar = float(np.mean(mobility_field))
    mdev = mobility_field - mbar

    # grad(Delta u), using the already available Fourier representation.
    lap_u_hat = -K2 * u_hat
    lap_ux, lap_uy = gradient_from_hat(lap_u_hat)

    # grad f(u).
    f = bulk_derivative(u)
    f_hat = np.fft.fft2(f)
    fx, fy = gradient_from_hat(f_hat)

    # Variable-mobility fourth-order flux.
    div4_hat = divergence_of_dealiased_flux_hat(
        mdev * lap_ux,
        mdev * lap_uy,
    )

    # Nonlinear chemical-potential flux.
    div2_hat = divergence_of_dealiased_flux_hat(
        mobility_field * fx,
        mobility_field * fy,
    )

    remainder_hat = (
        -EPS**2 * div4_hat
        + div2_hat
    )

    denominator = (
        1.0
        + DT
        * mbar
        * (
            EPS**2 * K4
            + STABILIZATION * K2
        )
    )

    # Explicit subtraction of the stabilization term added implicitly.
    numerator = (
        u_hat
        + DT
        * (
            remainder_hat
            + mbar * STABILIZATION * K2 * u_hat
        )
    )

    u_hat_new = numerator / denominator

    # Preserve the mean film thickness exactly.
    u_hat_new[0, 0] = u_hat[0, 0]

    return np.fft.ifft2(u_hat_new).real


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

    u_hat = np.fft.fft2(u)
    ux, uy = gradient_from_hat(u_hat)

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
    Integrate from t=0 to t=T.
    """

    num_steps = int(round(T / DT))

    if not np.isclose(num_steps * DT, T):
        raise ValueError("T must be an integer multiple of DT.")

    snapshot_steps = {
        int(round(t / DT)): t
        for t in SNAPSHOT_TIMES
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

        if np.min(u) < -0.05:
            raise FloatingPointError(
                f"Film thickness became strongly negative at step {step}: "
                f"min(u)={np.min(u):.8e}. Reduce DT."
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

    mass_drift = np.max(
        np.abs(mass_values - mass_values[0])
    )

    print("High-resolution thin-film / dewetting simulation complete")
    print(f"  grid                     : {M} x {M}")
    print(f"  dx                       : {dx:.8e}")
    print(f"  dt                       : {DT:.8e}")
    print(f"  time steps               : {num_steps}")
    print(f"  final time               : {T:.8e}")
    print(f"  epsilon                  : {EPS:.8e}")
    print(f"  mean film thickness      : {MEAN_FILM:.8e}")
    print(f"  mobility exponent        : {MOBILITY_EXPONENT:.8e}")
    print(f"  stabilization            : {STABILIZATION:.8e}")
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
    Common physical color limits for every snapshot.
    """

    snapshot_steps = [
        int(round(t / DT))
        for t in SNAPSHOT_TIMES
    ]

    umin = min(
        np.min(snapshots[n])
        for n in snapshot_steps
    )
    umax = max(
        np.max(snapshots[n])
        for n in snapshot_steps
    )

    return float(umin), float(umax)


def save_individual_snapshots(
    snapshots: dict[int, np.ndarray],
) -> None:
    """
    Save the individual high-resolution images used by LaTeX.
    """

    snapshot_steps = [
        int(round(t / DT))
        for t in SNAPSHOT_TIMES
    ]

    umin, umax = snapshot_color_limits(snapshots)

    for label, step in zip(
        SNAPSHOT_LABELS,
        snapshot_steps,
    ):
        fig, ax = plt.subplots(
            figsize=(5.0, 5.0)
        )

        ax.imshow(
            snapshots[step].T,
            extent=[0.0, L, 0.0, L],
            origin="lower",
            cmap=CMAP,
            vmin=umin,
            vmax=umax,
            interpolation="bicubic",
        )

        ax.axis("off")

        fig.savefig(
            f"5-TF_t{label}.png",
            dpi=FIG_DPI,
            bbox_inches="tight",
            pad_inches=0.015,
        )

        plt.close(fig)


def plot_timelapse(
    snapshots: dict[int, np.ndarray],
) -> None:
    """
    Save a common-scale 2 x 4 time-lapse montage.
    """

    snapshot_steps = [
        int(round(t / DT))
        for t in SNAPSHOT_TIMES
    ]

    umin, umax = snapshot_color_limits(snapshots)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14.5, 7.1),
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
            cmap=CMAP,
            vmin=umin,
            vmax=umax,
            interpolation="bicubic",
        )

        ax.set_title(
            rf"$t={t:.2f}$",
            fontsize=14,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Thin-film dewetting and coarsening",
        fontsize=17,
    )

    cbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.88,
        pad=0.018,
    )
    cbar.set_label(
        "film thickness",
        fontsize=13,
    )

    fig.savefig(
        "5-TF_timelapse.png",
        dpi=FIG_DPI,
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
    Save one scalar diagnostic.
    """

    fig, ax = plt.subplots(
        figsize=(7.5, 4.5)
    )

    ax.plot(
        times,
        values,
        linewidth=2.0,
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=FIG_DPI,
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

    fig, ax = plt.subplots(
        figsize=(7.5, 4.5)
    )

    ax.plot(
        times,
        min_values,
        label="minimum",
        linewidth=2.0,
    )
    ax.plot(
        times,
        max_values,
        label="maximum",
        linewidth=2.0,
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u$")
    ax.set_title(
        "Thin-film equation: minimum/maximum thickness"
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        "5-TF_minmax.png",
        dpi=FIG_DPI,
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

    save_individual_snapshots(
        snapshots
    )
    plot_timelapse(
        snapshots
    )

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