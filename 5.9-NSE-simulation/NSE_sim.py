#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Two-dimensional incompressible Navier--Stokes equation on a periodic square.

Equation

--------

    u_t + (u . grad)u = -grad p + nu Delta u,

    div u = 0,

on

    Omega = [0,L) x [0,L),

with periodic boundary conditions.

Numerical method

----------------

A Fourier pseudo-spectral projection method is used.

At each time step:

    1. spatial derivatives are computed spectrally;

    2. the nonlinear advection terms are formed in physical space;

    3. the nonlinear terms are filtered by the 2/3 de-aliasing rule;

    4. advection and diffusion are advanced by Forward Euler;

    5. the intermediate velocity is projected onto the divergence-free

       Fourier subspace using the Leray projection.

Thus, the time discretization is first order.  The Fourier differentiation

is spectrally accurate for smooth periodic solutions.

Diagnostics

-----------

The code tracks

    E(t) = 1/2 int_Omega |u|^2 dx,

    Z(t) = 1/2 int_Omega omega^2 dx,

where

    omega = v_x - u_y,

together with the L2 norm of div u.  For the periodic viscous problem,

kinetic energy and enstrophy should decay, while the divergence should

remain near roundoff.

Outputs

-------

    5-NSE_snapshots.png

    5-NSE_energy.png

    5-NSE_enstrophy.png

    5-NSE_divergence.png

    5-NSE_t0.png

    5-NSE_t1.png

    ...

    5-NSE_t8.png

"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt



# ============================================================

# Parameters

# ============================================================

M = 128

L = 2.0 * np.pi

NU = 0.01

DT = 0.005

T = 8.0

CFL_MAX = 0.50

SNAPSHOT_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)



# ============================================================

# Spatial grid and Fourier wave numbers

# ============================================================

dx = L / M

x = np.linspace(0.0, L, M, endpoint=False)

y = np.linspace(0.0, L, M, endpoint=False)

X, Y = np.meshgrid(x, y, indexing="ij")

kx = 2.0 * np.pi * np.fft.fftfreq(M, d=dx)

ky = 2.0 * np.pi * np.fft.fftfreq(M, d=dx)

KX, KY = np.meshgrid(kx, ky, indexing="ij")

K2 = KX**2 + KY**2

K2_SAFE = K2.copy()

K2_SAFE[0, 0] = 1.0

# 2/3 de-aliasing mask.

mode_numbers = np.fft.fftfreq(M) * M

MX, MY = np.meshgrid(mode_numbers, mode_numbers, indexing="ij")

DEALIAS_MASK = (

    (np.abs(MX) <= M / 3.0)

    & (np.abs(MY) <= M / 3.0)

)



# ============================================================

# Initial divergence-free velocity

# ============================================================

def initial_velocity() -> tuple[np.ndarray, np.ndarray]:

    """

    Construct a smooth divergence-free velocity from a stream function

        u =  psi_y,

        v = -psi_x.

    """

    psi = (

        np.sin(X) * np.sin(Y)

        + 0.4 * np.sin(2.0 * X + 0.3) * np.sin(Y)

        + 0.3 * np.sin(X) * np.sin(2.0 * Y - 0.5)

    )

    psi_hat = np.fft.fft2(psi)

    u_hat = 1j * KY * psi_hat

    v_hat = -1j * KX * psi_hat

    u = np.fft.ifft2(u_hat).real

    v = np.fft.ifft2(v_hat).real

    return project_velocity(u, v)



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

    The true zero Fourier mode is retained exactly: K2[0,0] = 0.

    """

    return np.fft.ifft2(-K2 * np.fft.fft2(f)).real



def divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:

    """

    Velocity divergence.

    """

    return ddx(u) + ddy(v)



def vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:

    """

    Scalar vorticity

        omega = v_x - u_y.

    """

    return ddx(v) - ddy(u)



# ============================================================

# Leray projection

# ============================================================

def project_velocity(

    u: np.ndarray,

    v: np.ndarray,

) -> tuple[np.ndarray, np.ndarray]:

    """

    Project a periodic velocity field onto the divergence-free subspace.

    In Fourier variables,

        P_k w_hat

        = w_hat - k (k . w_hat) / |k|^2,

    for k != 0.  The zero mode is left unchanged.

    """

    u_hat = np.fft.fft2(u)

    v_hat = np.fft.fft2(v)

    k_dot_u = KX * u_hat + KY * v_hat

    u_hat_proj = u_hat - KX * k_dot_u / K2_SAFE

    v_hat_proj = v_hat - KY * k_dot_u / K2_SAFE

    u_hat_proj[0, 0] = u_hat[0, 0]

    v_hat_proj[0, 0] = v_hat[0, 0]

    u_proj = np.fft.ifft2(u_hat_proj).real

    v_proj = np.fft.ifft2(v_hat_proj).real

    return u_proj, v_proj



# ============================================================

# De-aliased nonlinear term

# ============================================================

def dealias(f: np.ndarray) -> np.ndarray:

    """

    Apply the 2/3 Fourier de-aliasing filter.

    """

    f_hat = np.fft.fft2(f)

    f_hat *= DEALIAS_MASK

    return np.fft.ifft2(f_hat).real



def advection(

    u: np.ndarray,

    v: np.ndarray,

) -> tuple[np.ndarray, np.ndarray]:

    """

    Compute the convective terms

        (u . grad)u,

        (u . grad)v,

    using Fourier derivatives and 2/3 de-aliasing.

    """

    ux = ddx(u)

    uy = ddy(u)

    vx = ddx(v)

    vy = ddy(v)

    adv_u = u * ux + v * uy

    adv_v = u * vx + v * vy

    return dealias(adv_u), dealias(adv_v)



# ============================================================

# Diagnostics

# ============================================================

def kinetic_energy(u: np.ndarray, v: np.ndarray) -> float:

    """

    Discrete kinetic energy

        E_h = (dx^2/2) sum_{i,j} (u_ij^2 + v_ij^2).

    """

    return float(

        0.5 * dx**2 * np.sum(u**2 + v**2)

    )



def enstrophy(u: np.ndarray, v: np.ndarray) -> float:

    """

    Discrete enstrophy

        Z_h = (dx^2/2) sum_{i,j} omega_ij^2.

    """

    omega = vorticity(u, v)

    return float(

        0.5 * dx**2 * np.sum(omega**2)

    )



def divergence_l2(u: np.ndarray, v: np.ndarray) -> float:

    """

    Discrete L2 norm of div u.

    """

    div = divergence(u, v)

    return float(

        np.sqrt(dx**2 * np.sum(div**2))

    )



def cfl_number(u: np.ndarray, v: np.ndarray) -> float:

    """

    Advective CFL estimate

        CFL = dt * max(|u| + |v|) / dx.

    """

    return float(

        DT * np.max(np.abs(u) + np.abs(v)) / dx

    )



# ============================================================

# One Forward Euler projection step

# ============================================================

def time_step(

    u: np.ndarray,

    v: np.ndarray,

) -> tuple[np.ndarray, np.ndarray]:

    """

    Advance one time step.

    The intermediate velocity is

        u* = u^n + dt[-(u^n . grad)u^n + nu Delta u^n],

    followed by the Leray projection

        u^{n+1} = P u*.

    """

    adv_u, adv_v = advection(u, v)

    diff_u = NU * laplacian(u)

    diff_v = NU * laplacian(v)

    u_star = u + DT * (-adv_u + diff_u)

    v_star = v + DT * (-adv_v + diff_v)

    return project_velocity(u_star, v_star)



# ============================================================

# Time integration

# ============================================================

def solve_navier_stokes():

    """

    Integrate the velocity field from t=0 to t=T.

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

    u, v = initial_velocity()

    snapshots = {0: vorticity(u, v).copy()}

    times = np.empty(num_steps + 1)

    energies = np.empty(num_steps + 1)

    enstrophies = np.empty(num_steps + 1)

    divergences = np.empty(num_steps + 1)

    times[0] = 0.0

    energies[0] = kinetic_energy(u, v)

    enstrophies[0] = enstrophy(u, v)

    divergences[0] = divergence_l2(u, v)

    max_cfl = cfl_number(u, v)

    for n in range(1, num_steps + 1):

        cfl = cfl_number(u, v)

        max_cfl = max(max_cfl, cfl)

        if cfl > CFL_MAX:

            raise RuntimeError(

                f"CFL condition violated at step {n}: "

                f"CFL={cfl:.6f} > CFL_MAX={CFL_MAX:.6f}. "

                "Reduce DT."

            )

        u, v = time_step(u, v)

        t = n * DT

        times[n] = t

        energies[n] = kinetic_energy(u, v)

        enstrophies[n] = enstrophy(u, v)

        divergences[n] = divergence_l2(u, v)

        if n in snapshot_steps:

            snapshots[n] = vorticity(u, v).copy()

    print("2D incompressible Navier--Stokes simulation complete")

    print(f"  grid                     : {M} x {M}")

    print(f"  dx                       : {dx:.8e}")

    print(f"  dt                       : {DT:.8e}")

    print(f"  viscosity                : {NU:.8e}")

    print(f"  final time               : {T:.8e}")

    print(f"  maximum advective CFL    : {max_cfl:.8e}")

    print(

        f"  initial/final energy     : "

        f"{energies[0]:.12e} / {energies[-1]:.12e}"

    )

    print(

        f"  initial/final enstrophy  : "

        f"{enstrophies[0]:.12e} / {enstrophies[-1]:.12e}"

    )

    print(

        f"  maximum divergence L2    : "

        f"{np.max(divergences):.12e}"

    )

    return snapshots, times, energies, enstrophies, divergences



# ============================================================

# Plotting

# ============================================================

def plot_snapshot_panel(

    snapshots: dict[int, np.ndarray],

) -> None:

    """

    Save a common-scale panel of selected vorticity snapshots.

    """

    snapshot_steps = [int(round(t / DT)) for t in SNAPSHOT_TIMES]

    omega_abs_max = max(

        np.max(np.abs(snapshots[n]))

        for n in snapshot_steps

    )

    fig, axes = plt.subplots(

        2,

        4,

        figsize=(12, 6),

        constrained_layout=True,

    )

    for ax, t, n in zip(

        axes.ravel(),

        SNAPSHOT_TIMES,

        snapshot_steps,

    ):

        image = ax.imshow(

            snapshots[n].T,

            extent=[0.0, L, 0.0, L],

            origin="lower",

            cmap="RdBu_r",

            vmin=-omega_abs_max,

            vmax=omega_abs_max,

            interpolation="nearest",

        )

        ax.set_title(rf"$t={t:g}$")

        ax.set_xlabel(r"$x$")

        ax.set_ylabel(r"$y$")

    fig.colorbar(

        image,

        ax=axes.ravel().tolist(),

        label=r"$\omega(x,y,t)$",

        shrink=0.9,

    )

    fig.savefig(

        "5-NSE_snapshots.png",

        dpi=300,

        bbox_inches="tight",

    )

    plt.close(fig)



def save_individual_snapshots(

    snapshots: dict[int, np.ndarray],

) -> None:

    """

    Save each vorticity snapshot using one common symmetric color scale.

    """

    snapshot_steps = [int(round(t / DT)) for t in SNAPSHOT_TIMES]

    omega_abs_max = max(

        np.max(np.abs(snapshots[n]))

        for n in snapshot_steps

    )

    for t, n in zip(SNAPSHOT_TIMES, snapshot_steps):

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.imshow(

            snapshots[n].T,

            extent=[0.0, L, 0.0, L],

            origin="lower",

            cmap="RdBu_r",

            vmin=-omega_abs_max,

            vmax=omega_abs_max,

            interpolation="nearest",

        )

        ax.axis("off")

        fig.savefig(

            f"5-NSE_t{int(t)}.png",

            dpi=300,

            bbox_inches="tight",

            pad_inches=0.02,

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

    fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close(fig)



# ============================================================

# Main

# ============================================================

def main() -> None:

    (

        snapshots,

        times,

        energies,

        enstrophies,

        divergences,

    ) = solve_navier_stokes()

    plot_snapshot_panel(snapshots)

    save_individual_snapshots(snapshots)

    plot_diagnostic(

        times,

        energies,

        r"$E_h(t)$",

        "2D Navier--Stokes: kinetic-energy decay",

        "5-NSE_energy.png",

    )

    plot_diagnostic(

        times,

        enstrophies,

        r"$Z_h(t)$",

        "2D Navier--Stokes: enstrophy decay",

        "5-NSE_enstrophy.png",

    )

    plot_diagnostic(

        times,

        divergences,

        r"$\|\nabla\cdot\mathbf{u}\|_{L^2,h}$",

        "2D Navier--Stokes: divergence diagnostic",

        "5-NSE_divergence.png",

    )



if __name__ == "__main__":

    main()