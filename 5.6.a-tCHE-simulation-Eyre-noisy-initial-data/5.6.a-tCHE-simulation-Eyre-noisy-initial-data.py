#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Book-ready 2D ternary Cahn--Hilliard simulation.

PDE:
    partial_t c = kappa * Mmob * Delta(mu),

with periodic boundary conditions on Omega=[0,L)^2, Gibbs-simplex
concentrations c=(c1,c2,c3), Eyre-noisy initial data, and polynomial energy

    W(c) = A sum_i c_i^2(1-c_i)^2
           + chi12 c1 c2 + chi13 c1 c3 + chi23 c2 c3.

The first-order Eyre convex--concave split is

    mu_i^{n+1,n}
      = -eps^2 Delta_h c_i^{n+1}
        + 4A(c_i^{n+1})^3 + S c_i^{n+1}
        - 6A(c_i^n)^2 + (2A-S)c_i^n
        + sum_{j != i} chi_ij c_j^n,

and

    (c^{n+1}-c^n)/dt
      = kappa * Mmob * Delta_h(mu^{n+1,n}).

Because the cubic convex contribution is implicit, every time step is a
nonlinear elliptic algebraic solve.  It is solved here by fixed-point
iteration.  After convergence, the provisional state is mass-corrected and
projected pointwise onto the Gibbs simplex.

For A=1 and chi_ij=1.5, S=5 satisfies the convenient sufficient condition
S >= 2A + 2 max(chi_ij) for the chosen convex--concave decomposition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class Parameters:
    M: int = 128
    L: float = 1.0

    dt: float = 5.0e-8
    steps: int = 50_000
    save_every: int = 500

    eps: float = 0.01
    kappa: float = 1.0

    A: float = 1.0
    chi: tuple[float, float, float] = (1.5, 1.5, 1.5)
    stabilization: float = 5.0

    means: tuple[float, float, float] = (0.4, 0.4, 0.2)
    noise: float = 0.03
    seed: int | None = 1

    max_fixed_point_iters: int = 20
    fixed_point_tol: float = 1.0e-10

    projection_iters: int = 50
    mass_tol: float = 1.0e-10

    @property
    def dx(self) -> float:
        return self.L / self.M

    @property
    def gamma(self) -> float:
        return self.eps**2

    @property
    def terminal_time(self) -> float:
        return self.steps * self.dt


P = Parameters()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
DATA_FILE = os.path.join(SCRIPT_DIR, "final_state_polynomial_eyre.npz")
os.makedirs(IMAGE_DIR, exist_ok=True)


# ============================================================
# Mobility matrix
# ============================================================

MOBILITY_MATRIX = (1.0 / 3.0) * np.array(
    [
        [2.0, -1.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -1.0, 2.0],
    ],
    dtype=np.float64,
)


# ============================================================
# Validation
# ============================================================

def validate_parameters(p: Parameters) -> None:
    if p.M < 4:
        raise ValueError("M must be at least 4.")
    if p.L <= 0.0 or p.dt <= 0.0 or p.steps < 1:
        raise ValueError("L, dt, and steps must be positive.")
    if p.eps <= 0.0 or p.kappa <= 0.0 or p.A <= 0.0:
        raise ValueError("eps, kappa, and A must be positive.")

    means = np.asarray(p.means, dtype=float)
    if means.shape != (3,) or np.any(means < 0.0):
        raise ValueError("means must contain three nonnegative values.")
    if not np.isclose(np.sum(means), 1.0):
        raise ValueError("means must sum to 1.")

    required = 2.0 * p.A + 2.0 * max(p.chi)
    if p.stabilization < required:
        raise ValueError(
            f"stabilization={p.stabilization} is too small; "
            f"use at least {required}."
        )


# ============================================================
# Periodic finite differences
# ============================================================

def laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    """Second-order five-point periodic Laplacian."""
    return (
        np.roll(u, 1, axis=-2)
        + np.roll(u, -1, axis=-2)
        + np.roll(u, 1, axis=-1)
        + np.roll(u, -1, axis=-1)
        - 4.0 * u
    ) / dx**2


def gradx(u: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(u, -1, axis=-2)
        - np.roll(u, 1, axis=-2)
    ) / (2.0 * dx)


def grady(u: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(u, -1, axis=-1)
        - np.roll(u, 1, axis=-1)
    ) / (2.0 * dx)


# ============================================================
# Simplex projection and mass correction
# ============================================================

def project_simplex_pointwise(c: np.ndarray) -> np.ndarray:
    """
    Euclidean projection of each (c1,c2,c3) onto
        {x in R^3 : x_i >= 0, sum_i x_i = 1}.
    """
    if c.ndim != 3 or c.shape[0] != 3:
        raise ValueError("Expected c with shape (3,M,M).")

    shape = c.shape
    v = c.reshape(3, -1)

    u = np.sort(v, axis=0)[::-1]
    cssv = np.cumsum(u, axis=0) - 1.0
    ind = np.arange(1, 4, dtype=float).reshape(3, 1)
    cond = u - cssv / ind > 0.0
    rho = np.sum(cond, axis=0) - 1
    theta = cssv[rho, np.arange(v.shape[1])] / (rho + 1.0)

    return np.maximum(v - theta, 0.0).reshape(shape)


def mass_corrected_projection(
    c: np.ndarray,
    target_means: Sequence[float],
    max_iter: int,
    mass_tol: float,
) -> np.ndarray:
    """
    Alternate simplex projection with uniform component shifts so that the
    pointwise simplex constraint and component means are both recovered.
    """
    target_means = np.asarray(target_means, dtype=float)
    c = project_simplex_pointwise(c)

    for _ in range(max_iter):
        current = np.mean(c, axis=(1, 2))
        drift = target_means - current

        if np.max(np.abs(drift)) <= mass_tol:
            return c

        c = c + drift.reshape(3, 1, 1)
        c = project_simplex_pointwise(c)

    drift = target_means - np.mean(c, axis=(1, 2))
    if np.max(np.abs(drift)) > 10.0 * mass_tol:
        raise RuntimeError("Mass-corrected simplex projection did not converge.")

    return c


# ============================================================
# Eyre-noisy initial data
# ============================================================

def make_initial_condition(p: Parameters) -> np.ndarray:
    """
    Small uniform perturbations of (0.4,0.4,0.2).

    c1 and c2 are perturbed independently; c3 is determined from
    c1+c2+c3=1 and the final field is projected onto the simplex.
    """
    rng = np.random.default_rng(p.seed)

    c = np.zeros((3, p.M, p.M), dtype=np.float64)

    c[0] = p.means[0] + rng.uniform(
        -p.noise, p.noise, size=(p.M, p.M)
    )
    c[1] = p.means[1] + rng.uniform(
        -p.noise, p.noise, size=(p.M, p.M)
    )
    c[2] = 1.0 - c[0] - c[1]

    return project_simplex_pointwise(c)


# ============================================================
# Polynomial energy and Eyre splitting
# ============================================================

def bulk_energy_density(
    c: np.ndarray,
    A: float,
    chi: Sequence[float],
) -> np.ndarray:
    c1, c2, c3 = c
    chi12, chi13, chi23 = chi

    return (
        A * np.sum(c**2 * (1.0 - c)**2, axis=0)
        + chi12 * c1 * c2
        + chi13 * c1 * c3
        + chi23 * c2 * c3
    )


def convex_derivative(c_new: np.ndarray, p: Parameters) -> np.ndarray:
    """
    Implicit derivative:
        f_i^c(c_i) = 4 A c_i^3 + S c_i.
    """
    return 4.0 * p.A * c_new**3 + p.stabilization * c_new


def concave_derivative(c_old: np.ndarray, p: Parameters) -> np.ndarray:
    """
    Explicit derivative:
        f_i^e
          = -6 A c_i^2 + (2A-S)c_i
            + sum_{j != i} chi_ij c_j.
    """
    c1, c2, c3 = c_old
    chi12, chi13, chi23 = p.chi
    lin = 2.0 * p.A - p.stabilization

    out = np.empty_like(c_old)

    out[0] = -6.0*p.A*c1**2 + lin*c1 + chi12*c2 + chi13*c3
    out[1] = -6.0*p.A*c2**2 + lin*c2 + chi12*c1 + chi23*c3
    out[2] = -6.0*p.A*c3**2 + lin*c3 + chi13*c1 + chi23*c2

    return out


def chemical_potential(
    c_new: np.ndarray,
    c_old: np.ndarray,
    p: Parameters,
) -> np.ndarray:
    """
    Eyre semi-implicit chemical potential:
        mu^{n+1,n}
          = -eps^2 Delta_h c^{n+1}
            + f^c(c^{n+1})
            + f^e(c^n).
    """
    return (
        -p.gamma * laplacian(c_new, p.dx)
        + convex_derivative(c_new, p)
        + concave_derivative(c_old, p)
    )


def mobility_apply(mu: np.ndarray, dx: float) -> np.ndarray:
    lap_mu = laplacian(mu, dx)
    return np.einsum("ab,bxy->axy", MOBILITY_MATRIX, lap_mu)


# ============================================================
# Nonlinear semi-implicit Eyre step
# ============================================================

def step(
    c_old: np.ndarray,
    p: Parameters,
    target_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Solve
        c^{n+1}
          = c^n + dt*kappa*Mmob*Delta_h(mu^{n+1,n})
    by fixed-point iteration.

    Projection is applied only after the nonlinear solve has converged.
    """
    c_iter = c_old.copy()
    fp_error = np.inf

    for iteration in range(1, p.max_fixed_point_iters + 1):
        mu = chemical_potential(c_iter, c_old, p)
        rhs = p.kappa * mobility_apply(mu, p.dx)

        c_next = c_old + p.dt * rhs
        fp_error = float(np.max(np.abs(c_next - c_iter)))
        c_iter = c_next

        if fp_error <= p.fixed_point_tol:
            break
    else:
        raise RuntimeError(
            "Fixed-point solve failed to converge: "
            f"error={fp_error:.3e} after {p.max_fixed_point_iters} iterations."
        )

    c_new = mass_corrected_projection(
        c_iter,
        target_means=target_means,
        max_iter=p.projection_iters,
        mass_tol=p.mass_tol,
    )

    mu_final = chemical_potential(c_new, c_old, p)

    return c_new, mu_final, iteration, fp_error


# ============================================================
# Energy and diagnostics
# ============================================================

def total_energy(c: np.ndarray, p: Parameters) -> float:
    W = bulk_energy_density(c, p.A, p.chi)
    gx = gradx(c, p.dx)
    gy = grady(c, p.dx)

    grad_part = 0.5 * p.gamma * np.sum(gx**2 + gy**2, axis=0)

    return float(p.dx**2 * np.sum(W + grad_part))


def simplex_error(c: np.ndarray) -> float:
    return float(np.max(np.abs(np.sum(c, axis=0) - 1.0)))


def component_means(c: np.ndarray) -> np.ndarray:
    return np.mean(c, axis=(1, 2))


# ============================================================
# Output
# ============================================================

def save_phi_image(c: np.ndarray, step_num: int, dpi: int = 300) -> None:
    rgb = np.clip(np.moveaxis(c, 0, -1), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)

    fig.savefig(
        os.path.join(IMAGE_DIR, f"phi_{step_num:05d}.png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def plot_scalar(times, values, ylabel, title, filename) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, values, linewidth=2.0)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(IMAGE_DIR, filename),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_component_minmax(times, mins, maxs) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = (r"$c_1$", r"$c_2$", r"$c_3$")

    for i, label in enumerate(labels):
        ax.plot(times, mins[:, i], ":", linewidth=1.6, label=f"min {label}")
        ax.plot(times, maxs[:, i], "--", linewidth=1.6, label=f"max {label}")

    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.axhline(1.0, linewidth=1.0, alpha=0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time")
    ax.set_ylabel("component value")
    ax.set_title("Component minima and maxima")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(IMAGE_DIR, "component_minmax.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_component_means(times, means) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(3):
        ax.plot(times, means[:, i], linewidth=1.8, label=rf"$\bar c_{i+1}$")

    ax.set_xlabel("time")
    ax.set_ylabel("component mean")
    ax.set_title("Component-mass conservation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(IMAGE_DIR, "component_mass.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Simulation
# ============================================================

def simulate(p: Parameters = P) -> dict[str, np.ndarray]:
    validate_parameters(p)

    c = make_initial_condition(p)
    target_means = component_means(c)

    c = mass_corrected_projection(
        c,
        target_means,
        p.projection_iters,
        p.mass_tol,
    )

    save_phi_image(c, 0)

    mu = chemical_potential(c, c, p)

    times = [0.0]
    energies = [total_energy(c, p)]
    means_hist = [component_means(c)]
    mins_hist = [np.min(c, axis=(1, 2))]
    maxs_hist = [np.max(c, axis=(1, 2))]
    simplex_hist = [simplex_error(c)]

    fp_times = []
    fp_iters_hist = []
    fp_errors_hist = []

    print("Ternary Cahn--Hilliard: Eyre semi-implicit polynomial simulation")
    print("----------------------------------------------------------------")
    print(f"grid                : {p.M} x {p.M}")
    print(f"domain              : [0,{p.L}) x [0,{p.L})")
    print(f"h                   : {p.dx:.8e}")
    print(f"dt                  : {p.dt:.8e}")
    print(f"N                   : {p.steps}")
    print(f"T                   : {p.terminal_time:.8e}")
    print(f"epsilon             : {p.eps:.8e}")
    print(f"A                   : {p.A:.8e}")
    print(f"chi                 : {p.chi}")
    print(f"Eyre stabilization  : {p.stabilization:.8e}")
    print(f"initial means       : {target_means}")
    print()

    for n in range(1, p.steps + 1):
        c, mu, fp_iters, fp_error = step(c, p, target_means)

        if not np.all(np.isfinite(c)):
            raise FloatingPointError(f"Non-finite concentration at step {n}.")

        fp_times.append(n * p.dt)
        fp_iters_hist.append(fp_iters)
        fp_errors_hist.append(fp_error)

        if n % p.save_every == 0 or n == p.steps:
            t = n * p.dt
            E = total_energy(c, p)
            means_now = component_means(c)
            mins_now = np.min(c, axis=(1, 2))
            maxs_now = np.max(c, axis=(1, 2))
            s_err = simplex_error(c)

            times.append(t)
            energies.append(E)
            means_hist.append(means_now)
            mins_hist.append(mins_now)
            maxs_hist.append(maxs_now)
            simplex_hist.append(s_err)

            save_phi_image(c, n)

            print(
                f"{n:6d}/{p.steps}  "
                f"t={t:.6e}  "
                f"E={E:.10e}  "
                f"means=({means_now[0]:.6f},{means_now[1]:.6f},{means_now[2]:.6f})  "
                f"min=({mins_now[0]:.4f},{mins_now[1]:.4f},{mins_now[2]:.4f})  "
                f"max=({maxs_now[0]:.4f},{maxs_now[1]:.4f},{maxs_now[2]:.4f})  "
                f"simplex={s_err:.2e}  FP={fp_iters}  FPerr={fp_error:.2e}"
            )

    result = {
        "phi": c,
        "mu": mu,
        "times": np.asarray(times),
        "energies": np.asarray(energies),
        "means": np.asarray(means_hist),
        "mins": np.asarray(mins_hist),
        "maxs": np.asarray(maxs_hist),
        "simplex_errors": np.asarray(simplex_hist),
        "fp_times": np.asarray(fp_times),
        "fp_iterations": np.asarray(fp_iters_hist),
        "fp_errors": np.asarray(fp_errors_hist),
        "target_means": np.asarray(target_means),
        "dx": np.asarray(p.dx),
        "dt": np.asarray(p.dt),
        "terminal_time": np.asarray(p.terminal_time),
    }

    return result


def main() -> None:
    result = simulate(P)

    plot_scalar(
        result["times"],
        result["energies"],
        r"$E_h$",
        "Ternary Cahn--Hilliard polynomial free energy",
        "lyapunov_energy.png",
    )

    plot_component_minmax(
        result["times"],
        result["mins"],
        result["maxs"],
    )

    plot_component_means(
        result["times"],
        result["means"],
    )

    plot_scalar(
        result["times"],
        np.maximum(result["simplex_errors"], np.finfo(float).tiny),
        r"$\max |c_1+c_2+c_3-1|$",
        "Gibbs-simplex constraint error",
        "simplex_error.png",
    )

    plot_scalar(
        result["fp_times"],
        result["fp_iterations"],
        "fixed-point iterations",
        "Nonlinear Eyre solve: iterations per time step",
        "fixed_point_iterations.png",
    )

    dE = np.diff(result["energies"])

    print("\nSimulation summary")
    print("------------------")
    print("Initial means:          ", result["means"][0])
    print("Final means:            ", result["means"][-1])
    print("Mass drift:             ", result["means"][-1] - result["means"][0])
    print("Initial energy:         ", result["energies"][0])
    print("Final energy:           ", result["energies"][-1])
    if dE.size:
        print("Maximum recorded dE:    ", np.max(dE))
        print("Recorded energy monotone:", np.all(dE <= 1.0e-10))
    print("Final minima:           ", result["mins"][-1])
    print("Final maxima:           ", result["maxs"][-1])
    print("Max simplex error:      ", np.max(result["simplex_errors"]))
    print("Max FP iterations:      ", np.max(result["fp_iterations"]))
    print("Max final FP error:     ", np.max(result["fp_errors"]))

    np.savez_compressed(DATA_FILE, **result)

    print(f"\nSaved images to: {IMAGE_DIR}")
    print(f"Saved data to:   {DATA_FILE}")


if __name__ == "__main__":
    main()
