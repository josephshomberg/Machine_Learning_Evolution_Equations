"""
Book-ready 2D ternary Cahn--Hilliard simulation on a periodic square.

Model
-----
Let c = (c1,c2,c3) with c_i >= 0 and c1+c2+c3 = 1.  On
Omega = [0,L]^2 with periodic boundary conditions, we evolve

    partial_t c_i = M * Delta mu_i,      i = 1,2,3,

with projected chemical potentials

    mu_i = dW/dc_i - eps^2 Delta c_i,
    mu_i <- mu_i - (mu_1+mu_2+mu_3)/3,

and polynomial bulk energy

    W(c) = A sum_i c_i^2 (1-c_i)^2
           + chi12 c1 c2 + chi13 c1 c3 + chi23 c2 c3.

Spatial discretization
----------------------
A uniform M_grid x M_grid periodic grid is used, with

    dx = L / M_grid,

and the standard second-order five-point Laplacian.

Time discretization
-------------------
The raw update is Forward Euler,

    c^{n+1,*} = c^n + dt * RHS(c^n).

Because Forward Euler is not positivity preserving for Cahn--Hilliard,
the tentative state is followed by a mass-corrected pointwise projection
onto the ternary simplex.  The method should therefore be described as

    Forward Euler with mass-corrected physical projection,

not as unmodified Forward Euler.

For dataset generation, every nominal time interval dt is advanced by
one or more accepted substeps whose lengths sum exactly to dt.  If the
projected state violates the optional energy guard, the current substep
is bisected.  This preserves the same physical terminal time

    T = steps * dt

for every realization, even when adaptive substepping is triggered.

The equal scalar mobility assumption is deliberate.  With unequal
component mobilities, the simple pointwise chemical-potential projection
used here no longer guarantees the same constraint structure.

Outputs
-------
- RGB concentration snapshots
- physical snapshot times
- component masses
- discrete energy
- component minima and maxima
- accepted substep sizes
- compressed NumPy archive containing the trajectory diagnostics
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

OUT_DIR = "images"
DATA_FILE = "pair_sample_projected.npz"


@dataclass(frozen=True)
class SimulationParameters:
    M_grid: int = 128
    L: float = 1.0
    dt: float = 5.0e-8
    steps: int = 50_000

    eps: float = 0.01
    mobility: float = 1.0
    A: float = 1.0
    chi: tuple[float, float, float] = (1.5, 1.5, 1.5)

    means: tuple[float, float, float] = (0.4, 0.4, 0.2)
    noise: float = 0.03
    seed: int | None = 1

    save_every: int = 500

    # Projection controls
    projection_iters: int = 50
    mass_tol: float = 1.0e-10

    # Energy-guard controls.  Set use_energy_guard=False to run fixed-step
    # Forward Euler with projection and merely monitor the energy.
    use_energy_guard: bool = True
    energy_atol: float = 1.0e-10
    energy_rtol: float = 1.0e-12

    # Adaptive substepping inside each nominal interval dt.  The sum of all
    # accepted substeps over one nominal step is exactly dt.
    max_bisections: int = 20
    min_substep: float = 1.0e-14


# ============================================================
# Validation
# ============================================================

def validate_parameters(p: SimulationParameters) -> None:
    if p.M_grid < 4:
        raise ValueError("M_grid must be at least 4.")
    if p.L <= 0.0:
        raise ValueError("L must be positive.")
    if p.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if p.steps < 1:
        raise ValueError("steps must be positive.")
    if p.eps <= 0.0:
        raise ValueError("eps must be positive.")
    if p.mobility <= 0.0:
        raise ValueError("mobility must be positive.")
    if p.save_every < 1:
        raise ValueError("save_every must be positive.")
    if p.noise < 0.0:
        raise ValueError("noise must be nonnegative.")

    means = np.asarray(p.means, dtype=float)
    if means.shape != (3,):
        raise ValueError("means must contain exactly three components.")
    if np.any(means < 0.0):
        raise ValueError("means must be nonnegative.")
    if not np.isclose(np.sum(means), 1.0):
        raise ValueError("means must sum to 1.")


# ============================================================
# Periodic finite differences
# ============================================================

def laplacian_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """Second-order five-point periodic Laplacian."""
    return (
        np.roll(u, 1, axis=0)
        + np.roll(u, -1, axis=0)
        + np.roll(u, 1, axis=1)
        + np.roll(u, -1, axis=1)
        - 4.0 * u
    ) / dx**2


def forward_gradient_periodic(
    u: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-order forward periodic differences used in the energy."""
    ux = (np.roll(u, -1, axis=0) - u) / dx
    uy = (np.roll(u, -1, axis=1) - u) / dx
    return ux, uy


# ============================================================
# Free energy and chemical potentials
# ============================================================

def bulk_energy_density(
    c: np.ndarray,
    A: float,
    chi: Sequence[float],
) -> np.ndarray:
    """Polynomial ternary bulk-energy density W(c)."""
    c1, c2, c3 = c
    chi12, chi13, chi23 = chi

    return (
        A * np.sum(c**2 * (1.0 - c)**2, axis=0)
        + chi12 * c1 * c2
        + chi13 * c1 * c3
        + chi23 * c2 * c3
    )


def bulk_derivative(
    c: np.ndarray,
    A: float,
    chi: Sequence[float],
) -> np.ndarray:
    """Component derivatives dW/dc_i of the polynomial bulk energy."""
    c1, c2, c3 = c
    chi12, chi13, chi23 = chi

    dW = np.empty_like(c)

    # d/dc [c^2(1-c)^2] = 2c - 6c^2 + 4c^3.
    dW[0] = A * (2*c1 - 6*c1**2 + 4*c1**3) + chi12*c2 + chi13*c3
    dW[1] = A * (2*c2 - 6*c2**2 + 4*c2**3) + chi12*c1 + chi23*c3
    dW[2] = A * (2*c3 - 6*c3**2 + 4*c3**3) + chi13*c1 + chi23*c2

    return dW


def chemical_potential(
    c: np.ndarray,
    dx: float,
    eps: float,
    A: float,
    chi: Sequence[float],
) -> np.ndarray:
    """
    Projected ternary chemical potential.

        mu_i = dW/dc_i - eps^2 Delta_h c_i,

    followed by the pointwise projection

        mu_i <- mu_i - (mu_1+mu_2+mu_3)/3.
    """
    mu = bulk_derivative(c, A=A, chi=chi)

    for i in range(3):
        mu[i] -= eps**2 * laplacian_periodic(c[i], dx)

    mu -= np.mean(mu, axis=0, keepdims=True)
    return mu


def energy_functional(
    c: np.ndarray,
    dx: float,
    eps: float,
    A: float,
    chi: Sequence[float],
) -> float:
    """
    Discrete ternary free energy

        E_h(c) = dx^2 sum_{i,j}
                 [W(c_ij) + eps^2/2 sum_k |grad_h c_k|^2].
    """
    W = bulk_energy_density(c, A=A, chi=chi)
    grad_part = np.zeros_like(W)

    for i in range(3):
        cx, cy = forward_gradient_periodic(c[i], dx)
        grad_part += 0.5 * eps**2 * (cx**2 + cy**2)

    return float(dx**2 * np.sum(W + grad_part))


# ============================================================
# Initial condition
# ============================================================

def initialize_ternary(
    M_grid: int,
    means: Sequence[float],
    noise: float,
    seed: int | None,
) -> np.ndarray:
    """
    Noisy ternary-simplex initial condition.

    c1 and c2 are independently perturbed around their prescribed means;
    c3 is then determined by c1+c2+c3=1 before the final physical
    projection.
    """
    rng = np.random.default_rng(seed)
    means = np.asarray(means, dtype=float)

    c = np.zeros((3, M_grid, M_grid), dtype=np.float64)
    c[0] = means[0] + rng.uniform(-noise, noise, size=(M_grid, M_grid))
    c[1] = means[1] + rng.uniform(-noise, noise, size=(M_grid, M_grid))
    c[2] = 1.0 - c[0] - c[1]

    return project_simplex_pointwise(c)


# ============================================================
# Simplex projection and mass correction
# ============================================================

def project_simplex_pointwise(c: np.ndarray) -> np.ndarray:
    """
    Euclidean projection of every vector (c1,c2,c3) onto

        {x in R^3 : x_i >= 0, sum_i x_i = 1}.
    """
    if c.ndim != 3 or c.shape[0] != 3:
        raise ValueError("Expected c with shape (3, M_grid, M_grid).")

    original_shape = c.shape
    v = c.reshape(3, -1)

    u = np.sort(v, axis=0)[::-1]
    cssv = np.cumsum(u, axis=0) - 1.0
    ind = np.arange(1, 4, dtype=float).reshape(3, 1)
    cond = u - cssv / ind > 0.0

    rho = np.sum(cond, axis=0) - 1
    theta = cssv[rho, np.arange(v.shape[1])] / (rho + 1.0)

    w = np.maximum(v - theta, 0.0)
    return w.reshape(original_shape)


def mass_corrected_projection(
    c: np.ndarray,
    target_masses: Sequence[float],
    max_iter: int,
    mass_tol: float,
) -> np.ndarray:
    """
    Alternate pointwise simplex projection with uniform component shifts.

    The output satisfies the simplex constraints pointwise and restores
    the prescribed component means to the requested tolerance whenever
    the alternating correction converges.
    """
    target_masses = np.asarray(target_masses, dtype=float)
    c = project_simplex_pointwise(c)

    for _ in range(max_iter):
        current_masses = np.mean(c, axis=(1, 2))
        drift = target_masses - current_masses

        if np.max(np.abs(drift)) <= mass_tol:
            return c

        c = c + drift.reshape(3, 1, 1)
        c = project_simplex_pointwise(c)

    final_drift = target_masses - np.mean(c, axis=(1, 2))
    if np.max(np.abs(final_drift)) > 10.0 * mass_tol:
        raise RuntimeError(
            "Mass-corrected simplex projection did not converge to the "
            "requested tolerance."
        )

    return c


# ============================================================
# Forward Euler right-hand side
# ============================================================

def forward_euler_rhs(
    c: np.ndarray,
    dx: float,
    eps: float,
    mobility: float,
    A: float,
    chi: Sequence[float],
) -> np.ndarray:
    """Compute dc/dt = mobility * Delta_h(mu) componentwise."""
    mu = chemical_potential(c, dx=dx, eps=eps, A=A, chi=chi)

    dc = np.empty_like(c)
    for i in range(3):
        dc[i] = mobility * laplacian_periodic(mu[i], dx)

    return dc


# ============================================================
# Single projected trial step
# ============================================================

def projected_euler_trial(
    c: np.ndarray,
    sub_dt: float,
    dx: float,
    p: SimulationParameters,
    target_masses: np.ndarray,
) -> np.ndarray:
    """Take one Forward Euler substep followed by physical projection."""
    dc = forward_euler_rhs(
        c,
        dx=dx,
        eps=p.eps,
        mobility=p.mobility,
        A=p.A,
        chi=p.chi,
    )

    trial = c + sub_dt * dc

    return mass_corrected_projection(
        trial,
        target_masses=target_masses,
        max_iter=p.projection_iters,
        mass_tol=p.mass_tol,
    )


# ============================================================
# Advance exactly one nominal interval dt
# ============================================================

def advance_nominal_step(
    c: np.ndarray,
    current_energy: float,
    dx: float,
    p: SimulationParameters,
    target_masses: np.ndarray,
) -> tuple[np.ndarray, float, list[float]]:
    """
    Advance exactly p.dt units of physical time.

    Rejected substeps are bisected, but the accepted substeps always sum
    to p.dt.  Hence every realization reaches the same terminal time.
    """
    remaining = p.dt
    accepted_dts: list[float] = []

    # The current candidate substep is never larger than the remaining
    # portion of the nominal interval.
    candidate_dt = remaining
    bisections = 0

    while remaining > 0.0:
        candidate_dt = min(candidate_dt, remaining)

        trial = projected_euler_trial(
            c,
            sub_dt=candidate_dt,
            dx=dx,
            p=p,
            target_masses=target_masses,
        )

        trial_energy = energy_functional(
            trial,
            dx=dx,
            eps=p.eps,
            A=p.A,
            chi=p.chi,
        )

        tolerance = p.energy_atol + p.energy_rtol * abs(current_energy)
        energy_ok = (
            (not p.use_energy_guard)
            or (trial_energy <= current_energy + tolerance)
        )

        if energy_ok:
            c = trial
            current_energy = trial_energy
            accepted_dts.append(candidate_dt)

            # Avoid accumulated floating-point residue at the end of the
            # nominal interval.
            remaining = max(0.0, remaining - candidate_dt)

            # After a successful small substep, allow the next candidate
            # to grow, but never beyond the unadvanced remainder.
            if remaining > 0.0:
                candidate_dt = min(2.0 * candidate_dt, remaining)

            bisections = 0
            continue

        candidate_dt *= 0.5
        bisections += 1

        if candidate_dt < p.min_substep:
            raise RuntimeError(
                "Adaptive substep fell below min_substep.  Reduce the "
                "nominal dt or disable the energy guard."
            )

        if bisections > p.max_bisections:
            raise RuntimeError(
                "Exceeded max_bisections while trying to advance one "
                "nominal Forward Euler interval."
            )

    return c, current_energy, accepted_dts


# ============================================================
# Image and diagnostic output
# ============================================================

def save_ternary(c: np.ndarray, filename: str, dpi: int = 180) -> None:
    """Save c1,c2,c3 as RGB channels."""
    rgb = np.clip(np.moveaxis(c, 0, -1), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_energy_plot(
    times: np.ndarray,
    energies: np.ndarray,
    filename: str,
    dpi: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, energies, linewidth=2.0)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$E_h$")
    ax.set_title("Ternary Cahn--Hilliard energy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_component_minmax_plot(
    times: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    filename: str,
    dpi: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = (r"$c_1$", r"$c_2$", r"$c_3$")
    for i, label in enumerate(labels):
        ax.plot(times, mins[:, i], linestyle=":", linewidth=1.6,
                label=f"min {label}")
        ax.plot(times, maxs[:, i], linestyle="--", linewidth=1.6,
                label=f"max {label}")

    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.axhline(1.0, linewidth=1.0, alpha=0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time")
    ax.set_ylabel("component value")
    ax.set_title("Component minima and maxima")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_substep_plot(
    accepted_dts: np.ndarray,
    filename: str,
    dpi: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(accepted_dts) + 1), accepted_dts, linewidth=1.3)
    ax.set_xlabel("accepted Euler substep")
    ax.set_ylabel(r"$\Delta t_{\rm sub}$")
    ax.set_title("Accepted Forward Euler substeps")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main simulation
# ============================================================

def simulate_ternary_ch_projected(
    p: SimulationParameters,
) -> dict[str, np.ndarray]:
    """Run the projected Forward Euler ternary Cahn--Hilliard simulation."""
    validate_parameters(p)
    os.makedirs(OUT_DIR, exist_ok=True)

    dx = p.L / p.M_grid
    domain_area = p.L**2

    c = initialize_ternary(
        M_grid=p.M_grid,
        means=p.means,
        noise=p.noise,
        seed=p.seed,
    )

    # Preserve the actually realized component means.  This avoids
    # artificially changing the initial sample after it has been generated.
    target_masses = np.mean(c, axis=(1, 2))

    c = mass_corrected_projection(
        c,
        target_masses=target_masses,
        max_iter=p.projection_iters,
        mass_tol=p.mass_tol,
    )

    snapshots: list[np.ndarray] = []
    times: list[float] = []
    masses: list[np.ndarray] = []
    energies: list[float] = []
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    accepted_substeps: list[float] = []

    current_energy = energy_functional(
        c,
        dx=dx,
        eps=p.eps,
        A=p.A,
        chi=p.chi,
    )

    def record(step: int) -> None:
        snapshots.append(c.copy())
        times.append(step * p.dt)
        masses.append(np.mean(c, axis=(1, 2)))
        energies.append(current_energy)
        mins.append(np.min(c, axis=(1, 2)))
        maxs.append(np.max(c, axis=(1, 2)))

        save_ternary(c, os.path.join(OUT_DIR, f"frame_step_{step:06d}.png"))

    record(0)

    for step in range(1, p.steps + 1):
        c, current_energy, substeps = advance_nominal_step(
            c,
            current_energy=current_energy,
            dx=dx,
            p=p,
            target_masses=target_masses,
        )
        accepted_substeps.extend(substeps)

        if step % p.save_every == 0 or step == p.steps:
            record(step)

        if step % p.save_every == 0:
            current_masses = np.mean(c, axis=(1, 2))
            mass_drift = current_masses - target_masses
            print(
                f"step={step:6d}/{p.steps}  "
                f"t={step*p.dt:.6e}  "
                f"E={current_energy:.10e}  "
                f"max|mass drift|={np.max(np.abs(mass_drift)):.3e}  "
                f"substeps={len(substeps)}"
            )

    result = {
        "snapshots": np.asarray(snapshots),
        "times": np.asarray(times),
        "masses": np.asarray(masses),
        "energies": np.asarray(energies),
        "mins": np.asarray(mins),
        "maxs": np.asarray(maxs),
        "accepted_substeps": np.asarray(accepted_substeps),
        "target_masses": np.asarray(target_masses),
        "dx": np.asarray(dx),
        "domain_area": np.asarray(domain_area),
        "terminal_time": np.asarray(p.steps * p.dt),
    }

    return result


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    params = SimulationParameters(
        M_grid=128,
        L=1.0,
        dt=5.0e-8,
        steps=50_000,
        eps=0.01,
        mobility=1.0,
        A=1.0,
        chi=(1.5, 1.5, 1.5),
        means=(0.4, 0.4, 0.2),
        noise=0.03,
        seed=1,
        save_every=500,
        projection_iters=50,
        mass_tol=1.0e-10,
        use_energy_guard=True,
        energy_atol=1.0e-10,
        energy_rtol=1.0e-12,
        max_bisections=20,
        min_substep=1.0e-14,
    )

    result = simulate_ternary_ch_projected(params)

    snapshots = result["snapshots"]
    times = result["times"]
    masses = result["masses"]
    energies = result["energies"]
    mins = result["mins"]
    maxs = result["maxs"]
    accepted_substeps = result["accepted_substeps"]

    print("\nSimulation summary")
    print("------------------")
    print("Initial masses: ", masses[0])
    print("Final masses:   ", masses[-1])
    print("Mass drift:     ", masses[-1] - masses[0])
    print("Initial energy: ", energies[0])
    print("Final energy:   ", energies[-1])
    print("Energy change:  ", energies[-1] - energies[0])
    print("Final minima:   ", mins[-1])
    print("Final maxima:   ", maxs[-1])
    print("Terminal time:  ", float(result["terminal_time"]))
    print("Nominal dt:     ", params.dt)
    print("Min substep:    ", np.min(accepted_substeps))
    print("Max substep:    ", np.max(accepted_substeps))

    save_energy_plot(
        times,
        energies,
        os.path.join(OUT_DIR, "energy_plot.png"),
    )

    save_component_minmax_plot(
        times,
        mins,
        maxs,
        os.path.join(OUT_DIR, "component_minmax.png"),
    )

    save_substep_plot(
        accepted_substeps,
        os.path.join(OUT_DIR, "accepted_substeps.png"),
    )

    np.savez_compressed(
        DATA_FILE,
        u0=snapshots[0],
        uT=snapshots[-1],
        **result,
    )

    print(f"\nSaved images to: {os.path.abspath(OUT_DIR)}")
    print(f"Saved data to:   {os.path.abspath(DATA_FILE)}")