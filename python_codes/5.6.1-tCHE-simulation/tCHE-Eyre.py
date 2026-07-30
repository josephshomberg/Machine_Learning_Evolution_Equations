"""
Forward Euler simulation for the ternary Cahn--Hilliard equation
with periodic boundary conditions.

Optimized dataset-friendly version:
    - periodic finite differences
    - Forward Euler time stepping
    - projected chemical potentials
    - physical cutoff/projection after each Euler step
    - enforces 0 <= c_i <= 1 and c1+c2+c3 = 1 pointwise
    - approximately preserves initial component masses
    - rejects steps that increase energy beyond a tolerance
    - saves grayscale frames
    - saves energy plot separately
    - saves component min/max plot separately

Important mathematical note:
    The raw Forward Euler Cahn--Hilliard scheme conserves mass under periodic
    boundary conditions, but it is not positivity preserving.

    This script adds a physical projection/cutoff after each Euler step.
    That makes the output much better for machine-learning datasets, but the
    resulting method should be described as

        Forward Euler with mass-corrected physical projection,

    not as the unmodified Forward Euler scheme.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Output directory
# ============================================================

OUT_DIR = "images"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Periodic finite differences
# ============================================================

def laplacian_periodic(u, dx):
    """
    Second-order five-point periodic Laplacian on a square grid.
    """

    return (
        np.roll(u, 1, axis=0)
        + np.roll(u, -1, axis=0)
        + np.roll(u, 1, axis=1)
        + np.roll(u, -1, axis=1)
        - 4.0 * u
    ) / dx**2


# ============================================================
# Free energy and chemical potentials
# ============================================================

def bulk_derivative(c, A=1.0, chi=(1.0, 1.0, 1.0)):
    """
    Derivative of ternary bulk free energy.

    c has shape (3, N, N).

    Bulk energy density:

        W(c) = A sum_i c_i^2 (1-c_i)^2
             + chi12 c1 c2 + chi13 c1 c3 + chi23 c2 c3.
    """

    c1, c2, c3 = c
    chi12, chi13, chi23 = chi

    dW = np.zeros_like(c)

    # d/dc of c^2(1-c)^2 = 2c - 6c^2 + 4c^3
    dW[0] = A * (2*c1 - 6*c1**2 + 4*c1**3) + chi12*c2 + chi13*c3
    dW[1] = A * (2*c2 - 6*c2**2 + 4*c2**3) + chi12*c1 + chi23*c3
    dW[2] = A * (2*c3 - 6*c3**2 + 4*c3**3) + chi13*c1 + chi23*c2

    return dW


def chemical_potential(c, dx, eps=0.01, A=1.0, chi=(1.0, 1.0, 1.0)):
    """
    Projected chemical potential:

        mu_i = dW/dc_i - eps^2 Delta c_i.

    The pointwise projection removes the component-wise mean of mu.
    This keeps the dynamics compatible with c1+c2+c3 = 1.
    """

    mu = bulk_derivative(c, A=A, chi=chi)

    for i in range(3):
        mu[i] -= eps**2 * laplacian_periodic(c[i], dx)

    mu -= np.mean(mu, axis=0, keepdims=True)

    return mu


# ============================================================
# Energy functional
# ============================================================

def energy_functional(c, dx, eps=0.01, A=1.0, chi=(1.0, 1.0, 1.0)):
    """
    Discrete ternary Cahn--Hilliard energy:

        E(c) = integral [ W(c) + eps^2/2 sum_i |grad c_i|^2 ] dx.
    """

    c1, c2, c3 = c
    chi12, chi13, chi23 = chi

    W = (
        A * np.sum(c**2 * (1.0 - c)**2, axis=0)
        + chi12 * c1 * c2
        + chi13 * c1 * c3
        + chi23 * c2 * c3
    )

    grad_part = 0.0

    for i in range(3):
        cx = (np.roll(c[i], -1, axis=0) - c[i]) / dx
        cy = (np.roll(c[i], -1, axis=1) - c[i]) / dx
        grad_part += 0.5 * eps**2 * (cx**2 + cy**2)

    return dx**2 * np.sum(W + grad_part)


# ============================================================
# Eyre noisy initial condition
# ============================================================

def initialize_ternary(N, means=(0.4, 0.4, 0.2), noise=0.01, seed=None):
    """
    Eyre-style noisy ternary simplex initial condition.

    At each grid point:
        c1 ≈ means[0],
        c2 ≈ means[1],
        c3 = 1 - c1 - c2.

    This gives small random perturbations of the homogeneous mixture while
    enforcing the ternary constraint pointwise.
    """

    rng = np.random.default_rng(seed)

    c = np.zeros((3, N, N), dtype=np.float64)

    c[0] = means[0] + rng.uniform(-noise, noise, size=(N, N))
    c[1] = means[1] + rng.uniform(-noise, noise, size=(N, N))
    c[2] = 1.0 - c[0] - c[1]

    c = np.clip(c, 1.0e-8, 1.0)
    c /= np.sum(c, axis=0, keepdims=True)

    return c


# ============================================================
# Simplex projection and mass correction
# ============================================================

def project_simplex_pointwise(c):
    """
    Project each grid point vector (c1,c2,c3) onto the probability simplex:

        c_i >= 0,
        c1 + c2 + c3 = 1.

    This enforces physical ternary concentrations pointwise.
    """

    shape = c.shape
    assert shape[0] == 3, "Expected c with shape (3,N,N)."

    v = c.reshape(3, -1)

    # Projection onto simplex {x >= 0, sum x = 1}
    u = np.sort(v, axis=0)[::-1]
    cssv = np.cumsum(u, axis=0) - 1.0

    ind = np.arange(1, 4).reshape(3, 1)
    cond = u - cssv / ind > 0

    rho = np.sum(cond, axis=0) - 1
    theta = cssv[rho, np.arange(v.shape[1])] / (rho + 1)

    w = np.maximum(v - theta, 0.0)

    return w.reshape(shape)


def mass_corrected_projection(
    c,
    target_masses,
    max_iter=50,
    mass_tol=1e-10,
):
    """
    Alternate between:
        1. pointwise simplex projection,
        2. global component mass correction.

    This keeps c physical and brings component averages back close to
    target_masses.

    The final result satisfies:
        0 <= c_i <= 1,
        c1+c2+c3 = 1 pointwise,
        mean(c_i) approximately target_masses[i].
    """

    c = project_simplex_pointwise(c)

    target_masses = np.asarray(target_masses)

    for _ in range(max_iter):
        current_masses = np.mean(c, axis=(1, 2))
        drift = target_masses - current_masses

        if np.max(np.abs(drift)) < mass_tol:
            break

        # Uniform correction to each component.
        c = c + drift.reshape(3, 1, 1)

        # Restore physical pointwise ternary constraint.
        c = project_simplex_pointwise(c)

    return c


# ============================================================
# Forward Euler update
# ============================================================

def forward_euler_rhs(c, dx, eps, mobility, A, chi):
    """
    Compute dc/dt for the ternary Cahn--Hilliard system.
    """

    mu = chemical_potential(c, dx, eps=eps, A=A, chi=chi)

    dc = np.zeros_like(c)

    for i in range(3):
        dc[i] = mobility[i] * laplacian_periodic(mu[i], dx)

    return dc


# ============================================================
# Forward Euler solver
# ============================================================

def simulate_ternary_ch_projected(
    N=128,
    L=1.0,
    dt=5e-8,
    steps=50000,
    eps=0.01,
    mobility=(1.0, 1.0, 1.0),
    A=1.0,
    chi=(1.5, 1.5, 1.5),
    means=(0.4, 0.4, 0.2),
    noise=0.03,
    seed=None,
    save_every=500,
    energy_tol=1e-10,
    max_retries=20,
    min_dt=1e-14,
    projection_iters=50,
    mass_tol=1e-10,
    allow_dt_growth=True,
    dt_growth=1.02,
    dt_max=None,
):
    """
    Forward Euler ternary Cahn--Hilliard simulation with mass-corrected
    physical projection.

    A proposed step is:
        c_trial = c + dt * RHS(c)

    Then c_trial is projected back into the physical ternary simplex while
    approximately preserving initial component masses.

    If the projected step increases energy beyond energy_tol, the step is
    rejected and retried with half the time step.

    Returns:
        snapshots:    shape (num_snapshots, 3, N, N)
        masses:       shape (num_snapshots, 3)
        energies:     shape (num_snapshots,)
        mins:         shape (num_snapshots, 3)
        maxs:         shape (num_snapshots, 3)
        accepted_dts: shape (steps,)
    """

    dx = L / N

    c = initialize_ternary(N, means=means, seed=seed)
    c = mass_corrected_projection(
        c,
        target_masses=np.array(means),
        max_iter=projection_iters,
        mass_tol=mass_tol,
    )

    target_masses = np.mean(c, axis=(1, 2))

    mobility = np.array(mobility).reshape(3, 1, 1)

    if dt_max is None:
        dt_max = dt

    snapshots = []
    masses = []
    energies = []
    mins = []
    maxs = []
    accepted_dts = []

    current_dt = dt
    current_energy = energy_functional(c, dx, eps=eps, A=A, chi=chi)

    def record(step=None):
        snapshots.append(c.copy())
        masses.append(np.mean(c, axis=(1, 2)))
        energies.append(energy_functional(c, dx, eps=eps, A=A, chi=chi))
        mins.append(np.min(c, axis=(1, 2)))
        maxs.append(np.max(c, axis=(1, 2)))

        # Save image immediately when snapshot is recorded
        if step is not None:
            filename = f"{OUT_DIR}/frame_step_{step:06d}.png"
            save_ternary(c, filename)

    record(step=0)
    
    for n in range(1, steps + 1):

        accepted = False

        for _ in range(max_retries + 1):

            dc = forward_euler_rhs(
                c,
                dx=dx,
                eps=eps,
                mobility=mobility,
                A=A,
                chi=chi,
            )

            c_trial = c + current_dt * dc

            c_trial = mass_corrected_projection(
                c_trial,
                target_masses=target_masses,
                max_iter=projection_iters,
                mass_tol=mass_tol,
            )

            trial_energy = energy_functional(c_trial, dx, eps=eps, A=A, chi=chi)

            # Energy guard: accept nearly flat or decreasing energy.
            if trial_energy <= current_energy + energy_tol:
                c = c_trial
                current_energy = trial_energy
                accepted = True
                accepted_dts.append(current_dt)

                if allow_dt_growth:
                    current_dt = min(dt_max, current_dt * dt_growth)

                break

            current_dt *= 0.5

            if current_dt < min_dt:
                raise RuntimeError(
                    f"dt fell below min_dt at step {n}. "
                    f"Try smaller dt, smaller noise, larger energy_tol, "
                    f"or use a semi-implicit method."
                )

        if not accepted:
            raise RuntimeError(f"Failed to accept a stable step at step {n}.")

        if n % save_every == 0:
            record(step=n)

    return (
        np.array(snapshots),
        np.array(masses),
        np.array(energies),
        np.array(mins),
        np.array(maxs),
        np.array(accepted_dts),
    )


# ============================================================
# Image saving
# ============================================================

def save_ternary(c, filename, dpi=150):
    """
    Save ternary concentration field as an RGB image.

    Mapping:
        c1 -> Red
        c2 -> Green
        c3 -> Blue

    Assumes:
        c.shape = (3, N, N)
        0 <= c_i <= 1
        c1 + c2 + c3 = 1
    """

    c1, c2, c3 = c

    rgb = np.stack([c1, c2, c3], axis=-1)

    # Safety clipping
    rgb = np.clip(rgb, 0.0, 1.0)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, origin="lower", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()


def save_energy_plot(energies, filename, dpi=200):
    """
    Save energy plot separately.
    """

    t = np.arange(len(energies))

    plt.figure(figsize=(8, 5))
    plt.plot(t, energies, color="black", linewidth=2.2)
    plt.xlabel("saved snapshot index")
    plt.ylabel("energy")
    plt.title("Ternary Cahn--Hilliard energy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_component_minmax_plot(mins, maxs, filename, dpi=200):
    """
    Save component min/max plot on one shared axis.

        c1: red
        c2: green
        c3: blue

    Minima use dotted lines.
    Maxima use dashed lines.
    """

    t = np.arange(len(mins))

    labels = ["c1", "c2", "c3"]
    colors = ["red", "green", "blue"]

    plt.figure(figsize=(8, 5))

    for i in range(3):
        plt.plot(
            t,
            mins[:, i],
            color=colors[i],
            linestyle=":",
            linewidth=1.8,
            label=f"min {labels[i]}",
        )
        plt.plot(
            t,
            maxs[:, i],
            color=colors[i],
            linestyle="--",
            linewidth=1.8,
            label=f"max {labels[i]}",
        )

    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    plt.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)

    plt.ylim(-0.02, 1.02)
    plt.xlabel("saved snapshot index")
    plt.ylabel("component value")
    plt.title("Component minima and maxima")
    plt.grid(alpha=0.25)
    plt.legend(loc="center right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_dt_plot(accepted_dts, filename, dpi=200):
    """
    Save accepted dt plot.
    """

    t = np.arange(len(accepted_dts))

    plt.figure(figsize=(8, 5))
    plt.plot(t, accepted_dts, color="black", linewidth=1.5)
    plt.xlabel("Euler step")
    plt.ylabel("accepted dt")
    plt.title("Accepted forward Euler time steps")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()

# ============================================================
# Run test
# ============================================================

if __name__ == "__main__":

    snapshots, masses, energies, mins, maxs, accepted_dts = simulate_ternary_ch_projected(
        N=128,
        L=1.0,
        dt=5e-8,
        steps=50000,
        eps=0.01,
        mobility=(1.0, 1.0, 1.0),
        A=1.0,
        chi=(1.5, 1.5, 1.5),
        means=(0.4, 0.4, 0.2),   # Ce--Co--Fe style composition
        noise=0.03,
        seed=1,
        save_every=500,
        energy_tol=1e-10,
        max_retries=20,
        min_dt=1e-14,
        projection_iters=50,
        mass_tol=1e-10,
        allow_dt_growth=True,
        dt_growth=1.02,
        dt_max=5e-8,
    )

    print("Initial masses:", masses[0])
    print("Final masses:  ", masses[-1])
    print("Mass drift:    ", masses[-1] - masses[0])

    print("Initial energy:", energies[0])
    print("Final energy:  ", energies[-1])
    print("Energy change: ", energies[-1] - energies[0])

    print("Final mins:    ", mins[-1])
    print("Final maxs:    ", maxs[-1])

    print("Initial dt:    ", accepted_dts[0])
    print("Final dt:      ", accepted_dts[-1])
    print("Minimum dt:    ", np.min(accepted_dts))

    # Save all saved snapshots as grayscale frames
    for k, c in enumerate(snapshots):
        save_ternary(c, f"{OUT_DIR}/frame_{k:04d}.png")

    # Save separate diagnostic figures
    save_energy_plot(
        energies,
        f"{OUT_DIR}/energy_plot.png",
    )

    save_component_minmax_plot(
        mins,
        maxs,
        f"{OUT_DIR}/component_minmax.png",
    )

    save_dt_plot(
        accepted_dts,
        f"{OUT_DIR}/accepted_dt.png",
    )

    # Save raw arrays
    np.savez(
        "pair_sample_projected.npz",
        u0=snapshots[0],
        uT=snapshots[-1],
        snapshots=snapshots,
        masses=masses,
        energies=energies,
        mins=mins,
        maxs=maxs,
        accepted_dts=accepted_dts,
    )

    print(f"Saved images to: {os.path.abspath(OUT_DIR)}")
    print("Saved energy plot to: images/energy_plot.png")
    print("Saved min/max plot to: images/component_minmax.png")
    print("Saved dt plot to: images/accepted_dt.png")
    print("Saved data to: pair_sample_projected.npz")

