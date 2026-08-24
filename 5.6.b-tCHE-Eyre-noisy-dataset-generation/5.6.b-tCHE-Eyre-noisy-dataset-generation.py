#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified RGB ternary Cahn--Hilliard dataset pipeline.

This single script replaces:

    1. make_data.py
    2. merge_npz.py
    3. check_npz.py

Pipeline
--------
    A. Generate projected Forward Euler ternary Cahn--Hilliard pairs.
    B. Save temporary one-sample files.
    C. Merge temporary files into batch files.
    D. Merge all batch files produced by THIS run into one final dataset.
    E. Validate the final dataset and save preview/diagnostic figures.

Inverse-pair convention
-----------------------
    src = c(T)   final/evolved state
    tar = c(0)   initial state

Stored ternary arrays use channel-first format

    (batch, 3, M, M).

The numerical core is the same projected Forward Euler scheme used in
the source dataset maker:

    * periodic five-point finite-difference Laplacian,
    * polynomial ternary bulk energy,
    * tangent-space projection of the chemical potentials,
    * explicit Forward Euler evolution,
    * mass-corrected pointwise Gibbs-simplex projection,
    * adaptive energy guard with time-step bisection.

The final merged file also retains diagnostics such as component masses,
energies, extrema, accepted dt ranges, and seeds.

Run
---
    python tCHE_dataset_pipeline.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# ============================================================
# USER CONTROLS
# ============================================================

M = 128
L = 1.0

START = 0
END = 39000

BATCH_SIZE = 1000
NSTEPS = 11000

DT = 5.0e-8

EPS = 0.01
A = 1.0
CHI = (1.5, 1.5, 1.5)

# This preserves the numerical model used by the supplied dataset maker.
MOBILITY = (1.0, 1.0, 1.0)

MEANS = (0.4, 0.4, 0.2)
NOISE = 0.01

ENERGY_TOL = 1.0e-10
MAX_RETRIES = 20
MIN_DT = 1.0e-14

PROJECTION_ITERS = 50
MASS_TOL = 1.0e-10

ALLOW_DT_GROWTH = True
DT_GROWTH = 1.02
DT_MAX = DT

LOG_EVERY = 250

DATA_DIR = Path("data")
TMP_DIR = Path("tmp_tch_rgb")
PREVIEW_DIR = Path("dataset_preview")

# Which sample from the final merged dataset to inspect.
PREVIEW_SAMPLE = 0

# Pipeline stages.  Set any of these to False if rerunning only part.
RUN_GENERATION = True
RUN_FINAL_MERGE = True
RUN_PREVIEW = True

# If True, old temporary files from this pipeline are removed before generation.
CLEAR_TMP_AT_START = True

# Batch prefix is made specific to this experiment so final merging cannot
# accidentally ingest unrelated .npz files.
BATCH_STEM = (
    f"128x128_PBC_tCH_eyre_noisy_404020_"
    f"iters{NSTEPS}"
)

FINAL_STEM = (
    f"dataset_128x128_PBC_tCH_eyre_noisy_404020_"
    f"iters{NSTEPS}"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Periodic differential operators
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


# ============================================================
# Free energy and chemical potential
# ============================================================

def bulk_derivative(
    c: np.ndarray,
    A: float = A,
    chi: tuple[float, float, float] = CHI,
) -> np.ndarray:
    """
    Derivative of the polynomial ternary bulk free energy

        W(c)
          = A sum_i c_i^2 (1-c_i)^2
            + chi12 c1 c2
            + chi13 c1 c3
            + chi23 c2 c3.
    """

    c1, c2, c3 = c
    chi12, chi13, chi23 = chi

    dW = np.zeros_like(c)

    dW[0] = (
        A * (2.0*c1 - 6.0*c1**2 + 4.0*c1**3)
        + chi12*c2
        + chi13*c3
    )

    dW[1] = (
        A * (2.0*c2 - 6.0*c2**2 + 4.0*c2**3)
        + chi12*c1
        + chi23*c3
    )

    dW[2] = (
        A * (2.0*c3 - 6.0*c3**2 + 4.0*c3**3)
        + chi13*c1
        + chi23*c2
    )

    return dW


def chemical_potential(
    c: np.ndarray,
    dx: float,
    eps: float = EPS,
    A: float = A,
    chi: tuple[float, float, float] = CHI,
) -> np.ndarray:
    """
    Discrete chemical potential followed by tangent-space projection.
    """

    mu = bulk_derivative(c, A=A, chi=chi)

    for i in range(3):
        mu[i] -= eps**2 * laplacian_periodic(c[i], dx)

    # Pointwise tangent-space projection for c1+c2+c3=1.
    mu -= np.mean(mu, axis=0, keepdims=True)

    return mu


def energy_functional(
    c: np.ndarray,
    dx: float,
    eps: float = EPS,
    A: float = A,
    chi: tuple[float, float, float] = CHI,
) -> float:
    """Discrete polynomial ternary Cahn--Hilliard free energy."""

    c1, c2, c3 = c
    chi12, chi13, chi23 = chi

    W = (
        A * np.sum(c**2 * (1.0 - c)**2, axis=0)
        + chi12*c1*c2
        + chi13*c1*c3
        + chi23*c2*c3
    )

    grad_part = 0.0

    for i in range(3):
        cx = (
            np.roll(c[i], -1, axis=0)
            - c[i]
        ) / dx

        cy = (
            np.roll(c[i], -1, axis=1)
            - c[i]
        ) / dx

        grad_part += (
            0.5
            * eps**2
            * (cx**2 + cy**2)
        )

    return float(
        dx**2
        * np.sum(W + grad_part)
    )


# ============================================================
# Eyre-noisy initial condition
# ============================================================

def initialize_ternary_eyre_noisy(
    M: int = M,
    means: tuple[float, float, float] = MEANS,
    noise: float = NOISE,
    seed: int | None = None,
) -> np.ndarray:
    """
    Eyre-noisy initial condition near the 40--40--20 mean composition.
    """

    rng = np.random.default_rng(seed)

    c = np.zeros(
        (3, M, M),
        dtype=np.float64,
    )

    c[0] = (
        means[0]
        + rng.uniform(
            -noise,
            noise,
            size=(M, M),
        )
    )

    c[1] = (
        means[1]
        + rng.uniform(
            -noise,
            noise,
            size=(M, M),
        )
    )

    c[2] = 1.0 - c[0] - c[1]

    c = np.clip(
        c,
        1.0e-8,
        1.0,
    )

    c /= np.sum(
        c,
        axis=0,
        keepdims=True,
    )

    return c


# ============================================================
# Gibbs-simplex projection
# ============================================================

def project_simplex_pointwise(c: np.ndarray) -> np.ndarray:
    """
    Euclidean projection at each grid point onto

        G = {v in R^3 : v_i >= 0, sum_i v_i = 1}.
    """

    shape = c.shape
    v = c.reshape(3, -1)

    u = np.sort(v, axis=0)[::-1]
    cssv = np.cumsum(u, axis=0) - 1.0
    ind = np.arange(1, 4).reshape(3, 1)

    cond = (
        u
        - cssv / ind
        > 0.0
    )

    rho = (
        np.sum(cond, axis=0)
        - 1
    )

    theta = (
        cssv[
            rho,
            np.arange(v.shape[1]),
        ]
        / (rho + 1.0)
    )

    w = np.maximum(
        v - theta,
        0.0,
    )

    return w.reshape(shape)


def mass_corrected_projection(
    c: np.ndarray,
    target_masses: np.ndarray,
    max_iter: int = PROJECTION_ITERS,
    mass_tol: float = MASS_TOL,
) -> np.ndarray:
    """
    Alternate simplex projection and uniform component shifts to restore
    prescribed component means.
    """

    c = project_simplex_pointwise(c)
    target_masses = np.asarray(
        target_masses,
        dtype=np.float64,
    )

    for _ in range(max_iter):
        current_masses = np.mean(
            c,
            axis=(1, 2),
        )

        drift = (
            target_masses
            - current_masses
        )

        if (
            np.max(np.abs(drift))
            < mass_tol
        ):
            break

        c = (
            c
            + drift.reshape(3, 1, 1)
        )

        c = project_simplex_pointwise(c)

    return c


# ============================================================
# Forward Euler RHS
# ============================================================

def forward_euler_rhs(
    c: np.ndarray,
    dx: float,
    eps: float,
    mobility: tuple[float, float, float] | np.ndarray,
    A: float,
    chi: tuple[float, float, float],
) -> np.ndarray:
    """Explicit projected-chemical-potential ternary CH right-hand side."""

    mu = chemical_potential(
        c,
        dx,
        eps=eps,
        A=A,
        chi=chi,
    )

    mobility_array = np.asarray(
        mobility,
        dtype=np.float64,
    ).reshape(3)

    dc = np.zeros_like(c)

    for i in range(3):
        dc[i] = (
            mobility_array[i]
            * laplacian_periodic(
                mu[i],
                dx,
            )
        )

    return dc


# ============================================================
# Simulate one inverse pair
# ============================================================

def simulate_one_pair(
    M: int = M,
    L: float = L,
    dt: float = DT,
    steps: int = NSTEPS,
    eps: float = EPS,
    mobility: tuple[float, float, float] = MOBILITY,
    A: float = A,
    chi: tuple[float, float, float] = CHI,
    means: tuple[float, float, float] = MEANS,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate one inverse pair

        tar = c(0),
        src = c(T).
    """

    dx = L / M

    c = initialize_ternary_eyre_noisy(
        M=M,
        means=means,
        noise=NOISE,
        seed=seed,
    )

    c = mass_corrected_projection(
        c,
        target_masses=np.asarray(means),
        max_iter=PROJECTION_ITERS,
        mass_tol=MASS_TOL,
    )

    c0 = c.copy()

    # Preserve the actually realized initial masses.
    target_masses = np.mean(
        c0,
        axis=(1, 2),
    )

    current_dt = dt

    current_energy = energy_functional(
        c,
        dx,
        eps=eps,
        A=A,
        chi=chi,
    )

    energy0 = current_energy

    accepted_dt_min = current_dt
    accepted_dt_max = current_dt

    for step in range(steps):
        accepted = False

        for _ in range(
            MAX_RETRIES + 1
        ):
            dc = forward_euler_rhs(
                c,
                dx,
                eps,
                mobility,
                A,
                chi,
            )

            c_trial = (
                c
                + current_dt * dc
            )

            c_trial = mass_corrected_projection(
                c_trial,
                target_masses=target_masses,
                max_iter=PROJECTION_ITERS,
                mass_tol=MASS_TOL,
            )

            trial_energy = energy_functional(
                c_trial,
                dx,
                eps=eps,
                A=A,
                chi=chi,
            )

            if (
                trial_energy
                <= current_energy + ENERGY_TOL
            ):
                c = c_trial
                current_energy = trial_energy
                accepted = True

                accepted_dt_min = min(
                    accepted_dt_min,
                    current_dt,
                )

                accepted_dt_max = max(
                    accepted_dt_max,
                    current_dt,
                )

                if ALLOW_DT_GROWTH:
                    current_dt = min(
                        DT_MAX,
                        current_dt * DT_GROWTH,
                    )

                break

            current_dt *= 0.5

            if current_dt < MIN_DT:
                raise RuntimeError(
                    f"dt below MIN_DT at step {step}"
                )

        if not accepted:
            raise RuntimeError(
                f"Failed stable step at {step}"
            )

    cT = c.copy()

    diagnostics = {
        "mass0": np.mean(
            c0,
            axis=(1, 2),
        ),
        "massT": np.mean(
            cT,
            axis=(1, 2),
        ),
        "energy0": energy0,
        "energyT": current_energy,
        "minT": np.min(
            cT,
            axis=(1, 2),
        ),
        "maxT": np.max(
            cT,
            axis=(1, 2),
        ),
        "dt_min": accepted_dt_min,
        "dt_max": accepted_dt_max,
    }

    return c0, cT, diagnostics


# ============================================================
# Temporary sample files
# ============================================================

def temp_filename(index: int) -> Path:
    return (
        TMP_DIR
        / f"tmp_{index:05d}.npz"
    )


def save_temp_pair(
    index: int,
    src: np.ndarray,
    tar: np.ndarray,
    diagnostics: dict,
    seed: int,
) -> None:
    """Save one temporary inverse pair."""

    np.savez_compressed(
        temp_filename(index),
        src=src.astype(np.float32),
        tar=tar.astype(np.float32),
        mass0=np.asarray(
            diagnostics["mass0"],
            dtype=np.float64,
        ),
        massT=np.asarray(
            diagnostics["massT"],
            dtype=np.float64,
        ),
        energy0=np.asarray(
            diagnostics["energy0"],
            dtype=np.float64,
        ),
        energyT=np.asarray(
            diagnostics["energyT"],
            dtype=np.float64,
        ),
        minT=np.asarray(
            diagnostics["minT"],
            dtype=np.float64,
        ),
        maxT=np.asarray(
            diagnostics["maxT"],
            dtype=np.float64,
        ),
        dt_min=np.asarray(
            diagnostics["dt_min"],
            dtype=np.float64,
        ),
        dt_max=np.asarray(
            diagnostics["dt_max"],
            dtype=np.float64,
        ),
        seed=np.asarray(
            seed,
            dtype=np.int64,
        ),
    )


def clear_temporary_files() -> None:
    """Remove only tmp_*.npz files from this pipeline."""

    for path in TMP_DIR.glob(
        "tmp_*.npz"
    ):
        path.unlink()


# ============================================================
# Temporary -> batch merge
# ============================================================

def batch_path(
    batch_idx: int,
) -> Path:
    return (
        DATA_DIR
        / f"{BATCH_STEM}_batch_{batch_idx:03d}.npz"
    )


def merge_temp_to_batch(
    batch_idx: int,
) -> Path | None:
    """Merge all current temporary sample files into one batch."""

    files = sorted(
        TMP_DIR.glob("tmp_*.npz")
    )

    if not files:
        return None

    fields = {
        "src": [],
        "tar": [],
        "mass0": [],
        "massT": [],
        "energy0": [],
        "energyT": [],
        "minT": [],
        "maxT": [],
        "dt_min": [],
        "dt_max": [],
        "seed": [],
    }

    for path in files:
        with np.load(path) as data:
            for key in fields:
                fields[key].append(
                    np.asarray(data[key])
                )

    src = np.stack(
        fields["src"],
        axis=0,
    ).astype(np.float32)

    tar = np.stack(
        fields["tar"],
        axis=0,
    ).astype(np.float32)

    out_file = batch_path(
        batch_idx
    )

    np.savez_compressed(
        out_file,
        src=src,
        tar=tar,
        mass0=np.stack(
            fields["mass0"],
            axis=0,
        ),
        massT=np.stack(
            fields["massT"],
            axis=0,
        ),
        energy0=np.stack(
            fields["energy0"],
            axis=0,
        ),
        energyT=np.stack(
            fields["energyT"],
            axis=0,
        ),
        minT=np.stack(
            fields["minT"],
            axis=0,
        ),
        maxT=np.stack(
            fields["maxT"],
            axis=0,
        ),
        dt_min=np.stack(
            fields["dt_min"],
            axis=0,
        ),
        dt_max=np.stack(
            fields["dt_max"],
            axis=0,
        ),
        seeds=np.stack(
            fields["seed"],
            axis=0,
        ),
        M=np.asarray(M),
        L=np.asarray(L),
        steps=np.asarray(NSTEPS),
        dt=np.asarray(DT),
        eps=np.asarray(EPS),
        means=np.asarray(MEANS),
        chi=np.asarray(CHI),
        mobility=np.asarray(MOBILITY),
    )

    for path in files:
        path.unlink()

    print()
    print(
        f"[Batch {batch_idx}] saved:"
    )
    print(
        f"  {out_file}"
    )
    print(
        f"  src shape = {src.shape}"
    )

    return out_file


# ============================================================
# Dataset generation stage
# ============================================================

def generate_dataset_batches() -> list[Path]:
    """Generate all samples and save batch files."""

    if CLEAR_TMP_AT_START:
        clear_temporary_files()

    print()
    print("=" * 72)
    print("STAGE 1: GENERATE DATASET BATCHES")
    print("=" * 72)
    print(f"M              = {M}")
    print(f"L              = {L}")
    print(f"dx             = {L/M:.8e}")
    print(f"NSTEPS         = {NSTEPS}")
    print(f"DT             = {DT:.8e}")
    print(f"nominal T      = {NSTEPS*DT:.8e}")
    print(f"START          = {START}")
    print(f"END            = {END}")
    print(f"BATCH_SIZE     = {BATCH_SIZE}")
    print(f"batch stem     = {BATCH_STEM}")

    batch_idx = 0
    tmp_counter = 0
    batch_files: list[Path] = []

    for run in range(
        START,
        END,
    ):
        seed = run

        try:
            c0, cT, diagnostics = (
                simulate_one_pair(
                    M=M,
                    L=L,
                    dt=DT,
                    steps=NSTEPS,
                    eps=EPS,
                    mobility=MOBILITY,
                    A=A,
                    chi=CHI,
                    means=MEANS,
                    seed=seed,
                )
            )

        except RuntimeError as error:
            print(
                f"[seed {seed}] skipped: {error}"
            )
            continue

        # Inverse-learning convention.
        src = cT
        tar = c0

        save_temp_pair(
            tmp_counter,
            src,
            tar,
            diagnostics,
            seed,
        )

        tmp_counter += 1

        if tmp_counter == BATCH_SIZE:
            path = merge_temp_to_batch(
                batch_idx
            )

            if path is not None:
                batch_files.append(path)

            batch_idx += 1
            tmp_counter = 0

        if (
            (run + 1) % LOG_EVERY
            == 0
        ):
            print(
                f"{run+1}/{END} attempted"
            )

    if tmp_counter > 0:
        path = merge_temp_to_batch(
            batch_idx
        )

        if path is not None:
            batch_files.append(path)

    print()
    print(
        f"Generation complete: "
        f"{len(batch_files)} batch file(s)."
    )

    return batch_files


# ============================================================
# Shape handling for final merge / preview
# ============================================================

def normalize_ternary_array(
    x: np.ndarray,
    name: str = "array",
) -> np.ndarray:
    """
    Normalize ternary arrays to

        (batch, 3, M, M).
    """

    x = np.asarray(x)

    if (
        x.ndim == 3
        and x.shape[0] == 3
    ):
        x = x[np.newaxis, ...]

    elif (
        x.ndim == 4
        and x.shape[1] == 3
    ):
        pass

    elif (
        x.ndim == 3
        and x.shape[-1] == 3
    ):
        x = np.transpose(
            x,
            (2, 0, 1),
        )[np.newaxis, ...]

    elif (
        x.ndim == 4
        and x.shape[-1] == 3
    ):
        x = np.transpose(
            x,
            (0, 3, 1, 2),
        )

    else:
        raise ValueError(
            f"Cannot interpret {name} shape "
            f"{x.shape} as ternary RGB data."
        )

    if (
        x.shape[2] != M
        or x.shape[3] != M
    ):
        raise ValueError(
            f"Unexpected spatial shape for {name}: "
            f"{x.shape}; expected (*,3,{M},{M})."
        )

    return x.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Batch -> final dataset merge
# ============================================================

def discover_pipeline_batches() -> list[Path]:
    """
    Return only batch files associated with this experiment.
    """

    pattern = str(
        DATA_DIR
        / f"{BATCH_STEM}_batch_*.npz"
    )

    return [
        Path(path)
        for path in sorted(
            glob.glob(pattern)
        )
    ]


def merge_batches_to_final(
    batch_files: list[Path] | None = None,
) -> Path:
    """
    Merge this experiment's batch files into one final dataset.

    Unlike the old standalone merge script, diagnostic arrays are
    retained whenever they are present in every batch.
    """

    print()
    print("=" * 72)
    print("STAGE 2: MERGE BATCHES")
    print("=" * 72)

    if not batch_files:
        batch_files = discover_pipeline_batches()

    if not batch_files:
        raise FileNotFoundError(
            "No matching pipeline batch files found."
        )

    print(
        f"Found {len(batch_files)} batch file(s)."
    )

    all_src = []
    all_tar = []

    optional_keys = [
        "mass0",
        "massT",
        "energy0",
        "energyT",
        "minT",
        "maxT",
        "dt_min",
        "dt_max",
        "seeds",
    ]

    optional_data = {
        key: []
        for key in optional_keys
    }

    have_key = {
        key: True
        for key in optional_keys
    }

    for index, path in enumerate(
        batch_files,
        start=1,
    ):
        print(
            f"[{index}/{len(batch_files)}] "
            f"Loading {path}"
        )

        with np.load(
            path,
            mmap_mode="r",
        ) as data:
            if (
                "src" not in data
                or "tar" not in data
            ):
                raise KeyError(
                    f"{path} does not contain "
                    "both src and tar."
                )

            src = normalize_ternary_array(
                data["src"],
                name=f"src from {path}",
            )

            tar = normalize_ternary_array(
                data["tar"],
                name=f"tar from {path}",
            )

            if (
                src.shape[0]
                != tar.shape[0]
            ):
                raise ValueError(
                    f"Batch-size mismatch in {path}: "
                    f"src={src.shape}, tar={tar.shape}"
                )

            if (
                np.any(np.isnan(src))
                or np.any(np.isnan(tar))
            ):
                raise ValueError(
                    f"NaNs detected in {path}"
                )

            all_src.append(src)
            all_tar.append(tar)

            for key in optional_keys:
                if key in data:
                    optional_data[key].append(
                        np.asarray(data[key])
                    )
                else:
                    have_key[key] = False

    X_src = np.concatenate(
        all_src,
        axis=0,
    )

    X_tar = np.concatenate(
        all_tar,
        axis=0,
    )

    count = X_src.shape[0]

    final_file = (
        DATA_DIR
        / f"{FINAL_STEM}_count{count}.npz"
    )

    save_dict = {
        "src": X_src,
        "tar": X_tar,
        "M": np.asarray(M),
        "L": np.asarray(L),
        "steps": np.asarray(NSTEPS),
        "dt": np.asarray(DT),
        "eps": np.asarray(EPS),
        "A": np.asarray(A),
        "means": np.asarray(MEANS),
        "chi": np.asarray(CHI),
        "mobility": np.asarray(MOBILITY),
    }

    for key in optional_keys:
        if (
            have_key[key]
            and len(optional_data[key])
            == len(batch_files)
        ):
            save_dict[key] = np.concatenate(
                optional_data[key],
                axis=0,
            )

    print()
    print("Final merged shapes:")
    print("  src:", X_src.shape)
    print("  tar:", X_tar.shape)

    print()
    print("Global min/max:")
    print(
        "  src:",
        float(np.min(X_src)),
        float(np.max(X_src)),
    )
    print(
        "  tar:",
        float(np.min(X_tar)),
        float(np.max(X_tar)),
    )

    print()
    print("Component means:")
    print(
        "  src:",
        np.mean(
            X_src,
            axis=(0, 2, 3),
        ),
    )
    print(
        "  tar:",
        np.mean(
            X_tar,
            axis=(0, 2, 3),
        ),
    )

    print()
    print(
        "Simplex check max |sum_i c_i - 1|:"
    )
    print(
        "  src:",
        float(
            np.max(
                np.abs(
                    np.sum(
                        X_src,
                        axis=1,
                    )
                    - 1.0
                )
            )
        ),
    )
    print(
        "  tar:",
        float(
            np.max(
                np.abs(
                    np.sum(
                        X_tar,
                        axis=1,
                    )
                    - 1.0
                )
            )
        ),
    )

    np.savez_compressed(
        final_file,
        **save_dict,
    )

    print()
    print(
        f"Final dataset saved to:"
    )
    print(
        f"  {final_file}"
    )

    return final_file


# ============================================================
# Preview plotting helpers
# ============================================================

def ternary_rgb(
    c: np.ndarray,
) -> np.ndarray:
    rgb = np.stack(
        [
            c[0],
            c[1],
            c[2],
        ],
        axis=-1,
    )

    return np.clip(
        rgb,
        0.0,
        1.0,
    )


def save_ternary(
    c: np.ndarray,
    filename: Path,
    dpi: int = 220,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.imshow(
        ternary_rgb(c),
        origin="lower",
        interpolation="nearest",
    )

    ax.axis("off")

    if title is not None:
        ax.set_title(title)

    fig.savefig(
        filename,
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
    )

    plt.close(fig)


def save_phase_map(
    phase: np.ndarray,
    filename: Path,
    dpi: int = 220,
    title: str | None = None,
) -> None:
    cmap = ListedColormap(
        [
            "red",
            "lime",
            "blue",
        ]
    )

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.imshow(
        phase,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=2,
    )

    ax.axis("off")

    if title is not None:
        ax.set_title(title)

    fig.savefig(
        filename,
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
    )

    plt.close(fig)


def save_triplet(
    src: np.ndarray,
    tar: np.ndarray,
    filename: Path,
    dpi: int = 220,
) -> None:
    """
    Save final RGB, initial RGB, and componentwise absolute difference.
    """

    # Absolute difference is still a three-channel field and is displayed
    # through the same RGB channel convention.
    difference = np.abs(
        src - tar
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
    )

    items = [
        (
            src,
            r"src = final/evolved $c_T$",
        ),
        (
            tar,
            r"tar = initial $c_0$",
        ),
        (
            difference,
            r"$|c_T-c_0|$",
        ),
    ]

    for ax, (
        field,
        title,
    ) in zip(
        axes,
        items,
    ):
        ax.imshow(
            ternary_rgb(field),
            origin="lower",
            interpolation="nearest",
        )

        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        filename,
        bbox_inches="tight",
        dpi=dpi,
    )

    plt.close(fig)


def save_component_panel(
    c: np.ndarray,
    filename: Path,
    title_prefix: str,
    dpi: int = 220,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
    )

    names = [
        "c1 / red",
        "c2 / green",
        "c3 / blue",
    ]

    for i, ax in enumerate(axes):
        im = ax.imshow(
            c[i],
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(
            f"{title_prefix}: {names[i]}"
        )

        ax.axis("off")

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

    fig.tight_layout()

    fig.savefig(
        filename,
        bbox_inches="tight",
        dpi=dpi,
    )

    plt.close(fig)


# ============================================================
# Final dataset validation / preview
# ============================================================

def get_sample_value(
    data,
    key: str,
    number: int,
):
    if key not in data:
        return None

    arr = np.asarray(
        data[key]
    )

    if arr.ndim == 0:
        return arr.item()

    return arr[number]


def preview_final_dataset(
    final_file: Path,
    number: int = PREVIEW_SAMPLE,
) -> None:
    """Validate one sample and save preview figures."""

    print()
    print("=" * 72)
    print("STAGE 3: VALIDATE / PREVIEW FINAL DATASET")
    print("=" * 72)

    with np.load(
        final_file,
        mmap_mode="r",
    ) as data:
        print(
            "Keys:",
            list(data.keys()),
        )

        if (
            "src" not in data
            or "tar" not in data
        ):
            raise KeyError(
                "Final dataset must contain src and tar."
            )

        src_all = normalize_ternary_array(
            data["src"],
            name="src",
        )

        tar_all = normalize_ternary_array(
            data["tar"],
            name="tar",
        )

        if (
            np.any(np.isnan(src_all))
            or np.any(np.isnan(tar_all))
        ):
            raise ValueError(
                "NaNs found in final dataset."
            )

        batch_size = src_all.shape[0]

        if (
            number < 0
            or number >= batch_size
        ):
            raise IndexError(
                f"PREVIEW_SAMPLE={number} is out "
                f"of range for count={batch_size}."
            )

        src = np.asarray(
            src_all[number],
            dtype=np.float64,
        )

        tar = np.asarray(
            tar_all[number],
            dtype=np.float64,
        )

        _, n1, n2 = src.shape

        if n1 != n2:
            raise ValueError(
                f"Expected square sample; got {src.shape}."
            )

        dx = L / n1

        print()
        print("Final file:")
        print(
            f"  {final_file}"
        )

        print()
        print("Dataset shapes:")
        print(
            "  src:",
            src_all.shape,
        )
        print(
            "  tar:",
            tar_all.shape,
        )

        print()
        print(
            f"Preview sample: {number}"
        )

        print()
        print("=== GLOBAL STATS ===")
        print(
            "src max/min:",
            np.max(src_all),
            np.min(src_all),
        )
        print(
            "tar max/min:",
            np.max(tar_all),
            np.min(tar_all),
        )
        print(
            "src mean/var:",
            np.mean(src_all),
            np.var(src_all),
        )
        print(
            "tar mean/var:",
            np.mean(tar_all),
            np.var(tar_all),
        )

        print()
        print(
            "=== COMPONENT STATS: SRC / FINAL ==="
        )

        for i in range(3):
            print(
                f"c{i+1}: "
                f"mean={np.mean(src[i]):.8f}, "
                f"min={np.min(src[i]):.8f}, "
                f"max={np.max(src[i]):.8f}, "
                f"var={np.var(src[i]):.8f}"
            )

        print()
        print(
            "=== COMPONENT STATS: TAR / INITIAL ==="
        )

        for i in range(3):
            print(
                f"c{i+1}: "
                f"mean={np.mean(tar[i]):.8f}, "
                f"min={np.min(tar[i]):.8f}, "
                f"max={np.max(tar[i]):.8f}, "
                f"var={np.var(tar[i]):.8f}"
            )

        print()
        print("=== SIMPLEX CHECK ===")

        src_sum = np.sum(
            src,
            axis=0,
        )

        tar_sum = np.sum(
            tar,
            axis=0,
        )

        print(
            "max |sum(src)-1|:",
            np.max(
                np.abs(
                    src_sum - 1.0
                )
            ),
        )

        print(
            "max |sum(tar)-1|:",
            np.max(
                np.abs(
                    tar_sum - 1.0
                )
            ),
        )

        print(
            "src range:",
            float(src.min()),
            float(src.max()),
        )

        print(
            "tar range:",
            float(tar.min()),
            float(tar.max()),
        )

        print()
        print("=== MASS DRIFT ===")

        mass_src = np.mean(
            src,
            axis=(1, 2),
        )

        mass_tar = np.mean(
            tar,
            axis=(1, 2),
        )

        print(
            "mass tar / initial:",
            mass_tar,
        )
        print(
            "mass src / final:  ",
            mass_src,
        )
        print(
            "mass drift:        ",
            mass_src - mass_tar,
        )

        print()
        print("=== ENERGY CHECK ===")

        E_tar = energy_functional(
            tar,
            dx=dx,
        )

        E_src = energy_functional(
            src,
            dx=dx,
        )

        print(
            "recomputed E(tar initial):",
            E_tar,
        )
        print(
            "recomputed E(src final):  ",
            E_src,
        )
        print(
            "recomputed energy change: ",
            E_src - E_tar,
        )

        energy0_val = get_sample_value(
            data,
            "energy0",
            number,
        )

        energyT_val = get_sample_value(
            data,
            "energyT",
            number,
        )

        print(
            "stored energy0:",
            energy0_val,
        )
        print(
            "stored energyT:",
            energyT_val,
        )

        if (
            energy0_val is not None
            and energyT_val is not None
        ):
            print(
                "stored energy change:",
                energyT_val
                - energy0_val,
            )

        src_phase = np.argmax(
            src,
            axis=0,
        )

        tar_phase = np.argmax(
            tar,
            axis=0,
        )

        print()
        print("=== PHASE FRACTIONS ===")

        for label, phase in [
            ("src", src_phase),
            ("tar", tar_phase),
        ]:
            counts = np.asarray(
                [
                    (phase == i).sum()
                    for i in range(3)
                ]
            )

            print(
                f"{label} argmax fractions:",
                counts / counts.sum(),
            )

        save_ternary(
            src,
            PREVIEW_DIR / "phiNE.png",
            title="SRC final/evolved RGB",
        )

        save_ternary(
            tar,
            PREVIEW_DIR / "phiIC.png",
            title="TAR initial RGB",
        )

        save_phase_map(
            src_phase,
            PREVIEW_DIR / "phiNE_phase.png",
            title="SRC final argmax phase",
        )

        save_phase_map(
            tar_phase,
            PREVIEW_DIR / "phiIC_phase.png",
            title="TAR initial argmax phase",
        )

        save_triplet(
            src,
            tar,
            PREVIEW_DIR
            / "triplet_src_tar_difference.png",
        )

        save_component_panel(
            src,
            PREVIEW_DIR
            / "src_components.png",
            "src final",
        )

        save_component_panel(
            tar,
            PREVIEW_DIR
            / "tar_components.png",
            "tar initial",
        )

    print()
    print(
        "Preview images saved to:"
    )
    print(
        f"  {PREVIEW_DIR.resolve()}"
    )


# ============================================================
# Pipeline
# ============================================================

def main() -> None:
    print()
    print("=" * 72)
    print("TERNARY CAHN--HILLIARD DATASET PIPELINE")
    print("=" * 72)

    batch_files = None
    final_file = None

    if RUN_GENERATION:
        batch_files = (
            generate_dataset_batches()
        )

    if RUN_FINAL_MERGE:
        final_file = (
            merge_batches_to_final(
                batch_files
            )
        )

    if RUN_PREVIEW:
        if final_file is None:
            candidates = sorted(
                DATA_DIR.glob(
                    f"{FINAL_STEM}_count*.npz"
                )
            )

            if not candidates:
                raise FileNotFoundError(
                    "No final merged dataset was found "
                    "for the preview stage."
                )

            final_file = candidates[-1]

        preview_final_dataset(
            final_file,
            PREVIEW_SAMPLE,
        )

    print()
    print("=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)

    if final_file is not None:
        print(
            f"Final dataset: {final_file}"
        )


if __name__ == "__main__":
    main()
