#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full testing/evaluation script for inverse Chafee--Infante reconstruction.

Inverse problem
---------------
Given the forward-evolved state

    u_T = Phi_T(u_0),

a trained generator reconstructs an approximation

    u_hat_0 = G(u_T)

to the unknown initial condition u_0.

The PDE is

    u_t - gamma Delta u + kappa (u^3 - u) = 0

on

    Omega = [-L,L] x [-L,L],

with homogeneous Dirichlet boundary conditions.

Residual horizon
----------------
The dynamical residual uses the equal-time short-horizon definition

    R_tau(u_hat_0,u_0)
        =
        || Phi_tau(u_hat_0) - Phi_tau(u_0) ||_1,

where

    tau = RESIDUAL_STEPS * DT.

Both initial conditions are evolved with the same differentiable/numerical
Forward Euler surrogate.  The residual does NOT compare a short rollout
with the full terminal state u_T.

Empirical solution criterion
----------------------------
For tolerances epsilon > 0 and delta > 0, define

    A_epsilon = { MAE(u_hat_0,u_0) <= epsilon },

    B_delta   = { R_tau(u_hat_0,u_0) <= delta }.

A tested model satisfies the empirical (epsilon,delta,X)-solution criterion
when

    P_hat(A_epsilon intersect B_delta) >= X.

The script reports the empirical probability, a Wilson 95% confidence
interval, the marginal probabilities, conditional probabilities, and the
MAE/residual correlation.

Threshold policy
----------------
If --fixed-eps and --fixed-delta are supplied, those values are used.

Otherwise epsilon and delta are calibrated ONCE from a reference model
(default: best_mae) using the requested empirical quantiles.  The same
thresholds are then applied to every evaluated model, allowing a meaningful
best-MAE versus best-residual comparison.

Diagnostics
-----------
Per test sample, the script computes

    - reconstruction MAE (scaled and physical),
    - short-horizon residual (scaled and physical),
    - relative Lyapunov-energy error,
    - mean error,
    - variance error,
    - first-difference error,
    - generator minimum and maximum.

Outputs
-------
For each evaluated model:
    metrics_per_sample.csv
    report.json
    qualitative.png
    individual_images/
        src0.png, gen0.png, tar0.png
        src1.png, gen1.png, tar1.png
        src2.png, gen2.png, tar2.png
    *_hist.png
    *_cdf.png
    mae_vs_residual.png

At the top level:
    comparison_summary.txt
    comparison_summary.csv
    solution_definition.json

Typical use:
python evaluate_inverse_CIRDE.py \
    --train-npz ../../../../CIRDE_Datasets/128x128_kappa=4.7_DBC_Poisson_smooth_data_400iterations/train-dataset_128x128_DBC_kappa=4.7_PS400_count=50000.npz \
    --test-npz ../../../../CIRDE_Datasets/128x128_kappa=4.7_DBC_Poisson_smooth_data_400iterations/test-dataset_128x128_DBC_kappa=4.7_PS400_count=10000.npz \
    --best-mae-model training_output/models/best_generator_mae.keras \
    --best-residual-model training_output/models/best_generator_residual.keras
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full test evaluator for inverse Chafee--Infante generators "
            "with an equal-time short-horizon residual."
        )
    )

    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--test-npz", required=True)

    parser.add_argument(
        "--model-dir",
        default="training_output/models",
        help="Directory containing best_generator_mae.keras and optionally "
             "best_generator_residual.keras.",
    )
    parser.add_argument(
        "--best-mae-model",
        default=None,
        help="Optional explicit path to the best-MAE generator.",
    )
    parser.add_argument(
        "--best-residual-model",
        default=None,
        help="Optional explicit path to the best-residual generator.",
    )

    parser.add_argument(
        "--outdir",
        default="full_model_evaluation_CIRDE",
    )

    # PDE/discretization parameters.
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.005)
    parser.add_argument("--kappa", type=float, default=4.7)
    parser.add_argument("--dataset-steps", type=int, default=400)
    parser.add_argument("--residual-steps", type=int, default=100)

    # Evaluation batching.
    parser.add_argument("--predict-batch", type=int, default=16)
    parser.add_argument("--residual-batch", type=int, default=16)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Test-sample indices for qualitative plots.",
    )

    # Empirical solution definition.
    parser.add_argument("--target-p-mae", type=float, default=0.95)
    parser.add_argument("--target-p-res", type=float, default=0.95)
    parser.add_argument("--target-p-joint", type=float, default=0.90)

    parser.add_argument("--fixed-eps", type=float, default=None)
    parser.add_argument("--fixed-delta", type=float, default=None)

    parser.add_argument(
        "--threshold-reference",
        choices=["best_mae", "best_residual"],
        default="best_mae",
        help="Reference model used to calibrate epsilon and delta when "
             "fixed thresholds are not supplied.",
    )

    return parser.parse_args()


# =============================================================================
# Keras custom layer used by the saved generator
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="MLEE")
class EnforceDirichletBoundary(Layer):
    """
    Set all image-boundary values to zero.
    """

    def call(self, inputs):
        interior = inputs[:, 1:-1, 1:-1, :]

        middle = tf.pad(
            interior,
            paddings=[
                [0, 0],
                [0, 0],
                [1, 1],
                [0, 0],
            ],
            mode="CONSTANT",
        )

        output = tf.pad(
            middle,
            paddings=[
                [0, 0],
                [1, 1],
                [0, 0],
                [0, 0],
            ],
            mode="CONSTANT",
        )

        return tf.ensure_shape(output, inputs.shape)

    def get_config(self):
        return super().get_config()


# =============================================================================
# Dataset utilities
# =============================================================================

def prepare_array(array: np.ndarray, M: int) -> np.ndarray:
    """
    Convert a stored scalar-field dataset to shape (N,M,M,1).
    """
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 3:
        array = array[..., np.newaxis]

    elif array.ndim == 4 and array.shape[-1] == 1:
        pass

    elif array.ndim == 4 and array.shape[1] == 1:
        array = np.transpose(array, (0, 2, 3, 1))

    else:
        raise ValueError(f"Unsupported dataset shape: {array.shape}")

    if array.shape[1:] != (M, M, 1):
        raise ValueError(
            f"Expected image shape ({M},{M},1); received {array.shape[1:]}."
        )

    return array


def load_dataset(
    train_path: str,
    test_path: str,
    M: int,
):
    """
    Load train/test data and reproduce the training-set scaling convention.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Testing dataset not found: {test_path}")

    with np.load(train_path, mmap_mode="r") as data:
        train_src = prepare_array(data["src"], M)
        train_tar = prepare_array(data["tar"], M)

    with np.load(test_path, mmap_mode="r") as data:
        test_src = prepare_array(data["src"], M)
        test_tar = prepare_array(data["tar"], M)

    ne_scale = float(np.max(np.abs(train_src)))
    ic_scale = float(np.max(np.abs(train_tar)))

    if ne_scale <= 0.0 or ic_scale <= 0.0:
        raise ValueError("Dataset scaling factor is zero.")

    test_A = (test_src / ne_scale).astype(np.float32, copy=False)
    test_B = (test_tar / ic_scale).astype(np.float32, copy=False)

    return test_A, test_B, ne_scale, ic_scale


# =============================================================================
# Statistics
# =============================================================================

def wilson_ci(k: int, n: int, z: float = 1.96):
    """
    Wilson 95% confidence interval for a binomial proportion.
    """
    if n <= 0:
        return np.nan, np.nan, np.nan

    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = (
        z
        / denom
        * np.sqrt(
            p * (1.0 - p) / n
            + z**2 / (4.0 * n**2)
        )
    )

    return (
        p,
        max(0.0, center - half),
        min(1.0, center + half),
    )


def metric_summary(values: np.ndarray) -> Dict[str, float]:
    """
    Standard descriptive statistics.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]

    if v.size == 0:
        return {
            "N": 0,
            "mean": np.nan,
            "std": np.nan,
            "var": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "q05": np.nan,
            "q95": np.nan,
        }

    return {
        "N": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "var": float(np.var(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "median": float(np.median(v)),
        "q05": float(np.quantile(v, 0.05)),
        "q95": float(np.quantile(v, 0.95)),
    }


def quantile_threshold(values: np.ndarray, target_p: float) -> float:
    """
    Empirical left-tail threshold achieving at least target_p.
    """
    if not 0.0 < target_p <= 1.0:
        raise ValueError("target probability must lie in (0,1].")

    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]

    if v.size == 0:
        return np.nan

    v = np.sort(v)
    idx = int(np.ceil(target_p * v.size)) - 1
    idx = max(0, min(idx, v.size - 1))

    return float(v[idx])


def probability_report(values: np.ndarray, threshold: float):
    """
    Empirical P(value <= threshold) with Wilson 95% interval.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    mask = np.isfinite(v)
    v = v[mask]

    k = int(np.sum(v <= threshold))
    n = int(v.size)

    p, lo, hi = wilson_ci(k, n)

    return {
        "threshold": float(threshold),
        "k": k,
        "n": n,
        "p": float(p),
        "wilson95": [float(lo), float(hi)],
    }


def joint_solution_report(
    mae: np.ndarray,
    residual: np.ndarray,
    epsilon: float,
    delta: float,
    target_x: float,
):
    """
    Compute the empirical (epsilon,delta,X)-solution statistics.
    """
    mae = np.asarray(mae, dtype=np.float64).ravel()
    residual = np.asarray(residual, dtype=np.float64).ravel()

    if mae.size != residual.size:
        raise ValueError("MAE and residual arrays must have equal length.")

    finite = np.isfinite(mae) & np.isfinite(residual)
    mae = mae[finite]
    residual = residual[finite]

    A = mae <= epsilon
    B = residual <= delta
    J = A & B

    n = int(mae.size)
    kA = int(np.sum(A))
    kB = int(np.sum(B))
    kJ = int(np.sum(J))

    pA = wilson_ci(kA, n)
    pB = wilson_ci(kB, n)
    pJ = wilson_ci(kJ, n)

    corr = np.nan
    if (
        n > 1
        and np.std(mae) > 0.0
        and np.std(residual) > 0.0
    ):
        corr = float(np.corrcoef(mae, residual)[0, 1])

    return {
        "epsilon": float(epsilon),
        "delta": float(delta),
        "X": float(target_x),
        "P_MAE_leq_epsilon": {
            "p": float(pA[0]),
            "wilson95": [float(pA[1]), float(pA[2])],
            "k": kA,
            "n": n,
        },
        "P_RES_leq_delta": {
            "p": float(pB[0]),
            "wilson95": [float(pB[1]), float(pB[2])],
            "k": kB,
            "n": n,
        },
        "P_joint": {
            "p": float(pJ[0]),
            "wilson95": [float(pJ[1]), float(pJ[2])],
            "k": kJ,
            "n": n,
        },
        "P_RES_given_MAE": float(kJ / kA) if kA else np.nan,
        "P_MAE_given_RES": float(kJ / kB) if kB else np.nan,
        "corr_MAE_RES": corr,
        "solution_satisfied_empirically": bool(
            pJ[0] >= target_x
        ),
        "solution_satisfied_wilson_lower95": bool(
            pJ[1] >= target_x
        ),
    }


# =============================================================================
# Chafee--Infante operators
# =============================================================================

def enforce_dirichlet_tf(u: tf.Tensor) -> tf.Tensor:
    """
    Enforce homogeneous Dirichlet values on a full (B,M,M,1) tensor.
    """
    interior = u[:, 1:-1, 1:-1, :]

    return tf.pad(
        interior,
        paddings=[
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 0],
        ],
        mode="CONSTANT",
    )


@tf.function
def forward_euler_surrogate(
    u0_physical: tf.Tensor,
    nsteps: int,
    dt: float,
    gamma: float,
    kappa: float,
    hx: float,
    hy: float,
) -> tf.Tensor:
    """
    Forward Euler Chafee--Infante surrogate.

    Input/output are full fields of shape (B,M,M,1); homogeneous
    Dirichlet boundary values are imposed at every step.
    """
    u = tf.cast(u0_physical, tf.float32)
    u = enforce_dirichlet_tf(u)

    dt_tf = tf.cast(dt, tf.float32)
    gamma_tf = tf.cast(gamma, tf.float32)
    kappa_tf = tf.cast(kappa, tf.float32)
    hx2 = tf.cast(hx**2, tf.float32)
    hy2 = tf.cast(hy**2, tf.float32)

    for _ in tf.range(nsteps):
        interior = u[:, 1:-1, 1:-1, 0]

        left = u[:, 1:-1, :-2, 0]
        right = u[:, 1:-1, 2:, 0]
        down = u[:, :-2, 1:-1, 0]
        up = u[:, 2:, 1:-1, 0]

        lap = (
            (left - 2.0 * interior + right) / hx2
            + (down - 2.0 * interior + up) / hy2
        )

        reaction = -kappa_tf * (interior**3 - interior)

        interior_new = (
            interior
            + dt_tf * (
                gamma_tf * lap
                + reaction
            )
        )

        u = tf.pad(
            interior_new[..., tf.newaxis],
            paddings=[
                [0, 0],
                [1, 1],
                [1, 1],
                [0, 0],
            ],
            mode="CONSTANT",
        )

    return u


def residual_per_sample(
    pred_scaled: np.ndarray,
    true_scaled: np.ndarray,
    ic_scale: float,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equal-time short-horizon residual in scaled and physical units.
    """
    hx = 2.0 * args.L / (args.M - 1)
    hy = hx

    scaled_values: List[np.ndarray] = []
    physical_values: List[np.ndarray] = []

    for start in range(0, pred_scaled.shape[0], args.residual_batch):
        end = min(
            pred_scaled.shape[0],
            start + args.residual_batch,
        )

        pred_phys = (
            tf.convert_to_tensor(
                pred_scaled[start:end],
                dtype=tf.float32,
            )
            * tf.cast(ic_scale, tf.float32)
        )

        true_phys = (
            tf.convert_to_tensor(
                true_scaled[start:end],
                dtype=tf.float32,
            )
            * tf.cast(ic_scale, tf.float32)
        )

        pred_tau = forward_euler_surrogate(
            pred_phys,
            args.residual_steps,
            args.dt,
            args.gamma,
            args.kappa,
            hx,
            hy,
        )

        true_tau = forward_euler_surrogate(
            true_phys,
            args.residual_steps,
            args.dt,
            args.gamma,
            args.kappa,
            hx,
            hy,
        )

        # Interior comparison: the boundaries are identically zero.
        diff = tf.abs(
            pred_tau[:, 1:-1, 1:-1, :]
            - true_tau[:, 1:-1, 1:-1, :]
        )

        r_phys = tf.reduce_mean(
            diff,
            axis=[1, 2, 3],
        )

        r_scaled = (
            r_phys
            / tf.cast(ic_scale, tf.float32)
        )

        physical_values.append(
            r_phys.numpy().astype(np.float64)
        )
        scaled_values.append(
            r_scaled.numpy().astype(np.float64)
        )

    return (
        np.concatenate(scaled_values),
        np.concatenate(physical_values),
    )


def lyapunov_energy_per_sample(
    u_scaled: np.ndarray,
    ic_scale: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """
    Discrete Chafee--Infante Lyapunov energy.
    """
    u = (
        tf.convert_to_tensor(
            u_scaled,
            dtype=tf.float32,
        )
        * tf.cast(ic_scale, tf.float32)
    )

    phi = tf.squeeze(u, axis=-1)

    hx = 2.0 * args.L / (args.M - 1)
    hy = hx
    cell_area = hx * hy

    dx_field = (
        phi[:, 1:, :]
        - phi[:, :-1, :]
    ) / hx

    dy_field = (
        phi[:, :, 1:]
        - phi[:, :, :-1]
    ) / hy

    gradient_energy = (
        0.5
        * args.gamma
        * (
            tf.reduce_sum(
                dx_field**2,
                axis=[1, 2],
            )
            + tf.reduce_sum(
                dy_field**2,
                axis=[1, 2],
            )
        )
    )

    potential = (
        0.25 * phi**4
        - 0.5 * phi**2
    )

    potential_energy = (
        args.kappa
        * tf.reduce_sum(
            potential,
            axis=[1, 2],
        )
    )

    energy = (
        gradient_energy
        + potential_energy
    ) * cell_area

    return energy.numpy().astype(np.float64)


def diagnostic_batch(
    true_scaled: np.ndarray,
    pred_scaled: np.ndarray,
    ic_scale: float,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    """
    Reconstruction and structural diagnostics for one batch.
    """
    true = tf.convert_to_tensor(
        true_scaled,
        dtype=tf.float32,
    )
    pred = tf.convert_to_tensor(
        pred_scaled,
        dtype=tf.float32,
    )

    mae_scaled = tf.reduce_mean(
        tf.abs(pred - true),
        axis=[1, 2, 3],
    )

    mae_physical = (
        mae_scaled
        * tf.cast(ic_scale, tf.float32)
    )

    true_mean = tf.reduce_mean(
        true,
        axis=[1, 2, 3],
    )
    pred_mean = tf.reduce_mean(
        pred,
        axis=[1, 2, 3],
    )

    mean_error = tf.abs(
        pred_mean - true_mean
    )

    true_var = tf.math.reduce_variance(
        true,
        axis=[1, 2, 3],
    )
    pred_var = tf.math.reduce_variance(
        pred,
        axis=[1, 2, 3],
    )

    variance_error = tf.abs(
        pred_var - true_var
    )

    true_dx = true[:, 1:, :, :] - true[:, :-1, :, :]
    pred_dx = pred[:, 1:, :, :] - pred[:, :-1, :, :]

    true_dy = true[:, :, 1:, :] - true[:, :, :-1, :]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    gradient_error = (
        tf.reduce_mean(
            tf.abs(pred_dx - true_dx),
            axis=[1, 2, 3],
        )
        + tf.reduce_mean(
            tf.abs(pred_dy - true_dy),
            axis=[1, 2, 3],
        )
    )

    e_true = lyapunov_energy_per_sample(
        true_scaled,
        ic_scale,
        args,
    )
    e_pred = lyapunov_energy_per_sample(
        pred_scaled,
        ic_scale,
        args,
    )

    energy_abs = np.abs(e_pred - e_true)
    energy_rel = (
        energy_abs
        / (np.abs(e_true) + 1.0e-12)
    )

    return {
        "mae_scaled": mae_scaled.numpy().astype(np.float64),
        "mae_physical": mae_physical.numpy().astype(np.float64),
        "mean_error_scaled": mean_error.numpy().astype(np.float64),
        "variance_error_scaled": variance_error.numpy().astype(np.float64),
        "gradient_error_scaled": gradient_error.numpy().astype(np.float64),
        "energy_abs_physical": energy_abs.astype(np.float64),
        "energy_rel": energy_rel.astype(np.float64),
        "energy_true": e_true.astype(np.float64),
        "energy_pred": e_pred.astype(np.float64),
        "generator_min_scaled": tf.reduce_min(
            pred,
            axis=[1, 2, 3],
        ).numpy().astype(np.float64),
        "generator_max_scaled": tf.reduce_max(
            pred,
            axis=[1, 2, 3],
        ).numpy().astype(np.float64),
    }


# =============================================================================
# Plotting
# =============================================================================

def save_hist_cdf(
    values: np.ndarray,
    name: str,
    outdir: Path,
) -> None:
    """
    Save histogram and empirical CDF.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]

    if v.size == 0:
        return

    stem = name.lower()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(v, bins=50)
    ax.set_xlabel(name)
    ax.set_ylabel("count")
    ax.set_title(f"{name}: histogram")
    fig.tight_layout()
    fig.savefig(
        outdir / f"{stem}_hist.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    vs = np.sort(v)
    cdf = np.arange(1, vs.size + 1) / vs.size

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vs, cdf)
    ax.set_xlabel(name)
    ax.set_ylabel(r"$\widehat P(\mathrm{value}\leq t)$")
    ax.set_title(f"{name}: empirical CDF")
    fig.tight_layout()
    fig.savefig(
        outdir / f"{stem}_cdf.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_mae_residual_scatter(
    mae: np.ndarray,
    residual: np.ndarray,
    epsilon: float,
    delta: float,
    outpath: Path,
) -> None:
    """
    Scatter plot for the two defining solution metrics.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        mae,
        residual,
        s=14,
        alpha=0.65,
    )

    ax.axvline(
        epsilon,
        linestyle="--",
        linewidth=1.5,
        label=rf"$\varepsilon={epsilon:.4g}$",
    )
    ax.axhline(
        delta,
        linestyle="--",
        linewidth=1.5,
        label=rf"$\delta={delta:.4g}$",
    )

    ax.set_xlabel("scaled reconstruction MAE")
    ax.set_ylabel("scaled residual-horizon error")
    ax.set_title("MAE versus residual-horizon error")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_qualitative_plot(
    source_scaled: np.ndarray,
    pred_scaled: np.ndarray,
    true_scaled: np.ndarray,
    sample_ids: np.ndarray,
    mae_values: np.ndarray,
    residual_values: np.ndarray,
    model_name: str,
    outpath: Path,
) -> None:
    """
    Plot terminal input, reconstructed initial state, and true initial state.
    """
    nrows = len(sample_ids)

    if nrows == 0:
        return

    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(9, max(3.5, 3.0 * nrows)),
        squeeze=False,
    )

    titles = [
        r"given $u_T$",
        r"generated $\widehat u_0$",
        r"true $u_0$",
    ]

    for j, title in enumerate(titles):
        axes[0, j].set_title(title)

    # Symmetric color limits for inverse initial-condition comparison.
    ic_abs_max = max(
        np.max(np.abs(pred_scaled)),
        np.max(np.abs(true_scaled)),
    )

    src_abs_max = np.max(
        np.abs(source_scaled)
    )

    for i in range(nrows):
        axes[i, 0].imshow(
            source_scaled[i, :, :, 0].T,
            origin="lower",
            cmap="magma",
            vmin=-src_abs_max,
            vmax=src_abs_max,
            interpolation="nearest",
        )

        for j, field in enumerate(
            [pred_scaled[i], true_scaled[i]],
            start=1,
        ):
            axes[i, j].imshow(
                field[:, :, 0].T,
                origin="lower",
                cmap="magma",
                vmin=-ic_abs_max,
                vmax=ic_abs_max,
                interpolation="nearest",
            )

        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

        axes[i, 1].set_ylabel(
            f"sample {sample_ids[i]}\n"
            f"MAE={mae_values[i]:.4g}\n"
            f"RES={residual_values[i]:.4g}"
        )

    fig.suptitle(
        f"Inverse Chafee--Infante reconstruction: {model_name}"
    )

    fig.tight_layout()
    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_individual_qualitative_images(
    source_scaled: np.ndarray,
    pred_scaled: np.ndarray,
    true_scaled: np.ndarray,
    sample_ids: np.ndarray,
    outdir: Path,
) -> None:
    """
    Save each selected field as a true 1200 x 1200 pixel PNG with
    300-dpi metadata.  No plt.imsave() is used.
    """
    image_dir = outdir / "individual_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    ic_abs_max = max(
        float(np.max(np.abs(pred_scaled))),
        float(np.max(np.abs(true_scaled))),
    )
    src_abs_max = float(np.max(np.abs(source_scaled)))

    def save_field(field: np.ndarray, filename: Path, abs_max: float) -> None:
        # 4 inches x 4 inches at 300 dpi = exactly 1200 x 1200 pixels.
        fig = plt.figure(figsize=(4.0, 4.0), dpi=300)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

        ax.imshow(
            field.T,
            origin="lower",
            cmap="magma",
            vmin=-abs_max,
            vmax=abs_max,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_axis_off()

        # IMPORTANT: no bbox_inches="tight"; that can alter raster dimensions.
        fig.savefig(
            filename,
            dpi=300,
            format="png",
            metadata={"Software": "MLEE CIRDE evaluator"},
        )
        plt.close(fig)

        # Verify the file we actually wrote.
        from PIL import Image
        with Image.open(filename) as check:
            dpi = check.info.get("dpi", (0.0, 0.0))
            if check.size != (1200, 1200):
                raise RuntimeError(
                    f"{filename} was saved at {check.size}, expected (1200, 1200)."
                )
            if not (299.0 <= dpi[0] <= 301.0 and 299.0 <= dpi[1] <= 301.0):
                raise RuntimeError(
                    f"{filename} has DPI metadata {dpi}, expected approximately (300, 300)."
                )

    for i, sample_id in enumerate(sample_ids):
        save_field(
            source_scaled[i, :, :, 0],
            image_dir / f"src{sample_id}.png",
            src_abs_max,
        )
        save_field(
            pred_scaled[i, :, :, 0],
            image_dir / f"gen{sample_id}.png",
            ic_abs_max,
        )
        save_field(
            true_scaled[i, :, :, 0],
            image_dir / f"tar{sample_id}.png",
            ic_abs_max,
        )


# =============================================================================
# Models
# =============================================================================

def model_specs(args: argparse.Namespace):
    """
    Resolve best-MAE and optional best-residual generator paths.
    """
    model_dir = Path(args.model_dir)

    best_mae = (
        Path(args.best_mae_model)
        if args.best_mae_model
        else model_dir / "best_generator_mae.keras"
    )

    best_res = (
        Path(args.best_residual_model)
        if args.best_residual_model
        else model_dir / "best_generator_residual.keras"
    )

    if not best_mae.exists():
        raise FileNotFoundError(
            f"Required best-MAE model not found: {best_mae}"
        )

    specs = [
        {
            "name": "best_mae",
            "path": best_mae,
        }
    ]

    if best_res.exists():
        specs.append(
            {
                "name": "best_residual",
                "path": best_res,
            }
        )
    else:
        print(
            "Optional best-residual model not found; "
            f"skipping: {best_res}"
        )

    return specs


# =============================================================================
# Model evaluation
# =============================================================================

def evaluate_model_raw(
    spec: Dict[str, object],
    test_A: np.ndarray,
    test_B: np.ndarray,
    ne_scale: float,
    ic_scale: float,
    args: argparse.Namespace,
):
    """
    Run a model and collect all per-sample metrics before thresholding.
    """
    name = str(spec["name"])
    path = Path(spec["path"])

    model_outdir = Path(args.outdir) / name
    model_outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"[{name}] Loading {path}")

    model = tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "EnforceDirichletBoundary": EnforceDirichletBoundary,
        },
    )

    n_total = test_A.shape[0]
    n_eval = (
        min(n_total, args.max_test_samples)
        if args.max_test_samples is not None
        else n_total
    )

    print(
        f"[{name}] Evaluating {n_eval} test samples "
        f"in batches of {args.predict_batch}"
    )

    metric_chunks: Dict[str, List[np.ndarray]] = {}

    qualitative_ids = np.array(
        [
            s for s in args.samples
            if 0 <= s < n_eval
        ],
        dtype=int,
    )

    q_src: List[np.ndarray] = []
    q_pred: List[np.ndarray] = []
    q_true: List[np.ndarray] = []
    q_mae: List[float] = []
    q_res: List[float] = []
    q_ids: List[int] = []

    for start in range(0, n_eval, args.predict_batch):
        end = min(
            n_eval,
            start + args.predict_batch,
        )

        src_b = np.asarray(
            test_A[start:end],
            dtype=np.float32,
        )
        true_b = np.asarray(
            test_B[start:end],
            dtype=np.float32,
        )

        pred_b = model.predict(
            src_b,
            batch_size=args.predict_batch,
            verbose=0,
        ).astype(np.float32)

        # Enforce the exact DBC at evaluation as an additional safety check.
        pred_b[:, 0, :, :] = 0.0
        pred_b[:, -1, :, :] = 0.0
        pred_b[:, :, 0, :] = 0.0
        pred_b[:, :, -1, :] = 0.0

        diag = diagnostic_batch(
            true_b,
            pred_b,
            ic_scale,
            args,
        )

        res_scaled, res_physical = residual_per_sample(
            pred_b,
            true_b,
            ic_scale,
            args,
        )

        diag["residual_scaled"] = res_scaled
        diag["residual_physical"] = res_physical

        for key, arr in diag.items():
            metric_chunks.setdefault(
                key,
                [],
            ).append(
                np.asarray(
                    arr,
                    dtype=np.float64,
                )
            )

        for sid in qualitative_ids:
            if start <= sid < end:
                loc = int(sid - start)

                q_src.append(src_b[loc])
                q_pred.append(pred_b[loc])
                q_true.append(true_b[loc])
                q_mae.append(
                    float(diag["mae_scaled"][loc])
                )
                q_res.append(
                    float(res_scaled[loc])
                )
                q_ids.append(int(sid))

        print(
            f"[{name}] processed {end}/{n_eval}"
        )

    metrics = {
        key: np.concatenate(chunks)
        for key, chunks in metric_chunks.items()
    }

    if q_ids:
        q_src_array = np.stack(q_src)
        q_pred_array = np.stack(q_pred)
        q_true_array = np.stack(q_true)
        q_ids_array = np.asarray(q_ids)

        # Combined 3 x 3 qualitative figure.
        save_qualitative_plot(
            q_src_array,
            q_pred_array,
            q_true_array,
            q_ids_array,
            np.asarray(q_mae),
            np.asarray(q_res),
            name,
            model_outdir / "qualitative.png",
        )

        # Nine separate publication-ready images.
        save_individual_qualitative_images(
            q_src_array,
            q_pred_array,
            q_true_array,
            q_ids_array,
            model_outdir,
        )

    return metrics


def write_human_readable_report(
    name: str,
    report: dict,
    outdir: Path,
) -> None:
    """
    Write a human-readable report in the same style as the earlier
    Chafee--Infante testing script.
    """
    metrics = report["metrics"]
    p_mae = report["probability_MAE"]
    p_res = report["probability_residual"]
    joint = report["joint_solution_statistics"]

    mae = metrics["mae_scaled"]
    res = metrics["residual_scaled"]

    epsilon = report["empirical_solution_definition"]["epsilon"]
    delta = report["empirical_solution_definition"]["delta"]
    X = report["empirical_solution_definition"]["X"]

    p_joint = joint["P_joint"]["p"]
    joint_lo, joint_hi = joint["P_joint"]["wilson95"]

    p_res_given_mae = joint["P_RES_given_MAE"]
    p_mae_given_res = joint["P_MAE_given_RES"]
    corr = joint["corr_MAE_RES"]
    satisfied = joint["solution_satisfied_empirically"]

    lines = [
        r"\begin{reportbox}",
        r"=== Chafee--Infante Inversion Experiment with Poisson Smooth Initial Data ===",
        "",
        f"Model name : {name.replace('_', r'\_')}",
        "",
        "[Full test-set MAE: scaled IC units]",
        f"  mean +/- std        = {mae['mean']:.8f} +/- {mae['std']:.8f}",
        f"  min / median / max  = {mae['min']:.8f} / {mae['median']:.8f} / {mae['max']:.8f}",
        f"  q05 / q95           = {mae['q05']:.8f} / {mae['q95']:.8f}",
        f"  epsilon             = {epsilon:.8f}",
        (
            "  P(MAE\\_scaled <= epsilon) = "
            f"{p_mae['p']:.6f} "
            f"(Wilson95 [{p_mae['wilson95'][0]}, {p_mae['wilson95'][1]}])"
        ),
        "",
        "[Full test-set residual]",
        f"  mean +/- std        = {res['mean']:.8f} +/- {res['std']:.8f}",
        f"  min / median / max  = {res['min']:.8f} / {res['median']:.8f} / {res['max']:.8f}",
        f"  q05 / q95           = {res['q05']:.8f} / {res['q95']:.8f}",
        f"  delta               = {delta:.8f}",
        (
            "  P(Residual <= delta) = "
            f"{p_res['p']:.6f} "
            f"(Wilson95 [{p_res['wilson95'][0]}, {p_res['wilson95'][1]}])"
        ),
        "",
        "[Joint probabilities]",
        (
            "  P(MAE\\_scaled <= epsilon and Residual <= delta) = "
            f"{p_joint:.6f} "
            f"(Wilson95 [{joint_lo}, {joint_hi}])"
        ),
        (
            "  P(Residual <= delta | MAE\\_scaled <= epsilon)   = "
            f"{p_res_given_mae:.6f}"
        ),
        (
            "  P(MAE\\_scaled <= epsilon | Residual <= delta)   = "
            f"{p_mae_given_res:.6f}"
        ),
        (
            "  corr(MAE\\_scaled, Residual)                     = "
            f"{corr:.6f}"
        ),
        "",
        "[Machine-learned inverse solution criterion]",
        (
            "  Criterion: "
            f"P(MAE\\_scaled <= {epsilon:.6f} and Residual <= {delta:.6f}) >= {X}"
        ),
        f"  Empirical value = {p_joint:.6f}",
        f"  Satisfied = {satisfied}",
        "",
        r"[Machine-learned inverse solution criterion (in \LaTeX)]",
        (
            r"\noindent On the held-out test set, "
            rf"$\Pr(\mathrm{{MAE}}_{{\mathrm{{scaled}}}}\le {epsilon:.6f})"
            rf"\approx {p_mae['p']:.4f}$, "
            rf"$\Pr(\mathrm{{Res}}\le {delta:.6f})"
            rf"\approx {p_res['p']:.4f}$, and "
            rf"$\Pr(\mathrm{{MAE}}_{{\mathrm{{scaled}}}}\le {epsilon:.6f}"
            rf"\ \wedge\ \mathrm{{Res}}\le {delta:.6f})"
            rf"\approx {p_joint:.4f}$."
        ),
        r"\end{reportbox}",
        "",
    ]

    report_text = "\n".join(lines)

    (outdir / "report.txt").write_text(report_text)
    (outdir / "report.tex").write_text(report_text)


def save_model_report(
    name: str,
    metrics: Dict[str, np.ndarray],
    epsilon: float,
    delta: float,
    ne_scale: float,
    ic_scale: float,
    args: argparse.Namespace,
):
    """
    Save all model statistics after common thresholds are known.
    """
    model_outdir = Path(args.outdir) / name

    joint = joint_solution_report(
        metrics["mae_scaled"],
        metrics["residual_scaled"],
        epsilon,
        delta,
        args.target_p_joint,
    )

    report = {
        "model_name": name,
        "problem": {
            "equation": (
                "u_t - gamma*Delta u + kappa*(u^3-u) = 0"
            ),
            "boundary_condition": "homogeneous Dirichlet",
            "domain": [
                -float(args.L),
                float(args.L),
                -float(args.L),
                float(args.L),
            ],
            "grid": [
                int(args.M),
                int(args.M),
            ],
            "gamma": float(args.gamma),
            "kappa": float(args.kappa),
            "dt": float(args.dt),
            "dataset_steps": int(args.dataset_steps),
            "dataset_horizon": float(
                args.dataset_steps * args.dt
            ),
            "residual_steps": int(args.residual_steps),
            "residual_horizon": float(
                args.residual_steps * args.dt
            ),
            "residual_definition": (
                "mean absolute difference between equal-time "
                "short-horizon Forward Euler rollouts of predicted "
                "and true initial conditions"
            ),
        },
        "scales": {
            "NE_SCALE": float(ne_scale),
            "IC_SCALE": float(ic_scale),
        },
        "empirical_solution_definition": {
            "epsilon": float(epsilon),
            "delta": float(delta),
            "X": float(args.target_p_joint),
            "criterion": (
                "P_hat(MAE <= epsilon and residual <= delta) >= X"
            ),
        },
        "metrics": {
            key: metric_summary(values)
            for key, values in metrics.items()
        },
        "probability_MAE": probability_report(
            metrics["mae_scaled"],
            epsilon,
        ),
        "probability_residual": probability_report(
            metrics["residual_scaled"],
            delta,
        ),
        "joint_solution_statistics": joint,
    }

    with (
        model_outdir / "report.json"
    ).open("w") as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    # Per-sample CSV.
    keys = list(metrics.keys())
    n = len(metrics[keys[0]])

    with (
        model_outdir / "metrics_per_sample.csv"
    ).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["sample"] + keys
        )

        for i in range(n):
            writer.writerow(
                [i]
                + [
                    float(metrics[key][i])
                    for key in keys
                ]
            )

    # Plots for the most important metrics.
    plot_metrics = [
        "mae_scaled",
        "residual_scaled",
        "energy_rel",
        "mean_error_scaled",
        "variance_error_scaled",
        "gradient_error_scaled",
    ]

    for key in plot_metrics:
        save_hist_cdf(
            metrics[key],
            key,
            model_outdir,
        )

    save_mae_residual_scatter(
        metrics["mae_scaled"],
        metrics["residual_scaled"],
        epsilon,
        delta,
        model_outdir / "mae_vs_residual.png",
    )

    write_human_readable_report(
        name,
        report,
        model_outdir,
    )

    return report


# =============================================================================
# Comparison outputs
# =============================================================================

def write_comparison(
    reports: Dict[str, dict],
    outdir: Path,
) -> None:
    """
    Write concise model-comparison tables.
    """
    txt_path = outdir / "comparison_summary.txt"
    csv_path = outdir / "comparison_summary.csv"

    lines = [
        "=== Inverse Chafee--Infante model comparison ===",
        "",
    ]

    rows = []

    for name, report in reports.items():
        metrics = report["metrics"]
        joint = report["joint_solution_statistics"]

        row = {
            "model": name,
            "mae_mean": metrics["mae_scaled"]["mean"],
            "mae_std": metrics["mae_scaled"]["std"],
            "residual_mean": metrics["residual_scaled"]["mean"],
            "residual_std": metrics["residual_scaled"]["std"],
            "energy_rel_mean": metrics["energy_rel"]["mean"],
            "joint_probability": joint["P_joint"]["p"],
            "joint_wilson_low": joint["P_joint"]["wilson95"][0],
            "joint_wilson_high": joint["P_joint"]["wilson95"][1],
            "empirical_solution": joint[
                "solution_satisfied_empirically"
            ],
            "wilson_lower95_solution": joint[
                "solution_satisfied_wilson_lower95"
            ],
        }

        rows.append(row)

        lines.extend(
            [
                f"[{name}]",
                (
                    "MAE mean +/- std = "
                    f"{row['mae_mean']:.8g} +/- "
                    f"{row['mae_std']:.8g}"
                ),
                (
                    "Residual mean +/- std = "
                    f"{row['residual_mean']:.8g} +/- "
                    f"{row['residual_std']:.8g}"
                ),
                (
                    "Relative energy error mean = "
                    f"{row['energy_rel_mean']:.8g}"
                ),
                (
                    "Joint probability = "
                    f"{row['joint_probability']:.6f}"
                ),
                (
                    "Joint Wilson 95% = "
                    f"[{row['joint_wilson_low']:.6f}, "
                    f"{row['joint_wilson_high']:.6f}]"
                ),
                (
                    "Empirical solution criterion satisfied = "
                    f"{row['empirical_solution']}"
                ),
                (
                    "Wilson-lower-bound solution criterion satisfied = "
                    f"{row['wilson_lower95_solution']}"
                ),
                "",
            ]
        )

    txt_path.write_text(
        "\n".join(lines)
    )

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        (args.fixed_eps is None)
        != (args.fixed_delta is None)
    ):
        raise ValueError(
            "Supply both --fixed-eps and --fixed-delta, or neither."
        )

    if args.residual_steps <= 0:
        raise ValueError(
            "--residual-steps must be positive."
        )

    # TensorFlow device configuration.
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except (RuntimeError, ValueError):
            pass

    test_A, test_B, ne_scale, ic_scale = load_dataset(
        args.train_npz,
        args.test_npz,
        args.M,
    )

    print()
    print("Inverse Chafee--Infante full test evaluation")
    print("============================================")
    print(f"test src/tar shape:      {test_A.shape} / {test_B.shape}")
    print(f"NE_SCALE:                {ne_scale:.12e}")
    print(f"IC_SCALE:                {ic_scale:.12e}")
    print(
        f"dataset horizon:          "
        f"{args.dataset_steps * args.dt:.6f} "
        f"({args.dataset_steps} steps)"
    )
    print(
        f"residual horizon:         "
        f"{args.residual_steps * args.dt:.6f} "
        f"({args.residual_steps} steps)"
    )

    specs = model_specs(args)

    # First pass: obtain raw metrics for every model.
    raw_metrics: Dict[str, Dict[str, np.ndarray]] = {}

    for spec in specs:
        raw_metrics[str(spec["name"])] = evaluate_model_raw(
            spec,
            test_A,
            test_B,
            ne_scale,
            ic_scale,
            args,
        )

    # Calibrate ONE common pair of thresholds unless fixed by the user.
    if (
        args.fixed_eps is not None
        and args.fixed_delta is not None
    ):
        epsilon = float(args.fixed_eps)
        delta = float(args.fixed_delta)
        threshold_source = "fixed by command line"

    else:
        reference = args.threshold_reference

        if reference not in raw_metrics:
            raise ValueError(
                f"Threshold reference '{reference}' was not evaluated."
            )

        epsilon = quantile_threshold(
            raw_metrics[reference]["mae_scaled"],
            args.target_p_mae,
        )

        delta = quantile_threshold(
            raw_metrics[reference]["residual_scaled"],
            args.target_p_res,
        )

        threshold_source = (
            f"{reference}: empirical "
            f"{args.target_p_mae:.3f} MAE quantile and "
            f"{args.target_p_res:.3f} residual quantile"
        )

    print()
    print("Empirical solution definition")
    print("-----------------------------")
    print(f"epsilon:                  {epsilon:.12e}")
    print(f"delta:                    {delta:.12e}")
    print(f"X:                        {args.target_p_joint:.6f}")
    print(f"threshold source:         {threshold_source}")

    solution_definition = {
        "epsilon": float(epsilon),
        "delta": float(delta),
        "X": float(args.target_p_joint),
        "threshold_source": threshold_source,
        "target_p_mae_for_calibration": float(args.target_p_mae),
        "target_p_res_for_calibration": float(args.target_p_res),
        "criterion": (
            "P_hat(MAE <= epsilon and residual <= delta) >= X"
        ),
        "residual_horizon": float(
            args.residual_steps * args.dt
        ),
        "residual_steps": int(args.residual_steps),
        "residual_definition": (
            "R_tau(u0_pred,u0_true) = "
            "mean_abs(Phi_tau(u0_pred)-Phi_tau(u0_true))"
        ),
    }

    with (
        outdir / "solution_definition.json"
    ).open("w") as file:
        json.dump(
            solution_definition,
            file,
            indent=2,
        )

    # Second pass: apply common thresholds and write final reports.
    reports: Dict[str, dict] = {}

    for name, metrics in raw_metrics.items():
        reports[name] = save_model_report(
            name,
            metrics,
            epsilon,
            delta,
            ne_scale,
            ic_scale,
            args,
        )

    write_comparison(
        reports,
        outdir,
    )

    print()
    print(
        f"Evaluation complete. Results written to: {outdir}"
    )


if __name__ == "__main__":
    main()
