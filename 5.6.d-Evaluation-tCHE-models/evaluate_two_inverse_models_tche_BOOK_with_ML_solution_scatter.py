#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_two_inverse_models_tche.py

Clean full-test evaluator for inverse ternary Cahn--Hilliard generators.

Inverse problem
---------------
    c_T  --->  c_0

Data convention
---------------
    src = terminal state c_T
    tar = initial state c_0

Diagnostics
-----------
For each generator:
    - MAE
    - short-horizon dynamical residual
    - relative ternary free-energy error
    - component mass error
    - component variance error
    - purity error
    - simplex error
    - discrete interfacial-density error

Short-horizon residual
----------------------
For tau = residual_steps * dt,

    R_tau(c0_pred, c0_true)
        = || Phi_tau(c0_pred) - Phi_tau(c0_true) ||_1.

This deliberately compares equal-time short rollouts of the generated and
true initial conditions. It does not compare a short rollout against the
full terminal state c_T.

Typical usage
python evaluate_two_inverse_models_tche.py \
  --train-npz ../../../tCHE_Datasets/Eyre_noisy_128x128-iterations=11000/train-dataset_128x128_tCH_PBC_Eyer_40-40-20_iters=11000_count=50000.npz \
  --test-npz ../../../tCHE_Datasets/Eyre_noisy_128x128-iterations=11000/test-dataset_128x128_tCH_PBC_Eyer_40-40-20_iters=11000_count=10000.npz \
  --model-dir tCHE_3color_WGANGP_stats_energy_models \
  --outdir full_model_evaluation_Eyre-noisy \
  --residual-steps 200 \
  --max-test-samples 10000
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ============================================================
# Command-line arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean evaluator for inverse ternary Cahn--Hilliard models."
    )

    p.add_argument("--train-npz", required=True)
    p.add_argument("--test-npz", required=True)
    p.add_argument("--src-key", default="src")
    p.add_argument("--tar-key", default="tar")

    p.add_argument("--model-dir", default="tCHE_3color_WGANGP_stats_energy_models")
    p.add_argument("--best-mae-model", default=None)
    p.add_argument("--best-mae-metric", default=None)
    p.add_argument("--best-residual-model", default=None)
    p.add_argument("--best-residual-metric", default=None)
    p.add_argument("--outdir", default="full_model_evaluation_tche")

    p.add_argument("--N", type=int, default=None)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=5.0e-8)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--A", type=float, default=1.0)
    p.add_argument("--chi12", type=float, default=1.5)
    p.add_argument("--chi13", type=float, default=1.5)
    p.add_argument("--chi23", type=float, default=1.5)
    p.add_argument("--mobility", type=float, default=1.0)
    p.add_argument("--residual-steps", type=int, default=200)

    p.add_argument("--predict-batch", type=int, default=4)
    p.add_argument("--residual-batch", type=int, default=4)
    p.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2])

    p.add_argument("--target-p-mae", type=float, default=0.95)
    p.add_argument("--target-p-res", type=float, default=0.95)
    p.add_argument("--target-p-joint", type=float, default=0.90)

    p.add_argument("--fixed-eps", type=float, default=None)
    p.add_argument("--fixed-delta", type=float, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)

    return p.parse_args()


# ============================================================
# Data utilities
# ============================================================

def to_channel_last(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim == 3:
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))[None, ...]
        elif x.shape[-1] == 3:
            x = x[None, ...]
        else:
            raise ValueError(f"Cannot interpret 3D ternary shape {x.shape}")

    elif x.ndim == 4:
        if x.shape[-1] == 3:
            pass
        elif x.shape[1] == 3:
            x = np.transpose(x, (0, 2, 3, 1))
        else:
            raise ValueError(f"Cannot interpret 4D ternary shape {x.shape}")
    else:
        raise ValueError(f"Expected 3D or 4D array, got {x.shape}")

    return x.astype("float32", copy=False)


def normalize_simplex_np(c: np.ndarray, eps0: float = 1.0e-8) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    s = np.sum(c, axis=-1, keepdims=True)
    return (c / (s + eps0)).astype("float32", copy=False)


def load_npz_arrays(path: str, src_key: str, tar_key: str):
    z = np.load(path, mmap_mode="r")

    if src_key not in z or tar_key not in z:
        raise KeyError(
            f"Expected keys '{src_key}' and '{tar_key}' in {path}; "
            f"found {list(z.keys())}"
        )

    return z[src_key], z[tar_key]


def batch_take(arr, start: int, end: int) -> np.ndarray:
    return normalize_simplex_np(to_channel_last(arr[start:end]))


# ============================================================
# Statistical helpers
# ============================================================

def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return np.nan, np.nan, np.nan

    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(
        p * (1.0 - p) / n + z**2 / (4.0 * n**2)
    )

    return p, max(0.0, center - half), min(1.0, center + half)


def metric_summary(values: np.ndarray) -> Dict[str, float]:
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
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]

    if v.size == 0:
        return np.nan

    v = np.sort(v)
    idx = int(np.ceil(target_p * v.size)) - 1
    idx = max(0, min(idx, v.size - 1))
    return float(v[idx])


def prob_leq(values: np.ndarray, threshold: float):
    v = np.asarray(values, dtype=np.float64).ravel()
    k = int(np.sum(v <= threshold))
    n = int(v.size)
    p, lo, hi = wilson_ci(k, n)

    return {
        "threshold": float(threshold),
        "k": k,
        "n": n,
        "p": p,
        "wilson95": [lo, hi],
    }


def joint_probs(mae, residual, epsilon, delta):
    mae = np.asarray(mae, dtype=np.float64).ravel()
    residual = np.asarray(residual, dtype=np.float64).ravel()

    if mae.size != residual.size:
        raise ValueError("MAE and residual arrays must have equal length.")

    A = mae <= epsilon
    B = residual <= delta

    n = mae.size
    kA = int(np.sum(A))
    kB = int(np.sum(B))
    kJ = int(np.sum(A & B))

    pA = wilson_ci(kA, n)
    pB = wilson_ci(kB, n)
    pJ = wilson_ci(kJ, n)

    corr = np.nan
    if n > 1 and np.std(mae) > 0.0 and np.std(residual) > 0.0:
        corr = float(np.corrcoef(mae, residual)[0, 1])

    return {
        "P_MAE_leq_eps": {
            "p": pA[0], "wilson95": [pA[1], pA[2]], "k": kA, "n": int(n)
        },
        "P_RES_leq_delta": {
            "p": pB[0], "wilson95": [pB[1], pB[2]], "k": kB, "n": int(n)
        },
        "P_joint": {
            "p": pJ[0], "wilson95": [pJ[1], pJ[2]], "k": kJ, "n": int(n)
        },
        "P_RES_given_MAE": float(kJ / kA) if kA else np.nan,
        "P_MAE_given_RES": float(kJ / kB) if kB else np.nan,
        "corr_MAE_RES": corr,
    }


# ============================================================
# Ternary Cahn--Hilliard operators
# ============================================================

def normalize_simplex_tf(c, eps0=1.0e-8):
    c = tf.clip_by_value(c, 0.0, 1.0)
    s = tf.reduce_sum(c, axis=-1, keepdims=True)
    return c / (s + eps0)


def simplex_error_per_sample_tf(c):
    positivity = tf.reduce_mean(tf.nn.relu(-c), axis=[1, 2, 3])
    upper = tf.reduce_mean(tf.nn.relu(c - 1.0), axis=[1, 2, 3])
    sum_error = tf.reduce_mean(
        tf.abs(tf.reduce_sum(c, axis=-1) - 1.0),
        axis=[1, 2],
    )
    return positivity + upper + sum_error


def laplacian_periodic_tf(u, dx):
    return (
        tf.roll(u, 1, axis=1)
        + tf.roll(u, -1, axis=1)
        + tf.roll(u, 1, axis=2)
        + tf.roll(u, -1, axis=2)
        - 4.0 * u
    ) / (dx * dx)


def bulk_derivative_tf(c, A, chi):
    c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2]
    chi12, chi13, chi23 = chi

    dW1 = A * (2*c1 - 6*c1**2 + 4*c1**3) + chi12*c2 + chi13*c3
    dW2 = A * (2*c2 - 6*c2**2 + 4*c2**3) + chi12*c1 + chi23*c3
    dW3 = A * (2*c3 - 6*c3**2 + 4*c3**3) + chi13*c1 + chi23*c2

    return tf.stack([dW1, dW2, dW3], axis=-1)


def chemical_potential_tf(c, dx, eps, A, chi):
    mu0 = bulk_derivative_tf(c, A=A, chi=chi)

    mu = tf.stack(
        [
            mu0[..., i] - eps**2 * laplacian_periodic_tf(c[..., i], dx)
            for i in range(3)
        ],
        axis=-1,
    )

    return mu - tf.reduce_mean(mu, axis=-1, keepdims=True)


def forward_euler_rhs_tf(c, dx, eps, A, chi, mobility):
    mu = chemical_potential_tf(c, dx=dx, eps=eps, A=A, chi=chi)

    return tf.stack(
        [
            tf.cast(mobility, tf.float32) * laplacian_periodic_tf(mu[..., i], dx)
            for i in range(3)
        ],
        axis=-1,
    )


def ternary_energy_per_sample_tf(c, dx, eps, A, chi):
    c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2]
    chi12, chi13, chi23 = chi

    bulk = (
        A * tf.reduce_sum(c**2 * (1.0 - c)**2, axis=-1)
        + chi12*c1*c2
        + chi13*c1*c3
        + chi23*c2*c3
    )

    grad_part = 0.0

    for i in range(3):
        ci = c[..., i]
        cx = (tf.roll(ci, -1, axis=1) - ci) / dx
        cy = (tf.roll(ci, -1, axis=2) - ci) / dx
        grad_part += 0.5 * eps**2 * (cx**2 + cy**2)

    return dx**2 * tf.reduce_sum(bulk + grad_part, axis=[1, 2])


def interfacial_density_per_sample_tf(c, dx, domain_area):
    rho = 0.0

    for k in range(3):
        ck = c[..., k]
        cx = (tf.roll(ck, -1, axis=1) - ck) / dx
        cy = (tf.roll(ck, -1, axis=2) - ck) / dx

        grad_mag = tf.sqrt(cx**2 + cy**2 + 1.0e-12)
        rho += tf.reduce_sum(grad_mag, axis=[1, 2])

    return (dx**2 / domain_area) * rho


def forward_sim_tche_tf(c0, nsteps, dt, dx, eps, A, chi, mobility):
    """
    Eager short-horizon Forward Euler surrogate.
    """

    c = normalize_simplex_tf(c0)

    for _ in range(int(nsteps)):
        rhs = forward_euler_rhs_tf(
            c,
            dx=dx,
            eps=eps,
            A=A,
            chi=chi,
            mobility=mobility,
        )

        c = c + tf.cast(dt, tf.float32) * rhs
        c = normalize_simplex_tf(c)

    return c


# ============================================================
# Diagnostics
# ============================================================

def diagnostics_per_batch(true_np, pred_np, dx, domain_area, eps, A, chi):
    true = tf.convert_to_tensor(true_np, dtype=tf.float32)
    pred = tf.convert_to_tensor(pred_np, dtype=tf.float32)

    mae = tf.reduce_mean(tf.abs(pred - true), axis=[1, 2, 3])

    E_true = ternary_energy_per_sample_tf(true, dx, eps, A, chi)
    E_pred = ternary_energy_per_sample_tf(pred, dx, eps, A, chi)

    energy_abs = tf.abs(E_pred - E_true)
    energy_rel = energy_abs / (tf.abs(E_true) + 1.0e-8)

    mass_true = tf.reduce_mean(true, axis=[1, 2])
    mass_pred = tf.reduce_mean(pred, axis=[1, 2])
    mass_error = tf.reduce_mean(tf.abs(mass_pred - mass_true), axis=1)

    var_true = tf.math.reduce_variance(true, axis=[1, 2])
    var_pred = tf.math.reduce_variance(pred, axis=[1, 2])
    variance_error = tf.reduce_mean(tf.abs(var_pred - var_true), axis=1)

    purity = tf.reduce_sum(pred**2, axis=-1)
    purity_error = tf.reduce_mean(tf.abs(1.0 - purity), axis=[1, 2])

    simplex_error = simplex_error_per_sample_tf(pred)

    rho_true = interfacial_density_per_sample_tf(true, dx, domain_area)
    rho_pred = interfacial_density_per_sample_tf(pred, dx, domain_area)
    interfacial_error = tf.abs(rho_pred - rho_true)

    sums = tf.reduce_sum(pred, axis=-1)
    gen_mass = tf.reduce_mean(pred, axis=[1, 2])
    gen_var = tf.math.reduce_variance(pred, axis=[1, 2])

    return {
        "mae": mae.numpy().astype(np.float64),
        "energy_abs": energy_abs.numpy().astype(np.float64),
        "energy_rel": energy_rel.numpy().astype(np.float64),
        "mass_error": mass_error.numpy().astype(np.float64),
        "variance_error": variance_error.numpy().astype(np.float64),
        "purity_error": purity_error.numpy().astype(np.float64),
        "simplex_error": simplex_error.numpy().astype(np.float64),
        "interfacial_density_error": interfacial_error.numpy().astype(np.float64),
        "rho_true": rho_true.numpy().astype(np.float64),
        "rho_pred": rho_pred.numpy().astype(np.float64),
        "gen_min": tf.reduce_min(pred, axis=[1, 2, 3]).numpy().astype(np.float64),
        "gen_max": tf.reduce_max(pred, axis=[1, 2, 3]).numpy().astype(np.float64),
        "gen_sum_min": tf.reduce_min(sums, axis=[1, 2]).numpy().astype(np.float64),
        "gen_sum_max": tf.reduce_max(sums, axis=[1, 2]).numpy().astype(np.float64),
        "gen_mass_c1": gen_mass[:, 0].numpy().astype(np.float64),
        "gen_mass_c2": gen_mass[:, 1].numpy().astype(np.float64),
        "gen_mass_c3": gen_mass[:, 2].numpy().astype(np.float64),
        "gen_var_c1": gen_var[:, 0].numpy().astype(np.float64),
        "gen_var_c2": gen_var[:, 1].numpy().astype(np.float64),
        "gen_var_c3": gen_var[:, 2].numpy().astype(np.float64),
    }


def residual_per_sample(
    pred_np,
    true_np,
    residual_steps,
    residual_batch,
    dt,
    dx,
    eps,
    A,
    chi,
    mobility,
):
    out = []

    for start in range(0, pred_np.shape[0], residual_batch):
        end = min(pred_np.shape[0], start + residual_batch)

        c0_pred = tf.convert_to_tensor(pred_np[start:end], dtype=tf.float32)
        c0_true = tf.convert_to_tensor(true_np[start:end], dtype=tf.float32)

        pred_tau = forward_sim_tche_tf(
            c0_pred, residual_steps, dt, dx, eps, A, chi, mobility
        )
        true_tau = forward_sim_tche_tf(
            c0_true, residual_steps, dt, dx, eps, A, chi, mobility
        )

        res = tf.reduce_mean(tf.abs(pred_tau - true_tau), axis=[1, 2, 3])
        out.append(res.numpy().astype(np.float64))

    return np.concatenate(out, axis=0)


# ============================================================
# Plot / report helpers
# ============================================================

def save_hist_cdf(values, name, outdir):
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]

    if v.size == 0:
        return

    stem = name.lower()

    fig, ax = plt.subplots()
    ax.hist(v, bins=60)
    ax.set_title(f"{name} histogram")
    ax.set_xlabel(name)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}_hist.png"), dpi=200)
    plt.close(fig)

    vs = np.sort(v)
    cdf = np.arange(1, vs.size + 1) / vs.size

    fig, ax = plt.subplots()
    ax.plot(vs, cdf)
    ax.set_title(f"{name} empirical CDF")
    ax.set_xlabel(name)
    ax.set_ylabel("P(value <= t)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}_cdf.png"), dpi=200)
    plt.close(fig)


def save_mae_residual_scatter(mae, residual, epsilon, delta, outdir):
    mae = np.asarray(mae, dtype=np.float64).ravel()
    residual = np.asarray(residual, dtype=np.float64).ravel()
    finite = np.isfinite(mae) & np.isfinite(residual)
    mae = mae[finite]
    residual = residual[finite]

    if mae.size == 0:
        return

    joint_success = (mae <= epsilon) & (residual <= delta)

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=300)
    ax.scatter(
        mae[~joint_success],
        residual[~joint_success],
        s=9,
        alpha=0.45,
        color="0.45",
        edgecolors="none",
        label="outside joint criterion",
    )
    ax.scatter(
        mae[joint_success],
        residual[joint_success],
        s=9,
        alpha=0.55,
        color="tab:green",
        edgecolors="none",
        label=r"$\mathrm{MAE}\leq\varepsilon$ and $\mathrm{Residual}\leq\delta$",
    )
    ax.axvline(
        epsilon,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=rf"$\varepsilon={epsilon:.4g}$",
    )
    ax.axhline(
        delta,
        color="tab:blue",
        linestyle="--",
        linewidth=1.5,
        label=rf"$\delta={delta:.4g}$",
    )
    ax.set_xlabel("MAE")
    ax.set_ylabel("Residual")
    ax.set_title("MAE--residual joint criterion")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, "mae_residual_scatter.png"),
        dpi=300,
    )
    plt.close(fig)


def read_scalar_text(path: Optional[str]) -> Optional[float]:
    if path is None or not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return None


def save_qualitative_plot(
    src,
    pred,
    true,
    sample_ids,
    panel_values,
    model_name,
    outpng,
):
    nrows = len(sample_ids)

    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(9, max(4, 3.2*nrows + 1.8)),
    )

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["given $c_T$", "generated $\\hat c_0$", "true $c_0$"]

    for j, title in enumerate(titles):
        axes[0, j].set_title(title)

    for i, _sid in enumerate(sample_ids):
        fields = [src[i], pred[i], true[i]]

        for j, field in enumerate(fields):
            axes[i, j].imshow(
                np.clip(field, 0.0, 1.0),
                origin="lower",
                interpolation="nearest",
            )
            axes[i, j].axis("off")

    lines = [
        f"Model = {model_name}",
        "Per-row MAE / short residual / interfacial error:",
    ]

    for i, sid in enumerate(sample_ids):
        lines.append(
            f"  sample {int(sid)}: "
            f"MAE={panel_values['mae'][i]:.6g}, "
            f"RES={panel_values['residual'][i]:.6g}, "
            f"INT={panel_values['interfacial'][i]:.6g}"
        )

    fig.text(
        0.02,
        0.02,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
    )

    fig.tight_layout(rect=[0.0, 0.12, 1.0, 1.0])
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_separate_qualitative_images(
    src,
    pred,
    true,
    sample_ids,
    outdir,
):
    for i, sid in enumerate(sample_ids):
        fields = {
            "src": src[i],
            "gen": pred[i],
            "true": true[i],
        }

        for stem, field in fields.items():
            fig = plt.figure(
                figsize=(6.0, 6.0),
                dpi=300,
                frameon=False,
            )
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(
                np.clip(field, 0.0, 1.0),
                origin="lower",
                interpolation="nearest",
            )
            ax.axis("off")
            fig.savefig(
                os.path.join(outdir, f"{stem}{int(sid)}.png"),
                dpi=300,
                bbox_inches=None,
                pad_inches=0,
            )
            plt.close(fig)


def write_text_report(report, outpath):
    """Write the complete human-readable evaluation report."""

    model_name = report["model_name"]
    config = report["config"]
    metrics = report["metrics"]
    epsilon = report["thresholds"]["epsilon"]
    delta = report["thresholds"]["delta"]
    p_mae = report["P_MAE_leq_eps"]
    p_res = report["P_RES_leq_delta"]
    joint = report["joint"]
    p_joint = joint["P_joint"]
    criterion = report["ml_solution_definition_check"]

    saved_metric_name = (
        "validation_mae_saved_during_training"
        if model_name == "best_mae"
        else "validation_residual_saved_during_training"
    )

    lines = [
        "=== Ternary Cahn--Hilliard Inverse Model Evaluation Report ===",
        f"Model name : {model_name}",
        f"Train set  : {config['TRAIN_NPZ']}",
        f"Test set   : {config['TEST_NPZ']}",
        f"Model path : {config['model_path']}",
        "",
        "[Saved training metric]",
        f"  name  = {saved_metric_name}",
        f"  value = {report['saved_training_metric']}",
        "  NOTE  = This is the scalar saved during training/validation, not epsilon/delta.",
        "",
        "[Residual horizon]",
        f"  N_res  = {config['residual_steps']}",
        f"  tau    = {config['residual_steps'] * config['dt']:.8g}",
        "  type   = short-horizon equal-time dynamical consistency",
        "  formula = mean|S_h^N_res(c0_pred) - S_h^N_res(c0_true)|",
        "",
    ]

    def append_metric(title, key):
        m = metrics[key]
        lines.extend([
            f"[{title}]",
            f"  mean +/- std        = {m['mean']:.8g} +/- {m['std']:.8g}",
            f"  min / median / max  = {m['min']:.8g} / {m['median']:.8g} / {m['max']:.8g}",
            f"  q05 / q95           = {m['q05']:.8g} / {m['q95']:.8g}",
            "",
        ])

    append_metric("Full test-set MAE", "mae")
    append_metric("Full test-set forward residual", "residual")
    append_metric("Relative Lyapunov energy error", "energy_rel")
    append_metric("Component mass error", "mass_error")
    append_metric("Simplex error", "simplex_error")
    append_metric("Purity error", "purity_error")

    lines.extend([
        "[Thresholds]",
        f"  epsilon = {epsilon:.8g}",
        f"  delta   = {delta:.8g}",
        f"  P(MAE <= epsilon)      = {p_mae['p']:.6f}  Wilson95={p_mae['wilson95']}",
        f"  P(Residual <= delta)   = {p_res['p']:.6f}  Wilson95={p_res['wilson95']}",
        f"  P(joint)               = {p_joint['p']:.6f}  Wilson95={p_joint['wilson95']}",
        f"  corr(MAE, Residual)    = {joint['corr_MAE_RES']:.6f}",
        "",
        "[Machine-learned inverse solution criterion]",
        f"  Criterion: P(MAE <= {epsilon:.6g} and Residual <= {delta:.6g}) >= {criterion['X']:.6g}",
        f"  Empirical value = {criterion['empirical_joint_probability']:.6f}",
        f"  Satisfied = {criterion['satisfied']}",
        "",
        "[LaTeX-ready sentence]",
        (
            f"\\noindent On the held-out test set, "
            f"$\\Pr(\\mathrm{{MAE}}\\le {epsilon:.6g})\\approx {p_mae['p']:.4f}$, "
            f"$\\Pr(\\mathrm{{Res}}_{{{config['residual_steps']}}}\\le {delta:.6g})\\approx {p_res['p']:.4f}$, "
            f"and $\\Pr(\\mathrm{{MAE}}\\le {epsilon:.6g}\\ \\wedge\\ "
            f"\\mathrm{{Res}}_{{{config['residual_steps']}}}\\le {delta:.6g})\\approx {p_joint['p']:.4f}$."
        ),
    ])

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# Model selection
# ============================================================

def model_specs(args):
    best_mae_model = (
        args.best_mae_model
        or os.path.join(args.model_dir, "best_generator_mae.keras")
    )
    best_mae_metric = (
        args.best_mae_metric
        or os.path.join(args.model_dir, "best_mae.txt")
    )

    best_res_model = (
        args.best_residual_model
        or os.path.join(args.model_dir, "best_generator_residual.keras")
    )
    best_res_metric = (
        args.best_residual_metric
        or os.path.join(args.model_dir, "best_residual.txt")
    )

    specs = []

    if not os.path.exists(best_mae_model):
        raise FileNotFoundError(f"Required model not found: {best_mae_model}")

    specs.append(
        {
            "name": "best_mae",
            "model_path": best_mae_model,
            "metric_path": best_mae_metric,
        }
    )

    if os.path.exists(best_res_model):
        specs.append(
            {
                "name": "best_residual",
                "model_path": best_res_model,
                "metric_path": best_res_metric,
            }
        )
    else:
        print(f"Optional residual model not found; skipping: {best_res_model}")

    return specs


# ============================================================
# Evaluation
# ============================================================

def evaluate_one_model(spec, test_src, test_tar, args, dx):
    model_name = spec["name"]
    model_outdir = os.path.join(args.outdir, model_name)
    os.makedirs(model_outdir, exist_ok=True)

    print(f"\n[{model_name}] Loading {spec['model_path']}")

    model = tf.keras.models.load_model(spec["model_path"], compile=False)
    saved_metric = read_scalar_text(spec["metric_path"])

    n_total = int(test_src.shape[0])
    n_eval = (
        min(n_total, args.max_test_samples)
        if args.max_test_samples
        else n_total
    )

    print(
        f"[{model_name}] Evaluating {n_eval} test samples "
        f"in batches of {args.predict_batch}"
    )

    chi = (args.chi12, args.chi13, args.chi23)
    domain_area = args.L**2

    metric_arrays: Dict[str, List[np.ndarray]] = {}

    sample_ids = np.array(
        [s for s in args.samples if 0 <= s < n_eval],
        dtype=int,
    )

    sample_src = []
    sample_pred = []
    sample_true = []
    sample_mae = []
    sample_res = []
    sample_int = []

    for start in range(0, n_eval, args.predict_batch):
        end = min(n_eval, start + args.predict_batch)

        src_b = batch_take(test_src, start, end)
        tar_b = batch_take(test_tar, start, end)

        pred_b = model.predict(
            src_b,
            batch_size=args.predict_batch,
            verbose=0,
        )
        pred_b = normalize_simplex_np(to_channel_last(pred_b))

        diag = diagnostics_per_batch(
            tar_b,
            pred_b,
            dx=dx,
            domain_area=domain_area,
            eps=args.eps,
            A=args.A,
            chi=chi,
        )

        residual = residual_per_sample(
            pred_b,
            tar_b,
            residual_steps=args.residual_steps,
            residual_batch=args.residual_batch,
            dt=args.dt,
            dx=dx,
            eps=args.eps,
            A=args.A,
            chi=chi,
            mobility=args.mobility,
        )

        diag["residual"] = residual

        for key, arr in diag.items():
            metric_arrays.setdefault(key, []).append(
                np.asarray(arr, dtype=np.float64)
            )

        for sid in sample_ids:
            if start <= sid < end:
                loc = int(sid - start)

                sample_src.append(src_b[loc])
                sample_pred.append(pred_b[loc])
                sample_true.append(tar_b[loc])
                sample_mae.append(diag["mae"][loc])
                sample_res.append(residual[loc])
                sample_int.append(diag["interfacial_density_error"][loc])

        print(f"[{model_name}] processed {end}/{n_eval}")

    metrics = {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in metric_arrays.items()
    }

    epsilon = (
        float(args.fixed_eps)
        if args.fixed_eps is not None
        else quantile_threshold(metrics["mae"], args.target_p_mae)
    )

    delta = (
        float(args.fixed_delta)
        if args.fixed_delta is not None
        else quantile_threshold(metrics["residual"], args.target_p_res)
    )

    joint = joint_probs(metrics["mae"], metrics["residual"], epsilon, delta)

    report = {
        "model_name": model_name,
        "saved_training_metric": saved_metric,
        "config": {
            "TRAIN_NPZ": args.train_npz,
            "TEST_NPZ": args.test_npz,
            "model_path": spec["model_path"],
            "N": int(args.N),
            "L": float(args.L),
            "dx": float(dx),
            "dt": float(args.dt),
            "eps": float(args.eps),
            "A": float(args.A),
            "chi": [float(x) for x in chi],
            "mobility": float(args.mobility),
            "residual_steps": int(args.residual_steps),
            "residual_type": "short-horizon equal-time dynamical consistency",
            "N_evaluated": int(n_eval),
        },
        "metrics": {
            key: metric_summary(values)
            for key, values in metrics.items()
        },
        "thresholds": {
            "epsilon": float(epsilon),
            "delta": float(delta),
        },
        "P_MAE_leq_eps": prob_leq(metrics["mae"], epsilon),
        "P_RES_leq_delta": prob_leq(metrics["residual"], delta),
        "joint": joint,
        "ml_solution_definition_check": {
            "X": float(args.target_p_joint),
            "empirical_joint_probability": float(joint["P_joint"]["p"]),
            "satisfied": bool(
                joint["P_joint"]["p"] >= args.target_p_joint
            ),
        },
    }

    for key, arr in metrics.items():
        np.save(
            os.path.join(model_outdir, f"{key}_per_sample.npy"),
            arr,
        )
        save_hist_cdf(arr, key, model_outdir)

    save_mae_residual_scatter(
        metrics["mae"],
        metrics["residual"],
        epsilon,
        delta,
        model_outdir,
    )

    if sample_pred:
        separate_images_outdir = os.path.join(
            model_outdir,
            "nine_images_300dpi",
        )
        os.makedirs(separate_images_outdir, exist_ok=True)

        save_separate_qualitative_images(
            src=np.stack(sample_src, axis=0),
            pred=np.stack(sample_pred, axis=0),
            true=np.stack(sample_true, axis=0),
            sample_ids=sample_ids[:len(sample_pred)],
            outdir=separate_images_outdir,
        )

        save_qualitative_plot(
            src=np.stack(sample_src, axis=0),
            pred=np.stack(sample_pred, axis=0),
            true=np.stack(sample_true, axis=0),
            sample_ids=sample_ids[:len(sample_pred)],
            panel_values={
                "mae": np.asarray(sample_mae),
                "residual": np.asarray(sample_res),
                "interfacial": np.asarray(sample_int),
            },
            model_name=model_name,
            outpng=os.path.join(
                model_outdir,
                f"{model_name}_qualitative.png",
            ),
        )

    with open(os.path.join(model_outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    write_text_report(
        report,
        os.path.join(model_outdir, "report.txt"),
    )

    print(f"[{model_name}] evaluation complete.")

    return report


# ============================================================
# Comparison
# ============================================================

def write_comparison(outdir, reports):
    path = os.path.join(outdir, "comparison_summary.txt")

    lines = [
        "=== tCHE inverse-model comparison ===",
        "",
    ]

    for name, report in reports.items():
        m = report["metrics"]

        lines.append(f"[{name}]")
        lines.append(
            f"MAE mean +/- std = "
            f"{m['mae']['mean']:.8g} +/- {m['mae']['std']:.8g}"
        )
        lines.append(
            f"Residual mean +/- std = "
            f"{m['residual']['mean']:.8g} +/- {m['residual']['std']:.8g}"
        )
        lines.append(
            f"Energy relative error mean = {m['energy_rel']['mean']:.8g}"
        )
        lines.append(
            f"Mass error mean = {m['mass_error']['mean']:.8g}"
        )
        lines.append(
            f"Interfacial-density error mean = "
            f"{m['interfacial_density_error']['mean']:.8g}"
        )
        lines.append(
            f"Joint probability = {report['joint']['P_joint']['p']:.6f}"
        )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved comparison: {path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    train_src, train_tar = load_npz_arrays(
        args.train_npz,
        args.src_key,
        args.tar_key,
    )

    test_src, test_tar = load_npz_arrays(
        args.test_npz,
        args.src_key,
        args.tar_key,
    )

    _, H, W, C = to_channel_last(test_src[:1]).shape

    if H != W:
        raise ValueError(f"Expected square grid, got H={H}, W={W}")

    if C != 3:
        raise ValueError(f"Expected three channels, got C={C}")

    if args.N is None:
        args.N = H

    if args.N != H:
        raise ValueError(
            f"--N={args.N} does not match dataset grid H={H}"
        )

    dx = args.L / args.N

    print("Data summary")
    print(
        f"  train src/tar raw shapes: "
        f"{train_src.shape} / {train_tar.shape}"
    )
    print(
        f"  test  src/tar raw shapes: "
        f"{test_src.shape} / {test_tar.shape}"
    )
    print(f"  inferred channel-last image shape: {(H, W, C)}")
    print(
        f"  dx={dx}, dt={args.dt}, eps={args.eps}, "
        f"residual_steps={args.residual_steps}"
    )

    reports = {}

    for spec in model_specs(args):
        reports[spec["name"]] = evaluate_one_model(
            spec=spec,
            test_src=test_src,
            test_tar=test_tar,
            args=args,
            dx=dx,
        )

    write_comparison(args.outdir, reports)


if __name__ == "__main__":
    main()
