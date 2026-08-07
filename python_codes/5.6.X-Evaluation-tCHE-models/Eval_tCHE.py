#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_two_inverse_models_tche.py

Unified full-test evaluator for ternary Cahn--Hilliard inverse generators.

Designed to mirror the Chafee--Infante two-model evaluator, but for the
3-channel ternary Cahn--Hilliard WGAN-GP training script.

Inverse problem:
    c_T  --->  c_0

Data convention:
    src = final ternary state c_T
    tar = initial ternary state c_0

Accepted NPZ array shapes:
    (N, H, W, 3)
    (N, 3, H, W)

For each available model, the script:
  - loads the generator,
  - evaluates it on the entire test set in batches,
  - computes per-sample MAE and physics residual,
  - computes ternary diagnostics: energy error, mass error, mean/variance error,
    purity error, gradient error, simplex error,
  - saves qualitative RGB plots,
  - saves histograms and CDFs,
  - computes epsilon and delta from empirical test-set quantiles,
  - computes joint probabilities,
  - writes model-specific JSON/TXT reports and a comparison summary.

Typical use:

    python evaluate_two_inverse_models_tche.py \
        --train-npz TRAINING_DATASET_PATH \
        --test-npz  TESTING_DATASET_PATH \
        --model-dir tCHE_3color_WGANGP_stats_energy_models \
        --outdir full_model_evaluation_Eyre-noisy \
        --residual-steps 200

For the 64-on-128 and Eyre noisy runs, change only the dataset paths, output
folder, and residual step count/physical parameters if needed.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ============================================================
# Command-line configuration
# ============================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-test evaluator for ternary Cahn--Hilliard inverse models."
    )

    # Datasets / models
    p.add_argument("--train-npz", required=True, help="Training NPZ, used for metadata/checks.")
    p.add_argument("--test-npz", required=True, help="Held-out test NPZ.")
    p.add_argument("--src-key", default="src")
    p.add_argument("--tar-key", default="tar")
    p.add_argument("--model-dir", default="tCHE_3color_WGANGP_stats_energy_models")
    p.add_argument("--best-mae-model", default=None)
    p.add_argument("--best-mae-metric", default=None)
    p.add_argument("--best-residual-model", default=None)
    p.add_argument("--best-residual-metric", default=None)
    p.add_argument("--outdir", default="4_full_model_evaluation_tche")

    # Numerical parameters: match the training script unless the dataset used different values.
    p.add_argument("--N", type=int, default=None, help="Grid size. If omitted, inferred from test data.")
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=5.0e-8)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--A", type=float, default=1.0)
    p.add_argument("--chi12", type=float, default=1.5)
    p.add_argument("--chi13", type=float, default=1.5)
    p.add_argument("--chi23", type=float, default=1.5)
    p.add_argument("--mob1", type=float, default=1.0)
    p.add_argument("--mob2", type=float, default=1.0)
    p.add_argument("--mob3", type=float, default=1.0)
    p.add_argument("--residual-steps", type=int, default=200)

    # Evaluation parameters
    p.add_argument("--predict-batch", type=int, default=4)
    p.add_argument("--residual-batch", type=int, default=4)
    p.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--target-p-mae", type=float, default=0.95)
    p.add_argument("--target-p-res", type=float, default=0.95)
    p.add_argument("--target-p-joint", type=float, default=0.90)
    p.add_argument("--fixed-eps", type=float, default=None)
    p.add_argument("--fixed-delta", type=float, default=None)
    p.add_argument("--max-test-samples", type=int, default=None,
                   help="Optional cap for a quick smoke test. Omit for full held-out test set.")

    return p.parse_args()


# ============================================================
# Shape / simplex utilities
# ============================================================


def to_channel_last(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))[None, ...]
        elif x.shape[-1] == 3:
            x = x[None, ...]
        else:
            raise ValueError(f"Cannot interpret 3D ternary shape {x.shape}.")
    elif x.ndim == 4:
        if x.shape[-1] == 3:
            pass
        elif x.shape[1] == 3:
            x = np.transpose(x, (0, 2, 3, 1))
        else:
            raise ValueError(f"Cannot interpret 4D ternary shape {x.shape}.")
    else:
        raise ValueError(f"Expected 3D or 4D ternary array, got shape {x.shape}.")
    return x.astype("float32", copy=False)


def normalize_simplex_np(c: np.ndarray, eps0: float = 1.0e-8) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    s = np.sum(c, axis=-1, keepdims=True)
    return (c / (s + eps0)).astype("float32", copy=False)


def load_npz_arrays(path: str, src_key: str, tar_key: str):
    z = np.load(path, mmap_mode="r")
    if src_key not in z or tar_key not in z:
        raise KeyError(
            f"Expected keys '{src_key}' and '{tar_key}' in {path}. "
            f"Available keys: {list(z.keys())}"
        )
    return z[src_key], z[tar_key]


def batch_take(arr, start: int, end: int) -> np.ndarray:
    return normalize_simplex_np(to_channel_last(arr[start:end]))


def infer_shape(arr) -> Tuple[int, int, int, int]:
    sample = to_channel_last(arr[:1])
    return sample.shape


# ============================================================
# Statistics helpers
# ============================================================


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2))
    return p, max(0.0, center - half), min(1.0, center + half)


def metric_summary(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    return {
        "N": int(n),
        "mean": float(v.mean()) if n else np.nan,
        "std": float(v.std(ddof=1)) if n > 1 else 0.0,
        "var": float(v.var(ddof=1)) if n > 1 else 0.0,
        "min": float(v.min()) if n else np.nan,
        "max": float(v.max()) if n else np.nan,
        "median": float(np.median(v)) if n else np.nan,
        "q05": float(np.quantile(v, 0.05)) if n else np.nan,
        "q95": float(np.quantile(v, 0.95)) if n else np.nan,
    }


def quantile_threshold(values: np.ndarray, target_p: float) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    idx = int(np.ceil(target_p * v.size)) - 1
    idx = max(0, min(idx, v.size - 1))
    return float(v[idx])


def choose_threshold(values, fixed_thr, target_p, name):
    if fixed_thr is not None:
        return float(fixed_thr), {"mode": "fixed", "metric": name, "value": float(fixed_thr)}
    thr = quantile_threshold(values, target_p)
    return thr, {"mode": "target_probability", "metric": name, "target_p": float(target_p), "value": thr}


def prob_leq(values, thr):
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.size
    k = int(np.sum(v <= thr))
    p, lo, hi = wilson_ci(k, n)
    return {"threshold": float(thr), "k": k, "n": int(n), "p": p, "wilson95": [lo, hi]}


def joint_probs(mae, res, eps, delta):
    mae = np.asarray(mae, dtype=np.float64).ravel()
    res = np.asarray(res, dtype=np.float64).ravel()
    n = mae.size
    A = mae <= eps
    B = res <= delta
    kA = int(np.sum(A))
    kB = int(np.sum(B))
    kJ = int(np.sum(A & B))
    pA = wilson_ci(kA, n)
    pB = wilson_ci(kB, n)
    pJ = wilson_ci(kJ, n)
    return {
        "P_MAE_leq_eps": {"p": pA[0], "wilson95": [pA[1], pA[2]], "k": kA, "n": n},
        "P_RES_leq_delta": {"p": pB[0], "wilson95": [pB[1], pB[2]], "k": kB, "n": n},
        "P_joint": {"p": pJ[0], "wilson95": [pJ[1], pJ[2]], "k": kJ, "n": n},
        "P_RES_given_MAE": float(kJ / kA) if kA else np.nan,
        "P_MAE_given_RES": float(kJ / kB) if kB else np.nan,
        "corr_MAE_RES": float(np.corrcoef(mae, res)[0, 1]) if n > 1 else np.nan,
    }


# ============================================================
# TensorFlow ternary Cahn--Hilliard operators
# ============================================================


def make_tche_ops(dx, dt, eps, A, chi, mobility):
    chi = tuple(float(x) for x in chi)
    mobility = tuple(float(x) for x in mobility)

    def normalize_simplex_tf(c, eps0=1e-8):
        c = tf.clip_by_value(c, 0.0, 1.0)
        s = tf.reduce_sum(c, axis=-1, keepdims=True)
        return c / (s + eps0)

    def simplex_error_per_sample_tf(c):
        positivity = tf.reduce_mean(tf.nn.relu(-c), axis=[1, 2, 3])
        upper = tf.reduce_mean(tf.nn.relu(c - 1.0), axis=[1, 2, 3])
        sum_err = tf.reduce_mean(tf.abs(tf.reduce_sum(c, axis=-1) - 1.0), axis=[1, 2])
        return positivity + upper + sum_err

    def laplacian_periodic_tf(u):
        return (
            tf.roll(u, 1, axis=1) + tf.roll(u, -1, axis=1)
            + tf.roll(u, 1, axis=2) + tf.roll(u, -1, axis=2)
            - 4.0 * u
        ) / (dx * dx)

    def bulk_derivative_tf(c):
        c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2]
        chi12, chi13, chi23 = chi
        dW1 = A * (2*c1 - 6*c1**2 + 4*c1**3) + chi12*c2 + chi13*c3
        dW2 = A * (2*c2 - 6*c2**2 + 4*c2**3) + chi12*c1 + chi23*c3
        dW3 = A * (2*c3 - 6*c3**2 + 4*c3**3) + chi13*c1 + chi23*c2
        return tf.stack([dW1, dW2, dW3], axis=-1)

    def chemical_potential_tf(c):
        mu0 = bulk_derivative_tf(c)
        mus = []
        for i in range(3):
            mus.append(mu0[..., i] - eps**2 * laplacian_periodic_tf(c[..., i]))
        mu = tf.stack(mus, axis=-1)
        return mu - tf.reduce_mean(mu, axis=-1, keepdims=True)

    def forward_euler_rhs_tf(c):
        mu = chemical_potential_tf(c)
        rhs = []
        for i in range(3):
            rhs.append(tf.cast(mobility[i], tf.float32) * laplacian_periodic_tf(mu[..., i]))
        rhs = tf.stack(rhs, axis=-1)
        return rhs - tf.reduce_mean(rhs, axis=-1, keepdims=True)

    def ternary_energy_per_sample_tf(c):
        c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2]
        chi12, chi13, chi23 = chi
        bulk = (
            A * tf.reduce_sum(c**2 * (1.0 - c)**2, axis=-1)
            + chi12*c1*c2 + chi13*c1*c3 + chi23*c2*c3
        )
        grad_part = 0.0
        for i in range(3):
            ci = c[..., i]
            cx = (tf.roll(ci, -1, axis=1) - ci) / dx
            cy = (tf.roll(ci, -1, axis=2) - ci) / dx
            grad_part = grad_part + 0.5 * eps**2 * (cx**2 + cy**2)
        return dx**2 * tf.reduce_sum(bulk + grad_part, axis=[1, 2])

    @tf.function
    def forward_sim_tche_tf(c0, nsteps):
        c = normalize_simplex_tf(c0)
        for _ in tf.range(nsteps):
            rhs = forward_euler_rhs_tf(c)
            c = c + tf.cast(dt, tf.float32) * rhs
            c = normalize_simplex_tf(c)
        return c

    return {
        "normalize_simplex_tf": normalize_simplex_tf,
        "simplex_error_per_sample_tf": simplex_error_per_sample_tf,
        "ternary_energy_per_sample_tf": ternary_energy_per_sample_tf,
        "forward_sim_tche_tf": forward_sim_tche_tf,
    }


# ============================================================
# Per-batch diagnostics
# ============================================================


def diagnostics_per_batch(real_np: np.ndarray, fake_np: np.ndarray, ops) -> Dict[str, np.ndarray]:
    real = tf.convert_to_tensor(real_np, dtype=tf.float32)
    fake = tf.convert_to_tensor(fake_np, dtype=tf.float32)

    mae = tf.reduce_mean(tf.abs(fake - real), axis=[1, 2, 3])

    E_real = ops["ternary_energy_per_sample_tf"](real)
    E_fake = ops["ternary_energy_per_sample_tf"](fake)
    energy_abs = tf.abs(E_fake - E_real)
    energy_rel = energy_abs / (tf.abs(E_real) + 1.0e-8)

    mean_real = tf.reduce_mean(real, axis=[1, 2])
    mean_fake = tf.reduce_mean(fake, axis=[1, 2])
    var_real = tf.math.reduce_variance(real, axis=[1, 2])
    var_fake = tf.math.reduce_variance(fake, axis=[1, 2])
    mean_err = tf.reduce_mean(tf.abs(mean_fake - mean_real), axis=1)
    var_err = tf.reduce_mean(tf.abs(var_fake - var_real), axis=1)
    mass_err = mean_err

    purity = tf.reduce_sum(fake**2, axis=-1)
    purity_err = tf.reduce_mean(tf.abs(1.0 - purity), axis=[1, 2])

    real_dx = real[:, 1:, :, :] - real[:, :-1, :, :]
    fake_dx = fake[:, 1:, :, :] - fake[:, :-1, :, :]
    real_dy = real[:, :, 1:, :] - real[:, :, :-1, :]
    fake_dy = fake[:, :, 1:, :] - fake[:, :, :-1, :]
    grad_err = (
        tf.reduce_mean(tf.abs(fake_dx - real_dx), axis=[1, 2, 3])
        + tf.reduce_mean(tf.abs(fake_dy - real_dy), axis=[1, 2, 3])
    )

    simplex_err = ops["simplex_error_per_sample_tf"](fake)

    sums = tf.reduce_sum(fake, axis=-1)
    gen_mean = tf.reduce_mean(fake, axis=[1, 2])
    gen_var = tf.math.reduce_variance(fake, axis=[1, 2])

    return {
        "mae": mae.numpy().astype(np.float64),
        "energy_abs": energy_abs.numpy().astype(np.float64),
        "energy_rel": energy_rel.numpy().astype(np.float64),
        "mean_error": mean_err.numpy().astype(np.float64),
        "variance_error": var_err.numpy().astype(np.float64),
        "mass_error": mass_err.numpy().astype(np.float64),
        "purity_error": purity_err.numpy().astype(np.float64),
        "gradient_error": grad_err.numpy().astype(np.float64),
        "simplex_error": simplex_err.numpy().astype(np.float64),
        "gen_min": tf.reduce_min(fake, axis=[1, 2, 3]).numpy().astype(np.float64),
        "gen_max": tf.reduce_max(fake, axis=[1, 2, 3]).numpy().astype(np.float64),
        "gen_sum_min": tf.reduce_min(sums, axis=[1, 2]).numpy().astype(np.float64),
        "gen_sum_max": tf.reduce_max(sums, axis=[1, 2]).numpy().astype(np.float64),
        "gen_mean_c1": gen_mean[:, 0].numpy().astype(np.float64),
        "gen_mean_c2": gen_mean[:, 1].numpy().astype(np.float64),
        "gen_mean_c3": gen_mean[:, 2].numpy().astype(np.float64),
        "gen_var_c1": gen_var[:, 0].numpy().astype(np.float64),
        "gen_var_c2": gen_var[:, 1].numpy().astype(np.float64),
        "gen_var_c3": gen_var[:, 2].numpy().astype(np.float64),
    }


def residual_per_sample(pred_np: np.ndarray, src_np: np.ndarray, ops, residual_steps: int, residual_batch: int) -> np.ndarray:
    out = []
    for start in range(0, pred_np.shape[0], residual_batch):
        end = min(pred_np.shape[0], start + residual_batch)
        c0 = tf.convert_to_tensor(pred_np[start:end], dtype=tf.float32)
        cT = tf.convert_to_tensor(src_np[start:end], dtype=tf.float32)
        cT_sim = ops["forward_sim_tche_tf"](c0, tf.constant(residual_steps, dtype=tf.int32))
        r = tf.reduce_mean(tf.abs(cT_sim - cT), axis=[1, 2, 3])
        out.append(r.numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


# ============================================================
# Plotting / IO helpers
# ============================================================


def read_scalar_text(path: Optional[str]) -> Optional[float]:
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return None


def save_hist_cdf(values: np.ndarray, name: str, outdir: str):
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return

    stem = name.lower()
    plt.figure()
    plt.hist(v, bins=60)
    plt.title(f"{name} histogram")
    plt.xlabel(name)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{stem}_hist.png"), dpi=200)
    plt.close()

    vs = np.sort(v)
    cdf = np.arange(1, vs.size + 1) / vs.size
    plt.figure()
    plt.plot(vs, cdf)
    plt.title(f"{name} empirical CDF")
    plt.xlabel(name)
    plt.ylabel("P(value <= t)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{stem}_cdf.png"), dpi=200)
    plt.close()


def save_qualitative_plot(src, pred, true, outpng, samples, model_name, training_metric_name=None, training_metric_value=None, panel_values=None, individual_dir=None):
    if individual_dir is not None:
        os.makedirs(individual_dir, exist_ok=True)

    nrows = len(samples)
    fig, axes = plt.subplots(nrows, 3, figsize=(9, max(4, 3.2 * nrows + 1.8)))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.18, wspace=0.05, hspace=0.10)
    col_titles = ["given $c_T$", "generated $\\hat c_0$", "true $c_0$"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13)

    for i, idx in enumerate(samples):
        fields = [src[i], pred[i], true[i]]
        for j, field in enumerate(fields):
            axes[i, j].imshow(np.clip(field, 0.0, 1.0), origin="lower", interpolation="nearest")
            axes[i, j].axis("off")

        if individual_dir is not None:
            for label, field in zip(["SRC_cT", "GEN_c0", "TRUE_c0"], fields):
                plt.figure(figsize=(4, 4))
                plt.imshow(np.clip(field, 0.0, 1.0), origin="lower", interpolation="nearest")
                plt.axis("off")
                plt.savefig(os.path.join(individual_dir, f"sample_{int(idx):04d}_{label}.png"), dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close()

    lines = [f"Model = {model_name}"]
    if training_metric_name is not None and training_metric_value is not None:
        lines.append(f"{training_metric_name} = {training_metric_value:.8g}")
    if panel_values is not None:
        lines.append("Per-row MAE / residual:")
        for i, idx in enumerate(samples):
            lines.append(f"  sample {int(idx)}: MAE={panel_values['mae'][i]:.6g}, RES={panel_values['residual'][i]:.6g}")

    fig.text(
        0.02, 0.03, "\n".join(lines), ha="left", va="bottom", fontsize=9,
        family="monospace", bbox=dict(facecolor="white", alpha=0.95, edgecolor="black", boxstyle="round,pad=0.3")
    )
    plt.savefig(outpng, dpi=200)
    plt.close(fig)


def write_txt_report(path: str, report: Dict):
    m = report["metrics"]
    lines = []
    lines.append("=== Ternary Cahn--Hilliard Inverse Model Evaluation Report ===")
    lines.append(f"Model name : {report['model_name']}")
    lines.append(f"Train set  : {report['config']['TRAIN_NPZ']}")
    lines.append(f"Test set   : {report['config']['TEST_NPZ']}")
    lines.append(f"Model path : {report['config']['model_path']}")
    lines.append("")
    lines.append("[Saved training metric]")
    lines.append(f"  name  = {report['saved_training_metric']['name']}")
    lines.append(f"  value = {report['saved_training_metric']['value']}")
    lines.append("  NOTE  = This is the scalar saved during training/validation, not epsilon/delta.")
    lines.append("")

    for key, title in [
        ("mae", "Full test-set MAE"),
        ("residual", "Full test-set forward residual"),
        ("energy_rel", "Relative Lyapunov energy error"),
        ("mass_error", "Component mass error"),
        ("simplex_error", "Simplex error"),
        ("purity_error", "Purity error"),
    ]:
        s = m[key]
        lines.append(f"[{title}]")
        lines.append(f"  mean +/- std        = {s['mean']:.8g} +/- {s['std']:.8g}")
        lines.append(f"  min / median / max  = {s['min']:.8g} / {s['median']:.8g} / {s['max']:.8g}")
        lines.append(f"  q05 / q95           = {s['q05']:.8g} / {s['q95']:.8g}")
        lines.append("")

    lines.append("[Thresholds]")
    lines.append(f"  epsilon = {report['thresholds']['epsilon']:.8g}")
    lines.append(f"  delta   = {report['thresholds']['delta']:.8g}")
    lines.append(f"  P(MAE <= epsilon)      = {report['P_MAE_leq_eps']['p']:.6f}  Wilson95={report['P_MAE_leq_eps']['wilson95']}")
    lines.append(f"  P(Residual <= delta)   = {report['P_RES_leq_delta']['p']:.6f}  Wilson95={report['P_RES_leq_delta']['wilson95']}")
    lines.append(f"  P(joint)               = {report['joint']['P_joint']['p']:.6f}  Wilson95={report['joint']['P_joint']['wilson95']}")
    lines.append(f"  corr(MAE, Residual)    = {report['joint']['corr_MAE_RES']:.6f}")
    lines.append("")
    d = report["ml_solution_definition_check"]
    lines.append("[Machine-learned inverse solution criterion]")
    lines.append(f"  Criterion: P(MAE <= {d['epsilon']:.6g} and Residual <= {d['delta']:.6g}) >= {d['X']:.6g}")
    lines.append(f"  Empirical value = {d['empirical_joint_probability']:.6f}")
    lines.append(f"  Satisfied = {d['satisfied']}")
    lines.append("")
    lines.append("[LaTeX-ready sentence]")
    lines.append(
        r"\noindent On the held-out test set, "
        + rf"$P(\mathrm{{MAE}}\le {report['thresholds']['epsilon']:.6g})\approx {report['P_MAE_leq_eps']['p']:.4f}$, "
        + rf"$P(\mathrm{{Res}}\le {report['thresholds']['delta']:.6g})\approx {report['P_RES_leq_delta']['p']:.4f}$, "
        + rf"and $P(\mathrm{{MAE}}\le {report['thresholds']['epsilon']:.6g}\ \wedge\ \mathrm{{Res}}\le {report['thresholds']['delta']:.6g})\approx {report['joint']['P_joint']['p']:.4f}$."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ============================================================
# Evaluation core
# ============================================================


def model_specs(args):
    best_mae_model = args.best_mae_model or os.path.join(args.model_dir, "best_generator_mae.keras")
    best_mae_metric = args.best_mae_metric or os.path.join(args.model_dir, "best_mae.txt")
    best_res_model = args.best_residual_model or os.path.join(args.model_dir, "best_generator_residual.keras")
    best_res_metric = args.best_residual_metric or os.path.join(args.model_dir, "best_residual.txt")

    specs = []
    if os.path.exists(best_mae_model):
        specs.append({
            "name": "best_mae",
            "model_path": best_mae_model,
            "metric_path": best_mae_metric,
            "training_metric_name": "validation_mae_saved_during_training",
        })
    else:
        raise FileNotFoundError(f"Could not find required MAE model: {best_mae_model}")

    if os.path.exists(best_res_model):
        specs.append({
            "name": "best_residual",
            "model_path": best_res_model,
            "metric_path": best_res_metric,
            "training_metric_name": "validation_residual_saved_during_training",
        })
    else:
        print(f"Optional residual model not found; skipping: {best_res_model}")
    return specs


def evaluate_one_model(spec, train_src, train_tar, test_src, test_tar, args, ops, dx):
    model_name = spec["name"]
    model_outdir = os.path.join(args.outdir, model_name)
    os.makedirs(model_outdir, exist_ok=True)
    individual_dir = os.path.join(model_outdir, "individual")

    print(f"\n[{model_name}] Loading {spec['model_path']}")
    model = tf.keras.models.load_model(spec["model_path"], compile=False)
    saved_training_metric = read_scalar_text(spec.get("metric_path"))

    n_total = int(test_src.shape[0])
    n_eval = min(n_total, args.max_test_samples) if args.max_test_samples else n_total
    print(f"[{model_name}] Evaluating {n_eval} test samples in batches of {args.predict_batch}")

    metric_arrays: Dict[str, List[np.ndarray]] = {}

    # Qualitative sample cache.
    sample_ids = np.array([s for s in args.samples if 0 <= s < n_eval], dtype=int)
    sample_src = batch_take(test_src, sample_ids[0], sample_ids[0] + 1) if len(sample_ids) else None
    sample_pred = []
    sample_true = []
    sample_srcs = []
    sample_mae = []
    sample_res = []

    for start in range(0, n_eval, args.predict_batch):
        end = min(n_eval, start + args.predict_batch)
        src_b = batch_take(test_src, start, end)
        tar_b = batch_take(test_tar, start, end)
        pred_b = model.predict(src_b, batch_size=args.predict_batch, verbose=0)
        pred_b = normalize_simplex_np(to_channel_last(pred_b))

        diag = diagnostics_per_batch(tar_b, pred_b, ops)
        res = residual_per_sample(pred_b, src_b, ops, args.residual_steps, args.residual_batch)
        diag["residual"] = res

        for key, arr in diag.items():
            metric_arrays.setdefault(key, []).append(np.asarray(arr, dtype=np.float64))

        # Cache requested samples for plotting.
        for sid in sample_ids:
            if start <= sid < end:
                loc = int(sid - start)
                sample_srcs.append(src_b[loc])
                sample_pred.append(pred_b[loc])
                sample_true.append(tar_b[loc])
                sample_mae.append(diag["mae"][loc])
                sample_res.append(res[loc])

        print(f"[{model_name}] processed {end}/{n_eval}")

    metrics = {k: np.concatenate(v, axis=0) for k, v in metric_arrays.items()}

    eps, eps_info = choose_threshold(metrics["mae"], args.fixed_eps, args.target_p_mae, "MAE")
    delta, delta_info = choose_threshold(metrics["residual"], args.fixed_delta, args.target_p_res, "Residual")
    joint = joint_probs(metrics["mae"], metrics["residual"], eps, delta)

    report = {
        "model_name": model_name,
        "config": {
            "TRAIN_NPZ": args.train_npz,
            "TEST_NPZ": args.test_npz,
            "model_path": spec["model_path"],
            "metric_path": spec.get("metric_path"),
            "src_key": args.src_key,
            "tar_key": args.tar_key,
            "N": int(args.N),
            "L": float(args.L),
            "dx": float(dx),
            "dt": float(args.dt),
            "eps": float(args.eps),
            "A": float(args.A),
            "chi": [float(args.chi12), float(args.chi13), float(args.chi23)],
            "mobility": [float(args.mob1), float(args.mob2), float(args.mob3)],
            "RESIDUAL_STEPS": int(args.residual_steps),
            "PREDICT_BATCH": int(args.predict_batch),
            "RESIDUAL_BATCH": int(args.residual_batch),
            "TARGET_P_MAE": float(args.target_p_mae),
            "TARGET_P_RES": float(args.target_p_res),
            "TARGET_P_JOINT": float(args.target_p_joint),
            "FIXED_EPS": args.fixed_eps,
            "FIXED_DELTA": args.fixed_delta,
            "N_EVALUATED": int(n_eval),
        },
        "saved_training_metric": {
            "name": spec.get("training_metric_name", "saved_training_metric"),
            "value": saved_training_metric,
            "note": "Saved during training/validation; not the same as test-set epsilon/delta.",
        },
        "metrics": {k: metric_summary(v) for k, v in metrics.items()},
        "thresholds": {
            "epsilon": float(eps),
            "delta": float(delta),
            "epsilon_selection": eps_info,
            "delta_selection": delta_info,
        },
        "P_MAE_leq_eps": prob_leq(metrics["mae"], eps),
        "P_RES_leq_delta": prob_leq(metrics["residual"], delta),
        "joint": joint,
        "ml_solution_definition_check": {
            "definition": "Declare success if P(MAE <= epsilon and Residual <= delta) >= X.",
            "epsilon": float(eps),
            "delta": float(delta),
            "X": float(args.target_p_joint),
            "empirical_joint_probability": float(joint["P_joint"]["p"]),
            "satisfied": bool(joint["P_joint"]["p"] >= args.target_p_joint),
        },
    }

    # Save raw per-sample arrays and metric plots.
    for key, arr in metrics.items():
        np.save(os.path.join(model_outdir, f"{key}_per_sample.npy"), arr)
        save_hist_cdf(arr, key, model_outdir)

    # Save qualitative panel.
    if len(sample_pred) > 0:
        panel_values = {"mae": np.asarray(sample_mae), "residual": np.asarray(sample_res)}
        save_qualitative_plot(
            np.stack(sample_srcs, axis=0), np.stack(sample_pred, axis=0), np.stack(sample_true, axis=0),
            os.path.join(model_outdir, f"{model_name}_qualitative_plot.png"),
            sample_ids[:len(sample_pred)], model_name,
            training_metric_name=report["saved_training_metric"]["name"],
            training_metric_value=saved_training_metric,
            panel_values=panel_values,
            individual_dir=individual_dir,
        )

    json_path = os.path.join(model_outdir, "report.json")
    txt_path = os.path.join(model_outdir, "report.txt")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    write_txt_report(txt_path, report)

    print(f"[{model_name}] Saved report: {txt_path}")
    print(f"[{model_name}] Saved JSON  : {json_path}")
    return report


def write_comparison(outdir: str, reports: Dict[str, Dict]):
    comp_json = os.path.join(outdir, "comparison_summary.json")
    comp_txt = os.path.join(outdir, "comparison_summary.txt")
    with open(comp_json, "w") as f:
        json.dump(reports, f, indent=2)

    lines = ["=== Comparison Summary: ternary best-MAE vs best-Residual generators ===", ""]
    for key, r in reports.items():
        m = r["metrics"]["mae"]
        rs = r["metrics"]["residual"]
        er = r["metrics"]["energy_rel"]
        se = r["metrics"]["simplex_error"]
        lines.append(f"[{key}]")
        lines.append(f"  saved training metric ({r['saved_training_metric']['name']}) = {r['saved_training_metric']['value']}")
        lines.append(f"  full test MAE mean +/- std       = {m['mean']:.8g} +/- {m['std']:.8g}")
        lines.append(f"  full test Residual mean +/- std  = {rs['mean']:.8g} +/- {rs['std']:.8g}")
        lines.append(f"  energy_rel mean +/- std          = {er['mean']:.8g} +/- {er['std']:.8g}")
        lines.append(f"  simplex_error mean +/- std       = {se['mean']:.8g} +/- {se['std']:.8g}")
        lines.append(f"  epsilon = {r['thresholds']['epsilon']:.8g}, delta = {r['thresholds']['delta']:.8g}")
        lines.append(f"  joint probability = {r['joint']['P_joint']['p']:.6f}")
        lines.append("")

    with open(comp_txt, "w") as f:
        f.write("\n".join(lines))
    print("\n[comparison] Saved:")
    print(f"  {comp_json}")
    print(f"  {comp_txt}")


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

    train_src, train_tar = load_npz_arrays(args.train_npz, args.src_key, args.tar_key)
    test_src, test_tar = load_npz_arrays(args.test_npz, args.src_key, args.tar_key)

    _, H, W, C = infer_shape(test_src)
    if C != 3 or H != W:
        raise ValueError(f"Expected square 3-channel ternary data; got H={H}, W={W}, C={C}.")
    if args.N is None:
        args.N = H
    dx = args.L / args.N

    print("Data summary")
    print(f"  train src/tar raw shapes: {train_src.shape} / {train_tar.shape}")
    print(f"  test  src/tar raw shapes: {test_src.shape} / {test_tar.shape}")
    print(f"  inferred channel-last image shape: {(H, W, C)}")
    print(f"  dx={dx}, dt={args.dt}, eps={args.eps}, residual_steps={args.residual_steps}")

    ops = make_tche_ops(
        dx=dx,
        dt=args.dt,
        eps=args.eps,
        A=args.A,
        chi=(args.chi12, args.chi13, args.chi23),
        mobility=(args.mob1, args.mob2, args.mob3),
    )

    reports = {}
    for spec in model_specs(args):
        reports[spec["name"]] = evaluate_one_model(spec, train_src, train_tar, test_src, test_tar, args, ops, dx)

    write_comparison(args.outdir, reports)


if __name__ == "__main__":
    main()
