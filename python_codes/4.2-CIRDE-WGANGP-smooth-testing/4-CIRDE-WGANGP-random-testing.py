"""
evaluate_two_inverse_models_chafee_infante.py

One unified evaluation script for TWO generators:
    1. best_generator_mae.keras
    2. best_generator_residual.keras

For each model, the script:
  - loads the generator,
  - evaluates it on the ENTIRE test set,
  - saves a 3x3 qualitative plot,
  - computes full-test per-sample MAE and residual,
  - reports summary statistics,
  - computes epsilon and delta from empirical quantiles
    (or fixed thresholds if provided),
  - computes joint probabilities.

It then writes:
  - one report for best-MAE generator,
  - one report for best-residual generator,
  - one comparison summary.

Important distinction:
  - saved_best_metric_from_training:
        value read from best_mae.txt / best_residual.txt
  - epsilon / delta:
        thresholds chosen from held-out TEST-set quantiles
        so that P(metric <= threshold) ~= target probability

These are DIFFERENT quantities and should not be expected to match.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# USER CONFIGURATION
# ============================================================

SIM_STEPS = 400

TRAIN_NPZ       = '../../../../CIRDE_Datasets/128x128_kappa=4.7_DBC_uniformly_random_data_400iterations/train-dataset_128x128_DBC_kappa=4.7_UR400_count=50000.npz'

TEST_NPZ        = '../../../../CIRDE_Datasets/128x128_kappa=4.7_DBC_uniformly_random_data_400iterations/test-dataset_128x128_DBC_kappa=4.7_UR400_count=10000.npz'

BEST_MAE_MODEL_PATH      = '3_models/best_generator_mae.keras'
BEST_MAE_METRIC_PATH     = '3_models/best_mae.txt'

BEST_RES_MODEL_PATH      = '3_models/best_generator_residual.keras'
BEST_RES_METRIC_PATH     = '3_models/best_residual.txt'


MODEL_SPECS = []

# Always try MAE model first
if os.path.exists(BEST_MAE_MODEL_PATH):
    MODEL_SPECS.append({
        "name": "best_mae",
        "model_path": BEST_MAE_MODEL_PATH,
        "metric_path": BEST_MAE_METRIC_PATH,
        "training_metric_name": "validation_mae_saved_during_training",
    })
else:
    raise FileNotFoundError(f"Could not find required MAE model: {BEST_MAE_MODEL_PATH}")

# Residual model is optional
if os.path.exists(BEST_RES_MODEL_PATH):
    MODEL_SPECS.append({
        "name": "best_residual",
        "model_path": BEST_RES_MODEL_PATH,
        "metric_path": BEST_RES_METRIC_PATH,
        "training_metric_name": "validation_residual_saved_during_training",
    })
    
OUTDIR = '4_full_model_evaluation'
os.makedirs(OUTDIR, exist_ok=True)

BATCH_SIZE = 8
RESIDUAL_BATCH = 32
SAMPLES = [0, 1, 2]

NPZ_KEY_INPUT = 'src'
NPZ_KEY_TARGET = 'tar'


# ------------------------------------------------------------
# THRESHOLD SELECTION
# ------------------------------------------------------------
# These define TEST-SET quantile thresholds:
# epsilon = quantile of test-set MAE values
# delta   = quantile of test-set residual values
TARGET_P_MAE = 0.95
TARGET_P_RES = 0.95

# If you want fixed thresholds instead, set these to numbers.
FIXED_EPS = None
FIXED_DELTA = None

# Optional success criterion for joint event
TARGET_P_JOINT = 0.90


# ============================================================
# PDE / FORWARD SIMULATION PARAMETERS
# ============================================================

dt = 0.001
gamma = 0.005
kappa = 4.7

L = 1.0
M = 128
h = 2.0 * L / M


# ============================================================
# HELPERS: STATS
# ============================================================

def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return p, lo, hi


def metric_summary(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    n = values.size
    mean = float(values.mean()) if n else np.nan
    var = float(values.var(ddof=1)) if n > 1 else 0.0
    std = float(np.sqrt(var))
    return {
        "N": int(n),
        "mean": mean,
        "std": std,
        "var": var,
        "min": float(values.min()) if n else np.nan,
        "max": float(values.max()) if n else np.nan,
        "median": float(np.median(values)) if n else np.nan,
        "q05": float(np.quantile(values, 0.05)) if n else np.nan,
        "q95": float(np.quantile(values, 0.95)) if n else np.nan,
    }


def prob_leq(values, thr):
    values = np.asarray(values, dtype=np.float64).ravel()
    n = values.size
    k = int(np.sum(values <= thr))
    p, lo, hi = wilson_ci(k, n)
    return {
        "threshold": float(thr),
        "k": k,
        "n": int(n),
        "p": p,
        "wilson95": [lo, hi],
    }


def quantile_threshold(values, target_p):
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = v.size
    if n == 0:
        return np.nan
    idx = int(np.ceil(target_p * n)) - 1
    idx = max(0, min(idx, n - 1))
    return float(v[idx])


def choose_threshold(values, fixed_thr, target_p, name):
    if fixed_thr is not None:
        return float(fixed_thr), {
            "mode": "fixed",
            "metric": name,
            "value": float(fixed_thr),
        }

    if target_p is not None:
        thr = quantile_threshold(values, target_p)
        return float(thr), {
            "mode": "target_probability",
            "metric": name,
            "target_p": float(target_p),
            "value": float(thr),
        }

    return None, {
        "mode": "none",
        "metric": name,
        "value": None,
    }


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

    p_res_given_mae = (kJ / kA) if kA > 0 else np.nan
    p_mae_given_res = (kJ / kB) if kB > 0 else np.nan
    corr = float(np.corrcoef(mae, res)[0, 1]) if n > 1 else np.nan

    return {
        "P_MAE_leq_eps": {"p": pA[0], "wilson95": [pA[1], pA[2]], "k": kA, "n": n},
        "P_RES_leq_delta": {"p": pB[0], "wilson95": [pB[1], pB[2]], "k": kB, "n": n},
        "P_joint": {"p": pJ[0], "wilson95": [pJ[1], pJ[2]], "k": kJ, "n": n},
        "P_RES_given_MAE": p_res_given_mae,
        "P_MAE_given_RES": p_mae_given_res,
        "corr_MAE_RES": corr,
    }


# ============================================================
# HELPERS: IO
# ============================================================

def load_npz_xy(npz_path, key_x, key_y):
    data = np.load(npz_path, mmap_mode="r")
    if key_x not in data or key_y not in data:
        raise KeyError(f"NPZ keys not found. Available keys: {list(data.keys())}")
    return data[key_x], data[key_y]


def load_generator_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def read_scalar_text(path):
    if (path is None) or (not os.path.exists(path)):
        return None
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return None


# ============================================================
# HELPERS: PLOTTING
# ============================================================

def save_hist_cdf(values, name, outdir):
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return

    plt.figure()
    plt.hist(v, bins=60)
    plt.title(f"{name} histogram")
    plt.xlabel(name)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name.lower()}_hist.png"), dpi=200)
    plt.close()

    vs = np.sort(v)
    cdf = np.arange(1, vs.size + 1) / vs.size

    plt.figure()
    plt.plot(vs, cdf)
    plt.title(f"{name} empirical CDF")
    plt.xlabel(name)
    plt.ylabel("P(value <= t)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name.lower()}_cdf.png"), dpi=200)
    plt.close()

# ============================================================
# HELPERS: PLOTTING
# ============================================================

def save_hist_cdf(values, name, outdir):
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return

    plt.figure()
    plt.hist(v, bins=60)
    plt.title(f"{name} histogram")
    plt.xlabel(name)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name.lower()}_hist.png"), dpi=200)
    plt.close()

    vs = np.sort(v)
    cdf = np.arange(1, vs.size + 1) / vs.size

    plt.figure()
    plt.plot(vs, cdf)
    plt.title(f"{name} empirical CDF")
    plt.xlabel(name)
    plt.ylabel("P(value <= t)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name.lower()}_cdf.png"), dpi=200)
    plt.close()


def save_qualitative_plot(
    x_src, x_gen, x_true,
    ne_scale, ic_scale,
    outpng, samples,
    training_metric_name=None,
    training_metric_value=None,
    model_name=None,
    panel_metric_name="MAE",
    panel_metric_values=None,
    save_individual=True,
    individual_dir=None,
):
    if individual_dir is None:
        individual_dir = INDIVPICOUTDIR

    if save_individual:
        os.makedirs(individual_dir, exist_ok=True)

    nrows = len(samples)

    # Make figure tall enough to reserve a true footer area.
    fig, axes = plt.subplots(nrows, 3, figsize=(9, 11.5))

    # If only one row, axes may come back 1D; force 2D shape.
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Reserve bottom space for text block so it does NOT overlap images.
    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.94,
        bottom=0.20,
        wspace=0.05,
        hspace=0.10,
    )

    col_titles = ["NE (scaled)", "Gen IC (scaled)", "True IC (scaled)"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=14)

    for i in range(nrows):
        ne = x_src[i, :, :, 0]
        gen = x_gen[i, :, :, 0]
        true = x_true[i, :, :, 0]

        axes[i, 0].imshow(ne, cmap="Greys", interpolation="none")
        axes[i, 1].imshow(gen, cmap="Greys", interpolation="none")
        axes[i, 2].imshow(true, cmap="Greys", interpolation="none")

        for j in range(3):
            axes[i, j].axis("off")

        # Save individual images INSIDE the model-specific directory.
        if save_individual:
            idx = int(samples[i])

            for name, img in zip(
                ["NE", "GEN", "TRUE"],
                [ne, gen, true]
            ):
                fig_one = plt.figure(figsize=(4, 4))
                plt.imshow(img, cmap="Greys", interpolation="none")
                plt.axis("off")

                fname = os.path.join(
                    individual_dir,
                    f"sample_{idx:04d}_{name}.png"
                )
                plt.savefig(fname, dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close(fig_one)

    # Footer text block
    lines = []

    if model_name is not None:
        lines.append(f"Model = {model_name}")

    if (training_metric_name is not None) and (training_metric_value is not None):
        lines.append(f"{training_metric_name} = {training_metric_value:.6f}")

    lines.append(f"Scales: NE_SCALE = {ne_scale:.6g}    IC_SCALE = {ic_scale:.6g}")

    if panel_metric_values is not None:
        lines.append(f"Per-row {panel_metric_name} values:")
        for i, idx in enumerate(samples):
            lines.append(f"  sample {int(idx)}: {panel_metric_values[i]:.6f}")

    fig.text(
        0.02,
        0.03,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(
            facecolor="white",
            alpha=0.95,
            edgecolor="black",
            boxstyle="round,pad=0.3",
        ),
    )

    # IMPORTANT: do NOT use bbox_inches="tight" on the full grid figure.
    # That was part of what caused the footer/text layout trouble.
    plt.savefig(outpng, dpi=200)
    plt.close(fig)

# ============================================================
# HELPERS: PHYSICS RESIDUAL
# ============================================================

def make_dirichlet_mask_tf(H, W, dtype):
    mask = np.ones((H, W), dtype=np.float32)
    mask[0, :] = 0.0
    mask[-1, :] = 0.0
    mask[:, 0] = 0.0
    mask[:, -1] = 0.0
    mask = tf.constant(mask, dtype=dtype)
    return mask[None, ..., None]


def residual_per_sample(y_pred_np: np.ndarray, y_obs_np: np.ndarray) -> np.ndarray:
    y_pred = tf.convert_to_tensor(y_pred_np, dtype=tf.float32)
    y_obs = tf.convert_to_tensor(y_obs_np, dtype=tf.float32)

    N, H, W, C = y_pred.shape
    assert C == 1, "Expected channel dimension C=1."

    lap = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=np.float32) / (h * h)
    lap_k = tf.constant(lap.reshape(3, 3, 1, 1), dtype=tf.float32)

    dbc_mask = make_dirichlet_mask_tf(H, W, tf.float32)

    @tf.function
    def forward_sim(u0):
        u = u0 * dbc_mask
        for _ in tf.range(SIM_STEPS):
            du = tf.nn.conv2d(u, lap_k, strides=1, padding="SAME")
            reaction = (u * u * u - u)
            u = u + dt * (gamma * du - kappa * reaction)
            u = u * dbc_mask
        return u

    res_list = []
    for start in range(0, int(N), RESIDUAL_BATCH):
        end = min(int(N), start + RESIDUAL_BATCH)
        u0_b = y_pred[start:end]
        y_b = y_obs[start:end]
        uT_b = forward_sim(u0_b)
        r_b = tf.reduce_mean(tf.abs(uT_b - y_b), axis=[1, 2, 3])
        res_list.append(r_b.numpy())

    return np.concatenate(res_list, axis=0).astype(np.float64)


def enforce_dbc_numpy(u):
    u = np.array(u, dtype=np.float32, copy=True)
    u[:, 0, :, :] = 0.0
    u[:, -1, :, :] = 0.0
    u[:, :, 0, :] = 0.0
    u[:, :, -1, :] = 0.0
    return u


# ============================================================
# EVALUATION CORE
# ============================================================

def evaluate_one_model(spec, X_test, Y_true, NE_SCALE, IC_SCALE):
    model_name = spec["name"]
    model_path = spec["model_path"]
    metric_path = spec.get("metric_path", None)
    training_metric_name = spec.get("training_metric_name", "saved_training_metric")

    model_outdir = os.path.join(OUTDIR, model_name)
    os.makedirs(model_outdir, exist_ok=True)

    model = load_generator_model(model_path)
    saved_training_metric = read_scalar_text(metric_path)

    Y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    Y_pred = np.asarray(Y_pred, dtype=np.float32)
    Y_pred = enforce_dbc_numpy(Y_pred)

    if Y_pred.shape != Y_true.shape:
        raise ValueError(
            f"[{model_name}] Predicted shape {Y_pred.shape} does not match target shape {Y_true.shape}."
        )

    # Entire test set metrics
    mae_scaled = np.mean(np.abs(Y_pred - Y_true), axis=(1, 2, 3)).astype(np.float64)
    mae_physical = (IC_SCALE * mae_scaled).astype(np.float64)
    residual = residual_per_sample(Y_pred, X_test)

    mae_scaled_stats = metric_summary(mae_scaled)
    mae_physical_stats = metric_summary(mae_physical)
    residual_stats = metric_summary(residual)

    # Thresholds from THIS model's TEST-set metric distribution
    eps, eps_info = choose_threshold(mae_scaled, FIXED_EPS, TARGET_P_MAE, "MAE_scaled")
    delta, delta_info = choose_threshold(residual, FIXED_DELTA, TARGET_P_RES, "Residual")

    # Qualitative plot samples
    sam = np.array(SAMPLES, dtype=int)
    x_src_sam = X_test[sam]
    x_gen_sam = Y_pred[sam]
    x_true_sam = Y_true[sam]

    mae_vals = np.mean(np.abs(x_gen_sam - x_true_sam), axis=(1, 2, 3)).astype(np.float64)
    res_vals = residual_per_sample(x_gen_sam, x_src_sam)

    if model_name == "best_residual":
        panel_metric_name = "RES"
        panel_metric_values = res_vals
    else:
        panel_metric_name = "MAE"
        panel_metric_values = mae_vals

    # FIX 1: define qualitative plot path
    qual_png = os.path.join(model_outdir, f"{model_name}_qualitative_plot.png")

    # FIX 2: put individual images INSIDE this model's directory
    individual_dir = os.path.join(model_outdir, "individual")

    save_qualitative_plot(
        x_src=x_src_sam,
        x_gen=x_gen_sam,
        x_true=x_true_sam,
        ne_scale=NE_SCALE,
        ic_scale=IC_SCALE,
        outpng=qual_png,
        samples=SAMPLES,
        training_metric_name=training_metric_name,
        training_metric_value=saved_training_metric,
        model_name=model_name,
        panel_metric_name=panel_metric_name,
        panel_metric_values=panel_metric_values,
        save_individual=True,
        individual_dir=individual_dir,
    )

    report = {
        "model_name": model_name,
        "config": {
            "TRAIN_NPZ": TRAIN_NPZ,
            "TEST_NPZ": TEST_NPZ,
            "model_path": model_path,
            "metric_path": metric_path,
            "training_metric_name": training_metric_name,
            "BATCH_SIZE": BATCH_SIZE,
            "RESIDUAL_BATCH": RESIDUAL_BATCH,
            "SIM_STEPS": SIM_STEPS,
            "dt": dt,
            "gamma": gamma,
            "kappa": kappa,
            "L": L,
            "M": M,
            "h": h,
            "TARGET_P_MAE": TARGET_P_MAE,
            "TARGET_P_RES": TARGET_P_RES,
            "TARGET_P_JOINT": TARGET_P_JOINT,
            "FIXED_EPS": FIXED_EPS,
            "FIXED_DELTA": FIXED_DELTA,
            "NE_SCALE": NE_SCALE,
            "IC_SCALE": IC_SCALE,
        },
        "saved_training_metric": {
            "name": training_metric_name,
            "value": saved_training_metric,
            "note": (
                "This is the scalar saved during training/validation. "
                "It is NOT the same object as the test-set quantile threshold epsilon/delta."
            ),
        },
        "full_test_metrics": {
            "mae_scaled_summary": mae_scaled_stats,
            "mae_physical_summary": mae_physical_stats,
            "residual_summary": residual_stats,
        },
        "thresholds": {
            "epsilon": eps,
            "delta": delta,
            "epsilon_selection": eps_info,
            "delta_selection": delta_info,
            "note": (
                "epsilon and delta are chosen from the FULL TEST-SET metric distributions "
                "so that P(metric <= threshold) is approximately the target probability."
            ),
        },
    }

    if eps is not None:
        report["P_MAE_leq_eps"] = prob_leq(mae_scaled, eps)

    if delta is not None:
        report["P_RES_leq_delta"] = prob_leq(residual, delta)

    if (eps is not None) and (delta is not None):
        report["joint"] = joint_probs(mae_scaled, residual, eps, delta)

    if ("joint" in report) and (TARGET_P_JOINT is not None):
        empirical_joint = report["joint"]["P_joint"]["p"]
        report["ml_solution_definition_check"] = {
            "definition": (
                "Declare success if P(MAE <= epsilon and Residual <= delta) >= X."
            ),
            "epsilon": float(eps),
            "delta": float(delta),
            "X": float(TARGET_P_JOINT),
            "empirical_joint_probability": float(empirical_joint),
            "satisfied": bool(empirical_joint >= TARGET_P_JOINT),
        }

    # Save raw arrays
    np.save(os.path.join(model_outdir, "mae_scaled_per_sample.npy"), mae_scaled)
    np.save(os.path.join(model_outdir, "mae_physical_per_sample.npy"), mae_physical)
    np.save(os.path.join(model_outdir, "residual_per_sample.npy"), residual)

    # Save metric plots
    save_hist_cdf(mae_scaled, "MAE_scaled", model_outdir)
    save_hist_cdf(mae_physical, "MAE_physical", model_outdir)
    save_hist_cdf(residual, "Residual", model_outdir)

    # Save JSON
    json_path = os.path.join(model_outdir, "report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save TXT
    txt_lines = []
    txt_lines.append("=== Unified Inverse Solution Evaluation Report ===")
    txt_lines.append(f"Model name : {model_name}")
    txt_lines.append(f"Train set  : {TRAIN_NPZ}")
    txt_lines.append(f"Test set   : {TEST_NPZ}")
    txt_lines.append(f"Model path : {model_path}")
    txt_lines.append("")

    txt_lines.append("[Saved training metric]")
    txt_lines.append(f"  name  = {training_metric_name}")
    txt_lines.append(f"  value = {saved_training_metric}")
    txt_lines.append("  NOTE  = This is the metric saved during training/validation.")
    txt_lines.append("          It is NOT the same as epsilon or delta.")
    txt_lines.append("")

    txt_lines.append("[Full test-set MAE: scaled IC units]")
    txt_lines.append(
        f"  mean +/- std        = {mae_scaled_stats['mean']:.8f} +/- {mae_scaled_stats['std']:.8f}"
    )
    txt_lines.append(
        f"  min / median / max  = {mae_scaled_stats['min']:.8f} / {mae_scaled_stats['median']:.8f} / {mae_scaled_stats['max']:.8f}"
    )
    txt_lines.append(
        f"  q05 / q95           = {mae_scaled_stats['q05']:.8f} / {mae_scaled_stats['q95']:.8f}"
    )
    if eps is not None:
        txt_lines.append(f"  epsilon             = {eps:.8f}")
        txt_lines.append(
            f"  P(MAE_scaled <= epsilon) = {report['P_MAE_leq_eps']['p']:.6f} "
            f"(Wilson95 {report['P_MAE_leq_eps']['wilson95']})"
        )
    txt_lines.append("")

    txt_lines.append("[Full test-set MAE: physical IC units]")
    txt_lines.append(
        f"  mean +/- std        = {mae_physical_stats['mean']:.8f} +/- {mae_physical_stats['std']:.8f}"
    )
    txt_lines.append(
        f"  min / median / max  = {mae_physical_stats['min']:.8f} / {mae_physical_stats['median']:.8f} / {mae_physical_stats['max']:.8f}"
    )
    txt_lines.append(
        f"  q05 / q95           = {mae_physical_stats['q05']:.8f} / {mae_physical_stats['q95']:.8f}"
    )
    txt_lines.append("")

    txt_lines.append("[Full test-set residual]")
    txt_lines.append(
        f"  mean +/- std        = {residual_stats['mean']:.8f} +/- {residual_stats['std']:.8f}"
    )
    txt_lines.append(
        f"  min / median / max  = {residual_stats['min']:.8f} / {residual_stats['median']:.8f} / {residual_stats['max']:.8f}"
    )
    txt_lines.append(
        f"  q05 / q95           = {residual_stats['q05']:.8f} / {residual_stats['q95']:.8f}"
    )
    if delta is not None:
        txt_lines.append(f"  delta               = {delta:.8f}")
        txt_lines.append(
            f"  P(Residual <= delta) = {report['P_RES_leq_delta']['p']:.6f} "
            f"(Wilson95 {report['P_RES_leq_delta']['wilson95']})"
        )

    if "joint" in report:
        txt_lines.append("")
        txt_lines.append("[Joint probabilities]")
        txt_lines.append(
            f"  P(MAE_scaled <= epsilon and Residual <= delta) = "
            f"{report['joint']['P_joint']['p']:.6f} "
            f"(Wilson95 {report['joint']['P_joint']['wilson95']})"
        )
        txt_lines.append(
            f"  P(Residual <= delta | MAE_scaled <= epsilon)   = "
            f"{report['joint']['P_RES_given_MAE']:.6f}"
        )
        txt_lines.append(
            f"  P(MAE_scaled <= epsilon | Residual <= delta)   = "
            f"{report['joint']['P_MAE_given_RES']:.6f}"
        )
        txt_lines.append(
            f"  corr(MAE_scaled, Residual)                     = "
            f"{report['joint']['corr_MAE_RES']:.6f}"
        )

    if "ml_solution_definition_check" in report:
        d = report["ml_solution_definition_check"]
        txt_lines.append("")
        txt_lines.append("[Machine-learned inverse solution criterion]")
        txt_lines.append(
            f"  Criterion: P(MAE_scaled <= {d['epsilon']:.6g} and Residual <= {d['delta']:.6g}) >= {d['X']:.6g}"
        )
        txt_lines.append(f"  Empirical value = {d['empirical_joint_probability']:.6f}")
        txt_lines.append(f"  Satisfied = {d['satisfied']}")

    txt_lines.append("")
    txt_lines.append("[LaTeX-ready sentence]")
    if "joint" in report:
        txt_lines.append(
            r"\noindent On the held-out test set, "
            + rf"$\Pr(\mathrm{{MAE}}_{{\mathrm{{scaled}}}}\le {eps:.6g})\approx {report['P_MAE_leq_eps']['p']:.4f}$, "
            + rf"$\Pr(\mathrm{{Res}}\le {delta:.6g})\approx {report['P_RES_leq_delta']['p']:.4f}$, "
            + rf"and $\Pr(\mathrm{{MAE}}_{{\mathrm{{scaled}}}}\le {eps:.6g}\ \wedge\ \mathrm{{Res}}\le {delta:.6g})\approx {report['joint']['P_joint']['p']:.4f}$."
        )

    txt_path = os.path.join(model_outdir, "report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))

    print(f"\n[{model_name}] Saved:")
    print(f"  {qual_png}")
    print(f"  {individual_dir}")
    print(f"  {os.path.join(model_outdir, 'mae_scaled_hist.png')}")
    print(f"  {os.path.join(model_outdir, 'mae_scaled_cdf.png')}")
    print(f"  {os.path.join(model_outdir, 'mae_physical_hist.png')}")
    print(f"  {os.path.join(model_outdir, 'mae_physical_cdf.png')}")
    print(f"  {os.path.join(model_outdir, 'residual_hist.png')}")
    print(f"  {os.path.join(model_outdir, 'residual_cdf.png')}")
    print(f"  {os.path.join(model_outdir, 'mae_scaled_per_sample.npy')}")
    print(f"  {os.path.join(model_outdir, 'mae_physical_per_sample.npy')}")
    print(f"  {os.path.join(model_outdir, 'residual_per_sample.npy')}")
    print(f"  {json_path}")
    print(f"  {txt_path}")

    return report

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load training data only for scaling
    train_src, train_tar = load_npz_xy(TRAIN_NPZ, NPZ_KEY_INPUT, NPZ_KEY_TARGET)
    NE_SCALE = float(np.max(np.abs(train_src)))
    IC_SCALE = float(np.max(np.abs(train_tar)))

    # Load and scale test data
    X_test_raw, Y_true_raw = load_npz_xy(TEST_NPZ, NPZ_KEY_INPUT, NPZ_KEY_TARGET)

    X_test = np.asarray(X_test_raw, dtype=np.float32) / NE_SCALE
    Y_true = np.asarray(Y_true_raw, dtype=np.float32) / IC_SCALE

    if X_test.ndim == 3:
        X_test = X_test[..., np.newaxis]
    if Y_true.ndim == 3:
        Y_true = Y_true[..., np.newaxis]

    all_reports = {}
    for spec in MODEL_SPECS:
        all_reports[spec["name"]] = evaluate_one_model(spec, X_test, Y_true, NE_SCALE, IC_SCALE)

    # Comparison summary
    comparison = {
        "best_mae_model": all_reports.get("best_mae", {}),
        "best_residual_model": all_reports.get("best_residual", {}),
    }

    comp_json = os.path.join(OUTDIR, "comparison_summary.json")
    with open(comp_json, "w") as f:
        json.dump(comparison, f, indent=2)

    comp_txt = os.path.join(OUTDIR, "comparison_summary.txt")
    lines = []
    lines.append("=== Comparison Summary: best-MAE vs best-Residual generators ===")
    lines.append("")

    for key in ["best_mae", "best_residual"]:
        if key not in all_reports:
            continue
        r = all_reports[key]
        m = r["full_test_metrics"]["mae_scaled_summary"]
        mp = r["full_test_metrics"]["mae_physical_summary"]
        rs = r["full_test_metrics"]["residual_summary"]

        lines.append(f"[{key}]")
        lines.append(f"  saved training metric ({r['saved_training_metric']['name']}) = {r['saved_training_metric']['value']}")
        lines.append(f"  full test MAE_scaled mean +/- std   = {m['mean']:.8f} +/- {m['std']:.8f}")
        lines.append(f"  full test MAE_physical mean +/- std = {mp['mean']:.8f} +/- {mp['std']:.8f}")
        lines.append(f"  full test Residual mean +/- std     = {rs['mean']:.8f} +/- {rs['std']:.8f}")
        if "P_MAE_leq_eps" in r:
            lines.append(f"  epsilon = {r['thresholds']['epsilon']:.8f}   with P(MAE_scaled <= epsilon) = {r['P_MAE_leq_eps']['p']:.6f}")
        if "P_RES_leq_delta" in r:
            lines.append(f"  delta   = {r['thresholds']['delta']:.8f}   with P(Residual <= delta) = {r['P_RES_leq_delta']['p']:.6f}")
        if "joint" in r:
            lines.append(f"  joint probability = {r['joint']['P_joint']['p']:.6f}")
        lines.append("")

    with open(comp_txt, "w") as f:
        f.write("\n".join(lines))

    print("\n[comparison] Saved:")
    print(f"  {comp_json}")
    print(f"  {comp_txt}")


if __name__ == "__main__":
    main()
