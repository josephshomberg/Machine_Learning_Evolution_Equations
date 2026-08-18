#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tCHE_WGANGP_stats_energy_residual.py

Conditional WGAN-GP for the inverse ternary Cahn--Hilliard problem.

Inverse problem:
    c_T  --->  c_0

Data:
    src = final ternary state c_T
    tar = initial ternary state c_0

Accepted data shapes:
    (N, 128, 128, 3)
    (N, 3, 128, 128)

This version follows the backwards Chafee--Infante WGAN structure more closely:
    - WGAN-GP adversarial loss
    - MAE loss
    - Lyapunov / ternary Cahn--Hilliard energy loss
    - field mean loss
    - field variance loss
    - component mass loss
    - purity loss for pure ternary phases
    - optional gradient/interface loss
    - optional 3000-step physics residual in the generator loss
    - validation residual is recorded even if LAMBDA_RESIDUAL = 0
    - best-MAE and best-residual models are both saved

Recommended first run:
    LAMBDA_RESIDUAL = 0.0
    VALIDATE_RESIDUAL = True

Then later:
    LAMBDA_RESIDUAL = 0.005 or 0.01
"""

import os
import csv
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Input, Concatenate, Conv2D, LeakyReLU, Activation,
    Dropout, LayerNormalization, UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# Dataset paths
# ============================================================

TRAIN_NPZ = "PATH/TO/TRAIN/DATASET.npz"

TEST_NPZ  = "PATH/TO/TESTING/DATASET.npz"

SRC_KEY = "src"     # final state c_T
TAR_KEY = "tar"     # initial state c_0

# ============================================================
# Physical / numerical parameters
# ============================================================

N = 128
L = 1.0
dx = L / N

dt = 5.0e-8
eps = 0.01
A = 1.0
chi = (1.5, 1.5, 1.5)
mobility = (1.0, 1.0, 1.0)

RESIDUAL_STEPS = 200

# ============================================================
# Loss weights
# Eyre dataset WGAN baseline / MAE-only style
# No residual, no energy
# ============================================================

LAMBDA_ADV      = 0.15
LAMBDA_GP       = 10.0

LAMBDA_MAE      = 100.0

ENERGY_WT       = 0.005

LAMBDA_VAR      = 5.0
LAMBDA_MASS     = 5.0

LAMBDA_PURITY   = 0.25
LAMBDA_GRAD     = 0.0
LAMBDA_SIMPLEX  = 0.25

LAMBDA_RESIDUAL = 0.005
LAMBDA_ENTROPY  = 0.0

LAMBDA_HF       = 50.0

# ============================================================
# Training / validation parameters
# ============================================================

MAX_STEPS      = 50000  #  =(dataset size)*(max epochs)/(batch size)
MAX_EPOCHS     = 10
BATCH_SIZE     = 4
VAL_BATCH_SIZE = 4
NUM_SAMPLES    = 8
SAVEEVERY      = 100

G_LEARNING = 1.0e-4
C_LEARNING = 5.0e-5

CRITIC_STEPS = 1
CRITIC_SIZE  = 4

# Records and saves best residual models even if LAMBDA_RESIDUAL = 0.
# Expensive: runs RESIDUAL_STEPS forward Euler steps on validation samples.
VALIDATE_RESIDUAL = True
RESIDUAL_NUM_SAMPLES = 1  # 16

# ============================================================
# Output directories
# ============================================================

RUN_PREFIX = "tCHE_3color_WGANGP_stats_energy"

LATEST_DIR        = f"{RUN_PREFIX}_checkpoints_latest"
BEST_MAE_DIR      = f"{RUN_PREFIX}_checkpoints_best_mae"
BEST_RESIDUAL_DIR = f"{RUN_PREFIX}_checkpoints_best_residual"
MODEL_DIR         = f"{RUN_PREFIX}_models"
PLOT_DIR          = f"{RUN_PREFIX}_plots"

for d in [LATEST_DIR, BEST_MAE_DIR, BEST_RESIDUAL_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_CSV = f"{RUN_PREFIX}_training_data.csv"
TEST_CSV  = f"{RUN_PREFIX}_testing_data.csv"

# ============================================================
# GPU setup
# ============================================================

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# ============================================================
# CSV initialization
# ============================================================

if not os.path.exists(TRAIN_CSV):
    with open(TRAIN_CSV, mode="w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iteration", "critic_loss", "generator_loss", "adversarial_loss",
            "mae_loss", "energy_loss", "mean_loss", "variance_loss",
            "mass_loss", "purity_loss", "gradient_loss", "simplex_loss",
            "residual_loss", "entropy_loss", "hf_loss",
        ])

if not os.path.exists(TEST_CSV):
    with open(TEST_CSV, mode="w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iteration", "samples", "mae", "energy_error", "mean_error",
            "variance_error", "mass_error", "purity_error", "gradient_error",
            "simplex_error", "residual_error", "gen_min", "gen_max",
            "gen_sum_min", "gen_sum_max", "gen_mean_c1", "gen_mean_c2",
            "gen_mean_c3", "gen_var_c1", "gen_var_c2", "gen_var_c3",
        ])

# ============================================================
# Data utilities
# ============================================================

def to_channel_last(x):
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
        raise ValueError(f"Expected 3D or 4D array, got shape {x.shape}")
    return x.astype("float32")


def normalize_simplex_np(c, eps0=1e-8):
    c = np.clip(c, 0.0, 1.0)
    s = np.sum(c, axis=-1, keepdims=True)
    return c / (s + eps0)


def load_dataset(path, src_key=SRC_KEY, tar_key=TAR_KEY):
    z = np.load(path, mmap_mode="r")
    if src_key not in z or tar_key not in z:
        raise KeyError(
            f"Expected keys '{src_key}' and '{tar_key}' in {path}. "
            f"Found keys: {list(z.keys())}"
        )
    Xsrc = normalize_simplex_np(to_channel_last(z[src_key]))
    Xtar = normalize_simplex_np(to_channel_last(z[tar_key]))
    print(f"Loaded {path}")
#    print("  src shape:", Xsrc.shape, "range:", Xsrc.min(), Xsrc.max())
#    print("  tar shape:", Xtar.shape, "range:", Xtar.min(), Xtar.max())
#    print("  src pointwise sum range:", Xsrc.sum(axis=-1).min(), Xsrc.sum(axis=-1).max())
#    print("  tar pointwise sum range:", Xtar.sum(axis=-1).min(), Xtar.sum(axis=-1).max())
#    print("  tar component means:", np.mean(Xtar, axis=(0, 1, 2)))
#    print("  tar component vars: ", np.var(Xtar, axis=(0, 1, 2)))
    return Xsrc, Xtar


print("Loading datasets...")
trainA, trainB = load_dataset(TRAIN_NPZ)
testA, testB   = load_dataset(TEST_NPZ)

dataset = [trainA, trainB]
test_data = [testA, testB]
image_shape = trainA.shape[1:]
print("image_shape:", image_shape)
if image_shape[-1] != 3:
    raise ValueError("This script requires 3-channel ternary data.")

# ============================================================
# TensorFlow ternary Cahn--Hilliard operators and losses
# ============================================================

def simplex_error_tf(c):
    positivity = tf.reduce_mean(tf.nn.relu(-c))
    upper = tf.reduce_mean(tf.nn.relu(c - 1.0))
    sum_err = tf.reduce_mean(tf.abs(tf.reduce_sum(c, axis=-1) - 1.0))
    return positivity + upper + sum_err


def normalize_simplex_tf(c, eps0=1e-8):
    c = tf.clip_by_value(c, 0.0, 1.0)
    s = tf.reduce_sum(c, axis=-1, keepdims=True)
    return c / (s + eps0)


def purity_loss_tf(fake):
    purity = tf.reduce_sum(fake**2, axis=-1)
    return tf.reduce_mean(tf.abs(1.0 - purity))


def entropy_loss_tf(fake, eps0=1e-8):
    return tf.reduce_mean(
        -tf.reduce_sum(fake * tf.math.log(fake + eps0), axis=-1)
    )
    

def local_variance_loss_tf(real, fake):
    real_blur = tf.nn.avg_pool2d(real, ksize=5, strides=1, padding="SAME")
    fake_blur = tf.nn.avg_pool2d(fake, ksize=5, strides=1, padding="SAME")

    real_hf = real - real_blur
    fake_hf = fake - fake_blur

    return tf.reduce_mean(tf.abs(real_hf - fake_hf))


def field_variance_loss_tf(real, fake):
    var_real = tf.math.reduce_variance(real, axis=[1, 2])
    var_fake = tf.math.reduce_variance(fake, axis=[1, 2])
    return tf.reduce_mean(tf.abs(var_real - var_fake))


def mass_loss_tf(real, fake):
    real_mass = tf.reduce_mean(real, axis=[1, 2])
    fake_mass = tf.reduce_mean(fake, axis=[1, 2])
    return tf.reduce_mean(tf.abs(real_mass - fake_mass))


def gradient_l1_loss_tf(real, fake):
    real_dx = real[:, 1:, :, :] - real[:, :-1, :, :]
    fake_dx = fake[:, 1:, :, :] - fake[:, :-1, :, :]
    real_dy = real[:, :, 1:, :] - real[:, :, :-1, :]
    fake_dy = fake[:, :, 1:, :] - fake[:, :, :-1, :]
    return tf.reduce_mean(tf.abs(fake_dx - real_dx)) + tf.reduce_mean(tf.abs(fake_dy - real_dy))


def laplacian_periodic_tf(u, dx):
    return (
        tf.roll(u, 1, axis=1) + tf.roll(u, -1, axis=1)
        + tf.roll(u, 1, axis=2) + tf.roll(u, -1, axis=2)
        - 4.0 * u
    ) / (dx * dx)


def bulk_derivative_tf(c, A=1.0, chi=(1.5, 1.5, 1.5)):
    c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2]
    chi12, chi13, chi23 = chi
    dW1 = A * (2*c1 - 6*c1**2 + 4*c1**3) + chi12*c2 + chi13*c3
    dW2 = A * (2*c2 - 6*c2**2 + 4*c2**3) + chi12*c1 + chi23*c3
    dW3 = A * (2*c3 - 6*c3**2 + 4*c3**3) + chi13*c1 + chi23*c2
    return tf.stack([dW1, dW2, dW3], axis=-1)


def chemical_potential_tf(c, dx, eps=0.01, A=1.0, chi=(1.5, 1.5, 1.5)):
    mu = bulk_derivative_tf(c, A=A, chi=chi)
    mus = []
    for i in range(3):
        ci = c[..., i]
        mui = mu[..., i] - eps**2 * laplacian_periodic_tf(ci, dx)
        mus.append(mui)
    mu = tf.stack(mus, axis=-1)
    return mu - tf.reduce_mean(mu, axis=-1, keepdims=True)


def forward_euler_rhs_tf(c, dx, eps, mobility, A, chi):
    mu = chemical_potential_tf(c, dx=dx, eps=eps, A=A, chi=chi)
    rhs = []
    for i in range(3):
        mi = tf.cast(mobility[i], tf.float32)
        rhs.append(mi * laplacian_periodic_tf(mu[..., i], dx))
    rhs = tf.stack(rhs, axis=-1)
    return rhs - tf.reduce_mean(rhs, axis=-1, keepdims=True)


def ternary_lyapunov_energy_tf(c, dx, eps=0.01, A=1.0, chi=(1.5, 1.5, 1.5)):
    c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2]
    chi12, chi13, chi23 = chi
    bulk = (
        A * tf.reduce_sum(c**2 * (1.0 - c)**2, axis=-1)
        + chi12 * c1 * c2 + chi13 * c1 * c3 + chi23 * c2 * c3
    )
    grad_part = 0.0
    for i in range(3):
        ci = c[..., i]
        cx = (tf.roll(ci, -1, axis=1) - ci) / dx
        cy = (tf.roll(ci, -1, axis=2) - ci) / dx
        grad_part = grad_part + 0.5 * eps**2 * (cx**2 + cy**2)
    return dx**2 * tf.reduce_sum(bulk + grad_part, axis=[1, 2])


def lyapunov_energy_loss_tf(real, fake):
    E_fake = ternary_lyapunov_energy_tf(fake, dx=dx, eps=eps, A=A, chi=chi)
    E_real = ternary_lyapunov_energy_tf(real, dx=dx, eps=eps, A=A, chi=chi)
    return tf.reduce_mean(tf.abs(E_fake - E_real)) / (tf.reduce_mean(tf.abs(E_real)) + 1.0e-8)


@tf.function
def forward_sim_tche_tf(c0, nsteps, dt=dt, dx=dx, eps=eps, mobility=mobility, A=A, chi=chi):
    c = normalize_simplex_tf(c0)
    for _ in tf.range(nsteps):
        rhs = forward_euler_rhs_tf(c, dx=dx, eps=eps, mobility=mobility, A=A, chi=chi)
        c = c + tf.cast(dt, tf.float32) * rhs
        c = normalize_simplex_tf(c)
    return c


@tf.function
def residual_loss_tf(c0_pred, c0_true, nsteps=RESIDUAL_STEPS):
    """
    Short-horizon dynamical consistency loss.

    Instead of comparing a short Forward Euler rollout of c0_pred
    directly with the full terminal state c_T, evolve both the predicted
    and true initial conditions for the same number of short-horizon
    steps and compare the resulting states.

        R_tau(c0_pred, c0_true)
            = || Phi_tau(c0_pred) - Phi_tau(c0_true) ||_1.

    This is a differentiable surrogate physics loss.
    """

    c_pred_tau = forward_sim_tche_tf(
        c0_pred,
        nsteps=nsteps,
    )

    c_true_tau = forward_sim_tche_tf(
        c0_true,
        nsteps=nsteps,
    )

    # The true trajectory is a fixed target; gradients are needed only
    # through the predicted trajectory.
    c_true_tau = tf.stop_gradient(c_true_tau)

    return tf.reduce_mean(tf.abs(c_pred_tau - c_true_tau))


# ============================================================
# Models
# ============================================================

def encoder_block(layer_in, n_filters, norm=True):
    g = Conv2D(n_filters, 4, strides=2, padding="same", kernel_initializer="he_uniform", use_bias=False)(layer_in)
    if norm:
        g = LayerNormalization()(g)
    return LeakyReLU(negative_slope=0.2)(g)


def decoder_block(layer_in, skip_in, n_filters, dropout=False):
    g = UpSampling2D(size=(2, 2), interpolation="bilinear")(layer_in)
    g = Conv2D(n_filters, 3, strides=1, padding="same", kernel_initializer="he_uniform", use_bias=False)(g)
    g = LayerNormalization()(g)
    if dropout:
        g = Dropout(0.5)(g)
    g = Activation("relu")(g)
    return Concatenate()([g, skip_in])


def generator(image_shape):
    in_image = Input(shape=image_shape)
    e1 = Conv2D(64, 4, strides=2, padding="same", kernel_initializer="he_uniform", use_bias=False)(in_image)
    e1 = LeakyReLU(negative_slope=0.2)(e1)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)
    b = Conv2D(512, 4, strides=2, padding="same", kernel_initializer="he_uniform", use_bias=False)(e5)
    b = Activation("relu")(b)
    d1 = decoder_block(b, e5, 512, dropout=True)
    d2 = decoder_block(d1, e4, 512, dropout=True)
    d3 = decoder_block(d2, e3, 256, dropout=False)
    d4 = decoder_block(d3, e2, 128, dropout=False)
    d5 = decoder_block(d4, e1, 64, dropout=False)
    g = UpSampling2D(size=(2, 2), interpolation="bilinear")(d5)
    g = Conv2D(3, 3, strides=1, padding="same", kernel_initializer="he_uniform")(g)
    out_image = Activation("softmax")(g)
    return Model(in_image, out_image, name="ternary_unet_generator")


def critic(image_shape, base_filters=64):
    in_src = Input(shape=image_shape)
    in_tgt = Input(shape=image_shape)
    x = Concatenate()([in_src, in_tgt])
    def disc_block(x, filters, stride):
        x = Conv2D(filters, CRITIC_SIZE, strides=stride, padding="same", kernel_initializer="he_uniform", use_bias=True)(x)
        return LeakyReLU(0.2)(x)
    x = disc_block(x, base_filters * 1, 2)
    x = disc_block(x, base_filters * 2, 2)
    x = disc_block(x, base_filters * 4, 2)
    x = disc_block(x, base_filters * 8, 2)
    patch_out = Conv2D(1, CRITIC_SIZE, strides=1, padding="same")(x)
    return Model([in_src, in_tgt], patch_out, name="conditional_wasserstein_critic")


def wgan(g_model, c_model, image_shape):
    for layer in c_model.layers:
        layer.trainable = False
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    c_out = c_model([in_src, gen_out])
    model = Model(inputs=in_src, outputs=[c_out, gen_out], name="ternary_wgan")
    for layer in c_model.layers:
        layer.trainable = True
    return model

# ============================================================
# WGAN-GP utility
# ============================================================

def generate_real_samples(dataset, n_samples):
    X1, X2 = dataset
    ix = np.random.randint(0, X1.shape[0], n_samples)
    return X1[ix], X2[ix]


def gradient_penalty(c_model, real_A, real_B, fake_B):
    real_A = tf.convert_to_tensor(real_A, dtype=tf.float32)
    real_B = tf.convert_to_tensor(real_B, dtype=tf.float32)
    fake_B = tf.convert_to_tensor(fake_B, dtype=tf.float32)
    alpha = tf.random.uniform([tf.shape(real_B)[0], 1, 1, 1], 0.0, 1.0, dtype=tf.float32)
    interpolated = alpha * real_B + (1.0 - alpha) * fake_B
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = c_model([real_A, interpolated], training=True)
        pred = tf.reduce_sum(pred, axis=[1, 2, 3])
    grads = tape.gradient(pred, interpolated)
    if grads is None:
        raise RuntimeError("Gradient penalty gradients are None.")
    grads = tf.reshape(grads, [tf.shape(grads)[0], -1])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1.0e-12)
    return tf.reduce_mean((slopes - 1.0) ** 2)

# ============================================================
# Visualization and validation
# ============================================================

def save_rgb_triplet(src, pred, tar, filename):
    src = np.clip(src, 0.0, 1.0)
    pred = np.clip(pred, 0.0, 1.0)
    tar = np.clip(tar, 0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["given $c_T$", "generated $\\hat c_0$", "true $c_0$"]
    for ax, title, field in zip(axes, titles, [src, pred, tar]):
        ax.imshow(field, origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()


def summarize_performance(step, g_model, test_data, num_samples=NUM_SAMPLES, val_batch_size=VAL_BATCH_SIZE):
    model_num = step + 1
    print("Running validation at step", model_num)
    test_num = test_data[0].shape[0]
    sam = np.random.choice(test_num, size=min(num_samples, test_num), replace=False)
    maes, energy_errors, mean_errors, variance_errors = [], [], [], []
    mass_errors, purity_errors, grad_errors, simplex_errors, residual_errors = [], [], [], [], []
    gen_mins, gen_maxs, gen_sum_mins, gen_sum_maxs = [], [], [], []
    gen_means, gen_vars = [], []
    made_plot = False
    residual_count = 0

    for start in range(0, num_samples, val_batch_size):
        end = min(start + val_batch_size, num_samples)
        batch_ids = sam[start:end]
        src_batch = test_data[0][batch_ids]
        tar_batch = test_data[1][batch_ids]
        pred_batch = normalize_simplex_np(g_model.predict(src_batch, verbose=0))
        src_tf = tf.convert_to_tensor(src_batch, dtype=tf.float32)
        tar_tf = tf.convert_to_tensor(tar_batch, dtype=tf.float32)
        pred_tf = tf.convert_to_tensor(pred_batch, dtype=tf.float32)

        mae = tf.reduce_mean(tf.abs(pred_tf - tar_tf)).numpy()
        energy_err = lyapunov_energy_loss_tf(tar_tf, pred_tf).numpy()
        mean_err, var_err = field_mean_variance_loss_tf(tar_tf, pred_tf)
        mass_err = mass_loss_tf(tar_tf, pred_tf).numpy()
        purity_err = purity_loss_tf(pred_tf).numpy()
        grad_err = gradient_l1_loss_tf(tar_tf, pred_tf).numpy()
        simp_err = simplex_error_tf(pred_tf).numpy()

        maes.append(float(mae))
        energy_errors.append(float(energy_err))
        mean_errors.append(float(mean_err.numpy()))
        variance_errors.append(float(var_err.numpy()))
        mass_errors.append(float(mass_err))
        purity_errors.append(float(purity_err))
        grad_errors.append(float(grad_err))
        simplex_errors.append(float(simp_err))

        if VALIDATE_RESIDUAL and residual_count < RESIDUAL_NUM_SAMPLES:
            remaining = RESIDUAL_NUM_SAMPLES - residual_count
            take = min(remaining, len(batch_ids))
            res_err = residual_loss_tf(pred_tf[:take], tar_tf[:take], nsteps=RESIDUAL_STEPS).numpy()
            residual_errors.append(float(res_err))
            residual_count += take

        gen_mins.append(float(np.min(pred_batch)))
        gen_maxs.append(float(np.max(pred_batch)))
        sums = np.sum(pred_batch, axis=-1)
        gen_sum_mins.append(float(np.min(sums)))
        gen_sum_maxs.append(float(np.max(sums)))
        gen_means.append(np.mean(pred_batch, axis=(0, 1, 2)))
        gen_vars.append(np.var(pred_batch, axis=(0, 1, 2)))

        if not made_plot:
            save_rgb_triplet(src_batch[0], pred_batch[0], tar_batch[0], f"{PLOT_DIR}/validation_step_{model_num:08d}.png")
            made_plot = True

    current_mae = float(np.mean(maes))
    current_res = float(np.mean(residual_errors)) if residual_errors else float("inf")
    mean_gen_mean = np.mean(np.stack(gen_means, axis=0), axis=0)
    mean_gen_var = np.mean(np.stack(gen_vars, axis=0), axis=0)

    with open(TEST_CSV, mode="a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            model_num, sam.tolist(), current_mae, float(np.mean(energy_errors)),
            float(np.mean(mean_errors)), float(np.mean(variance_errors)),
            float(np.mean(mass_errors)), float(np.mean(purity_errors)),
            float(np.mean(grad_errors)), float(np.mean(simplex_errors)), current_res,
            float(np.mean(gen_mins)), float(np.mean(gen_maxs)),
            float(np.mean(gen_sum_mins)), float(np.mean(gen_sum_maxs)),
            float(mean_gen_mean[0]), float(mean_gen_mean[1]), float(mean_gen_mean[2]),
            float(mean_gen_var[0]), float(mean_gen_var[1]), float(mean_gen_var[2]),
        ])

    print(
        f"Validation: MAE={current_mae:.6f}, energy={np.mean(energy_errors):.6f}, "
        f"mean={np.mean(mean_errors):.6f}, var={np.mean(variance_errors):.6f}, "
        f"mass={np.mean(mass_errors):.6f}, purity={np.mean(purity_errors):.6f}, "
        f"grad={np.mean(grad_errors):.6f}, simplex={np.mean(simplex_errors):.6e}, "
        f"residual={current_res:.6f}"
    )
    return current_mae, current_res

# ============================================================
# Training step
# ============================================================

@tf.function
def train_step(real_A, real_B, generator_model, critic_model, g_optimizer, c_optimizer):
    real_A = tf.convert_to_tensor(real_A, dtype=tf.float32)
    real_B = tf.convert_to_tensor(real_B, dtype=tf.float32)
    c_loss_accum = 0.0

    for _ in tf.range(CRITIC_STEPS):
        with tf.GradientTape() as tape:
            fake_B = generator_model(real_A, training=True)
            c_real = critic_model([real_A, real_B], training=True)
            c_fake = critic_model([real_A, fake_B], training=True)
            gp = gradient_penalty(critic_model, real_A, real_B, fake_B)
            c_loss = tf.reduce_mean(c_fake) - tf.reduce_mean(c_real) + LAMBDA_GP * gp
        grads = tape.gradient(c_loss, critic_model.trainable_variables)
        c_optimizer.apply_gradients(zip(grads, critic_model.trainable_variables))
        c_loss_accum += c_loss
    c_loss_accum = c_loss_accum / tf.cast(CRITIC_STEPS, tf.float32)

    with tf.GradientTape() as tape:
        fake_B = generator_model(real_A, training=True)
        d_fake = critic_model([real_A, fake_B], training=True)
        adv_loss = -tf.reduce_mean(d_fake)
        mae_loss = tf.reduce_mean(tf.abs(fake_B - real_B))
        energy_loss = lyapunov_energy_loss_tf(real_B, fake_B)
        var_loss = field_variance_loss_tf(real_B, fake_B)
        mass_loss = mass_loss_tf(real_B, fake_B)
        purity_loss = purity_loss_tf(fake_B)
        grad_loss = gradient_l1_loss_tf(real_B, fake_B)
        simplex_loss = simplex_error_tf(fake_B)
        entropy_loss = entropy_loss_tf(fake_B)
        hf_loss = local_variance_loss_tf(real_B, fake_B)
        if LAMBDA_RESIDUAL > 0.0:
            residual_loss = residual_loss_tf(fake_B, real_B, nsteps=RESIDUAL_STEPS)
        else:
            residual_loss = tf.constant(0.0, dtype=tf.float32)
        g_loss = (
            LAMBDA_ADV * adv_loss
            + LAMBDA_MAE * mae_loss
            + ENERGY_WT * energy_loss
            + LAMBDA_VAR * var_loss
            + LAMBDA_MASS * mass_loss
            + LAMBDA_PURITY * purity_loss
            + LAMBDA_GRAD * grad_loss
            + LAMBDA_SIMPLEX * simplex_loss
            + LAMBDA_RESIDUAL * residual_loss
            + LAMBDA_ENTROPY * entropy_loss
            + LAMBDA_HF * hf_loss
        )
    grads = tape.gradient(g_loss, generator_model.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator_model.trainable_variables))
    return (
    c_loss_accum, g_loss, adv_loss, mae_loss, energy_loss, mean_loss,
    var_loss, mass_loss, purity_loss, grad_loss, simplex_loss,
            residual_loss, entropy_loss, hf_loss,
    )

# ============================================================
# Main training loop
# ============================================================

def train(c_model, g_model, wgan_model, dataset, max_epochs, n_batch=BATCH_SIZE):
    g_optimizer = Adam(learning_rate=G_LEARNING, beta_1=0.0, beta_2=0.9)
    c_optimizer = Adam(learning_rate=C_LEARNING, beta_1=0.0, beta_2=0.9)
    step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
    best_mae_var = tf.Variable(np.inf, dtype=tf.float32, trainable=False, name="best_mae")
    best_res_var = tf.Variable(np.inf, dtype=tf.float32, trainable=False, name="best_residual")
    checkpoint = tf.train.Checkpoint(
        step=step, best_mae=best_mae_var, best_residual=best_res_var,
        g_optimizer=g_optimizer, c_optimizer=c_optimizer, generator=g_model, critic=c_model,
    )
    latest_manager = tf.train.CheckpointManager(checkpoint, directory=LATEST_DIR, max_to_keep=2)
    best_mae_manager = tf.train.CheckpointManager(checkpoint, directory=BEST_MAE_DIR, max_to_keep=1)
    best_res_manager = tf.train.CheckpointManager(checkpoint, directory=BEST_RESIDUAL_DIR, max_to_keep=1)

    if latest_manager.latest_checkpoint:
        checkpoint.restore(latest_manager.latest_checkpoint).expect_partial()
        print(
            "Restored LATEST checkpoint:", latest_manager.latest_checkpoint,
            "step =", int(step.numpy()), "best_mae =", float(best_mae_var.numpy()),
            "best_residual =", float(best_res_var.numpy()),
        )
    else:
        print("No LATEST checkpoint found. Training from scratch.")

    last_mae = float(best_mae_var.numpy())
    last_res = float(best_res_var.numpy())
    trainA, _ = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * max_epochs
    start_i = int(step.numpy())
    print("Starting iteration index =", start_i, "of", n_steps)

    for i in range(start_i, min(n_steps, MAX_STEPS)):
        X_realA, X_realB = generate_real_samples(dataset, n_batch)
        (
        c_loss, g_loss, adv_loss, mae_loss, energy_loss, mean_loss,
        var_loss, mass_loss, purity_loss, grad_loss, simplex_loss,
        residual_loss, entropy_loss, hf_loss
        ) = train_step(X_realA, X_realB, g_model, c_model, g_optimizer, c_optimizer)
        step.assign(i + 1)

        if (i + 1) == 1 or (i + 1) % SAVEEVERY == 0:
            with open(TRAIN_CSV, mode="a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    i + 1, float(c_loss.numpy()), float(g_loss.numpy()),
                    float(adv_loss.numpy()), float(mae_loss.numpy()),
                    float(energy_loss.numpy()), float(mean_loss.numpy()),
                    float(var_loss.numpy()), float(mass_loss.numpy()),
                    float(purity_loss.numpy()), float(grad_loss.numpy()),
                    float(simplex_loss.numpy()), float(residual_loss.numpy()),
                    float(entropy_loss.numpy()),
                    float(hf_loss.numpy()),
                ])
            try:
                latest_manager.save(checkpoint_number=int(step.numpy()))
                print(f"Saved LATEST at step={int(step.numpy())}")
            except tf.errors.ResourceExhaustedError as e:
                print("Could not save latest checkpoint:", e)

            current_mae, current_res = summarize_performance(i, g_model, test_data)

            if current_mae < last_mae:
                last_mae = current_mae
                best_mae_var.assign(last_mae)
                g_model.save(f"{MODEL_DIR}/best_generator_mae.keras")
                c_model.save(f"{MODEL_DIR}/best_critic_mae.keras")
                with open(f"{MODEL_DIR}/best_mae.txt", "w") as f:
                    f.write(str(last_mae))
                try:
                    best_mae_manager.save(checkpoint_number=int(step.numpy()))
                    print(f"New BEST(MAE): mae={last_mae:.6f} at step={int(step.numpy())}")
                except tf.errors.ResourceExhaustedError as e:
                    print("Could not archive BEST(MAE):", e)

            # Evaluated even when LAMBDA_RESIDUAL = 0, if VALIDATE_RESIDUAL=True.
            if np.isfinite(current_res) and current_res < last_res:
                last_res = current_res
                best_res_var.assign(last_res)
                g_model.save(f"{MODEL_DIR}/best_generator_residual.keras")
                c_model.save(f"{MODEL_DIR}/best_critic_residual.keras")
                with open(f"{MODEL_DIR}/best_residual.txt", "w") as f:
                    f.write(str(last_res))
                try:
                    best_res_manager.save(checkpoint_number=int(step.numpy()))
                    print(f"New BEST(RESIDUAL): res={last_res:.6f} at step={int(step.numpy())}")
                except tf.errors.ResourceExhaustedError as e:
                    print("Could not archive BEST(RESIDUAL):", e)

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    c_model = critic(image_shape)
    g_model = generator(image_shape)
    wgan_model = wgan(g_model, c_model, image_shape)
    print("Critic output shape:", c_model.output_shape)
    print("Generator output shape:", g_model.output_shape)
    c_model.summary()
    g_model.summary()
    wgan_model.summary()
    try:
        plot_model(c_model, to_file=f"{RUN_PREFIX}_critic_model_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(g_model, to_file=f"{RUN_PREFIX}_generator_model_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(wgan_model, to_file=f"{RUN_PREFIX}_wgan_model_plot.png", show_shapes=True, show_layer_names=True)
    except Exception as e:
        print("Model plotting skipped:", e)
    train(c_model, g_model, wgan_model, dataset, MAX_EPOCHS)
