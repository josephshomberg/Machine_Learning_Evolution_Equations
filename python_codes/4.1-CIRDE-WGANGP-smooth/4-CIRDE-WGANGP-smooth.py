"""
train_inverse_wgangp_chafee_infante_smooth_400.py

Train a Wasserstein GAN with gradient penalty (WGAN-GP) to reconstruct
initial conditions for the 2D Chafee--Infante equation from near-equilibrium
states.

Problem setup
-------------
Input  : near-equilibrium state u_T
Output : predicted initial state u_0

The generator is a U-Net style encoder-decoder with skip connections.
The critic is a PatchGAN-style convolutional critic.

The generator loss combines:
    - Wasserstein adversarial loss
    - normalized Lyapunov-energy mismatch
    - physics-informed residual loss
    - mean absolute error (MAE)
    - mean error penalty
    - variance error penalty
    - gradient mismatch penalty

The script:
    1. loads training/testing datasets,
    2. scales input/output fields,
    3. builds generator and critic,
    4. trains with checkpointing,
    5. evaluates on fixed test samples,
    6. saves best models by MAE and residual.

Boundary conditions
-------------------
Homogeneous Dirichlet boundary conditions are imposed by zeroing the
boundary of generated states and by zero-padding in the forward simulator.
"""

import os
import csv
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Concatenate,
    Conv2D,
    LeakyReLU,
    Activation,
    Dropout,
    LayerNormalization,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

SIM_STEPS = 400

TRAIN_NPZ = "YOUR/PATH/TO/TRAINING-DATASET.npz"
TEST_NPZ  = "YOUR/PATH/TO/TESTING-DATASET.npz"

# Physical parameters
DT = 0.001
GAMMA = 0.005
KAPPA = 4.7
L = 1.0
M = 128
HX = 2.0 * L / M
HY = HX
H  = HX * HY
MI = M - 2
J  = MI * MI

# Initial best values
FIRST_MAE      = float("inf")
FIRST_RESIDUAL = float("inf")

# Gradient loss weight
LAMBDA_GRAD     = 2.0
# Physical loss weights
ENERGY_WT       = 0.1
LAMBDA_RESIDUAL = 0.25
# Stats loss weights
LAMBDA_MEAN     = 0.25
LAMBDA_VAR      = 1.0
# Pixel loss weight
LAMBDA_MAE      = 3.0
# Gradient penalty weight
LAMBDA_GP       = 10.0
# Adversarial loss weight
LAMBDA_ADV      = 1.5

# Critic parameters
C_LEARNING    = 1e-4
CRITIC_SIZE   = 4
CRITIC_STEPS  = 5

# Generator parameters
G_LEARNING = 1e-4

# Training parameters
MAX_EPOCHS      = 10
SAVEEVERY       = 250
NUM_SAMPLES     = 128
VAL_BATCH_SIZE  = 16
BATCH_SIZE      = 8

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Output folders
LATEST_DIR         = "3_checkpoints_latest"
BEST_MAE_DIR       = "3_checkpoints_best_mae"
BEST_RESIDUAL_DIR  = "3_checkpoints_best_residual"
MODEL_DIR          = "3_models"

for folder in [LATEST_DIR, BEST_MAE_DIR, BEST_RESIDUAL_DIR, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

# Optional GPU memory growth
tf.experimental.numpy.experimental_enable_numpy_behavior()
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

TRAIN_CSV = "3_training_data.csv"
TEST_CSV  = "4_testing_data.csv"


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv_files() -> None:
    """Create training/testing CSV files with headers if they do not exist."""
    if not os.path.exists(TRAIN_CSV):
        with open(TRAIN_CSV, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration",
                "critic_loss",
                "generator_loss",
                "adversarial_loss",
                "energy_loss",
                "residual_loss",
                "mean_loss",
                "variance_loss",
                "gradient_loss",
                "mae_loss",
            ])

    if not os.path.exists(TEST_CSV):
        with open(TEST_CSV, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration",
                "samples",
                "gen_max_scaled",
                "gen_min_scaled",
                "mae_scaled",
                "mean_err_scaled",
                "var_err_scaled",
                "residual_scaled",
                "gen_max_phys_IC",
                "gen_min_phys_IC",
                "mae_phys_IC",
                "mean_err_phys_IC",
                "var_err_phys_IC",
                "energy_err_phys",
                "residual_phys_NE",
            ])


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(train_npz_path: str, test_npz_path: str):
    """
    Load training and testing datasets.

    Returns
    -------
    dataset : list
        [trainA, trainB], where
        trainA = scaled NE inputs,
        trainB = scaled IC targets.
    test_data : list
        [testA, testB] with the same scaling.
    image_shape : tuple
        Shape of one scaled input sample.
    ne_scale : float
        Physical scaling factor for NE data.
    ic_scale : float
        Physical scaling factor for IC data.
    """
    train_npz = np.load(train_npz_path, mmap_mode="r")
    x_src = train_npz["src"]   # near-equilibrium
    x_tar = train_npz["tar"]   # initial condition

    print("Training dataset loaded.")
    print("  src shape:", x_src.shape)
    print("  tar shape:", x_tar.shape)

    ne_scale = np.max(np.abs(x_src))
    ic_scale = np.max(np.abs(x_tar))

    trainA = (x_src / ne_scale)[..., None]
    trainB = (x_tar / ic_scale)[..., None]

    print("Training NE scale:", ne_scale)
    print("Training IC scale:", ic_scale)
    print("Training NE range:", trainA.min(), trainA.max())
    print("Training IC range:", trainB.min(), trainB.max())

    dataset = [trainA, trainB]
    image_shape = trainA.shape[1:]

    test_npz = np.load(test_npz_path, mmap_mode="r")
    testA = (test_npz["src"] / ne_scale)[..., None]
    testB = (test_npz["tar"] / ic_scale)[..., None]

    print("Testing dataset loaded.")
    print("  src shape:", test_npz["src"].shape)
    print("  tar shape:", test_npz["tar"].shape)
    print("Testing dataset uses training scales.")

    test_data = [testA, testB]
    return dataset, test_data, image_shape, ne_scale, ic_scale


# ============================================================
# PHYSICS UTILITIES
# ============================================================


def enforce_dbc_tensor(u: tf.Tensor) -> tf.Tensor:
    """
    Enforce homogeneous Dirichlet boundary conditions on a batch of images.

    Parameters
    ----------
    u : tf.Tensor
        Shape (B,H,W,1)

    Returns
    -------
    tf.Tensor
        Same shape, with zero boundary values.
    """
    interior = u[:, 1:-1, 1:-1, :]
    top_bottom = tf.zeros_like(u[:, :1, :, :])
    left_right = tf.zeros_like(u[:, 1:-1, :1, :])

    middle = tf.concat([left_right, interior, left_right], axis=2)
    return tf.concat([top_bottom, middle, top_bottom], axis=1)


def MLENERGY(phi_scaled: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Compute the discrete Lyapunov energy in physical units.

    Parameters
    ----------
    phi_scaled : tf.Tensor
        Shape (B,M,M,1) or (B,M,M), scaled data.
    scale : float
        Physical scale converting scaled values to physical values.

    Returns
    -------
    tf.Tensor
        Batch of energy values, shape (B,).
    """
    phi = tf.cast(scale, tf.float32) * tf.cast(phi_scaled, tf.float32)

    if phi.shape.rank == 4 and phi.shape[-1] == 1:
        phi = tf.squeeze(phi, axis=-1)
    if phi.shape.rank != 3:
        raise ValueError(f"Expected shape (batch, M, M), got {phi.shape}")

    bulkx = (phi[:, 2:, 1:-1] - phi[:, :-2, 1:-1]) / (2 * HX)
    bulky = (phi[:, 1:-1, 2:] - phi[:, 1:-1, :-2]) / (2 * HY)

    top    = (phi[:, 1, :] - phi[:, 0, :]) / HX
    bottom = (phi[:, -1, :] - phi[:, -2, :]) / HX
    left   = (phi[:, :, 1] - phi[:, :, 0]) / HY
    right  = (phi[:, :, -1] - phi[:, :, -2]) / HY

    Dx = GAMMA * 0.5 * H * (
        tf.reduce_sum(top**2, axis=1)
        + tf.reduce_sum(bottom**2, axis=1)
        + tf.reduce_sum(bulkx**2, axis=[1, 2])
    )
    Dy = GAMMA * 0.5 * H * (
        tf.reduce_sum(left**2, axis=1)
        + tf.reduce_sum(right**2, axis=1)
        + tf.reduce_sum(bulky**2, axis=[1, 2])
    )

    L4 = KAPPA * 0.25 * H * tf.reduce_sum(phi**4, axis=[1, 2])
    L2 = KAPPA * 0.5  * H * tf.reduce_sum(phi**2, axis=[1, 2])

    return Dx + Dy + L4 - L2


@tf.function
def forward_sim_interior(u0_phys, nsteps, dt, gamma, kappa, hx, hy):
    """
    Forward Euler simulation on the interior grid.

    Parameters
    ----------
    u0_phys : tf.Tensor
        Shape (B,128,128,1), physical units, zero boundary assumed.

    Returns
    -------
    tf.Tensor
        Interior solution after nsteps, shape (B,126,126).
    """
    u = u0_phys[:, 1:-1, 1:-1, 0]

    hx2 = hx * hx
    hy2 = hy * hy

    for _ in tf.range(nsteps):
        u_pad = tf.pad(
            u,
            paddings=[[0, 0], [1, 1], [1, 1]],
            mode="CONSTANT",
            constant_values=0.0,
        )

        u_c  = u_pad[:, 1:-1, 1:-1]
        u_up = u_pad[:, :-2, 1:-1]
        u_dn = u_pad[:, 2:, 1:-1]
        u_lf = u_pad[:, 1:-1, :-2]
        u_rt = u_pad[:, 1:-1, 2:]

        lap = (u_lf - 2.0 * u_c + u_rt) / hx2 + (u_up - 2.0 * u_c + u_dn) / hy2
        reaction = -kappa * (tf.pow(u_c, 3) - u_c)

        u = u_c + dt * (gamma * lap + reaction)

    return u


@tf.function
def residual_loss_interior(
    u0_pred_scaled,
    uT_target_scaled,
    nsteps,
    dt=DT,
    gamma=GAMMA,
    kappa=KAPPA,
    hx=HX,
    hy=HY,
    scale_u0=1.0,
    scale_uT=1.0,
    reduce_batch=True
):
    """
    Compute the physics-informed residual loss on the interior.

    Returns
    -------
    tf.Tensor
        Scalar MAE between forward-simulated prediction and target at time T,
        measured in physical NE units.
    """
    u0_phys = tf.cast(scale_u0, tf.float32) * tf.cast(u0_pred_scaled, tf.float32)
    uT_phys = tf.cast(scale_uT, tf.float32) * tf.cast(uT_target_scaled, tf.float32)

    uT_sim_int = forward_sim_interior(u0_phys, nsteps, dt, gamma, kappa, hx, hy)
    uT_tar_int = uT_phys[:, 1:-1, 1:-1, 0]

    per_sample = tf.reduce_mean(tf.abs(uT_sim_int - uT_tar_int), axis=[1, 2])

    if reduce_batch:
        return tf.reduce_mean(per_sample)
    return per_sample


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def critic(image_shape, base_filters=64) -> Model:
    """
    Build the PatchGAN critic.

    Parameters
    ----------
    image_shape : tuple
        Input shape (H,W,C).
    base_filters : int
        Number of filters in the first convolution block.

    Returns
    -------
    Model
        Critic model mapping (source, target) -> patch scores.
    """
    in_src = Input(shape=image_shape)
    in_tgt = Input(shape=image_shape)

    x = Concatenate()([in_src, in_tgt])

    def disc_block(x, filters, stride):
        x = Conv2D(
            filters,
            kernel_size=CRITIC_SIZE,
            strides=stride,
            padding="same",
            kernel_initializer="he_uniform",
            use_bias=True,
        )(x)
        x = LeakyReLU(0.2)(x)
        return x

    x = disc_block(x, base_filters * 1, 2)
    x = disc_block(x, base_filters * 2, 2)
    x = disc_block(x, base_filters * 4, 2)
    x = disc_block(x, base_filters * 8, 2)

    patch_out = Conv2D(1, kernel_size=CRITIC_SIZE, strides=1, padding="same")(x)
    return Model([in_src, in_tgt], patch_out, name="critic")


def encoder_block(layer_in, n_filters, norm=True):
    """One encoder block for the generator."""
    g = Conv2D(
        n_filters,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(layer_in)
    if norm:
        g = LayerNormalization()(g)
    g = LeakyReLU(negative_slope=0.2)(g)
    return g


def decoder_block(layer_in, skip_in, n_filters, dropout=False):
    """One decoder block for the generator."""
    g = UpSampling2D(size=(2, 2), interpolation="bilinear")(layer_in)
    g = Conv2D(
        n_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(g)
    g = LayerNormalization()(g)
    if dropout:
        g = Dropout(0.5)(g)
    g = Activation("relu")(g)
    g = Concatenate()([g, skip_in])
    return g


def generator(image_shape) -> Model:
    """
    Build the U-Net style generator.

    Parameters
    ----------
    image_shape : tuple
        Input shape (H,W,C).

    Returns
    -------
    Model
        Generator mapping NE -> predicted IC.
    """
    in_image = Input(shape=image_shape)

    # Encoder
    e1 = Conv2D(
        64,
        4,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(in_image)
    e1 = LeakyReLU(negative_slope=0.2)(e1)

    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)

    # Bottleneck
    b = Conv2D(
        512,
        4,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(e5)
    b = Activation("relu")(b)

    # Decoder
    d1 = decoder_block(b,  e5, 512, dropout=True)
    d2 = decoder_block(d1, e4, 512, dropout=True)
    d3 = decoder_block(d2, e3, 256, dropout=False)
    d4 = decoder_block(d3, e2, 128, dropout=False)
    d5 = decoder_block(d4, e1, 64,  dropout=False)

    g = UpSampling2D(size=(2, 2), interpolation="bilinear")(d5)
    g = Conv2D(
        1,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
    )(g)
    out_image = Activation("tanh")(g)

    return Model(in_image, out_image, name="generator")


def wgan(g_model: Model, c_model: Model, image_shape) -> Model:
    """
    Build the composite generator-critic model.

    The critic is frozen inside this composite model.
    """
    for layer in c_model.layers:
        layer.trainable = False

    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    c_out = c_model([in_src, gen_out])
    model = Model(inputs=in_src, outputs=[c_out, gen_out], name="wgan")

    for layer in c_model.layers:
        layer.trainable = True

    return model


# ============================================================
# LOSSES
# ============================================================

def generate_real_samples(dataset, n_samples):
    """
    Sample a random batch from the dataset.
    """
    X1, X2 = dataset
    ix = np.random.randint(0, X1.shape[0], n_samples)
    return X1[ix], X2[ix]


def stat_loss_scaled(real_scaled: tf.Tensor, fake_scaled: tf.Tensor):
    """
    Mean/variance matching loss in scaled units.
    """
    mean_real = tf.reduce_mean(real_scaled, axis=[1, 2, 3])
    mean_fake = tf.reduce_mean(fake_scaled, axis=[1, 2, 3])
    var_real  = tf.math.reduce_variance(real_scaled, axis=[1, 2, 3])
    var_fake  = tf.math.reduce_variance(fake_scaled, axis=[1, 2, 3])

    mean_loss = tf.reduce_mean(tf.abs(mean_real - mean_fake))
    var_loss  = tf.reduce_mean(tf.abs(var_real - var_fake))
    return mean_loss, var_loss


def gradient_l1_loss(real_scaled: tf.Tensor, fake_scaled: tf.Tensor) -> tf.Tensor:
    """
    L1 loss on first spatial differences in scaled IC units.
    """
    real_dx = real_scaled[:, 1:, :, :] - real_scaled[:, :-1, :, :]
    fake_dx = fake_scaled[:, 1:, :, :] - fake_scaled[:, :-1, :, :]

    real_dy = real_scaled[:, :, 1:, :] - real_scaled[:, :, :-1, :]
    fake_dy = fake_scaled[:, :, 1:, :] - fake_scaled[:, :, :-1, :]

    loss_dx = tf.reduce_mean(tf.abs(fake_dx - real_dx))
    loss_dy = tf.reduce_mean(tf.abs(fake_dy - real_dy))

    return loss_dx + loss_dy


def gradient_penalty(c_model: Model, real_A, real_B, fake_B) -> tf.Tensor:
    """
    Compute the WGAN-GP gradient penalty.
    """
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
        tf.print(
            "Gradient penalty failed:",
            "real_A", tf.shape(real_A),
            "real_B", tf.shape(real_B),
            "fake_B", tf.shape(fake_B),
        )
        raise RuntimeError("Gradient penalty gradients are None.")

    grads = tf.reshape(grads, [tf.shape(grads)[0], -1])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean((slopes - 1.0) ** 2)


TRAIN_STEPS = int(SIM_STEPS)


@tf.function
def train_step(
    real_A,
    real_B,
    generator_model,
    critic_model,
    g_optimizer,
    c_optimizer,
    ic_scale,
    ne_scale,
    lambda_adv=LAMBDA_ADV,
    lambda_gp=LAMBDA_GP,
    lambda_residual=LAMBDA_RESIDUAL,
    lambda_mae=LAMBDA_MAE,
    lambda_mean=LAMBDA_MEAN,
    lambda_var=LAMBDA_VAR,
    lambda_grad=LAMBDA_GRAD,
    energy_weight=ENERGY_WT,
):
    """
    Perform one WGAN-GP training step:
        - multiple critic updates
        - one generator update
    """
    real_A = tf.convert_to_tensor(real_A, dtype=tf.float32)
    real_B = tf.convert_to_tensor(real_B, dtype=tf.float32)

    # ---------------- Critic update(s) ----------------
    for _ in range(CRITIC_STEPS):
        with tf.GradientTape() as tape:
            fake_B = generator_model(real_A, training=True)
            fake_B = enforce_dbc_tensor(fake_B)

            c_real = critic_model([real_A, real_B], training=True)
            c_fake = critic_model([real_A, fake_B], training=True)
            gp = gradient_penalty(critic_model, real_A, real_B, fake_B)

            c_loss = tf.reduce_mean(c_fake) - tf.reduce_mean(c_real) + lambda_gp * gp

        grads = tape.gradient(c_loss, critic_model.trainable_variables)
        c_optimizer.apply_gradients(zip(grads, critic_model.trainable_variables))

    # ---------------- Generator update ----------------
    with tf.GradientTape() as tape:
        fake_B = generator_model(real_A, training=True)
        fake_B = enforce_dbc_tensor(fake_B)

        d_fake = critic_model([real_A, fake_B], training=True)
        adv_loss = -tf.reduce_mean(d_fake)

        mae = tf.reduce_mean(tf.abs(fake_B - real_B))
        mean_loss, var_loss = stat_loss_scaled(real_B, fake_B)
        grad_loss = gradient_l1_loss(real_B, fake_B)

        res_phys = residual_loss_interior(
            fake_B,
            real_A,
            TRAIN_STEPS,
            scale_u0=ic_scale,
            scale_uT=ne_scale,
            reduce_batch=True
        )
        res_loss = res_phys / tf.cast(ne_scale, tf.float32)

        E_fake = MLENERGY(fake_B, ic_scale)
        E_real = MLENERGY(real_B, ic_scale)
        energy_num = tf.reduce_mean(tf.abs(E_fake - E_real))
        energy_den = tf.reduce_mean(tf.abs(E_real)) + 1.0e-8
        energy_loss = energy_num / energy_den

        g_loss = (
            lambda_adv * adv_loss
            + energy_weight * energy_loss
            + lambda_residual * res_loss
            + lambda_mae * mae
            + lambda_mean * mean_loss
            + lambda_var * var_loss
            + lambda_grad * grad_loss
        )

    grads = tape.gradient(g_loss, generator_model.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator_model.trainable_variables))

    return c_loss, g_loss, adv_loss, energy_loss, res_loss, mean_loss, var_loss, grad_loss, mae


# ============================================================
# VALIDATION
# ============================================================

def summarize_performance(
    step,
    g_model: Model,
    test_data,
    ic_scale: float,
    ne_scale: float,
    num_samples=NUM_SAMPLES,
    sim_steps=SIM_STEPS,
    val_batch_size=VAL_BATCH_SIZE
):
    """
    Evaluate the current generator on a small fixed subset of test samples.

    Returns
    -------
    current_mae : float
        Mean scaled MAE over the selected samples.
    current_residual : float
        Mean scaled residual over the selected samples.
    """
    model_num = step + 1
    print("Running validation on the testing dataset.")
    print("Using current generator model:", model_num)

    test_num = test_data[0].shape[0]
    sam = np.random.choice(test_num, size=num_samples, replace=False)
    predict_data = g_model.predict(test_data[0][sam], verbose=0)
    predict_data = enforce_dbc_tensor(predict_data)

    max_list = np.zeros(num_samples)
    min_list = np.zeros(num_samples)
    means, vars_, energies, maes = [], [], [], []
    res_scaled_list = []

    for counter in range(num_samples):
        real_tar = test_data[1][sam][counter, :, :, 0]
        tar_mean = np.mean(real_tar)
        tar_var  = np.var(real_tar)
        tar_phi  = test_data[1][sam][counter:counter + 1]
        tar_en   = MLENERGY(tar_phi, ic_scale)

        gen     = predict_data[counter, :, :, 0]
        gen_phi = predict_data[counter:counter + 1]

        gen_max  = np.max(gen)
        gen_min  = np.min(gen)
        gen_mean = np.mean(gen)
        gen_var  = np.var(gen)
        gen_en   = MLENERGY(gen_phi, ic_scale)

        test_mean = np.abs(tar_mean - gen_mean)
        test_var  = np.abs(tar_var - gen_var)
        test_en   = float(tf.reduce_mean(tf.abs(tar_en - gen_en)).numpy())
        test_mae  = float(np.mean(np.abs(real_tar - gen)))

        realA_phi = test_data[0][sam][counter:counter + 1]
        res_phys = residual_loss_interior(
            tf.convert_to_tensor(gen_phi, dtype=tf.float32),
            tf.convert_to_tensor(realA_phi, dtype=tf.float32),
            sim_steps,
            scale_u0=ic_scale,
            scale_uT=ne_scale,
            reduce_batch=False
        ).numpy()
        test_res_scaled_dimless = float(res_phys) / float(ne_scale)

        max_list[counter] = gen_max
        min_list[counter] = gen_min
        means.append(test_mean)
        vars_.append(test_var)
        energies.append(test_en)
        maes.append(test_mae)
        res_scaled_list.append(test_res_scaled_dimless)

    gen_max_scaled   = float(np.mean(max_list))
    gen_min_scaled   = float(np.mean(min_list))
    mae_scaled       = float(np.mean(maes))
    mean_err_scaled  = float(np.mean(means))
    var_err_scaled   = float(np.mean(vars_))
    energy_err_phys  = float(np.mean(energies))

    residual_scaled  = float(np.mean(res_scaled_list))
    residual_phys_NE = float(ne_scale) * residual_scaled

    gen_max_phys_IC  = float(ic_scale) * gen_max_scaled
    gen_min_phys_IC  = float(ic_scale) * gen_min_scaled
    mae_phys_IC      = float(ic_scale) * mae_scaled
    mean_err_phys_IC = float(ic_scale) * mean_err_scaled
    var_err_phys_IC  = (float(ic_scale) ** 2) * var_err_scaled

    with open(TEST_CSV, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_num,
            sam,
            gen_max_scaled,
            gen_min_scaled,
            mae_scaled,
            mean_err_scaled,
            var_err_scaled,
            residual_scaled,
            gen_max_phys_IC,
            gen_min_phys_IC,
            mae_phys_IC,
            mean_err_phys_IC,
            var_err_phys_IC,
            energy_err_phys,
            residual_phys_NE,
        ])

    return mae_scaled, residual_scaled


# ============================================================
# TRAINING LOOP
# ============================================================

def train(c_model, g_model, wgan_model, dataset, test_data, max_epochs, ic_scale, ne_scale, n_batch=BATCH_SIZE):
    """
    Main training loop with:
        - rolling checkpoints,
        - best MAE archive,
        - best residual archive.
    """
    del wgan_model  # composite model is constructed for completeness; training uses explicit tapes

    g_optimizer = Adam(learning_rate=G_LEARNING, beta_1=0.0, beta_2=0.9)
    c_optimizer = Adam(learning_rate=C_LEARNING, beta_1=0.0, beta_2=0.9)

    step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
    best_mae_var = tf.Variable(np.inf, dtype=tf.float32, trainable=False, name="best_mae")
    best_res_var = tf.Variable(np.inf, dtype=tf.float32, trainable=False, name="best_residual")

    checkpoint = tf.train.Checkpoint(
        step=step,
        best_mae=best_mae_var,
        best_residual=best_res_var,
        g_optimizer=g_optimizer,
        c_optimizer=c_optimizer,
        generator=g_model,
        critic=c_model,
    )

    latest_manager = tf.train.CheckpointManager(checkpoint, directory=LATEST_DIR, max_to_keep=2)
    best_manager = tf.train.CheckpointManager(checkpoint, directory=BEST_MAE_DIR, max_to_keep=1)
    best_res_manager = tf.train.CheckpointManager(checkpoint, directory=BEST_RESIDUAL_DIR, max_to_keep=1)

    if latest_manager.latest_checkpoint:
        checkpoint.restore(latest_manager.latest_checkpoint).expect_partial()
        print("Restored LATEST checkpoint:", latest_manager.latest_checkpoint)
        print("  step =", int(step.numpy()))
        print("  best_mae =", float(best_mae_var.numpy()))
        print("  best_res =", float(best_res_var.numpy()))
    else:
        print("No LATEST checkpoint found. Training from scratch.")

    last_mae = float(best_mae_var.numpy())
    last_res = float(best_res_var.numpy())

    trainA, _ = dataset
    bat_per_epoch = int(len(trainA) / n_batch)
    n_steps = bat_per_epoch * max_epochs

    start_i = int(step.numpy())
    print("Starting iteration index =", start_i, "of", n_steps)

    for i in range(start_i, n_steps):
        X_realA, X_realB = generate_real_samples(dataset, n_batch)

        c_loss, g_loss, adv_loss, en_loss, res_loss, mean_loss, var_loss, grad_loss, mae = train_step(
            X_realA,
            X_realB,
            g_model,
            c_model,
            g_optimizer,
            c_optimizer,
            ic_scale,
            ne_scale,
        )

        step.assign(i + 1)

        if (i + 1) == 10 or (i + 1) % SAVE_EVERY == 0:
            with open(TRAIN_CSV, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    i + 1,
                    float(c_loss.numpy()),
                    float(g_loss.numpy()),
                    float(adv_loss.numpy()),
                    float(en_loss.numpy()),
                    float(res_loss.numpy()),
                    float(mean_loss.numpy()),
                    float(var_loss.numpy()),
                    float(grad_loss.numpy()),
                    float(mae.numpy()),
                ])

            try:
                latest_manager.save(checkpoint_number=int(step.numpy()))
                print(f"Saved LATEST checkpoint at step={int(step.numpy())}")
            except tf.errors.ResourceExhaustedError as e:
                print("Could not save LATEST checkpoint (disk full):", e)

            current_mae, current_res = summarize_performance(
                i,
                g_model,
                test_data,
                ic_scale,
                ne_scale,
                sim_steps=SIM_STEPS,
            )

            # ----- Best by MAE -----
            if current_mae < last_mae:
                last_mae = current_mae
                best_mae_var.assign(last_mae)

                g_model.save(os.path.join(MODEL_DIR, "best_generator_mae.keras"))
                c_model.save(os.path.join(MODEL_DIR, "best_critic_mae.keras"))

                with open(os.path.join(MODEL_DIR, "best_mae.txt"), "w") as f:
                    f.write(str(last_mae))

                try:
                    best_manager.save(checkpoint_number=int(step.numpy()))
                    print(f"New BEST(MAE): mae={last_mae:.6f} at step={int(step.numpy())}")
                except tf.errors.ResourceExhaustedError as e:
                    print("Could not archive BEST(MAE) checkpoint:", e)

            # ----- Best by residual -----
            if current_res < last_res:
                last_res = current_res
                best_res_var.assign(last_res)

                g_model.save(os.path.join(MODEL_DIR, "best_generator_residual.keras"))
                c_model.save(os.path.join(MODEL_DIR, "best_critic_residual.keras"))

                with open(os.path.join(MODEL_DIR, "best_residual.txt"), "w") as f:
                    f.write(str(last_res))

                try:
                    best_res_manager.save(checkpoint_number=int(step.numpy()))
                    print(f"New BEST(RESIDUAL): res={last_res:.6f} at step={int(step.numpy())}")
                except tf.errors.ResourceExhaustedError as e:
                    print("Could not archive BEST(RESIDUAL) checkpoint:", e)


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the full training pipeline."""
    initialize_csv_files()

    dataset, test_data, image_shape, ne_scale, ic_scale = load_dataset(TRAIN_NPZ, TEST_NPZ)

    c_model = critic(image_shape)
    print("PatchGAN critic output shape:", c_model.output_shape)
    print("Number of patches per image:", c_model.output_shape[1])

    g_model = generator(image_shape)
    wgan_model = wgan(g_model, c_model, image_shape)

    c_model.summary()
    g_model.summary()
    wgan_model.summary()

    plot_model(c_model, to_file="3_critic_model_plot.png", show_shapes=True, show_layer_names=True)
    plot_model(g_model, to_file="3_generator_model_plot.png", show_shapes=True, show_layer_names=True)
    plot_model(wgan_model, to_file="3_wgan_model_plot.png", show_shapes=True, show_layer_names=True)

    train(
        c_model,
        g_model,
        wgan_model,
        dataset,
        test_data,
        MAX_EPOCHS,
        ic_scale,
        ne_scale,
        n_batch=BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
