"""
Inverse Chafee--Infante Reconstruction with a Physics-Informed WGAN-GP

This script trains a Wasserstein generative adversarial network with
gradient penalty (WGAN-GP) to reconstruct initial conditions for the
two-dimensional Chafee--Infante equation

    u_t - gamma * Delta u + kappa * (u^3 - u) = 0

on

    Omega = [-L, L] x [-L, L],

subject to homogeneous Dirichlet boundary conditions.

The inverse-learning problem is

    input  : forward-evolved state u_T,
    output : reconstructed initial state u_0.

The training dataset is assumed to contain paired arrays

    src = u_T,
    tar = u_0.

The generator is a U-Net-style encoder-decoder with skip connections.
Homogeneous Dirichlet boundary conditions are enforced directly in the
generator output.

The critic is a conditional PatchGAN-style critic acting on pairs

    (u_T, u_0).

The generator loss combines

    1. Wasserstein adversarial loss,
    2. normalized Lyapunov-energy mismatch,
    3. physics-informed forward-consistency residual,
    4. mean absolute error,
    5. mean mismatch,
    6. variance mismatch,
    7. first-difference mismatch.

Important numerical note
------------------------
The canonical datasets may be generated with the Eyre-type semi-implicit
scheme. During neural-network training, however, the physics-informed
residual uses a differentiable Forward Euler simulator as a computationally
tractable surrogate for the forward map.

Thus the physics loss should be interpreted as a differentiable
forward-consistency regularizer rather than an exact reproduction of
the numerical scheme used to generate the dataset.

Requirements
------------
numpy
tensorflow
pydot / graphviz   (optional, for model plots)
"""

from pathlib import Path
import csv
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv2D,
    Dropout,
    Input,
    Layer,
    LayerNormalization,
    LeakyReLU,
    UpSampling2D,
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


# =============================================================================
# Global configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

TRAIN_NPZ = Path(
    "YOUR/PATH/TO/TRAINING-DATASET.npz"
)

TEST_NPZ = Path(
    "YOUR/PATH/TO/TESTING-DATASET.npz"
)


# -----------------------------------------------------------------------------
# PDE parameters
# -----------------------------------------------------------------------------

SIM_STEPS = 400

DT = 0.001

GAMMA = 0.005
KAPPA = 4.7

L = 1.0
M = 128

# M grid points include both endpoints of [-L,L].
HX = 2.0 * L / (M - 1)
HY = HX

CELL_AREA = HX * HY


# -----------------------------------------------------------------------------
# Generator-loss weights
# -----------------------------------------------------------------------------

LAMBDA_ADV = 1.5

ENERGY_WEIGHT = 0.1
LAMBDA_RESIDUAL = 0.25

LAMBDA_MAE = 3.0

LAMBDA_MEAN = 0.25
LAMBDA_VAR = 1.0

LAMBDA_GRAD = 2.0


# -----------------------------------------------------------------------------
# WGAN-GP parameters
# -----------------------------------------------------------------------------

LAMBDA_GP = 10.0

CRITIC_FILTERS = 64
CRITIC_KERNEL_SIZE = 4
CRITIC_STEPS = 5

CRITIC_LEARNING_RATE = 1.0e-4
GENERATOR_LEARNING_RATE = 1.0e-4


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

MAX_EPOCHS = 10

BATCH_SIZE = 8

SAVE_EVERY = 250

NUM_VALIDATION_SAMPLES = 128
VALIDATION_BATCH_SIZE = 16


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("training_output")

LATEST_DIR = (
    OUTPUT_DIR
    / "checkpoints_latest"
)

BEST_MAE_DIR = (
    OUTPUT_DIR
    / "checkpoints_best_mae"
)

BEST_RESIDUAL_DIR = (
    OUTPUT_DIR
    / "checkpoints_best_residual"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "models"
)

for directory in (
    OUTPUT_DIR,
    LATEST_DIR,
    BEST_MAE_DIR,
    BEST_RESIDUAL_DIR,
    MODEL_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


TRAIN_CSV = (
    OUTPUT_DIR
    / "training_history.csv"
)

TEST_CSV = (
    OUTPUT_DIR
    / "validation_history.csv"
)


# =============================================================================
# TensorFlow device configuration
# =============================================================================

def configure_tensorflow():
    """
    Configure available TensorFlow GPU devices.

    Memory growth is requested when supported. Failure to enable memory
    growth is not fatal because some TensorFlow backends manage device
    memory differently.
    """

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        "TensorFlow version:",
        tf.__version__,
    )

    print(
        "Detected GPU devices:",
        gpus,
    )

    for gpu in gpus:

        try:

            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )

        except (
            RuntimeError,
            ValueError,
        ):

            pass


# =============================================================================
# CSV initialization
# =============================================================================

def initialize_csv_files():
    """
    Create training and validation CSV files if necessary.
    """

    if not TRAIN_CSV.exists():

        with TRAIN_CSV.open(
            mode="w",
            newline="",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
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
                ]
            )

    if not TEST_CSV.exists():

        with TEST_CSV.open(
            mode="w",
            newline="",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "iteration",
                    "mae_scaled",
                    "residual_scaled",
                    "mean_error_scaled",
                    "variance_error_scaled",
                    "energy_error_physical",
                    "mae_physical_IC",
                    "residual_physical",
                    "generator_min_scaled",
                    "generator_max_scaled",
                ]
            )


# =============================================================================
# Dataset loading
# =============================================================================

def prepare_array(array):
    """
    Convert a stored dataset array to shape (N,M,M,1).

    Accepted input shapes are

        (N,M,M),
        (N,M,M,1),
        (N,1,M,M).
    """

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    if array.ndim == 3:

        array = array[
            ...,
            np.newaxis,
        ]

    elif (
        array.ndim == 4
        and array.shape[-1] == 1
    ):

        pass

    elif (
        array.ndim == 4
        and array.shape[1] == 1
    ):

        array = np.transpose(
            array,
            (
                0,
                2,
                3,
                1,
            ),
        )

    else:

        raise ValueError(
            "Unsupported dataset shape: "
            f"{array.shape}"
        )

    if array.shape[1:] != (
        M,
        M,
        1,
    ):

        raise ValueError(
            "Expected image shape "
            f"({M},{M},1); "
            f"received {array.shape[1:]}."
        )

    return array


def load_dataset(
    train_path,
    test_path,
):
    """
    Load and scale the training and testing datasets.

    Scaling factors are computed from the training dataset only.

    Returns
    -------
    train_data : tuple
        Scaled (src, tar) training arrays.
    test_data : tuple
        Scaled (src, tar) testing arrays.
    image_shape : tuple
        Shape (M,M,1).
    ne_scale : float
        Physical scale for forward-evolved states.
    ic_scale : float
        Physical scale for initial conditions.
    """

    if not train_path.exists():

        raise FileNotFoundError(
            f"Training dataset not found: "
            f"{train_path}"
        )

    if not test_path.exists():

        raise FileNotFoundError(
            f"Testing dataset not found: "
            f"{test_path}"
        )

    with np.load(
        train_path
    ) as data:

        train_src = prepare_array(
            data["src"]
        )

        train_tar = prepare_array(
            data["tar"]
        )

    with np.load(
        test_path
    ) as data:

        test_src = prepare_array(
            data["src"]
        )

        test_tar = prepare_array(
            data["tar"]
        )

    ne_scale = float(
        np.max(
            np.abs(
                train_src
            )
        )
    )

    ic_scale = float(
        np.max(
            np.abs(
                train_tar
            )
        )
    )

    if ne_scale <= 0.0:

        raise ValueError(
            "Training src scale is zero."
        )

    if ic_scale <= 0.0:

        raise ValueError(
            "Training tar scale is zero."
        )

    train_A = (
        train_src
        / ne_scale
    ).astype(
        np.float32
    )

    train_B = (
        train_tar
        / ic_scale
    ).astype(
        np.float32
    )

    test_A = (
        test_src
        / ne_scale
    ).astype(
        np.float32
    )

    test_B = (
        test_tar
        / ic_scale
    ).astype(
        np.float32
    )

    print()
    print("Training dataset")
    print("----------------")
    print("src shape:", train_A.shape)
    print("tar shape:", train_B.shape)

    print()
    print("Training scales")
    print("----------------")
    print("Forward-state scale:", ne_scale)
    print("Initial-state scale:", ic_scale)

    print()
    print("Scaled training ranges")
    print("----------------------")

    print(
        "src:",
        train_A.min(),
        train_A.max(),
    )

    print(
        "tar:",
        train_B.min(),
        train_B.max(),
    )

    print()
    print("Testing dataset")
    print("----------------")
    print("src shape:", test_A.shape)
    print("tar shape:", test_B.shape)

    return (
        (train_A, train_B),
        (test_A, test_B),
        train_A.shape[1:],
        ne_scale,
        ic_scale,
    )


# =============================================================================
# Homogeneous Dirichlet boundary layer
# =============================================================================

@tf.keras.utils.register_keras_serializable(
    package="MLEE"
)
class EnforceDirichletBoundary(Layer):
    """
    Set all image-boundary values equal to zero.

    The layer leaves interior pixels unchanged and therefore ensures that
    every generator output satisfies homogeneous Dirichlet boundary
    conditions.
    """

    def call(
        self,
        inputs,
    ):

        interior = inputs[
            :,
            1:-1,
            1:-1,
            :,
        ]

        batch_size = tf.shape(
            inputs
        )[0]

        channels = tf.shape(
            inputs
        )[3]

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

        output = tf.ensure_shape(
            output,
            inputs.shape,
        )

        del batch_size
        del channels

        return output

    def get_config(
        self,
    ):

        return super().get_config()


# =============================================================================
# Physics: discrete Lyapunov energy
# =============================================================================

def lyapunov_energy_tensor(
    phi_scaled,
    scale,
):
    r"""
    Compute the discrete Chafee--Infante Lyapunov energy.

    The physical field is

        phi = scale * phi_scaled.

    The discrete energy matches the finite-difference diagnostic used by
    the canonical simulation code:

        E(phi)
        =
        [
            gamma/2 sum |D_x phi|^2
            + gamma/2 sum |D_y phi|^2
            + kappa sum (phi^4/4 - phi^2/2)
        ] h_x h_y.

    Parameters
    ----------
    phi_scaled : tf.Tensor
        Shape (B,M,M,1).
    scale : float
        Physical scaling factor.

    Returns
    -------
    tf.Tensor
        Energy for each sample, shape (B,).
    """

    phi = (
        tf.cast(
            scale,
            tf.float32,
        )
        *
        tf.cast(
            phi_scaled,
            tf.float32,
        )
    )

    phi = tf.squeeze(
        phi,
        axis=-1,
    )

    dx = (
        phi[:, 1:, :]
        - phi[:, :-1, :]
    ) / HX

    dy = (
        phi[:, :, 1:]
        - phi[:, :, :-1]
    ) / HY

    gradient_energy = (
        0.5
        * GAMMA
        * (
            tf.reduce_sum(
                dx**2,
                axis=[1, 2],
            )
            +
            tf.reduce_sum(
                dy**2,
                axis=[1, 2],
            )
        )
    )

    potential = (
        0.25
        * phi**4
        -
        0.5
        * phi**2
    )

    potential_energy = (
        KAPPA
        *
        tf.reduce_sum(
            potential,
            axis=[1, 2],
        )
    )

    return (
        gradient_energy
        + potential_energy
    ) * CELL_AREA


# =============================================================================
# Physics: differentiable Forward Euler surrogate
# =============================================================================

@tf.function
def forward_euler_surrogate(
    u0_physical,
    nsteps,
):
    r"""
    Evolve a batch using explicit Forward Euler.

    This function is used only as a differentiable physics surrogate during
    neural-network optimization.

    It does not claim to reproduce the Eyre discretization used to generate
    the canonical dataset.

    Parameters
    ----------
    u0_physical : tf.Tensor
        Shape (B,M,M,1), physical units.
    nsteps : int
        Number of Forward Euler steps.

    Returns
    -------
    tf.Tensor
        Interior solution after ``nsteps`` steps,
        shape (B,M-2,M-2).
    """

    u = tf.cast(
        u0_physical[
            :,
            1:-1,
            1:-1,
            0,
        ],
        tf.float32,
    )

    hx2 = tf.cast(
        HX**2,
        tf.float32,
    )

    hy2 = tf.cast(
        HY**2,
        tf.float32,
    )

    dt = tf.cast(
        DT,
        tf.float32,
    )

    gamma = tf.cast(
        GAMMA,
        tf.float32,
    )

    kappa = tf.cast(
        KAPPA,
        tf.float32,
    )

    for _ in tf.range(
        nsteps
    ):

        padded = tf.pad(
            u,
            paddings=[
                [0, 0],
                [1, 1],
                [1, 1],
            ],
            mode="CONSTANT",
        )

        center = padded[
            :,
            1:-1,
            1:-1,
        ]

        up = padded[
            :,
            :-2,
            1:-1,
        ]

        down = padded[
            :,
            2:,
            1:-1,
        ]

        left = padded[
            :,
            1:-1,
            :-2,
        ]

        right = padded[
            :,
            1:-1,
            2:,
        ]

        laplacian = (
            (
                left
                - 2.0 * center
                + right
            )
            / hx2
            +
            (
                up
                - 2.0 * center
                + down
            )
            / hy2
        )

        reaction = (
            -kappa
            * (
                center**3
                - center
            )
        )

        u = (
            center
            +
            dt
            * (
                gamma
                * laplacian
                +
                reaction
            )
        )

    return u


@tf.function
def forward_consistency_loss(
    u0_pred_scaled,
    uT_target_scaled,
    ic_scale,
    ne_scale,
    reduce_batch=True,
):
    """
    Compute the differentiable forward-consistency residual.

    A predicted initial condition is mapped forward by the Forward Euler
    surrogate and compared with the observed forward state.

    The returned quantity is normalized by the physical scale of u_T.
    """

    u0_physical = (
        tf.cast(
            ic_scale,
            tf.float32,
        )
        *
        tf.cast(
            u0_pred_scaled,
            tf.float32,
        )
    )

    uT_physical = (
        tf.cast(
            ne_scale,
            tf.float32,
        )
        *
        tf.cast(
            uT_target_scaled,
            tf.float32,
        )
    )

    simulated = forward_euler_surrogate(
        u0_physical,
        SIM_STEPS,
    )

    target = uT_physical[
        :,
        1:-1,
        1:-1,
        0,
    ]

    per_sample_physical = (
        tf.reduce_mean(
            tf.abs(
                simulated
                - target
            ),
            axis=[1, 2],
        )
    )

    per_sample_scaled = (
        per_sample_physical
        /
        tf.cast(
            ne_scale,
            tf.float32,
        )
    )

    if reduce_batch:

        return tf.reduce_mean(
            per_sample_scaled
        )

    return per_sample_scaled


# =============================================================================
# Generator
# =============================================================================

def encoder_block(
    layer_input,
    filters,
    normalize=True,
):
    """
    Construct one U-Net encoder block.
    """

    x = Conv2D(
        filters,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=not normalize,
    )(
        layer_input
    )

    if normalize:

        x = LayerNormalization()(
            x
        )

    x = LeakyReLU(
        negative_slope=0.2
    )(
        x
    )

    return x


def decoder_block(
    layer_input,
    skip_input,
    filters,
    dropout=False,
):
    """
    Construct one U-Net decoder block.
    """

    x = UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
    )(
        layer_input
    )

    x = Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(
        x
    )

    x = LayerNormalization()(
        x
    )

    if dropout:

        x = Dropout(
            0.5
        )(
            x
        )

    x = Activation(
        "relu"
    )(
        x
    )

    x = Concatenate()(
        [
            x,
            skip_input,
        ]
    )

    return x


def build_generator(
    image_shape,
):
    """
    Build the U-Net inverse generator.

    The final EnforceDirichletBoundary layer guarantees that the saved
    generator itself produces zero boundary values.
    """

    inputs = Input(
        shape=image_shape,
        name="forward_state",
    )

    # Encoder
    e1 = Conv2D(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(
        inputs
    )

    e1 = LeakyReLU(
        negative_slope=0.2
    )(
        e1
    )

    e2 = encoder_block(
        e1,
        128,
    )

    e3 = encoder_block(
        e2,
        256,
    )

    e4 = encoder_block(
        e3,
        512,
    )

    e5 = encoder_block(
        e4,
        512,
    )

    # Bottleneck
    bottleneck = Conv2D(
        512,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(
        e5
    )

    bottleneck = Activation(
        "relu"
    )(
        bottleneck
    )

    # Decoder
    d1 = decoder_block(
        bottleneck,
        e5,
        512,
        dropout=True,
    )

    d2 = decoder_block(
        d1,
        e4,
        512,
        dropout=True,
    )

    d3 = decoder_block(
        d2,
        e3,
        256,
    )

    d4 = decoder_block(
        d3,
        e2,
        128,
    )

    d5 = decoder_block(
        d4,
        e1,
        64,
    )

    output = UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
    )(
        d5
    )

    output = Conv2D(
        1,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_uniform",
    )(
        output
    )

    output = Activation(
        "tanh"
    )(
        output
    )

    output = EnforceDirichletBoundary(
        name="homogeneous_dirichlet_boundary"
    )(
        output
    )

    return Model(
        inputs,
        output,
        name="inverse_generator",
    )


# =============================================================================
# Critic
# =============================================================================

def critic_block(
    layer_input,
    filters,
    stride,
):
    """
    Construct one convolutional critic block.
    """

    x = Conv2D(
        filters,
        kernel_size=CRITIC_KERNEL_SIZE,
        strides=stride,
        padding="same",
        kernel_initializer="he_uniform",
    )(
        layer_input
    )

    x = LeakyReLU(
        negative_slope=0.2
    )(
        x
    )

    return x


def build_critic(
    image_shape,
):
    """
    Build a conditional PatchGAN critic.

    The critic maps

        (forward state, initial state)

    to a spatial field of Wasserstein scores.
    """

    source = Input(
        shape=image_shape,
        name="forward_state",
    )

    target = Input(
        shape=image_shape,
        name="initial_state",
    )

    x = Concatenate()(
        [
            source,
            target,
        ]
    )

    x = critic_block(
        x,
        CRITIC_FILTERS,
        2,
    )

    x = critic_block(
        x,
        2 * CRITIC_FILTERS,
        2,
    )

    x = critic_block(
        x,
        4 * CRITIC_FILTERS,
        2,
    )

    x = critic_block(
        x,
        8 * CRITIC_FILTERS,
        2,
    )

    output = Conv2D(
        1,
        kernel_size=CRITIC_KERNEL_SIZE,
        strides=1,
        padding="same",
        name="wasserstein_score",
    )(
        x
    )

    return Model(
        [
            source,
            target,
        ],
        output,
        name="conditional_critic",
    )


# =============================================================================
# Auxiliary losses
# =============================================================================

def statistical_losses(
    real,
    fake,
):
    """
    Compute sample-wise mean and variance mismatch.
    """

    mean_real = tf.reduce_mean(
        real,
        axis=[1, 2, 3],
    )

    mean_fake = tf.reduce_mean(
        fake,
        axis=[1, 2, 3],
    )

    var_real = tf.math.reduce_variance(
        real,
        axis=[1, 2, 3],
    )

    var_fake = tf.math.reduce_variance(
        fake,
        axis=[1, 2, 3],
    )

    mean_loss = tf.reduce_mean(
        tf.abs(
            mean_real
            - mean_fake
        )
    )

    variance_loss = tf.reduce_mean(
        tf.abs(
            var_real
            - var_fake
        )
    )

    return (
        mean_loss,
        variance_loss,
    )


def gradient_l1_loss(
    real,
    fake,
):
    """
    Compute L1 mismatch of first spatial differences.
    """

    real_dx = (
        real[:, 1:, :, :]
        - real[:, :-1, :, :]
    )

    fake_dx = (
        fake[:, 1:, :, :]
        - fake[:, :-1, :, :]
    )

    real_dy = (
        real[:, :, 1:, :]
        - real[:, :, :-1, :]
    )

    fake_dy = (
        fake[:, :, 1:, :]
        - fake[:, :, :-1, :]
    )

    loss_x = tf.reduce_mean(
        tf.abs(
            fake_dx
            - real_dx
        )
    )

    loss_y = tf.reduce_mean(
        tf.abs(
            fake_dy
            - real_dy
        )
    )

    return (
        loss_x
        + loss_y
    )


# =============================================================================
# WGAN-GP gradient penalty
# =============================================================================

def gradient_penalty(
    critic_model,
    real_source,
    real_target,
    fake_target,
):
    """
    Compute the WGAN-GP interpolation penalty.
    """

    batch_size = tf.shape(
        real_target
    )[0]

    alpha = tf.random.uniform(
        shape=[
            batch_size,
            1,
            1,
            1,
        ],
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    interpolated = (
        alpha
        * real_target
        +
        (
            1.0
            - alpha
        )
        * fake_target
    )

    with tf.GradientTape() as tape:

        tape.watch(
            interpolated
        )

        prediction = critic_model(
            [
                real_source,
                interpolated,
            ],
            training=True,
        )

        prediction = tf.reduce_sum(
            prediction,
            axis=[1, 2, 3],
        )

    gradients = tape.gradient(
        prediction,
        interpolated,
    )

    if gradients is None:

        raise RuntimeError(
            "Unable to compute critic gradients "
            "for WGAN-GP."
        )

    gradients = tf.reshape(
        gradients,
        [
            batch_size,
            -1,
        ],
    )

    slopes = tf.sqrt(
        tf.reduce_sum(
            gradients**2,
            axis=1,
        )
        + 1.0e-12
    )

    return tf.reduce_mean(
        (
            slopes
            - 1.0
        )**2
    )


# =============================================================================
# Batch sampling
# =============================================================================

def generate_real_samples(
    dataset,
    batch_size,
):
    """
    Sample one random training batch.
    """

    source, target = dataset

    indices = np.random.randint(
        0,
        source.shape[0],
        size=batch_size,
    )

    return (
        source[indices],
        target[indices],
    )


# =============================================================================
# Training step
# =============================================================================

@tf.function
def train_step(
    real_A,
    real_B,
    generator_model,
    critic_model,
    generator_optimizer,
    critic_optimizer,
    ic_scale,
    ne_scale,
):
    """
    Perform multiple critic updates followed by one generator update.
    """

    real_A = tf.convert_to_tensor(
        real_A,
        dtype=tf.float32,
    )

    real_B = tf.convert_to_tensor(
        real_B,
        dtype=tf.float32,
    )

    # -------------------------------------------------------------------------
    # Critic updates
    # -------------------------------------------------------------------------

    critic_loss = tf.constant(
        0.0,
        dtype=tf.float32,
    )

    for _ in range(
        CRITIC_STEPS
    ):

        fake_B = generator_model(
            real_A,
            training=True,
        )

        fake_B = tf.stop_gradient(
            fake_B
        )

        with tf.GradientTape() as tape:

            critic_real = critic_model(
                [
                    real_A,
                    real_B,
                ],
                training=True,
            )

            critic_fake = critic_model(
                [
                    real_A,
                    fake_B,
                ],
                training=True,
            )

            gp = gradient_penalty(
                critic_model,
                real_A,
                real_B,
                fake_B,
            )

            critic_loss = (
                tf.reduce_mean(
                    critic_fake
                )
                -
                tf.reduce_mean(
                    critic_real
                )
                +
                LAMBDA_GP
                * gp
            )

        gradients = tape.gradient(
            critic_loss,
            critic_model.trainable_variables,
        )

        critic_optimizer.apply_gradients(
            zip(
                gradients,
                critic_model.trainable_variables,
            )
        )

    # -------------------------------------------------------------------------
    # Generator update
    # -------------------------------------------------------------------------

    with tf.GradientTape() as tape:

        fake_B = generator_model(
            real_A,
            training=True,
        )

        critic_fake = critic_model(
            [
                real_A,
                fake_B,
            ],
            training=False,
        )

        adversarial_loss = (
            -tf.reduce_mean(
                critic_fake
            )
        )

        mae_loss = tf.reduce_mean(
            tf.abs(
                fake_B
                - real_B
            )
        )

        (
            mean_loss,
            variance_loss,
        ) = statistical_losses(
            real_B,
            fake_B,
        )

        spatial_gradient_loss = (
            gradient_l1_loss(
                real_B,
                fake_B,
            )
        )

        residual_loss = (
            forward_consistency_loss(
                fake_B,
                real_A,
                ic_scale,
                ne_scale,
                reduce_batch=True,
            )
        )

        fake_energy = (
            lyapunov_energy_tensor(
                fake_B,
                ic_scale,
            )
        )

        real_energy = (
            lyapunov_energy_tensor(
                real_B,
                ic_scale,
            )
        )

        energy_numerator = (
            tf.reduce_mean(
                tf.abs(
                    fake_energy
                    - real_energy
                )
            )
        )

        energy_denominator = (
            tf.reduce_mean(
                tf.abs(
                    real_energy
                )
            )
            + 1.0e-8
        )

        energy_loss = (
            energy_numerator
            / energy_denominator
        )

        generator_loss = (
            LAMBDA_ADV
            * adversarial_loss

            + ENERGY_WEIGHT
            * energy_loss

            + LAMBDA_RESIDUAL
            * residual_loss

            + LAMBDA_MAE
            * mae_loss

            + LAMBDA_MEAN
            * mean_loss

            + LAMBDA_VAR
            * variance_loss

            + LAMBDA_GRAD
            * spatial_gradient_loss
        )

    gradients = tape.gradient(
        generator_loss,
        generator_model.trainable_variables,
    )

    generator_optimizer.apply_gradients(
        zip(
            gradients,
            generator_model.trainable_variables,
        )
    )

    return (
        critic_loss,
        generator_loss,
        adversarial_loss,
        energy_loss,
        residual_loss,
        mean_loss,
        variance_loss,
        spatial_gradient_loss,
        mae_loss,
    )


# =============================================================================
# Validation set
# =============================================================================

def create_validation_indices(
    test_size,
):
    """
    Construct one fixed validation subset for model selection.
    """

    sample_count = min(
        NUM_VALIDATION_SAMPLES,
        test_size,
    )

    rng = np.random.default_rng(
        SEED + 1
    )

    return np.sort(
        rng.choice(
            test_size,
            size=sample_count,
            replace=False,
        )
    )


# =============================================================================
# Validation
# =============================================================================

def summarize_performance(
    iteration,
    generator_model,
    test_data,
    validation_indices,
    ic_scale,
    ne_scale,
):
    """
    Evaluate the generator on the fixed validation subset.

    Returns
    -------
    mae_scaled : float
        Mean scaled reconstruction MAE.
    residual_scaled : float
        Mean scaled Forward Euler surrogate residual.
    """

    test_A, test_B = test_data

    source = test_A[
        validation_indices
    ]

    target = test_B[
        validation_indices
    ]

    prediction = generator_model.predict(
        source,
        batch_size=VALIDATION_BATCH_SIZE,
        verbose=0,
    )

    mae_scaled = float(
        np.mean(
            np.abs(
                prediction
                - target
            )
        )
    )

    generator_min = float(
        np.mean(
            np.min(
                prediction,
                axis=(1, 2, 3),
            )
        )
    )

    generator_max = float(
        np.mean(
            np.max(
                prediction,
                axis=(1, 2, 3),
            )
        )
    )

    target_means = np.mean(
        target,
        axis=(1, 2, 3),
    )

    prediction_means = np.mean(
        prediction,
        axis=(1, 2, 3),
    )

    mean_error_scaled = float(
        np.mean(
            np.abs(
                target_means
                - prediction_means
            )
        )
    )

    target_variances = np.var(
        target,
        axis=(1, 2, 3),
    )

    prediction_variances = np.var(
        prediction,
        axis=(1, 2, 3),
    )

    variance_error_scaled = float(
        np.mean(
            np.abs(
                target_variances
                - prediction_variances
            )
        )
    )

    real_energy = (
        lyapunov_energy_tensor(
            tf.convert_to_tensor(
                target,
                dtype=tf.float32,
            ),
            ic_scale,
        )
        .numpy()
    )

    fake_energy = (
        lyapunov_energy_tensor(
            tf.convert_to_tensor(
                prediction,
                dtype=tf.float32,
            ),
            ic_scale,
        )
        .numpy()
    )

    energy_error_physical = float(
        np.mean(
            np.abs(
                fake_energy
                - real_energy
            )
        )
    )

    # -------------------------------------------------------------------------
    # Residual evaluation in manageable batches
    # -------------------------------------------------------------------------

    residual_values = []

    total = source.shape[0]

    for start in range(
        0,
        total,
        VALIDATION_BATCH_SIZE,
    ):

        stop = min(
            start
            + VALIDATION_BATCH_SIZE,
            total,
        )

        residual_batch = (
            forward_consistency_loss(
                tf.convert_to_tensor(
                    prediction[
                        start:stop
                    ],
                    dtype=tf.float32,
                ),
                tf.convert_to_tensor(
                    source[
                        start:stop
                    ],
                    dtype=tf.float32,
                ),
                ic_scale,
                ne_scale,
                reduce_batch=False,
            )
            .numpy()
        )

        residual_values.extend(
            residual_batch.tolist()
        )

    residual_scaled = float(
        np.mean(
            residual_values
        )
    )

    mae_physical = (
        float(ic_scale)
        * mae_scaled
    )

    residual_physical = (
        float(ne_scale)
        * residual_scaled
    )

    print()

    print(
        f"Validation at iteration "
        f"{iteration}"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Scaled MAE:       "
        f"{mae_scaled:.8f}"
    )

    print(
        f"Physical IC MAE:  "
        f"{mae_physical:.8e}"
    )

    print(
        f"Scaled residual:  "
        f"{residual_scaled:.8f}"
    )

    print(
        f"Physical residual:"
        f" {residual_physical:.8e}"
    )

    with TEST_CSV.open(
        mode="a",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                iteration,
                mae_scaled,
                residual_scaled,
                mean_error_scaled,
                variance_error_scaled,
                energy_error_physical,
                mae_physical,
                residual_physical,
                generator_min,
                generator_max,
            ]
        )

    return (
        mae_scaled,
        residual_scaled,
    )


# =============================================================================
# Model plots
# =============================================================================

def save_model_plots(
    generator_model,
    critic_model,
):
    """
    Save architecture diagrams when Graphviz is available.
    """

    try:

        plot_model(
            generator_model,
            to_file=str(
                OUTPUT_DIR
                / "generator_architecture.png"
            ),
            show_shapes=True,
            show_layer_names=True,
        )

        plot_model(
            critic_model,
            to_file=str(
                OUTPUT_DIR
                / "critic_architecture.png"
            ),
            show_shapes=True,
            show_layer_names=True,
        )

    except Exception as error:

        print(
            "Model diagrams were not generated:"
        )

        print(
            error
        )


# =============================================================================
# Training loop
# =============================================================================

def train(
    critic_model,
    generator_model,
    dataset,
    test_data,
    max_epochs,
    ic_scale,
    ne_scale,
):
    """
    Train the physics-informed WGAN-GP.

    Checkpoints are maintained for

        1. latest training state,
        2. best validation MAE,
        3. best validation residual.
    """

    generator_optimizer = Adam(
        learning_rate=GENERATOR_LEARNING_RATE,
        beta_1=0.0,
        beta_2=0.9,
    )

    critic_optimizer = Adam(
        learning_rate=CRITIC_LEARNING_RATE,
        beta_1=0.0,
        beta_2=0.9,
    )

    global_step = tf.Variable(
        0,
        dtype=tf.int64,
        trainable=False,
        name="global_step",
    )

    best_mae = tf.Variable(
        np.inf,
        dtype=tf.float32,
        trainable=False,
        name="best_mae",
    )

    best_residual = tf.Variable(
        np.inf,
        dtype=tf.float32,
        trainable=False,
        name="best_residual",
    )

    checkpoint = tf.train.Checkpoint(
        global_step=global_step,
        best_mae=best_mae,
        best_residual=best_residual,
        generator_optimizer=generator_optimizer,
        critic_optimizer=critic_optimizer,
        generator=generator_model,
        critic=critic_model,
    )

    latest_manager = (
        tf.train.CheckpointManager(
            checkpoint,
            directory=str(
                LATEST_DIR
            ),
            max_to_keep=2,
        )
    )

    best_mae_manager = (
        tf.train.CheckpointManager(
            checkpoint,
            directory=str(
                BEST_MAE_DIR
            ),
            max_to_keep=1,
        )
    )

    best_residual_manager = (
        tf.train.CheckpointManager(
            checkpoint,
            directory=str(
                BEST_RESIDUAL_DIR
            ),
            max_to_keep=1,
        )
    )

    if latest_manager.latest_checkpoint:

        checkpoint.restore(
            latest_manager.latest_checkpoint
        )

        print(
            "Restored checkpoint:",
            latest_manager.latest_checkpoint,
        )

        print(
            "Global step:",
            int(
                global_step.numpy()
            ),
        )

        print(
            "Best MAE:",
            float(
                best_mae.numpy()
            ),
        )

        print(
            "Best residual:",
            float(
                best_residual.numpy()
            ),
        )

    else:

        print(
            "No checkpoint found. "
            "Starting from scratch."
        )

    train_A, _ = dataset

    batches_per_epoch = (
        train_A.shape[0]
        // BATCH_SIZE
    )

    total_iterations = (
        batches_per_epoch
        * max_epochs
    )

    validation_indices = (
        create_validation_indices(
            test_data[0].shape[0]
        )
    )

    print()
    print(
        "Fixed validation sample count:",
        len(validation_indices),
    )

    print(
        "Starting iteration:",
        int(
            global_step.numpy()
        ),
    )

    print(
        "Total target iterations:",
        total_iterations,
    )

    # -------------------------------------------------------------------------
    # Main iteration
    # -------------------------------------------------------------------------

    for iteration in range(
        int(
            global_step.numpy()
        ),
        total_iterations,
    ):

        (
            real_A,
            real_B,
        ) = generate_real_samples(
            dataset,
            BATCH_SIZE,
        )

        losses = train_step(
            real_A,
            real_B,
            generator_model,
            critic_model,
            generator_optimizer,
            critic_optimizer,
            ic_scale,
            ne_scale,
        )

        (
            critic_loss,
            generator_loss,
            adversarial_loss,
            energy_loss,
            residual_loss,
            mean_loss,
            variance_loss,
            spatial_gradient_loss,
            mae_loss,
        ) = losses

        current_iteration = (
            iteration + 1
        )

        global_step.assign(
            current_iteration
        )

        # ---------------------------------------------------------------------
        # Console progress
        # ---------------------------------------------------------------------

        if (
            current_iteration == 1
            or current_iteration % 25 == 0
        ):

            print(
                f"iter {current_iteration:7d} | "
                f"critic {float(critic_loss.numpy()): .5f} | "
                f"generator {float(generator_loss.numpy()): .5f} | "
                f"adv {float(adversarial_loss.numpy()): .5f} | "
                f"energy {float(energy_loss.numpy()):.5f} | "
                f"res {float(residual_loss.numpy()):.5f} | "
                f"mae {float(mae_loss.numpy()):.5f}"
            )

        # ---------------------------------------------------------------------
        # Checkpoint and validation
        # ---------------------------------------------------------------------

        if (
            current_iteration == 10
            or current_iteration
            % SAVE_EVERY
            == 0
        ):

            with TRAIN_CSV.open(
                mode="a",
                newline="",
            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow(
                    [
                        current_iteration,
                        float(
                            critic_loss.numpy()
                        ),
                        float(
                            generator_loss.numpy()
                        ),
                        float(
                            adversarial_loss.numpy()
                        ),
                        float(
                            energy_loss.numpy()
                        ),
                        float(
                            residual_loss.numpy()
                        ),
                        float(
                            mean_loss.numpy()
                        ),
                        float(
                            variance_loss.numpy()
                        ),
                        float(
                            spatial_gradient_loss.numpy()
                        ),
                        float(
                            mae_loss.numpy()
                        ),
                    ]
                )

            latest_manager.save(
                checkpoint_number=current_iteration
            )

            print(
                "Saved latest checkpoint "
                f"at iteration {current_iteration}."
            )

            (
                current_mae,
                current_residual,
            ) = summarize_performance(
                current_iteration,
                generator_model,
                test_data,
                validation_indices,
                ic_scale,
                ne_scale,
            )

            # -----------------------------------------------------------------
            # Best by reconstruction MAE
            # -----------------------------------------------------------------

            if (
                current_mae
                < float(
                    best_mae.numpy()
                )
            ):

                best_mae.assign(
                    current_mae
                )

                generator_model.save(
                    MODEL_DIR
                    / "best_generator_mae.keras"
                )

                critic_model.save(
                    MODEL_DIR
                    / "best_critic_mae.keras"
                )

                with (
                    MODEL_DIR
                    / "best_mae.txt"
                ).open(
                    "w"
                ) as file:

                    file.write(
                        f"{current_mae:.12e}\n"
                    )

                best_mae_manager.save(
                    checkpoint_number=current_iteration
                )

                print(
                    "NEW BEST MAE: "
                    f"{current_mae:.8f}"
                )

            # -----------------------------------------------------------------
            # Best by forward residual
            # -----------------------------------------------------------------

            if (
                current_residual
                < float(
                    best_residual.numpy()
                )
            ):

                best_residual.assign(
                    current_residual
                )

                generator_model.save(
                    MODEL_DIR
                    / "best_generator_residual.keras"
                )

                critic_model.save(
                    MODEL_DIR
                    / "best_critic_residual.keras"
                )

                with (
                    MODEL_DIR
                    / "best_residual.txt"
                ).open(
                    "w"
                ) as file:

                    file.write(
                        f"{current_residual:.12e}\n"
                    )

                best_residual_manager.save(
                    checkpoint_number=current_iteration
                )

                print(
                    "NEW BEST RESIDUAL: "
                    f"{current_residual:.8f}"
                )


# =============================================================================
# Main program
# =============================================================================

def main():
    """
    Run the complete inverse-learning pipeline.
    """

    configure_tensorflow()

    initialize_csv_files()

    print()
    print(
        "Inverse Chafee--Infante WGAN-GP"
    )

    print(
        "================================"
    )

    print(
        f"Grid:              {M} x {M}"
    )

    print(
        f"Domain:            "
        f"[-{L},{L}] x [-{L},{L}]"
    )

    print(
        f"Mesh size:         "
        f"{HX:.8e}"
    )

    print(
        f"Dataset time:      "
        f"T = {SIM_STEPS * DT:.6f}"
    )

    print(
        f"gamma:             "
        f"{GAMMA}"
    )

    print(
        f"kappa:             "
        f"{KAPPA}"
    )

    print(
        "Physics residual:  "
        "Forward Euler surrogate"
    )

    print()

    (
        dataset,
        test_data,
        image_shape,
        ne_scale,
        ic_scale,
    ) = load_dataset(
        TRAIN_NPZ,
        TEST_NPZ,
    )

    critic_model = (
        build_critic(
            image_shape
        )
    )

    generator_model = (
        build_generator(
            image_shape
        )
    )

    print()
    print(
        "Critic output shape:",
        critic_model.output_shape,
    )

    print()

    generator_model.summary()

    print()

    critic_model.summary()

    save_model_plots(
        generator_model,
        critic_model,
    )

    train(
        critic_model,
        generator_model,
        dataset,
        test_data,
        MAX_EPOCHS,
        ic_scale,
        ne_scale,
    )


if __name__ == "__main__":
    main()