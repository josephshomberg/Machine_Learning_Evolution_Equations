"""
MAE-Dominant WGAN-GP Baseline for the Inverse Chafee--Infante Problem

This script trains a simplified inverse model for the two-dimensional
Chafee--Infante equation.

The inverse-learning problem is

    input  : forward-evolved state u_T,
    output : reconstructed initial state u_0.

The generator is trained using only

    1. a Wasserstein adversarial loss,
    2. a mean absolute error (MAE) reconstruction loss.

Thus, unlike the full physics-informed model, this baseline does not include

    - Lyapunov-energy matching,
    - forward-consistency residual loss,
    - mean or variance penalties,
    - spatial-gradient penalties.

The purpose of the script is to provide a simpler baseline against which
the physics-informed WGAN-GP model can be compared.

Homogeneous Dirichlet boundary conditions are imposed directly in the
generator output.

Requirements
------------
numpy
tensorflow
"""

from pathlib import Path
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    Layer,
    LayerNormalization,
    LeakyReLU,
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# =============================================================================
# Configuration
# =============================================================================

TRAIN_NPZ = Path(
    "YOUR/PATH/TO/TRAINING-DATASET.npz"
)

TEST_NPZ = Path(
    "YOUR/PATH/TO/TESTING-DATASET.npz"
)


# -----------------------------------------------------------------------------
# Loss weights
# -----------------------------------------------------------------------------

LAMBDA_MAE = 25.0
LAMBDA_ADV = 2.0
LAMBDA_GP = 10.0


# -----------------------------------------------------------------------------
# Training parameters
# -----------------------------------------------------------------------------

BATCH_SIZE = 8
MAX_EPOCHS = 20

CRITIC_STEPS = 4

GENERATOR_LEARNING_RATE = 1.0e-4
CRITIC_LEARNING_RATE = 2.0e-4

VALIDATION_SAMPLES = 128


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path(
    "mae_wgangp_baseline"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Dataset utilities
# =============================================================================

def prepare_array(array):
    """
    Convert stored dataset arrays to shape (N,H,W,1).
    """

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    if array.ndim == 3:

        return array[
            ...,
            np.newaxis,
        ]

    if (
        array.ndim == 4
        and array.shape[-1] == 1
    ):

        return array

    if (
        array.ndim == 4
        and array.shape[1] == 1
    ):

        return np.transpose(
            array,
            (
                0,
                2,
                3,
                1,
            ),
        )

    raise ValueError(
        "Unsupported dataset shape: "
        f"{array.shape}"
    )


def load_dataset(
    train_path,
    test_path,
):
    """
    Load and scale training and testing datasets.

    Scaling constants are computed from the training data only.

    Returns
    -------
    train_data : tuple
        Scaled training (src, tar).
    test_data : tuple
        Scaled testing (src, tar).
    image_shape : tuple
        Shape of one sample.
    ne_scale : float
        Physical scale for forward states.
    ic_scale : float
        Physical scale for initial states.
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
            "Training source scale is zero."
        )

    if ic_scale <= 0.0:

        raise ValueError(
            "Training target scale is zero."
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

    print(
        "Training source shape:",
        train_A.shape,
    )

    print(
        "Training target shape:",
        train_B.shape,
    )

    print(
        "Testing source shape:",
        test_A.shape,
    )

    print(
        "Testing target shape:",
        test_B.shape,
    )

    print(
        f"Forward-state scale: "
        f"{ne_scale:.8e}"
    )

    print(
        f"Initial-state scale: "
        f"{ic_scale:.8e}"
    )

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
    Set all boundary pixels equal to zero.
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

        middle = tf.pad(
            interior,
            paddings=[
                [0, 0],
                [0, 0],
                [1, 1],
                [0, 0],
            ],
        )

        output = tf.pad(
            middle,
            paddings=[
                [0, 0],
                [1, 1],
                [0, 0],
                [0, 0],
            ],
        )

        return tf.ensure_shape(
            output,
            inputs.shape,
        )

    def get_config(
        self,
    ):

        return super().get_config()


# =============================================================================
# Generator
# =============================================================================

def build_generator(
    image_shape,
):
    """
    Build the simplified U-Net-style inverse generator.
    """

    inputs = Input(
        shape=image_shape,
        name="forward_state",
    )

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------

    e1 = Conv2D(
        64,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
    )(
        inputs
    )

    e1 = LeakyReLU(
        negative_slope=0.2
    )(
        e1
    )

    e2 = Conv2D(
        128,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(
        e1
    )

    e2 = LayerNormalization()(
        e2
    )

    e2 = LeakyReLU(
        negative_slope=0.2
    )(
        e2
    )

    # -------------------------------------------------------------------------
    # Bottleneck
    # -------------------------------------------------------------------------

    bottleneck = Conv2D(
        256,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
    )(
        e2
    )

    bottleneck = Activation(
        "relu"
    )(
        bottleneck
    )

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------

    d1 = Conv2DTranspose(
        128,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
    )(
        bottleneck
    )

    d1 = Concatenate()(
        [
            d1,
            e2,
        ]
    )

    d2 = Conv2DTranspose(
        64,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
    )(
        d1
    )

    d2 = Concatenate()(
        [
            d2,
            e1,
        ]
    )

    output = Conv2DTranspose(
        1,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
    )(
        d2
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
        name="mae_wgangp_generator",
    )


# =============================================================================
# Critic
# =============================================================================

def build_critic(
    image_shape,
):
    """
    Build the conditional PatchGAN critic.
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

    x = Conv2D(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
    )(
        x
    )

    x = LeakyReLU(
        negative_slope=0.2
    )(
        x
    )

    x = Conv2D(
        128,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer="he_uniform",
    )(
        x
    )

    x = LeakyReLU(
        negative_slope=0.2
    )(
        x
    )

    output = Conv2D(
        1,
        kernel_size=4,
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
        name="mae_wgangp_critic",
    )


# =============================================================================
# WGAN-GP gradient penalty
# =============================================================================

def gradient_penalty(
    critic_model,
    real_A,
    real_B,
    fake_B,
):
    """
    Compute the WGAN-GP interpolation penalty.
    """

    batch_size = tf.shape(
        real_B
    )[0]

    alpha = tf.random.uniform(
        [
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
        alpha * real_B
        +
        (
            1.0
            - alpha
        )
        * fake_B
    )

    with tf.GradientTape() as tape:

        tape.watch(
            interpolated
        )

        prediction = critic_model(
            [
                real_A,
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
            "Gradient penalty calculation failed."
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
):
    """
    Perform critic updates followed by one generator update.

    Generator loss:

        L_G
        =
        lambda_adv L_adv
        +
        lambda_MAE L_MAE.
    """

    real_A = tf.convert_to_tensor(
        real_A,
        dtype=tf.float32,
    )

    real_B = tf.convert_to_tensor(
        real_B,
        dtype=tf.float32,
    )

    critic_loss = tf.constant(
        0.0,
        dtype=tf.float32,
    )

    # -------------------------------------------------------------------------
    # Critic updates
    # -------------------------------------------------------------------------

    for _ in range(
        CRITIC_STEPS
    ):

        fake_B = generator_model(
            real_A,
            training=True,
        )

        # The critic update does not require generator gradients.
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
                real_B
                - fake_B
            )
        )

        generator_loss = (
            LAMBDA_ADV
            * adversarial_loss
            +
            LAMBDA_MAE
            * mae_loss
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
        mae_loss,
    )


# =============================================================================
# Fixed validation subset
# =============================================================================

def create_validation_subset(
    test_data,
):
    """
    Construct a reproducible validation subset.
    """

    test_A, test_B = test_data

    count = min(
        VALIDATION_SAMPLES,
        test_A.shape[0],
    )

    rng = np.random.default_rng(
        SEED + 1
    )

    indices = np.sort(
        rng.choice(
            test_A.shape[0],
            size=count,
            replace=False,
        )
    )

    return (
        test_A[
            indices
        ],
        test_B[
            indices
        ],
    )


# =============================================================================
# Validation
# =============================================================================

def validation_mae(
    generator_model,
    validation_data,
):
    """
    Compute mean MAE on the fixed validation subset.
    """

    val_A, val_B = (
        validation_data
    )

    prediction = (
        generator_model.predict(
            val_A,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
    )

    return float(
        np.mean(
            np.abs(
                prediction
                - val_B
            )
        )
    )


# =============================================================================
# Training loop
# =============================================================================

def train(
    train_data,
    test_data,
    image_shape,
):
    """
    Train the simplified MAE-dominant WGAN-GP baseline.
    """

    train_A, train_B = (
        train_data
    )

    generator_model = (
        build_generator(
            image_shape
        )
    )

    critic_model = (
        build_critic(
            image_shape
        )
    )

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

    validation_data = (
        create_validation_subset(
            test_data
        )
    )

    best_validation_mae = np.inf

    batches_per_epoch = (
        train_A.shape[0]
        // BATCH_SIZE
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_mae = []

        epoch_generator_loss = []

        epoch_critic_loss = []

        for _ in range(
            batches_per_epoch
        ):

            indices = np.random.randint(
                0,
                train_A.shape[0],
                size=BATCH_SIZE,
            )

            real_A = train_A[
                indices
            ]

            real_B = train_B[
                indices
            ]

            (
                critic_loss,
                generator_loss,
                adversarial_loss,
                mae_loss,
            ) = train_step(
                real_A,
                real_B,
                generator_model,
                critic_model,
                generator_optimizer,
                critic_optimizer,
            )

            del adversarial_loss

            epoch_critic_loss.append(
                float(
                    critic_loss.numpy()
                )
            )

            epoch_generator_loss.append(
                float(
                    generator_loss.numpy()
                )
            )

            epoch_mae.append(
                float(
                    mae_loss.numpy()
                )
            )

        current_validation_mae = (
            validation_mae(
                generator_model,
                validation_data,
            )
        )

        print(
            f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
            f"critic = "
            f"{np.mean(epoch_critic_loss): .6f} | "
            f"generator = "
            f"{np.mean(epoch_generator_loss): .6f} | "
            f"train MAE = "
            f"{np.mean(epoch_mae):.6f} | "
            f"validation MAE = "
            f"{current_validation_mae:.6f}"
        )

        if (
            current_validation_mae
            < best_validation_mae
        ):

            best_validation_mae = (
                current_validation_mae
            )

            generator_model.save(
                MODEL_DIR
                / "best_generator_mae_baseline.keras"
            )

            critic_model.save(
                MODEL_DIR
                / "best_critic_mae_baseline.keras"
            )

            (
                MODEL_DIR
                / "best_validation_mae.txt"
            ).write_text(
                f"{best_validation_mae:.12e}\n"
            )

            print(
                "  New best validation MAE: "
                f"{best_validation_mae:.8f}"
            )

    return (
        generator_model,
        critic_model,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Run the MAE-dominant WGAN-GP baseline experiment.
    """

    print(
        "MAE-Dominant WGAN-GP Baseline"
    )

    print(
        "======================================"
    )

    (
        train_data,
        test_data,
        image_shape,
        ne_scale,
        ic_scale,
    ) = load_dataset(
        TRAIN_NPZ,
        TEST_NPZ,
    )

    print(
        f"Image shape: "
        f"{image_shape}"
    )

    print(
        f"NE scale: "
        f"{ne_scale:.8e}"
    )

    print(
        f"IC scale: "
        f"{ic_scale:.8e}"
    )

    train(
        train_data,
        test_data,
        image_shape,
    )


if __name__ == "__main__":
    main()