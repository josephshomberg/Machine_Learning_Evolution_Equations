"""
Full-Test Evaluation of Two Inverse Chafee--Infante Generators

This script evaluates two trained inverse models for the two-dimensional
Chafee--Infante equation:

    1. the generator selected by validation MAE,
    2. the generator selected by validation forward residual.

The inverse problem is

    input  : forward-evolved state u_T,
    output : reconstructed initial state u_0.

For each model, the script

    - evaluates the complete held-out test set,
    - computes per-sample reconstruction MAE,
    - computes a differentiable Forward Euler surrogate residual,
    - computes descriptive statistics,
    - computes empirical probabilities for predeclared thresholds,
    - computes Wilson confidence intervals,
    - computes joint and conditional probabilities,
    - saves qualitative figures,
    - saves empirical histograms and CDFs,
    - saves raw metric arrays,
    - writes JSON and text reports.

Important numerical note
------------------------
The canonical dataset may be generated using the Eyre-type semi-implicit
scheme. The residual used here is the same Forward Euler surrogate used
during neural-network training. It should therefore be interpreted as a
forward-consistency diagnostic rather than as the exact residual of the
dataset-generating numerical scheme.

Important statistical note
--------------------------
The thresholds epsilon and delta used in a machine-learned solution
criterion should be specified independently of the held-out test set.

If FIXED_EPS and FIXED_DELTA are supplied below, the script evaluates

    P(MAE <= epsilon),
    P(Residual <= delta),

and the corresponding joint event directly on the complete held-out test
set.

The script deliberately does NOT select epsilon and delta from quantiles
of the same test set on which those probabilities are reported.

Requirements
------------
numpy
tensorflow
matplotlib
"""

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer


# =============================================================================
# User configuration
# =============================================================================

TRAIN_NPZ = Path(
    "YOUR/PATH/TO/TRAINING-DATASET.npz"
)

TEST_NPZ = Path(
    "YOUR/PATH/TO/TESTING-DATASET.npz"
)

BEST_MAE_MODEL_PATH = Path(
    "training_output/models/best_generator_mae.keras"
)

BEST_MAE_METRIC_PATH = Path(
    "training_output/models/best_mae.txt"
)

BEST_RESIDUAL_MODEL_PATH = Path(
    "training_output/models/best_generator_residual.keras"
)

BEST_RESIDUAL_METRIC_PATH = Path(
    "training_output/models/best_residual.txt"
)


# =============================================================================
# Thresholds for the empirical solution criterion
# =============================================================================

# These thresholds must be specified independently of the held-out test set
# if they are to be used in an inferential success criterion.
#
# Example:
#
# FIXED_EPS = 0.10
# FIXED_DELTA = 0.05
#
# Leave either value as None if only descriptive statistics are desired.

FIXED_EPS = None
FIXED_DELTA = None

# Desired lower bound for the joint empirical success criterion.
TARGET_P_JOINT = 0.90


# =============================================================================
# Evaluation parameters
# =============================================================================

PREDICTION_BATCH_SIZE = 8
RESIDUAL_BATCH_SIZE = 32

QUALITATIVE_SAMPLES = (
    0,
    1,
    2,
)

OUTDIR = Path(
    "full_model_evaluation"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# PDE parameters
# =============================================================================

SIM_STEPS = 400

DT = 0.001

GAMMA = 0.005
KAPPA = 4.7

L = 1.0
M = 128

# M grid points include both endpoints.
HX = 2.0 * L / (M - 1)
HY = HX


# =============================================================================
# Saved-model custom layer
# =============================================================================

@tf.keras.utils.register_keras_serializable(
    package="MLEE"
)
class EnforceDirichletBoundary(Layer):
    """
    Enforce homogeneous Dirichlet boundary conditions.

    This class matches the custom layer used by the canonical generator
    training script and is included here so that saved Keras models can be
    deserialized independently.
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

        return tf.ensure_shape(
            output,
            inputs.shape,
        )

    def get_config(
        self,
    ):

        return super().get_config()


# =============================================================================
# Model specifications
# =============================================================================

MODEL_SPECS = [
    {
        "name": "best_mae",
        "model_path": BEST_MAE_MODEL_PATH,
        "metric_path": BEST_MAE_METRIC_PATH,
        "training_metric_name": (
            "validation_mae_saved_during_training"
        ),
    },
    {
        "name": "best_residual",
        "model_path": BEST_RESIDUAL_MODEL_PATH,
        "metric_path": BEST_RESIDUAL_METRIC_PATH,
        "training_metric_name": (
            "validation_residual_saved_during_training"
        ),
    },
]


# =============================================================================
# Dataset utilities
# =============================================================================

def prepare_array(
    array,
):
    """
    Convert a stored dataset array to shape (N,M,M,1).
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
            "Expected sample shape "
            f"({M},{M},1); "
            f"received {array.shape[1:]}."
        )

    return array


def load_dataset_arrays(
    path,
):
    """
    Load src and tar arrays from one NPZ dataset.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with np.load(
        path
    ) as data:

        if (
            "src" not in data
            or "tar" not in data
        ):

            raise KeyError(
                f"{path} must contain "
                "'src' and 'tar'."
            )

        src = prepare_array(
            data["src"]
        )

        tar = prepare_array(
            data["tar"]
        )

    return (
        src,
        tar,
    )


# =============================================================================
# Scaling
# =============================================================================

def compute_training_scales(
    train_src,
    train_tar,
):
    """
    Compute scaling constants from the training dataset only.
    """

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

    return (
        ne_scale,
        ic_scale,
    )


# =============================================================================
# IO utilities
# =============================================================================

def read_scalar_text(
    path,
):
    """
    Read one floating-point value from a text file.
    """

    if (
        path is None
        or not path.exists()
    ):

        return None

    try:

        return float(
            path.read_text().strip()
        )

    except (
        OSError,
        ValueError,
    ):

        return None


def load_generator_model(
    path,
):
    """
    Load one saved generator.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Generator model not found: "
            f"{path}"
        )

    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "EnforceDirichletBoundary":
                EnforceDirichletBoundary,
        },
    )


# =============================================================================
# Basic statistics
# =============================================================================

def metric_summary(
    values,
):
    """
    Compute descriptive statistics for one metric.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    ).ravel()

    values = values[
        np.isfinite(
            values
        )
    ]

    n = values.size

    if n == 0:

        return {
            "N": 0,
            "mean": None,
            "std": None,
            "var": None,
            "min": None,
            "max": None,
            "median": None,
            "q05": None,
            "q95": None,
        }

    variance = (
        float(
            values.var(
                ddof=1
            )
        )
        if n > 1
        else 0.0
    )

    return {
        "N": int(n),

        "mean": float(
            values.mean()
        ),

        "std": float(
            np.sqrt(
                variance
            )
        ),

        "var": variance,

        "min": float(
            values.min()
        ),

        "max": float(
            values.max()
        ),

        "median": float(
            np.median(
                values
            )
        ),

        "q05": float(
            np.quantile(
                values,
                0.05,
            )
        ),

        "q95": float(
            np.quantile(
                values,
                0.95,
            )
        ),
    }


# =============================================================================
# Wilson confidence interval
# =============================================================================

def wilson_interval(
    successes,
    total,
    z=1.96,
):
    """
    Compute the Wilson confidence interval for a binomial proportion.
    """

    if total <= 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    proportion = (
        successes
        / total
    )

    z2 = z**2

    denominator = (
        1.0
        + z2 / total
    )

    center = (
        proportion
        + z2 / (2.0 * total)
    ) / denominator

    half_width = (
        z
        / denominator
        * np.sqrt(
            proportion
            * (
                1.0
                - proportion
            )
            / total
            +
            z2
            / (
                4.0
                * total**2
            )
        )
    )

    lower = max(
        0.0,
        center - half_width,
    )

    upper = min(
        1.0,
        center + half_width,
    )

    return (
        proportion,
        lower,
        upper,
    )


# =============================================================================
# Probability calculations
# =============================================================================

def probability_leq(
    values,
    threshold,
):
    """
    Estimate P(metric <= threshold).
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    ).ravel()

    total = values.size

    successes = int(
        np.sum(
            values
            <= threshold
        )
    )

    (
        probability,
        lower,
        upper,
    ) = wilson_interval(
        successes,
        total,
    )

    return {
        "threshold": float(
            threshold
        ),

        "successes": successes,

        "N": int(
            total
        ),

        "probability": float(
            probability
        ),

        "wilson95": [
            float(
                lower
            ),
            float(
                upper
            ),
        ],
    }


def joint_probabilities(
    mae,
    residual,
    epsilon,
    delta,
):
    """
    Compute marginal, joint, and conditional empirical probabilities.
    """

    mae = np.asarray(
        mae,
        dtype=np.float64,
    ).ravel()

    residual = np.asarray(
        residual,
        dtype=np.float64,
    ).ravel()

    if mae.size != residual.size:

        raise ValueError(
            "MAE and residual arrays must have equal length."
        )

    event_mae = (
        mae
        <= epsilon
    )

    event_residual = (
        residual
        <= delta
    )

    joint_event = (
        event_mae
        & event_residual
    )

    total = mae.size

    count_mae = int(
        np.sum(
            event_mae
        )
    )

    count_residual = int(
        np.sum(
            event_residual
        )
    )

    count_joint = int(
        np.sum(
            joint_event
        )
    )

    p_mae = wilson_interval(
        count_mae,
        total,
    )

    p_residual = wilson_interval(
        count_residual,
        total,
    )

    p_joint = wilson_interval(
        count_joint,
        total,
    )

    conditional_res_given_mae = (
        count_joint / count_mae
        if count_mae > 0
        else np.nan
    )

    conditional_mae_given_res = (
        count_joint / count_residual
        if count_residual > 0
        else np.nan
    )

    correlation = (
        float(
            np.corrcoef(
                mae,
                residual,
            )[0, 1]
        )
        if total > 1
        else np.nan
    )

    return {
        "P_MAE_leq_epsilon": {
            "probability": float(
                p_mae[0]
            ),
            "wilson95": [
                float(
                    p_mae[1]
                ),
                float(
                    p_mae[2]
                ),
            ],
            "successes": count_mae,
            "N": total,
        },

        "P_RES_leq_delta": {
            "probability": float(
                p_residual[0]
            ),
            "wilson95": [
                float(
                    p_residual[1]
                ),
                float(
                    p_residual[2]
                ),
            ],
            "successes": count_residual,
            "N": total,
        },

        "P_joint": {
            "probability": float(
                p_joint[0]
            ),
            "wilson95": [
                float(
                    p_joint[1]
                ),
                float(
                    p_joint[2]
                ),
            ],
            "successes": count_joint,
            "N": total,
        },

        "P_RES_given_MAE": float(
            conditional_res_given_mae
        ),

        "P_MAE_given_RES": float(
            conditional_mae_given_res
        ),

        "corr_MAE_RES": float(
            correlation
        ),
    }


# =============================================================================
# Forward Euler residual surrogate
# =============================================================================

@tf.function
def forward_euler_surrogate(
    u0_physical,
):
    """
    Evolve physical initial data with the Forward Euler surrogate.

    Parameters
    ----------
    u0_physical : tf.Tensor
        Shape (B,M,M,1).

    Returns
    -------
    tf.Tensor
        Physical interior state after SIM_STEPS steps.
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
        SIM_STEPS
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


def residual_per_sample(
    predicted_scaled,
    observed_scaled,
    ic_scale,
    ne_scale,
):
    """
    Compute the Forward Euler surrogate residual for each sample.

    The generator output and observed state are first converted from scaled
    values back to physical units.

    Returns
    -------
    residual_scaled : ndarray
        Physical residual divided by the physical NE scaling factor.
    residual_physical : ndarray
        Physical mean absolute forward residual.
    """

    predicted_scaled = np.asarray(
        predicted_scaled,
        dtype=np.float32,
    )

    observed_scaled = np.asarray(
        observed_scaled,
        dtype=np.float32,
    )

    total = predicted_scaled.shape[0]

    scaled_results = []
    physical_results = []

    for start in range(
        0,
        total,
        RESIDUAL_BATCH_SIZE,
    ):

        stop = min(
            start
            + RESIDUAL_BATCH_SIZE,
            total,
        )

        predicted_physical = (
            tf.convert_to_tensor(
                predicted_scaled[
                    start:stop
                ],
                dtype=tf.float32,
            )
            *
            tf.cast(
                ic_scale,
                tf.float32,
            )
        )

        observed_physical = (
            tf.convert_to_tensor(
                observed_scaled[
                    start:stop
                ],
                dtype=tf.float32,
            )
            *
            tf.cast(
                ne_scale,
                tf.float32,
            )
        )

        simulated = (
            forward_euler_surrogate(
                predicted_physical
            )
        )

        target = observed_physical[
            :,
            1:-1,
            1:-1,
            0,
        ]

        physical = tf.reduce_mean(
            tf.abs(
                simulated
                - target
            ),
            axis=[1, 2],
        )

        scaled = (
            physical
            /
            tf.cast(
                ne_scale,
                tf.float32,
            )
        )

        physical_results.append(
            physical.numpy()
        )

        scaled_results.append(
            scaled.numpy()
        )

    residual_scaled = np.concatenate(
        scaled_results
    ).astype(
        np.float64
    )

    residual_physical = np.concatenate(
        physical_results
    ).astype(
        np.float64
    )

    return (
        residual_scaled,
        residual_physical,
    )


# =============================================================================
# Plot utilities
# =============================================================================

def save_histogram_and_cdf(
    values,
    name,
    outdir,
):
    """
    Save histogram and empirical CDF plots.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    ).ravel()

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:

        return

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.hist(
        values,
        bins=60,
    )

    ax.set_title(
        f"{name} Histogram"
    )

    ax.set_xlabel(
        name
    )

    ax.set_ylabel(
        "Count"
    )

    fig.tight_layout()

    fig.savefig(
        outdir
        / f"{name.lower()}_hist.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    sorted_values = np.sort(
        values
    )

    cdf = (
        np.arange(
            1,
            sorted_values.size + 1,
        )
        / sorted_values.size
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.plot(
        sorted_values,
        cdf,
    )

    ax.set_title(
        f"{name} Empirical CDF"
    )

    ax.set_xlabel(
        name
    )

    ax.set_ylabel(
        "Empirical probability"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    fig.tight_layout()

    fig.savefig(
        outdir
        / f"{name.lower()}_cdf.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Qualitative comparison
# =============================================================================

def save_qualitative_plot(
    source,
    prediction,
    target,
    sample_indices,
    model_name,
    metric_name,
    metric_values,
    outfile,
):
    """
    Save source / reconstructed IC / true IC comparisons.
    """

    nrows = len(
        sample_indices
    )

    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(
            9,
            3.3 * nrows,
        ),
    )

    if nrows == 1:

        axes = np.expand_dims(
            axes,
            axis=0,
        )

    titles = [
        r"Forward state $u_T$",
        r"Reconstructed $u_0$",
        r"True $u_0$",
    ]

    for column, title in enumerate(
        titles
    ):

        axes[
            0,
            column,
        ].set_title(
            title
        )

    for row, sample_index in enumerate(
        sample_indices
    ):

        images = [
            source[
                sample_index,
                :,
                :,
                0,
            ],

            prediction[
                sample_index,
                :,
                :,
                0,
            ],

            target[
                sample_index,
                :,
                :,
                0,
            ],
        ]

        for column, image in enumerate(
            images
        ):

            axes[
                row,
                column,
            ].imshow(
                image,
                cmap="Greys",
                interpolation="none",
            )

            axes[
                row,
                column,
            ].axis(
                "off"
            )

        axes[
            row,
            1,
        ].set_ylabel(
            f"sample {sample_index}\n"
            f"{metric_name} = "
            f"{metric_values[row]:.5f}",
            fontsize=9,
        )

    fig.suptitle(
        model_name,
        fontsize=14,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Report formatting
# =============================================================================

def format_summary(
    name,
    summary,
):
    """
    Return human-readable summary lines.
    """

    return [
        f"[{name}]",

        (
            "  N                   = "
            f"{summary['N']}"
        ),

        (
            "  mean +/- std        = "
            f"{summary['mean']:.8f} +/- "
            f"{summary['std']:.8f}"
        ),

        (
            "  min / median / max  = "
            f"{summary['min']:.8f} / "
            f"{summary['median']:.8f} / "
            f"{summary['max']:.8f}"
        ),

        (
            "  q05 / q95           = "
            f"{summary['q05']:.8f} / "
            f"{summary['q95']:.8f}"
        ),

        "",
    ]


# =============================================================================
# Evaluate one model
# =============================================================================

def evaluate_one_model(
    specification,
    test_source,
    test_target,
    ne_scale,
    ic_scale,
):
    """
    Evaluate one generator over the complete held-out test set.
    """

    model_name = (
        specification[
            "name"
        ]
    )

    model_path = (
        specification[
            "model_path"
        ]
    )

    metric_path = (
        specification[
            "metric_path"
        ]
    )

    training_metric_name = (
        specification[
            "training_metric_name"
        ]
    )

    model_outdir = (
        OUTDIR
        / model_name
    )

    model_outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Evaluating model: "
        f"{model_name}"
    )

    print(
        "----------------------------------------"
    )

    model = load_generator_model(
        model_path
    )

    saved_training_metric = (
        read_scalar_text(
            metric_path
        )
    )

    prediction = model.predict(
        test_source,
        batch_size=PREDICTION_BATCH_SIZE,
        verbose=1,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    if prediction.shape != test_target.shape:

        raise ValueError(
            f"Prediction shape "
            f"{prediction.shape} "
            f"does not match target shape "
            f"{test_target.shape}."
        )

    # -------------------------------------------------------------------------
    # Reconstruction MAE
    # -------------------------------------------------------------------------

    mae_scaled = np.mean(
        np.abs(
            prediction
            - test_target
        ),
        axis=(1, 2, 3),
    ).astype(
        np.float64
    )

    mae_physical = (
        ic_scale
        * mae_scaled
    ).astype(
        np.float64
    )

    # -------------------------------------------------------------------------
    # Physics residual
    # -------------------------------------------------------------------------

    (
        residual_scaled,
        residual_physical,
    ) = residual_per_sample(
        prediction,
        test_source,
        ic_scale,
        ne_scale,
    )

    # -------------------------------------------------------------------------
    # Descriptive statistics
    # -------------------------------------------------------------------------

    mae_scaled_summary = (
        metric_summary(
            mae_scaled
        )
    )

    mae_physical_summary = (
        metric_summary(
            mae_physical
        )
    )

    residual_scaled_summary = (
        metric_summary(
            residual_scaled
        )
    )

    residual_physical_summary = (
        metric_summary(
            residual_physical
        )
    )

    # -------------------------------------------------------------------------
    # Qualitative examples
    # -------------------------------------------------------------------------

    sample_indices = np.asarray(
        QUALITATIVE_SAMPLES,
        dtype=int,
    )

    if (
        np.any(
            sample_indices < 0
        )
        or np.any(
            sample_indices
            >= test_source.shape[0]
        )
    ):

        raise IndexError(
            "QUALITATIVE_SAMPLES contains "
            "an invalid test index."
        )

    if model_name == "best_residual":

        panel_name = (
            "Residual"
        )

        panel_values = (
            residual_scaled[
                sample_indices
            ]
        )

    else:

        panel_name = (
            "MAE"
        )

        panel_values = (
            mae_scaled[
                sample_indices
            ]
        )

    save_qualitative_plot(
        test_source,
        prediction,
        test_target,
        sample_indices,
        model_name,
        panel_name,
        panel_values,
        model_outdir
        / "qualitative_comparison.png",
    )

    # -------------------------------------------------------------------------
    # Metric plots
    # -------------------------------------------------------------------------

    save_histogram_and_cdf(
        mae_scaled,
        "MAE_scaled",
        model_outdir,
    )

    save_histogram_and_cdf(
        mae_physical,
        "MAE_physical",
        model_outdir,
    )

    save_histogram_and_cdf(
        residual_scaled,
        "Residual_scaled",
        model_outdir,
    )

    save_histogram_and_cdf(
        residual_physical,
        "Residual_physical",
        model_outdir,
    )

    # -------------------------------------------------------------------------
    # Base report
    # -------------------------------------------------------------------------

    report = {
        "model_name": model_name,

        "configuration": {
            "train_dataset": str(
                TRAIN_NPZ
            ),

            "test_dataset": str(
                TEST_NPZ
            ),

            "model_path": str(
                model_path
            ),

            "SIM_STEPS": SIM_STEPS,
            "DT": DT,
            "GAMMA": GAMMA,
            "KAPPA": KAPPA,

            "L": L,
            "M": M,

            "HX": HX,
            "HY": HY,

            "NE_SCALE": ne_scale,
            "IC_SCALE": ic_scale,

            "FIXED_EPS": FIXED_EPS,
            "FIXED_DELTA": FIXED_DELTA,

            "TARGET_P_JOINT":
                TARGET_P_JOINT,

            "residual_definition": (
                "Differentiable Forward Euler "
                "surrogate evaluated in physical "
                "units and normalized by NE_SCALE."
            ),
        },

        "saved_training_metric": {
            "name":
                training_metric_name,

            "value":
                saved_training_metric,

            "note": (
                "This is the validation metric "
                "used during model selection. "
                "It is distinct from the "
                "full-test statistics reported here."
            ),
        },

        "full_test_metrics": {
            "mae_scaled":
                mae_scaled_summary,

            "mae_physical":
                mae_physical_summary,

            "residual_scaled":
                residual_scaled_summary,

            "residual_physical":
                residual_physical_summary,
        },
    }

    # -------------------------------------------------------------------------
    # Predeclared threshold probabilities
    # -------------------------------------------------------------------------

    if FIXED_EPS is not None:

        report[
            "P_MAE_leq_epsilon"
        ] = probability_leq(
            mae_scaled,
            FIXED_EPS,
        )

    if FIXED_DELTA is not None:

        report[
            "P_RES_leq_delta"
        ] = probability_leq(
            residual_scaled,
            FIXED_DELTA,
        )

    if (
        FIXED_EPS is not None
        and FIXED_DELTA is not None
    ):

        joint = joint_probabilities(
            mae_scaled,
            residual_scaled,
            FIXED_EPS,
            FIXED_DELTA,
        )

        report[
            "joint_probabilities"
        ] = joint

        empirical_joint = (
            joint[
                "P_joint"
            ][
                "probability"
            ]
        )

        lower_joint = (
            joint[
                "P_joint"
            ][
                "wilson95"
            ][0]
        )

        report[
            "machine_learned_solution_criterion"
        ] = {
            "criterion": (
                "P(MAE_scaled <= epsilon "
                "and Residual_scaled <= delta) "
                ">= X"
            ),

            "epsilon":
                FIXED_EPS,

            "delta":
                FIXED_DELTA,

            "X":
                TARGET_P_JOINT,

            "empirical_probability":
                empirical_joint,

            "wilson95":
                joint[
                    "P_joint"
                ][
                    "wilson95"
                ],

            "empirical_criterion_satisfied":
                bool(
                    empirical_joint
                    >= TARGET_P_JOINT
                ),

            "wilson_lower_bound_satisfies":
                bool(
                    lower_joint
                    >= TARGET_P_JOINT
                ),
        }

    # -------------------------------------------------------------------------
    # Save raw arrays
    # -------------------------------------------------------------------------

    np.save(
        model_outdir
        / "mae_scaled_per_sample.npy",
        mae_scaled,
    )

    np.save(
        model_outdir
        / "mae_physical_per_sample.npy",
        mae_physical,
    )

    np.save(
        model_outdir
        / "residual_scaled_per_sample.npy",
        residual_scaled,
    )

    np.save(
        model_outdir
        / "residual_physical_per_sample.npy",
        residual_physical,
    )

    # -------------------------------------------------------------------------
    # JSON report
    # -------------------------------------------------------------------------

    json_path = (
        model_outdir
        / "report.json"
    )

    with json_path.open(
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Human-readable report
    # -------------------------------------------------------------------------

    lines = [
        (
            "=== Inverse Chafee--Infante "
            "Full-Test Evaluation ==="
        ),
        "",
        f"Model: {model_name}",
        f"Model path: {model_path}",
        f"Training dataset: {TRAIN_NPZ}",
        f"Testing dataset: {TEST_NPZ}",
        "",
        "[Saved model-selection metric]",
        (
            f"  {training_metric_name} = "
            f"{saved_training_metric}"
        ),
        "",
    ]

    lines.extend(
        format_summary(
            "Scaled reconstruction MAE",
            mae_scaled_summary,
        )
    )

    lines.extend(
        format_summary(
            "Physical reconstruction MAE",
            mae_physical_summary,
        )
    )

    lines.extend(
        format_summary(
            "Scaled Forward Euler residual",
            residual_scaled_summary,
        )
    )

    lines.extend(
        format_summary(
            "Physical Forward Euler residual",
            residual_physical_summary,
        )
    )

    if FIXED_EPS is not None:

        probability = report[
            "P_MAE_leq_epsilon"
        ]

        lines.extend(
            [
                "[Predeclared MAE threshold]",
                (
                    f"  epsilon = "
                    f"{FIXED_EPS:.8f}"
                ),
                (
                    "  P(MAE_scaled <= epsilon) = "
                    f"{probability['probability']:.6f}"
                ),
                (
                    "  Wilson 95% interval = "
                    f"{probability['wilson95']}"
                ),
                "",
            ]
        )

    if FIXED_DELTA is not None:

        probability = report[
            "P_RES_leq_delta"
        ]

        lines.extend(
            [
                "[Predeclared residual threshold]",
                (
                    f"  delta = "
                    f"{FIXED_DELTA:.8f}"
                ),
                (
                    "  P(Residual_scaled <= delta) = "
                    f"{probability['probability']:.6f}"
                ),
                (
                    "  Wilson 95% interval = "
                    f"{probability['wilson95']}"
                ),
                "",
            ]
        )

    if (
        "joint_probabilities"
        in report
    ):

        joint = report[
            "joint_probabilities"
        ]

        criterion = report[
            "machine_learned_solution_criterion"
        ]

        lines.extend(
            [
                "[Joint empirical criterion]",

                (
                    "  P(MAE_scaled <= epsilon "
                    "and Residual_scaled <= delta) = "
                    f"{joint['P_joint']['probability']:.6f}"
                ),

                (
                    "  Wilson 95% interval = "
                    f"{joint['P_joint']['wilson95']}"
                ),

                (
                    "  Target X = "
                    f"{TARGET_P_JOINT:.6f}"
                ),

                (
                    "  Empirical criterion satisfied = "
                    f"{criterion['empirical_criterion_satisfied']}"
                ),

                (
                    "  Wilson lower bound >= X = "
                    f"{criterion['wilson_lower_bound_satisfies']}"
                ),

                "",

                "[Conditional probabilities]",

                (
                    "  P(Residual <= delta | "
                    "MAE <= epsilon) = "
                    f"{joint['P_RES_given_MAE']:.6f}"
                ),

                (
                    "  P(MAE <= epsilon | "
                    "Residual <= delta) = "
                    f"{joint['P_MAE_given_RES']:.6f}"
                ),

                (
                    "  corr(MAE, Residual) = "
                    f"{joint['corr_MAE_RES']:.6f}"
                ),

                "",

                "[LaTeX-ready sentence]",

                (
                    r"\noindent On the held-out test set, "
                    rf"$\Pr(\mathrm{{MAE}}\le "
                    rf"{FIXED_EPS:.6g})"
                    rf"\approx "
                    rf"{joint['P_MAE_leq_epsilon']['probability']:.4f}$, "
                    rf"$\Pr(\mathrm{{Res}}\le "
                    rf"{FIXED_DELTA:.6g})"
                    rf"\approx "
                    rf"{joint['P_RES_leq_delta']['probability']:.4f}$, "
                    rf"and "
                    rf"$\Pr(\mathrm{{MAE}}\le "
                    rf"{FIXED_EPS:.6g},\,"
                    rf"\mathrm{{Res}}\le "
                    rf"{FIXED_DELTA:.6g})"
                    rf"\approx "
                    rf"{joint['P_joint']['probability']:.4f}$."
                ),

                "",
            ]
        )

    else:

        lines.extend(
            [
                "[Probability criterion]",
                (
                    "  epsilon and/or delta were not "
                    "predeclared."
                ),
                (
                    "  Descriptive test statistics "
                    "were computed, but no empirical "
                    "success criterion was asserted."
                ),
                "",
            ]
        )

    text_path = (
        model_outdir
        / "report.txt"
    )

    text_path.write_text(
        "\n".join(
            lines
        )
    )

    print(
        f"Saved report: "
        f"{text_path}"
    )

    return report


# =============================================================================
# Comparison report
# =============================================================================

def save_comparison_report(
    reports,
):
    """
    Save compact comparison of all evaluated models.
    """

    json_path = (
        OUTDIR
        / "comparison_summary.json"
    )

    with json_path.open(
        "w"
    ) as file:

        json.dump(
            reports,
            file,
            indent=2,
        )

    lines = [
        (
            "=== Comparison of Inverse "
            "Chafee--Infante Generators ==="
        ),
        "",
    ]

    for model_name, report in reports.items():

        metrics = report[
            "full_test_metrics"
        ]

        mae = metrics[
            "mae_scaled"
        ]

        residual = metrics[
            "residual_scaled"
        ]

        lines.extend(
            [
                f"[{model_name}]",

                (
                    "  full-test scaled MAE "
                    "mean +/- std = "
                    f"{mae['mean']:.8f} +/- "
                    f"{mae['std']:.8f}"
                ),

                (
                    "  full-test scaled residual "
                    "mean +/- std = "
                    f"{residual['mean']:.8f} +/- "
                    f"{residual['std']:.8f}"
                ),
            ]
        )

        if (
            "machine_learned_solution_criterion"
            in report
        ):

            criterion = report[
                "machine_learned_solution_criterion"
            ]

            lines.append(
                "  joint success probability = "
                f"{criterion['empirical_probability']:.6f}"
            )

            lines.append(
                "  Wilson95 = "
                f"{criterion['wilson95']}"
            )

        lines.append(
            ""
        )

    text_path = (
        OUTDIR
        / "comparison_summary.txt"
    )

    text_path.write_text(
        "\n".join(
            lines
        )
    )

    print()
    print(
        f"Comparison saved to: "
        f"{text_path}"
    )


# =============================================================================
# Main program
# =============================================================================

def main():
    """
    Run full-test evaluation for all available inverse generators.
    """

    print(
        "Inverse Chafee--Infante "
        "full-test evaluation"
    )

    print(
        "========================================"
    )

    print(
        f"Grid:        {M} x {M}"
    )

    print(
        f"Mesh size:   {HX:.8e}"
    )

    print(
        f"T:           "
        f"{SIM_STEPS * DT:.6f}"
    )

    print(
        f"gamma:       {GAMMA}"
    )

    print(
        f"kappa:       {KAPPA}"
    )

    print()

    # -------------------------------------------------------------------------
    # Training data are used only to recover the scaling constants.
    # -------------------------------------------------------------------------

    train_source, train_target = (
        load_dataset_arrays(
            TRAIN_NPZ
        )
    )

    (
        ne_scale,
        ic_scale,
    ) = compute_training_scales(
        train_source,
        train_target,
    )

    del train_source
    del train_target

    print(
        f"NE scale:    {ne_scale:.8e}"
    )

    print(
        f"IC scale:    {ic_scale:.8e}"
    )

    # -------------------------------------------------------------------------
    # Held-out test set
    # -------------------------------------------------------------------------

    test_source_raw, test_target_raw = (
        load_dataset_arrays(
            TEST_NPZ
        )
    )

    test_source = (
        test_source_raw
        / ne_scale
    ).astype(
        np.float32
    )

    test_target = (
        test_target_raw
        / ic_scale
    ).astype(
        np.float32
    )

    del test_source_raw
    del test_target_raw

    print(
        f"Test samples: "
        f"{test_source.shape[0]}"
    )

    # -------------------------------------------------------------------------
    # Evaluate available models
    # -------------------------------------------------------------------------

    reports = {}

    for specification in MODEL_SPECS:

        model_path = (
            specification[
                "model_path"
            ]
        )

        if not model_path.exists():

            print()
            print(
                "Skipping missing model:"
            )

            print(
                f"  {model_path}"
            )

            continue

        name = specification[
            "name"
        ]

        reports[
            name
        ] = evaluate_one_model(
            specification,
            test_source,
            test_target,
            ne_scale,
            ic_scale,
        )

    if not reports:

        raise FileNotFoundError(
            "No generator models were available "
            "for evaluation."
        )

    save_comparison_report(
        reports
    )

    print()
    print(
        "Full-test evaluation complete."
    )


if __name__ == "__main__":
    main()