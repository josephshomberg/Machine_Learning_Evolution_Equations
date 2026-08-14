"""
Full-Test Evaluation of Trained Inverse Chafee--Infante Generators

This script evaluates one or two trained inverse generators for the
two-dimensional Chafee--Infante problem.

For each available model, it

    - evaluates the complete held-out test set,
    - computes per-sample reconstruction MAE,
    - computes a Forward Euler surrogate residual in physical units,
    - computes descriptive statistics,
    - optionally evaluates predeclared empirical success thresholds,
    - saves qualitative comparisons,
    - saves metric histograms and empirical CDFs,
    - saves raw per-sample metric arrays,
    - writes JSON and text reports.

The inverse problem is

    input  : forward-evolved state u_T,
    output : reconstructed initial state u_0.

Important numerical note
------------------------
The dataset may be generated with the Eyre-type semi-implicit solver,
whereas the residual used here is the same differentiable Forward Euler
surrogate used during training. It is therefore a forward-consistency
diagnostic rather than the exact residual of the dataset-generating scheme.

Important statistical note
--------------------------
If FIXED_EPS and FIXED_DELTA are used in a success criterion, they should
be specified independently of the held-out test set.

Requirements
------------
numpy
tensorflow
matplotlib
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer


# =============================================================================
# User configuration
# =============================================================================

TRAIN_NPZ = Path(
    "../../../../CIRDE_Datasets/"
    "128x128_kappa=4.7_DBC_Poisson_smooth_data_400iterations/"
    "train-dataset_128x128_DBC_kappa=4.7_PS400_count=50000.npz"
)

TEST_NPZ = Path(
    "../../../../CIRDE_Datasets/"
    "128x128_kappa=4.7_DBC_Poisson_smooth_data_400iterations/"
    "test-dataset_128x128_DBC_kappa=4.7_PS400_count=10000.npz"
)

BEST_MAE_MODEL_PATH = Path(
    "3_models/best_generator_mae.keras"
)

BEST_MAE_METRIC_PATH = Path(
    "3_models/best_mae.txt"
)

BEST_RESIDUAL_MODEL_PATH = Path(
    "3_models/best_generator_residual.keras"
)

BEST_RESIDUAL_METRIC_PATH = Path(
    "3_models/best_residual.txt"
)

OUTDIR = Path(
    "4_full_model_evaluation"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Evaluation controls
# =============================================================================

PREDICTION_BATCH_SIZE = 8
RESIDUAL_BATCH_SIZE = 32

QUALITATIVE_SAMPLES = (
    0,
    1,
    2,
)

# Predeclared thresholds for empirical success criteria.
FIXED_EPS = None
FIXED_DELTA = None

TARGET_P_JOINT = 0.90


# =============================================================================
# PDE parameters
# =============================================================================

SIM_STEPS = 400

DT = 0.001

GAMMA = 0.005
KAPPA = 4.7

L = 1.0
M = 128

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
        "metric_name": "validation_mae",
    },
    {
        "name": "best_residual",
        "model_path": BEST_RESIDUAL_MODEL_PATH,
        "metric_path": BEST_RESIDUAL_METRIC_PATH,
        "metric_name": "validation_residual",
    },
]


# =============================================================================
# Dataset helpers
# =============================================================================

def prepare_array(
    array,
):
    """
    Convert stored data to shape (N,M,M,1).
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
            f"Unsupported array shape: "
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


def load_dataset(
    path,
):
    """
    Load src and tar arrays from one NPZ dataset.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{path}"
        )

    with np.load(
        path
    ) as data:

        if (
            "src" not in data
            or "tar" not in data
        ):

            raise KeyError(
                "Dataset must contain "
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


def compute_scales(
    train_src,
    train_tar,
):
    """
    Compute scaling constants from the training data only.
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
            "NE scale is zero."
        )

    if ic_scale <= 0.0:
        raise ValueError(
            "IC scale is zero."
        )

    return (
        ne_scale,
        ic_scale,
    )


# =============================================================================
# Model IO
# =============================================================================

def load_generator(
    path,
):
    """
    Load one saved generator.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Model not found: "
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


def read_scalar(
    path,
):
    """
    Read one saved scalar metric.
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


# =============================================================================
# Statistics
# =============================================================================

def metric_summary(
    values,
):
    """
    Compute descriptive statistics.
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

        raise ValueError(
            "Metric array is empty."
        )

    variance = (
        float(
            np.var(
                values,
                ddof=1,
            )
        )
        if n > 1
        else 0.0
    )

    return {
        "N": int(n),

        "mean": float(
            np.mean(
                values
            )
        ),

        "std": float(
            np.sqrt(
                variance
            )
        ),

        "var": variance,

        "min": float(
            np.min(
                values
            )
        ),

        "median": float(
            np.median(
                values
            )
        ),

        "max": float(
            np.max(
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


def wilson_interval(
    successes,
    total,
    z=1.96,
):
    """
    Compute a Wilson confidence interval.
    """

    if total <= 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    p = (
        successes
        / total
    )

    z2 = z**2

    denominator = (
        1.0
        + z2 / total
    )

    center = (
        p
        + z2 / (2.0 * total)
    ) / denominator

    half_width = (
        z
        / denominator
        * np.sqrt(
            p
            * (
                1.0 - p
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

    return (
        p,
        max(
            0.0,
            center - half_width,
        ),
        min(
            1.0,
            center + half_width,
        ),
    )


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

    successes = int(
        np.sum(
            values
            <= threshold
        )
    )

    total = values.size

    (
        probability,
        lower,
        upper,
    ) = wilson_interval(
        successes,
        total,
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "probability":
            float(
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

        "successes":
            successes,

        "N":
            int(
                total
            ),
    }


def joint_probability(
    mae,
    residual,
    epsilon,
    delta,
):
    """
    Compute the empirical joint success probability.
    """

    event = (
        (
            mae
            <= epsilon
        )
        &
        (
            residual
            <= delta
        )
    )

    successes = int(
        np.sum(
            event
        )
    )

    total = len(
        mae
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
        "probability":
            float(
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

        "successes":
            successes,

        "N":
            total,
    }


# =============================================================================
# Forward Euler surrogate residual
# =============================================================================

@tf.function
def forward_euler_surrogate(
    u0_physical,
):
    """
    Evolve physical initial states by Forward Euler.
    """

    u = u0_physical[
        :,
        1:-1,
        1:-1,
        0,
    ]

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
    prediction_scaled,
    source_scaled,
    ic_scale,
    ne_scale,
):
    """
    Compute normalized Forward Euler residual per sample.

    The predicted IC and observed forward state are converted back to
    physical units before the PDE simulation is performed.
    """

    total = (
        prediction_scaled.shape[0]
    )

    residuals = []

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

        prediction_physical = (
            tf.convert_to_tensor(
                prediction_scaled[
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

        source_physical = (
            tf.convert_to_tensor(
                source_scaled[
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
                prediction_physical
            )
        )

        observed = source_physical[
            :,
            1:-1,
            1:-1,
            0,
        ]

        physical_residual = (
            tf.reduce_mean(
                tf.abs(
                    simulated
                    - observed
                ),
                axis=[
                    1,
                    2,
                ],
            )
        )

        scaled_residual = (
            physical_residual
            /
            tf.cast(
                ne_scale,
                tf.float32,
            )
        )

        residuals.append(
            scaled_residual.numpy()
        )

    return np.concatenate(
        residuals
    ).astype(
        np.float64
    )


# =============================================================================
# Plots
# =============================================================================

def save_histogram_and_cdf(
    values,
    name,
    outdir,
):
    """
    Save histogram and empirical CDF.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.hist(
        values,
        bins=60,
    )

    ax.set_xlabel(
        name
    )

    ax.set_ylabel(
        "Count"
    )

    ax.set_title(
        f"{name} Histogram"
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
            len(
                sorted_values
            )
            + 1,
        )
        /
        len(
            sorted_values
        )
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.plot(
        sorted_values,
        cdf,
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

    ax.set_title(
        f"{name} Empirical CDF"
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


def save_qualitative_plot(
    source,
    prediction,
    target,
    sample_indices,
    model_name,
    outfile,
):
    """
    Save source / reconstruction / target comparison.
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

    column_titles = [
        r"Forward state $u_T$",
        r"Reconstructed $u_0$",
        r"True $u_0$",
    ]

    for column, title in enumerate(
        column_titles
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
                cmap="magma",
                interpolation="none",
            )

            axes[
                row,
                column,
            ].axis(
                "off"
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
# Evaluate one model
# =============================================================================

def evaluate_model(
    spec,
    test_source,
    test_target,
    ne_scale,
    ic_scale,
):
    """
    Evaluate one trained generator.
    """

    name = spec[
        "name"
    ]

    model_path = spec[
        "model_path"
    ]

    metric_path = spec[
        "metric_path"
    ]

    model_outdir = (
        OUTDIR
        / name
    )

    model_outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_generator(
        model_path
    )

    saved_metric = read_scalar(
        metric_path
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
            "Prediction shape mismatch."
        )

    # -------------------------------------------------------------------------
    # MAE
    # -------------------------------------------------------------------------

    mae_scaled = np.mean(
        np.abs(
            prediction
            - test_target
        ),
        axis=(
            1,
            2,
            3,
        ),
    ).astype(
        np.float64
    )

    mae_physical = (
        ic_scale
        * mae_scaled
    )

    # -------------------------------------------------------------------------
    # Residual
    # -------------------------------------------------------------------------

    residual_scaled = (
        residual_per_sample(
            prediction,
            test_source,
            ic_scale,
            ne_scale,
        )
    )

    # -------------------------------------------------------------------------
    # Summaries
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

    residual_summary = (
        metric_summary(
            residual_scaled
        )
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    save_qualitative_plot(
        test_source,
        prediction,
        test_target,
        QUALITATIVE_SAMPLES,
        name,
        model_outdir
        / "qualitative_comparison.png",
    )

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

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report = {
        "model_name":
            name,

        "model_path":
            str(
                model_path
            ),

        "saved_training_metric":
            saved_metric,

        "scales": {
            "NE_SCALE":
                ne_scale,

            "IC_SCALE":
                ic_scale,
        },

        "PDE": {
            "DT":
                DT,

            "GAMMA":
                GAMMA,

            "KAPPA":
                KAPPA,

            "L":
                L,

            "M":
                M,

            "HX":
                HX,

            "SIM_STEPS":
                SIM_STEPS,
        },

        "metrics": {
            "mae_scaled":
                mae_scaled_summary,

            "mae_physical":
                mae_physical_summary,

            "residual_scaled":
                residual_summary,
        },
    }

    # -------------------------------------------------------------------------
    # Optional empirical criterion
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

        joint = joint_probability(
            mae_scaled,
            residual_scaled,
            FIXED_EPS,
            FIXED_DELTA,
        )

        report[
            "P_joint"
        ] = joint

        report[
            "criterion_satisfied"
        ] = bool(
            joint[
                "probability"
            ]
            >= TARGET_P_JOINT
        )

        report[
            "wilson_lower_bound_satisfies"
        ] = bool(
            joint[
                "wilson95"
            ][0]
            >= TARGET_P_JOINT
        )

    # -------------------------------------------------------------------------
    # Save arrays
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

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------

    with (
        model_outdir
        / "report.json"
    ).open(
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Save text report
    # -------------------------------------------------------------------------

    lines = [
        (
            "=== Inverse Chafee--Infante "
            "Test Report ==="
        ),
        "",
        f"Model: {name}",
        (
            "Saved validation metric: "
            f"{saved_metric}"
        ),
        "",
        (
            "Scaled MAE mean +/- std: "
            f"{mae_scaled_summary['mean']:.8f} +/- "
            f"{mae_scaled_summary['std']:.8f}"
        ),
        (
            "Physical MAE mean +/- std: "
            f"{mae_physical_summary['mean']:.8f} +/- "
            f"{mae_physical_summary['std']:.8f}"
        ),
        (
            "Scaled residual mean +/- std: "
            f"{residual_summary['mean']:.8f} +/- "
            f"{residual_summary['std']:.8f}"
        ),
        "",
    ]

    if (
        FIXED_EPS is not None
        and FIXED_DELTA is not None
    ):

        joint = report[
            "P_joint"
        ]

        lines.extend(
            [
                (
                    "Empirical joint criterion:"
                ),
                (
                    f"  epsilon = "
                    f"{FIXED_EPS}"
                ),
                (
                    f"  delta   = "
                    f"{FIXED_DELTA}"
                ),
                (
                    "  P(MAE <= epsilon and "
                    "Residual <= delta) = "
                    f"{joint['probability']:.6f}"
                ),
                (
                    "  Wilson 95% interval = "
                    f"{joint['wilson95']}"
                ),
                (
                    "  empirical criterion satisfied = "
                    f"{report['criterion_satisfied']}"
                ),
                (
                    "  Wilson lower bound satisfies = "
                    f"{report['wilson_lower_bound_satisfies']}"
                ),
                "",
                (
                    r"\noindent On the held-out test set, "
                    rf"$\Pr(\mathrm{{MAE}}\le {FIXED_EPS:.6g},\,"
                    rf"\mathrm{{Res}}\le {FIXED_DELTA:.6g})"
                    rf"\approx {joint['probability']:.4f}$."
                ),
            ]
        )

    (
        model_outdir
        / "report.txt"
    ).write_text(
        "\n".join(
            lines
        )
    )

    print()
    print(
        f"[{name}]"
    )

    print(
        "Scaled MAE mean:",
        mae_scaled_summary[
            "mean"
        ],
    )

    print(
        "Scaled residual mean:",
        residual_summary[
            "mean"
        ],
    )

    return report


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Evaluate all available trained generators.
    """

    print(
        "Inverse Chafee--Infante "
        "test-set evaluation"
    )

    print(
        "========================================"
    )

    # -------------------------------------------------------------------------
    # Training dataset only determines scaling constants.
    # -------------------------------------------------------------------------

    train_src, train_tar = (
        load_dataset(
            TRAIN_NPZ
        )
    )

    (
        ne_scale,
        ic_scale,
    ) = compute_scales(
        train_src,
        train_tar,
    )

    del train_src
    del train_tar

    # -------------------------------------------------------------------------
    # Held-out test dataset
    # -------------------------------------------------------------------------

    test_src_raw, test_tar_raw = (
        load_dataset(
            TEST_NPZ
        )
    )

    test_source = (
        test_src_raw
        / ne_scale
    ).astype(
        np.float32
    )

    test_target = (
        test_tar_raw
        / ic_scale
    ).astype(
        np.float32
    )

    del test_src_raw
    del test_tar_raw

    print(
        f"Test samples: "
        f"{test_source.shape[0]}"
    )

    print(
        f"NE scale: "
        f"{ne_scale:.8e}"
    )

    print(
        f"IC scale: "
        f"{ic_scale:.8e}"
    )

    reports = {}

    for spec in MODEL_SPECS:

        model_path = spec[
            "model_path"
        ]

        if not model_path.exists():

            print(
                "Skipping missing model: "
                f"{model_path}"
            )

            continue

        reports[
            spec["name"]
        ] = evaluate_model(
            spec,
            test_source,
            test_target,
            ne_scale,
            ic_scale,
        )

    if not reports:

        raise FileNotFoundError(
            "No trained generator models were found."
        )

    comparison_path = (
        OUTDIR
        / "comparison_summary.json"
    )

    with comparison_path.open(
        "w"
    ) as file:

        json.dump(
            reports,
            file,
            indent=2,
        )

    print()
    print(
        "Testing complete."
    )

    print(
        f"Comparison report: "
        f"{comparison_path}"
    )


if __name__ == "__main__":
    main()