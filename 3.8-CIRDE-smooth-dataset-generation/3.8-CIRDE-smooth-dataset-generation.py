"""
Dataset Generation for the Backward Chafee--Infante Problem
with Homogeneous Dirichlet Boundary Conditions

This script generates paired data

    (src, tar) = (S_N(u_0), u_0),

where u_0 is a smooth random initial condition satisfying homogeneous
Dirichlet boundary conditions and S_N(u_0) is the numerical solution after
N Eyre-type semi-implicit time steps of the two-dimensional
Chafee--Infante equation

    u_t - gamma * Delta u + kappa * (u^3 - u) = 0

on

    Omega = [-L, L] x [-L, L].

The time discretization is

    u^{n+1}
    - delta_t * gamma * Delta_h u^{n+1}
    + delta_t * kappa * (u^{n+1})^3

    =
    (1 + delta_t * kappa) * u^n.

The nonlinear system at each time level is solved by Newton's method.

Smooth initial data are generated from the discrete Poisson problem

    -Delta_h u = f,

where f is a reproducible random source. The solution is normalized to
have prescribed maximum absolute amplitude.

The final dataset contains

    src : evolved states,
    tar : corresponding initial states,

with array shape

    (number_of_samples, M, M).

Intermediate batch files are written to disk so that the complete
dataset does not need to reside in memory during generation.

Requirements
------------
numpy
scipy
"""

from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import factorized, spsolve


# =============================================================================
# Dataset parameters
# =============================================================================

START = 0
END = 10000

BATCH_SIZE = 1000

# Number of Eyre time steps used to construct src = S_N(u_0).
ITERS_SRC = 400


# =============================================================================
# PDE and spatial-discretization parameters
# =============================================================================

DELTA_T = 0.001

L = 1.0
M = 128

GAMMA = 0.005
KAPPA = 4.7

INITIAL_AMPLITUDE = 0.02

# Base seed used to construct reproducible sample-specific seeds.
BASE_RANDOM_SEED = 42


# =============================================================================
# Newton parameters
# =============================================================================

NEWTON_MAX_ITERS = 12
NEWTON_TOL = 1.0e-10


# =============================================================================
# Output directories
# =============================================================================

DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "batches"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Output file names
# =============================================================================

DATASET_NAME = (
    f"CI_128x128_DBC_"
    f"kappa={KAPPA:g}_"
    f"steps={ITERS_SRC}"
)

FINAL_OUTFILE = (
    DATA_DIR
    / (
        f"{DATASET_NAME}_"
        f"count={END - START}.npz"
    )
)


# =============================================================================
# Grid
# =============================================================================

X = np.linspace(
    -L,
    L,
    M,
)

Y = np.linspace(
    -L,
    L,
    M,
)

HX = X[1] - X[0]
HY = Y[1] - Y[0]

if not np.isclose(HX, HY):
    raise ValueError(
        "This implementation assumes a square mesh with HX = HY."
    )

MI = M - 2
N_INTERIOR = MI * MI


# =============================================================================
# Discrete Dirichlet Laplacian
# =============================================================================

def build_dirichlet_laplacian_1d(
    n_points,
    mesh_size,
):
    """
    Construct the one-dimensional second-difference Laplacian.

    Homogeneous Dirichlet boundary values are imposed externally,
    so the matrix acts only on the interior unknowns.

    Parameters
    ----------
    n_points : int
        Number of interior grid points.
    mesh_size : float
        Spatial mesh width.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse one-dimensional Laplacian.
    """

    main_diagonal = -2.0 * np.ones(
        n_points,
    )

    off_diagonal = np.ones(
        n_points - 1,
    )

    laplacian = diags(
        [
            off_diagonal,
            main_diagonal,
            off_diagonal,
        ],
        offsets=[
            -1,
            0,
            1,
        ],
        shape=(
            n_points,
            n_points,
        ),
        format="csr",
    )

    return (
        laplacian
        / mesh_size**2
    )


def build_dirichlet_laplacian_2d(
    total_grid_points,
    mesh_size,
):
    """
    Construct the two-dimensional five-point Dirichlet Laplacian.

    The operator is the Kronecker sum

        Delta_h = I kron T + T kron I,

    where T is the one-dimensional second-difference operator.

    Parameters
    ----------
    total_grid_points : int
        Total number of grid points in each spatial direction,
        including boundary points.
    mesh_size : float
        Spatial mesh width.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse two-dimensional Laplacian acting on interior nodes.
    """

    n_interior = (
        total_grid_points - 2
    )

    laplacian_1d = (
        build_dirichlet_laplacian_1d(
            n_interior,
            mesh_size,
        )
    )

    identity = eye(
        n_interior,
        format="csr",
    )

    laplacian_2d = (
        kron(
            identity,
            laplacian_1d,
            format="csr",
        )
        +
        kron(
            laplacian_1d,
            identity,
            format="csr",
        )
    )

    return laplacian_2d


LAPLACIAN = (
    build_dirichlet_laplacian_2d(
        M,
        HX,
    )
)

IDENTITY = eye(
    N_INTERIOR,
    format="csr",
)


# =============================================================================
# Poisson solver for initial-data generation
# =============================================================================

# The matrix -Delta_h is fixed for every sample, so its factorization
# can be computed once and reused.
POISSON_SOLVER = factorized(
    (-LAPLACIAN).tocsc()
)


def sample_seed(sample_index):
    """
    Construct a deterministic seed for one dataset sample.

    Using a sample-specific seed makes the dataset reproducible even if
    generation is restarted from a nonzero value of START.
    """

    seed_sequence = np.random.SeedSequence(
        [
            BASE_RANDOM_SEED,
            int(sample_index),
        ]
    )

    return seed_sequence


def smooth_poisson_initial_condition(
    sample_index,
    amplitude=INITIAL_AMPLITUDE,
):
    r"""
    Generate smooth random initial data from a discrete Poisson problem.

    A random interior source f is generated and

        -Delta_h u = f

    is solved subject to homogeneous Dirichlet boundary conditions.

    The resulting field is normalized according to

        u <- amplitude * u / ||u||_infinity.

    Parameters
    ----------
    sample_index : int
        Global dataset sample index.
    amplitude : float
        Desired maximum absolute value of the initial field.

    Returns
    -------
    u : ndarray
        Full array of shape (M, M) with zero boundary values.
    """

    if amplitude <= 0.0:
        raise ValueError(
            "amplitude must be positive."
        )

    rng = np.random.default_rng(
        sample_seed(sample_index)
    )

    source = rng.standard_normal(
        N_INTERIOR
    )

    u_interior = POISSON_SOLVER(
        source
    )

    maximum = np.max(
        np.abs(u_interior)
    )

    if maximum == 0.0:
        raise RuntimeError(
            "The Poisson solution has zero magnitude "
            "and cannot be normalized."
        )

    u_interior *= (
        amplitude / maximum
    )

    u = np.zeros(
        (M, M),
        dtype=np.float64,
    )

    u[1:-1, 1:-1] = (
        u_interior.reshape(
            MI,
            MI,
        )
    )

    return u


# =============================================================================
# Eyre-type semi-implicit time step
# =============================================================================

def eyre_step(
    u_previous,
):
    r"""
    Advance the solution by one Eyre-type time step.

    The scheme is

        u^{n+1}
        - delta_t gamma Delta_h u^{n+1}
        + delta_t kappa (u^{n+1})^3

        =
        (1 + delta_t kappa) u^n.

    Newton's method is used to solve the nonlinear interior system.

    Parameters
    ----------
    u_previous : ndarray
        Full numerical solution at the previous time level.

    Returns
    -------
    u : ndarray
        Full numerical solution at the new time level.
    """

    previous_interior = (
        u_previous[1:-1, 1:-1]
    )

    rhs = (
        1.0
        + DELTA_T * KAPPA
    ) * previous_interior

    u_interior = (
        previous_interior.copy()
    )

    converged = False
    residual_norm = np.inf

    for _ in range(
        NEWTON_MAX_ITERS
    ):

        u_flat = (
            u_interior.ravel()
        )

        residual = (
            u_flat
            - DELTA_T
            * GAMMA
            * (
                LAPLACIAN
                @ u_flat
            )
            + DELTA_T
            * KAPPA
            * u_flat**3
            - rhs.ravel()
        )

        residual_norm = (
            np.linalg.norm(
                residual,
                ord=2,
            )
        )

        if (
            residual_norm
            < NEWTON_TOL
        ):
            converged = True
            break

        jacobian_diagonal = (
            3.0
            * DELTA_T
            * KAPPA
            * u_flat**2
        )

        jacobian = (
            IDENTITY
            - DELTA_T
            * GAMMA
            * LAPLACIAN
            + diags(
                jacobian_diagonal,
                offsets=0,
                format="csr",
            )
        )

        correction = spsolve(
            jacobian,
            -residual,
        )

        u_interior = (
            u_flat + correction
        ).reshape(
            MI,
            MI,
        )

    if not converged:
        raise RuntimeError(
            "Newton iteration failed to converge; "
            f"final residual norm = "
            f"{residual_norm:.3e}."
        )

    u = np.zeros(
        (M, M),
        dtype=np.float64,
    )

    u[1:-1, 1:-1] = (
        u_interior
    )

    return u


# =============================================================================
# Forward evolution
# =============================================================================

def evolve_forward(
    u_initial,
    num_steps,
):
    """
    Evolve one initial condition for a prescribed number of Eyre steps.

    Parameters
    ----------
    u_initial : ndarray
        Initial condition of shape (M, M).
    num_steps : int
        Number of time steps.

    Returns
    -------
    u : ndarray
        Numerical solution after ``num_steps`` time steps.
    """

    if num_steps < 0:
        raise ValueError(
            "num_steps must be nonnegative."
        )

    u = np.asarray(
        u_initial,
        dtype=np.float64,
    ).copy()

    for _ in range(
        num_steps
    ):
        u = eyre_step(u)

    return u


# =============================================================================
# Sample validation
# =============================================================================

def validate_sample(
    src,
    tar,
):
    """
    Verify basic structural properties of one dataset pair.
    """

    if src.shape != (M, M):
        raise ValueError(
            f"src has shape {src.shape}; "
            f"expected {(M, M)}."
        )

    if tar.shape != (M, M):
        raise ValueError(
            f"tar has shape {tar.shape}; "
            f"expected {(M, M)}."
        )

    if not np.all(
        np.isfinite(src)
    ):
        raise FloatingPointError(
            "Nonfinite value detected in src."
        )

    if not np.all(
        np.isfinite(tar)
    ):
        raise FloatingPointError(
            "Nonfinite value detected in tar."
        )

    src_boundary = max(
        np.max(
            np.abs(src[0, :])
        ),
        np.max(
            np.abs(src[-1, :])
        ),
        np.max(
            np.abs(src[:, 0])
        ),
        np.max(
            np.abs(src[:, -1])
        ),
    )

    tar_boundary = max(
        np.max(
            np.abs(tar[0, :])
        ),
        np.max(
            np.abs(tar[-1, :])
        ),
        np.max(
            np.abs(tar[:, 0])
        ),
        np.max(
            np.abs(tar[:, -1])
        ),
    )

    if src_boundary > 1.0e-14:
        raise ValueError(
            "src violates the homogeneous "
            "Dirichlet boundary condition."
        )

    if tar_boundary > 1.0e-14:
        raise ValueError(
            "tar violates the homogeneous "
            "Dirichlet boundary condition."
        )


# =============================================================================
# Batch file handling
# =============================================================================

def batch_filename(
    batch_index,
):
    """
    Return the filename for one dataset batch.
    """

    return (
        BATCH_DIR
        / (
            f"{DATASET_NAME}_"
            f"batch_{batch_index:04d}.npz"
        )
    )


def save_batch(
    batch_index,
    src_batch,
    tar_batch,
    sample_indices,
):
    """
    Save one compressed dataset batch.

    Arrays are converted to float32 only when written to disk. The
    numerical simulation itself is performed in float64.
    """

    output_file = (
        batch_filename(
            batch_index
        )
    )

    np.savez_compressed(
        output_file,
        src=np.asarray(
            src_batch,
            dtype=np.float32,
        ),
        tar=np.asarray(
            tar_batch,
            dtype=np.float32,
        ),
        sample_indices=np.asarray(
            sample_indices,
            dtype=np.int64,
        ),
    )

    print(
        f"Saved batch {batch_index:04d}: "
        f"{output_file}"
    )

    print(
        f"    src shape = "
        f"{src_batch.shape}"
    )

    print(
        f"    tar shape = "
        f"{tar_batch.shape}"
    )


# =============================================================================
# Dataset generation
# =============================================================================

def generate_dataset():
    """
    Generate all requested dataset samples and save them in batches.
    """

    if END <= START:
        raise ValueError(
            "END must be greater than START."
        )

    if BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE must be positive."
        )

    print(
        "Backward Chafee--Infante "
        "dataset generation"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Samples:       "
        f"{START} through {END - 1}"
    )

    print(
        f"Sample count:  "
        f"{END - START}"
    )

    print(
        f"Grid:          "
        f"{M} x {M}"
    )

    print(
        f"Domain:        "
        f"[-{L}, {L}] x [-{L}, {L}]"
    )

    print(
        f"Mesh size:     "
        f"{HX:.8e}"
    )

    print(
        f"Time step:     "
        f"{DELTA_T:.8e}"
    )

    print(
        f"Eyre steps:    "
        f"{ITERS_SRC}"
    )

    print(
        f"Evolution time:"
        f" {ITERS_SRC * DELTA_T:.6f}"
    )

    print(
        f"gamma:         "
        f"{GAMMA}"
    )

    print(
        f"kappa:         "
        f"{KAPPA}"
    )

    print(
        f"IC amplitude:  "
        f"{INITIAL_AMPLITUDE}"
    )

    print()

    src_batch = np.empty(
        (
            BATCH_SIZE,
            M,
            M,
        ),
        dtype=np.float32,
    )

    tar_batch = np.empty(
        (
            BATCH_SIZE,
            M,
            M,
        ),
        dtype=np.float32,
    )

    sample_indices = np.empty(
        BATCH_SIZE,
        dtype=np.int64,
    )

    batch_index = 0
    batch_count = 0

    for sample_index in range(
        START,
        END,
    ):

        # ---------------------------------------------------------------------
        # Target: initial condition
        # ---------------------------------------------------------------------

        tar = (
            smooth_poisson_initial_condition(
                sample_index,
                amplitude=INITIAL_AMPLITUDE,
            )
        )

        # ---------------------------------------------------------------------
        # Source: forward-evolved state
        # ---------------------------------------------------------------------

        src = evolve_forward(
            tar,
            num_steps=ITERS_SRC,
        )

        # ---------------------------------------------------------------------
        # Verify sample
        # ---------------------------------------------------------------------

        validate_sample(
            src,
            tar,
        )

        # ---------------------------------------------------------------------
        # Store sample in current batch
        # ---------------------------------------------------------------------

        src_batch[
            batch_count
        ] = src.astype(
            np.float32
        )

        tar_batch[
            batch_count
        ] = tar.astype(
            np.float32
        )

        sample_indices[
            batch_count
        ] = sample_index

        batch_count += 1

        # ---------------------------------------------------------------------
        # Save full batch
        # ---------------------------------------------------------------------

        if (
            batch_count
            == BATCH_SIZE
        ):

            save_batch(
                batch_index,
                src_batch,
                tar_batch,
                sample_indices,
            )

            batch_index += 1
            batch_count = 0

        if (
            (sample_index - START + 1)
            % 25
            == 0
        ):

            print(
                f"{sample_index - START + 1}"
                f"/{END - START} "
                "samples generated"
            )

    # -------------------------------------------------------------------------
    # Save incomplete final batch
    # -------------------------------------------------------------------------

    if batch_count > 0:

        save_batch(
            batch_index,
            src_batch[
                :batch_count
            ],
            tar_batch[
                :batch_count
            ],
            sample_indices[
                :batch_count
            ],
        )

    print()

    print(
        "Dataset batch generation complete."
    )


# =============================================================================
# Merge batch files
# =============================================================================

def merge_all_batches():
    """
    Merge the batch files belonging to this dataset.

    Only files with the exact dataset prefix are used, preventing unrelated
    NPZ files in the data directory from being incorporated accidentally.
    """

    pattern = (
        f"{DATASET_NAME}_"
        f"batch_*.npz"
    )

    batch_files = sorted(
        BATCH_DIR.glob(
            pattern
        )
    )

    if not batch_files:
        raise FileNotFoundError(
            "No batch files were found."
        )

    print()

    print(
        f"Merging {len(batch_files)} "
        "dataset batches..."
    )

    total_samples = 0

    for path in batch_files:

        with np.load(
            path
        ) as data:

            total_samples += (
                data["src"].shape[0]
            )

    X_src = np.empty(
        (
            total_samples,
            M,
            M,
        ),
        dtype=np.float32,
    )

    X_tar = np.empty(
        (
            total_samples,
            M,
            M,
        ),
        dtype=np.float32,
    )

    indices = np.empty(
        total_samples,
        dtype=np.int64,
    )

    offset = 0

    for path in batch_files:

        with np.load(
            path
        ) as data:

            src = np.asarray(
                data["src"],
                dtype=np.float32,
            )

            tar = np.asarray(
                data["tar"],
                dtype=np.float32,
            )

            batch_indices = np.asarray(
                data["sample_indices"],
                dtype=np.int64,
            )

        count = src.shape[0]

        X_src[
            offset : offset + count
        ] = src

        X_tar[
            offset : offset + count
        ] = tar

        indices[
            offset : offset + count
        ] = batch_indices

        offset += count

        print(
            f"Loaded {path.name}"
        )

    # Ensure samples are stored in their intended global order.
    order = np.argsort(
        indices
    )

    X_src = X_src[
        order
    ]

    X_tar = X_tar[
        order
    ]

    indices = indices[
        order
    ]

    np.savez_compressed(
        FINAL_OUTFILE,
        src=X_src,
        tar=X_tar,
        sample_indices=indices,
        M=np.int64(M),
        L=np.float64(L),
        delta_t=np.float64(
            DELTA_T
        ),
        gamma=np.float64(
            GAMMA
        ),
        kappa=np.float64(
            KAPPA
        ),
        num_steps=np.int64(
            ITERS_SRC
        ),
        initial_amplitude=np.float64(
            INITIAL_AMPLITUDE
        ),
        base_random_seed=np.int64(
            BASE_RANDOM_SEED
        ),
    )

    print()

    print(
        "Final dataset saved to:"
    )

    print(
        f"    {FINAL_OUTFILE}"
    )

    print(
        f"src shape: {X_src.shape}"
    )

    print(
        f"tar shape: {X_tar.shape}"
    )

    print(
        f"src range: "
        f"[{X_src.min():.6f}, "
        f"{X_src.max():.6f}]"
    )

    print(
        f"tar range: "
        f"[{X_tar.min():.6f}, "
        f"{X_tar.max():.6f}]"
    )


# =============================================================================
# Main program
# =============================================================================

def main():
    """
    Generate and merge the backward Chafee--Infante dataset.
    """

    generate_dataset()
    merge_all_batches()

    print()

    print(
        "All dataset construction "
        "steps completed."
    )


if __name__ == "__main__":
    main()