import os
import numpy as np
from numpy import savez_compressed
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve


# ================================================================
#  DATASET GENERATION FOR THE BACKWARD CHAFEE--INFANTE PROBLEM
# ================================================================
#
#  This script generates pairs
#      (src, tar) = (S_N(u_0), u_0),
#  where u_0 is a smooth initial condition satisfying homogeneous
#  Dirichlet boundary conditions, and S_N(u_0) is the result of
#  evolving u_0 forward by N explicit forward Euler steps for the
#  Chafee--Infante equation
#
#      u_t = gamma * Delta u - kappa * (u^3 - u).
#
#  The script supports two stages:
#
#  (1) batchwise generation of compressed .npz files,
#  (2) optional merging of all batch files into one final dataset.
#
# ================================================================


# ================================================================
#  GLOBAL PARAMETERS
# ================================================================

# Number of forward Euler steps used to produce src = S_N(u_0)
ITERS_SRC = 400

# Output controls
DATA_DIR = "data"
TMP_DIR = "tmp_pairs"
PREFIX = os.path.join(DATA_DIR, "128x128_DBC_kappa=4.7")

# Range of samples to generate
START = 0
END = 10000

# Number of temporary samples merged into each batch file
BATCH_SIZE = 1000

# Final merged output file
FINAL_OUTFILE = os.path.join(
    DATA_DIR,
    f"dataset_128x128_DBC_kappa=4.7_count={END - START}.npz"
)

# PDE / discretization parameters
deltat = 0.001
L = 1.0
M = 128
hx = 2.0 * L / M
gamma = 0.005
kappa = 4.7

# Smooth initial-data parameters
initial_amplitude = 0.02
ell = 0.03   # correlation length for screened Poisson smoothing

# Directory setup
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Interior grid size
Mi = M - 2
J = Mi * Mi


# ================================================================
#  DISCRETE DIRICHLET LAPLACIAN
# ================================================================

def build_dirichlet_laplacian_1d(Mi, hx):
    """
    Construct the standard second-order finite-difference 1D Laplacian
    with homogeneous Dirichlet boundary conditions on the interior grid.
    """
    main = -2.0 * np.ones(Mi)
    off = 1.0 * np.ones(Mi - 1)
    T = diags([off, main, off], [-1, 0, 1], shape=(Mi, Mi))
    return T / (hx * hx)


def build_dirichlet_laplacian_2d(M, hx):
    """
    Construct the 2D five-point Laplacian with homogeneous Dirichlet
    boundary conditions on an M x M grid, restricted to interior nodes.
    """
    Mi = M - 2
    T = build_dirichlet_laplacian_1d(Mi, hx)
    I = eye(Mi)
    return kron(I, T) + kron(T, I)


L2D = build_dirichlet_laplacian_2d(M, hx).tocsr()


# ================================================================
#  SMOOTH INITIAL CONDITIONS VIA SCREENED POISSON
# ================================================================

def smooth_dbc_ic_poisson(M, L2D, ell=0.5, amp=0.02, seed=None):
    """
    Generate a smooth random initial condition satisfying homogeneous
    Dirichlet boundary conditions by solving the screened Poisson problem

        (I - ell^2 * Delta) u = noise

    on the interior grid.

    The returned array has shape (M, M), with zero boundary values.
    """
    rng = np.random.default_rng(seed)
    Mi = M - 2
    J = Mi * Mi

    noise_int = rng.standard_normal((Mi, Mi)).astype(np.float64)
    rhs = noise_int.ravel()

    A = eye(J, format="csr") - (ell ** 2) * L2D
    u_int = spsolve(A, rhs).reshape(Mi, Mi)

    # Rescale so that the maximum absolute amplitude is approximately amp.
    max_abs = np.max(np.abs(u_int))
    if max_abs > 0:
        u_int = (amp / max_abs) * u_int

    u_full = np.zeros((M, M), dtype=np.float64)
    u_full[1:-1, 1:-1] = u_int
    return u_full.astype(np.float32)


# ================================================================
#  FORWARD EULER TIME STEPPER
# ================================================================

def forward_euler_step(u_int, L2D, dt, gamma, kappa):
    """
    Advance one explicit forward Euler step for the interior values of

        u_t = gamma * Delta u - kappa * (u^3 - u).
    """
    u = u_int.ravel()
    lap = L2D @ u
    reaction = -kappa * (u ** 3 - u)
    u_new = u + dt * (gamma * lap + reaction)
    return u_new.reshape(Mi, Mi)


def evolve_forward(u0_full, L2D, num_steps, dt, gamma, kappa):
    """
    Evolve a full M x M state forward for num_steps explicit Euler steps,
    maintaining homogeneous Dirichlet boundary conditions.
    """
    phi = u0_full.copy()

    for _ in range(num_steps):
        u_int = forward_euler_step(phi[1:-1, 1:-1], L2D, dt, gamma, kappa)

        phi_new = np.zeros((M, M), dtype=np.float32)
        phi_new[1:-1, 1:-1] = u_int
        phi = phi_new

    return phi


# ================================================================
#  SHAPE NORMALIZATION
# ================================================================

def normalize_array_shape(x):
    """
    Normalize stored arrays so that they always have shape (N, M, M).
    """
    x = np.asarray(x)

    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0, :, :]

    if x.ndim == 4 and x.shape[-1] == 1:
        return x[:, :, :, 0]

    if x.ndim == 2:
        return x[np.newaxis, :, :]

    if x.ndim == 3:
        return x

    raise ValueError(f"Unhandled array shape: {x.shape}")


# ================================================================
#  TEMPORARY STORAGE AND BATCH MERGING
# ================================================================

def save_temp_pair(tmp_dir, i, src, tar):
    """
    Save a single temporary sample pair to disk.
    """
    filename = os.path.join(tmp_dir, f"tmp_{i:05d}.npz")
    savez_compressed(filename, src=src, tar=tar)


def merge_temp_pairs(tmp_dir, batch_idx, prefix):
    """
    Merge all temporary sample files currently in tmp_dir into one
    compressed batch file, then delete the temporary files.
    """
    files = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".npz"))
    if not files:
        return

    all_src = []
    all_tar = []

    for fname in files:
        path = os.path.join(tmp_dir, fname)
        data = np.load(path)
        all_src.append(normalize_array_shape(data["src"]))
        all_tar.append(normalize_array_shape(data["tar"]))

    X_src = np.concatenate(all_src, axis=0)
    X_tar = np.concatenate(all_tar, axis=0)

    out_file = f"{prefix}_batch_{batch_idx:03d}.npz"
    savez_compressed(out_file, src=X_src, tar=X_tar)

    for fname in files:
        os.remove(os.path.join(tmp_dir, fname))

    print(f"[Batch {batch_idx}] saved -> {out_file}   src shape = {X_src.shape}")


def merge_all_batches(data_dir, outfile):
    """
    Merge all dataset .npz files in data_dir into one final compressed file.
    """
    npz_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".npz") and not f.startswith("dataset_")
    )

    if not npz_files:
        print("No batch files found to merge.")
        return

    print(f"Found {len(npz_files)} batch files.")

    all_src = []
    all_tar = []

    for fname in npz_files:
        path = os.path.join(data_dir, fname)
        data = np.load(path)

        src = normalize_array_shape(data["src"])
        tar = normalize_array_shape(data["tar"])

        all_src.append(src)
        all_tar.append(tar)

    X_src = np.concatenate(all_src, axis=0)
    X_tar = np.concatenate(all_tar, axis=0)

    savez_compressed(outfile, src=X_src, tar=X_tar)
    print(f"Final merged dataset saved to: {outfile}")
    print(f"Final shapes: src = {X_src.shape}, tar = {X_tar.shape}")


# ================================================================
#  MAIN DATASET GENERATION LOOP
# ================================================================

def generate_dataset():
    """
    Generate the dataset in temporary samples and merge them batchwise.
    """
    batch_idx = 0
    tmp_counter = 0

    print("Generating dataset using explicit forward Euler...")

    for run in range(START, END):
        # Step 1: create smooth Dirichlet initial condition u_0
        tar = smooth_dbc_ic_poisson(
            M=M,
            L2D=L2D,
            ell=ell,
            amp=initial_amplitude,
            seed=None
        )

        # Enforce boundary conditions explicitly
        tar[0, :] = 0.0
        tar[-1, :] = 0.0
        tar[:, 0] = 0.0
        tar[:, -1] = 0.0

        # Step 2: evolve forward to obtain src = S_N(u_0)
        src = evolve_forward(
            u0_full=tar,
            L2D=L2D,
            num_steps=ITERS_SRC,
            dt=deltat,
            gamma=gamma,
            kappa=kappa
        )

        # Add batch dimension so each sample has shape (1, M, M)
        src = src[np.newaxis, :, :]
        tar = tar[np.newaxis, :, :]

        # Step 3: save temporary sample
        save_temp_pair(TMP_DIR, tmp_counter, src, tar)
        tmp_counter += 1

        # Step 4: merge once batch is full
        if tmp_counter == BATCH_SIZE:
            merge_temp_pairs(TMP_DIR, batch_idx, PREFIX)
            batch_idx += 1
            tmp_counter = 0

        if (run + 1) % 25 == 0:
            print(f"{run + 1}/{END} samples generated")

    # Merge any remaining temporary samples
    if tmp_counter > 0:
        merge_temp_pairs(TMP_DIR, batch_idx, PREFIX)

    print("Finished generating batch files.")


# ================================================================
#  SCRIPT ENTRY POINT
# ================================================================

if __name__ == "__main__":
    generate_dataset()
    merge_all_batches(DATA_DIR, FINAL_OUTFILE)
    print("All dataset construction steps completed.")
