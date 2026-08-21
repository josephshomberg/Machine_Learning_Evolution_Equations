import numpy as np
from numpy import savez_compressed
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import os

# ================================================================
# ---------------------- GLOBAL CONTROL ---------------------------
# ================================================================

ITERS_SRC  = 400
PREFIX     = 'data/128x128_DBC_kappa=4.7'
START      = 0
END        = 10000
BATCH_SIZE = 1000         # keeps RAM + disk safe

# Simulation parameters (EXPLICIT FORWARD EULER)
deltat = 0.001
L = 1.0
M = 128
hx = 2.0 * L / M
gamma = 0.005
kappa = 4.7
initial_amplitude = 0.02

# dirs
os.makedirs('data', exist_ok=True)
TMP_DIR = 'tmp_pairs'
os.makedirs(TMP_DIR, exist_ok=True)

Mi = M - 2          # interior
J  = Mi * Mi

# ================================================================
# ------------------   SMOOTH POISSON INITIAL DATA   -------------
# ================================================================

def smooth_dbc_ic_poisson(M, L2D, ell=0.5, amp=0.02, seed=None):
    """
    Smooth DBC IC via screened Poisson:
      (I - ell^2 Δ) u = noise, with homogeneous DBC (implicit via interior-only L2D).
    Returns full (M,M) with boundary = 0.
    """
    rng = np.random.default_rng(seed)
    Mi = M - 2
    J = Mi * Mi

    noise_int = rng.standard_normal((Mi, Mi)).astype(np.float64)
    rhs = noise_int.ravel()

    A = (eye(J, format='csr') - (ell**2) * L2D)  # L2D already includes 1/h^2 scaling
    u_int = spsolve(A, rhs).reshape(Mi, Mi)

    # scale to desired amplitude (peak-ish control)
    m = np.max(np.abs(u_int))
    if m > 0:
        u_int = (amp / m) * u_int

    u_full = np.zeros((M, M), dtype=np.float64)
    u_full[1:-1, 1:-1] = u_int
    return u_full.astype(np.float32)

# ================================================================
# ------------------   LAPLACIAN OPERATORS   ---------------------
# ================================================================

def build_dirichlet_laplacian_1d(Mi, hx):
    main = -2.0 * np.ones(Mi)
    off  =  1.0 * np.ones(Mi-1)
    T = diags([off, main, off], [-1,0,1], shape=(Mi,Mi))
    return T / (hx*hx)

def build_dirichlet_laplacian_2d(M, hx):
    Mi = M - 2
    T = build_dirichlet_laplacian_1d(Mi, hx)
    I = eye(Mi)
    return kron(I, T) + kron(T, I)

L2D = build_dirichlet_laplacian_2d(M, hx).tocsr()

# ================================================================
# ------------------ FORWARD EULER TIME STEPPER ------------------
# ================================================================

def forward_euler_step(u_int, L2D, dt, gamma, kappa):
    """
    Explicit forward Euler for:
        u_t = gamma Δu - kappa (u^3 - u)
    """
    u = u_int.ravel()
    lap = L2D @ u
    reaction = -kappa * (u**3 - u)
    unew = u + dt*(gamma*lap + reaction)
    return unew.reshape(Mi, Mi)

# ================================================================
# ------------------ TEMP + BATCH MERGING I/O --------------------
# ================================================================

def save_temp_pair(tmp_dir, i, src, tar):
    filename = os.path.join(tmp_dir, f'tmp_{i:05d}.npz')
    savez_compressed(filename, src=src, tar=tar)

def merge_and_delete(tmp_dir, batch_idx, prefix):
    files = sorted([f for f in os.listdir(tmp_dir) if f.endswith('.npz')])
    if not files:
        return

    all_src = []
    all_tar = []

    for f in files:
        data = np.load(os.path.join(tmp_dir, f))
        all_src.append(data['src'])
        all_tar.append(data['tar'])

    # Concatenate batch
    all_src = np.concatenate(all_src, axis=0)
    all_tar = np.concatenate(all_tar, axis=0)

    out_file = f'{prefix}_batch_{batch_idx:03d}.npz'
    savez_compressed(out_file, src=all_src, tar=all_tar)

    # Delete temporary individual files
    for f in files:
        os.remove(os.path.join(tmp_dir, f))

    print(f"[Batch {batch_idx}] Saved → {out_file} (src shape = {all_src.shape})")


# ================================================================
# --------------------------- MAIN LOOP --------------------------
# ================================================================

batch_idx   = 0
tmp_counter = 0

print("Generating dataset using explicit Forward Euler...")

for run in range(START, END):

    # ------------------ CREATE SMOOTH POISSON INITIAL DATA -----------
    # Smooth DBC initial condition via screened Poisson
    # Choose ell to set correlation length (in physical units on [-L,L]^2).
    # Typical: ell ~ 0.03 to 0.10 for M=128.
    ell = 0.03

    tar = smooth_dbc_ic_poisson(
        M=M,
        L2D=L2D,
        ell=ell,
        amp=initial_amplitude,
        seed=None  # run  # 'run' makes each sample reproducible but different
    )

    # (already DBC, but keep explicit)
    tar[0,:] = tar[-1,:] = 0.0
    tar[:,0] = tar[:,-1] = 0.0

    phi = tar.copy()

    # ------------------ TIME EVOLUTION --------------------------
    for _ in range(ITERS_SRC):
        u_int = forward_euler_step(phi[1:-1, 1:-1], L2D, deltat, gamma, kappa)

        phi = np.zeros((M, M), dtype=np.float32)
        phi[1:-1,1:-1] = u_int

    src = phi.copy()

    # shape normalization for batching
    src = src[np.newaxis, :, :]
    tar = tar[np.newaxis, :, :]

    # ------------------ SAVE TEMP -------------------------------
    save_temp_pair(TMP_DIR, tmp_counter, src, tar)
    tmp_counter += 1

    # merge batch if full
    if tmp_counter == BATCH_SIZE:
        merge_and_delete(TMP_DIR, batch_idx, PREFIX)
        batch_idx += 1
        tmp_counter = 0

    if (run+1) % 25 == 0:
        print(f"{run+1}/{END} generated")

# ------------------ MERGE REMAINING ------------------------------
if tmp_counter > 0:
    merge_and_delete(TMP_DIR, batch_idx, PREFIX)

print("\nFINISHED ALL DATASETS.")
