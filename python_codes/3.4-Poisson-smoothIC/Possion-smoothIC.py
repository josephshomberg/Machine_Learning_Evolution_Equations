"""
2D Dirichlet Laplace Solver on a Cartesian Grid.

Generates smooth initial data by solving a Poisson equation with a random
source term and homogeneous Dirichlet boundary conditions.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

# =============================================================================
# PARAMETERS
# =============================================================================

M = 128                  # Total grid size including boundary
L = 1.0                  # Domain is [-L, L]^2
AMP = 0.02               # Amplitude scaling
SEED = 42                # Reproducibility seed

np.random.seed(SEED)

# =============================================================================
# GRID SETUP
# =============================================================================

hx = 2.0 * L / (M - 1)
mi = M - 2               # Number of interior points in one direction
n_int = mi * mi          # Total number of interior unknowns

# =============================================================================
# BUILD 2D DIRICHLET LAPLACIAN
# =============================================================================
# Construct the 1D second-derivative operator

main_diag = -2.0 * np.ones(mi)
off_diag = 1.0 * np.ones(mi - 1)
t_matrix = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(mi, mi))
t_matrix /= hx * hx

# Extend to 2D using Kronecker products
identity_matrix = eye(mi)
l_2d = kron(identity_matrix, t_matrix) + kron(t_matrix, identity_matrix)

# =============================================================================
# RANDOM SOURCE TERM FOR POISSON PROBLEM
# Solve: -Delta u = f in Omega, u = 0 on boundary
# =============================================================================

f_source = np.random.randn(n_int)

# Solve the linear system
u_interior = spsolve(-l_2d, f_source)

# Reshape interior solution to a 2D mesh grid
u_interior = u_interior.reshape((mi, mi))

# =============================================================================
# NORMALIZE AND EMBED INTO FULL GRID
# =============================================================================
# Normalize maximum magnitude to 1.0

u_interior /= np.max(np.abs(u_interior))

# Embed into full grid with zero boundary conditions
u0 = np.zeros((M, M), dtype=np.float64)
u0[1:-1, 1:-1] = AMP * u_interior

# =============================================================================
# DISPLAY & VERIFICATION
# =============================================================================

print(f"u0 shape: {u0.shape}")
print(f"min/max:  {u0.min():.6f} / {u0.max():.6f}")

# Calculate exact max absolute error on boundaries
boundary_max = max(
    np.max(np.abs(u0[0, :])),
    np.max(np.abs(u0[-1, :])),
    np.max(np.abs(u0[:, 0])),
    np.max(np.abs(u0[:, -1]))
)
print(f"Boundary max absolute value: {boundary_max}")

# Visualize the field
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(u0, cmap="coolwarm", origin="lower", extent=[-L, L, -L, L])
fig.colorbar(im, ax=ax, label="Amplitude")
ax.set_title("Smooth Poisson Initial Data")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()
plt.show()
