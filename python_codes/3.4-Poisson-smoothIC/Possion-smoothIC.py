import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================
M = 128                  # total grid size including boundary
L = 1.0                  # domain is [-L,L]^2
amp = 0.02               # amplitude scaling
seed = 42                # reproducibility

np.random.seed(seed)

# ============================================================
# GRID SETUP
# ============================================================
hx = 2.0 * L / (M - 1)
Mi = M - 2               # number of interior points in one direction
Nint = Mi * Mi           # total number of interior unknowns

# ============================================================
# BUILD 2D DIRICHLET LAPLACIAN
# ============================================================
main = -2.0 * np.ones(Mi)
off  =  1.0 * np.ones(Mi - 1)
T = diags([off, main, off], [-1, 0, 1], shape=(Mi, Mi)) / (hx * hx)
I = eye(Mi)
L2D = kron(I, T) + kron(T, I)

# ============================================================
# RANDOM SOURCE TERM FOR POISSON PROBLEM
# Solve: -Delta u = f in Omega,  u = 0 on boundary
# ============================================================
f = np.random.randn(Nint)

# Solve Poisson system
u_int = spsolve(-L2D, f)

# Reshape interior solution
u_int = u_int.reshape((Mi, Mi))

# ============================================================
# NORMALIZE AND EMBED INTO FULL GRID
# ============================================================
u_int = u_int / np.max(np.abs(u_int))
u0 = np.zeros((M, M), dtype=np.float64)
u0[1:-1, 1:-1] = amp * u_int

# ============================================================
# DISPLAY
# ============================================================
print("u0 shape:", u0.shape)
print("min/max:", u0.min(), u0.max())
print("boundary max abs:", np.max(np.abs([
    u0[0, :], u0[-1, :], u0[:, 0], u0[:, -1]
])))

plt.imshow(u0, cmap="coolwarm", origin="lower")
plt.colorbar()
plt.title("Smooth Poisson Initial Data")
plt.show()
