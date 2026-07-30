"""
2D Allen--Cahn equation with convex--concave splitting
and homogeneous periodic boundary conditions.

Equation:
    u_t = Delta u - (u^3 - u)

Scheme:
    (I - dt * Delta) u^{n+1} = u^n + dt * (u^n - (u^n)^3)

This is a semi-implicit convex splitting method:
    - Laplacian treated implicitly
    - Nonlinear term explicit

Outputs:
    - snapshots
    - energy decay
    - min/max tracking
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve


# ============================================================
# PARAMETERS
# ============================================================

T = 1.0
N = 10000
snapshot_stride = 100
X = 20  # number of energy & interface steps to display

dt = 5.0e-4  # time t = n dt; end/total time t = N dt = 5.
L = 1.0
M = 128

hx = 2.0 * L / M
hy = hx
h = hx * hy

os.makedirs("images", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ============================================================
# GRID
# ============================================================

J = M * M


# ============================================================
# LAPLACIAN
# ============================================================

def build_laplacian_periodic(M, h):
    e = np.ones(M)

    # 1D periodic Laplacian
    T = diags([e, -2*e, e], [-1, 0, 1], shape=(M, M)).tolil()
    T[0, -1] = 1
    T[-1, 0] = 1
    T = T / h**2

    I = eye(M)
    L = kron(I, T) + kron(T, I)

    return L.tocsr()

L2D = build_laplacian_periodic(M, hx)
I_big = eye(J, format='csr')


# ============================================================
# INITIAL DATA
# ============================================================

def phase_ic(M, seed=0):
    rng = np.random.default_rng(seed)
    u = np.zeros((M, M))

    block_size = 4
    for i in range(0, M, block_size):
        for j in range(0, M, block_size):
            u[i:i+block_size, j:j+block_size] = rng.choice([-1.0, 1.0])

    u += 0.3  # bias toward +1 phase
    u += 0.05 * rng.standard_normal((M, M)) # additative Gaussian noise

    return u

# ============================================================
# ENERGY
# ============================================================

def lyapunov_energy(u):
    ux = (u[1:, :] - u[:-1, :]) / hx
    uy = (u[:, 1:] - u[:, :-1]) / hy

    grad_sq = np.sum(ux**2) + np.sum(uy**2)
    potential = 0.25 * (u**2 - 1.0)**2

    return (0.5 * grad_sq + np.sum(potential)) * h


# ============================================================
# INTERFACE MEASURE
# ============================================================

def interface_measure(u):
    ux = np.diff(u, axis=0) / hx
    uy = np.diff(u, axis=1) / hy
    return np.sum(np.sqrt(ux[:, :-1]**2 + uy[:-1, :]**2)) * h

# ============================================================
# INITIALIZE
# ============================================================

phi = np.zeros((N + 1, M, M))
phi[0] = np.tanh(2 * (np.random.randn(M, M) + 0.5))

print(
    f"Step {0:5d}/{N} | "
    f"min={np.min(phi[0]):.4f}, max={np.max(phi[0]):.4f} | "
    f"E={lyapunov_energy(phi[0]):.6f}"
    f"I={interface_measure(phi[0]):.6f}"
)

plt.imshow(phi[0], cmap='twilight_shifted', interpolation='none')
plt.colorbar()
plt.savefig(f'images/phi00000.png')
plt.close()


energy_vals = np.zeros(N + 1)
u_min_vals = np.zeros(N + 1)
u_max_vals = np.zeros(N + 1)
interface_vals = np.zeros(N + 1)

energy_vals[0] = lyapunov_energy(phi[0])
u_min_vals[0] = np.min(phi[0])
u_max_vals[0] = np.max(phi[0])
interface_vals[0] = interface_measure(phi[0])

# ============================================================
# PRECOMPUTE MATRIX
# ============================================================

A = (I_big - dt * L2D).tocsr()


# ============================================================
# TIME STEPPING
# ============================================================

for n in range(1, N + 1):
    u_prev = phi[n - 1]
    u_flat = u_prev.ravel()

    rhs = u_flat + dt * (u_flat - u_flat**3)

    u_new_flat = spsolve(A, rhs)
    u_new = u_new_flat.reshape(M, M)

    phi[n] = u_new

    energy_vals[n] = lyapunov_energy(u_new)
    u_min_vals[n] = np.min(u_new)
    u_max_vals[n] = np.max(u_new)
    interface_vals[n] = interface_measure(u_new)
    
    print(
        f"Step {n:5d}/{N} | "
        f"min={u_min_vals[n]:.4f}, max={u_max_vals[n]:.4f} | "
        f"E={energy_vals[n]:.6f} | "
        f"I={interface_vals[n]:.6f}"
    )

    if n % snapshot_stride == 0:
        plt.imshow(u_new, cmap='twilight_shifted', interpolation='none')
        plt.colorbar()
        plt.savefig(f'images/phi{str(n).zfill(5)}.png')
        plt.close()

# ============================================================
# FINAL PLOTS
# ============================================================

plt.plot(u_max_vals, label='max')
plt.plot(u_min_vals, label='min')
plt.legend()
plt.savefig('data/maxmin.png')
plt.close()

plt.plot(energy_vals[:X])
plt.title('Lyapunov Energy (first 20 steps)')
plt.savefig('data/energy_first20.png')
plt.close()

plt.plot(interface_vals[:X])
plt.title('Interface Measure (first 20 steps)')
plt.xlabel('Time step')
plt.ylabel('Approximate interface length')
plt.savefig('data/interface_first20.png')
plt.close()

print("Finished.")
