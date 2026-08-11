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

# ============================================================ #
# PARAMETERS                                                   #
# ============================================================ #

snapshot_stride = 100
X = 20  # number of energy & interface steps to display
ITERS = 10
N = 20000
dt = 5.0e-4  # N*dt=10
L = 1.0
M = 128
hx = 2.0 * L / M  # square (periodic) domain [-1,1]^2
hy = hx
h = hx * hy

os.makedirs('images', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ============================================================ #
# GRID                                                         #
# ============================================================ #

J = M * M

# ============================================================ #
# LAPLACIAN                                                    #
# ============================================================ #

def build_laplacian_periodic(M, h):
    e = np.ones(M) # 1D periodic Laplacian
    T = diags([e, -2*e, e], [-1, 0, 1], shape=(M, M)).tolil()
    T[0, -1] = 1
    T[-1, 0] = 1
    T = T / h**2
    I = eye(M)
    L = kron(I, T) + kron(T, I)
    return L.tocsr()

L2D = build_laplacian_periodic(M, hx)
I_big = eye(J, format='csr')

# ============================================================ #
# ENERGY                                                       #
# ============================================================ #

def lyapunov_energy(u):
    ux = (np.roll(u, -1, axis=0) - u) / hx
    uy = (np.roll(u, -1, axis=1) - u) / hy
    grad_sq = np.sum(ux**2) + np.sum(uy**2)
    potential = 0.25 * (u**2 - 1.0)**2
    return (0.5 * grad_sq + np.sum(potential)) * h

# ============================================================ #
# INTERFACE LENGTH                                             #
# ============================================================ #

def interface_length(u, eps=0.05):
    ux = (np.roll(u, -1, axis=0) - u) / hx
    uy = (np.roll(u, -1, axis=1) - u) / hy
    grad_mag = np.sqrt(ux**2 + uy**2)
    delta = np.exp(-u**2 / (2 * eps**2))
    delta /= np.sqrt(2 * np.pi) * eps
    return np.sum(delta * grad_mag) * h

# ============================================================ #
# INITIALIZE                                                   #
# ============================================================ #

phi = np.tanh(2 * (np.random.randn(M, M) + 0.5))  # initial condition
print(f"Step {0:5d}/{N} | min={np.min(phi):.4f}, max={np.max(phi):.4f} | E={lyapunov_energy(phi):.6f} | I={interface_length(phi):.6f}")

plt.imshow(phi, cmap='twilight_shifted', interpolation='none')
plt.colorbar()
plt.axis('off')
plt.savefig(f"images/phi00000.png", dpi=300)
plt.close()

energy_vals = np.zeros(N + 1)
u_min_vals = np.zeros(N + 1)
u_max_vals = np.zeros(N + 1)
interface_vals = np.zeros(N + 1)

energy_vals[0] = lyapunov_energy(phi)
u_min_vals[0] = np.min(phi)
u_max_vals[0] = np.max(phi)
interface_vals[0] = interface_length(phi)

# ============================================================ #
# TIME STEPPING                                                #
# ============================================================ #
# True Convex Splitting Matrix (Pre-built outside loop for speed)
# (I - dt * Delta) u^{n+1} = (1 + dt) * u^n - dt * (u^n)^3

A_linear = I_big - dt * L2D

for n in range(1, N + 1):
    u = phi.copy()
    
    # Strictly applying the explicit-implicit convex splitting step
    rhs = ((1.0 + dt) * u - dt * (u**3)).ravel()
    u_new = spsolve(A_linear, rhs).reshape(M, M)
    
    phi = u_new
    
    energy_vals[n] = lyapunov_energy(phi)
    u_min_vals[n] = np.min(phi)
    u_max_vals[n] = np.max(phi)
    interface_vals[n] = interface_length(phi)
    
    if n % 100 == 0:
        print(f"Step {n:5d}/{N} | min={u_min_vals[n]:.4f} | max={u_max_vals[n]:.4f} | E={energy_vals[n]:.6f} | I={interface_vals[n]:.6f}")
        
    if n % snapshot_stride == 0:
        plt.imshow(phi, cmap='twilight_shifted', interpolation='none')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(f"images/phi{str(n).zfill(5)}.png", dpi=300)
        plt.close()

# ============================================================ #
# FINAL PLOTS                                                  #
# ============================================================ #

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
