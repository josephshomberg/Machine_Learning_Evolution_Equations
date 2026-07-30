import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Viscous Burgers forward solver
#
#     u_t + u u_x = nu u_xx
#
# Periodic domain: 0 <= x < L
# Transport step: local Lax--Friedrichs flux
# Diffusion step: exact Fourier heat update
# ============================================================

N = 512
L = 2*np.pi
nu = 0.03

T = 2.0
dt = 1.0e-3
num_steps = int(T/dt)

x = np.linspace(0,L,N,endpoint=False)
dx = L/N

k = 2*np.pi*np.fft.fftfreq(N,d=dx)

# ------------------------------------------------------------
# Smooth random periodic initial condition
# ------------------------------------------------------------

np.random.seed(3)

u0 = np.sin(x) + 0.35*np.sin(2*x)

for m in range(3,8):
    a = np.random.randn()
    b = np.random.randn()

    u0 += 0.12*(
        a*np.cos(m*x)
        + b*np.sin(m*x)
    )/m**2

u = u0.copy()

# ------------------------------------------------------------
# Flux for inviscid Burgers
# ------------------------------------------------------------

def flux(v):
    return 0.5*v**2

def transport_step(u,dt,dx):
    """
    Conservative local Lax--Friedrichs step for

        u_t + (u^2/2)_x = 0.

    Periodic boundary conditions are imposed with np.roll.
    """

    uR = np.roll(u,-1)
    uL = np.roll(u,1)

    alpha_R = np.maximum(np.abs(u),np.abs(uR))
    alpha_L = np.maximum(np.abs(uL),np.abs(u))

    F_R = 0.5*(flux(u) + flux(uR)) - 0.5*alpha_R*(uR-u)
    F_L = 0.5*(flux(uL) + flux(u)) - 0.5*alpha_L*(u-uL)

    return u - (dt/dx)*(F_R-F_L)

def diffusion_step(u,dt,nu,k):
    """
    Exact Fourier update for

        u_t = nu u_xx.
    """

    uhat = np.fft.fft(u)
    uhat = np.exp(-nu*k**2*dt)*uhat

    return np.real(np.fft.ifft(uhat))

# ------------------------------------------------------------
# Times to save
# ------------------------------------------------------------

save_times = [0.0,0.4,1.0,2.0]
save_steps = [int(t/dt) for t in save_times]

snapshots = {0: u.copy()}

space_time = []
time_values = []

# ------------------------------------------------------------
# Time evolution
# ------------------------------------------------------------

for n in range(1,num_steps+1):

    # nonlinear steepening
    u = transport_step(u,dt,dx)

    # viscous smoothing
    u = diffusion_step(u,dt,nu,k)

    if n in save_steps:
        snapshots[n] = u.copy()

    if n % 5 == 0:
        space_time.append(u.copy())
        time_values.append(n*dt)

space_time = np.array(space_time)

# ------------------------------------------------------------
# Plot solution snapshots
# ------------------------------------------------------------

plt.figure(figsize=(11,6))

for t,n in zip(save_times,save_steps):
    plt.plot(x,snapshots[n],label=f"t = {t}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Viscous Burgers Solution Snapshots")
plt.legend()
plt.tight_layout()
plt.savefig("5-VBE_snapshots.png",dpi=300)
plt.show()

# ------------------------------------------------------------
# Plot spatiotemporal diagram
# ------------------------------------------------------------

plt.figure(figsize=(11,6))

plt.imshow(
    space_time,
    extent=[0,L,time_values[-1],time_values[0]],
    aspect="auto"
)

plt.xlabel("x")
plt.ylabel("time")
plt.title("Viscous Burgers Spatiotemporal Evolution")
plt.colorbar(label="u(x,t)")
plt.tight_layout()
plt.savefig("5-VBE_spacetime.png",dpi=300)
plt.show()

