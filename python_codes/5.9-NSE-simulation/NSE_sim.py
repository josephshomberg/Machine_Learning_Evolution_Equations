import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Two-dimensional incompressible Navier--Stokes solver
#
#     u_t + (u . grad)u = -grad p + nu Delta u
#     div u = 0
#
# Periodic domain: [0,L] x [0,L]
# Method: projection method
# Spatial derivatives: Fourier spectral
# Time stepping: explicit advection/diffusion + pressure projection
# ============================================================

N = 128
L = 2*np.pi
nu = 0.01

dt = 0.005
T = 8.0
num_steps = int(T/dt)

x = np.linspace(0,L,N,endpoint=False)
y = np.linspace(0,L,N,endpoint=False)

X,Y = np.meshgrid(x,y,indexing="ij")

kx = 2*np.pi*np.fft.fftfreq(N,d=L/N)
ky = 2*np.pi*np.fft.fftfreq(N,d=L/N)

KX,KY = np.meshgrid(kx,ky,indexing="ij")
K2 = KX**2 + KY**2
K2[0,0] = 1.0

# ------------------------------------------------------------
# Initial divergence-free velocity field
# generated from a smooth stream function psi.
#
#     u =  psi_y
#     v = -psi_x
# ------------------------------------------------------------

np.random.seed(2)

psi = (
    np.sin(X)*np.sin(Y)
    + 0.4*np.sin(2*X + 0.3)*np.sin(Y)
    + 0.3*np.sin(X)*np.sin(2*Y - 0.5)
)

psihat = np.fft.fft2(psi)

uvel = np.real(np.fft.ifft2(1j*KY*psihat))
vvel = np.real(np.fft.ifft2(-1j*KX*psihat))

# ------------------------------------------------------------
# Spectral derivative operators
# ------------------------------------------------------------

def ddx(f):
    return np.real(np.fft.ifft2(1j*KX*np.fft.fft2(f)))

def ddy(f):
    return np.real(np.fft.ifft2(1j*KY*np.fft.fft2(f)))

def laplacian(f):
    return np.real(np.fft.ifft2(-K2*np.fft.fft2(f)))

def divergence(u,v):
    return ddx(u) + ddy(v)

def curl(u,v):
    return ddx(v) - ddy(u)

# ------------------------------------------------------------
# Pressure projection
# ------------------------------------------------------------

def project(u,v,dt):
    """
    Project the velocity field onto the divergence-free subspace.

    We solve

        Delta p = (1/dt) div u_star

    in Fourier space, then set

        u_new = u_star - dt p_x,
        v_new = v_star - dt p_y.
    """

    div_u = divergence(u,v)
    div_hat = np.fft.fft2(div_u)

    phat = -div_hat/(dt*K2)
    phat[0,0] = 0.0

    px = np.real(np.fft.ifft2(1j*KX*phat))
    py = np.real(np.fft.ifft2(1j*KY*phat))

    u = u - dt*px
    v = v - dt*py

    return u,v

# project initial data to remove roundoff divergence
uvel,vvel = project(uvel,vvel,dt)

# ------------------------------------------------------------
# Save snapshots
# ------------------------------------------------------------

save_times = [0.0,2.0,5.0,8.0]
save_steps = [int(t/dt) for t in save_times]

snapshots = {0: curl(uvel,vvel)}

vorticity_history = []
time_values = []

# ------------------------------------------------------------
# Time evolution
# ------------------------------------------------------------

for n in range(1,num_steps+1):

    ux = ddx(uvel)
    uy = ddy(uvel)
    vx = ddx(vvel)
    vy = ddy(vvel)

    adv_u = uvel*ux + vvel*uy
    adv_v = uvel*vx + vvel*vy

    diff_u = nu*laplacian(uvel)
    diff_v = nu*laplacian(vvel)

    # intermediate velocity
    ustar = uvel + dt*(-adv_u + diff_u)
    vstar = vvel + dt*(-adv_v + diff_v)

    # pressure projection
    uvel,vvel = project(ustar,vstar,dt)

    if n in save_steps:
        snapshots[n] = curl(uvel,vvel)

    if n % 5 == 0:
        vorticity_history.append(curl(uvel,vvel))
        time_values.append(n*dt)

vorticity_history = np.array(vorticity_history)

# ------------------------------------------------------------
# Plot vorticity snapshots
# ------------------------------------------------------------

fig,axes = plt.subplots(1,len(save_times),figsize=(14,3))

for ax,t,n in zip(axes,save_times,save_steps):
    im = ax.imshow(
        snapshots[n],
        extent=[0,L,0,L],
        origin="lower"
    )
    ax.set_title(f"t = {t}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

# ------------------------------------------------------------
# Save snapshots
# ------------------------------------------------------------

save_times = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,8.0]
save_steps = [int(t/dt) for t in save_times]

snapshots = {0: curl(uvel,vvel)}

# ------------------------------------------------------------
# Time evolution
# ------------------------------------------------------------

for n in range(1,num_steps+1):

    ux = ddx(uvel)
    uy = ddy(uvel)
    vx = ddx(vvel)
    vy = ddy(vvel)

    adv_u = uvel*ux + vvel*uy
    adv_v = uvel*vx + vvel*vy

    diff_u = nu*laplacian(uvel)
    diff_v = nu*laplacian(vvel)

    ustar = uvel + dt*(-adv_u + diff_u)
    vstar = vvel + dt*(-adv_v + diff_v)

    uvel,vvel = project(ustar,vstar,dt)

    if n in save_steps:
        snapshots[n] = curl(uvel,vvel)

# ------------------------------------------------------------
# Save individual vorticity images
# ------------------------------------------------------------

vmin = min(np.min(snapshots[n]) for n in save_steps)
vmax = max(np.max(snapshots[n]) for n in save_steps)

for t,n in zip(save_times,save_steps):

    plt.figure(figsize=(4,4))

    plt.imshow(
        snapshots[n],
        extent=[0,L,0,L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"5-NSE_t{int(t)}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.close()
