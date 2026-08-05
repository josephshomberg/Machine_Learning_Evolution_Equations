import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Thin-film / dewetting forward simulation
#
#     u_t = div( M(u) grad mu )
#     mu  = -eps^2 Delta u + u^3 - u
#     M(u)= u^n
#
# Periodic domain: [0,L] x [0,L]
# Method: semi-implicit Fourier treatment of fourth-order term
# ============================================================

N = 128
L = 1.0

eps = 0.018
n_mobility = 3.0

dt = 5.0e-5
T = 0.25
num_steps = int(T/dt)

x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
dx = L/N

X, Y = np.meshgrid(x, y, indexing="ij")

kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
ky = 2*np.pi*np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2
K4 = K2**2

# ------------------------------------------------------------
# Smooth random perturbation of a nearly flat film
# ------------------------------------------------------------

np.random.seed(7)

eta = np.random.randn(N, N)

for _ in range(12):
    eta = (
        eta
        + np.roll(eta, 1, axis=0)
        + np.roll(eta, -1, axis=0)
        + np.roll(eta, 1, axis=1)
        + np.roll(eta, -1, axis=1)
    )/5.0

eta = eta/np.max(np.abs(eta))

u = 0.55 + 0.12*eta
u = np.maximum(u, 0.05)

u0 = u.copy()

# ------------------------------------------------------------
# Save times
# ------------------------------------------------------------

save_times = [0.0, 0.03, 0.06, 0.09, 0.13, 0.17, 0.21, 0.25]
save_steps = [int(t/dt) for t in save_times]
save_labels = [0, 1, 2, 3, 4, 5, 6, 8]

snapshots = {0: u.copy()}

mass_values = []
time_values = []

# ------------------------------------------------------------
# Time evolution
# ------------------------------------------------------------

for step in range(1, num_steps + 1):

    uhat = np.fft.fft2(u)

    # nonlinear chemical potential part: u^3 - u
    nonlinear = u**3 - u
    nonlinear_hat = np.fft.fft2(nonlinear)

    # mobility evaluated explicitly
    mobility = np.maximum(u, 0.02)**n_mobility
    mobility_mean = np.mean(mobility)

    # semi-implicit update:
    #
    # u_t = -M eps^2 Delta^2 u + div(M grad(u^3-u))
    #
    rhs_hat = -mobility_mean*K2*nonlinear_hat

    uhat_new = (uhat + dt*rhs_hat)/(1.0 + dt*mobility_mean*eps**2*K4)

    u = np.real(np.fft.ifft2(uhat_new))

    # positivity floor for qualitative film-thickness visualization
    u = np.maximum(u, 0.02)

    if step in save_steps:
        snapshots[step] = u.copy()

    if step % 100 == 0:
        mass_values.append(np.mean(u))
        time_values.append(step*dt)

# ------------------------------------------------------------
# Common color scale
# ------------------------------------------------------------

umin = min(np.min(snapshots[s]) for s in save_steps)
umax = max(np.max(snapshots[s]) for s in save_steps)

cmap_name = "viridis"

# ------------------------------------------------------------
# Save individual images for LaTeX
# ------------------------------------------------------------

for label, step in zip(save_labels, save_steps):

    plt.figure(figsize=(4, 4))

    plt.imshow(
        snapshots[step],
        extent=[0, L, 0, L],
        origin="lower",
        cmap=cmap_name,
        vmin=umin,
        vmax=umax
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        f"5-TF_t{label}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.close()

# ------------------------------------------------------------
# Optional montage
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax, label, step, t in zip(axes.flat, save_labels, save_steps, save_times):

    im = ax.imshow(
        snapshots[step],
        extent=[0, L, 0, L],
        origin="lower",
        cmap=cmap_name,
        vmin=umin,
        vmax=umax
    )

    ax.set_title(f"t = {t:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Thin-Film Dewetting and Coarsening")

fig.colorbar(
    im,
    ax=axes.ravel().tolist(),
    shrink=0.85,
    label="film thickness"
)

plt.tight_layout()
plt.savefig("5-TF_timelapse.png", dpi=300, bbox_inches="tight")
plt.show()
