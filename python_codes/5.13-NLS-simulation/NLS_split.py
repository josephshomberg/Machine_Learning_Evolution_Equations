import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Nonlinear Schrodinger equation
#
#     i u_t + u_xx + kappa |u|^2 u = 0
#
# Periodic domain: [-L/2,L/2)
# Spatial method:  Fourier spectral
# Time method:     split-step Fourier / Strang splitting
# Observable:      intensity |u|^2
# ============================================================

N = 1024
L = 40.0

kappa = 1.0

dt = 0.002
T = 8.0
num_steps = int(T/dt)

x = np.linspace(-L/2,L/2,N,endpoint=False)
dx = L/N

k = 2*np.pi*np.fft.fftfreq(N,d=dx)

# ------------------------------------------------------------
# Initial condition: two localized wave packets
# ------------------------------------------------------------

A1 = 1.0
A2 = 0.9

x0 = 7.0
sigma = 1.4

c1 = 1.5
c2 = -1.2

u = (
    A1*np.exp(-((x+x0)**2)/(2*sigma**2))*np.exp(1j*c1*x)
    +
    A2*np.exp(-((x-x0)**2)/(2*sigma**2))*np.exp(1j*c2*x)
)

u0 = u.copy()

# ------------------------------------------------------------
# Split-step factors
# ------------------------------------------------------------

# For i u_t + u_xx = 0, the Fourier update is
#
#     uhat(t+dt) = exp(-i k^2 dt) uhat(t).
#
# Strang splitting uses half linear steps.
linear_half = np.exp(-0.5j*k**2*dt)

# ------------------------------------------------------------
# Times to save
# ------------------------------------------------------------

save_times = [0,1,2,3,4,5,6,8]
save_steps = [int(t/dt) for t in save_times]

snapshots = {0: np.abs(u)**2}

mass_values = []
time_values = []

# ------------------------------------------------------------
# Time evolution
# ------------------------------------------------------------

for n in range(1,num_steps+1):

    # half linear dispersive step
    uhat = np.fft.fft(u)
    uhat = linear_half*uhat
    u = np.fft.ifft(uhat)

    # full nonlinear phase step
    u = np.exp(1j*kappa*np.abs(u)**2*dt)*u

    # second half linear dispersive step
    uhat = np.fft.fft(u)
    uhat = linear_half*uhat
    u = np.fft.ifft(uhat)

    if n in save_steps:
        snapshots[n] = np.abs(u)**2

    if n % 20 == 0:
        mass = np.sum(np.abs(u)**2)*dx
        mass_values.append(mass)
        time_values.append(n*dt)

# ------------------------------------------------------------
# Common vertical scale for all snapshot plots
# ------------------------------------------------------------

rho_max = max(np.max(snapshots[n]) for n in save_steps)

# ------------------------------------------------------------
# Save individual intensity snapshots
# ------------------------------------------------------------

for t,n in zip(save_times,save_steps):

    plt.figure(figsize=(5,3))

    plt.plot(x,snapshots[n],linewidth=2)

    plt.xlim(-L/2,L/2)
    plt.ylim(0,1.05*rho_max)

    plt.xlabel("x")
    plt.ylabel(r"$|u(x,t)|^2$")
    plt.title(f"NLS intensity, t = {t}")

    plt.tight_layout()

    plt.savefig(
        f"5-NLS_t{int(t)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------------------------------------
# Optional montage for quick inspection
# ------------------------------------------------------------

fig,axes = plt.subplots(2,4,figsize=(14,6))

for ax,t,n in zip(axes.flat,save_times,save_steps):

    ax.plot(x,snapshots[n],linewidth=1.8)

    ax.set_xlim(-L/2,L/2)
    ax.set_ylim(0,1.05*rho_max)

    ax.set_title(f"t = {t}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|u|^2$")

fig.suptitle("Nonlinear Schrodinger Intensity Time-Lapse")
plt.tight_layout()

plt.savefig(
    "5-NLS_timelapse.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# Optional mass diagnostic
# ------------------------------------------------------------

plt.figure(figsize=(7,4))
plt.plot(time_values,mass_values,linewidth=2)

plt.xlabel("time")
plt.ylabel("mass")
plt.title("Numerical Mass Conservation")

plt.tight_layout()
plt.savefig("5-NLS_mass.png",dpi=300,bbox_inches="tight")
plt.show()

