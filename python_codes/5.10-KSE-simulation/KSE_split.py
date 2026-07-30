import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Kuramoto--Sivashinsky forward solver
#
#     u_t + u u_x + u_xx + u_xxxx = 0
#
# Periodic domain: 0 <= x < L
# Spatial method:  Fourier spectral
# Time method:     semi-implicit Euler
# ============================================================

N = 256
L = 32*np.pi
dt = 0.25
T = 300.0

num_steps = int(T/dt)

x = np.linspace(0,L,N,endpoint=False)
k = 2*np.pi*np.fft.fftfreq(N,d=L/N)

# ------------------------------------------------------------
# Smooth random Fourier initial condition
# ------------------------------------------------------------

np.random.seed(4)

u = 0.2*np.cos(2*np.pi*x/L)

for m in range(1,9):
    a = np.random.randn()
    b = np.random.randn()

    u += 0.15*(
        a*np.cos(2*np.pi*m*x/L)
        + b*np.sin(2*np.pi*m*x/L)
    )/m**2

u0 = u.copy()

# ------------------------------------------------------------
# Fourier representation and linear multiplier
# ------------------------------------------------------------

uhat = np.fft.fft(u)

# For u_t = -u u_x - u_xx - u_xxxx,
# the linear Fourier multiplier is k^2 - k^4.
linear = k**2 - k**4

# ------------------------------------------------------------
# Times to save for plotting
# ------------------------------------------------------------

save_times = [0,50,150,300]
save_steps = [int(t/dt) for t in save_times]

snapshots = {0: u0.copy()}

space_time = []
time_values = []

# ------------------------------------------------------------
# Time evolution
# ------------------------------------------------------------

for n in range(1,num_steps+1):

    u = np.real(np.fft.ifft(uhat))
    ux = np.real(np.fft.ifft(1j*k*uhat))

    nonlinear = -u*ux
    nonlinear_hat = np.fft.fft(nonlinear)

    # Semi-implicit Euler update:
    #
    #   uhat^{n+1}
    #   =
    #   (uhat^n + dt*Nhat^n)/(1 - dt*(k^2-k^4)).
    #
    # The stiff fourth-order term is handled implicitly.
    uhat = (uhat + dt*nonlinear_hat)/(1 - dt*linear)

    if n in save_steps:
        snapshots[n] = np.real(np.fft.ifft(uhat)).copy()

    if n % 4 == 0:
        space_time.append(np.real(np.fft.ifft(uhat)).copy())
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
plt.title("Kuramoto--Sivashinsky Solution Snapshots")
plt.legend()
plt.tight_layout()
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
plt.title("Kuramoto--Sivashinsky Spatiotemporal Dynamics")
plt.colorbar(label="u(x,t)")
plt.tight_layout()
plt.show()
