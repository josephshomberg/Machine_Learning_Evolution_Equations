"""
Example: Euler's Method for a Nonlinear Differential Equation

We approximate the solution of the initial value problem y' = y^2, y(0) = 1/1.1
on the interval [0, 1]. The exact solution is y(x) = 1 / (1.1 - x) which grows
rapidly as x approaches 1. This makes the problem a useful demonstration of
how numerical errors accumulate when using Euler's method.
"""

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------
# Numerical parameters
# -----------------------------------------------------------
h = 0.1  # step size
N = 10  # number of Euler steps (right endpoint is x = h*N = 1)

# -----------------------------------------------------------
# Storage for the numerical solution
# -----------------------------------------------------------
x = np.zeros(N + 1)  # grid points
y = np.zeros(N + 1)  # Euler approximation

# Initial conditions
x[0] = 0.0
y[0] = 1.0 / 1.1


# -----------------------------------------------------------
# Differential equation
# -----------------------------------------------------------
def f(x, y):
    """Right-hand side of the differential equation y' = f(x,y) = y^2."""
    return y**2


# -----------------------------------------------------------
# Euler method
# -----------------------------------------------------------
# Euler update formula:
# y_{i+1} = y_i + h * f(x_i, y_i)
for i in range(N):
    x[i + 1] = (i + 1) * h
    y[i + 1] = y[i] + h * f(x[i], y[i])

# -----------------------------------------------------------
# Exact solution for comparison
# -----------------------------------------------------------
z = np.linspace(0, 1, 1000)
y_exact = 1.0 / (1.1 - z)

# -----------------------------------------------------------
# Plot the numerical and exact solutions
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(x, y, "o-", linewidth=2, label=f"Euler approximation (N = {N})")
ax.plot(z, y_exact, linewidth=2, label="Exact solution")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Euler Method for $y' = y^2$")
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.6)  # Added grid for textbook presentation

plt.tight_layout()
plt.show()
