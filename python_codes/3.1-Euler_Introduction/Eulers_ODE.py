"""
Euler's Method for a Nonlinear Differential Equation

This example approximates the solution of the initial value problem

    y' = y^2,
    y(0) = 1/1.1,

on the interval [0, 1] using Euler's method.

The exact solution is

    y(x) = 1 / (1.1 - x).

The solution grows rapidly as x approaches the singularity at x = 1.1,
making this problem a useful illustration of error accumulation in an
explicit time-stepping method.
"""

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Numerical parameters
# ============================================================

x_start = 0.0
x_end = 1.0
h = 0.1

N = int((x_end - x_start) / h)


# ============================================================
# Differential equation
# ============================================================

def f(x, y):
    """
    Right-hand side of the differential equation

        y' = f(x, y) = y^2.
    """
    return y**2


def exact_solution(x):
    """
    Exact solution of the initial value problem.
    """
    return 1.0 / (1.1 - x)


# ============================================================
# Euler approximation
# ============================================================

x = np.linspace(x_start, x_end, N + 1)
y = np.zeros(N + 1)

y[0] = 1.0 / 1.1

for i in range(N):
    y[i + 1] = y[i] + h * f(x[i], y[i])


# ============================================================
# Exact solution for comparison
# ============================================================

x_exact = np.linspace(x_start, x_end, 1000)
y_exact = exact_solution(x_exact)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    x,
    y,
    "o-",
    linewidth=2,
    label=f"Euler approximation ($N={N}$)",
)

ax.plot(
    x_exact,
    y_exact,
    linewidth=2,
    label="Exact solution",
)

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Euler Method for $y' = y^2$")
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
