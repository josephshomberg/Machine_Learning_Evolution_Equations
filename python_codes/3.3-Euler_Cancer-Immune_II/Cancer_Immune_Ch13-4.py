"""
Simulate the cancer-immune interaction model by the forward Euler method
for several values of the parameter A, where gamma = 200 / A.

The model is
    dC/dt   = lambda_c * C * (1 - C/C_0) - mu_c * T * C
    dM1/dt  = k_1 - gamma * M1 * C/(K_1 + C) - mu * M1
    dM2/dt  = gamma * M1 * C/(K_1 + C) - mu * M2
    dT/dt   = k_T * M1/(K_2 + M2) - mu_T * T

Parameters
----------
A_values : iterable
    Values of A defining gamma = 200/A.
t0, tend : float
    Time interval.
N : int
    Number of time steps.

Returns
-------
t : ndarray
    Time grid of length N+1.
gamma_values : ndarray
    Values gamma = 200/A used in the simulations.
C, M1, M2, T : ndarray
    Arrays of shape (num_cases, N+1).
delta_t : float
    Time-step size.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def cancer_immune_ode(t, y, gamma, lambda_c, mu_c, C_0, mu, k_1, mu_T, k_T, K_1, K_2):
    C, M1, M2, T = y
    
    # Right-hand side of ODEs
    dC_dt = lambda_c * C * (1.0 - C / C_0) - mu_c * T * C
    dM1_dt = k_1 - gamma * M1 * C / (K_1 + C) - mu * M1
    dM2_dt = gamma * M1 * C / (K_1 + C) - mu * M2
    dT_dt = k_T * M1 / (K_2 + M2) - mu_T * T
    
    return [dC_dt, dM1_dt, dM2_dt, dT_dt]

def simulate_cancer_immune_parameter_sweep(
    A_values=(2, 5, 10, 100),
    t0=0.0,
    tend=600.0,
    num_evals=32769,
    lambda_c=1.0e-2,
    mu_c=1.0e-5,
    C_0=1.0e6,
    mu=0.3,
    k_1=3000.0,
    mu_T=0.2,
    k_T=3300.0,
    K_1=None,
    K_2=1.0e5,
    C_init=1.0e2,
    M1_init=5.0e4,
    M2_init=0.0,
    T_init=0.0
):
    A_values = np.asarray(A_values, dtype=float)
    if K_1 is None:
        K_1 = 0.05 * C_0
        
    num_cases = len(A_values)
    gamma_values = 200.0 / A_values
    t_eval = np.linspace(t0, tend, num_evals)
    delta_t = (tend - t0) / (num_evals - 1)
    
    C = np.zeros((num_cases, num_evals), dtype=float)
    M1 = np.zeros((num_cases, num_evals), dtype=float)
    M2 = np.zeros((num_cases, num_evals), dtype=float)
    T = np.zeros((num_cases, num_evals), dtype=float)
    
    y0 = [C_init, M1_init, M2_init, T_init]
    
    for k, gamma in enumerate(gamma_values):
        sol = solve_ivp(
            fun=lambda t, y: cancer_immune_ode(t, y, gamma, lambda_c, mu_c, C_0, mu, k_1, mu_T, k_T, K_1, K_2),
            t_span=(t0, tend),
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        C[k, :] = sol.y[0]
        M1[k, :] = sol.y[1]
        M2[k, :] = sol.y[2]
        T[k, :] = sol.y[3]
        
    return t_eval, gamma_values, C, M1, M2, T, delta_t


def make_case_labels(A_values, gamma_values):
    """
    Build legend labels for each parameter choice.
    """
    labels = []
    for A, gamma in zip(A_values, gamma_values):
        if float(A).is_integer():
            A_text = str(int(A))
        else:
            A_text = f"{A:g}"
        labels.append(rf"$A={A_text}$, $\gamma={gamma:.4g}$")
    return labels


def save_individual_plots(t, values, labels, ylabel, title, filename, outdir="images"):
    """
    Save a single figure containing one state variable for all parameter choices.
    """
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    for curve, label in zip(values, labels):
        plt.plot(t, curve, linewidth=1.5, label=label)

    plt.xlabel("Time (days)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def save_quad_plot(t, C, M1, M2, T, labels, outdir="images", filename="cancer_immune_parameter_sweep_quad.png"):
    """
    Save a 2x2 panel figure with separate vertical scales for each state variable.
    """
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    panel_data = [
        (axes[0, 0], C,  r"(a) Cancer cells $C(t)$"),
        (axes[0, 1], M1, r"(b) Pro-inflammatory macrophages $M_1(t)$"),
        (axes[1, 0], M2, r"(c) Anti-inflammatory macrophages $M_2(t)$"),
        (axes[1, 1], T,  r"(d) T-cells $T(t)$"),
    ]

    for ax, values, title in panel_data:
        for curve, label in zip(values, labels):
            ax.plot(t, curve, linewidth=1.5, label=label)
        ax.set_title(title)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Density (cells/cm$^3$)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Forward Euler parameter sweep for the cancer-immune system",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    A_values = np.array([2, 5, 10, 100], dtype=float)

    t, gamma_values, C, M1, M2, T, delta_t = simulate_cancer_immune_parameter_sweep(
        A_values=A_values
    )

    labels = make_case_labels(A_values, gamma_values)

    print(f"Step size delta_t = {delta_t:.8f}")
    print("Gamma values used:")
    for label in labels:
        print(f"  {label}")

    save_individual_plots(
        t=t,
        values=C,
        labels=labels,
        ylabel="Density (cells/cm$^3$)",
        title="Cancer cell density for varying $A$",
        filename="C_parameter_sweep.png",
        outdir="images",
    )

    save_individual_plots(
        t=t,
        values=M1,
        labels=labels,
        ylabel="Density (cells/cm$^3$)",
        title="Pro-inflammatory macrophages for varying $A$",
        filename="M1_parameter_sweep.png",
        outdir="images",
    )

    save_individual_plots(
        t=t,
        values=M2,
        labels=labels,
        ylabel="Density (cells/cm$^3$)",
        title="Anti-inflammatory macrophages for varying $A$",
        filename="M2_parameter_sweep.png",
        outdir="images",
    )

    save_individual_plots(
        t=t,
        values=T,
        labels=labels,
        ylabel="Density (cells/cm$^3$)",
        title="T-cell density for varying $A$",
        filename="T_parameter_sweep.png",
        outdir="images",
    )

    save_quad_plot(
        t=t,
        C=C,
        M1=M1,
        M2=M2,
        T=T,
        labels=labels,
        outdir="images",
        filename="cancer_immune_parameter_sweep_quad.png",
    )

    print("\nSaved figures:")
    print("  images/C_parameter_sweep.png")
    print("  images/M1_parameter_sweep.png")
    print("  images/M2_parameter_sweep.png")
    print("  images/T_parameter_sweep.png")
    print("  images/cancer_immune_parameter_sweep_quad.png")


if __name__ == "__main__":
    main()
