import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# PARAMETERS
# =============================================================================
M = 128
AMP = 0.02
SEED = 42

# Initialize the modern, recommended random number generator
rng = np.random.default_rng(SEED)

# =============================================================================
# GENERATE RANDOM INITIAL DATA
# =============================================================================
# Create empty grid matrix with strict float64 specification
u0 = np.zeros((M, M), dtype=np.float64)

# Populate inner grid elements with scaled uniform noise, leaving boundaries 0
u0[1:-1, 1:-1] = AMP * rng.uniform(-1.0, 1.0, size=(M - 2, M - 2))

# =============================================================================
# DISPLAY LOGS AND PLOT
# =============================================================================
print(f"u0 shape: {u0.shape}")
print(f"min/max:  {u0.min():.4f} / {u0.max():.4f}")

# Check boundaries using correct mathematical negative slicing
boundary_max_abs = np.max(
    np.abs([u0[0, :], u0[-1, :], u0[:, 0], u0[:, -1]])
)
print(f"Boundary max absolute value: {boundary_max_abs}")

# Visualize the initial data matrix
plt.figure(figsize=(6, 5))
plt.imshow(u0, cmap="coolwarm", origin="lower")
plt.colorbar(label="Amplitude")
plt.title("Random Initial Data Grid")
plt.tight_layout()
plt.show()
