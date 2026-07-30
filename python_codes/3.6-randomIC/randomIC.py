import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================
M = 128
amp = 0.02
seed = 42

np.random.seed(seed)

# ============================================================
# GENERATE RANDOM INITIAL DATA
# ============================================================
u0 = np.zeros((M, M), dtype=np.float64)
u0[1:-1, 1:-1] = amp * np.random.uniform(-1.0, 1.0, size=(M - 2, M - 2))

# ============================================================
# DISPLAY
# ============================================================
print("u0 shape:", u0.shape)
print("min/max:", u0.min(), u0.max())
print("boundary max abs:", np.max(np.abs([
    u0[0, :], u0[-1, :], u0[:, 0], u0[:, -1]
])))

plt.imshow(u0, cmap="coolwarm", origin="lower")
plt.colorbar()
plt.title("Random Initial Data")
plt.show()
