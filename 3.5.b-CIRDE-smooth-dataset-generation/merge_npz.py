import numpy as np
import os

DATA_DIR = 'data'
OUTFILE = 'dataset_128x128_DBC_kappa=4.7_count=XXXXX.npz'

def normalize(x):
    x = np.asarray(x)

    # If shape (N, 1, 128, 128) → squeeze the channel
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0, :, :]   # → (N,128,128)
        return x

    # If (N,128,128,1) → squeeze last dim
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[:, :, :, 0]   # → (N,128,128)
        return x

    # If (1,128,128) → add batch dim
    if x.ndim == 3 and x.shape[0] == 1:
        return x  # already (1,128,128)

    # If (128,128) → add batch dim
    if x.ndim == 2:
        x = x[np.newaxis, :, :]
        return x

    # If already (N,128,128)
    if x.ndim == 3:
        return x

    raise ValueError(f"Unhandled shape {x.shape}")


# ---------------------------------------------------------
# Merge all .npz files
# ---------------------------------------------------------

npz_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npz')])
print(f"Found {len(npz_files)} files.")

all_src = []
all_tar = []

for fname in npz_files:
    path = os.path.join(DATA_DIR, fname)
    data = np.load(path)

    src = normalize(data['src'])
    tar = normalize(data['tar'])

    all_src.append(src)
    all_tar.append(tar)

X_src = np.concatenate(all_src, axis=0)
X_tar = np.concatenate(all_tar, axis=0)

print("Final merged shapes:", X_src.shape, X_tar.shape)

np.savez_compressed(OUTFILE, src=X_src, tar=X_tar)
print(f"Merged dataset saved to: {OUTFILE}")
