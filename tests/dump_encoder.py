import numpy as np

arr = np.load("encoder_block0_out.npy")

# REMOVE batch dimension
if arr.ndim == 3:
    arr = arr[0]

print("Shape:", arr.shape)
print(arr[:5,:5])