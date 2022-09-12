# %%
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
# %%

def test_rotation(dim: int=10, small_val: float = 0.01, num_large_vals: int = 2, orders_of_magnitude: int = 2, dtype = np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """
    dim: Dimension of space
    small_val: Value for the "base" values
    num_large_values: How many big values at the start
    orders_of_magnitude: How many orders of magnitude should the large values be than the small ones
    dtype: dtype to do this all in
    """
    assert num_large_vals < dim

    # If we can't even store the large number to begin
    # with, this test isn't interesting
    if 10 ** orders_of_magnitude > np.finfo(dtype).max:
        return (None, None)

    vec = np.ones((dim,), dtype=dtype) * small_val
    vec[0:num_large_vals] *= 10.0 ** orders_of_magnitude

    # Generate a random rotation matrix
    rot = np.linalg.svd(np.random.randn(dim, dim), full_matrices=True)[0].astype(dtype)

    # Rotate vec out and back again
    vec_result = vec @ rot @ rot.T
    return vec, vec_result

# %%
# TODO: Need to use torch for bfloat16?
types = [np.float16, np.float32, np.float64]
res = {typ: [] for typ in types}
for dtype in types:
    for i in range(20):
        num_large_vals = 3 
        out, back = test_rotation(num_large_vals=num_large_vals, orders_of_magnitude=i, dtype=dtype, small_val=1)
        if back is not None:
            back_std = back[num_large_vals:].std()
            res[dtype].append(back_std)

for k, l in res.items():
    plt.plot(l, label=k)

plt.yscale('log')
plt.xlabel("relative orders of magnitude")
plt.ylabel("std of results")
plt.legend()
# %%
