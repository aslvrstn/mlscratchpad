# %%
import numpy as np
import torch as t
from typing import Tuple
import matplotlib.pyplot as plt
# %%

def test_rotation(dim: int=10, small_val: float = 0.01, num_large_vals: int = 2, orders_of_magnitude: int = 2, dtype = t.float32) -> Tuple[np.ndarray, np.ndarray]:
    """
    dim: Dimension of space
    small_val: Value for the "base" values
    num_large_values: How many big values at the start
    orders_of_magnitude: How many orders of magnitude should the large values be than the small ones
    dtype: dtype to do this all in
    """
    assert num_large_vals < dim

    # Handle both torch and numpy dtypes in this function
    backend = t if isinstance(dtype, t.dtype) else np
    backend_randn = t.randn if backend == t else np.random.randn

    # If we can't even store the large number to begin
    # with, this test isn't interesting
    if 10 ** orders_of_magnitude > backend.finfo(dtype).max:
        return (None, None)

    vec = backend.ones((dim,), dtype=dtype) * small_val
    vec[0:num_large_vals] *= 10.0 ** orders_of_magnitude

    # Generate a random rotation matrix
    rot = backend.linalg.svd(backend_randn(dim, dim), full_matrices=True)[0]
    rot = rot.to(dtype) if backend == t else rot.astype(dtype)

    # Rotate vec out and back again
    vec_result = vec @ rot @ rot.T
    return vec, vec_result

# %%
types = [t.bfloat16, np.float16, np.float32, t.float32, t.float64, np.float64]
printed = []
res = {typ: [] for typ in types}
for dtype in types:
    for i in range(20):
        num_large_vals = 3 
        out, back = test_rotation(num_large_vals=num_large_vals, orders_of_magnitude=i, dtype=dtype, small_val=1)
        if back is not None:
            # What's the std of the small values? If there's no loss in precision it would be 0.
            back_std = back[num_large_vals:].std()
            # matplotlib is unhappy plotting bfloat16
            if dtype == t.bfloat16:
                back_std = back_std.to(t.float32)
            res[dtype].append(back_std)

for k, l in res.items():
    plt.plot(l, label=k)

plt.yscale('log')
plt.xlabel("relative orders of magnitude")
plt.xticks(range(0, 20, 2))
plt.ylabel("std of results")
plt.legend()
# %%

def test_constant_rotation(dim: int=10, val: float = 10, dtype = t.float32) -> Tuple[np.ndarray, np.ndarray]:
    """
    dim: Dimension of space
    val: Constant value
    dtype: dtype to do this all in
    """
    # Handle both torch and numpy dtypes in this function
    backend = t if isinstance(dtype, t.dtype) else np
    backend_randn = t.randn if backend == t else np.random.randn

    vec = backend.ones((dim,), dtype=dtype) * val

    # Generate a random rotation matrix
    rot = backend.linalg.svd(backend_randn(dim, dim), full_matrices=True)[0]
    rot = rot.to(dtype) if backend == t else rot.astype(dtype)

    # Rotate vec out and back again
    vec_result = vec @ rot @ rot.T
    return vec, vec_result

print("t.bfloat16", test_constant_rotation(dtype=t.bfloat16))
print("np.float16", test_constant_rotation(dtype=np.float16))
print("t.float32", test_constant_rotation(dtype=t.float32))
# %%
