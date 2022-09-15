# %%
from re import A
from tkinter import W
import numpy as np
import torch as t
from typing import Tuple, Union
import matplotlib.pyplot as plt
# %%

def as_type(arr: Union[np.ndarray, t.Tensor], dtype: Union[np.dtype, t.dtype]) -> Union[np.ndarray, t.Tensor]:
    if isinstance(arr, np.ndarray):
        return arr.astype(dtype)
    elif isinstance(arr, t.Tensor):
        return arr.to(dtype)
    else:
        raise ValueError(f"Unknown array type: {arr.dtype}")

def random_rotate_precise(arr: Union[np.ndarray, t.Tensor], downcast_in_rotated_space=True) -> Union[np.ndarray, t.Tensor]:
    # Generate a random rotation matrix of appropriate size.
    # And make the rotation matrix high precision so that nothing is lost in
    # the rotation, since that's not what we're testing
    dim = len(arr)
    rot = t.linalg.svd(t.randn((dim, dim), dtype=t.float64), full_matrices=True)[0]
    if not isinstance(arr, t.Tensor):
        rot = rot.numpy()

    high_prec_type = t.float64 if isinstance(arr, t.Tensor) else np.float64

    # Apply the rotation after first upcasting our vector
    rotated_arr = as_type(arr, high_prec_type) @ rot
    # Downcast to the type of `arr` to get the loss from representing in that type
    if downcast_in_rotated_space:
        rotated_arr = as_type(rotated_arr, arr.dtype)
        # And upcast again for the rotation back
        rotated_arr = as_type(rotated_arr, high_prec_type)
    rotated_back_arr = rotated_arr @ rot.T
    # And downcast one more time
    return as_type(rotated_back_arr, arr.dtype)


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

    # If we can't even store the large number to begin
    # with, this test isn't interesting
    if 10 ** orders_of_magnitude > backend.finfo(dtype).max:
        return (None, None)

    vec = backend.ones((dim,), dtype=dtype) * small_val
    vec[0:num_large_vals] *= 10.0 ** orders_of_magnitude

    return vec, random_rotate_precise(vec)

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

# Yes I have to write this test, because the original version of the rotation
# stuff was busted and wouldn't have passed (it did rotation in the imprecise dtype instead of upscaling.)
def test_constant_rotation(dim: int=10, val: float = 10, dtype = t.float32) -> None:
    backend = t if isinstance(dtype, t.dtype) else np

    vec = backend.ones((dim,), dtype=dtype) * val
    if backend is t:
        # Torch ends up with some weird tiny tiny differences. The tolerances
        # on this check are tighter than the effect we're looking for
        assert t.allclose(vec, random_rotate_precise(vec, False))
    else:
        np.testing.assert_array_equal(vec, random_rotate_precise(vec, False))

test_constant_rotation(dtype=t.bfloat16)
test_constant_rotation(dtype=np.float16)
test_constant_rotation(dtype=t.float32)
test_constant_rotation(dtype=np.float32)
# %%
def test_with_separate_rotations(dim: int=10, num_large_vals: int = 2, orders_of_magnitude=2, dtype=t.float32):
    backend = t if isinstance(dtype, t.dtype) else np
    vec = backend.ones((dim,), dtype=dtype)
    vec[0:num_large_vals] *= 10.0 ** orders_of_magnitude

    rot = t.linalg.svd(t.randn((dim, dim), dtype=t.float64), full_matrices=True)[0]
    if backend is np:
        rot = rot.numpy()

    high_prec_type = t.float64 if backend is t else np.float64

    # Rotate all as one
    rotated_vec = as_type(vec, high_prec_type) @ rot
    # Then as completely separate features. This will cause the separate
    # features to not interfere (be combined into a single float) and so should
    # let us measure the loss coming from downcasting in a rotated basis separately
    # from the loss from combining representations
    diag_vec = vec.diag() if backend is t else np.diag(vec)
    rotated_split_vec = as_type(diag_vec, high_prec_type) @ rot

    # Make sure that splitting and recombining is equivalent
    # np.testing.assert_array_almost_equal(rotated_vec, rotated_split_vec.sum(0), decimal=10)

    orig_downcast_together = as_type(as_type(rotated_vec, dtype), high_prec_type) @ rot.T
    orig_downcast_separate = as_type(as_type(rotated_split_vec, dtype), high_prec_type) @ rot.T

    # The separated version should recover a higher precision version of the
    # original along the diagonal, and the off diagonals should be near zero.
    return orig_downcast_together, orig_downcast_separate.diagonal()


test_with_separate_rotations(dim=10, orders_of_magnitude=5, dtype=np.float16)
test_with_separate_rotations(dim=10, orders_of_magnitude=5, dtype=t.float32)

types = [t.bfloat16, np.float16, np.float32, t.float32, t.float64, np.float64]
printed = []
back_together_stds = {typ: [] for typ in types}
back_separate_stds = {typ: [] for typ in types}
for dtype in types:
    for i in range(20):
        num_large_vals = 3 
        back_together, back_separate = test_with_separate_rotations(num_large_vals=num_large_vals, orders_of_magnitude=i, dtype=dtype)
        if back_separate is not None:
            # What's the std of the small values? If there's no loss in precision it would be 0.
            back_together_std = back_together[num_large_vals:].std()
            back_separate_std = back_separate[num_large_vals:].std()
            # matplotlib is unhappy plotting bfloat16
            if dtype == t.bfloat16:
                back_together_std = back_together_std.to(t.float32)
                back_separate_std = back_separate_std.to(t.float32)
            back_together_stds[dtype].append(back_together_std)
            back_separate_stds[dtype].append(back_separate_std)

for k, l in back_separate_stds.items():
    plt.plot(l, label=k)

plt.yscale('log')
plt.xlabel("relative orders of magnitude")
plt.xticks(range(0, 20, 2))
plt.ylabel("std of results (not combined in rotated basis)")
plt.title("Not combined in rotated basis")
plt.legend()

plt.figure()

for k, l in back_together_stds.items():
    plt.plot(l, label=k)

plt.yscale('log')
plt.xlabel("relative orders of magnitude")
plt.xticks(range(0, 20, 2))
plt.ylabel("std of results")
plt.title("Combined in rotated basis")
plt.legend()
# %%