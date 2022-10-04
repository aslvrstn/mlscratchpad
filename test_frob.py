# %%

import torch as t


def test_thing(A, B):
    return (
        t.linalg.matrix_norm(A @ B) / t.linalg.matrix_norm(A) / t.linalg.matrix_norm(B)
    )


A = t.randn(5).repeat(5).reshape(5, 5).T
B = t.randn(5).repeat(5).reshape(5, 5)

print(test_thing(A, B))
print(A)
print(B)
# %%
