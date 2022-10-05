import unittest
from hypothesis import given
from hypothesis.extra.numpy import mutually_broadcastable_shapes
import numpy as np
import hypothesis
import hypothesis.strategies
from numpy.testing import assert_array_equal

import torch as t


@hypothesis.strategies.composite
def where_inputs(draw, xy_dtype = int):
    # TODO: Could we do this with a ufunc sig? `where` doesn't seem to have one
    # @given(mutually_broadcastable_shapes(signature=np.where.signature))
    shapes, result_shape = draw(mutually_broadcastable_shapes(num_shapes=3))
    cond = draw(hypothesis.extra.numpy.arrays(bool, shape=shapes[0]))
    x = draw(hypothesis.extra.numpy.arrays(xy_dtype, shape=shapes[1]))
    y = draw(hypothesis.extra.numpy.arrays(xy_dtype, shape=shapes[2]))
    return cond,x,y


class TestMPS(unittest.TestCase):
    @given(where_inputs())
    @hypothesis.settings(max_examples=5000)
    # TODO: Can I unpack the tuple??
    def test_where_consistent(self, where_inputs):
        dev = "mps"

        cond, x, y = where_inputs
        t_cond, t_x, t_y = t.from_numpy(cond), t.from_numpy(x), t.from_numpy(y)

        np_out = np.where(cond, x, y)
        torch_cpu_out = t.where(t_cond, t_x, t_y)
        torch_mps_out = t.where(t_cond.to(dev), t_x.to(dev), t_y.to(dev))

        assert_array_equal(np_out, torch_cpu_out.numpy())
        assert_array_equal(np_out, torch_mps_out.cpu().numpy())


if __name__ == "__main__":
    unittest.main()