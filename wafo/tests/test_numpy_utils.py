from numpy.testing import (
    TestCase, assert_, assert_array_equal, assert_raises,)
#     run_module_suite,
#      assert_allclose, assert_array_max_ulp, assert_warns,
#     assert_equal, assert_array_almost_equal, assert_almost_equal,
#     )

import unittest as local_unittest
from wafo.numpy_utils import (rotation_matrix, rotate_2d, spaceline,
                              args_flat, sub2index, index2sub, piecewise)
import numpy as np


class TestPiecewise(TestCase):
    def test_condition_is_single_bool_list(self):
        assert_raises(ValueError, piecewise, [0, 0], [True, False], [1])

    def test_condition_is_list_of_single_bool_list(self):
        x = piecewise([0, 0], [[True, False]], [1])
        assert_array_equal(x, [1, 0])

    def test_conditions_is_list_of_single_bool_array(self):
        x = piecewise([0, 0], [np.array([True, False])], [1])
        assert_array_equal(x, [1, 0])

    def test_condition_is_single_int_array(self):
        assert_raises(ValueError, piecewise, [0, 0], np.array([1, 0]), [1])

    def test_condition_is_list_of_single_int_array(self):
        x = piecewise([0, 0], [np.array([1, 0])], [1])
        assert_array_equal(x, [1, 0])

    def test_simple(self):
        x = piecewise([0, 0], [[False, True]], [lambda x:-1])
        assert_array_equal(x, [0, -1])

        x = piecewise([1, 2], [[True, False], [False, True]], [3, 4])
        assert_array_equal(x, [3, 4])

    def test_default(self):
        # No value specified for x[1], should be 0
        x = piecewise([1, 2], [[True, False]], [2])
        assert_array_equal(x, [2, 0])

        # Should set x[1] to 3
        x = piecewise([1, 2], [[True, False]], [2, 3])
        assert_array_equal(x, [2, 3])

    def test_0d(self):
        x = np.array(3)
        y = piecewise(x, [x > 3], [4, 0])
        assert_(y.ndim == 0)
        assert_(y == 0)

        x = 5
        y = piecewise(x, [[True], [False]], [1, 0])
        assert_(y.ndim == 0)
        assert_(y == 1)

    def test_abs_function(self):
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise((x,), [x < 0, x >= 0], [lambda x: -x, lambda x: x])
        assert_array_equal(vals,
                           [2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    def test_abs_function_with_scalar(self):
        x = np.array(-2.5)
        vals = piecewise((x,), [x < 0, x >= 0], [lambda x: -x, lambda x: x])
        assert_(vals == 2.5)

    def test_otherwise_condition(self):
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise((x,), [x < 0, ], [lambda x: -x, lambda x: x])
        assert_array_equal(vals, [2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    def test_passing_further_args_to_fun(self):
        def fun0(x, y, scale=1.):
            return -x*y/scale

        def fun1(x, y, scale=1.):
            return x*y/scale
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise((x,), [x < 0, ], [fun0, fun1], args=(2.,), scale=2.)
        assert_array_equal(vals, [2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    def test_step_function(self):
        x = np.linspace(-2.5, 2.5, 6)
        vals = piecewise(x, [x < 0, x >= 0], [-1, 1])
        assert_array_equal(vals, [-1., -1., -1.,  1.,  1.,  1.])

    def test_step_function_with_scalar(self):
        x = 1
        vals = piecewise(x, [x < 0, x >= 0], [-1, 1])
        assert_(vals == 1)

    def test_function_with_two_args(self):
        x = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, x)
        vals = piecewise(
            (X, Y), [X * Y < 0, ], [lambda x, y: -x * y, lambda x, y: x * y])
        assert_array_equal(vals, [[4.,  2., -0.,  2.,  4.],
                                  [2.,  1., -0.,  1.,  2.],
                                  [-0., -0.,  0.,  0.,  0.],
                                  [2.,  1.,  0.,  1.,  2.],
                                  [4.,  2.,  0.,  2.,  4.]])

    def test_fill_value_and_function_with_two_args(self):
        x = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, x)
        vals = piecewise((X, Y), [X * Y < -0.5, X * Y > 0.5],
                         [lambda x, y: -x * y, lambda x, y: x * y],
                         fill_value=np.nan)
        nan = np.nan
        assert_array_equal(vals, [[4.,   2.,  nan,   2.,   4.],
                                  [2.,   1.,  nan,   1.,   2.],
                                  [nan,  nan,  nan,  nan,  nan],
                                  [2.,   1.,  nan,   1.,   2.],
                                  [4.,   2.,  nan,   2.,   4.]])

    def test_fill_value2_and_function_with_two_args(self):
        x = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, x)
        vals = piecewise((X, Y), [X * Y < -0.5, X * Y > 0.5],
                         [lambda x, y: -x * y, lambda x, y: x * y, np.nan])
        nan = np.nan
        assert_array_equal(vals, [[4.,   2.,  nan,   2.,   4.],
                                  [2.,   1.,  nan,   1.,   2.],
                                  [nan,  nan,  nan,  nan,  nan],
                                  [2.,   1.,  nan,   1.,   2.],
                                  [4.,   2.,  nan,   2.,   4.]])


class TestRotationMatrix(TestCase):

    def test_h0_p0_r0(self):
        vals = rotation_matrix(heading=0, pitch=0, roll=0).tolist()
        truevals = [[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]]
        self.assertListEqual(vals, truevals)

    def test_h180_p0_r0(self):
        vals = rotation_matrix(heading=180, pitch=0, roll=0).tolist()
        truevals = [[-1.0, -1.2246467991473532e-16, 0.0],
                    [1.2246467991473532e-16, -1.0, 0.0],
                    [-0.0, 0.0, 1.0]]
        self.assertListEqual(vals, truevals)

    def test_h0_p180_r0(self):
        vals = rotation_matrix(heading=0, pitch=180, roll=0).tolist()
        truevals = [[-1.0, 0.0, 1.2246467991473532e-16],
                    [-0.0, 1.0, 0.0],
                    [-1.2246467991473532e-16, -0.0, -1.0]]
        self.assertListEqual(vals, truevals)

    def test_h0_p0_r180(self):
        vals = rotation_matrix(heading=0, pitch=180, roll=0).tolist()
        truevals = [[-1.0, 0.0, 1.2246467991473532e-16],
                    [-0.0, 1.0, 0.0],
                    [-1.2246467991473532e-16, -0.0, -1.0]]
        self.assertListEqual(vals, truevals)


class TestRotate2d(TestCase):

    def test_rotate_0deg(self):
        vals = list(rotate_2d(x=1, y=0, angle_deg=0))
        truevals = [1.0, 0.0]
        self.assertListEqual(vals, truevals)

    def test_rotate_90deg(self):
        vals = list(rotate_2d(x=1, y=0, angle_deg=90))
        truevals = [6.123233995736766e-17, 1.0]
        self.assertListEqual(vals, truevals)

    def test_rotate_180deg(self):
        vals = list(rotate_2d(x=1, y=0, angle_deg=180))
        truevals = [-1.0, 1.2246467991473532e-16]
        self.assertListEqual(vals, truevals)

    def test_rotate_360deg(self):
        vals = list(rotate_2d(x=1, y=0, angle_deg=360))
        truevals = [1.0, -2.4492935982947064e-16]
        self.assertListEqual(vals, truevals)


class TestSpaceLine(TestCase):

    def test_space_line(self):
        vals = spaceline((2, 0, 0), (3, 0, 0), num=5).tolist()
        truevals = [[2., 0., 0.],
                    [2.25, 0., 0.],
                    [2.5, 0., 0.],
                    [2.75, 0., 0.],
                    [3., 0., 0.]]
        self.assertListEqual(vals, truevals)


class TestArgsFlat(TestCase):

    def test_1_vector_and_2_scalar_args(self):
        x = [1, 2, 3]
        pos, c_shape = args_flat(x, 2, 3)
        truepos = [[1, 2, 3],
                   [2, 2, 3],
                   [3, 2, 3]]
        truec_shape = [3, ]
        self.assertListEqual(pos.tolist(), truepos)
        self.assertListEqual(list(c_shape), truec_shape)

    def test_1_vector_args(self):
        pos1, c_shape1 = args_flat([1, 2, 3])
        truepos1 = [[1, 2, 3]]
        truec_shape1 = None
        self.assertListEqual(pos1.tolist(), truepos1)
        self.assertIs(c_shape1, truec_shape1)

    def test_3_scalar_args(self):
        pos1, c_shape1 = args_flat(1, 2, 3)
        truepos1 = [[1, 2, 3]]
        truec_shape1 = []
        self.assertListEqual(pos1.tolist(), truepos1)
        self.assertListEqual(list(c_shape1), truec_shape1)

    def test_3_scalar_args_version2(self):
        pos1, c_shape1 = args_flat([1], 2, 3)
        truepos1 = [[1, 2, 3]]
        truec_shape1 = [1, ]
        self.assertListEqual(pos1.tolist(), truepos1)
        self.assertListEqual(list(c_shape1), truec_shape1)


class TestSub2index2Sub(TestCase):

    def test_sub2index_and_index2sub(self):
        shape = (3, 3, 4)
        a = np.arange(np.prod(shape)).reshape(shape)
        trueval = a[1, 2, 3]
        order = 'C'
        i = sub2index(shape, 1, 2, 3, order=order)
        self.assertEquals(i, 23)

        val = a.ravel(order)[i]
        self.assertEquals(val, trueval)

        sub = index2sub(shape, i, order=order)
        for j, true_sub_j in enumerate([[1], [2], [3]]):
            self.assertEquals(sub[j].tolist(), true_sub_j)


if __name__ == '__main__':
    runner = local_unittest.TextTestRunner()  # get_test_runner()
    local_unittest.main(testRunner=runner)
