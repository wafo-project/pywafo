

import unittest
import numpy as np
from numpy import cos, pi
import numpy.testing as npt
from numpy.testing import assert_array_almost_equal
from wafo.padua import (padua_points, example_functions, padua_fit,
                        padua_fit2,
                        padua_cubature, padua_val)


class PaduaTestCase(unittest.TestCase):

    def test_padua_points_degree0(self):
        pad = padua_points(0)
        expected = [[-1], [-1]]
        assert_array_almost_equal(pad, expected, 15)

    def test_padua_points_degree1(self):
        pad = padua_points(1)
        expected = [cos(np.r_[0, 1, 1] * pi),
                    cos(np.r_[1, 0, 2] * pi / 2)]

        assert_array_almost_equal(pad, expected, 15)

    def test_padua_points_degree2(self):
        pad = padua_points(2, domain=[0, 1, 0, 2])
        expected = [(cos(np.r_[0, 0, 1, 1, 2, 2] * pi / 2) + 1) / 2,
                    cos(np.r_[1, 3, 0, 2, 1, 3] * pi / 3) + 1]

        assert_array_almost_equal(pad, expected, 15)

    def test_testfunct(self):
        vals = [example_functions(0, 0, id_) for id_ in range(12)]
        expected = [7.664205912849231e-01, 0.7071067811865476, 0,
                    1.6487212707001282, 1.9287498479639178e-22, 1.0,
                    1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        assert_array_almost_equal(vals, expected, 15)

    def test_padua_fit_even_degree(self):
        points = padua_points(10)
        C0f, _abs_error = padua_fit(points, example_functions, 6)
        expected = np.zeros((11, 11))
        expected[0, 0] = 1
        assert_array_almost_equal(C0f, expected, 15)

    def test_padua_fit_odd_degree(self):
        points = padua_points(9)
        C0f, _abs_error = padua_fit(points, example_functions, 6)
        expected = np.zeros((10, 10))
        expected[0, 0] = 1
        assert_array_almost_equal(C0f, expected, 15)

# TODO: padua_fit2 does not work correctly
#    def test_padua_fit_odd_degree2(self):
#        points = padua_points(9)
#        C0f, _abs_error = padua_fit2(points, example_functions, 6)
#        expected = np.zeros((10, 10))
#        expected[0, 0] = 1
#        assert_array_almost_equal(C0f, expected, 15)

    def test_padua_cubature(self):
        domain = [0, 1, 0, 1]
        points = padua_points(500, domain)
        C0f, _abs_error = padua_fit(points, example_functions, 0)
        val = padua_cubature(C0f, domain)
        expected = 4.06969589491556e-01
        assert_array_almost_equal(val, expected, 15)

    def test_padua_val_unordered(self):
        domain = [0, 1, 0, 1]
        points = padua_points(20, domain)
        C0f, _abs_error = padua_fit(points, example_functions, 0)
        X = [0, 0.5, 1]

        val = padua_val(X, X, C0f, domain)
        expected = [7.664205912849228e-01, 3.2621734202884815e-01,
                    3.587865112678535e-02]
        assert_array_almost_equal(val, expected, 14)

    def test_padua_val_grid(self):
        domain = [0, 1, 0, 1]
        a, b, c, d = domain
        points = padua_points(21, domain)
        C0f, _abs_error = padua_fit(points, example_functions, 0)
        X1 = np.linspace(a, b, 2)
        X2 = np.linspace(c, d, 2)
        val = padua_val(X1, X2, C0f, domain, use_meshgrid=True)
        expected = [[7.664205912849229e-01, 1.0757071952145181e-01],
                    [2.703371615911344e-01, 3.5734971024838565e-02]]
        assert_array_almost_equal(val, expected, 14)


if __name__ == "__main__":
    npt.run_module_suite()
