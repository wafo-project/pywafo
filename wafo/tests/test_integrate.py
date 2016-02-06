'''
Created on 23. okt. 2014

@author: pab
'''
import unittest
import numpy as np
from numpy import exp, Inf
from numpy.testing import assert_array_almost_equal
from wafo.integrate import gaussq


class Gaussq(unittest.TestCase):
    '''
    1 : p(x) = 1                       a =-1,   b = 1   Gauss-Legendre
    2 : p(x) = exp(-x^2)               a =-inf, b = inf Hermite
    3 : p(x) = x^alpha*exp(-x)         a = 0,   b = inf Laguerre
    4 : p(x) = (x-a)^alpha*(b-x)^beta  a =-1,   b = 1 Jacobi
    5 : p(x) = 1/sqrt((x-a)*(b-x)),    a =-1,   b = 1 Chebyshev 1'st kind
    6 : p(x) = sqrt((x-a)*(b-x)),      a =-1,   b = 1 Chebyshev 2'nd kind
    7 : p(x) = sqrt((x-a)/(b-x)),      a = 0,   b = 1
    8 : p(x) = 1/sqrt(b-x),            a = 0,   b = 1
    9 : p(x) = sqrt(b-x),              a = 0,   b = 1
    '''

    def test_gauss_legendre(self):
        val, _err = gaussq(exp, 0, 1)
        self.assertAlmostEqual(val, exp(1)-exp(0))

        a, b, y = [0, 0], [1, 1], np.array([1., 2.])
        val, _err = gaussq(lambda x, y: x * y, a, b, args=(y, ))
        assert_array_almost_equal(val, 0.5*y)

    def test_gauss_hermite(self):
        f = lambda x: x
        val, _err = gaussq(f, -Inf, Inf, wfun=2)
        self.assertAlmostEqual(val, 0)

    def test_gauss_laguerre(self):
        f = lambda x: x
        val, _err = gaussq(f, 0, Inf, wfun=3, alpha=1)
        self.assertAlmostEqual(val, 2)

    def test_gauss_jacobi(self):
        f = lambda x: x
        val, _err = gaussq(f, -1, 1, wfun=4, alpha=-0.5, beta=-0.5)
        self.assertAlmostEqual(val, 0)

    def test_gauss_wfun5_6(self):
        f = lambda x: x
        for i in [5, 6]:
            val, _err = gaussq(f, -1, 1, wfun=i)
            self.assertAlmostEqual(val, 0)

    def test_gauss_wfun7(self):
        f = lambda x: x
        val, _err = gaussq(f, 0, 1, wfun=7)
        self.assertAlmostEqual(val, 1.17809725)

    def test_gauss_wfun8(self):
        f = lambda x: x
        val, _err = gaussq(f, 0, 1, wfun=8)
        self.assertAlmostEqual(val, 1.33333333)

    def test_gauss_wfun9(self):
        f = lambda x: x
        val, _err = gaussq(f, 0, 1, wfun=9)
        self.assertAlmostEqual(val, 0.26666667)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
