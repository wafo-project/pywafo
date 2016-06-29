'''
Created on 23. okt. 2014

@author: pab
'''
import unittest
import numpy as np
from numpy import exp, Inf
from numpy.testing import assert_array_almost_equal
from wafo.integrate import gaussq, quadgr, clencurt, romberg


class TestIntegrators(unittest.TestCase):
    def test_clencurt(self):
        val, err = clencurt(np.exp, 0, 2)
        assert_array_almost_equal(val, np.expm1(2))
        self.assert_(err < 1e-10)

    def test_romberg(self):
        tol = 1e-7
        q, err = romberg(np.sqrt, 0, 10, 0, abseps=tol)
        assert_array_almost_equal(q, 2.0/3 * 10**(3./2))
        self.assert_(err < tol)


class TestGaussq(unittest.TestCase):
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
        assert_array_almost_equal(val, exp(1)-exp(0))

        a, b, y = [0, 0], [1, 1], np.array([1., 2.])
        val, _err = gaussq(lambda x, y: x * y, a, b, args=(y, ))
        assert_array_almost_equal(val, 0.5*y)

    def test_gauss_hermite(self):
        val, _err = gaussq(lambda x: x, -Inf, Inf, wfun=2)
        assert_array_almost_equal(val, 0)

    def test_gauss_laguerre(self):
        val, _err = gaussq(lambda x: x, 0, Inf, wfun=3, alpha=1)
        assert_array_almost_equal(val, 2)

    def test_gauss_jacobi(self):
        val, _err = gaussq(lambda x: x, -1, 1, wfun=4, alpha=-0.5, beta=-0.5)
        assert_array_almost_equal(val, 0)

    def test_gauss_wfun5_6(self):
        for i in [5, 6]:
            val, _err = gaussq(lambda x: x, -1, 1, wfun=i)
            assert_array_almost_equal(val, 0)

    def test_gauss_wfun7(self):
        val, _err = gaussq(lambda x: x, 0, 1, wfun=7)
        assert_array_almost_equal(val, 1.17809725)

    def test_gauss_wfun8(self):
        val, _err = gaussq(lambda x: x, 0, 1, wfun=8)
        assert_array_almost_equal(val, 1.33333333)

    def test_gauss_wfun9(self):
        val, _err = gaussq(lambda x: x, 0, 1, wfun=9)
        assert_array_almost_equal(val, 0.26666667)


class TestQuadgr(unittest.TestCase):
    def test_log(self):
        Q, err = quadgr(np.log, 0, 1)
        assert_array_almost_equal(Q, -1)
        self.assert_(err < 1e-5)

    def test_exp(self):
        Q, err = quadgr(np.exp, 0, 9999*1j*np.pi)
        assert_array_almost_equal(Q, -2.0000000000122662)
        self.assert_(err < 1.0e-8)

    def test_integral3(self):
        tol = 1e-12
        Q, err = quadgr(lambda x: np.sqrt(4-x**2), 0, 2, tol)
        assert_array_almost_equal(Q, np.pi)
        self.assert_(err < tol)
        # (3.1415926535897811, 1.5809575870662229e-13)

    def test_integral4(self):
        Q, err = quadgr(lambda x: 1./x**0.75, 0, 1)
        assert_array_almost_equal(Q, 4)
        self.assert_(err < 1.0e-12)

    def test_integrand4(self):
        tol = 1e-10
        Q, err = quadgr(lambda x: 1./np.sqrt(1-x**2), -1, 1, tol)
        assert_array_almost_equal(Q, np.pi)
        self.assert_(err < tol)
        # (3.141596056985029, 6.2146261559092864e-06)

    def test_integrand5(self):
        tol = 1e-9
        Q, err = quadgr(lambda x: np.exp(-x**2), -np.inf, np.inf, tol)

        assert_array_almost_equal(Q, np.sqrt(np.pi))
        self.assert_(err < tol)
        # (1.7724538509055152, 1.9722334876348668e-11)

    def test_integrand6(self):
        tol = 1e-9
        Q, err = quadgr(lambda x: np.cos(x)*np.exp(-x), 0, np.inf, tol)
        assert_array_almost_equal(Q, 0.5)
        self.assert_(err < tol)
        # (0.50000000000000044, 7.3296813063450372e-11)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
