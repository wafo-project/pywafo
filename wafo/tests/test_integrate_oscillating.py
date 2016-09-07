'''
Created on 31. aug. 2015

@author: pab
'''
from __future__ import division
import numpy as np
import mpmath as mp
import unittest
from wafo.integrate_oscillating import (adaptive_levin_points,
                                        chebyshev_extrema,
                                        chebyshev_roots, tanh_sinh_nodes,
                                        tanh_sinh_open_nodes,
                                        AdaptiveLevin, poly_basis,
                                        chebyshev_basis,
                                        EvansWebster, QuadOsc)
# import numdifftools as nd
from numpy.testing import assert_allclose
from scipy.special import gamma, digamma
_EPS = np.finfo(float).eps


class TestBasis(unittest.TestCase):
    def test_poly(self):
        t = 1
        vals = [poly_basis.derivative(t, k, n=1) for k in range(3)]
        assert_allclose(vals, range(3))
        vals = [poly_basis.derivative(0, k, n=1) for k in range(3)]
        assert_allclose(vals, [0, 1, 0])
        vals = [poly_basis.derivative(0, k, n=2) for k in range(3)]
        assert_allclose(vals, [0, 0, 2])

    def test_chebyshev(self):
        t = 1
        vals = [chebyshev_basis.derivative(t, k, n=1) for k in range(3)]
        assert_allclose(vals, np.arange(3)**2)
        vals = [chebyshev_basis.derivative(0, k, n=1) for k in range(3)]
        assert_allclose(vals, [0, 1, 0])
        vals = [chebyshev_basis.derivative(0, k, n=2) for k in range(3)]
        assert_allclose(vals, [0, 0, 4])


class TestLevinPoints(unittest.TestCase):

    def test_adaptive(self):
        M = 11
        delta = 100
        x = adaptive_levin_points(M, delta)
        true_x = [-1., -0.99, -0.98, -0.97, -0.96,  0.,
                  0.96,  0.97,  0.98,    0.99,  1.]
        assert_allclose(x, true_x)

    def test_chebyshev_extrema(self):
        M = 11
        delta = 100
        x = chebyshev_extrema(M, delta)
        true_x = [1.000000e+00,   9.510565e-01,   8.090170e-01,   5.877853e-01,
                  3.090170e-01,   6.123234e-17,  -3.090170e-01,  -5.877853e-01,
                  -8.090170e-01,  -9.510565e-01,  -1.000000e+00]
        assert_allclose(x, true_x)

    def test_chebyshev_roots(self):
        M = 11
        delta = 100
        x = chebyshev_roots(M, delta)

        true_x = [9.89821442e-01, 9.09631995e-01, 7.55749574e-01,
                  5.40640817e-01, 2.81732557e-01, 2.83276945e-16,
                  -2.81732557e-01,  -5.40640817e-01, -7.55749574e-01,
                  -9.09631995e-01,  -9.89821442e-01]
        assert_allclose(x, true_x)

    def test_tanh_sinh_nodes(self):
        for n in 2**np.arange(1, 5) + 1:
            x = tanh_sinh_nodes(n)
            # self.assertEqual(n, len(x))

    def test_tanh_sinh_open_nodes(self):
        for n in 2**np.arange(1, 5) + 1:
            x = tanh_sinh_open_nodes(n)
            # self.assertEqual(n, len(x))


class LevinQuadrature(unittest.TestCase):
    def test_exp_4t_exp_jw_gamma_t_exp_4t(self):
        def f(t):
            return np.exp(4 * t)  # amplitude function

        def g(t):
            return t + np.exp(4 * t) * gamma(t)  # phase function

        def dg(t):
            return 1 + (4 + digamma(t)) * np.exp(4 * t) * gamma(t)
        a = 1
        b = 2
        omega = 100

        def ftot(t):
            exp4t = mp.exp(4*t)
            return exp4t * mp.exp(1j * omega * (t+exp4t*mp.gamma(t)))

        _true_val, _err = mp.quadts(ftot, [a, (a+b)/2, b], error=True)

        true_val = 0.00435354129735323908804 + 0.00202865398517716214366j
        # quad = AdaptiveLevin(f, g, dg,  a=a, b=b, s=1, full_output=True)
        for quadfun in [EvansWebster, QuadOsc, AdaptiveLevin]:
            quad = quadfun(f, g, dg,  a=a, b=b, full_output=True)
            val, info = quad(omega)
            assert_allclose(val, true_val)
            self.assert_(info.error_estimate < 1e-11)
            # assert_allclose(info.n, 9)

    def test_exp_jw_t(self):
        def g(t):
            return t

        def dg(t):
            return np.ones(np.shape(t))

        def true_F(t):
            return np.exp(1j*omega*g(t))/(1j*omega)

        val, _err = mp.quadts(g, [0, 1], error=True)
        a = 1
        b = 2
        omega = 1
        true_val = true_F(b)-true_F(a)

        for quadfun in [QuadOsc, AdaptiveLevin, EvansWebster]:
            quad = quadfun(dg, g, dg, a, b, full_output=True)
            val, info = quad(omega)

            assert_allclose(val, true_val)
            self.assert_(info.error_estimate < 1e-12)
            # assert_allclose(info.n, 21)

    def test_I1_1_p_ln_x_exp_jw_xlnx(self):
        def g(t):
            return t*np.log(t)

        def dg(t):
            return 1 + np.log(t)

        def true_F(t):
            return np.exp(1j*(omega*g(t)))/(1j*omega)

        a = 100
        b = 200
        omega = 1
        true_val = true_F(b)-true_F(a)
        for quadfun in [AdaptiveLevin, QuadOsc, EvansWebster]:
            quad = quadfun(dg, g, dg, a, b, full_output=True)

            val, info = quad(omega)

            assert_allclose(val, true_val)
            self.assert_(info.error_estimate < 1e-10)
            # assert_allclose(info.n, 11)

    def test_I4_ln_x_exp_jw_30x(self):
        n = 7

        def g(t):
            return t**n

        def dg(t):
            return n*t**(n-1)

        def f(t):
            return dg(t)*np.log(g(t))

        a = 0
        b = (2 * np.pi)**(1./n)
        omega = 30

        def ftot(t):
            return n*t**(n-1)*mp.log(t**n) * mp.exp(1j * omega * t**n)

        _true_val, _err = mp.quadts(ftot, [a, b], error=True, maxdegree=8)
        # true_val = (-0.052183048684992 - 0.193877275099872j)
        true_val = (-0.0521830486849921 - 0.193877275099871j)

        for quadfun in [QuadOsc, EvansWebster, AdaptiveLevin]:
            quad = quadfun(f, g, dg, a, b, full_output=True)
            val, info = quad(omega)
            assert_allclose(val, true_val)
            self.assert_(info.error_estimate < 1e-5)

    def test_I5_coscost_sint_exp_jw_sint(self):
        a = 0
        b = np.pi/2
        omega = 100

        def f(t):
            return np.cos(np.cos(t))*np.sin(t)

        def g(t):
            return np.sin(t)

        def dg(t):
            return np.cos(t)

        def ftot(t):
            return mp.cos(mp.cos(t)) * mp.sin(t) * mp.exp(1j * omega *
                                                          mp.sin(t))

        _true_val, _err = mp.quadts(ftot, [a, 0.5, 1, b], maxdegree=9,
                                    error=True)

        true_val = 0.0325497765499959-0.121009052128827j
        for quadfun in [QuadOsc, EvansWebster, AdaptiveLevin]:
            quad = quadfun(f, g, dg, a, b, full_output=True)

            val, info = quad(omega)

            assert_allclose(val, true_val)
            self.assert_(info.error_estimate < 1e-9)

    def test_I6_exp_jw_td_1_m_t(self):
        a = 0
        b = 1
        omega = 1

        def f(t):
            return np.ones(np.shape(t))

        def g(t):
            return t/(1-t)

        def dg(t):
            return 1./(1-t)**2

        def ftot(t):
            return mp.exp(1j * omega * t/(1-t))

        true_val = (0.3785503757641866423607342717846606761068353230802945830 +
                    0.3433779615564270328325330038583124340012440194999075192j)
        for quadfun in [QuadOsc, EvansWebster, AdaptiveLevin]:
            quad = quadfun(f, g, dg, a, b, endpoints=False, full_output=True)

            val, info = quad(omega)

            assert_allclose(val, true_val)
            self.assert_(info.error_estimate < 1e-10)

    def test_I8_cos_47pix2d4_exp_jw_x(self):
        def f(t):
            return np.cos(47*np.pi/4*t**2)

        def g(t):
            return t

        def dg(t):
            return 1

        a = -1
        b = 1
        omega = 451*np.pi/4

        true_val = 2.3328690362927e-3
        s = 15
        for quadfun in [QuadOsc, EvansWebster]:  # , AdaptiveLevin]:
            quad = quadfun(f, g, dg, a, b, s=s, endpoints=False,
                           full_output=True)
            val, _info = quad(omega)
            assert_allclose(val.real, true_val)
            s = 1 if s <= 2 else s // 2
            # self.assert_(info.error_estimate < 1e-10)
            # assert_allclose(info.n, 11)

    def test_I9_exp_tant_sec2t_exp_jw_tant(self):
        a = 0
        b = np.pi/2
        omega = 100

        def f(t):
            return np.exp(-np.tan(t))/np.cos(t)**2

        def g(t):
            return np.tan(t)

        def dg(t):
            return 1./np.cos(t)**2

        true_val = (0.0000999900009999000099990000999900009999000099990000999 +
                    0.009999000099990000999900009999000099990000999900009999j)
        for quadfun in [QuadOsc, EvansWebster, AdaptiveLevin]:
            quad = quadfun(f, g, dg, a, b, endpoints=False, full_output=True)

            val, info = quad(omega)

            assert_allclose(val, true_val)
            self.assert_(info.error_estimate < 1e-8)

    def test_exp_zdcos2t_dcos2t_exp_jw_cos_t_b_dcos2t(self):
        x1 = 20
        y1 = 50
        z1 = 10
        beta = np.abs(np.arctan(y1/x1))
        R = np.sqrt(x1**2+y1**2)

        def f(t, beta, z1):
            cos2t = np.cos(t)**2
            return np.where(cos2t == 0, 0, np.exp(-z1/cos2t)/cos2t)

        def g(t, beta, z1):
            return np.cos(t-beta)/np.cos(t)**2

        def dg(t, beta, z1=0):
            cos3t = np.cos(t)**3
            return 0.5*(3*np.sin(beta)-np.sin(beta-2*t))/cos3t

        def append_dg_zero(zeros, g1, beta):
            signs = [1, ] if np.abs(g1) <= _EPS else [-1, 1]
            for sgn1 in signs:
                tn = np.arccos(sgn1 * g1)
                if -np.pi / 2 <= tn <= np.pi / 2:
                    for sgn2 in [-1, 1]:
                        t = sgn2 * tn
                        if np.abs(dg(t, beta)) < 10*_EPS:
                            zeros.append(t)
            return zeros

        def zeros_dg(beta):
            k0 = (9*np.cos(2*beta)-7)
            if k0 < 0:  # No stationary points
                return ()
            k1 = 3*np.cos(2*beta)-5
            g0 = np.sqrt(2)*np.sqrt(np.cos(beta)**2*k0)
            zeros = []

            if g0+k1 < _EPS:
                g1 = 1./2*np.sqrt(-g0-k1)
                zeros = append_dg_zero(zeros, g1, beta)
            if _EPS < g0-k1:
                g2 = 1./2*np.sqrt(g0-k1)
                zeros = append_dg_zero(zeros, g2, beta)
            if np.abs(g0+k1) <= _EPS or np.abs(g0-k1) <= _EPS:
                zeros = append_dg_zero(zeros, 0, beta)
            return tuple(zeros)

        a = -np.pi/2
        b = np.pi/2
        omega = R

        def ftot(t):
            cos2t = mp.cos(t)**2
            return (mp.exp(-z1/cos2t) / cos2t *
                    mp.exp(1j * omega * mp.cos(t-beta)/cos2t))

        zdg = zeros_dg(beta)
        ab = (a, ) + zdg + (b, )
        true_val, _err = mp.quadts(ftot, ab, maxdegree=9, error=True)
        # true_val3, err3 = mp.quadgl(ftot, ab, maxdegree=9, error=True)
        if False:
            import matplotlib.pyplot as plt
            t = np.linspace(a, b, 5*513)
            plt.subplot(2, 1, 1)
            f2 = f(t, beta, z1)*np.exp(1j*R*g(t, beta, z1))

            true_val2 = np.trapz(f2, t)
            plt.plot(t, f2.real, label='f.real')
            plt.plot(t, f2.imag, 'r', label='f.imag')
            plt.title('integral=%g+1j%g,\n'
                      '(%g+1j%g)' % (true_val2.real, true_val2.imag,
                                     true_val.real, true_val.imag))
            plt.legend(loc='best', framealpha=0.5)
            plt.subplot(2, 1, 2)
            plt.plot(t, dg(t, beta, z1), 'r',
                     label='dg(t,b={},z={})'.format(beta, z1))
            plt.plot(t, g(t, beta, z1), label='g(t,b,z)')
            plt.hlines(0, a, b)
            plt.axis([a, b, -5, 5])
            plt.title('beta=%g' % beta)
            print(np.trapz(f2, t))
            plt.legend(loc='best', framealpha=0.5)
            plt.show('hold')
        # true_val = 0.00253186684281+0.004314054498j
        # s = 15
        for quadfun in [QuadOsc]:  # , EvansWebster]:  # ,  AdaptiveLevin]:
            # EvansWebster]:  # , AdaptiveLevin, ]:
            quad = quadfun(f, g, dg, a, b, precision=10, endpoints=False,
                           full_output=True)
            val, _info = quad(omega, beta, z1)  # @UnusedVariable
            print(quadfun.__name__)
            assert_allclose(val, complex(true_val), rtol=1e-3)
            # s = 1 if s<=1 else s//2
            pass
        # assert(False)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
