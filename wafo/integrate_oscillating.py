'''
Created on 20. aug. 2015

@author: pab
'''
from __future__ import division
import numpy as np
import warnings
import numdifftools as nd  # @UnresolvedImport
import numdifftools.nd_algopy as nda  # @UnresolvedImport
from numdifftools.limits import Limit  # @UnresolvedImport
from numpy import linalg
from numpy.polynomial.chebyshev import chebval, Chebyshev
from numpy.polynomial import polynomial
from wafo.misc import piecewise, findcross, ecross
from collections import namedtuple

EPS = np.finfo(float).eps
_EPS = EPS
finfo = np.finfo(float)
_TINY = finfo.tiny
_HUGE = finfo.max
dea3 = nd.dea3


class PolyBasis(object):
    @staticmethod
    def _derivative(c, m):
        return polynomial.polyder(c, m)

    @staticmethod
    def eval(t, c):
        return polynomial.polyval(t, c)

    @staticmethod
    def _coefficients(k):
        c = np.zeros(k + 1)
        c[k] = 1
        return c

    def derivative(self, t, k, n=1):
        c = self._coefficients(k)
        dc = self._derivative(c, m=n)
        return self.eval(t, dc)

    def __call__(self, t, k):
        return t**k
poly_basis = PolyBasis()


class ChebyshevBasis(PolyBasis):

    @staticmethod
    def _derivative(c, m):
        cheb = Chebyshev(c)
        dcheb = cheb.deriv(m=m)
        return dcheb.coef

    @staticmethod
    def eval(t, c):
        return chebval(t, c)

    def __call__(self, t, k):
        c = self._coefficients(k)
        return self.eval(t, c)
chebyshev_basis = ChebyshevBasis()


def richardson(Q, k):
    # license BSD
    # Richardson extrapolation with parameter estimation
    c = np.real((Q[k - 1] - Q[k - 2]) / (Q[k] - Q[k - 1])) - 1.
    # The lower bound 0.07 admits the singularity x.^-0.9
    c = max(c, 0.07)
    R = Q[k] + (Q[k] - Q[k - 1]) / c
    return R


def evans_webster_weights(omega, gg, dgg, x, basis, *args, **kwds):

    def Psi(t, k):
        return dgg(t, *args, **kwds) * basis(t, k)

    j_w = 1j * omega
    nn = len(x)
    A = np.zeros((nn, nn),  dtype=complex)
    F = np.zeros((nn,), dtype=complex)

    dbasis = basis.derivative
    lim_gg = Limit(gg)
    b1 = np.exp(j_w*lim_gg(1, *args, **kwds))
    if np.isnan(b1):
        b1 = 0.0
    a1 = np.exp(j_w*lim_gg(-1, *args, **kwds))
    if np.isnan(a1):
        a1 = 0.0

    lim_Psi = Limit(Psi)
    for k in range(nn):
        F[k] = basis(1, k)*b1 - basis(-1, k)*a1
        A[k] = (dbasis(x, k, n=1) + j_w * lim_Psi(x, k))

    LS = linalg.lstsq(A, F)
    return LS[0]


def osc_weights(omega, g, dg, x, basis, ab, *args, **kwds):
    def gg(t):
        return g(scale * t + offset, *args, **kwds)

    def dgg(t):
        return scale * dg(scale * t + offset, *args, **kwds)

    w = []

    for a, b in zip(ab[::2], ab[1::2]):
        scale = (b - a) / 2
        offset = (a + b) / 2

        w.append(evans_webster_weights(omega, gg, dgg, x, basis))

    return np.asarray(w).ravel()


class _Integrator(object):
    info = namedtuple('info', ['error_estimate', 'n'])

    def __init__(self, f, g, dg=None, a=-1, b=1, basis=chebyshev_basis, s=1,
                 precision=10, endpoints=True, full_output=False):
        self.f = f
        self.g = g
        self.dg = nd.Derivative(g) if dg is None else dg
        self.basis = basis
        self.a = a
        self.b = b
        self.s = s
        self.endpoints = endpoints
        self.precision = precision
        self.full_output = full_output


class QuadOsc(_Integrator):
    def __init__(self, f, g, dg=None, a=-1, b=1, basis=chebyshev_basis, s=15,
                 precision=10, endpoints=False, full_output=False, maxiter=17):
        self.maxiter = maxiter
        super(QuadOsc, self).__init__(f, g, dg=dg, a=a, b=b, basis=basis, s=s,
                                      precision=precision, endpoints=endpoints,
                                      full_output=full_output)

    @staticmethod
    def _change_interval_to_0_1(f, g, dg, a, b):
        def f1(t, *args, **kwds):
            den = 1-t
            return f(a + t / den, *args, **kwds) / den ** 2

        def g1(t, *args, **kwds):
            return g(a + t / (1 - t), *args, **kwds)

        def dg1(t, *args, **kwds):
            den = 1-t
            return dg(a + t / den, *args, **kwds) / den ** 2
        return f1, g1, dg1, 0., 1.

    @staticmethod
    def _change_interval_to_m1_0(f, g, dg, a, b):
        def f2(t, *args, **kwds):
            den = 1 + t
            return f(b + t / den, *args, **kwds) / den ** 2

        def g2(t, *args, **kwds):
            return g(b + t / (1 + t), *args, **kwds)

        def dg2(t, *args, **kwds):
            den = 1 + t
            return dg(b + t / den, *args, **kwds) / den ** 2
        return f2, g2, dg2, -1.0, 0.0

    @staticmethod
    def _change_interval_to_m1_1(f, g, dg, a, b):
        def f2(t, *args, **kwds):
            den = (1 - t**2)
            return f(t / den, *args, **kwds) * (1+t**2) / den ** 2

        def g2(t, *args, **kwds):
            den = (1 - t**2)
            return g(t / den, *args, **kwds)

        def dg2(t, *args, **kwds):
            den = (1 - t**2)
            return dg(t / den, *args, **kwds) * (1+t**2) / den ** 2
        return f2, g2, dg2, -1., 1.

    def _get_functions(self):
        a, b = self.a, self.b
        reverse = np.real(a) > np.real(b)
        if reverse:
            a, b = b, a
        f, g, dg = self.f, self.g, self.dg

        if a == b:
            pass
        elif np.isinf(a) | np.isinf(b):
            # Check real limits
            if ~np.isreal(a) | ~np.isreal(b) | np.isnan(a) | np.isnan(b):
                raise ValueError('Infinite intervals must be real.')
            # Change of variable
            if np.isfinite(a) & np.isinf(b):
                f, g, dg, a, b = self._change_interval_to_0_1(f, g, dg, a, b)
            elif np.isinf(a) & np.isfinite(b):
                f, g, dg, a, b = self._change_interval_to_m1_0(f, g, dg, a, b)
            else:  # -inf to inf
                f, g, dg, a, b = self._change_interval_to_m1_1(f, g, dg, a, b)

        return f, g, dg, a, b, reverse

    def __call__(self, omega, *args, **kwds):
        f, g, dg, a, b, reverse = self._get_functions()

        val, err = self._quad_osc(f, g, dg, a, b, omega, *args, **kwds)

        if reverse:
            val = -val
        if self.full_output:
            return val, err
        return val

    @staticmethod
    def _get_best_estimate(k, q0, q1, q2):
        if k >= 5:
            qv = np.hstack((q0[k], q1[k], q2[k]))
            qw = np.hstack((q0[k - 1], q1[k - 1], q2[k - 1]))
        elif k >= 3:
            qv = np.hstack((q0[k], q1[k]))
            qw = np.hstack((q0[k - 1], q1[k - 1]))
        else:
            qv = np.atleast_1d(q0[k])
            qw = q0[k - 1]
        errors = np.atleast_1d(abs(qv - qw))
        j = np.nanargmin(errors)
        return qv[j], errors[j]

    def _extrapolate(self, k, q0, q1, q2):
        if k >= 4:
            q1[k] = dea3(q0[k - 2], q0[k - 1], q0[k])[0]
            q2[k] = dea3(q1[k - 2], q1[k - 1], q1[k])[0]
        elif k >= 2:
            q1[k] = dea3(q0[k - 2], q0[k - 1], q0[k])[0]
        #         # Richardson extrapolation
        #         if k >= 4:
        #             q1[k] = richardson(q0, k)
        #             q2[k] = richardson(q1, k)
        #         elif k >= 2:
        #             q1[k] = richardson(q0, k)
        q, err = self._get_best_estimate(k, q0, q1, q2)
        return q, err

    def _quad_osc(self, f, g, dg, a, b, omega, *args, **kwds):
        if a == b:
            Q = b - a
            err = b - a
            return Q, err

        abseps = 10**-self.precision
        max_iter = self.maxiter
        basis = self.basis
        if self.endpoints:
            xq = chebyshev_extrema(self.s)
        else:
            xq = chebyshev_roots(self.s)
            # xq = tanh_sinh_open_nodes(self.s)

        # One interval
        hh = (b - a) / 2
        x = (a + b) / 2 + hh * xq      # Nodes

        dtype = complex
        Q0 = np.zeros((max_iter, 1), dtype=dtype)  # Quadrature
        Q1 = np.zeros((max_iter, 1), dtype=dtype)  # First extrapolation
        Q2 = np.zeros((max_iter, 1), dtype=dtype)  # Second extrapolation

        lim_f = Limit(f)
        ab = np.hstack([a, b])
        wq = osc_weights(omega, g, dg, xq, basis, ab, *args, **kwds)
        Q0[0] = hh * np.sum(wq * lim_f(x, *args, **kwds))

        # Successive bisection of intervals
        nq = len(xq)
        n = nq
        for k in range(1, max_iter):
            n += nq*2**k

            hh = hh / 2
            x = np.hstack([x + a, x + b]) / 2
            ab = np.hstack([ab + a, ab + b]) / 2
            wq = osc_weights(omega, g, dg, xq, basis, ab, *args, **kwds)

            Q0[k] = hh * np.sum(wq * lim_f(x, *args, **kwds))

            Q, err = self._extrapolate(k, Q0, Q1, Q2)

            convergence = (err <= abseps) | ~np.isfinite(Q)
            if convergence:
                break
        else:
            warnings.warn('Max number of iterations reached '
                          'without convergence.')

        if ~np.isfinite(Q):
            warnings.warn('Integral approximation is Infinite or NaN.')

        # The error estimate should not be zero
        err += 2 * np.finfo(Q).eps
        return Q, self.info(err, n)


def adaptive_levin_points(M, delta):
    m = M - 1
    prm = 0.5
    while prm * m / delta >= 1:
        delta = 2 * delta
    k = np.arange(M)
    x = piecewise([k < prm * m, k == np.ceil(prm * m)],
                  [-1 + k / delta, 0 * k, 1 - (m - k) / delta])
    return x


def open_levin_points(M, delta):
    return adaptive_levin_points(M+2, delta)[1:-1]


def chebyshev_extrema(M, delta=None):
    k = np.arange(M)
    x = np.cos(k * np.pi / (M-1))
    return x

_EPS = np.finfo(float).eps


def tanh_sinh_nodes(M, delta=None, tol=_EPS):
    tmax = np.arcsinh(np.arctanh(1-_EPS)*2/np.pi)
    # tmax = 3.18
    m = int(np.floor(-np.log2(tmax/max(M-1, 1)))) - 1
    h = 2.0**-m
    t = np.arange((M+1)//2+1)*h
    x = np.tanh(np.pi/2*np.sinh(t))
    k = np.flatnonzero(np.abs(x - 1) <= 10*tol)
    y = x[:k[0]+1] if len(k) else x
    return np.hstack((-y[:0:-1], y))


def tanh_sinh_open_nodes(M, delta=None, tol=_EPS):
    return tanh_sinh_nodes(M+1, delta, tol)[1:-1]


def chebyshev_roots(M, delta=None):
    k = np.arange(1, 2*M, 2) * 0.5
    x = np.cos(k * np.pi / M)
    return x


class AdaptiveLevin(_Integrator):
    '''Return integral for the Levin-type and adaptive Levin-type methods'''

    @staticmethod
    def aLevinTQ(omega, ff, gg, dgg, x, s, basis, *args, **kwds):

        def Psi(t, k):
            return dgg(t, *args, **kwds) * basis(t, k)

        j_w = 1j * omega
        nu = np.ones((len(x),), dtype=int)
        nu[0] = nu[-1] = s
        S = np.cumsum(np.hstack((nu, 0)))
        S[-1] = 0
        nn = int(S[-2])
        A = np.zeros((nn, nn),  dtype=complex)
        F = np.zeros((nn,))
        dff = Limit(nda.Derivative(ff))
        dPsi = Limit(nda.Derivative(Psi))
        dbasis = basis.derivative
        for r, t in enumerate(x):
            for j in range(S[r - 1], S[r]):
                order = ((j - S[r - 1]) % nu[r])  # derivative order
                dff.fun.n = order
                F[j] = dff(t, *args, **kwds)
                dPsi.fun.n = order
                for k in range(nn):
                    A[j, k] = (dbasis(t, k, n=order+1) + j_w * dPsi(t, k))
        k1 = np.flatnonzero(1-np.isfinite(F))
        if k1.size > 0:  # Remove singularities
            warnings.warn('Singularities detected! ')
            A[k1] = 0
            F[k1] = 0
        LS = linalg.lstsq(A, F)
        v = basis.eval([-1, 1], LS[0])

        lim_gg = Limit(gg)
        gb = np.exp(j_w * lim_gg(1, *args, **kwds))
        if np.isnan(gb):
            gb = 0
        ga = np.exp(j_w * lim_gg(-1, *args, **kwds))
        if np.isnan(ga):
            ga = 0
        NR = (v[1] * gb - v[0] * ga)
        return NR

    def _get_integration_limits(self, omega, args, kwds):
        a, b = self.a, self.b
        M = 30
        ab = [a]
        scale = (b - a) / 2
        n = 30
        x = np.linspace(a, b, n + 1)
        dg_x = np.asarray([scale * omega * self.dg(xi, *args, **kwds)
                           for xi in x])
        i10 = findcross(dg_x, M)
        i1 = findcross(dg_x, 1)
        i0 = findcross(dg_x, 0)
        im1 = findcross(dg_x, -1)
        im10 = findcross(dg_x, -M)
        x10 = ecross(x, dg_x, i10, M) if len(i10) else ()
        x1 = ecross(x, dg_x, i1, 1) if len(i1) else ()
        x0 = ecross(x, dg_x, i0, 0) if len(i0) else ()
        xm1 = ecross(x, dg_x, im1, -1) if len(im1) else ()
        xm10 = ecross(x, dg_x, im10, -M) if len(im10) else ()

        for i in np.unique(np.hstack((x10, x1, x0, xm1, xm10))):
            if x[0] < i < x[n]:
                ab.append(i)
        ab.append(b)
        return ab

    def __call__(self, omega, *args, **kwds):
        ab = self._get_integration_limits(omega, args, kwds)
        s = self.s
        val = 0
        n = 0
        err = 0
        for ai, bi in zip(ab[:-1], ab[1:]):
            vali, infoi = self._QaL(s, ai, bi, omega, *args, **kwds)
            val += vali
            err += infoi.error_estimate
            n += infoi.n
        if self.full_output:
            info = self.info(err, n)
            return val, info
        return val

    @staticmethod
    def _get_num_points(s, prec, betam):
        return 1 if s > 1 else int(prec / max(np.log10(betam + 1), 1) + 1)

    def _QaL(self, s, a, b, omega, *args, **kwds):
        '''if s>1,the integral is computed by Q_s^L'''
        scale = (b - a) / 2
        offset = (a + b) / 2
        prec = self.precision  # desired precision

        def ff(t, *args, **kwds):
            return scale * self.f(scale * t + offset, *args, **kwds)

        def gg(t, *args, **kwds):
            return self.g(scale * t + offset, *args, **kwds)

        def dgg(t, *args, **kwds):
            return scale * self.dg(scale * t + offset, *args, **kwds)
        dg_a = abs(omega * dgg(-1, *args, **kwds))
        dg_b = abs(omega * dgg(1, *args, **kwds))
        g_a = abs(omega * gg(-1, *args, **kwds))
        g_b = abs(omega * gg(1, *args, **kwds))
        delta, alpha = min(dg_a, dg_b), min(g_a, g_b)

        betam = delta   # * scale
        if self.endpoints:
            if (delta < 10 or alpha <= 10 or s > 1):
                points = chebyshev_extrema
            else:
                points = adaptive_levin_points
        elif (delta < 10 or alpha <= 10 or s > 1):
            points = chebyshev_roots
        else:
            points = open_levin_points  # tanh_sinh_open_nodes

        m = self._get_num_points(s, prec, betam)
        abseps = 10*10.0**-prec
        num_collocation_point_list = m*2**np.arange(1, 5) + 1
        basis = self.basis

        Q = 1e+300
        n = 0
        ni = 0
        for num_collocation_points in num_collocation_point_list:
            ni_old = ni
            Q_old = Q
            x = points(num_collocation_points, betam)
            ni = len(x)
            if ni > ni_old:
                Q = self.aLevinTQ(omega, ff, gg, dgg, x, s, basis, *args,
                                  **kwds)
                n += ni
                err = np.abs(Q-Q_old)
                if err <= abseps:
                    break
        info = self.info(err, n)
        return Q, info


class EvansWebster(AdaptiveLevin):
    '''Return integral for the Evans Webster method'''

    def __init__(self, f, g, dg=None, a=-1, b=1, basis=chebyshev_basis, s=8,
                 precision=10, endpoints=False, full_output=False):
        super(EvansWebster,
              self).__init__(f, g, dg=dg, a=a, b=b, basis=basis, s=s,
                             precision=precision, endpoints=endpoints,
                             full_output=full_output)

    def aLevinTQ(self, omega, ff, gg, dgg, x, s, basis, *args, **kwds):
        w = evans_webster_weights(omega, gg, dgg, x, basis, *args, **kwds)

        f = Limit(ff)(x, *args, **kwds)
        NR = np.sum(f*w)
        return NR

    def _get_num_points(self, s, prec, betam):
        return 8 if s > 1 else int(prec / max(np.log10(betam + 1), 1) + 1)


if __name__ == '__main__':
    tanh_sinh_nodes(16)
