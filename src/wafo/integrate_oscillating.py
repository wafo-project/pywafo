"""
Created on 20. aug. 2015

@author: pab
"""
from collections import namedtuple
import warnings
import numdifftools as nd
import numdifftools.nd_algopy as nda
from numdifftools.extrapolation import dea3
from numdifftools.limits import Limit
import numpy as np
from numpy import linalg
from numpy.polynomial.chebyshev import chebval, Chebyshev
from numpy.polynomial import polynomial
from wafo.misc import piecewise, findcross, ecross

_FINFO = np.finfo(float)
EPS = _FINFO.eps
_EPS = EPS
_TINY = _FINFO.tiny
_HUGE = _FINFO.max


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


def _assert_warn(cond, msg):
    if not cond:
        warnings.warn(msg)


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
        d_c = self._derivative(c, m=n)
        return self.eval(t, d_c)

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


def richardson(q_val, k):
    # license BSD
    # Richardson extrapolation with parameter estimation
    c = np.real((q_val[k - 1] - q_val[k - 2]) / (q_val[k] - q_val[k - 1])) - 1.
    # The lower bound 0.07 admits the singularity x.^-0.9
    c = max(c, 0.07)
    return q_val[k] + (q_val[k] - q_val[k - 1]) / c


def evans_webster_weights(omega, g, d_g, x, basis, *args, **kwds):

    def psi(t, k):
        return d_g(t, *args, **kwds) * basis(t, k)

    j_w = 1j * omega
    n = len(x)
    a_matrix = np.zeros((n, n), dtype=complex)
    rhs = np.zeros((n,), dtype=complex)

    dbasis = basis.derivative
    lim_g = Limit(g)
    b_1 = np.exp(j_w * lim_g(1, *args, **kwds))
    if np.isnan(b_1):
        b_1 = 0.0
    a_1 = np.exp(j_w * lim_g(-1, *args, **kwds))
    if np.isnan(a_1):
        a_1 = 0.0

    lim_psi = Limit(psi)
    for k in range(n):
        rhs[k] = basis(1, k) * b_1 - basis(-1, k) * a_1
        a_matrix[k] = (dbasis(x, k, n=1) + j_w * lim_psi(x, k))

    solution = linalg.lstsq(a_matrix, rhs, rcond=None)
    return solution[0]


def osc_weights(omega, g, d_g, x, basis, a_b, *args, **kwds):
    def _g(t):
        return g(scale * t + offset, *args, **kwds)

    def _d_g(t):
        return scale * d_g(scale * t + offset, *args, **kwds)

    w = []

    for a, b in zip(a_b[::2], a_b[1::2]):
        scale = (b - a) / 2
        offset = (a + b) / 2

        w.append(evans_webster_weights(omega, _g, _d_g, x, basis))

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
    def _change_interval_to_0_1(f, g, d_g, a, _b):
        def f_01(t, *args, **kwds):
            den = 1 - t
            return f(a + t / den, *args, **kwds) / den ** 2

        def g_01(t, *args, **kwds):
            return g(a + t / (1 - t), *args, **kwds)

        def d_g_01(t, *args, **kwds):
            den = 1 - t
            return d_g(a + t / den, *args, **kwds) / den ** 2
        return f_01, g_01, d_g_01, 0., 1.

    @staticmethod
    def _change_interval_to_m1_0(f, g, d_g, _a, b):
        def f_m10(t, *args, **kwds):
            den = 1 + t
            return f(b + t / den, *args, **kwds) / den ** 2

        def g_m10(t, *args, **kwds):
            return g(b + t / (1 + t), *args, **kwds)

        def d_g_m10(t, *args, **kwds):
            den = 1 + t
            return d_g(b + t / den, *args, **kwds) / den ** 2
        return f_m10, g_m10, d_g_m10, -1.0, 0.0

    @staticmethod
    def _change_interval_to_m1_1(f, g, d_g, _a, _b):
        def f_m11(t, *args, **kwds):
            den = (1 - t**2)
            return f(t / den, *args, **kwds) * (1 + t**2) / den ** 2

        def g_m11(t, *args, **kwds):
            den = (1 - t**2)
            return g(t / den, *args, **kwds)

        def d_g_m11(t, *args, **kwds):
            den = (1 - t**2)
            return d_g(t / den, *args, **kwds) * (1 + t**2) / den ** 2
        return f_m11, g_m11, d_g_m11, -1., 1.

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
    def _get_best_estimate(k, q_0, q_1, q_2):
        if k >= 5:
            q_v = np.hstack((q_0[k], q_1[k], q_2[k]))
            q_w = np.hstack((q_0[k - 1], q_1[k - 1], q_2[k - 1]))
        elif k >= 3:
            q_v = np.hstack((q_0[k], q_1[k]))
            q_w = np.hstack((q_0[k - 1], q_1[k - 1]))
        else:
            q_v = np.atleast_1d(q_0[k])
            q_w = q_0[k - 1]
        errors = np.atleast_1d(abs(q_v - q_w))
        j = np.nanargmin(errors)
        return q_v[j], errors[j]

    def _extrapolate(self, k, q_0, q_1, q_2):
        if k >= 4:
            q_1[k] = dea3(q_0[k - 2], q_0[k - 1], q_0[k])[0]
            q_2[k] = dea3(q_1[k - 2], q_1[k - 1], q_1[k])[0]
        elif k >= 2:
            q_1[k] = dea3(q_0[k - 2], q_0[k - 1], q_0[k])[0]
        #         # Richardson extrapolation
        #         if k >= 4:
        #             q_1[k] = richardson(q_0, k)
        #             q_2[k] = richardson(q_1, k)
        #         elif k >= 2:
        #             q_1[k] = richardson(q_0, k)
        q, err = self._get_best_estimate(k, q_0, q_1, q_2)
        return q, err

    def _quad_osc(self, f, g, dg, a, b, omega, *args, **kwds):
        if a == b:
            q_val = b - a
            err = np.abs(b - a)
            return q_val, err

        abseps = 10**-self.precision
        max_iter = self.maxiter
        basis = self.basis
        if self.endpoints:
            x_n = chebyshev_extrema(self.s)
        else:
            x_n = chebyshev_roots(self.s)
            # x_n = tanh_sinh_open_nodes(self.s)

        # One interval
        hh = (b - a) / 2
        x = (a + b) / 2 + hh * x_n      # Nodes

        dtype = complex
        val0 = np.zeros((max_iter, 1), dtype=dtype)  # Quadrature
        val1 = np.zeros((max_iter, 1), dtype=dtype)  # First extrapolation
        val2 = np.zeros((max_iter, 1), dtype=dtype)  # Second extrapolation

        lim_f = Limit(f)
        a_b = np.hstack([a, b])
        wq = osc_weights(omega, g, dg, x_n, basis, a_b, *args, **kwds)
        val0[0] = hh * np.sum(wq * lim_f(x, *args, **kwds))

        # Successive bisection of intervals
        nq = len(x_n)
        n = nq
        for k in range(1, max_iter):
            n += nq * 2**k

            hh = hh / 2
            x = np.hstack([x + a, x + b]) / 2
            a_b = np.hstack([a_b + a, a_b + b]) / 2
            wq = osc_weights(omega, g, dg, x_n, basis, a_b, *args, **kwds)

            val0[k] = hh * np.sum(wq * lim_f(x, *args, **kwds))

            q_val, err = self._extrapolate(k, val0, val1, val2)

            converged = (err <= abseps) | ~np.isfinite(q_val)
            if converged:
                break
        _assert_warn(converged, 'Max number of iterations reached '
                     'without convergence.')
        _assert_warn(np.isfinite(q_val),
                     'Integral approximation is Infinite or NaN.')

        # The error estimate should not be zero
        err += 2 * np.finfo(q_val).eps
        return q_val, self.info(err, n)


def adaptive_levin_points(m, delta):
    m_1 = m - 1
    prm = 0.5
    while prm * m_1 / delta >= 1:
        delta = 2 * delta
    k = np.arange(m)
    x = piecewise([k < prm * m_1, k == np.ceil(prm * m_1)],
                  [-1 + k / delta, 0 * k, 1 - (m_1 - k) / delta])
    return x


def open_levin_points(m, delta):
    return adaptive_levin_points(m + 2, delta)[1:-1]


def chebyshev_extrema(m, delta=None):
    k = np.arange(m)
    x = np.cos(k * np.pi / (m - 1))
    return x


def tanh_sinh_nodes(m, delta=None, tol=_EPS):
    tmax = np.arcsinh(np.arctanh(1 - _EPS) * 2 / np.pi)
    # tmax = 3.18
    m_1 = int(np.floor(-np.log2(tmax / max(m - 1, 1)))) - 1
    h = 2.0**-m_1
    t = np.arange((m + 1) // 2 + 1) * h
    x = np.tanh(np.pi / 2 * np.sinh(t))
    k = np.flatnonzero(np.abs(x - 1) <= 10 * tol)
    y = x[:k[0] + 1] if len(k) else x
    return np.hstack((-y[:0:-1], y))


def tanh_sinh_open_nodes(m, delta=None, tol=_EPS):
    return tanh_sinh_nodes(m + 1, delta, tol)[1:-1]


def chebyshev_roots(m, delta=None):
    k = np.arange(1, 2 * m, 2) * 0.5
    x = np.cos(k * np.pi / m)
    return x


class AdaptiveLevin(_Integrator):
    """Return integral for the Levin-type and adaptive Levin-type methods"""

    @staticmethod
    def _a_levin(omega, f, g, d_g, x, s, basis, *args, **kwds):

        def psi(t, k):
            return d_g(t, *args, **kwds) * basis(t, k)

        j_w = 1j * omega
        nu = np.ones((len(x),), dtype=int)
        nu[0] = nu[-1] = s
        S = np.cumsum(np.hstack((nu, 0)))
        S[-1] = 0
        n = int(S[-2])
        a_matrix = np.zeros((n, n), dtype=complex)
        rhs = np.zeros((n,))
        dff = Limit(nda.Derivative(f))
        d_psi = Limit(nda.Derivative(psi))
        dbasis = basis.derivative
        for r, t in enumerate(x):
            for j in range(S[r - 1], S[r]):
                order = ((j - S[r - 1]) % nu[r])  # derivative order
                dff.fun.n = order
                rhs[j] = dff(t, *args, **kwds)
                d_psi.fun.n = order
                for k in range(n):
                    a_matrix[j, k] = (dbasis(t, k, n=order + 1) +
                                      j_w * d_psi(t, k))
        k1 = np.flatnonzero(1 - np.isfinite(rhs))
        if k1.size > 0:  # Remove singularities
            warnings.warn('Singularities detected! ')
            a_matrix[k1] = 0
            rhs[k1] = 0
        solution = linalg.lstsq(a_matrix, rhs)
        v = basis.eval([-1, 1], solution[0])

        lim_g = Limit(g)
        g_b = np.exp(j_w * lim_g(1, *args, **kwds))
        if np.isnan(g_b):
            g_b = 0
        g_a = np.exp(j_w * lim_g(-1, *args, **kwds))
        if np.isnan(g_a):
            g_a = 0
        return v[1] * g_b - v[0] * g_a

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
        """if s>1,the integral is computed by Q_s^L"""
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
            if delta < 10 or alpha <= 10 or s > 1:
                points = chebyshev_extrema
            else:
                points = adaptive_levin_points
        elif delta < 10 or alpha <= 10 or s > 1:
            points = chebyshev_roots
        else:
            points = open_levin_points  # tanh_sinh_open_nodes

        m = self._get_num_points(s, prec, betam)
        abseps = 10 * 10.0**-prec
        num_collocation_point_list = m * 2**np.arange(1, 5) + 1
        basis = self.basis

        q_val = 1e+300
        num_function_evaluations = 0
        n = 0
        for num_collocation_points in num_collocation_point_list:
            n_old = n
            q_old = q_val
            x = points(num_collocation_points, betam)
            n = len(x)
            if n > n_old:
                q_val = self._a_levin(omega, ff, gg, dgg, x, s, basis, *args,
                                      **kwds)
                num_function_evaluations += n
                err = np.abs(q_val - q_old)
                if err <= abseps:
                    break
        info = self.info(err, num_function_evaluations)
        return q_val, info


class EvansWebster(AdaptiveLevin):
    """Return integral for the Evans Webster method"""

    def __init__(self, f, g, dg=None, a=-1, b=1, basis=chebyshev_basis, s=8,
                 precision=10, endpoints=False, full_output=False):
        super(EvansWebster,
              self).__init__(f, g, dg=dg, a=a, b=b, basis=basis, s=s,
                             precision=precision, endpoints=endpoints,
                             full_output=full_output)

    def _a_levin(self, omega, ff, gg, dgg, x, s, basis, *args, **kwds):
        w = evans_webster_weights(omega, gg, dgg, x, basis, *args, **kwds)

        f = Limit(ff)(x, *args, **kwds)
        return np.sum(f * w)

    def _get_num_points(self, s, prec, betam):
        return 8 if s > 1 else int(prec / max(np.log10(betam + 1), 1) + 1)


if __name__ == '__main__':
    tanh_sinh_nodes(16)
