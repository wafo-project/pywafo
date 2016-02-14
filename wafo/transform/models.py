'''
Transform Gaussian models
-------------------------
TrHermite
TrOchi
TrLinear
'''
# !/usr/bin/env python
from __future__ import division, absolute_import
from scipy.optimize import brentq  # @UnresolvedImport
from numpy import (sqrt, atleast_1d, abs, imag, sign, where, cos, arccos, ceil,
                   expm1, log1p, pi)
import numpy as np
import warnings
from .core import TrCommon, TrData
__all__ = ['TrHermite', 'TrLinear', 'TrOchi']

_example = '''
    >>> import numpy as np
    >>> import wafo.spectrum.models as sm
    >>> import wafo.transform.models as tm
    >>> std = 7./4
    >>> g = tm.<generic>(sigma=std, ysigma=std)

    Simulate a Transformed Gaussian process:
    >>> Sj = sm.Jonswap(Hm0=4*std, Tp=11)
    >>> w = np.linspace(0,4,256)
    >>> S = Sj.tospecdata(w) # Make spectrum object from numerical values
    >>> ys = S.sim(ns=15000) # Simulated in the Gaussian world

    >>> me, va, sk, ku = S.stats_nl(moments='mvsk')
    >>> g2 = tm.<generic>(mean=me, var=va, skew=sk, kurt=ku, ysigma=std)
    >>> xs = g2.gauss2dat(ys[:,1:]) # Transformed to the real world
    '''


class TrCommon2(TrCommon):
    __doc__ = TrCommon.__doc__  # @ReservedAssignment

    def trdata(self, x=None, xnmin=-5, xnmax=5, n=513):
        """
        Return a discretized transformation model.

        Parameters
        ----------
        x : vector  (default sigma*linspace(xnmin,xnmax,n)+mean)
        xnmin : real, scalar
            minimum on normalized scale
        xnmax : real, scalar
            maximum on normalized scale
        n : integer, scalar
            number of evaluation points

        Returns
        -------
        t0 : real, scalar
            a measure of departure from the Gaussian model calculated as
            trapz((xn-g(x))**2., xn) where int. limits is given by X.
        """
        if x is None:
            xn = np.linspace(xnmin, xnmax, n)
            x = self.sigma * xn + self.mean
        else:
            xn = (x - self.mean) / self.sigma

        yn = (self._dat2gauss(x) - self.ymean) / self.ysigma

        return TrData(yn, x, mean=self.mean, sigma=self.sigma)


class TrHermite(TrCommon2):
    __doc__ = TrCommon2.__doc__.replace('<generic>', 'Hermite'
                                        ) + """
    pardef : scalar, integer
        1  Winterstein et. al. (1994) parametrization [1]_ (default)
        2  Winterstein (1988) parametrization [2]_

    Description
    -----------
    The hermite transformation model is monotonic cubic polynomial, calibrated
    such that the first 4 moments of the transformed model G(y)=g^-1(y) match
    the moments of the true process. The model is given as:

        g(x) =  xn - c3(xn**2-1) - c4*(xn**3-3*xn)

    for kurt<3 (hardening model) where
        xn = (x-mean)/sigma
        c3 = skew/6
        c4 = (kurt-3)/24.

    or
        G(y) = mean + K*sigma*[ y + c3(y**2-1) + c4*(y**3-3*y) ]

    for kurt>=3 (softening model) where
        y  = g(x) = G**-1(x)
        K  = 1/sqrt(1+2*c3^2+6*c4^2)
        If pardef = 1 :
            c3  = skew/6*(1-0.015*abs(skew)+0.3*skew^2)/(1+0.2*(kurt-3))
            c4  = 0.1*((1+1.25*(kurt-3))^(1/3)-1)*c41
            c41 = (1-1.43*skew^2/(kurt-3))^(1-0.1*(kurt)^0.8)
        If pardef = 2 :
            c3 = skew/(6*(1+6*c4))
            c4 = [sqrt(1+1.5*(kurt-3))-1]/18


    Example:
    --------
    """ + _example.replace('<generic>', 'TrHermite') + """
    >>> np.allclose(g.dist2gauss(), 0.88230868748851499)
    True
    >>> np.allclose(g2.dist2gauss(), 1.1411663205144991)
    True

    See also
    --------
    SpecData1d.stats_nl
    wafo.transform.TrOchi
    wafo.objects.LevelCrossings.trdata
    wafo.objects.TimeSeries.trdata

    References
    ----------
    .. [1] Winterstein, S.R, Ude, T.C. and Kleiven, G. (1994)
           "Springing and slow drift responses:
           predicted extremes and fatigue vs. simulation"
           In Proc. 7th International behaviour of Offshore structures, (BOSS)
           Vol. 3, pp.1-15
    .. [2] Winterstein, S.R. (1988)
           'Nonlinear vibration models for extremes and fatigue.'
           J. Engng. Mech., ASCE, Vol 114, No 10, pp 1772-1790
    """

    def __init__(self, *args, **kwds):
        super(TrHermite, self).__init__(*args, **kwds)
        self.pardef = kwds.get('pardef', 1)
        self._c3 = None
        self._c4 = None
        self._forward = None
        self._backward = None
        self._x_limit = None
        self.set_poly()

    def _poly_par_from_stats(self):
        skew = self.skew
        ga2 = self.kurt - 3.0
        if ga2 <= 0:
            self._c4 = ga2 / 24.
            self._c3 = skew / 6.
        elif self.pardef == 2:
            # Winterstein 1988 parametrization
            if skew ** 2 > 8 * (ga2 + 3.) / 9.:
                warnings.warn('Kurtosis too low compared to the skewness')

            self._c4 = (sqrt(1. + 1.5 * ga2) - 1.) / 18.
            self._c3 = skew / (6. * (1 + 6. * self._c4))
        else:
            # Winterstein et. al. 1994 parametrization intended to
            # apply for the range:  0 <= ga2 < 12 and 0<= skew^2 < 2*ga2/3
            if skew ** 2 > 2 * (ga2) / 3:
                warnings.warn('Kurtosis too low compared to the skewness')

            if (ga2 < 0) or (12 < ga2):
                warnings.warn('Kurtosis must be between 0 and 12')

            self._c3 = skew / 6 * \
                (1 - 0.015 * abs(skew) + 0.3 * skew ** 2) / (1 + 0.2 * ga2)
            if ga2 == 0.:
                self._c4 = 0.0
            else:
                expon = 1. - 0.1 * (ga2 + 3.) ** 0.8
                c41 = (1. - 1.43 * skew ** 2. / ga2) ** (expon)
                self._c4 = 0.1 * ((1. + 1.25 * ga2) ** (1. / 3.) - 1.) * c41

        if not np.isfinite(self._c3) or not np.isfinite(self._c4):
            raise ValueError('Unable to calculate the polynomial')

    def set_poly(self):
        '''
        Set poly function from stats (i.e., mean, sigma, skew and kurt)
        '''

        if self._c3 is None:
            self._poly_par_from_stats()
        eps = np.finfo(float).eps
        c3 = self._c3
        c4 = self._c4
        ma = self.mean
        sa = self.sigma
        if abs(c4) < sqrt(eps):
            c4 = 0.0

        #  gdef = self.kurt-3.0
        if self.kurt < 3.0:
            p = np.poly1d([-c4, -c3, 1. + 3. * c4, c3])  # forward, g
            self._forward = p
            self._backward = None
        else:
            Km1 = np.sqrt(1. + 2. * c3 ** 2 + 6 * c4 ** 2)
            # backward G
            p = np.poly1d(np.r_[c4, c3, 1. - 3. * c4, -c3] / Km1)
            self._forward = None
            self._backward = p

        # Check if it is a strictly increasing function.
        dp = p.deriv(m=1)  # % Derivative
        r = dp.r  # % Find roots of the derivative
        r = r[where(abs(imag(r)) < eps)]  # Keep only real roots

        if r.size > 0:
            # Compute where it is possible to invert the polynomial
            if self.kurt < 3.:
                self._x_limit = r
            else:
                self._x_limit = sa * p(r) + ma
            txt1 = '''
                The polynomial is not a strictly increasing function.
                The derivative of g(x) is infinite at x = %g''' % self._x_limit
            warnings.warn(txt1)
        return

    def check_forward(self, x):
        if not (self._x_limit is None):
            x00 = self._x_limit
            txt2 = 'for the given interval x = [%g, %g]' % (x[0], x[-1])

            if any(np.logical_and(x[0] <= x00, x00 <= x[-1])):
                cdef = 1
            else:
                cdef = sum(np.logical_xor(x00 <= x[0], x00 <= x[-1]))

            if np.mod(cdef, 2):
                errtxt = 'Unable to invert the polynomial \n %s' % txt2
                raise ValueError(errtxt)
            np.disp(
                'However, successfully inverted the polynomial\n %s' % txt2)

    def _dat2gauss(self, x, *xi):
        if len(xi) > 0:
            raise ValueError('Transforming derivatives is not implemented!')
        xn = atleast_1d(x)
        self.check_forward(xn)

        xn = (xn - self.mean) / self.sigma

        if self._forward is None:
            # Inverting the polynomial
            yn = self._poly_inv(self._backward, xn)
        else:
            yn = self._forward(xn)
        return yn * self.ysigma + self.ymean

    def _gauss2dat(self, y, *yi):
        if len(yi) > 0:
            raise ValueError('Transforming derivatives is not implemented!')
        yn = (atleast_1d(y) - self.ymean) / self.ysigma
        # self.check_forward(y)

        if self._backward is None:
            # Inverting the polynomial
            xn = self._poly_inv(self._forward, yn)
        else:
            xn = self._backward(yn)
        return self.sigma * xn + self.mean

    def _poly_inv(self, p, xn):
        '''
        Invert polynomial
        '''
        if p.order < 2:
            return xn
        elif p.order == 2:
            # Quadratic: Solve a*u**2+b*u+c = xn
            coefs = p.coeffs
            a = coefs[0]
            b = coefs[1]
            c = coefs[2] - xn
            t = 0.5 * (b + sign(b) * sqrt(b ** 2 - 4 * a * c))
            # so1 = t/a # largest solution
            so2 = -c / t  # smallest solution
            return so2
        elif p.order == 3:
            # Solve
            # K*(c4*u^3+c3*u^2+(1-3*c4)*u-c3) = xn = (x-ma)/sa
            # -c4*xn^3-c3*xn^2+(1+3*c4)*xn+c3 = u
            coefs = p.coeffs[1::] / p.coeffs[0]
            a = coefs[0]
            b = coefs[1]
            c = coefs[2] - xn / p.coeffs[0]

            x0 = a / 3.
            # substitue xn = z-x0  and divide by c4 => z^3 + 3*p1*z+2*q0  = 0
            p1 = b / 3 - x0 ** 2
            # p1 = (b-a**2/3)/3

            # q0 = (c + x0*(2.*x0/3.-b))/2.
            # q0 = x0**3 -a*b/6 +c/2
            q0 = x0 * (x0 ** 2 - b / 2) + c / 2
# z^3+3*p1*z+2*q0=0

#            c3 = self._c3
#            c4 = self._c4
#            b1 = 1./(3.*c4)
# x0 = c3*b1
# % substitue u = z-x0  and divide by c4 => z^3 + 3*c*z+2*q0  = 0
# p1  = b1-1.-x0**2.
#            Km1 = np.sqrt(1.+2.*c3**2+6*c4**2)
#            q0 = x0**3-1.5*b1*(x0+xn*Km1)
            # q0 = x0**3-1.5*b1*(x0+xn)
            if not (self._x_limit is None):  # % Three real roots
                d = sqrt(-p1)
                theta1 = arccos(-q0 / d ** 3) / 3
                th2 = np.r_[0, -2 * pi / 3, 2 * pi / 3]
                x1 = abs(2 * d * cos(theta1[ceil(len(xn) / 2)] + th2) - x0)
                ix = x1.argmin()  # % choose the smallest solution
                return 2. * d * cos(theta1 + th2[ix]) - x0
            else:                # %Only one real root exist
                q1 = sqrt((q0) ** 2 + p1 ** 3)
                # Find the real root of the monic polynomial
                A0 = (q1 - q0) ** (1. / 3.)
                B0 = -(q1 + q0) ** (1. / 3.)
                return A0 + B0 - x0  # % real root
                # The other complex roots are given by
                # x= -(A0+B0)/2+(A0-B0)*sqrt(3)/2-x0
                # x=-(A0+B0)/2+(A0-B0)*sqrt(-3)/2-x0


class TrLinear(TrCommon2):
    __doc__ = TrCommon2.__doc__.replace('<generic>', 'Linear'
                                        ) + """
    Description
    -----------
    The linear transformation model is monotonic linear polynomial, calibrated
    such that the first 2 moments of the transformed model G(y)=g^-1(y) match
    the moments of the true process.

    Example:
    --------
    """ + _example.replace('<generic>', 'TrLinear') + """
    >>> np.allclose(g.dist2gauss(), 0)
    True
    >>> np.allclose(g2.dist2gauss(), 0)
    True

    See also
    --------
    TrOchi
    TrHermite
    SpecData1D.stats_nl
    LevelCrossings.trdata
    TimeSeries.trdata
    spec2skew, ochitr, lc2tr, dat2tr

    """

    def _dat2gauss(self, x, *xi):
        sratio = atleast_1d(self.ysigma / self.sigma)
        y = (atleast_1d(x) - self.mean) * sratio + self.ymean
        if len(xi) > 0:
            y = [y, ] + [ix * sratio for ix in xi]
        return y

    def _gauss2dat(self, y, *yi):
        sratio = atleast_1d(self.sigma / self.ysigma)
        x = (atleast_1d(y) - self.ymean) * sratio + self.mean
        if len(yi) > 0:
            x = [x, ] + [iy * sratio for iy in yi]
        return x


class TrOchi(TrCommon2):
    __doc__ = TrCommon2.__doc__.replace('<generic>', 'Ochi'
                                        ) + """

    Description
    -----------
    The Ochi transformation model is a monotonic exponential function,
    calibrated such that the first 3 moments of the transformed model
    G(y)=g^-1(y) match the moments of the true  process. However, the
    skewness is limited by ABS(SKEW)<2.82. According to Ochi it is
    appropriate for a process with very strong non-linear characteristics.
    The model is given as:
        g(x) = ((1-exp(-gamma*(x-mean)/sigma))/gamma-mean2)/sigma2
    where
        gamma  = 1.28*a  for x>=mean
                 3*a     otherwise
        mean,
        sigma  = standard deviation and mean, respectively, of the process.
        mean2,
        sigma2 = normalizing parameters in the transformed world, i.e., to
                make the gaussian process in the transformed world is N(0,1).

    The unknown parameters a, mean2 and sigma2 are found by solving the
    following non-linear equations:

        a*(sigma2^2+mean2^2)+mean2 = 0
           sigma2^2-2*a^2*sigma2^4 = 1
    2*a*sigma2^4*(3-8*a^2*sigma2^2) = skew

    Note
    ----
    Transformation, g, does not have continous derivatives of 2'nd order or
    higher.

    Example
    -------
    """ + _example.replace('<generic>', 'TrOchi') + """
    >>> np.allclose(g.dist2gauss(), 1.410698801056657)
    True
    >>> np.allclose(g2.dist2gauss(), 1.988807188766706)
    True

    See also
    --------
    spec2skew, hermitetr, lc2tr, dat2tr

    References
    ----------
    Ochi, M.K. and Ahn, K. (1994)
    'Non-Gaussian probability distribution of coastal waves.'
    In Proc. 24th Conf. Coastal Engng, Vol. 1, pp 482-496

    Michel K. Ochi (1998),
    "OCEAN WAVES, The stochastic approach",
    OCEAN TECHNOLOGY series 6, Cambridge, pp 255-275.
    """

    def __init__(self, *args, **kwds):
        super(TrOchi, self).__init__(*args, **kwds)
        self.kurt = None
        self._phat = None
        self._par_from_stats()

    def _par_from_stats(self):
        skew = self.skew
        if abs(skew) > 2.82842712474619:
            raise ValueError('Skewness must be less than 2.82842')

        mean1 = self.mean
        sigma1 = self.sigma

        if skew == 0:
            self._phat = [sigma1, mean1, 0, 0, 1, 0]
            return

        # Solve the equations to obtain the gamma parameters:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #          a*(sig2^2+ma2^2)+ma2 = 0
        #           sig2^2-2*a^2*sig2^4 = E(y^2) % =1
        #   2*a*sig2^4*(3-8*a^2*sig2^2) = E(y^3) % = skew

        # Let x = [a sig2^2 ]
        # Set up the 2D non-linear equations for a and sig2^2:
        # g1='[x(2)-2.*x(1).^2.*x(2).^2-P1,
        #      2.*x(1).*x(2).^2.*(3-8.*x(1).^2.*x(2))-P2  ]'
        # Or solve the following 1D non-linear equation for sig2^2:
        def g2(x):
            return (-sqrt(abs(x - 1) * 2) * (3. * x - 4 * abs(x - 1)) +
                    abs(skew))

        a1 = 1.  # Start interval where sig2^2 is located.
        a2 = 2.

        sig22 = brentq(g2, a1, a2)  # % smallest solution for sig22
        a = sign(skew) * sqrt(abs(sig22 - 1) / 2) / sig22
        gam_a = 1.28 * a
        gam_b = 3 * a
        sigma2 = sqrt(sig22)

        # Solve the following 2nd order equation to obtain ma2
        #        a*(sig2^2+ma2^2)+ma2 = 0
        my2 = (-1. - sqrt(1. - 4. * a ** 2 * sig22)) / a  # % Largest mean
        mean2 = a * sig22 / my2  # % choose the smallest mean

        self._phat = [sigma1, mean1, gam_a, gam_b, sigma2, mean2]
        return

    def _get_par(self):
        '''
        Returns ga, gb, sigma2, mean2
        '''
        if (self._phat is None or self.sigma != self._phat[0] or
                self.mean != self._phat[1]):
            self._par_from_stats()
        # sigma1 = self._phat[0]
        # mean1 = self._phat[1]
        ga = self._phat[2]
        gb = self._phat[3]
        sigma2 = self._phat[4]
        mean2 = self._phat[5]
        return ga, gb, sigma2, mean2

    def _dat2gauss(self, x, *xi):
        if len(xi) > 0:
            raise ValueError('Transforming derivatives is not implemented!')
        ga, gb, sigma2, mean2 = self._get_par()
        mean = self.mean
        sigma = self.sigma
        xn = atleast_1d(x)
        shape0 = xn.shape
        xn = (xn.ravel() - mean) / sigma
        igp, = where(0 <= xn)
        igm, = where(xn < 0)

        g = xn.copy()

        if ga != 0:
            np.put(g, igp, (-expm1(-ga * xn[igp])) / ga)

        if gb != 0:
            np.put(g, igm, (-expm1(-gb * xn[igm])) / gb)

        g.shape = shape0
        return (g - mean2) * self.ysigma / sigma2 + self.ymean

    def _gauss2dat(self, y, *yi):
        if len(yi) > 0:
            raise ValueError('Transforming derivatives is not implemented!')

        ga, gb, sigma2, mean2 = self._get_par()
        mean = self.mean
        sigma = self.sigma

        yn = (atleast_1d(y) - self.ymean) / self.ysigma
        xn = sigma2 * yn.ravel() + mean2

        igp, = where(0 <= xn)
        igm, = where(xn < 0)

        if ga != 0:
            np.put(xn, igp, -log1p(-ga * xn[igp]) / ga)

        if gb != 0:
            np.put(xn, igm, -log1p(-gb * xn[igm]) / gb)

        xn.shape = yn.shape
        return sigma * xn + mean


def main():
    import pylab
    g = TrHermite(skew=0.1, kurt=3.01)
    g.dist2gauss()
    # g = TrOchi(skew=0.56)
    x = np.linspace(-5, 5)
    y = g(x)
    pylab.plot(np.abs(x - g.gauss2dat(y)))
    # pylab.plot(x,y,x,x,':',g.gauss2dat(y),y,'r')

    pylab.show()
    np.disp('finito')

if __name__ == '__main__':
    if True:  # False: #
        import doctest
        doctest.testmod()
    else:
        main()
