"""
Dispersion relation module
--------------------------
k2w - Translates from wave number to frequency
w2k - Translates from frequency to wave number
"""
import warnings
import numpy as np
from wafo.misc import lazywhere
from numpy import (atleast_1d, sqrt, ones_like, zeros_like, arctan2, where,
                   tanh, sin, cos, sign, inf,
                   flatnonzero, cosh)

__all__ = ['k2w', 'w2k']


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


def _assert_warn(cond, msg):
    if not cond:
        warnings.warn(msg)


def k2w(k1, k2=0e0, h=inf, g=9.81, u1=0e0, u2=0e0):
    """ Translates from wave number to frequency using the dispersion relation

    Parameters
    ----------
    k1 : array-like
        wave numbers [rad/m].
    k2 : array-like, optional
        second dimension wave number
    h : real scalar, optional
        water depth [m].
    g : real scalar, optional
        acceleration of gravity, see gravity
    u1, u2 : real scalars, optional
        current velocity [m/s] along dimension 1 and 2.
        note: when u1!=0 | u2!=0 then theta is not calculated correctly

    Returns
    -------
    w : ndarray
        angular frequency [rad/s].
    theta : ndarray
        direction [rad].

    Dispersion relation
    -------------------
        w     = sqrt(g*K*tanh(K*h))   (  0 <   w   < inf)
        theta = arctan2(k2,k1)        (-pi < theta <  pi)
    where
        K = sqrt(k1**2+k2**2)

    The shape of w and theta is the common shape of k1 and k2 according to the
    numpy broadcasting rules.

    See also
    --------
    w2k

    Examples
    --------
    >>> from numpy import arange
    >>> import wafo.wave_theory.dispersion_relation as wsd
    >>> wsd.k2w(arange(0.01,.5,0.2))[0]
    array([ 0.3132092 ,  1.43530485,  2.00551739])
    >>> wsd.k2w(arange(0.01,.5,0.2),h=20)[0]
    array([ 0.13914927,  1.43498213,  2.00551724])
    """

    k1i, k2i, hi, gi, u1i, u2i = np.broadcast_arrays(k1, k2, h, g, u1, u2)

    if np.size(k1i) == 0:
        return zeros_like(k1i)
    ku1 = k1i * u1i
    ku2 = k2i * u2i

    theta = arctan2(k2, k1)

    k = sqrt(k1i ** 2 + k2i ** 2)
    w = where(k > 0, ku1 + ku2 + sqrt(gi * k * tanh(k * hi)), 0.0)

    cond = (0 <= w)
    _assert_warn(np.all(cond), """
        Waves and current are in opposite directions
        making some of the frequencies negative.
        Here we are forcing the negative frequencies to zero.
        """)

    w = where(cond, w, 0.0)  # force w to zero
    return w, theta


def w2k(w, theta=0.0, h=inf, g=9.81, count_limit=100, rtol=1e-7, atol=1e-14):
    """
    Translates from frequency to wave number using the dispersion relation

    Parameters
    ----------
    w : array-like
        angular frequency [rad/s].
    theta : array-like, optional
        direction [rad].
    h : real scalar, optional
        water depth [m].
    g : real scalar or array-like of size 2.
        constant of gravity [m/s**2] or 3D normalizing constant

    Returns
    -------
    k1, k2 : ndarray
        wave numbers [rad/m] along dimension 1 and 2.

    Description
    -----------
    Uses Newton Raphson method to find the wave number k in the dispersion
    relation
        w**2= g*k*tanh(k*h).
    The solution k(w) => k1 = k(w)*cos(theta)
                         k2 = k(w)*sin(theta)
    The size of k1,k2 is the common shape of w and theta according to numpy
    broadcasting rules. If w or theta is scalar it functions as a constant
    matrix of the same shape as the other.

    Examples
    --------
    >>> import pylab as plb
    >>> import wafo.wave_theory.dispersion_relation as wsd
    >>> w = plb.linspace(0,3);
    >>> wsd.w2k(range(4))[0]
    array([ 0.        ,  0.1019368 ,  0.4077472 ,  0.91743119])
    >>> wsd.w2k(range(4),h=20)[0]
    array([ 0.        ,  0.10503601,  0.40774726,  0.91743119])

    h = plb.plot(w,w2k(w)[0])
    plb.close('all')

    See also
    --------
    k2w
    """
    gi = atleast_1d(g)
    wi, th, hi = np.broadcast_arrays(w, theta, h)
    if wi.size == 0:
        return zeros_like(wi)

    k = 1.0 * sign(wi) * wi ** 2.0 / gi[0]  # deep water
    if (hi > 1e25).all():
        k2 = k * sin(th) * gi[0] / gi[-1]  # size np x nf
        k1 = k * cos(th)
        return k1, k2
    _assert(gi.size == 1, 'Finite depth in combination with 3D normalization'
            ' (len(g)=2) is not implemented yet.')

    oshape = k.shape
    wi, k, hi = wi.ravel(), k.ravel(), hi.ravel()

    # Newton's Method
    # Permit no more than count_limit iterations.
    hi = hi * ones_like(k)
    hn = zeros_like(k)
    ix = flatnonzero(((wi < 0) | (0 < wi)) & (hi < 1e25))

    # Break out of the iteration loop for three reasons:
    #  1) the last update is very small (compared to k*rtol)
    #  2) the last update is very small (compared to atol)
    #  3) There are more than 100 iterations. This should NEVER happen.
    count = 0
    while (ix.size > 0 and count < count_limit):
        ki = k[ix]
        kh = ki * hi[ix]
        coshkh2 = lazywhere(np.abs(kh) < 350, (kh, ),
                            lambda kh: cosh(kh) ** 2.0, fillvalue=np.inf)
        hn[ix] = (ki * tanh(kh) - wi[ix] ** 2.0 / gi) / (tanh(kh) + kh / coshkh2)
        knew = ki - hn[ix]
        # Make sure that the current guess is not zero.
        # When Newton's Method suggests steps that lead to zero guesses
        # take a step 9/10ths of the way to zero:
        ksmall = flatnonzero((np.abs(knew) == 0) | (np.isnan(knew)) )
        if ksmall.size > 0:
            knew[ksmall] = ki[ksmall] / 10.0
            hn[ix[ksmall]] = ki[ksmall] - knew[ksmall]

        k[ix] = knew
        # disp(['Iteration ',num2str(count),'  Number of points left:  '
        # num2str(length(ix)) ]),

        ix = flatnonzero((np.abs(hn) > rtol * np.abs(k)) *
                         (np.abs(hn) > atol))
        count += 1
    max_err = np.max(hn[ix]) if np.any(ix) else 0
    _assert_warn(count < count_limit, 'W2K did not converge. '
                 'Maximum error in the last step was: %13.8f' % max_err)

    k.shape = oshape

    k2 = k * sin(th)
    k1 = k * cos(th)
    return k1, k2


if __name__ == '__main__':
    from wafo.testing import test_docstrings
    test_docstrings(__file__)
