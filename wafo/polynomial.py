"""
    Extended functions to operate on polynomials
"""
# -------------------------------------------------------------------------
# Name:        polynomial
# Purpose:     Functions to operate on polynomials.
#
# Author:      pab
# polyXXX functions are based on functions found in the matlab toolbox polyutil
# written by
# Author:      Peter J. Acklam
# E-mail:      pjacklam@online.no
# WWW URL:     http://home.online.no/~pjacklam
#
# Created:     30.12.2008
# Copyright:   (c) pab 2008
# Licence:     LGPL
# -------------------------------------------------------------------------
# !/usr/bin/env python
from __future__ import absolute_import
import warnings  # @UnusedImport
from numpy.polynomial import polyutils as pu
from .plotbackend import plotbackend as plt
import numpy as np
from numpy import (zeros, asarray, newaxis, arange,
                   logical_or, any, pi, cos, round, diff, all, exp,
                   where, extract, linalg, sign, concatenate, floor,
                   linspace, sum, meshgrid)

from scipy.fftpack import dct, idct as _idct
from numpy.lib.polynomial import *  # @UnusedWildImport
from scipy.misc.common import pade  # @UnresolvedImport
__all__ = np.lib.polynomial.__all__
__all__ = __all__ + ['pade', 'padefit', 'polyreloc', 'polyrescl', 'polytrim',
                     'poly2hstr', 'poly2str', 'polyshift', 'polyishift',
                     'map_from_intervall', 'map_to_intervall', 'cheb2poly',
                     'chebextr', 'chebroot', 'chebpoly', 'chebfit', 'chebval',
                     'chebder', 'chebint', 'Cheb1d', 'dct', 'idct']


def polyint(p, m=1, k=None):
    """
    Return an antiderivative (indefinite integral) of a polynomial.

    The returned order `m` antiderivative `P` of polynomial `p` satisfies
    :math:`\\frac{d^m}{dx^m}P(x) = p(x)` and is defined up to `m - 1`
    integration constants `k`. The constants determine the low-order
    polynomial part

    .. math:: \\frac{k_{m-1}}{0!} x^0 + \\ldots + \\frac{k_0}{(m-1)!}x^{m-1}

    of `P` so that :math:`P^{(j)}(0) = k_{m-j-1}`.

    Parameters
    ----------
    p : {array_like, poly1d}
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of the antiderivative. (Default: 1)
    k : {None, list of `m` scalars, scalar}, optional
        Integration constants. They are given in the order of integration:
        those corresponding to highest-order terms come first.

        If ``None`` (default), all constants are assumed to be zero.
        If `m = 1`, a single scalar can be given instead of a list.

    See Also
    --------
    polyder : derivative of a polynomial
    poly1d.integ : equivalent method

    Examples
    --------
    The defining property of the antiderivative:

    >>> p = np.poly1d([1,1,1])
    >>> P = np.polyint(p)
    >>> P
    poly1d([ 0.33333333,  0.5       ,  1.        ,  0.        ])
    >>> np.polyder(P) == p
    True

    The integration constants default to zero, but can be specified:

    >>> P = np.polyint(p, 3)
    >>> P(0)
    0.0
    >>> np.polyder(P)(0)
    0.0
    >>> np.polyder(P, 2)(0)
    0.0
    >>> P = np.polyint(p, 3, k=[6, 5, 3])
    >>> P.coefficients.tolist()
    [0.016666666666666666, 0.041666666666666664, 0.16666666666666666, 3.0,
    5.0, 3.0]

    Note that 3 = 6 / 2!, and that the constants are given in the order of
    integrations. Constant of the highest-order polynomial term comes first:

    >>> np.polyder(P, 2)(0)
    6.0
    >>> np.polyder(P, 1)(0)
    5.0
    >>> P(0)
    3.0

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    if k is None:
        k = zeros(m, float)
    k = np.atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0] * np.ones(m, float)
    if len(k) < m:
        raise ValueError(
            "k must be a scalar or a rank-1 array of length 1 or >m.")
    truepoly = isinstance(p, poly1d)
    p = asarray(p)
    if m == 0:
        if truepoly:
            return poly1d(p)
        return p
    else:
        ix = arange(len(p), 0, -1)
        if p.ndim > 1:
            ix = ix[..., newaxis]
            pieces = p.shape[-1]
            k0 = k[0] * np.ones((1, pieces), dtype=int)
        else:
            k0 = [k[0]]
        y = np.concatenate((p.__truediv__(ix), k0), axis=0)

        val = polyint(y, m - 1, k=k[1:])
        if truepoly:
            return poly1d(val)
        return val


def polyder(p, m=1):
    """
    Return the derivative of the specified order of a polynomial.

    Parameters
    ----------
    p : poly1d or sequence
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of differentiation (default: 1)

    Returns
    -------
    der : poly1d
        A new polynomial representing the derivative.

    See Also
    --------
    polyint : Anti-derivative of a polynomial.
    poly1d : Class for one-dimensional polynomials.

    Examples
    --------
    The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:

    >>> p = np.poly1d([1,1,1,1])
    >>> p2 = np.polyder(p)
    >>> p2
    poly1d([3, 2, 1])

    which evaluates to:

    >>> p2(2.)
    17.0

    We can verify this, approximating the derivative with
    ``(f(x + h) - f(x))/h``:

    >>> (p(2. + 0.001) - p(2.)) / 0.001
    17.007000999997857

    The fourth-order derivative of a 3rd-order polynomial is zero:

    >>> np.polyder(p, 2)
    poly1d([6, 2])
    >>> np.polyder(p, 3)
    poly1d([6])
    >>> np.polyder(p, 4)
    poly1d([ 0.])

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")
    truepoly = isinstance(p, poly1d)
    p = asarray(p)
    if m == 0:
        if truepoly:
            return poly1d(p)
        return p
    else:
        n = len(p) - 1
        ix = arange(n, 0, -1)
        if p.ndim > 1:
            ix = ix[..., newaxis]
        y = ix * p[:-1]
        val = polyder(y, m - 1)
        if truepoly:
            return poly1d(val)
        return val


def polydeg(x, y):
    '''
    Return optimal degree for polynomial fitting


    N = POLYDEG(X,Y) finds the optimal degree for polynomial fitting,
    according to the Akaike's information criterion.

    Assuming that you want to find the degree N of a polynomial that fits
    the data Y(X) best in a least-squares sense, the Akaike's information
    criterion is defined by:
        2*(N + 1) + n * (log(2 * pi * RSS / n) + 1)
    where n is the number of points and RSS is the residual sum of squares.
    The optimal degree N is defined here as that which minimizes AIC:
    http://en.wikipedia.org/wiki/Akaike_Information_Criterion

    Notes:
    -----
    If the number of data is small, POLYDEG may tend to return:
    N = (number of points)-1.

    ORTHOFIT is more appropriate than POLYFIT for polynomial fitting with
    relatively high degrees.

    Example:
    -------
    >>> x = np.linspace(0,10,300)
    >>> y = np.sin(x ** 3 / 100) ** 2 + 0.05 * np.random.randn(x.size)
    >>> n = polydeg(x,y)
    >>> n
    21

    ys = orthofit(x,y,n);
    plt.plot(x, y, '.', x, ys, 'k')

    See also
    --------
    polyfit, orthofit
    '''
    x, y = np.atleast_1d(x, y)
    x = x.ravel()
    y = y.ravel()
    N = len(x)

    # Search the optimal degree minimizing the Akaike's information criterion
    #  y(x) are fitted in a least-squares sense using a polynomial of degree n
    #  developed in a series of orthogonal polynomials.
    ys = np.ones((N,)) * y.mean()
    # correction for small sample sizes
    logsum2 = (np.log(2 * pi * ((ys - y) ** 2).sum() / N) + 1)
    AIC = 2 + N * logsum2 + 4 / (N - 2)

    n = 1
    nit = 0

    # While-loop is stopped when a minimum is detected. 3 more steps are
    # required to take AIC noise into account and to ensure that this minimum
    # is a (likely) global minimum.

    while nit < 3:
        p = orthofit(x, y, n)
        ys = orthoval(p, x)
        # -- Akaike's Information Criterion
        aic = (2 * (n + 1) * (1 + (n + 2) / (N - n - 2)) +
               N * (np.log(2 * pi * sum((ys - y) ** 2) / N) + 1))

        if aic >= AIC:
            nit += 1
        else:
            nit = 0
            AIC = aic

        n = n + 1

        if n >= N:
            break
    n = n - nit - 1
    return n


def orthoval(p, x):
    '''
    Evaluation of orthogonal polynomial

    Parameters
    ----------
    p : array_like
       2D array of polynomial coefficients (including coefficients equal
       to zero) from highest degree to the constant term.
    x : array_like
       A number or a 1D array of numbers "at" which to evaluate `p`.

    Returns
    -------
    values : ndarray

    See Also
    --------
    orthofit
    '''
    p = np.atleast_2d(p)
    n = p.shape[1] - 1
    xi = np.atleast_1d(x)
    shape0 = xi.shape
    if n == 0:
        return np.ones(shape0) * p[0]
    xi = xi.ravel()
    xn = np.ones((n + 1, len(xi)))
    xn[1] = xi - p[1, 1]
    for i in range(2, n + 1):
        xn[i, :] = (xi - p[1, i]) * xn[i - 1, :] - p[2, i] * xn[i - 2, :]
    ys = np.dot(p[0], xn)
    return ys.reshape(shape0)


def ortho2poly(p):
    """
    Converts orthogonal polynomial to ordinary polynomial coefficients

    Parameters
    ----------
    p : array-like
        orthogonal polynomial coefficients

    Returns
    -------
    p : ndarray
        ordinary polynomial coefficients

    It is not advised to do this for p.shape[1]>10 due to numerical
    cancellations.

    See also
    --------
    orthoval
    orthofit

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> p = orthofit(x, y, 3)
    >>> p
    array([[ 0.        , -0.30285714, -0.16071429,  0.08703704],
           [ 0.        ,  2.5       ,  2.5       ,  2.5       ],
           [ 0.        ,  0.        ,  2.91666667,  2.13333333]])
    >>> ortho2poly(p)
    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254])
    >>> np.polyfit(x, y, 3)
    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254])

    References
    ----------
    """
    p = np.atleast_2d(p)
    n = p.shape[1] - 1
    if n == 0:
        return p[0]
    x = [1, ] * (n + 1)
    x[1] = np.array([1, - p[1, 1]])
    for i in range(2, n + 1):
        x[i] = polyadd(polymul([1, - p[1, i]], x[i - 1]), - p[2, i] * x[i - 2])
    for i in range(n + 1):
        x[i] *= p[0, i]
    return reduce(polyadd, x)


def orthofit(x, y, n):
    '''
    Fit orthogonal polynomial to data.

    Parameters
    ---------
    x, y : arrays
        data Y(X) to fit to a polynomial.
    n : integer
        degree of fitted polynomial.

    Returns
    -------
    p : array
        orthogonal polynomial

    Notes:
    -----
    Orthofit smooths/fits data using a polynomial of degree N developed in
    a sequence of orthogonal polynomials. ORTHOFIT is more appropriate than
    polyfit for polynomial fitting and smoothing since this method does not
    involve any matrix linear system but a simple recursive procedure.
    Degrees much higher than 30 could be used with orthogonal polynomials,
    whereas badly conditioned matrices may appear with a classical
    polynomial fitting of degree typically higher than 10.

    To avoid using unnecessarily high degrees, you may let the function
    POLYDEG choose it for you. POLYDEG finds an optimal polynomial degree
    according to the Akaike's information criterion.

    Example:
    -------
    >>> x = np.linspace(0,10,300);
    >>> y = np.sin(x**3/100)**2 + 0.05*np.random.randn(x.size)
    >>> p = orthofit(x, y, 25)
    >>> ys = orthoval(p, x)

    plot(x, y,'.',x, ys, 'k')

    See also
    --------
    polydeg, polyfit, polyval

    Reference:
    ---------
    Methodes de calcul numerique 2. JP Nougier. Hermes Science
    Publications, 2001. Section 4.7 pp 116-121
    '''
    x, y = np.atleast_1d(x, y)
    x = x.ravel()
    y = y.ravel()
    # Particular case: n=0
    if n == 0:
        return y.mean()

    # p = Coefficients of the orthogonal polynomials
    p = np.zeros((3, n + 1))
    p[1, 1] = x.mean()

    N = len(x)
    PL = np.ones((n + 1, N))
    PL[1] = x - p[1, 1]

    for i in range(2, n + 1):
        p[1, i] = np.dot(x, PL[i - 1] ** 2) / sum(PL[i - 1] ** 2)
        p[2, i] = np.dot(x, PL[i - 2] * PL[i - 1]) / sum(PL[i - 2] ** 2)
        PL[i] = (x - p[1, i]) * PL[i - 1] - p[2, i] * PL[i - 2]
    p[0, :] = np.dot(PL, y) / sum(PL ** 2, axis=1)
    return p
    # ys = np.dot(p[0, :], PL)  # smoothed y


def polyreloc(p, x, y=0.0):
    """
    Relocate polynomial

    The polynomial `p` is relocated by "moving" it `x`
    units along the x-axis and `y` units along the y-axis.
    So the polynomial `r` is relative to the point (x,y) as
    the polynomial `p` is relative to the point (0,0).

    Parameters
    ----------
    p : array-like, poly1d
        vector or matrix of column vectors of polynomial coefficients to
        relocate. (Polynomial coefficients are in decreasing order.)
    x : scalar
        distance to relocate P along x-axis
    y : scalar
        distance to relocate P along y-axis (default 0)

    Returns
    -------
    r : ndarray, poly1d
        vector/matrix/poly1d of relocated polynomial coefficients.

    See also
    --------
    polyrescl

    Example
    -------
    >>> import numpy as np
    >>> p = np.arange(6); p.shape = (2,-1)
    >>> np.polyval(p,0)
    array([3, 4, 5])
    >>> np.polyval(p,1)
    array([3, 5, 7])
    >>> r = polyreloc(p,-1) # move to the left along x-axis
    >>> np.polyval(r,-1)    # = polyval(p,0)
    array([3, 4, 5])
    >>> np.polyval(r,0)     # = polyval(p,1)
    array([3, 5, 7])
    """

    truepoly = isinstance(p, poly1d)
    r = np.atleast_1d(p).copy()
    n = r.shape[0]

    # Relocate polynomial using Horner's algorithm
    for ii in range(n, 1, -1):
        for i in range(1, ii):
            r[i] = r[i] - x * r[i - 1]
    r[-1] = r[-1] + y
    if r.ndim > 1 and r.shape[-1] == 1:
        r.shape = (r.size,)
    if truepoly:
        r = poly1d(r)
    return r


def polyrescl(p, x, y=1.0):
    """
    Rescale polynomial.

    Parameters
    ----------
    p : array-like, poly1d
        vector or matrix of column vectors of polynomial coefficients to
        rescale. (Polynomial coefficients are in decreasing order.)
    x,y : scalars
        defining the factors to rescale the polynomial `p`  in
        x-direction and y-direction, respectively.

    Returns
    -------
    r : ndarray, poly1d
        vector/matrix/poly1d of rescaled polynomial coefficients.

    See also
    --------
    polyreloc

    Example
    -------
    >>> import numpy as np
    >>> p = np.arange(6); p.shape = (2,-1)
    >>> np.polyval(p,0)
    array([3, 4, 5])
    >>> np.polyval(p,1)
    array([3, 5, 7])
    >>> r = polyrescl(p,2)  # scale by 2 along x-axis
    >>> np.polyval(r,0)     # = polyval(p,0)
    array([ 3.,  4.,  5.])
    >>> np.polyval(r,2)     # = polyval(p,1)
    array([ 3.,  5.,  7.])
    """

    truepoly = isinstance(p, poly1d)
    r = np.atleast_1d(p)
    n = r.shape[0]

    xscale = (float(x) ** arange(1 - n, 1))
    if r.ndim == 1:
        q = y * r * xscale
    else:
        q = y * r * xscale[:, newaxis]
    if truepoly:
        q = poly1d(q)
    return q


def polytrim(p):
    """
    Trim polynomial by stripping off leading zeros.

    Parameters
    ----------
    p : array-like, poly1d
        vector or matrix of column vectors of polynomial coefficients in
        decreasing order.

    Returns
    -------
    r : ndarray, poly1d
        vector/matrix/poly1d of trimmed polynomial coefficients.

    Example
    -------
    >>> p = [0,1,2]
    >>> polytrim(p)
    array([1, 2])
    >>> p1 = [[0,0],[1,2],[3,4]]
    >>> polytrim(p1)
    array([[1, 2],
           [3, 4]])
    """

    truepoly = isinstance(p, poly1d)
    if truepoly:
        return p
    else:
        r = np.atleast_1d(p).copy()
        # Remove leading zeros
        is_not_lead_zeros = logical_or.accumulate(r != 0, axis=0)
        if r.ndim == 1:
            r = r[is_not_lead_zeros]
        else:
            is_not_lead_zeros = any(is_not_lead_zeros, axis=1)
            r = r[is_not_lead_zeros, :]
        return r


def poly2hstr(p, variable='x'):
    """
    Return polynomial as a Horner represented string.

    Parameters
    ----------
    p : array-like poly1d
        vector of polynomial coefficients in decreasing order.
    variable : string
        display character for variable

    Returns
    -------
    p_str : string
        consisting of the polynomial coefficients in the vector P multiplied
        by powers of the given `variable`.

    Examples
    --------
    >>> poly2hstr([1, 1, 2], 's' )
    '(s + 1)*s + 2'

    See also
    --------
    poly2str
    """
    var = variable

    coefs = polytrim(np.atleast_1d(p))
    order = len(coefs) - 1  # Order of polynomial.
    s = ''    # Initialize output string.
    ix = 1
    for expon in range(order, -1, -1):
        coef = coefs[order - expon]
        # There is no point in adding a zero term (except if it's the only
        # term, but we'll take care of that later).
        if coef == 0:
            ix += 1
        else:
            # Append exponent if necessary.
            if ix > 1:
                exponstr = '%.0f' % ix
                s = '%s**%s' % (s, exponstr)
                ix = 1
            # Is it the first term?
            isfirst = s == ''

            # We need the coefficient only if it is different from 1 or -1 or
            # when it is the constant term.
            needcoef = (
                (abs(coef) != 1) | (
                    expon == 0) & isfirst) | 1 - isfirst

            # We need the variable except in the constant term.
            needvar = (expon != 0)

            # Add sign, but we don't need a leading plus-sign.
            if isfirst:
                if coef < 0:
                    s = '-'  # % Unary minus.
            else:
                if coef < 0:
                    s = '%s - ' % s  # % Binary minus (subtraction).
                else:
                    s = '%s + ' % s  # % Binary plus (addition).

            # Append the coefficient if it is different from one or when it is
            # the constant term.
            if needcoef:
                coefstr = '%.20g' % abs(coef)
                s = '%s%s' % (s, coefstr)

            # Append variable if necessary.
            if needvar:
                # Append a multiplication sign if necessary.
                if needcoef:
                    if 1 - isfirst:
                        s = '(%s)' % s
                    s = '%s*' % s
                s = '%s%s' % (s, var)

    # Now treat the special case where the polynomial is zero.
    if s == '':
        s = '0'
    return s


def poly2str(p, variable='x'):
    """
    Return polynomial as a string.

    Parameters
    ----------
    p : array-like poly1d
        vector of polynomial coefficients in decreasing order.
    variable : string
        display character for variable

    Returns
    -------
    p_str : string
        consisting of the polynomial coefficients in the vector P multiplied
        by powers of the given `variable`.

    See also
    --------
    poly2hstr

    Examples
    --------
    >>> poly2str([1, 1, 2], 's' )
    's**2 + s + 2'
    """
    thestr = "0"
    var = variable

    # Remove leading zeros
    coeffs = polytrim(np.atleast_1d(p))

    N = len(coeffs) - 1

    for k in range(N+1):
        coefstr = '%.4g' % abs(coeffs[k])
        if coefstr[-4:] == '0000':
            coefstr = coefstr[:-5]
        power = (N - k)
        if power == 0:
            if coefstr != '0':
                newstr = '%s' % (coefstr,)
            else:
                newstr = '0' if k == 0 else ''
        elif power == 1:
            if coefstr == '0':
                newstr = ''
            elif coefstr in ['b', '1']:
                newstr = var
            else:
                newstr = '%s*%s' % (coefstr, var)
        else:
            if coefstr == '0':
                newstr = ''
            elif coefstr in ['b', '1']:
                newstr = '%s**%d' % (var, power,)
            else:
                newstr = '%s*%s**%d' % (coefstr, var, power)

        if k > 0:
            if newstr != '':
                if coeffs[k] < 0:
                    thestr = "%s - %s" % (thestr, newstr)
                else:
                    thestr = "%s + %s" % (thestr, newstr)
        elif (k == 0) and (newstr != '') and (coeffs[k] < 0):
            thestr = "-%s" % (newstr,)
        else:
            thestr = newstr
    return thestr


def polyshift(py, a=-1, b=1):
    """
    Polynomial coefficient shift

    Polyshift shift the polynomial coefficients by a variable shift:

    Y = 2*(X-.5*(b+a))/(b-a)

    i.e., the interval -1 <= Y <= 1 is mapped to the interval a <= X <= b

    Parameters
    ----------
    py : array-like
        polynomial coefficients for the variable y.
    a,b : scalars
        lower and upper limit.

    Returns
    -------
    px : ndarray
        polynomial coefficients for the variable x.

    See also
    --------
    polyishift

    Example
    -------
    >>> py = [1, 0]
    >>> px = polyshift(py,0,5)
    >>> polyval(px,[0, 2.5, 5])  #% This is the same as the line below
    array([-1.,  0.,  1.])
    >>> polyval(py,[-1, 0, 1 ])
    array([-1,  0,  1])
    """

    if (a == -1) & (b == 1):
        return py
    L = b - a
    return polyishift(py, -(2. + b + a) / L, (2. - b - a) / L)


def polyishift(px, a=-1, b=1):
    """
    Inverse polynomial coefficient shift

    Polyishift does the inverse of Polyshift,
    shift the polynomial coefficients by a variable shift:

    Y = 2*(X-.5*(b+a)/(b-a)

    i.e., the interval a <= X <= b is mapped to the interval -1 <= Y <= 1

    Parameters
    ----------
    px : array-like
        polynomial coefficients for the variable x.
    a,b : scalars
        lower and upper limit.

    Returns
    -------
    py : ndarray
        polynomial coefficients for the variable y.

    See also
    --------
    polyishift

    Example
    -------
    >>> px = [1, 0]
    >>> py = polyishift(px,0,5);
    >>> polyval(px,[0, 2.5, 5])  #% This is the same as the line below
    array([ 0. ,  2.5,  5. ])
    >>> polyval(py,[-1, 0, 1])
    array([ 0. ,  2.5,  5. ])
    """
    if (a == -1) & (b == 1):
        return px
    L = b - a
    xscale = 2. / L
    xloc = -float(a + b) / L
    return polyreloc(polyrescl(px, xscale), xloc)


def map_from_interval(x, a, b):
    """F(x), where F: [a,b] -> [-1,1]."""
    return (x - (b + a) / 2.0) * (2.0 / (b - a))


def map_to_interval(x, a, b):
    """F(x), where F: [-1,1] -> [a,b]."""
    return (x * (b - a) + (b + a)) / 2.0


def poly2cheb(p, a=-1, b=1):
    """
    Convert polynomial coefficients into Chebyshev coefficients

    Parameters
    ----------
    p : array-like
        polynomial coefficients
    a,b : real scalars
        lower and upper limits (Default -1,1)

    Returns
    -------
    ck : ndarray
        Chebychef coefficients

    POLY2CHEB do the inverse of CHEB2POLY: given a vector of polynomial
    coefficients AK, returns an equivalent vector of Chebyshev
    coefficients CK.

    This is useful for economization of power series.
    The steps for doing so:
    1. Convert polynomial coefficients to Chebychev coefficients, CK.
    2. Truncate the CK series to a smaller number of terms, using the
    coefficient of the first neglected Chebychev polynomial as an error
    estimate.
    3 Convert back to a polynomial by CHEB2POLY

    See also
    --------
    cheb2poly
    chebval
    chebfit

    Examples
    --------
    >>> import numpy as np
    >>> p = np.arange(5)
    >>> ck = poly2cheb(p)
    >>> cheb2poly(ck)
    array([ 1.,  2.,  3.,  4.])

    Reference
    ---------
    William H. Press, Saul Teukolsky,
    William T. Wetterling and Brian P. Flannery (1997)
    "Numerical recipes in Fortran 77", Vol. 1, pp 184-194
    """
    f = poly1d(p)
    n = len(f.coeffs)
    return chebfit(f, n, a, b)


def cheb2poly(ck, a=-1, b=1):
    """
    Converts Chebyshev coefficients to polynomial coefficients

    Parameters
    ----------
    ck : array-like
        Chebychef coefficients
    a,b : real, scalars
        lower and upper limits (Default -1,1)

    Returns
    -------
    p : ndarray
        polynomial coefficients

    It is not advised to do this for len(ck)>10 due to numerical cancellations.

    See also
    --------
    chebval
    chebfit

    Examples
    --------
    >>> import numpy as np
    >>> p = np.arange(5)
    >>> ck = poly2cheb(p)
    >>> cheb2poly(ck)
    array([ 1.,  2.,  3.,  4.])


    References
    ----------
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    http://en.wikipedia.org/wiki/Chebyshev_form
    http://en.wikipedia.org/wiki/Clenshaw_algorithm
    """

    n = len(ck)

    b_Nmi = zeros(1)
    b_Nmip1 = zeros(1)
    y = np.r_[2 / (b - a), -(a + b) / (b - a)]
    y2 = 2. * y

    # Clenshaw recurence
    for ix in range(n - 1):
        tmp = b_Nmi
        b_Nmi = polymul(y2, b_Nmi)  # polynomial multiplication
        nb = len(b_Nmip1)
        b_Nmip1[-1] = b_Nmip1[-1] - ck[ix]
        b_Nmi[-nb::] = b_Nmi[-nb::] - b_Nmip1
        b_Nmip1 = tmp

    p = polymul(y, b_Nmi)  # polynomial multiplication
    nb = len(b_Nmip1)
    b_Nmip1[-1] = b_Nmip1[-1] - ck[n - 1]
    p[-nb::] = p[-nb::] - b_Nmip1
    return polytrim(p)


def chebextr(n):
    """
    Return roots of derivative of Chebychev polynomial of the first kind.

    All local extreme values of the polynomial are either -1 or 1. So,
    CHEBPOLY( N, CHEBEXTR(N) ) ) return the same as (-1).^(N:-1:0)
    except for the numerical noise in the former.

    Because the extreme values of Chebychev polynomials of the first
    kind are either -1 or 1, their roots are often used as starting
    values for the nodes in minimax approximations.


    Parameters
    ----------
    n : scalar, integer
        degree of Chebychev polynomial.

    Examples
    --------
    >>> x = chebextr(4)
    >>> chebpoly(4,x)
    array([ 1., -1.,  1., -1.,  1.])


    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_nodes
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    """
    return - cos((pi * arange(n + 1)) / n)


def chebroot(n, kind=1):
    """
    Return roots of Chebychev polynomial of the first or second kind.

    The roots of the Chebychev polynomial of the first kind form a particularly
    good set of nodes for polynomial interpolation because the resulting
    interpolation polynomial minimizes the problem of Runge's phenomenon.

    Parameters
    ----------
    n : scalar, integer
        degree of Chebychev polynomial.
    kind: 1 or 2, optional
        kind of Chebychev polynomial (default 1)

    Examples
    --------
    >>> import numpy as np
    >>> x = chebroot(3)
    >>> np.abs(chebpoly(3,x))<1e-15
    array([ True,  True,  True], dtype=bool)
    >>> chebpoly(3)
    array([ 4.,  0., -3.,  0.])
    >>> x2 = chebroot(4,kind=2)
    >>> np.abs(chebpoly(4,x2,kind=2))<1e-15
    array([ True,  True,  True,  True], dtype=bool)
    >>> chebpoly(4,kind=2)
    array([ 16.,   0., -12.,   0.,   1.])


    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_nodes
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    """
    if kind not in (1, 2):
        raise ValueError('kind must be 1 or 2')
    return - cos(pi * (arange(n) + 0.5 * kind) / (n + kind - 1))


def chebpoly(n, x=None, kind=1):
    """
    Return Chebyshev polynomial of the first or second kind.

    These polynomials are orthogonal on the interval [-1,1], with
    respect to the weight function w(x) = (1-x**2)**(-1/2+kind-1).

    chebpoly(n) returns coefficients of the Chebychev polynomial of degree N.
    chebpoly(n,x) returns the Chebychev polynomial of degree N evaluated at X.

    Parameters
    ----------
    n : integer, scalar
        degree of Chebychev polynomial.
    x : array-like, optional
        evaluation points
    kind: 1 or 2, optional
        kind of Chebychev polynomial (default 1)

    Returns
    -------
    p : ndarray
        polynomial coefficients if x is None.
        Chebyshev polynomial evaluated at x otherwise

    Examples
    --------
    >>> import numpy as np
    >>> x = chebroot(3)
    >>> np.abs(chebpoly(3,x))<1e-15
    array([ True,  True,  True], dtype=bool)
    >>> chebpoly(3)
    array([ 4.,  0., -3.,  0.])
    >>> x2 = chebroot(4,kind=2)
    >>> np.abs(chebpoly(4,x2,kind=2))<1e-15
    array([ True,  True,  True,  True], dtype=bool)
    >>> chebpoly(4,kind=2)
    array([ 16.,   0., -12.,   0.,   1.])


    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    """
    if x is None:  # Calculate coefficients.
        if n == 0:
            p = np.ones(1)
        else:
            p = round(pow(2, n - 2 + kind) * poly(chebroot(n, kind=kind)))
            p[1::2] = 0
        return p
    else:  # Evaluate polynomial in chebychev form
        ck = zeros(n + 1)
        ck[0] = 1.
        return _chebval(np.atleast_1d(x), ck, kind=kind)


def chebfit(fun, n=10, a=-1, b=1, trace=False):
    """
    Computes the Chebyshevs coefficients

    so that f(x) can be approximated by:

                  n-1
           f(x) = sum ck*Tk(x)
                  k=0

    where Tk is the k'th Chebyshev polynomial of the first kind.

    Parameters
    ----------
    fun : callable
        function to approximate
    n : integer, scalar, optional
        number of base points (abscissas). Default n=10 (maximum 50)
    a,b : real, scalars, optional
        integration limits

    Returns
    -------
    ck : ndarray
        polynomial coefficients in Chebychev form.

    Examples
    --------
    Fit exp(x)

    >>> import matplotlib.pyplot as plt
    >>> a = 0; b = 2
    >>> ck = chebfit(np.exp,7,a,b);
    >>> x = np.linspace(0,4);
    >>> h=plt.plot(x, np.exp(x), 'r', x, chebval(x,ck,a,b), 'g.')
    >>> x1 = chebroot(9)*(b-a)/2+(b+a)/2
    >>> ck1 = chebfit(np.exp(x1))
    >>> h = plt.plot(x,np.exp(x), 'r', x, chebval(x,ck1,a,b),'g.')

    >>> plt.close()

    See also
    --------
    chebval

    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_nodes
    http://mathworld.wolfram.com/ChebyshevApproximationFormula.html

    W. Fraser (1965)
    "A Survey of Methods of Computing Minimax and Near-Minimax Polynomial
    Approximations for Functions of a Single Independent Variable"
    Journal of the ACM (JACM), Vol. 12 ,  Issue 3, pp 295 - 314
    """

    if (n > 50):
        warnings.warn('CHEBFIT should only be used for n<50')

    if hasattr(fun, '__call__'):
        x = map_to_interval(chebroot(n), a, b)
        f = fun(x)
        if trace:
            plt.plot(x, f, '+')
    else:
        f = fun
        n = len(f)
    #                     N-1
    #       c[k] = (2/N) sum w[n] f[n]*cos(pi*k*(2n+1)/(2N)), 0 <= k < N.
    #                    n=0
    #
    # w[0] = 0.5, w[n]=1 for n>0

    ck = dct(f[::-1]) / n
    ck[0] = ck[0] / 2.
    return ck[::-1]


def chebfit_dct(f, n=(10, ), domain=None):
    """
    Fit Chebyshev series to N-dimensional function
    so that f(x1, x2,..., xn) can be approximated by:

    .. math:: f(x_1, x_2,...,x_n) =
                    \\sum_{i,j,...k} c_i T_i(x_1)*...*c_k T_k(x_n) ,

    where Tk is the k'th Chebyshev polynomial of the first kind.

    Parameters
    ----------
    f : callable
        function to approximate
    n : list of integers, optional
        number of base points (abscissas) used for each dimension.
        Default n=10 (maximum 50)
    domain : vector [a1,b1,a2,b2 ,..., an, bn], optional
        defining the rectangle [a1, b1] x [a2, b2] x ...x [an, bn].
        (default domain = (-1,1) * len(n))

    Returns
    -------
    ck : ndarray
        polynomial coefficients in Chebychev form.

    Examples
    --------
    Fit exponential function

    >>> import matplotlib.pyplot as plt
    >>> domain = (0, 2)
    >>> ck = chebfit_dct(np.exp, 7, domain)
    >>> np.allclose(ck, [3.44152387e+00,   3.07252345e+00,   7.38000848e-01,
    ...                  1.20520053e-01,   1.48805268e-02,   1.47579673e-03,
    ...                  1.21719524e-04])
    True
    >>> x1 = map_to_interval(chebroot(9), *domain)
    >>> ck1 = chebfit(np.exp(x1))
    >>> np.allclose(ck1, [5.40019009e-07,   8.69418381e-06,   1.22261037e-04,
    ...                   1.47582673e-03,   1.48805283e-02,   1.20520053e-01,
    ...                   7.38000848e-01,   3.07252345e+00,   3.44152387e+00])
    True

    x = np.linspace(0,4)
    h = plt.plot(x, np.exp(x), 'r', x, chebvalnd(ck, x,ck,a,b), 'g.')
    h = plt.plot(x, np.exp(x), 'r', x, chebvalnd(ck1, x,ck1,a,b),'b.')
    plt.close()

    See also
    --------
    chebval, chebvalnd

    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_nodes
    http://mathworld.wolfram.com/ChebyshevApproximationFormula.html

    W. Fraser (1965)
    "A Survey of Methods of Computing Minimax and Near-Minimax Polynomial
    Approximations for Functions of a Single Independent Variable"
    Journal of the ACM (JACM), Vol. 12 ,  Issue 3, pp 295 - 314
    """
    n = np.atleast_1d(n)
    if np.any(n > 50):
        warnings.warn('CHEBFIT should only be used for n<50')

    if hasattr(f, '__call__'):
        if domain is None:
            domain = (-1, 1) * len(n)
        domain = np.atleast_2d(domain).reshape((-1, 2))
        xi = [map_to_interval(chebroot(ni), d[0], d[1])
              for ni, d in zip(n, domain)]
        Xi = np.meshgrid(*xi)
        ck = f(*Xi)
    else:
        ck = f
        n = ck.shape

    ndim = len(n)
    for i in range(ndim):
        ck = dct(ck[..., ::-1])
        ck[..., 0] = ck[..., 0] / 2.
        if i < ndim-1:
            ck = np.rollaxis(ck, axis=-1)
    return ck / np.product(n)


def idct(x, n=None):
    """
    Inverse Discrete Cosine Transform

                       N-1
           x[k] = 1/N sum w[n]*y[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                       n=0

           w(0) = 1/2
           w(n) = 1 for n>0

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.abs(x-idct(dct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(x-dct(idct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)

    Reference
    ---------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
    """
    return _idct(x, n=n, norm=None)*0.5/len(x)


def _chebval(x, ck, kind=1):
    """
    Evaluate polynomial in Chebyshev form.

    A polynomial of degree N in Chebyshev form is a polynomial p(x):

                 N
        p(x) =  sum ck*Tk(x)
                k=0
    or
                 N
        p(x) =  sum ck*Uk(x)
                k=0

    where Tk and Uk are the k'th Chebyshev polynomial of the first and second
    kind, respectively.

    References
    ----------
    http://en.wikipedia.org/wiki/Clenshaw_algorithm
    http://mathworld.wolfram.com/ClenshawRecurrenceFormula.html
    """
    n = len(ck)
    b_Nmi = zeros(x.shape)  # b_(N-i)
    b_Nmip1 = b_Nmi.copy()    # b_(N-i+1)
    x2 = 2 * x
    # Clenshaw reccurence
    for ix in range(n - 1):
        tmp = b_Nmi
        b_Nmi = x2 * b_Nmi - b_Nmip1 + ck[ix]
        b_Nmip1 = tmp
    return kind * x * b_Nmi - b_Nmip1 + ck[n - 1]


def chebval(x, ck, a=-1, b=1, kind=1, fill=None):
    """
    Evaluate polynomial in Chebyshev form at X

    A polynomial of degree N in Chebyshev form is a polynomial p(x) of the form

             N
    p(x) =  sum ck*Tk(x)
            k=0

    where Tk is the k'th Chebyshev polynomial of the first or second kind.

    Paramaters
    ----------
    x : array-like
        points to evaluate
    ck : array-like
        polynomial coefficients in Chebyshev form ordered from highest degree
        to zero
    a,b : real, scalars, optional
        limits for polynomial (Default -1,1)
    kind: 1 or 2, optional
        kind of Chebychev polynomial (default 1)
    fill : scalar, optional
        If provided, define value to return for `x < a` or `b < x`.

    Examples
    --------
    Plot Chebychev polynomial of the first kind and order 4:
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-1,1)
    >>> ck = np.zeros(5); ck[-1]=1
    >>> h = plt.plot(x,chebval(x,ck),x,chebpoly(4,x),'.')
    >>> plt.close()

    Fit exponential function:
    >>> import matplotlib.pyplot as plt
    >>> ck = chebfit(np.exp,7,0,2)
    >>> x = np.linspace(0,4);
    >>> h=plt.plot(x,chebval(x,ck,0,2),'g',x,np.exp(x))
    >>> plt.close()

    See also
    --------
    chebfit

    References
    ----------
    http://en.wikipedia.org/wiki/Clenshaw_algorithm
    http://mathworld.wolfram.com/ClenshawRecurrenceFormula.html
    """

    y = map_from_interval(atleast_1d(x), a, b)
    if fill is None:
        f = _chebval(y, ck, kind=kind)
    else:
        cond = (abs(y) <= 1)
        f = where(cond, 0, fill)
        if any(cond):
            yk = extract(cond, y)
            f[cond] = _chebval(yk, ck, kind=kind)
    return f


def chebder(ck, a=-1, b=1):
    """
    Differentiate Chebyshev polynomial

    Parameters
    ----------
    ck : array-like
        polynomial coefficients in Chebyshev form of function to differentiate
    a,b : real, scalars
        limits for polynomial(Default -1,1)

    Return
    ------
    cder : ndarray
        polynomial coefficients in Chebyshev form of the derivative

    Examples
    --------

    Fit exponential function:
    >>> import matplotlib.pyplot as plt
    >>> ck = chebfit(np.exp,7,0,2)
    >>> x = np.linspace(0,4)
    >>> ck2 = chebder(ck,0,2);
    >>> h = plt.plot(x,chebval(x,ck,0,2),'g',x,np.exp(x),'r')
    >>> plt.close()

    See also
    --------
    chebint
    chebfit

    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_polynomials

    W. Fraser (1965)
    "A Survey of Methods of Computing Minimax and Near-Minimax Polynomial
    Approximations for Functions of a Single Independent Variable"
    Journal of the ACM (JACM), Vol. 12 ,  Issue 3, pp 295 - 314
    """

    n = len(ck) - 1
    cder = zeros(n, dtype=asarray(ck).dtype)
    cder[0] = 2 * n * ck[0]
    cder[1] = 2 * (n - 1) * ck[1]
    for j in range(2, n):
        cder[j] = cder[j - 2] + 2 * (n - j) * ck[j]

    return cder * 2. / (b - a)  # Normalize to the interval b-a.


def chebint(ck, a=-1, b=1):
    """
    Integrate Chebyshev polynomial

    Parameters
    ----------
    ck : array-like
        polynomial coefficients in Chebyshev form of function to integrate.
    a,b : real, scalars
        limits for polynomial(Default -1,1)

    Return
    ------
    cint : ndarray
        polynomial coefficients in Chebyshev form of the integrated function

    Examples
    --------
    Fit exponential function:
    >>> import matplotlib.pyplot as plt
    >>> ck = chebfit(np.exp,7,0,2)
    >>> x = np.linspace(0,4)
    >>> ck2 = chebint(ck,0,2);
    >>> h=plt.plot(x,chebval(x,ck,0,2),'g',x,np.exp(x),'r.')
    >>> plt.close()

    See also
    --------
    chebder
    chebfit

    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_polynomials

    W. Fraser (1965)
    "A Survey of Methods of Computing Minimax and Near-Minimax Polynomial
    Approximations for Functions of a Single Independent Variable"
    Journal of the ACM (JACM), Vol. 12 ,  Issue 3, pp 295 - 314
    """

# int T0(x) = T1(x)+1
# int T1(x) = 0.5*(T2(x)/2-T0/2)
# int Tn(x) dx = 0.5*{Tn+1(x)/(n+1) - Tn-1(x)/(n-1)}
#             N
#    p(x) =  sum cn*Tn(x)
#            n=0

# int p(x) dx = sum cn * int(Tn(x)dx) =
# 0.5*sum cn *{Tn+1(x)/(n+1) - Tn-1(x)/(n-1)} = 0.5 sum (cn-1-cn+1)*Tn/n n>0

    n = len(ck)

    cint = zeros(n)
    con = 0.25 * (b - a)

    dif1 = diff(ck[-1::-2])
    ix1 = np.r_[1:n - 1:2]
    cint[ix1] = -(con * dif1) / ix1
    if n > 3:
        dif2 = diff(ck[-2::-2])
        ix2 = np.r_[2:n - 1:2]
        cint[ix2] = -(con * dif2) / ix2
    cint = cint[::-1]
    # cint(n) is a special case
    cint[-1] = (con * ck[n - 2]) / (n - 1)
    # Set integration constant
    cint[0] = 2 * np.sum((-1) ** np.r_[0:n - 1] * cint[-2::-1])
    return cint


class Cheb1d(object):
    coeffs = None
    order = None
    a = None
    b = None
    kind = None

    def __init__(self, ck, a=-1, b=1, kind=1):
        if isinstance(ck, Cheb1d):
            for key in ck.__dict__.keys():
                self.__dict__[key] = ck.__dict__[key]
            return
        cki = trim_zeros(atleast_1d(ck), 'b')
        if len(cki.shape) > 1:
            raise ValueError("Polynomial must be 1d only.")
        self.__dict__['coeffs'] = cki
        self.__dict__['order'] = len(cki) - 1
        self.__dict__['a'] = a
        self.__dict__['b'] = b
        self.__dict__['kind'] = kind

    def __call__(self, x):
        return chebval(x, self.coeffs, self.a, self.b, self.kind)

    def __array__(self, t=None):
        if t:
            return asarray(self.coeffs, t)
        else:
            return asarray(self.coeffs)

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "Cheb1d(%s)" % vals

    def __len__(self):
        return self.order

    def __str__(self):
        pass

    def __neg__(self):
        new = Cheb1d(self)
        new.coeffs = -self.coeffs
        return new

    def __pos__(self):
        return self

    def __add__(self, other):
        other = Cheb1d(other)
        new = Cheb1d(self)
        new.coeffs = polyadd(self.coeffs, other.coeffs)
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = Cheb1d(other)
        new = Cheb1d(self)
        new.coeffs = polysub(self.coeffs, other.coeffs)
        return new

    def __rsub__(self, other):
        other = Cheb1d(other)
        new = Cheb1d(self)
        new.coeffs = polysub(other.coeffs, new.coeffs)
        return new

    def __eq__(self, other):
        other = Cheb1d(other)
        return (all(self.coeffs == other.coeffs) and (self.a == other.a) and
                (self.b == other.b) and (self.kind == other.kind))

    def __ne__(self, other):
        return any(self.coeffs != other.coeffs) or (self.a != other.a) or (
            self.b != other.b) or (self.kind != other.kind)

    def __setattr__(self, key, val):
        raise ValueError("Attributes cannot be changed this way.")

    def __getattr__(self, key):
        if key in ['c', 'coef', 'coefficients']:
            return self.coeffs
        elif key in ['o']:
            return self.order
        elif key in ['a']:
            return self.a
        elif key in ['b']:
            return self.b
        elif key in ['k']:
            return self.kind
        else:
            try:
                return self.__dict__[key]
            except KeyError:
                raise AttributeError(
                    "'%s' has no attribute '%s'" %
                    (self.__class__, key))

    def __getitem__(self, val):
        if val > self.order:
            return 0
        if val < 0:
            return 0
        return self.coeffs[val]

    def __setitem__(self, key, val):
        # ind = self.order - key
        if key < 0:
            raise ValueError("Does not support negative powers.")
        if key > self.order:
            zr = zeros(key - self.order, self.coeffs.dtype)
            self.__dict__['coeffs'] = concatenate((self.coeffs, zr))
            self.__dict__['order'] = key
        self.__dict__['coeffs'][key] = val
        return

    def __iter__(self):
        return iter(self.coeffs)

    def integ(self, m=1):
        """
        Return an antiderivative (indefinite integral) of this polynomial.

        Refer to `chebint` for full documentation.

        See Also
        --------
        chebint : equivalent function

        """
        integ = Cheb1d(self)
        integ.coeffs = chebint(self.coeffs, self.a, self.b)
        return integ

    def deriv(self, m=1):
        """
        Return a derivative of this polynomial.

        Refer to `chebder` for full documentation.

        See Also
        --------
        chebder : equivalent function

        """
        der = Cheb1d(self)
        der.coeffs = chebder(self.coeffs, self.a, self.b)
        return der


def padefit(c, m=None):
    """
    Rational polynomial fitting from polynomial coefficients

    Parameters
    ----------
    c : array-like
        coefficients of power series expansion from highest degree to zero.
    m : scalar integer
        order of denominator polynomial. (Default floor((len(c)-1)/2))

    Returns
    -------
    num, den : poly1d
        numerator and denominator polynomials for the pade approximation

    If the function is well approximated by
              M+N+1
       f(x) = sum c(2*n+2-k)*x^k
              k=0

    then the pade approximation is given by
               M
              sum c1(n-k+1)*x^k
              k=0
    f(x) = ------------------------
              N
              sum c2(n-k+1)*x^k
              k=0

    Note: c must be ordered for direct use with polyval

    Example
    -------
    Pade approximation to exp(x)
    >>> import scipy.special as sp
    >>> import matplotlib.pyplot as plt
    >>> c = poly1d(1./sp.gamma(np.r_[6+1:0:-1]))
    >>> [p, q] = padefit(c)
    >>> p; q
    poly1d([ 0.00277778,  0.03333333,  0.2       ,  0.66666667,  1.        ])
    poly1d([ 0.03333333, -0.33333333,  1.        ])

    >>> x = np.linspace(0,4);
    >>> h = plt.plot(x,c(x),x,p(x)/q(x),'g-', x,np.exp(x),'r.')
    >>> plt.close()

    See also
    --------
    scipy.misc.pade

    """
    if not m:
        m = int(floor((len(c) - 1) * 0.5))
    c = asarray(c)
    return pade(c[::-1], m)


def test_pade():
    cof = array(([1.0, 1.0, 1.0 / 2, 1. / 6, 1. / 24]))
    p, q = pade(cof, 2)
    t = arange(0, 2, 0.1)
    assert(all(abs(p(t) / q(t) - exp(t)) < 0.3))


def padefitlsq(fun, m, k, a=-1, b=1, trace=False, x=None, end_points=True):
    """
    Rational polynomial fitting. A minimax solution by least squares.

    Parameters
    ----------
    fun : callable or or a two column matrix
           f=[x,f(x)]  where length(x)>(m+k+1)*8.
    m, k : integer
        number of coefficients of the numerator and denominater, respectively.
    a, b : real scalars
        evaluation limits, (default a=-1,b=1)

    Returns
    -------
    num, den : poly1d
        numerator and denominator polynomials for the pade approximation
    dev : ndarray
        maximum absolute deviation of the approximation

    The pade approximation is given by
               m
              sum c1[m-i]*x**i
              i=0
    f(x) = ------------------------
               k
              sum c2[k-i]*x**i
              i=0

    If F is a two column matrix, [x f(x)], a good choice for x is:

    x = cos(pi/(N-1)*(N-1:-1:0))*(b-a)/2+ (a+b)/2, where N = (m+k+1)*8;

    Note: c1 and c2 are ordered for direct use with polyval

    Example
    -------

    Pade approximation to exp(x) between 0 and 2
    >>> import matplotlib.pyplot as plt
    >>> [c1, c2] = padefitlsq(np.exp,3,3,0,2)
    >>> c1; c2
    poly1d([ 0.01443847,  0.128842  ,  0.55284547,  0.99999962])
    poly1d([-0.0049658 ,  0.07610473, -0.44716929,  1.        ])

    >>> x = np.linspace(0,4)
    >>> h = plt.plot(x, polyval(c1,x)/polyval(c2,x),'g')
    >>> h = plt.plot(x, np.exp(x), 'r')

    See also
    --------
    padefit

    Reference
    ---------
    William H. Press, Saul Teukolsky,
    William T. Wetterling and Brian P. Flannery (1997)
    "Numerical recipes in Fortran 77", Vol. 1, pp 197-20
    """

    NFAC = 8
    BIG = 1e30
    MAXIT = 5

    smallest_devmax = BIG
    ncof = m + k + 1
    # Number of points where function is evaluated, i.e. fineness of mesh
    npt = NFAC * ncof

    if x is None:
        if end_points:
            # Use the location of the local extreme values of
            # the Chebychev polynomial of the first kind of degree NPT-1.
            x = map_to_interval(chebextr(npt - 1), a, b)
        else:
            # Use the roots of the Chebychev polynomial of the first kind of
            # degree NPT. Note this is useful if there are singularities close
            # to the endpoints.
            x = map_to_interval(chebroot(npt, kind=1), a, b)

    if hasattr(fun, '__call__'):
        fs = fun(x)
    else:
        fs = fun
        n = len(fs)
        if n < npt:
            warnings.warn(
                'Check the result! ' +
                'Number of function values should be at least: %d' % npt)

    if trace:
        plt.plot(x, fs, '+')

    wt = np.ones((npt))
    ee = np.ones((npt))
    mad = 0

    u = zeros((npt, ncof))
    for ix in range(MAXIT):
        # Set up design matrix for least squares fit.
        pow1 = wt
        bb = pow1 * (fs + abs(mad) * sign(ee))

        for jx in range(m + 1):
            u[:, jx] = pow1
            pow1 = pow1 * x

        pow1 = -bb
        for jx in range(m + 1, ncof):
            pow1 = pow1 * x
            u[:, jx] = pow1

        [u1, w, v] = linalg.svd(u, full_matrices=False)
        cof = where(w == 0, 0.0, np.dot(bb, u1) / w)
        cof = np.dot(cof, v)

        # Tabulate the deviations and revise the weights
        ee = polyval(cof[m::-1], x) / \
            polyval(cof[ncof:m:-1].tolist() + [1, ], x) - fs

        wt = np.abs(ee)
        devmax = max(wt)
        mad = wt.mean()  # mean absolute deviation

        # Save only the best coefficients found
        if (devmax <= smallest_devmax):
            smallest_devmax = devmax
            c1 = cof[m::-1]
            c2 = cof[ncof:m:-1].tolist() + [1, ]

        if trace:
            print('Iteration=%d,  max error=%g' % (ix, devmax))
            plt.plot(x, fs, x, ee + fs)
    return poly1d(c1), poly1d(c2)


def main():

    [c1, c2] = padefitlsq(exp, 3, 3, 0, 2)

    x = linspace(0, 4)
    plt.plot(x, polyval(c1, x) / polyval(c2, x), 'g')
    plt.plot(x, exp(x), 'r')

    import scipy.special as sp

    p = [[1, 1, 1], [2, 2, 2]]
    pi = polyint(p, 1)
    _pr = polyreloc(p, 2)
    _pd = polyder(p)
    _st = poly2str(p)
    c = poly1d(
        1. /
        sp.gamma(
            np.r_[
                6 +
                1:0:-
                1]))  # polynomial coeff exponential function
    [p, q] = padefit(c)
    x = linspace(0, 4)
    plt.plot(x, c(x), x, p(x) / q(x), 'g-', x, exp(x), 'r.')
    plt.close()
    x = arange(4)
    dx = dct(x)
    _idx = idct(dx)

    a = 0
    b = 2
    ck = chebfit(exp, 6, a, b)
    _t = chebval(0, ck, a, b)
    x = linspace(0, 2, 6)
    plt.plot(x, exp(x), 'r', x, chebval(x, ck, a, b), 'g.')
    # x1 = chebroot(9).'*(b-a)/2+(b+a)/2 ;
    # ck1 =chebfit([x1 exp(x1)],9,a,b);
    # plot(x,exp(x),'r'), hold on
    # plot(x,chebval(x,ck1,a,b),'g'), hold off

    _t = poly2hstr([1, 1, 2])
    py = [1, 0]
    px = polyshift(py, 0, 5)
    _t1 = polyval(px, [0, 2.5, 5])  # % This is the same as the line below
    _t2 = polyval(py, [-1, 0, 1])

    px = [1, 0]
    py = polyishift(px, 0, 5)
    t1 = polyval(px, [0, 2.5, 5])  # % This is the same as the line below
    t2 = polyval(py, [-1, 0, 1])
    print(t1, t2)


def test_polydeg():
    x = np.linspace(0, 10, 300)
    y = np.sin(x ** 3 / 100) ** 2 + 0.05 * np.random.randn(x.size)
    n = polydeg(x, y)
    # n = 2
    p = orthofit(x, y, n)
    xi = linspace(x.min(), x.max())
    ys0 = orthoval(p, x)
    ys = orthoval(p, xi)

    ys2 = orthoval(p, xi)
    plt.plot(x, y, '.', x, ys0, 'k', xi, ys, 'r', xi, ys2, 'r.')
    p0 = ortho2poly(p)
    p1 = polyfit(x, ys0, n)
    plt.plot(xi, polyval(p0, xi), 'g-.', xi, polyval(p1, xi), 'go')
    plt.show('hold')


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


def chebvandernd(deg, *xi):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x1, x2, ..., xn)`. If `l, m, n` are the given degrees in
    `x1, x2, x3`, then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = T_i(x1)*T_j(x2)*T_k(x3),

    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= k <= n`.  The leading
    indices of `V` index the points `(x, y, z)` and the last index encodes
    the degrees of the Chebyshev polynomials.

    If ``V = chebvandernd([xdeg, ydeg, zdeg], x, y, z)``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and ``np.dot(V, c.flat)`` and ``chebvalnd(c, x, y, z)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of N-D Chebyshev
    series of the same degrees and sample points.

    Parameters
    ----------
    deg : list of ints
        List of maximum degrees of the form [x1_deg, x2_deg, ...,xn_deg].
    x1, x2, ..., xn : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.

    Returns
    -------
    vander : ndarray
        The shape of the returned matrix is ``x1.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)*...*(deg[n-1]+1)`.  The dtype
        will be the same as the converted `x1`, `x2`, ... `xn`.

    See Also
    --------
    chebvander, chebvalnd, chebfitnd
    """
    ideg = [int(d) for d in deg]
    is_valid = np.array([di == d and di >= 0 for di, d in zip(ideg, deg)])
    if np.any(is_valid != 1):
        raise ValueError("degrees must be non-negative integers")
    ndim = len(xi)
    if len(ideg) != ndim:
        msg = 'length of deg must be the same as number of dimensions'
        raise ValueError(msg)

    xi = np.array(xi, copy=0) + 0.0
    chebvander = np.polynomial.chebyshev.chebvander
    shape0 = xi[0].shape
    s0 = (1,) * ndim
    vxi = [chebvander(x, d).reshape(shape0 + s0[:i] + (-1,) + s0[i + 1::])
           for i, (d, x) in enumerate(zip(ideg, xi))]

    v = reduce(np.multiply, vxi)

    return v.reshape(v.shape[:-ndim] + (-1,))


def chebfitnd(xi, f, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Chebyshev series to N-dimensional data.
    Return the coefficients of a Chebyshev series of degree `deg` that is the
    least squares fit to the data values `f` given at points
    `x1`, `x2`,..., `xn`

    The fitted polynomial(s) are in the form
    .. math::  p(x,y) = c_00 + c_11 * T_1(x)*T_1(y) + ..c_ij * T_i(x)*T_j(y).
                        + c_nm * T_n(x)*T_m(y),
    where `n`, `m` is `deg`.

    Parameters
    ----------
    xi: tuple
        x1-, x2-,....xn-coordinates of the sample points.
    f : array_like
        function values at the sample points ``(x1[i], x2[i], ..., xn[i])``.
    deg : list
        Degrees of the fitting series in the x1, x2, ..., xn directions,
        respectively.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is size(x1)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, optional
        Weights. If not None, the contribution of each point
        ``(x1[i], x2[i], ..., xn[i])`` to the fit is weighted by `w[i]`.
        Ideally the weights are chosen so that the errors of the products
        ``w[i]*f[i]`` all have the same variance.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M1, M2,..., Mn)
        Chebyshev coefficients ordered from low to high.
    [residuals, rank, singular_values, rcond] : list
        These values are only returned if `full` = True
        resid -- sum of squared residuals of the least squares fit
        rank -- the numerical rank of the scaled Vandermonde matrix
        sv -- singular values of the scaled Vandermonde matrix
        rcond -- value of `rcond`.
        For more details, see `linalg.lstsq`.
    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if `full` = False.  The
        warnings can be turned off by
        >>> import warnings
        >>> warnings.simplefilter('ignore', RankWarning)

    See Also
    --------
    chebvalnd, chebgridnd

    Notes
    -----
    The solution is the coefficients of the Chebyshev series `p` that
    minimizes the sum of the weighted squared errors
    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
    where :math:`w_j` are the weights. This problem is solved by setting up
    as the (typically) overdetermined matrix equation
    .. math:: V(x, y) * c = w * y,
    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.
    If some of the singular values of `V` are so small that they are
    neglected, then a `RankWarning` will be issued. This means that the
    coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.
    Fits using Chebyshev series are usually better conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting
    Examples
    --------
    """
    # xi = np.array(xi, copy=0) + 0.0
    z = np.array(f)
    degrees = np.asarray(deg, dtype=int)
    orders = degrees + 1
    order = np.product(orders)

    ndims = np.array([x.ndim for x in xi])
    ndim = len(ndims)
    sizes = np.array([x.size for x in xi])
    if np.any(ndims != ndim) or z.ndim != ndim:
        raise TypeError("expected %dD array for x1, x2,...,xn and f" % ndim)
    if np.any(sizes == 0):
        raise TypeError("expected non-empty vector for xi")

    lhs = chebvandernd(degrees, *xi).reshape((-1, order))
    rhs = z.ravel()
    if w is not None:
        w = np.asarray(w).ravel() + 0.0
        if len(lhs) != len(w):
            raise TypeError("expected x and w to have same length")
        lhs = lhs * w
        rhs = rhs * w

    if rcond is None:
        rcond = xi[0].size * np.finfo(x.dtype).eps

    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(axis=0))
    else:
        scl = np.sqrt(np.square(lhs).sum(axis=0))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    c, resids, rank, s = np.linalg.lstsq(lhs/scl, rhs, rcond)
    c = (c/scl).reshape(orders)

    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, pu.RankWarning)

    if full:
        return c, [resids, rank, s, rcond]
    else:
        return c


def chebvalnd(c, *xi):
    """
    Evaluate a N-D Chebyshev series at points (x1, x2, ..., xn).

    This function returns the values:

    .. math:: p(x1,x2,...,xn) =
            \\sum_{i,j,...,k} c_{i,j,...,k} * T_i(x1) * T_j(x2)*...* T_k(xn)

    The parameters `x1`, `x2`, ...., `xn` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x1`, `x2`, ..., `xn` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than N dimensions, ones are implicitly appended to its
    shape to make it N-D. The shape of the result will be c.shape[3:] +
    x1.shape.

    Parameters
    ----------
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,...,k is contained in ``c[i,j,...,k]``. If `c` has
        dimension greater than N the remaining indices enumerate multiple sets
        of coefficients.
    x1, x2,..., xn : array_like, compatible object
        The N dimensional series is evaluated at the points
        `(x1, x2,...,xn)`, where `x1`, `x2`,..., `xn` must have the same shape.
        If any of `x1`, `x2`, ..., `xn` is a list or tuple, it is first
        converted to an ndarray, otherwise it is left unchanged and if it isn't
        an ndarray it is  treated as a scalar.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    chebval, chebgridnd, chebfitnd
    """
    try:
        xi = np.array(xi, copy=0)
    except:
        raise ValueError('x, y, z are incompatible')
    chebval = np.polynomial.chebyshev.chebval
    c = chebval(xi[0], c)
    for x in xi[1:]:
        c = chebval(x, c, tensor=False)
    return c


def chebgridnd(c, *xi):
    """
    Evaluate a N-D Chebyshev series on the Cartesian product of x1, x2,..., xn.

    This function returns the values:

    .. math:: p(a,b,...) = \\sum_{i,j,...} c_{i,j,...} * T_i(a) * T_j(b) *  ...

    where the points `(a, b, ...)` consist of all points formed by taking
    `a` from `x1`, `b` from `x2`, and so on. The resulting points form
    a grid with `x1` in the first dimension, `x2` in the second, and so on.

    The parameters `x1`, `x2`, ... and `xn` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x1`, `x2`,... and `xn` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than N dimensions, ones are implicitly appended to
    its shape to make it N-D. The shape of the result will be c.shape[3:] +
    x1.shape + x2.shape + ... + xn.shape

    Parameters
    ----------
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.
    x1, x2,..., xn : ndarray, compatible object
        1-D arrays representing the coordinates of a grid.
        The N dimensional series is evaluated at the points in the
        Cartesian product of `x1`, `x2`, ... and `xn`.  If `xi`, is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.

    Returns
    -------
    values : ndarray, compatible object
        The values of the N dimensional polynomial at points in the Cartesian
        product of `x1`, `x2`, ... and `xn`.

    See Also
    --------
    chebval, chebvalnd, chebfitnd
    """
    chebval = np.polynomial.chebyshev.chebval
    for x in xi:
        c = chebval(x, c)
    return c


def test_chebfit1d():
    def f(x):
        return np.exp(-x**2)

    # x = chebroot(n=64, kind=1)
    # z = f(x)

    c = chebfit(f, n=64)[::-1]

    xi = np.linspace(-1, 1, 151)
    zi = np.polynomial.chebyshev.chebval(xi, c)

    # plt.plot(xi, zi,'.', xi, f(xi))
    plt.semilogy(xi, np.abs(zi-f(xi)))
    plt.show('hold')


def test_chebfit2d():
    n = 3
    xorder, yorder = n-1, n-1
    x = chebroot(n=n, kind=1)
    xgrid, ygrid = meshgrid(x, x)

    def f(x, y):
        return np.exp(-x**2-6*y**2)
    zgrid = f(xgrid, ygrid)

    # v2d = np.polynomial.chebyshev.chebvander2d(xgrid, ygrid,
    #                   [xorder,yorder]).reshape((-1, (xorder+1)*(yorder+1)))
    # coeff, residuals, rank, s = np.linalg.lstsq(v2d, zgrid.ravel())
    # doeff = coeff.reshape(xorder+1,yorder+1)
    _dcoeff2 = chebfitnd((xgrid, ygrid), zgrid, [xorder, yorder])
    dcoeff = chebfit_dct(f, n=(xorder+1, yorder+1))

    xi = np.linspace(-1, 1, 151)
    Xi, Yi = np.meshgrid(xi, xi)
    Zi = f(Xi, Yi)
    zzi = chebvalnd(dcoeff, Xi, Yi)
    _devi = Zi - zzi
    # plot residuals
    # zz = np.polynomial.chebyshev.chebval2d(xgrid, ygrid, dcoeff)
    zz = chebvalnd(dcoeff, xgrid, ygrid)
    dev = zgrid - zz
    # plt.spy(np.abs(dcoeff)>1e-13)
    plt.contourf(xgrid, ygrid, np.abs(dev))
    # plt.contourf(Xi, Yi, np.abs(devi))
    plt.colorbar()
    # plt.semilogy(np.abs(devi.ravel()))
    plt.show('hold')


if __name__ == '__main__':
    if False:  # True: #
        main()
    else:
        test_chebfit2d()
        # test_docstrings()
        # test_polydeg()
