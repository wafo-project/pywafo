"""
    Extended functions to operate on polynomials
"""
#-------------------------------------------------------------------------------
# Name:        polynomial
# Purpose:     Functions to operate on polynomials.
#
# Author:      pab
# polyXXX functions are based on functions found in the matlab toolbox polyutil written by
# Author:      Peter J. Acklam
# E-mail:      pjacklam@online.no
# WWW URL:     http://home.online.no/~pjacklam
#
# Created:     30.12.2008
# Copyright:   (c) pab 2008
# Licence:     LGPL
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import warnings
import pylab as plb
import numpy as np
from numpy.fft import fft, ifft
from numpy import (zeros, ones, zeros_like, atleast_1d, array, asarray, newaxis, arange, #@UnresolvedImport 
    logical_or, abs, any, pi, cos, round, diff, all, r_, exp, hstack, trim_zeros, #@UnresolvedImport
    where, extract, dot, linalg, sign, concatenate, floor, isreal, conj, remainder, #@UnresolvedImport
    linspace) #@UnresolvedImport
from numpy.lib.polynomial import * #@UnusedWildImport
from scipy.misc.common import pade
__all__ = np.lib.polynomial.__all__
__all__ = __all__ + ['pade', 'padefit', 'polyreloc', 'polyrescl', 'polytrim', 'poly2hstr', 'poly2str',
    'polyshift', 'polyishift', 'map_from_intervall', 'map_to_intervall',
    'cheb2poly', 'chebextr', 'chebroot', 'chebpoly', 'chebfit', 'chebval',
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
    >>> P = np.polyint(p, 3, k=[6,5,3])
    >>> P
    poly1d([ 0.01666667,  0.04166667,  0.16666667,  3.        ,  5.        ,  3.        ])

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
        raise ValueError, "Order of integral must be positive (see polyder)"
    if k is None:
        k = zeros(m, float)
    k = atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0] * ones(m, float)
    if len(k) < m:
        raise ValueError, \
              "k must be a scalar or a rank-1 array of length 1 or >m."
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
            k0 = k[0] * ones((1, pieces), dtype=int)
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
        raise ValueError, "Order of derivative must be positive (see polyint)"
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
        vector or matrix of column vectors of polynomial coefficients to relocate.
        (Polynomial coefficients are in decreasing order.)
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
    r = atleast_1d(p).copy()
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
        vector or matrix of column vectors of polynomial coefficients to rescale.
        (Polynomial coefficients are in decreasing order.)
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
    r = atleast_1d(p)
    n = r.shape[0]

    xscale = (float(x) ** arange(1 - n , 1))
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
        r = atleast_1d(p).copy()
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

    coefs = polytrim(atleast_1d(p))
    order = len(coefs) - 1 # Order of polynomial.
    s = ''    # Initialize output string.
    ix = 1;
    for expon in range(order, -1, -1):
        coef = coefs[order - expon]
        #% There is no point in adding a zero term (except if it's the only
        #% term, but we'll take care of that later).
        if coef == 0:
            ix += 1
        else:
        #% Append exponent if necessary.
            if ix > 1:
                exponstr = '%.0f' % ix
                s = '%s**%s' % (s, exponstr);
                ix = 1
            #% Is it the first term?
            isfirst = s == ''

            # We need the coefficient only if it is different from 1 or -1 or
            # when it is the constant term.
            needcoef = ((abs(coef) != 1) | (expon == 0) & isfirst) | 1 - isfirst

            # We need the variable except in the constant term.
            needvar = (expon != 0)

            #% Add sign, but we don't need a leading plus-sign.
            if isfirst:
                if coef < 0:
                    s = '-'  #        % Unary minus.
            else:
                if coef < 0:
                    s = '%s - ' % s  #    % Binary minus (subtraction).
                else:
                    s = '%s + ' % s  #  % Binary plus (addition).


            #% Append the coefficient if it is different from one or when it is
            #% the constant term.
            if needcoef:
                coefstr = '%.20g' % abs(coef)
                s = '%s%s' % (s, coefstr)

            #% Append variable if necessary.
            if needvar:
                #% Append a multiplication sign if necessary.
                if needcoef:
                    if 1 - isfirst:
                        s = '(%s)' % s
                    s = '%s*' % s
                s = '%s%s' % (s, var)

    #% Now treat the special case where the polynomial is zero.
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
    coeffs = polytrim(atleast_1d(p))

    N = len(coeffs) - 1

    for k in range(len(coeffs)):
        coefstr = '%.4g' % abs(coeffs[k])
        if coefstr[-4:] == '0000':
            coefstr = coefstr[:-5]
        power = (N - k)
        if power == 0:
            if coefstr != '0':
                newstr = '%s' % (coefstr,)
            else:
                if k == 0:
                    newstr = '0'
                else:
                    newstr = ''
        elif power == 1:
            if coefstr == '0':
                newstr = ''
            elif coefstr == 'b' or coefstr == '1':
                newstr = var
            else:
                newstr = '%s*%s' % (coefstr, var)
        else:
            if coefstr == '0':
                newstr = ''
            elif coefstr == 'b' or coefstr == '1':
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

def polyshift(py, a= -1, b=1):
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

def polyishift(px, a= -1, b=1):
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

def map_from_interval(x, a, b) :
    """F(x), where F: [a,b] -> [-1,1]."""
    return (x - (b + a) / 2.0) * (2.0 / (b - a))

def map_to_interval(x, a, b) :
    """F(x), where F: [-1,1] -> [a,b]."""
    return (x * (b - a) + (b + a)) / 2.0

def poly2cheb(p, a= -1, b=1):
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

def cheb2poly(ck, a= -1, b=1):
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
    y = r_[2 / (b - a), -(a + b) / (b - a)]
    y2 = 2. * y

    # Clenshaw recurence
    for ix in xrange(n - 1):
        tmp = b_Nmi
        b_Nmi = polymul(y2, b_Nmi) # polynomial multiplication
        nb = len(b_Nmip1)
        b_Nmip1[-1] = b_Nmip1[-1] - ck[ix]
        b_Nmi[-nb::] = b_Nmi[-nb::] - b_Nmip1
        b_Nmip1 = tmp

    p = polymul(y, b_Nmi) # polynomial multiplication
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
    return - cos((pi * arange(n + 1)) / n);

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
    return - cos(pi * (arange(n) + 0.5 * kind) / (n + kind - 1));


def chebpoly(n, x=None, kind=1):
    """
    Return Chebyshev polynomial of the first or second kind.

    These polynomials are orthogonal on the interval [-1,1], with
    respect to the weight function w(x) = (1-x^2)^(-1/2+kind-1).

    chebpoly(n) returns the coefficients of the Chebychev polynomial of degree N.
    chebpoly(n,x) returns the Chebychev polynomial of degree N evaluated in X.

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
            p = ones(1)
        else:
            p = round(pow(2, n - 2 + kind) * poly(chebroot(n, kind=kind)))
            p[1::2] = 0;
        return p
    else: #   Evaluate polynomial in chebychev form
        ck = zeros(n + 1)
        ck[0] = 1.
        return _chebval(atleast_1d(x), ck, kind=kind)

def chebfit(fun, n=10, a= -1, b=1, trace=False):
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

    >>> import pylab as pb
    >>> a = 0; b = 2
    >>> ck = chebfit(pb.exp,7,a,b);
    >>> x = pb.linspace(0,4);
    >>> h=pb.plot(x,pb.exp(x),'r',x,chebval(x,ck,a,b),'g.')
    >>> x1 = chebroot(9)*(b-a)/2+(b+a)/2
    >>> ck1 = chebfit(pb.exp(x1))
    >>> h=pb.plot(x,pb.exp(x),'r',x,chebval(x,ck1,a,b),'g.')

    >>> pb.close()

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
        f = fun(x);
        if trace:
            plb.plot(x, f, '+')
    else:
        f = fun
        n = len(f)
        #raise ValueError('Function must be callable!')
    #                     N-1
    #       c(k) = (2/N) sum w(n) f(n)*cos(pi*k*(2n+1)/(2N)), 0 <= k < N.
    #                    n=0
    #
    # w(0) = 0.5, w(n)=1 for n>0
    ck = dct(f[::-1]) / n
    ck[0] = ck[0] / 2.
    return ck[::-1]

def dct(x, n=None):
    """
    Discrete Cosine Transform

                      N-1
           y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                      n=0

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
    
    x = atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    if x.shape[-1] < n:
        n_shape = x.shape[:-1] + (n - x.shape[-1],)
        xx = hstack((x, zeros(n_shape)))
    else:
        xx = x[..., :n]

    real_x = all(isreal(xx))
    if (real_x and (remainder(n, 2) == 0)):
        xp = 2 * fft(hstack((xx[..., ::2], xx[..., ::-2])))
    else:
        xp = fft(hstack((xx, xx[..., ::-1])))
        xp = xp[..., :n]

    w = exp(-1j * arange(n) * pi / (2 * n))

    y = xp * w

    if real_x:
        return y.real
    else:
        return y

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

    
    x = atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    w = exp(1j * arange(n) * pi / (2 * n))

    if x.shape[-1] < n:
        n_shape = x.shape[:-1] + (n - x.shape[-1],)
        xx = hstack((x, zeros(n_shape))) * w
    else:
        xx = x[..., :n] * w

    real_x = all(isreal(x))
    if (real_x and (remainder(n, 2) == 0)):
        xx[..., 0] = xx[..., 0] * 0.5
        yp = ifft(xx)
        y = zeros(xx.shape, dtype=complex)
        y[..., ::2] = yp[..., :n / 2]
        y[..., ::-2] = yp[..., n / 2::]
    else:
        yp = ifft(hstack((xx, zeros_like(xx[..., 0]), conj(xx[..., :0:-1]))))
        y = yp[..., :n]

    if real_x:
        return y.real
    else:
        return y

def _chebval(x, ck, kind=1):
    """
    Evaluate polynomial in Chebyshev form.

    A polynomial of degree N in Chebyshev form is a polynomial p(x) of the form:

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
    b_Nmi = zeros(x.shape) # b_(N-i)
    b_Nmip1 = b_Nmi.copy()    # b_(N-i+1)
    x2 = 2 * x
    # Clenshaw reccurence
    for ix in xrange(n - 1):
        tmp = b_Nmi
        b_Nmi = x2 * b_Nmi - b_Nmip1 + ck[ix]
        b_Nmip1 = tmp
    return kind * x * b_Nmi - b_Nmip1 + ck[n - 1]


def chebval(x, ck, a= -1, b=1, kind=1, fill=None):
    """
    Evaluate polynomial in Chebyshev form at X

    A polynomial of degree N in Chebyshev form is a polynomial p(x) of the form:

             N
    p(x) =  sum ck*Tk(x)
            k=0

    where Tk is the k'th Chebyshev polynomial of the first or second kind.

    Paramaters
    ----------
    x : array-like
        points to evaluate
    ck : array-like
        polynomial coefficients in Chebyshev form ordered from highest degree to zero
    a,b : real, scalars, optional
        limits for polynomial (Default -1,1)
    kind: 1 or 2, optional
        kind of Chebychev polynomial (default 1)
    fill : scalar, optional
        If provided, define value to return for `x < a` or `b < x`.

    Examples
    --------
    Plot Chebychev polynomial of the first kind and order 4:
    >>> import pylab as pb
    >>> x = pb.linspace(-1,1)
    >>> ck = pb.zeros(5); ck[-1]=1
    >>> h = pb.plot(x,chebval(x,ck),x,chebpoly(4,x),'.')
    >>> pb.close()

    Fit exponential function:
    >>> import pylab as pb
    >>> ck = chebfit(pb.exp,7,0,2)
    >>> x = pb.linspace(0,4);
    >>> h=pb.plot(x,chebval(x,ck,0,2),'g',x,pb.exp(x))
    >>> pb.close()

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


def chebder(ck, a= -1, b=1):
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
    >>> import pylab as pb
    >>> ck = chebfit(pb.exp,7,0,2)
    >>> x = pb.linspace(0,4)
    >>> ck2 = chebder(ck,0,2);
    >>> h = pb.plot(x,chebval(x,ck,0,2),'g',x,pb.exp(x),'r')
    >>> pb.close()

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
    for j in xrange(2, n):
        cder[j] = cder[j - 2] + 2 * (n - j) * ck[j]

    return cder * 2. / (b - a) # Normalize to the interval b-a.

def chebint(ck, a= -1, b=1):
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
    >>> import pylab as pb
    >>> ck = chebfit(pb.exp,7,0,2)
    >>> x = pb.linspace(0,4)
    >>> ck2 = chebint(ck,0,2);
    >>> h=pb.plot(x,chebval(x,ck,0,2),'g',x,pb.exp(x),'r.')
    >>> pb.close()

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

# int p(x) dx = sum cn * int(Tn(x)dx) = 0.5*sum cn *{Tn+1(x)/(n+1) - Tn-1(x)/(n-1)}
# = 0.5 sum (cn-1-cn+1)*Tn/n n>0

    n = len(ck)

    cint = zeros(n)
    con = 0.25 * (b - a)

    dif1 = diff(ck[-1::-2])
    ix1 = r_[1:n - 1:2]
    cint[ix1] = -(con * dif1) / ix1
    if n > 3:
        dif2 = diff(ck[-2::-2])
        ix2 = r_[2:n - 1:2]
        cint[ix2] = -(con * dif2) / ix2
    cint = cint[::-1]
    #% cint(n) is a special case
    cint[-1] = (con * ck[n - 2]) / (n - 1)
    cint[0] = 2 * np.sum((-1) ** r_[0:n - 1] * cint[-2::-1]) # Set integration constant    
    return cint

class Cheb1d(object):
    coeffs = None
    order = None
    a = None
    b = None
    kind = None
    def __init__(self, ck, a= -1, b=1, kind=1):
        if isinstance(ck, Cheb1d):
            for key in ck.__dict__.keys():
                self.__dict__[key] = ck.__dict__[key]
            return
        cki = trim_zeros(atleast_1d(ck), 'b')
        if len(cki.shape) > 1:
            raise ValueError, "Polynomial must be 1d only."
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
        return (all(self.coeffs == other.coeffs) and (self.a == other.a)
        and (self.b == other.b) and (self.kind == other.kind))

    def __ne__(self, other):
        return any(self.coeffs != other.coeffs) or (self.a != other.a) or (self.b != other.b) or (self.kind != other.kind)

    def __setattr__(self, key, val):
        raise ValueError, "Attributes cannot be changed this way."

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
                raise AttributeError("'%s' has no attribute '%s'" % (self.__class__, key))
    def __getitem__(self, val):
        if val > self.order:
            return 0
        if val < 0:
            return 0
        return self.coeffs[val]

    def __setitem__(self, key, val):
        #ind = self.order - key
        if key < 0:
            raise ValueError, "Does not support negative powers."
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
    >>> import pylab as plb
    >>> c = poly1d(1./sp.gamma(plb.r_[6+1:0:-1]))  #polynomial coeff exponential function
    >>> [p, q] = padefit(c)
    >>> p; q
    poly1d([ 0.00277778,  0.03333333,  0.2       ,  0.66666667,  1.        ])
    poly1d([ 0.03333333, -0.33333333,  1.        ])
    
    >>> x = plb.linspace(0,4);
    >>> h = plb.plot(x,c(x),x,p(x)/q(x),'g-', x,plb.exp(x),'r.')
    >>> plb.close()

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

def padefitlsq(fun, m, k, a= -1, b=1, trace=False, x=None, end_points=True):
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
    >>> import pylab as plb
    >>> [c1, c2] = padefitlsq(plb.exp,3,3,0,2)
    >>> c1; c2
    poly1d([ 0.01443847,  0.128842  ,  0.55284547,  0.99999962])
    poly1d([-0.0049658 ,  0.07610473, -0.44716929,  1.        ])
    
    >>> x = plb.linspace(0,4)
    >>> h = plb.plot(x, polyval(c1,x)/polyval(c2,x),'g')
    >>> h = plb.plot(x, plb.exp(x), 'r')

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
    npt = NFAC * ncof # % Number of points where function is evaluated, i.e. fineness of mesh

    if x is None:
        if end_points:
            # Use the location of the local extreme values of
            # the Chebychev polynomial of the first kind of degree NPT-1.
            x = map_to_interval(chebextr(npt - 1), a, b)
        else:
            # Use the roots of the Chebychev polynomial of the first kind of degree NPT.
            # Note this is useful if there are singularities close to the endpoints.
            x = map_to_interval(chebroot(npt, kind=1), a, b)


    if hasattr(fun, '__call__'):
        fs = fun(x)
    else:
        fs = fun
        n = len(fs)
        if n < npt:
            warnings.warn('Check the result! Number of function values should be at least: %d' % npt)

    if trace:
        import pylab as plb
        plb.plot(x, fs, '+')

    wt = ones((npt))
    ee = ones((npt))
    mad = 0

    u = zeros((npt, ncof))
    for ix in xrange(MAXIT):
        #% Set up design matrix for least squares fit.
        pow = wt
        bb = pow * (fs + abs(mad) * sign(ee))

        for jx in xrange(m + 1):
            u[:, jx] = pow
            pow = pow * x

        pow = -bb
        for jx in xrange(m + 1, ncof):
            pow = pow * x
            u[:, jx] = pow


        [u1, w, v] = linalg.svd(u, full_matrices=False)
        cof = where(w == 0, 0.0, dot(bb, u1) / w)
        cof = dot(cof, v)

        #% Tabulate the deviations and revise the weights
        ee = polyval(cof[m::-1], x) / polyval(cof[ncof:m:-1].tolist() + [1, ], x) - fs

        wt = abs(ee)
        devmax = max(wt)
        mad = wt.mean() #% mean absolute deviation

        if (devmax <= smallest_devmax): #% Save only the best coefficients found
            smallest_devmax = devmax
            c1 = cof[m::-1]
            c2 = cof[ncof:m:-1].tolist() + [1, ]

        if trace:
            print('Iteration=%d,  max error=%g' % (ix, devmax))
            plb.plot(x, fs, x, ee + fs)
    #c1=c1(:)
    #c2=c2(:)
    return poly1d(c1), poly1d(c2)





def main():

    [c1, c2] = padefitlsq(exp, 3, 3, 0, 2)

    x = linspace(0, 4)
    plb.plot(x, polyval(c1, x) / polyval(c2, x), 'g')
    plb.plot(x, exp(x), 'r')

    import scipy.special as sp
    
    p = [[1, 1, 1], [2, 2, 2]]
    pi = polyint(p, 1)
    pr = polyreloc(p, 2)
    pd = polyder(p)
    st = poly2str(p)
    c = poly1d(1. / sp.gamma(plb.r_[6 + 1:0:-1]))  #polynomial coeff exponential function
    [p, q] = padefit(c)
    x = linspace(0, 4);
    plb.plot(x, c(x), x, p(x) / q(x), 'g-', x, exp(x), 'r.')
    plb.close()
    x = arange(4)
    dx = dct(x)
    idx = idct(dx)
    
    a = 0;
    b = 2;
    ck = chebfit(exp, 6, a, b);
    t = chebval(0, ck, a, b)
    x = linspace(0, 2, 6);
    plb.plot(x, exp(x), 'r', x, chebval(x, ck, a, b), 'g.')
    #x1 = chebroot(9).'*(b-a)/2+(b+a)/2 ;
    #ck1 =chebfit([x1 exp(x1)],9,a,b);
    #plot(x,exp(x),'r'), hold on
    #plot(x,chebval(x,ck1,a,b),'g'), hold off


    t = poly2hstr([1, 1, 2])
    py = [1, 0]
    px = polyshift(py, 0, 5);
    t1 = polyval(px, [0, 2.5, 5])  #% This is the same as the line below
    t2 = polyval(py, [-1, 0, 1 ])

    px = [1, 0]
    py = polyishift(px, 0, 5);
    t1 = polyval(px, [0, 2.5, 5])  #% This is the same as the line below
    t2 = polyval(py, [-1, 0, 1 ])
    print(t1, t2)

if __name__ == '__main__':
    if False:
        main()
    else:
        import doctest
        doctest.testmod()
