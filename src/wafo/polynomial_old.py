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

"""
    Extended functions to operate on polynomials
"""
import warnings
import numpy as np
from numpy.lib.polynomial import *
__all__ = np.lib.polynomial.__all__
__all__ = __all__ + ['polyreloc', 'polyrescl', 'polytrim', 'poly2hstr', 'poly2str',
    'polyshift', 'polyishift', 'map_from_intervall', 'map_to_intervall',
    'cheb2poly', 'chebextr', 'chebroot', 'chebpoly', 'chebfit', 'chebval',
    'chebder', 'chebint', 'Cheb1d', 'dct', 'idct']

def polyreloc(p,x,y=0.0):
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
    r = np.atleast_1d(p).copy()
    n = r.shape[0]

    # Relocate polynomial using Horner's algorithm
    for ii in range(n,1,-1):
        for i in range(1,ii):
            r[i] = r[i] - x*r[i-1]
    r[-1] = r[-1] + y
    if r.ndim>1 and r.shape[-1]==1:
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
    r = np.atleast_1d(p).copy()
    n = r.shape[0]

    xscale =(float(x)**np.arange(1-n , 1))
    if r.ndim==1:
        q = y*r*xscale
    else:
        q = y*r*xscale[:,np.newaxis]
    if truepoly:
        q = poly1d(q)
    return q

def polytrim(p):
    """
    Trim polynomial by stripping off leading zeros.

    Parameters
    ----------
    p : array-like, poly1d
        vector of polynomial coefficients in decreasing order.

    Returns
    -------
    r : ndarray, poly1d
        vector/matrix/poly1d of trimmed polynomial coefficients.

    Example
    -------
    >>> p = [0,1,2]
    >>> polytrim(p)
    array([1, 2])
    """

    truepoly = isinstance(p, poly1d)
    if truepoly:
        return p
    else:
        r = np.atleast_1d(p).copy()
        # Remove leading zeros
        is_not_lead_zeros =np.logical_or.accumulate(r != 0,axis=0)
        if r.ndim==1:
            r = r[is_not_lead_zeros]
        else:
            is_not_lead_zeros = np.any(is_not_lead_zeros,axis=1)
            r = r[is_not_lead_zeros,:]
        return r

def poly2hstr(p, variable='x' ):
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
    order = len(coefs)-1 # Order of polynomial.
    s = ''    # Initialize output string.
    ix = 1;
    for expon in range(order,-1,-1):
        coef = coefs[order-expon]
        #% There is no point in adding a zero term (except if it's the only
        #% term, but we'll take care of that later).
        if coef == 0:
          ix = ix+1
        else:
        #% Append exponent if necessary.
            if ix>1:
                exponstr = '%.0f' % ix
                s = '%s**%s' % (s,exponstr);
                ix = 1
            #% Is it the first term?
            isfirst = s == ''

            # We need the coefficient only if it is different from 1 or -1 or
            # when it is the constant term.
            needcoef = (( abs(coef) != 1 ) | ( expon == 0 ) & isfirst) | 1-isfirst

            # We need the variable except in the constant term.
            needvar = ( expon != 0 )

            #% Add sign, but we don't need a leading plus-sign.
            if isfirst:
                if coef < 0:
                    s = '-'  #        % Unary minus.
            else:
                if coef < 0:
                    s = '%s - ' %  s  #    % Binary minus (subtraction).
                else:
                    s = '%s + ' %  s  #  % Binary plus (addition).


            #% Append the coefficient if it is different from one or when it is
            #% the constant term.
            if needcoef:
                coefstr = '%.20g' % abs(coef)
                s = '%s%s' % (s,coefstr)

            #% Append variable if necessary.
            if needvar:
                #% Append a multiplication sign if necessary.
                if needcoef:
                    if 1-isfirst:
                        s = '(%s)' % s
                    s = '%s*' % s
                s = '%s%s' % ( s, var )

    #% Now treat the special case where the polynomial is zero.
    if s=='':
       s = '0'
    return s

def poly2str(p,variable='x'):
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

    N = len(coeffs)-1

    for k in range(len(coeffs)):
        coefstr = '%.4g' % abs(coeffs[k])
        if coefstr[-4:] == '0000':
            coefstr = coefstr[:-5]
        power = (N-k)
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

def polyshift(py,a=-1,b=1):
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

    if (a==-1) & (b ==1):
        return py
    L = b-a
    return polyishift(py,-(2.+b+a)/L,(2.-b-a)/L)

def polyishift(px,a=-1,b=1):
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
    if (a==-1) & (b ==1):
        return px
    L = b-a
    xscale = 2./L
    xloc = -float(a+b)/L
    return polyreloc( polyrescl(px,xscale),xloc)

def map_from_interval(x,a,b) :
    """F(x), where F: [a,b] -> [-1,1]."""
    return (x - (b + a)/2.0)*(2.0/(b - a))

def map_to_interval(x,a,b) :
    """F(x), where F: [-1,1] -> [a,b]."""
    return (x*(b - a) + (b + a))/2.0

def poly2cheb(p,a=-1,b=1):
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
    return chebfit(f,n,a,b)

def cheb2poly(ck,a=-1,b=1):
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

    b_Nmi = np.zeros(1)
    b_Nmip1 = np.zeros(1)
    y = np.r_[2/(b-a), -(a+b)/(b-a)]
    y2 = 2.*y

    # Clenshaw recurence
    for ix in range(n-1,0,-1):
        tmp = b_Nmi
        b_Nmi = polymul(y2,b_Nmi) # polynomial multiplication
        nb = len(b_Nmip1)
        b_Nmip1[-1] = b_Nmip1[-1]-ck[ix]
        b_Nmi[-nb::] = b_Nmi[-nb::]-b_Nmip1
        b_Nmip1 = tmp

    p = polymul(y,b_Nmi) # polynomial multiplication
    nb = len(b_Nmip1)
    b_Nmip1[-1] = b_Nmip1[-1]-ck[0]
    p[-nb::] = p[-nb::]-b_Nmip1
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
    return -np.cos((np.pi*np.arange(n+1))/n);

def chebroot(n,kind=1):
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
    if kind not in (1,2):
        raise ValueError('kind must be 1 or 2')
    return -np.cos(np.pi*(np.arange(n)+0.5*kind)/(n+kind-1));


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
          p = np.ones(1)
       else:
          p = np.round( pow(2,n-2+kind) * np.poly( chebroot(n,kind=kind) ) )
          p[1::2] = 0;
       return p
    else: #   Evaluate polynomial in chebychev form
        ck = np.zeros(n+1)
        ck[n] = 1.
        return _chebval(np.atleast_1d(x),ck,kind=kind)

def chebfit(fun,n=10,a=-1,b=1,trace=False):
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

    William H. Press, Saul Teukolsky,
    William T. Wetterling and Brian P. Flannery (1997)
    "Numerical recipes in Fortran 77", Vol. 1, pp 184-194
    """

    if (n>50):
      warnings.warn('CHEBFIT should only be used for n<50')

    if hasattr(fun,'__call__'):
        x = map_to_interval(chebroot(n),a,b)
        f = fun(x);
        if trace:
            import pylab as plb
            plb.plot(x,f,'+')
    else:
        f = fun
        n = len(f)
        #raise ValueError('Function must be callable!')
    #                     N-1
    #       c(k) = (2/N) sum w(n) f(n)*cos(pi*k*(2n+1)/(2N)), 0 <= k < N.
    #                    n=0
    #
    # w(0) = 0.5, w(n)=1 for n>0
    ck = dct(f[::-1])/n
    ck[0] = ck[0]/2.
    return ck

def dct(x,n=None):
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
    fft = np.fft.fft
    x = np.atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    if x.shape[-1]<n:
        n_shape = x.shape[:-1] + (n-x.shape[-1],)
        xx = np.hstack((x,np.zeros(n_shape)))
    else:
        xx = x[...,:n]

    real_x = np.all(np.isreal(xx))
    if (real_x and (np.remainder(n,2) == 0)):
        xp = 2 * fft(np.hstack( (xx[...,::2], xx[...,::-2]) ))
    else:
        xp = fft(np.hstack((xx, xx[...,::-1])))
        xp = xp[...,:n]

    w = np.exp(-1j * np.arange(n) * np.pi/(2*n))

    y = xp*w

    if real_x:
        return y.real
    else:
        return y

def idct(x,n=None):
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

    ifft = np.fft.ifft
    x = np.atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    w = np.exp(1j * np.arange(n) * np.pi/(2*n))

    if x.shape[-1]<n:
        n_shape = x.shape[:-1] + (n-x.shape[-1],)
        xx = np.hstack((x,np.zeros(n_shape)))*w
    else:
        xx = x[...,:n]*w

    real_x = np.all(np.isreal(x))
    if (real_x and (np.remainder(n,2) == 0)):
        xx[...,0] = xx[...,0]*0.5
        yp = ifft(xx)
        y  = np.zeros(xx.shape,dtype=complex)
        y[...,::2] = yp[...,:n/2]
        y[...,::-2] = yp[...,n/2::]
    else:
        yp = ifft(np.hstack((xx, np.zeros_like(xx[...,0]), np.conj(xx[...,:0:-1]))))
        y = yp[...,:n]

    if real_x:
        return y.real
    else:
        return y

def _chebval(x,ck,kind=1):
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
    b_Nmi = np.zeros(x.shape) # b_(N-i)
    b_Nmip1 = b_Nmi.copy()    # b_(N-i+1)
    x2 = 2*x
    # Clenshaw reccurence
    for ix in range(n-1,0,-1):
        tmp = b_Nmi
        b_Nmi = x2*b_Nmi - b_Nmip1 + ck[ix]
        b_Nmip1 = tmp
    return kind*x*b_Nmi - b_Nmip1 + ck[0]


def chebval(x,ck,a=-1,b=1,kind=1,fill=None):
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
        polynomial coefficients in Chebyshev form.
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

    y = map_from_interval(np.atleast_1d(x),a,b)
    if fill is None:
        f = _chebval(y,ck,kind=kind)
    else:
        cond = (np.abs(y)<=1)
        f = np.where(cond,0,fill)
        if np.any(cond):
            yk = np.extract(cond,y)
            f[cond] = _chebval(yk,ck,kind=kind)
    return f


def chebder(ck,a=-1,b=1):
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
    >>> h=pb.plot(x,chebval(x,ck,0,2),'g',x,pb.exp(x),'r')
    >>> pb.close()

    See also
    --------
    chebint
    chebfit
    """

    n = len(ck)
    cder = np.zeros(n)
    # n and n-1 are special cases.
    # cder(n-1)=0;
    cder[-2] = 2*(n-1)*ck[-1]
    for j in range(n-3,-1,-1):
        cder[j] = cder[j+2]+2*j*ck[j+1]

    return cder*2./(b-a) # Normalize to the interval b-a.

def chebint(ck,a=-1,b=1):
    """
    Integrate Chebyshef polynomial

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
    """


    n    = len(ck)

    cint = np.zeros(n);
    con  = 0.25*(b-a);

    dif1 = np.diff(ck[::2])
    ix1= np.r_[1:n-1:2]
    cint[ix1] = -(con*dif1)/(ix1-1)
    if n>3:
        dif2 = np.diff(ck[1::2])
        ix2=np.r_[2:n-1:2]
        cint[ix2] = -(con*dif2)/(ix2-1)

    #% cint(n) is a special case
    cint[-1] = (con*ck[n-2])/(n-1)
    cint[0] = np.sum((-1)**np.r_[1:n]*cint[1::]) # Set constant of integration


    return cint

class Cheb1d(object):
    coeffs = None
    order = None
    a = None
    b = None
    def __init__(self,ck,a=-1,b=1):
        if isinstance(ck, poly1d):
            for key in ck.__dict__.keys():
                self.__dict__[key] = ck.__dict__[key]
            return
        cki = np.trim_zeros(np.atleast_1d(ck),'b')
        if len(cki.shape) > 1:
            raise ValueError, "Polynomial must be 1d only."
        self.__dict__['coeffs'] = cki
        self.__dict__['order'] = len(cki) - 1
        self.__dict__['a'] = a
        self.__dict__['b'] = b


    def __call__(self,x):
        return chebval(x,self.coeffs,self.a,self.b)

    def __array__(self, t=None):
        if t:
            return np.asarray(self.coeffs, t)
        else:
            return np.asarray(self.coeffs)

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "Cheb1d(%s)" % vals

    def __len__(self):
        return self.order

    def __str__(self):
        pass
    def __neg__(self):
        return Cheb1d(-self.coeffs,self.a,self.b)

    def __pos__(self):
        return self


    def __add__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __radd__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __sub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(self.coeffs, other.coeffs))

    def __rsub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(other.coeffs, self.coeffs))

    def __eq__(self, other):
        return np.alltrue(self.coeffs == other.coeffs)

    def __ne__(self, other):
        return np.any(self.coeffs != other.coeffs) or (self.a!=other.a) or (self.b !=other.b)

    def __setattr__(self, key, val):
        raise ValueError, "Attributes cannot be changed this way."

    def __getattr__(self, key):
        if key in ['c','coef','coefficients']:
            return self.coeffs
        elif key in ['o']:
            return self.order
        elif key in ['a']:
            return self.a
        elif key in ['b']:
            return self.b
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
        ind = self.order - key
        if key < 0:
            raise ValueError, "Does not support negative powers."
        if key > self.order:
            zr = NX.zeros(key-self.order, self.coeffs.dtype)
            self.__dict__['coeffs'] = NX.concatenate((self.coeffs,zr))
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
        return Cheb1d(chebint(self.coeffs, self.a,self.b))

    def deriv(self, m=1):
        """
        Return a derivative of this polynomial.

        Refer to `chebder` for full documentation.

        See Also
        --------
        chebder : equivalent function

        """
        return Cheb1d(chebder(self.coeffs,self.a,self.b))
def main():
    if  False: #True: #
        x = np.arange(4)
        dx = dct(x)
        idx = idct(dx)
        import pylab as plb
        a = 0;
        b = 2;
        ck = chebfit(np.exp,6,a,b);
        t = chebval(0,ck,a,b)
        x=np.linspace(0,2,6);
        plb.plot(x,np.exp(x),'r', x,chebval(x,ck,a,b),'g.')
        #x1 = chebroot(9).'*(b-a)/2+(b+a)/2 ;
        #ck1 =chebfit([x1 exp(x1)],9,a,b);
        #plot(x,exp(x),'r'), hold on
        #plot(x,chebval(x,ck1,a,b),'g'), hold off


        t = poly2hstr([1,1,2])
        py = [1, 0]
        px = polyshift(py,0,5);
        t1=polyval(px,[0, 2.5, 5])  #% This is the same as the line below
        t2=polyval(py,[-1, 0, 1 ])

        px = [1, 0]
        py = polyishift(px,0,5);
        t1 = polyval(px,[0, 2.5, 5])  #% This is the same as the line below
        t2 = polyval(py,[-1, 0, 1 ])
        print(t1,t2)
    else:
        import doctest
        doctest.testmod()
if __name__== '__main__':
    main()
