#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      pab
#
# Created:     30.12.2008
# Copyright:   (c) pab 2008
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg #@UnusedImport
from numpy.ma.core import ones, zeros, prod, sin
from numpy import diff, pi, inf #@UnresolvedImport
from numpy.lib.shape_base import vstack
from numpy.lib.function_base import linspace
import polynomial as pl

__all__ = ['PPform', 'SmoothSpline', 'stineman_interp']

class PPform(object):
    """The ppform of the piecewise polynomials is given in terms of coefficients
    and breaks.  The polynomial in the ith interval is
    x_{i} <= x < x_{i+1}

    S_i = sum(coefs[m,i]*(x-breaks[i])^(k-m), m=0..k)
    where k is the degree of the polynomial.

    Example
    -------
    >>> coef = np.array([[1,1]]) # unit step function
    >>> coef = np.array([[1,1],[0,1]]) # linear from 0 to 2
    >>> coef = np.array([[1,1],[1,1],[0,2]]) # linear from 0 to 2
    >>> breaks = [0,1,2]
    >>> self = PPform(coef, breaks)
    >>> x = linspace(-1,3)
    >>> plot(x,self(x))
    """
    def __init__(self, coeffs, breaks, fill=0.0, sort=False, a=None, b=None):
        if sort:
            self.breaks = np.sort(breaks)
        else:
            self.breaks = np.asarray(breaks)
        if a is None:
            a = self.breaks[0]
        if b is None:
            b = self.breaks[-1]
        self.coeffs = np.asarray(coeffs)
        self.order = self.coeffs.shape[0]
        self.fill = fill
        self.a = a
        self.b = b

    def __call__(self, xnew):
        saveshape = np.shape(xnew)
        xnew = np.ravel(xnew)
        res = np.empty_like(xnew)
        mask = (self.a <= xnew) & (xnew <= self.b)
        res[~mask] = self.fill
        xx = xnew.compress(mask)
        indxs = np.searchsorted(self.breaks[:-1], xx) - 1
        indxs = indxs.clip(0, len(self.breaks))
        pp = self.coeffs
        dx = xx - self.breaks.take(indxs)
        if True:
            v = pp[0, indxs]
            for i in xrange(1, self.order):
                v = dx * v + pp[i, indxs]
            values = v
        else:
            V = np.vander(dx, N=self.order)
            # values = np.diag(dot(V,pp[:,indxs]))
            dot = np.dot
            values = np.array([dot(V[k, :], pp[:, indxs[k]]) for k in xrange(len(xx))])
        
        res[mask] = values
        res.shape = saveshape
        return res
    
    def linear_extrapolate(self, output=True):
        '''
        Return a 1D PPform which extrapolate linearly outside its basic interval
        '''
    
        max_order = 2
    
        if self.order <= max_order:
            if output:
                return self
            else: 
                return
        breaks = self.breaks.copy()
        coefs = self.coeffs.copy()
        #pieces = len(breaks) - 1
        
        # Add new breaks beyond each end
        breaks2add = breaks[[0, -1]] + np.array([-1, 1])
        newbreaks = np.hstack([breaks2add[0], breaks, breaks2add[1]])
    
        dx = newbreaks[[0, -2]] - breaks[[0, -2]]
    
        dx = dx.ravel()
       
        # Get coefficients for the new last polynomial piece (a_n)
        # by just relocate the previous last polynomial and
        # then set all terms of order > maxOrder to zero
        
        a_nn = coefs[:, -1]
        dxN = dx[-1]
         
        a_n = pl.polyreloc(a_nn, -dxN) # Relocate last polynomial
        #set to zero all terms of order > maxOrder 
        a_n[0:self.order - max_order] = 0
    
        #Get the coefficients for the new first piece (a_1)
        # by first setting all terms of order > maxOrder to zero and then
        # relocate the polynomial.

    
        #Set to zero all terms of order > maxOrder, i.e., not using them
        a_11 = coefs[self.order - max_order::, 0]
        dx1 = dx[0]
    
        a_1 = pl.polyreloc(a_11, -dx1) # Relocate first polynomial 
        a_1 = np.hstack([zeros(self.order - max_order), a_1])
      
        newcoefs = np.hstack([ a_1.reshape(-1, 1), coefs, a_n.reshape(-1, 1)])
        if output:
            return PPform(newcoefs, newbreaks, a= -inf, b=inf)
        else:
            self.coeffs = newcoefs
            self.breaks = newbreaks
            self.a = -inf
            self.b = inf
    
    def derivative(self):
        """
        Return first derivative of the piecewise polynomial
        """
        
        cof = pl.polyder(self.coeffs)
        brks = self.breaks.copy()
        return PPform(cof, brks, fill=self.fill)


    def integrate(self):
        """
        Return the indefinite integral of the piecewise polynomial
        """
        cof = pl.polyint(self.coeffs)        

        pieces = len(self.breaks) - 1
        if 1 < pieces :
            # evaluate each integrated polynomial at the right endpoint of its interval
            xs = diff(self.breaks[:-1, ...], axis=0)
            index = np.arange(pieces - 1)
            
            vv = xs * cof[0, index]
            k = self.order
            for i in xrange(1, k):
                vv = xs * (vv + cof[i, index])
          
            cof[-1] = np.hstack((0, vv)).cumsum()

        return PPform(cof, self.breaks, fill=self.fill)



##    def fromspline(cls, xk, cvals, order, fill=0.0):
##        N = len(xk)-1
##        sivals = np.empty((order+1,N), dtype=float)
##        for m in xrange(order,-1,-1):
##            fact = spec.gamma(m+1)
##            res = _fitpack._bspleval(xk[:-1], xk, cvals, order, m)
##            res /= fact
##            sivals[order-m,:] = res
##        return cls(sivals, xk, fill=fill)

class SmoothSpline(PPform):
    """
    Cubic Smoothing Spline.

    Parameters
    ----------
    x : array-like
        x-coordinates of data. (vector)
    y : array-like
        y-coordinates of data. (vector or matrix)
    p : real scalar
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    lin_extrap : bool
        if False regular smoothing spline 
        if True a smoothing spline with a constraint on the ends to
        ensure linear extrapolation outside the range of the data (default)
    var : array-like
        variance of each y(i) (default  1)

    Returns
    -------
    pp : ppform
        If xx is not given, return self-form of the spline.

    Given the approximate values

        y(i) = g(x(i))+e(i)

    of some smooth function, g, where e(i) is the error. SMOOTH tries to
    recover g from y by constructing a function, f, which  minimizes

      p * sum (Y(i) - f(X(i)))^2/d2(i)  +  (1-p) * int (f'')^2


    Example
    -------
    >>> import numpy as np
    >>> x = np.linspace(0,1)
    >>> y = exp(x)+1e-1*np.random.randn(x.shape)
    >>> pp9 = SmoothSpline(x, y, p=.9)
    >>> pp99 = SmoothSpline(x, y, p=.99, var=0.01)
    >>> plot(x,y, x,pp99(x),'g', x,pp9(x),'k', x,exp(x),'r')

    See also
    --------
    lc2tr, dat2tr


    References
    ----------
    Carl de Boor (1978)
    'Practical Guide to Splines'
    Springer Verlag
    Uses EqXIV.6--9, self 239
    """
    def __init__(self, xx, yy, p=None, lin_extrap=True, var=1):
        coefs, brks = self._compute_coefs(xx, yy, p, var)
        super(SmoothSpline, self).__init__(coefs, brks)
        if lin_extrap:
            self.linear_extrapolate(output=False)
        
    def _compute_coefs(self, xx, yy, p=None, var=1):
        x, y = np.atleast_1d(xx, yy)
        x = x.ravel()
        dx = np.diff(x)
        must_sort = (dx < 0).any()
        if must_sort:
            ind = x.argsort()
            x = x[ind]
            y = y[..., ind]
            dx = np.diff(x)
    
        n = len(x)
    
        #ndy = y.ndim
        szy = y.shape
    
        nd = prod(szy[:-1])
        ny = szy[-1]
       
        if n < 2:
            raise ValueError('There must be >=2 data points.')
        elif (dx <= 0).any():
            raise ValueError('Two consecutive values in x can not be equal.')
        elif n != ny:
            raise ValueError('x and y must have the same length.')
    
        dydx = np.diff(y) / dx
    
        if (n == 2) : #% straight line
            coefs = np.vstack([dydx.ravel(), y[0, :]])
        else:
           
            dx1 = 1. / dx
            D = sp.spdiags(var * ones(n), 0, n, n)  # The variance
    
            u, p = self._compute_u(p, D, dydx, dx, dx1, n)
            dx1.shape = (n - 1, -1)
            dx.shape = (n - 1, -1)
            zrs = zeros(nd)
            if p < 1:
                ai = (y - (6 * (1 - p) * D * diff(vstack([zrs,
                                               diff(vstack([zrs, u, zrs]), axis=0) * dx1,
                                               zrs]), axis=0)).T).T #faster than yi-6*(1-p)*Q*u
            else:
                ai = y.reshape(n, -1)
    
            # The piecewise polynominals are written as
            # fi=ai+bi*(x-xi)+ci*(x-xi)^2+di*(x-xi)^3
            # where the derivatives in the knots according to Carl de Boor are:
            #    ddfi  = 6*p*[0;u] = 2*ci;
            #    dddfi = 2*diff([ci;0])./dx = 6*di;
            #    dfi   = diff(ai)./dx-(ci+di.*dx).*dx = bi;
    
            ci = np.vstack([zrs, 3 * p * u])  
            di = (diff(vstack([ci, zrs]), axis=0) * dx1 / 3);
            bi = (diff(ai, axis=0) * dx1 - (ci + di * dx) * dx)
            ai = ai[:n - 1, ...] 
            if nd > 1:
                di = di.T
                ci = ci.T
                ai = ai.T
                #end
            if not any(di):
                if not any(ci):
                    coefs = vstack([bi.ravel(), ai.ravel()])
                else:
                    coefs = vstack([ci.ravel(), bi.ravel(), ai.ravel()]) 
                    #end
            else:
                coefs = vstack([di.ravel(), ci.ravel(), bi.ravel(), ai.ravel()]) 
                
        return coefs, x
       
    def _compute_u(self, p, D, dydx, dx, dx1, n):
        if p is None or p != 0:
            data = [dx[1:n - 1], 2 * (dx[:n - 2] + dx[1:n - 1]), dx[:n - 2]]
            R = sp.spdiags(data, [-1, 0, 1], n - 2, n - 2)
        
        if p is None or p < 1:
            Q = sp.spdiags([dx1[:n - 2], -(dx1[:n - 2] + dx1[1:n - 1]), dx1[1:n - 1]], [0, -1, -2], n, n - 2)
            QDQ = (Q.T * D * Q) 
            if p is None or p < 0:
                # Estimate p
                p = 1. / (1. + QDQ.diagonal().sum() / (100. * R.diagonal().sum()** 2));
            
            if p == 0:
                QQ = 6 * QDQ
            else:
                QQ = (6 * (1 - p)) * (QDQ) + p * R
        else:
            QQ = R 
            
        # Make sure it uses symmetric matrix solver
        ddydx = diff(dydx, axis=0)
        sp.linalg.use_solver(useUmfpack=True)
        u = 2 * sp.linalg.spsolve((QQ + QQ.T), ddydx)
        #faster than u=QQ\(Q' * yi);
        return u.reshape(n - 2, -1), p
 

def slopes(x, y):
    """
    :func:`slopes` calculates the slope *y*'(*x*)

    The slope is estimated using the slope obtained from that of a
    parabola through any three consecutive points.

    This method should be superior to that described in the appendix
    of A CONSISTENTLY WELL BEHAVED METHOD OF INTERPOLATION by Russel
    W. Stineman (Creative Computing July 1980) in at least one aspect:

      Circles for interpolation demand a known aspect ratio between
      *x*- and *y*-values.  For many functions, however, the abscissa
      are given in different dimensions, so an aspect ratio is
      completely arbitrary.

    The parabola method gives very similar results to the circle
    method for most regular cases but behaves much better in special
    cases.

    Norbert Nemec, Institute of Theoretical Physics, University or
    Regensburg, April 2006 Norbert.Nemec at physik.uni-regensburg.de

    (inspired by a original implementation by Halldor Bjornsson,
    Icelandic Meteorological Office, March 2006 halldor at vedur.is)
    """
    # Cast key variables as float.
    x = np.asarray(x, np.float_)
    y = np.asarray(y, np.float_)

    yp = np.zeros(y.shape, np.float_)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dydx = dy / dx
    yp[1:-1] = (dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1]) / (dx[1:] + dx[:-1])
    yp[0] = 2.0 * dy[0] / dx[0] - yp[1]
    yp[-1] = 2.0 * dy[-1] / dx[-1] - yp[-2]
    return yp


def stineman_interp(xi, x, y, yp=None):
    """
    Given data vectors *x* and *y*, the slope vector *yp* and a new
    abscissa vector *xi*, the function :func:`stineman_interp` uses
    Stineman interpolation to calculate a vector *yi* corresponding to
    *xi*.

    Here's an example that generates a coarse sine curve, then
    interpolates over a finer abscissa::

      x = linspace(0,2*pi,20);  y = sin(x); yp = cos(x)
      xi = linspace(0,2*pi,40);
      yi = stineman_interp(xi,x,y,yp);
      plot(x,y,'o',xi,yi)

    The interpolation method is described in the article A
    CONSISTENTLY WELL BEHAVED METHOD OF INTERPOLATION by Russell
    W. Stineman. The article appeared in the July 1980 issue of
    Creative Computing with a note from the editor stating that while
    they were:

      not an academic journal but once in a while something serious
      and original comes in adding that this was
      "apparently a real solution" to a well known problem.

    For *yp* = *None*, the routine automatically determines the slopes
    using the :func:`slopes` routine.

    *x* is assumed to be sorted in increasing order.

    For values ``xi[j] < x[0]`` or ``xi[j] > x[-1]``, the routine
    tries an extrapolation.  The relevance of the data obtained from
    this, of course, is questionable...

    Original implementation by Halldor Bjornsson, Icelandic
    Meteorolocial Office, March 2006 halldor at vedur.is

    Completely reworked and optimized for Python by Norbert Nemec,
    Institute of Theoretical Physics, University or Regensburg, April
    2006 Norbert.Nemec at physik.uni-regensburg.de
    """

    # Cast key variables as float.
    x = np.asarray(x, np.float_)
    y = np.asarray(y, np.float_)
    assert x.shape == y.shape
    #N = len(y)

    if yp is None:
        yp = slopes(x, y)
    else:
        yp = np.asarray(yp, np.float_)

    xi = np.asarray(xi, np.float_)
    #yi = np.zeros(xi.shape, np.float_)

    # calculate linear slopes
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    s = dy / dx  #note length of s is N-1 so last element is #N-2

    # find the segment each xi is in
    # this line actually is the key to the efficiency of this implementation
    idx = np.searchsorted(x[1:-1], xi)

    # now we have generally: x[idx[j]] <= xi[j] <= x[idx[j]+1]
    # except at the boundaries, where it may be that xi[j] < x[0] or xi[j] > x[-1]

    # the y-values that would come out from a linear interpolation:
    sidx = s.take(idx)
    xidx = x.take(idx)
    yidx = y.take(idx)
    xidxp1 = x.take(idx + 1)
    yo = yidx + sidx * (xi - xidx)

    # the difference that comes when using the slopes given in yp
    dy1 = (yp.take(idx) - sidx) * (xi - xidx)       # using the yp slope of the left point
    dy2 = (yp.take(idx + 1) - sidx) * (xi - xidxp1) # using the yp slope of the right point

    dy1dy2 = dy1 * dy2
    # The following is optimized for Python. The solution actually
    # does more calculations than necessary but exploiting the power
    # of numpy, this is far more efficient than coding a loop by hand
    # in Python
    dy1mdy2 = np.where(dy1dy2,dy1-dy2,np.inf)
    dy1pdy2 = np.where(dy1dy2,dy1+dy2,np.inf)
    yi = yo + dy1dy2 * np.choose(np.array(np.sign(dy1dy2), np.int32) + 1,
                                 ((2 * xi - xidx - xidxp1) / ((dy1mdy2) * (xidxp1 - xidx)),
                                  0.0,
                                  1 / (dy1pdy2)))
    return yi

def test_smoothing_spline():
    x = linspace(0, 2 * pi + pi / 4, 20) 
    y = sin(x) #+ np.random.randn(x.size)
    pp = SmoothSpline(x, y, p=1)
    x1 = linspace(-1, 2 * pi + pi / 4 + 1, 20) 
    y1 = pp(x1)
    pp1 = pp.derivative()
    pp0 = pp1.integrate()
    dy1 = pp1(x1)
    y01 = pp0(x1)
    #dy = y-y1
    import pylab as plb
   
    plb.plot(x, y, x1, y1, '.', x1, dy1, 'ro', x1, y01, 'r-')
    plb.show()
    pass
    #tck = interpolate.splrep(x, y, s=len(x)) 

def main():
    from scipy import interpolate
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.interactive(True)
    
    coef = np.array([[1, 1], [0, 1]]) # linear from 0 to 2
    #coef = np.array([[1,1],[1,1],[0,2]]) # linear from 0 to 2
    breaks = [0, 1, 2]
    pp = PPform(coef, breaks, a= -100, b=100)
    x = linspace(-1, 3, 20)
    y = pp(x)
    
    x = linspace(0, 2 * pi + pi / 4, 20) 
    y = x + np.random.randn(x.size)
    tck = interpolate.splrep(x, y, s=len(x)) 
    xnew = linspace(0, 2 * pi, 100) 
    ynew = interpolate.splev(xnew, tck, der=0)
    tck0 = interpolate.splmake(xnew, ynew, order=3, kind='smoothest', conds=None)
    pp = interpolate.ppform.fromspline(*tck0)
     
    plt.plot(x, y, "x", xnew, ynew, xnew, sin(xnew), x, y, "b") 
    plt.legend(['Linear', 'Cubic Spline', 'True'])  
    plt.title('Cubic-spline interpolation') 
     
    
    t = np.arange(0, 1.1, .1)
    x = np.sin(2 * np.pi * t)
    y = np.cos(2 * np.pi * t)
    tck1, u = interpolate.splprep([t, y], s=0)
    tck2 = interpolate.splrep(t, y, s=len(t), task=0)
    #interpolate.spl
    tck = interpolate.splmake(t, y, order=3, kind='smoothest', conds=None)
    self = interpolate.ppform.fromspline(*tck2)
    plt.plot(t, self(t))
    pass

def test_pp():
    import polynomial as pl
    coef = np.array([[1, 1], [0, 0]]) # linear from 0 to 2

    coef = np.array([[1, 1], [1, 1], [0, 2]]) # quadratic from 0 to 1 and 1 to 2.
    dc = pl.polyder(coef, 1)
    c2 = pl.polyint(dc, 1)
    breaks = [0, 1, 2]
    pp = PPform(coef, breaks)
    pp(0.5)
    pp(1)
    pp(1.5)
    dpp = pp.derivative()
    import pylab as plb
    x = plb.linspace(-1, 3)
    plb.plot(x, pp(x), x, dpp(x), '.')
    plb.show()
    
if __name__ == '__main__':
    #main()
    test_smoothing_spline()
