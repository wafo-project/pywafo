#!/usr/bin/env python
from __future__ import absolute_import, division
import numpy as np
import scipy.signal
# import scipy.sparse.linalg  # @UnusedImport
import scipy.sparse as sparse
from numpy import ones, zeros, prod, sin, diff, pi, inf, vstack, linspace
from scipy.interpolate import PiecewisePolynomial, interp1d

from . import polynomial as pl


__all__ = [
    'PPform', 'savitzky_golay', 'savitzky_golay_piecewise', 'sgolay2d',
    'SmoothSpline', 'pchip_slopes', 'slopes', 'stineman_interp', 'Pchip',
    'StinemanInterp', 'CubicHermiteSpline']


def savitzky_golay(y, window_size, order, deriv=0):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        order of the derivative to compute (default = 0 means only smoothing)

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The test_doctstrings idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = savitzky_golay(y, window_size=31, order=4)
    >>> import matplotlib.pyplot as plt
    >>> h=plt.plot(t, y, label='Noisy signal')
    >>> h=plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> h=plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> h=plt.legend()
    >>> plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range]
               for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m, y, mode='valid')


def savitzky_golay_piecewise(xvals, data, kernel=11, order=4):
    '''
    One of the most popular applications of S-G filter, apart from smoothing
    UV-VIS and IR spectra, is smoothing of curves obtained in electroanalytical
    experiments. In cyclic voltammetry, voltage (being the abcissa) changes
    like a triangle wave. And in the signal there are cusps at the turning
    points (at switching potentials) which should never be smoothed.
    In this case, Savitzky-Golay smoothing should be
    done piecewise, ie. separately on pieces monotonic in x

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> n = 1e3
    >>> x = np.linspace(0, 25, n)
    >>> y = np.round(sin(x))
    >>> sig2 = linspace(0,0.5,50)

    # As an example, this figure shows the effect of an additive noise with a
    # variance of 0.2 (original signal (black), noisy signal (red) and filtered
    # signal (blue dots)).

    >>> yn = y + np.sqrt(0.2)*np.random.randn(*x.shape)
    >>> yr = savitzky_golay_piecewise(x, yn, kernel=11, order=4)
    >>> h=plt.plot(x, yn, 'r', x, y, 'k', x, yr, 'b.')
    '''
    turnpoint = 0
    last = len(xvals)
    if xvals[1] > xvals[0]:  # x is increasing?
        for i in range(1, last):  # yes
            if xvals[i] < xvals[i - 1]:  # search where x starts to fall
                turnpoint = i
                break
    else:  # no, x is decreasing
        for i in range(1, last):  # search where it starts to rise
            if xvals[i] > xvals[i - 1]:
                turnpoint = i
                break
    if turnpoint == 0:  # no change in direction of x
        return savitzky_golay(data, kernel, order)
    else:
        # smooth the first piece
        firstpart = savitzky_golay(data[0:turnpoint], kernel, order)
        # recursively smooth the rest
        rest = savitzky_golay_piecewise(
            xvals[turnpoint:], data[turnpoint:], kernel, order)
        return np.concatenate((firstpart, rest))


def sgolay2d(z, window_size, order, derivative=None):
    """
    Savitsky - Golay filters can also be used to smooth two dimensional data
    affected by noise. The algorithm is exactly the same as for the one
    dimensional case, only the math is a bit more tricky. The basic algorithm
    is as follow: for each point of the two dimensional matrix extract a sub
    - matrix, centered at that point and with a size equal to an odd number
    "window_size". for this sub - matrix compute a least - square fit of a
    polynomial surface, defined as
    p(x, y) = a0 + a1 * x + a2 * y + a3 * x2 + a4 * y2 + a5 * x * y + ... .

    Note that x and y are equal to zero at the central point.
    replace the initial central point with the value computed with the fit.
    Note that because the fit coefficients are linear with respect to the data
    spacing, they can pre - computed for efficiency. Moreover, it is important
    to appropriately pad the borders of the data, with a mirror image of the
    data itself, so that the evaluation of the fit at the borders of the data
    can happen smoothly.
    Here is the code for two dimensional filtering.

    Example
    -------
    # create some sample twoD data
    >>> x = np.linspace(-3,3,100)
    >>> y = np.linspace(-3,3,100)
    >>> X, Y = np.meshgrid(x,y)
    >>> Z = np.exp( -(X**2+Y**2))

    # add noise
    >>> Zn = Z + np.random.normal( 0, 0.2, Z.shape )

    # filter it
    >>> Zf = sgolay2d( Zn, window_size=29, order=4)

    # do some plotting
    >>> import matplotlib.pyplot as plt
    >>> h=plt.matshow(Z)
    >>> h=plt.matshow(Zn)
    >>> h=plt.matshow(Zf)
    """
    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size ** 2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size ** 2,)

    # build matrix of system of equation
    A = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - \
        np.abs(np.flipud(z[1:half_size + 1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + \
        np.abs(np.flipud(z[-half_size - 1:-1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - \
        np.abs(np.fliplr(z[:, 1:half_size + 1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + \
        np.abs(np.fliplr(z[:, -half_size - 1:-1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - \
        np.abs(
            np.flipud(np.fliplr(z[1:half_size + 1, 1:half_size + 1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + \
        np.abs(np.flipud(np.fliplr(z[-half_size - 1:-1, -half_size - 1:-1])) -
               band)

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - \
        np.abs(
            np.flipud(Z[half_size + 1:2 * half_size + 1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - \
        np.abs(
            np.fliplr(Z[-half_size:, half_size + 1:2 * half_size + 1]) - band)

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return (scipy.signal.fftconvolve(Z, -r, mode='valid'),
                scipy.signal.fftconvolve(Z, -c, mode='valid'))


class PPform(object):

    """The ppform of the piecewise polynomials
                    is given in terms of coefficients and breaks.
    The polynomial in the ith interval is
        x_{i} <= x < x_{i+1}

    S_i = sum(coefs[m,i]*(x-breaks[i])^(k-m), m=0..k)
    where k is the degree of the polynomial.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> coef = np.array([[1,1]]) # unit step function
    >>> coef = np.array([[1,1],[0,1]]) # linear from 0 to 2
    >>> coef = np.array([[1,1],[1,1],[0,2]]) # linear from 0 to 2
    >>> breaks = [0,1,2]
    >>> self = PPform(coef, breaks)
    >>> x = linspace(-1,3)
    >>> h=plt.plot(x,self(x))
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

        v = pp[0, indxs]
        for i in range(1, self.order):
            v = dx * v + pp[i, indxs]
        values = v

        res[mask] = values
        res.shape = saveshape
        return res

    def linear_extrapolate(self, output=True):
        '''
        Return 1D PPform which extrapolate linearly outside its basic interval
        '''

        max_order = 2

        if self.order <= max_order:
            if output:
                return self
            else:
                return
        breaks = self.breaks.copy()
        coefs = self.coeffs.copy()
        # pieces = len(breaks) - 1

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

        a_n = pl.polyreloc(a_nn, -dxN)  # Relocate last polynomial
        # set to zero all terms of order > maxOrder
        a_n[0:self.order - max_order] = 0

        # Get the coefficients for the new first piece (a_1)
        # by first setting all terms of order > maxOrder to zero and then
        # relocate the polynomial.

        # Set to zero all terms of order > maxOrder, i.e., not using them
        a_11 = coefs[self.order - max_order::, 0]
        dx1 = dx[0]

        a_1 = pl.polyreloc(a_11, -dx1)  # Relocate first polynomial
        a_1 = np.hstack([zeros(self.order - max_order), a_1])

        newcoefs = np.hstack([a_1.reshape(-1, 1), coefs, a_n.reshape(-1, 1)])
        if output:
            return PPform(newcoefs, newbreaks, a=-inf, b=inf)
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
        if 1 < pieces:
            # evaluate each integrated polynomial at the right endpoint of its
            # interval
            xs = diff(self.breaks[:-1, ...], axis=0)
            index = np.arange(pieces - 1)

            vv = xs * cof[0, index]
            k = self.order
            for i in range(1, k):
                vv = xs * (vv + cof[i, index])

            cof[-1] = np.hstack((0, vv)).cumsum()

        return PPform(cof, self.breaks, fill=self.fill)

#     def fromspline(self, xk, cvals, order, fill=0.0):
#         N = len(xk) - 1
#         sivals = np.empty((order + 1, N), dtype=float)
#         for m in range(order, -1, -1):
#             fact = spec.gamma(m + 1)
#             res = _fitpack._bspleval(xk[:-1], xk, cvals, order, m)
#             res /= fact
#             sivals[order - m, :] = res
#         return self(sivals, xk, fill=fill)


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
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0,1)
    >>> y = np.exp(x)+1e-1*np.random.randn(x.size)
    >>> pp9 = SmoothSpline(x, y, p=.9)
    >>> pp99 = SmoothSpline(x, y, p=.99, var=0.01)
    >>> h=plt.plot(x,y, x,pp99(x),'g', x,pp9(x),'k', x,np.exp(x),'r')

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

        # ndy = y.ndim
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

        if (n == 2):  # % straight line
            coefs = np.vstack([dydx.ravel(), y[0, :]])
        else:

            dx1 = 1. / dx
            D = sparse.spdiags(var * ones(n), 0, n, n)  # The variance

            u, p = self._compute_u(p, D, dydx, dx, dx1, n)
            dx1.shape = (n - 1, -1)
            dx.shape = (n - 1, -1)
            zrs = zeros(nd)
            if p < 1:
                # faster than yi-6*(1-p)*Q*u
                Qu = D * diff(vstack([zrs, diff(vstack([zrs, u, zrs]),
                                                axis=0) * dx1, zrs]), axis=0)
                ai = (y - (6 * (1 - p) * Qu).T).T
            else:
                ai = y.reshape(n, -1)

            # The piecewise polynominals are written as
            # fi=ai+bi*(x-xi)+ci*(x-xi)^2+di*(x-xi)^3
            # where the derivatives in the knots according to Carl de Boor are:
            #    ddfi  = 6*p*[0;u] = 2*ci;
            #    dddfi = 2*diff([ci;0])./dx = 6*di;
            #    dfi   = diff(ai)./dx-(ci+di.*dx).*dx = bi;

            ci = np.vstack([zrs, 3 * p * u])
            di = (diff(vstack([ci, zrs]), axis=0) * dx1 / 3)
            bi = (diff(ai, axis=0) * dx1 - (ci + di * dx) * dx)
            ai = ai[:n - 1, ...]
            if nd > 1:
                di = di.T
                ci = ci.T
                ai = ai.T
            if not any(di):
                if not any(ci):
                    coefs = vstack([bi.ravel(), ai.ravel()])
                else:
                    coefs = vstack([ci.ravel(), bi.ravel(), ai.ravel()])
            else:
                coefs = vstack(
                    [di.ravel(), ci.ravel(), bi.ravel(), ai.ravel()])

        return coefs, x

    def _compute_u(self, p, D, dydx, dx, dx1, n):
        if p is None or p != 0:
            data = [dx[1:n - 1], 2 * (dx[:n - 2] + dx[1:n - 1]), dx[:n - 2]]
            R = sparse.spdiags(data, [-1, 0, 1], n - 2, n - 2)

        if p is None or p < 1:
            Q = sparse.spdiags(
                [dx1[:n - 2], -(dx1[:n - 2] + dx1[1:n - 1]), dx1[1:n - 1]],
                [0, -1, -2], n, n - 2)
            QDQ = (Q.T * D * Q)
            if p is None or p < 0:
                # Estimate p
                p = 1. / \
                    (1. + QDQ.diagonal().sum() /
                     (100. * R.diagonal().sum() ** 2))

            if p == 0:
                QQ = 6 * QDQ
            else:
                QQ = (6 * (1 - p)) * (QDQ) + p * R
        else:
            QQ = R

        # Make sure it uses symmetric matrix solver
        ddydx = diff(dydx, axis=0)
        # sp.linalg.use_solver(useUmfpack=True)
        u = 2 * sparse.linalg.spsolve((QQ + QQ.T), ddydx)  # @UndefinedVariable
        return u.reshape(n - 2, -1), p


def _edge_case(m0, d1):
    return np.where((d1 == 0) | (m0 == 0), 0.0, 1.0 / (1.0 / m0 + 1.0 / d1))


def pchip_slopes(x, y):
    # Determine the derivatives at the points y_k, d_k, by using
    #  PCHIP algorithm is:
    # We choose the derivatives at the point x_k by
    # Let m_k be the slope of the kth segment (between k and k+1)
    # If m_k=0 or m_{k-1}=0 or sgn(m_k) != sgn(m_{k-1}) then d_k == 0
    # else use weighted harmonic mean:
    #   w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
    #   1/d_k = 1/(w_1 + w_2)*(w_1 / m_k + w_2 / m_{k-1})
    #   where h_k is the spacing between x_k and x_{k+1}

    hk = x[1:] - x[:-1]
    mk = (y[1:] - y[:-1]) / hk
    smk = np.sign(mk)
    condition = ((smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0))

    w1 = 2 * hk[1:] + hk[:-1]
    w2 = hk[1:] + 2 * hk[:-1]
    whmean = 1.0 / (w1 + w2) * (w1 / mk[1:] + w2 / mk[:-1])

    dk = np.zeros_like(y)
    dk[1:-1][condition] = 0.0
    dk[1:-1][~condition] = 1.0 / whmean[~condition]

    # For end-points choose d_0 so that 1/d_0 = 1/m_0 + 1/d_1 unless
    #  one of d_1 or m_0 is 0, then choose d_0 = 0

    dk[0] = _edge_case(mk[0], dk[1])
    dk[-1] = _edge_case(mk[-1], dk[-2])
    return dk


def slopes(x, y, method='parabola', tension=0, monotone=False):
    '''
    Return estimated slopes y'(x)

    Parameters
    ----------
    x, y : array-like
        array containing the x- and y-data, respectively.
        x must be sorted low to high... (no repeats) while
        y can have repeated values.
    method : string
        defining method of estimation for yp. Valid options are:
        'Catmull-Rom'  yp = (y[k+1]-y[k-1])/(x[k+1]-x[k-1])
        'Cardinal'     yp = (1-tension) * (y[k+1]-y[k-1])/(x[k+1]-x[k-1])
        'parabola'
        'secant' average secants
           yp = 0.5*((y[k+1]-y[k])/(x[k+1]-x[k]) + (y[k]-y[k-1])/(x[k]-x[k-1]))
    tension : real scalar between 0 and 1.
        tension parameter used in Cardinal method
    monotone : bool
        If True modifies yp to preserve monoticity

    Returns
    -------
    yp : ndarray
        estimated slope

    References:
    -----------
    Wikipedia:  Monotone cubic interpolation
                Cubic Hermite spline

    '''
    x = np.asarray(x, np.float_)
    y = np.asarray(y, np.float_)
    yp = np.zeros(y.shape, np.float_)

    dx = x[1:] - x[:-1]
    # Compute the slopes of the secant lines between successive points
    dydx = (y[1:] - y[:-1]) / dx

    method = method.lower()
    if method.startswith('p'):  # parabola'):
        yp[1:-1] = (dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1]) / \
            (dx[1:] + dx[:-1])
        yp[0] = 2.0 * dydx[0] - yp[1]
        yp[-1] = 2.0 * dydx[-1] - yp[-2]
    else:
        # At the endpoints - use one-sided differences
        yp[0] = dydx[0]
        yp[-1] = dydx[-1]
        if method.startswith('s'):  # secant'):
            # In the middle - use the average of the secants
            yp[1:-1] = (dydx[:-1] + dydx[1:]) / 2.0
        else:  # Cardinal or Catmull-Rom method
            yp[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
            if method.startswith('car'):  # cardinal'):
                yp = (1 - tension) * yp

    if monotone:
        # Special case: intervals where y[k] == y[k+1]
        # Setting these slopes to zero guarantees the spline connecting
        # these points will be flat which preserves monotonicity
        ii, = (dydx == 0.0).nonzero()
        yp[ii] = 0.0
        yp[ii + 1] = 0.0

        alpha = yp[:-1] / dydx
        beta = yp[1:] / dydx
        dist = alpha ** 2 + beta ** 2
        tau = 3.0 / np.sqrt(dist)

        # To prevent overshoot or undershoot, restrict the position vector
        # (alpha, beta) to a circle of radius 3.  If (alpha**2 +  beta**2)>9,
        # then set m[k] = tau[k]alpha[k]delta[k] and
        #  m[k+1] =  tau[k]beta[b]delta[k]
        # where tau = 3/sqrt(alpha**2 + beta**2).

        # Find the indices that need adjustment
        indices_to_fix, = (dist > 9.0).nonzero()
        for ii in indices_to_fix:
            yp[ii] = tau[ii] * alpha[ii] * dydx[ii]
            yp[ii + 1] = tau[ii] * beta[ii] * dydx[ii]

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
    # N = len(y)

    if yp is None:
        yp = slopes(x, y)
    else:
        yp = np.asarray(yp, np.float_)

    xi = np.asarray(xi, np.float_)
    # yi = np.zeros(xi.shape, np.float_)

    # calculate linear slopes
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    s = dy / dx  # note length of s is N-1 so last element is #N-2

    # find the segment each xi is in
    # this line actually is the key to the efficiency of this implementation
    idx = np.searchsorted(x[1:-1], xi)

    # now we have generally: x[idx[j]] <= xi[j] <= x[idx[j]+1]
    # except at the boundaries, where it may be that xi[j] < x[0] or xi[j] >
    # x[-1]

    # the y-values that would come out from a linear interpolation:
    sidx = s.take(idx)
    xidx = x.take(idx)
    yidx = y.take(idx)
    xidxp1 = x.take(idx + 1)
    yo = yidx + sidx * (xi - xidx)

    # the difference that comes when using the slopes given in yp
    # using the yp slope of the left point
    dy1 = (yp.take(idx) - sidx) * (xi - xidx)
    # using the yp slope of the right point
    dy2 = (yp.take(idx + 1) - sidx) * (xi - xidxp1)

    dy1dy2 = dy1 * dy2
    # The following is optimized for Python. The solution actually
    # does more calculations than necessary but exploiting the power
    # of numpy, this is far more efficient than coding a loop by hand
    # in Python
    dy1mdy2 = np.where(dy1dy2, dy1 - dy2, np.inf)
    dy1pdy2 = np.where(dy1dy2, dy1 + dy2, np.inf)
    yi = yo + dy1dy2 * np.choose(
        np.array(np.sign(dy1dy2), np.int32) + 1,
        ((2 * xi - xidx - xidxp1) / ((dy1mdy2) * (xidxp1 - xidx)), 0.0,
         1 / (dy1pdy2)))
    return yi


class StinemanInterp(object):

    '''
    Returns an interpolating function
        that runs through a set of points according to the algorithm of
        Stineman (1980).

    Parameters
    ----------
    x,y : array-like
        coordinates of points defining the interpolating function.
    yp : array-like
        slopes of the interpolating function at x.
        Optional: only given if they are known, else the argument is not used.
    method : string
        method for computing the slope at the given points if the slope is not
        known. With method= "parabola" calculates the slopes from a parabola
        through every three points.

    Notes
    -----
    The interpolation method is described by Russell W. Stineman (1980)

    According to Stineman, the interpolation procedure has "the following
    properties:

    If values of the ordinates of the specified points change monotonically,
    and the slopes of the line segments joining the points change
    monotonically, then the interpolating curve and its slope will change
    monotonically. If the slopes of the line segments joining the specified
    points change monotonically, then the slopes of the interpolating curve
    will change monotonically. Suppose that the conditions in (1) or (2) are
    satisfied by a set of points, but a small change in the ordinate or slope
    at one of the points will result conditions(1) or (2) being not longer
    satisfied. Then making this small change in the ordinate or slope at a
    point will cause no more than a small change in the interpolating
    curve." The method is based on rational interpolation with specially chosen
    rational functions to satisfy the above three conditions.

    Slopes computed at the given points with the methods provided by the
    `StinemanInterp' function satisfy Stineman's requirements.
    The original method suggested by Stineman(method="scaledstineman", the
    default, and "stineman") result in lower slopes near abrupt steps or spikes
    in the point sequence, and therefore a smaller tendency for overshooting.
    The method based on a second degree polynomial(method="parabola") provides
    better approximation to smooth functions, but it results in in higher
    slopes near abrupt steps or spikes and can lead to some overshooting where
    Stineman's method does not. Both methods lead to much less tendency for
    `spurious' oscillations than traditional interplation methods based on
    polynomials, such as splines
    (see the examples section).

    Stineman states that "The complete assurance that the procedure will never
    generate `wild' points makes it attractive as a general purpose procedure".

    This interpolation method has been implemented in Matlab and R in addition
    to Python.

    Examples
    --------
    >>> import wafo.interpolate as wi
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0,2*pi,20)
    >>> y = np.sin(x); yp = np.cos(x)
    >>> xi = np.linspace(0,2*pi,40);
    >>> yi = wi.StinemanInterp(x,y)(xi)
    >>> yi1 = wi.CubicHermiteSpline(x,y, yp)(xi)
    >>> yi2 = wi.Pchip(x,y, method='parabola')(xi)
    >>> h=plt.subplot(211)
    >>> h=plt.plot(x,y,'o',xi,yi,'r', xi,yi1, 'g', xi,yi1, 'b')
    >>> h=plt.subplot(212)
    >>> h=plt.plot(xi,np.abs(sin(xi)-yi), 'r',
    ...            xi,  np.abs(sin(xi)-yi1), 'g',
    ...            xi, np.abs(sin(xi)-yi2), 'b')

    References
    ----------
    Stineman, R. W. A Consistently Well Behaved Method of Interpolation.
    Creative Computing (1980), volume 6, number 7, p. 54-57.

    See Also
    --------
    slopes, Pchip
    '''
    def __init__(self, x, y, yp=None, method='parabola', monotone=False):
        if yp is None:
            yp = slopes(x, y, method, monotone=monotone)
        self.x = np.asarray(x, np.float_)
        self.y = np.asarray(y, np.float_)
        self.yp = np.asarray(yp, np.float_)

    def __call__(self, xi):
        xi = np.asarray(xi, np.float_)
        x = self.x
        y = self.y
        yp = self.yp
        # calculate linear slopes
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        s = dy / dx  # note length of s is N-1 so last element is #N-2

        # find the segment each xi is in
        # this line actually is the key to the efficiency of this
        # implementation
        idx = np.searchsorted(x[1:-1], xi)

        # now we have generally: x[idx[j]] <= xi[j] <= x[idx[j]+1]
        # except at the boundaries, where it may be that xi[j] < x[0] or xi[j]
        # > x[-1]

        # the y-values that would come out from a linear interpolation:
        sidx = s.take(idx)
        xidx = x.take(idx)
        yidx = y.take(idx)
        xidxp1 = x.take(idx + 1)
        yo = yidx + sidx * (xi - xidx)

        # the difference that comes when using the slopes given in yp
        # using the yp slope of the left point
        dy1 = (yp.take(idx) - sidx) * (xi - xidx)
        # using the yp slope of the right point
        dy2 = (yp.take(idx + 1) - sidx) * (xi - xidxp1)

        dy1dy2 = dy1 * dy2
        # The following is optimized for Python. The solution actually
        # does more calculations than necessary but exploiting the power
        # of numpy, this is far more efficient than coding a loop by hand
        # in Python
        dy1mdy2 = np.where(dy1dy2, dy1 - dy2, np.inf)
        dy1pdy2 = np.where(dy1dy2, dy1 + dy2, np.inf)
        yi = yo + dy1dy2 * np.choose(
            np.array(np.sign(dy1dy2), np.int32) + 1,
            ((2 * xi - xidx - xidxp1) / ((dy1mdy2) * (xidxp1 - xidx)), 0.0,
             1 / (dy1pdy2)))
        return yi


class StinemanInterp2(PiecewisePolynomial):

    def __init__(self, x, y, yp=None, method='parabola', monotone=False):
        if yp is None:
            yp = slopes(x, y, method, monotone=monotone)
        super(StinemanInterp2, self).__init__(x, zip(y, yp))


class CubicHermiteSpline(PiecewisePolynomial):

    '''
    Piecewise Cubic Hermite Interpolation using Catmull-Rom
    method for computing the slopes.
    '''

    def __init__(self, x, y, yp=None, method='Catmull-Rom'):
        if yp is None:
            yp = slopes(x, y, method, monotone=False)
        super(CubicHermiteSpline, self).__init__(x, zip(y, yp), orders=3)


class Pchip(PiecewisePolynomial):

    """PCHIP 1-d monotonic cubic interpolation

    Description
    -----------
    x and y are arrays of values used to approximate some function f:
       y = f(x)
    This class factory function returns a callable class whose __call__ method
    uses monotonic cubic, interpolation to find the value of new points.

    Parameters
    ----------
    x : array
        A 1D array of monotonically increasing real values.  x cannot
        include duplicate values (otherwise f is overspecified)
    y : array
        A 1-D array of real values.  y's length along the interpolation
        axis must be equal to the length of x.
    yp : array
        slopes of the interpolating function at x.
        Optional: only given if they are known, else the argument is not used.
    method : string
        method for computing the slope at the given points if the slope is not
        known. With method="parabola" calculates the slopes from a parabola
        through every three points.

    Assumes x is sorted in monotonic order (e.g. x[1] > x[0])

    Example
    -------
    >>> import wafo.interpolate as wi

    # Create a step function (will demonstrate monotonicity)
    >>> x = np.arange(7.0) - 3.0
    >>> y = np.array([-1.0, -1,-1,0,1,1,1])

    # Interpolate using monotonic piecewise Hermite cubic spline
    >>> xvec = np.arange(599.)/100. - 3.0
    >>> yvec = wi.Pchip(x, y)(xvec)

    # Call the Scipy cubic spline interpolator
    >>> from scipy.interpolate import interpolate
    >>> function = interpolate.interp1d(x, y, kind='cubic')
    >>> yvec1 = function(xvec)

    # Non-montonic cubic Hermite spline interpolator using
    # Catmul-Rom method for computing slopes...
    >>> yvec2 = wi.CubicHermiteSpline(x,y)(xvec)

    >>> yvec3 = wi.StinemanInterp(x, y)(xvec)

    # Plot the results
    >>> import matplotlib.pyplot as plt
    >>> h=plt.plot(x,    y,     'ro')
    >>> h=plt.plot(xvec, yvec,  'b')
    >>> h=plt.plot(xvec, yvec1, 'k')
    >>> h=plt.plot(xvec, yvec2, 'g')
    >>> h=plt.plot(xvec, yvec3, 'm')
    >>> h=plt.title("pchip() step function test")

    >>> h=plt.xlabel("X")
    >>> h=plt.ylabel("Y")
    >>> txt = "Comparing pypchip() vs. Scipy interp1d() vs. non-monotonic CHS"
    >>> h=plt.title(txt)
    >>> legends = ["Data", "pypchip()", "interp1d","CHS", 'SI']
    >>> h=plt.legend(legends, loc="upper left")
    >>> plt.show()

    """

    def __init__(self, x, y, yp=None, method='secant'):
        if yp is None:
            yp = slopes(x, y, method=method, monotone=True)
        super(Pchip, self).__init__(x, zip(y, yp), orders=3)


def interp3(x, y, z, v, xi, yi, zi, method='cubic'):
    """Interpolation on 3-D. x, y, xi, yi should be 1-D
    and z.shape == (len(x), len(y), len(z))"""
    q = (x, y, z)
    qi = (xi, yi, zi)
    for j in range(3):
        pp = interp1d(q[j], v, axis=j, kind=method)
        v = pp(qi[j])
    return v


def somefunc(x, y, z):
    return x**2 + y**2 - z**2 + x*y*z


def test_interp3():
    # some input data
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 2, 6)
    z = np.linspace(0, 3, 7)
    v = somefunc(x[:, None, None], y[None, :, None], z[None, None, :])

    # interpolate
    xi = np.linspace(0, 1, 45)
    yi = np.linspace(0, 2, 46)
    zi = np.linspace(0, 3, 47)
    vi = interp3(x, y, z, v, xi, yi, zi)

    import matplotlib.pyplot as plt
    X, Y = np.meshgrid(xi, yi)
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.pcolor(X, Y, vi[:, :, 12].T)
    plt.title('interpolated')
    plt.subplot(1, 2, 2)
    plt.pcolor(X, Y, somefunc(xi[:, None], yi[None, :], zi[12]).T)
    plt.title('exact')
    plt.show('hold')


def test_smoothing_spline():
    x = linspace(0, 2 * pi + pi / 4, 20)
    y = sin(x)  # + np.random.randn(x.size)
    pp = SmoothSpline(x, y, p=1)
    x1 = linspace(-1, 2 * pi + pi / 4 + 1, 20)
    y1 = pp(x1)
    pp1 = pp.derivative()
    pp0 = pp1.integrate()
    dy1 = pp1(x1)
    y01 = pp0(x1)
    # dy = y-y1
    import matplotlib.pyplot as plt

    plt.plot(x, y, x1, y1, '.', x1, dy1, 'ro', x1, y01, 'r-')
    plt.show('hold')
    pass
    # tck = interpolate.splrep(x, y, s=len(x))


def compare_methods():
    #
    # Sine wave test
    #
    fun = np.sin
    # Create a example vector containing a sine wave.
    x = np.arange(30.0) / 10.
    y = fun(x)

    # Interpolate the data above to the grid defined by "xvec"
    xvec = np.arange(250.) / 100.

    # Initialize the interpolator slopes
    # Create the pchip slopes
    m = slopes(x, y,  method='parabola', monotone=True)
    m1 = slopes(x, y, method='parabola', monotone=False)
    m2 = slopes(x, y, method='catmul', monotone=False)
    m3 = pchip_slopes(x, y)

    # Call the monotonic piece-wise Hermite cubic interpolator
    yvec = Pchip(x, y, m)(xvec)
    yvec1 = Pchip(x, y, m1)(xvec)
    yvec2 = Pchip(x, y, m2)(xvec)
    yvec3 = Pchip(x, y, m3)(xvec)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y, 'ro', xvec, fun(xvec), 'r')
    plt.title("pchip() Sin test code")

    # Plot the interpolated points
    plt.plot(xvec, yvec, xvec, yvec1, xvec, yvec2, 'g.', xvec, yvec3)
    plt.legend(
        ['true', 'true', 'parbola_monoton', 'parabola', 'catmul', 'pchip'],
        frameon=False, loc=0)
    plt.ioff()
    plt.show()


def demo_monoticity():
    # Step function test...
    import matplotlib.pyplot as plt
    plt.figure(2)
    plt.title("pchip() step function test")
    # Create a step function (will demonstrate monotonicity)
    x = np.arange(7.0) - 3.0
    y = np.array([-1.0, -1, -1, 0, 1, 1, 1])

    # Interpolate using monotonic piecewise Hermite cubic spline
    xvec = np.arange(599.) / 100. - 3.0

    # Create the pchip slopes
    m = slopes(x, y, monotone=True)
#    m1 = slopes(x, y, monotone=False)
#    m2  = slopes(x,y,method='catmul',monotone=False)
    m3 = pchip_slopes(x, y)
    # Interpolate...
    yvec = Pchip(x, y, m)(xvec)

    # Call the Scipy cubic spline interpolator
    from scipy.interpolate import interpolate as ip
    function = ip.interp1d(x, y, kind='cubic')
    yvec2 = function(xvec)

    # Non-montonic cubic Hermite spline interpolator using
    # Catmul-Rom method for computing slopes...
    yvec3 = CubicHermiteSpline(x, y)(xvec)
    yvec4 = StinemanInterp(x, y)(xvec)
    yvec5 = Pchip(x, y, m3)(xvec)  # @UnusedVariable

    # Plot the results
    plt.plot(x,    y,     'ro', label='Data')
    plt.plot(xvec, yvec,  'b', label='Pchip')
    plt.plot(xvec, yvec2, 'k', label='interp1d')
    plt.plot(xvec, yvec3, 'g', label='CHS')
    plt.plot(xvec, yvec4, 'm', label='Stineman')
    # plt.plot(xvec, yvec5, 'yo', label='Pchip2')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Comparing Pchip() vs. Scipy interp1d() vs. non-monotonic CHS")
#    legends = ["Data", "Pchip()", "interp1d","CHS", 'Stineman']
    plt.legend(loc="upper left", frameon=False)
    plt.ioff()
    plt.show()


def test_func():
    from scipy import interpolate
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.interactive(False)

    coef = np.array([[1, 1], [0, 1]])  # linear from 0 to 2
    # coef = np.array([[1,1],[1,1],[0,2]]) # linear from 0 to 2
    breaks = [0, 1, 2]
    pp = PPform(coef, breaks, a=-100, b=100)
    x = linspace(-1, 3, 20)
    y = pp(x)  # @UnusedVariable

    x = linspace(0, 2 * pi + pi / 4, 20)
    y = sin(x) + np.random.randn(x.size)
    tck = interpolate.splrep(x, y, s=len(x))  # @UndefinedVariable
    xnew = linspace(0, 2 * pi, 100)
    ynew = interpolate.splev(xnew, tck, der=0)  # @UndefinedVariable
    tck0 = interpolate.splmake(  # @UndefinedVariable
        xnew, ynew, order=3, kind='smoothest', conds=None)
    pp = interpolate.ppform.fromspline(*tck0)  # @UndefinedVariable

    plt.plot(x, y, "x", xnew, ynew, xnew, sin(xnew), x, y, "b", x, pp(x), 'g')
    plt.legend(['Linear', 'Cubic Spline', 'True'])
    plt.title('Cubic-spline interpolation')
    plt.show()

    t = np.arange(0, 1.1, .1)
    x = np.sin(2 * np.pi * t)
    y = np.cos(2 * np.pi * t)
    _tck1, _u = interpolate.splprep([t, y], s=0)  # @UndefinedVariable
    tck2 = interpolate.splrep(t, y, s=len(t), task=0)  # @UndefinedVariable
    # interpolate.spl
    tck = interpolate.splmake(t, y, order=3, kind='smoothest', conds=None)
    self = interpolate.ppform.fromspline(*tck2)  # @UndefinedVariable
    plt.plot(t, self(t))
    plt.show('hold')
    pass


def test_pp():
    coef = np.array([[1, 1], [0, 0]])  # linear from 0 to 2 @UnusedVariable

    # quadratic from 0 to 1 and 1 to 2.
    coef = np.array([[1, 1], [1, 1], [0, 2]])
    dc = pl.polyder(coef, 1)
    c2 = pl.polyint(dc, 1)  # @UnusedVariable
    breaks = [0, 1, 2]
    pp = PPform(coef, breaks)
    pp(0.5)
    pp(1)
    pp(1.5)
    dpp = pp.derivative()
    import matplotlib.pyplot as plt
    x = np.linspace(-1, 3)
    plt.plot(x, pp(x), x, dpp(x), '.')
    plt.show()


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    # test_func()
    # test_doctstrings()
    test_smoothing_spline()
    # compare_methods()
    # demo_monoticity()
    # test_interp3()
