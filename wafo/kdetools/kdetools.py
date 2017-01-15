#!/usr/bin/env python
# -------------------------------------------------------------------------
# Name:        kdetools
# Purpose:
#
# Author:      pab
#
# Created:     01.11.2008
# Copyright:   (c) pab 2008
# Licence:     LGPL
# -------------------------------------------------------------------------

from __future__ import absolute_import, division
# from abc import ABCMeta, abstractmethod
import copy
import warnings
import numpy as np
import scipy.stats
from scipy import interpolate, linalg, special
from numpy import sqrt, atleast_2d, meshgrid
from numpy.fft import fftn, ifftn
from wafo.misc import nextpow2
from wafo.containers import PlotData
from wafo.testing import test_docstrings
from wafo.kdetools.kernels import iqrange, qlevels, Kernel
from wafo.kdetools.gridding import gridcount

__all__ = ['TKDE', 'KDE', 'test_docstrings', 'KRegression', 'BKRegression']

_TINY = np.finfo(float).machar.tiny
# _REALMIN = np.finfo(float).machar.xmin
_REALMAX = np.finfo(float).machar.xmax
_EPS = np.finfo(float).eps


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


def _assert_warn(cond, msg):
    if not cond:
        warnings.warn(msg)


def _invnorm(q):
    return special.ndtri(q)


def _logit(p):
    pc = p.clip(min=0, max=1)
    return (np.log(pc) - np.log1p(-pc)).clip(min=-40, max=40)


# def _logitinv(x):
#     return 1.0 / (np.exp(-x) + 1)


class _KDE(object):

    def __init__(self, data, kernel=None, xmin=None, xmax=None):
        self.dataset = data
        self.xmin = xmin
        self.xmax = xmax
        self.kernel = kernel if kernel else Kernel('gauss')

    @property
    def inc(self):
        return self._inc

    @inc.setter
    def inc(self, inc):
        self._inc = inc

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, data):
        self._dataset = atleast_2d(data)

    @property
    def n(self):
        return self.dataset.shape[1]

    @property
    def d(self):
        return self.dataset.shape[0]

    @property
    def sigma(self):
        """minimum(stdev, 0.75 * interquartile-range)"""
        iqr = iqrange(self.dataset, axis=-1)
        sigma = np.minimum(np.std(self.dataset, axis=-1, ddof=1), iqr / 1.34)
        return sigma

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, xmin):
        if xmin is None:
            xmin = self.dataset.min(axis=-1) - 2 * self.sigma
        self._xmin = self._check_xmin(xmin*np.ones(self.d))

    def _check_xmin(self, xmin):
        return xmin

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, xmax):
        if xmax is None:
            xmax = self.dataset.max(axis=-1) + 2 * self.sigma

        self._xmax = self._check_xmax(xmax * np.ones(self.d))

    def _check_xmax(self, xmax):
        return xmax

    def eval_grid_fast(self, *args, **kwds):
        """Evaluate the estimated pdf on a grid using fft.

        Parameters
        ----------
        arg_0,arg_1,... arg_d-1 : vectors
            Alternatively, if no vectors is passed in then
             arg_i = linspace(self.xmin[i], self.xmax[i], self.inc)
        output : string optional
            'value' if value output
            'data' if object output

        Returns
        -------
        values : array-like
            The values evaluated at meshgrid(*args).

        """
        return self.eval_grid_fun(self._eval_grid_fast, *args, **kwds)

    def _eval_grid_fast(self, *args, **kwds):
        pass

    def eval_grid(self, *args, **kwds):
        """Evaluate the estimated pdf on a grid.

        Parameters
        ----------
        arg_0,arg_1,... arg_d-1 : vectors
            Alternatively, if no vectors is passed in then
             arg_i = linspace(self.xmin[i], self.xmax[i], self.inc)
        output : string optional
            'value' if value output
            'data' if object output

        Returns
        -------
        values : array-like
            The values evaluated at meshgrid(*args).

        """
        return self.eval_grid_fun(self._eval_grid, *args, **kwds)

    def _eval_grid(self, *args, **kwds):
        pass

    def _add_contour_levels(self, wdata):
        p_levels = np.r_[10:90:20, 95, 99, 99.9]
        try:
            c_levels = qlevels(wdata.data, p=p_levels)
            wdata.clevels = c_levels
            wdata.plevels = p_levels
        except Exception as e:
            msg = "Could not calculate contour levels!. ({})".format(str(e))
            warnings.warn(msg)

    def _make_object(self, f, **kwds):
        titlestr = 'Kernel density estimate ({})'.format(self.kernel.name)
        kwds2 = dict(title=titlestr)
        kwds2['plot_kwds'] = dict(plotflag=1)
        kwds2.update(**kwds)
        args = self.args
        if self.d == 1:
            args = args[0]
        wdata = PlotData(f, args, **kwds2)
        if self.d > 1:
            self._add_contour_levels(wdata)
        return wdata

    def get_args(self, xmin=None, xmax=None):
        sxmin = self.xmin
        if xmin is not None:
            sxmin = np.minimum(xmin, sxmin)

        sxmax = self.xmax
        if xmax is not None:
            sxmax = np.maximum(xmax, sxmax)

        args = []
        inc = self.inc
        for i in range(self.d):
            args.append(np.linspace(sxmin[i], sxmax[i], inc))
        return args

    def eval_grid_fun(self, eval_grd, *args, **kwds):
        if len(args) == 0:
            args = self.get_args()
        self.args = args
        output = kwds.pop('output', 'value')
        f = eval_grd(*args, **kwds)
        if output == 'value':
            return f
        return self._make_object(f, **kwds)

    def _check_shape(self, points):
        points = atleast_2d(points)
        d, m = points.shape
        if d != self.d:
            _assert(d == 1 and m == self.d, "points have dimension {}, "
                    "dataset has dimension {}".format(d, self.d))
            # points was passed in as a row vector
            points = np.reshape(points, (self.d, 1))
        return points

    def eval_points(self, points, **kwds):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError if the dimensionality of the input points is different than
        the dimensionality of the KDE.

        """

        points = self._check_shape(points)
        return self._eval_points(points, **kwds)

    def _eval_points(self, points, **kwds):
        pass

    __call__ = eval_grid


class TKDE(_KDE):

    """ Transformation Kernel-Density Estimator.

    Parameters
    ----------
    dataset : (# of dims, # of data)-array
        datapoints to estimate from
    hs : array-like (optional)
        smooting parameter vector/matrix.
        (default compute from data using kernel.get_smoothing function)
    kernel :  kernel function object.
        kernel must have get_smoothing method
    alpha : real scalar (optional)
        sensitivity parameter               (default 0 regular KDE)
        A good choice might be alpha = 0.5 ( or 1/D)
        alpha = 0      Regular  KDE (hs is constant)
        0 < alpha <= 1 Adaptive KDE (Make hs change)
    xmin, xmax  : vectors
        specifying the default argument range for the kde.eval_grid methods.
        For the kde.eval_grid_fast methods the values must cover the range of
        the data. (default min(data)-range(data)/4, max(data)-range(data)/4)
        If a single value of xmin or xmax is given then the boundary is the is
        the same for all dimensions.
    inc :  scalar integer
        defining the default dimension of the output from kde.eval_grid methods
        (default 512)
        (For kde.eval_grid_fast: A value below 50 is very fast to compute but
        may give some inaccuracies. Values between 100 and 500 give very
        accurate results)
    L2 : array-like
        vector of transformation parameters (default 1 no transformation)
        t(xi;L2) = xi^L2*sign(L2)   for L2(i) ~= 0
        t(xi;L2) = log(xi)          for L2(i) == 0
        If single value of L2 is given then the transformation is the same in
        all directions.

    Members
    -------
    d : int
        number of dimensions
    n : int
        number of datapoints

    Methods
    -------
    kde.eval_grid_fast(x0, x1,..., xd) : array
        evaluate the estimated pdf on meshgrid(x0, x1,..., xd)
    kde.eval_grid(x0, x1,..., xd) : array
        evaluate the estimated pdf on meshgrid(x0, x1,..., xd)
    kde.eval_points(points) : array
        evaluate the estimated pdf on a provided set of points
    kde(x0, x1,..., xd) : array
        same as kde.eval_grid(x0, x1,..., xd)

    Example
    -------
    N = 20
    data = np.random.rayleigh(1, size=(N,))
    >>> data = np.array([
    ...        0.75355792,  0.72779194,  0.94149169,  0.07841119,2.32291887,
    ...        1.10419995,  0.77055114,  0.60288273,  1.36883635,  1.74754326,
    ...        1.09547561,  1.01671133,  0.73211143,  0.61891719,  0.75903487,
    ...        1.8919469 ,  0.72433808,  1.92973094,  0.44749838,  1.36508452])

    >>> import wafo.kdetools as wk
    >>> x = np.linspace(0.01, max(data.ravel()) + 1, 10)
    >>> kde = wk.TKDE(data, hs=0.5, L2=0.5)
    >>> f = kde(x)
    >>> f
    array([ 1.03982714,  0.45839018,  0.39514782,  0.32860602,  0.26433318,
            0.20717946,  0.15907684,  0.1201074 ,  0.08941027,  0.06574882])

    >>> kde.eval_grid(x)
    array([ 1.03982714,  0.45839018,  0.39514782,  0.32860602,  0.26433318,
            0.20717946,  0.15907684,  0.1201074 ,  0.08941027,  0.06574882])

    >>> kde.eval_grid_fast(x)
    array([ 1.04018924,  0.45838973,  0.39514689,  0.32860532,  0.26433301,
            0.20717976,  0.15907697,  0.1201077 ,  0.08941129,  0.06574899])

    import pylab as plb
    h1 = plb.plot(x, f) #  1D probability density plot
    t = np.trapz(f, x)
    """

    def __init__(self, data, hs=None, kernel=None, alpha=0.0,
                 xmin=None, xmax=None, inc=512, L2=None):
        self.L2 = L2
        super(TKDE, self).__init__(data, kernel, xmin, xmax)

        tdataset = self._transform(self.dataset)
        txmin = np.ravel(self._transform(np.reshape(self.xmin, (-1, 1))))
        txmax = np.ravel(self._transform(np.reshape(self.xmax, (-1, 1))))
        self.tkde = KDE(tdataset, hs, self.kernel, alpha, txmin, txmax, inc)

    def _check_xmin(self, xmin):
        if self.L2 is not None:
            amin = self.dataset.min(axis=-1)
            L2 = np.atleast_1d(self.L2) * np.ones(self.d)
            xmin = np.where(L2 != 1, np.maximum(xmin, amin / 100.0), xmin)
        return xmin

    @property
    def inc(self):
        return self.tkde.inc

    @inc.setter
    def inc(self, inc):
        self.tkde.inc = inc

    @property
    def hs(self):
        return self.tkde.hs

    @hs.setter
    def hs(self, hs):
        self.tkde.hs = hs

    def _transform(self, points):
        if self.L2 is None:
            return points  # default no transformation

        L2 = np.atleast_1d(self.L2) * np.ones(self.d)

        tpoints = copy.copy(points)
        for i, v2 in enumerate(L2.tolist()):
            tpoints[i] = np.log(points[i]) if v2 == 0 else points[i] ** v2
        return tpoints

    def _inverse_transform(self, tpoints):
        if self.L2 is None:
            return tpoints  # default no transformation

        L2 = np.atleast_1d(self.L2) * np.ones(self.d)

        points = copy.copy(tpoints)
        for i, v2 in enumerate(L2.tolist()):
            points[i] = np.exp(
                tpoints[i]) if v2 == 0 else tpoints[i] ** (1.0 / v2)
        return points

    def _scale_pdf(self, pdf, points):
        if self.L2 is None:
            return pdf
        # default no transformation
        L2 = np.atleast_1d(self.L2) * np.ones(self.d)
        for i, v2 in enumerate(L2.tolist()):
            factor = v2 * np.sign(v2) if v2 else 1
            pdf *= np.where(v2 == 1, 1, points[i] ** (v2 - 1) * factor)

        _assert_warn((np.abs(np.diff(pdf)).max() < 10).all(), '''
        Numerical problems may have occured due to the power transformation.
        Check the KDE for spurious spikes''')
        return pdf

    def _interpolate(self, points, f, *args, **kwds):
        ipoints = meshgrid(*args)  # if self.d > 1 else args
        for i in range(self.d):
            points[i].shape = -1,
        points = np.asarray(points).T

        fi = interpolate.griddata(points, np.ravel(f), tuple(ipoints),
                                  method='linear', fill_value=0.0)
        self.args = args
        r = kwds.get('r', 0)
        if r == 0:
            return fi * (fi > 0)
        return fi

    def _get_targs(self, args):
        targs = []
        if len(args):
            targs0 = self._transform(list(args))
            xmin = [min(t) for t in targs0]
            xmax = [max(t) for t in targs0]
            targs = self.tkde.get_args(xmin, xmax)
        return targs

    def _eval_grid_fast(self, *args, **kwds):
        if self.L2 is None:
            f = self.tkde.eval_grid_fast(*args, **kwds)
            self.args = self.tkde.args
            return f
        targs = self._get_targs(args)
        tf = self.tkde.eval_grid_fast(*targs)

        self.args = self._inverse_transform(list(self.tkde.args))
        points = meshgrid(*self.args)
        f = self._scale_pdf(tf, points)
        if len(args):
            return self._interpolate(points, f, *args, **kwds)
        return f

    def _eval_grid(self, *args, **kwds):
        if self.L2 is None:
            return self.tkde.eval_grid(*args, **kwds)
        targs = self._transform(list(args))
        tf = self.tkde.eval_grid(*targs, **kwds)
        points = meshgrid(*args)
        f = self._scale_pdf(tf, points)
        return f

    def _eval_points(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError if the dimensionality of the input points is different than
        the dimensionality of the KDE.

        """
        if self.L2 is None:
            return self.tkde.eval_points(points)

        tpoints = self._transform(points)
        tf = self.tkde.eval_points(tpoints)
        f = self._scale_pdf(tf, points)
        return f


class KDE(_KDE):

    """ Kernel-Density Estimator.

    Parameters
    ----------
    data : (# of dims, # of data)-array
        datapoints to estimate from
    hs : array-like (optional)
        smooting parameter vector/matrix.
        (default compute from data using kernel.get_smoothing function)
    kernel :  kernel function object.
        kernel must have get_smoothing method
    alpha : real scalar (optional)
        sensitivity parameter               (default 0 regular KDE)
        A good choice might be alpha = 0.5 ( or 1/D)
        alpha = 0      Regular  KDE (hs is constant)
        0 < alpha <= 1 Adaptive KDE (Make hs change)
    xmin, xmax  : vectors
        specifying the default argument range for the kde.eval_grid methods.
        For the kde.eval_grid_fast methods the values must cover the range of
        the data.
        (default min(data)-range(data)/4, max(data)-range(data)/4)
        If a single value of xmin or xmax is given then the boundary is the is
        the same for all dimensions.
    inc :  scalar integer (default 512)
        defining the default dimension of the output from kde.eval_grid methods
        (For kde.eval_grid_fast: A value below 50 is very fast to compute but
        may give some inaccuracies. Values between 100 and 500 give very
        accurate results)

    Members
    -------
    d : int
        number of dimensions
    n : int
        number of datapoints

    Methods
    -------
    kde.eval_grid_fast(x0, x1,..., xd) : array
        evaluate the estimated pdf on meshgrid(x0, x1,..., xd)
    kde.eval_grid(x0, x1,..., xd) : array
        evaluate the estimated pdf on meshgrid(x0, x1,..., xd)
    kde.eval_points(points) : array
        evaluate the estimated pdf on a provided set of points
    kde(x0, x1,..., xd) : array
        same as kde.eval_grid(x0, x1,..., xd)

    Example
    -------
    N = 20
    data = np.random.rayleigh(1, size=(N,))
    >>> data = np.array([
    ...        0.75355792,  0.72779194,  0.94149169,  0.07841119,  2.32291887,
    ...        1.10419995,  0.77055114,  0.60288273,  1.36883635,  1.74754326,
    ...        1.09547561,  1.01671133,  0.73211143,  0.61891719,  0.75903487,
    ...        1.8919469 ,  0.72433808,  1.92973094,  0.44749838,  1.36508452])

    >>> x = np.linspace(0, max(data.ravel()) + 1, 10)
    >>> import wafo.kdetools as wk
    >>> kde = wk.KDE(data, hs=0.5, alpha=0.5)
    >>> f = kde(x)
    >>> f
    array([ 0.17252055,  0.41014271,  0.61349072,  0.57023834,  0.37198073,
            0.21409279,  0.12738463,  0.07460326,  0.03956191,  0.01887164])

    >>> kde.eval_grid(x)
    array([ 0.17252055,  0.41014271,  0.61349072,  0.57023834,  0.37198073,
            0.21409279,  0.12738463,  0.07460326,  0.03956191,  0.01887164])
    >>> kde.eval_grid_fast(x)
    array([ 0.20729484,  0.39865044,  0.53716945,  0.5169322 ,  0.39060223,
            0.26441126,  0.16388801,  0.08388527,  0.03227164,  0.00883579])

    >>> kde0 = wk.KDE(data, hs=0.5, alpha=0.0)
    >>> kde0.eval_points(x)
    array([ 0.2039735 ,  0.40252503,  0.54595078,  0.52219649,  0.3906213 ,
            0.26381501,  0.16407362,  0.08270612,  0.02991145,  0.00720821])

    >>> kde0.eval_grid(x)
    array([ 0.2039735 ,  0.40252503,  0.54595078,  0.52219649,  0.3906213 ,
            0.26381501,  0.16407362,  0.08270612,  0.02991145,  0.00720821])
    >>> f = kde0.eval_grid(x, output='plotobj')
    >>> f.data
    array([ 0.2039735 ,  0.40252503,  0.54595078,  0.52219649,  0.3906213 ,
            0.26381501,  0.16407362,  0.08270612,  0.02991145,  0.00720821])

    >>> f = kde0.eval_grid_fast()
    >>> np.allclose(np.interp(x, kde0.args[0], f),
    ...    [ 0.20398034,  0.40252166,  0.54593292,  0.52218993,  0.39062245,
    ...     0.26381651,  0.16407487,  0.08270847,  0.02991439,  0.00882095])
    True
    >>> f1 = kde0.eval_grid_fast(output='plot')
    >>> np.allclose(np.interp(x, f1.args, f1.data),
    ...   [ 0.20398034,  0.40252166,  0.54593292,  0.52218993,  0.39062245,
    ...     0.26381651,  0.16407487,  0.08270847,  0.02991439,  0.00882095])
    True

    h = f1.plot()
    import pylab as plb
    h1 = plb.plot(x, f) #  1D probability density plot
    t = np.trapz(f, x)
    """

    def __init__(self, data, hs=None, kernel=None, alpha=0.0, xmin=None,
                 xmax=None, inc=512):
        super(KDE, self).__init__(data, kernel, xmin, xmax)
        self.hs = hs
        self.inc = inc
        self.alpha = alpha

    def _replace_negatives_with_default_hs(self, h):
        get_default_hs = self.kernel.get_smoothing
        ind, = np.where(h <= 0)
        for i in ind.tolist():
            h[i] = get_default_hs(self.dataset[i])

    def _check_hs(self, h):
        """make sure it has the correct dimension and replace negative vals"""
        h = np.atleast_1d(h)
        if (len(h.shape) == 1) or (self.d == 1):
            h = h * np.ones(self.d) if max(h.shape) == 1 else h.reshape(self.d)
            self._replace_negatives_with_default_hs(h)
        return h

    def _invert_hs(self, h):
        if (len(h.shape) == 1) or (self.d == 1):
            determinant = h.prod()
            inv_hs = np.diag(1.0 / h)
        else:  # fully general smoothing matrix
            determinant = linalg.det(h)
            _assert(0 < determinant,
                    'bandwidth matrix h must be positive definit!')
            inv_hs = linalg.inv(h)
        return inv_hs, determinant

    @property
    def hs(self):
        return self._hs

    @hs.setter
    def hs(self, h):
        if h is None:
            h = self.kernel.get_smoothing(self.dataset)
        h = self._check_hs(h)
        self._inv_hs, deth = self._invert_hs(h)
        self._norm_factor = deth * self.n
        self._hs = h

    @property
    def inc(self):
        return self._inc

    @inc.setter
    def inc(self, inc):
        if inc is None:
            _tau, tau = self.kernel.effective_support()
            xyzrange = 8 * self.sigma
            L1 = 10
            inc = max(48, (L1 * xyzrange / (tau * self.hs)).max())
            inc = 2 ** nextpow2(inc)
        self._inc = inc

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self._lambda = np.ones(self.n)
        if alpha > 0:
            f = self.eval_points(self.dataset)  # pilot estimate
            g = np.exp(np.mean(np.log(f)))
            self._lambda = (f / g) ** (-alpha)

    @staticmethod
    def _make_flat_grid(dx, d, inc):
        Xn = []
        x0 = np.linspace(-inc, inc, 2 * inc + 1)
        for i in range(d):
            Xn.append(x0[:-1] * dx[i])

        Xnc = meshgrid(*Xn)

        for i in range(d):
            Xnc[i].shape = (-1,)
        return np.vstack(Xnc)

    def _kernel_weights(self, Xn, dx, d, inc):
        kw = self.kernel(Xn)
        norm_fact0 = (kw.sum() * dx.prod() * self.n)
        norm_fact = (self._norm_factor * self.kernel.norm_factor(d, self.n))
        if np.abs(norm_fact0 - norm_fact) > 0.05 * norm_fact:
            warnings.warn(
                'Numerical inaccuracy due to too low discretization. ' +
                'Increase the discretization of the evaluation grid ' +
                '(inc={})!'.format(inc))
            norm_fact = norm_fact0

        kw = kw / norm_fact
        return kw

    def _eval_grid_fast(self, *args, **kwds):
        X = np.vstack(args)
        d, inc = X.shape
        dx = X[:, 1] - X[:, 0]

        Xnc = self._make_flat_grid(dx, d, inc)

        Xn = np.dot(self._inv_hs, Xnc)
        kw = self._kernel_weights(Xn, dx, d, inc)

        r = kwds.get('r', 0)
        if r != 0:
            fun = self._moment_fun(r)
            kw *= fun(np.vstack(Xnc))
        kw.shape = (2 * inc, ) * d
        kw = np.fft.ifftshift(kw)

        y = kwds.get('y', 1.0)
        if self.alpha > 0:
            warnings.warn('alpha parameter is not used for binned kde!')

        # Find the binned kernel weights, c.
        c = gridcount(self.dataset, X, y=y)
        # Perform the convolution.
        z = np.real(ifftn(fftn(c, s=kw.shape) * fftn(kw)))

        ix = (slice(0, inc),) * d
        if r == 0:
            return z[ix] * (z[ix] > 0.0)
        return z[ix]

    def _eval_grid(self, *args, **kwds):

        grd = meshgrid(*args)
        shape0 = grd[0].shape
        d = len(grd)
        for i in range(d):
            grd[i] = grd[i].ravel()
        f = self.eval_points(np.vstack(grd), **kwds)
        return f.reshape(shape0)

    def _moment_fun(self, r):
        if r == 0:
            return lambda x: 1
        return lambda x: (x ** r).sum(axis=0)

    @property
    def norm_factor(self):
        return self._norm_factor * self.kernel.norm_factor(self.d, self.n)

    def _loop_over_data(self, data, points, y, r):
        fun = self._moment_fun(r)
        d, m = points.shape
        inv_hs, lambda_ = self._inv_hs, self._lambda
        kernel = self.kernel

        y_d_lambda = y / lambda_ ** d
        result = np.zeros((m,))
        for i in range(self.n):
            dxi = points - data[:, i, np.newaxis]
            tdiff = np.dot(inv_hs / lambda_[i], dxi)
            result += fun(dxi) * kernel(tdiff) * y_d_lambda[i]
        return result / self.norm_factor

    def _loop_over_points(self, data, points, y, r):
        fun = self._moment_fun(r)
        d, m = points.shape
        inv_hs, lambda_ = self._inv_hs, self._lambda
        kernel = self.kernel

        y_d_lambda = y / lambda_ ** d
        result = np.zeros((m,))
        for i in range(m):
            dxi = points[:, i, np.newaxis] - data
            tdiff = np.dot(inv_hs, dxi / lambda_[np.newaxis, :])
            result[i] = np.sum(fun(dxi) * kernel(tdiff) * y_d_lambda, axis=-1)
        return result / self.norm_factor

    def _eval_points(self, points, **kwds):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError if the dimensionality of the input points is different than
        the dimensionality of the KDE.

        """
        d, m = points.shape
        _assert(d == self.d, "d={} expected, got {}".format(self.d, d))

        y = kwds.get('y', 1)
        r = kwds.get('r', 0)

        more_points_than_data = m >= self.n
        if more_points_than_data:
            return self._loop_over_data(self.dataset, points, y, r)
        return self._loop_over_points(self.dataset, points, y, r)


class KRegression(object):

    """ Kernel-Regression

    Parameters
    ----------
    data : (# of dims, # of data)-array
        datapoints to estimate from
    y : # of data - array
        response variable
    p : scalar integer (0 or 1)
        Nadaraya-Watson estimator if p=0,
        local linear estimator if p=1.
    hs : array-like (optional)
        smooting parameter vector/matrix.
        (default compute from data using kernel.get_smoothing function)
    kernel :  kernel function object.
        kernel must have get_smoothing method
    alpha : real scalar (optional)
        sensitivity parameter               (default 0 regular KDE)
        A good choice might be alpha = 0.5 ( or 1/D)
        alpha = 0      Regular  KDE (hs is constant)
        0 < alpha <= 1 Adaptive KDE (Make hs change)
    xmin, xmax  : vectors
        specifying the default argument range for the kde.eval_grid methods.
        For the kde.eval_grid_fast methods the values must cover the range of
        the data. (default min(data)-range(data)/4, max(data)-range(data)/4)
        If a single value of xmin or xmax is given then the boundary is the is
        the same for all dimensions.
    inc :  scalar integer   (default 128)
        defining the default dimension of the output from kde.eval_grid methods
        (For kde.eval_grid_fast: A value below 50 is very fast to compute but
        may give some inaccuracies. Values between 100 and 500 give very
        accurate results)

    Members
    -------
    d : int
        number of dimensions
    n : int
        number of datapoints

    Methods
    -------
    kde.eval_grid_fast(x0, x1,..., xd) : array
        evaluate the estimated pdf on meshgrid(x0, x1,..., xd)
    kde.eval_grid(x0, x1,..., xd) : array
        evaluate the estimated pdf on meshgrid(x0, x1,..., xd)
    kde.eval_points(points) : array
        evaluate the estimated pdf on a provided set of points
    kde(x0, x1,..., xd) : array
        same as kde.eval_grid(x0, x1,..., xd)


    Example
    -------
    >>> import wafo.kdetools as wk
    >>> N = 100
    >>> x = np.linspace(0, 1, N)
    >>> ei = np.random.normal(loc=0, scale=0.075, size=(N,))
    >>> ei = np.sqrt(0.075) * np.sin(100*x)

    >>> y = 2*np.exp(-x**2/(2*0.3**2))+3*np.exp(-(x-1)**2/(2*0.7**2)) + ei
    >>> kreg = wk.KRegression(x, y)
    >>> f = kreg(output='plotobj', title='Kernel regression', plotflag=1)
    >>> np.allclose(f.data[:5],
    ...     [ 3.18670593,  3.18678088,  3.18682196,  3.18682932,  3.18680337])
    True

    h = f.plot(label='p=0')
    """

    def __init__(self, data, y, p=0, hs=None, kernel=None, alpha=0.0,
                 xmin=None, xmax=None, inc=128, L2=None):

        self.tkde = TKDE(data, hs=hs, kernel=kernel,
                         alpha=alpha, xmin=xmin, xmax=xmax, inc=inc, L2=L2)
        self.y = np.atleast_1d(y)
        self.p = p

    def eval_grid_fast(self, *args, **kwds):
        self._grdfun = self.tkde.eval_grid_fast
        return self.tkde.eval_grid_fun(self._eval_gridfun, *args, **kwds)

    def eval_grid(self, *args, **kwds):
        self._grdfun = self.tkde.eval_grid
        return self.tkde.eval_grid_fun(self._eval_gridfun, *args, **kwds)

    def _eval_gridfun(self, *args, **kwds):
        grdfun = self._grdfun
        s0 = grdfun(*args, r=0)
        t0 = grdfun(*args, r=0, y=self.y)
        if self.p == 0:
            return (t0 / (s0 + _TINY)).clip(min=-_REALMAX, max=_REALMAX)
        elif self.p == 1:
            s1 = grdfun(*args, r=1)
            s2 = grdfun(*args, r=2)
            t1 = grdfun(*args, r=1, y=self.y)
            return ((s2 * t0 - s1 * t1) /
                    (s2 * s0 - s1 ** 2)).clip(min=-_REALMAX, max=_REALMAX)
    __call__ = eval_grid_fast


class BKRegression(object):

    '''
    Kernel-Regression on binomial data

    method : {'beta', 'wilson'}
        method is one of the following
        'beta', return Bayesian Credible interval using beta-distribution.
        'wilson', return Wilson score interval
    a, b : scalars
        parameters of the beta distribution defining the apriori distribution
        of p, i.e., the Bayes estimator for p: p = (y+a)/(n+a+b).
        Setting a=b=0.5 gives Jeffreys interval.
    '''

    def __init__(self, data, y, method='beta', a=0.05, b=0.05, p=0, hs_e=None,
                 hs=None, kernel=None, alpha=0.0, xmin=None, xmax=None,
                 inc=128, L2=None):
        self.method = method
        self.a = max(a, _TINY)
        self.b = max(b, _TINY)
        self.kreg = KRegression(data, y, p=p, hs=hs, kernel=kernel,
                                alpha=alpha, xmin=xmin, xmax=xmax, inc=inc,
                                L2=L2)
        # defines bin width (i.e. smoothing) in empirical estimate
        self.hs_e = hs_e

    @property
    def hs_e(self):
        return self._hs_e

    @hs_e.setter
    def hs_e(self, hs_e):
        if hs_e is None:
            hs1 = self._get_max_smoothing('hste')[0]
            hs2 = self._get_max_smoothing('hos')[0]
            hs_e = sqrt(hs1 * hs2)
        self._hs_e = hs_e

    def _set_smoothing(self, hs):
        self.kreg.tkde.hs = hs

    x = property(fget=lambda cls: cls.kreg.tkde.dataset.squeeze())
    y = property(fget=lambda cls: cls.kreg.y)
    kernel = property(fget=lambda cls: cls.kreg.tkde.kernel)
    hs = property(fset=_set_smoothing, fget=lambda cls: cls.kreg.tkde.hs)

    def _get_max_smoothing(self, fun=None):
        """Return maximum value for smoothing parameter."""
        x = self.x
        y = self.y
        if fun is None:
            get_smoothing = self.kernel.get_smoothing
        else:
            get_smoothing = getattr(self.kernel, fun)

        hs1 = get_smoothing(x)
        # hx = np.median(np.abs(x-np.median(x)))/0.6745*(4.0/(3*n))**0.2
        if (y == 1).any():
            hs2 = get_smoothing(x[y == 1])
            # hy = np.median(np.abs(y-np.mean(y)))/0.6745*(4.0/(3*n))**0.2
        else:
            hs2 = 4 * hs1
            # hy = 4*hx

        hopt = sqrt(hs1 * hs2)
        return hopt, hs1, hs2

    def get_grid(self, hs_e=None):
        if hs_e is None:
            hs_e = self.hs_e
        x = self.x
        xmin, xmax = x.min(), x.max()
        ni = max(2 * int((xmax - xmin) / hs_e) + 3, 5)
        sml = hs_e  # *0.1
        xi = np.linspace(xmin - sml, xmax + sml, ni)
        return xi

    def _wilson_score(self, n, p, alpha):
        # Wilson score
        z0 = -_invnorm(alpha / 2)
        den = 1 + (z0 ** 2. / n)
        xc = (p + (z0 ** 2) / (2 * n)) / den
        halfwidth = (z0 * sqrt((p * (1 - p) / n) +
                               (z0 ** 2 / (4 * (n ** 2))))) / den
        plo = xc - halfwidth.clip(min=0)  # wilson score
        pup = xc + halfwidth.clip(max=1.0)  # wilson score
        return plo, pup

    def _credible_interval(self, n, p, alpha):
        # Jeffreys intervall a=b=0.5
        # st.beta.isf(alpha/2, y+a, n-y+b) y = n*p, n-y = n*(1-p)
        a, b = self.a, self.b
        st = scipy.stats
        pup = np.where(p == 1, 1,
                       st.beta.isf(alpha / 2, n * p + a, n * (1 - p) + b))
        plo = np.where(p == 0, 0,
                       st.beta.isf(1 - alpha / 2, n * p + a, n * (1 - p) + b))
        return plo, pup

    def prb_ci(self, n, p, alpha=0.05):
        """Return Confidence Interval for the binomial probability p.

        Parameters
        ----------
        n : array-like
            number of Bernoulli trials
        p : array-like
            estimated probability of success in each trial
        alpha : scalar
            confidence level
        """
        if self.method.startswith('w'):
            return self._wilson_score(n, p, alpha)
        return self._credible_interval(n, p, alpha)

    def prb_empirical(self, xi=None, hs_e=None, alpha=0.05, color='r', **kwds):
        """Returns empirical binomial probabiltity.

        Parameters
        ----------
        x : ndarray
            position vector
        y : ndarray
            binomial response variable (zeros and ones)
        alpha : scalar
            confidence level
        color:
            used in plot

        Returns
        -------
        P(x) : PlotData object
            empirical probability

        """
        if xi is None:
            xi = self.get_grid(hs_e)

        x = self.x
        y = self.y

        c = gridcount(x, xi)  # + self.a + self.b # count data
        if np.any(y == 1):
            c0 = gridcount(x[y == 1], xi)  # + self.a # count success
        else:
            c0 = np.zeros(np.shape(xi))
        prb = np.where(c == 0, 0, c0 / (c + _TINY))  # assume prb==0 for c==0
        CI = np.vstack(self.prb_ci(c, prb, alpha))

        prb_e = PlotData(prb, xi, plotmethod='plot', plot_args=['.'],
                         plot_kwds=dict(markersize=6, color=color, picker=5))
        prb_e.dataCI = CI.T
        prb_e.count = c
        return prb_e

    def prb_smoothed(self, prb_e, hs, alpha=0.05, color='r', label=''):
        """Return smoothed binomial probability.

        Parameters
        ----------
        prb_e : PlotData object with empirical binomial probabilites
        hs : smoothing parameter
        alpha : confidence level
        color : color of plot object
        label : label for plot object

        """

        x_e = prb_e.args
        n_e = len(x_e)
        dx_e = x_e[1] - x_e[0]
        n = self.x.size

        x_s = np.linspace(x_e[0], x_e[-1], 10 * n_e + 1)
        self.hs = hs

        prb_s = self.kreg(x_s, output='plotobj', title='', plot_kwds=dict(
            color=color, linewidth=2))  # dict(plotflag=7))
        m_nan = np.isnan(prb_s.data)
        if m_nan.any():  # assume 0/0 division
            prb_s.data[m_nan] = 0.0

        # prb_s.data[np.isnan(prb_s.data)] = 0
        # expected number of data in each bin
        c_s = self.kreg.tkde.eval_grid_fast(x_s) * dx_e * n
        plo, pup = self.prb_ci(c_s, prb_s.data, alpha)

        prb_s.dataCI = np.vstack((plo, pup)).T
        prb_s.prediction_error_avg = (np.trapz(pup - plo, x_s) /
                                      (x_s[-1] - x_s[0]))

        if label:
            prb_s.plot_kwds['label'] = label
        prb_s.children = [PlotData([plo, pup], x_s,
                                   plotmethod='fill_between',
                                   plot_kwds=dict(alpha=0.2, color=color)),
                          prb_e]

        p_e = prb_e.eval_points(x_s)
        p_s = prb_s.data
        dp_s = np.sign(np.diff(p_s))
        k = (dp_s[:-1] != dp_s[1:]).sum()  # numpeaks

        sigmai = _logit(pup) - _logit(plo) + _EPS
        aicc = ((((_logit(p_e) - _logit(p_s)) / sigmai) ** 2).sum() +
                2 * k * (k + 1) / np.maximum(n_e - k + 1, 1) +
                np.abs((p_e - pup).clip(min=0) -
                       (p_e - plo).clip(max=0)).sum())

        prb_s.aicc = aicc
        return prb_s

    def prb_search_best(self, prb_e=None, hsvec=None, hsfun='hste',
                        alpha=0.05, color='r', label=''):
        """Return best smoothed binomial probability.

        Parameters
        ----------
        prb_e : PlotData object with empirical binomial probabilites
        hsvec : arraylike  (default np.linspace(hsmax*0.1,hsmax,55))
            vector smoothing parameters
        hsfun :
            method for calculating hsmax

        """
        if prb_e is None:
            prb_e = self.prb_empirical(alpha=alpha, color=color)
        if hsvec is None:
            hsmax = max(self._get_max_smoothing(hsfun)[0], self.hs_e)
            hsvec = np.linspace(hsmax * 0.2, hsmax, 55)

        hs_best = hsvec[-1] + 0.1
        prb_best = self.prb_smoothed(prb_e, hs_best, alpha, color, label)
        aicc = np.zeros(np.size(hsvec))
        for i, hi in enumerate(hsvec):
            f = self.prb_smoothed(prb_e, hi, alpha, color, label)
            aicc[i] = f.aicc
            if f.aicc <= prb_best.aicc:
                prb_best = f
                hs_best = hi
        prb_best.score = PlotData(aicc, hsvec)
        prb_best.hs = hs_best
        self._set_smoothing(hs_best)
        return prb_best


if __name__ == '__main__':
    test_docstrings(__file__)
