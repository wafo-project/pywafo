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
from numpy import pi, sqrt, atleast_2d, exp, meshgrid
from wafo.misc import nextpow2
from wafo.containers import PlotData
from wafo.dctpack import dctn, idctn  # , dstn, idstn
from wafo.plotbackend import plotbackend as plt
from wafo.testing import test_docstrings
from wafo.kdetools.kernels import iqrange, qlevels, Kernel
from wafo.kdetools.gridding import gridcount
import time

try:
    from wafo import fig
except ImportError:
    warnings.warn('fig import only supported on Windows')

__all__ = ['TKDE', 'KDE', 'kde_demo1', 'kde_demo2', 'test_docstrings',
           'KRegression', 'KDEgauss']


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


def _invnorm(q):
    return special.ndtri(q)


class _KDE(object):

    """ Kernel-Density Estimator base class.

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
    """

    def __init__(self, data, hs=None, kernel=None, alpha=0.0, xmin=None,
                 xmax=None, inc=512):
        self.dataset = atleast_2d(data)
        self.kernel = kernel if kernel else Kernel('gauss')
        self.xmin = xmin
        self.xmax = xmax
        self.hs = hs
        self.inc = inc
        self.alpha = alpha
        self.initialize()

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
            self._xmin = self.dataset.min(axis=-1) - 2 * self.sigma
        else:
            self._xmin = xmin * np.ones(self.d)

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, xmax):
        if xmax is None:
            self._xmax = self.dataset.max(axis=-1) + 2 * self.sigma
        else:
            self._xmax = xmax * np.ones(self.d)

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
        inv_hs, deth = self._invert_hs(h)

        self._norm_factor = deth * self.n
        self._inv_hs = inv_hs
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

    def initialize(self):
        if self.n > 1:
            self._initialize()

    def _initialize(self):
        pass

    def get_args(self, xmin=None, xmax=None):
        if xmin is None:
            xmin = self.xmin
        else:
            xmin = [min(i, j) for i, j in zip(xmin, self.xmin)]
        if xmax is None:
            xmax = self.xmax
        else:
            xmax = [max(i, j) for i, j in zip(xmax, self.xmax)]
        args = []
        inc = self.inc
        for i in range(self.d):
            args.append(np.linspace(xmin[i], xmax[i], inc))
        return args

    def eval_grid_fast(self, *args, **kwds):
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
        if len(args) == 0:
            args = self.get_args()
        self.args = args
        return self._eval_grid_fun(self._eval_grid_fast, *args, **kwds)

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
        if len(args) == 0:
            args = self.get_args()
        self.args = args
        return self._eval_grid_fun(self._eval_grid, *args, **kwds)

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

    def _eval_grid_fun(self, eval_grd, *args, **kwds):
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

    def __init__(self, data, hs=None, kernel=None, alpha=0.0, xmin=None,
                 xmax=None, inc=512, L2=None):
        self.L2 = L2
        super(TKDE, self).__init__(data, hs, kernel, alpha, xmin, xmax, inc)

    def _initialize(self):
        self._check_xmin()
        tdataset = self._dat2gaus(self.dataset)
        xmin = self.xmin
        if xmin is not None:
            xmin = self._dat2gaus(np.reshape(xmin, (-1, 1)))
        xmax = self.xmax
        if xmax is not None:
            xmax = self._dat2gaus(np.reshape(xmax, (-1, 1)))
        self.tkde = KDE(tdataset, self.hs, self.kernel, self.alpha, xmin, xmax,
                        self.inc)
        if self.inc is None:
            self.inc = self.tkde.inc

    def _check_xmin(self):
        if self.L2 is not None:
            amin = self.dataset.min(axis=-1)
            # default no transformation
            L2 = np.atleast_1d(self.L2) * np.ones(self.d)
            self.xmin = np.where(L2 != 1,
                                 np.maximum(self.xmin, amin / 100.0),
                                 self.xmin).reshape((-1, 1))

    def _dat2gaus(self, points):
        if self.L2 is None:
            return points  # default no transformation

        # default no transformation
        L2 = np.atleast_1d(self.L2) * np.ones(self.d)

        tpoints = copy.copy(points)
        for i, v2 in enumerate(L2.tolist()):
            tpoints[i] = np.log(points[i]) if v2 == 0 else points[i] ** v2
        return tpoints

    def _gaus2dat(self, tpoints):
        if self.L2 is None:
            return tpoints  # default no transformation

        # default no transformation
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
        if (np.abs(np.diff(pdf)).max() > 10).any():
            msg = ''' Numerical problems may have occured due to the power
                    transformation. Check the KDE for spurious spikes'''
            warnings.warn(msg)
        return pdf

    def eval_grid_fast2(self, *args, **kwds):
        """Evaluate the estimated pdf on a grid.

        Parameters
        ----------
        arg_0,arg_1,... arg_d-1 : vectors
           Alternatively, if no vectors is passed in then
            arg_i = gauss2dat(linspace(dat2gauss(self.xmin[i]),
                                       dat2gauss(self.xmax[i]), self.inc))
        output : string optional
            'value' if value output
            'data' if object output

        Returns
        -------
        values : array-like
           The values evaluated at meshgrid(*args).

        """
        return self._eval_grid_fun(self._eval_grid_fast, *args, **kwds)

    def _interpolate(self, points, f, *args, **kwds):
        ipoints = meshgrid(*args) if self.d > 1 else args
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

    def _eval_grid_fast(self, *args, **kwds):
        if self.L2 is None:
            f = self.tkde.eval_grid_fast(*args, **kwds)
            self.args = self.tkde.args
            return f
        targs = []
        if len(args):
            targs0 = self._dat2gaus(list(args))
            xmin = [min(t) for t in targs0]
            xmax = [max(t) for t in targs0]
            targs = self.tkde.get_args(xmin, xmax)
        tf = self.tkde.eval_grid_fast(*targs)
        self.args = self._gaus2dat(list(self.tkde.args))
        points = meshgrid(*self.args) if self.d > 1 else self.args
        f = self._scale_pdf(tf, points)
        if len(args):
            return self._interpolate(points, f, *args, **kwds)
        return f

    def _eval_grid(self, *args, **kwds):
        if self.L2 is None:
            return self.tkde.eval_grid(*args, **kwds)
        targs = self._dat2gaus(list(args))
        tf = self.tkde.eval_grid(*targs, **kwds)
        points = meshgrid(*args) if self.d > 1 else list(args)
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

        tpoints = self._dat2gaus(points)
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
    array([ 0.21720891,  0.43308789,  0.59017626,  0.55847998,  0.39681482,
            0.23987473,  0.13113066,  0.06062029,  0.02160104,  0.00559028])

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

    def _eval_grid_fast(self, *args, **kwds):
        X = np.vstack(args)
        d, inc = X.shape
        dx = X[:, 1] - X[:, 0]

        Xn = []
        nfft0 = 2 * inc
        nfft = (nfft0,) * d
        x0 = np.linspace(-inc, inc, nfft0 + 1)
        for i in range(d):
            Xn.append(x0[:-1] * dx[i])

        Xnc = meshgrid(*Xn)  # if d > 1 else Xn

        shape0 = Xnc[0].shape
        for i in range(d):
            Xnc[i].shape = (-1,)

        Xn = np.dot(self._inv_hs, np.vstack(Xnc))

        # Obtain the kernel weights.
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
        r = kwds.get('r', 0)
        if r != 0:
            kw *= np.vstack(Xnc) ** r if d > 1 else Xnc[0] ** r
        kw.shape = shape0
        kw = np.fft.ifftshift(kw)
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn

        y = kwds.get('y', 1.0)
        if self.alpha > 0:
            y = y / self._lambda**d

        # Find the binned kernel weights, c.
        c = gridcount(self.dataset, X, y=y)
        # Perform the convolution.
        z = np.real(ifftn(fftn(c, s=nfft) * fftn(kw)))
#        opt = dict(type=1, norm=None)
#        z = idctn(dctn(c, shape=(inc,)*d, **opt) * dctn(kw[:inc], **opt),
#                  **opt)/(inc-1)/2
#         # if r is odd
#         op2 = dict(type=3, norm=None)
#         z3 = idstn(dctn(c, shape=(inc,)*d, **op2) * dstn(kw[1:inc+1], **op2),
#                    **op2)/(inc-1)/2

        ix = (slice(0, inc),) * d
        if r == 0:
            return z[ix] * (z[ix] > 0.0)
        return z[ix]

    def _eval_grid(self, *args, **kwds):

        grd = meshgrid(*args) if len(args) > 1 else list(args)
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


class KDEgauss(KDE):

    """ Kernel-Density Estimator base class.

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
        If a single value of xmin or xmax is given, then the boundary is the
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
    kde(x0, x1,..., xd) : array
        same as kde.eval_grid_fast(x0, x1,..., xd)
    """
    def _eval_grid_fast(self, *args, **kwds):
        X = np.vstack(args)
        d, inc = X.shape
        # dx = X[:, 1] - X[:, 0]
        R = X.max(axis=-1) - X.min(axis=-1)

        t_star = (self.hs / R) ** 2
        I = (np.asfarray(np.arange(0, inc)) * pi) ** 2
        In = []
        for i in range(d):
            In.append(I * t_star[i] * 0.5)

        r = kwds.get('r', 0)
        fun = self._moment_fun(r)

        Inc = meshgrid(*In) if d > 1 else In
        kw = np.zeros((inc,) * d)
        for i in range(d):
            kw += exp(-Inc[i]) * fun(Inc[i])

        y = kwds.get('y', 1.0)
        d, n = self.dataset.shape
        # Find the binned kernel weights, c.
        c = gridcount(self.dataset, X, y=y)
        # Perform the convolution.
        at = dctn(c) * kw / n
        z = idctn(at) * (at.size-1) / np.prod(R)
        return z * (z > 0.0)

    __call__ = _KDE.eval_grid_fast


class KRegression(_KDE):

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
        self.y = y
        self.p = p

    def eval_grid_fast(self, *args, **kwds):
        self._grdfun = self.tkde.eval_grid_fast
        return self.tkde._eval_grid_fun(self._eval_gridfun, *args, **kwds)

    def eval_grid(self, *args, **kwds):
        self._grdfun = self.tkde.eval_grid
        return self.tkde._eval_grid_fun(self._eval_gridfun, *args, **kwds)

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

    def __init__(self, *args, **kwds):
        self.method = kwds.pop('method', 'beta')
        self.a = max(kwds.pop('a', 0.5), _TINY)
        self.b = max(kwds.pop('b', 0.5), _TINY)
        self.kreg = KRegression(*args, **kwds)
        # defines bin width (i.e. smoothing) in empirical estimate
        self.hs_e = None
#        self.x = self.kreg.tkde.dataset
#        self.y = self.kreg.y

    def _set_smoothing(self, hs):
        self.kreg.tkde.hs = hs
        self.kreg.tkde.initialize()

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
            if self.hs_e is None:
                hs1 = self._get_max_smoothing('hste')[0]
                hs2 = self._get_max_smoothing('hos')[0]
                self.hs_e = sqrt(hs1 * hs2)
            hs_e = self.hs_e
        x = self.x
        xmin, xmax = x.min(), x.max()
        ni = max(2 * int((xmax - xmin) / hs_e) + 3, 5)
        sml = hs_e  # *0.1
        xi = np.linspace(xmin - sml, xmax + sml, ni)
        return xi

    def prb_ci(self, n, p, alpha=0.05, **kwds):
        """Return Confidence Interval for the binomial probability p.

        Parameters
        ----------
        n : array-like
            number of Bernoulli trials
        p : array-like
            estimated probability of success in each trial
        alpha : scalar
            confidence level
        method : {'beta', 'wilson'}
            method is one of the following
            'beta', return Bayesian Credible interval using beta-distribution.
            'wilson', return Wilson score interval
        a, b : scalars
            parameters of the beta distribution defining the apriori
            distribution of p, i.e.,
            the Bayes estimator for p: p = (y+a)/(n+a+b).
            Setting a=b=0.5 gives Jeffreys interval.

        """
        if self.method.startswith('w'):
            # Wilson score
            z0 = -_invnorm(alpha / 2)
            den = 1 + (z0 ** 2. / n)
            xc = (p + (z0 ** 2) / (2 * n)) / den
            halfwidth = (z0 * sqrt((p * (1 - p) / n) +
                                   (z0 ** 2 / (4 * (n ** 2))))) / den
            plo = (xc - halfwidth).clip(min=0)  # wilson score
            pup = (xc + halfwidth).clip(max=1.0)  # wilson score
        else:
            # Jeffreys intervall a=b=0.5
            # st.beta.isf(alpha/2, y+a, n-y+b) y = n*p, n-y = n*(1-p)
            a = self.a
            b = self.b
            st = scipy.stats
            pup = np.where(p == 1, 1,
                           st.beta.isf(alpha / 2, n * p + a, n * (1 - p) + b))
            plo = np.where(p == 0, 0,
                           st.beta.isf(1 - alpha / 2,
                                       n * p + a, n * (1 - p) + b))
        return plo, pup

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
        if (y == 1).any():
            c0 = gridcount(x[y == 1], xi)  # + self.a # count success
        else:
            c0 = np.zeros(np.shape(xi))
        prb = np.where(c == 0, 0, c0 / (c + _TINY))  # assume prb==0 for c==0
        CI = np.vstack(self.prb_ci(c, prb, alpha, **kwds))

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
        prb_s.prediction_error_avg = np.trapz(
            pup - plo, x_s) / (x_s[-1] - x_s[0])

        if label:
            prb_s.plot_kwds['label'] = label
        prb_s.children = [PlotData([plo, pup], x_s,
                                   plotmethod='fill_between',
                                   plot_kwds=dict(alpha=0.2, color=color)),
                          prb_e]

        # empirical oversmooths the data
#        p_s = prb_s.eval_points(self.x)
#        dp_s = np.diff(prb_s.data)
# k = (dp_s[:-1]*dp_s[1:]<0).sum() # numpeaks
#        p_e = self.y
#        n_s = interpolate.interp1d(x_s, c_s)(self.x)
#        plo, pup = self.prb_ci(n_s, p_s, alpha)
#        sigmai = (pup-plo)
#        aicc = (((p_e-p_s)/sigmai)**2).sum()+ 2*k*(k+1)/np.maximum(n-k+1,1)

        p_e = prb_e.eval_points(x_s)
        p_s = prb_s.data
        dp_s = np.sign(np.diff(p_s))
        k = (dp_s[:-1] != dp_s[1:]).sum()  # numpeaks

        # sigmai = (pup-plo)+_EPS
        # aicc = (((p_e-p_s)/sigmai)**2).sum()+ 2*k*(k+1)/np.maximum(n_e-k+1,1)
        # + np.abs((p_e-pup).clip(min=0)-(p_e-plo).clip(max=0)).sum()
        sigmai = _logit(pup) - _logit(plo) + _EPS
        aicc = ((((_logit(p_e) - _logit(p_s)) / sigmai) ** 2).sum() +
                2 * k * (k + 1) / np.maximum(n_e - k + 1, 1) +
                np.abs((p_e - pup).clip(min=0) -
                       (p_e - plo).clip(max=0)).sum())

        prb_s.aicc = aicc
        # prb_s.labels.title = ''
        # prb_s.labels.title='perr=%1.3f,aicc=%1.3f, n=%d, hs=%1.3f' %
        # (prb_s.prediction_error_avg,aicc,n,hs)

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
            prb_e = self.prb_empirical(
                hs_e=self.hs_e, alpha=alpha, color=color)
        if hsvec is None:
            hsmax = self._get_max_smoothing(hsfun)[0]  # @UnusedVariable
            hsmax = max(hsmax, self.hs_e)
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


def kde_demo1():
    """KDEDEMO1 Demonstrate the smoothing parameter impact on KDE.

    KDEDEMO1 shows the true density (dotted) compared to KDE based on 7
    observations (solid) and their individual kernels (dashed) for 3
    different values of the smoothing parameter, hs.

    """
    st = scipy.stats
    x = np.linspace(-4, 4, 101)
    x0 = x / 2.0
    data = np.random.normal(loc=0, scale=1.0, size=7)
    kernel = Kernel('gauss')
    hs = kernel.hns(data)
    hVec = [hs / 2, hs, 2 * hs]

    for ix, h in enumerate(hVec):
        plt.figure(ix)
        kde = KDE(data, hs=h, kernel=kernel)
        f2 = kde(x, output='plot', title='h_s = {0:2.2f}'.format(h),
                 ylab='Density')
        f2.plot('k-')

        plt.plot(x, st.norm.pdf(x, 0, 1), 'k:')
        n = len(data)
        plt.plot(data, np.zeros(data.shape), 'bx')
        y = kernel(x0) / (n * h * kernel.norm_factor(d=1, n=n))
        for i in range(n):
            plt.plot(data[i] + x0 * h, y, 'b--')
            plt.plot([data[i], data[i]], [0, np.max(y)], 'b')

        plt.axis([min(x), max(x), 0, 0.5])


def kde_demo2():
    '''Demonstrate the difference between transformation- and ordinary-KDE.

    KDEDEMO2 shows that the transformation KDE is a better estimate for
    Rayleigh distributed data around 0 than the ordinary KDE.
    '''
    st = scipy.stats
    data = st.rayleigh.rvs(scale=1, size=300)

    x = np.linspace(1.5e-2, 5, 55)

    kde = KDE(data)
    f = kde(output='plot', title='Ordinary KDE (hs={0:g})'.format(kde.hs))
    plt.figure(0)
    f.plot()

    plt.plot(x, st.rayleigh.pdf(x, scale=1), ':')

    # plotnorm((data).^(L2))  # gives a straight line => L2 = 0.5 reasonable

    tkde = TKDE(data, L2=0.5)
    ft = tkde(x, output='plot',
              title='Transformation KDE (hs={0:g})'.format(tkde.tkde.hs))
    plt.figure(1)
    ft.plot()

    plt.plot(x, st.rayleigh.pdf(x, scale=1), ':')

    plt.figure(0)


def kde_demo3():
    '''Demonstrate the difference between transformation and ordinary-KDE in 2D

    KDEDEMO3 shows that the transformation KDE is a better estimate for
    Rayleigh distributed data around 0 than the ordinary KDE.
    '''
    st = scipy.stats
    data = st.rayleigh.rvs(scale=1, size=(2, 300))

    # x = np.linspace(1.5e-3, 5, 55)

    kde = KDE(data)
    f = kde(output='plot', title='Ordinary KDE', plotflag=1)
    plt.figure(0)
    f.plot()

    plt.plot(data[0], data[1], '.')

    # plotnorm((data).^(L2)) % gives a straight line => L2 = 0.5 reasonable

    tkde = TKDE(data, L2=0.5)
    ft = tkde.eval_grid_fast(
        output='plot', title='Transformation KDE', plotflag=1)

    plt.figure(1)
    ft.plot()

    plt.plot(data[0], data[1], '.')

    plt.figure(0)


def kde_demo4(N=50):
    '''Demonstrate that the improved Sheather-Jones plug-in (hisj) is superior
       for 1D multimodal distributions

    KDEDEMO4 shows that the improved Sheather-Jones plug-in smoothing is a
    better compared to normal reference rules (in this case the hns)
    '''
    st = scipy.stats

    data = np.hstack((st.norm.rvs(loc=5, scale=1, size=(N,)),
                      st.norm.rvs(loc=-5, scale=1, size=(N,))))

    # x = np.linspace(1.5e-3, 5, 55)

    kde = KDE(data, kernel=Kernel('gauss', 'hns'))
    f = kde(output='plot', title='Ordinary KDE', plotflag=1)

    kde1 = KDE(data, kernel=Kernel('gauss', 'hisj'))
    f1 = kde1(output='plot', label='Ordinary KDE', plotflag=1)

    plt.figure(0)
    f.plot('r', label='hns={0:g}'.format(kde.hs))
    # plt.figure(2)
    f1.plot('b', label='hisj={0:g}'.format(kde1.hs))
    x = np.linspace(-4, 4)
    for loc in [-5, 5]:
        plt.plot(x + loc, st.norm.pdf(x, 0, scale=1) / 2, 'k:',
                 label='True density')
    plt.legend()


def kde_demo5(N=500):
    '''Demonstrate that the improved Sheather-Jones plug-in (hisj) is superior
       for 2D multimodal distributions

    KDEDEMO5 shows that the improved Sheather-Jones plug-in smoothing is better
    compared to normal reference rules (in this case the hns)
    '''
    st = scipy.stats

    data = np.hstack((st.norm.rvs(loc=5, scale=1, size=(2, N,)),
                      st.norm.rvs(loc=-5, scale=1, size=(2, N,))))
    kde = KDE(data, kernel=Kernel('gauss', 'hns'))
    f = kde(output='plot', plotflag=1,
            title='Ordinary KDE (hns={0:s}'.format(str(kde.hs.tolist())))

    kde1 = KDE(data, kernel=Kernel('gauss', 'hisj'))
    f1 = kde1(output='plot', plotflag=1,
              title='Ordinary KDE (hisj={0:s})'.format(str(kde1.hs.tolist())))

    plt.figure(0)
    plt.clf()
    f.plot()
    plt.plot(data[0], data[1], '.')
    plt.figure(1)
    plt.clf()
    f1.plot()
    plt.plot(data[0], data[1], '.')


def kreg_demo1(hs=None, fast=False, fun='hisj'):
    """"""
    N = 100
    # ei = np.random.normal(loc=0, scale=0.075, size=(N,))
    ei = np.array([
        -0.08508516, 0.10462496, 0.07694448, -0.03080661, 0.05777525,
        0.06096313, -0.16572389, 0.01838912, -0.06251845, -0.09186784,
        -0.04304887, -0.13365788, -0.0185279, -0.07289167, 0.02319097,
        0.06887854, -0.08938374, -0.15181813, 0.03307712, 0.08523183,
        -0.0378058, -0.06312874, 0.01485772, 0.06307944, -0.0632959,
        0.18963205, 0.0369126, -0.01485447, 0.04037722, 0.0085057,
        -0.06912903, 0.02073998, 0.1174351, 0.17599277, -0.06842139,
        0.12587608, 0.07698113, -0.0032394, -0.12045792, -0.03132877,
        0.05047314, 0.02013453, 0.04080741, 0.00158392, 0.10237899,
        -0.09069682, 0.09242174, -0.15445323, 0.09190278, 0.07138498,
        0.03002497, 0.02495252, 0.01286942, 0.06449978, 0.03031802,
        0.11754861, -0.02322272, 0.00455867, -0.02132251, 0.09119446,
        -0.03210086, -0.06509545, 0.07306443, 0.04330647, 0.078111,
        -0.04146907, 0.05705476, 0.02492201, -0.03200572, -0.02859788,
        -0.05893749, 0.00089538, 0.0432551, 0.04001474, 0.04888828,
        -0.17708392, 0.16478644, 0.1171006, 0.11664846, 0.01410477,
        -0.12458953, -0.11692081, 0.0413047, -0.09292439, -0.07042327,
        0.14119701, -0.05114335, 0.04994696, -0.09520663, 0.04829406,
        -0.01603065, -0.1933216, 0.19352763, 0.11819496, 0.04567619,
        -0.08348306, 0.00812816, -0.00908206, 0.14528945, 0.02901065])
    x = np.linspace(0, 1, N)

    y0 = 2 * np.exp(-x ** 2 / (2 * 0.3 ** 2)) + \
        3 * np.exp(-(x - 1) ** 2 / (2 * 0.7 ** 2))
    y = y0 + ei
    kernel = Kernel('gauss', fun=fun)
    hopt = kernel.hisj(x)
    kreg = KRegression(
        x, y, p=0, hs=hs, kernel=kernel, xmin=-2 * hopt, xmax=1 + 2 * hopt)
    if fast:
        kreg.__call__ = kreg.eval_grid_fast

    f = kreg(output='plot', title='Kernel regression', plotflag=1)
    plt.figure(0)
    f.plot(label='p=0')

    kreg.p = 1
    f1 = kreg(output='plot', title='Kernel regression', plotflag=1)
    f1.plot(label='p=1')
    # print(f1.data)
    plt.plot(x, y, '.', label='data')
    plt.plot(x, y0, 'k', label='True model')
    plt.legend()

    plt.show()

    print(kreg.tkde.tkde._inv_hs)
    print(kreg.tkde.tkde.hs)

_TINY = np.finfo(float).machar.tiny
_REALMIN = np.finfo(float).machar.xmin
_REALMAX = np.finfo(float).machar.xmax
_EPS = np.finfo(float).eps


def _logit(p):
    pc = p.clip(min=0, max=1)
    return (np.log(pc) - np.log1p(-pc)).clip(min=-40, max=40)


def _logitinv(x):
    return 1.0 / (np.exp(-x) + 1)


def _get_data(n=100, symmetric=False, loc1=1.1, scale1=0.6, scale2=1.0):
    st = scipy.stats
    # from sg_filter import SavitzkyGolay
    dist = st.norm

    norm1 = scale2 * (dist.pdf(-loc1, loc=-loc1, scale=scale1) +
                      dist.pdf(-loc1, loc=loc1, scale=scale1))

    def fun1(x):
        return ((dist.pdf(x, loc=-loc1, scale=scale1) +
                 dist.pdf(x, loc=loc1, scale=scale1)) / norm1).clip(max=1.0)

    x = np.sort(6 * np.random.rand(n, 1) - 3, axis=0)

    y = (fun1(x) > np.random.rand(n, 1)).ravel()
    # y = (np.cos(x)>2*np.random.rand(n, 1)-1).ravel()
    x = x.ravel()

    if symmetric:
        xi = np.hstack((x.ravel(), -x.ravel()))
        yi = np.hstack((y, y))
        i = np.argsort(xi)
        x = xi[i]
        y = yi[i]
    return x, y, fun1


def kreg_demo2(n=100, hs=None, symmetric=False, fun='hisj', plotlog=False):
    x, y, fun1 = _get_data(n, symmetric)
    kreg_demo3(x, y, fun1, hs=None, fun='hisj', plotlog=False)


def kreg_demo3(x, y, fun1, hs=None, fun='hisj', plotlog=False):
    st = scipy.stats

    alpha = 0.1
    z0 = -_invnorm(alpha / 2)

    n = x.size
    hopt, hs1, hs2 = _get_regression_smooting(x, y, fun='hos')
    if hs is None:
        hs = hopt

    forward = _logit
    reverse = _logitinv
    # forward = np.log
    # reverse = np.exp

    xmin, xmax = x.min(), x.max()
    ni = max(2 * int((xmax - xmin) / hopt) + 3, 5)
    print(ni)
    print(xmin, xmax)
    sml = hopt * 0.1
    xi = np.linspace(xmin - sml, xmax + sml, ni)
    xiii = np.linspace(xmin - sml, xmax + sml, 4 * ni + 1)

    c = gridcount(x, xi)
    if (y == 1).any():
        c0 = gridcount(x[y == 1], xi)
    else:
        c0 = np.zeros(np.shape(xi))
    yi = np.where(c == 0, 0, c0 / c)

    kreg = KRegression(x, y, hs=hs, p=0)
    fiii = kreg(xiii)
    yiii = interpolate.interp1d(xi, yi)(xiii)
    fit = fun1(xiii).clip(max=1.0)
    df = np.diff(fiii)
    eerr = np.abs((yiii - fiii)).std() + 0.5 * (df[:-1] * df[1:] < 0).sum() / n
    err = (fiii - fit).std()
    msg = '{} err={1:1.3f},eerr={2:1.3f}, n={:d}, hs={:1.3f}, hs1={:1.3f}, '\
        'hs2={:1.3f}'
    title = (msg.format(fun, err, eerr, n, hs, hs1, hs2))
    f = kreg(xiii, output='plotobj', title=title, plotflag=1)

    # yi[yi==0] = 1.0/(c[c!=0].min()+4)
    # yi[yi==1] = 1-1.0/(c[c!=0].min()+4)
    # yi[yi==0] = fi[yi==0]
    # yi[yi==0] = np.exp(stineman_interp(xi[yi==0], xi[yi>0],np.log(yi[yi>0])))
    # yi[yi==0] = fun1(xi[yi==0])
    try:
        yi[yi == 0] = yi[yi > 0].min() / sqrt(n)
    except:
        yi[yi == 0] = 1. / n
    yi[yi == 1] = 1 - (1 - yi[yi < 1].max()) / sqrt(n)

    logity = forward(yi)

    gkreg = KRegression(xi, logity, hs=hs, xmin=xmin - hopt, xmax=xmax + hopt)
    fg = gkreg.eval_grid(
        xi, output='plotobj', title='Kernel regression', plotflag=1)
    sa = (fg.data - logity).std()
    sa2 = iqrange(fg.data - logity) / 1.349
    # print('sa=%g %g' % (sa, sa2))
    sa = min(sa, sa2)

#    plt.figure(1)
#    plt.plot(xi, slogity-logity,'r.')
# plt.plot(xi, logity-,'b.')
#    plt.plot(xi, fg.data-logity, 'b.')
#    plt.show()
#    return

    fg = gkreg.eval_grid(
        xiii, output='plotobj', title='Kernel regression', plotflag=1)
    pi = reverse(fg.data)

    dx = xi[1] - xi[0]
    ckreg = KDE(x, hs=hs)
    # ci = ckreg.eval_grid_fast(xi)*n*dx
    ciii = ckreg.eval_grid_fast(xiii) * dx * x.size  # n*(1+symmetric)

#    sa1 = np.sqrt(1./(ciii*pi*(1-pi)))
#    plo3 = reverse(fg.data-z0*sa)
#    pup3 = reverse(fg.data+z0*sa)
    fg.data = pi
    pi = f.data

    # ref Casella and Berger (1990) "Statistical inference" pp444
#    a = 2*pi + z0**2/(ciii+1e-16)
#    b = 2*(1+z0**2/(ciii+1e-16))
#    plo2 = ((a-sqrt(a**2-2*pi**2*b))/b).clip(min=0,max=1)
#    pup2 = ((a+sqrt(a**2-2*pi**2*b))/b).clip(min=0,max=1)
    # Jeffreys intervall a=b=0.5
    # st.beta.isf(alpha/2, x+a, n-x+b)
    ab = 0.07  # 0.055
    pi1 = pi  # fun1(xiii)
    pup2 = np.where(pi == 1,
                    1,
                    st.beta.isf(alpha / 2,
                                ciii * pi1 + ab,
                                ciii * (1 - pi1) + ab))
    plo2 = np.where(pi == 0,
                    0,
                    st.beta.isf(1 - alpha / 2,
                                ciii * pi1 + ab,
                                ciii * (1 - pi1) + ab))

    averr = np.trapz(pup2 - plo2, xiii) / \
        (xiii[-1] - xiii[0]) + 0.5 * (df[:-1] * df[1:] < 0).sum()

    # f2 = kreg_demo4(x, y, hs, hopt)
    # Wilson score
    den = 1 + (z0 ** 2. / ciii)
    xc = (pi1 + (z0 ** 2) / (2 * ciii)) / den
    halfwidth = (z0 * sqrt((pi1 * (1 - pi1) / ciii) +
                           (z0 ** 2 / (4 * (ciii ** 2))))) / den
    plo = (xc - halfwidth).clip(min=0)  # wilson score
    pup = (xc + halfwidth).clip(max=1.0)  # wilson score
    # pup = (pi + z0*np.sqrt(pi*(1-pi)/ciii)).clip(min=0,max=1) # dont use
    # plo = (pi - z0*np.sqrt(pi*(1-pi)/ciii)).clip(min=0,max=1)

    # mi = kreg.eval_grid(x)
    # sigma = (stineman_interp(x, xiii, pup)-stineman_interp(x, xiii, plo))/4
    # aic = np.abs((y-mi)/sigma).std()+ 0.5*(df[:-1]*df[1:]<0).sum()/n
    # aic = np.abs((yiii-fiii)/(pup-plo)).std() + \
    #                0.5*(df[:-1]*df[1:]<0).sum() + \
    #            ((yiii-pup).clip(min=0)-(yiii-plo).clip(max=0)).sum()

    k = (df[:-1] * df[1:] < 0).sum()  # numpeaks
    sigmai = (pup - plo)
    aic = (((yiii - fiii) / sigmai) ** 2).sum() + \
        2 * k * (k + 1) / np.maximum(ni - k + 1, 1) + \
        np.abs((yiii - pup).clip(min=0) - (yiii - plo).clip(max=0)).sum()

    # aic = (((yiii-fiii)/sigmai)**2).sum()+ 2*k*(k+1)/(ni-k+1) + \
    #        np.abs((yiii-pup).clip(min=0)-(yiii-plo).clip(max=0)).sum()

    # aic = averr + ((yiii-pup).clip(min=0)-(yiii-plo).clip(max=0)).sum()

    fg.plot(label='KReg grid aic={:2.3f}'.format(aic))
    f.plot(label='KReg averr={:2.3f} '.format(averr))
    labtxt = '%d CI' % (int(100 * (1 - alpha)))
    plt.fill_between(xiii, pup, plo, alpha=0.20,
                     color='r', linestyle='--', label=labtxt)
    plt.fill_between(xiii, pup2, plo2, alpha=0.20, color='b', linestyle=':',
                     label='{:d} CI2'.format(int(100 * (1 - alpha))))
    plt.plot(xiii, fun1(xiii), 'r', label='True model')
    plt.scatter(xi, yi, label='data')
    print('maxp = {:g}'.format(np.nanmax(f.data)))
    print('hs = {:g}'.format(kreg.tkde.tkde.hs))
    plt.legend()
    h = plt.gca()
    if plotlog:
        plt.setp(h, yscale='log')
    # plt.show()
    return hs1, hs2


def kreg_demo4(x, y, hs, hopt, alpha=0.05):
    st = scipy.stats

    n = x.size
    xmin, xmax = x.min(), x.max()
    ni = max(2 * int((xmax - xmin) / hopt) + 3, 5)

    sml = hopt * 0.1
    xi = np.linspace(xmin - sml, xmax + sml, ni)
    xiii = np.linspace(xmin - sml, xmax + sml, 4 * ni + 1)

    kreg = KRegression(x, y, hs=hs, p=0)

    dx = xi[1] - xi[0]
    ciii = kreg.tkde.eval_grid_fast(xiii) * dx * x.size
#    ckreg = KDE(x,hs=hs)
# ciiii = ckreg.eval_grid_fast(xiii)*dx* x.size #n*(1+symmetric)

    f = kreg(xiii, output='plotobj')  # , plot_kwds=dict(plotflag=7))
    pi = f.data

    # Jeffreys intervall a=b=0.5
    # st.beta.isf(alpha/2, x+a, n-x+b)
    ab = 0.07  # 0.5
    pi1 = pi
    pup = np.where(pi1 == 1, 1, st.beta.isf(
        alpha / 2, ciii * pi1 + ab, ciii * (1 - pi1) + ab))
    plo = np.where(pi1 == 0, 0, st.beta.isf(
        1 - alpha / 2, ciii * pi1 + ab, ciii * (1 - pi1) + ab))

    # Wilson score
    # z0 = -_invnorm(alpha/2)
#    den = 1+(z0**2./ciii);
#    xc=(pi1+(z0**2)/(2*ciii))/den;
#    halfwidth=(z0*sqrt((pi1*(1-pi1)/ciii)+(z0**2/(4*(ciii**2)))))/den
# plo2 = (xc-halfwidth).clip(min=0) # wilson score
# pup2 = (xc+halfwidth).clip(max=1.0) # wilson score
    # f.dataCI = np.vstack((plo,pup)).T
    f.prediction_error_avg = np.trapz(pup - plo, xiii) / (xiii[-1] - xiii[0])
    fiii = f.data

    c = gridcount(x, xi)
    if (y == 1).any():
        c0 = gridcount(x[y == 1], xi)
    else:
        c0 = np.zeros(np.shape(xi))
    yi = np.where(c == 0, 0, c0 / c)

    f.children = [PlotData([plo, pup], xiii, plotmethod='fill_between',
                           plot_kwds=dict(alpha=0.2, color='r')),
                  PlotData(yi, xi, plotmethod='scatter',
                           plot_kwds=dict(color='r', s=5))]

    yiii = interpolate.interp1d(xi, yi)(xiii)
    df = np.diff(fiii)
    k = (df[:-1] * df[1:] < 0).sum()  # numpeaks
    sigmai = (pup - plo)
    aicc = (((yiii - fiii) / sigmai) ** 2).sum() + \
        2 * k * (k + 1) / np.maximum(ni - k + 1, 1) + \
        np.abs((yiii - pup).clip(min=0) - (yiii - plo).clip(max=0)).sum()

    f.aicc = aicc
    f.labels.title = ('perr={:1.3f},aicc={:1.3f}, n={:d}, '
                      'hs={:1.3f}'.format(f.prediction_error_avg, aicc, n, hs))

    return f


def check_kreg_demo3():

    plt.ion()
    k = 0
    for n in [50, 100, 300, 600, 4000]:
        x, y, fun1 = _get_data(
            n, symmetric=True, loc1=1.0, scale1=0.6, scale2=1.25)
        k0 = k

        for fun in ['hste', ]:
            hsmax, _hs1, _hs2 = _get_regression_smooting(x, y, fun=fun)
            for hi in np.linspace(hsmax * 0.25, hsmax, 9):
                plt.figure(k)
                k += 1
                unused = kreg_demo3(x, y, fun1, hs=hi, fun=fun, plotlog=False)

            # kreg_demo2(n=n,symmetric=True,fun='hste', plotlog=False)
        fig.tile(range(k0, k))
    plt.ioff()
    plt.show()


def check_kreg_demo4():
    plt.ion()
    # test_docstrings()
    # kde_demo2()
    # kreg_demo1(fast=True)
    # kde_gauss_demo()
    # kreg_demo2(n=120,symmetric=True,fun='hste', plotlog=True)
    k = 0
    for _i, n in enumerate([100, 300, 600, 4000]):
        x, y, fun1 = _get_data(
            n, symmetric=True, loc1=0.1, scale1=0.6, scale2=0.75)
        # k0 = k
        hopt1, _h1, _h2 = _get_regression_smooting(x, y, fun='hos')
        hopt2, _h1, _h2 = _get_regression_smooting(x, y, fun='hste')
        hopt = sqrt(hopt1 * hopt2)
        # hopt = _get_regression_smooting(x,y,fun='hos')[0]
        for _j, fun in enumerate(['hste']):  # , 'hisj', 'hns', 'hstt'
            hsmax, _hs1, _hs2 = _get_regression_smooting(x, y, fun=fun)

            fmax = kreg_demo4(x, y, hsmax + 0.1, hopt)
            for hi in np.linspace(hsmax * 0.1, hsmax, 55):
                f = kreg_demo4(x, y, hi, hopt)
                if f.aicc <= fmax.aicc:
                    fmax = f
            plt.figure(k)
            k += 1
            fmax.plot()
            plt.plot(x, fun1(x), 'r')

            # kreg_demo2(n=n,symmetric=True,fun='hste', plotlog=False)
    fig.tile(range(0, k))
    plt.ioff()
    plt.show()


def check_regression_bin():
    plt.ion()
    # test_docstrings()
    # kde_demo2()
    # kreg_demo1(fast=True)
    # kde_gauss_demo()
    # kreg_demo2(n=120,symmetric=True,fun='hste', plotlog=True)
    k = 0
    for _i, n in enumerate([100, 300, 600, 4000]):
        x, y, fun1 = _get_data(
            n, symmetric=True, loc1=0.1, scale1=0.6, scale2=0.75)
        fbest = regressionbin(x, y, alpha=0.05, color='g', label='Transit_D')

        figk = plt.figure(k)
        ax = figk.gca()
        k += 1
        fbest.labels.title = 'N = {:d}'.format(n)
        fbest.plot(axis=ax)
        ax.plot(x, fun1(x), 'r')
        ax.legend(frameon=False, markerscale=4)
        # ax = plt.gca()
        ax.set_yticklabels(ax.get_yticks() * 100.0)
        ax.grid(True)

    fig.tile(range(0, k))
    plt.ioff()
    plt.show()


def check_bkregression():
    plt.ion()
    k = 0
    for _i, n in enumerate([50, 100, 300, 600]):
        x, y, fun1 = _get_data(
            n, symmetric=True, loc1=0.1, scale1=0.6, scale2=0.75)
        bkreg = BKRegression(x, y)
        fbest = bkreg.prb_search_best(
            hsfun='hste', alpha=0.05, color='g', label='Transit_D')

        figk = plt.figure(k)
        ax = figk.gca()
        k += 1
#        fbest.score.plot(axis=ax)
#        axsize = ax.axis()
#        ax.vlines(fbest.hs,axsize[2]+1,axsize[3])
#        ax.set(yscale='log')
        fbest.labels.title = 'N = {:d}'.format(n)
        fbest.plot(axis=ax)
        ax.plot(x, fun1(x), 'r')
        ax.legend(frameon=False, markerscale=4)
        # ax = plt.gca()
        ax.set_yticklabels(ax.get_yticks() * 100.0)
        ax.grid(True)

    fig.tile(range(0, k))
    plt.ioff()
    plt.show()


def _get_regression_smooting(x, y, fun='hste'):
    hs1 = Kernel('gauss', fun=fun).get_smoothing(x)
    # hx = np.median(np.abs(x-np.median(x)))/0.6745*(4.0/(3*n))**0.2
    if (y == 1).any():
        hs2 = Kernel('gauss', fun=fun).get_smoothing(x[y == 1])
        # hy = np.median(np.abs(y-np.mean(y)))/0.6745*(4.0/(3*n))**0.2
    else:
        hs2 = 4 * hs1
        # hy = 4*hx

    # hy2 = Kernel('gauss', fun=fun).get_smoothing(y)
    # kernel = Kernel('gauss',fun=fun)
    # hopt = (hs1+2*hs2)/3
    # hopt = (hs1+4*hs2)/5 #kernel.get_smoothing(x)
    # hopt = hs2
    hopt = sqrt(hs1 * hs2)
    return hopt, hs1, hs2


def empirical_bin_prb(x, y, hopt, color='r'):
    """Returns empirical binomial probabiltity.

    Parameters
    ----------
    x : ndarray
        position ve
    y : ndarray
        binomial response variable (zeros and ones)

    Returns
    -------
    P(x) : PlotData object
        empirical probability

    """
    xmin, xmax = x.min(), x.max()
    ni = max(2 * int((xmax - xmin) / hopt) + 3, 5)

    sml = hopt  # *0.1
    xi = np.linspace(xmin - sml, xmax + sml, ni)

    c = gridcount(x, xi)
    if (y == 1).any():
        c0 = gridcount(x[y == 1], xi)
    else:
        c0 = np.zeros(np.shape(xi))
    yi = np.where(c == 0, 0, c0 / c)
    return PlotData(yi, xi, plotmethod='scatter',
                    plot_kwds=dict(color=color, s=5))


def smoothed_bin_prb(x, y, hs, hopt, alpha=0.05, color='r', label='',
                     bin_prb=None):
    '''
    Parameters
    ----------
    x,y
    hs : smoothing parameter
    hopt : spacing in empirical_bin_prb
    alpha : confidence level
    color : color of plot object
    bin_prb : PlotData object with empirical bin prb
    '''
    if bin_prb is None:
        bin_prb = empirical_bin_prb(x, y, hopt, color)

    xi = bin_prb.args
    yi = bin_prb.data
    ni = len(xi)
    dxi = xi[1] - xi[0]

    n = x.size

    xiii = np.linspace(xi[0], xi[-1], 10 * ni + 1)

    kreg = KRegression(x, y, hs=hs, p=0)
    # expected number of data in each bin
    ciii = kreg.tkde.eval_grid_fast(xiii) * dxi * n

    f = kreg(xiii, output='plotobj')  # , plot_kwds=dict(plotflag=7))
    pi = f.data

    st = scipy.stats
    # Jeffreys intervall a=b=0.5
    # st.beta.isf(alpha/2, x+a, n-x+b)
    ab = 0.07  # 0.5
    pi1 = pi
    pup = np.where(pi1 == 1, 1, st.beta.isf(
        alpha / 2, ciii * pi1 + ab, ciii * (1 - pi1) + ab))
    plo = np.where(pi1 == 0, 0, st.beta.isf(
        1 - alpha / 2, ciii * pi1 + ab, ciii * (1 - pi1) + ab))

    # Wilson score
    # z0 = -_invnorm(alpha/2)
#    den = 1+(z0**2./ciii);
#    xc=(pi1+(z0**2)/(2*ciii))/den;
#    halfwidth=(z0*sqrt((pi1*(1-pi1)/ciii)+(z0**2/(4*(ciii**2)))))/den
# plo2 = (xc-halfwidth).clip(min=0) # wilson score
# pup2 = (xc+halfwidth).clip(max=1.0) # wilson score
    # f.dataCI = np.vstack((plo,pup)).T
    f.prediction_error_avg = np.trapz(pup - plo, xiii) / (xiii[-1] - xiii[0])
    fiii = f.data

    f.plot_kwds['color'] = color
    f.plot_kwds['linewidth'] = 2
    if label:
        f.plot_kwds['label'] = label
    f.children = [PlotData([plo, pup], xiii, plotmethod='fill_between',
                           plot_kwds=dict(alpha=0.2, color=color)),
                  bin_prb]

    yiii = interpolate.interp1d(xi, yi)(xiii)
    df = np.diff(fiii)
    k = (df[:-1] * df[1:] < 0).sum()  # numpeaks
    sigmai = (pup - plo)
    aicc = (((yiii - fiii) / sigmai) ** 2).sum() + \
        2 * k * (k + 1) / np.maximum(ni - k + 1, 1) + \
        np.abs((yiii - pup).clip(min=0) - (yiii - plo).clip(max=0)).sum()

    f.aicc = aicc
    f.fun = kreg
    f.labels.title = ('perr={:1.3f},aicc={:1.3f}, n={:d}, '
                      'hs={:1.3f}'.format(f.prediction_error_avg, aicc, n, hs))

    return f


def regressionbin(x, y, alpha=0.05, color='r', label=''):
    """Return kernel regression estimate for binomial data.

    Parameters
    ----------
    x : arraylike
        positions
    y : arraylike
        of 0 and 1

    """

    hopt1, _h1, _h2 = _get_regression_smooting(x, y, fun='hos')
    hopt2, _h1, _h2 = _get_regression_smooting(x, y, fun='hste')
    hopt = sqrt(hopt1 * hopt2)

    fbest = smoothed_bin_prb(x, y, hopt2 + 0.1, hopt, alpha, color, label)
    bin_prb = fbest.children[-1]
    for fun in ['hste']:  # , 'hisj', 'hns', 'hstt'
        hsmax, _hs1, _hs2 = _get_regression_smooting(x, y, fun=fun)
        for hi in np.linspace(hsmax * 0.1, hsmax, 55):
            f = smoothed_bin_prb(x, y, hi, hopt, alpha, color, label, bin_prb)
            if f.aicc <= fbest.aicc:
                fbest = f
                # hbest = hi
    return fbest


def kde_gauss_demo(n=50):
    """KDEDEMO Demonstrate the KDEgauss.

    KDEDEMO1 shows the true density (dotted) compared to KDE based on 7
    observations (solid) and their individual kernels (dashed) for 3
    different values of the smoothing parameter, hs.

    """

    st = scipy.stats
    # x = np.linspace(-4, 4, 101)
    # data = np.random.normal(loc=0, scale=1.0, size=n)
    # data = np.random.exponential(scale=1.0, size=n)
#    n1 = 128
#    I = (np.arange(n1)*pi)**2 *0.01*0.5
#    kw = exp(-I)
#    plt.plot(idctn(kw))
#    return
    dist = st.norm
    # dist = st.expon
    data = dist.rvs(loc=0, scale=1.0, size=n)
    d, _N = np.atleast_2d(data).shape

    if d == 1:
        plot_options = [dict(color='red', label='KDE hste'),
                        dict(color='green', label='TKDE hisj'),
                        dict(color='black', label='KDEgauss hste')]
    else:
        plot_options = [dict(colors='red'), dict(colors='green'),
                        dict(colors='black')]

    plt.figure(1)
    t0 = time.time()
    kde0 = KDE(data, kernel=Kernel('gauss', 'hste'))
    f0 = kde0.eval_grid_fast(output='plot', ylab='Density', r=0)
    t1 = time.time()
    total1 = t1-t0

    f0.plot('.', **plot_options[0])
    if dist.name != 'norm':
        kde1 = TKDE(data, kernel=Kernel('gauss', 'hisj'), L2=.5)
        f1 = kde1.eval_grid_fast(output='plot', ylab='Density', r=0)
        f1.plot(**plot_options[1])
    else:
        kde1 = kde0
        f1 = f0
    t1 = time.time()
    kde2 = KDEgauss(data)
    f2 = kde2(output='plot', ylab='Density', r=0)
    t2 = time.time()
    total2 = t2-t1

    x = f2.args
    f2.plot(**plot_options[2])

    fmax = dist.pdf(x, 0, 1).max()
    if d == 1:
        plt.plot(x, dist.pdf(x, 0, 1), 'k:', label='True pdf')
        plt.axis([x.min(), x.max(), 0, fmax])
    plt.legend()
    plt.show()
    print(fmax / f2.data.max())
    try:
        print('hs0={:s} hs1={:s} hs2={:s}'.format(str(kde0.hs.tolist()),
                                                  str(kde1.tkde.hs.tolist()),
                                                  str(kde2.hs.tolist())))
    except:
        pass
    print('inc0 = {:d}, inc1 = {:d}, inc2 = {:d}'.format(kde0.inc, kde1.inc,
                                                         kde2.inc))
    print(np.trapz(f0.data, f0.args), np.trapz(f2.data, f2.args))
    print(total1, total2)


def test_kde():
    data = np.array([
        0.75355792, 0.72779194, 0.94149169, 0.07841119, 2.32291887,
        1.10419995, 0.77055114, 0.60288273, 1.36883635, 1.74754326,
        1.09547561, 1.01671133, 0.73211143, 0.61891719, 0.75903487,
        1.8919469, 0.72433808, 1.92973094, 0.44749838, 1.36508452])

    x = np.linspace(0.01, max(data + 1), 10)
    kde = TKDE(data, hs=0.5, L2=0.5)
    _f = kde(x)
    # f = array([1.03982714, 0.45839018, 0.39514782, 0.32860602, 0.26433318,
    #   0.20717946,  0.15907684,  0.1201074 ,  0.08941027,  0.06574882])

    _f1 = kde.eval_grid(x)
    # array([ 1.03982714,  0.45839018,  0.39514782,  0.32860602,  0.26433318,
    #        0.20717946,  0.15907684,  0.1201074 ,  0.08941027,  0.06574882])

    _f2 = kde.eval_grid_fast(x)
    # array([ 1.06437223,  0.46203314,  0.39593137,  0.32781899,  0.26276433,
    #        0.20532206,  0.15723498,  0.11843998,  0.08797755,  0.        ])


if __name__ == '__main__':
    if True:
        test_docstrings(__file__)
    else:
        # test_kde()
        # check_bkregression()
        # check_regression_bin()
        # check_kreg_demo3()
        # check_kreg_demo4()

        # kde_demo2()
        # kreg_demo1(fast=True)
        kde_gauss_demo(n=50)
        # kreg_demo2(n=120,symmetric=True,fun='hste', plotlog=True)
        plt.show('hold')
