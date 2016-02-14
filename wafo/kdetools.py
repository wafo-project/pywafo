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
#!/usr/bin/env python  # @IgnorePep8
from __future__ import absolute_import, division
import copy
import numpy as np
import scipy
import warnings
from itertools import product
from scipy import interpolate, linalg, optimize, sparse, special, stats
from scipy.special import gamma
from numpy import pi, sqrt, atleast_2d, exp, newaxis  # @UnresolvedImport

from .misc import meshgrid, nextpow2, tranproc  # , trangood
from .containers import PlotData
from .dctpack import dct, dctn, idctn
from .plotbackend import plotbackend as plt
try:
    from . import fig
except ImportError:
    warnings.warn('fig import only supported on Windows')


def _invnorm(q):
    return special.ndtri(q)

_stats_epan = (1. / 5, 3. / 5, np.inf)
_stats_biwe = (1. / 7, 5. / 7, 45. / 2)
_stats_triw = (1. / 9, 350. / 429, np.inf)
_stats_rect = (1. / 3, 1. / 2, np.inf)
_stats_tria = (1. / 6, 2. / 3, np.inf)
_stats_lapl = (2, 1. / 4, np.inf)
_stats_logi = (pi ** 2 / 3, 1. / 6, 1 / 42)
_stats_gaus = (1, 1. / (2 * sqrt(pi)), 3. / (8 * sqrt(pi)))

__all__ = ['sphere_volume', 'TKDE', 'KDE', 'Kernel', 'accum', 'qlevels',
           'iqrange', 'gridcount', 'kde_demo1', 'kde_demo2', 'test_docstrings']


def sphere_volume(d, r=1.0):
    """
    Returns volume of  d-dimensional sphere with radius r

    Parameters
    ----------
    d : scalar or array_like
        dimension of sphere
    r : scalar or array_like
        radius of sphere (default 1)

    Example
    -------
    >>> sphere_volume(2., r=2.)
    12.566370614359172
    >>> sphere_volume(2., r=1.)
    3.1415926535897931

    Reference
    ---------
    Wand,M.P. and Jones, M.C. (1995)
    'Kernel smoothing'
    Chapman and Hall, pp 105
    """
    return (r ** d) * 2.0 * pi ** (d / 2.0) / (d * gamma(d / 2.0))


class KDEgauss(object):

    """ Kernel-Density Estimator base class.

    Parameters
    ----------
    data : (# of dims, # of data)-array
        datapoints to estimate from
    hs : array-like (optional)
        smooting parameter vector/matrix.
        (default compute from data using kernel.get_smoothing function)
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
    kde(x0, x1,..., xd) : array
        same as kde.eval_grid_fast(x0, x1,..., xd)
    """

    def __init__(self, data, hs=None, kernel=None, alpha=0.0, xmin=None,
                 xmax=None, inc=512):
        self.dataset = atleast_2d(data)
        self.hs = hs
        self.kernel = kernel if kernel else Kernel('gauss')
        self.alpha = alpha
        self.xmin = xmin
        self.xmax = xmax
        self.inc = inc
        self.initialize()

    def initialize(self):
        self.d, self.n = self.dataset.shape
        self._set_xlimits()
        self._initialize()

    def _initialize(self):
        self._compute_smoothing()

    def _compute_smoothing(self):
        """Computes the smoothing matrix."""
        get_smoothing = self.kernel.get_smoothing
        h = self.hs
        if h is None:
            h = get_smoothing(self.dataset)
        h = np.atleast_1d(h)
        hsiz = h.shape

        if (len(hsiz) == 1) or (self.d == 1):
            if max(hsiz) == 1:
                h = h * np.ones(self.d)
            else:
                h.shape = (self.d,)  # make sure it has the correct dimension

            # If h negative calculate automatic values
            ind, = np.where(h <= 0)
            for i in ind.tolist():
                h[i] = get_smoothing(self.dataset[i])
            deth = h.prod()
            self.inv_hs = np.diag(1.0 / h)
        else:  # fully general smoothing matrix
            deth = linalg.det(h)
            if deth <= 0:
                raise ValueError(
                    'bandwidth matrix h must be positive definit!')
            self.inv_hs = linalg.inv(h)
        self.hs = h
        self._norm_factor = deth * self.n

    def _set_xlimits(self):
        amin = self.dataset.min(axis=-1)
        amax = self.dataset.max(axis=-1)
        iqr = iqrange(self.dataset, axis=-1)
        sigma = np.minimum(np.std(self.dataset, axis=-1, ddof=1), iqr / 1.34)
        # xyzrange = amax - amin
        # offset = xyzrange / 4.0
        offset = 2 * sigma
        if self.xmin is None:
            self.xmin = amin - offset
        else:
            self.xmin = self.xmin * np.ones((self.d, 1))
        if self.xmax is None:
            self.xmax = amax + offset
        else:
            self.xmax = self.xmax * np.ones((self.d, 1))

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
            args = []
            for i in range(self.d):
                args.append(np.linspace(self.xmin[i], self.xmax[i], self.inc))
        self.args = args
        return self._eval_grid_fun(self._eval_grid_fast, *args, **kwds)

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

        Inc = meshgrid(*In) if d > 1 else In

        kw = np.zeros((inc,) * d)
        for i in range(d):
            kw += exp(-Inc[i])
        y = kwds.get('y', 1.0)
        d, n = self.dataset.shape
        # Find the binned kernel weights, c.
        c = gridcount(self.dataset, X, y=y) / n
        # Perform the convolution.
        at = dctn(c) * kw
        z = idctn(at) * at.size / np.prod(R)
        return z * (z > 0.0)

    def _eval_grid_fun(self, eval_grd, *args, **kwds):
        output = kwds.pop('output', 'value')
        f = eval_grd(*args, **kwds)
        if output == 'value':
            return f
        else:
            titlestr = 'Kernel density estimate (%s)' % self.kernel.name
            kwds2 = dict(title=titlestr)
            kwds2['plot_kwds'] = dict(plotflag=1)
            kwds2.update(**kwds)
            args = self.args
            if self.d == 1:
                args = args[0]
            wdata = PlotData(f, args, **kwds2)
            if self.d > 1:
                PL = np.r_[10:90:20, 95, 99, 99.9]
                try:
                    ql = qlevels(f, p=PL)
                    wdata.clevels = ql
                    wdata.plevels = PL
                except:
                    pass
            return wdata

    def _check_shape(self, points):
        points = atleast_2d(points)
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
            else:
                msg = "points have dimension %s, dataset has dimension %s"
                raise ValueError(msg % (d, self.d))
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

    __call__ = eval_grid_fast


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
        self.hs = hs
        self.kernel = kernel if kernel else Kernel('gauss')
        self.alpha = alpha
        self.xmin = xmin
        self.xmax = xmax
        self.inc = inc
        self.initialize()

    def initialize(self):
        self.d, self.n = self.dataset.shape
        if self.n > 1:
            self._set_xlimits()
            self._initialize()

    def _initialize(self):
        pass

    def _set_xlimits(self):
        amin = self.dataset.min(axis=-1)
        amax = self.dataset.max(axis=-1)
        iqr = iqrange(self.dataset, axis=-1)
        self._sigma = np.minimum(
            np.std(self.dataset, axis=-1, ddof=1), iqr / 1.34)
        # xyzrange = amax - amin
        # offset = xyzrange / 4.0
        offset = self._sigma
        if self.xmin is None:
            self.xmin = amin - offset
        else:
            self.xmin = self.xmin * np.ones((self.d, 1))
        if self.xmax is None:
            self.xmax = amax + offset
        else:
            self.xmax = self.xmax * np.ones((self.d, 1))

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
        for i in range(self.d):
            args.append(np.linspace(xmin[i], xmax[i], self.inc))
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
            args = []
            for i in range(self.d):
                args.append(np.linspace(self.xmin[i], self.xmax[i], self.inc))
        self.args = args
        return self._eval_grid_fun(self._eval_grid, *args, **kwds)

    def _eval_grid(self, *args):
        pass

    def _eval_grid_fun(self, eval_grd, *args, **kwds):
        output = kwds.pop('output', 'value')
        f = eval_grd(*args, **kwds)
        if output == 'value':
            return f
        else:
            titlestr = 'Kernel density estimate (%s)' % self.kernel.name
            kwds2 = dict(title=titlestr)

            kwds2['plot_kwds'] = kwds.pop('plot_kwds', dict(plotflag=1))
            kwds2.update(**kwds)
            args = self.args
            if self.d == 1:
                args = args[0]
            wdata = PlotData(f, args, **kwds2)
            if self.d > 1:
                PL = np.r_[10:90:20, 95, 99, 99.9]
                try:
                    ql = qlevels(f, p=PL)
                    wdata.clevels = ql
                    wdata.plevels = PL
                except:
                    pass
            return wdata

    def _check_shape(self, points):
        points = atleast_2d(points)
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
            else:
                msg = "points have dimension %s, dataset has dimension %s"
                raise ValueError(msg % (d, self.d))
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
            self.xmin = np.where(L2 != 1, np.maximum(
                self.xmin, amin / 100.0), self.xmin).reshape((-1, 1))

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
            ipoints = meshgrid(*args) if self.d > 1 else args
            # shape0 = points[0].shape
            # shape0i = ipoints[0].shape
            for i in range(self.d):
                points[i].shape = (-1,)
                # ipoints[i].shape = (-1,)
            points = np.asarray(points).T
            # ipoints = np.asarray(ipoints).T
            fi = interpolate.griddata(points, f.ravel(), tuple(ipoints),
                                      method='linear',
                                      fill_value=0.0)
            # fi.shape = shape0i
            self.args = args
            r = kwds.get('r', 0)
            if r == 0:
                return fi * (fi > 0)
            else:
                return fi
        return f

    def _eval_grid(self, *args, **kwds):
        if self.L2 is None:
            return self.tkde.eval_grid(*args, **kwds)
        targs = self._dat2gaus(list(args))
        tf = self.tkde.eval_grid(*targs, **kwds)
        points = meshgrid(*args) if self.d > 1 else self.args
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
    ...    [ 0.20397743,  0.40252228,  0.54594119,  0.52219025,  0.39062189,
    ...      0.2638171 ,  0.16407487,  0.08270755,  0.04784434,  0.04784434])
    True
    >>> f1 = kde0.eval_grid_fast(output='plot')
    >>> np.allclose(np.interp(x, f1.args, f1.data),
    ...   [ 0.20397743,  0.40252228,  0.54594119,  0.52219025,  0.39062189,
    ...     0.2638171 ,  0.16407487,  0.08270755,  0.04784434,  0.04784434])
    True
    >>> h = f1.plot()

    import pylab as plb
    h1 = plb.plot(x, f) #  1D probability density plot
    t = np.trapz(f, x)
    """

    def __init__(self, data, hs=None, kernel=None, alpha=0.0, xmin=None,
                 xmax=None, inc=512):
        super(KDE, self).__init__(data, hs, kernel, alpha, xmin, xmax, inc)

    def _initialize(self):
        self._compute_smoothing()
        self._lambda = np.ones(self.n)
        if self.alpha > 0:
            # pilt = KDE(self.dataset, hs=self.hs, kernel=self.kernel, alpha=0)
            # f = pilt.eval_points(self.dataset) # get a pilot estimate by
            # regular KDE (alpha=0)
            f = self.eval_points(self.dataset)  # pilot estimate
            g = np.exp(np.mean(np.log(f)))
            self._lambda = (f / g) ** (-self.alpha)

        if self.inc is None:
            unused_tau, tau = self.kernel.effective_support()
            xyzrange = 8 * self._sigma
            L1 = 10
            self.inc = 2 ** nextpow2(
                max(48, (L1 * xyzrange / (tau * self.hs)).max()))
            pass

    def _compute_smoothing(self):
        """Computes the smoothing matrix."""
        get_smoothing = self.kernel.get_smoothing
        h = self.hs
        if h is None:
            h = get_smoothing(self.dataset)
        h = np.atleast_1d(h)
        hsiz = h.shape

        if (len(hsiz) == 1) or (self.d == 1):
            if max(hsiz) == 1:
                h = h * np.ones(self.d)
            else:
                h.shape = (self.d,)  # make sure it has the correct dimension

            # If h negative calculate automatic values
            ind, = np.where(h <= 0)
            for i in ind.tolist():
                h[i] = get_smoothing(self.dataset[i])
            deth = h.prod()
            self.inv_hs = np.diag(1.0 / h)
        else:  # fully general smoothing matrix
            deth = linalg.det(h)
            if deth <= 0:
                raise ValueError(
                    'bandwidth matrix h must be positive definit!')
            self.inv_hs = linalg.inv(h)
        self.hs = h
        self._norm_factor = deth * self.n

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

        Xnc = meshgrid(*Xn) if d > 1 else Xn

        shape0 = Xnc[0].shape
        for i in range(d):
            Xnc[i].shape = (-1,)

        Xn = np.dot(self.inv_hs, np.vstack(Xnc))

        # Obtain the kernel weights.
        kw = self.kernel(Xn)

        # plt.plot(kw)
        # plt.draw()
        # plt.show()
        norm_fact0 = (kw.sum() * dx.prod() * self.n)
        norm_fact = (self._norm_factor * self.kernel.norm_factor(d, self.n))
        if np.abs(norm_fact0 - norm_fact) > 0.05 * norm_fact:
            warnings.warn(
                'Numerical inaccuracy due to too low discretization. ' +
                'Increase the discretization of the evaluation grid ' +
                '(inc=%d)!' % inc)
            norm_fact = norm_fact0

        kw = kw / norm_fact
        r = kwds.get('r', 0)
        if r != 0:
            kw *= np.vstack(Xnc) ** r if d > 1 else Xnc[0]
        kw.shape = shape0
        kw = np.fft.ifftshift(kw)
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn

        y = kwds.get('y', 1.0)
        # if self.alpha>0:
        #    y = y / self._lambda**d

        # Find the binned kernel weights, c.
        c = gridcount(self.dataset, X, y=y)
        # Perform the convolution.
        z = np.real(ifftn(fftn(c, s=nfft) * fftn(kw)))

        ix = (slice(0, inc),) * d
        if r == 0:
            return z[ix] * (z[ix] > 0.0)
        else:
            return z[ix]

    def _eval_grid(self, *args, **kwds):

        grd = meshgrid(*args) if len(args) > 1 else list(args)
        shape0 = grd[0].shape
        d = len(grd)
        for i in range(d):
            grd[i] = grd[i].ravel()
        f = self.eval_points(np.vstack(grd), **kwds)
        return f.reshape(shape0)

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

        result = np.zeros((m,))

        r = kwds.get('r', 0)
        if r == 0:
            def fun(xi):
                return 1
        else:
            def fun(xi):
                return (xi ** r).sum(axis=0)

        if m >= self.n:
            y = kwds.get('y', np.ones(self.n))
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_hs / self._lambda[i], diff)
                result += y[i] * \
                    fun(diff) * self.kernel(tdiff) / self._lambda[i] ** d
        else:
            y = kwds.get('y', 1)
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_hs, diff / self._lambda[np.newaxis, :])
                tmp = y * fun(diff) * self.kernel(tdiff) / self._lambda ** d
                result[i] = tmp.sum(axis=-1)

        result /= (self._norm_factor * self.kernel.norm_factor(d, self.n))

        return result


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
    >>> N = 100
    >>> ei = np.random.normal(loc=0, scale=0.075, size=(N,))

    >>> x = np.linspace(0, 1, N)
    >>> import wafo.kdetools as wk

    >>> y = 2*np.exp(-x**2/(2*0.3**2))+3*np.exp(-(x-1)**2/(2*0.7**2)) + ei
    >>> kreg = wk.KRegression(x, y)
    >>> f = kreg(output='plotobj', title='Kernel regression', plotflag=1)
    >>> h = f.plot(label='p=0')
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
            st = stats
            pup = np.where(
                p == 1, 1, st.beta.isf(alpha / 2, n * p + a, n * (1 - p) + b))
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
            c0 = np.zeros(xi.shape)
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


class _Kernel(object):

    def __init__(self, r=1.0, stats=None):
        self.r = r  # radius of kernel
        self.stats = stats

    def norm_factor(self, d=1, n=None):
        return 1.0

    def norm_kernel(self, x):
        X = np.atleast_2d(x)
        return self._kernel(X) / self.norm_factor(*X.shape)

    def kernel(self, x):
        return self._kernel(np.atleast_2d(x))

    def deriv4_6_8_10(self, t, numout=4):
        raise Exception('Method not implemented for this kernel!')

    def effective_support(self):
        """Return the effective support of kernel.

        The kernel must be symmetric and compactly supported on [-tau tau]
        if the kernel has infinite support then the kernel must have the
        effective support in [-tau tau], i.e., be negligible outside the range

        """
        return self._effective_support()

    def _effective_support(self):
        return - self.r, self.r
    __call__ = kernel


class _KernelMulti(_Kernel):
    # p=0;  %Sphere = rect for 1D
    # p=1;  %Multivariate Epanechnikov kernel.
    # p=2;  %Multivariate Bi-weight Kernel
    # p=3;  %Multi variate Tri-weight Kernel
    # p=4;  %Multi variate Four-weight Kernel

    def __init__(self, r=1.0, p=1, stats=None):
        self.r = r
        self.p = p
        self.stats = stats

    def norm_factor(self, d=1, n=None):
        r = self.r
        p = self.p
        c = 2 ** p * np.prod(np.r_[1:p + 1]) * sphere_volume(d, r) / np.prod(
            np.r_[(d + 2):(2 * p + d + 1):2])  # normalizing constant
        return c

    def _kernel(self, x):
        r = self.r
        p = self.p
        x2 = x ** 2
        return ((1.0 - x2.sum(axis=0) / r ** 2).clip(min=0.0)) ** p

mkernel_epanechnikov = _KernelMulti(p=1, stats=_stats_epan)
mkernel_biweight = _KernelMulti(p=2, stats=_stats_biwe)
mkernel_triweight = _KernelMulti(p=3, stats=_stats_triw)


class _KernelProduct(_KernelMulti):
    # p=0;  %rectangular
    # p=1;  %1D product Epanechnikov kernel.
    # p=2;  %1D product Bi-weight Kernel
    # p=3;  %1D product Tri-weight Kernel
    # p=4;  %1D product Four-weight Kernel

    def norm_factor(self, d=1, n=None):
        r = self.r
        p = self.p
        c = (2 ** p * np.prod(np.r_[1:p + 1]) * sphere_volume(1, r) /
             np.prod(np.r_[(1 + 2):(2 * p + 2):2]))
        return c ** d

    def _kernel(self, x):
        r = self.r  # radius
        pdf = (1 - (x / r) ** 2).clip(min=0.0)
        return pdf.prod(axis=0)

mkernel_p1epanechnikov = _KernelProduct(p=1, stats=_stats_epan)
mkernel_p1biweight = _KernelProduct(p=2, stats=_stats_biwe)
mkernel_p1triweight = _KernelProduct(p=3, stats=_stats_triw)


class _KernelRectangular(_Kernel):

    def _kernel(self, x):
        return np.where(np.all(np.abs(x) <= self.r, axis=0), 1, 0.0)

    def norm_factor(self, d=1, n=None):
        r = self.r
        return (2 * r) ** d
mkernel_rectangular = _KernelRectangular(stats=_stats_rect)


class _KernelTriangular(_Kernel):

    def _kernel(self, x):
        pdf = (1 - np.abs(x)).clip(min=0.0)
        return pdf.prod(axis=0)
mkernel_triangular = _KernelTriangular(stats=_stats_tria)


class _KernelGaussian(_Kernel):

    def _kernel(self, x):
        sigma = self.r / 4.0
        x2 = (x / sigma) ** 2
        return exp(-0.5 * x2.sum(axis=0))

    def norm_factor(self, d=1, n=None):
        sigma = self.r / 4.0
        return (2 * pi * sigma) ** (d / 2.0)

    def deriv4_6_8_10(self, t, numout=4):
        """Returns 4th, 6th, 8th and 10th derivatives of the kernel
        function."""
        phi0 = exp(-0.5 * t ** 2) / sqrt(2 * pi)
        p4 = [1, 0, -6, 0, +3]
        p4val = np.polyval(p4, t) * phi0
        if numout == 1:
            return p4val
        out = [p4val]
        pn = p4
        for unusedix in range(numout - 1):
            pnp1 = np.polyadd(-np.r_[pn, 0], np.polyder(pn))
            pnp2 = np.polyadd(-np.r_[pnp1, 0], np.polyder(pnp1))
            out.append(np.polyval(pnp2, t) * phi0)
            pn = pnp2
        return out

mkernel_gaussian = _KernelGaussian(r=4.0, stats=_stats_gaus)

# def mkernel_gaussian(X):
#    x2 = X ** 2
#    d = X.shape[0]
#    return (2 * pi) ** (-d / 2) * exp(-0.5 * x2.sum(axis=0))


class _KernelLaplace(_Kernel):

    def _kernel(self, x):
        absX = np.abs(x)
        return exp(-absX.sum(axis=0))

    def norm_factor(self, d=1, n=None):
        return 2 ** d
mkernel_laplace = _KernelLaplace(r=7.0, stats=_stats_lapl)


class _KernelLogistic(_Kernel):

    def _kernel(self, x):
        s = exp(-x)
        return np.prod(1.0 / (s + 1) ** 2, axis=0)
mkernel_logistic = _KernelLogistic(r=7.0, stats=_stats_logi)

_MKERNEL_DICT = dict(
    epan=mkernel_epanechnikov,
    biwe=mkernel_biweight,
    triw=mkernel_triweight,
    p1ep=mkernel_p1epanechnikov,
    p1bi=mkernel_p1biweight,
    p1tr=mkernel_p1triweight,
    rect=mkernel_rectangular,
    tria=mkernel_triangular,
    lapl=mkernel_laplace,
    logi=mkernel_logistic,
    gaus=mkernel_gaussian
)
_KERNEL_EXPONENT_DICT = dict(
    re=0, sp=0, ep=1, bi=2, tr=3, fo=4, fi=5, si=6, se=7)


class Kernel(object):

    """Multivariate kernel.

    Parameters
    ----------
    name : string
        defining the kernel. Valid options are:
        'epanechnikov'  - Epanechnikov kernel.
        'biweight'      - Bi-weight kernel.
        'triweight'     - Tri-weight kernel.
        'p1epanechnikov' - product of 1D Epanechnikov kernel.
        'p1biweight'    - product of 1D Bi-weight kernel.
        'p1triweight'   - product of 1D Tri-weight kernel.
        'triangular'    - Triangular kernel.
        'gaussian'      - Gaussian kernel
        'rectangular'   - Rectangular kernel.
        'laplace'       - Laplace kernel.
        'logistic'      - Logistic kernel.
    Note that only the first 4 letters of the kernel name is needed.

    Examples
    --------
     N = 20
    data = np.random.rayleigh(1, size=(N,))
    >>> data = np.array([
    ...        0.75355792,  0.72779194,  0.94149169,  0.07841119,  2.32291887,
    ...        1.10419995,  0.77055114,  0.60288273,  1.36883635,  1.74754326,
    ...        1.09547561,  1.01671133,  0.73211143,  0.61891719,  0.75903487,
    ...        1.8919469 ,  0.72433808,  1.92973094,  0.44749838,  1.36508452])

    >>> import wafo.kdetools as wk
    >>> gauss = wk.Kernel('gaussian')
    >>> gauss.stats()
    (1, 0.28209479177387814, 0.21157109383040862)
    >>> np.allclose(gauss.hscv(data), 0.21779575)
    True
    >>> np.allclose(gauss.hstt(data), 0.16341135)
    True
    >>> np.allclose(gauss.hste(data), 0.19179399)
    True
    >>> np.allclose(gauss.hldpi(data), 0.22502733)
    True
    >>> wk.Kernel('laplace').stats()
    (2, 0.25, inf)

    >>> triweight = wk.Kernel('triweight')
    >>> np.allclose(triweight.stats(),
    ...            (0.1111111111111111, 0.81585081585081587, np.inf))
    True
    >>> np.allclose(triweight(np.linspace(-1,1,11)),
    ...   [ 0.,  0.046656,  0.262144,  0.592704,  0.884736,  1.,
    ...     0.884736,  0.592704,  0.262144,  0.046656,  0.])
    True
    >>> np.allclose(triweight.hns(data), 0.82, rtol=1e-2)
    True
    >>> np.allclose(triweight.hos(data), 0.88, rtol=1e-2)
    True
    >>> np.allclose(triweight.hste(data), 0.57, rtol=1e-2)
    True
    >>> np.allclose(triweight.hscv(data), 0.648, rtol=1e-2)
    True

    See also
    --------
    mkernel

    References
    ----------
    B. W. Silverman (1986)
    'Density estimation for statistics and data analysis'
     Chapman and Hall, pp. 43, 76

    Wand, M. P. and Jones, M. C. (1995)
    'Density estimation for statistics and data analysis'
     Chapman and Hall, pp 31, 103,  175

    """

    def __init__(self, name, fun='hste'):  # 'hns'):
        self.kernel = _MKERNEL_DICT[name[:4]]
        # self.name = self.kernel.__name__.replace('mkernel_', '').title()
        try:
            self.get_smoothing = getattr(self, fun)
        except:
            self.get_smoothing = self.hste

    def _get_name(self):
        return self.kernel.__class__.__name__.replace('_Kernel', '').title()
    name = property(_get_name)

    def get_smoothing(self, *args, **kwds):
        pass

    def stats(self):
        """Return some 1D statistics of the kernel.

        Returns
        -------
        mu2 : real scalar
            2'nd order moment, i.e.,int(x^2*kernel(x))
        R : real scalar
            integral of squared kernel, i.e., int(kernel(x)^2)
        Rdd  : real scalar
            integral of squared double derivative of kernel,
            i.e., int( (kernel''(x))^2 ).

        Reference
        ---------
        Wand,M.P. and Jones, M.C. (1995)
        'Kernel smoothing'
        Chapman and Hall, pp 176.

        """
        return self.kernel.stats

    def deriv4_6_8_10(self, t, numout=4):
        return self.kernel.deriv4_6_8_10(t, numout)

    def effective_support(self):
        return self.kernel.effective_support()

    def hns(self, data):
        """Returns Normal Scale Estimate of Smoothing Parameter.

        Parameter
        ---------
        data : 2D array
            shape d x n (d = # dimensions )

        Returns
        -------
        h : array-like
            one dimensional optimal value for smoothing parameter
            given the data and kernel.  size D

        HNS only gives an optimal value with respect to mean integrated
        square error, when the true underlying distribution
        is Gaussian. This works reasonably well if the data resembles a
        Gaussian distribution. However if the distribution is asymmetric,
        multimodal or have long tails then HNS may  return a to large
        smoothing parameter, i.e., the KDE may be oversmoothed and mask
        important features of the data. (=> large bias).
        One way to remedy this is to reduce H by multiplying with a constant
        factor, e.g., 0.85. Another is to try different values for H and make a
        visual check by eye.

        Example:
          data = rndnorm(0, 1,20,1)
          h = hns(data,'epan')

        See also:
        ---------
        hste, hbcv, hboot, hos, hldpi, hlscv, hscv, hstt, kde

        Reference:
        ---------
        B. W. Silverman (1986)
        'Density estimation for statistics and data analysis'
        Chapman and Hall, pp 43-48
        Wand,M.P. and Jones, M.C. (1995)
        'Kernel smoothing'
        Chapman and Hall, pp 60--63

        """

        A = np.atleast_2d(data)
        n = A.shape[1]

        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, unusedRdd = self.stats()
        AMISEconstant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        iqr = iqrange(A, axis=1)  # interquartile range
        stdA = np.std(A, axis=1, ddof=1)
        # use of interquartile range guards against outliers.
        # the use of interquartile range is better if
        # the distribution is skew or have heavy tails
        # This lessen the chance of oversmoothing.
        return np.where(iqr > 0,
                        np.minimum(stdA, iqr / 1.349), stdA) * AMISEconstant

    def hos(self, data):
        """Returns Oversmoothing Parameter.

        Parameter
        ---------
        data   = data matrix, size N x D (D = # dimensions )

        Returns
        -------
        h : vector size 1 x D
            one dimensional maximum smoothing value for smoothing parameter
            given the data and kernel.

        The oversmoothing or maximal smoothing principle relies on the fact
        that there is a simple upper bound for the AMISE-optimal bandwidth for
        estimation of densities with a fixed value of a particular scale
        measure. While HOS will give too large bandwidth for optimal estimation
        of a general density it provides an excellent starting point for
        subjective choice of bandwidth. A sensible strategy is to plot an
        estimate with bandwidth HOS and then sucessively look at plots based on
        convenient fractions of HOS to see what features are present in the
        data for various amount of smoothing. The relation to HNS is given by:

                HOS = HNS/0.93

        Example:
        --------
        data = rndnorm(0, 1,20,1)
        h = hos(data,'epan');

        See also  hste, hbcv, hboot, hldpi, hlscv, hscv, hstt, kde, kdefun

        Reference
        ---------
        B. W. Silverman (1986)
        'Density estimation for statistics and data analysis'
        Chapman and Hall, pp 43-48

        Wand,M.P. and Jones, M.C. (1986)
        'Kernel smoothing'
        Chapman and Hall, pp 60--63

        """
        return self.hns(data) / 0.93

    def hmns(self, data):
        """Returns Multivariate Normal Scale Estimate of Smoothing Parameter.

         CALL:  h = hmns(data,kernel)

           h      = M dimensional optimal value for smoothing parameter
                    given the data and kernel.  size D x D
           data   = data matrix, size D x N (D = # dimensions )
           kernel = 'epanechnikov'  - Epanechnikov kernel.
                    'biweight'      - Bi-weight kernel.
                    'triweight'     - Tri-weight kernel.
                    'gaussian'      - Gaussian kernel

          Note that only the first 4 letters of the kernel name is needed.

         HMNS  only gives  a optimal value with respect to mean integrated
         square error, when the true underlying distribution is Multivariate
         Gaussian. This works reasonably well if the data resembles a
         Multivariate Gaussian distribution. However if the distribution is
         asymmetric, multimodal or have long tails then HNS is maybe more
         appropriate.

          Example:
            data = rndnorm(0, 1,20,2)
            h = hmns(data,'epan')

         See also
         --------

        hns, hste, hbcv, hboot, hos, hldpi, hlscv, hscv, hstt

         Reference
         ----------
          B. W. Silverman (1986)
         'Density estimation for statistics and data analysis'
          Chapman and Hall, pp 43-48, 87

          Wand,M.P. and Jones, M.C. (1995)
         'Kernel smoothing'
          Chapman and Hall, pp 60--63, 86--88

        """
        # TODO: implement more kernels

        A = np.atleast_2d(data)
        d, n = A.shape

        if d == 1:
            return self.hns(data)
        name = self.name[:4].lower()
        if name == 'epan':        # Epanechnikov kernel
            a = (8.0 * (d + 4.0) * (2 * sqrt(pi)) ** d /
                 sphere_volume(d)) ** (1. / (4.0 + d))
        elif name == 'biwe':  # Bi-weight kernel
            a = 2.7779
            if d > 2:
                raise ValueError('not implemented for d>2')
        elif name == 'triw':  # Triweight
            a = 3.12
            if d > 2:
                raise ValueError('not implemented for d>2')
        elif name == 'gaus':  # Gaussian kernel
            a = (4.0 / (d + 2.0)) ** (1. / (d + 4.0))
        else:
            raise ValueError('Unknown kernel.')

        covA = scipy.cov(A)

        return a * linalg.sqrtm(covA).real * n ** (-1. / (d + 4))

    def hste(self, data, h0=None, inc=128, maxit=100, releps=0.01, abseps=0.0):
        '''HSTE 2-Stage Solve the Equation estimate of smoothing parameter.

         CALL:  hs = hste(data,kernel,h0)

               hs = one dimensional value for smoothing parameter
                    given the data and kernel.  size 1 x D
           data   = data matrix, size N x D (D = # dimensions )
           kernel = 'gaussian'  - Gaussian kernel (default)
                     ( currently the only supported kernel)
               h0 = initial starting guess for hs (default h0=hns(A,kernel))

          Example:
           x  = rndnorm(0,1,50,1);
           hs = hste(x,'gauss');

         See also  hbcv, hboot, hos, hldpi, hlscv, hscv, hstt, kde, kdefun

         Reference
         ---------
          B. W. Silverman (1986)
         'Density estimation for statistics and data analysis'
          Chapman and Hall, pp 57--61

          Wand,M.P. and Jones, M.C. (1986)
         'Kernel smoothing'
          Chapman and Hall, pp 74--75
        '''
        # TODO: NB: this routine can be made faster:
        # TODO: replace the iteration in the end with a Newton Raphson scheme

        A = np.atleast_2d(data)
        d, n = A.shape

        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, unusedRdd = self.stats()

        AMISEconstant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        STEconstant = R / (mu2 ** (2) * n)

        sigmaA = self.hns(A) / AMISEconstant
        if h0 is None:
            h0 = sigmaA * AMISEconstant

        h = np.asarray(h0, dtype=float)

        nfft = inc * 2
        amin = A.min(axis=1)  # Find the minimum value of A.
        amax = A.max(axis=1)  # Find the maximum value of A.
        arange = amax - amin  # Find the range of A.

        # xa holds the x 'axis' vector, defining a grid of x values where
        # the k.d. function will be evaluated.

        ax1 = amin - arange / 8.0
        bx1 = amax + arange / 8.0

        kernel2 = Kernel('gauss')
        mu2, R, unusedRdd = kernel2.stats()
        STEconstant2 = R / (mu2 ** (2) * n)
        fft = np.fft.fft
        ifft = np.fft.ifft

        for dim in range(d):
            s = sigmaA[dim]
            ax = ax1[dim]
            bx = bx1[dim]

            xa = np.linspace(ax, bx, inc)
            xn = np.linspace(0, bx - ax, inc)

            c = gridcount(A[dim], xa)

            # Step 1
            psi6NS = -15 / (16 * sqrt(pi) * s ** 7)
            psi8NS = 105 / (32 * sqrt(pi) * s ** 9)

            # Step 2
            k40, k60 = kernel2.deriv4_6_8_10(0, numout=2)
            g1 = (-2 * k40 / (mu2 * psi6NS * n)) ** (1.0 / 7)
            g2 = (-2 * k60 / (mu2 * psi8NS * n)) ** (1.0 / 9)

            # Estimate psi6 given g2.
            # kernel weights.
            kw4, kw6 = kernel2.deriv4_6_8_10(xn / g2, numout=2)
            # Apply fftshift to kw.
            kw = np.r_[kw6, 0, kw6[-1:0:-1]]
            z = np.real(ifft(fft(c, nfft) * fft(kw)))     # convolution.
            psi6 = np.sum(c * z[:inc]) / (n * (n - 1) * g2 ** 7)

            # Estimate psi4 given g1.
            kw4 = kernel2.deriv4_6_8_10(xn / g1, numout=1)  # kernel weights.
            kw = np.r_[kw4, 0, kw4[-1:0:-1]]  # Apply 'fftshift' to kw.
            z = np.real(ifft(fft(c, nfft) * fft(kw)))  # convolution.
            psi4 = np.sum(c * z[:inc]) / (n * (n - 1) * g1 ** 5)

            h1 = h[dim]
            h_old = 0
            count = 0

            while ((abs(h_old - h1) > max(releps * h1, abseps)) and
                   (count < maxit)):
                count += 1
                h_old = h1

                # Step 3
                gamma = ((2 * k40 * mu2 * psi4 * h1 ** 5) /
                         (-psi6 * R)) ** (1.0 / 7)

                # Now estimate psi4 given gamma.
                # kernel weights.
                kw4 = kernel2.deriv4_6_8_10(xn / gamma, numout=1)
                kw = np.r_[kw4, 0, kw4[-1:0:-1]]  # Apply 'fftshift' to kw.
                z = np.real(ifft(fft(c, nfft) * fft(kw)))  # convolution.

                psi4Gamma = np.sum(c * z[:inc]) / (n * (n - 1) * gamma ** 5)

                # Step 4
                h1 = (STEconstant2 / psi4Gamma) ** (1.0 / 5)

            # Kernel other than Gaussian scale bandwidth
            h1 = h1 * (STEconstant / STEconstant2) ** (1.0 / 5)

            if count >= maxit:
                warnings.warn('The obtained value did not converge.')

            h[dim] = h1
        # end for dim loop
        return h

    def hisj(self, data, inc=512, L=7):
        '''
        HISJ Improved Sheather-Jones estimate of smoothing parameter.

        Unlike many other implementations, this one is immune to problems
        caused by multimodal densities with widely separated modes. The
        estimation does not deteriorate for multimodal densities, because
        it do not assume a parametric model for the data.

        Parameters
        ----------
        data - a vector of data from which the density estimate is constructed
        inc  - the number of mesh points used in the uniform discretization

        Returns
        -------
        bandwidth - the optimal bandwidth

        Reference
        ---------
        Kernel density estimation via diffusion
        Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
        Annals of Statistics, Volume 38, Number 5, pages 2916-2957.
        '''
        A = np.atleast_2d(data)
        d, n = A.shape

        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, unusedRdd = self.stats()
        STEconstant = R / (n * mu2 ** 2)

        amin = A.min(axis=1)  # Find the minimum value of A.
        amax = A.max(axis=1)  # Find the maximum value of A.
        arange = amax - amin  # Find the range of A.

        # xa holds the x 'axis' vector, defining a grid of x values where
        # the k.d. function will be evaluated.

        ax1 = amin - arange / 8.0
        bx1 = amax + arange / 8.0

        kernel2 = Kernel('gauss')
        mu2, R, unusedRdd = kernel2.stats()
        STEconstant2 = R / (mu2 ** (2) * n)

        def fixed_point(t, N, I, a2):
            ''' this implements the function t-zeta*gamma^[L](t)'''

            prod = np.prod
            # L = 7
            logI = np.log(I)
            f = 2 * pi ** (2 * L) * \
                (a2 * exp(L * logI - I * pi ** 2 * t)).sum()
            for s in range(L - 1, 1, -1):
                K0 = prod(np.r_[1:2 * s:2]) / sqrt(2 * pi)
                const = (1 + (1. / 2) ** (s + 1. / 2)) / 3
                time = (2 * const * K0 / N / f) ** (2. / (3 + 2 * s))
                f = 2 * pi ** (2 * s) * \
                    (a2 * exp(s * logI - I * pi ** 2 * time)).sum()
            return t - (2 * N * sqrt(pi) * f) ** (-2. / 5)

        h = np.empty(d)
        for dim in range(d):
            ax = ax1[dim]
            bx = bx1[dim]
            xa = np.linspace(ax, bx, inc)
            R = bx - ax

            c = gridcount(A[dim], xa)
            N = len(set(A[dim]))
            # a = dct(c/c.sum(), norm=None)
            a = dct(c / len(A[dim]), norm=None)

            # now compute the optimal bandwidth^2 using the referenced method
            I = np.asfarray(np.arange(1, inc)) ** 2
            a2 = (a[1:] / 2) ** 2

            def fun(t):
                return fixed_point(t, N, I, a2)
            x = np.linspace(0, 0.1, 150)
            ai = x[0]
            f0 = fun(ai)
            for bi in x[1:]:
                f1 = fun(bi)
                if f1 * f0 <= 0:
                    # print('ai = %g, bi = %g' % (ai,bi))
                    break
                else:
                    ai = bi
            # y = np.asarray([fun(j) for j in x])
            # plt.figure(1)
            # plt.plot(x,y)
            # plt.show()

            # use  fzero to solve the equation t=zeta*gamma^[5](t)
            try:
                t_star = optimize.brentq(fun, a=ai, b=bi)
            except:
                t_star = 0.28 * N ** (-2. / 5)
                warnings.warn('Failure in obtaining smoothing parameter')

            # smooth the discrete cosine transform of initial data using t_star
            # a_t = a*exp(-np.arange(inc)**2*pi**2*t_star/2)
            # now apply the inverse discrete cosine transform
            # density = idct(a_t)/R;

            # take the rescaling of the data into account
            bandwidth = sqrt(t_star) * R

            # Kernel other than Gaussian scale bandwidth
            h[dim] = bandwidth * (STEconstant / STEconstant2) ** (1.0 / 5)
        # end  for dim loop
        return h

    def hstt(self, data, h0=None, inc=128, maxit=100, releps=0.01, abseps=0.0):
        '''HSTT Scott-Tapia-Thompson estimate of smoothing parameter.

         CALL: hs = hstt(data,kernel)

               hs = one dimensional value for smoothing parameter
                    given the data and kernel.  size 1 x D
           data   = data matrix, size N x D (D = # dimensions )
           kernel = 'epanechnikov'  - Epanechnikov kernel. (default)
                    'biweight'      - Bi-weight kernel.
                    'triweight'     - Tri-weight kernel.
                    'triangular'    - Triangular kernel.
                    'gaussian'      - Gaussian kernel
                    'rectangular'   - Rectangular kernel.
                    'laplace'       - Laplace kernel.
                    'logistic'      - Logistic kernel.

         HSTT returns Scott-Tapia-Thompson (STT) estimate of smoothing
         parameter. This is a Solve-The-Equation rule (STE).
         Simulation studies shows that the STT estimate of HS
         is a good choice under a variety of models. A comparison with
         likelihood cross-validation (LCV) indicates that LCV performs slightly
         better for short tailed densities.
         However, STT method in contrast to LCV is insensitive to outliers.

        Example
        -------
           x  = rndnorm(0,1,50,1);
           hs = hstt(x,'gauss');

        See also
        --------
        hste, hbcv, hboot, hos, hldpi, hlscv, hscv, kde, kdebin

        Reference
        ---------
        B. W. Silverman (1986)
         'Density estimation for statistics and data analysis'
          Chapman and Hall, pp 57--61
        '''
        A = np.atleast_2d(data)
        d, n = A.shape

        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, unusedRdd = self.stats()

        AMISEconstant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        STEconstant = R / (mu2 ** (2) * n)

        sigmaA = self.hns(A) / AMISEconstant
        if h0 is None:
            h0 = sigmaA * AMISEconstant

        h = np.asarray(h0, dtype=float)

        nfft = inc * 2
        amin = A.min(axis=1)  # Find the minimum value of A.
        amax = A.max(axis=1)  # Find the maximum value of A.
        arange = amax - amin  # Find the range of A.

        # xa holds the x 'axis' vector, defining a grid of x values where
        # the k.d. function will be evaluated.

        ax1 = amin - arange / 8.0
        bx1 = amax + arange / 8.0

        fft = np.fft.fft
        ifft = np.fft.ifft
        for dim in range(d):
            s = sigmaA[dim]
            datan = A[dim] / s
            ax = ax1[dim] / s
            bx = bx1[dim] / s

            xa = np.linspace(ax, bx, inc)
            xn = np.linspace(0, bx - ax, inc)

            c = gridcount(datan, xa)

            count = 1
            h_old = 0
            h1 = h[dim] / s
            delta = (bx - ax) / (inc - 1)
            while ((abs(h_old - h1) > max(releps * h1, abseps)) and
                   (count < maxit)):
                count += 1
                h_old = h1

                kw4 = self.kernel(xn / h1) / (n * h1 * self.norm_factor(d=1))
                kw = np.r_[kw4, 0, kw4[-1:0:-1]]  # Apply 'fftshift' to kw.
                f = np.real(ifft(fft(c, nfft) * fft(kw)))  # convolution.

                # Estimate psi4=R(f'') using simple finite differences and
                # quadrature.
                ix = np.arange(1, inc - 1)
                z = ((f[ix + 1] - 2 * f[ix] + f[ix - 1]) / delta ** 2) ** 2
                psi4 = delta * z.sum()
                h1 = (STEconstant / psi4) ** (1. / 5)

            if count >= maxit:
                warnings.warn('The obtained value did not converge.')

            h[dim] = h1 * s
        # end % for dim loop
        return h

    def hscv(self, data, hvec=None, inc=128, maxit=100, fulloutput=False):
        '''
        HSCV Smoothed cross-validation estimate of smoothing parameter.

         CALL: [hs,hvec,score] = hscv(data,kernel,hvec)

           hs     = smoothing parameter
           hvec   = vector defining possible values of hs
                     (default linspace(0.25*h0,h0,100), h0=0.62)
           score  = score vector
           data   = data vector
           kernel = 'gaussian'      - Gaussian kernel the only supported

          Note that only the first 4 letters of the kernel name is needed.

          Example:
            data = rndnorm(0,1,20,1)
             [hs hvec score] = hscv(data,'epan');
             plot(hvec,score)
         See also  hste, hbcv, hboot, hos, hldpi, hlscv, hstt, kde, kdefun

         Wand,M.P. and Jones, M.C. (1986)
         'Kernel smoothing'
          Chapman and Hall, pp 75--79
        '''
        # TODO: Add support for other kernels than Gaussian
        A = np.atleast_2d(data)
        d, n = A.shape

        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, unusedRdd = self.stats()

        AMISEconstant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        STEconstant = R / (mu2 ** (2) * n)

        sigmaA = self.hns(A) / AMISEconstant
        if hvec is None:
            H = AMISEconstant / 0.93
            hvec = np.linspace(0.25 * H, H, maxit)
        hvec = np.asarray(hvec, dtype=float)

        steps = len(hvec)
        score = np.zeros(steps)

        nfft = inc * 2
        amin = A.min(axis=1)  # Find the minimum value of A.
        amax = A.max(axis=1)  # Find the maximum value of A.
        arange = amax - amin  # Find the range of A.

        # xa holds the x 'axis' vector, defining a grid of x values where
        # the k.d. function will be evaluated.

        ax1 = amin - arange / 8.0
        bx1 = amax + arange / 8.0

        kernel2 = Kernel('gauss')
        mu2, R, unusedRdd = kernel2.stats()
        STEconstant2 = R / (mu2 ** (2) * n)
        fft = np.fft.fft
        ifft = np.fft.ifft

        h = np.zeros(d)
        hvec = hvec * (STEconstant2 / STEconstant) ** (1. / 5.)

        k40, k60, k80, k100 = kernel2.deriv4_6_8_10(0, numout=4)
        psi8 = 105 / (32 * sqrt(pi))
        psi12 = 3465. / (512 * sqrt(pi))
        g1 = (-2. * k60 / (mu2 * psi8 * n)) ** (1. / 9.)
        g2 = (-2. * k100 / (mu2 * psi12 * n)) ** (1. / 13.)

        for dim in range(d):
            s = sigmaA[dim]
            ax = ax1[dim] / s
            bx = bx1[dim] / s
            datan = A[dim] / s

            xa = np.linspace(ax, bx, inc)
            xn = np.linspace(0, bx - ax, inc)

            c = gridcount(datan, xa)

            kw4, kw6 = kernel2.deriv4_6_8_10(xn / g1, numout=2)
            kw = np.r_[kw6, 0, kw6[-1:0:-1]]
            z = np.real(ifft(fft(c, nfft) * fft(kw)))
            psi6 = np.sum(c * z[:inc]) / (n ** 2 * g1 ** 7)

            kw4, kw6, kw8, kw10 = kernel2.deriv4_6_8_10(xn / g2, numout=4)
            kw = np.r_[kw10, 0, kw10[-1:0:-1]]
            z = np.real(ifft(fft(c, nfft) * fft(kw)))
            psi10 = np.sum(c * z[:inc]) / (n ** 2 * g2 ** 11)

            g3 = (-2. * k40 / (mu2 * psi6 * n)) ** (1. / 7.)
            g4 = (-2. * k80 / (mu2 * psi10 * n)) ** (1. / 11.)

            kw4 = kernel2.deriv4_6_8_10(xn / g3, numout=1)
            kw = np.r_[kw4, 0, kw4[-1:0:-1]]
            z = np.real(ifft(fft(c, nfft) * fft(kw)))
            psi4 = np.sum(c * z[:inc]) / (n ** 2 * g3 ** 5)

            kw4, kw6, kw8 = kernel2.deriv4_6_8_10(xn / g3, numout=3)
            kw = np.r_[kw8, 0, kw8[-1:0:-1]]
            z = np.real(ifft(fft(c, nfft) * fft(kw)))
            psi8 = np.sum(c * z[:inc]) / (n ** 2 * g4 ** 9)

            const = (441. / (64 * pi)) ** (1. / 18.) * \
                (4 * pi) ** (-1. / 5.) * \
                psi4 ** (-2. / 5.) * psi8 ** (-1. / 9.)

            M = np.atleast_2d(datan)

            Y = (M - M.T).ravel()

            for i in range(steps):
                g = const * n ** (-23. / 45) * hvec[i] ** (-2)
                sig1 = sqrt(2 * hvec[i] ** 2 + 2 * g ** 2)
                sig2 = sqrt(hvec[i] ** 2 + 2 * g ** 2)
                sig3 = sqrt(2 * g ** 2)
                term2 = np.sum(kernel2(Y / sig1) / sig1 - 2 * kernel2(
                    Y / sig2) / sig2 + kernel2(Y / sig3) / sig3)

                score[i] = 1. / (n * hvec[i] * 2. * sqrt(pi)) + term2 / n ** 2

            idx = score.argmin()
            # Kernel other than Gaussian scale bandwidth
            h[dim] = hvec[idx] * (STEconstant / STEconstant2) ** (1 / 5)
            if idx == 0:
                warnings.warn(
                    'Optimum is probably lower than hs=%g for dim=%d' %
                    (h[dim] * s, dim))
            elif idx == maxit - 1:
                warnings.warn(
                    'Optimum is probably higher than hs=%g for dim=%d' %
                    (h[dim] * s, dim))

        hvec = hvec * (STEconstant / STEconstant2) ** (1 / 5)
        if fulloutput:
            return h * sigmaA, score, hvec, sigmaA
        else:
            return h * sigmaA

    def hldpi(self, data, L=2, inc=128):
        '''HLDPI L-stage Direct Plug-In estimate of smoothing parameter.

         CALL: hs = hldpi(data,kernel,L)

               hs = one dimensional value for smoothing parameter
                    given the data and kernel.  size 1 x D
           data   = data matrix, size N x D (D = # dimensions )
           kernel = 'epanechnikov'  - Epanechnikov kernel.
                    'biweight'      - Bi-weight kernel.
                    'triweight'     - Tri-weight kernel.
                    'triangluar'    - Triangular kernel.
                    'gaussian'      - Gaussian kernel
                    'rectangular'   - Rectanguler kernel.
                    'laplace'       - Laplace kernel.
                    'logistic'      - Logistic kernel.
                L = 0,1,2,3,...   (default 2)

          Note that only the first 4 letters of the kernel name is needed.

          Example:
           x  = rndnorm(0,1,50,1);
           hs = hldpi(x,'gauss',1);

         See also  hste, hbcv, hboot, hos, hlscv, hscv, hstt, kde, kdefun

          Wand,M.P. and Jones, M.C. (1995)
         'Kernel smoothing'
          Chapman and Hall, pp 67--74
        '''
        A = np.atleast_2d(data)
        d, n = A.shape

        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, unusedRdd = self.stats()

        AMISEconstant = (8 * sqrt(pi) * R / (3 * n * mu2 ** 2)) ** (1. / 5)
        STEconstant = R / (n * mu2 ** 2)

        sigmaA = self.hns(A) / AMISEconstant

        nfft = inc * 2
        amin = A.min(axis=1)  # Find the minimum value of A.
        amax = A.max(axis=1)  # Find the maximum value of A.
        arange = amax - amin  # Find the range of A.

        # xa holds the x 'axis' vector, defining a grid of x values where
        # the k.d. function will be evaluated.

        ax1 = amin - arange / 8.0
        bx1 = amax + arange / 8.0

        kernel2 = Kernel('gauss')
        mu2, unusedR, unusedRdd = kernel2.stats()

        fft = np.fft.fft
        ifft = np.fft.ifft

        h = np.zeros(d)
        for dim in range(d):
            s = sigmaA[dim]
            datan = A[dim]  # / s
            ax = ax1[dim]  # / s
            bx = bx1[dim]  # / s

            xa = np.linspace(ax, bx, inc)
            xn = np.linspace(0, bx - ax, inc)

            c = gridcount(datan, xa)

            r = 2 * L + 4
            rd2 = L + 2

            # Eq. 3.7 in Wand and Jones (1995)
            PSI_r = (-1) ** (rd2) * np.prod(
                np.r_[rd2 + 1:r + 1]) / (sqrt(pi) * (2 * s) ** (r + 1))
            PSI = PSI_r
            if L > 0:
                # High order derivatives of the Gaussian kernel
                Kd = kernel2.deriv4_6_8_10(0, numout=L)

                # L-stage iterations to estimate PSI_4
                for ix in range(L, 0, -1):
                    gi = (-2 * Kd[ix - 1] /
                          (mu2 * PSI * n)) ** (1. / (2 * ix + 5))

                    # Obtain the kernel weights.
                    KW0 = kernel2.deriv4_6_8_10(xn / gi, numout=ix)
                    if ix > 1:
                        KW0 = KW0[-1]
                    # Apply 'fftshift' to kw.
                    kw = np.r_[KW0, 0, KW0[inc - 1:0:-1]]

                    # Perform the convolution.
                    z = np.real(ifft(fft(c, nfft) * fft(kw)))

                    PSI = np.sum(c * z[:inc]) / (n ** 2 * gi ** (2 * ix + 3))
                    # end
                # end
            h[dim] = (STEconstant / PSI) ** (1. / 5)
        return h

    def norm_factor(self, d=1, n=None):
        return self.kernel.norm_factor(d, n)

    def eval_points(self, points):
        return self.kernel(np.atleast_2d(points))
    __call__ = eval_points


def mkernel(X, kernel):
    """MKERNEL Multivariate Kernel Function.

    Paramaters
    ----------
    X : array-like
        matrix  size d x n (d = # dimensions, n = # evaluation points)
    kernel : string
        defining kernel
        'epanechnikov'  - Epanechnikov kernel.
        'biweight'      - Bi-weight kernel.
        'triweight'     - Tri-weight kernel.
        'p1epanechnikov' - product of 1D Epanechnikov kernel.
        'p1biweight'    - product of 1D Bi-weight kernel.
        'p1triweight'   - product of 1D Tri-weight kernel.
        'triangular'    - Triangular kernel.
        'gaussian'      - Gaussian kernel
        'rectangular'   - Rectangular kernel.
        'laplace'       - Laplace kernel.
        'logistic'      - Logistic kernel.
    Note that only the first 4 letters of the kernel name is needed.

    Returns
    -------
    z : ndarray
        kernel function values evaluated at X

    See also
    --------
    kde, kdefun, kdebin

    References
    ----------
    B. W. Silverman (1986)
    'Density estimation for statistics and data analysis'
     Chapman and Hall, pp. 43, 76

    Wand, M. P. and Jones, M. C. (1995)
    'Density estimation for statistics and data analysis'
     Chapman and Hall, pp 31, 103,  175

    """
    fun = _MKERNEL_DICT[kernel[:4]]
    return fun(np.atleast_2d(X))


def accumsum(accmap, a, size, dtype=None):
    if dtype is None:
        dtype = a.dtype
    size = np.atleast_1d(size)
    if len(size) > 1:
        binx = accmap[:, 0]
        biny = accmap[:, 1]
        out = sparse.coo_matrix(
            (a.ravel(), (binx, biny)), shape=size, dtype=dtype).tocsr()
    else:
        binx = accmap.ravel()
        zero = np.zeros(len(binx))
        out = sparse.coo_matrix(
            (a.ravel(), (binx, zero)), shape=(size, 1), dtype=dtype).tocsr()
    return out


def accumsum2(accmap, a, size):
    return np.bincount(accmap.ravel(), a.ravel(), np.array(size).max())


def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    """An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    >>> s
    array([ 9,  7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
    ...        [[0,0],[0,0],[0,1]],
    ...        [[0,0],[0,0],[0,1]],
    ...        [[1,0],[1,0],[1,1]]])
    >>> # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)

    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError(
            "The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])
    return out


def qlevels(pdf, p=(10, 30, 50, 70, 90, 95, 99, 99.9), x1=None, x2=None):
    """QLEVELS Calculates quantile levels which encloses P% of PDF.

      CALL: [ql PL] = qlevels(pdf,PL,x1,x2);

            ql    = the discrete quantile levels.
            pdf   = joint point density function matrix or vector
            PL    = percent level (default [10:20:90 95 99 99.9])
            x1,x2 = vectors of the spacing of the variables
                   (Default unit spacing)

    QLEVELS numerically integrates PDF by decreasing height and find the
    quantile levels which  encloses P% of the distribution. If X1 and
    (or) X2 is unspecified it is assumed that dX1 and dX2 is constant.
    NB! QLEVELS normalizes the integral of PDF to N/(N+0.001) before
    calculating QL in order to reflect the sampling of PDF is finite.
    Currently only able to handle 1D and 2D PDF's if dXi is not constant
    (i=1,2).

    Example
    -------
    >>> import wafo.stats as ws
    >>> x = np.linspace(-8,8,2001);
    >>> PL = np.r_[10:90:20, 90, 95, 99, 99.9]
    >>> qlevels(ws.norm.pdf(x),p=PL, x1=x);
    array([ 0.39591707,  0.37058719,  0.31830968,  0.23402133,  0.10362052,
            0.05862129,  0.01449505,  0.00178806])

    # compared with the exact values
    >>> ws.norm.pdf(ws.norm.ppf((100-PL)/200))
    array([ 0.39580488,  0.370399  ,  0.31777657,  0.23315878,  0.10313564,
            0.05844507,  0.01445974,  0.00177719])

    See also
    --------
    qlevels2, tranproc

    """

    norm = 1  # normalize cdf to unity
    pdf = np.atleast_1d(pdf)
    if any(pdf.ravel() < 0):
        raise ValueError(
            'This is not a pdf since one or more values of pdf is negative')

    fsiz = pdf.shape
    fsizmin = min(fsiz)
    if fsizmin == 0:
        return []

    N = np.prod(fsiz)
    d = len(fsiz)
    if x1 is None or ((x2 is None) and d > 2):
        fdfi = pdf.ravel()
    else:
        if d == 1:  # pdf in one dimension
            dx22 = np.ones(1)
        else:  # % pdf in two dimensions
            dx2 = np.diff(x2.ravel()) * 0.5
            dx22 = np.r_[0, dx2] + np.r_[dx2, 0]

        dx1 = np.diff(x1.ravel()) * 0.5
        dx11 = np.r_[0, dx1] + np.r_[dx1, 0]
        dx1x2 = dx22[:, None] * dx11
        fdfi = (pdf * dx1x2).ravel()

    p = np.atleast_1d(p)

    if np.any((p < 0) | (100 < p)):
        raise ValueError('PL must satisfy 0 <= PL <= 100')

    p2 = p / 100.0
    ind = np.argsort(pdf.ravel())  # sort by height of pdf
    ind = ind[::-1]
    fi = pdf.flat[ind]

    # integration in the order of decreasing height of pdf
    Fi = np.cumsum(fdfi[ind])

    if norm:  # %normalize Fi to make sure int pdf dx1 dx2 approx 1
        Fi = Fi / Fi[-1] * N / (N + 1.5e-8)

    maxFi = np.max(Fi)
    if maxFi > 1:
        warnings.warn('this is not a pdf since cdf>1! normalizing')

        Fi = Fi / Fi[-1] * N / (N + 1.5e-8)

    elif maxFi < .95:
        msg = '''The given pdf is too sparsely sampled since cdf<.95.
        Thus QL is questionable'''
        warnings.warn(msg)

    # make sure Fi is strictly increasing by not considering duplicate values
    ind, = np.where(np.diff(np.r_[Fi, 1]) > 0)
    # calculating the inverse of Fi to find the index
    ui = tranproc(Fi[ind], fi[ind], p2)
    # to the desired quantile level
    # ui=smooth(Fi(ind),fi(ind),1,p2(:),1) % alternative
    # res=ui-ui2

    if np.any(ui >= max(pdf.ravel())):
        warnings.warn('The lowest percent level is too close to 0%')

    if np.any(ui <= min(pdf.ravel())):
        msg = '''The given pdf is too sparsely sampled or
       the highest percent level is too close to 100%'''
        warnings.warn(msg)
        ui[ui < 0] = 0.0

    return ui


def qlevels2(data, p=(10, 30, 50, 70, 90, 95, 99, 99.9), method=1):
    """QLEVELS2 Calculates quantile levels which encloses P% of data.

     CALL: [ql PL] = qlevels2(data,PL,method);

       ql   = the discrete quantile levels, size D X Np
    Parameters
    ----------
    data : data matrix, size D x N (D = # of dimensions)
    p : percent level vector, length Np (default [10:20:90 95 99 99.9])
    method : integer
        1 Interpolation so that F(X_(k)) == (k-0.5)/n. (default)
        2 Interpolation so that F(X_(k)) == k/(n+1).
        3 Based on the empirical distribution.

    Returns
    -------

    QLEVELS2 sort the columns of data in ascending order and find the
             quantile levels for each column which encloses  P% of the data.

    Examples :  Finding quantile levels enclosing P% of data:
    --------
    >>> import wafo.stats as ws
    >>> PL = np.r_[10:90:20, 90, 95, 99, 99.9]
    >>> xs = ws.norm.rvs(size=2500000)
    >>> np.allclose(qlevels2(ws.norm.pdf(xs), p=PL),
    ...  [0.3958, 0.3704, 0.3179, 0.2331, 0.1031, 0.05841, 0.01451, 0.001751],
    ...   rtol=1e-1)
    True

    # compared with the exact values
    >>> ws.norm.pdf(ws.norm.ppf((100-PL)/200))
    array([ 0.39580488,  0.370399  ,  0.31777657,  0.23315878,  0.10313564,
            0.05844507,  0.01445974,  0.00177719])

    # Finding the median of xs:
    >>> '%2.2f' % np.abs(qlevels2(xs,50)[0])
    '0.00'

    See also
    --------
    qlevels

    """
    q = 100 - np.atleast_1d(p)
    return percentile(data, q, axis=-1, method=method)


_PKDICT = {1: lambda k, w, n: (k - w) / (n - 1),
           2: lambda k, w, n: (k - w / 2) / n,
           3: lambda k, w, n: k / n,
           4: lambda k, w, n: k / (n + 1),
           5: lambda k, w, n: (k - w / 3) / (n + 1 / 3),
           6: lambda k, w, n: (k - w * 3 / 8) / (n + 1 / 4)}


def _compute_qth_weighted_percentile(a, q, axis, out, method, weights,
                                     overwrite_input):
    # normalise weight vector such that sum of the weight vector equals to n
    q = np.atleast_1d(q) / 100.0
    if (q < 0).any() or (q > 1).any():
        raise ValueError("percentile must be in the range [0,100]")

    shape0 = a.shape
    if axis is None:
        sorted_ = a.ravel()
    else:
        taxes = range(a.ndim)
        taxes[-1], taxes[axis] = taxes[axis], taxes[-1]
        sorted_ = np.transpose(a, taxes).reshape(-1, shape0[axis])

    ind = sorted_.argsort(axis=-1)
    if overwrite_input:
        sorted_.sort(axis=-1)
    else:
        sorted_ = np.sort(sorted_, axis=-1)

    w = np.atleast_1d(weights)
    n = len(w)
    w = w * n / w.sum()

    # Work on each column separately because of weight vector
    m = sorted_.shape[0]
    nq = len(q)
    y = np.zeros((m, nq))
    pk_fun = _PKDICT.get(method, 1)
    for i in range(m):
        sortedW = w[ind[i]]            # rearrange the weight according to ind
        k = sortedW.cumsum()           # cumulative weight
        # different algorithm to compute percentile
        pk = pk_fun(k, sortedW, n)
        # Interpolation between pk and sorted_ for given value of q
        y[i] = np.interp(q, pk, sorted_[i])
    if axis is None:
        return np.squeeze(y)
    else:
        shape1 = list(shape0)
        shape1[axis], shape1[-1] = shape1[-1], nq
        return np.squeeze(np.transpose(y.reshape(shape1), taxes))

# method=1: p(k) = k/(n-1)
# method=2: p(k) = (k+0.5)/n.
# method=3: p(k) = (k+1)/n
# method=4: p(k) = (k+1)/(n+1)
# method=5: p(k) = (k+2/3)/(n+1/3)
# method=6: p(k) = (k+5/8)/(n+1/4)

_KDICT = {1: lambda p, n: p * (n - 1),
          2: lambda p, n: p * n - 0.5,
          3: lambda p, n: p * n - 1,
          4: lambda p, n: p * (n + 1) - 1,
          5: lambda p, n: p * (n + 1. / 3) - 2. / 3,
          6: lambda p, n: p * (n + 1. / 4) - 5. / 8}


def _compute_qth_percentile(sorted_, q, axis, out, method):
    if not np.isscalar(q):
        p = [_compute_qth_percentile(sorted_, qi, axis, None, method)
             for qi in q]
        if out is not None:
            out.flat = p
        return p

    q = q / 100.0
    if (q < 0) or (q > 1):
        raise ValueError("percentile must be in the range [0,100]")

    indexer = [slice(None)] * sorted_.ndim
    Nx = sorted_.shape[axis]
    k_fun = _KDICT.get(method, 1)
    index = np.clip(k_fun(q, Nx), 0, Nx - 1)
    i = int(index)
    if i == index:
        indexer[axis] = slice(i, i + 1)
        weights1 = np.array(1)
        sumval = 1.0
    else:
        indexer[axis] = slice(i, i + 2)
        j = i + 1
        weights1 = np.array([(j - index), (index - i)], float)
        wshape = [1] * sorted_.ndim
        wshape[axis] = 2
        weights1.shape = wshape
        sumval = weights1.sum()

    # Use add.reduce in both cases to coerce data type as well as
    # check and use out array.
    return np.add.reduce(sorted_[indexer] * weights1,
                         axis=axis, out=out) / sumval


def percentile(a, q, axis=None, out=None, overwrite_input=False, method=1,
               weights=None):
    """Compute the qth percentile of the data along the specified axis.

    Returns the qth percentile of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : float in range of [0,100] (or sequence of floats)
        percentile to compute which must be between 0 and 100 inclusive
    axis : {None, int}, optional
        Axis along which the percentiles are computed. The default (axis=None)
        is to compute the median along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : {False, True}, optional
       If True, then allow use of memory of input array (a) for
       calculations. The input array will be modified by the call to
       median. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. Note that, if `overwrite_input` is True and the input
       is not already an ndarray, an error will be raised.
    method : scalar integer
        defining the interpolation method. Valid options are
        1 : p[k] = k/(n-1). In this case, p[k] = mode[F(x[k])].
                 This is used by S. (default)
        2 : p[k] = (k+0.5)/n. That is a piecewise linear function where
                 the knots are the values midway through the steps of the
                 empirical cdf. This is popular amongst hydrologists.
                 Matlab also uses this formula.
        3 : p[k] = (k+1)/n. That is, linear interpolation of the empirical cdf.
        4 : p[k] = (k+1)/(n+1). Thus p[k] = E[F(x[k])].
                 This is used by Minitab and by SPSS.
        5 : p[k] = (k+2/3)/(n+1/3). Then p[k] =~ median[F(x[k])].
                 The resulting quantile estimates are approximately
                 median-unbiased regardless of the distribution of x.
        6 : p[k] = (k+5/8)/(n+1/4). The resulting quantile estimates are
                 approximately unbiased for the expected order statistics
                 if x is normally distributed.

    Returns
    -------
    pcntile : ndarray
        A new array holding the result (unless `out` is specified, in
        which case that array is returned instead).  If the input contains
        integers, or floats of smaller precision than 64, then the output
        data-type is float64.  Otherwise, the output data-type is the same
        as that of the input.

    See Also
    --------
    mean, median

    Notes
    -----
    Given a vector V of length N, the qth percentile of V is the qth ranked
    value in a sorted copy of V.  A weighted average of the two nearest
    neighbors is used if the normalized ranking does not match q exactly.
    The same as the median if q is 0.5; the same as the min if q is 0;
    and the same as the max if q is 1

    Examples
    --------
    >>> import wafo.kdetools as wk
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> wk.percentile(a, 50)
    3.5
    >>> wk.percentile(a, 50, axis=0)
    array([ 6.5,  4.5,  2.5])
    >>> wk.percentile(a, 50, axis=0, weights=np.ones(2))
    array([ 6.5,  4.5,  2.5])
    >>> wk.percentile(a, 50, axis=1)
    array([ 7.,  2.])
    >>> wk.percentile(a, 50, axis=1, weights=np.ones(3))
    array([ 7.,  2.])
    >>> m = wk.percentile(a, 50, axis=0)
    >>> out = np.zeros_like(m)
    >>> wk.percentile(a, 50, axis=0, out=m)
    array([ 6.5,  4.5,  2.5])
    >>> m
    array([ 6.5,  4.5,  2.5])
    >>> b = a.copy()
    >>> wk.percentile(b, 50, axis=1, overwrite_input=True)
    array([ 7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> wk.percentile(b, 50, axis=None, overwrite_input=True)
    3.5
    >>> np.all(a==b)
    True

    """
    a = np.asarray(a)
    try:
        if q == 0:
            return a.min(axis=axis, out=out)
        elif q == 100:
            return a.max(axis=axis, out=out)
    except:
        pass
    if weights is not None:
        return _compute_qth_weighted_percentile(a, q, axis, out, method,
                                                weights, overwrite_input)
    elif overwrite_input:
        if axis is None:
            sorted_ = np.sort(a, axis=axis)
        else:
            a.sort(axis=axis)
            sorted_ = a
    else:
        sorted_ = np.sort(a, axis=axis)
    if axis is None:
        axis = 0

    return _compute_qth_percentile(sorted_, q, axis, out, method)


def iqrange(data, axis=None):
    """Returns the Inter Quartile Range of data.

    Parameters
    ----------
    data : array-like
        Input array or object that can be converted to an array.
    axis : {None, int}, optional
        Axis along which the percentiles are computed. The default (axis=None)
        is to compute the median along a flattened version of the array.

    Returns
    -------
    r : array-like
        abs(np.percentile(data, 75, axis)-np.percentile(data, 25, axis))

    Notes
    -----
    IQRANGE is a robust measure of spread. The use of interquartile range
    guards against outliers if the distribution have heavy tails.

    Example
    -------
    >>> a = np.arange(101)
    >>> iqrange(a)
    50.0

    See also
    --------
    np.std

    """
    return np.abs(np.percentile(data, 75, axis=axis) -
                  np.percentile(data, 25, axis=axis))


def bitget(int_type, offset):
    """Returns the value of the bit at the offset position in int_type.

    Example
    -------
    >>> bitget(5, np.r_[0:4])
    array([1, 0, 1, 0])

    """
    return np.bitwise_and(int_type, 1 << offset) >> offset


def gridcount(data, X, y=1):
    '''
    Returns D-dimensional histogram using linear binning.

    Parameters
    ----------
    data = column vectors with D-dimensional data, shape D x Nd
    X    = row vectors defining discretization, shape D x N
            Must include the range of the data.

    Returns
    -------
    c    = gridcount,  shape N x N x ... x N

    GRIDCOUNT obtains the grid counts using linear binning.
    There are 2 strategies: simple- or linear- binning.
    Suppose that an observation occurs at x and that the nearest point
    below and above is y and z, respectively. Then simple binning strategy
    assigns a unit weight to either y or z, whichever is closer. Linear
    binning, on the other hand, assigns the grid point at y with the weight
    of (z-x)/(z-y) and the gridpoint at z a weight of (y-x)/(z-y).

    In terms of approximation error of using gridcounts as pdf-estimate,
    linear binning is significantly more accurate than simple binning.

     NOTE: The interval [min(X);max(X)] must include the range of the data.
           The order of C is permuted in the same order as
           meshgrid for D==2 or D==3.

    Example
    -------
    >>> import numpy as np
    >>> import wafo.kdetools as wk
    >>> import pylab as plb
    >>> N = 200
    >>> data  = np.random.rayleigh(1,N)
    >>> x = np.linspace(0,max(data)+1,50)
    >>> dx = x[1]-x[0]

    >>> c = wk.gridcount(data,x)

    >>> h = plb.plot(x,c,'.')   # 1D histogram
    >>> pdf = c/dx/N
    >>> h1 = plb.plot(x, pdf) #  1D probability density plot
    >>> '%1.2f' % np.trapz(pdf, x)
    '1.00'

    See also
    --------
    bincount, accum, kdebin

    Reference
    ----------
    Wand,M.P. and Jones, M.C. (1995)
    'Kernel smoothing'
    Chapman and Hall, pp 182-192
    '''
    dat = np.atleast_2d(data)
    x = np.atleast_2d(X)
    y = np.atleast_1d(y).ravel()
    d = dat.shape[0]
    d1, inc = x.shape

    if d != d1:
        raise ValueError('Dimension 0 of data and X do not match.')

    dx = np.diff(x[:, :2], axis=1)
    xlo = x[:, 0]
    xup = x[:, -1]

    datlo = dat.min(axis=1)
    datup = dat.max(axis=1)
    if ((datlo < xlo) | (xup < datup)).any():
        raise ValueError('X does not include whole range of the data!')

    csiz = np.repeat(inc, d)
    use_sparse = False
    if use_sparse:
        acfun = accumsum  # faster than accum
    else:
        acfun = accumsum2  # accum

    binx = np.asarray(np.floor((dat - xlo[:, newaxis]) / dx), dtype=int)
    w = dx.prod()
    abs = np.abs  # @ReservedAssignment
    if d == 1:
        x.shape = (-1,)
        c = np.asarray((acfun(binx, (x[binx + 1] - dat) * y, size=(inc, )) +
                        acfun(binx + 1, (dat - x[binx]) * y, size=(inc, ))) /
                       w).ravel()
    else:  # d>2

        Nc = csiz.prod()
        c = np.zeros((Nc,))

        fact2 = np.asarray(np.reshape(inc * np.arange(d), (d, -1)), dtype=int)
        fact1 = np.asarray(
            np.reshape(csiz.cumprod() / inc, (d, -1)), dtype=int)
        # fact1 = fact1(ones(n,1),:);
        bt0 = [0, 0]
        X1 = X.ravel()
        for ir in range(2 ** (d - 1)):
            bt0[0] = np.reshape(bitget(ir, np.arange(d)), (d, -1))
            bt0[1] = 1 - bt0[0]
            for ix in range(2):
                one = np.mod(ix, 2)
                two = np.mod(ix + 1, 2)
                # Convert to linear index
                # linear index to c
                b1 = np.sum((binx + bt0[one]) * fact1, axis=0)
                bt2 = bt0[two] + fact2
                b2 = binx + bt2                     # linear index to X
                c += acfun(
                    b1, abs(np.prod(X1[b2] - dat, axis=0)) * y, size=(Nc,))

        c = np.reshape(c / w, csiz, order='F')

        T = range(d)
        T[1], T[0] = T[0], T[1]
        # make sure c is stored in the same way as meshgrid
        c = c.transpose(*T)
    return c


def kde_demo1():
    """KDEDEMO1 Demonstrate the smoothing parameter impact on KDE.

    KDEDEMO1 shows the true density (dotted) compared to KDE based on 7
    observations (solid) and their individual kernels (dashed) for 3
    different values of the smoothing parameter, hs.

    """

    import scipy.stats as st
    x = np.linspace(-4, 4, 101)
    x0 = x / 2.0
    data = np.random.normal(loc=0, scale=1.0, size=7)
    kernel = Kernel('gauss')
    hs = kernel.hns(data)
    hVec = [hs / 2, hs, 2 * hs]

    for ix, h in enumerate(hVec):
        plt.figure(ix)
        kde = KDE(data, hs=h, kernel=kernel)
        f2 = kde(x, output='plot', title='h_s = %2.2f' % h, ylab='Density')
        f2.plot('k-')

        plt.plot(x, st.norm.pdf(x, 0, 1), 'k:')
        n = len(data)
        plt.plot(data, np.zeros(data.shape), 'bx')
        y = kernel(x0) / (n * h * kernel.norm_factor(d=1, n=n))
        for i in range(n):
            plt.plot(data[i] + x0 * h, y, 'b--')
            plt.plot([data[i], data[i]], [0, np.max(y)], 'b')

        plt.axis([x.min(), x.max(), 0, 0.5])


def kde_demo2():
    '''Demonstrate the difference between transformation- and ordinary-KDE.

    KDEDEMO2 shows that the transformation KDE is a better estimate for
    Rayleigh distributed data around 0 than the ordinary KDE.
    '''
    import scipy.stats as st
    data = st.rayleigh.rvs(scale=1, size=300)

    x = np.linspace(1.5e-2, 5, 55)

    kde = KDE(data)
    f = kde(output='plot', title='Ordinary KDE (hs=%g)' % kde.hs)
    plt.figure(0)
    f.plot()

    plt.plot(x, st.rayleigh.pdf(x, scale=1), ':')

    # plotnorm((data).^(L2)) % gives a straight line => L2 = 0.5 reasonable

    tkde = TKDE(data, L2=0.5)
    ft = tkde(x, output='plot', title='Transformation KDE (hs=%g)' %
              tkde.tkde.hs)
    plt.figure(1)
    ft.plot()

    plt.plot(x, st.rayleigh.pdf(x, scale=1), ':')

    plt.figure(0)


def kde_demo3():
    '''Demonstrate the difference between transformation and ordinary-KDE in 2D

    KDEDEMO3 shows that the transformation KDE is a better estimate for
    Rayleigh distributed data around 0 than the ordinary KDE.
    '''
    import scipy.stats as st
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
    import scipy.stats as st

    data = np.hstack((st.norm.rvs(loc=5, scale=1, size=(N,)),
                      st.norm.rvs(loc=-5, scale=1, size=(N,))))

    # x = np.linspace(1.5e-3, 5, 55)

    kde = KDE(data, kernel=Kernel('gauss', 'hns'))
    f = kde(output='plot', title='Ordinary KDE', plotflag=1)

    kde1 = KDE(data, kernel=Kernel('gauss', 'hisj'))
    f1 = kde1(output='plot', label='Ordinary KDE', plotflag=1)

    plt.figure(0)
    f.plot('r', label='hns=%g' % kde.hs)
    # plt.figure(2)
    f1.plot('b', label='hisj=%g' % kde1.hs)
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
    import scipy.stats as st

    data = np.hstack((st.norm.rvs(loc=5, scale=1, size=(2, N,)),
                      st.norm.rvs(loc=-5, scale=1, size=(2, N,))))
    kde = KDE(data, kernel=Kernel('gauss', 'hns'))
    f = kde(output='plot', title='Ordinary KDE (hns=%g %g)' %
            tuple(kde.hs.tolist()), plotflag=1)

    kde1 = KDE(data, kernel=Kernel('gauss', 'hisj'))
    f1 = kde1(output='plot', title='Ordinary KDE (hisj=%g %g)' %
              tuple(kde1.hs.tolist()), plotflag=1)

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

    print(kreg.tkde.tkde.inv_hs)
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
    import scipy.stats as st
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
    st = stats

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
        c0 = np.zeros(xi.shape)
    yi = np.where(c == 0, 0, c0 / c)

    kreg = KRegression(x, y, hs=hs, p=0)
    fiii = kreg(xiii)
    yiii = interpolate.interp1d(xi, yi)(xiii)
    fit = fun1(xiii).clip(max=1.0)
    df = np.diff(fiii)
    eerr = np.abs((yiii - fiii)).std() + 0.5 * (df[:-1] * df[1:] < 0).sum() / n
    err = (fiii - fit).std()
    f = kreg(
        xiii, output='plotobj',
        title='%s err=%1.3f,eerr=%1.3f, n=%d, hs=%1.3f, hs1=%1.3f, hs2=%1.3f' %
        (fun, err, eerr, n, hs, hs1, hs2), plotflag=1)

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

    fg.plot(label='KReg grid aic=%2.3f' % (aic))
    f.plot(label='KReg averr=%2.3f ' % (averr))
    labtxt = '%d CI' % (int(100 * (1 - alpha)))
    plt.fill_between(xiii, pup, plo, alpha=0.20,
                     color='r', linestyle='--', label=labtxt)
    plt.fill_between(xiii, pup2, plo2, alpha=0.20, color='b',
                     linestyle=':', label='%d CI2' % (int(100 * (1 - alpha))))
    plt.plot(xiii, fun1(xiii), 'r', label='True model')
    plt.scatter(xi, yi, label='data')
    print('maxp = %g' % (np.nanmax(f.data)))
    print('hs = %g' % (kreg.tkde.tkde.hs))
    plt.legend()
    h = plt.gca()
    if plotlog:
        plt.setp(h, yscale='log')
    # plt.show()
    return hs1, hs2


def kreg_demo4(x, y, hs, hopt, alpha=0.05):
    st = stats

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
        c0 = np.zeros(xi.shape)
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
    f.labels.title = 'perr=%1.3f,aicc=%1.3f, n=%d, hs=%1.3f' % (
        f.prediction_error_avg, aicc, n, hs)

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
        fbest.labels.title = 'N = %d' % n
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
        fbest.labels.title = 'N = %d' % n
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
        c0 = np.zeros(xi.shape)
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

    st = stats
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
    f.labels.title = 'perr=%1.3f,aicc=%1.3f, n=%d, hs=%1.3f' % (
        f.prediction_error_avg, aicc, n, hs)

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

    st = stats
    # x = np.linspace(-4, 4, 101)
    # data = np.random.normal(loc=0, scale=1.0, size=n)
    # data = np.random.exponential(scale=1.0, size=n)
#    n1 = 128
#    I = (np.arange(n1)*pi)**2 *0.01*0.5
#    kw = exp(-I)
#    plt.plot(idctn(kw))
#    return
    # dist = st.norm
    dist = st.expon
    data = dist.rvs(loc=0, scale=1.0, size=n)
    d, _N = np.atleast_2d(data).shape

    if d == 1:
        plot_options = [dict(color='red'), dict(
            color='green'), dict(color='black')]
    else:
        plot_options = [dict(colors='red'), dict(colors='green'),
                        dict(colors='black')]

    plt.figure(1)
    kde0 = KDE(data, kernel=Kernel('gauss', 'hste'))
    f0 = kde0.eval_grid_fast(output='plot', ylab='Density')
    f0.plot(**plot_options[0])

    kde1 = TKDE(data, kernel=Kernel('gauss', 'hisj'), L2=.5)
    f1 = kde1.eval_grid_fast(output='plot', ylab='Density')
    f1.plot(**plot_options[1])

    kde2 = KDEgauss(data)
    f2 = kde2(output='plot', ylab='Density')
    x = f2.args
    f2.plot(**plot_options[2])

    fmax = dist.pdf(x, 0, 1).max()
    if d == 1:
        plt.plot(x, dist.pdf(x, 0, 1), 'k:')
        plt.axis([x.min(), x.max(), 0, fmax])
    plt.show()
    print(fmax / f2.data.max())
    format_ = ''.join(('%g, ') * d)
    format_ = 'hs0=%s hs1=%s hs2=%s' % (format_, format_, format_)
    print(format_ % tuple(kde0.hs.tolist() +
                          kde1.tkde.hs.tolist() + kde2.hs.tolist()))
    print('inc0 = %d, inc1 = %d, inc2 = %d' % (kde0.inc, kde1.inc, kde2.inc))


def test_kde():
    data = np.array([
        0.75355792, 0.72779194, 0.94149169, 0.07841119, 2.32291887,
        1.10419995, 0.77055114, 0.60288273, 1.36883635, 1.74754326,
        1.09547561, 1.01671133, 0.73211143, 0.61891719, 0.75903487,
        1.8919469, 0.72433808, 1.92973094, 0.44749838, 1.36508452])

    x = np.linspace(0.01, max(data.ravel()) + 1, 10)
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


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    test_docstrings()
    # test_kde()
    # check_bkregression()
    # check_regression_bin()
    # check_kreg_demo3()
    # check_kreg_demo4()

    # test_smoothn_1d()
    # test_smoothn_2d()

    # kde_demo2()
    # kreg_demo1(fast=True)
    # kde_gauss_demo()
    # kreg_demo2(n=120,symmetric=True,fun='hste', plotlog=True)
    # plt.show('hold')
