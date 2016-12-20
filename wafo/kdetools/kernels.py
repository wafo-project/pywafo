'''
Created on 15. des. 2016

@author: pab
'''
from __future__ import division
from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
from numpy import pi, sqrt, exp, percentile
from scipy import optimize, linalg
from scipy.special import gamma
from wafo.misc import tranproc  # , trangood
from wafo.kdetools.gridding import gridcount
from wafo.dctpack import dct
from wafo.testing import test_docstrings

__all__ = ['Kernel', 'sphere_volume', 'qlevels', 'iqrange', 'percentile']


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)

# stats = (mu2, R, Rdd) where
#     mu2 : 2'nd order moment, i.e.,int(x^2*kernel(x))
#     R :  integral of squared kernel, i.e., int(kernel(x)^2)
#     Rdd  : int( (kernel''(x))^2 ).
_stats_epan = (1. / 5, 3. / 5, np.inf)
_stats_biwe = (1. / 7, 5. / 7, 45. / 2)
_stats_triw = (1. / 9, 350. / 429, np.inf)
_stats_rect = (1. / 3, 1. / 2, np.inf)
_stats_tria = (1. / 6, 2. / 3, np.inf)
_stats_lapl = (2, 1. / 4, np.inf)
_stats_logi = (pi ** 2 / 3, 1. / 6, 1 / 42)
_stats_gaus = (1, 1. / (2 * sqrt(pi)), 3. / (8 * sqrt(pi)))


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
    _assert(not any(pdf.ravel() < 0), 'This is not a pdf since one or more '
            'values of pdf is negative')

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
    _assert(not np.any((p < 0) | (100 < p)), 'PL must satisfy 0 <= PL <= 100')

    p2 = p / 100.0
    ind = np.argsort(pdf.ravel())  # sort by height of pdf
    ind = ind[::-1]
    fi = pdf.flat[ind]

    # integration in the order of decreasing height of pdf
    Fi = np.cumsum(fdfi[ind])

    if norm:  # normalize Fi to make sure int pdf dx1 dx2 approx 1
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
        1 Interpolation so that F(X_[k]) == k/(n-1). (linear default)
        2 Interpolation so that F(X_[k]) == (k+0.5)/n. (midpoint)
        3 Interpolation so that F(X_[k]) == (k+1)/n.   (lower)
        4 Interpolation so that F(X_[k]) == k/n.       (higher)

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
    _assert(0 < method < 5,
            'Method must be between 1 to 4. Got method={}.'.format(method))
    interpolation = ['', 'linear', 'midpoint', 'lower', 'higher'][method]
    q = 100 - np.atleast_1d(p)
    return percentile(data, q, axis=-1, interpolation=interpolation)


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


class _Kernel(object):
    __metaclass__ = ABCMeta

    def __init__(self, r=1.0, stats=None):
        self.r = r  # radius of kernel
        self.stats = stats

    def norm_factor(self, d=1, n=None):
        _assert(0 < d, "D")
        _assert(0 < n, "Number of samples too few (n={})".format(n))
        return 1.0

    @abstractmethod
    def _kernel(self, x):
        pass

    def norm_kernel(self, x):
        X = np.atleast_2d(x)
        return self._kernel(X) / self.norm_factor(*X.shape)

    def kernel(self, x):
        return self._kernel(np.atleast_2d(x))

    def deriv4_6_8_10(self, t, numout=4):
        raise NotImplementedError('Method not implemented for this kernel!')

    def effective_support(self):
        """Return the effective support of kernel.

        The kernel must be symmetric and compactly supported on [-tau tau]
        if the kernel has infinite support then the kernel must have the
        effective support in [-tau tau], i.e., be negligible outside the range

        """
        return self._effective_support()

    def _effective_support(self):
        return -self.r, self.r
    __call__ = kernel


class _KernelMulti(_Kernel):
    """
    p=0;  Sphere = rect for 1D
    p=1;  Multivariate Epanechnikov kernel.
    p=2;  Multivariate Bi-weight Kernel
    p=3;  Multi variate Tri-weight Kernel
    p=4;  Multi variate Four-weight Kernel
    """
    def __init__(self, r=1.0, p=1, stats=None):
        self.p = p
        super(_KernelMulti, self).__init__(r, stats)

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
    """
    p=0;  rectangular
    p=1;  1D product Epanechnikov kernel.
    p=2;  1D product Bi-weight Kernel
    p=3;  1D product Tri-weight Kernel
    p=4;  1D product Four-weight Kernel
    """
    def norm_factor(self, d=1, n=None):
        r = self.r
        p = self.p
        c = (2 ** p * np.prod(np.r_[1:p + 1]) * sphere_volume(1, r) /
             np.prod(np.r_[(1 + 2):(2 * p + 2):2]))
        return c ** d

    def _kernel(self, x):
        r = self.r  # radius
        pdf = (1 - (x / r) ** 2).clip(min=0.0) ** self.p
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
        for _i in range(numout - 1):
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
        s = exp(x)
        return np.prod(s / (s + 1) ** 2, axis=0)
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
        self.get_smoothing = getattr(self, fun)

    @property
    def name(self):
        return self.kernel.__class__.__name__.replace('_Kernel', '').title()

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

        a = np.atleast_2d(data)
        n = a.shape[1]

        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, _Rdd = self.stats()
        amise_constant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        iqr = iqrange(a, axis=1)  # interquartile range
        stdA = np.std(a, axis=1, ddof=1)
        # use of interquartile range guards against outliers.
        # the use of interquartile range is better if
        # the distribution is skew or have heavy tails
        # This lessen the chance of oversmoothing.
        return np.where(iqr > 0,
                        np.minimum(stdA, iqr / 1.349), stdA) * amise_constant

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

    def _hmns_scale(self, d):
        name = self.name[:4].lower()
        if name == 'epan':  # Epanechnikov kernel
            a = (8.0 * (d + 4.0) * (2 * sqrt(pi)) ** d /
                 sphere_volume(d)) ** (1. / (4.0 + d))
        elif name == 'biwe':  # Bi-weight kernel
            a = 2.7779
            if d > 2:
                raise NotImplementedError('Not implemented for d>2')
        elif name == 'triw':  # Triweight
            a = 3.12
            if d > 2:
                raise NotImplementedError('not implemented for d>2')
        elif name == 'gaus':  # Gaussian kernel
            a = (4.0 / (d + 2.0)) ** (1. / (d + 4.0))
        else:
            raise ValueError('Unknown kernel.')
        return a

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

        a = np.atleast_2d(data)
        d, n = a.shape
        if d == 1:
            return self.hns(data)
        scale = self._hmns_scale(d)
        cov_a = np.cov(a)
        return scale * linalg.sqrtm(cov_a).real * n ** (-1. / (d + 4))

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

        # R = int(mkernel(x)^2),  mu2 = int(x^2*mkernel(x))
        mu2, R, _Rdd = self.stats()

        amise_constant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        ste_constant = R / (mu2 ** (2) * n)

        sigmaA = self.hns(A) / amise_constant
        if h0 is None:
            h0 = sigmaA * amise_constant

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
        mu2, R, _Rdd = kernel2.stats()
        ste_constant2 = R / (mu2 ** (2) * n)
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
                gamma_ = ((2 * k40 * mu2 * psi4 * h1 ** 5) /
                          (-psi6 * R)) ** (1.0 / 7)

                # Now estimate psi4 given gamma_.
                # kernel weights.
                kw4 = kernel2.deriv4_6_8_10(xn / gamma_, numout=1)
                kw = np.r_[kw4, 0, kw4[-1:0:-1]]  # Apply 'fftshift' to kw.
                z = np.real(ifft(fft(c, nfft) * fft(kw)))  # convolution.

                psi4Gamma = np.sum(c * z[:inc]) / (n * (n - 1) * gamma_ ** 5)

                # Step 4
                h1 = (ste_constant2 / psi4Gamma) ** (1.0 / 5)

            # Kernel other than Gaussian scale bandwidth
            h1 = h1 * (ste_constant / ste_constant2) ** (1.0 / 5)

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

        # R = int(mkernel(x)^2),  mu2 = int(x^2*mkernel(x))
        mu2, R, _Rdd = self.stats()
        ste_constant = R / (n * mu2 ** 2)

        amin = A.min(axis=1)  # Find the minimum value of A.
        amax = A.max(axis=1)  # Find the maximum value of A.
        arange = amax - amin  # Find the range of A.

        # xa holds the x 'axis' vector, defining a grid of x values where
        # the k.d. function will be evaluated.

        ax1 = amin - arange / 8.0
        bx1 = amax + arange / 8.0

        kernel2 = Kernel('gauss')
        mu2, R, _Rdd = kernel2.stats()
        ste_constant2 = R / (mu2 ** (2) * n)

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
            h[dim] = bandwidth * (ste_constant / ste_constant2) ** (1.0 / 5)
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
        mu2, R, _Rdd = self.stats()

        amise_constant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        ste_constant = R / (mu2 ** (2) * n)

        sigmaA = self.hns(A) / amise_constant
        if h0 is None:
            h0 = sigmaA * amise_constant

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
                h1 = (ste_constant / psi4) ** (1. / 5)

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
        mu2, R, _Rdd = self.stats()

        amise_constant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        ste_constant = R / (mu2 ** (2) * n)

        sigmaA = self.hns(A) / amise_constant
        if hvec is None:
            H = amise_constant / 0.93
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
        mu2, R, _Rdd = kernel2.stats()
        ste_constant2 = R / (mu2 ** (2) * n)
        fft = np.fft.fft
        ifft = np.fft.ifft

        h = np.zeros(d)
        hvec = hvec * (ste_constant2 / ste_constant) ** (1. / 5.)

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
            h[dim] = hvec[idx] * (ste_constant / ste_constant2) ** (1 / 5)
            if idx == 0:
                warnings.warn("Optimum is probably lower than "
                              "hs={0:g} for dim={1:d}".format(h[dim] * s, dim))
            elif idx == maxit - 1:
                msg = "Optimum is probably higher than hs={0:g] for dim={1:d}"
                warnings.warn(msg.format(h[dim] * s, dim))

        hvec = hvec * (ste_constant / ste_constant2) ** (1 / 5)
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
        mu2, R, _Rdd = self.stats()

        amise_constant = (8 * sqrt(pi) * R / (3 * n * mu2 ** 2)) ** (1. / 5)
        ste_constant = R / (n * mu2 ** 2)

        sigmaA = self.hns(A) / amise_constant

        nfft = inc * 2
        amin = A.min(axis=1)  # Find the minimum value of A.
        amax = A.max(axis=1)  # Find the maximum value of A.
        arange = amax - amin  # Find the range of A.

        # xa holds the x 'axis' vector, defining a grid of x values where
        # the k.d. function will be evaluated.

        ax1 = amin - arange / 8.0
        bx1 = amax + arange / 8.0

        kernel2 = Kernel('gauss')
        mu2, _R, _Rdd = kernel2.stats()

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
            psi_r = (-1) ** (rd2) * np.prod(
                np.r_[rd2 + 1:r + 1]) / (sqrt(pi) * (2 * s) ** (r + 1))
            psi = psi_r
            if L > 0:
                # High order derivatives of the Gaussian kernel
                Kd = kernel2.deriv4_6_8_10(0, numout=L)

                # L-stage iterations to estimate PSI_4
                for ix in range(L, 0, -1):
                    gi = (-2 * Kd[ix - 1] /
                          (mu2 * psi * n)) ** (1. / (2 * ix + 5))

                    # Obtain the kernel weights.
                    kw0 = kernel2.deriv4_6_8_10(xn / gi, numout=ix)
                    if ix > 1:
                        kw0 = kw0[-1]
                    # Apply 'fftshift' to kw.
                    kw = np.r_[kw0, 0, kw0[inc - 1:0:-1]]

                    # Perform the convolution.
                    z = np.real(ifft(fft(c, nfft) * fft(kw)))

                    psi = np.sum(c * z[:inc]) / (n ** 2 * gi ** (2 * ix + 3))
                    # end
                # end
            h[dim] = (ste_constant / psi) ** (1. / 5)
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


if __name__ == '__main__':
    test_docstrings(__file__)
