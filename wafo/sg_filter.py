from __future__ import absolute_import, division
import numpy as np
# from math import pow
# from numpy import zeros,dot
from numpy import (pi, abs, size, convolve, linalg, concatenate, sqrt)
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, expm
from scipy.signal import medfilt
from .dctpack import dctn, idctn
from .plotbackend import plotbackend as plt
import scipy.optimize as optimize
from scipy.signal import _savitzky_golay
from scipy.ndimage import convolve1d
from scipy.ndimage.morphology import distance_transform_edt
import warnings


__all__ = ['SavitzkyGolay', 'Kalman', 'HodrickPrescott', 'smoothn']


class SavitzkyGolay(object):
    r"""Smooth and optionally differentiate data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    n : int
        the size of the smoothing window is 2*n+1.
    degree : int
        the degree of the polynomial used in the filtering.
        Must be less than `window_size` - 1, i.e, less than 2*n.
    diff_order : int
        order of the derivative to compute (default = 0 means only smoothing)
        0 means that filter results in smoothing of function
        1 means that filter results in smoothing the first derivative of the
          function and so on ...
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.  Default is 1.0.
    axis : int, optional
        The axis of the array `x` along which the filter is to be applied.
        Default is -1.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly suited for
    smoothing noisy data. The main idea behind this approach is to make for
    each point a least-square fit with a polynomial of high order over a
    odd-sized window centered at the point.

    Details on the `mode` options:

        'mirror':
            Repeats the values at the edges in reverse order.  The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'constant':
            The extension contains the value given by the `cval` argument.
        'wrap':
            The extension contains the values from the other end of the array.

    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
    `window_length` is 7, the following shows the extended data for
    the various `mode` options (assuming `cval` is 0)::

        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3

    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = SavitzkyGolay(n=20, degree=2).smooth(y)
    >>> import matplotlib.pyplot as plt
    >>> h = plt.plot(t, y, label='Noisy signal')
    >>> h1 = plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> h2 = plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> h3 = plt.legend()
    >>> h4 = plt.title('Savitzky-Golay')
    plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    def __init__(self, n, degree=1, diff_order=0, delta=1.0,  axis=-1,
                 mode='interp', cval=0.0):
        self.n = n
        self.degree = degree
        self.diff_order = diff_order
        self.mode = mode
        self.cval = cval
        self.axis = axis
        self.delta = delta
        window_length = 2 * n + 1
        self._coeff = _savitzky_golay.savgol_coeffs(window_length,
                                                    degree, deriv=diff_order,
                                                    delta=delta)

    def smooth_last(self, signal, k=0):
        coeff = self._coeff
        n = size(coeff - 1) // 2
        y = np.squeeze(signal)
        if n == 0:
            return y
        if y.ndim > 1:
            coeff.shape = (-1, 1)
        first_vals = y[0] - abs(y[n:0:-1] - y[0])
        last_vals = y[-1] + abs(y[-2:-n - 2:-1] - y[-1])
        y = concatenate((first_vals, y, last_vals))
        return (y[-2 * n - 1 - k:-k] * coeff).sum(axis=0)

    def __call__(self, signal):
        return self.smooth(signal)

    def smooth(self, signal):
        x = np.asarray(signal)
        if x.dtype != np.float64 and x.dtype != np.float32:
            x = x.astype(np.float64)

        coeffs = self._coeff
        mode, axis = self.mode, self.axis
        if mode == "interp":
            window_length, polyorder = self.n * 2 + 1, self.degree
            deriv, delta = self.diff_order, self.delta
            y = convolve1d(x, coeffs, axis=axis, mode="constant")
            _savitzky_golay._fit_edges_polyfit(x, window_length, polyorder,
                                               deriv, delta, axis, y)
        else:
            y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=self.cval)
        return y

    def _smooth(self, signal, pad=True):
        """
        Returns smoothed signal (or it's n-th derivative).

        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        pad : bool
           pad first and last values to lessen the end effects.

        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        """
        coeff = self._coeff
        n = size(coeff - 1) // 2
        y = np.squeeze(signal)
        if n == 0:
            return y
        if pad:
            first_vals = y[0] - abs(y[n:0:-1] - y[0])
            last_vals = y[-1] + abs(y[-2:-n - 2:-1] - y[-1])
            y = concatenate((first_vals, y, last_vals))
            n *= 2
        d = y.ndim
        if d > 1:
            y1 = y.reshape(y.shape[0], -1)
            res = []
            for i in range(y1.shape[1]):
                res.append(convolve(y1[:, i], coeff)[n:-n])
            res = np.asarray(res).T
        else:
            res = convolve(y, coeff)[n:-n]
        return res


def evar(y):
    """Noise variance estimation. Assuming that the deterministic function Y
    has additive Gaussian noise, EVAR(Y) returns an estimated variance of this
    noise.

    Note:
    ----
    A thin-plate smoothing spline model is used to smooth Y. It is assumed
    that the model whose generalized cross-validation score is minimum can
    provide the variance of the additive noise. A few tests showed that
    EVAR works very well with "not too irregular" functions.

    Examples:
    --------
    1D signal
    >>> n = 1e6
    >>> x = np.linspace(0,100,n);
    >>> y = np.cos(x/10)+(x/50)
    >>> var0 = 0.02   #  noise variance
    >>> yn = y + sqrt(var0)*np.random.randn(*y.shape)
    >>> s = evar(yn)  # estimated variance
    >>> np.abs(s-var0)/var0 < 3.5/np.sqrt(n)
    True

    2D function
    >>> xp = np.linspace(0,1,50)
    >>> x, y = np.meshgrid(xp,xp)
    >>> f = np.exp(x+y) + np.sin((x-2*y)*3)
    >>> var0 = 0.04 #  noise variance
    >>> fn = f + sqrt(var0)*np.random.randn(*f.shape)
    >>> s = evar(fn)  # estimated variance
    >>> np.abs(s-var0)/var0 < 3.5/np.sqrt(50)
    True

    3D function
    >>> yp = np.linspace(-2,2,50)
    >>> [x,y,z] = meshgrid(yp,yp,yp, sparse=True)
    >>> f = x*exp(-x**2-y**2-z**2)
    >>> var0 = 0.5  # noise variance
    >>> fn = f + sqrt(var0)*np.random.randn(*f.shape)
    >>> s = evar(fn)  # estimated variance
    >>> np.abs(s-var0)/var0 < 3.5/np.sqrt(50)
    True

    Other example
    -------------
    http://www.biomecardio.com/matlab/evar.html

    Note:
    ----
    EVAR is only adapted to evenly-gridded 1-D to N-D data.

    See also
    --------
    VAR, STD, SMOOTHN

    """

    # Damien Garcia -- 2008/04, revised 2009/10
    y = np.atleast_1d(y)
    d = y.ndim
    sh0 = y.shape

    S = np.zeros(sh0)
    sh1 = np.ones((d,))
    cos = np.cos
    pi = np.pi
    for i in range(d):
        ni = sh0[i]
        sh1[i] = ni
        t = np.arange(ni).reshape(sh1) / ni
        S += cos(pi * t)
        sh1[i] = 1

    S2 = 2 * (d - S).ravel()
    # N-D Discrete Cosine Transform of Y
    dcty2 = dctn(y).ravel() ** 2

    def score_fun(L, S2, dcty2):
        # Generalized cross validation score
        M = 1 - 1. / (1 + 10 ** L * S2)
        noisevar = (dcty2 * M ** 2).mean()
        return noisevar / M.mean() ** 2
    # fun = lambda x : score_fun(x, S2, dcty2)
    Lopt = optimize.fminbound(score_fun, -38, 38, args=(S2, dcty2))
    M = 1.0 - 1.0 / (1 + 10 ** Lopt * S2)
    noisevar = (dcty2 * M ** 2).mean()
    return noisevar


class _Filter(object):
    def __init__(self, y, z0, weightstr, weights, s, robust, maxiter, tolz):
        self.y = y
        self.z0 = z0
        self.weightstr = weightstr
        self.s = s
        self.robust = robust
        self.maxiter = maxiter
        self.tolz = tolz

        self.auto_smooth = s is None
        self.is_finite = np.isfinite(y)
        self.nof = self.is_finite.sum()  # number of finite elements
        self.W = self._normalized_weights(weights, self.is_finite)

        self.gamma = self._gamma_fun(y)

        self.N = self._tensor_rank(y)
        self.s_min, self.s_max = self._smoothness_limits(self.N)

        # Initialize before iterating
        self.Wtot = self.W
        self.is_weighted = (self.W < 1).any()  # Weighted or missing data?

        self.z0 = self._get_start_condition(y, z0)

        self.y[~self.is_finite] = 0  # arbitrary values for missing y-data

        # Error on p. Smoothness parameter s = 10^p
        self.errp = 0.1

        # Relaxation factor RF: to speedup convergence
        self.RF = 1.75 if self.is_weighted else 1.0

    @staticmethod
    def _tensor_rank(y):
        """tensor rank of the y-array"""
        return (np.array(y.shape) != 1).sum()

    @staticmethod
    def _smoothness_limits(n):
        """
        Return upper and lower bound for the smoothness parameter

        The average leverage (h) is by definition in [0 1]. Weak smoothing
        occurs if h is close to 1, while over-smoothing appears when h is
        near 0. Upper and lower bounds for h are given to avoid under- or
        over-smoothing. See equation relating h to the smoothness parameter
        (Equation #12 in the referenced CSDA paper).
        """
        h_min = 1e-6 ** (2. / n)
        h_max = 0.99 ** (2. / n)

        s_min = (((1 + sqrt(1 + 8 * h_max)) / 4. / h_max) ** 2 - 1) / 16
        s_max = (((1 + sqrt(1 + 8 * h_min)) / 4. / h_min) ** 2 - 1) / 16
        return s_min, s_max

    @staticmethod
    def _lambda_tensor(y):
        """
        Return the Lambda tensor

        Lambda contains the eigenvalues of the difference matrix used in this
        penalized least squares process.
        """
        d = y.ndim
        Lambda = np.zeros(y.shape)
        shape0 = [1, ] * d
        for i in range(d):
            shape0[i] = y.shape[i]
            Lambda = Lambda + \
                np.cos(pi * np.arange(y.shape[i]) / y.shape[i]).reshape(shape0)
            shape0[i] = 1
        Lambda = -2 * (d - Lambda)
        return Lambda

    def _gamma_fun(self, y):
        Lambda = self._lambda_tensor(y)

        def gamma(s):
            return 1. / (1 + s * Lambda ** 2)
        return gamma

    @staticmethod
    def _initial_guess(y, I):
        # Initial Guess with weighted/missing data
        # nearest neighbor interpolation (in case of missing values)
        z = y
        if (1 - I).any():
            notI = ~I
            z, L = distance_transform_edt(notI,  return_indices=True)
            z[notI] = y[L.flat[notI]]

        # coarse fast smoothing using one-tenth of the DCT coefficients
        shape = z.shape
        d = z.ndim
        z = dctn(z)
        for k in range(d):
            z[int((shape[k] + 0.5) / 10) + 1::, ...] = 0
            z = z.reshape(np.roll(shape, -k))
            z = z.transpose(np.roll(range(d), -1))
            # z = shiftdim(z,1);
        return idctn(z)

    def _get_start_condition(self, y, z0):
        # Initial conditions for z
        if self.is_weighted:
            # With weighted/missing data
            # An initial guess is provided to ensure faster convergence. For
            # that purpose, a nearest neighbor interpolation followed by a
            # coarse smoothing are performed.
            if z0 is None:
                z = self._initial_guess(y, self.is_finite)
            else:
                z = z0  # an initial guess (z0) has been provided
        else:
            z = np.zeros(y.shape)
        return z

    @staticmethod
    def _normalized_weights(weight, is_finite):
        """ Return normalized weights.

        Zero weights are assigned to not finite values (Inf or NaN),
        (Inf/NaN values = missing data).
        """
        weights = weight * is_finite
        if (weights < 0).any():
            raise ValueError('Weights must all be >=0')
        return weights / weights.max()

    @staticmethod
    def _studentized_residuals(r, I, h):
        median_abs_deviation = np.median(abs(r[I] - np.median(r[I])))
        return abs(r / (1.4826 * median_abs_deviation) / sqrt(1 - h))

    def robust_weights(self, r, I, h):
        """Return weights for robust smoothing."""
        def bisquare(u):
            c = 4.685
            return (1 - (u / c) ** 2) ** 2 * ((u / c) < 1)

        def talworth(u):
            c = 2.795
            return u < c

        def cauchy(u):
            c = 2.385
            return 1. / (1 + (u / c) ** 2)

        u = self._studentized_residuals(r, I, h)

        wfun = {'cauchy': cauchy, 'talworth': talworth}.get(self.weightstr,
                                                            bisquare)
        weights = wfun(u)

        weights[np.isnan(weights)] = 0
        return weights

    @staticmethod
    def _average_leverage(s, N):
        h = sqrt(1 + 16 * s)
        h = sqrt(1 + h) / sqrt(2) / h
        return h ** N

    def check_smooth_parameter(self, s):
        if self.auto_smooth:
            if abs(np.log10(s) - np.log10(self.s_min)) < self.errp:
                warnings.warn('''s = %g: the lower bound for s has been reached.
            Put s as an input variable if required.''' % s)
            elif abs(np.log10(s) - np.log10(self.s_max)) < self.errp:
                warnings.warn('''s = %g: the Upper bound for s has been reached.
            Put s as an input variable if required.''' % s)

    def gcv(self, p, aow, DCTy, y, Wtot):
        # Search the smoothing parameter s that minimizes the GCV score
        s = 10.0 ** p
        Gamma = self.gamma(s)
        if aow > 0.9:
            # aow = 1 means that all of the data are equally weighted
            # very much faster: does not require any inverse DCT
            residual = DCTy.ravel() * (Gamma.ravel() - 1)
        else:
            # take account of the weights to calculate RSS:
            is_finite = self.is_finite
            yhat = idctn(Gamma * DCTy)
            residual = sqrt(Wtot[is_finite]) * (y[is_finite] - yhat[is_finite])

        TrH = Gamma.sum()
        RSS = linalg.norm(residual)**2  # Residual sum-of-squares
        GCVscore = RSS / self.nof / (1.0 - TrH / y.size) ** 2
        return GCVscore

    def __call__(self, z, s):
        auto_smooth = self.auto_smooth
        norm = linalg.norm
        y = self.y
        Wtot = self.Wtot
        Gamma = 1
        if s is not None:
            Gamma = self.gamma(s)
        # "amount" of weights (see the function GCVscore)
        aow = Wtot.sum() / y.size  # 0 < aow <= 1
        for nit in range(self.maxiter):
            DCTy = dctn(Wtot * (y - z) + z)
            if auto_smooth and not np.remainder(np.log2(nit + 1), 1):
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter s that minimizes the GCV
                # score i.e. s = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when nit is a power of 2)
                log10s = optimize.fminbound(
                    self.gcv, np.log10(self.s_min), np.log10(self.s_max),
                    args=(aow, DCTy, y, Wtot),
                    xtol=self.errp, full_output=False, disp=False)
                s = 10 ** log10s
                Gamma = self.gamma(s)
            z0 = z
            z = self.RF * idctn(Gamma * DCTy) + (1 - self.RF) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = norm(z0.ravel() - z.ravel()) / norm(z.ravel())
            converged = tol <= self.tolz or not self.is_weighted
            if converged:
                break
        if self.robust:
            # -- Robust Smoothing: iteratively re-weighted process
            h = self._average_leverage(s, self.N)
            self.Wtot = self.W * self.robust_weights(y - z, self.is_finite, h)
            # re-initialize for another iterative weighted process
            self.is_weighted = True
        return z, s, converged


def smoothn(data, s=None, weight=None, robust=False, z0=None, tolz=1e-3,
            maxiter=100, fulloutput=False):
    '''
    SMOOTHN fast and robust spline smoothing for 1-D to N-D data.

    Parameters
    ----------
    data : array like
        uniformly-sampled data array to smooth. Non finite values (NaN or Inf)
        are treated as missing values.
    s : real positive scalar
        smooting parameter. The larger S is, the smoother the output will be.
        Default value is automatically determined using the generalized
        cross-validation (GCV) method.
    weight : string or array weights
        weighting array of real positive values, that must have the same size
        as DATA. Note that a zero weight corresponds to a missing value.
    robust : bool
        If true carry out a robust smoothing that minimizes the influence of
        outlying data.
    tolz : real positive scalar
        Termination tolerance on Z (default = 1e-3)
    maxiter :  scalar integer
        Maximum number of iterations allowed (default = 100)
    z0 : array-like
        Initial value for the iterative process (default = original data)

    Returns
    -------
    z : array like
        smoothed data

    To be made
    ----------
    Estimate the confidence bands (see Wahba 1983, Nychka 1988).

    Reference
    ---------
    Garcia D, Robust smoothing of gridded data in one and higher dimensions
    with missing values. Computational Statistics & Data Analysis, 2010.
    http://www.biomecardio.com/pageshtm/publi/csda10.pdf

    Examples:
    --------

    1-D example
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0,100,2**8)
    >>> y = np.cos(x/10)+(x/50)**2 + np.random.randn(*x.shape)/10
    >>> y[np.r_[70, 75, 80]] = np.array([5.5, 5, 6])
    >>> z = smoothn(y) # Regular smoothing
    >>> zr = smoothn(y,robust=True) #  Robust smoothing
    >>> h=plt.subplot(121),
    >>> h = plt.plot(x,y,'r.',x,z,'k',linewidth=2)
    >>> h=plt.title('Regular smoothing')
    >>> h=plt.subplot(122)
    >>> h=plt.plot(x,y,'r.',x,zr,'k',linewidth=2)
    >>> h=plt.title('Robust smoothing')

     2-D example
    >>> xp = np.r_[0:1:.02]
    >>> [x,y] = np.meshgrid(xp,xp)
    >>> f = np.exp(x+y) + np.sin((x-2*y)*3);
    >>> fn = f + np.random.randn(*f.shape)*0.5;
    >>> fs = smoothn(fn);
    >>> h=plt.subplot(121),
    >>> h=plt.contourf(xp,xp,fn)
    >>> h=plt.subplot(122)
    >>> h=plt.contourf(xp,xp,fs)

     2-D example with missing data
    n = 256;
    y0 = peaks(n);
    y = y0 + rand(size(y0))*2;
    I = randperm(n^2);
    y(I(1:n^2*0.5)) = NaN;  lose 1/2 of data
    y(40:90,140:190) = NaN;  create a hole
    z = smoothn(y);  smooth data
    subplot(2,2,1:2), imagesc(y), axis equal off
    title('Noisy corrupt data')
    subplot(223), imagesc(z), axis equal off
    title('Recovered data ...')
    subplot(224), imagesc(y0), axis equal off
    title('... compared with original data')

     3-D example
    [x,y,z] = meshgrid(-2:.2:2);
    xslice = [-0.8,1]; yslice = 2; zslice = [-2,0];
    vn = x.*exp(-x.^2-y.^2-z.^2) + randn(size(x))*0.06;
    subplot(121), slice(x,y,z,vn,xslice,yslice,zslice,'cubic')
    title('Noisy data')
    v = smoothn(vn);
    subplot(122), slice(x,y,z,v,xslice,yslice,zslice,'cubic')
    title('Smoothed data')

    Cardioid

    t = linspace(0,2*pi,1000);
    x = 2*cos(t).*(1-cos(t)) + randn(size(t))*0.1;
    y = 2*sin(t).*(1-cos(t)) + randn(size(t))*0.1;
    z = smoothn(complex(x,y));
    plot(x,y,'r.',real(z),imag(z),'k','linewidth',2)
    axis equal tight

     Cellular vortical flow
    [x,y] = meshgrid(linspace(0,1,24));
    Vx = cos(2*pi*x+pi/2).*cos(2*pi*y);
    Vy = sin(2*pi*x+pi/2).*sin(2*pi*y);
    Vx = Vx + sqrt(0.05)*randn(24,24);  adding Gaussian noise
    Vy = Vy + sqrt(0.05)*randn(24,24);  adding Gaussian noise
    I = randperm(numel(Vx));
    Vx(I(1:30)) = (rand(30,1)-0.5)*5;  adding outliers
    Vy(I(1:30)) = (rand(30,1)-0.5)*5;  adding outliers
    Vx(I(31:60)) = NaN;  missing values
    Vy(I(31:60)) = NaN;  missing values
    Vs = smoothn(complex(Vx,Vy),'robust');  automatic smoothing
    subplot(121), quiver(x,y,Vx,Vy,2.5), axis square
    title('Noisy velocity field')
    subplot(122), quiver(x,y,real(Vs),imag(Vs)), axis square
    title('Smoothed velocity field')

    See also SMOOTH, SMOOTH3, DCTN, IDCTN.

    -- Damien Garcia -- 2009/03, revised 2010/11
    Visit
    http://www.biomecardio.com/matlab/smoothn.html
    for more details about SMOOTHN
    '''
    return SmoothNd(s, weight, robust, z0, tolz, maxiter, fulloutput)(data)


class SmoothNd(object):
    def __init__(self, s=None, weight=None, robust=False, z0=None, tolz=1e-3,
                 maxiter=100, fulloutput=False):
        self.s = s
        self.weight = weight
        self.robust = robust
        self.z0 = z0
        self.tolz = tolz
        self.maxiter = maxiter
        self.fulloutput = fulloutput

    @property
    def weightstr(self):
        if isinstance(self._weight, str):
            return self._weight.lower()
        return 'bisquare'

    @property
    def weight(self):
        if self._weight is None or isinstance(self._weight, str):
            return 1.0
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    def _init_filter(self, y):
        return _Filter(y, self.z0, self.weightstr, self.weight, self.s,
                       self.robust, self.maxiter, self.tolz)

    @property
    def num_steps(self):
        return 3 if self.robust else 1

    def __call__(self, data):

        y = np.atleast_1d(data)
        if y.size < 2:
            return data

        _filter = self._init_filter(y)
        z = _filter.z0
        s = _filter.s
        converged = False
        for _i in range(self.num_steps):
            z, s, converged = _filter(z, s)

        if not converged:
            msg = '''Maximum number of iterations (%d) has been exceeded.
            Increase MaxIter option or decrease TolZ value.''' % (self.maxiter)
            warnings.warn(msg)

        _filter.check_smooth_parameter(s)

        if self.fulloutput:
            return z, s
        return z


def test_smoothn_1d():
    x = np.linspace(0, 100, 2 ** 8)
    y = np.cos(x / 10) + (x / 50) ** 2 + np.random.randn(x.size) / 10
    y[np.r_[70, 75, 80]] = np.array([5.5, 5, 6])
    z = smoothn(y)  # Regular smoothing
    zr = smoothn(y, robust=True)  # Robust smoothing
    plt.subplot(121),
    unused_h = plt.plot(x, y, 'r.', x, z, 'k', linewidth=2)
    plt.title('Regular smoothing')
    plt.subplot(122)
    plt.plot(x, y, 'r.', x, zr, 'k', linewidth=2)
    plt.title('Robust smoothing')
    plt.show('hold')


def test_smoothn_2d():

    # import mayavi.mlab as plt
    xp = np.r_[0:1:.02]
    [x, y] = np.meshgrid(xp, xp)
    f = np.exp(x + y) + np.sin((x - 2 * y) * 3)
    fn = f + np.random.randn(*f.shape) * 0.5
    fs, s = smoothn(fn, fulloutput=True)  # @UnusedVariable
    fs2 = smoothn(fn, s=2 * s)
    plt.subplot(131),
    plt.contourf(xp, xp, fn)
    plt.subplot(132),
    plt.contourf(xp, xp, fs2)
    plt.subplot(133),
    plt.contourf(xp, xp, f)
    plt.show('hold')


def test_smoothn_cardioid():
    t = np.linspace(0, 2 * pi, 1000)
    cos = np.cos
    sin = np.sin
    randn = np.random.randn
    x0 = 2 * cos(t) * (1 - cos(t))
    x = x0 + randn(t.size) * 0.1
    y0 = 2 * sin(t) * (1 - cos(t))
    y = y0 + randn(t.size) * 0.1
    z = smoothn(x + 1j * y, robust=False)
    plt.plot(x0, y0, 'y',
             x, y, 'r.',
             z.real, z.imag, 'k', linewidth=2)
    plt.show('hold')


class HodrickPrescott(object):

    '''Smooth data with a Hodrick-Prescott filter.

    The Hodrick-Prescott filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameter
    ---------
    w : real scalar
        smooting parameter. Larger w means more smoothing. Values usually
        in the [100, 20000] interval. As w approach infinity H-P will approach
        a line.

    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = HodrickPrescott(w=10000)(y)
    >>> import matplotlib.pyplot as plt
    >>> h = plt.plot(t, y, label='Noisy signal')
    >>> h1 = plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> h2 = plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> h3 = plt.legend()
    >>> h4 = plt.title('Hodrick-Prescott')
    >>> plt.show()

    References
    ----------
    .. [1] E. T. Whittaker, On a new method of graduation. In proceedings of
        the Edinburgh Mathematical association., 1923, 78, pp 88-89.
    .. [2] R. Hodrick and E. Prescott, Postwar U.S. business cycles: an
        empirical investigation,
        Journal of money, credit and banking, 1997, 29 (1), pp 1-16.
    .. [3] Kim Hyeongwoo, Hodrick-Prescott filter,
        2004, www.auburn.edu/~hzk0001/hpfilter.pdf
    '''

    def __init__(self, w=100):
        self.w = w

    def _get_matrix(self, n):
        w = self.w
        diag_matrix = np.repeat(
            np.atleast_2d([w, -4 * w, 6 * w + 1, -4 * w, w]).T, n, axis=1)
        A = spdiags(diag_matrix, np.arange(-2, 2 + 1), n, n).tocsr()
        A[0, 0] = A[-1, -1] = 1 + w
        A[1, 1] = A[-2, -2] = 1 + 5 * w
        A[0, 1] = A[1, 0] = A[-2, -1] = A[-1, -2] = -2 * w
        return A

    def __call__(self, x):
        x = np.atleast_1d(x).flatten()
        n = len(x)
        if n < 4:
            return x.copy()

        A = self._get_matrix(n)
        return spsolve(A, x)


class Kalman(object):

    '''
    Kalman filter object - updates a system state vector estimate based upon an
              observation, using a discrete Kalman filter.

    The Kalman filter is "optimal" under a variety of
    circumstances.  An excellent paper on Kalman filtering at
    the introductory level, without detailing the mathematical
    underpinnings, is:

    "An Introduction to the Kalman Filter"
    Greg Welch and Gary Bishop, University of North Carolina
    http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

    PURPOSE:
    The purpose of each iteration of a Kalman filter is to update
    the estimate of the state vector of a system (and the covariance
    of that vector) based upon the information in a new observation.
    The version of the Kalman filter in this function assumes that
    observations occur at fixed discrete time intervals. Also, this
    function assumes a linear system, meaning that the time evolution
    of the state vector can be calculated by means of a state transition
    matrix.

    USAGE:
    filt = Kalman(R, x, P, A, B=0, Q, H)
    x = filt(z, u=0)

    filt is a "system" object containing various fields used as input
    and output. The state estimate "x" and its covariance "P" are
    updated by the function. The other fields describe the mechanics
    of the system and are left unchanged. A calling routine may change
    these other fields as needed if state dynamics are time-dependent;
    otherwise, they should be left alone after initial values are set.
    The exceptions are the observation vector "z" and the input control
    (or forcing function) "u." If there is an input function, then
    "u" should be set to some nonzero value by the calling routine.

    System dynamics
    ---------------

    The system evolves according to the following difference equations,
    where quantities are further defined below:

    x = Ax + Bu + w  meaning the state vector x evolves during one time
                     step by premultiplying by the "state transition
                     matrix" A. There is optionally (if nonzero) an input
                     vector u which affects the state linearly, and this
                     linear effect on the state is represented by
                     premultiplying by the "input matrix" B. There is also
                     gaussian process noise w.
    z = Hx + v       meaning the observation vector z is a linear function
                     of the state vector, and this linear relationship is
                     represented by premultiplication by "observation
                     matrix" H. There is also gaussian measurement
                     noise v.
    where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
          v ~ N(0,R) meaning v is gaussian noise with covariance R

    VECTOR VARIABLES:

    s.x = state vector estimate. In the input struct, this is the
          "a priori" state estimate (prior to the addition of the
          information from the new observation). In the output struct,
          this is the "a posteriori" state estimate (after the new
          measurement information is included).
    z = observation vector
    u = input control vector, optional (defaults to zero).

    MATRIX VARIABLES:

    s.A = state transition matrix (defaults to identity).
    s.P = covariance of the state vector estimate. In the input struct,
          this is "a priori," and in the output it is "a posteriori."
          (required unless autoinitializing as described below).
    s.B = input matrix, optional (defaults to zero).
    s.Q = process noise covariance (defaults to zero).
    s.R = measurement noise covariance (required).
    s.H = observation matrix (defaults to identity).

    NORMAL OPERATION:

    (1) define all state definition fields: A,B,H,Q,R
    (2) define intial state estimate: x,P
    (3) obtain observation and control vectors: z,u
    (4) call the filter to obtain updated state estimate: x,P
    (5) return to step (3) and repeat

    INITIALIZATION:

    If an initial state estimate is unavailable, it can be obtained
    from the first observation as follows, provided that there are the
    same number of observable variables as state variables. This "auto-
    intitialization" is done automatically if s.x is absent or NaN.

    x = inv(H)*z
    P = inv(H)*R*inv(H')

    This is mathematically equivalent to setting the initial state estimate
    covariance to infinity.

    Example (Automobile Voltimeter):
    -------
    # Define the system as a constant of 12 volts:
    >>> V0 = 12
    >>> h = 1      # voltimeter measure the voltage itself
    >>> q = 1e-5   # variance of process noise s the car operates
    >>> r = 0.1**2 # variance of measurement error
    >>> b = 0      # no system input
    >>> u = 0      # no system input
    >>> filt = Kalman(R=r, A=1, Q=q, H=h, B=b)

    # Generate random voltages and watch the filter operate.
    >>> n = 50
    >>> truth = np.random.randn(n)*np.sqrt(q) + V0
    >>> z = truth + np.random.randn(n)*np.sqrt(r) # measurement
    >>> x = np.zeros(n)

    >>> for i, zi in enumerate(z):
    ...     x[i] = filt(zi, u) #  perform a Kalman filter iteration

    >>> import matplotlib.pyplot as plt
    >>> hz = plt.plot(z,'r.', label='observations')

    # a-posteriori state estimates:
    >>> hx = plt.plot(x,'b-', label='Kalman output')
    >>> ht = plt.plot(truth,'g-', label='true voltage')
    >>> h = plt.legend()
    >>> h1 = plt.title('Automobile Voltimeter Example')
    >>> plt.show()

    '''

    def __init__(self, R, x=None, P=None, A=None, B=0, Q=None, H=None):
        self.R = R  # Estimated error in measurements.
        self.x = x  # Initial state estimate.
        self.P = P  # Initial covariance estimate.
        self.A = A  # State transition matrix.
        self.B = B  # Control matrix.
        self.Q = Q  # Estimated error in process.
        self.H = H  # Observation matrix.
        self.reset()

    def reset(self):
        self._filter = self._filter_first

    def _set_A(self, n):
        if self.A is None:
            self.A = np.eye(n)
        self.A = np.atleast_2d(self.A)

    def _set_Q(self, n):
        if self.Q is None:
            self.Q = np.zeros((n, n))
        self.Q = np.atleast_2d(self.Q)

    def _set_H(self, n):
        if self.H is None:
            self.H = np.eye(n)
        self.H = np.atleast_2d(self.H)

    def _set_P(self, HI):
        if self.P is None:
            self.P = np.dot(np.dot(HI, self.R), HI.T)
        self.P = np.atleast_2d(self.P)

    def _init_first(self, n):
        self._set_A(n)
        self._set_Q(n)
        self._set_H(n)
        try:
            HI = np.linalg.inv(self.H)
        except:
            HI = np.eye(n)
        self._set_P(HI)
        return HI

    def _first_state(self, z):
        n = np.size(z)
        HI = self._init_first(n)
        # initialize state estimate from first observation
        x = np.dot(HI, z)
        return x

    def _filter_first(self, z, u):

        self._filter = self._filter_main

        if self.x is None:
            self.x = self._first_state(z)
            return self.x

        n = np.size(self.x)
        self._init_first(n)
        return self._filter_main(z, u)

    def _predict_state(self, x, u):
        return np.dot(self.A, x) + np.dot(self.B, u)

    def _predict_covariance(self, P):
        A = self.A
        return np.dot(np.dot(A, P), A.T) + self.Q

    def _compute_gain(self, P):
        """Kalman gain factor."""
        H = self.H
        PHT = np.dot(P, H.T)
        innovation_covariance = np.dot(H, PHT) + self.R
        # return np.linalg.solve(PHT, innovation_covariance)
        return np.dot(PHT, np.linalg.inv(innovation_covariance))

    def _update_state_from_observation(self, x, z, K):
        innovation = z - np.dot(self.H, x)
        return x + np.dot(K, innovation)

    def _update_covariance(self, P, K):
        return P - np.dot(K, np.dot(self.H, P))
        return np.dot(np.eye(len(P)) - K * self.H, P)

    def _filter_main(self, z, u):
        ''' This is the code which implements the discrete Kalman filter:
        '''
        P = self._predict_covariance(self.P)
        x = self._predict_state(self.x, u)

        K = self._compute_gain(P)

        self.P = self._update_covariance(P, K)
        self.x = self._update_state_from_observation(x, z, K)

        return self.x

    def __call__(self, z, u=0):
        return self._filter(z, u)


def test_kalman():
    V0 = 12
    h = np.atleast_2d(1)  # voltimeter measure the voltage itself
    q = 1e-9  # variance of process noise as the car operates
    r = 0.05 ** 2  # variance of measurement error
    b = 0  # no system input
    u = 0  # no system input
    filt = Kalman(R=r, A=1, Q=q, H=h, B=b)

    # Generate random voltages and watch the filter operate.
    n = 50
    truth = np.random.randn(n) * np.sqrt(q) + V0
    z = truth + np.random.randn(n) * np.sqrt(r)  # measurement
    x = np.zeros(n)

    for i, zi in enumerate(z):
        x[i] = filt(zi, u)  # perform a Kalman filter iteration

    _hz = plt.plot(z, 'r.', label='observations')
    # a-posteriori state estimates:
    _hx = plt.plot(x, 'b-', label='Kalman output')
    _ht = plt.plot(truth, 'g-', label='true voltage')
    plt.legend()
    plt.title('Automobile Voltimeter Example')
    plt.show('hold')


def lti_disc(F, L=None, Q=None, dt=1):
    """LTI_DISC  Discretize LTI ODE with Gaussian Noise.

    Syntax:
     [A,Q] = lti_disc(F,L,Qc,dt)

    In:
     F  - NxN Feedback matrix
     L  - NxL Noise effect matrix        (optional, default identity)
     Qc - LxL Diagonal Spectral Density  (optional, default zeros)
     dt - Time Step                      (optional, default 1)

    Out:
     A - Transition matrix
     Q - Discrete Process Covariance

    Description:
     Discretize LTI ODE with Gaussian Noise. The original
     ODE model is in form

       dx/dt = F x + L w,  w ~ N(0,Qc)

     Result of discretization is the model

       x[k] = A x[k-1] + q, q ~ N(0,Q)

     Which can be used for integrating the model
     exactly over time steps, which are multiples
     of dt.

    """
    n = np.shape(F)[0]
    if L is None:
        L = np.eye(n)

    if Q is None:
        Q = np.zeros((n, n))
    # Closed form integration of transition matrix
    A = expm(F * dt)

    # Closed form integration of covariance
    # by matrix fraction decomposition

    Phi = np.vstack((np.hstack((F, np.dot(np.dot(L, Q), L.T))),
                     np.hstack((np.zeros((n, n)), -F.T))))
    AB = np.dot(expm(Phi * dt), np.vstack((np.zeros((n, n)), np.eye(n))))
    # Q = AB[:n, :] / AB[n:(2 * n), :]
    Q = np.linalg.solve(AB[n:(2 * n), :].T, AB[:n, :].T)
    return A, Q


def test_kalman_sine():
    """Kalman Filter demonstration with sine signal."""
    sd = 0.5
    dt = 0.1
    w = 1
    T = np.arange(0, 30 + dt / 2, dt)
    n = len(T)
    X = 3*np.sin(w * T)
    Y = X + sd * np.random.randn(n)

    ''' Initialize KF to values
       x = 0
       dx/dt = 0
     with great uncertainty in derivative
    '''
    M = np.zeros((2, 1))
    P = np.diag([0.1, 2])
    R = sd ** 2
    H = np.atleast_2d([1, 0])
    q = 0.1
    F = np.atleast_2d([[0, 1],
                       [0, 0]])
    A, Q = lti_disc(F, L=None, Q=np.diag([0, q]), dt=dt)

    # Track and animate
    m = M.shape[0]
    _MM = np.zeros((m, n))
    _PP = np.zeros((m, m, n))
    '''In this demonstration we estimate a stationary sine signal from noisy
    measurements by using the classical Kalman filter.'
    '''
    filt = Kalman(R=R, x=M, P=P, A=A, Q=Q, H=H, B=0)

    # Generate random voltages and watch the filter operate.
    # n = 50
    # truth = np.random.randn(n) * np.sqrt(q) + V0
    # z = truth + np.random.randn(n) * np.sqrt(r)  # measurement
    truth = X
    z = Y
    x = np.zeros((n, m))

    for i, zi in enumerate(z):
        x[i] = filt(zi, u=0).ravel()

    _hz = plt.plot(z, 'r.', label='observations')
    # a-posteriori state estimates:
    _hx = plt.plot(x[:, 0], 'b-', label='Kalman output')
    _ht = plt.plot(truth, 'g-', label='true voltage')
    plt.legend()
    plt.title('Automobile Voltimeter Example')
    plt.show('hold')

#     for k in range(m):
#         [M,P] = kf_predict(M,P,A,Q);
#         [M,P] = kf_update(M,P,Y(k),H,R);
#
#         MM(:,k) = M;
#         PP(:,:,k) = P;
#
#       %
#       % Animate
#       %
#       if rem(k,10)==1
#         plot(T,X,'b--',...
#              T,Y,'ro',...
#              T(k),M(1),'k*',...
#              T(1:k),MM(1,1:k),'k-');
#         legend('Real signal','Measurements','Latest estimate',
#                'Filtered estimate')
#         title('Estimating a noisy sine signal with Kalman filter.');
#         drawnow;
#
#         pause;
#       end
#     end
#
#     clc;
#     disp('In this demonstration we estimate a stationary sine signal '
#        'from noisy measurements by using the classical Kalman filter.');
#     disp(' ');
#     disp('The filtering results are now displayed sequantially for 10 time '
#            'step at a time.');
#     disp(' ');
#     disp('<push any key to see the filtered and smoothed results together>')
#     pause;
#     %
#     % Apply Kalman smoother
#     %
#     SM = rts_smooth(MM,PP,A,Q);
#     plot(T,X,'b--',...
#          T,MM(1,:),'k-',...
#          T,SM(1,:),'r-');
#     legend('Real signal','Filtered estimate','Smoothed estimate')
#     title('Filtered and smoothed estimate of the original signal');
#
#     clc;
#     disp('The filtered and smoothed estimates of the signal are now '
#        'displayed.')
#     disp(' ');
#     disp('RMS errors:');
#     %
#     % Errors
#     %
#     fprintf('KF = %.3f\nRTS = %.3f\n',...
#             sqrt(mean((MM(1,:)-X(1,:)).^2)),...
#             sqrt(mean((SM(1,:)-X(1,:)).^2)));


class HampelFilter(object):
    """Hampel Filter.

    HAMPEL(X,Y,DX,T,varargin) returns the Hampel filtered values of the
    elements in Y. It was developed to detect outliers in a time series,
    but it can also be used as an alternative to the standard median
    filter.

    X,Y are row or column vectors with an equal number of elements.
    The elements in Y should be Gaussian distributed.

    Parameters
    ----------
    dx : positive scalar (default 3 * median(diff(X))
        which defines the half width of the filter window. Dx should be
        dimensionally equivalent to the values in X.
    t : positive scalar (default 3)
        which defines the threshold value used in the equation
        |Y - Y0| > T * S0.
    adaptive: real scalar
        if greater than 0 it uses an experimental adaptive Hampel filter.
        If none it uses a standard Hampel filter
    fulloutput: bool
        if True also the vectors: outliers, Y0,LB,UB,ADX, which corresponds to
        the mask of the replaced values, nominal data, lower and upper bounds
        on the Hampel filter and the relative half size of the local window,
        respectively. outliers.sum() gives the number of outliers detected.

    Examples
    ---------
    Hampel filter removal of outliers
    >>> import numpy as np
    >>> randint = np.random.randint
    >>> Y = 5000 + np.random.randn(1000)
    >>> outliers = randint(0,1000, size=(10,))
    >>> Y[outliers] = Y[outliers] + randint(1000, size=(10,))
    >>> YY, res = HampelFilter(fulloutput=True)(Y)
    >>> YY1, res1 = HampelFilter(dx=1, t=3, adaptive=0.1, fulloutput=True)(Y)
    >>> YY2, res2 = HampelFilter(dx=3, t=0, fulloutput=True)(Y)  # Y0 = median

    X = np.arange(len(YY))
    plt.plot(X, Y, 'b.')  # Original Data
    plt.plot(X, YY, 'r')  # Hampel Filtered Data
    plt.plot(X, res['Y0'], 'b--')  # Nominal Data
    plt.plot(X, res['LB'], 'r--')  # Lower Bounds on Hampel Filter
    plt.plot(X, res['UB'], 'r--')  # Upper Bounds on Hampel Filter
    i = res['outliers']
    plt.plot(X[i], Y[i], 'ks')  # Identified Outliers
    plt.show('hold')

     References
    ----------
    Chapters 1.4.2, 3.2.2 and 4.3.4 in Mining Imperfect Data: Dealing with
    Contamination and Incomplete Records by Ronald K. Pearson.

    Acknowledgements
    I would like to thank Ronald K. Pearson for the introduction to moving
    window filters. Please visit his blog at:
    http://exploringdatablog.blogspot.com/2012/01/moving-window-filters-and
    -pracma.html

    """
    def __init__(self, dx=None, t=3, adaptive=None, fulloutput=False):
        self.dx = dx
        self.t = t
        self.adaptive = adaptive
        self.fulloutput = fulloutput

    def _check(self, dx):
        if not np.isscalar(dx):
            raise ValueError('DX must be a scalar.')
        if dx < 0:
            raise ValueError('DX must be larger than zero.')

    @staticmethod
    def localwindow(X, Y, DX, i):
        mask = (X[i] - DX <= X) & (X <= X[i] + DX)
        Y0 = np.median(Y[mask])
        # Calculate Local Scale of Natural Variation
        S0 = 1.4826 * np.median(np.abs(Y[mask] - Y0))
        return Y0, S0

    @staticmethod
    def smgauss(X, V, DX):
        Xj = X
        Xk = np.atleast_2d(X).T
        Wjk = np.exp(-((Xj - Xk) / (2 * DX)) ** 2)
        G = np.dot(Wjk, V) / np.sum(Wjk, axis=0)
        return G

    def _adaptive(self, Y, X, dx):
        localwindow = self.localwindow
        Y0, S0, ADX = self._init(Y, dx)
        Y0Tmp = np.nan * np.zeros(Y.shape)
        S0Tmp = np.nan * np.zeros(Y.shape)
        DXTmp = np.arange(1, len(S0) + 1) * dx
        # Integer variation of Window Half Size
        # Calculate Initial Guess of Optimal Parameters Y0, S0, ADX
        for i in range(len(Y)):
            j = 0
            S0Rel = np.inf
            while S0Rel > self.adaptive:
                Y0Tmp[j], S0Tmp[j] = localwindow(X, Y, DXTmp[j], i)
                if j > 0:
                    S0Rel = abs((S0Tmp[j - 1] - S0Tmp[j]) /
                                (S0Tmp[j - 1] + S0Tmp[j]) / 2)
                j += 1

            Y0[i] = Y0Tmp[j - 2]
            S0[i] = S0Tmp[j - 2]
            ADX[i] = DXTmp[j - 2] / dx

    # Gaussian smoothing of relevant parameters
        DX = 2 * np.median(np.diff(X))
        ADX = self.smgauss(X, ADX, DX)
        S0 = self.smgauss(X, S0, DX)
        Y0 = self.smgauss(X, Y0, DX)
        return Y0, S0, ADX

    def _init(self, Y, dx):
        S0 = np.nan * np.zeros(Y.shape)
        Y0 = np.nan * np.zeros(Y.shape)
        ADX = dx * np.ones(Y.shape)
        return Y0, S0, ADX

    def _fixed(self, Y, X, dx):
        localwindow = self.localwindow
        Y0, S0, ADX = self._init(Y, dx)
        for i in range(len(Y)):
            Y0[i], S0[i] = localwindow(X, Y, dx, i)
        return Y0, S0, ADX

    def _filter(self, Y, X, dx):
        if len(X) <= 1:
            Y0, S0, ADX = self._init(Y, dx)
        elif self.adaptive is None:
            Y0, S0, ADX = self._fixed(Y, X, dx)
        else:
            Y0, S0, ADX = self._adaptive(Y, X, dx)  # 'adaptive'
        return Y0, S0, ADX

    def __call__(self, y, x=None):
        Y = np.atleast_1d(y).ravel()
        if x is None:
            x = range(len(Y))
        X = np.atleast_1d(x).ravel()

        dx = 3 * np.median(np.diff(X)) if self.dx is None else self.dx
        self._check(dx)

        Y0, S0, ADX = self._filter(Y, X, dx)
        YY = Y.copy()
        T = self.t
        # Prepare Output
        self.UB = Y0 + T * S0
        self.LB = Y0 - T * S0
        outliers = np.abs(Y - Y0) > T * S0  # possible outliers
        np.putmask(YY, outliers, Y0)  # YY[outliers] = Y0[outliers]
        self.outliers = outliers
        self.num_outliers = outliers.sum()
        self.ADX = ADX
        self.Y0 = Y0
        if self.fulloutput:
            return YY, dict(outliers=outliers, Y0=Y0,
                            LB=self.LB, UB=self.UB, ADX=ADX)
        return YY


def demo_hampel():
    randint = np.random.randint
    Y = 5000 + np.random.randn(1000)
    outliers = randint(0, 1000, size=(10,))
    Y[outliers] = Y[outliers] + randint(1000, size=(10,))
    YY, res = HampelFilter(dx=3, t=3, fulloutput=True)(Y)
    YY1, res1 = HampelFilter(dx=1, t=3, adaptive=0.1, fulloutput=True)(Y)
    YY2, res2 = HampelFilter(dx=3, t=0, fulloutput=True)(Y)  # median
    plt.figure(1)
    plot_hampel(Y, YY, res)
    plt.title('Standard HampelFilter')
    plt.figure(2)
    plot_hampel(Y, YY1, res1)
    plt.title('Adaptive HampelFilter')
    plt.figure(3)
    plot_hampel(Y, YY2, res2)
    plt.title('Median filter')
    plt.show('hold')


def plot_hampel(Y, YY, res):
    X = np.arange(len(YY))
    plt.plot(X, Y, 'b.')  # Original Data
    plt.plot(X, YY, 'r')  # Hampel Filtered Data
    plt.plot(X, res['Y0'], 'b--')  # Nominal Data
    plt.plot(X, res['LB'], 'r--')  # Lower Bounds on Hampel Filter
    plt.plot(X, res['UB'], 'r--')  # Upper Bounds on Hampel Filter
    i = res['outliers']
    plt.plot(X[i], Y[i], 'ks')  # Identified Outliers
    # plt.show('hold')


def test_tide_filter():
    # import statsmodels.api as sa
    import wafo.spectrum.models as sm
    sd = 10
    Sj = sm.Jonswap(Hm0=4.*sd)
    S = Sj.tospecdata()

    q = (0.1 * sd) ** 2   # variance of process noise s the car operates
    r = (100 * sd) ** 2  # variance of measurement error
    b = 0  # no system input
    u = 0  # no system input

    from scipy.signal import butter, filtfilt, lfilter_zi  # lfilter,
    freq_tide = 1. / (12 * 60 * 60)
    freq_wave = 1. / 10
    freq_filt = freq_wave / 10
    dt = 1.
    freq = 1. / dt
    fn = (freq / 2)

    P = 10 * np.diag([1, 0.01])
    R = r
    H = np.atleast_2d([1, 0])

    F = np.atleast_2d([[0, 1],
                       [0, 0]])
    A, Q = lti_disc(F, L=None, Q=np.diag([0, q]), dt=dt)

    t = np.arange(0, 60 * 12, 1. / freq)
    w = 2 * np.pi * freq  # 1 Hz
    tide = 100 * np.sin(freq_tide * w * t + 2 * np.pi / 4) + 100
    y = tide + S.sim(len(t), dt=1. / freq)[:, 1].ravel()
#     lowess = sa.nonparametric.lowess
#     y2 = lowess(y, t, frac=0.5)[:,1]

    filt = Kalman(R=R, x=np.array([[tide[0]], [0]]), P=P, A=A, Q=Q, H=H, B=b)
    filt2 = Kalman(R=R, x=np.array([[tide[0]], [0]]), P=P, A=A, Q=Q, H=H, B=b)
    # y = tide + 0.5 * np.sin(freq_wave * w * t)
    # Butterworth filter
    b, a = butter(9, (freq_filt / fn), btype='low')
    # y2 = [lowess(y[max(i-60,0):i + 1], t[max(i-60,0):i + 1], frac=.3)[-1,1]
    #    for i in range(len(y))]
    # y2 = [lfilter(b, a, y[:i + 1])[i] for i in range(len(y))]
    # y3 = filtfilt(b, a, y[:16]).tolist() + [filtfilt(b, a, y[:i + 1])[i]
    #    for i in range(16, len(y))]
    # y0 = medfilt(y, 41)
    _zi = lfilter_zi(b, a)
    # y2 = lfilter(b, a, y)#, zi=y[0]*zi)  # standard filter
    y3 = filtfilt(b, a, y)  # filter with phase shift correction
    y4 = []
    y5 = []
    for _i, j in enumerate(y):
        tmp = filt(j, u=u).ravel()
        tmp = filt2(tmp[0], u=u).ravel()
#         if i==0:
#             print(filt.x)
#             print(filt2.x)
        y4.append(tmp[0])
        y5.append(tmp[1])
    _y0 = medfilt(y4, 41)
    print(filt.P)
    # plot

    plt.plot(t, y, 'r.-', linewidth=2, label='raw data')
    # plt.plot(t, y2, 'b.-', linewidth=2, label='lowess @ %g Hz' % freq_filt)
    # plt.plot(t, y2, 'b.-', linewidth=2, label='filter @ %g Hz' % freq_filt)
    plt.plot(t, y3, 'g.-', linewidth=2, label='filtfilt @ %g Hz' % freq_filt)
    plt.plot(t, y4, 'k.-', linewidth=2, label='kalman')
    # plt.plot(t, y5, 'k.', linewidth=2, label='kalman2')
    plt.plot(t, tide, 'y-', linewidth=2, label='True tide')
    plt.legend(frameon=False, fontsize=14)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show('hold')


def test_smooth():
    t = np.linspace(-4, 4, 500)
    y = np.exp(-t ** 2) + np.random.normal(0, 0.05, t.shape)
    n = 11
    ysg = SavitzkyGolay(n, degree=1, diff_order=0)(y)

    plt.plot(t, y, t, ysg, '--')
    plt.show('hold')


def test_hodrick_cardioid():
    t = np.linspace(0, 2 * np.pi, 1000)
    cos = np.cos
    sin = np.sin
    randn = np.random.randn
    x0 = 2 * cos(t) * (1 - cos(t))
    x = x0 + randn(t.size) * 0.1
    y0 = 2 * sin(t) * (1 - cos(t))
    y = y0 + randn(t.size) * 0.1
    smooth = HodrickPrescott(w=20000)
    # smooth = HampelFilter(adaptive=50)
    z = smooth(x) + 1j * smooth(y)
    plt.plot(x0, y0, 'y',
             x, y, 'r.',
             z.real, z.imag, 'k', linewidth=2)
    plt.show('hold')


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

if __name__ == '__main__':
    # test_docstrings()
    # test_kalman_sine()
    # test_tide_filter()
    # demo_hampel()
    # test_kalman()
    # test_smooth()
    # test_hodrick_cardioid()
    test_smoothn_1d()
    # test_smoothn_cardioid()
