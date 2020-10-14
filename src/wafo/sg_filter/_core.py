import numpy as np
from numpy import (pi, linalg, concatenate, sqrt)
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import scipy.optimize as optimize
from scipy.signal import _savitzky_golay
from scipy.ndimage import convolve1d
from scipy.ndimage.morphology import distance_transform_edt
import warnings
from wafo.dctpack import dctn, idctn


__all__ = ['SavitzkyGolay', 'Kalman', 'HodrickPrescott', 'smoothn',
           'HampelFilter', 'SmoothNd', 'noise']

# noise = np.random.randn(2**8)/10
noise = [-0.0490483773397234, 0.07101522794824691, 0.043129450693516064, 0.07858516767729644, -0.04489848540755172, -0.012710090966021995, 0.022967442347004003, -0.1593564930543959, 0.14752458454255937, -0.1220055819473534, -0.030151822649201642, 0.009880871420067841, 0.0401050562035102, -0.10931262882008379, -0.14550620919429919, -0.06632845063372966, 0.07773893951749064, -0.009527784302072342, 0.06002486046176557, 0.11972670522904964, -0.14436696992162384, 0.06009486605688445, -0.05802790838575894, 0.16964239368289297, 0.09088881573238144, -0.003398259264109856, 0.059830811447018004, -0.08189024981767952, -0.05455483548325317, 0.056651518536760745, -0.05211609539593189, -0.07848323826083178, -0.03921692262168154, -0.04755275276447492, -0.05855172473750038, 0.06480280696345982, -0.05237889271019207, -0.05891912551792037, -0.04045907452295067, -0.09058522124919187, 0.1406515441218336, 0.15557979603588584, -0.09096515320242772, 0.1724190189462715, -0.04978942687488187, -0.0855435866249914, 0.09439718859306868, -0.14758639479507882, -0.07225230856508442, 0.008364508824556314, 0.06704423745152435, -0.01718113731784587, 0.07473943576290255, 0.028133087670974395, 0.026270590730899095, 0.13175770484080895, -0.01821821552644416, 0.11325945472394446, 0.04694754851273185, -0.23899404962137366, -0.1528175431702195, 0.151870532421663, -0.07353204927616248, 0.11604199430172217, -0.09111623325687843, -0.11887366073405607, -0.029872397510562025, 0.047672685028458936, -0.18340065977268627, 0.06896217941210328, 0.042997912112300564, 0.15416998299846174, -0.0386283794526545, 0.14070600624229804, 0.020984623041646142, -0.1892741373898864, 0.03253519397457513, -0.06182705494266229, -0.1326495728975159, 0.026234150321195537, 0.0550541170409239, 0.029275813927566702, 0.042742104678489906, -0.2170004668366198, -0.00035991761313413197, -0.0638872684868346, -0.11769436550364845, -0.017792813824766808, -0.022786402363044914, -0.10668279890162544, 0.05979507681729831, -0.1008100479486818, 0.0703474638610785, 0.1630534776572414, 0.06682406484481357, -0.0527228810042394, -0.046515310355062636, 0.04609515154732255, 0.11503753838360875, 0.11517599661346192, -0.05596425736274815, -0.06149119758833357, 0.10599964719188917, -0.012076380140185552, 0.0436828262270732, -0.03910174791470852, -0.03263251315745414, -0.012513843545007558, 0.004590611827089213, 0.0762719171282112, 0.06497715695411535, -0.003280826953794463, 0.13524154885565484, -0.020441364843140027, -0.09488214173137496, 0.1385755359902911, -0.23883052310744746, -0.10110537386421652, -0.1588981058869149, 0.06645444828058467, -0.2103306051703948, 0.15215327561190056, -0.03582175680076989, 0.013593833383013293, -0.11542058494732854, -0.05613268116816099, 0.012711037661355899, 0.04242805633100794, -0.011799315325220794, 0.12141794601099387, 0.054285270560662645, 0.07549385527022169, -0.04549437694653443, 0.11009856942530691, 0.05233482224379645, -0.042246830306136955, -0.1737197924666796, -0.10589427330127077, 0.04895472597843757, 0.06756519832636187, 0.083376600742245, -0.07502859751328732, -0.09493802498812245, -0.01058967186080922, -0.23759763247649018, 0.08439637862616411, -0.2021754550870607, 0.07365816800912013, 0.07435401663661081, 0.047992791325423556, -0.005250092450514997, 0.1693610927865244, 0.030338113772413154, -0.18010537945928004, 0.01744129379023785, 0.1902505975745975, -0.004598733688659104, 0.13663542585715657, -0.04100719174496187, -0.15406303185009937, -0.05297118247908407, 0.04435144348234146, 0.022377061632995063, 0.05491057192661079, -0.08473062163887303, -0.03907641665824873, 0.008686833182075315, -0.06053451866471732, -0.051735892949367854, -0.1902071038920444, 0.11508817132666356, 0.08903045262390544, -0.028537865059606825, -0.07160660523436188, 0.05994760363400714, 0.03637820115278829, 0.027604828657436364, 0.04168122074675033, -0.021707671111253164, 0.06770739385070886, -0.04848505599153394, -0.14377853380839264, 0.17448368721141166, -0.05972663746675887, -0.1615729579782888, -0.09508063624538736, -0.05501964872264433, -0.14370852991216054, -0.1025241548369181, -0.14751000180775747, -0.05402976681470177, -0.05847606145915367, 0.015603559358987138, 0.040327317968149784, 0.015596571983936361, 0.08721780106901023, 0.13669912032986667, -0.07070030973798198, 0.04821782065785363, 0.05266507025196321, -0.013775127999269254, 0.07032239356769251, 0.04685048562398681, 0.004648720572365418, -0.19364418622487742, 0.013662994215276983, 0.04703494294810789, 0.04863794676207257, -0.09883919097676001, -0.004798538894290822, -0.22183503742087135, 0.062096556899520906, 0.07098373434409047, -0.05335639719762188, -0.09150459514627822, -0.1329311651202703, -0.037376442133682145, 0.1238732233009325, -0.01232052797514208, 0.007151238520555889, -0.04772828461473576, -0.029830395387364726, -0.03277336781995001, 0.09964048194066656, 0.09306408040020697, -0.03761782769337173, 0.07059549032551317, -0.15490333414875848, 0.12599077783991805, 0.23520519946427365, 0.021640305946603107, 0.014851729969403227, -0.039035437601777224, -0.12087588583684257, -0.07207855860199022, -0.002800081649022032, 0.2543907308881692, -0.07966223382328289, -0.1014419766425384, -0.11243061225437859, -0.08744845956375621, -0.05540140267769189, -0.04995531421885231, -0.13274847220288336, 0.06435474943034288, 0.015640361472736924, -0.11210644205346465, -0.04080648821849449, -0.011452694652695428, 0.22044736923317904, 0.024322228245949113, 0.09622705616884256, 0.05793212184654495, -0.10620553812614748, 0.06762504431789758, 0.19135075519983785]  # nopep8


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


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
    >>> import wafo.sg_filter as ws
    >>> t = np.linspace(-4, 4, 500)
    >>> noise = np.random.normal(0, 0.05, t.shape)
    >>> noise = np.sqrt(0.05)*np.sin(100*t)
    >>> y = np.exp( -t**2 ) + noise
    >>> ysg = SavitzkyGolay(n=20, degree=2).smooth(y)
    >>> np.allclose(ysg[:3], [ 0.01345312,  0.01164172,  0.00992839])
    True
    >>> ysm = SavitzkyGolay(n=20, degree=2, mode='mirror').smooth(y)
    >>> np.allclose(ysm[:3], [-0.01604804, -0.00592883,  0.0035858 ])
    True
    >>> ysc = SavitzkyGolay(n=20, degree=2, mode='constant').smooth(y)
    >>> np.allclose(ysc[:3], [-0.00279797,  0.00519541,  0.00666146])
    True
    >>> ysn = SavitzkyGolay(n=20, degree=2, mode='nearest').smooth(y)
    >>> np.allclose(ysn[:3], [ 0.08711171,  0.0846945 ,  0.07587448])
    True
    >>> ysw = SavitzkyGolay(n=20, degree=2, mode='wrap').smooth(y)
    >>> np.allclose(ysw[:3], [-0.00208422, -0.00201491,  0.00201772])
    True
    >>> np.allclose(SavitzkyGolay(n=20, degree=2).smooth_last(y),
    ...             0.004921382626100505)
    True

    import matplotlib.pyplot as plt
    h = plt.plot(t, y, label='Noisy signal')
    h1 = plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    h2 = plt.plot(t, ysg, 'r', label='Filtered signal')
    h3 = plt.legend()
    h4 = plt.title('Savitzky-Golay')
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

    def __init__(self, n, degree=1, diff_order=0, delta=1.0, axis=-1,
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

    def smooth_last(self, signal, k=1):
        coeff = self._coeff
        n = (np.size(coeff) - 1) // 2
        y = np.squeeze(signal)
        if n == 0:
            return y
        if y.ndim > 1:
            coeff.shape = (-1, 1)
        first_vals = y[0] - np.abs(y[n:0:-1] - y[0])
        last_vals = y[-1] + np.abs(y[-2:-n - 2:-1] - y[-1])
        y = concatenate((first_vals, y, last_vals))
        return (y[-2 * n - 1 - k:-k] * coeff).sum(axis=0)

    def __call__(self, signal):
        return self.smooth(signal)

    def smooth(self, signal):
        dtype = np.result_type(signal, np.float)
        x = np.asarray(signal, dtype=dtype)

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
    >>> [x,y,z] = np.meshgrid(yp, yp, yp, sparse=True)
    >>> f = x*np.exp(-x**2-y**2-z**2)
    >>> var0 = 0.5  # noise variance
    >>> fn = f + np.sqrt(var0)*np.random.randn(*f.shape)
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
    sh1 = np.ones((d,), dtype=np.int64)
    cos = np.cos
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
    def _smoothness_par(h):
        return (((1 + sqrt(1 + 8 * h)) / 4. / h) ** 2 - 1) / 16

    def _smoothness_limits(self, n):
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

        s_min = self._smoothness_par(h_max)
        s_max = self._smoothness_par(h_min)
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
            if s is None:
                return 1.0
            return 1. / (1 + s * Lambda ** 2)
        return gamma

    @staticmethod
    def _initial_guess(y, I):
        # Initial Guess with weighted/missing data
        # nearest neighbor interpolation (in case of missing values)
        z = y
        if (1 - I).any():
            notI = ~I
            z, L = distance_transform_edt(notI, return_indices=True)
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
        _assert(np.all(0 <= weights), 'Weights must all be >=0')
        return weights / weights.max()

    @staticmethod
    def _studentized_residuals(r, I, h):
        median_abs_deviation = np.median(np.abs(r[I] - np.median(r[I])))
        return np.abs(r / (1.4826 * median_abs_deviation) / sqrt(1 - h))

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
            if np.abs(np.log10(s) - np.log10(self.s_min)) < self.errp:
                warnings.warn("""s = %g: the lower bound for s has been reached.
            Put s as an input variable if required.""" % s)
            elif np.abs(np.log10(s) - np.log10(self.s_max)) < self.errp:
                warnings.warn("""s = %g: the Upper bound for s has been reached.
            Put s as an input variable if required.""" % s)

    def gcv(self, p, aow, DCTy, y, Wtot):
        # Search the smoothing parameter s that minimizes the GCV score
        s = 10.0 ** p
        gamma_s = self.gamma(s)
        if aow > 0.9:
            # aow = 1 means that all of the data are equally weighted
            # very much faster: does not require any inverse DCT
            residual = DCTy.ravel() * (gamma_s.ravel() - 1)
        else:
            # take account of the weights to calculate RSS:
            is_finite = self.is_finite
            yhat = idctn(gamma_s * DCTy)
            residual = sqrt(Wtot[is_finite]) * (y[is_finite] - yhat[is_finite])

        TrH = gamma_s.sum()
        RSS = linalg.norm(residual)**2  # Residual sum-of-squares
        GCVscore = RSS / self.nof / (1.0 - TrH / y.size) ** 2
        return GCVscore

    def _smooth(self, z, s):
        auto_smooth = self.auto_smooth
        norm = linalg.norm
        y = self.y
        Wtot = self.Wtot
        gamma_s = self.gamma(s)
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
                gamma_s = self.gamma(s)
            z0 = z
            z = self.RF * idctn(gamma_s * DCTy) + (1 - self.RF) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = norm(z0.ravel() - z.ravel()) / norm(z.ravel())
            converged = tol <= self.tolz or not self.is_weighted
            if converged:
                break
        return z, s, converged

    def __call__(self, z, s):
        z, s, converged = self._smooth(z, s)
        if self.robust:
            # -- Robust Smoothing: iteratively re-weighted process
            h = self._average_leverage(s, self.N)
            self.Wtot = self.W * self.robust_weights(self.y - z,
                                                     self.is_finite, h)
            # re-initialize for another iterative weighted process
            self.is_weighted = True
        return z, s, converged


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
            msg = """Maximum number of iterations (%d) has been exceeded.
            Increase MaxIter option or decrease TolZ value.""" % (self.maxiter)
            warnings.warn(msg)

        _filter.check_smooth_parameter(s)

        if self.fulloutput:
            return z, s
        return z


def smoothn(data, s=None, weight=None, robust=False, z0=None, tolz=1e-3,
            maxiter=100, fulloutput=False):
    """
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
    >>> import wafo.sg_filter as ws
    >>> x = np.linspace(0, 100, 2**8)
    >>> noise = np.random.randn(2**8)/10
    >>> noise = ws.noise
    >>> y = np.cos(x/10)+(x/50)**2 + noise
    >>> y[np.r_[70, 75, 80]] = np.array([5.5, 5, 6])
    >>> y[181] = np.nan
    >>> z = ws.smoothn(y) # Regular smoothing
    >>> np.allclose(z[:3], [ 0.99517904,  0.99372346,  0.99079798])
    True
    >>> zr = ws.smoothn(y,robust=True) #  Robust smoothing
    >>> np.allclose(zr[:3], [ 1.01190564,  1.00976197,  1.00513244])
    True

    h=plt.subplot(121),
    h = plt.plot(x,y,'r.',x,z,'k',linewidth=2)
    h=plt.title('Regular smoothing')
    h=plt.subplot(122)
    h=plt.plot(x,y,'r.',x,zr,'k',linewidth=2)
    h=plt.title('Robust smoothing')

     2-D example
    >>> xp = np.r_[0:1:.02]
    >>> [x,y] = np.meshgrid(xp,xp)
    >>> f = np.exp(x+y) + np.sin((x-2*y)*3)
    >>> fn = f + np.random.randn(*f.shape)*0.5
    >>> fs = smoothn(fn)

    h=plt.subplot(121),
    h=plt.contourf(xp, xp, fn)
    h=plt.subplot(122)
    h=plt.contourf(xp, xp, fs)

    2-D example with missing data
    >>> import wafo.demos as wd
    >>> n = 256
    >>> x0, y0, z0 = wd.peaks(n)

    z = z0 + rand(size(y0))*2
    I = randperm(n**2)
    z[I(1:n^2*0.5)] = np.NaN;  # lose 1/2 of data
    z[40:90, 140:190] = np.NaN;  # create a hole
    zs = smoothn(z)

    plt.subplot(2,2,1)
    plt.imagesc(y)  # , axis equal off
    plt.title('Noisy corrupt data')
    plt.subplot(223)
    plt.imagesc(z)   # , axis equal off
    plt.title('Recovered data ...')
    plt.subplot(224)
    plt.imagesc(y0)  # , axis equal off
    plt.title('... compared with original data')

     3-D example
    [x,y,z] = meshgrid(-2:.2:2);
    xslice = [-0.8,1]; yslice = 2; zslice = [-2,0];
    vn = x.*exp(-x.^2-y.^2-z.^2) + randn(size(x))*0.06;
    subplot(121), slice(x,y,z,vn,xslice,yslice,zslice,'cubic')
    title('Noisy data')
    v = smoothn(vn);
    subplot(122), slice(x,y,z,v,xslice,yslice,zslice,'cubic')
    title('Smoothed data')


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

    See also
    -------
    SmoothNd

    -- Damien Garcia -- 2009/03, revised 2010/11
    Visit
    http://www.biomecardio.com/matlab/smoothn.html
    for more details about SMOOTHN
    """
    return SmoothNd(s, weight, robust, z0, tolz, maxiter, fulloutput)(data)


class HodrickPrescott(object):

    """Smooth data with a Hodrick-Prescott filter.

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
    >>> import wafo.sg_filter as ws
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = ws.HodrickPrescott(w=10000)(y)

    import matplotlib.pyplot as plt
    h = plt.plot(t, y, label='Noisy signal')
    h1 = plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    h2 = plt.plot(t, ysg, 'r', label='Filtered signal')
    h3 = plt.legend()
    h4 = plt.title('Hodrick-Prescott')
    plt.show()

    References
    ----------
    .. [1] E. T. Whittaker, On a new method of graduation. In proceedings of
        the Edinburgh Mathematical association., 1923, 78, pp 88-89.
    .. [2] R. Hodrick and E. Prescott, Postwar U.S. business cycles: an
        empirical investigation,
        Journal of money, credit and banking, 1997, 29 (1), pp 1-16.
    .. [3] Kim Hyeongwoo, Hodrick-Prescott filter,
        2004, www.auburn.edu/~hzk0001/hpfilter.pdf
    """

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

    """
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

    Examples 
    --------
    # Automobile Voltimeter:
    >>> import wafo.sg_filter as ws
    >>> V0 = 12    # Define the system as a constant of 12 volts
    >>> h = 1      # voltimeter measure the voltage itself
    >>> q = 1e-5   # variance of process noise s the car operates
    >>> r = 0.1**2 # variance of measurement error
    >>> b = 0      # no system input
    >>> u = 0      # no system input
    >>> filt = ws.Kalman(R=r, A=1, Q=q, H=h, B=b)

    # Generate random voltages and watch the filter operate.
    >>> n = 50
    >>> truth = np.random.randn(n)*np.sqrt(q) + V0
    >>> z = truth + np.random.randn(n)*np.sqrt(r) # measurement
    >>> x = np.zeros(n)

    >>> for i, zi in enumerate(z):
    ...     x[i] = filt(zi, u) #  perform a Kalman filter iteration

    import matplotlib.pyplot as plt
    hz = plt.plot(z,'r.', label='observations')

    # a-posteriori state estimates:
    hx = plt.plot(x,'b-', label='Kalman output')
    ht = plt.plot(truth,'g-', label='true voltage')
    h = plt.legend()
    h1 = plt.title('Automobile Voltimeter Example')
    plt.show()

    """

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

    @staticmethod
    def _none_or_atleast_2d(a):
        if a is not None:
            return np.atleast_2d(a)
        return a

    @property
    def A(self):
        return self._a

    @A.setter
    def A(self, a):
        self._a = self._none_or_atleast_2d(a)

    def _set_A(self, n):
        if self.A is None:
            self.A = np.eye(n)

    @property
    def Q(self):
        return self._q

    @Q.setter
    def Q(self, q):
        self._q = self._none_or_atleast_2d(q)

    def _set_Q(self, n):
        if self.Q is None:
            self.Q = np.zeros((n, n))

    @property
    def H(self):
        return self._h

    @H.setter
    def H(self, h):
        self._h = self._none_or_atleast_2d(h)

    def _set_H(self, n):
        if self.H is None:
            self.H = np.eye(n)

    @property
    def P(self):
        return self._p

    @P.setter
    def P(self, p):
        self._p = self._none_or_atleast_2d(p)

    def _set_P(self, HI):
        if self.P is None:
            self.P = np.dot(np.dot(HI, self.R), HI.T)

    def _init_first(self, n):
        self._set_A(n)
        self._set_Q(n)
        self._set_H(n)
        try:
            HI = np.linalg.inv(self.H)
        except Exception:
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
        # return np.dot(np.eye(len(P)) - K * self.H, P)

    def _filter_main(self, z, u):
        """ This is the code which implements the discrete Kalman filter:
        """
        P = self._predict_covariance(self.P)
        x = self._predict_state(self.x, u)

        K = self._compute_gain(P)

        self.P = self._update_covariance(P, K)
        self.x = self._update_state_from_observation(x, z, K)

        return self.x

    def __call__(self, z, u=0):
        return self._filter(z, u)


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

    @staticmethod
    def _check(dx):
        _assert(np.isscalar(dx), 'DX must be a scalar.')
        _assert(0 < dx, 'DX must be larger than zero.')

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
                    S0Rel = np.abs((S0Tmp[j - 1] - S0Tmp[j]) /
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
        np.putmask(YY, outliers, Y0)        # YY[outliers] = Y0[outliers]
        self.outliers = outliers
        self.num_outliers = outliers.sum()
        self.ADX = ADX
        self.Y0 = Y0
        if self.fulloutput:
            return YY, dict(outliers=outliers, Y0=Y0,
                            LB=self.LB, UB=self.UB, ADX=ADX)
        return YY


if __name__ == '__main__':
    from wafo.testing import test_docstrings
    test_docstrings(__file__)
