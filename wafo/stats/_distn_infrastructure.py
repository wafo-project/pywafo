from __future__ import absolute_import
from scipy.stats._distn_infrastructure import *  # @UnusedWildImport
from scipy.stats._distn_infrastructure import (_skew,  # @UnusedImport
    _kurtosis, _lazywhere, _ncx2_log_pdf,  # @IgnorePep8 @UnusedImport
    _ncx2_pdf,  _ncx2_cdf)  # @UnusedImport @IgnorePep8
from .estimation import FitDistribution
from ._constants import _XMAX


_doc_default_example = """\
Examples
--------
>>> from wafo.stats import %(name)s
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate a few first moments:

%(set_vals_stmt)s
>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Display the probability density function (``pdf``):

>>> x = np.linspace(%(name)s.ppf(0.01, %(shapes)s),
...                 %(name)s.ppf(0.99, %(shapes)s), 100)
>>> ax.plot(x, %(name)s.pdf(x, %(shapes)s),
...        'r-', lw=5, alpha=0.6, label='%(name)s pdf')

Alternatively, the distribution object can be called (as a function)
to fix the shape, location and scale parameters. This returns a "frozen"
RV object holding the given parameters fixed.

Freeze the distribution and display the frozen ``pdf``:

>>> rv = %(name)s(%(shapes)s)
>>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

Check accuracy of ``cdf`` and ``ppf``:

>>> vals = %(name)s.ppf([0.001, 0.5, 0.999], %(shapes)s)
>>> np.allclose([0.001, 0.5, 0.999], %(name)s.cdf(vals, %(shapes)s))
True

Generate random numbers:

>>> r = %(name)s.rvs(%(shapes)s, size=1000)

And compare the histogram:

>>> ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()

Compare ML and MPS method
>>> phat = %(name)s.fit2(R, method='ml');>>> phat.plotfitsummary()
>>> plt.figure(plt.gcf().number+1)
>>> phat2 = %(name)s.fit2(R, method='mps')
>>> phat2.plotfitsummary(); plt.figure(plt.gcf().number+1)

Fix loc=0 and estimate shapes and scale
>>> phat3 = %(name)s.fit2(R, scale=1, floc=0, method='mps')
>>> phat3.plotfitsummary(); plt.figure(plt.gcf().number+1)

Accurate confidence interval with profile loglikelihood
>>> lp = phat3.profile()
>>> lp.plot()
>>> pci = lp.get_bounds()

"""


# Frozen RV class
class rv_frozen(object):
    ''' Frozen continous or discrete 1D Random Variable object (RV)

    Methods
    -------
    rvs(size=1)
        Random variates.
    pdf(x)
        Probability density function.
    cdf(x)
        Cumulative density function.
    sf(x)
        Survival function (1-cdf --- sometimes more accurate).
    ppf(q)
        Percent point function (inverse of cdf --- percentiles).
    isf(q)
        Inverse survival function (inverse of sf).
    stats(moments='mv')
        Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
    moment(n)
        n-th order non-central moment of distribution.
    entropy()
        (Differential) entropy of the RV.
    interval(alpha)
        Confidence interval with equal areas around the median.
    expect(func, lb, ub, conditional=False)
        Calculate expected value of a function with respect to the
        distribution.
    '''
    def __init__(self, dist, *args, **kwds):
        # create a new instance
        self.dist = dist  # .__class__(**dist._ctor_param)
        shapes, loc, scale = self.dist._parse_args(*args, **kwds)
        if isinstance(dist, rv_continuous):
            self.par = shapes + (loc, scale)
        else:  # rv_discrete
            self.par = shapes + (loc,)
        self.a = self.dist.a
        self.b = self.dist.b
        self.shapes = self.dist.shapes
    # @property
    # def shapes(self):
    #     return self.dist.shapes

    @property
    def random_state(self):
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self.dist._random_state = check_random_state(seed)

    def pdf(self, x):
        ''' Probability density function at x of the given RV.'''
        return self.dist.pdf(x, *self.par)

    def logpdf(self, x):
        return self.dist.logpdf(x, *self.par)

    def cdf(self, x):
        '''Cumulative distribution function at x of the given RV.'''
        return self.dist.cdf(x, *self.par)

    def logcdf(self, x):
        return self.dist.logcdf(x, *self.par)

    def ppf(self, q):
        '''Percent point function (inverse of cdf) at q of the given RV.'''
        return self.dist.ppf(q, *self.par)

    def isf(self, q):
        '''Inverse survival function at q of the given RV.'''
        return self.dist.isf(q, *self.par)

    def rvs(self, size=None, random_state=None):
        kwds = {'size': size, 'random_state': random_state}
        return self.dist.rvs(*self.par, **kwds)

    def sf(self, x):
        '''Survival function (1-cdf) at x of the given RV.'''
        return self.dist.sf(x, *self.par)

    def logsf(self, x):
        return self.dist.logsf(x, *self.par)

    def stats(self, moments='mv'):
        ''' Some statistics of the given RV'''
        kwds = dict(moments=moments)
        return self.dist.stats(*self.par, **kwds)

    def median(self):
        return self.dist.median(*self.par)

    def mean(self):
        return self.dist.mean(*self.par)

    def var(self):
        return self.dist.var(*self.par)

    def std(self):
        return self.dist.std(*self.par)

    def moment(self, n):
        return self.dist.moment(n, *self.par)

    def entropy(self):
        return self.dist.entropy(*self.par)

    def pmf(self, k):
        '''Probability mass function at k of the given RV'''
        return self.dist.pmf(k, *self.par)

    def logpmf(self, k):
        return self.dist.logpmf(k, *self.par)

    def interval(self, alpha):
        return self.dist.interval(alpha, *self.par)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        if isinstance(self.dist, rv_continuous):
            a, loc, scale = self.par[:-2], self.par[:-2], self.par[-1]
            return self.dist.expect(func, a, loc, scale, lb, ub, conditional,
                                    **kwds)

        a, loc = self.par[:-1], self.par[-1]
        if kwds:
            raise ValueError("Discrete expect does not accept **kwds.")
        return self.dist.expect(func, a, loc, lb, ub, conditional)


def freeze(self, *args, **kwds):
    """Freeze the distribution for the given arguments.

    Parameters
    ----------
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution.  Should include all
        the non-optional arguments, may include ``loc`` and ``scale``.

    Returns
    -------
    rv_frozen : rv_frozen instance
        The frozen distribution.

    """
    return rv_frozen(self, *args, **kwds)


def link(self, x, logSF, theta, i):
    '''
    Return theta[i] as function of quantile, survival probability and
        theta[j] for j!=i.

    Parameters
    ----------
    x : quantile
    logSF : logarithm of the survival probability
    theta : list
        all distribution parameters including location and scale.

    Returns
    -------
    theta[i] : real scalar
        fixed distribution parameter theta[i] as function of x, logSF and
        theta[j] where j != i.

    LINK is a function connecting the fixed distribution parameter theta[i]
    with the quantile (x) and the survival probability (SF) and the
    remaining free distribution parameters theta[j] for j!=i, i.e.:
        theta[i] = link(x, logSF, theta, i),
    where logSF = log(Prob(X>x; theta)).

    See also
    estimation.Profile
    '''
    return self._link(x, logSF, theta, i)


def _link(self, x, logSF, theta, i):
    msg = ('Link function not implemented for the %s distribution' %
           self.name)
    raise NotImplementedError(msg)


def nlogps(self, theta, x):
    """ Moran's negative log Product Spacings statistic

        where theta are the parameters (including loc and scale)

        Note the data in x must be sorted

    References
    -----------

    R. C. H. Cheng; N. A. K. Amin (1983)
    "Estimating Parameters in Continuous Univariate Distributions with a
    Shifted Origin.",
    Journal of the Royal Statistical Society. Series B (Methodological),
    Vol. 45, No. 3. (1983), pp. 394-403.

    R. C. H. Cheng; M. A. Stephens (1989)
    "A Goodness-Of-Fit Test Using Moran's Statistic with Estimated
    Parameters", Biometrika, 76, 2, pp 385-392

    Wong, T.S.T. and Li, W.K. (2006)
    "A note on the estimation of extreme value distributions using maximum
    product of spacings.",
    IMS Lecture Notes Monograph Series 2006, Vol. 52, pp. 272-283
    """

    try:
        loc = theta[-2]
        scale = theta[-1]
        args = tuple(theta[:-2])
    except IndexError:
        raise ValueError("Not enough input arguments.")
    if not self._argcheck(*args) or scale <= 0:
        return inf
    x = asarray((x - loc) / scale)
    cond0 = (x <= self.a) | (self.b <= x)
    Nbad = np.sum(cond0)
    if Nbad > 0:
        x = argsreduce(~cond0, x)[0]

    lowertail = True
    if lowertail:
        prb = np.hstack((0.0, self.cdf(x, *args), 1.0))
        dprb = np.diff(prb)
    else:
        prb = np.hstack((1.0, self.sf(x, *args), 0.0))
        dprb = -np.diff(prb)

    logD = log(dprb)
    dx = np.diff(x, axis=0)
    tie = (dx == 0)
    if any(tie):
        # TODO : implement this method for treating ties in data:
        # Assume measuring error is delta. Then compute
        # yL = F(xi-delta,theta)
        # yU = F(xi+delta,theta)
        # and replace
        # logDj = log((yU-yL)/(r-1)) for j = i+1,i+2,...i+r-1

        # The following is OK when only minimization of T is wanted
        i_tie, = np.nonzero(tie)
        tiedata = x[i_tie]
        logD[i_tie + 1] = log(self._pdf(tiedata, *args)) - log(scale)

    finiteD = np.isfinite(logD)
    nonfiniteD = 1 - finiteD
    Nbad += np.sum(nonfiniteD, axis=0)
    if Nbad > 0:
        T = -np.sum(logD[finiteD], axis=0) + 100.0 * np.log(_XMAX) * Nbad
    else:
        T = -np.sum(logD, axis=0)
    return T


def _reduce_func(self, args, options):
    # First of all, convert fshapes params to fnum: eg for stats.beta,
    # shapes='a, b'. To fix `a`, can specify either `f1` or `fa`.
    # Convert the latter into the former.
    kwds = options.copy()
    if self.shapes:
        shapes = self.shapes.replace(',', ' ').split()
        for j, s in enumerate(shapes):
            val = kwds.pop('f' + s, None) or kwds.pop('fix_' + s, None)
            if val is not None:
                key = 'f%d' % j
                if key in kwds:
                    raise ValueError("Duplicate entry for %s." % key)
                else:
                    kwds[key] = val

    args = list(args)
    Nargs = len(args)
    fixedn = []
    names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
    x0 = []
    for n, key in enumerate(names):
        if key in kwds:
            fixedn.append(n)
            args[n] = kwds.pop(key)
        else:
            x0.append(args[n])
    method = kwds.pop('method', 'ml').lower()
    if method.startswith('mps'):
        fitfun = self.nlogps
    else:
        fitfun = self._penalized_nnlf

    if len(fixedn) == 0:
        func = fitfun
        restore = None
    else:
        if len(fixedn) == Nargs:
            raise ValueError(
                "All parameters fixed. There is nothing to optimize.")

        def restore(args, theta):
            # Replace with theta for all numbers not in fixedn
            # This allows the non-fixed values to vary, but
            #  we still call self.nnlf with all parameters.
            i = 0
            for n in range(Nargs):
                if n not in fixedn:
                    args[n] = theta[i]
                    i += 1
            return args

        def func(theta, x):
            newtheta = restore(args[:], theta)
            return fitfun(newtheta, x)

    return x0, func, restore, args


def fit(self, data, *args, **kwargs):
    """
    Return ML/MPS estimate for shape, location, and scale parameters from data.

    ML and MPS stands for Maximum Likelihood and Maximum Product Spacing,
    respectively.  Starting estimates for
    the fit are given by input arguments; for any arguments not provided
    with starting estimates, ``self._fitstart(data)`` is called to generate
    such.

    One can hold some parameters fixed to specific values by passing in
    keyword arguments ``f0``, ``f1``, ..., ``fn`` (for shape parameters)
    and ``floc`` and ``fscale`` (for location and scale parameters,
    respectively).

    Parameters
    ----------
    data : array_like
        Data to use in calculating the MLEs.
    args : floats, optional
        Starting value(s) for any shape-characterizing arguments (those not
        provided will be determined by a call to ``_fitstart(data)``).
        No default value.
    kwds : floats, optional
        Starting values for the location and scale parameters; no default.
        Special keyword arguments are recognized as holding certain
        parameters fixed:

        - f0...fn : hold respective shape parameters fixed.
          Alternatively, shape parameters to fix can be specified by name.
          For example, if ``self.shapes == "a, b"``, ``fa``and ``fix_a``
          are equivalent to ``f0``, and ``fb`` and ``fix_b`` are
          equivalent to ``f1``.

        - floc : hold location parameter fixed to specified value.

        - fscale : hold scale parameter fixed to specified value.

        - optimizer : The optimizer to use.  The optimizer must take ``func``,
          and starting position as the first two arguments,
          plus ``args`` (for extra arguments to pass to the
          function to be optimized) and ``disp=0`` to suppress
          output as keyword arguments.

    Returns
    -------
    shape, loc, scale : tuple of floats
        MLEs for any shape statistics, followed by those for location and
        scale.

    Notes
    -----
    This fit is computed by maximizing a log-likelihood function, with
    penalty applied for samples outside of range of the distribution. The
    returned answer is not guaranteed to be the globally optimal MLE, it
    may only be locally optimal, or the optimization may fail altogether.


    Examples
    --------

    Generate some data to fit: draw random variates from the `beta`
    distribution

    >>> from wafo.stats import beta
    >>> a, b = 1., 2.
    >>> x = beta.rvs(a, b, size=1000)

    Now we can fit all four parameters (``a``, ``b``, ``loc`` and ``scale``):

    >>> a1, b1, loc1, scale1 = beta.fit(x)

    We can also use some prior knowledge about the dataset: let's keep
    ``loc`` and ``scale`` fixed:

    >>> a1, b1, loc1, scale1 = beta.fit(x, floc=0, fscale=1)
    >>> loc1, scale1
    (0, 1)

    We can also keep shape parameters fixed by using ``f``-keywords. To
    keep the zero-th shape parameter ``a`` equal 1, use ``f0=1`` or,
    equivalently, ``fa=1``:

    >>> a1, b1, loc1, scale1 = beta.fit(x, fa=1, floc=0, fscale=1)
    >>> a1
    1

    """
    Narg = len(args)
    if Narg > self.numargs:
        raise TypeError("Too many input arguments.")

    kwds = kwargs.copy()
    start = [None]*2
    if (Narg < self.numargs) or not ('loc' in kwds and
                                     'scale' in kwds):
        # get distribution specific starting locations
        start = self._fitstart(data)
        args += start[Narg:-2]
    loc = kwds.pop('loc', start[-2])
    scale = kwds.pop('scale', start[-1])
    args += (loc, scale)
    x0, func, restore, args = self._reduce_func(args, kwds)

    optimizer = kwds.pop('optimizer', optimize.fmin)
    # convert string to function in scipy.optimize
    if not callable(optimizer) and isinstance(optimizer, string_types):
        if not optimizer.startswith('fmin_'):
            optimizer = "fmin_"+optimizer
        if optimizer == 'fmin_':
            optimizer = 'fmin'
        try:
            optimizer = getattr(optimize, optimizer)
        except AttributeError:
            raise ValueError("%s is not a valid optimizer" % optimizer)

    # by now kwds must be empty, since everybody took what they needed
    if kwds:
        raise TypeError("Unknown arguments: %s." % kwds)

    vals = optimizer(func, x0, args=(ravel(data),), disp=0)
    if restore is not None:
        vals = restore(args, vals)
    vals = tuple(vals)
    return vals


def fit2(self, data, *args, **kwds):
    ''' Return Maximum Likelihood or Maximum Product Spacing estimator object

    Parameters
    ----------
    data : array-like
        Data to use in calculating the ML or MPS estimators
    args : optional
        Starting values for any shape arguments (those not specified
        will be determined by dist._fitstart(data))
    kwds : loc, scale
        Starting values for the location and scale parameters
        Special keyword arguments are recognized as holding certain
        parameters fixed:
            f0..fn : hold respective shape paramters fixed
            floc : hold location parameter fixed to specified value
            fscale : hold scale parameter fixed to specified value
        method : of estimation. Options are
            'ml' : Maximum Likelihood method (default)
            'mps': Maximum Product Spacing method
        alpha : scalar, optional
            Confidence coefficent  (default=0.05)
        search : bool
            If true search for best estimator (default),
            otherwise return object with initial distribution parameters
        copydata : bool
            If true copydata (default)
        optimizer : The optimizer to use.  The optimizer must take func,
                     and starting position as the first two arguments,
                     plus args (for extra arguments to pass to the
                     function to be optimized) and disp=0 to suppress
                     output as keyword arguments.

    Return
    ------
    phat : FitDistribution object
        Fitted distribution object with following member variables:
        LLmax  : loglikelihood function evaluated using par
        LPSmax : log product spacing function evaluated using par
        pvalue : p-value for the fit
        par : distribution parameters (fixed and fitted)
        par_cov : covariance of distribution parameters
        par_fix : fixed distribution parameters
        par_lower : lower (1-alpha)% confidence bound for the parameters
        par_upper : upper (1-alpha)% confidence bound for the parameters

    Note
    ----
    `data` is sorted using this function, so if `copydata`==False the data
    in your namespace will be sorted as well.
    '''
    return FitDistribution(self, data, args, **kwds)


rv_generic.freeze = freeze
rv_discrete.freeze = freeze
rv_continuous.freeze = freeze
rv_continuous.link = link
rv_continuous._link = _link
rv_continuous.nlogps = nlogps
rv_continuous._reduce_func = _reduce_func
rv_continuous.fit = fit
rv_continuous.fit2 = fit2
