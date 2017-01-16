from __future__ import absolute_import
from scipy.stats._distn_infrastructure import *  # @UnusedWildImport
from scipy.stats._distn_infrastructure import (_skew,  # @UnusedImport
    _kurtosis,  _ncx2_log_pdf,  # @IgnorePep8 @UnusedImport
    _ncx2_pdf,  _ncx2_cdf)  # @UnusedImport @IgnorePep8
from .estimation import FitDistribution, rv_frozen  # @Reimport
from ._constants import _XMAX, _XMIN
from wafo.misc import lazyselect as _lazyselect  # @UnusedImport
from wafo.misc import lazywhere as _lazywhere  # @UnusedImport

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


def _resolve_ties(self, log_dprb, x, args, scale):
    dx = np.diff(x, axis=0)
    tie = dx == 0
    if any(tie):  # TODO : implement this method for treating ties in data:
        # Assume measuring error is delta. Then compute
        # yL = F(xi-delta,theta)
        # yU = F(xi+delta,theta)
        # and replace
        # logDj = log((yU-yL)/(r-1)) for j = i+1,i+2,...i+r-1
        # The following is OK when only minimization of T is wanted
        i_tie, = np.nonzero(tie)
        log_dprb[i_tie + 1] = log(self._pdf(x[i_tie], *args)) - log(scale)
    return log_dprb


def _log_dprb(self, x, args, scale, lowertail=True):

    if lowertail:
        prb = np.hstack((0.0, self.cdf(x, *args), 1.0))
        dprb = np.diff(prb)
    else:
        prb = np.hstack((1.0, self.sf(x, *args), 0.0))
        dprb = -np.diff(prb)
    log_dprb = log(dprb + _XMIN)
    log_dprb = _resolve_ties(self, log_dprb, x, args, scale)
    return log_dprb


def _nlogps_and_penalty(self, x, scale, args):
    cond0 = ~self._support_mask(x)
    n_bad = np.sum(cond0)
    if n_bad > 0:
        x = argsreduce(~cond0, x)[0]
    log_dprb = _log_dprb(self, x, args, scale)
    finite_log_dprb = np.isfinite(log_dprb)
    n_bad += np.sum(~finite_log_dprb, axis=0)
    if n_bad > 0:
        penalty = 100.0 * np.log(_XMAX) * n_bad
        return -np.sum(log_dprb[finite_log_dprb], axis=0) + penalty
    return -np.sum(log_dprb, axis=0)


def _penalized_nlogps(self, theta, x):
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
    loc, scale, args = _unpack_loc_scale(theta)
    if not self._argcheck(*args) or scale <= 0:
        return inf
    x = asarray((x - loc) / scale)
    return _nlogps_and_penalty(self, x, scale, args)


def _unpack_loc_scale(theta):
    try:
        loc = theta[-2]
        scale = theta[-1]
        args = tuple(theta[:-2])
    except IndexError:
        raise ValueError("Not enough input arguments.")
    return loc, scale, args


def _nnlf_and_penalty(self, x, args):
    cond0 = ~self._support_mask(x)
    n_bad = sum(cond0)
    if n_bad > 0:
        x = argsreduce(~cond0, x)[0]
    logpdf = self._logpdf(x, *args)
    finite_logpdf = np.isfinite(logpdf)
    n_bad += np.sum(~finite_logpdf, axis=0)
    if n_bad > 0:
        penalty = n_bad * log(_XMAX) * 100
        return -np.sum(logpdf[finite_logpdf], axis=0) + penalty
    return -np.sum(logpdf, axis=0)


def _penalized_nnlf(self, theta, x):
    ''' Return negative loglikelihood function,
    i.e., - sum (log pdf(x, theta), axis=0)
       where theta are the parameters (including loc and scale)
    '''
    loc, scale, args = _unpack_loc_scale(theta)
    if not self._argcheck(*args) or scale <= 0:
        return inf
    x = asarray((x-loc) / scale)
    n_log_scale = len(x) * log(scale)
    return _nnlf_and_penalty(self, x, args) + n_log_scale


def _convert_fshapes2num(self, kwds):
    # First of all, convert fshapes params to fnum: eg for stats.beta,
    # shapes='a, b'. To fix `a`, can specify either `f1` or `fa`.
    # Convert the latter into the former.
    if self.shapes:
        shapes = self.shapes.replace(',', ' ').split()
        for j, s in enumerate(shapes):
            val = kwds.pop('f' + s, None) or kwds.pop('fix_' + s, None)
            if val is not None:
                key = 'f{0:d}'.format(j)
                if key in kwds:
                    raise ValueError("Duplicate entry for {0:s}.".format(key))
                else:
                    kwds[key] = val
    return kwds


def _unpack_args_kwds(self, args, kwds):
    kwds = _convert_fshapes2num(self, kwds)
    args = list(args)
    fixedn = []
    names = ['f%d' % n for n in range(len(args) - 2)] + ['floc', 'fscale']
    x0 = []
    for n, key in enumerate(names):
        if key in kwds:
            fixedn.append(n)
            args[n] = kwds.pop(key)
        else:
            x0.append(args[n])
    return x0, args, fixedn


def _reduce_func(self, args, kwds):
    method = kwds.pop('method', 'ml').lower()
    if method.startswith('mps'):
        fitfun = self._penalized_nlogps
    else:
        fitfun = self._penalized_nnlf

    x0, args, fixedn = _unpack_args_kwds(self, args, kwds)

    nargs = len(args)

    if len(fixedn) == 0:
        func = fitfun
        restore = None
    else:
        if len(fixedn) == nargs:
            raise ValueError(
                "All parameters fixed. There is nothing to optimize.")

        def restore(args, theta):
            # Replace with theta for all numbers not in fixedn
            # This allows the non-fixed values to vary, but
            #  we still call self.nnlf with all parameters.
            i = 0
            for n in range(nargs):
                if n not in fixedn:
                    args[n] = theta[i]
                    i += 1
            return args

        def func(theta, x):
            newtheta = restore(args[:], theta)
            return fitfun(newtheta, x)

    return x0, func, restore, args, fixedn


def _get_optimizer(kwds):
    optimizer = kwds.pop('optimizer', optimize.fmin)
# convert string to function in scipy.optimize
    if not callable(optimizer) and isinstance(optimizer, string_types):
        if not optimizer.startswith('fmin_'):
                optimizer = '_'.join(("fmin", optimizer))
        try:
            optimizer = getattr(optimize, optimizer)
        except AttributeError:
            raise ValueError("{} is not a valid optimizer".format(optimizer))
    return optimizer


def _warn_if_no_success(warnflag):
    if warnflag == 1:
        warnings.warn("The maximum number of iterations was exceeded.")
    elif warnflag == 2:
        warnings.warn("Did not converge")


def _fitstart(self, data, args, kwds):
    narg = len(args)
    if narg > self.numargs:
        raise TypeError("Too many input arguments.")
    start = [None] * 2
    if (narg < self.numargs) or not ('loc' in kwds and 'scale' in kwds):
        # get distribution specific starting locations
        start = self._fitstart(data)
        args += start[narg:-2]
    loc = kwds.pop('loc', start[-2])
    scale = kwds.pop('scale', start[-1])
    args += loc, scale
    return args, kwds


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

        - method : of estimation. Options are
            'ml' : Maximum Likelihood method (default)
            'mps': Maximum Product Spacing method

        - alpha :  Confidence coefficent  (default=0.05)

        - search : bool
            If true search for best estimator (default),
            otherwise return object with initial distribution parameters

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
    vals, _ = self._fit(data, *args, **kwargs)
    return vals


def _fit(self, data, *args, **kwargs):
    args, kwds = _fitstart(self, data, args, kwargs.copy())
    x0, func, restore, args, fixedn = self._reduce_func(args, kwds)
    if kwds.pop('search', True):
        optimizer = _get_optimizer(kwds)

        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError("Unknown arguments: {}.".format(kwds))

        output = optimizer(func, x0, args=(ravel(data),), full_output=True,
                           disp=0)
        if output[-1] != 0:
            output = optimizer(func, output[0], args=(ravel(data),),
                               full_output=True)

        _warn_if_no_success(output[-1])
        vals = tuple(output[0])
    else:
        vals = tuple(x0)

    if restore is not None:
        vals = restore(args, vals)
    vals = tuple(vals)
    return vals, fixedn


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


def _support_mask(self, x):
    return (self.a <= x) & (x <= self.b)


def _open_support_mask(self, x):
    return (self.a < x) & (x < self.b)


rv_generic.freeze = freeze
rv_discrete.freeze = freeze
rv_continuous.freeze = freeze

rv_continuous._penalized_nlogps = _penalized_nlogps
rv_continuous._penalized_nnlf = _penalized_nnlf
rv_continuous._reduce_func = _reduce_func
rv_continuous.fit = fit
rv_continuous._fit = _fit
rv_continuous.fit2 = fit2
rv_continuous._support_mask = _support_mask
rv_continuous._open_support_mask = _open_support_mask
