'''
Contains FitDistribution and Profile class, which are

important classes for fitting to various Continous and Discrete Probability
Distributions

Author:  Per A. Brodtkorb 2008
'''

from __future__ import division, absolute_import
import warnings

from wafo.plotbackend import plotbackend
from wafo.misc import ecross, findcross, argsreduce
from wafo.stats._constants import _EPS, _XMAX
from wafo.stats._distn_infrastructure import rv_frozen, rv_continuous
from scipy._lib.six import string_types
import numdifftools as nd  # @UnresolvedImport
from scipy import special
from scipy.linalg import pinv2
from scipy import optimize

import numpy as np
from numpy import (alltrue, arange, zeros, log, sqrt, exp,
                   atleast_1d, any, asarray, nan, pi, isfinite)
from numpy import flatnonzero as nonzero


__all__ = ['Profile', 'FitDistribution']

floatinfo = np.finfo(float)

arr = asarray
all = alltrue  # @ReservedAssignment


def chi2isf(p, df):
    return special.chdtri(df, p)


def chi2sf(x, df):
    return special.chdtrc(df, x)


def norm_ppf(q):
    return special.ndtri(q)


# internal class to profile parameters of a given distribution
class Profile(object):

    ''' Profile Log- likelihood or Product Spacing-function.
            which can be used for constructing confidence interval for
            either phat[i], probability or quantile.

    Parameters
    ----------
    fit_dist : FitDistribution object
        with ML or MPS estimated distribution parameters.
    **kwds : named arguments with keys
        i : scalar integer
            defining which distribution parameter to keep fixed in the
            profiling process (default first non-fixed parameter)
        pmin, pmax : real scalars
            Interval for either the parameter, phat(i), prb, or x, used in the
            optimization of the profile function (default is based on the
            100*(1-alpha)% confidence interval computed with the delta method.)
        N : scalar integer
            Max number of points used in Lp (default 100)
        x : real scalar
            Quantile (return value) (default None)
        logSF : real scalar
            log survival probability,i.e., SF = Prob(X>x;phat) (default None)
        link : function connecting the x-quantile and the survival probability
            (SF) with the fixed distribution parameter, i.e.:
            self.par[i] = link(x, logSF, self.par, i), where
                logSF = log(Prob(X>x;phat)).
            This means that if:
                1) x is not None then x is profiled
                2) logSF is not None then logSF is profiled
                3) x and logSF are None then self.par[i] is profiled (default)
        alpha : real scalar
            confidence coefficent (default 0.05)
    Returns
    -------
    Lp : Profile log-likelihood function with parameters phat given
        the data, phat(i), probability (prb) and quantile (x) (if given), i.e.,
            Lp = max(log(f(phat|data,phat(i)))),
        or
            Lp = max(log(f(phat|data,phat(i),x,prb)))

    Member methods
    -------------
    plot() : Plot profile function with 100(1-alpha)% confidence interval
    get_bounds() : Return 100(1-alpha)% confidence interval

    Member variables
    ----------------
    fit_dist : FitDistribution data object.
    data : profile function values
    args : profile function arguments
    alpha : confidence coefficient
    Lmax : Maximum value of profile function
    alpha_cross_level :

    PROFILE is a utility function for making inferences either on a particular
    component of the vector phat or the quantile, x, or the probability, SF.
    This is usually more accurate than using the delta method assuming
    asymptotic normality of the ML estimator or the MPS estimator.

    Examples
    --------
    # MLE
    >>> import wafo.stats as ws
    >>> R = ws.weibull_min.rvs(1,size=100);
    >>> phat = FitDistribution(ws.weibull_min, R, 1, scale=1, floc=0.0)

    # Better CI for phat.par[i=0]
    >>> Lp = Profile(phat, i=0)
    >>> Lp.plot()
    >>> phat_ci = Lp.get_bounds(alpha=0.1)

    >>> SF = 1./990
    >>> x = phat.isf(SF)

    # CI for x
    >>> Lx = phat.profile(i=0, x=x, link=phat.dist.link)
    >>> Lx.plot()
    >>> x_ci = Lx.get_bounds(alpha=0.2)

    # CI for logSF=log(SF)
    >>> Lsf = phat.profile(i=0, logSF=log(SF), link=phat.dist.link)
    >>> Lsf.plot()
    >>> sf_ci = Lsf.get_bounds(alpha=0.2)
    '''

    def __init__(self, fit_dist, **kwds):

        try:
            i0 = (1 - np.isfinite(fit_dist.par_fix)).argmax()
        except:
            i0 = 0
        self.fit_dist = fit_dist
        self.data = None
        self.args = None
        self.title = ''
        self.xlabel = ''
        self.ylabel = 'Profile log'
        (self.i_fixed, self.N, self.alpha, self.pmin, self.pmax, self.x,
         self.logSF, self.link) = [kwds.get(name, val)
                                   for name, val in zip(
            ['i', 'N', 'alpha', 'pmin', 'pmax', 'x', 'logSF', 'link'],
            [i0, 100, 0.05, None, None, None, None, None])]

        self.title = '%g%s CI' % (100 * (1.0 - self.alpha), '%')
        if fit_dist.method.startswith('ml'):
            self.ylabel = self.ylabel + 'likelihood'
            Lmax = fit_dist.LLmax
        elif fit_dist.method.startswith('mps'):
            self.ylabel = self.ylabel + ' product spacing'
            Lmax = fit_dist.LPSmax
        else:
            raise ValueError(
                "PROFILE is only valid for ML- or MPS- estimators")

        if fit_dist.par_fix is None:
            isnotfixed = np.ones(fit_dist.par.shape, dtype=bool)
        else:
            isnotfixed = 1 - np.isfinite(fit_dist.par_fix)

        self.i_notfixed = nonzero(isnotfixed)

        self.i_fixed = atleast_1d(self.i_fixed)

        if 1 - isnotfixed[self.i_fixed]:
            raise IndexError(
                "Index i must be equal to an index to one of the free " +
                "parameters.")

        isfree = isnotfixed
        isfree[self.i_fixed] = False
        self.i_free = nonzero(isfree)

        self.Lmax = Lmax
        self.alpha_Lrange = 0.5 * chi2isf(self.alpha, 1)
        self.alpha_cross_level = Lmax - self.alpha_Lrange
        # lowLevel = self.alpha_cross_level - self.alpha_Lrange / 7.0

        phatv = fit_dist.par.copy()
        self._par = phatv.copy()

        # Set up variable to profile and _local_link function
        self.profile_x = self.x is not None
        self.profile_logSF = not (self.logSF is None or self.profile_x)
        self.profile_par = not (self.profile_x or self.profile_logSF)

        if self.link is None:
            self.link = self.fit_dist.dist.link
        if self.profile_par:
            self._local_link = self._par_link
            self.xlabel = 'phat(%d)' % self.i_fixed
            p_opt = self._par[self.i_fixed]
        elif self.profile_x:
            self.logSF = fit_dist.logsf(self.x)
            self._local_link = self._x_link
            self.xlabel = 'x'
            p_opt = self.x
        elif self.profile_logSF:
            p_opt = self.logSF
            self.x = fit_dist.isf(exp(p_opt))
            self._local_link = self._logSF_link
            self.xlabel = 'log(SF)'
        else:
            raise ValueError(
                "You must supply a non-empty quantile (x) or probability " +
                "(logSF) in order to profile it!")

        self.xlabel = self.xlabel + ' (' + fit_dist.dist.name + ')'

        phatfree = phatv[self.i_free].copy()
        self._set_profile(phatfree, p_opt)

    def _par_link(self, fix_par, par):
        return fix_par

    def _x_link(self, fix_par, par):
        return self.link(fix_par, self.logSF, par, self.i_fixed)

    def _logSF_link(self, fix_par, par):
        return self.link(self.x, fix_par, par, self.i_fixed)

    def _correct_Lmax(self, Lmax):
        if Lmax > self.Lmax:  # foundNewphat = True
            warnings.warn(
                'The fitted parameters does not provide the optimum fit. ' +
                'Something wrong with fit')
            dL = self.Lmax - Lmax
            self.alpha_cross_level -= dL
            self.Lmax = Lmax

    def _profile_optimum(self, phatfree0, p_opt):
        phatfree = optimize.fmin(
            self._profile_fun, phatfree0, args=(p_opt,), disp=0)
        Lmax = -self._profile_fun(phatfree, p_opt)
        self._correct_Lmax(Lmax)
        return Lmax, phatfree

    def _set_profile(self, phatfree0, p_opt):
        pvec = self._get_pvec(phatfree0, p_opt)

        self.data = np.ones_like(pvec) * nan
        k1 = (pvec >= p_opt).argmax()

        for size, step in ((-1, -1), (pvec.size, 1)):
            phatfree = phatfree0.copy()
            for ix in range(k1, size, step):
                Lmax, phatfree = self._profile_optimum(phatfree, pvec[ix])
                self.data[ix] = Lmax
                if self.data[ix] < self.alpha_cross_level:
                    break
        np.putmask(pvec, np.isnan(self.data), nan)
        self.args = pvec

        self._prettify_profile()

    def _prettify_profile(self):
        pvec = self.args
        ix = nonzero(np.isfinite(pvec))
        self.data = self.data[ix]
        self.args = pvec[ix]
        cond = self.data == -np.inf
        if any(cond):
            ind, = cond.nonzero()
            self.data.put(ind, floatinfo.min / 2.0)
            ind1 = np.where(ind == 0, ind, ind - 1)
            cl = self.alpha_cross_level - self.alpha_Lrange / 2.0
            t0 = ecross(self.args, self.data, ind1, cl)
            self.data.put(ind, cl)
            self.args.put(ind, t0)

    def _get_variance(self):
        if self.profile_par:
            pvar = self.fit_dist.par_cov[self.i_fixed, :][:, self.i_fixed]
        else:
            i_notfixed = self.i_notfixed
            phatv = self._par

            if self.profile_x:
                gradfun = nd.Gradient(self._myinvfun)
            else:
                gradfun = nd.Gradient(self._myprbfun)
            drl = gradfun(phatv[self.i_notfixed])

            pcov = self.fit_dist.par_cov[i_notfixed, :][:, i_notfixed]
            pvar = np.sum(np.dot(drl, pcov) * drl)
        return pvar

    def _get_pvec(self, phatfree0, p_opt):
        ''' return proper interval for the variable to profile
        '''

        linspace = np.linspace
        if self.pmin is None or self.pmax is None:

            pvar = self._get_variance()

            if pvar <= 1e-5 or np.isnan(pvar):
                pvar = max(abs(p_opt) * 0.5, 0.5)

            p_crit = (-norm_ppf(self.alpha / 2.0) *
                      sqrt(np.ravel(pvar)) * 1.5)
            if self.pmin is None:
                self.pmin = self._search_pmin(phatfree0,
                                              p_opt - 5.0 * p_crit, p_opt)
            p_crit_low = (p_opt - self.pmin) / 5

            if self.pmax is None:
                self.pmax = self._search_pmax(phatfree0,
                                              p_opt + 5.0 * p_crit, p_opt)
            p_crit_up = (self.pmax - p_opt) / 5

            N4 = np.floor(self.N / 4.0)

            pvec1 = linspace(self.pmin, p_opt - p_crit_low, N4 + 1)
            pvec2 = linspace(
                p_opt - p_crit_low, p_opt + p_crit_up, self.N - 2 * N4)
            pvec3 = linspace(p_opt + p_crit_up, self.pmax, N4 + 1)
            pvec = np.unique(np.hstack((pvec1, p_opt, pvec2, pvec3)))

        else:
            pvec = linspace(self.pmin, self.pmax, self.N)
        return pvec

    def _search_pmin(self, phatfree0, p_min0, p_opt):
        phatfree = phatfree0.copy()

        dp = p_opt - p_min0
        if dp < 1e-2:
            dp = 0.1
        p_min_opt = p_min0
        Lmax, phatfree = self._profile_optimum(phatfree, p_opt)
        for _j in range(50):
            p_min = p_opt - dp
            Lmax, phatfree = self._profile_optimum(phatfree, p_min)
            if np.isnan(Lmax):
                dp *= 0.33
            elif Lmax < self.alpha_cross_level - self.alpha_Lrange * 2:
                p_min_opt = p_min
                dp *= 0.33
            elif Lmax < self.alpha_cross_level:
                p_min_opt = p_min
                break
            else:
                dp *= 1.67
        return p_min_opt

    def _search_pmax(self, phatfree0, p_max0, p_opt):
        phatfree = phatfree0.copy()

        dp = p_max0 - p_opt
        if dp < 1e-2:
            dp = 0.1
        p_max_opt = p_max0
        Lmax, phatfree = self._profile_optimum(phatfree, p_opt)
        for _j in range(50):
            p_max = p_opt + dp
            Lmax, phatfree = self._profile_optimum(phatfree, p_max)
            if np.isnan(Lmax):
                dp *= 0.33
            elif Lmax < self.alpha_cross_level - self.alpha_Lrange * 2:
                p_max_opt = p_max
                dp *= 0.33
            elif Lmax < self.alpha_cross_level:
                p_max_opt = p_max
                break
            else:
                dp *= 1.67
        return p_max_opt

    def _myinvfun(self, phatnotfixed):
        mphat = self._par.copy()
        mphat[self.i_notfixed] = phatnotfixed
        prb = exp(self.logSF)
        return self.fit_dist.dist.isf(prb, *mphat)

    def _myprbfun(self, phatnotfixed):
        mphat = self._par.copy()
        mphat[self.i_notfixed] = phatnotfixed
        logSF = self.fit_dist.dist.logsf(self.x, *mphat)
        return np.where(np.isfinite(logSF), logSF, np.nan)

    def _profile_fun(self, free_par, fix_par):
        ''' Return negative of loglike or logps function

           free_par - vector of free parameters
           fix_par  - fixed parameter, i.e., either quantile (return level),
                      probability (return period) or distribution parameter
        '''
        par = self._par.copy()
        par[self.i_free] = free_par
        # _local_link: connects fixed quantile or probability with fixed
        # distribution parameter
        par[self.i_fixed] = self._local_link(fix_par, par)
        return self.fit_dist.fitfun(par)

    def get_bounds(self, alpha=0.05):
        '''Return confidence interval for profiled parameter
        '''
        if alpha < self.alpha:
            warnings.warn(
                'Might not be able to return CI with alpha less than %g' %
                self.alpha)
        cross_level = self.Lmax - 0.5 * chi2isf(alpha, 1)
        ind = findcross(self.data, cross_level)
        N = len(ind)
        if N == 0:
            warnings.warn('''Number of crossings is zero, i.e.,
            upper and lower bound is not found!''')
            CI = (self.pmin, self.pmax)

        elif N == 1:
            x0 = ecross(self.args, self.data, ind, cross_level)
            isUpcrossing = self.data[ind] > self.data[ind + 1]
            if isUpcrossing:
                CI = (x0, self.pmax)
                warnings.warn('Upper bound is larger')
            else:
                CI = (self.pmin, x0)
                warnings.warn('Lower bound is smaller')

        elif N == 2:
            CI = ecross(self.args, self.data, ind, cross_level)
        else:
            warnings.warn('Number of crossings too large! Something is wrong!')
            CI = ecross(self.args, self.data, ind[[0, -1]], cross_level)
        return CI

    def plot(self, axis=None):
        ''' Plot profile function with 100(1-alpha)% CI
        '''
        if axis is None:
            axis = plotbackend.gca()

        p_ci = self.get_bounds(self.alpha)
        axis.plot(
            self.args, self.data,
            self.args[[0, -1]], [self.Lmax, ] * 2, 'r--',
            self.args[[0, -1]], [self.alpha_cross_level, ] * 2, 'r--')
        axis.vlines(p_ci, ymin=axis.get_ylim()[0],
                    ymax=self.Lmax,  # self.alpha_cross_level,
                    color='r', linestyles='--')
        axis.set_title(self.title)
        axis.set_ylabel(self.ylabel)
        axis.set_xlabel(self.xlabel)


class FitDistribution(rv_frozen):

    '''
    Return estimators to shape, location, and scale from data

    Starting points for the fit are given by input arguments.  For any
    arguments not given starting points, dist._fitstart(data) is called
    to get the starting estimates.

    You can hold some parameters fixed to specific values by passing in
    keyword arguments f0..fn for shape paramters and floc, fscale for
    location and scale parameters.

    Parameters
    ----------
    dist : scipy distribution object
        distribution to fit to data
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

    Examples
    --------
    Estimate distribution parameters for weibull_min distribution.
    >>> import wafo.stats as ws
    >>> R = ws.weibull_min.rvs(1,size=100);
    >>> phat = FitDistribution(ws.weibull_min, R, 1, scale=1, floc=0.0)

    #Plot various diagnostic plots to asses quality of fit.
    >>> phat.plotfitsummary()

    #phat.par holds the estimated parameters
    #phat.par_upper upper CI for parameters
    #phat.par_lower lower CI for parameters

    #Better CI for phat.par[0]
    >>> Lp = phat.profile(i=0)
    >>> Lp.plot()
    >>> p_ci = Lp.get_bounds(alpha=0.1)

    >>> SF = 1./990
    >>> x = phat.isf(SF)

    # CI for x
    >>> Lx = phat.profile(i=0,x=x,link=phat.dist.link)
    >>> Lx.plot()
    >>> x_ci = Lx.get_bounds(alpha=0.2)

     # CI for logSF=log(SF)
    >>> Lsf = phat.profile(i=0, logSF=log(SF), link=phat.dist.link)
    >>> Lsf.plot()
    >>> sf_ci = Lsf.get_bounds(alpha=0.2)
    '''

    def __init__(self, dist, data, args=(), method='ML', alpha=0.05,
                 par_fix=None, search=True, copydata=True, **kwds):
        extradoc = '''
    plotfitsummary()
         Plot various diagnostic plots to asses quality of fit.
    plotecdf()
        Plot Empirical and fitted Cumulative Distribution Function
    plotesf()
        Plot Empirical and fitted Survival Function
    plotepdf()
        Plot Empirical and fitted Probability Distribution Function
    plotresq()
        Displays a residual quantile plot.
    plotresprb()
        Displays a residual probability plot.

    profile()
        Return Profile Log- likelihood or Product Spacing-function.

    Parameters
    ----------
    x : array-like
        quantiles
    q : array-like
        lower or upper tail probability
    size : int or tuple of ints, optional
        shape of random variates (default computed from input arguments )
    moments : str, optional
        composed of letters ['mvsk'] specifying which moments to compute where
        'm' = mean, 'v' = variance, 's' = (Fisher's) skew and
        'k' = (Fisher's) kurtosis. (default='mv')
       '''
#    Member variables
#    ----------------
#    data - data used in fitting
#    alpha - confidence coefficient
#    method - method used
#    LLmax  - loglikelihood function evaluated using par
#    LPSmax - log product spacing function evaluated using par
#    pvalue - p-value for the fit
#    search - True if search for distribution parameters (default)
#    copydata - True if copy input data (default)
#
#    par     - parameters (fixed and fitted)
#    par_cov - covariance of parameters
#    par_fix - fixed parameters
#    par_lower - lower (1-alpha)% confidence bound for the parameters
#    par_upper - upper (1-alpha)% confidence bound for the parameters
#
#        '''
        self.__doc__ = str(rv_frozen.__doc__) + extradoc
        self.dist = dist
        numargs = dist.numargs

        self.method = method
        self.alpha = alpha
        self.par_fix = par_fix
        self.search = search
        self.copydata = copydata

        if self.method.lower()[:].startswith('mps'):
            self._fitfun = self._nlogps
        else:
            self._fitfun = self._nnlf

        self.data = np.ravel(data)
        if self.copydata:
            self.data = self.data.copy()
        self.data.sort()
        par, fixedn = self._fit(*args, **kwds.copy())
        super(FitDistribution, self).__init__(dist, *par)
        self.par = arr(par)
        somefixed = len(fixedn) > 0
        if somefixed:
            self.par_fix = [nan, ] * len(self.par)
            for i in fixedn:
                self.par_fix[i] = self.par[i]

            self.i_notfixed = nonzero(1 - isfinite(self.par_fix))
            self.i_fixed = nonzero(isfinite(self.par_fix))

        numpar = numargs + 2
        self.par_cov = zeros((numpar, numpar))
        self._compute_cov()

        # Set confidence interval for parameters
        pvar = np.diag(self.par_cov)
        zcrit = -norm_ppf(self.alpha / 2.0)
        self.par_lower = self.par - zcrit * sqrt(pvar)
        self.par_upper = self.par + zcrit * sqrt(pvar)

        self.LLmax = -self._nnlf(self.par, self.data)
        self.LPSmax = -self._nlogps(self.par, self.data)
        self.pvalue = self._pvalue(self.par, self.data, unknown_numpar=numpar)

    def __repr__(self):
        params = ['alpha', 'method', 'LLmax', 'LPSmax', 'pvalue',
                  'par', 'par_lower', 'par_upper', 'par_fix', 'par_cov']
        t = ['%s:\n' % self.__class__.__name__]
        for par in params:
            t.append('%s = %s\n' % (par, str(getattr(self, par))))
        return ''.join(t)

    def _reduce_func(self, args, kwds):
        # First of all, convert fshapes params to fnum: eg for stats.beta,
        # shapes='a, b'. To fix `a`, can specify either `f1` or `fa`.
        # Convert the latter into the former.
        shapes = self.dist.shapes
        if shapes:
            shapes = shapes.replace(',', ' ').split()
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

        fitfun = self._fitfun

        if len(fixedn) == 0:
            func = fitfun
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError("All parameters fixed. " +
                                 "There is nothing to optimize.")

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

        return x0, func, restore, args, fixedn

    @staticmethod
    def _hessian(nnlf, theta, data, eps=None):
        ''' approximate hessian of nnlf where theta are the parameters
        (including loc and scale)
        '''
        if eps is None:
            eps = (_EPS) ** 0.4
        num_par = len(theta)
        # pab 07.01.2001: Always choose the stepsize h so that
        # it is an exactly representable number.
        # This is important when calculating numerical derivatives and is
        #  accomplished by the following.
        delta = (eps + 2.0) - 2.0
        delta2 = delta ** 2.0
        # Approximate 1/(nE( (d L(x|theta)/dtheta)^2)) with
        #              1/(d^2 L(theta|x)/dtheta^2)
        # using central differences

        LL = nnlf(theta, data)
        H = zeros((num_par, num_par))   # Hessian matrix
        theta = tuple(theta)
        for ix in range(num_par):
            sparam = list(theta)
            sparam[ix] = theta[ix] + delta
            fp = nnlf(sparam, data)

            sparam[ix] = theta[ix] - delta
            fm = nnlf(sparam, data)

            H[ix, ix] = (fp - 2 * LL + fm) / delta2
            for iy in range(ix + 1, num_par):
                sparam[ix] = theta[ix] + delta
                sparam[iy] = theta[iy] + delta
                fpp = nnlf(sparam, data)

                sparam[iy] = theta[iy] - delta
                fpm = nnlf(sparam, data)

                sparam[ix] = theta[ix] - delta
                fmm = nnlf(sparam, data)

                sparam[iy] = theta[iy] + delta
                fmp = nnlf(sparam, data)

                H[ix, iy] = ((fpp + fmm) - (fmp + fpm)) / (4. * delta2)
                H[iy, ix] = H[ix, iy]
                sparam[iy] = theta[iy]
        return -H

    def _nnlf(self, theta, x):
        return self.dist._penalized_nnlf(theta, x)

    def _nlogps(self, theta, x):
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
        n = 2 if isinstance(self.dist, rv_continuous) else 1
        try:
            loc = theta[-n]
            scale = theta[-1]
            args = tuple(theta[:-n])
        except IndexError:
            raise ValueError("Not enough input arguments.")
        if not isinstance(self.dist, rv_continuous):
            scale = 1
        if not self.dist._argcheck(*args) or scale <= 0:
            return np.inf
        dist = self.dist
        x = asarray((x - loc) / scale)
        cond0 = (x <= dist.a) | (dist.b <= x)
        Nbad = np.sum(cond0)
        if Nbad > 0:
            x = argsreduce(~cond0, x)[0]

        lowertail = True
        if lowertail:
            prb = np.hstack((0.0, dist.cdf(x, *args), 1.0))
            dprb = np.diff(prb)
        else:
            prb = np.hstack((1.0, dist.sf(x, *args), 0.0))
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
            logD[i_tie + 1] = log(dist._pdf(tiedata, *args)) - log(scale)

        finiteD = np.isfinite(logD)
        nonfiniteD = 1 - finiteD
        Nbad += np.sum(nonfiniteD, axis=0)
        if Nbad > 0:
            T = -np.sum(logD[finiteD], axis=0) + 100.0 * log(_XMAX) * Nbad
        else:
            T = -np.sum(logD, axis=0)
        return T

    def _fit(self, *args, **kwds):

        dist = self.dist
        data = self.data

        Narg = len(args)
        if Narg > dist.numargs:
                raise ValueError("Too many input arguments.")
        start = [None] * 2
        if (Narg < dist.numargs) or not ('loc' in kwds and 'scale' in kwds):
            # get distribution specific starting locations
            start = dist._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args, fixedn = self._reduce_func(args, kwds)
        if self.search:
            optimizer = kwds.pop('optimizer', optimize.fmin)
            # convert string to function in scipy.optimize
            if not callable(optimizer) and isinstance(optimizer, string_types):
                if not optimizer.startswith('fmin_'):
                    optimizer = "fmin_" + optimizer
                if optimizer == 'fmin_':
                    optimizer = 'fmin'
                try:
                    optimizer = getattr(optimize, optimizer)
                except AttributeError:
                    raise ValueError("%s is not a valid optimizer" % optimizer)
            # by now kwds must be empty, since everybody took what they needed
            if kwds:
                raise TypeError("Unknown arguments: %s." % kwds)
            vals = optimizer(func, x0, args=(np.ravel(data),), disp=0)
            vals = tuple(vals)
        else:
            vals = tuple(x0)
        if restore is not None:
            vals = restore(args, vals)
        return vals, fixedn

    def _compute_cov(self):
        '''Compute covariance
        '''
        somefixed = (self.par_fix is not None) and any(isfinite(self.par_fix))
        H = np.asmatrix(self._hessian(self._fitfun, self.par, self.data))
        self.H = H
        try:
            if somefixed:
                allfixed = all(isfinite(self.par_fix))
                if allfixed:
                    self.par_cov[:, :] = 0
                else:
                    pcov = -pinv2(H[self.i_notfixed, :][..., self.i_notfixed])
                    for row, ix in enumerate(list(self.i_notfixed)):
                        self.par_cov[ix, self.i_notfixed] = pcov[row, :]
            else:
                self.par_cov = -pinv2(H)
        except:
            self.par_cov[:, :] = nan

    def fitfun(self, phat):
        return self._fitfun(phat, self.data)

    def profile(self, **kwds):
        ''' Profile Log- likelihood or Log Product Spacing- function,
            which can be used for constructing confidence interval for
            either phat(i), probability or quantile.

        Parameters
        ----------
        **kwds : named arguments with keys
        i : scalar integer
            defining which distribution parameter to profile, i.e. which
            parameter to keep fixed (default first non-fixed parameter)
        pmin, pmax : real scalars
            Interval for either the parameter, phat(i), prb, or x, used in the
            optimization of the profile function (default is based on the
            100*(1-alpha)% confidence interval computed with the delta method.)
        N : scalar integer
            Max number of points used in Lp (default 100)
        x : real scalar
            Quantile (return value) (default None)
        logSF : real scalar
            log survival probability,i.e., SF = Prob(X>x;phat) (default None)
        link : function connecting the x-quantile and the survival probability
            (SF) with the fixed distribution parameter, i.e.:
            self.par[i] = link(x,logSF,self.par,i), where
            logSF = log(Prob(X>x;phat)).
            This means that if:
                1) x is not None then x is profiled
                2) logSF is not None then logSF is profiled
                3) x and logSF are None then self.par[i] is profiled (default)
        alpha : real scalar
            confidence coefficent (default 0.05)
        Returns
        -------
        Lp : Profile log-likelihood function with parameters phat given
            the data, phat(i), probability (prb) and quantile (x), i.e.,
                Lp = max(log(f(phat|data,phat(i)))),
            or
                Lp = max(log(f(phat|data,phat(i),x,prb)))

        Member methods
        -------------
        plot() : Plot profile function with 100(1-alpha)% confidence interval
        get_bounds() : Return 100(1-alpha)% confidence interval

        Member variables
        ----------------
        fit_dist : FitDistribution data object.
        data : profile function values
        args : profile function arguments
        alpha : confidence coefficient
        Lmax : Maximum value of profile function
        alpha_cross_level :

        PROFILE is a utility function for making inferences either on a
        particular component of the vector phat or the quantile, x, or the
        probability, SF. This is usually more accurate than using the delta
        method assuming asymptotic normality of the ML estimator or the MPS
        estimator.

        Examples
        --------
        # MLE
        >>> import wafo.stats as ws
        >>> R = ws.weibull_min.rvs(1,size=100);
        >>> phat = FitDistribution(ws.weibull_min, R, 1, scale=1, floc=0.0)

        # Better CI for phat.par[i=0]
        >>> Lp = Profile(phat, i=0)
        >>> Lp.plot()
        >>> phat_ci = Lp.get_bounds(alpha=0.1)

        >>> SF = 1./990
        >>> x = phat.isf(SF)

        # CI for x
        >>> Lx = phat.profile(i=0, x=x, link=phat.dist.link)
        >>> Lx.plot()
        >>> x_ci = Lx.get_bounds(alpha=0.2)

        # CI for logSF=log(SF)
        >>> Lsf = phat.profile(i=0, logSF=log(SF), link=phat.dist.link)
        >>> Lsf.plot()
        >>> sf_ci = Lsf.get_bounds(alpha=0.2)

        See also
        --------
        Profile
        '''
        return Profile(self, **kwds)

    def plotfitsummary(self):
        ''' Plot various diagnostic plots to asses the quality of the fit.

        PLOTFITSUMMARY displays probability plot, density plot, residual
        quantile plot and residual probability plot.
        The purpose of these plots is to graphically assess whether the data
        could come from the fitted distribution. If so the empirical- CDF and
        PDF should follow the model and the residual plots will be linear.
        Other distribution types will introduce curvature in the residual
        plots.
        '''
        plotbackend.subplot(2, 2, 1)
        # self.plotecdf()
        self.plotesf()
        plotbackend.subplot(2, 2, 2)
        self.plotepdf()
        plotbackend.subplot(2, 2, 3)
        self.plotresq()
        plotbackend.subplot(2, 2, 4)
        self.plotresprb()

        fixstr = ''
        if self.par_fix is not None:
            numfix = len(self.i_fixed)
            if numfix > 0:
                format0 = ', '.join(['%d'] * numfix)
                format1 = ', '.join(['%g'] * numfix)
                phatistr = format0 % tuple(self.i_fixed)
                phatvstr = format1 % tuple(self.par[self.i_fixed])
                fixstr = 'Fixed: phat[%s] = %s ' % (phatistr, phatvstr)

        infostr = 'Fit method: %s, Fit p-value: %2.2f %s' % (
            self.method, self.pvalue, fixstr)
        try:
            plotbackend.figtext(0.05, 0.01, infostr)
        except:
            pass

    def plotesf(self, symb1='r-', symb2='b.'):
        '''  Plot Empirical and fitted Survival Function

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution.
        If so the empirical CDF should resemble the model CDF.
        Other distribution types will introduce deviations in the plot.
        '''
        n = len(self.data)
        SF = (arange(n, 0, -1)) / n
        plotbackend.semilogy(
            self.data, SF, symb2, self.data, self.sf(self.data), symb1)
        # plotbackend.plot(self.data,SF,'b.',self.data,self.sf(self.data),'r-')
        plotbackend.xlabel('x')
        plotbackend.ylabel('F(x) (%s)' % self.dist.name)
        plotbackend.title('Empirical SF plot')

    def plotecdf(self, symb1='r-', symb2='b.'):
        '''  Plot Empirical and fitted Cumulative Distribution Function

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution.
        If so the empirical CDF should resemble the model CDF.
        Other distribution types will introduce deviations in the plot.
        '''
        n = len(self.data)
        F = (arange(1, n + 1)) / n
        plotbackend.plot(self.data, F, symb2,
                         self.data, self.cdf(self.data), symb1)
        plotbackend.xlabel('x')
        plotbackend.ylabel('F(x) (%s)' % self.dist.name)
        plotbackend.title('Empirical CDF plot')

    def _get_grid(self, odd=False):
        x = np.atleast_1d(self.data)
        n = np.ceil(4 * np.sqrt(np.sqrt(len(x))))
        mn = x.min()
        mx = x.max()
        d = (mx - mn) / n * 2
        e = np.floor(np.log(d) / np.log(10))
        m = np.floor(d / 10 ** e)
        if m > 5:
            m = 5
        elif m > 2:
            m = 2
        d = m * 10 ** e
        mn = (np.floor(mn / d) - 1) * d - odd * d / 2
        mx = (np.ceil(mx / d) + 1) * d + odd * d / 2
        limits = np.arange(mn, mx, d)
        return limits

    def _staircase(self, x, y):
        xx = x.reshape(-1, 1).repeat(3, axis=1).ravel()[1:-1]
        yy = y.reshape(-1, 1).repeat(3, axis=1)
        # yy[0,0] = 0.0 # pdf
        yy[:, 0] = 0.0  # histogram
        yy.shape = (-1,)
        yy = np.hstack((yy, 0.0))
        return xx, yy

    def _get_empirical_pdf(self):
        limits = self._get_grid()
        pdf, x = np.histogram(self.data, bins=limits, normed=True)
        return self._staircase(x, pdf)

    def plotepdf(self, symb1='r-', symb2='b-'):
        '''Plot Empirical and fitted Probability Density Function

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution.
        If so the histogram should resemble the model density.
        Other distribution types will introduce deviations in the plot.
        '''
        x, pdf = self._get_empirical_pdf()
        ymax = pdf.max()
        # plotbackend.hist(self.data,normed=True,fill=False)
        plotbackend.plot(self.data, self.pdf(self.data), symb1,
                         x, pdf, symb2)
        ax = list(plotbackend.axis())
        ax[3] = min(ymax * 1.3, ax[3])
        plotbackend.axis(ax)
        plotbackend.xlabel('x')
        plotbackend.ylabel('f(x) (%s)' % self.dist.name)
        plotbackend.title('Density plot')

    def plotresq(self, symb1='r-', symb2='b.'):
        '''PLOTRESQ displays a residual quantile plot.

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution. If so the
        plot will be linear. Other distribution types will introduce
        curvature in the plot.
        '''
        n = len(self.data)
        eprob = (arange(1, n + 1) - 0.5) / n
        y = self.ppf(eprob)
        y1 = self.data[[0, -1]]
        plotbackend.plot(self.data, y, symb2, y1, y1, symb1)
        plotbackend.xlabel('Empirical')
        plotbackend.ylabel('Model (%s)' % self.dist.name)
        plotbackend.title('Residual Quantile Plot')
        plotbackend.axis('tight')
        plotbackend.axis('equal')

    def plotresprb(self, symb1='r-', symb2='b.'):
        ''' PLOTRESPRB displays a residual probability plot.

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution. If so the
        plot will be linear. Other distribution types will introduce curvature
        in the plot.
        '''
        n = len(self.data)
        # ecdf = (0.5:n-0.5)/n;
        ecdf = arange(1, n + 1) / (n + 1)
        mcdf = self.cdf(self.data)
        p1 = [0, 1]
        plotbackend.plot(ecdf, mcdf, symb2,
                         p1, p1, symb1)
        plotbackend.xlabel('Empirical')
        plotbackend.ylabel('Model (%s)' % self.dist.name)
        plotbackend.title('Residual Probability Plot')
        plotbackend.axis('equal')
        plotbackend.axis([0, 1, 0, 1])

    def _pvalue(self, theta, x, unknown_numpar=None):
        ''' Return P-value for the fit using Moran's negative log Product
        Spacings statistic

            where theta are the parameters (including loc and scale)

            Note: the data in x must be sorted
        '''
        dx = np.diff(x, axis=0)
        tie = (dx == 0)
        if any(tie):
            warnings.warn(
                'P-value is on the conservative side (i.e. too large) due to' +
                ' ties in the data!')

        T = self.dist.nlogps(theta, x)

        n = len(x)
        np1 = n + 1
        if unknown_numpar is None:
            k = len(theta)
        else:
            k = unknown_numpar

        isParUnKnown = True
        m = (np1) * (log(np1) + 0.57722) - 0.5 - 1.0 / (12. * (np1))
        v = (np1) * (pi ** 2. / 6.0 - 1.0) - 0.5 - 1.0 / (6. * (np1))
        C1 = m - sqrt(0.5 * n * v)
        C2 = sqrt(v / (2.0 * n))
        # chi2 with n degrees of freedom
        Tn = (T + 0.5 * k * isParUnKnown - C1) / C2
        pvalue = chi2sf(Tn, n)  # _WAFODIST.chi2.sf(Tn, n)
        return pvalue


def test_doctstrings():
    import doctest
    doctest.testmod()


def test1():
    import wafo.stats as ws
    dist = ws.weibull_min
    # dist = ws.bradford
    R = dist.rvs(0.3, size=1000)
    phat = FitDistribution(dist, R, method='ml')

# Better CI for phat.par[i=0]
    Lp1 = Profile(phat, i=0)  # @UnusedVariable
    Lp1.plot()
    import matplotlib.pyplot as plt
    plt.show()
#    Lp2 = Profile(phat, i=2)
#    SF = 1./990
#    x = phat.isf(SF)
#
# CI for x
#    Lx = Profile(phat, i=0,x=x,link=phat.dist.link)
#    Lx.plot()
#    x_ci = Lx.get_bounds(alpha=0.2)
#
# CI for logSF=log(SF)
#    Lsf = phat.profile(i=0, logSF=log(SF), link=phat.dist.link)
#    Lsf.plot()
#    sf_ci = Lsf.get_bounds(alpha=0.2)
#    pass


#    _WAFODIST = ppimport('wafo.stats.distributions')
# nbinom(10, 0.75).rvs(3)
#    import matplotlib
#    matplotlib.interactive(True)
#    t = _WAFODIST.bernoulli(0.75).rvs(3)
#    x = np.r_[5, 10]
#    npr = np.r_[9, 9]
#    t2 = _WAFODIST.bd0(x, npr)
# Examples   MLE and better CI for phat.par[0]
#    R = _WAFODIST.weibull_min.rvs(1, size=100);
#    phat = _WAFODIST.weibull_min.fit(R, 1, 1, par_fix=[nan, 0, nan])
#    Lp = phat.profile(i=0)
#    Lp.plot()
#    Lp.get_bounds(alpha=0.1)
#    R = 1. / 990
#    x = phat.isf(R)
#
# CI for x
#    Lx = phat.profile(i=0, x=x)
#    Lx.plot()
#    Lx.get_bounds(alpha=0.2)
#
# CI for logSF=log(SF)
#    Lpr = phat.profile(i=0, logSF=log(R), link=phat.dist.link)
#    Lpr.plot()
#    Lpr.get_bounds(alpha=0.075)
#
#    _WAFODIST.dlaplace.stats(0.8, loc=0)
# pass
#    t = _WAFODIST.planck(0.51000000000000001)
#    t.ppf(0.5)
#    t = _WAFODIST.zipf(2)
#    t.ppf(0.5)
#    import pylab as plb
#    _WAFODIST.rice.rvs(1)
#    x = plb.linspace(-5, 5)
#    y = _WAFODIST.genpareto.cdf(x, 0)
# plb.plot(x,y)
# plb.show()
#
#
#    on = ones((2, 3))
#    r = _WAFODIST.genpareto.rvs(0, size=100)
#    pht = _WAFODIST.genpareto.fit(r, 1, par_fix=[0, 0, nan])
#    lp = pht.profile()
if __name__ == '__main__':
    test1()
    # test_doctstrings()
