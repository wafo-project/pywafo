#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
# NOTE: To look at history using `git blame`, use `git blame -M -C -C`
#       instead of `git blame -Lxxx,+x`.
#
from copy import copy
from functools import partial
from ._distn_infrastructure import entropy, rv_discrete, rv_continuous, rv_frozen

from scipy.stats import _continuous_distns
from scipy.stats import _discrete_distns
from scipy.stats._constants import _EULER
# from scipy.stats._continuous_distns import *
# from scipy.stats._discrete_distns import *
import scipy.special as sc
import numpy as np
from scipy import optimize
from scipy.stats.mstats_basic import mode


def _betaprime_fitstart(self, data, fitstart):  # pab
    g1 = np.mean(data)
    g2 = mode(data)[0]

    def func(x):
        a, b = x
        me = a / (b - 1) if 1 < b else 1e100
        mo = (a - 1) / (b + 1) if 1 <= a else 0
        return [me - g1, mo - g2]
    a, b = optimize.fsolve(func, (1.0, 1.5))
    return fitstart(data, args=(a, b))


def _bradford_fitstart(self, data, fitstart):  # pab
    loc = data.min() - 1e-4
    scale = (data - loc).max()
    m = np.mean((data - loc) / scale)
    fun = lambda c: (c - sc.log1p(c)) / (c * sc.log1p(c)) - m
    res = optimize.root(fun, 0.3)
    c = res.x
    return c, loc, scale


def _chi_fitstart(self, data, fitstart):  # pab
    m = data.mean()
    v = data.var()
    # Supply a starting guess with method of moments:
    df = max(int(v + m ** 2), 1)
    return fitstart(data, args=(df,))

def _f_fitstart(self, data, fitstart):  # pab
    m = data.mean()
    v = data.var()
    # Supply a starting guess with method of moments:
    dfd = max(np.round(2 * m / (m - 1)), 5)
    dfn = max(
        np.round(2 * dfd * dfd * (dfd - 2) /
                 (v * (dfd - 4) * (dfd - 2) ** 2 - 2 * dfd * dfd)), 1)
    return fitstart(data, args=(dfn, dfd,))


def _weibull_min_fitstart(self, data, fitstart):  # pab
    loc = data.min() - 0.01  # *np.std(data)
    chat = 1. / (6 ** (1 / 2) / np.pi * np.std(np.log(data - loc)))
    scale = np.mean((data - loc) ** chat) ** (1. / chat)
    return chat, loc, scale


def _genpareto_fitstart(self, data, fitstart):  # pab
    d = np.asarray(data)
    loc = d.min() - 0.01 * d.std()
    # moments estimator
    d1 = d - loc
    m = d1.mean()
    s = d1.std()

    shape = ((m / s) ** 2 - 1) / 2
    scale = m * ((m / s) ** 2 + 1) / 2
    return shape, loc, scale


def _genextreme_fitstart(self, data, fitstart): # pab
    d = np.asarray(data)
    # Probability weighted moments
    log = np.log
    n = len(d)
    d.sort()
    koeff1 = np.r_[0:n] / (n - 1)
    koeff2 = koeff1 * (np.r_[0:n] - 1) / (n - 2)
    b2 = np.dot(koeff2, d) / n
    b1 = np.dot(koeff1, d) / n
    b0 = d.mean()
    z = (2 * b1 - b0) / (3 * b2 - b0) - log(2) / log(3)
    shape = 7.8590 * z + 2.9554 * z ** 2
    scale = (2 * b1 - b0) * shape / (np.exp(sc.gammaln(1 + shape)) * (1 - 2 ** (-shape)))
    loc = b0 + scale * (sc.expm1(sc.gammaln(1 + shape))) / shape
    return shape, loc, scale


def _lognorm_fitstart(self, data, fitstart):  # TODO: delete! pab
    scale = data.std()
    loc = data.min() - 0.001
    logd = np.log(data - loc)
    m = logd.mean()
    s = np.sqrt((logd ** 2).mean() - m ** 2)
    return s, loc, scale


def _gilbrat_fitstart(self, data, fitstart): # pab
    scale = data.std()
    loc = data.min() - 0.001
    return loc, scale


def _ncx2_fitstart(self, data, fitstart): # pab
    m = data.mean()
    v = data.var()
    # Supply a starting guess with method of moments:
    nc = (v / 2 - m) / 2
    df = m - nc
    return fitstart(data, args=(df, nc))


def _nct_fitstart(self, data, fitstart):  # pab
    me = np.mean(data)
    # g2 = mode(data)[0]
    sa = np.std(data)

    def func(df):
        return ((df - 2) * (4 * df - 1) -
                (4 * df - 1) * df / (sa ** 2 + me ** 2) +
                me ** 2 / (sa ** 2 + me ** 2) * (df * (4 * df - 1) -
                                                 3 * df))

    df0 = np.maximum(2 * sa / (sa - 1), 1)
    df = optimize.fsolve(func, df0)
    mu = me * (1 - 3 / (4 * df - 1))
    return fitstart(data, args=(df, mu))


def _reciprocal_fitstart(self, data, fitstart):
    a = np.min(data)
    a -= 0.01 * np.abs(a)
    b = np.max(data)
    b += 0.01 * np.abs(b)
    if a <= 0:
        da = np.abs(a) + 0.001
        a += da
        b += da
    return fitstart(data, args=(a, b))


class truncrayleigh_gen(rv_continuous):

    r"""A truncated Rayleigh continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `truncrayleigh` is::

    .. math::

        f(x, c) = 1 - \exp(-((x+c)^2-c^2)/2)

    for :math:`x \ge 0`, :math:`c \ge 0`. This class takes
    :math:`c` as truncation parameters. %(after_notes)s

    %(example)s


    """
    _support_mask = rv_continuous._open_support_mask

    def _argcheck(self, c):
        return (c >= 0)

    def _fitstart(self, data, args=None):
        if args is None:
            n = len(data)
            args = (np.sqrt(np.sum(data**2)/n/2),) # Initial guess (MLE with c=0)

        return args + self.fit_loc_scale(data, *args)

    def _pdf(self, r, c):
        rc = r + c
        return rc * np.exp(-(rc * rc - c * c) / 2.0)

    def _logpdf(self, r, c):
        rc = r + c
        return np.log(rc) - (rc * rc - c * c) / 2.0

    def _cdf(self, r, c):
        rc = r + c
        return - sc.expm1(-(rc * rc - c * c) / 2.0)  #pylint: disable=no-member

    def _logsf(self, r, c):
        rc = r + c
        return -(rc * rc - c * c) / 2.0

    def _sf(self, r, c):
        return np.exp(self._logsf(r, c))

    def _ppf(self, q, c):
        return np.sqrt(c * c - 2 * sc.log1p(-q)) - c  #pylint: disable=no-member

    def _stats(self, c):
        # TODO: correct this it is wrong!
        pi = np.pi
        val = 4 - pi
        return (np.sqrt(pi / 2),
                val / 2,
                2 * (pi - 3) * np.sqrt(pi) / val ** 1.5,
                6 * pi / val - 16 / val ** 2)

    def _entropy(self, c):
        # TODO: correct this it is wrong!
        return _EULER / 2.0 + 1 - 0.5 * np.log(2)


truncrayleigh = truncrayleigh_gen(a=0.0, name="truncrayleigh", shapes='c')
truncrayleigh.__doc__ = truncrayleigh.__doc__.replace("scipy.stats", "wafo.stats")

_loc = locals()

_patch_fit_start_dict = dict(betaprime=_betaprime_fitstart,
                             bradford=_bradford_fitstart,
                             chi=_chi_fitstart,
                             f=_f_fitstart,
                             weibull_min=_weibull_min_fitstart,
                             genpareto=_genpareto_fitstart,
                             genextreme=_genextreme_fitstart,
                             lognorm=_lognorm_fitstart,
                             gilbrat=_gilbrat_fitstart,
                             ncx2=_ncx2_fitstart,
                             nct=_nct_fitstart,
                             reciprocal=_reciprocal_fitstart,
                             )


for distname in _continuous_distns._distn_names + ["rv_histogram"]:  # pylint: disable=protected-access
    _loc[distname] = dist = copy(getattr(_continuous_distns, distname))
    dist.__doc__ = dist.__doc__.replace("scipy.stats", "wafo.stats")
    if distname in _patch_fit_start_dict:
        dist._fitstart = partial(_patch_fit_start_dict[distname], dist, fitstart=dist._fitstart)

for distname in _discrete_distns._distn_names:  # pylint: disable=protected-access
    _loc[distname] = dist = copy(getattr(_discrete_distns, distname))

    dist.__doc__ = dist.__doc__.replace("scipy.stats", "wafo.stats")

# For backwards compatibility e.g. pymc expects distributions.__all__.
__all__ = ['entropy', 'rv_discrete', 'rv_continuous', 'rv_histogram', "truncrayleigh"]

# Add only the distribution names, not the *_gen names.
__all__ += _continuous_distns._distn_names + _discrete_distns._distn_names  #pylint: disable=protected-access
