#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
# NOTE: To look at history using `git blame`, use `git blame -M -C -C`
#       instead of `git blame -Lxxx,+x`.
#
from copy import copy
from ._distn_infrastructure import entropy, rv_discrete, rv_continuous, rv_frozen

from scipy.stats import _continuous_distns
from scipy.stats import _discrete_distns
from scipy.stats._constants import _EULER
# from scipy.stats._continuous_distns import *
# from scipy.stats._discrete_distns import *
import scipy.special as sc
import numpy as np


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

for distname in _continuous_distns._distn_names + ["rv_histogram"]:  #pylint: disable=protected-access
    _loc[distname] = dist = copy(getattr(_continuous_distns, distname))
    dist.__doc__ = dist.__doc__.replace("scipy.stats", "wafo.stats")

for distname in _discrete_distns._distn_names:  #pylint: disable=protected-access
    _loc[distname] = dist = copy(getattr(_discrete_distns, distname))

    dist.__doc__ = dist.__doc__.replace("scipy.stats", "wafo.stats")

# For backwards compatibility e.g. pymc expects distributions.__all__.
__all__ = ['entropy', 'rv_discrete', 'rv_continuous', 'rv_histogram', "truncrayleigh"]

# Add only the distribution names, not the *_gen names.
__all__ += _continuous_distns._distn_names + _discrete_distns._distn_names  #pylint: disable=protected-access
