#@PydevCodeAnalysisIgnore

import scipy.stats as ss
from scipy.stats import *
from .core import *
from .distributions import *
from . import estimation
__doc__ = ss.__doc__.replace('scipy.stats',
                             'wafo.stats').replace("a growing library of statistical functions.",
                                                   """a growing library of statistical functions. Most of its
functionality is taken from scipy.stats. However, wafo.stats extends
scipy.stats with the possibility to use maximum spacing (MS)
estimator as an alternative to the maximum likelihood (ML) estimator and
the technique known as profile likelihood to construct better confidence intervals.
See https://en.wikipedia.org/wiki/Maximum_spacing_estimation and
https://journals.sagepub.com/doi/pdf/10.1177/1536867X0700700305

To use the extended functionality for parameter estimation you must call the dist.fit2 method
or alternatively use the FitDistribution class in wafo.stats.estimation.
To construct better confidence intervalls for the estimated distribution parameters, probability or quantile
you must use the Profile-, ProfileProbability- or ProfileQuantile-class, respectively, in wafo.stats.estimation module.

""")


# remove vonmises_cython from __all__, I don't know why it is included
__all__ = [s for s in dir() if not (s.startswith('_') or s.endswith('cython'))]

from numpy.testing import Tester
test = Tester().test
