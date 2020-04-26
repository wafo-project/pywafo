
import os

import numpy as np
from numpy.testing import assert_allclose, suppress_warnings  # @UnresolvedImport
import pytest
from wafo import stats

from scipy.stats.tests.test_continuous_basic import distcont

# this is not a proper statistical test for convergence, but only
# verifies that the estimate and true values don't differ by too much
DISTCONT = distcont + [['truncrayleigh', (3.5704770516650459,)]]
FIT_SIZES = [1000, 5000]  # sample sizes to try

THRESH_PERCENT = 0.25  # percent of true parameters for fail cut-off
THRESH_MIN = 0.75  # minimum difference estimate - true to fail test

FAILING_FITS_MLE = [
        'gausshyper',
        'genexpon',
        'kappa4',
        'ksone',
        'levy_stable',  # extremely slow
        'trapz',
        'truncexpon',
        'wrapcauchy'
]

FAILING_FITS_MPS = [
        'gausshyper',  # extremely slow
        'kappa4',
        'ksone',
        'levy_stable',  # extremely slow
        'trapz',
        'wrapcauchy'
]

# Don't run the fit test on these:
SKIP_FIT = [
    'erlang',  # Subclass of gamma, generates a warning.
]

SKIP_FIT_MPS = {"geninvgauss",
                "norminvgauss",
                "skewnorm",
                }

def cases_test_cont_fit():
    # this tests the closeness of the estimated parameters to the true
    # parameters with fit method of continuous distributions
    # Note: is slow, some distributions don't converge with sample size <= 10000


    for distname, arg in DISTCONT:
        if distname not in SKIP_FIT:
            yield distname, arg, 'ml'
            if distname not in SKIP_FIT_MPS:
                yield distname, arg, 'mps'


@pytest.mark.slow
@pytest.mark.parametrize('distname,arg,method', cases_test_cont_fit())
def test_cont_fit(distname, arg, method):
    options = dict(method=method)
    failing_fits = dict(mps=FAILING_FITS_MPS, ml=FAILING_FITS_MLE)[method]
    if distname in failing_fits:
        # Skip failing fits unless overridden
        try:
            xfail = not int(os.environ['SCIPY_XFAIL'])
        except Exception:
            xfail = True
        if xfail:
            msg = "Fitting %s doesn't work reliably yet" % distname
            msg += " [Set environment variable SCIPY_XFAIL=1 to run this test nevertheless.]"
            pytest.xfail(msg)
            options['floc']=0.
            options['fscale']=1.

    distfn = getattr(stats, distname)

    truearg = np.hstack([arg, [0.0, 1.0]])
    diffthreshold = np.max(np.vstack([truearg*THRESH_PERCENT,
                                      np.full(distfn.numargs+2, THRESH_MIN)]),
                           0)
    opt = options.copy()
    for fit_size in FIT_SIZES:
        # Note that if a fit succeeds, the other FIT_SIZES are skipped
        np.random.seed(1234)

        with np.errstate(all='ignore'), suppress_warnings() as sup:
            sup.filter(category=DeprecationWarning, message=".*frechet_")
            rvs = distfn.rvs(size=fit_size, *arg)
            # phat = distfn.fit2(rvs)

            phat = distfn.fit2(rvs, **opt)

            est = phat.par
            # est = distfn.fit(rvs)  # start with default values

        diff = est - truearg

        # threshold for location
        diffthreshold[-2] = np.max([np.abs(rvs.mean())*THRESH_PERCENT,THRESH_MIN])

        if np.any(np.isnan(est)):
            raise AssertionError('nan returned in fit')
        else:
            if np.all(np.abs(diff) <= diffthreshold) or phat.pvalue > 0.05:
                break
    else:
        txt = 'parameter: %s\n' % str(truearg)
        txt += 'estimated: %s\n' % str(est)
        txt += 'diff     : %s\n' % str(diff)
        txt += 'pvalue   : %s\n' % str(phat.pvalue)
        raise AssertionError('fit not very good in %s\n' % distfn.name + txt)


def _check_loc_scale_mle_fit(name, data, desired, atol=None):
    d = getattr(stats, name)
    actual = d.fit(data)[-2:]
    assert_allclose(actual, desired, atol=atol,
                    err_msg='poor mle fit of (loc, scale) in %s' % name)


def test_non_default_loc_scale_mle_fit():
    data = np.array([1.01, 1.78, 1.78, 1.78, 1.88, 1.88, 1.88, 2.00])
    _check_loc_scale_mle_fit('uniform', data, [1.01, 0.99], 1e-3)
    _check_loc_scale_mle_fit('expon', data, [1.01, 0.73875], 1e-3)


def test_expon_fit():
    """gh-6167"""
    data = [0, 0, 0, 0, 2, 2, 2, 2]
    phat = stats.expon.fit(data, floc=0)
    assert_allclose(phat, [0, 1.0], atol=1e-3)

