'''
Created on 8. mai 2014

@author: pab
'''
from __future__ import absolute_import
from .core import TrData
from .models import TrHermite, TrOchi, TrLinear
from ..stats import edf, skew, kurtosis
from ..interpolate import SmoothSpline
from scipy.special import ndtri as invnorm
from scipy.integrate import cumtrapz
import warnings
import numpy as np
floatinfo = np.finfo(float)


class TransformEstimator(object):
    '''
    Estimate transformation, g, from ovserved data.
        Assumption: a Gaussian process, Y, is related to the
                            non-Gaussian process, X, by Y = g(X).

    Parameters
    ----------
    method : string
        estimation method. Options are:
        'nonlinear' : smoothed crossing intensity (default)
        'mnonlinear': smoothed marginal cumulative distribution
        'hermite'   : cubic Hermite polynomial
        'ochi'      : exponential function
        'linear'    : identity.
    chkDer : bool
        False: No check on the derivative of the transform.
        True: Check if transform have positive derivative
    csm, gsm : real scalars
        defines the smoothing of the logarithm of crossing intensity and
        the transformation g, respectively. Valid values must be
            0<=csm,gsm<=1. (default csm=0.9, gsm=0.05)
        Smaller values gives smoother functions.
    param : vector (default (-5, 5, 513))
        defines the region of variation of the data X. If X(t) is likely to
        cross levels higher than 5 standard deviations then the vector param
        has to be modified. For example if X(t) is unlikely to cross a level
        of 7 standard deviations one can use param = (-7, 7, 513).
    crossdef : string
        Crossing definition used in the crossing spectrum:
         'u'   or 1: only upcrossings
         'uM'  or 2: upcrossings and Maxima (default)
         'umM' or 3: upcrossings, minima, and Maxima.
         'um'  or 4: upcrossings and minima.
    plotflag : int
        0 no plotting (Default)
        1 plots empirical and smoothed g(u) and the theoretical for a
            Gaussian model.
        2 monitor the development of the estimation
    Delay : real scalar
        Delay time for each plot when PLOTFLAG==2.
    linextrap: int
        0 use a regular smoothing spline
        1 use a smoothing spline with a constraint on the ends to ensure
            linear extrapolation outside the range of the data. (default)
    cvar: real scalar
        Variances for the the crossing intensity. (default  1)
    gvar: real scalar
        Variances for the empirical transformation, g. (default  1)
    ne : int
        Number of extremes (maxima & minima) to remove from the estimation
        of the transformation. This makes the estimation more robust
        against outliers. (default 7)
    ntr : int
        Maximum length of empirical crossing intensity or CDF. The
        empirical crossing intensity or CDF is interpolated linearly before
        smoothing if their lengths exceeds Ntr. A reasonable NTR will
        significantly speed up the estimation for long time series without
        loosing any accuracy. NTR should be chosen greater than PARAM(3).
        (default 10000)
    multip : Bool
        False: the data in columns belong to the same seastate (default).
        True: the data in columns are from separate seastates.
    '''

    def __init__(self, method='nonlinear', chkder=True, plotflag=False,
                 csm=.95, gsm=.05, param=(-5, 5, 513), delay=2, ntr=10000,
                 linextrap=True, ne=7, cvar=1, gvar=1, multip=False,
                 crossdef='uM', monitor=False):
        self.method = method
        self.chkder = chkder
        self.plotflag = plotflag
        self.csm = csm
        self.gsm = gsm
        self.param = param
        self.delay = delay
        self.ntr = ntr
        self.linextrap = linextrap
        self.ne = ne
        self.cvar = cvar
        self.gvar = gvar
        self.multip = multip
        self.crossdef = crossdef

    def _check_tr(self, tr, tr_raw):
        eps = floatinfo.eps
        x = tr.args
        mean = tr.mean
        sigma = tr.sigma
        for ix in range(5):
            dy = np.diff(tr.data)
            if (dy <= 0).any():
                dy[dy > 0] = eps
                gvar = -(np.hstack((dy, 0)) + np.hstack((0, dy))) / 2 + eps
                pp_tr = SmoothSpline(tr_raw.args, tr_raw.data, p=1,
                                     lin_extrap=self.linextrap,
                                     var=ix * gvar)
                tr = TrData(pp_tr(x), x, mean=mean, sigma=sigma)
            else:
                break
        else:
            msg = '''
            The estimated transfer function, g, is not
            a strictly increasing function.
            The transfer function is possibly not sufficiently smoothed.
            '''
            warnings.warn(msg)
        return tr

    def _trdata_lc(self, level_crossings, mean=None, sigma=None):
        '''
        Estimate transformation, g, from observed crossing intensity.

        Assumption: a Gaussian process, Y, is related to the
                    non-Gaussian process, X, by Y = g(X).

        Parameters
        ----------
        mean, sigma : real scalars
            mean and standard deviation of the process
        **options :
        csm, gsm : real scalars
            defines the smoothing of the crossing intensity and the
            transformation g.
            Valid values must be 0<=csm,gsm<=1. (default csm = 0.9 gsm=0.05)
            Smaller values gives smoother functions.
        param :
            vector which defines the region of variation of the data X.
                     (default [-5, 5, 513]).
        monitor : bool
            if true monitor development of estimation
        linextrap : bool
            if true use a smoothing spline with a constraint on the ends to
            ensure linear extrapolation outside the range of data. (default)
            otherwise use a regular smoothing spline
        cvar, gvar : real scalars
            Variances for the crossing intensity and the empirical
            transformation, g. (default  1)
        ne : scalar integer
            Number of extremes (maxima & minima) to remove from the estimation
            of the transformation. This makes the estimation more robust
            against outliers. (default 7)
        ntr :  scalar integer
            Maximum length of empirical crossing intensity. The empirical
            crossing intensity is interpolated linearly  before smoothing if
            the length exceeds ntr. A reasonable NTR (eg. 1000) will
            significantly speed up the estimation for long time series without
            loosing any accuracy. NTR should be chosen greater than PARAM(3).
            (default inf)

        Returns
        -------
        gs, ge : TrData objects
            smoothed and empirical estimate of the transformation g.

        Notes
        -----
        The empirical crossing intensity is usually very irregular.
        More than one local maximum of the empirical crossing intensity
        may cause poor fit of the transformation. In such case one
        should use a smaller value of GSM or set a larger variance for GVAR.
        If X(t) is likely to cross levels higher than 5 standard deviations
        then the vector param has to be modified.  For example if X(t) is
        unlikely to cross a level of 7 standard deviations one can use
        param = [-7 7 513].

        Example
        -------
        >>> import wafo.spectrum.models as sm
        >>> import wafo.transform.models as tm
        >>> from wafo.objects import mat2timeseries
        >>> Hs = 7.0
        >>> Sj = sm.Jonswap(Hm0=Hs)
        >>> S = Sj.tospecdata()   #Make spectrum object from numerical values
        >>> S.tr = tm.TrOchi(mean=0, skew=0.16, kurt=0,
        ...        sigma=Hs/4, ysigma=Hs/4)
        >>> xs = S.sim(ns=2**16, iseed=10)
        >>> ts = mat2timeseries(xs)
        >>> tp = ts.turning_points()
        >>> mm = tp.cycle_pairs()
        >>> lc = mm.level_crossings()
        >>> g0, g0emp = lc.trdata(monitor=True) # Monitor the development
        >>> g1, g1emp = lc.trdata(gvar=0.5 ) # Equal weight on all points
        >>> g2, g2emp = lc.trdata(gvar=[3.5, 0.5, 3.5])  # Less weight on ends
        >>> int(S.tr.dist2gauss()*100)
        141
        >>> int(g0emp.dist2gauss()*100)
        380995
        >>> int(g0.dist2gauss()*100)
        143
        >>> int(g1.dist2gauss()*100)
        162
        >>> int(g2.dist2gauss()*100)
        120

        g0.plot() # Check the fit.

        See also
          troptset, dat2tr, trplot, findcross, smooth

        NB! the transformated data will be N(0,1)

        Reference
        ---------
        Rychlik , I., Johannesson, P., and Leadbetter, M.R. (1997)
        "Modelling and statistical analysis of ocean wavedata
        using a transformed Gaussian process",
        Marine structures, Design, Construction and Safety,
        Vol 10, pp 13--47
        '''
        if mean is None:
            mean = level_crossings.mean
        if sigma is None:
            sigma = level_crossings.sigma
        lc1, lc2 = level_crossings.args, level_crossings.data
        intensity = level_crossings.intensity

        Ne = self.ne
        ncr = len(lc2)
        if ncr > self.ntr and self.ntr > 0:
            x0 = np.linspace(lc1[Ne], lc1[-1 - Ne], self.ntr)
            lc1, lc2 = x0, np.interp(x0, lc1, lc2)
            Ne = 0
            Ner = self.ne
            ncr = self.ntr
        else:
            Ner = 0

        ng = len(np.atleast_1d(self.gvar))
        if ng == 1:
            gvar = self.gvar * np.ones(ncr)
        else:
            gvar = np.interp(np.linspace(0, 1, ncr),
                             np.linspace(0, 1, ng), self.gvar)

        uu = np.linspace(*self.param)
        g1 = sigma * uu + mean

        if Ner > 0:  # Compute correction factors
            cor1 = np.trapz(lc2[0:Ner + 1], lc1[0:Ner + 1])
            cor2 = np.trapz(lc2[-Ner - 1::], lc1[-Ner - 1::])
        else:
            cor1 = 0
            cor2 = 0

        lc22 = np.hstack((0, cumtrapz(lc2, lc1) + cor1))

        if intensity:
            lc22 = (lc22 + 0.5 / ncr) / (lc22[-1] + cor2 + 1. / ncr)
        else:
            lc22 = (lc22 + 0.5) / (lc22[-1] + cor2 + 1)

        lc11 = (lc1 - mean) / sigma

        lc22 = invnorm(lc22)  # - ymean

        g2 = TrData(lc22.copy(), lc1.copy(), mean=mean, sigma=sigma)
        g2.setplotter('step')
        # NB! the smooth function does not always extrapolate well outside the
        # edges causing poor estimate of g
        # We may alleviate this problem by: forcing the extrapolation
        # to be linear outside the edges or choosing a lower value for csm2.

        inds = slice(Ne, ncr - Ne)  # indices to points we are smoothing over
        slc22 = SmoothSpline(lc11[inds], lc22[inds], self.gsm, self.linextrap,
                             gvar[inds])(uu)

        g = TrData(slc22.copy(), g1.copy(), mean=mean, sigma=sigma)

        if self.chkder:
            tr_raw = TrData(lc22[inds], lc11[inds], mean=mean, sigma=sigma)
            g = self._check_tr(g, tr_raw)

        if self.plotflag > 0:
            g.plot()
            g2.plot()

        return g, g2

    def _trdata_cdf(self, data):
        '''
        Estimate transformation, g, from observed marginal CDF.
        Assumption: a Gaussian process, Y, is related to the
                            non-Gaussian process, X, by Y = g(X).
        Parameters
        ----------
        options = options structure defining how the smoothing is done.
                     (See troptset for default values)
        Returns
        -------
        tr, tr_emp  = smoothed and empirical estimate of the transformation g.

        The empirical CDF is usually very irregular. More than one local
        maximum of the empirical CDF may cause poor fit of the transformation.
        In such case one should use a smaller value of GSM or set a larger
        variance for GVAR.  If X(t) is likely to cross levels higher than 5
        standard deviations then the vector param has to be modified. For
        example if X(t) is unlikely to cross a level of 7 standard deviations
        one can use  param = [-7 7 513].
        '''
        mean = data.mean()
        sigma = data.std()
        cdf = edf(data.ravel())
        Ne = self.ne
        nd = len(cdf.data)
        if nd > self.ntr and self.ntr > 0:
            x0 = np.linspace(cdf.args[Ne], cdf.args[nd - 1 - Ne], self.ntr)
            cdf.data = np.interp(x0, cdf.args, cdf.data)
            cdf.args = x0
            Ne = 0
        uu = np.linspace(*self.param)

        ncr = len(cdf.data)
        ng = len(np.atleast_1d(self.gvar))
        if ng == 1:
            gvar = self.gvar * np.ones(ncr)
        else:
            self.gvar = np.atleast_1d(self.gvar)
            gvar = np.interp(np.linspace(0, 1, ncr),
                             np.linspace(0, 1, ng), self.gvar.ravel())

        ind = np.flatnonzero(np.diff(cdf.args) > 0)  # remove equal points
        nd = len(ind)
        ind1 = ind[Ne:nd - Ne]
        tmp = invnorm(cdf.data[ind])

        x = sigma * uu + mean
        pp_tr = SmoothSpline(cdf.args[ind1], tmp[Ne:nd - Ne], p=self.gsm,
                             lin_extrap=self.linextrap, var=gvar[ind1])
        tr = TrData(pp_tr(x), x, mean=mean, sigma=sigma)
        tr_emp = TrData(tmp, cdf.args[ind], mean=mean, sigma=sigma)
        tr_emp.setplotter('step')

        if self.chkder:
            tr_raw = TrData(tmp[Ne:nd - Ne], cdf.args[ind1], mean=mean,
                            sigma=sigma)
            tr = self._check_tr(tr, tr_raw)

        if self.plotflag > 0:
            tr.plot()
            tr_emp.plot()
        return tr, tr_emp

    def trdata(self, timeseries):
        '''

        Returns
        -------
        tr, tr_emp : TrData objects
            with the smoothed and empirical transformation, respectively.

        TRDATA estimates the transformation in a transformed Gaussian model.
        Assumption: a Gaussian process, Y, is related to the
        non-Gaussian process, X, by Y = g(X).

        The empirical crossing intensity is usually very irregular.
        More than one local maximum of the empirical crossing intensity may
        cause poor fit of the transformation. In such case one should use a
        smaller value of CSM. In order to check the effect of smoothing it is
        recomended to also plot g and g2 in the same plot or plot the smoothed
        g against an interpolated version of g (when CSM=GSM=1).

        Example
        -------
        >>> import wafo.spectrum.models as sm
        >>> import wafo.transform.models as tm
        >>> from wafo.objects import mat2timeseries
        >>> Hs = 7.0
        >>> Sj = sm.Jonswap(Hm0=Hs)
        >>> S = Sj.tospecdata()   #Make spectrum object from numerical values
        >>> S.tr = tm.TrOchi(mean=0, skew=0.16, kurt=0,
        ...        sigma=Hs/4, ysigma=Hs/4)
        >>> xs = S.sim(ns=2**16, iseed=10)
        >>> ts = mat2timeseries(xs)
        >>> g0, g0emp = ts.trdata(monitor=True)
        >>> g1, g1emp = ts.trdata(method='m', gvar=0.5 )
        >>> g2, g2emp = ts.trdata(method='n', gvar=[3.5, 0.5, 3.5])
        >>> int(S.tr.dist2gauss()*100)
        141
        >>> int(g0emp.dist2gauss()*100)
        217949
        >>> int(g0.dist2gauss()*100)
        93
        >>> int(g1.dist2gauss()*100)
        66
        >>> int(g2.dist2gauss()*100)
        84

        See also
        --------
        LevelCrossings.trdata
        wafo.transform.models

        References
        ----------
        Rychlik, I. , Johannesson, P and Leadbetter, M. R. (1997)
        "Modelling and statistical analysis of ocean wavedata using
        transformed Gaussian process."
        Marine structures, Design, Construction and Safety, Vol. 10, No. 1,
        pp 13--47

        Brodtkorb, P, Myrhaug, D, and Rue, H (1999)
        "Joint distribution of wave height and crest velocity from
        reconstructed data"
        in Proceedings of 9th ISOPE Conference, Vol III, pp 66-73
        '''

        data = np.atleast_1d(timeseries.data)
        ma = data.mean()
        sa = data.std()
        method = self.method[0]
        if method == 'l':
            return TrLinear(mean=ma, sigma=sa), TrLinear(mean=ma, sigma=sa)
        if method == 'n':
            tp = timeseries.turning_points()
            mM = tp.cycle_pairs()
            lc = mM.level_crossings(self.crossdef)
            return self._trdata_lc(lc)
        elif method == 'm':
            return self._trdata_cdf(data)
        elif method == 'h':
            ga1 = skew(data)
            ga2 = kurtosis(data, fisher=True)  # kurt(xx(n+1:end))-3;
            up = min(4 * (4 * ga1 / 3) ** 2, 13)
            lo = (ga1 ** 2) * 3 / 2
            kurt1 = min(up, max(ga2, lo)) + 3
            return TrHermite(mean=ma, var=sa ** 2, skew=ga1, kurt=kurt1)
        elif method[0] == 'o':
            ga1 = skew(data)
            return TrOchi(mean=ma, var=sa ** 2, skew=ga1)

    __call__ = trdata
