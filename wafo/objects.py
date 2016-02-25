

# Name:        module1
# Purpose:
#
# Author:      pab
#
# Created:     16.09.2008
# Copyright:   (c) pab 2008
# Licence:     <your licence>

# !/usr/bin/env python


from __future__ import absolute_import, division
from .transform.core import TrData
from .transform.estimation import TransformEstimator
from .stats import distributions
from .misc import (nextpow2, findtp, findrfc, findtc, findcross,
                   ecross, JITImport, DotDict, gravity, findrfc_astm)
from .interpolate import stineman_interp
from .containers import PlotData
from .plotbackend import plotbackend
from scipy.integrate import trapz
from scipy.signal import welch, lfilter
from scipy.signal.windows import get_window  # @UnusedImport
from scipy import special
from scipy.interpolate.interpolate import interp1d
from scipy.special import ndtr as cdfnorm
import warnings
import numpy as np
from numpy import (inf, pi, zeros, ones, sqrt, where, log, exp, cos, sin,
                   arcsin, mod,
                   linspace, arange, sort, all, abs, vstack, hstack,
                   atleast_1d, finfo, polyfit, r_, nonzero,
                   cumsum, ravel, isnan, ceil, diff, array)
from numpy.fft import fft  # @UnusedImport
from numpy.random import randn
from matplotlib.mlab import psd, detrend_mean


floatinfo = finfo(float)

_wafocov = JITImport('wafo.covariance')
_wafocov_estimation = JITImport('wafo.covariance.estimation')
_wafospec = JITImport('wafo.spectrum')

__all__ = ['TimeSeries', 'LevelCrossings', 'CyclePairs', 'TurningPoints',
           'sensortypeid', 'sensortype']


def _invchi2(q, df):
    return special.chdtri(df, q)


class LevelCrossings(PlotData):

    '''
    Container class for Level crossing data objects in WAFO

    Member variables
    ----------------
    data : array-like
        number of upcrossings or upcrossingintensity
    args : array-like
        crossing levels

    Examples
    --------
    >>> import wafo.data
    >>> import wafo.objects as wo
    >>> x = wafo.data.sea()
    >>> ts = wo.mat2timeseries(x)

    >>> tp = ts.turning_points()
    >>> mm = tp.cycle_pairs()

    >>> lc = mm.level_crossings()
    >>> h2 = lc.plot()
    '''

    def __init__(self, *args, **kwds):
        options = dict(title='Level crossing spectrum',
                       xlab='Levels', ylab='Count',
                       plotmethod='semilogy',
                       plot_args=['b'],
                       plot_args_children=['r--'])
        options.update(**kwds)
        super(LevelCrossings, self).__init__(*args, **options)
        self.intensity = kwds.get('intensity', False)
        self.sigma = kwds.get('sigma', None)
        self.mean = kwds.get('mean', None)
        # self.setplotter(plotmethod='step')

        icmax = self.data.argmax()
        if self.data is not None:
            if self.sigma is None or self.mean is None:
                logcros = where(self.data == 0.0, inf, -log(self.data))
                logcmin = logcros[icmax]
                logcros = sqrt(2 * abs(logcros - logcmin))
                logcros[0:icmax + 1] = 2 * logcros[
                    icmax] - logcros[0:icmax + 1]
                ncr = 10
                # least square fit
                p = polyfit(self.args[ncr:-ncr], logcros[ncr:-ncr], 1)
                if self.sigma is None:
                    # estimated standard deviation of x
                    self.sigma = 1.0 / p[0]
                if self.mean is None:
                    self.mean = -p[1] / p[0]  # self.args[icmax]
            cmax = self.data[icmax]
            x = (self.args - self.mean) / self.sigma
            y = cmax * exp(-x ** 2 / 2.0)
            self.children = [PlotData(y, self.args)]

    def extrapolate(self, u_min=None, u_max=None, method='ml', dist='genpar',
                    plotflag=0):
        '''
        Returns an extrapolated level crossing spectrum

        Parameters
        -----------
        u_min, u_max : real scalars
            extrapolate below u_min and above u_max.
        method : string
            describing the method of estimation. Options are:
            'ml' : Maximum Likelihood method (default)
            'mps': Maximum Product Spacing method
        dist : string
            defining distribution function. Options are:
            genpareto : Generalized Pareto distribution (GPD)
            expon : Exponential distribution (GPD with k=0)
            rayleigh : truncated Rayleigh distribution
        plotflag : scalar integer
            1: Diagnostic plots. (default)
            0: Don't plot diagnostic plots.

        Returns
        -------
        lc : LevelCrossing object
            with the estimated level crossing spectrum
            Est      = Estimated parameters. [struct array]

        Extrapolates the level crossing spectrum (LC) for high and for low
        levels.
        The tails of the LC is fitted to a survival function of a GPD.
           H(x) = (1-k*x/s)^(1/k)               (GPD)
        The use of GPD is motivated by POT methods in extreme value theory.
        For k=0 the GPD is the exponential distribution
           H(x) = exp(-x/s),  k=0               (expon)
        The tails with the survival function of a truncated Rayleigh
        distribution.
           H(x) = exp(-((x+x0).^2-x0^2)/s^2)    (rayleigh)
        where x0 is the distance from the truncation level to where the LC has
        its maximum.
        The method 'gpd' uses the GPD. We recommend the use of 'gpd,ml'.
        The method 'exp' uses the Exp.
        The method 'ray' uses Ray, and should be used if the load is a
        Gaussian process.

        Example
        -------
        >>> import wafo.data
        >>> import wafo.objects as wo
        >>> x = wafo.data.sea()
        >>> ts = wo.mat2timeseries(x)

        >>> tp = ts.turning_points()
        >>> mm = tp.cycle_pairs()
        >>> lc = mm.level_crossings()

        >>> s = x[:,1].std()
        >>> lc_gpd = lc.extrapolate(-2*s, 2*s)
        >>> lc_exp = lc.extrapolate(-2*s, 2*s, dist='expon')
        >>> lc_ray = lc.extrapolate(-2*s, 2*s, dist='rayleigh')

        lc.plot()
        lc_gpd.plot()
        lc_exp.plot()
        lc_ray.plot()


        See also
        --------
        cmat2extralc, rfmextrapolate, lc2rfmextreme, extralc, fitgenpar

        References
        ----------
        Johannesson, P., and Thomas, J-.J. (2000):
        Extrapolation of Rainflow Matrices.
        Preprint 2000:82, Mathematical statistics, Chalmers, pp. 18.
        '''

        i_max = self.data.argmax()
        c_max = self.data[i_max]
        lc_max = self.args[i_max]

        if u_min is None or u_max is None:
            fraction = sqrt(c_max)
            i = np.flatnonzero(self.data > fraction)
            if u_min is None:
                u_min = self.args[i.min()]
            if u_max is None:
                u_max = self.args[i.max()]
        lcf, lcx = self.data, self.args
        # Extrapolate LC for high levels
        [lc_High, phat_high] = self._extrapolate(lcx, lcf, u_max,
                                                 u_max - lc_max, method, dist)
        # Extrapolate LC for low levels
        [lcEst1, phat_low] = self._extrapolate(-lcx[::-1], lcf[::-1], -u_min,
                                               lc_max - u_min, method, dist)
        lc_Low = lcEst1[::-1, :]  # [-lcEst1[::-1, 0], lcEst1[::-1, 1::]]
        lc_Low[:, 0] *= -1

        if plotflag:
            plotbackend.semilogx(lcf, lcx,
                                 lc_High[:, 1], lc_High[:, 0],
                                 lc_Low[:, 1], lc_Low[:, 0])
        i_mask = (u_min < lcx) & (lcx < u_max)
        f = np.hstack((lc_Low[:, 1], lcf[i_mask], lc_High[:, 1]))
        x = np.hstack((lc_Low[:, 0], lcx[i_mask], lc_High[:, 0]))
        lc_out = LevelCrossings(f, x, sigma=self.sigma, mean=self.mean)
        lc_out.phat_high = phat_high
        lc_out.phat_low = phat_low
        return lc_out

    def _extrapolate(self, lcx, lcf, u, offset, method, dist):
        # Extrapolate the level crossing spectra for high levels

        method = method.lower()
        dist = dist.lower()

        # Excedences over level u
        Iu = lcx > u
        lcx1, lcf1 = lcx[Iu], lcf[Iu]
        lcf2, lcx2 = self._make_increasing(lcf1[::-1], lcx1[::-1])

        nim1 = 0
        x = []
        for xk, ni in zip(lcx2.tolist(), lcf2.tolist()):
            x.append(ones(ni - nim1) * xk)
            nim1 = ni

        x = np.hstack(x) - u

        df = 0.01
        xF = np.arange(0.0, 4 + df / 2, df)
        lcu = np.interp(u, lcx, lcf) + 1
        # Estimate tail
        if dist.startswith('gen'):
            genpareto = distributions.genpareto
            phat = genpareto.fit2(x, floc=0, method=method)
            SF = phat.sf(xF)

            covar = phat.par_cov[::2, ::2]
            # Calculate 90 # confidence region, an ellipse, for (k,s)
            D, B = np.linalg.eig(covar)
            b = phat.par[::2]
            if b[0] > 0:
                phat.upperlimit = u + b[1] / b[0]

            r = sqrt(-2 * log(1 - 90 / 100))  # 90 # confidence sphere
            Nc = 16 + 1
            ang = linspace(0, 2 * pi, Nc)
            # 90# Circle
            c0 = np.vstack(
                (r * sqrt(D[0]) * sin(ang), r * sqrt(D[1]) * cos(ang)))
        #    plot(c0(1,:),c0(2,:))

            # * ones((1, len(c0))) # Transform to ellipse for (k,s)
            c1 = np.dot(B, c0) + b[:, None]
        #    plot(c1(1,:),c1(2,:)), hold on

            # Calculate conf.int for lcu
            # Assumtion: lcu is Poisson distributed
            # Poissin distr. approximated by normal when calculating conf. int.
            dXX = 1.64 * sqrt(lcu)  # 90 # quantile for lcu

            lcEstCu = zeros((len(xF), Nc))
            lcEstCl = zeros((len(xF), Nc))
            for i in range(Nc):
                k = c1[0, i]
                s = c1[1, i]
                SF2 = genpareto.sf(xF, k, scale=s)
                lcEstCu[:, i] = (lcu + dXX) * (SF2)
                lcEstCl[:, i] = (lcu - dXX) * (SF2)
            # end

            lcEst = np.vstack((xF + u, lcu * (SF),
                               lcEstCl.min(axis=1), lcEstCu.max(axis=1))).T
        elif dist.startswith('exp'):
            expon = distributions.expon
            phat = expon.fit2(x, floc=0, method=method)
            SF = phat.sf(xF)
            lcEst = np.vstack((xF + u, lcu * (SF))).T

        elif dist.startswith('ray') or dist.startswith('trun'):
            phat = distributions.truncrayleigh.fit2(x, floc=0, method=method)
            SF = phat.sf(xF)
#            if False:
#                n = len(x)
#                Sx = sum((x + offset) ** 2 - offset ** 2)
# s = sqrt(Sx / n);  # Shape parameter
#                F = -np.expm1(-((xF + offset) ** 2 - offset ** 2) / s ** 2)
            lcEst = np.vstack((xF + u, lcu * (SF))).T
        else:
            raise ValueError()

        return lcEst, phat
        # End extrapolate

    def _make_increasing(self, f, t=None):
        # Makes the signal f strictly increasing.

        n = len(f)
        if t is None:
            t = np.arange(n)
        ff = [f[0], ]
        tt = [t[0], ]

        for i in range(1, n):
            if f[i] > ff[-1]:
                ff.append(f[i])
                tt.append(t[i])

        return np.asarray(ff), np.asarray(tt)

    def sim(self, ns, alpha):
        """
        Simulates process with given irregularity factor and crossing spectrum

        Parameters
        ----------
        ns : scalar, integer
            number of sample points.
        alpha : real scalar
            irregularity factor, 0<alpha<1, small  alpha  gives
            irregular process.

        Returns
        --------
        ts : timeseries object
            with times and values of the simulated process.

        Example
        -------
        >>> import wafo.spectrum.models as sm
        >>> from wafo.objects import mat2timeseries
        >>> Sj = sm.Jonswap(Hm0=7)
        >>> S = Sj.tospecdata()   #Make spectrum object from numerical values
        >>> alpha = S.characteristic('alpha')[0]
        >>> n = 10000
        >>> xs = S.sim(ns=n)
        >>> ts = mat2timeseries(xs)
        >>> tp = ts.turning_points()
        >>> mm = tp.cycle_pairs()
        >>> lc = mm.level_crossings()

        >>> xs2 = lc.sim(n,alpha)
        >>> ts2 = mat2timeseries(xs2)
        >>> Se  = ts2.tospecdata(L=324)

        >>> alpha2 = Se.characteristic('alpha')[0]
        >>> np.round(alpha2*10)
        array([ 7.])
        >>> np.abs(alpha-alpha2)<0.03
        array([ True], dtype=bool)

        >>> h0 = S.plot('b')
        >>> h1 = Se.plot('r')

        >>> lc2 = ts2.turning_points().cycle_pairs().level_crossings()

        >>> import pylab as plt
        >>> h = plt.subplot(211)
        >>> h2 = lc2.plot()
        >>> h = plt.subplot(212)
        >>> h0 = lc.plot()

        """

        # TODO: add a good example
        f = linspace(0, 0.49999, 1000)
        rho_st = 2. * sin(f * pi) ** 2 - 1.
        tmp = alpha * arcsin(sqrt((1. + rho_st) / 2))
        tmp = sin(tmp) ** 2
        a2 = (tmp - rho_st) / (1 - tmp)
        y = vstack((a2 + rho_st, 1 - a2)).min(axis=0)
        maxidx = y.argmax()
        # [maximum,maxidx]=max(y)

        rho_st = rho_st[maxidx]
        a2 = a2[maxidx]
        a1 = 2. * rho_st + a2 - 1.
        r0 = 1.
        r1 = -a1 / (1. + a2)
        r2 = (a1 ** 2 - a2 - a2 ** 2) / (1 + a2)
        sigma2 = r0 + a1 * r1 + a2 * r2
        # randn = np.random.randn
        e = randn(ns) * sqrt(sigma2)
        e[:2] = 0.0
        L0 = randn(1)
        L0 = hstack((L0, r1 * L0 + sqrt(1 - r2 ** 2) * randn(1)))
        # Simulate the process, starting in L0
        z0 = lfilter([1, a1, a2], ones(1), L0)
        L, unused_zf = lfilter(ones(1), [1, a1, a2], e, axis=0, zi=z0)

        epsilon = 1.01
        min_L = min(L)
        max_L = max(L)
        maxi = max(abs(r_[min_L, max_L])) * epsilon
        mini = -maxi

        u = linspace(mini, maxi, 101)
        G = cdfnorm(u)  # (1 + erf(u / sqrt(2))) / 2
        G = G * (1 - G)

        x = linspace(0, r1, 100)
        factor1 = 1. / sqrt(1 - x ** 2)
        factor2 = 1. / (1 + x)
        integral = zeros(u.shape, dtype=float)
        for i in range(len(integral)):
            y = factor1 * exp(-u[i] * u[i] * factor2)
            integral[i] = trapz(y, x)
        # end
        G = G - integral / (2 * pi)
        G = G / max(G)

        Z = ((u >= 0) * 2 - 1) * sqrt(-2 * log(G))

        sumcr = trapz(self.data, self.args)
        lc = self.data / sumcr
        lc1 = self.args
        mcr = trapz(lc1 * lc, lc1) if self.mean is None else self.mean
        if self.sigma is None:
            scr = trapz(lc1 ** 2 * lc, lc1)
            scr = sqrt(scr - mcr ** 2)
        else:
            scr = self.sigma
        lc2 = LevelCrossings(lc, lc1, mean=mcr, sigma=scr, intensity=True)

        g = lc2.trdata()[0]

        f = g.gauss2dat(Z)
        G = TrData(f, u)

        process = G.dat2gauss(L)
        return np.vstack((arange(len(process)), process)).T

#
#
# Check the result without reference to getrfc:
#        LCe = dat2lc(process)
# max(lc(:,2))
# max(LCe(:,2))
#
# clf
# plot(lc(:,1),lc(:,2)/max(lc(:,2)))
# hold on
# plot(LCe(:,1),LCe(:,2)/max(LCe(:,2)),'-.')
#        title('Relative crossing intensity')
#
# %% Plot made by the function funplot_4, JE 970707
# %param = [min(process(:,2)) max(process(:,2)) 100]
# %plot(lc(:,1),lc(:,2)/max(lc(:,2)))
# %hold on
# %plot(levels(param),mu/max(mu),'--')
# %hold off
# %title('Crossing intensity')
# %watstamp
#
# % Temporarily
# %funplot_4(lc,param,mu)
    def trdata(self, mean=None, sigma=None, **options):
        '''
        Estimate transformation, g, from observed crossing intensity, version2.

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
        >>> g0, g0emp = lc.trdata(plotflag=1)
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
        estimate = TransformEstimator(**options)
        return estimate._trdata_lc(self, mean, sigma)


def test_levelcrossings_extrapolate():
    import wafo.data
    x = wafo.data.sea()
    ts = mat2timeseries(x)

    tp = ts.turning_points()
    mm = tp.cycle_pairs()
    lc = mm.level_crossings()

    s = x[:, 1].std()
    lc_gpd = lc.extrapolate(-2 * s, 2 * s, dist='rayleigh')  # @UnusedVariable


class CyclePairs(PlotData):
    '''
    Container class for Cycle Pairs data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D

    Examples
    --------
    >>> import wafo.data
    >>> import wafo.objects as wo
    >>> x = wafo.data.sea()
    >>> ts = wo.mat2timeseries(x)

    >>> tp = ts.turning_points()
    >>> mm = tp.cycle_pairs()
    >>> h1 = mm.plot(marker='x')
    '''

    def __init__(self, *args, **kwds):
        self.kind = kwds.pop('kind', 'min2max')
        self.sigma = kwds.pop('sigma', None)
        self.mean = kwds.pop('mean', None)
        self.time = kwds.pop('time', 1)

        options = dict(title=self.kind + ' cycle pairs',
                       xlab='min', ylab='max',
                       plot_args=['b.'])
        options.update(**kwds)
        super(CyclePairs, self).__init__(*args, **options)

    def amplitudes(self):
        return (self.data - self.args) / 2.

    def damage(self, beta, K=1):
        """
        Calculates the total Palmgren-Miner damage of cycle pairs.

        Parameters
        ----------
        beta : array-like, size m
            Beta-values, material parameter.
        K : scalar, optional
            K-value, material parameter.

        Returns
        -------
        D : ndarray, size m
            Damage.

        Notes
        -----
        The damage is calculated according to
           D[i] = sum ( K * a**beta[i] ),  with  a = (max-min)/2

        Examples
        --------
        >>> import wafo
        >>> from matplotlib import pyplot as plt
        >>> ts = wafo.objects.mat2timeseries(wafo.data.sea())
        >>> tp = ts.turning_points()
        >>> mm = tp.cycle_pairs()
        >>> h = mm.plot(marker='.')
        >>> bv = range(3,9)
        >>> D = mm.damage(beta=bv)
        >>> D
        array([ 138.5238799 ,  117.56050788,  108.99265423,  107.86681126,
                112.3791076 ,  122.08375071])
        >>> h = plt.plot(bv,D,'x-')

        See also
        --------
        SurvivalCycleCount
        """
        amp = abs(self.amplitudes())
        return atleast_1d([K * np.sum(amp ** betai) for betai in beta])

    def get_minima_and_maxima(self):
        index, = nonzero(self.args <= self.data)
        if index.size == 0:
            index, = nonzero(self.args >= self.data)
            M = self.args[index]
            m = self.data[index]
        else:
            m = self.args[index]
            M = self.data[index]
        return m, M

    def level_crossings(self, kind='uM', intensity=False):
        """ Return level crossing spectrum from a cycle count.

        Parameters
        ----------
        kind : int or string
            defining crossing type, options are
            0,'u'  : only upcrossings.
            1,'uM' : upcrossings and maxima (default).
            2,'umM': upcrossings, minima, and maxima.
            3,'um' : upcrossings and minima.
        intensity : bool
            True if level crossing intensity spectrum
            False if level crossing count spectrum
        Return
        ------
        lc : level crossing object
            with levels and number of upcrossings.

        Calculates the number of upcrossings from a cycle pairs, e.g.
        min2Max cycles or rainflow cycles.

        Example:
        --------
        >>> import wafo
        >>> ts = wafo.objects.mat2timeseries(wafo.data.sea())
        >>> tp = ts.turning_points()
        >>> mm = tp.cycle_pairs()
        >>> h = mm.plot(marker='.')
        >>> lc = mm.level_crossings()
        >>> h2 = lc.plot()

        See also
        --------
        TurningPoints
        LevelCrossings
        """

        if isinstance(kind, str):
            t = dict(u=0, uM=1, umM=2, um=3)
            defnr = t.get(kind, 1)
        else:
            defnr = kind

        if ((defnr < 0) or (defnr > 3)):
            raise ValueError('kind must be one of (1,2,3,4).')

        m, M = self.get_minima_and_maxima()

        ncc = len(m)
        minima = vstack((m, ones(ncc), zeros(ncc), ones(ncc)))
        maxima = vstack((M, -ones(ncc), ones(ncc), zeros(ncc)))

        extremes = hstack((maxima, minima))
        index = extremes[0].argsort()
        extremes = extremes[:, index]

        ii = 0
        n = extremes.shape[1]
        extr = zeros((4, n))
        extr[:, 0] = extremes[:, 0]
        for i in range(1, n):
            if extremes[0, i] == extr[0, ii]:
                extr[1:4, ii] = extr[1:4, ii] + extremes[1:4, i]
            else:
                ii += 1
                extr[:, ii] = extremes[:, i]

        nx = extr[0].argmax() + 1
        levels = extr[0, 0:nx]
        if defnr == 2:  # This are upcrossings + maxima
            dcount = cumsum(extr[1, 0:nx]) + extr[2, 0:nx] - extr[3, 0:nx]
        elif defnr == 4:  # This are upcrossings + minima
            dcount = cumsum(extr[1, 0:nx])
            dcount[nx - 1] = dcount[nx - 2]
        elif defnr == 1:  # This are only upcrossings
            dcount = cumsum(extr[1, 0:nx]) - extr[3, 0:nx]
        elif defnr == 3:  # This are upcrossings + minima + maxima
            dcount = cumsum(extr[1, 0:nx]) + extr[2, 0:nx]
        ylab = 'Count'
        if intensity:
            dcount = dcount / self.time
            ylab = 'Intensity [count/sec]'
        return LevelCrossings(dcount, levels, mean=self.mean, sigma=self.sigma,
                              ylab=ylab, intensity=intensity)


class TurningPoints(PlotData):

    '''
    Container class for Turning Points data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D

      Examples
    --------
    >>> import wafo.data
    >>> import wafo.objects as wo
    >>> x = wafo.data.sea()
    >>> ts = wo.mat2timeseries(x)

    >>> tp = ts.turning_points()
    >>> h1 = tp.plot(marker='x')
    '''

    def __init__(self, *args, **kwds):
        self.name_ = kwds.pop('name', 'WAFO TurningPoints Object')
        self.sigma = kwds.pop('sigma', None)
        self.mean = kwds.pop('mean', None)

        options = dict(title='Turning points')
        options.update(**kwds)
        super(TurningPoints, self).__init__(*args, **options)

        if not any(self.args):
            n = len(self.data)
            self.args = range(0, n)
        else:
            self.args = ravel(self.args)
        self.data = ravel(self.data)

    def rainflow_filter(self, h=0.0, method='clib'):
        '''
        Return rainflow filtered turning points (tp).

        Parameters
        ----------
        h  : scalar
            a threshold
             if  h<=0, then  tp  is a sequence of turning points (default)
             if  h>0, then all rainflow cycles with height smaller than
                      h  are removed.

        Returns
        -------
        tp : TurningPoints object
            with times and turning points.

        Example:
        >>> import wafo.data
        >>> x = wafo.data.sea()
        >>> x1 = x[:200,:]
        >>> ts1 = mat2timeseries(x1)
        >>> tp = ts1.turning_points(wavetype='Mw')
        >>> tph = tp.rainflow_filter(h=0.3)
        >>> hs = ts1.plot()
        >>> hp = tp.plot('ro')
        >>> hph = tph.plot('k.')

        See also
        ---------
        findcross,
        findrfc
        findtp
        '''
        ind = findrfc(self.data, max(h, 0.0), method)
        try:
            t = self.args[ind]
        except:
            t = ind
        mean = self.mean
        sigma = self.sigma
        return TurningPoints(self.data[ind], t, mean=mean, sigma=sigma)

    def cycle_pairs(self, h=0, kind='min2max', method='clib'):
        """ Return min2Max or Max2min cycle pairs from turning points

        Parameters
        ----------
        kind : string
            type of cycles to return options are 'min2max' or 'max2min'
        method : string
            specify which library to use
            'clib' for wafo's c_library
            'None' for wafo's Python functions

        Return
        ------
        mm : cycles object
            with min2Max or Max2min cycle pairs.

        Example
        -------
        >>> import wafo
        >>> x = wafo.data.sea()
        >>> ts = wafo.objects.mat2timeseries(x)
        >>> tp = ts.turning_points()
        >>> mM = tp.cycle_pairs()
        >>> h = mM.plot(marker='x')


        See also
        --------
        TurningPoints
        SurvivalCycleCount
        """

        if h > 0:
            ind = findrfc(self.data, h, method=method)
            data = self.data[ind]
        else:
            data = self.data
        if data[0] > data[1]:
            im = 1
            iM = 0
        else:
            im = 0
            iM = 1

        # Extract min-max and max-min cycle pairs
        if kind.lower().startswith('min2max'):
            m = data[im:-1:2]
            M = data[im + 1::2]
        else:
            kind = 'max2min'
            M = data[iM:-1:2]
            m = data[iM + 1::2]

        time = self.args[-1] - self.args[0]

        return CyclePairs(M, m, kind=kind, mean=self.mean, sigma=self.sigma,
                          time=time)

    def cycle_astm(self):
        """
        Rainflow counted cycles according to Nieslony's ASTM implementation

        Parameters
        ----------

        Returns
        -------
        sig_rfc : array-like
            array of shape (n,3) with:
            sig_rfc[:,0] Cycles amplitude
            sig_rfc[:,1] Cycles mean value
            sig_rfc[:,2] Cycle type, half (=0.5) or full (=1.0)

        References
        ----------
        Adam Nieslony, "Determination of fragments of multiaxial service
        loading strongly influencing the fatigue of machine components",
        Mechanical Systems and Signal Processing 23, no. 8 (2009): 2712-2721.

        and is based on the following standard:
        ASTM E 1049-85 (Reapproved 1997), Standard practices for cycle counting
        in fatigue analysis, in: Annual Book of ASTM Standards,
        vol. 03.01, ASTM, Philadelphia, 1999, pp. 710-718.

        Copyright (c) 1999-2002 by Adam Nieslony
        Ported to Python by David Verelst

        Example
        -------
        >>> import wafo
        >>> x = wafo.data.sea()
        >>> sig_ts = wafo.objects.mat2timeseries(x)
        >>> sig_tp = sig_ts.turning_points(h=0, wavetype='astm')
        >>> sig_cp = sig_tp.cycle_astm()
        """

        # output of Nieslony's algorithm is organised differently with
        # respect to wafo's approach
        # TODO: integrate ASTM method into the CyclyPairs class?
        return findrfc_astm(self.data)


def mat2timeseries(x):
    """
    Convert 2D arrays to TimeSeries object
        assuming 1st column is time and the remaining columns contain data.
    """
    return TimeSeries(x[:, 1::], x[:, 0].ravel())


class TimeSeries(PlotData):
    '''
    Container class for 1D TimeSeries data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D, list of vectors for 2D, 3D, ...

    sensortypes : list of integers or strings
        sensor type for time series (default ['n']    : Surface elevation)
        see sensortype for more options
    position : vector of size 3
        instrument position relative to the coordinate system

    Examples
    --------
    >>> import wafo.data
    >>> import wafo.objects as wo
    >>> x = wafo.data.sea()
    >>> ts = wo.mat2timeseries(x)
    >>> rf = ts.tocovdata(lag=150)
    >>> h = rf.plot()

    >>> S = ts.tospecdata()
    >>> tp = ts.turning_points()
    >>> mm = tp.cycle_pairs()
    >>> h1 = mm.plot(marker='x')

    >>> lc = mm.level_crossings()
    >>> h2 = lc.plot()
    '''

    def __init__(self, *args, **kwds):
        self.name_ = kwds.pop('name', 'WAFO TimeSeries Object')
        self.sensortypes = kwds.pop('sensortypes', ['n', ])
        self.position = kwds.pop('position', [zeros(3), ])

        super(TimeSeries, self).__init__(*args, **kwds)

        if not any(self.args):
            n = len(self.data)
            self.args = range(0, n)

    def sampling_period(self):
        '''
        Returns sampling interval

        Returns
        -------
        dt : scalar
            sampling interval, unit:
            [s] if lagtype=='t'
            [m] otherwise

        See also
        '''
        t_vec = self.args
        dt1 = t_vec[1] - t_vec[0]
        n = len(t_vec) - 1
        t = t_vec[-1] - t_vec[0]
        dt = t / n
        if abs(dt - dt1) > 1e-10:
            warnings.warn('Data is not uniformly sampled!')
        return dt

    def tocovdata(self, lag=None, tr=None, detrend=detrend_mean,
                  window='boxcar', flag='biased', norm=False, dt=None):
        '''
        Return auto covariance function from data.

        Parameters
        ----------
        lag : scalar, int
            maximum time-lag for which the ACF is estimated. (Default lag=n-1)
        flag : string, 'biased' or 'unbiased'
            If 'unbiased' scales the raw correlation by 1/(n-abs(k)),
            where k is the index into the result, otherwise scales the raw
            cross-correlation by 1/n. (default)
        norm : bool
            True if normalize output to one
        dt : scalar
            time-step between data points (default see sampling_period).

        Return
        -------
        R : CovData1D object
            with attributes:
            data : ACF vector length L+1
            args : time lags  length L+1
            sigma : estimated large lag standard deviation of the estimate
                     assuming x is a Gaussian process:
                     if R(k)=0 for all lags k>q then an approximation
                     of the variance for large samples due to Bartlett
                     var(R(k))=1/N*(R(0)^2+2*R(1)^2+2*R(2)^2+ ..+2*R(q)^2)
                     for  k>q and where  N=length(x). Special case is
                     white noise where it equals R(0)^2/N for k>0
            norm : bool
                If false indicating that R is not normalized

         Example:
         --------
         >>> import wafo.data
         >>> import wafo.objects as wo
         >>> x = wafo.data.sea()
         >>> ts = wo.mat2timeseries(x)
         >>> acf = ts.tocovdata(150)
         >>> h = acf.plot()
        '''
        estimate_cov = _wafocov_estimation.CovarianceEstimator(
            lag=lag, tr=tr, detrend=detrend, window=window, flag=flag,
            norm=norm, dt=dt)
        return estimate_cov(self)

    def _specdata(self, L=None, tr=None, method='cov', detrend=detrend_mean,
                  window='parzen', noverlap=0, pad_to=None):
        """
        Obsolete: Delete?
        Return power spectral density by Welches average periodogram method.

        Parameters
        ----------
        NFFT : int, scalar
            if len(data) < NFFT, it will be zero padded to `NFFT`
            before estimation. Must be even; a power 2 is most efficient.
        detrend : function
        window : vector of length NFFT or function
            To create window vectors see numpy.blackman, numpy.hamming,
            numpy.bartlett, scipy.signal, scipy.signal.get_window etc.
        noverlap : scalar int
             gives the length of the overlap between segments.

        Returns
        -------
        S : SpecData1D
            Power Spectral Density

        Notes
        -----
        The data vector is divided into NFFT length segments.  Each segment
        is detrended by function detrend and windowed by function window.
        noverlap gives the length of the overlap between segments.  The
        absolute(fft(segment))**2 of each segment are averaged to compute Pxx,
        with a scaling to correct for power loss due to windowing.

        Reference
        ---------
        Bendat & Piersol (1986) Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons
        """
        dt = self.sampling_period()
        yy = self.data.ravel() if tr is None else tr.dat2gauss(
            self.data.ravel())
        yy = detrend(yy) if hasattr(detrend, '__call__') else yy

        S, f = psd(yy, Fs=1. / dt, NFFT=L, detrend=detrend, window=window,
                   noverlap=noverlap, pad_to=pad_to, scale_by_freq=True)
        fact = 2.0 * pi
        w = fact * f
        return _wafospec.SpecData1D(S / fact, w)

    def _get_bandwidth_and_dof(self, wname, n, L, dt):
        '''Returns bandwidth (rad/sec) and degrees of freedom
            used in chi^2 distribution
        '''
        Be = v = None
        if isinstance(wname, tuple):
            wname = wname[0]
        if wname == 'parzen':
            v = int(3.71 * n / L)
            Be = 2 * pi * 1.33 / (L * dt)
        elif wname == 'hanning':
            v = int(2.67 * n / L)
            Be = 2 * pi / (L * dt)
        elif wname == 'bartlett':
            v = int(3 * n / L)
            Be = 2 * pi * 1.33 / (L * dt)
        return Be, v

    def tospecdata(self, L=None, tr=None, method='cov', detrend=detrend_mean,
                   window='parzen', noverlap=0, ftype='w', alpha=None):
        '''
        Estimate one-sided spectral density from data.

        Parameters
        ----------
        L : scalar integer
            maximum lag size of the window function. As L decreases the
            estimate becomes smoother and Bw increases. If we want to resolve
            peaks in S which is Bf (Hz or rad/sec) apart then Bw < Bf. If no
            value is given the lag size is set to be the lag where the auto
            correlation is less than 2 standard deviations. (maximum 300)
        tr : transformation object
            the transformation assuming that x is a sample of a transformed
            Gaussian process. If g is None then x  is a sample of a Gaussian
            process (Default)
        method : string
            defining estimation method. Options are
            'cov' :  Frequency smoothing using the window function
                    on the estimated autocovariance function.  (default)
            'psd' : Welch's averaged periodogram method with no overlapping
                batches
        detrend : function
            defining detrending performed on the signal before estimation.
            (default detrend_mean)
        window : vector of length NFFT or function
            To create window vectors see numpy.blackman, numpy.hamming,
            numpy.bartlett, scipy.signal, scipy.signal.get_window etc.
        noverlap : scalar int
             gives the length of the overlap between segments.
        ftype : character
            defining frequency type: 'w' or 'f'  (default 'w')

        Returns
        ---------
        spec : SpecData1D  object

        Example
        -------
        x = load('sea.dat');
        S = dat2spec(x);
        specplot(S)

        See also
        --------
        dat2tr, dat2cov

        References:
        -----------
        Georg Lindgren and Holger Rootzen (1986)
        "Stationara stokastiska processer",  pp 173--176.

        Gareth Janacek and Louise Swift (1993)
        "TIME SERIES forecasting, simulation, applications",
        pp 75--76 and 261--268

        Emanuel Parzen (1962),
        "Stochastic Processes", HOLDEN-DAY,
        pp 66--103
        '''

        nugget = 1e-12
        rate = 2  # interpolationrate for frequency
        dt = self.sampling_period()

        yy = self.data.ravel()
        if not (tr is None):
            yy = tr.dat2gauss(yy)
        yy = detrend(yy) if hasattr(detrend, '__call__') else yy
        n = len(yy)

        estimate_L = L is None
        if method == 'cov' or estimate_L:
            tsy = TimeSeries(yy, self.args)
            R = tsy.tocovdata(lag=L, window=window)
            L = len(R.data) - 1
            if method == 'cov':
                # add a nugget effect to ensure that round off errors
                # do not result in negative spectral estimates
                spec = R.tospecdata(rate=rate, nugget=nugget)
        L = min(L, n - 1)
        if method == 'psd':
            nfft = 2 ** nextpow2(L)
            pad_to = rate * nfft  # Interpolate the spectrum with rate
            f, S = welch(yy, fs=1.0 / dt, window=window, nperseg=nfft,
                         noverlap=noverlap, nfft=pad_to, detrend=detrend,
                         return_onesided=True, scaling='density', axis=-1)
#             S, f = psd(yy, Fs=1. / dt, NFFT=nfft, detrend=detrend,
#                        window=win, noverlap=noverlap, pad_to=pad_to,
#                        scale_by_freq=True)
            fact = 2.0 * pi
            w = fact * f
            spec = _wafospec.SpecData1D(S / fact, w)
        elif method == 'cov':
            pass
        else:
            raise ValueError('Unknown method (%s)' % method)

        Be, v = self._get_bandwidth_and_dof(window, n, L, dt)
        spec.Bw = Be
        if ftype == 'f':
            spec.Bw = Be / (2 * pi)  # bandwidth in Hz

        if alpha is not None:
            # Confidence interval constants
            spec.CI = [v / _invchi2(1 - alpha / 2, v),
                       v / _invchi2(alpha / 2, v)]

        spec.tr = tr
        spec.L = L
        spec.norm = False
        spec.note = 'method=%s' % method
        return spec

    def trdata(self, method='nonlinear', **options):
        '''
        Estimate transformation, g, from data.

        Parameters
        ----------
        method : string defining transform based on:
            'nonlinear' : smoothed crossing intensity (default)
            'mnonlinear': smoothed marginal distribution
            'hermite'   : cubic Hermite polynomial
            'ochi'      : exponential function
            'linear'    : identity.

        options : keyword with the following fields:
        csm, gsm : real scalars
            defines the smoothing of the logarithm of crossing intensity and
            the transformation g, respectively. Valid values must be
                0<=csm,gsm<=1. (default csm=0.9, gsm=0.05)
            Smaller values gives smoother functions.
        param : vector (default see lc2tr)
            which defines the region of variation of the data x.
         plotflag : int
            0 no plotting (Default)
            1 plots empirical and smoothed g(u) and the theoretical for a
                Gaussian model.
            2 monitor the development of the estimation
        linextrap: int
            0 use a regular smoothing spline
            1 use a smoothing spline with a constraint on the ends to ensure
                linear extrapolation outside the range of the data. (default)
        gvar: real scalar
            Variances for the empirical transformation, g. (default  1)
        ne - Number of extremes (maxima & minima) to remove from the
                    estimation of the transformation. This makes the
                    estimation more robust against outliers. (default 7)
              ntr - Maximum length of empirical crossing intensity or CDF.
                    The empirical crossing intensity or CDF is interpolated
                    linearly  before smoothing if their lengths exceeds Ntr.
                    A reasonable NTR will significantly speed up the
                    estimation for long time series without loosing any
                    accuracy. NTR should be chosen greater than
                    PARAM(3). (default 1000)

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
        If x is likely to cross levels higher than 5 standard deviations
        then the vector param has to be modified.  For example if x is
        unlikely to cross a level of 7 standard deviations one can use
        PARAM=[-7 7 513].

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
        >>> g0, g0emp = ts.trdata(plotflag=1)
        >>> g1, g1emp = ts.trdata(method='mnonlinear', gvar=0.5 )
        >>> g2, g2emp = ts.trdata(method='nonlinear', gvar=[3.5, 0.5, 3.5])
        >>> 100 < S.tr.dist2gauss()*100 < 200
        True
        >>> 2000 < g0emp.dist2gauss() < 4000
        True
        >>> 80 < g0.dist2gauss()*100 < 150
        True
        >>> 50 < g1.dist2gauss()*100 < 100
        True
        >>> 70 < g2.dist2gauss()*100 < 140
        True

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
        estimate = TransformEstimator(method=method, **options)
        return estimate.trdata(self)

    def turning_points(self, h=0.0, wavetype=None):
        '''
        Return turning points (tp) from data, optionally rainflowfiltered.

        Parameters
        ----------
        h  : scalar
            a threshold
             if  h<=0, then  tp  is a sequence of turning points (default)
             if  h>0, then all rainflow cycles with height smaller than
                      h  are removed.
        wavetype : string
            defines the type of wave. Possible options are
            'astm' 'mw' 'Mw' or 'none'.
            If None all rainflow filtered min and max
            will be returned, otherwise only the rainflow filtered
            min and max, which define a wave according to the
            wave definition, will be returned.
            'astm' forces to have the first data point of the load history as
            the first turning point. To be used in combination with
            TurningPoints.cycle_astm()

        Returns
        -------
        tp : TurningPoints object
            with times and turning points.

        Example:
        >>> import wafo.data
        >>> x = wafo.data.sea()
        >>> x1 = x[:200,:]
        >>> ts1 = mat2timeseries(x1)
        >>> tp = ts1.turning_points(wavetype='Mw')
        >>> tph = ts1.turning_points(h=0.3,wavetype='Mw')
        >>> hs = ts1.plot()
        >>> hp = tp.plot('ro')
        >>> hph = tph.plot('k.')

        See also
        ---------
        findcross,
        findrfc
        findtp
        '''
        ind = findtp(self.data, max(h, 0.0), wavetype)
        try:
            t = self.args[ind]
        except:
            t = ind
        mean = self.data.mean()
        sigma = self.data.std()
        return TurningPoints(self.data[ind], t, mean=mean, sigma=sigma)

    def trough_crest(self, v=None, wavetype=None):
        """
        Return trough and crest turning points

        Parameters
        -----------
        v : scalar
            reference level (default  v = mean of x).

        wavetype : string
            defines the type of wave. Possible options are
            'dw', 'uw', 'tw', 'cw' or None.
            If None indices to all troughs and crests will be returned,
            otherwise only the paired ones will be returned
            according to the wavedefinition.

        Returns
        --------
        tc : TurningPoints object
            with trough and crest turningpoints
        """
        ind = findtc(self.data, v, wavetype)[0]
        try:
            t = self.args[ind]
        except:
            t = ind
        mean = self.data.mean()
        sigma = self.data.std()
        return TurningPoints(self.data[ind], t, mean=mean, sigma=sigma)

    def wave_parameters(self, rate=1):
        '''
        Returns several wave parameters from data.

        Parameters
        ----------
        rate : scalar integer
            interpolation rate. Interpolates with spline if greater than one.

        Returns
        -------
        parameters : dict
            wave parameters such as
            Ac, At : Crest and trough amplitude, respectively
            Tcf, Tcb : Crest front and crest (rear) back period, respectively
            Hu, Hd : zero-up- and down-crossing wave height, respectively.
            Tu, Td : zero-up- and down-crossing wave period, respectively.

        The definition of g, Ac,At, Tcf, etc. are given in gravity and
        wafo.definitions.

        Example
        -------
        >>> import wafo.data as wd
        >>> import wafo.objects as wo
        >>> x = wd.sea()
        >>> ts = wo.mat2timeseries(x)
        >>> wp = ts.wave_parameters()
        >>> for name in ['Ac', 'At', 'Hu', 'Hd', 'Tu', 'Td', 'Tcf', 'Tcb']:
        ...    print('%s' % name, wp[name][:2])
        ('Ac', array([ 0.25950546,  0.34950546]))
        ('At', array([ 0.16049454,  0.43049454]))
        ('Hu', array([ 0.69,  0.86]))
        ('Hd', array([ 0.42,  0.78]))
        ('Tu', array([ 6.10295202,  3.36978685]))
        ('Td', array([ 3.84377468,  6.35707656]))
        ('Tcf', array([ 0.42656819,  0.57361617]))
        ('Tcb', array([ 0.93355982,  1.04063638]))

        >>> import pylab as plt
        >>> h = plt.plot(wp['Td'],wp['Hd'],'.')
        >>> h = plt.xlabel('Td [s]')
        >>> h = plt.ylabel('Hd [m]')


        See also
        --------
        wafo.definitions
        '''
        dT = self.sampling_period()
        if rate > 1:
            dT = dT / rate
            t0, tn = self.args[0], self.args[-1]
            n = len(self.args)
            ti = linspace(t0, tn, int(rate * n))
            xi = interp1d(self.args, self.data.ravel(), kind='cubic')(ti)
        else:
            ti, xi = self.args, self.data.ravel()

        tc_ind, z_ind = findtc(xi, v=0, kind='tw')
        tc_a = xi[tc_ind]
        tc_t = ti[tc_ind]
        Ac = tc_a[1::2]  # crest amplitude
        At = -tc_a[0::2]  # trough  amplitude
        Hu = Ac + At[1:]
        Hd = Ac + At[:-1]
        tu = ecross(ti, xi, z_ind[1::2], v=0)
        Tu = diff(tu)  # Period zero-upcrossing waves
        td = ecross(ti, xi, z_ind[::2], v=0)
        Td = diff(td)  # Period zero-downcrossing waves
        Tcf = tc_t[1::2] - tu[:-1]
        Tcf[(Tcf == 0)] = dT  # avoiding division by zero
        Tcb = td[1:] - tc_t[1::2]
        Tcb[(Tcb == 0)] = dT  # avoiding division by zero
        return dict(Ac=Ac, At=At, Hu=Hu, Hd=Hd, Tu=Tu, Td=Td, Tcf=Tcf, Tcb=Tcb)

    def wave_height_steepness(self, method=1, rate=1, g=None):
        '''
        Returns waveheights and steepnesses from data.

        Parameters
        ----------
        rate : scalar integer
            interpolation rate. Interpolates with spline if greater than one.

        method : scalar integer (default 1)
            0 max(Vcf, Vcb) and corresponding wave height Hd or Hu in H
            1 crest front (rise) speed (Vcf) in S and wave height Hd in H.
           -1 crest back (fall) speed (Vcb) in S and waveheight Hu in H.
            2 crest front steepness in S and the wave height Hd in H.
           -2 crest back steepness in S and the wave height Hu in H.
            3 total wave steepness in S and the wave height Hd in H
                for zero-downcrossing waves.
           -3 total wave steepness in S and the wave height Hu in H.
                for zero-upcrossing waves.
        Returns
        -------
        S, H = Steepness and the corresponding wave height according to method


        The parameters are calculated as follows:
          Crest front speed (velocity) = Vcf = Ac/Tcf
          Crest back speed  (velocity) = Vcb = Ac/Tcb
          Crest front steepness  =  2*pi*Ac./Td/Tcf/g
          Crest back steepness   =  2*pi*Ac./Tu/Tcb/g
          Total wave steepness (zero-downcrossing wave) =  2*pi*Hd./Td.^2/g
          Total wave steepness (zero-upcrossing wave)   =  2*pi*Hu./Tu.^2/g

        The definition of g, Ac,At, Tcf, etc. are given in gravity and
        wafo.definitions.

        Example
        -------
        >>> import wafo.data as wd
        >>> import wafo.objects as wo
        >>> x = wd.sea()
        >>> ts = wo.mat2timeseries(x)
        >>> for i in range(-3,4):
        ...     S, H = ts.wave_height_steepness(method=i)
        ...     print(S[:2],H[:2])
        (array([ 0.01186982,  0.04852534]), array([ 0.69,  0.86]))
        (array([ 0.02918363,  0.06385979]), array([ 0.69,  0.86]))
        (array([ 0.27797411,  0.33585743]), array([ 0.69,  0.86]))
        (array([ 0.60835634,  0.60930197]), array([ 0.42,  0.78]))
        (array([ 0.60835634,  0.60930197]), array([ 0.42,  0.78]))
        (array([ 0.10140867,  0.06141156]), array([ 0.42,  0.78]))
        (array([ 0.01821413,  0.01236672]), array([ 0.42,  0.78]))

        >>> import pylab as plt
        >>> h = plt.plot(S,H,'.')
        >>> h = plt.xlabel('S')
        >>> h = plt.ylabel('Hd [m]')

        See also
        --------
        wafo.definitions
        '''

        dT = self.sampling_period()
        if g is None:
            g = gravity()  # % acceleration of gravity

        if rate > 1:
            dT = dT / rate
            t0, tn = self.args[0], self.args[-1]
            n = len(self.args)
            ti = linspace(t0, tn, int(rate * n))
            xi = interp1d(self.args, self.data.ravel(), kind='cubic')(ti)

        else:
            ti, xi = self.args, self.data.ravel()

        tc_ind, z_ind = findtc(xi, v=0, kind='tw')
        tc_a = xi[tc_ind]
        tc_t = ti[tc_ind]
        Ac = tc_a[1::2]  # crest amplitude
        At = -tc_a[0::2]  # trough  amplitude

        if (0 <= method and method <= 2):
            # time between zero-upcrossing and  crest  [s]
            tu = ecross(ti, xi, z_ind[1:-1:2], v=0)
            Tcf = tc_t[1::2] - tu
            Tcf[(Tcf == 0)] = dT  # avoiding division by zero
        if (0 >= method and method >= -2):
            # time between  crest and zero-downcrossing [s]
            td = ecross(ti, xi, z_ind[2::2], v=0)
            Tcb = td - tc_t[1::2]
            Tcb[(Tcb == 0)] = dT
            # % avoiding division by zero

        if method == 0:
            # max(Vcf, Vcr) and the corresponding wave height Hd or Hu in H
            Hu = Ac + At[1:]
            Hd = Ac + At[:-1]
            T = np.where(Tcf < Tcb, Tcf, Tcb)
            S = Ac / T
            H = np.where(Tcf < Tcb, Hd, Hu)
        elif method == 1:  # extracting crest front velocity [m/s] and
            # Zero-downcrossing wave height [m]
            H = Ac + At[:-1]  # Hd
            S = Ac / Tcf
        elif method == -1:  # extracting crest rear velocity [m/s] and
            # Zero-upcrossing wave height [m]
            H = Ac + At[1:]  # Hu
            S = Ac / Tcb
        # crest front steepness in S and the wave height Hd in H.
        elif method == 2:
            H = Ac + At[:-1]  # Hd
            Td = diff(ecross(ti, xi, z_ind[::2], v=0))
            S = 2 * pi * Ac / Td / Tcf / g
        # crest back steepness in S and the wave height Hu in H.
        elif method == -2:
            H = Ac + At[1:]
            Tu = diff(ecross(ti, xi, z_ind[1::2], v=0))
            S = 2 * pi * Ac / Tu / Tcb / g
        elif method == 3:  # total steepness in S and the wave height Hd in H
            # for zero-doewncrossing waves.
            H = Ac + At[:-1]
            # Period zero-downcrossing waves
            Td = diff(ecross(ti, xi, z_ind[::2], v=0))
            S = 2 * pi * H / Td ** 2 / g
        # total steepness in S and the wave height Hu in H for
        elif method == -3:
            # zero-upcrossing waves.
            H = Ac + At[1:]
            # Period zero-upcrossing waves
            Tu = diff(ecross(ti, xi, z_ind[1::2], v=0))
            S = 2 * pi * H / Tu ** 2 / g

        return S, H

    @staticmethod
    def _default_index(x, vh, wdef, pdef):
        if pdef in ('m2m', 'm2M', 'M2m', 'M2M'):
            index = findtp(x, vh, wdef)
        elif pdef in ('u2u', 'u2d', 'd2u', 'd2d'):
            index = findcross(x, vh, wdef)
        elif pdef in ('t2t', 't2c', 'c2t', 'c2c'):
            index = findtc(x, vh, wdef)[0]
        elif pdef in ('d2t', 't2u', 'u2c', 'c2d', 'all'):
            index, v_ind = findtc(x, vh, wdef)
            # sorting crossings and tp in sequence
            index = sort(r_[index, v_ind])
        else:
            raise ValueError('Unknown pdef option!')
        return index

    def _get_start_index(self, pdef, down_crossing_or_max):
        if down_crossing_or_max:
            if pdef in ('d2t', 'M2m', 'c2t', 'd2u', 'M2M', 'c2c', 'd2d', 'all'):
                start = 1
            elif pdef in ('t2u', 'm2M', 't2c', 'u2d', 'm2m', 't2t', 'u2u'):
                start = 2
            elif pdef in ('u2c'):
                start = 3
            elif pdef in ('c2d'):
                start = 4
            else:
                raise ValueError('Unknown pdef option!')
            # else first is up-crossing or min
        elif pdef in ('all', 'u2c', 'm2M', 't2c', 'u2d', 'm2m', 't2t', 'u2u'):
            start = 0
        elif pdef in ('c2d', 'M2m', 'c2t', 'd2u', 'M2M', 'c2c', 'd2d'):
            start = 1
        elif pdef in ('d2t'):
            start = 2
        elif pdef in ('t2u'):
            start = 3
        else:
            raise ValueError('Unknown pdef option!')
        return start


    def _get_step(self, pdef):
    # determine the steps between wanted periods
        if pdef in ('d2t', 't2u', 'u2c', 'c2d'):
            step = 4
        elif pdef in ('all'):
            step = 1 # % secret option!
        else:
            step = 2
        return step

    def wave_periods(self, vh=None, pdef='d2d', wdef=None, index=None, rate=1):
        """
        Return sequence of wave periods/lengths from data.

        Parameters
        ----------
        vh : scalar
            reference level ( default v=mean(x(:,2)) ) or
            rainflow filtering height (default h=0)
        pdef : string
            defining type of waveperiod (wavelength) returned:
            Level v separated 't2c', 'c2t', 't2t' or 'c2c' -waveperiod.
            Level v 'd2d', 'u2u', 'd2u' or 'u2d' -waveperiod.
            Rain flow filtered (with height greater than h)
            'm2M', 'M2m', 'm2m' or 'M2M' -waveperiod.
            Explanation to the abbreviations:
            M=Max, m=min, d=down-crossing, u=up-crossing ,
            t=trough and c=crest.
            Thus 'd2d' means period between a down-crossing to the
            next down-crossing and 'u2c' means period between a
            u-crossing to the following crest.
        wdef : string
            defining type of wave. Possible options are
            'mw','Mw','dw', 'uw', 'tw', 'cw' or None.
            If wdef is None all troughs and crests will be used,
            otherwise only the troughs and crests which define a
            wave according to the wavedefinition are used.

        index : vector
            index sequence of one of the following :
            -level v-crossings (indices to "du" are required to
                calculate 'd2d', 'd2u', 'u2d' or 'u2u' waveperiods)
            -level v separated trough and crest turningpoints
                (indices to 'tc' are required to calculate
                't2t', 't2c', 'c2t' or 'c2c' waveperiods)
            -level v crossings and level v separated trough and
                crest turningpoints (indices to "dutc" are
                required to calculate t2u, u2c, c2d or d2t
                waveperiods)
            -rainflow filtered turningpoints with minimum rfc height h
               (indices to "mMtc" are required to calculate
               'm2m', 'm2M', 'M2m' or 'M2M' waveperiods)

        rate : scalar
            interpolation rate. If rate larger than one, then x is
            interpolated before extrating T

        Returns
        --------
        T : vector
            sequence of waveperiods (or wavelengths).
        index : vector
            of indices


        Example:
        --------
        Histogram of crest2crest waveperiods
        >>> import wafo.data as wd
        >>> import wafo.objects as wo
        >>> import pylab as plb
        >>> x = wd.sea()
        >>> ts = wo.mat2timeseries(x[0:400,:])
        >>> T, ix = ts.wave_periods(vh=0.0,pdef='c2c')
        >>> h = plb.hist(T)

        See also:
        --------
        findtp,
        findtc,
        findcross, perioddef
        """

# % This is a more flexible version than the dat2hwa or tp2wa routines.
# % There is a secret option: if pdef='all' the function returns
# % all the waveperiods 'd2t', 't2u', 'u2c' and 'c2d' in sequence.
# % It is up to the user to extract the right waveperiods.
# % If the first is a down-crossing then the first is a 'd2t' waveperiod.
# % If the first is a up-crossing then the first is a 'u2c' waveperiod.
# %
# %    Example:
# %        [T ind]=dat2wa(x,0,'all') %returns all waveperiods
# %        nn = length(T)
# %        % want to extract all t2u waveperiods
# %        if x(ind(1),2)>0 % if first is down-crossing
# %            Tt2u=T(2:4:nn)
# %        else         % first is up-crossing
# %            Tt2u=T(4:4:nn)
# %        end

        if rate > 1:  # % interpolate with spline
            n = ceil(self.data.size * rate)
            ti = linspace(self.args[0], self.args[-1], n)
            x = stineman_interp(ti, self.args, self.data.ravel())
        else:
            x = self.data
            ti = self.args

        if vh is None:
            if pdef[0] in ('m', 'M'):
                vh = 0
                print('   The minimum rfc height, h,  is set to: %g' % vh)
            else:
                vh = x.mean()
                print('   The level l is set to: %g' % vh)

        if index is None:
            index = self._default_index(x, vh, wdef, pdef)

        down_crossing_or_max = (x[index[0]] > x[index[1]])
        start = self._get_start_index(pdef, down_crossing_or_max)
        step = self._get_step(pdef)

        # determine the distance between min2min, t2t etc..
        if pdef in ('m2m', 't2t', 'u2u', 'M2M', 'c2c', 'd2d'):
            dist = 2
        else:
            dist = 1

        nn = len(index)
        # New call: (pab 28.06.2001)
        if pdef[0] in ('u', 'd'):
            t0 = ecross(ti, x, index[start:(nn - dist):step], vh)
        else:  # min, Max, trough, crest or all crossings wanted
            t0 = x[index[start:(nn - dist):step]]

        if pdef[2] in ('u', 'd'):
            t1 = ecross(ti, x, index[(start + dist):nn:step], vh)
        else:  # min, Max, trough, crest or all crossings wanted
            t1 = x[index[(start + dist):nn:step]]

        T = t1 - t0
        return T, index

    def reconstruct(self, inds=None, Nsim=20, L=None, def_='nonlinear',
                    **options):
        '''
        function [y,g,g2,test,tobs,mu1o, mu1oStd] = reconstruct(x,)
        RECONSTRUCT reconstruct the spurious/missing points of timeseries

         CALL: [y,g,g2,test,tobs,mu1o,mu1oStd]=
             reconstruct(x,inds,Nsim,L,def,options)

        Returns
        -------
        y   = reconstructed signal
        g,g2 = smoothed and empirical transformation, respectively
        test, tobs = test observator int(g(u)-u)^2 du and
                     int(g_new(u)-g_old(u))^2 du,
                     respectively, where int limits is given by param in lc2tr.
                     Test is a measure of departure from the Gaussian model for
                     the data. Tobs is a measure of the convergence of the
                     estimation of g.
        mu1o = expected surface elevation of the Gaussian model process.
        mu1o_std = standarddeviation of mu1o.

        Parameters
        ----------
        x : 2 column timeseries
            first column sampling times [sec]
            second column surface elevation [m]
        inds : integer array
            indices to spurious points of x
        Nsim = the maximum # of iterations before we stop

        L = lag size of the Parzen window function.
                     If no value is given the lag size is set to
                     be the lag where the auto correlation is less than
                     2 standard deviations. (maximum 200)
        def :
            'nonlinear' : transform from smoothed crossing intensity (default)
            'mnonlinear': transform from smoothed marginal distribution
            'linear'    : identity.
        options = options structure defining how the estimation of g is
                     done, see troptset.

         In order to reconstruct the data a transformed Gaussian random process
         is used for modelling and simulation of the missing/removed data
         conditioned on the other known observations.

         Estimates of standarddeviations of y is obtained by a call to tranproc
                Std = tranproc(mu1o+/-mu1oStd,fliplr(g));

         See also
         --------
         troptset, findoutliers, cov2csdat, dat2cov, dat2tr, detrendma

         Reference
         ---------
         Brodtkorb, P, Myrhaug, D, and Rue, H (2001)
         "Joint distribution of wave height and wave crest velocity from
         reconstructed data with application to ringing"
         Int. Journal of Offshore and Polar Engineering, Vol 11, No. 1,
         pp 23--32

         Brodtkorb, P, Myrhaug, D, and Rue, H (1999)
         "Joint distribution of wave height and wave crest velocity from
         reconstructed data
         in Proceedings of 9th ISOPE Conference, Vol III, pp 66-73
        '''

        opt = DotDict(chkder=True, plotflag=False, csm=0.9, gsm=.05,
                      param=(-5, 5, 513), delay=2, linextrap=True, ntr=10000,
                      ne=7, gvar=1)
        opt.update(options)

        xn = self.data.copy().ravel()
        n = len(xn)

        if n < 2:
            raise ValueError('The vector must have more than 2 elements!')

        param = opt.param
        plotflags = dict(none=0, off=0, final=1, iter=2)
        plotflag = plotflags.get(opt.plotflag, opt.plotflag)

        olddef = def_
        method = 'approx'
        ptime = opt.delay  # pause for ptime sec if plotflag=2

        expect1 = 1  # first reconstruction by expectation? 1=yes 0=no
        expect = 1   # reconstruct by expectation? 1=yes 0=no
        tol = 0.001  # absolute tolerance of e(g_new-g_old)

        cmvmax = 100 # if number of consecutive missing values (cmv) are longer they
                     # are not used in estimation of g, due to the fact that the
                     # conditional expectation approaches zero as the length to
                     # the closest known points increases, see below in the for loop
        dT = self.sampling_period()

        Lm = np.minimum([n, 200, int(200/dT)])  # Lagmax 200 seconds
        if L is not None:
            Lm = max(L, Lm)
        # Lma: size of the moving average window used for detrending the
        #     reconstructed signal
        Lma = 1500
        if inds is not None:
            xn[inds] = np.nan

        inds = isnan(xn)
        if not inds.any():
            raise ValueError('No spurious data given')

        endpos = np.diff(inds)
        strtpos = np.flatnonzero(endpos > 0)
        endpos = np.flatnonzero(endpos < 0)

        indg = np.flatnonzero(1-inds)  # indices to good points
        inds = np.flatnonzero(inds)  # indices to spurious points

        indNaN = []  # indices to points omitted in the covariance estimation
        indr = np.arange(n)  # indices to point used in the estimation of g

        # Finding more than cmvmax consecutive spurios points.
        # They will not be used in the estimation of g and are thus removed
        # from indr.

        if strtpos.size > 0 and (endpos.size == 0 or endpos[-1] < strtpos[-1]):
            if (n - strtpos[-1]) > cmvmax:
                indNaN = indr[strtpos[-1]+1:n]
                indr = indr[:strtpos[-1]+1]
            strtpos = strtpos[:-1]

        if endpos.size > 0 and (strtpos.size == 0 or endpos[0] < strtpos[0]):
            if endpos[0] > cmvmax:
                indNaN = np.hstack((indNaN, indr[:endpos[0]]))
                indr = indr[endpos[0]:]

            strtpos = strtpos-endpos[0]
            endpos = endpos-endpos[0]
            endpos = endpos[1:]

        for ix in range(len(strtpos)-1, -1, -1):
            if (endpos[ix]-strtpos[ix] > cmvmax):
                indNaN = np.hstack((indNaN, indr[strtpos[ix]+1:endpos[ix]]))
                # remove this when estimating the transform
                del indr[strtpos[ix]+1:endpos[ix]]

        if len(indr) < 0.1*n:
            raise ValueError('Not possible to reconstruct signal')

        if indNaN.any():
            indNaN = np.sort(indNaN)

        # initial reconstruction attempt
        # xn(indg,2) = detrendma(xn(indg,2),1500);
#         [g, test, cmax, irr, g2]  = dat2tr(xn(indg,:),def,opt);
#         xnt=xn;
#         xnt(indg,:)=dat2gaus(xn(indg,:),g);
#         xnt(inds,2)=NaN;
#         rwin=findrwin(xnt,Lm,L);
#         disp(['First reconstruction attempt,    e(g-u)=', num2str(test)] )
#         # old simcgauss
#         [samp ,mu1o, mu1oStd]  = cov2csdat(xnt(:,2),rwin,1,method,inds);
#         if expect1,# reconstruction by expectation
#           xnt(inds,2) =mu1o;
#         else
#           xnt(inds,2) =samp;
#         end
#         xn=gaus2dat(xnt,g);
#         xn(:,2)=detrendma(xn(:,2),Lma); # detrends the signal with a moving
#                                          # average of size Lma
#         g_old=g;
#
#         bias = mean(xn(:,2));
#         xn(:,2)=xn(:,2)-bias; # bias correction

#         if plotflag==2
#           clf
#           mind=1:min(1500,n);
#           waveplot(xn(mind,:),x(inds(mind),:), 6,1)
#           subplot(111)
#           pause(ptime)
#         end
#
#         test0=0;
#         for ix=1:Nsim,
#         #   if 0,#ix==2,
#         #     rwin=findrwin(xn,Lm,L);
#         #     xs=cov2sdat(rwin,[n 100 dT]);
#         #     [g0 test0 cmax irr g2]  = dat2tr(xs,def,opt);
#         #     [test0 ind0]=sort(test0);
#         #   end

#            if 1, #test>test0(end-5),
#              # 95# sure the data comes from a non-Gaussian process
#              def = olddef; #Non Gaussian process
#            else
#              def = 'linear'; # Gaussian process
#            end
#            # used for isope article
#            # indr =[1:27000 30000:39000];
#            # Too many consecutive missing values will influence the estimation of
#            # g. By default do not use consecutive missing values if there are more
#            # than cmvmax.
#
#            [g test cmax irr g2]  = dat2tr(xn(indr,:),def,opt);
#           if plotflag==2,
#             pause(ptime)
#           end
#
#           #tobs=sqrt((param(2)-param(1))/(param(3)-1)*sum((g_old(:,2)-g(:,2)).^2))
#           # new call
#           tobs=sqrt((param(2)-param(1))/(param(3)-1)....
#             *sum((g(:,2)-interp1(g_old(:,1)-bias, g_old(:,2),g(:,1),'spline')).^2));
#
#           if ix>1
#             if tol>tobs2 && tol>tobs,
#               break, #estimation of g converged break out of for loop
#             end
#           end
#
#           tobs2=tobs;
#
#           xnt=dat2gaus(xn,g);
#           if ~isempty(indNaN),    xnt(indNaN,2)=NaN;  end
#           rwin=findrwin(xnt,Lm,L);
#           disp(['Simulation nr: ', int2str(ix), ' of ' num2str(Nsim),'   e(g-g_old)=', num2str(tobs), ',  e(g-u)=', num2str(test)])
#           [samp ,mu1o, mu1oStd]  = cov2csdat(xnt(:,2),rwin,1,method,inds);
#
#           if expect,
#             xnt(inds,2) =mu1o;
#           else
#             xnt(inds,2) =samp;
#           end
#
#           xn=gaus2dat(xnt,g);
#           if ix<Nsim
#             bias=mean(xn(:,2));
#             xn(:,2) = (xn(:,2)-bias); # bias correction
#           end
#           g_old=g;# saving the last transform
#           if plotflag==2
#             waveplot(xn(mind,:),x(inds(mind),:),6,1,[])
#             subplot(111)
#             pause(ptime)
#           end
#         end # for loop
#
#         if 1, #test>test0(end-5)
#           xnt=dat2gaus(xn,g);
#           [samp ,mu1o, mu1oStd]  = cov2csdat(xnt(:,2),rwin,1,method,inds);
#           xnt(inds,2) =samp;
#           xn=gaus2dat(xnt,g);
#           bias=mean(xn(:,2));
#           xn(:,2) = (xn(:,2)-bias); # bias correction
#           g(:,1)=g(:,1)-bias;
#           g2(:,1)=g2(:,1)-bias;
#           gn=trangood(g);
#
#           #mu1o=mu1o-tranproc(bias,gn);
#           muUStd=tranproc(mu1o+2*mu1oStd,fliplr(gn));#
#           muLStd=tranproc(mu1o-2*mu1oStd,fliplr(gn));#
#         else
#           muLStd=mu1o-2*mu1oStd;
#           muUStd=mu1o+2*mu1oStd;
#         end
#
#         if  plotflag==2 && length(xn)<10000,
#           waveplot(xn,[xn(inds,1) muLStd ;xn(inds,1) muUStd ], 6,round(n/3000),[])
#           legend('reconstructed','2 stdev')
#           #axis([770 850 -1 1])
#           #axis([1300 1325 -1 1])
#         end
#         y=xn;
#         toc
#
#         return
#
#         function r=findrwin(xnt,Lm,L)
#           r=dat2cov(xnt,Lm);#computes  ACF
#           #finding where ACF is less than 2 st. deviations .
#           # in order to find a better L  value
#           if nargin<3||isempty(L)
#             L=find(abs(r.R)>2*r.stdev)+1;
#             if isempty(L), # pab added this check 09.10.2000
#               L = Lm;
#             else
#               L = min([floor(4/3*L(end)) Lm]);
#             end
#           end
#           win=parzen(2*L-1);
#           r.R(1:L)=win(L:2*L-1).*r.R(1:L);
#           r.R(L+1:end)=0;
#           return

    def plot_wave(self, sym1='k.', ts=None, sym2='k+', nfig=None, nsub=None,
                  sigma=None, vfact=3):
        '''
        Plots the surface elevation of timeseries.

        Parameters
        ----------
        sym1, sym2 : string
            plot symbol and color for data and ts, respectively
                      (see PLOT)  (default 'k.' and 'k+')
        ts : TimeSeries or TurningPoints object
            to overplot data. default zero-separated troughs and crests.
        nsub : scalar integer
            Number of subplots in each figure. By default nsub is such that
            there are about 20 mean down crossing waves in each subplot.
            If nfig is not given and nsub is larger than 6 then nsub is
            changed to nsub=min(6,ceil(nsub/nfig))
        nfig : scalar integer
            Number of figures. By default nfig=ceil(Nsub/6).
        sigma : real scalar
            standard deviation of data.
        vfact : real scalar
            how large in stdev the vertical scale should be (default 3)


        Example
        -------
        Plot x1 with red lines and mark troughs and crests with blue circles.
        >>> import wafo
        >>> x = wafo.data.sea()
        >>> ts150 = wafo.objects.mat2timeseries(x[:150,:])
        >>> h = ts150.plot_wave('r-', sym2='bo')

        See also
        --------
        findtc, plot
        '''

        nw = 20
        tn = self.args
        xn = self.data.ravel()
        indmiss = isnan(xn)  # indices to missing points
        indg = where(1 - indmiss)[0]
        if ts is None:
            tc_ix = findtc(xn[indg], 0, 'tw')[0]
            xn2 = xn[tc_ix]
            tn2 = tn[tc_ix]
        else:
            xn2 = ts.data
            tn2 = ts.args

        if sigma is None:
            sigma = xn[indg].std()

        if nsub is None:
            # about Nw mdc waves in each plot
            nsub = int(len(xn2) / (2 * nw)) + 1
        if nfig is None:
            nfig = int(ceil(nsub / 6))
            nsub = min(6, int(ceil(nsub / nfig)))

        n = len(xn)
        Ns = int(n / (nfig * nsub))
        ind = r_[0:Ns]
        if all(xn >= 0):
            vscale = [0, 2 * sigma * vfact]  # @UnusedVariable
        else:
            vscale = array([-1, 1]) * vfact * sigma  # @UnusedVariable

        XlblTxt = 'Time [sec]'
        dT = 1
        timespan = tn[ind[-1]] - tn[ind[0]]
        if abs(timespan) > 18000:  # more than 5 hours
            dT = 1 / (60 * 60)
            XlblTxt = 'Time (hours)'
        elif abs(timespan) > 300:  # more than 5 minutes
            dT = 1 / 60
            XlblTxt = 'Time (minutes)'

        if np.max(abs(xn[indg])) > 5 * sigma:
            XlblTxt = XlblTxt + ' (Spurious data since max > 5 std.)'

        plot = plotbackend.plot
        subplot = plotbackend.subplot
        figs = []
        for unused_iz in range(nfig):
            figs.append(plotbackend.figure())
            plotbackend.title('Surface elevation from mean water level (MWL).')
            for ix in range(nsub):
                if nsub > 1:
                    subplot(nsub, 1, ix)
                h_scale = array([tn[ind[0]], tn[ind[-1]]])
                ind2 = where((h_scale[0] <= tn2) & (tn2 <= h_scale[1]))[0]
                plot(tn[ind] * dT, xn[ind], sym1)
                if len(ind2) > 0:
                    plot(tn2[ind2] * dT, xn2[ind2], sym2)
                plot(h_scale * dT, [0, 0], 'k-')
                # plotbackend.axis([h_scale*dT, v_scale])
                for iy in [-2, 2]:
                    plot(h_scale * dT, iy * sigma * ones(2), ':')
                ind = ind + Ns
            plotbackend.xlabel(XlblTxt)

        return figs

    def plot_sp_wave(self, wave_idx_, *args, **kwds):
        """
        Plot specified wave(s) from timeseries

        Parameters
        ----------
        wave_idx : integer vector
            of indices to waves we want to plot, i.e., wave numbers.
        tz_idx : integer vector
            of indices to the beginning, middle and end of
            defining wave, i.e. for zero-downcrossing waves, indices to
            zerocrossings (default trough2trough wave)

        Examples
        --------
        Plot waves nr. 6,7,8 and waves nr. 12,13,...,17
        >>> import wafo
        >>> x = wafo.data.sea()
        >>> ts = wafo.objects.mat2timeseries(x[0:500,...])
        >>> h = ts.plot_sp_wave(np.r_[6:9,12:18])

        See also
        --------
        plot_wave, findtc
        """
        wave_idx = atleast_1d(wave_idx_).flatten()
        tz_idx = kwds.pop('tz_idx', None)
        if tz_idx is None:
            # finding trough to trough waves
            unused_tc_ind, tz_idx = findtc(self.data, 0, 'tw')

        dw = nonzero(abs(diff(wave_idx)) > 1)[0]
        Nsub = dw.size + 1
        Nwp = zeros(Nsub, dtype=int)
        if Nsub > 1:
            dw = dw + 1
            Nwp[Nsub - 1] = wave_idx[-1] - wave_idx[dw[-1]] + 1
            wave_idx[dw[-1] + 1:] = -2
            for ix in range(Nsub - 2, 1, -2):
                # of waves pr subplot
                Nwp[ix] = wave_idx[dw[ix] - 1] - wave_idx[dw[ix - 1]] + 1
                wave_idx[dw[ix - 1] + 1:dw[ix]] = -2

            Nwp[0] = wave_idx[dw[0] - 1] - wave_idx[0] + 1
            wave_idx[1:dw[0]] = -2
            wave_idx = wave_idx[wave_idx > -1]
        else:
            Nwp[0] = wave_idx[-1] - wave_idx[0] + 1

        Nsub = min(6, Nsub)
        Nfig = int(ceil(Nsub / 6))
        Nsub = min(6, int(ceil(Nsub / Nfig)))
        figs = []
        for unused_iy in range(Nfig):
            figs.append(plotbackend.figure())
            for ix in range(Nsub):
                plotbackend.subplot(Nsub, 1, mod(ix, Nsub) + 1)
                ind = r_[tz_idx[2 * wave_idx[ix] - 1]:tz_idx[
                    2 * wave_idx[ix] + 2 * Nwp[ix] - 1]]
                # indices to wave
                plotbackend.plot(self.args[ind], self.data[ind], *args, **kwds)
                plotbackend.hold('on')
                xi = [self.args[ind[0]], self.args[ind[-1]]]
                plotbackend.plot(xi, [0, 0])

                if Nwp[ix] == 1:
                    plotbackend.ylabel('Wave %d' % wave_idx[ix])
                else:
                    plotbackend.ylabel(
                        'Wave %d - %d' % (wave_idx[ix],
                                          wave_idx[ix] + Nwp[ix] - 1))
            plotbackend.xlabel('Time [sec]')
            # wafostamp
        return figs


def main():
    import wafo
    ts = wafo.objects.mat2timeseries(wafo.data.sea())
    _S = ts.tospecdata(method='psd')
    tp = ts.turning_points()
    mm = tp.cycle_pairs()
    lc = mm.level_crossings()
    lc.plot()
    T = ts.wave_periods(vh=0.0, pdef='c2c')  # @UnusedVariable

    # main()
    import wafo.spectrum.models as sm
    Sj = sm.Jonswap()
    S = Sj.tospecdata()

    R = S.tocovdata()
    x = R.sim(ns=1000, dt=0.2)  # @UnusedVariable
    S.characteristic(['hm0', 'tm02'])
    ns = 1000
    dt = .2
    x1 = S.sim(ns, dt=dt)

    ts = TimeSeries(x1[:, 1], x1[:, 0])
    tp = ts.turning_points(0.0)

    x = np.arange(-2, 2, 0.2)

    # Plot 2 objects in one call
    d2 = PlotData(np.sin(x), x, xlab='x', ylab='sin', title='sinus')

    d0 = d2.copy()
    d0.data = d0.data * 0.9
    d1 = d2.copy()
    d1.data = d1.data * 1.2
    d1.children = [d0]
    d2.children = [d1]

    d2.plot()
    print('Done')


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

if __name__ == '__main__':
    test_docstrings()
    # main()
    # test_levelcrossings_extrapolate()
