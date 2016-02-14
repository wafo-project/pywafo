'''
CovData1D
---------
data : Covariance function values. Size [ny nx nt], all singleton dim. removed.
args : Lag of first space dimension, length nx.
h : Water depth.
tr : Transformation function.
type : 'enc', 'rot' or 'none'.
v : Ship speed, if .type='enc'
phi : Rotation of coordinate system, e.g.  direction of ship
norm : Normalization flag, Logical 1 if autocorrelation, 0 if covariance.
Rx, ... ,Rtttt :  Obvious derivatives of .R.
note : Memorandum string.
date : Date and time of creation or change.
'''

from __future__ import division, absolute_import
import warnings
import numpy as np
from numpy import (zeros, ones, sqrt, inf, where, nan,
                   atleast_1d, hstack, r_, linspace, flatnonzero, size,
                   isnan, finfo, diag, ceil, random, pi)
from numpy.fft import fft
from numpy.random import randn
import scipy.interpolate as interpolate
from scipy.linalg import toeplitz, lstsq
from scipy import sparse
from pylab import stineman_interp

from ..containers import PlotData
from ..misc import sub_dict_select, nextpow2  # , JITImport
from .. import spectrum as _wafospec
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from scipy.sparse.base import issparse
from scipy.signal.windows import parzen
# _wafospec = JITImport('wafo.spectrum')

__all__ = ['CovData1D']


def _set_seed(iseed):
    if iseed is not None:
        try:
            random.set_state(iseed)
        except:
            random.seed(iseed)


def rndnormnd(mean, cov, cases=1):
    '''
    Random vectors from a multivariate Normal distribution

    Parameters
    ----------
    mean, cov : array-like
         mean and covariance, respectively.
    cases : scalar integer
        number of sample vectors

    Returns
    -------
    r : matrix of random numbers from the multivariate normal
        distribution with the given mean and covariance matrix.

    The covariance must be a symmetric, semi-positive definite matrix with
    shape equal to the size of the mean.

    Example
    -------
    >>> mu = [0, 5]
    >>> S = [[1 0.45], [0.45 0.25]]
    >>> r = rndnormnd(mu, S, 1)

    plot(r(:,1),r(:,2),'.')

    >>> d = 40
    >>> rho = 2 * np.random.rand(1,d)-1
    >>> mu = zeros(d)
    >>> S = (np.dot(rho.T, rho)-diag(rho.ravel()**2))+np.eye(d)
    >>> r = rndnormnd(mu, S, 100)

    See also
    --------
    np.random.multivariate_normal
    '''
    return np.random.multivariate_normal(mean, cov, cases)


class CovData1D(PlotData):

    """ Container class for 1D covariance data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D, list of vectors for 2D, 3D, ...

    type : string
        spectrum type, one of 'freq', 'k1d', 'enc' (default 'freq')
    lagtype : letter
        lag type, one of: 'x', 'y' or 't' (default 't')


    Examples
    --------
    >>> import numpy as np
    >>> import wafo.spectrum as sp
    >>> Sj = sp.models.Jonswap(Hm0=3,Tp=7)
    >>> w = np.linspace(0,4,256)
    >>> S = sp.SpecData1D(Sj(w),w) #Make spectrum object from numerical values

    See also
    --------
    PlotData
    CovData
    """

    def __init__(self, *args, **kwds):
        super(CovData1D, self).__init__(*args, **kwds)

        self.name = 'WAFO Covariance Object'
        self.type = 'time'
        self.lagtype = 't'
        self.h = inf
        self.tr = None
        self.phi = 0.
        self.v = 0.
        self.norm = 0
        somekeys = ['phi', 'name', 'h', 'tr', 'lagtype', 'v', 'type', 'norm']

        self.__dict__.update(sub_dict_select(kwds, somekeys))

        self.setlabels()

    def setlabels(self):
        ''' Set automatic title, x-,y- and z- labels

            based on type,
        '''

        N = len(self.type)
        if N == 0:
            raise ValueError(
                'Object does not appear to be initialized, it is empty!')

        labels = ['', 'ACF', '']

        if self.lagtype.startswith('t'):
            labels[0] = 'Lag [s]'
        else:
            labels[0] = 'Lag [m]'

        if self.norm:
            title = 'Auto Correlation Function '
            labels[0] = labels[0].split('[')[0]
        else:
            title = 'Auto Covariance Function '

        self.labels.title = title
        self.labels.xlab = labels[0]
        self.labels.ylab = labels[1]
        self.labels.zlab = labels[2]

    def tospecdata(self, rate=None, method='fft', nugget=0.0, trunc=1e-5,
                   fast=True):
        '''
        Computes spectral density from the auto covariance function

        Parameters
        ----------
        rate = scalar, int
            1,2,4,8...2^r, interpolation rate for f (default 1)
        method : string
            interpolation method 'stineman', 'linear', 'cubic', 'fft'
        nugget : scalar, real
            nugget effect to ensure that round off errors do not result in
            negative spectral estimates. Good choice might be 10^-12.
        trunc : scalar, real
            truncates all spectral values where S/max(S) < trunc
                      0 <= trunc <1   This is to ensure that high frequency
                      noise is not added to the spectrum.  (default 1e-5)
        fast : bool
             if True : zero-pad to obtain power of 2 length ACF (default)
             otherwise  no zero-padding of ACF, slower but more accurate.

        Returns
        --------
        S : SpecData1D object
            spectral density

         NB! This routine requires that the covariance is evenly spaced
             starting from zero lag. Currently only capable of 1D matrices.

        Example:
        >>> import wafo.spectrum.models as sm
        >>> import numpy as np
        >>> import scipy.signal as st
        >>> import pylab
        >>> L = 129
        >>> t = np.linspace(0,75,L)
        >>> R = np.zeros(L)
        >>> win = st.parzen(41)
        >>> R[0:21] = win[20:41]
        >>> R0 = CovData1D(R,t)
        >>> S0 = R0.tospecdata()

        >>> Sj = sm.Jonswap()
        >>> S = Sj.tospecdata()
        >>> R2 = S.tocovdata()
        >>> S1 = R2.tospecdata()
        >>> abs(S1.data-S.data).max()

        >>> S1.plot('r-')
        >>> S.plot('b:')
        >>> pylab.show()

        >>> all(abs(S1.data-S.data)<1e-4)

        See also
        --------
        spec2cov
        datastructures
        '''

        dt = self.sampling_period()
        # dt = time-step between data points.

        acf, unused_ti = atleast_1d(self.data, self.args)

        if self.lagtype in 't':
            spectype = 'freq'
            ftype = 'w'
        else:
            spectype = 'k1d'
            ftype = 'k'

        if rate is None:
            rate = 1  # interpolation rate
        else:
            rate = 2 ** nextpow2(rate)  # make sure rate is a power of 2

        # add a nugget effect to ensure that round off errors
        # do not result in negative spectral estimates
        acf[0] = acf[0] + nugget
        n = acf.size
        # embedding a circulant vector and Fourier transform

        nfft = 2 ** nextpow2(2 * n - 2) if fast else 2 * n - 2

        if method == 'fft':
            nfft *= rate

        nf = nfft / 2  # number of frequencies
        acf = r_[acf, zeros(nfft - 2 * n + 2), acf[n - 2:0:-1]]

        Rper = (fft(acf, nfft).real).clip(0)  # periodogram
        RperMax = Rper.max()
        Rper = where(Rper < trunc * RperMax, 0, Rper)

        S = abs(Rper[0:(nf + 1)]) * dt / pi
        w = linspace(0, pi / dt, nf + 1)
        So = _wafospec.SpecData1D(S, w, type=spectype, freqtype=ftype)
        So.tr = self.tr
        So.h = self.h
        So.norm = self.norm

        if method != 'fft' and rate > 1:
            So.args = linspace(0, pi / dt, nf * rate)
            if method == 'stineman':
                So.data = stineman_interp(So.args, w, S)
            else:
                intfun = interpolate.interp1d(w, S, kind=method)
                So.data = intfun(So.args)
            So.data = So.data.clip(0)  # clip negative values to 0
        return So

    def sampling_period(self):
        '''
        Returns sampling interval

        Returns
        ---------
        dt : scalar
            sampling interval, unit:
            [s] if lagtype=='t'
            [m] otherwise
        '''
        dt1 = self.args[1] - self.args[0]
        n = size(self.args) - 1
        t = self.args[-1] - self.args[0]
        dt = t / n
        if abs(dt - dt1) > 1e-10:
            warnings.warn('Data is not uniformly sampled!')
        return dt

    def _is_valid_acf(self):
        if self.data.argmax() != 0:
            raise ValueError('ACF does not have a maximum at zero lag')

    def sim(self, ns=None, cases=1, dt=None, iseed=None, derivative=False):
        '''
        Simulates a Gaussian process and its derivative from ACF

        Parameters
        ----------
        ns : scalar
            number of simulated points.  (default length(S)-1=n-1).
                     If ns>n-1 it is assummed that R(k)=0 for all k>n-1
        cases : scalar
            number of replicates (default=1)
        dt : scalar
            step in grid (default dt is defined by the Nyquist freq)
        iseed : int or state
            starting state/seed number for the random number generator
            (default none is set)
        derivative : bool
            if true : return derivative of simulated signal as well
            otherwise

        Returns
        -------
        xs    = a cases+1 column matrix  ( t,X1(t) X2(t) ...).
        xsder = a cases+1 column matrix  ( t,X1'(t) X2'(t) ...).

        Details
        -------
        Performs a fast and exact simulation of stationary zero mean
        Gaussian process through circulant embedding of the covariance matrix.

        If the ACF has a non-empty field .tr, then the transformation is
        applied to the simulated data, the result is a simulation of a
        transformed Gaussian process.

        Note: The simulation may give high frequency ripple when used with a
                small dt.

        Example:
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap()
        >>> S = Sj.tospecdata()   #Make spec
        >>> R = S.tocovdata()
        >>> x = R.sim(ns=1000,dt=0.2)

        See also
        --------
        spec2sdat, gaus2dat

        Reference
        -----------
        C.R Dietrich and G. N. Newsam (1997)
        "Fast and exact simulation of stationary
        Gaussian process through circulant embedding
        of the Covariance matrix"
        SIAM J. SCI. COMPT. Vol 18, No 4, pp. 1088-1107
        '''

        # TODO fix it, it does not work

        # Add a nugget effect to ensure that round off errors
        # do not result in negative spectral estimates
        nugget = 0  # 10**-12

        _set_seed(iseed)
        self._is_valid_acf()
        acf = self.data.ravel()
        n = acf.size
        acf.shape = (n, 1)

        dT = self.sampling_period()

        x = zeros((ns, cases + 1))

        if derivative:
            xder = x.copy()

        # add a nugget effect to ensure that round off errors
        # do not result in negative spectral estimates
        acf[0] = acf[0] + nugget

        # Fast and exact simulation of simulation of stationary
        # Gaussian process throug circulant embedding of the
        # Covariance matrix
        floatinfo = finfo(float)
        if (abs(acf[-1]) > floatinfo.eps):  # assuming acf(n+1)==0
            m2 = 2 * n - 1
            nfft = 2 ** nextpow2(max(m2, 2 * ns))
            acf = r_[acf, zeros((nfft - m2, 1)), acf[-1:0:-1, :]]
            # warnings,warn('I am now assuming that ACF(k)=0 for k>MAXLAG.')
        else:  # ACF(n)==0
            m2 = 2 * n - 2
            nfft = 2 ** nextpow2(max(m2, 2 * ns))
            acf = r_[acf, zeros((nfft - m2, 1)), acf[n - 1:1:-1, :]]

        # m2=2*n-2
        S = fft(acf, nfft, axis=0).real  # periodogram

        I = S.argmax()
        k = flatnonzero(S < 0)
        if k.size > 0:
            _msg = '''
                Not able to construct a nonnegative circulant vector from ACF.
                Apply parzen windowfunction to the ACF in order to avoid this.
                The returned result is now only an approximation.'''

            # truncating negative values to zero to ensure that
            # that this noise is not added to the simulated timeseries

            S[k] = 0.

            ix = flatnonzero(k > 2 * I)
            if ix.size > 0:
                # truncating all oscillating values above 2 times the peak
                # frequency to zero to ensure that
                # that high frequency noise is not added to
                # the simulated timeseries.
                ix0 = k[ix[0]]
                S[ix0:-ix0] = 0.0

        trunc = 1e-5
        maxS = S[I]
        k = flatnonzero(S[I:-I] < maxS * trunc)
        if k.size > 0:
            S[k + I] = 0.
            # truncating small values to zero to ensure that
            # that high frequency noise is not added to
            # the simulated timeseries

        cases1 = int(cases / 2)
        cases2 = int(ceil(cases / 2))
# Generate standard normal random numbers for the simulations

        # randn = np.random.randn
        epsi = randn(nfft, cases2) + 1j * randn(nfft, cases2)
        Ssqr = sqrt(S / (nfft))  # sqrt(S(wn)*dw )
        ephat = epsi * Ssqr  # [:,np.newaxis]
        y = fft(ephat, nfft, axis=0)
        x[:, 1:cases + 1] = hstack((y[2:ns + 2, 0:cases2].real,
                                    y[2:ns + 2, 0:cases1].imag))

        x[:, 0] = linspace(0, (ns - 1) * dT, ns)  # (0:dT:(dT*(np-1)))'

        if derivative:
            Ssqr = Ssqr * \
                r_[0:(nfft / 2 + 1), -(nfft / 2 - 1):0] * 2 * pi / nfft / dT
            ephat = epsi * Ssqr  # [:,newaxis]
            y = fft(ephat, nfft, axis=0)
            xder[:, 1:(cases + 1)] = hstack((y[2:ns + 2, 0:cases2].imag -
                                            y[2:ns + 2, 0:cases1].real))
            xder[:, 0] = x[:, 0]

        if self.tr is not None:
            print('   Transforming data.')
            g = self.tr
            if derivative:
                for ix in range(cases):
                    tmp = g.gauss2dat(x[:, ix + 1], xder[:, ix + 1])
                    x[:, ix + 1] = tmp[0]
                    xder[:, ix + 1] = tmp[1]
            else:
                for ix in range(cases):
                    x[:, ix + 1] = g.gauss2dat(x[:, ix + 1])

        if derivative:
            return x, xder
        else:
            return x

    def _get_lag_where_acf_is_almost_zero(self):
        acf = self.data.ravel()
        r0 = acf[0]
        n = len(acf)
        sigma = sqrt(r_[0, r0 ** 2,
                        r0 ** 2 + 2 * np.cumsum(acf[1:n - 1] ** 2)] / n)
        k = flatnonzero(np.abs(acf) > 0.1 * sigma)
        if k.size > 0:
            lag = min(k.max() + 3, n)
            return lag
        return n

    def _get_acf(self, smooth=False):
        self._is_valid_acf()
        acf = atleast_1d(self.data).ravel()
        n = self._get_lag_where_acf_is_almost_zero()
        if smooth:
            rwin = parzen(2 * n + 1)
            return acf[:n] * rwin[n:2 * n]
        else:
            return acf[:n]

    def _split_cov(self, sigma, i_known, i_unknown):
        '''
        Split covariance matrix between known/unknown observations

        Returns
        -------
        Soo  covariance between known observations
        S11 = covariance between unknown observations
        S1o = covariance between known and unknown obs
        '''
        Soo, So1 = sigma[i_known][:, i_known], sigma[i_known][:, i_unknown]
        S11 = sigma[i_unknown][:, i_unknown]
        return Soo, So1, S11

    def _update_window(self, idx, i_unknown, num_x, num_acf,
                       overlap, nw, num_restored):
        Nsig = len(idx)
        start_max = num_x - Nsig
        if (nw == 0) and (num_restored < len(i_unknown)):
            # move to the next missing data
            start_ix = min(i_unknown[num_restored + 1] - overlap, start_max)
        else:
            start_ix = min(idx[0] + num_acf, start_max)

        return idx + start_ix - idx[0]

    def simcond(self, xo, method='approx', i_unknown=None):
        """
        Simulate values conditionally on observed known values

        Parameters
        ----------
        x : vector
            timeseries including missing data.
            (missing data must be NaN if i_unknown is not given)
            Assumption: The covariance of x is equal to self and have the
            same sample period.
        method : string
            defining method used in the conditional simulation. Options are:
            'approximate': Condition only on the closest points. Quite fast
            'exact' : Exact simulation. Slow for large data sets, may not
                return any result due to near singularity of the covariance
                matrix.
        i_unknown : integers
            indices to spurious or missing data in x

        Returns
        -------
        sample : ndarray
            a random sample of the missing values conditioned on the observed
            data.
        mu, sigma : ndarray
            mean and standard deviation, respectively, of the missing values
            conditioned on the observed data.

        Notes
        -----
        SIMCOND generates the missing values from x conditioned on the observed
        values assuming x comes from a multivariate Gaussian distribution
        with zero expectation and Auto Covariance function R.

        See also
        --------
        CovData1D.sim
        TimeSeries.reconstruct,
        rndnormnd

        Reference
        ---------
        Brodtkorb, P, Myrhaug, D, and Rue, H (2001)
        "Joint distribution of wave height and wave crest velocity from
        reconstructed data with application to ringing"
        Int. Journal of Offshore and Polar Engineering, Vol 11, No. 1,
        pp 23--32

        Brodtkorb, P, Myrhaug, D, and Rue, H (1999)
        "Joint distribution of wave height and wave crest velocity from
        reconstructed data"
        in Proceedings of 9th ISOPE Conference, Vol III, pp 66-73
        """
        x = atleast_1d(xo).ravel()
        acf = self._get_acf()

        num_x = len(x)
        num_acf = len(acf)

        if i_unknown is not None:
            x[i_unknown] = nan
        i_unknown = flatnonzero(isnan(x))
        num_unknown = len(i_unknown)

        mu1o = zeros((num_unknown,))
        mu1o_std = zeros((num_unknown,))
        sample = zeros((num_unknown,))
        if num_unknown == 0:
            warnings.warn('No missing data, no point to continue.')
            return sample, mu1o, mu1o_std
        if num_unknown == num_x:
            warnings.warn('All data missing, returning sample from' +
                          ' the apriori distribution.')
            mu1o_std = ones(num_unknown) * sqrt(acf[0])
            return self.sim(ns=num_unknown, cases=1)[:, 1], mu1o, mu1o_std

        i_known = flatnonzero(1 - isnan(x))

        if method.startswith('exac'):
            # exact but slow. It also may not return any result
            if num_acf > 0.3 * num_x:
                Sigma = toeplitz(hstack((acf, zeros(num_x - num_acf))))
            else:
                acf[0] = acf[0] * 1.00001
                Sigma = sptoeplitz(hstack((acf, zeros(num_x - num_acf))))
            Soo, So1, S11 = self._split_cov(Sigma, i_known, i_unknown)

            if issparse(Sigma):
                So1 = So1.todense()
                S11 = S11.todense()
                S1o_Sooinv = spsolve(Soo + Soo.T, 2 * So1).T
            else:
                Sooinv_So1, _res, _rank, _s = lstsq(Soo + Soo.T, 2 * So1,
                                                    cond=1e-4)
                S1o_Sooinv = Sooinv_So1.T
            mu1o = S1o_Sooinv.dot(x[i_known])
            Sigma1o = S11 - S1o_Sooinv.dot(So1)
            if (diag(Sigma1o) < 0).any():
                raise ValueError('Failed to converge to a solution')

            mu1o_std = sqrt(diag(Sigma1o))
            sample[:] = rndnormnd(mu1o, Sigma1o, cases=1).ravel()

        elif method.startswith('appr'):
            # approximating by only condition on the closest points

            Nsig = min(2 * num_acf, num_x)

            Sigma = toeplitz(hstack((acf, zeros(Nsig - num_acf))))
            overlap = int(Nsig / 4)
            # indices to the points used
            idx = r_[0:Nsig] + max(0, min(i_unknown[0] - overlap,
                                          num_x - Nsig))
            mask_unknown = zeros(num_x, dtype=bool)
            # temporary storage of indices to missing points
            mask_unknown[i_unknown] = True
            t_unknown = where(mask_unknown[idx])[0]
            t_known = where(1 - mask_unknown[idx])[0]
            ns = len(t_unknown)  # number of missing data in the interval

            num_restored = 0  # number of previously simulated points
            x2 = x.copy()

            while ns > 0:
                Soo, So1, S11 = self._split_cov(Sigma, t_known, t_unknown)
                if issparse(Soo):
                    So1 = So1.todense()
                    S11 = S11.todense()
                    S1o_Sooinv = spsolve(Soo + Soo.T, 2 * So1).T
                else:
                    Sooinv_So1, _res, _rank, _s = lstsq(Soo + Soo.T, 2 * So1,
                                                        cond=1e-4)
                    S1o_Sooinv = Sooinv_So1.T
                Sigma1o = S11 - S1o_Sooinv.dot(So1)
                if (diag(Sigma1o) < 0).any():
                    raise ValueError('Failed to converge to a solution')

                ix = slice((num_restored), (num_restored + ns))
                # standard deviation of the expected surface
                mu1o_std[ix] = np.maximum(mu1o_std[ix], sqrt(diag(Sigma1o)))

                # expected surface conditioned on the closest known
                # observations from x
                mu1o[ix] = S1o_Sooinv.dot(x2[idx[t_known]])
                # sample conditioned on the known observations from x
                mu1os = S1o_Sooinv.dot(x[idx[t_known]])
                sample[ix] = rndnormnd(mu1os, Sigma1o, cases=1)
                if idx[-1] == num_x - 1:
                    ns = 0  # no more points to simulate
                else:
                    x2[idx[t_unknown]] = mu1o[ix]  # expected surface
                    x[idx[t_unknown]] = sample[ix]  # sampled surface
                    # removing indices to data which has been simulated
                    mask_unknown[idx[:-overlap]] = False
                    # data we want to simulate once more
                    nw = sum(mask_unknown[idx[-overlap:]] is True)
                    num_restored += ns - nw  # update # points simulated so far

                    idx = self._update_window(idx, i_unknown, num_x, num_acf,
                                              overlap, nw,  num_restored)

                    # find new interval with missing data
                    t_unknown = flatnonzero(mask_unknown[idx])
                    t_known = flatnonzero(1 - mask_unknown[idx])
                    ns = len(t_unknown)  # # missing data in the interval
        return sample, mu1o, mu1o_std


def sptoeplitz(x):
    k = flatnonzero(x)
    n = len(x)
    spdiags = sparse.dia_matrix
    data = x[k].reshape(-1, 1).repeat(n, axis=-1)
    offsets = k
    y = spdiags((data, offsets), shape=(n, n))
    if k[0] == 0:
        offsets = k[1::]
        data = data[1::, :]
    t = y + spdiags((data, -offsets), shape=(n, n))
    return t.tocsr()


def _test_covdata():
    import wafo.data
    x = wafo.data.sea()
    ts = wafo.objects.mat2timeseries(x)
    rf = ts.tocovdata(lag=150)
    rf.plot()


def main():
    import wafo.spectrum.models as sm
    import matplotlib
    matplotlib.interactive(True)
    Sj = sm.Jonswap()
    S = Sj.tospecdata()  # Make spec
    S.plot()
    R = S.tocovdata(rate=3)
    R.plot()
    x = R.sim(ns=1024 * 4)
    inds = np.hstack((21 + np.arange(20),
                     1000 + np.arange(20),
                     1024 * 4 - 21 + np.arange(20)))
    sample, mu1o, mu1o_std = R.simcond(x[:, 1], method='approx',
                                       i_unknown=inds)

    import matplotlib.pyplot as plt
    # inds = np.atleast_2d(inds).reshape((-1,1))
    plt.plot(x[:, 1], 'k.', label='observed values')
    plt.plot(inds, mu1o, '*', label='mu1o')
    plt.plot(inds, sample.ravel(), 'r+', label='samples')
    plt.plot(inds, mu1o - 2 * mu1o_std, 'r',
             inds, mu1o + 2 * mu1o_std, 'r', label='2 stdev')
    plt.legend()
    plt.show('hold')


if __name__ == '__main__':
    if False:  # True:  #
        import doctest
        doctest.testmod()
    else:
        main()
