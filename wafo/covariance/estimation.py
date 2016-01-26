'''
Created on 10. mai 2014

@author: pab
'''
import numpy as np
from numpy.fft import fft
from wafo.misc import nextpow2
from scipy.signal.windows import get_window
from wafo.containers import PlotData
from wafo.covariance import CovData1D
import warnings


def sampling_period(t_vec):
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
    dt1 = t_vec[1] - t_vec[0]
    n = len(t_vec) - 1
    t = t_vec[-1] - t_vec[0]
    dt = t / n
    if abs(dt - dt1) > 1e-10:
        warnings.warn('Data is not uniformly sampled!')
    return dt


class CovarianceEstimator(object):
    '''
    Class for estimating AutoCovariance from timeseries

    Parameters
    ----------
    lag : scalar, int
        maximum time-lag for which the ACF is estimated.
        (Default lag where ACF is zero)
    tr : transformation object
        the transformation assuming that x is a sample of a transformed
        Gaussian process. If g is None then x  is a sample of a Gaussian
        process (Default)
    detrend : function
        defining detrending performed on the signal before estimation.
        (default detrend_mean)
    window : vector of length NFFT or function
        To create window vectors see numpy.blackman, numpy.hamming,
        numpy.bartlett, scipy.signal, scipy.signal.get_window etc.
    flag : string, 'biased' or 'unbiased'
        If 'unbiased' scales the raw correlation by 1/(n-abs(k)),
        where k is the index into the result, otherwise scales the raw
        cross-correlation by 1/n. (default)
    norm : bool
        True if normalize output to one
    dt : scalar
        time-step between data points (default see sampling_period).
    '''
    def __init__(self, lag=None, tr=None, detrend=None, window='boxcar',
                 flag='biased', norm=False, dt=None):
        self.lag = lag
        self.tr = tr
        self.detrend = detrend
        self.window = window
        self.flag = flag
        self.norm = norm
        self.dt = dt

    def _estimate_lag(self, R, Ncens):
        Lmax = min(300, len(R) - 1)  # maximum lag if L is undetermined
        # finding where ACF is less than 2 st. deviations.
        sigma = np.sqrt(np.r_[0, R[0] ** 2,
                              R[0] ** 2 + 2 * np.cumsum(R[1:] ** 2)] / Ncens)
        lag = Lmax + 2 - (np.abs(R[Lmax::-1]) > 2 * sigma[Lmax::-1]).argmax()
        if self.window == 'parzen':
            lag = int(4 * lag / 3)
        # print('The default L is set to %d' % L)
        return lag

    def tocovdata(self, timeseries):
        '''
        Return auto covariance function from data.

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
        lag = self.lag
        window = self.window
        detrend = self.detrend

        try:
            x = timeseries.data.flatten('F')
            dt = timeseries.sampling_period()
        except Exception:
            x = timeseries[:, 1:].flatten('F')
            dt = sampling_period(timeseries[:, 0])
        if not (self.dt is None):
            dt = self.dt

        if not (self.tr is None):
            x = self.tr.dat2gauss(x)

        n = len(x)
        indnan = np.isnan(x)
        if any(indnan):
            x = x - x[1 - indnan].mean()
            Ncens = n - indnan.sum()
            x[indnan] = 0.
        else:
            Ncens = n
            x = x - x.mean()
        if hasattr(detrend, '__call__'):
            x = detrend(x)

        nfft = 2 ** nextpow2(n)
        Rper = abs(fft(x, nfft)) ** 2 / Ncens  # Raw periodogram
        R = np.real(fft(Rper)) / nfft  # ifft = fft/nfft since Rper is real!

        if self.flag.startswith('unbiased'):
            # unbiased result, i.e. divide by n-abs(lag)
            R = R[:Ncens] * Ncens / np.arange(Ncens, 1, -1)

        if self.norm:
            R = R / R[0]

        if lag is None:
            lag = self._estimate_lag(R, Ncens)
        lag = min(lag, n - 2)
        if isinstance(window, str) or type(window) is tuple:
            win = get_window(window, 2 * lag - 1)
        else:
            win = np.asarray(window)
        R[:lag] = R[:lag] * win[lag - 1::]
        R[lag] = 0
        lags = slice(0, lag + 1)
        t = np.linspace(0, lag * dt, lag + 1)
        acf = CovData1D(R[lags], t)
        acf.sigma = np.sqrt(np.r_[0, R[0] ** 2,
                            R[0] ** 2 + 2 * np.cumsum(R[1:] ** 2)] / Ncens)
        acf.children = [PlotData(-2. * acf.sigma[lags], t),
                        PlotData(2. * acf.sigma[lags], t)]
        acf.plot_args_children = ['r:']
        acf.norm = self.norm
        return acf

    __call__ = tocovdata
