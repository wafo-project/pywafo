#!/usr/bin/env python
"""
Models module
-------------

Dispersion relation
-------------------
k2w - Translates from wave number to frequency
w2k - Translates from frequency to wave number

Model spectra
-------------
Bretschneider    - Bretschneider spectral density.
Jonswap          - JONSWAP spectral density
McCormick        - McCormick spectral density.
OchiHubble       - OchiHubble bimodal spectral density model.
Tmaspec          - JONSWAP spectral density for finite water depth
Torsethaugen     - Torsethaugen  double peaked (swell + wind) spectrum model
Wallop           - Wallop spectral density.
demospec         - Loads a precreated spectrum of chosen type
jonswap_peakfact - Jonswap peakedness factor Gamma given Hm0 and Tp
jonswap_seastate - jonswap seastate from windspeed and fetch

Directional spreading functions
-------------------------------
Spreading - Directional spreading function.

"""

from __future__ import absolute_import, division

import warnings
from scipy.interpolate import interp1d
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.special as sp
from scipy.fftpack import fft
import numpy as np
from numpy import (inf, atleast_1d, newaxis, any, minimum, maximum, array,
                   asarray, exp, log, sqrt, where, pi, arange, linspace, sin,
                   cos, abs, sinh, isfinite, mod, expm1, tanh, cosh, finfo,
                   ones, ones_like, isnan, zeros_like, flatnonzero, sinc,
                   hstack, vstack, real, flipud, clip)
from ..wave_theory.dispersion_relation import w2k, k2w  # @UnusedImport
from .core import SpecData1D, SpecData2D


__all__ = ['Bretschneider', 'Jonswap', 'Torsethaugen', 'Wallop', 'McCormick',
           'OchiHubble', 'Tmaspec', 'jonswap_peakfact', 'jonswap_seastate',
           'spreading', 'w2k', 'k2w', 'phi1']

_EPS = finfo(float).eps


def sech(x):
    return 1.0 / cosh(x)


def _gengamspec(wn, N=5, M=4):
    ''' Return Generalized gamma spectrum in dimensionless form

    Parameters
    ----------
    wn : arraylike
        normalized frequencies, w/wp.
    N  : scalar
        defining the decay of the high frequency part.
    M  : scalar
        defining the spectral width around the peak.

    Returns
    -------
    S   : arraylike
        spectral values, same size as wn.

    The generalized gamma spectrum in non-
    dimensional form is defined as:

      S = G0.*wn.**(-N).*exp(-B*wn.**(-M))  for wn > 0
        = 0                              otherwise
    where
        B  = N/M
        C  = (N-1)/M
        G0 = B**C*M/gamma(C), Normalizing factor related to Bretschneider form

    Note that N = 5, M = 4 corresponds to a normalized
    Bretschneider spectrum.

    Examples
    --------
    >>> import wafo.spectrum.models as wsm
    >>> import numpy as np
    >>> wn = np.linspace(0,4,5)
    >>> wsm._gengamspec(wn, N=6, M=2)
    array([ 0.        ,  1.16765216,  0.17309961,  0.02305179,  0.00474686])

    See also
    --------
    Bretschneider
    Jonswap,
    Torsethaugen


    References
    ----------
    Torsethaugen, K. (2004)
    "Simplified Double Peak Spectral Model for Ocean Waves"
    In Proc. 14th ISOPE
    '''
    w = atleast_1d(wn)
    S = zeros_like(w)

    k = flatnonzero(w > 0.0)
    if k.size > 0:
        B = N / M
        C = (N - 1.0) / M

        # A = Normalizing factor related to Bretschneider form
        # A = B**C*M/gamma(C)
        # S[k] = A*wn[k]**(-N)*exp(-B*wn[k]**(-M))
        logwn = log(w.take(k))
        logA = (C * log(B) + log(M) - sp.gammaln(C))
        S.put(k, exp(logA - N * logwn - B * exp(-M * logwn)))
    return S


class ModelSpectrum(object):

    def __init__(self, Hm0=7.0, Tp=11.0, **kwds):
        self.Hm0 = Hm0
        self.Tp = Tp
        self.type = 'ModelSpectrum'

    def tospecdata(self, w=None, wc=None, nw=257):
        '''
        Return SpecData1D object from ModelSpectrum

        Parameter
        ---------
        w : arraylike
            vector of angular frequencies used in discretization of spectrum
        wc : scalar
            cut off frequency (default 33/Tp)
        nw : int
            number of frequencies

        Returns
        -------
        S : SpecData1D object
            member attributes of model spectrum are copied to S.workspace
        '''

        if w is None:
            if wc is None:
                wc = 33. / self.Tp
            w = linspace(0, wc, nw)
        S = SpecData1D(self.__call__(w), w)
        try:
            h = self.h
            S.h = h
        except:
            pass
        S.labels.title = self.type + ' ' + S.labels.title
        S.workspace = self.__dict__.copy()
        return S

    def chk_seastate(self):
        ''' Check if seastate is valid
        '''

        if self.Hm0 < 0:
            raise ValueError('Hm0 can not be negative!')

        if self.Tp <= 0:
            raise ValueError('Tp must be positve!')

        if self.Hm0 == 0.0:
            warnings.warn('Hm0 is zero!')

        self._chk_extra_param()

    def _chk_extra_param(self):
        pass


class Bretschneider(ModelSpectrum):

    '''
    Bretschneider spectral density model

    Member variables
    ----------------
    Hm0 : significant wave height (default 7 (m))
    Tp  : peak period             (default 11 (sec))
    N   : scalar defining decay of high frequency part.   (default 5)
    M   : scalar defining spectral width around the peak. (default 4)

    Parameters
    ----------
    w : array-like
        angular frequencies [rad/s]


    The Bretschneider spectrum is defined as

         S(w) = A * G0 * wn**(-N)*exp(-N/(M*wn**M))
    where
         G0  = Normalizing factor related to Bretschneider form
         A   = (Hm0/4)**2 / wp     (Normalization factor)
         wn  = w/wp
         wp  = 2*pi/Tp,  angular peak frequency

    This spectrum is a suitable model for fully developed sea,
    i.e. a sea state where the wind has been blowing long enough over a
    sufficiently open stretch of water, so that the high-frequency waves have
    reached an equilibrium. In the part of the spectrum where the frequency is
    greater than the peak frequency (w>wp), the energy distribution is
    proportional to w**-5.
    The spectrum is identical with ITTC (International Towing Tank
    Conference), ISSC (International Ship and Offshore Structures Congress)
    and Pierson-Moskowitz, wave spectrum given Hm0 and Tm01. It is also
    identical with JONSWAP when the peakedness factor, gamma, is one.
    For this spectrum, the following relations exist between the mean
    period Tm01 = 2*pi*m0/m1, the peak period Tp and the mean
    zero-upcrossing period Tz:

           Tm01 = 1.086*Tz, Tp = 1.408*Tz and Tp=1.2965*Tm01

    Examples
    --------
    >>> import wafo.spectrum.models as wsm
    >>> S = wsm.Bretschneider(Hm0=6.5,Tp=10)
    >>> S((0,1,2,3))
    array([ 0.        ,  1.69350993,  0.06352698,  0.00844783])

    See also
    --------
    Jonswap,
    Torsethaugen
    '''

    def __init__(self, Hm0=7.0, Tp=11.0, N=5, M=4, chk_seastate=True, **kwds):
        self.type = 'Bretschneider'
        self.Hm0 = Hm0
        self.Tp = Tp
        self.N = N
        self.M = M
        if chk_seastate:
            self.chk_seastate()

    def __call__(self, wi):
        ''' Return Bretschnieder spectrum
        '''
        w = atleast_1d(wi)
        if self.Hm0 > 0:
            wp = 2 * pi / self.Tp
            wn = w / wp
            S = (self.Hm0 / 4.0) ** 2 / wp * _gengamspec(wn, self.N, self.M)
        else:
            S = zeros_like(w)
        return S


def jonswap_peakfact(Hm0, Tp):
    ''' Jonswap peakedness factor, gamma, given Hm0 and Tp

    Parameters
    ----------
    Hm0 : significant wave height [m].
    Tp  : peak period [s]

    Returns
    -------
    gamma : Peakedness parameter of the JONSWAP spectrum

    Details
    -------
    A standard value for GAMMA is 3.3. However, a more correct approach is
    to relate GAMMA to Hm0 and Tp:
         D = 0.036-0.0056*Tp/sqrt(Hm0)
        gamma = exp(3.484*(1-0.1975*D*Tp**4/(Hm0**2)))
    This parameterization is based on qualitative considerations of deep water
    wave data from the North Sea, see Torsethaugen et. al. (1984)
    Here GAMMA is limited to 1..7.

    NOTE: The size of GAMMA is the common shape of Hm0 and Tp.

    Examples
    --------
    >>> import wafo.spectrum.models as wsm
    >>> import pylab as plb
    >>> Tp,Hs = plb.meshgrid(range(4,8),range(2,6))
    >>> gam = wsm.jonswap_peakfact(Hs,Tp)

    >>> Hm0 = plb.linspace(1,20)
    >>> Tp = Hm0
    >>> [T,H] = plb.meshgrid(Tp,Hm0)
    >>> gam = wsm.jonswap_peakfact(H,T)
    >>> v = plb.arange(0,8)
    >>> h = plb.contourf(Tp,Hm0,gam,v);h=plb.colorbar()

    >>> Hm0 = plb.arange(1,11)
    >>> Tp  = plb.linspace(2,16)
    >>> T,H = plb.meshgrid(Tp,Hm0)
    >>> gam = wsm.jonswap_peakfact(H,T)
    >>> h = plb.plot(Tp,gam.T)
    >>> h = plb.xlabel('Tp [s]')
    >>> h = plb.ylabel('Peakedness parameter')

    >>> plb.close('all')

    See also
    --------
    jonswap
    '''
    Hm0, Tp = atleast_1d(Hm0, Tp)

    x = Tp / sqrt(Hm0)

    gam = ones_like(x)

    k1 = flatnonzero(x <= 5.14285714285714)
    if k1.size > 0:  # limiting gamma to [1 7]
        xk = x.take(k1)
        D = 0.036 - 0.0056 * xk  # approx 5.061*Hm0**2/Tp**4*(1-0.287*log(gam))
        # gamma
        gam.put(k1, minimum(exp(3.484 * (1.0 - 0.1975 * D * xk ** 4.0)), 7.0))

    return gam


def jonswap_seastate(u10, fetch=150000., method='lewis', g=9.81,
                     output='dict'):
    '''
    Return Jonswap seastate from windspeed and fetch

    Parameters
    ----------
    U10 : real scalar
        windspeed at 10 m above mean water surface [m/s]
    fetch : real scalar
        fetch [m]
    method : 'hasselman73' seastate according to Hasselman et. al. 1973
             'hasselman76' seastate according to Hasselman et. al. 1976
             'lewis'       seastate according to Lewis and Allos 1990
    g : real scalar
        accelaration of gravity [m/s**2]
    output : 'dict' or 'list'

    Returns
    -------
    seastate: dict  where
            Hm0    : significant wave height [m]
            Tp     : peak period [s]
            gamma  : jonswap peak enhancement factor.
            sigmaA,
            sigmaB : jonswap spectral width parameters.
            Ag     : jonswap alpha, normalization factor.

    Example
    --------
    >>> import wafo.spectrum.models as wsm
    >>> fetch = 10000; u10 = 10
    >>> ss = wsm.jonswap_seastate(u10, fetch, output='dict')
    >>> for key in sorted(ss.keys()): key, ss[key]
    ('Ag', 0.016257903375341734)
    ('Hm0', 0.51083679198275533)
    ('Tp', 2.7727680999585265)
    ('gamma', 2.4824142635861119)
    ('sigmaA', 0.07531733139517202)
    ('sigmaB', 0.09191208451225134)
    >>> S = wsm.Jonswap(**ss)
    >>> S.Hm0
    0.51083679198275533

    # Alternatively
    >>> ss1 = wsm.jonswap_seastate(u10, fetch, output='list')
    >>> S1 = wsm.Jonswap(*ss1)
    >>> S1.Hm0
    0.51083679198275533

    See also
    --------
    Jonswap


    References
    ----------
    Lewis, A. W. and Allos, R.N. (1990)
    JONSWAP's parameters: sorting out the inconscistencies.
    Ocean Engng, Vol 17, No 4, pp 409-415

    Hasselmann et al. (1973)
    Measurements of Wind-Wave Growth and Swell Decay during the Joint
    North Sea Project (JONSWAP).
    Ergansungsheft, Reihe A(8), Nr. 12, Deutschen Hydrografischen Zeitschrift.

    Hasselmann et al. (1976)
    A parametric wave prediction model.
    J. phys. oceanogr. Vol 6, pp 200-228

    '''

    # The following formulas are from Lewis and Allos 1990:
    zeta = g * fetch / (u10 ** 2)  # dimensionless fetch, Table 1
    # zeta = min(zeta, 2.414655013429281e+004)
    if method.startswith('h'):
        if method[-1] == '3':  # Hasselman et.al (1973)
            A = 0.076 * zeta ** (-0.22)
            # dimensionless peakfrequency, Table 1
            ny = 3.5 * zeta ** (-0.33)
            # dimensionless surface variance, Table 1
            epsilon1 = 9.91e-8 * zeta ** 1.1
        else:  # Hasselman et.al (1976)
            A = 0.0662 * zeta ** (-0.2)
            ny = 2.84 * zeta ** (-0.3)  # dimensionless peakfrequency, Table 1
            # dimensionless surface variance, Eq.4
            epsilon1 = 1.6e-7 * zeta

        sa = 0.07
        sb = 0.09
        gam = 3.3
    else:
        A = 0.074 * zeta ** (-0.22)     # Eq. 10
        ny = 3.57 * zeta ** (-0.33)     # dimensionless peakfrequency, Eq. 11
        # dimensionless surface variance, Eq.12
        epsilon1 = 3.512e-4 * A * ny ** (-4.) * zeta ** (-0.1)
        sa = 0.05468 * ny ** (-0.32)      # Eq. 13
        sb = 0.078314 * ny ** (-0.16)     # Eq. 14
        gam = maximum(17.54 * zeta ** (-0.28384), 1)     # Eq. 15

    Tp = u10 / (ny * g)                          # Table 1
    Hm0 = 4 * sqrt(epsilon1) * u10 ** 2. / g            # Table 1
    if output[0] == 'l':
        return Hm0, Tp, gam, sa, sb, A
    else:
        return dict(Hm0=Hm0, Tp=Tp, gamma=gam, sigmaA=sa, sigmaB=sb, Ag=A)


class Jonswap(ModelSpectrum):

    '''
    Jonswap spectral density model

    Member variables
    ----------------
    Hm0    : significant wave height (default 7 (m))
    Tp     : peak period             (default 11 (sec))
    gamma  : peakedness factor determines the concentraton
            of the spectrum on the peak frequency.
            Usually in the range  1 <= gamma <= 7.
            default depending on Hm0, Tp, see jonswap_peakedness)
    sigmaA : spectral width parameter for w<wp (default 0.07)
    sigmaB : spectral width parameter for w<wp (default 0.09)
    Ag     : normalization factor used when gamma>1:
    N      : scalar defining decay of high frequency part.   (default 5)
    M      : scalar defining spectral width around the peak. (default 4)
    method : String defining method used to estimate Ag when gamma>1
            'integration': Ag = 1/gaussq(Gf*ggamspec(wn,N,M),0,wnc) (default)
            'parametric' : Ag = (1+f1(N,M)*log(gamma)**f2(N,M))/gamma
            'custom'     : Ag = Ag
    wnc    : wc/wp normalized cut off frequency used when calculating Ag
                by integration (default 6)
    Parameters
    ----------
    w : array-like
        angular frequencies [rad/s]

    Description
    -----------
     The JONSWAP spectrum is defined as

             S(w) = A * Gf * G0 * wn**(-N)*exp(-N/(M*wn**M))
        where
             G0  = Normalizing factor related to Bretschneider form
             A   = Ag * (Hm0/4)**2 / wp     (Normalization factor)
             Gf  = j**exp(-.5*((wn-1)/s)**2) (Peak enhancement factor)
             wn  = w/wp
             wp  = angular peak frequency
             s   = sigmaA      for wn <= 1
                   sigmaB      for 1  <  wn
             j   = gamma,     (j=1, => Bretschneider spectrum)

    The JONSWAP spectrum is assumed to be especially suitable for the North
    Sea, and does not represent a fully developed sea. It is a reasonable model
    for wind generated sea when the seastate is in the so called JONSWAP range,
    i.e., 3.6*sqrt(Hm0) < Tp < 5*sqrt(Hm0)

    The relation between the peak period and mean zero-upcrossing period
    may be approximated by
             Tz = Tp/(1.30301-0.01698*gamma+0.12102/gamma)

    Examples
    ---------
    >>> import pylab as plb
    >>> import wafo.spectrum.models as wsm
    >>> S = wsm.Jonswap(Hm0=7, Tp=11,gamma=1)
    >>> w = plb.linspace(0,5)
    >>> h = plb.plot(w,S(w))

    >>> S2 = wsm.Bretschneider(Hm0=7, Tp=11)
    >>> all(abs(S(w)-S2(w))<1.e-7)
    True
    >>> plb.close('all')

    See also
    --------
     Bretschneider
     Tmaspec
     Torsethaugen

    References
    -----------
     Torsethaugen et al. (1984)
     Characteristica for extreme Sea States on the Norwegian continental shelf.
     Report No. STF60 A84123. Norwegian Hydrodyn. Lab., Trondheim

     Hasselmann et al. (1973)
     Measurements of Wind-Wave Growth and Swell Decay during the Joint
     North Sea Project (JONSWAP).
     Ergansungsheft, Reihe A(8), Nr. 12, Deutschen Hydrografischen Zeitschrift.
    '''

    def __init__(self, Hm0=7.0, Tp=11.0, gamma=None, sigmaA=0.07, sigmaB=0.09,
                 Ag=None, N=5, M=4, method='integration', wnc=6.0,
                 chk_seastate=True):

        self.type = 'Jonswap'
        self.Hm0 = Hm0
        self.Tp = Tp
        self.N = N
        self.M = M
        self.sigmaA = sigmaA
        self.sigmaB = sigmaB
        self.gamma = gamma
        self.Ag = Ag
        self.method = method
        self.wnc = wnc

        if self.gamma is None or not isfinite(self.gamma) or self.gamma < 1:
            self.gamma = jonswap_peakfact(Hm0, Tp)

        self._preCalculateAg()

        if chk_seastate:
            self.chk_seastate()

    def _chk_extra_param(self):
        Tp = self.Tp
        Hm0 = self.Hm0
        gam = self.gamma
        outsideJonswapRange = Tp > 5 * sqrt(Hm0) or Tp < 3.6 * sqrt(Hm0)
        if outsideJonswapRange:
            txt0 = '''
            Hm0=%g,Tp=%g is outside the JONSWAP range.
            The validity of the spectral density is questionable.
            ''' % (Hm0, Tp)
            warnings.warn(txt0)

        if gam < 1 or 7 < gam:
            txt = '''
            The peakedness factor, gamma, is possibly too large.
            The validity of the spectral density is questionable.
            '''
            warnings.warn(txt)

    def _localspec(self, wn):
        Gf = self.peak_e_factor(wn)
        return Gf * _gengamspec(wn, self.N, self.M)

    def _preCalculateAg(self):
        ''' PRECALCULATEAG Precalculate normalization.
        '''
        if self.gamma == 1:
            self.Ag = 1.0
            self.method = 'parametric'
        elif self.Ag is not None:
            self.method = 'custom'
            if self.Ag <= 0:
                raise ValueError('Ag must be larger than 0!')
        elif self.method[0] == 'i':
            # normalizing by integration
            self.method = 'integration'
            if self.wnc < 1.0:
                raise ValueError('Normalized cutoff frequency, wnc, ' +
                                 'must be larger than one!')
            area1, unused_err1 = integrate.quad(self._localspec, 0, 1)
            area2, unused_err2 = integrate.quad(self._localspec, 1, self.wnc)
            area = area1 + area2
            self.Ag = 1.0 / area
        elif self.method[1] == 'p':
            self.method = 'parametric'
            # Original normalization
            # NOTE: that  Hm0**2/16 generally is not equal to intS(w)dw
            # with this definition of Ag if sa or sb are changed from the
            # default values
            N = self.N
            M = self.M
            gammai = self.gamma
            parametersOK = (3 <= N and N <= 50) or (
                2 <= M and M <= 9.5) and (1 <= gammai and gammai <= 20)
            if parametersOK:
                f1NM = 4.1 * \
                    (N - 2 * M ** 0.28 + 5.3) ** (-1.45 * M ** 0.1 + 0.96)
                f2NM = (2.2 * M ** (-3.3) + 0.57) * \
                    N ** (-0.58 * M ** 0.37 + 0.53) - 1.04 * M ** (-1.9) + 0.94
                self.Ag = (1 + f1NM * log(gammai) ** f2NM) / gammai

            # elseif N == 5 && M == 4,
            # options.Ag = (1+1.0*log(gammai).**1.16)/gammai
            # options.Ag = (1-0.287*log(gammai))
            # options.normalizeMethod = 'Three'
            # elseif  N == 4 && M == 4,
            # options.Ag = (1+1.1*log(gammai).**1.19)/gammai
            else:
                raise ValueError('Not knowing the normalization because N, ' +
                                 'M or peakedness parameter is out of bounds!')

            if self.sigmaA != 0.07 or self.sigmaB != 0.09:
                warnings.warn('Use integration to calculate Ag when ' +
                              'sigmaA!=0.07 or sigmaB!=0.09')

    def peak_e_factor(self, wn):
        ''' PEAKENHANCEMENTFACTOR
        '''
        w = maximum(atleast_1d(wn), 0.0)
        sab = where(w > 1, self.sigmaB, self.sigmaA)

        wnm12 = 0.5 * ((w - 1.0) / sab) ** 2.0
        Gf = self.gamma ** (exp(-wnm12))
        return Gf

    def __call__(self, wi):
        ''' JONSWAP spectral density
        '''
        w = atleast_1d(wi)
        if (self.Hm0 > 0.0):

            N = self.N
            M = self.M
            wp = 2 * pi / self.Tp
            wn = w / wp
            Ag = self.Ag
            Hm0 = self.Hm0
            Gf = self.peak_e_factor(wn)
            S = ((Hm0 / 4.0) ** 2 / wp * Ag) * Gf * _gengamspec(wn, N, M)
        else:
            S = zeros_like(w)
        return S


def phi1(wi, h, g=9.81):
    ''' Factor transforming spectra to finite water depth spectra.

    Input
    -----
         w : arraylike
            angular frequency [rad/s]
         h : scalar
            water depth [m]
         g : scalar
            acceleration of gravity [m/s**2]
    Returns
    -------
        tr : arraylike
            transformation factors

    Example:
    -------
    Transform a JONSWAP spectrum to a spectrum for waterdepth = 30 m
    >>> import wafo.spectrum.models as wsm
    >>> S = wsm.Jonswap()
    >>> w = np.arange(3.0)
    >>> S(w)*wsm.phi1(w,30.0)
    array([ 0.        ,  1.0358056 ,  0.03796281])


    Reference
    ---------
    Buows, E., Gunther, H., Rosenthal, W. and Vincent, C.L. (1985)
     'Similarity of the wind wave spectrum in finite depth water:
     1 spectral form.'
      J. Geophys. Res., Vol 90, No. C1, pp 975-986

    '''
    w = atleast_1d(wi)
    if h == inf:  # % special case infinite water depth
        return ones_like(w)

    k1 = w2k(w, 0, inf, g=g)[0]
    dw1 = 2.0 * w / g  # % dw/dk|h=inf
    k2 = w2k(w, 0, h, g=g)[0]

    k2h = k2 * h
    den = where(k1 == 0, 1, (tanh(k2h) + k2h / cosh(k2h) ** 2.0))
    dw2 = where(k1 == 0, 0, dw1 / den)  # dw/dk|h=h0
    return where(k1 == 0, 0, (k1 / k2) ** 3.0 * dw2 / dw1)


class Tmaspec(Jonswap):

    ''' JONSWAP spectrum for finite water depth

    Member variables
    ----------------
     h     = water depth             (default 42 [m])
     g     : acceleration of gravity [m/s**2]
     Hm0   = significant wave height (default  7 [m])
     Tp    = peak period (default 11 (sec))
     gamma = peakedness factor determines the concentraton
              of the spectrum on the peak frequency.
              Usually in the range  1 <= gamma <= 7.
              default depending on Hm0, Tp, see getjonswappeakedness)
     sigmaA = spectral width parameter for w<wp (default 0.07)
     sigmaB = spectral width parameter for w<wp (default 0.09)
     Ag     = normalization factor used when gamma>1:
     N      = scalar defining decay of high frequency part.   (default 5)
     M      = scalar defining spectral width around the peak. (default 4)
     method = String defining method used to estimate Ag when gamma>1
             'integrate' : Ag = 1/gaussq(Gf.*ggamspec(wn,N,M),0,wnc) (default)
              'parametric': Ag = (1+f1(N,M)*log(gamma)^f2(N,M))/gamma
              'custom'    : Ag = Ag
     wnc    = wc/wp normalized cut off frequency used when calculating Ag
               by integration (default 6)
    Parameters
    ----------
    w : array-like
        angular frequencies [rad/s]

    Description
    ------------
    The evaluated spectrum is
         S(w) = Sj(w)*phi(w,h)
    where
         Sj  = jonswap spectrum
         phi = modification due to water depth

    The concept is based on a similarity law, and its validity is verified
    through analysis of 3 data sets from: TEXEL, MARSEN projects (North
    Sea) and ARSLOE project (Duck, North Carolina, USA). The data include
    observations at water depths ranging from 6 m to 42 m.

    Example
    --------
    >>> import wafo.spectrum.models as wsm
    >>> import pylab as plb
    >>> w = plb.linspace(0,2.5)
    >>> S = wsm.Tmaspec(h=10,gamma=1) # Bretschneider spectrum Hm0=7, Tp=11
    >>> o=plb.plot(w,S(w))
    >>> o=plb.plot(w,S(w,h=21))
    >>> o=plb.plot(w,S(w,h=42))
    >>> plb.show()
    >>> plb.close('all')

    See also
    ---------
    Bretschneider,
    Jonswap,
    phi1,
    Torsethaugen

    References
    ----------
    Buows, E., Gunther, H., Rosenthal, W., and Vincent, C.L. (1985)
    'Similarity of the wind wave spectrum in finite depth water:
    1 spectral form.'
    J. Geophys. Res., Vol 90, No. C1, pp 975-986

    Hasselman et al. (1973)
    Measurements of Wind-Wave Growth and Swell Decay during the Joint
    North Sea Project (JONSWAP).
    Ergansungsheft, Reihe A(8), Nr. 12, deutschen Hydrografischen
    Zeitschrift.

    '''

    def __init__(self, Hm0=7.0, Tp=11.0, gamma=None, sigmaA=0.07, sigmaB=0.09,
                 Ag=None, N=5, M=4, method='integration', wnc=6.0,
                 chk_seastate=True, h=42, g=9.81):
        self.g = g
        self.h = h
        super(Tmaspec, self).__init__(Hm0, Tp, gamma, sigmaA, sigmaB, Ag, N,
                                      M, method, wnc, chk_seastate)
        self.type = 'TMA'

    def phi(self, w, h=None, g=None):
        if h is None:
            h = self.h
        if g is None:
            g = self.g
        return phi1(w, h, g)

    def __call__(self, w, h=None, g=None):
        jonswap = super(Tmaspec, self).__call__(w)
        return jonswap * self.phi(w, h, g)


class Torsethaugen(ModelSpectrum):

    '''
    Torsethaugen  double peaked (swell + wind) spectrum model

    Member variables
    ----------------
    Hm0   : significant wave height (default 7 (m))
    Tp    : peak period (default 11 (sec))
    wnc   : wc/wp normalized cut off frequency used when calculating Ag
            by integration (default 6)
    method : String defining method used to estimate normalization factors, Ag,
             in the the modified JONSWAP spectra when gamma>1
            'integrate' : Ag = 1/quad(Gf.*gengamspec(wn,N,M),0,wnc)
            'parametric': Ag = (1+f1(N,M)*log(gamma)**f2(N,M))/gamma
    Parameters
    ----------
    w : array-like
        angular frequencies [rad/s]

    Description
    -----------
    The double peaked (swell + wind) Torsethaugen spectrum is
    modelled as  S(w) = Ss(w) + Sw(w) where Ss and Sw are modified
    JONSWAP spectrums for swell and wind peak, respectively.
    The energy is divided between the two peaks according
    to empirical parameters, which peak that is primary depends on parameters.
    The empirical parameters are found for classes of Hm0 and Tp,
    originating from a dataset consisting of 20 000 spectra divided
    into 146 different classes of Hm0 and Tp. (Data measured at the
    Statfjord field in the North Sea in a period from 1980 to 1989.)
    The range of the measured  Hm0 and Tp for the dataset
    are from 0.5 to 11 meters and from 3.5 to 19 sec, respectively.

    Preliminary comparisons with spectra from other areas indicate that
    some of the empirical parameters are dependent on geographical location.
    Thus the model must be used with care for other areas than the
    North Sea and sea states outside the area where measured data
    are available.

    Example
    -------
    >>> import wafo.spectrum.models as wsm
    >>> import pylab as plb
    >>> w = plb.linspace(0,4)
    >>> S = wsm.Torsethaugen(Hm0=6, Tp=8)
    >>> h=plb.plot(w,S(w),w,S.wind(w),w,S.swell(w))

    See also
    --------
    Bretschneider
    Jonswap


    References
    ----------
    Torsethaugen, K. (2004)
    "Simplified Double Peak Spectral Model for Ocean Waves"
    In Proc. 14th ISOPE

    Torsethaugen, K. (1996)
    Model for a doubly peaked wave spectrum
    Report No. STF22 A96204. SINTEF Civil and Environm. Engineering, Trondheim

    Torsethaugen, K. (1994)
    'Model for a doubly peaked spectrum. Lifetime and fatigue strength
    estimation implications.'
    International Workshop on Floating Structures in Coastal zone,
    Hiroshima, November 1994.

    Torsethaugen, K. (1993)
    'A two peak wave spectral model.'
    In proceedings OMAE, Glasgow

    '''

    def __init__(self, Hm0=7, Tp=11, method='integration', wnc=6, gravity=9.81,
                 chk_seastate=True, **kwds):
        self.type = 'Torsethaugen'
        self.Hm0 = Hm0
        self.Tp = Tp
        self.method = method
        self.wnc = wnc
        self.gravity = gravity
        self.wind = None
        self.swell = None
        if chk_seastate:
            self.chk_seastate()

        self._init_spec()

    def __call__(self, w):
        ''' TORSETHAUGEN spectral density
        '''
        return self.wind(w) + self.swell(w)

    def _chk_extra_param(self):
        Hm0 = self.Hm0
        Tp = self.Tp
        if Hm0 > 11 or Hm0 > max((Tp / 3.6) ** 2, (Tp - 2) * 12 / 11):
            txt0 = '''Hm0 is outside the valid range.
                    The validity of the spectral density is questionable'''
            warnings.warn(txt0)

        if Tp > 20 or Tp < 3:
            txt1 = '''Tp is outside the valid range.
                    The validity of the spectral density is questionable'''
            warnings.warn(txt1)

    def _init_spec(self):
        ''' Initialize swell and wind part of Torsethaugen spectrum
        '''
        monitor = 0
        Hm0 = self.Hm0
        Tp = self.Tp
        gravity1 = self.gravity  # m/s**2

        min = minimum  # @ReservedAssignment
        max = maximum  # @ReservedAssignment

        # The parameter values below are found comparing the
        # model to average measured spectra for the Statfjord Field
        # in the Northern North Sea.
        Af = 6.6  # m**(-1/3)*sec
        AL = 2  # sec/sqrt(m)
        Au = 25  # sec
        KG = 35
        KG0 = 3.5
        KG1 = 1     # m
        r = 0.857  # 6/7
        K0 = 0.5  # 1/sqrt(m)
        K00 = 3.2

        M0 = 4
        B1 = 2  # sec
        B2 = 0.7
        B3 = 3.0  # m
        S0 = 0.08  # m**2*s
        S1 = 3  # m

        # Preliminary comparisons with spectra from other areas indicate that
        # the parameters on the line below can be dependent on geographical
        # location
        A10 = 0.7
        A1 = 0.5
        A20 = 0.6
        A2 = 0.3
        A3 = 6

        Tf = Af * (Hm0) ** (1.0 / 3.0)
        Tl = AL * sqrt(Hm0)   # lower limit
        Tu = Au             # upper limit

        # Non-dimensional scales
        # New call pab April 2005
        El = min(max((Tf - Tp) / (Tf - Tl), 0), 1)  # wind sea
        Eu = min(max((Tp - Tf) / (Tu - Tf), 0), 1)  # Swell

        if Tp < Tf:  # Wind dominated seas
            # Primary peak (wind dominated)
            Nw = K0 * sqrt(Hm0) + K00             # high frequency exponent
            Mw = M0                           # spectral width exponent
            Rpw = min((1 - A10) * exp(-(El / A1) ** 2) + A10, 1)
            Hpw = Rpw * Hm0                      # significant waveheight wind
            Tpw = Tp                           # primary peak period
            # peak enhancement factor
            gammaw = KG * (1 + KG0 * exp(-Hm0 / KG1)) * \
                (2 * pi / gravity1 * Rpw * Hm0 / (Tp ** 2)) ** r
            gammaw = max(gammaw, 1)
            # Secondary peak (swell)
            Ns = Nw                # high frequency exponent
            Ms = Mw                # spectral width exponent
            Rps = sqrt(1.0 - Rpw ** 2.0)
            Hps = Rps * Hm0           # significant waveheight swell
            Tps = Tf + B1
            gammas = 1.0

            if monitor:
                if Rps > 0.1:
                    print('     Spectrum for Wind dominated sea')
                else:
                    print('     Spectrum for pure wind sea')
        else:  # swell dominated seas

            # Primary peak (swell)
            Ns = K0 * sqrt(Hm0) + K00  # high frequency exponent
            Ms = M0  # spectral width exponent
            Rps = min((1 - A20) * exp(-(Eu / A2) ** 2) + A20, 1)
            Hps = Rps * Hm0                      # significant waveheight swell
            Tps = Tp                           # primary peak period
            # peak enhancement factor
            gammas = KG * (1 + KG0 * exp(-Hm0 / KG1)) * \
                (2 * pi / gravity1 * Hm0 / (Tf ** 2)) ** r * (1 + A3 * Eu)
            gammas = max(gammas, 1)

            # Secondary peak (wind)
            Nw = Ns                       # high frequency exponent
            Mw = M0 * (1 - B2 * exp(-Hm0 / B3))   # spectral width exponent
            Rpw = sqrt(1 - Rps ** 2)
            Hpw = Rpw * Hm0                  # significant waveheight wind

            C = (Nw - 1) / Mw
            B = Nw / Mw
            G0w = B ** C * Mw / sp.gamma(C)  # normalizing factor
            # G0w = exp(C*log(B)+log(Mw)-gammaln(C))
            # G0w  = Mw/((B)**(-C)*gamma(C))

            if Hpw > 0:
                Tpw = (16 * S0 * (1 - exp(-Hm0 / S1)) * (0.4) **
                       Nw / (G0w * Hpw ** 2)) ** (-1.0 / (Nw - 1.0))
            else:
                Tpw = inf

            # Tpw  = max(Tpw,2.5)
            gammaw = 1
            if monitor:
                if Rpw > 0.1:
                    print('     Spectrum for swell dominated sea')
                else:
                    print('     Spectrum for pure swell sea')

        if monitor:
            if (3.6 * sqrt(Hm0) <= Tp & Tp <= 5 * sqrt(Hm0)):
                print('     Jonswap range')

            print('Hm0 = %g' % Hm0)
            print('Ns, Ms = %g, %g  Nw, Mw = %g, %g' % (Ns, Ms, Nw, Mw))
            print('gammas = %g gammaw = ' % (gammas, gammaw))
            print('Rps = %g Rpw = %g' % (Rps, Rpw))
            print('Hps = %g Hpw = %g' % (Hps, Hpw))
            print('Tps = %g Tpw = %g' % (Tps, Tpw))

        # G0s=Ms/((Ns/Ms)**(-(Ns-1)/Ms)*gamma((Ns-1)/Ms )) #normalizing factor
        # Wind part
        self.wind = Jonswap(Hm0=Hpw, Tp=Tpw, gamma=gammaw, N=Nw, M=Mw,
                            method=self.method, chk_seastate=False)
        # Swell part
        self.swell = Jonswap(Hm0=Hps, Tp=Tps, gamma=gammas, N=Ns, M=Ms,
                             method=self.method, chk_seastate=False)


class McCormick(Bretschneider):

    ''' McCormick spectral density model

    Member variables
    ----------------
    Hm0  = significant wave height (default 7 (m))
    Tp   = peak period (default 11 (sec))
    Tz   = zero-down crossing period (default  0.8143*Tp)
    M    = scalar defining spectral width around the peak.
           (default depending on Tp and Tz)

    Parameters
    ----------
    w : array-like
        angular frequencies [rad/s]

    Description
    -----------
    The McCormick spectrum parameterization is a modification of the
    Bretschneider spectrum and defined as

        S(w) = (M+1)*(Hm0/4)^2/wp*(wp./w)^(M+1)*exp(-(M+1)/M*(wp/w)^M)
    where
        Tp/Tz=(1+1/M)^(1/M)/gamma(1+1/M)


    Example:
    --------
    >>> import wafo.spectrum.models as wsm
    >>> S = wsm.McCormick(Hm0=6.5,Tp=10)
    >>> S(range(4))
    array([ 0.        ,  1.87865908,  0.15050447,  0.02994663])


    See also
    --------
    Bretschneider
    Jonswap,
    Torsethaugen


    References:
    -----------
    M.E. McCormick (1999)
    "Application of the Generic Spectral Formula to Fetch-Limited Seas"
    Marine Technology Society, Vol 33, No. 3, pp 27-32
    '''

    def __init__(self, Hm0=7, Tp=11, Tz=None, M=None, chk_seastate=True):
        self.type = 'McCormick'
        self.Hm0 = Hm0
        self.Tp = Tp
        if Tz is None:
            Tz = 0.8143 * Tp

        self.Tz = Tz
        if chk_seastate:
            self.chk_seastate()

        if M is None and self.Hm0 > 0:
            self._TpdTz = Tp / Tz
            M = 1.0 / optimize.fminbound(self._localoptfun, 0.01, 5)
        self.M = M
        self.N = M + 1.0

    def _localoptfun(self, x):
        # LOCALOPTFUN Local function to optimize.
        y = 1.0 + x
        return (y ** (x) / sp.gamma(y) - self._TpdTz) ** 2.0


class OchiHubble(ModelSpectrum):

    ''' OchiHubble bimodal spectral density model.

    Member variables
    ----------------

    Hm0   : significant wave height (default 7 (m))
    par   : integer defining the parametrization (default 0)
            0 : The most probable spectrum
            1,2,...10 : gives 95% Confidence spectra

    The OchiHubble  bimodal spectrum is modelled as
    S(w) = Ss(w) + Sw(w) where Ss and Sw are modified Bretschneider
    spectra for swell and wind peak, respectively.

    The OH spectrum is a six parameter spectrum, all functions of Hm0.
    The values of these parameters are determined from a analysis of data
    obtained in the North Atlantic. The source of the data is the same as
    that for the development of the Pierson-Moskowitz spectrum, but
    analysis is carried out on over 800 spectra including those in
    partially developed seas and those having a bimodal shape. From a
    statistical analysis of the data, a family of wave spectra consisting
    of 11 members is generated for a desired sea severity (Hm0) with the
    coefficient of 0.95.
    A significant advantage of using a family of spectra for design  of
    marine systems is that one of the family members yields the largest
    response such as motions or wave induced forces for a specified sea
    severity, while another yields the smallest response with confidence
    coefficient of 0.95.

    Examples
    --------
    >>> import wafo.spectrum.models as wsm
    >>> S = wsm.OchiHubble(par=2)
    >>> S(range(4))
    array([ 0.        ,  0.90155636,  0.04185445,  0.00583207])


    See also
    --------
    Bretschneider,
    Jonswap,
    Torsethaugen

    References:
    ----------
    Ochi, M.K. and Hubble, E.N. (1976)
    'On six-parameter wave spectra.'
    In Proc. 15th Conf. Coastal Engng., Vol.1, pp301-328

    '''

    def __init__(self, Hm0=7, par=1, chk_seastate=True):
        self.type = 'Ochi Hubble'
        self.Hm0 = Hm0
        self.Tp = 1
        self.par = par
        self.wind = None
        self.swell = None

        if chk_seastate:
            self.chk_seastate()
        self._init_spec()

    def __call__(self, w):
        return self.wind(w) + self.swell(w)

    def _init_spec(self):

        hp = array([[0.84, 0.54],
                    [0.84, 0.54],
                    [0.84, 0.54],
                    [0.84, 0.54],
                    [0.84, 0.54],
                    [0.95, 0.31],
                    [0.65, 0.76],
                    [0.90, 0.44],
                    [0.77, 0.64],
                    [0.73, 0.68],
                    [0.92, 0.39]])
        wa = array([[0.7, 1.15],
                    [0.93, 1.5],
                    [0.41, 0.88],
                    [0.74, 1.3],
                    [0.62, 1.03],
                    [0.70, 1.50],
                    [0.61, 0.94],
                    [0.81, 1.60],
                    [0.54, 0.61],
                    [0.70, 0.99],
                    [0.70, 1.37]])
        wb = array([[0.046, 0.039],
                    [0.056, 0.046],
                    [0.016, 0.026],
                    [0.052, 0.039],
                    [0.039, 0.030],
                    [0.046, 0.046],
                    [0.039, 0.036],
                    [0.052, 0.033],
                    [0.039, 0.000],
                    [0.046, 0.039],
                    [0.046, 0.039]])
        Lpar = array([[3.00, 1.54, -0.062],
                      [3.00, 2.77, -0.112],
                      [2.55, 1.82, -0.089],
                      [2.65, 3.90, -0.085],
                      [2.60, 0.53, -0.069],
                      [1.35, 2.48, -0.102],
                      [4.95, 2.48, -0.102],
                      [1.80, 2.95, -0.105],
                      [4.50, 1.95, -0.082],
                      [6.40, 1.78, -0.069],
                      [0.70, 1.78, -0.069]])
        Hm0 = self.Hm0
        Lpari = Lpar[self.par]
        Li = hstack((Lpari[0], Lpari[1] * exp(Lpari[2] * Hm0)))

        Hm0i = hp[self.par] * Hm0
        Tpi = 2 * pi * exp(wb[self.par] * Hm0) / wa[self.par]
        Ni = 4 * Li + 1
        Mi = [4, 4]

        self.swell = Bretschneider(Hm0=Hm0i[0], Tp=Tpi[0], N=Ni[0], M=Mi[0])
        self.wind = Bretschneider(Hm0=Hm0i[1], Tp=Tpi[1], N=Ni[1], M=Mi[1])

    def _chk_extra_param(self):
        if self.par < 0 or 10 < self.par:
            raise ValueError('Par must be an integer from 0 to 10!')


class Wallop(Bretschneider):

    '''Wallop spectral density model.

    Member variables
    ----------------
    Hm0  = significant wave height (default 7 (m))
    Tp   = peak period (default 11 (sec))
    N   = shape factor, i.e. slope for the high frequency
%                     part (default depending on Hm0 and Tp, see below)

    Parameters
    ----------
    w : array-like
        angular frequencies [rad/s]

    Description
    -----------
    The WALLOP spectrum parameterization is a modification of the Bretschneider
    spectrum and defined as

            S(w) = A * G0 * wn**(-N)*exp(-N/(4*wn**4))
    where
        G0 = Normalizing factor related to Bretschneider form
        A  = (Hm0/4)^2 / wp     (Normalization factor)
        wn = w/wp
        wp = 2*pi/Tp,  angular peak frequency
        N  = abs((log(2*pi^2)+2*log(Hm0/4)-2*log(Lp))/log(2))
        Lp = wave length corresponding to the peak frequency, wp.

    If N=5 it becomes the same as the JONSWAP spectrum with
    peak enhancement factor gamma=1 or the Bretschneider
    (Pierson-Moskowitz) spectrum.

    Example:
    --------
    >>> import wafo.spectrum.models as wsm
    >>> S = wsm.Wallop(Hm0=6.5, Tp=10)
    >>> S(range(4))
    array([  0.00000000e+00,   9.36921871e-01,   2.76991078e-03,
             7.72996150e-05])

    See also
    --------
    Bretschneider
    Jonswap,
    Torsethaugen

    References:
    -----------
    Huang, N.E., Long, S.R., Tung, C.C, Yuen, Y. and Bilven, L.F. (1981)
    "A unified two parameter wave spectral model for a generous sea state"
    J. Fluid Mechanics, Vol.112, pp 203-224
    '''

    def __init__(self, Hm0=7, Tp=11, N=None, chk_seastate=True):
        self.type = 'Wallop'
        self.Hm0 = Hm0
        self.Tp = Tp
        self.M = 4
        if N is None:
            wp = 2. * pi / Tp
            kp = w2k(wp, 0, inf)[0]  # wavenumber at peak frequency
            Lp = 2. * pi / kp  # wave length at the peak frequency
            N = abs((log(2. * pi ** 2.) + 2 * log(Hm0 / 4) -
                     2.0 * log(Lp)) / log(2))

        self.N = N

        if chk_seastate:
            self.chk_seastate()


class Spreading(object):
    '''
    Directional spreading function.

    Parameters
    ----------
    theta, w : arrays
        angles and angular frequencies given in radians and rad/s,
        respectively. Lenghts are Nt and Nw.
    wc : real scalar
        cut over frequency

    Returns
    -------
    D : 2D array
        Directonal spreading function. size Nt X Nw.
        The principal direction of D is always along the x-axis.
    phi0 : real scalar
        Parameter defining the actual principal direction of D.

    Member variables
    ----------------
    type : string (default 'cos-2s')
        type of spreading function, see options below
        'cos-2s'    : N(S)*[cos((theta-theta0)/2)]**(2*S)  (0 < S)
        'Box-car'   : N(A)*I( -A < theta-theta0 < A)       (0 < A < pi)
        'von-Mises' : N(K)*exp(K*cos(theta-theta0))        (0 < K)
        'Poisson'   : N(X)/(1-2*X*cos(theta-theta0)+X**2)  (0 < X < 1)
        'sech-2'    : N(B)*sech(B*(theta-theta0))**2       (0 < B)
        'wrapped-normal':
            [1 + 2*sum exp(-(n*D1)^2/2)*cos(n*(theta-theta0))]/(2*pi) (0 < D1)
         (N(.) = normalization factor)
         (the first letter is enough for unique identification)

    theta0 : callable, matrix or a scalar
        defines average direction given in radians at every angular frequency.
                (length 1 or length == length(wn)) (default 0)
    method : string or integer
        Defines function used for direcional spreading parameter:
        0, None       : S(wn) = s_a, frequency independent
        1, 'mitsuyasu': S(wn) frequency dependent  (default)
            where  S(wn) = s_a *(wn)**m_a,  for wn_lo <= wn < wn_c
                         = s_b *(wn)**m_b,  for  wn_c  <= wn < wn_up
                         = 0                otherwise
        2, 'donelan'  : B(wn) frequency dependent
        3, 'banner'   : B(wn) frequency dependent
            where  B(wn) = S(wn)           for wn_lo <= wn < wn_up
                         = s_b*wn_up**m_b, for wn_up <= wn and method = 2
                         = sc*F(wn)        for wn_up <= wn and method = 3
               where F(wn) = 10^(-0.4+0.8393*exp(-0.567*log(wn^2))) and
               sc is scalefactor to make the spreading funtion continous.
    wn_lo, wn_c, wn_up: real scalars  (default 0, 1, inf)
        limits used in the function defining the directional spreading
        parameter, S() or B() defined above.
        wn_c is the normalized cutover frequency
    s_a, s_b : real scalars
        maximum spread parameters      (default [15 15])
    m_a, m_b : real scalars
        shape parameters                 (default [5 -2.5])

    SPREADING return a Directional spreading function.
    Here the S- or B-parameter, of the COS-2S and SECH-2 spreading function,
    respectively, is used as a measure of spread. All the parameters of the
    other distributions are related to this parameter through the first Fourier
    coefficient, R1, of the directional distribution as follows:
         R1 = S/(S+1) or S = R1/(1-R1).
    where
         Box-car spreading  : R1 = sin(A)/A
         Von Mises spreading: R1 = besseli(1,K)/besseli(0,K),
         Poisson spreading  : R1 = X
         sech-2 spreading   : R1 = pi/(2*B*sinh(pi/(2*B))
         Wrapped Normal     : R1 = exp(-D1^2/2)

    A value of S = 15 corresponds to
           'box'    : A=0.62,          'sech-2'  : B=0.89
           'von-mises'  : K=8.3,           'poisson': X=0.94
           'wrapped-normal': D=0.36

    The COS2S is the most frequently used spreading in engineering practice.
    Apart from the current meter/pressure cell data in WADIC all
    instruments seem to support the 'cos2s' distribution for heavier sea
    states, (Krogstad and Barstow, 1999). For medium sea states
    a spreading function between COS2S and POISSON seem appropriate,
    while POISSON seems appropriate for swell.
    For the COS2S Mitsuyasu et al. parameterized SPa = SPb =
    11.5*(U10/Cp) where Cp = g/wp is the deep water phase speed at wp and
    U10 the wind speed at reference height 10m. Hasselman et al. (1980)
    parameterized  mb = -2.33-1.45*(U10/Cp-1.17).
    Mitsuyasu et al. (1975) showed that SP for wind waves varies from
    5 to 30 being a function of dimensionless wind speed.
    However, Goda and Suzuki (1975) proposed SP = 10 for wind waves, SP = 25
    for swell with short decay distance and SP = 75 for long decay distance.
    Compared to experiments Krogstad et al. (1998) found that m_a = 5 +/- _EPS
    and that -1< m_b < -3.5.
    Values given in the litterature:  [s_a  s_b  m_a   m_b  wn_lo wn_c wn_up]
    (Mitsuyasu: s_a == s_b)  (cos-2s) [15   15   5    -2.5  0     1    3  ]
    (Hasselman: s_a ~= s_b)  (cos-2s) [6.97 9.77 4.06 -2.3  0     1.05 3  ]
    (Banner   : s_a ~= s_b)  (sech2)  [2.61 2.28 1.3  -1.3  0.56  0.95 1.6]

    Examples
    --------
    >>> import wafo.spectrum.models as wsm
    >>> import pylab as plb
    >>> D = wsm.Spreading('cos2s',s_a=10.0)

    # Make directionale spectrum
    >>> S = wsm.Jonswap().tospecdata()
    >>> SD = D.tospecdata2d(S)
    >>> h = SD.plot()

    >>> w = plb.linspace(0,3,257)
    >>> theta = plb.linspace(-pi,pi,129)
    >>> t = plb.contour(D(theta,w)[0].squeeze())

    # Make frequency dependent direction spreading
    >>> theta0 = lambda w: w*plb.pi/6.0
    >>> D2 = wsm.Spreading('cos2s',theta0=theta0)
    >>> t = plb.contour(D2(theta,w)[0])

    # Plot all spreading functions
    >>> alltypes = ('cos2s','box','mises','poisson','sech2','wrap_norm')
    >>> for ix in range(len(alltypes)):
    ...     D3 = wsm.Spreading(alltypes[ix])
    ...     t = plb.figure(ix)
    ...     t = plb.contour(D3(theta,w)[0])
    ...     t = plb.title(alltypes[ix])
    >>> plb.close('all')


    See also
    --------
    mkdspec, plotspec, spec2spec

    References
    ---------
    Krogstad, H.E. and Barstow, S.F. (1999)
    "Directional Distributions in Ocean Wave Spectra"
    Proceedings of the 9th ISOPE Conference, Vol III, pp. 79-86

    Goda, Y. (1999)
    "Numerical simulation of ocean waves for statistical analysis"
    Marine Tech. Soc. Journal, Vol. 33, No. 3, pp 5--14

    Banner, M.L. (1990)
    "Equilibrium spectra of wind waves."
    J. Phys. Ocean, Vol 20, pp 966--984
    Donelan M.A., Hamilton J, Hui W.H. (1985)
    "Directional spectra of wind generated waves."
    Phil. Trans. Royal Soc. London, Vol A315, pp 387--407

    Hasselmann D, Dunckel M, Ewing JA (1980)
    "Directional spectra observed during JONSWAP."
    J. Phys. Ocean, Vol.10, pp 1264--1280

    Mitsuyasu, H, et al. (1975)
    "Observation of the directional spectrum of ocean waves using a
    coverleaf buoy."
    J. Physical Oceanography, Vol.5, No.4, pp 750--760
    Some of this might be included in help header:
    cos-2s:
    NB! The generally strong frequency dependence in directional spread
    makes it questionable to run load tests of ships and structures with a
    directional spread independent of frequency (Krogstad and Barstow, 1999).
    '''
# Parameterization of B
#    def = 2 Donelan et al freq. parametrization for 'sech2'
#    def = 3 Banner freq. parametrization for 'sech2'
#    (spa ~= spb)  (sech-2)  [2.61 2.28 1.3  -1.3  0.56 0.95 1.6]
#

    def __init__(self, type='cos-2s', theta0=0,  # @ReservedAssignment
                 method='mitsuyasu', s_a=15., s_b=15., m_a=5., m_b=-2.5,
                 wn_lo=0.0, wn_c=1., wn_up=inf):

        self.type = type
        self.theta0 = theta0
        self.method = method
        self.s_a = s_a
        self.s_b = s_b
        self.m_a = m_a
        self.m_b = m_b
        self.wn_lo = wn_lo
        self.wn_c = wn_c
        self.wn_up = wn_up

        methods = dict(n=None, m='mitsuyasu', d='donelan', b='banner')
        methodslist = (None, 'mitsuyasu', 'donelan', 'banner')

        if isinstance(self.method, str):
            if not self.method[0] in methods:
                raise ValueError('Unknown method')
            self.method = methods[self.method[0]]
        elif self.method is None:
            pass
        else:
            if method < 0 or 3 < method:
                method = 2
            self.method = methodslist[method]

        self._spreadfun = dict(c=self.cos2s, b=self.box, m=self.mises,
                               v=self.mises,
                               p=self.poisson, s=self.sech2, w=self.wrap_norm)
        self._fourierdispatch = dict(b=self.fourier2a, m=self.fourier2k,
                                     v=self.fourier2k,
                                     p=self.fourier2x, s=self.fourier2b,
                                     w=self.fourier2d)

    def __call__(self, theta, w=1, wc=1):
        spreadfun = self._spreadfun[self.type[0]]
        return spreadfun(theta, w, wc)

    def chk_input(self, theta, w=1, wc=1):  # [s_par,TH,phi0,Nt] =
        ''' CHK_INPUT

        CALL [s_par,TH,phi0,Nt] = inputchk(theta,w,wc)
        '''

        wn = atleast_1d(w / wc)
        theta = theta.ravel()
        Nt = len(theta)

        # Make sure theta is from -pi to pi
        phi0 = 0.0
        theta = mod(theta + pi, 2 * pi) - pi

        if hasattr(self.theta0, '__call__'):
            th0 = self.theta0(wn.flatten())
        else:
            th0 = atleast_1d(self.theta0).flatten()

        Nt0 = th0.size
        Nw = wn.size
        isFreqDepDir = (Nt0 == Nw)
        if isFreqDepDir:
            # frequency dependent spreading and/or
            # frequency dependent direction
            # make sure -pi<=TH<pi
            TH = mod(theta[:, newaxis] - th0[newaxis, :] + pi, 2 * pi) - pi
        elif Nt0 != 1:
            raise ValueError(
                'The length of theta0 must equal to 1 or the length of w')
        else:
            TH = mod(theta - th0 + pi, 2 * pi) - pi  # make sure -pi<=TH<pi
            if self.method is not None:  # frequency dependent spreading
                TH = TH[:, newaxis]

        # Get spreading parameter
        s = self.spread_par_s(wn)

        if self.type[0] == 'c':  # cos2s
            s_par = s
        else:
            # First Fourier coefficient of the directional spreading function.
            r1 = abs(s / (s + 1))
            # Find distribution parameter from first Fourier coefficient.
            s_par = self.fourier2distpar(r1)
        if self.method is not None:
            s_par = s_par[newaxis, :]
        return s_par, TH, phi0, Nt

    def cos2s(self, theta, w=1, wc=1):  # [D, phi0] =
        ''' COS2S spreading function

        cos2s(theta,w) =  N(S)*[cos((theta-theta0)/2)]^(2*S)  (0 < S)

        where N() is a normalization factor and S is the spreading parameter
        possibly dependent on w.

        Parameters
        ----------
        theta, w : arrays
            angles and angular frequencies given in radians and rad/s,
            respectively. Lenghts are Nt and Nw.

        Returns
        -------
        D : 2D array
            Directonal spreading function. size Nt X Nw.
            The principal direction of D is always along the x-axis.
        phi0 : real scalar
            Parameter defining the actual principal direction of D.
        '''
        S, TH, phi0, unused_Nt = self.chk_input(theta, w, wc)

        gammaln = sp.gammaln

        D = (exp(gammaln(S + 1) - gammaln(S + 1.0 / 2.0)) / (2 * sqrt(pi))) * \
            cos(TH / 2.0) ** (2.0 * S)
        return D, phi0

    def poisson(self, theta, w=1, wc=1):  # [D,phi0] =
        ''' POISSON spreading function

        poisson(theta,w) =   N(X)/(1-2*X*cos(theta-theta0)+X^2)  (0 < X < 1)

        where N() is a normalization factor and X is the spreading parameter
        possibly dependent on w.

        Parameters
        ----------
        theta, w : arrays
            angles and angular frequencies given in radians and rad/s,
            respectively. Lenghts are Nt and Nw.

        Returns
        -------
        D : 2D array
            Directonal spreading function. size Nt X Nw.
            The principal direction of D is always along the x-axis.
        phi0 : real scalar
            Parameter defining the actual principal direction of D.
        '''
        [X, TH, phi0, unused_Nt] = self.chk_input(theta, w, wc)

        D = (1 - X ** 2.) / (1. - (2. * cos(TH) - X) * X) / (2. * pi)
        return D, phi0

    def wrap_norm(self, theta, w=1, wc=1):
        ''' Wrapped Normal spreading function

        wnormal(theta,w) = N(D1)*[1 +
                        2*sum exp(-(n*D1)^2/2)*cos(n*(theta-theta0))]  (0 < D1)

        where N() is a normalization factor and D1 is the spreading parameter
        possibly dependent on w.

        Parameters
        ----------
        theta, w : arrays
            angles and angular frequencies given in radians and rad/s,
            respectively. Lenghts are Nt and Nw.

        Returns
        -------
        D : 2D array
            Directonal spreading function. size Nt X Nw.
            The principal direction of D is always along the x-axis.
        phi0 : real scalar
            Parameter defining the actual principal direction of D.
        '''

        [par, TH, phi0, Nt] = self.chk_input(theta, w, wc)

        D1 = par ** 2. / 2.

        ix = arange(1, Nt)
        ix2 = ix ** 2
        Nd2 = D1.size
        Fcof = vstack((ones((1, Nd2)) / 2, exp(-ix2[:, newaxis] * D1))) / pi

        cor = exp(1j * ix[:, newaxis] * TH[0, :])
        # correction term to get
        Pcor = vstack((ones((1, TH.shape[1])), cor))
        # the correct integration limits
        Fcof = Fcof * Pcor.conj()
        D = real(fft(Fcof, axis=0))
        D[D < 0] = 0
        return D, phi0

    def sech2(self, theta, w=1, wc=1):
        '''SECH2 directonal spreading function

            sech2(theta,w) = N(B)*0.5*B*sech(B*(theta-theta0))^2 (0 < B)

        where N() is a normalization factor and X is the spreading parameter
        possibly dependent on w.

        Parameters
        ----------
        theta, w : arrays
            angles and angular frequencies given in radians and rad/s,
            respectively. Lenghts are Nt and Nw.

        Returns
        -------
        D : 2D array
            Directonal spreading function. size Nt X Nw.
            The principal direction of D is always along the x-axis.
        phi0 : real scalar
            Parameter defining the actual principal direction of D.
        '''

        [B, TH, phi0, unused_Nt] = self.chk_input(theta, w, wc)
        NB = tanh(pi * B)  # % Normalization factor.
        NB = where(NB == 0, 1.0, NB)  # Avoid division by zero

        D = 0.5 * B * sech(B * TH) ** 2. / NB
        return D, phi0

    def mises(self, theta, w=1, wc=1):
        '''Mises spreading function

        mises(theta,w) =  N(K)*exp(K*cos(theta-theta0))       (0 < K)

        where N() is a normalization factor and K is the spreading parameter
        possibly dependent on w.

        Parameters
        ----------
        theta, w : arrays
            angles and angular frequencies given in radians and rad/s,
            respectively. Lenghts are Nt and Nw.

        Returns
        -------
        D : 2D array
            Directonal spreading function. size Nt X Nw.
            The principal direction of D is always along the x-axis.
        phi0 : real scalar
            Parameter defining the actual principal direction of D.
        '''

        [K, TH, phi0, unused_Nt] = self.chk_input(theta, w, wc)

        D = exp(K * (cos(TH) - 1.)) / (2 * pi * sp.ive(0, K))
        return D, phi0

    def box(self, theta, w=1, wc=1):
        ''' Box car spreading function

            box(theta,w) =  N(A)*I( -A < theta-theta0 < A)      (0 < A < pi)

        where N() is a normalization factor and A is the spreading parameter
        possibly dependent on w.

        Parameters
        ----------
        theta, w : vectors
            angles and angular frequencies given in radians and rad/s,
            respectively. Lenghts are Nt and Nw.

        Returns
        -------
        D : 2D array
            Directonal spreading function. size Nt X Nw.
            The principal direction of D is always along the x-axis.
        phi0 : real scalar
            Parameter defining the actual principal direction of D.
        '''

        [A, TH, phi0, unused_Nt] = self.chk_input(theta, w, wc)
        D = ((-A <= TH) & (TH <= A)) / (2. * A)
        return D, phi0

# Local sub functions

    def fourier2distpar(self, r1):
        ''' Fourier coefficients to distribution parameter

        Parameters
        ----------
         r1   = corresponding fourier coefficient.
         type = string defining spreading function
                'box'
                'mises'
                'poisson'
                'sech2'
                'wnormal'
         Returns
          x    = distribution parameter

         The S-parameter of the COS-2S spreading function is used as a measure
         of spread in MKSPREADING. All the parameters of the other
         distributions are related to this S-parameter through the first
         Fourier coefficient, R1, of the directional distribution as follows:
                 R1 = S/(S+1) or S = R1/(1-R1).
         where
                 Box-car spreading  : R1 = sin(A)/A
                 Von Mises spreading: R1 = besseli(1,K)/besseli(0,K),
                 Poisson spreading  : R1 = X
                 sech-2 spreading   : R1 = pi/(2*B*sinh(pi/(2*B))
                 Wrapped Normal     : R1 = exp(-D1^2/2)
        '''
        fourierfun = self._fourierdispatch.get(self.type[0])
        return fourierfun(r1)

    def fourier2x(self, r1):
        ''' Returns the solution of r1 = x.
        '''
        X = r1
        if any(X >= 1):
            raise ValueError('POISSON spreading: X value must be less than 1')
        return X

    def fourier2a(self, r1):
        ''' Returns the solution of R1 = sin(A)/A.
        '''
        A0 = flipud(linspace(0, pi + 0.1, 1025))
        funA = interp1d(sinc(A0 / pi), A0)
        A0 = funA(r1.ravel())
        A = asarray(A0)

        # Newton-Raphson
        da = ones_like(r1)

        max_count = 100
        ix = flatnonzero(A)
        for unused_iy in range(max_count):
            Ai = A[ix]
            da[ix] = (sin(Ai) - Ai * r1[ix]) / (cos(Ai) - r1[ix])
            Ai = Ai - da[ix]
            # Make sure that the current guess is larger than zero.
            A[ix] = Ai + 0.5 * (da[ix] - Ai) * (Ai <= 0.0)

            ix = flatnonzero(
                (abs(da) > sqrt(_EPS) * abs(A)) * (abs(da) > sqrt(_EPS)))
            if ix.size == 0:
                if any(A > pi):
                    raise ValueError(
                        'BOX-CAR spreading: The A value must be less than pi')
                return A.clip(min=1e-16, max=pi)

        warnings.warn('Newton raphson method did not converge.')
        return A.clip(min=1e-16)  # Avoid division by zero

    def fourier2k(self, r1):
        '''
        Returns the solution of R1 = besseli(1,K)/besseli(0,K),
        '''
        def fun0(x):
            return sp.ive(1, x) / sp.ive(0, x)

        K0 = hstack((linspace(0, 10, 513), linspace(10.00001, 100)))
        funK = interp1d(fun0(K0), K0)
        K0 = funK(r1.ravel())
        k1 = flatnonzero(isnan(K0))
        if (k1.size > 0):
            K0[k1] = 0.0
            K0[k1] = K0.max()

        ix0 = flatnonzero(r1 != 0.0)
        K = zeros_like(r1)
        for ix in ix0:
            K[ix] = optimize.fsolve(lambda x: fun0(x) - r1[ix], K0[ix])
        return K

    def fourier2b(self, r1):
        ''' Returns the solution of R1 = pi/(2*B*sinh(pi/(2*B)).
        '''
        B0 = hstack((linspace(_EPS, 5, 513), linspace(5.0001, 100)))
        funB = interp1d(self._r1ofsech2(B0), B0)

        B0 = funB(r1.ravel())
        k1 = flatnonzero(isnan(B0))
        if (k1.size > 0):
            B0[k1] = 0.0
            B0[k1] = max(B0)

        ix0 = flatnonzero(r1 != 0.0)
        B = zeros_like(r1)

        def fun(x):
            return 0.5 * pi / (sinh(.5 * pi / x)) - x * r1[ix]
        for ix in ix0:
            B[ix] = abs(optimize.fsolve(fun, B0[ix]))
        return B

    def fourier2d(self, r1):
        ''' Returns the solution of R1 = exp(-D**2/2).
        '''
        r = clip(r1, 0., 1.0)
        return where(r <= 0, inf, sqrt(-2.0 * log(r)))

    def spread_par_s(self, wn):
        ''' Return spread parameter, S, of COS2S function

        Parameters
        ----------
        wn  : array_like
            normalized frequencies.
        Returns
        -------
        S   : ndarray
            spread parameter of COS2S functions
        '''
        if self.method is None:
            # no frequency dependent spreading,
            # but possible frequency dependent direction
            s = atleast_1d(self.s_a)
        else:
            wn_lo = self.wn_lo
            wn_up = self.wn_up
            wn_c = self.wn_c

            spa = self.s_a
            spb = self.s_b
            ma = self.m_a
            mb = self.m_b

            # Mitsuyasu et. al and Hasselman et. al parametrization   of
            # frequency dependent spreading
            s = where(wn <= wn_c, spa * wn ** ma, spb * wn ** mb)
            s[wn <= wn_lo] = 0.0

            k = flatnonzero(wn_up < wn)
            if k.size > 0:
                if self.method[0] == 'd':
                    # Donelan et. al. parametrization for B in SECH-2
                    s[k] = spb * (wn_up) ** mb

                    # Convert to S-paramater in COS-2S distribution
                    r1 = self.r1ofsech2(s)
                    s = r1 / (1. - r1)

                elif self.method[0] == 'b':
                    # Banner parametrization  for B in SECH-2
                    s3m = spb * (wn_up) ** mb
                    s3p = self._donelan(wn_up)
                    #  Scale so that parametrization will be continous
                    scale = s3m / s3p
                    s[k] = scale * self.donelan(wn[k])
                    r1 = self.r1ofsech2(s)

                    # Convert to S-paramater in COS-2S distribution
                    s = r1 / (1. - r1)
                else:
                    s[k] = 0.0

        if any(s < 0):
            raise ValueError('The COS2S spread parameter, S(w), ' +
                             'value must be larger than 0')
        return s

    def _donelan(self, wn):
        ''' High frequency decay of B of sech2 paramater
        '''
        return 10.0 ** (-0.4 + 0.8393 * exp(-0.567 * log(wn ** 2)))

    def _r1ofsech2(self, B):
        ''' R1OFSECH2   Computes R1 = pi./(2*B.*sinh(pi./(2*B)))
        '''
        realmax = finfo(float).max
        tiny = 1. / realmax
        x = clip(2. * B, tiny, realmax)
        xk = pi / x
        return where(x < 100., xk / sinh(xk),
                     -2. * xk / (exp(xk) * expm1(-2. * xk)))

    def tospecdata2d(self, specdata=None, theta=None, wc=0.52, nt=51):
        '''
         MKDSPEC Make a directional spectrum
                 frequency spectrum times spreading function

         CALL:  Snew=mkdspec(S,D,plotflag)

               Snew = directional spectrum (spectrum struct)
               S    = frequency spectrum (spectrum struct)
                          (default jonswap)
               D    = spreading function (special struct)
                          (default spreading([],'cos2s'))
               plotflag = 1: plot the spectrum, else: do not plot (default 0)

         Creates a directional spectrum through multiplication of a frequency
         spectrum and a spreading function: S(w,theta)=S(w)*D(w,theta)

         The spreading structure must contain the following fields:
           .S (size [np 1] or [np nf])  and  .theta (length np)
         optional fields: .w (length nf), .note (memo) .phi (rotation-azymuth)

         NB! S.w and D.w (if any) must be identical.

         Example
         -------
         >>> import wafo.spectrum.models as wsm
         >>> S = wsm.Jonswap().tospecdata()
         >>> D = wsm.Spreading('cos2s')
         >>> SD = D.tospecdata2d(S)
         >>> h = SD.plot()

         See also  spreading, rotspec, jonswap, torsethaugen
        '''

        if specdata is None:
            specdata = Jonswap().tospecdata()
        if theta is None:
            pi = np.pi
            theta = np.linspace(-pi, pi, nt)
        else:
            L = abs(theta[-1] - theta[0])
            if abs(L - pi) > _EPS:
                raise ValueError('theta must cover all angles -pi -> pi')
            nt = len(theta)

        if nt < 40:
            warnings.warn('Number of angles is less than 40. ' +
                          'Spreading too sparsely sampled!')

        w = specdata.args
        S = specdata.data
        D, phi0 = self(theta, w=w, wc=wc)
        if D.ndim != 2:  # frequency dependent spreading
            D = D[:, None]
        SD = D * S[None, :]

        Snew = SpecData2D(SD, (w, theta), type='dir',
                          freqtype=specdata.freqtype)
        Snew.tr = specdata.tr
        Snew.h = specdata.h
        Snew.phi = phi0
        Snew.norm = specdata.norm
        # Snew.note = specdata.note + ', spreading: %s' % self.type
        return Snew


def _test_some_spectra():
    S = Jonswap()

    w = arange(3.0)
    S(w) * phi1(w, 30.0)
    S1 = S.tospecdata(w)
    S1.plot()

    import pylab as plb
    w = plb.linspace(0, 2.5)
    S = Tmaspec(h=10, gamma=1)  # Bretschneider spectrum Hm0=7, Tp=11
    plb.plot(w, S(w))
    plb.plot(w, S(w, h=21))
    plb.plot(w, S(w, h=42))
    plb.show()
    plb.close('all')

    w, th = plb.ogrid[0:4, 0:6]
    k, k2 = w2k(w, th)

    plb.plot(w, k, w, k2)

    plb.show()

    plb.close('all')
    w = plb.linspace(0, 2, 100)
    S = Torsethaugen(Hm0=6, Tp=8)
    plb.plot(w, S(w), w, S.wind(w), w, S.swell(w))

    S1 = Jonswap(Hm0=7, Tp=11, gamma=1)
    w = plb.linspace(0, 2, 100)
    plb.plot(w, S1(w))
    plb.show()
    plb.close('all')

    Hm0 = plb.arange(1, 11)
    Tp = plb.linspace(2, 16)
    T, H = plb.meshgrid(Tp, Hm0)
    gam = jonswap_peakfact(H, T)
    plb.plot(Tp, gam.T)
    plb.xlabel('Tp [s]')
    plb.ylabel('Peakedness parameter')

    Hm0 = plb.linspace(1, 20)
    Tp = Hm0
    [T, H] = plb.meshgrid(Tp, Hm0)
    gam = jonswap_peakfact(H, T)
    v = plb.arange(0, 8)
    plb.contourf(Tp, Hm0, gam, v)
    plb.colorbar()
    plb.show()
    plb.close('all')


def _test_spreading():
    import pylab as plb
    pi = plb.pi
    w = plb.linspace(0, 3, 257)
    theta = plb.linspace(-pi, pi, 129)

    D2 = Spreading('cos2s', theta0=lambda w: w * plb.pi / 6.0)
    d1 = D2(theta, w)[0]
    _t = plb.contour(d1.squeeze())

    pi = plb.pi
    D = Spreading('wrap_norm', s_a=10.0)

    w = plb.linspace(0, 3, 257)
    theta = plb.linspace(-pi, pi, 129)
    d1 = D(theta, w)
    plb.contour(d1[0])
    plb.show()


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


def main():
    if False:  # True: #
        _test_some_spectra()
    else:
        test_docstrings()

if __name__ == '__main__':
    main()
