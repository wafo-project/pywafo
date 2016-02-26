from __future__ import absolute_import, division
import warnings
import os
import numpy as np
from numpy import (pi, inf, zeros, ones, where, nonzero,
                   flatnonzero, ceil, sqrt, exp, log, arctan2,
                   tanh, cosh, sinh, random, atleast_1d,
                   minimum, diff, isnan, any, r_, conj, mod,
                   hstack, vstack, interp, ravel, finfo, linspace,
                   arange, array, nan, newaxis, sign)
from numpy.fft import fft
from scipy.integrate import simps, trapz
from scipy.special import erf
from scipy.linalg import toeplitz
import scipy.interpolate as interpolate
from scipy.interpolate.interpolate import interp1d, interp2d
from ..misc import meshgrid, gravity, cart2polar, polar2cart
from ..objects import TimeSeries, mat2timeseries
from ..interpolate import stineman_interp
from ..wave_theory.dispersion_relation import w2k  # , k2w
from ..containers import PlotData, now
from ..misc import sub_dict_select, nextpow2, discretize, JITImport
from ..kdetools import qlevels

# from wafo.transform import TrData
from ..transform.models import TrLinear
from ..plotbackend import plotbackend

try:
    from ..gaussian import Rind
except ImportError:
    Rind = None
try:
    from .. import c_library
except ImportError:
    warnings.warn('Compile the c_library.pyd again!')
    c_library = None
try:
    from .. import cov2mod
except ImportError:
    warnings.warn('Compile the cov2mod.pyd again!')
    cov2mod = None

# Trick to avoid error due to circular import
_WAFOCOV = JITImport('wafo.covariance')


__all__ = ['SpecData1D', 'SpecData2D', 'plotspec']


_EPS = np.finfo(float).eps


def _set_seed(iseed):
    '''Set seed of random generator'''
    if iseed is not None:
        try:
            random.set_state(iseed)
        except:
            random.seed(iseed)


def qtf(w, h=inf, g=9.81):
    """
    Return Quadratic Transfer Function

    Parameters
    ------------
    w : array-like
        angular frequencies
    h : scalar
        water depth
    g : scalar
        acceleration of gravity

    Returns
    -------
    h_s   = sum frequency effects
    h_d   = difference frequency effects
    h_dii = diagonal of h_d
    """
    w = atleast_1d(w)
    num_w = w.size

    k_w = w2k(w, theta=0, h=h, g=g)[0]

    k_1, k_2 = meshgrid(k_w, k_w)

    if h == inf:  # go here for faster calculations
        h_s = 0.25 * (abs(k_1) + abs(k_2))
        h_d = -0.25 * abs(abs(k_1) - abs(k_2))
        h_dii = zeros(num_w)
        return h_s, h_d, h_dii

    [w_1, w_2] = meshgrid(w, w)

    w12 = (w_1 * w_2)
    w1p2 = (w_1 + w_2)
    w1m2 = (w_1 - w_2)
    k12 = (k_1 * k_2)
    k1p2 = (k_1 + k_2)
    k1m2 = abs(k_1 - k_2)

    if 0:  # Langley
        p_1 = (-2 * w1p2 * (k12 * g ** 2. - w12 ** 2.) +
               w_1 * (w_2 ** 4. - g ** 2 * k_2 ** 2) +
               w_2 * (w_1 ** 4 - g * 2. * k_1 ** 2)) / (4. * w12)
        p_2 = w1p2 ** 2. * cosh((k1p2) * h) - g * (k1p2) * sinh((k1p2) * h)

        h_s = (-p_1 / p_2 * w1p2 * cosh((k1p2) * h) / g -
               (k12 * g ** 2 - w12 ** 2.) / (4 * g * w12) +
               (w_1 ** 2 + w_2 ** 2) / (4 * g))

        p_3 = (-2 * w1m2 * (k12 * g ** 2 + w12 ** 2) -
               w_1 * (w_2 ** 4 - g ** 2 * k_2 ** 2) +
               w_2 * (w_1 ** 4 - g ** 2 * k_1 ** 2)) / (4. * w12)
        p_4 = w1m2 ** 2. * cosh(k1m2 * h) - g * (k1m2) * sinh((k1m2) * h)

        h_d = (-p_3 / p_4 * (w1m2) * cosh((k1m2) * h) / g -
               (k12 * g ** 2 + w12 ** 2) / (4 * g * w12) +
               (w_1 ** 2. + w_2 ** 2.) / (4. * g))

    else:  # Marthinsen & Winterstein
        tmp1 = 0.5 * g * k12 / w12
        tmp2 = 0.25 / g * (w_1 ** 2. + w_2 ** 2. + w12)
        h_s = (tmp1 - tmp2 + 0.25 * g * (w_1 * k_2 ** 2. + w_2 * k_1 ** 2) /
               (w12 * (w1p2))) / (1. - g * (k1p2) / (w1p2) ** 2. *
                                  tanh((k1p2) * h)) + tmp2 - 0.5 * tmp1  # OK

        tmp2 = 0.25 / g * (w_1 ** 2 + w_2 ** 2 - w12)  # OK
        h_d = (tmp1 - tmp2 - 0.25 * g * (w_1 * k_2 ** 2 - w_2 * k_1 ** 2) /
               (w12 * (w1m2))) / (1. - g * (k1m2) / (w1m2) ** 2. *
                                  tanh((k1m2) * h)) + tmp2 - 0.5 * tmp1  # OK

    # tmp1 = 0.5*g*k_w./(w.*sqrt(g*h))
    # tmp2 = 0.25*w.^2/g
    # Wave group velocity
    c_g = 0.5 * g * (tanh(k_w * h) + k_w * h * (1.0 - tanh(k_w * h) ** 2)) / w
    h_dii = (0.5 * (0.5 * g * (k_w / w) ** 2. - 0.5 * w ** 2 / g +
                    g * k_w / (w * c_g)) /
             (1. - g * h / c_g ** 2.) - 0.5 * k_w / sinh(2 * k_w * h))  # OK
    h_d.flat[0::num_w + 1] = h_dii

    # k    = find(w_1==w_2)
    # h_d(k) = h_dii

    # The NaN's occur due to division by zero. => Set the isnans to zero

    h_dii = where(isnan(h_dii), 0, h_dii)
    h_d = where(isnan(h_d), 0, h_d)
    h_s = where(isnan(h_s), 0, h_s)

    return h_s, h_d, h_dii


def plotspec(specdata, linetype='b-', flag=1):
    '''
    PLOTSPEC Plot a spectral density

    Parameters
    ----------
    S : SpecData1D or SpecData2D object
        defining spectral density.
    linetype : string
        defining color and linetype, see plot for possibilities
    flag : scalar integer
        defining the type of plot
        1D:
            1 plots the density, S, (default)
            2 plot 10log10(S)
            3 plots both the above plots
        2D:
        Directional spectra: S(w,theta), S(f,theta)
            1 polar plot S (default)
            2 plots spectral density and the directional
                    spreading, int S(w,theta) dw or int S(f,theta) df
            3 plots spectral density and the directional
                   spreading, int S(w,theta)/S(w) dw or int S(f,theta)/S(f) df
            4 mesh of S
            5 mesh of S in polar coordinates
            6 contour plot of S
            7 filled contour plot of S
        Wavenumber spectra: S(k1,k2)
            1 contour plot of S (default)
            2 filled contour plot of S

    Example
    -------
    >>> import numpy as np
    >>> import wafo.spectrum as ws
    >>> Sj = ws.models.Jonswap(Hm0=3, Tp=7)
    >>> S = Sj.tospecdata()
    >>> ws.plotspec(S,flag=1)

      S = demospec('dir'); S2 = mkdspec(jonswap,spreading);
      plotspec(S,2), hold on
      # Same as previous fig. due to frequency independent spreading
      plotspec(S,3,'g')
      # Not the same as previous figs. due to frequency dependent spreading
      plotspec(S2,2,'r')
      plotspec(S2,3,'m')
      # transform from angular frequency and radians to frequency and degrees
      Sf = ttspec(S,'f','d'); clf
      plotspec(Sf,2),

    See also
    dat2spec, createspec, simpson
    '''
    pass
#     # label the contour levels
#     txtFlag = 0
#     LegendOn = 1
#
#     ftype = specdata.freqtype  # options are 'f' and 'w' and 'k'
#     data = specdata.data
#     if data.ndim == 2:
#         freq = specdata.args[1]
#         theta = specdata.args[0]
#     else:
#         freq = specdata.args
#     if isinstance(specdata.args, (list, tuple)):
#
#     if ftype == 'w':
#         xlbl_txt = 'Frequency [rad/s]'
#         ylbl1_txt = 'S(w) [m^2 s / rad]'
#         ylbl3_txt = 'Directional Spectrum'
#         zlbl_txt = 'S(w,\theta) [m^2 s / rad^2]'
#         funit = ' [rad/s]'
#     Sunit     = ' [m^2 s / rad]'
#     elif ftype == 'f':
#         xlbl_txt = 'Frequency [Hz]'
#         ylbl1_txt = 'S(f) [m^2 s]'
#         ylbl3_txt = 'Directional Spectrum'
#         zlbl_txt = 'S(f,\theta) [m^2 s / rad]'
#         funit = ' [Hz]'
#     Sunit     = ' [m^2 s ]'
#     elif ftype == 'k':
#         xlbl_txt = 'Wave number [rad/m]'
#         ylbl1_txt = 'S(k) [m^3/ rad]'
#         funit = ' [rad/m]'
#     Sunit     = ' [m^3 / rad]'
#         ylbl4_txt = 'Wave Number Spectrum'
#
#     else:
#         raise ValueError('Frequency type unknown')
#
#
#     if hasattr(specdata, 'norm') and specdata.norm :
#         Sunit=[]
#         funit = []
#         ylbl1_txt = 'Normalized Spectral density'
#         ylbl3_txt = 'Normalized Directional Spectrum'
#         ylbl4_txt = 'Normalized Wave Number Spectrum'
#         if ftype == 'k':
#             xlbl_txt = 'Normalized Wave number'
#         else:
#             xlbl_txt = 'Normalized Frequency'
#
#     ylbl2_txt = 'Power spectrum (dB)'
#
#     phi = specdata.phi
#
#     spectype = specdata.type.lower()
#     stype = spectype[-3::]
#     if stype in ('enc', 'req', 'k1d') : #1D plot
#     Fn = freq[-1] # Nyquist frequency
#     indm = findpeaks(data, n=4)
#     maxS = data.max()
#     if isfield(S,'CI') && ~isempty(S.CI):
#         maxS  = maxS*S.CI(2)
#         txtCI = [num2str(100*S.p), '% CI']
#         #end
#
#     Fp = freq[indm]# %peak frequency/wave number
#
#     if len(indm) == 1:
#         txt = [('fp = %0.2g' % Fp) + funit]
#     else:
#         txt = []
#         for i, fp in enumerate(Fp.tolist()):
#             txt.append(('fp%d = %0.2g' % (i, fp)) + funit)
#
#     txt = ''.join(txt)
#     if (flag == 3):
#         plotbackend.subplot(2, 1, 1)
#     if (flag == 1) or (flag == 3):#  Plot in normal scale
#         plotbackend.plot(np.vstack([Fp, Fp]),
#             np.vstack([zeros(len(indm)), data.take(indm)]),
#             ':', label=txt)
#         plotbackend.plot(freq, data, linetype)
#         specdata.labels.labelfig()
#     if isfield(S,'CI'):
#         plot(freq,S.S*S.CI(1), 'r:' )
#         plot(freq,S.S*S.CI(2), 'r:' )
#
#         a = plotbackend.axis()
#
#         a1 = Fn
#         if (Fp > 0):
#             a1 = max(min(Fn, 10 * max(Fp)), a[1])
#
#         plotbackend.axis([0, a1 , 0, max(1.01 * maxS, a[3])])
#     plotbackend.title('Spectral density')
#     plotbackend.xlabel(xlbl_txt)
#     plotbackend.ylabel(ylbl1_txt)
#
#
#     if (flag == 3):
#         plotbackend.subplot(2, 1, 2)
#
#     if (flag == 2) or (flag == 3) : # Plot in logaritmic scale
#         ind = np.flatnonzero(data > 0)
#
#         plotbackend.plot(np.vstack([Fp, Fp]),
#                       np.vstack((min(10 * log10(data.take(ind) /
#                                       maxS)).repeat(len(Fp)),
#                       10 * log10(data.take(indm) / maxS))), ':',label=txt)
#     hold on
#     if isfield(S,'CI'):
#         plot(freq(ind),10*log10(S.S(ind)*S.CI(1)/maxS), 'r:' )
#         plot(freq(ind),10*log10(S.S(ind)*S.CI(2)/maxS), 'r:' )
#
#     plotbackend.plot(freq[ind], 10 * log10(data[ind] / maxS), linetype)
#
#     a = plotbackend.axis()
#
#     a1 = Fn
#     if (Fp > 0):
#         a1 = max(min(Fn, 10 * max(Fp)), a[1])
#
#     plotbackend.axis([0, a1 , -20, max(1.01 * 10 * log10(1), a[3])])
#
#     specdata.labels.labelfig()
#     plotbackend.title('Spectral density')
#     plotbackend.xlabel(xlbl_txt)
#     plotbackend.ylabel(ylbl2_txt)
#
#         if LegendOn:
#             plotbackend.legend()
#             if isfield(S,'CI'),
#                 legend(txt{:},txtCI,1)
#             else
#                 legend(txt{:},1)
#                 end
#         end
#       case {'k2d'}
#         if plotflag==1,
#           [c, h] = contour(freq,S.k2,S.S,'b')
#           z_level = clevels(c)
#
#
#           if txtFlag==1
#             textstart_x=0.05; textstart_y=0.94
#             cltext1(z_level,textstart_x,textstart_y)
#           else
#             cltext(z_level,0)
#           end
#         else
#           [c,h] = contourf(freq,S.k2,S.S)
#           %clabel(c,h), colorbar(c,h)
#           fcolorbar(c) % alternative
#         end
#         rotate(h,[0 0 1],-phi*180/pi)
#
#
#
#         xlabel(xlbl_txt)
#         ylabel(xlbl_txt)
#         title(ylbl4_txt)
#         # return
#         km=max([-freq(1) freq(end) S.k2(1) -S.k2(end)])
#         axis([-km km -km km])
#         hold on
#         plot([0 0],[ -km km],':')
#         plot([-km km],[0 0],':')
#         axis('square')
#
#
#         # cltext(z_level)
#         # axis('square')
#         if ~ih, hold off,end
#       case {'dir'}
#         thmin = S.theta(1)-phi;thmax=S.theta(end)-phi
#         if plotflag==1 % polar plot
#           if 0, % alternative but then z_level must be chosen beforehand
#             h = polar([0 2*pi],[0 freq(end)])
#             delete(h);hold on
#             [X,Y]=meshgrid(S.theta,freq)
#             [X,Y]=polar2cart(X,Y)
#             contour(X,Y,S.S',lintype)
#           else
#             if (abs(thmax-thmin)<3*pi), % angle given in radians
#               theta = S.theta
#             else
#               theta = S.theta*pi/180 % convert to radians
#               phi  = phi*pi/180
#             end
#             c = contours(theta,freq,S.S')%,Nlevel) % calculate levels
#             if isempty(c)
#               c = contours(theta,freq,S.S)%,Nlevel); % calculate levels
#             end
#             [z_level c] = clevels(c); % find contour levels
#             h = polar(c(1,:),c(2,:),lintype);
#             rotate(h,[0 0 1],-phi*180/pi)
#           end
#           title(ylbl3_txt)
#           % label the contour levels
#
#           if txtFlag==1
#             textstart_x = -0.1; textstart_y=1.00;
#             cltext1(z_level,textstart_x,textstart_y);
#           else
#             cltext(z_level,0)
#           end
#
#         elseif (plotflag==2) || (plotflag==3),
#           %ih = ishold;
#
#           subplot(211)
#
#           if ih, hold on, end
#
#           Sf = spec2spec(S,'freq'); % frequency spectrum
#           plotspec(Sf,1,lintype)
#
#           subplot(212)
#
#           Dtf        = S.S;
#           [Nt,Nf]    = size(S.S);
#           Sf         = Sf.S(:).';
#           ind        = find(Sf);
#
#           if plotflag==3, %Directional distribution  D(theta,freq))
#             Dtf(:,ind) = Dtf(:,ind)./Sf(ones(Nt,1),ind);
#           end
#           Dtheta  = simpson(freq,Dtf,2); %Directional spreading, D(theta)
#           Dtheta  = Dtheta/simpson(S.theta,Dtheta); % make sure int D(theta)dtheta = 1
#           [y,ind] = max(Dtheta);
#           Wdir    = S.theta(ind)-phi; % main wave direction
#           txtwdir = ['\theta_p=' num2pistr(Wdir,3)]; % convert to text string
#
#           plot([1 1]*S.theta(ind)-phi,[0 Dtheta(ind)],':'), hold on
#           if LegendOn
#             lh=legend(txtwdir,0);
#           end
#           plot(S.theta-phi,Dtheta,lintype)
#
#           fixthetalabels(thmin,thmax,'x',2)  % fix xticklabel and xlabel for theta
#           ylabel('D(\theta)')
#           title('Spreading function')
#           if ~ih, hold off, end
#           %legend(lh) % refresh current legend
#         elseif plotflag==4 % mesh
#           mesh(freq,S.theta-phi,S.S)
#           xlabel(xlbl_txt);
#           fixthetalabels(thmin,thmax,'y',3) % fix yticklabel and ylabel for theta
#           zlabel(zlbl_txt)
#           title(ylbl3_txt)
#         elseif plotflag==5 % mesh
#           %h=polar([0 2*pi],[0 freq(end)]);
#           %delete(h);hold on
#           [X,Y]=meshgrid(S.theta-phi,freq);
#           [X,Y]=polar2cart(X,Y);
#           mesh(X,Y,S.S')
#           % display the unit circle beneath the surface
#           hold on, mesh(X,Y,zeros(size(S.S'))),hold off
#           zlabel(zlbl_txt)
#           title(ylbl3_txt)
#           set(gca,'xticklabel','','yticklabel','')
#           lighting phong
#           %lighting gouraud
#           %light
#         elseif (plotflag==6) || (plotflag==7),
#           theta = S.theta-phi;
#           [c, h] = contour(freq,theta,S.S); %,Nlevel); % calculate levels
#           fixthetalabels(thmin,thmax,'y',2) % fix yticklabel and ylabel for theta
#           if plotflag==7,
#             hold on
#             [c,h] =    contourf(freq,theta,S.S); %,Nlevel); % calculate levels
#             %hold on
#           end
#
#           title(ylbl3_txt)
#           xlabel(xlbl_txt);
#           if 0,
#             [z_level] = clevels(c); % find contour levels
#             % label the contour levels
#             if txtFlag==1
#               textstart_x = 0.06; textstart_y=0.94;
#               cltext1(z_level,textstart_x,textstart_y) % a local variant of cltext
#             else
#               cltext(z_level)
#             end
#           else
#             colormap('jet')
#
#             if plotflag==7,
#               fcolorbar(c)
#             else
#               %clabel(c,h),
#               hcb = colorbar;
#             end
#             grid on
#           end
#         else
#           error('Unknown plot option')
#         end
#       otherwise, error('unknown spectral type')
#     end
#
#     if ~ih, hold off, end
#
#     #  The following two commands install point-and-click editing of
#     #   all the text objects (title, xlabel, ylabel) of the current figure:
#
#     #set(findall(gcf,'type','text'),'buttondownfcn','edtext')
#     #set(gcf,'windowbuttondownfcn','edtext(''hide'')')
#
#     return


class SpecData1D(PlotData):

    """
    Container class for 1D spectrum data objects in WAFO

    Member variables
    ----------------
    data : array-like
        One sided Spectrum values, size nf
    args : array-like
        freguency/wave-number-lag values of freqtype, size nf
    type : String
        spectrum type, one of 'freq', 'k1d', 'enc' (default 'freq')
    freqtype : letter
        frequency type, one of: 'f', 'w' or 'k' (default 'w')
    tr : Transformation function (default (none)).
    h : real scalar
        Water depth (default inf).
    v : real scalar
        Ship speed, if type = 'enc'.
    norm : bool
        Normalization flag, True if S is normalized, False if not
    date : string
        Date and time of creation or change.

    Examples
    --------
    >>> import numpy as np
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=3)
    >>> w = np.linspace(0,4,256)
    >>> S1 = Sj.tospecdata(w)   #Make spectrum object from numerical values
    >>> S = sm.SpecData1D(Sj(w),w) # Alternatively do it manually

    See also
    --------
    PlotData
    CovData
    """

    def __init__(self, *args, **kwds):
        self.name_ = kwds.pop('name', 'WAFO Spectrum Object')
        self.type = kwds.pop('type', 'freq')
        self.freqtype = kwds.pop('freqtype', 'w')
        self.angletype = ''
        self.h = kwds.pop('h', inf)
        self.tr = kwds.pop('tr', None)  # TrLinear()
        self.phi = kwds.pop('phi', 0.0)
        self.v = kwds.pop('v', 0.0)
        self.norm = kwds.pop('norm', False)
        super(SpecData1D, self).__init__(*args, **kwds)

        self.setlabels()

    def _get_default_dt_and_rate(self, dt):
        dt_old = self.sampling_period()
        if dt is None:
            return dt_old, 1
        rate = max(round(dt_old * 1. / dt), 1.)
        return dt, rate

    def _check_dt(self, dt):
        freq = self.args
        checkdt = 1.2 * min(diff(freq)) / 2. / pi
        if self.freqtype in 'f':
                checkdt *= 2 * pi
        if (checkdt < 2. ** -16 / dt):
            print('Step dt = %g in computation of the density is ' +
                  'too small.' % dt)
            print('The computed covariance (by FFT(2^K)) may differ from the')
            print('theoretical. Solution:')
            raise ValueError('use larger dt or sparser grid for spectrum.')

    def _check_cov_matrix(self, acfmat, nt, dt):
        eps0 = 0.0001
        if nt + 1 >= 5:
            cc2 = acfmat[0, 0] - acfmat[4, 0] * (acfmat[4, 0] / acfmat[0, 0])
            if (cc2 < eps0):
                warnings.warn('Step dt = %g in computation of the density ' +
                              'is too small.' % dt)
        cc1 = acfmat[0, 0] - acfmat[1, 0] * (acfmat[1, 0] / acfmat[0, 0])
        if (cc1 < eps0):
            warnings.warn('Step dt = %g is small, and may cause numerical ' +
                          'inaccuracies.' % dt)

    @property
    def lagtype(self):
        if self.freqtype in 'k':  # options are 'f' and 'w' and 'k'
            return 'x'
        return 't'

    def tocov_matrix(self, nr=0, nt=None, dt=None):
        '''
        Computes covariance function and its derivatives, alternative version

        Parameters
        ----------
        nr : scalar integer
            number of derivatives in output, nr<=4          (default 0)
        nt : scalar integer
            number in time grid, i.e., number of time-lags.
            (default rate*(n_f-1)) where rate = round(1/(2*f(end)*dt)) or
                     rate = round(pi/(w(n_f)*dt)) depending on S.
        dt : real scalar
            time spacing for acfmat

        Returns
        -------
        acfmat : [R0, R1,...Rnr], shape Nt+1 x Nr+1
            matrix with autocovariance and its derivatives, i.e., Ri (i=1:nr)
            are column vectors with the 1'st to nr'th derivatives of R0.

        NB! This routine requires that the spectrum grid is equidistant
           starting from zero frequency.

        Example
        -------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap()
        >>> S = Sj.tospecdata()
        >>> acfmat = S.tocov_matrix(nr=3, nt=256, dt=0.1)
        >>> np.round(acfmat[:2,:],3)
        array([[ 3.061,  0.   , -1.677,  0.   ],
               [ 3.052, -0.167, -1.668,  0.187]])

        See also
        --------
        cov,
        resample,
        objects
        '''


        dt, rate = self._get_default_dt_and_rate(dt)
        self._check_dt(dt)

        freq = self.args
        n_f = len(freq)
        if nt is None:
            nt = rate * (n_f - 1)
        else:  # %check if Nt is ok
            nt = minimum(nt, rate * (n_f - 1))

        spec = self.copy()
        spec.resample(dt)

        acf = spec.tocovdata(nr, nt, rate=1)
        acfmat = zeros((nt + 1, nr + 1), dtype=float)
        acfmat[:, 0] = acf.data[0:nt + 1]
        fieldname = 'R' + self.lagtype * nr
        for i in range(1, nr + 1):
            fname = fieldname[:i + 1]
            r_i = getattr(acf, fname)
            acfmat[:, i] = r_i[0:nt + 1]

        self._check_cov_matrix(acfmat, nt, dt)
        return acfmat

    def tocovdata(self, nr=0, nt=None, rate=None):
        '''
        Computes covariance function and its derivatives

        Parameters
        ----------
        nr : number of derivatives in output, nr<=4 (default = 0).
        nt : number in time grid, i.e., number of time-lags
              (default rate*(length(S.data)-1)).
        rate = 1,2,4,8...2**r, interpolation rate for R
               (default = 1, no interpolation)

        Returns
        -------
        R : CovData1D
            auto covariance function

        The input 'rate' with the spectrum gives the time-grid-spacing:
            dt=pi/(S.w[-1]*rate),
            S.w[-1] is the Nyquist freq.
        This results in the time-grid: 0:dt:Nt*dt.

        What output is achieved with different S and choices of Nt, Nx and Ny:
        1) S.type='freq' or 'dir', Nt set, Nx,Ny not set => R(time) (one-dim)
        2) S.type='k1d' or 'k2d', Nt set, Nx,Ny not set: => R(x) (one-dim)
        3) Any type, Nt and Nx set => R(x,time); Nt and Ny set => R(y,time)
        4) Any type, Nt, Nx and Ny set => R(x,y,time)
        5) Any type, Nt not set, Nx and/or Ny set
            => Nt set to default, goto 3) or 4)

        NB! This routine requires that the spectrum grid is equidistant
         starting from zero frequency.
        NB! If you are using a model spectrum, spec, with sharp edges
         to calculate covariances then you should probably round off the sharp
         edges like this:

        Example:
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap()
        >>> S = Sj.tospecdata()
        >>> S.data[0:40] = 0.0
        >>> S.data[100:-1] = 0.0
        >>> Nt = len(S.data)-1
        >>> acf = S.tocovdata(nr=0, nt=Nt)
        >>> S1 = acf.tospecdata()
        >>> h = S.plot('r')
        >>> h1 = S1.plot('b:')

        R   = spec2cov(spec,0,Nt)
        win = parzen(2*Nt+1)
        R.data = R.data.*win(Nt+1:end)
        S1  = cov2spec(acf)
        R2  = spec2cov(S1)
        figure(1)
        plotspec(S),hold on, plotspec(S1,'r')
        figure(2)
        covplot(R), hold on, covplot(R2,[],[],'r')
        figure(3)
        semilogy(abs(R2.data-R.data)), hold on,
        semilogy(abs(S1.data-S.data)+1e-7,'r')

        See also
        --------
        cov2spec
        '''

        freq = self.args
        n_f = len(freq)

        if freq[0] > 0:
            txt = '''Spectrum does not start at zero frequency/wave number.
            Correct it with resample, for example.'''
            raise ValueError(txt)
        d_w = abs(diff(freq, n=2, axis=0))
        if any(d_w > 1.0e-8):
            txt = '''Not equidistant frequencies/wave numbers in spectrum.
            Correct it with resample, for example.'''
            raise ValueError(txt)

        if rate is None:
            rate = 1  # %interpolation rate
        elif rate > 16:
            rate = 16
        else:  # make sure rate is a power of 2
            rate = 2 ** nextpow2(rate)

        if nt is None:
            nt = rate * (n_f - 1)
        else:  # check if Nt is ok
            nt = minimum(nt, rate * (n_f - 1))

        spec = self.copy()

        if self.freqtype in 'k':
            lagtype = 'x'
        else:
            lagtype = 't'

        d_t = spec.sampling_period()
        # normalize spec so that sum(specn)/(n_f-1)=acf(0)=var(X)
        specn = spec.data * freq[-1]
        if spec.freqtype in 'f':
            w = freq * 2 * pi
        else:
            w = freq

        nfft = rate * 2 ** nextpow2(2 * n_f - 2)

        # periodogram
        rper = r_[
            specn, zeros(nfft - (2 * n_f) + 2), conj(specn[n_f - 2:0:-1])]
        time = r_[0:nt + 1] * d_t * (2 * n_f - 2) / nfft

        r = fft(rper, nfft).real / (2 * n_f - 2)
        acf = _WAFOCOV.CovData1D(r[0:nt + 1], time, lagtype=lagtype)
        acf.tr = spec.tr
        acf.h = spec.h
        acf.norm = spec.norm

        if nr > 0:
            w = r_[w, zeros(nfft - 2 * n_f + 2), -w[n_f - 2:0:-1]]
            fieldname = 'R' + lagtype[0] * nr
            for i in range(1, nr + 1):
                rper = -1j * w * rper
                d_acf = fft(rper, nfft).real / (2 * n_f - 2)
                setattr(acf, fieldname[0:i + 1], d_acf[0:nt + 1])
        return acf

    def to_linspec(self, ns=None, dt=None, cases=20, iseed=None,
                   fn_limit=sqrt(2), gravity=9.81):
        '''
        Split the linear and non-linear component from the Spectrum
            according to 2nd order wave theory

        Returns
        -------
        SL, SN : SpecData1D objects
            with linear and non-linear components only, respectively.

        Parameters
        ----------
        ns : scalar integer
            giving ns load points.  (default length(S)-1=n-1).
            If np>n-1 it is assummed that S(k)=0 for all k>n-1
        cases : scalar integer
            number of cases (default=20)
        dt : real scalar
            step in grid (default dt is defined by the Nyquist freq)
        iseed : scalar integer
            starting seed number for the random number generator
                  (default none is set)
        fnLimit : real scalar
            normalized upper frequency limit of spectrum for 2'nd order
            components. The frequency is normalized with
                   sqrt(gravity*tanh(kbar*water_depth)/Amax)/(2*pi)
            (default sqrt(2), i.e., Convergence criterion).
            Generally this should be the same as used in the final
                   non-linear simulation (see example below).

        SPEC2LINSPEC separates the linear and non-linear component of the
        spectrum according to 2nd order wave theory. This is useful when
        simulating non-linear waves because:
        If the spectrum does not decay rapidly enough towards zero, the
        contribution from the 2nd order wave components at the upper tail can
        be very large and unphysical. Another option to ensure convergence of
        the perturbation series in the simulation, is to truncate the upper
        tail of the spectrum at FNLIMIT in the calculation of the 2nd order
        wave components, i.e., in the calculation of sum and difference
        frequency effects.

        Example:
        --------
        np = 10000
          iseed = 1
          pflag = 2
          S  = jonswap(10)
          fnLimit = inf
          [SL,SN] = spec2linspec(S,np,[],[],fnLimit)
          x0 = spec2nlsdat(SL,8*np,[],iseed,[],fnLimit)
          x1 = spec2nlsdat(S,8*np,[],iseed,[],fnLimit)
          x2 = spec2nlsdat(S,8*np,[],iseed,[],sqrt(2))
          Se0 = dat2spec(x0)
          Se1 = dat2spec(x1)
          Se2 = dat2spec(x2)
          clf
          plotspec(SL,'r',pflag),  % Linear components
           hold on
          plotspec(S,'b',pflag)    % target spectrum for simulated data
          plotspec(Se0,'m',pflag), % approx. same as S
          plotspec(Se1,'g',pflag)  % unphysical spectrum
          plotspec(Se2,'k',pflag)  % approx. same as S
          axis([0 10 -80 0])
          hold off

        See also
        --------
        spec2nlsdat

        References
        ----------
        P. A. Brodtkorb (2004),
        The probability of Occurrence of dangerous Wave Situations at Sea.
        Dr.Ing thesis, Norwegian University of Science and Technolgy, NTNU,
        Trondheim, Norway.

        Nestegaard, A  and Stokka T (1995)
        A Third Order Random Wave model.
        In proc.ISOPE conf., Vol III, pp 136-142.

        R. S Langley (1987)
        A statistical analysis of non-linear random waves.
        Ocean Engng, Vol 14, pp 389-407

        Marthinsen, T. and Winterstein, S.R (1992)
        'On the skewness of random surface waves'
        In proc. ISOPE Conf., San Francisco, 14-19 june.
        '''

        # by pab 13.08.2002

        # TODO % Replace inputs with options structure
        # TODO % Can be improved further.

        method = 'apstochastic'
        trace = 1  # % trace the convergence
        max_sim = 30
        tolerance = 5e-4

        L = 200  # maximum lag size of the window function used in estimate
        # ftype = self.freqtype #options are 'f' and 'w' and 'k'
#        switch ftype
#         case 'f',
#          ftype = 'w'
#          S = ttspec(S,ftype)
#        end
        Hm0 = self.characteristic('Hm0')
        Tm02 = self.characteristic('Tm02')

        if iseed is not None:
            _set_seed(iseed)  # set the the seed

        n = len(self.data)
        if ns is None:
            ns = max(n - 1, 5000)
        if dt is None:
            S = self.interp(dt)  # interpolate spectrum
        else:
            S = self.copy()

        ns = ns + mod(ns, 2)  # make sure np is even

        water_depth = abs(self.h)
        kbar = w2k(2 * pi / Tm02, 0, water_depth)[0]

        # Expected maximum amplitude for 10000 waves seastate
        num_waves = 10000  # Typical number of waves in 30 hour seastate
        Amax = sqrt(2 * log(num_waves)) * Hm0 / 4

        fLimitLo = sqrt(
            gravity * tanh(kbar * water_depth) * Amax / water_depth ** 3)

        freq = S.args
        eps = finfo(float).eps
        freq[-1] = freq[-1] - sqrt(eps)
        Hw2 = 0

        SL = S

        indZero = nonzero(freq < fLimitLo)[0]
        if len(indZero):
            SL.data[indZero] = 0

        maxS = max(S.data)
        # Fs = 2*freq(end)+eps  # sampling frequency

        for ix in range(max_sim):
            x2, x1 = self.sim_nl(ns=np, cases=cases, dt=None, iseed=iseed,
                                 method=method, fnlimit=fn_limit,
                                 output='timeseries')
            x2.data -= x1.data  # x2(:,2:end) = x2(:,2:end) -x1(:,2:end)
            S2 = x2.tospecdata(L)
            S1 = x1.tospecdata(L)

            # TODO: Finish spec.to_linspec
#             S2 = dat2spec(x2, L)
#             S1 = dat2spec(x1, L)
# %[tf21,fi] = tfe(x2(:,2),x1(:,2),1024,Fs,[],512)
# %Hw11 = interp1q(fi,tf21.*conj(tf21),freq)
            if True:
                Hw1 = exp(interp1d(log(abs(S1.data / S2.data)), S2.args)(freq))
            else:
                # Geometric mean
                Hw1 = exp((interp1d(log(abs(S1.data / S2.data)), S2.args)(freq)
                           + log(Hw2)) / 2)
                # end
            # Hw1  = (interp1q( S2.w,abs(S1.S./S2.S),freq)+Hw2)/2
            # plot(freq, abs(Hw11-Hw1),'g')
            # title('diff')
            # pause
            # clf

            # d1 = interp1q( S2.w,S2.S,freq)

            SL.data = (Hw1 * S.data)

            if len(indZero):
                SL.data[indZero] = 0
                # end
            k = nonzero(SL.data < 0)[0]
            if len(k):  # Make sure that the current guess is larger than zero
                # k
                # Hw1(k)
                Hw1[k] = min(S1.data[k] * 0.9, S.data[k])
                SL.data[k] = max(Hw1[k] * S.data[k], eps)
                # end
            Hw12 = Hw1 - Hw2
            maxHw12 = max(abs(Hw12))
            if trace == 1:
                plotbackend.figure(1),
                plotbackend.semilogy(freq, Hw1, 'r')
                plotbackend.title('Hw')
                plotbackend.figure(2),
                plotbackend.semilogy(freq, abs(Hw12), 'r')
                plotbackend.title('Hw-HwOld')

                # pause(3)
                plotbackend.figure(1),
                plotbackend.semilogy(freq, Hw1, 'b')
                plotbackend.title('Hw')
                plotbackend.figure(2),
                plotbackend.semilogy(freq, abs(Hw12), 'b')
                plotbackend.title('Hw-HwOld')
                # figtile
            # end

            print('Iteration : %d, Hw12 : %g  Hw12/maxS : %g' %
                  (ix, maxHw12, (maxHw12 / maxS)))
            if (maxHw12 < maxS * tolerance) and (Hw1[-1] < Hw2[-1]):
                break
            # end
            Hw2 = Hw1
        # end

        # Hw1(end)
        # maxS*1e-3
        # if Hw1[-1]*S.data>maxS*1e-3,
        #   warning('The Nyquist frequency of the spectrum may be too low')
        # end

        SL.date = now()  # datestr(now)
        # if nargout>1
        SN = SL.copy()
        SN.data = S.data - SL.data
        SN.note = SN.note + ' non-linear component (spec2linspec)'
        # end
        SL.note = SL.note + ' linear component (spec2linspec)'

        return SL, SN

    def to_mm_pdf(self, paramt=None, paramu=None, utc=None, nit=2, EPS=5e-5,
                  EPSS=1e-6, C=4.5, EPS0=1e-5, IAC=1, ISQ=0, verbose=False):
        '''
        nit    = order of numerical integration: 0,1,2,3,4,5.
        paramu = parameter vector defining discretization of min/max values.
        t      = grid of time points between maximum and minimum (to
             integrate out). interval between maximum and the following
             minimum,
        The variable ISQ marks which type of conditioning will be used ISQ=0
        means random time where the probability is minimum, ISQ=1 is the time
        where the variance of the residual process is minimal(ISQ=1 is faster).

        NIT, IAC are  described in CROSSPACK paper, EPS0 is the accuracy
        constant used in choosing the number of nodes in numerical integrations
        (XX1, H1 vectors). The nodes and weights and other parameters are
        read in the subroutine INITINTEG from files Z.DAT, H.DAT and ACCUR.DAT.


        NIT=0, IAC=1 then one uses RIND0 - subroutine, all other cases
        goes through RIND1, ...,RIND5. NIT=0, here means explicite formula
        approximation for XIND=E[Y^+1{ HH<BU(I)<0 for all I, I=1,...,N}], where
        BU(I) is deterministic function.

        NIT=1, leads tp call RIND1, IAC=0 is also explicit form approximation,
        while IAC=1 leads to maximum one dimensional integral.
        .......
        NIT=5, leads tp call RIND5, IAC is maximally 4-dimensional integral,
        while IAC=1 leads to maximum 5 dimensional integral.

        >>> import numpy as np
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=3)
        >>> w = np.linspace(0,4,256)
        >>> S1 = Sj.tospecdata(w)   #Make spectrum object from numerical values
        >>> S = sm.SpecData1D(Sj(w),w) # Alternatively do it manually
        mm = S.to_mm_pdf()
        mm.plot()
        mm.plot(plotflag=1)
        '''

        S = self.copy()
        S.normalize()
        m, unused_mtxt = self.moment(nr=4, even=True)
        A = sqrt(m[0] / m[1])

        if paramt is None:
            # (2.5 * mean distance between extremes)
            distanceBetweenExtremes = 5 * pi * sqrt(m[1] / m[2])
            paramt = [0, distanceBetweenExtremes, 43]

        if paramu is None:
            paramu = [-5 * sqrt(m[0]), 5 * sqrt(m[0]), 41]

        if self.tr is None:
            g = TrLinear(var=m[0])
        else:
            g = self.tr

        if utc is None:
            utc = g.gauss2dat(0)  # most frequent crossed level

        # transform reference level into Gaussian level
        u = g.dat2gauss(utc)
        if verbose:
            print('The level u for Gaussian process = %g' % u)

        unused_t0, tn, Nt = paramt
        t = linspace(0, tn / A, Nt)  # normalized times

        # Transform amplitudes to Gaussian levels:
        h = linspace(*paramu)
        dt = t[1] - t[0]
        nr = 4
        R = S.tocov_matrix(nr, Nt - 1, dt)

        # ulev = linspace(*paramu)
        # vlev = linspace(*paramu)

        trdata = g.trdata()
        Tg = trdata.args
        Xg = trdata.data

        cov2mod.initinteg(EPS, EPSS, EPS0, C, IAC, ISQ)
        uvdens = cov2mod.cov2mmpdfreg(t, R, h, h, Tg, Xg, nit)
        uvdens = np.rot90(uvdens, -2)

        dh = h[1] - h[0]
        uvdens *= dh * dh

        mmpdf = PlotData(uvdens, args=(h, h), xlab='max [m]', ylab='min [m]',
                         title='Joint density of maximum and minimum')
        try:
            pl = [10, 30, 50, 70, 90, 95, 99, 99.9]
            mmpdf.cl = qlevels(uvdens, pl, h, h)
            mmpdf.pl = pl
        except:
            pass
        return mmpdf

    def to_t_pdf(self, u=None, kind='Tc', paramt=None, **options):
        '''
        Density of crest/trough- period or length, version 2.

        Parameters
        ----------
        u : real scalar
            reference level (default the most frequently crossed level).
        kind : string, 'Tc', Tt', 'Lc' or 'Lt'
            'Tc',    gives half wave period, Tc (default).
            'Tt',    gives half wave period, Tt
            'Lc' and 'Lt' ditto for wave length.
        paramt : [t0, tn, nt]
            where t0, tn and nt is the first value, last value and the number
            of points, respectively, for which the density will be computed.
            paramt= [5, 5, 51] implies that the density is computed only for
            T=5 and using 51 equidistant points in the interval [0,5].
        options : optional parameters
            controlling the performance of the integration.
            See Rind for details.

        Notes
        -----
        SPEC2TPDF2 calculates pdf of halfperiods  Tc, Tt, Lc or Lt
        in a stationary Gaussian transform process X(t),
        where Y(t) = g(X(t)) (Y zero-mean Gaussian with spectrum given in S).
        The transformation, g, can be estimated using LC2TR,
        DAT2TR, HERMITETR or OCHITR.

        Example
        -------
        The density of Tc is computed by:
        >>> import pylab as plb
        >>> from wafo.spectrum import models as sm
        >>> w = np.linspace(0,3,100)
        >>> Sj = sm.Jonswap()
        >>> S = Sj.tospecdata()
        >>> f = S.to_t_pdf(pdef='Tc', paramt=(0, 10, 51), speed=7)
        >>> h = f.plot()

        estimated error bounds
        >>> h2 = plb.plot(f.args, f.data+f.err, 'r', f.args, f.data-f.err, 'r')

        >>> plb.close('all')

        See also
        --------
        Rind, spec2cov2, specnorm, dat2tr, dat2gaus,
        definitions.wave_periods,
        definitions.waves

        '''

        opts = dict(speed=9)
        opts.update(options)
        if kind[0] in ('l', 'L'):
            if self.type != 'k1d':
                raise ValueError('Must be spectrum of type: k1d')
        elif kind[0] in ('t', 'T'):
            if self.type != 'freq':
                raise ValueError('Must be spectrum of type: freq')
        else:
            raise ValueError('pdef must be Tc,Tt or Lc, Lt')
#        if strncmpi('l',kind,1)
#          spec=spec2spec(spec,'k1d')
#        elseif strncmpi('t',kind,1)
#          spec=spec2spec(spec,'freq')
#        else
#          error('Unknown kind')
#        end
        kind2defnr = dict(tc=1, lc=1, tt=-1, lt=-1)
        defnr = kind2defnr[kind.lower()]

        S = self.copy()
        S.normalize()
        m, unused_mtxt = self.moment(nr=2, even=True)
        A = sqrt(m[0] / m[1])

        if self.tr is None:
            g = TrLinear(var=m[0])
        else:
            g = self.tr

        if u is None:
            u = g.gauss2dat(0)  # % most frequently crossed level

        # transform reference level into Gaussian level
        un = g.dat2gauss(u)

        # disp(['The level u for Gaussian process = ', num2str(u)])

        if paramt is None:
            # z2 = u^2/2
            z = -sign(defnr) * un / sqrt(2)
            expectedMaxPeriod = 2 * \
                ceil(2 * pi * A * exp(z) * (0.5 + erf(z) / 2))
            paramt = [0, expectedMaxPeriod, 51]

        t0 = paramt[0]
        tn = paramt[1]
        Ntime = paramt[2]
        t = linspace(0, tn / A, Ntime)  # normalized times
        # index to starting point to evaluate
        Nstart = max(round(t0 / tn * (Ntime - 1)), 1)

        dt = t[1] - t[0]
        nr = 2
        R = S.tocov_matrix(nr, Ntime - 1, dt)
        # R  = spec2cov2(S,nr,Ntime-1,dt)

        xc = vstack((un, un))
        indI = -ones(4, dtype=int)
        Nd = 2
        Nc = 2
        XdInf = 100.e0 * sqrt(-R[0, 2])
        XtInf = 100.e0 * sqrt(R[0, 0])

        B_up = hstack([un + XtInf, XdInf, 0])
        B_lo = hstack([un, 0, -XdInf])
        # INFIN = [1 1 0]
        # BIG   = zeros((Ntime+2,Ntime+2))
        ex = zeros(Ntime + 2, dtype=float)
        # CC    = 2*pi*sqrt(-R(1,1)/R(1,3))*exp(un^2/(2*R(1,1)))
        #   XcScale = log(CC)
        opts['xcscale'] = log(
            2 * pi * sqrt(-R[0, 0] / R[0, 2])) + (un ** 2 / (2 * R[0, 0]))

        f = zeros(Ntime, dtype=float)
        err = zeros(Ntime, dtype=float)

        rind = Rind(**opts)
        # h11 = fwaitbar(0,[],sprintf('Please wait ...(start at: %s)',
        #        datestr(now)))
        for pt in range(Nstart, Ntime):
            Nt = pt - Nd + 1
            Ntd = Nt + Nd
            Ntdc = Ntd + Nc
            indI[1] = Nt - 1
            indI[2] = Nt
            indI[3] = Ntd - 1

            #  positive wave period
            BIG = self._covinput_t_pdf(pt, R)

            tmp = rind(BIG, ex[:Ntdc], B_lo, B_up, indI, xc, Nt)
            f[pt], err[pt] = tmp[:2]
            # fwaitbar(pt/Ntime,h11,sprintf('%s Ready: %d of %d',
            #                        datestr(now),pt,Ntime))
        # end
        # close(h11)

        titledict = dict(
            tc='Density of Tc', tt='Density of Tt', lc='Density of Lc',
            lt='Density of Lt')
        Htxt = titledict.get(kind.lower())

        if kind[0].lower() == 'l':
            xtxt = 'wave length [m]'
        else:
            xtxt = 'period [s]'

        Htxt = '%s_{v =%2.5g}' % (Htxt, u)
        pdf = PlotData(f / A, t * A, title=Htxt, xlab=xtxt)
        pdf.err = err / A
        pdf.u = u
        pdf.options = opts
        return pdf

    def _covinput_t_pdf(self, pt, R):
        """
        Return covariance matrix for Tc or Tt period problems

        Parameters
        ----------
        pt : scalar integer
            time
        R : array-like, shape Ntime x 3
            [R0,R1,R2] column vectors with autocovariance and its derivatives,
            i.e., R1 and R2 are vectors with the 1'st and 2'nd derivatives of
            R0, respectively.

        The order of the variables in the covariance matrix are organized as
        follows:
        For pt>1:
        ||X(t2)..X(ts),..X(tn-1)|| X'(t1) X'(tn)|| X(t1) X(tn) ||
        = [Xt                          Xd                    Xc]

        where

        Xt = time points in the indicator function
        Xd = derivatives
        Xc=variables to condition on

        Computations of all covariances follows simple rules:
            Cov(X(t),X(s))=r(t,s),
        then  Cov(X'(t),X(s))=dr(t,s)/dt.  Now for stationary X(t) we have
        a function r(tau) such that Cov(X(t),X(s))=r(s-t) (or r(t-s) will give
        the same result).

        Consequently
            Cov(X'(t),X(s))    = -r'(s-t)    = -sign(s-t)*r'(|s-t|)
            Cov(X'(t),X'(s))   = -r''(s-t)   = -r''(|s-t|)
            Cov(X''(t),X'(s))  =  r'''(s-t)  =  sign(s-t)*r'''(|s-t|)
            Cov(X''(t),X(s))   =  r''(s-t)   =   r''(|s-t|)
            Cov(X''(t),X''(s)) =  r''''(s-t) = r''''(|s-t|)

        """
        # cov(Xd)
        Sdd = -toeplitz(R[[0, pt], 2])
        # cov(Xc)
        Scc = toeplitz(R[[0, pt], 0])
        # cov(Xc,Xd)
        Scd = array([[0, R[pt, 1]], [-R[pt, 1], 0]])

        if pt > 1:
            # cov(Xt)
            # Cov(X(tn),X(ts))  = r(ts-tn)   = r(|ts-tn|)
            Stt = toeplitz(R[:pt - 1, 0])
            # cov(Xc,Xt)
            # Cov(X(tn),X(ts))  = r(ts-tn)   = r(|ts-tn|)
            Sct = R[1:pt, 0]
            Sct = vstack((Sct, Sct[::-1]))
            # Cov(Xd,Xt)
            # Cov(X'(t1),X(ts)) = -r'(ts-t1) = r(|s-t|)
            Sdt = -R[1:pt, 1]
            Sdt = vstack((Sdt, -Sdt[::-1]))
            # N   = pt + 3
            big = vstack((hstack((Stt, Sdt.T, Sct.T)),
                          hstack((Sdt, Sdd, Scd.T)),
                          hstack((Sct, Scd, Scc))))
        else:
            # N = 4
            big = vstack((hstack((Sdd, Scd.T)),
                          hstack((Scd, Scc))))
        return big

    def to_mmt_pdf(self, paramt=None, paramu=None, utc=None, kind='mm',
                   verbose=False, **options):
        ''' Returns joint density of Maximum, minimum and period.

        Parameters
        ----------
        u    = reference level (default the most frequently crossed level).
        kind : string
            defining density returned
            'Mm'    : maximum and the following minimum. (M,m) (default)
            'rfc'   : maximum and the rainflow minimum height.
            'AcAt'  : (crest,trough) heights.
            'vMm'   : level v separated Maximum and minimum   (M,m)_v
            'MmTMm' : maximum, minimum and period between (M,m,TMm)
            'vMmTMm': level v separated Maximum, minimum and period
                         between (M,m,TMm)_v
            'MmTMd' : level v separated Maximum, minimum and the period
                         from Max to level v-down-crossing (M,m,TMd)_v.
            'MmTdm' : level v separated Maximum, minimum and the period from
                         level v-down-crossing to min. (M,m,Tdm)_v
            NB! All 'T' above can be replaced by 'L' to get wave length
               instead.
        paramt : [0 tn Nt]
            defines discretization of half period: tn is the longest period
            considered while Nt is the number of points, i.e. (Nt-1)/tn is the
            sampling frequnecy. paramt= [0 10 51] implies that the halfperiods
            are considered at 51 linearly spaced points in the interval [0,10],
            i.e. sampling frequency is 5 Hz.
        paramu : [u, v, N]
            defines discretization of maxima and minima ranges:  u is the
            lowest minimum considered, v the highest maximum and N is the
            number of levels (u,v) included.
        options :
            rind-options structure containing optional parameters controlling
            the performance of the integration. See rindoptset for details.
        []    = default values are used.

        Returns
        -------
        f    = pdf (density structure) of crests (trough) heights

        TO_MMT_PDF calculates densities of wave characteristics in a
        stationary Gaussian transform process X(t) where
        Y(t) = g(X(t)) (Y zero-mean Gaussian with spectrum given in input spec)
        The tr.g can be estimated using lc2tr, dat2tr, hermitetr or ochitr.

        Examples
        --------
        The joint density of zero separated Max2min cycles in time (a);
        in space (b); AcAt in time for nonlinear sea model (c):

        Hm0=7;Tp=11
        S = jonswap(4*pi/Tp,[Hm0 Tp])
        Sk = spec2spec(S,'k1d')
        L0 = spec2mom(S,1)
        paramu = [sqrt(L0)*[-4 4] 41]
        ft = spec2mmtpdf(S,0,'vmm',[],paramu); pdfplot(ft)         % a)
        fs = spec2mmtpdf(Sk,0,'vmm');  figure, pdfplot(fs)         % b)
        [sk, ku, me]=spec2skew(S)
        g = hermitetr([],[sqrt(L0) sk ku me])
        Snorm=S; Snorm.S=S.S/L0; Snorm.tr=g
        ftg=spec2mmtpdf(Snorm,0,'AcAt',[],paramu); pdfplot(ftg)    % c)

        See also
        --------
        rindoptset, dat2tr, datastructures, wavedef, perioddef

        References
        ---------
        Podgorski et al. (2000)
        "Exact distributions for apparent waves in irregular seas"
        Ocean Engineering,  Vol 27, no 1, pp979-1016.

        P. A. Brodtkorb (2004),
        Numerical evaluation of multinormal expectations
        In Lund university report series
        and in the Dr.Ing thesis:
        The probability of Occurrence of dangerous Wave Situations at Sea.
        Dr.Ing thesis, Norwegian University of Science and Technolgy, NTNU,
        Trondheim, Norway.

        Per A. Brodtkorb (2006)
        "Evaluating Nearly Singular Multinormal Expectations with Application
        to Wave Distributions",
        Methodology And Computing In Applied Probability, Volume 8, Number 1,
        pp. 65-91(27)
        '''

        opts = dict(speed=4, nit=2, method=0)
        opts.update(**options)

        ftype = self.freqtype
        kind2defnr = dict(ac=-2, at=-2,
                          rfc=-1,
                          mm=0,
                          mmtmm=1, mmlmm=1,
                          vmm=2,
                          vmmtmm=3, vmmlmm=3,
                          mmtmd=4, vmmtmd=4,  mmlmd=4, vmmlmd=4,
                          mmtdm=5, vmmtdm=5, mmldm=5, vmmldm=5)
        defnr = kind2defnr.get(kind, 0)
        in_space = (ftype == 'k')  # distribution in space or time
        if defnr >= 3 or defnr == 1:
            in_space = (kind[-2].upper() == 'L')

        if in_space:
            # spec = spec2spec(spec,'k1d')
            ptxt = 'space'
        else:
            # spec = spec2spec(spec,'freq')
            ptxt = 'time'

        S = self.copy()
        S.normalize()
        m, unused_mtxt = self.moment(nr=4, even=True)
        A = sqrt(m[0] / m[1])

        if paramt is None:
            # (2.5 * mean distance between extremes)
            distanceBetweenExtremes = 5 * pi * sqrt(m[1] / m[2])
            paramt = [0, distanceBetweenExtremes, 43]

        if paramu is None:
            paramu = [-5 * sqrt(m[0]), 5 * sqrt(m[0]), 41]

        if self.tr is None:
            g = TrLinear(var=m[0])
        else:
            g = self.tr

        if utc is None:
            utc = g.gauss2dat(0)  # most frequent crossed level

        # transform reference level into Gaussian level
        u = g.dat2gauss(utc)
        if verbose:
            print('The level u for Gaussian process = %g' % u)

        t0, tn, Nt = paramt
        t = linspace(0, tn / A, Nt)  # normalized times

        # the starting point to evaluate
        Nstart = 1 + round(t0 / tn * (Nt - 1))

        Nx = paramu[2]
        if (defnr > 1):
            paramu[0] = max(0, paramu[0])
            if (paramu[1] < 0):
                raise ValueError(
                    'Discretization levels must be larger than zero')

        # Transform amplitudes to Gaussian levels:
        h = linspace(*paramu)

        if defnr > 1:  # level v separated Max2min densities
            hg = np.hstack((utc + h, utc - h))
            hg, der = g.dat2gauss(utc + h, ones(Nx))
            hg1, der1 = g.dat2gauss(utc - h, ones(Nx))
            der, der1 = np.abs(der), np.abs(der1)
            hg = np.hstack((hg, hg1))
        else:  # Max2min densities
            hg, der = np.abs(g.dat2gauss(h, ones(Nx)))
            der = der1 = np.abs(der)

        dt = t[1] - t[0]
        nr = 4
        R = S.tocov_matrix(nr, Nt - 1, dt)

        # NB!!! the spec2XXpdf.exe programmes are very sensitive to how you
        #  interpolate the covariances, especially where the process is very
        # dependent and the covariance matrix is nearly singular. (i.e. for
        # small t and high levels of u if Tc and low levels of u if Tt)
        # The best is to interpolate the spectrum linearly so that S.S>=0
        # This makes sure that the covariance matrix is positive
        # semi-definitt, since the circulant spectrum are the eigenvalues of
        # the circulant covariance matrix.

        callFortran = 0
        # %options.method<0
        # if callFortran, % call fortran
        # ftmp = cov2mmtpdfexe(R,dt,u,defnr,Nstart,hg,options)
        # err = repmat(nan,size(ftmp))
        # else
        ftmp, err, terr, options = self._cov2mmtpdf(R, dt, u, defnr, Nstart,
                                                    hg, options)

        # end
        note = ''
        if hasattr(self, 'note'):
            note = note + self.note
        tmp = 'L' if in_space else 'T'
        if Nx > 2:
            titledict = {
                '-2': 'Joint density of (Ac,At) in %s' % ptxt,
                '-1': 'Joint density of (M,m_{rfc}) in %s' % ptxt,
                '0': 'Joint density of (M,m) in %s' % ptxt,
                '1': 'Joint density of (M,m,%sMm) in %s' % (tmp, ptxt),
                '2': 'Joint density of (M,m)_{v=%2.5g} in %s' % (utc, ptxt),
                '3': 'Joint density of (M,m,%sMm)_{v=%2.5g} in %s' %
                (tmp, utc, ptxt),
                '4': 'Joint density of (M,m,%sMd)_{v=%2.5g} in %s' %
                (tmp, utc, ptxt),
                '5': 'Joint density of (M,m,%sdm)_{v=%2.5g} in %s' %
                (tmp, utc, ptxt)}
            title = titledict[defnr]
            labx = 'Max [m]'
            laby = 'min [m]'
            args = (h, h)
        else:
            note = note + 'Density is not scaled to unity'
            if defnr in (-2, -1, 0, 1):
                title = 'Density of (%sMm, M = %2.5g, m = %2.5g)' % (
                    tmp, h[1], h[0])
            elif defnr in (2, 3):
                title = 'Density of (%sMm, M = %2.5g, m = %2.5g)_{v=%2.5g}' % (
                    tmp, h[1], -h[1], utc)
            elif defnr == 4:
                title = 'Density of (%sMd, %sMm, M = %2.5g, m = %2.5g)_{v=%2.5g}' % (
                    tmp, tmp, h[1], -h[1], utc)
            elif defnr == 5:
                title = 'Density of (%sdm, %sMm, M = %2.5g, m = %2.5g)_{v=%2.5g}' % (
                    tmp, tmp, h[1], -h[1], utc)

        f = PlotData()
#        f.options = options
#        if defnr>1 or defnr==-2:
# f.u  = utc # save level u
#
#        if Nx>2  % amplitude distributions wanted
#        f.x{2}    = h
#        f.labx{2} = 'min [m]'
#
#
#        if defnr>2 || defnr==1
#            der0 = der1[:,None] * der[None,:]
#            ftmp = np.reshape(ftmp,Nx,Nx,Nt) * der0[:,:, None] / A
#            err  = np.reshape(err,Nx,Nx,Nt) * der0[:,:, None] / A
#
#            f.x{3} = t(:)*A
#            labz = 'wave length [m]' if in_space else 'period [sec]'
#
#        else
#            der0 = der[:,None] * der[None,:]
#            ftmp  = np.reshape(ftmp,Nx,Nx) * der0
#            err   = np.reshape(err,Nx,Nx) * der0
#
#            if (defnr==-1):
#                ftmp0 = fliplr(mctp2rfc(fliplr(ftmp)))
#                err  = abs(ftmp0-fliplr(mctp2rfc(fliplr(ftmp+err))))
#                ftmp = ftmp0
#            elif (defnr==-2):
#              ftmp0=fliplr(mctp2tc(fliplr(ftmp),utc,paramu))*sqrt(L4*L0)/L2
#              err =abs(ftmp0-fliplr(mctp2tc(fliplr(ftmp+err),utc,paramu))*sqrt(L4*L0)/L2)
#              index1=find(f.x{1}>0)
#              index2=find(f.x{2}<0)
#              ftmp=flipud(ftmp0(index2,index1))
#              err =flipud(err(index2,index1))
#              f.x{1} = f.x{1}(index1)
#              f.x{2} = abs(flipud(f.x{2}(index2)))
#            end
#        end
#        f.f = ftmp
#        f.err = err
#        else % Only time or wave length distributions wanted
#        f.f = ftmp/A
#        f.err = err/A
#        f.x{1}=A*t'
#        if strcmpi(def(1),'t')
#        f.labx{1} = 'period [sec]'
#        else
#        f.labx{1} = 'wave length [m]'
#        end
#        if defnr>3,
#        f.f   = reshape(f.f,[Nt, Nt])
#        f.err = reshape(f.err,[Nt, Nt])
#        f.x{2}= A*t'
#        if strcmpi(def(1),'t')
#          f.labx{2} = 'period [sec]'
#        else
#          f.labx{2} = 'wave length [m]'
#        end
#        end
#        end
#
#
#        try
#        [f.cl,f.pl]=qlevels(f.f,[10 30 50 70 90 95 99 99.9],f.x{1},f.x{2})
#        catch
#        warning('WAFO:SPEC2MMTPDF','Singularity likely in pdf')
#        end
#        %pdfplot(f)
#
#        %Test of spec2mmtpdf
#        % cd  f:\matlab\matlab\wafo\source\sp2thpdfalan
#        % addpath f:\matlab\matlab\wafo ,initwafo, addpath f:\matlab\matlab\graphutil
#        % Hm0=7;Tp=11; S = jonswap(4*pi/Tp,[Hm0 Tp])
#        % ft = spec2mmtpdf(S,0,'vMmTMm',[0.3,.4,11],[0 .00005 2])

        return f  # % main

    def _cov2mmtpdf(self, R, dt, u, def_nr, Nstart, hg, options):
        '''
        COV2MMTPDF Joint density of Maximum, minimum and period.

        CALL  [pdf, err, options] = cov2mmtpdf(R,dt,u,def,Nstart,hg,options)

        pdf     = calculated pdf size Nx x Ntime
        err     = error estimate
        terr    = truncation error
        options = requested and actual rindoptions used in integration.
        R       = [R0,R1,R2,R3,R4] column vectors with autocovariance and its
                  derivatives, i.e., Ri (i=1:4) are vectors with the 1'st to
                  4'th derivatives of R0.  size Ntime x Nr+1
        dt      = time spacing between covariance samples, i.e.,
                   between R0(1),R0(2).
        u       = crossing level
        def     = integer defining pdf calculated:
                   0 : maximum and the following minimum. (M,m) (default)
                   1 : level v separated Maximum and minimum   (M,m)_v
                   2 : maximum, minimum and period between (M,m,TMm)
                   3 : level v separated Maximum, minimum and period
                       between (M,m,TMm)_v
                   4 : level v separated Maximum, minimum and the period
                       from Max to level v-down-crossing (M,m,TMd)_v.
                   5 : level v separated Maximum, minimum and the period from
                       level v-down-crossing to min. (M,m,Tdm)_v
        Nstart  = index to where to start calculation, i.e., t0 = t(Nstart)
        hg      = vector of amplitudes length Nx or 0
        options = rind options structure defining the integration parameters

        COV2MMTPDF computes joint density of the  maximum and the following
        minimum or level u separated maxima and minima + period/wavelength

        For DEF = 0,1 : (Maxima, Minima and period/wavelength)
                = 2,3 : (Level v separated Maxima and Minima and
                         period/wavelength between them)

        If Nx==1 then the conditional  density for  period/wavelength between
        Maxima and Minima given the Max and Min is returned
        Y =
        X'(t2)..X'(ts)..X'(tn-1)|| X''(t1) X''(tn)|| X'(t1) X'(tn) X(t1) X(tn)
         = [       Xt                   Xd                    Xc            ]

         Nt = tn-2, Nd = 2, Nc = 4
         Xt = contains Nt time points in the indicator function
         Xd =    "     Nd    derivatives in Jacobian
         Xc =    "     Nc    variables to condition on

         There are 3 (NI=4) regions with constant barriers:
         (indI[0]=0);   for i in (indI[0],indI[1]]    Y[i]<0.
         (indI[1]=Nt);  for i in (indI[1]+1,indI[2]], Y[i]<0 (deriv. X''(t1))
         (indI[2]=Nt+1);  for i\in (indI[2]+1,indI[3]], Y[i]>0 (deriv. X''(tn))

         For DEF = 4,5 (Level v separated Maxima and Minima and
                        period/wavelength from Max to crossing)

         If Nx==1 then the conditional joint density for  period/wavelength
         between Maxima, Minima and Max to level v crossing given the Max and
         the min is returned

         Y=
         X'(t2)..X'(ts)..X'(tn-1)||X''(t1) X''(tn) X'(ts)|| X'(t1) X'(tn)  X(t1) X(tn) X(ts)
         = [       Xt                      Xd                     Xc           ]

         Nt = tn-2, Nd = 3, Nc = 5

         Xt= contains Nt time points in the indicator function
         Xd=    "     Nd    derivatives
         Xc=    "     Nc    variables to condition on

         There are 4 (NI=5) regions with constant barriers:
         (indI(1)=0);     for i\in (indI(1),indI(2)]    Y(i)<0.
         (indI(2)=Nt)  ;  for i\in (indI(2)+1,indI(3)], Y(i)<0 (deriv. X''(t1))
         (indI(3)=Nt+1);  for i\in (indI(3)+1,indI(4)], Y(i)>0 (deriv. X''(tn))
         (indI(4)=Nt+2);  for i\in (indI(4)+1,indI(5)], Y(i)<0 (deriv. X'(ts))
        '''
        R0, R1, R2, R3, R4 = R[:, :5].T
        covinput = self._covinput_mmt_pdf
        Ntime = len(R0)
        Nx0 = max(1, len(hg))
        Nx1 = Nx0
        # Nx0 = Nx1 #just plain Mm
        if def_nr > 1:
            Nx1 = Nx0 // 2
            # Nx0 = 2*Nx1 # level v separated max2min densities wanted
        # print('def = %d' % def_nr))

        # The bound 'infinity' is set to 100*sigma
        XdInf = 100.0 * sqrt(R4[0])
        XtInf = 100.0 * sqrt(-R2[0])
        Nc = 4
        NI = 4
        Nd = 2
        # Mb = 1
        # Nj = 0

        Nstart = max(2, Nstart)
        symmetry = 0
        isOdd = np.mod(Nx1, 2)
        if def_nr <= 1:  # just plain Mm
            Nx = Nx1 * (Nx1 - 1) / 2
            IJ = (Nx1 + isOdd) / 2
            if (hg[0] + hg[Nx1 - 1] == 0 and (hg[IJ - 1] == 0 or
                                              hg[IJ - 1] + hg[IJ] == 0)):
                symmetry = 0
                print(' Integration region symmetric')
                # May save Nx1-isOdd integrations in each time step
                # This is not implemented yet.
                # Nx = Nx1*(Nx1-1)/2-Nx1+isOdd
            # normalizing constant:
            # CC = 1/ expected number of zero-up-crossings of X'
            # CC = 2*pi*sqrt(-R2[0]/R4[0])
            #  XcScale = log(CC)
            XcScale = log(2 * pi * sqrt(-R2[0] / R4[0]))
        else:
            # level u separated Mm
            Nx = (Nx1 - 1) * (Nx1 - 1)
            if (abs(u) <= _EPS and (hg[0] + hg[Nx1] == 0) and
                    (hg[Nx1 - 1] + hg[2 * Nx1 - 1] == 0)):
                symmetry = 0
                print(' Integration region symmetric')
                # Not implemented for DEF <= 3
                # IF (DEF.LE.3) Nx = (Nx1-1)*(Nx1-2)/2

            if def_nr > 3:
                Nstart = max(Nstart, 3)
                Nc = 5
                NI = 5
                Nd = 3
            # CC = 1/ expected number of u-up-crossings of X
            # CC = 2*pi*sqrt(-R0(1)/R2(1))*exp(0.5D0*u*u/R0(1))
            XcScale = log(2 * pi * sqrt(-R0[0] / R2[0])) + 0.5 * u * u / R0[0]

        options['xcscale'] = XcScale
#         opt0 = [options[n] for n in ('SCIS', 'XcScale', 'ABSEPS', 'RELEPS',
#                                      'COVEPS', 'MAXPTS', 'MINPTS', 'seed',
#                                      'NIT1')]
        rind = Rind(**options)
        if (Nx > 1):
            # (M,m) or (M,m)v distribution wanted
            if def_nr in [0, 2]:
                asize = [Nx1, Nx1]
            else:
                # (M,m,TMm), (M,m,TMm)v  (M,m,TMd)v or (M,M,Tdm)v
                # distributions wanted
                asize = [Nx1, Nx1, Ntime]
        elif (def_nr > 3):
            # Conditional distribution for (TMd,TMm)v or (Tdm,TMm)v given (M,m)
            # wanted
            asize = [1, Ntime, Ntime]
        else:
            # Conditional distribution for  (TMm) or (TMm)v given (M,m) wanted
            asize = [1, 1, Ntime]
        # Initialization
        pdf = zeros(asize)
        err = zeros(asize)
        terr = zeros(asize)

        BIG = zeros(Ntime + Nc + 1, Ntime + Nc + 1)
        ex = zeros(1, Ntime + Nc + 1)
        # fxind = zeros(Nx,1)
        xc = zeros(Nc, Nx)

        indI = zeros(1, NI)
        a_up = zeros(1, NI - 1)
        a_lo = zeros(1, NI - 1)

        # INFIN =  INTEGER, array of integration limits flags: size 1 x Nb (in)
        #            if INFIN(I) < 0, Ith limits are (-infinity, infinity)
        #            if INFIN(I) = 0, Ith limits are (-infinity, Hup(I)]
        #            if INFIN(I) = 1, Ith limits are [Hlo(I), infinity)
        #            if INFIN(I) = 2, Ith limits are [Hlo(I), Hup(I)].
        # INFIN = repmat(0,1,NI-1)
        # INFIN(3)  = 1
        a_up[0, 2] = +XdInf
        a_lo[0, :2] = [-XtInf, -XdInf]
        if (def_nr > 3):
            a_lo[0, 3] = -XtInf

        IJ = 0
        if (def_nr <= 1):  # Max2min and period/wavelength
            for I in range(1, Nx1):
                J = IJ + I
                xc[2, IJ:J] = hg[I]
                xc[3, IJ:J] = hg[:I].T
                IJ = J
        else:
            # Level u separated Max2min
            xc[Nc, :] = u
            # Hg(1) = Hg(Nx1+1)= u => start do loop at I=2 since by definition
            # we must have:  minimum<u-level<Maximum
            for i in range(1, Nx1):
                J = IJ + Nx1
                xc[2, IJ:J] = hg[i]  # Max > u
                xc[3, IJ:J] = hg[Nx1 + 2: 2 * Nx1].T  # Min < u
                IJ = J
        if (def_nr <= 3):
            # h11 = fwaitbar(0,[],sprintf('Please wait ...(start at:
            # %s)',datestr(now)))

            for Ntd in range(Nstart, Ntime):
                # Ntd=tn
                Ntdc = Ntd + Nc
                Nt = Ntd - Nd
                indI[1] = Nt
                indI[2] = Nt + 1
                indI[3] = Ntd
                # positive wave period
                # self._covinput_mmt_pdf(BIG, R, tn, ts, tnold)
                BIG[:Ntdc, :Ntdc] = covinput(BIG[:Ntdc, :Ntdc], R, Ntd, 0)
                [fxind, err0, terr0] = rind(BIG[:Ntdc, :Ntdc], ex[:Ntdc],
                                            a_lo, a_up, indI, xc, Nt)
                # fxind  = CC*rind(BIG(1:Ntdc,1:Ntdc),ex(1:Ntdc),xc,Nt,NIT1,
                # speed1,indI,a_lo,a_up)
                if (Nx < 2):
                    # Density of TMm given the Max and the Min. Note that the
                    # density is not scaled to unity
                    pdf[0, 0, Ntd] = fxind[0]
                    err[0, 0, Ntd] = err0[0] ** 2
                    terr[0, 0, Ntd] = terr0[0]
                    # GOTO 100
                else:
                    IJ = 0
                    # joint density of (Ac,At),(M,m_rfc) or (M,m).
                    if def_nr in [-2, -1, 0]:
                        for i in range(1, Nx1):
                            J = IJ + i
                            pdf[:i, i, 0] += fxind[IJ:J].T * dt  # *CC
                            err[:i, i, 0] += (err0[IJ + 1:J].T * dt) ** 2
                            terr[:i, i, 0] += (terr0[IJ:J].T * dt)
                            IJ = J
                    elif def_nr == 1:  # joint density of (M,m,TMm)
                        for i in range(1, Nx1):
                            J = IJ + i
                            pdf[:i, i, Ntd] = fxind[IJ:J].T  # %*CC
                            err[:i, i, Ntd] = (err0[IJ:J].T) ** 2  # %*CC
                            terr[:i, i, Ntd] = (terr0[IJ:J].T)  # %*CC
                            IJ = J
                        # end %do
                    # joint density of level v separated (M,m)v
                    elif def_nr == 2:
                        for i in range(1, Nx1):
                            J = IJ + Nx1
                            pdf[1:Nx1, i, 0] += fxind[IJ:J].T * dt  # %*CC
                            err[1:Nx1, i, 0] += (err0[IJ:J].T * dt) ** 2
                            terr[1:Nx1, i, 0] += (terr0[IJ:J].T * dt)
                            IJ = J
                        # end %do
                    elif def_nr == 3:
                        # joint density of level v separated (M,m,TMm)v
                        for i in range(1, Nx1):
                            J = IJ + Nx1
                            pdf[1:Nx1, i, Ntd] += fxind[IJ:J].T  # %*CC
                            err[1:Nx1, i, Ntd] += (err0[IJ:J].T) ** 2
                            terr[1:Nx1, i, Ntd] += (terr0[IJ:J].T)
                            IJ = J
                        # end %do
                    # end % SELECT
                # end %ENDIF
                # waitTxt = sprintf('%s Ready: %d of %d',datestr(now),Ntd,Ntime)
                # fwaitbar(Ntd/Ntime,h11,waitTxt)

            # end %do
            # close(h11)
            err = sqrt(err)
            #   goto 800
        else:  # def_nr>3
            # 200 continue
            # waitTxt = sprintf('Please wait ...(start at: %s)',datestr(now))
            # h11 = fwaitbar(0,[],waitTxt)
            tnold = -1
            for tn in range(Nstart, Ntime):
                Ntd = tn + 1
                Ntdc = Ntd + Nc
                Nt = Ntd - Nd
                indI[1] = Nt
                indI[2] = Nt + 1
                indI[3] = Nt + 2
                indI[4] = Ntd

                if not symmetry:  # IF (SYMMETRY) GOTO 300
                    for ts in range(1, tn - 1):  # = 2:tn-1:
                        # positive wave period
                        BIG[:Ntdc, :Ntdc] = covinput(BIG[:Ntdc, :Ntdc],
                                                     R, tn, ts, tnold)
                        fxind, err0, terr0 = rind(BIG[:Ntdc, :Ntdc], ex[:Ntdc],
                                                  a_lo, a_up, indI, xc, Nt)

                        # tnold = tn
                        if def_nr in [3, 4]:
                            if (Nx == 1):
                                # Joint density (TMd,TMm) given the Max and the min.
                                # Note the density is not scaled to unity
                                pdf[0, ts, tn] = fxind[0]  # *CC
                                err[0, ts, tn] = err0[0] ** 2  # *CC
                                terr[0, ts, tn] = terr0[0]  # *CC
                            else:
                                # 4,  gives level u separated Max2min and wave period
                                # from Max to the crossing of level u
                                # (M,m,TMd).
                                IJ = 0
                                for i in range(1, Nx1):
                                    J = IJ + Nx1
                                    pdf[1:Nx1, i, ts] += fxind[IJ:J].T * dt
                                    err[1:Nx1, i, ts] += (err0[IJ:J].T * dt) ** 2
                                    terr[1:Nx1, i, ts] += (terr0[IJ:J].T * dt)
                                    IJ = J
                                # end %do
                            # end
                        elif def_nr == 5:
                            if (Nx == 1):
                                #  Joint density (Tdm,TMm) given the Max and the min.
                                #  Note the density is not scaled to unity
                                pdf[0, tn - ts, tn] = fxind[0]  #  %*CC
                                err[0, tn - ts, tn] = err0[0] ** 2
                                terr[0, tn - ts, tn] = terr0[0]
                            else:
                                #  5,  gives level u separated Max2min and wave period from
                                #  the crossing of level u to the min (M,m,Tdm).

                                IJ = 0
                                for i in range(1, Nx1):  # = 2:Nx1
                                    J = IJ + Nx1
                                    # %*CC
                                    pdf[1:Nx1, i, tn - ts] += fxind[IJ:J].T * dt
                                    err[1:Nx1, i, tn - ts] += (err0[IJ:J].T * dt) ** 2
                                    terr[1:Nx1, i, tn - ts] += (terr0[IJ:J].T * dt)
                                    IJ = J
                                # end %do
                            # end
                        # end % SELECT
                    # end%         enddo
                else:  # % exploit symmetry
                    # 300   Symmetry
                    for ts in range(1, Ntd // 2):  # = 2:floor(Ntd//2)
                        #  Using the symmetry since U = 0 and the transformation is
                        #  linear.
                        #  positive wave period
                        BIG[:Ntdc, :Ntdc] = covinput(BIG[:Ntdc, :Ntdc],
                                                     R, tn, ts, tnold)
                        fxind, err0, terr0 = rind(BIG[:Ntdc, :Ntdc], ex[:Ntdc],
                                                    a_lo, a_up, indI, xc, Nt)

                        #[fxind,err0] = rind(BIG(1:Ntdc,1:Ntdc),ex,a_lo,a_up,indI, xc,Nt,opt0{:})
                        # tnold = tn
                        if (Nx == 1):  # % THEN
                            #  Joint density of (TMd,TMm),(Tdm,TMm) given the max and
                            #  the min.
                            #  Note that the density is not scaled to unity
                            pdf[0, ts, tn] = fxind[0]  # %*CC
                            err[0, ts, tn] = err0[0] ** 2
                            err[0, ts, tn] = terr0(1)
                            if (ts < tn - ts):  # %THEN
                                pdf[0, tn - ts, tn] = fxind[0]  # *CC
                                err[0, tn - ts, tn] = err0[0] ** 2
                                terr[0, tn - ts, tn] = terr0[0]
                            # end
                            # GOTO 350
                        else:
                            IJ = 0
                            if def_nr == 4:
                                #  4,  gives level u separated Max2min and wave period from
                                #  Max to the crossing of level u (M,m,TMd).
                                for i in range(1, Nx1):
                                    J = IJ + Nx1
                                    pdf[1:Nx1, i, ts] += fxind[IJ:J] * dt  # %*CC
                                    err[1:Nx1, i, ts] += (err0[IJ:J] * dt) ** 2
                                    terr[1:Nx1, i, ts] += (terr0[IJ:J] * dt)
                                    if (ts < tn - ts):
                                        #  exploiting the symmetry
                                        # %*CC
                                        pdf[i, 1:Nx1, tn - ts] += fxind[IJ:J] * dt
                                        err[i, 1:Nx1, tn - ts] += (err0[IJ:J] * dt) ** 2
                                        terr[i, 1:Nx1, tn - ts] += (terr0[IJ:J] * dt)
                                    # end
                                    IJ = J
                                # end %do
                            elif def_nr == 5:
                                #  5,   gives level u separated Max2min and wave period
                                #  from the crossing of level u to min (M,m,Tdm).
                                for i in range(1, Nx1):  # = 2:Nx1,
                                    J = IJ + Nx1
                                    pdf[1:Nx1, i, tn - ts] += fxind[IJ:J] * dt
                                    err[1:Nx1, i, tn - ts] += (err0[IJ:J] * dt) ** 2
                                    terr[
                                        1:Nx1, i, tn - ts] += (terr0[IJ:J] * dt)
                                    if (ts < tn - ts + 1):
                                        # exploiting the symmetry
                                        pdf[i, 1:Nx1, ts] += fxind[IJ:J] * dt
                                        err[i, 1:Nx1, ts] += (err0[IJ:J] * dt) ** 2
                                        terr[i, 1:Nx1, ts] += (terr0[IJ:J] * dt)
                                    # end %ENDIF
                                    IJ = J
                                # end %do
                            # end %END SELECT
                        # end
                        # 350
                    # end %do
                # end
                # waitTxt = sprintf('%s Ready: %d of %d',datestr(now),tn,Ntime)
                # fwaitbar(tn/Ntime,h11,waitTxt)
                # 400     print *,'Ready: ',tn,' of ',Ntime
            # end %do
            # close(h11)
            err = sqrt(err)
        # end % if

        # Nx1,size(pdf) def  Ntime
        if (Nx > 1):  # % THEN
            IJ = 1
            if (def_nr > 2 or def_nr == 1):
                IJ = Ntime
            # end
            pdf = pdf[:Nx1, :Nx1, :IJ]
            err = err[:Nx1, :Nx1, :IJ]
            terr = terr[:Nx1, :Nx1, :IJ]
        else:
            IJ = 1
            if (def_nr > 3):
                IJ = Ntime
            # end
            pdf = np.squeeze(pdf[0, :IJ, :Ntime])
            err = np.squeeze(err[0, :IJ, :Ntime])
            terr = np.squeeze(terr[0, :IJ, :Ntime])
        # end
        return pdf, err, terr, options

    def _covinput_mmt_pdf(self, BIG, R, tn, ts, tnold=-1):
        """
        COVINPUT Sets up the covariance matrix

         CALL BIG = covinput(BIG, R0,R1,R2,R3,R4,tn,ts)

          BIG = covariance matrix for X = [Xt,Xd,Xc] in spec2mmtpdf problems.

        The order of the variables in the covariance matrix are organized as
        follows:
        for  ts <= 1:
        X'(t2)..X'(ts),...,X'(tn-1) X''(t1),X''(tn)  X'(t1),X'(tn),X(t1),X(tn)
        = [          Xt               |      Xd       |          Xc          ]

        for ts > =2:
        X'(t2)..X'(ts),...,X'(tn-1) X''(t1),X''(tn) X'(ts)  X'(t1),X'(tn),X(t1),X(tn) X(ts)
        = [          Xt               |      Xd               |          Xc             ]

        where

         Xt = time points in the indicator function
         Xd = derivatives
         Xc = variables to condition on

        Computations of all covariances follows simple rules: Cov(X(t),X(s)) = r(t,s),
        then  Cov(X'(t),X(s))=dr(t,s)/dt.  Now for stationary X(t) we have
        a function r(tau) such that Cov(X(t),X(s))=r(s-t) (or r(t-s) will give the same result).

        Consequently  Cov(X'(t),X(s))    = -r'(s-t)    = -sign(s-t)*r'(|s-t|)
                       Cov(X'(t),X'(s))   = -r''(s-t)   = -r''(|s-t|)
                       Cov(X''(t),X'(s))  =  r'''(s-t)  = sign(s-t)*r'''(|s-t|)
                       Cov(X''(t),X(s))   =  r''(s-t)   = r''(|s-t|)
        Cov(X''(t),X''(s)) =  r''''(s-t) = r''''(|s-t|)
        """
        R0, R1, R2, R3, R4 = R[:, :5].T
        if (ts > 1):
            shft = 1
            N = tn + 5 + shft
            # Cov(Xt,Xc)
            # for
            i = np.arange(tn - 2)  # 1:tn-2
            # j = abs(i+1-ts)
            # BIG(i,N)  = -sign(R1(j+1),R1(j+1)*dble(ts-i-1)) %cov(X'(ti+1),X(ts))
            j = i + 1 - ts
            tau = abs(j)
            # BIG(i,N)  = abs(R1(tau)).*sign(R1(tau).*j.')
            BIG[i, N] = R1[tau] * sign(j)
            # end do
            # Cov(Xc)
            BIG[N, N] = R0[0]       # cov(X(ts),X(ts))
            BIG[tn + shft + 1, N] = -R1[ts]      # cov(X'(t1),X(ts))
            BIG[tn + shft + 2, N] = R1[tn - ts]  # cov(X'(tn),X(ts))
            BIG[tn + shft + 3, N] = R0[ts]      # cov(X(t1),X(ts))
            BIG[tn + shft + 4, N] = R0[tn - ts]  # cov(X(tn),X(ts))
            # Cov(Xd,Xc)
            BIG[tn - 1, N] = R2[ts]  # %cov(X''(t1),X(ts))
            BIG[tn, N] = R2[tn - ts]  # %cov(X''(tn),X(ts))

            # ADD a level u crossing  at ts

            # Cov(Xt,Xd)
            # for
            i = np.arange(tn - 2)  # 1:tn-2
            j = abs(i + 1 - ts)
            BIG[i, tn + shft] = -R2[j]  #  %cov(X'(ti+1),X'(ts))
            # end do
            # Cov(Xd)
            BIG[tn + shft, tn + shft] = -R2[0]  #   %cov(X'(ts),X'(ts))
            BIG[tn - 1, tn + shft] = R3[ts]  #  %cov(X''(t1),X'(ts))
            BIG[tn, tn + shft] = -R3[tn - ts]  #   %cov(X''(tn),X'(ts))

            # Cov(Xd,Xc)
            BIG[tn + shft, N] = 0.0  # %cov(X'(ts),X(ts))
            #       % cov(X'(ts),X'(t1))
            BIG[tn + shft, tn + shft + 1] = -R2[ts]
            # % cov(X'(ts),X'(tn))
            BIG[tn + shft, tn + shft + 2] = -R2[tn - ts]
            BIG[tn + shft, tn + shft + 3] = R1[ts]  # % cov(X'(ts),X(t1))
            #  % cov(X'(ts),X(tn))
            BIG[tn + shft, tn + shft + 4] = -R1[tn - ts]

            if (tnold == tn):
                #  A previous call to covinput with tn==tnold has been made
                #  need only to update  row and column N and tn+1 of big:
                return BIG
        #       % make lower triangular part equal to upper and then return
        #       for j=1:tn+shft
        #          BIG(N,j)      = BIG(j,N)
        #          BIG(tn+shft,j) = BIG(j,tn+shft)
        #       end
        #       for j=tn+shft+1:N-1
        #          BIG(N,j) = BIG(j,N)
        #          BIG(j,tn+shft) = BIG(tn+shft,j)
        #       end
        #       return
        #   end %if
        #   %tnold = tn
        else:
            # N = tn+4
            shft = 0
        # end %if

        if (tn > 2):
            # for i=1:tn-2
            # cov(Xt)
            #    for j=i:tn-2
            #      BIG(i,j) = -R2(j-i+1)              % cov(X'(ti+1),X'(tj+1))
            #   end %do

            # % cov(Xt) =   % cov(X'(ti+1),X'(tj+1))
            BIG[:tn - 2, :tn - 2] = toeplitz(-R2[:tn - 2])

            # cov(Xt,Xc)
            BIG[:tn - 2, tn + shft] = -R2[1:tn - 1]  # cov(X'(ti+1),X'(t1))
            # cov(X'(ti+1),X'(tn))
            BIG[:tn - 2, tn + shft + 1] = -R2[tn - 2:0:-1]
            BIG[:tn - 2, tn + shft + 2] = R1[1:tn - 1]  # cov(X'(ti+1),X(t1))
            # cov(X'(ti+1),X(tn))
            BIG[:tn - 2, tn + shft + 3] = -R1[tn - 2:0:-1]

            # Cov(Xt,Xd)
            BIG[:tn - 2, tn - 2] = R3[1:tn - 1]     # cov(X'(ti+1),X''(t1))
            BIG[:tn - 2, tn - 1] = -R3[tn - 2:0:-1]  # cov(X'(ti+1),X''(tn))
            # end %do
        # end
        # cov(Xd)
        BIG[tn - 2, tn - 2] = R4[0]
        BIG[tn - 2, tn - 1] = R4[tn - 1]  # cov(X''(t1),X''(tn))
        BIG[tn - 1, tn - 1] = R4[0]

        # cov(Xc)
        BIG[tn + shft + 2, tn + shft + 2] = R0[0]        # cov(X(t1),X(t1))
        # cov(X(t1),X(tn))
        BIG[tn + shft + 2, tn + shft + 3] = R0[tn - 1]
        BIG[tn + shft + 1, tn + shft + 2] = 0.0        # cov(X(t1),X'(t1))
        # cov(X(t1),X'(tn))
        BIG[tn + shft + 1, tn + shft + 2] = R1[tn - 1]
        BIG[tn + shft + 3, tn + shft + 3] = R0[0]       # cov(X(tn),X(tn))
        BIG[tn + shft, tn + shft + 3] = -R1[tn - 1]       # cov(X(tn),X'(t1))
        BIG[tn + shft + 1, tn + shft + 3] = 0.0         # cov(X(tn),X'(tn))
        BIG[tn + shft, tn + shft] = -R2[0]        # cov(X'(t1),X'(t1))
        BIG[tn + shft, tn + shft + 1] = -R2[tn - 1]       # cov(X'(t1),X'(tn))
        BIG[tn + shft + 1, tn + shft + 1] = -R2[0]        # cov(X'(tn),X'(tn))
        # Xc=X(t1),X(tn),X'(t1),X'(tn)
        # Xd=X''(t1),X''(tn)
        # cov(Xd,Xc)
        BIG[tn - 2, tn + shft + 2] = R2[0]  # cov(X''(t1),X(t1))
        BIG[tn - 2, tn + shft + 3] = R2[tn - 1]  # cov(X''(t1),X(tn))
        BIG[tn - 2, tn + shft] = 0.0             # cov(X''(t1),X'(t1))
        BIG[tn - 2, tn + shft + 1] = R3[tn - 1]        # cov(X''(t1),X'(tn))
        BIG[tn - 1, tn + shft + 2] = R2[tn - 1]        # cov(X''(tn),X(t1))
        BIG[tn - 1, tn + shft + 3] = R2[0]           # cov(X''(tn),X(tn))
        BIG[tn - 1, tn + shft] = -R3[tn - 1]          # cov(X''(tn),X'(t1))
        BIG[tn - 1, tn + shft + 1] = 0.0            # cov(X''(tn),X'(tn))

        # make lower triangular part equal to upper
        # for j=1:N-1
        #   for i=j+1:N
        #      BIG(i,j) = BIG(j,i)
        # end #do
        # end #do
        lp = np.flatnonzero(np.tril(ones(BIG.shape)))  # indices to lower triangular part
        BIGT = BIG.T
        BIG[lp] = BIGT[lp]
        return BIG
        # END  SUBROUTINE COV_INPUT

    def _cov2mmtpdfexe(self, R, dt, u, defnr, Nstart, hg, options):
        # Write parameters to file
        Nx = max(1, len(hg))
        if defnr > 1:
            Nx = Nx // 2  # level v separated max2min densities wanted

        Ntime = R.shape[0]

        filenames = ['h.in', 'reflev.in']
        self._cleanup(*filenames)

        with open('h.in', 'wt') as f:
            f.write('%12.10f\n', hg)

        # XSPLT = options.xsplit
        nit = options.nit
        speed = options.speed
        seed = options.seed
        SCIS = abs(options.method)  # method<=0

        with open('reflev.in', 'wt') as fid:
            fid.write('%2.0f \n', Ntime)
            fid.write('%2.0f \n', Nstart)
            fid.write('%2.0f \n', nit)
            fid.write('%2.0f \n', speed)
            fid.write('%2.0f \n', SCIS)
            fid.write('%2.0f \n', seed)
            fid.write('%2.0f \n', Nx)
            fid.write('%12.10E \n', dt)
            fid.write('%12.10E \n', u)
            fid.write('%2.0f \n', defnr)

        filenames2 = self._writecov(R)

        print('   Starting Fortran executable.')
        # compiled cov2mmtpdf.f with rind70.f
        # dos([ wafoexepath 'cov2mmtpdf.exe'])

        dens =  1  # load('dens.out')

        self._cleanup(*filenames)
        self._cleanup(*filenames2)

        return dens

    def _cleanup(self, *files):
        '''Removes files from harddisk if they exist'''
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    def to_specnorm(self):
        S = self.copy()
        S.normalize()
        return S

    def sim(self, ns=None, cases=1, dt=None, iseed=None, method='random',
            derivative=False):
        ''' Simulates a Gaussian process and its derivative from spectrum

        Parameters
        ----------
        ns : scalar
            number of simulated points.  (default length(spec)-1=n-1).
                     If ns>n-1 it is assummed that acf(k)=0 for all k>n-1
        cases : scalar
            number of replicates (default=1)
        dt : scalar
            step in grid (default dt is defined by the Nyquist freq)
        iseed : int or state
            starting state/seed number for the random number generator
            (default none is set)
        method : string
            if 'exact'  : simulation using cov2sdat
            if 'random' : random phase and amplitude simulation (default)
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
        Gaussian process through circulant embedding of the covariance matrix
        or by summation of sinus functions with random amplitudes and random
        phase angle.

        If the spectrum has a non-empty field .tr, then the transformation is
        applied to the simulated data, the result is a simulation of a
        transformed Gaussian process.

        Note: The method 'exact' simulation may give high frequency ripple when
        used with a small dt. In this case the method 'random' works better.

        Example:
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap();S = Sj.tospecdata()
        >>> ns =100; dt = .2
        >>> x1 = S.sim(ns,dt=dt)

        >>> import numpy as np
        >>> import scipy.stats as st
        >>> x2 = S.sim(20000,20)
        >>> truth1 = [0,np.sqrt(S.moment(1)[0]),0., 0.]
        >>> funs = [np.mean,np.std,st.skew,st.kurtosis]
        >>> for fun,trueval in zip(funs,truth1):
        ...     res = fun(x2[:,1::],axis=0)
        ...     m = res.mean()
        ...     sa = res.std()
        ...     #trueval, m, sa
        ...     np.abs(m-trueval)<sa
        True
        array([ True], dtype=bool)
        True
        True

        waveplot(x1,'r',x2,'g',1,1)

        See also
        --------
        cov2sdat, gaus2dat

        Reference
        -----------
        C.S Dietrich and G. N. Newsam (1997)
        "Fast and exact simulation of stationary
        Gaussian process through circulant embedding
        of the Covariance matrix"
        SIAM J. SCI. COMPT. Vol 18, No 4, pp. 1088-1107

        Hudspeth, S.T. and Borgman, L.E. (1979)
        "Efficient FFT simulation of Digital Time sequences"
        Journal of the Engineering Mechanics Division, ASCE, Vol. 105, No. EM2,

        '''

        spec = self.copy()
        if dt is not None:
            spec.resample(dt)

        ftype = spec.freqtype
        freq = spec.args

        d_t = spec.sampling_period()
        Nt = freq.size

        if ns is None:
            ns = Nt - 1

        if method in 'exact':

            # nr=0,Nt=None,dt=None
            acf = spec.tocovdata(nr=0)
            T = Nt * d_t
            i = flatnonzero(acf.args > T)

            # Trick to avoid adding high frequency noise to the spectrum
            if i.size > 0:
                acf.data[i[0]::] = 0.0

            return acf.sim(ns=ns, cases=cases, iseed=iseed,
                           derivative=derivative)

        _set_seed(iseed)

        ns = ns + mod(ns, 2)  # make sure it is even

        f_i = freq[1:-1]
        s_i = spec.data[1:-1]
        if ftype in ('w', 'k'):
            fact = 2. * pi
            s_i = s_i * fact
            f_i = f_i / fact

        x = zeros((ns, cases + 1))

        d_f = 1 / (ns * d_t)

        # interpolate for freq.  [1:(N/2)-1]*d_f and create 2-sided, uncentered
        # spectra
        f = arange(1, ns / 2.) * d_f

        f_u = hstack((0., f_i, d_f * ns / 2.))
        s_u = hstack((0., abs(s_i) / 2., 0.))

        s_i = interp(f, f_u, s_u)
        s_u = hstack((0., s_i, 0, s_i[(ns / 2) - 2::-1]))
        del(s_i, f_u)

        # Generate standard normal random numbers for the simulations
        randn = random.randn
        z_r = randn((ns / 2) + 1, cases)
        z_i = vstack(
            (zeros((1, cases)), randn((ns / 2) - 1, cases), zeros((1, cases))))

        amp = zeros((ns, cases), dtype=complex)
        amp[0:(ns / 2 + 1), :] = z_r - 1j * z_i
        del(z_r, z_i)
        amp[(ns / 2 + 1):ns, :] = amp[ns / 2 - 1:0:-1, :].conj()
        amp[0, :] = amp[0, :] * sqrt(2.)
        amp[(ns / 2), :] = amp[(ns / 2), :] * sqrt(2.)

        # Make simulated time series
        T = (ns - 1) * d_t
        Ssqr = sqrt(s_u * d_f / 2.)

        # stochastic amplitude
        amp = amp * Ssqr[:, newaxis]

        # Deterministic amplitude
        # amp =
        # sqrt[1]*Ssqr(:,ones(1,cases)) * \
        #            exp(sqrt(-1)*atan2(imag(amp),real(amp)))
        del(s_u, Ssqr)

        x[:, 1::] = fft(amp, axis=0).real
        x[:, 0] = linspace(0, T, ns)  # ' %(0:d_t:(np-1)*d_t).'

        if derivative:
            xder = zeros(ns, cases + 1)
            w = 2. * pi * hstack((0, f, 0., -f[-1::-1]))
            amp = -1j * amp * w[:, newaxis]
            xder[:, 1:(cases + 1)] = fft(amp, axis=0).real
            xder[:, 0] = x[:, 0]

        if spec.tr is not None:
            # print('   Transforming data.')
            g = spec.tr
            if derivative:
                for i in range(cases):
                    x[:, i + 1], xder[:, i + 1] = g.gauss2dat(x[:, i + 1],
                                                              xder[:, i + 1])
            else:
                for i in range(cases):
                    x[:, i + 1] = g.gauss2dat(x[:, i + 1])

        if derivative:
            return x, xder
        else:
            return x

# function [x2,x,svec,dvec,amp]=spec2nlsdat(spec,np,dt,iseed,method,
#                                truncationLimit)
    def sim_nl(self, ns=None, cases=1, dt=None, iseed=None, method='random',
               fnlimit=1.4142, reltol=1e-3, g=9.81, verbose=False,
               output='timeseries'):
        """
        Simulates a Randomized 2nd order non-linear wave X(t)

        Parameters
        ----------
        ns : scalar
            number of simulated points.  (default length(spec)-1=n-1).
            If ns>n-1 it is assummed that R(k)=0 for all k>n-1
        cases : scalar
            number of replicates (default=1)
        dt : scalar
            step in grid (default dt is defined by the Nyquist freq)
        iseed : int or state
            starting state/seed number for the random number generator
            (default none is set)
        method : string
            'apStochastic'    : Random amplitude and phase (default)
            'aDeterministic'  : Deterministic amplitude and random phase
            'apDeterministic' : Deterministic amplitude and phase
        fnlimit : scalar
            normalized upper frequency limit of spectrum for 2'nd order
            components. The frequency is normalized with
            sqrt(gravity*tanh(kbar*water_depth)/amp_max)/(2*pi)
            (default sqrt(2), i.e., Convergence criterion [1]_).
            Other possible values are:
            sqrt(1/2)  : No bump in trough criterion
            sqrt(pi/7) : Wave steepness criterion
        reltol : scalar
            relative tolerance defining where to truncate spectrum for the
            sum and difference frequency effects


        Returns
        -------
        xs2 = a cases+1 column matrix  ( t,X1(t) X2(t) ...).
        xs1 = a cases+1 column matrix  ( t,X1'(t) X2'(t) ...).

        Details
        -------
        Performs a Fast simulation of Randomized 2nd order non-linear
        waves by summation of sinus functions with random amplitudes and
        phase angles. The extent to which the simulated result are applicable
        to real seastates are dependent on the validity of the assumptions:

        1.  Seastate is unidirectional
        2.  Surface elevation is adequately represented by 2nd order random
            wave theory
        3.  The first order component of the surface elevation is a Gaussian
            random process.

        If the spectrum does not decay rapidly enough towards zero, the
        contribution from the 2nd order wave components at the upper tail can
        be very large and unphysical. To ensure convergence of the perturbation
        series, the upper tail of the spectrum is truncated at FNLIMIT in the
        calculation of the 2nd order wave components, i.e., in the calculation
        of sum and difference frequency effects. This may also be combined with
        the elimination of second order effects from the spectrum, i.e.,
        extract the linear components from the spectrum. One way to do this is
        to use SPEC2LINSPEC.

        Example
        --------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap();S = Sj.tospecdata()
        >>> ns =100; dt = .2
        >>> x1 = S.sim_nl(ns,dt=dt)

        >>> import numpy as np
        >>> import scipy.stats as st
        >>> x2, x1 = S.sim_nl(ns=20000,cases=20)
        >>> truth1 = [0,np.sqrt(S.moment(1)[0][0])] + S.stats_nl(moments='sk')
        >>> truth1[-1] = truth1[-1]-3
        >>> np.round(truth1, 3)
        array([ 0.   ,  1.75 ,  0.187,  0.062])

        >>> funs = [np.mean,np.std,st.skew,st.kurtosis]
        >>> for fun,trueval in zip(funs,truth1):
        ...     res = fun(x2[:,1::], axis=0)
        ...     m = res.mean()
        ...     sa = res.std()
        ...     #trueval, m, sa
        ...     np.abs(m-trueval)<sa
        True
        True
        True
        True

        >>> x = []
        >>> for i in range(20):
        ...     x2, x1 = S.sim_nl(ns=20000,cases=1)
        ...     x.append(x2[:,1::])
        >>> x2 = np.hstack(x)
        >>> truth1 = [0,np.sqrt(S.moment(1)[0][0])] + S.stats_nl(moments='sk')
        >>> truth1[-1] = truth1[-1]-3
        >>> np.round(truth1,3)
        array([ 0.   ,  1.75 ,  0.187,  0.062])

        >>> funs = [np.mean,np.std,st.skew,st.kurtosis]
        >>> for fun,trueval in zip(funs,truth1):
        ...     res = fun(x2, axis=0)
        ...     m = res.mean()
        ...     sa = res.std()
        ...     #trueval, m, sa
        ...     np.abs(m-trueval)<sa
        True
        True
        True
        True

        assert(np.abs(m-trueval)<sa, fun.__name__)

        np =100; dt = .2
        [x1, x2] = spec2nlsdat(jonswap,np,dt)
        waveplot(x1,'r',x2,'g',1,1)

        See also
        --------
          spec2linspec, spec2sdat, cov2sdat

        References
        ----------
        .. [1] Nestegaard, amp  and Stokka T (1995)
                amp Third Order Random Wave model.
                In proc.ISOPE conf., Vol III, pp 136-142.

        .. [2] R. spec Langley (1987)
                amp statistical analysis of non-linear random waves.
                Ocean Engng, Vol 14, pp 389-407

        .. [3] Marthinsen, T. and Winterstein, spec.R (1992)
                'On the skewness of random surface waves'
                In proc. ISOPE Conf., San Francisco, 14-19 june.
        """

        # TODO % Check the methods: 'apdeterministic' and 'adeterministic'
        Hm0, Tm02 = self.characteristic(['Hm0', 'Tm02'])[0].tolist()

        _set_seed(iseed)

        spec = self.copy()
        if dt is not None:
            spec.resample(dt)

        ftype = spec.freqtype
        freq = spec.args

        d_t = spec.sampling_period()
        Nt = freq.size

        if ns is None:
            ns = Nt - 1

        ns = ns + mod(ns, 2)  # make sure it is even

        f_i = freq[1:-1]
        s_i = spec.data[1:-1]
        if ftype in ('w', 'k'):
            fact = 2. * pi
            s_i = s_i * fact
            f_i = f_i / fact

        s_max = max(s_i)
        water_depth = min(abs(spec.h), 10. ** 30)

        x = zeros((ns, cases + 1))

        df = 1 / (ns * d_t)

        # interpolate for freq.  [1:(N/2)-1]*df and create 2-sided, uncentered
        # spectra
        f = arange(1, ns / 2.) * df
        f_u = hstack((0., f_i, df * ns / 2.))
        w = 2. * pi * hstack((0., f, df * ns / 2.))
        kw = w2k(w, 0., water_depth, g)[0]
        s_u = hstack((0., abs(s_i) / 2., 0.))

        s_i = interp(f, f_u, s_u)
        nmin = (s_i > s_max * reltol).argmax()
        nmax = flatnonzero(s_i > 0).max()
        s_u = hstack((0., s_i, 0, s_i[(ns / 2) - 2::-1]))
        del(s_i, f_u)

        # Generate standard normal random numbers for the simulations
        randn = random.randn
        z_r = randn((ns / 2) + 1, cases)
        z_i = vstack((zeros((1, cases)),
                      randn((ns / 2) - 1, cases),
                      zeros((1, cases))))

        amp = zeros((ns, cases), dtype=complex)
        amp[0:(ns / 2 + 1), :] = z_r - 1j * z_i
        del(z_r, z_i)
        amp[(ns / 2 + 1):ns, :] = amp[ns / 2 - 1:0:-1, :].conj()
        amp[0, :] = amp[0, :] * sqrt(2.)
        amp[(ns / 2), :] = amp[(ns / 2), :] * sqrt(2.)

        # Make simulated time series
        T = (ns - 1) * d_t
        Ssqr = sqrt(s_u * df / 2.)

        if method.startswith('apd'):  # apdeterministic
            # Deterministic amplitude and phase
            amp[1:(ns / 2), :] = amp[1, 0]
            amp[(ns / 2 + 1):ns, :] = amp[1, 0].conj()
            amp = sqrt(2) * Ssqr[:, newaxis] * \
                exp(1J * arctan2(amp.imag, amp.real))
        elif method.startswith('ade'):  # adeterministic
            # Deterministic amplitude and random phase
            amp = sqrt(2) * Ssqr[:, newaxis] * \
                exp(1J * arctan2(amp.imag, amp.real))
        else:
            # stochastic amplitude
            amp = amp * Ssqr[:, newaxis]
        # Deterministic amplitude
        # amp =
        # sqrt(2)*Ssqr(:,ones(1,cases))* \
        #                exp(sqrt(-1)*atan2(imag(amp),real(amp)))
        del(s_u, Ssqr)

        x[:, 1::] = fft(amp, axis=0).real
        x[:, 0] = linspace(0, T, ns)  # ' %(0:d_t:(np-1)*d_t).'

        x2 = x.copy()

        # If the spectrum does not decay rapidly enough towards zero, the
        # contribution from the wave components at the  upper tail can be very
        # large and unphysical.
        # To ensure convergence of the perturbation series, the upper tail of
        # the spectrum is truncated in the calculation of sum and difference
        # frequency effects.
        # Find the critical wave frequency to ensure convergence.

        num_waves = 1000.  # Typical number of waves in 3 hour seastate
        kbar = w2k(2. * pi / Tm02, 0., water_depth)[0]
        # Expected maximum amplitude for 1000 waves seastate
        amp_max = sqrt(2 * log(num_waves)) * Hm0 / 4

        f_limit_up = fnlimit * \
            sqrt(g * tanh(kbar * water_depth) / amp_max) / (2 * pi)
        f_limit_lo = sqrt(g * tanh(kbar * water_depth) *
                          amp_max / water_depth) / (2 * pi * water_depth)

        nmax = min(flatnonzero(f <= f_limit_up).max(), nmax) + 1
        nmin = max(flatnonzero(f_limit_lo <= f).min(), nmin) + 1

        # if isempty(nmax),nmax = np/2end
        # if isempty(nmin),nmin = 2end % Must always be greater than 1
        f_limit_up = df * nmax
        f_limit_lo = df * nmin
        if verbose:
            print('2nd order frequency Limits = %g,%g' %
                  (f_limit_lo, f_limit_up))

# if nargout>3,
# #compute the sum and frequency effects separately
# [svec, dvec] = disufq((amp.'),w,kw,min(h,10^30),g,nmin,nmax)
# svec = svec.'
# dvec = dvec.'
##
# x2s  = fft(svec) % 2'nd order sum frequency component
# x2d  = fft(dvec) % 2'nd order difference frequency component
##
# # 1'st order + 2'nd order component.
# x2(:,2:end) =x(:,2:end)+ real(x2s(1:np,:))+real(x2d(1:np,:))
# else
        if False:
            # TODO: disufq does not work for cases>1
            amp = np.array(amp.T).ravel()
            rvec, ivec = c_library.disufq(amp.real, amp.imag, w, kw,
                                          water_depth,
                                          g, nmin, nmax, cases, ns)
            svec = rvec + 1J * ivec
        else:
            amp = amp.T
            svec = []
            for i in range(cases):
                rvec, ivec = c_library.disufq(amp[i].real, amp[i].imag, w, kw,
                                              water_depth,
                                              g, nmin, nmax, 1, ns)
                svec.append(rvec + 1J * ivec)
            svec = np.hstack(svec)
        svec.shape = (cases, ns)
        x2o = fft(svec, axis=1).T  # 2'nd order component

        # 1'st order + 2'nd order component.
        x2[:, 1::] = x[:, 1::] + x2o[0:ns, :].real
        if output == 'timeseries':
            xx2 = mat2timeseries(x2)
            xx = mat2timeseries(x)
            return xx2, xx
        return x2, x

    def stats_nl(self, h=None, moments='sk', method='approximate', g=9.81):
        """
        Statistics of 2'nd order waves to the leading order.

        Parameters
        ----------
        h : scalar
            water depth (default self.h)
        moments : string (default='sk')
            composed of letters ['mvsk'] specifying which moments to compute:
                   'm' = mean,
                   'v' = variance,
                   's' = skewness,
                   'k' = (Pearson's) kurtosis.
        method : string
            'approximate' method due to Marthinsen & Winterstein (default)
            'eigenvalue'  method due to Kac and Siegert

        Skewness = kurtosis-3 = 0 for a Gaussian process.
        The mean, sigma, skewness and kurtosis are determined as follows:
        method == 'approximate':  due to Marthinsen and Winterstein
        mean  = 2 * int Hd(w1,w1)*S(w1) dw1
        sigma = sqrt(int S(w1) dw1)
        skew  = 6 * int int [Hs(w1,w2)+Hd(w1,w2)]*S(w1)*S(w2) dw1*dw2/m0^(3/2)
        kurt  = (4*skew/3)^2

        where Hs = sum frequency effects  and Hd = difference frequency effects

        method == 'eigenvalue'

        mean  = sum(E)
        sigma = sqrt(sum(C^2)+2*sum(E^2))
        skew  = sum((6*C^2+8*E^2).*E)/sigma^3
        kurt  = 3+48*sum((C^2+E^2).*E^2)/sigma^4

        where
            h1 = sqrt(S*dw/2)
            C  = (ctranspose(V)*[h1;h1])
        and E and V is the eigenvalues and eigenvectors, respectively, of the
        2'order transfer matrix.
        S is the spectrum and dw is the frequency spacing of S.

        Example:
        --------
        # Simulate a Transformed Gaussian process:
        >>> import wafo.spectrum.models as sm
        >>> import wafo.transform.models as wtm
        >>> Hs = 7.
        >>> Sj = sm.Jonswap(Hm0=Hs, Tp=11)
        >>> S = Sj.tospecdata()
        >>> me, va, sk, ku = S.stats_nl(moments='mvsk')
        >>> g = wtm.TrHermite(mean=me, sigma=Hs/4, skew=sk, kurt=ku,
        ...                    ysigma=Hs/4)
        >>> ys = S.sim(15000)         # Simulated in the Gaussian world
        >>> xs = g.gauss2dat(ys[:,1]) # Transformed to the real world


        See also
        ---------
        transform.TrHermite
        transform.TrOchi
        objects.LevelCrossings.trdata
        objects.TimeSeries.trdata

        References:
        -----------
        Langley, RS (1987)
        'A statistical analysis of nonlinear random waves'
        Ocean Engineering, Vol 14, No 5, pp 389-407

        Marthinsen, T. and Winterstein, S.R (1992)
        'On the skewness of random surface waves'
        In proceedings of the 2nd ISOPE Conference, San Francisco, 14-19 june.

        Winterstein, S.R, Ude, T.C. and Kleiven, G. (1994)
        'Springing and slow drift responses:
        predicted extremes and fatigue vs. simulation'
        In Proc. 7th International behaviour of Offshore structures, (BOSS)
        Vol. 3, pp.1-15
        """

        #  default options
        if h is None:
            h = self.h

        # S = ttspec(S,'w')
        w = ravel(self.args)
        S = ravel(self.data)
        if self.freqtype in ['f', 'w']:
            # vari = 't'
            if self.freqtype == 'f':
                w = 2. * pi * w
                S = S / (2. * pi)
        # m0 = self.moment(nr=0)
        m0 = simps(S, w)
        sa = sqrt(m0)
        # Nw = w.size

        Hs, Hd, Hdii = qtf(w, h, g)

        # return
        # skew=6/sqrt(m0)^3*simpson(S.w,
        #            simpson(S.w,(Hs+Hd).*S1(:,ones(1,Nw))).*S1.')

        Hspd = trapz(trapz((Hs + Hd) * S[newaxis, :], w) * S, w)
        output = []
        # %approx : Marthinsen, T. and Winterstein, S.R (1992) method
        if method[0] == 'a':
            if 'm' in moments:
                output.append(2. * trapz(Hdii * S, w))
            if 'v' in moments:
                output.append(m0)
            skew = 6. / sa ** 3 * Hspd
            if 's' in moments:
                output.append(skew)
            if 'k' in moments:
                output.append((4. * skew / 3.) ** 2. + 3.)
        else:
            raise ValueError('Unknown option!')

# elif method[0]== 'q': #, #  quasi method
# Fn = self.nyquist_freq()
# dw = Fn/Nw
# tmp1 =sqrt(S[:,newaxis]*S[newaxis,:])*dw
# Hd = Hd*tmp1
# Hs = Hs*tmp1
# k = 6
# stop = 0
# while !stop:
# E = eigs([Hd,Hs;Hs,Hd],[],k)
# %stop = (length(find(abs(E)<1e-4))>0 | k>1200)
# %stop = (any(abs(E(:))<1e-4) | k>1200)
# stop = (any(abs(E(:))<1e-4) | k>=min(2*Nw,1200))
# k = min(2*k,2*Nw)
# end
##
##
# m02=2*sum(E.^2) % variance of 2'nd order contribution
##
# %Hstd = 16*trapz(S.w,(Hdii.*S1).^2)
# %Hstd = trapz(S.w,trapz(S.w,((Hs+Hd)+ 2*Hs.*Hd).*S1(:,ones(1,Nw))).*S1.')
# ma   = 2*trapz(S.w,Hdii.*S1)
# %m02  = Hstd-ma^2% variance of second order part
# sa   = sqrt(m0+m02)
# skew = 6/sa^3*Hspd
# kurt = (4*skew/3).^2+3
# elif method[0]== 'e': #, % Kac and Siegert eigenvalue analysis
# Fn = self.nyquist_freq()
# dw = Fn/Nw
# tmp1 =sqrt(S[:,newaxis]*S[newaxis,:])*dw
# Hd = Hd*tmp1
# Hs = Hs*tmp1
# k = 6
# stop = 0
##
##
# while (not stop):
# [V,D] = eigs([Hd,HsHs,Hd],[],k)
# E = diag(D)
# %stop = (length(find(abs(E)<1e-4))>0 | k>=min(2*Nw,1200))
# stop = (any(abs(E(:))<1e-4) | k>=min(2*Nw,1200))
# k = min(2*k,2*Nw)
# end
##
##
# h1 = sqrt(S*dw/2)
# C  = (ctranspose(V)*[h1;h1])
##
# E2 = E.^2
# C2 = C.^2
##
# ma   = sum(E)                     % mean
# sa   = sqrt(sum(C2)+2*sum(E2))    % standard deviation
# skew = sum((6*C2+8*E2).*E)/sa^3   % skewness
# kurt = 3+48*sum((C2+E2).*E2)/sa^4 % kurtosis
        return output

    def testgaussian(self, ns, test0=None, cases=100, method='nonlinear',
                     verbose=False, **opt):
        '''
        TESTGAUSSIAN Test if a stochastic process is Gaussian.

         CALL:  test1 = testgaussian(S,[ns,Ns],test0,def,options)

        Returns
        -------
         test1 : array,
            simulated values of e(g)=int (g(u)-u)^2 du, where int limits is
            given by OPTIONS.PARAM.

        Parameters
        ----------
        ns : int
            # of points simulated
        test0 : real scalar
            observed value of e(g)=int (g(u)-u)^2 du,
        cases : int
            # of independent simulations (default  100)
        method : string
            defines method of estimation of the transform
            nonlinear': from smoothed crossing intensity (default)
            'mnonlinear': from smoothed marginal distribution
        options = options structure defining how the estimation of the
                    transformation is done. (default troptset('dat2tr'))

         TESTGAUSSIAN simulates  e(g(u)-u) = int (g(u)-u)^2 du  for Gaussian
         processes given the spectral density, S. The result is plotted if
         test0 is given. This is useful for testing if the process X(t) is
         Gaussian. If 95% of TEST1 is less than TEST0 then X(t) is not Gaussian
         at a 5% level.

        Example:
        -------
        >>> import wafo.spectrum.models as sm
        >>> import wafo.transform.models as wtm
        >>> import wafo.objects as wo
        >>> Hs = 7
        >>> Sj = sm.Jonswap(Hm0=Hs)
        >>> S0 = Sj.tospecdata()
        >>> ns =100; dt = .2
        >>> x1 = S0.sim(ns, dt=dt)

        >>> S = S0.copy()
        >>> me, va, sk, ku = S.stats_nl(moments='mvsk')
        >>> S.tr = wtm.TrHermite(mean=me, sigma=Hs/4, skew=sk, kurt=ku,
        ...                        ysigma=Hs/4)
        >>> ys = wo.mat2timeseries(S.sim(ns=2**13))
        >>> g0, gemp = ys.trdata()
        >>> t0 = g0.dist2gauss()
        >>> t1 = S0.testgaussian(ns=2**13, t0=t0, cases=50)
        >>> sum(t1 > t0) < 5
        True

        See also
        --------
        cov2sdat, dat2tr, troptset
        '''

        maxsize = 200000  # must divide the computations due to limited memory
#        if nargin<5||isempty(opt):
#            opt = troptset('dat2tr')
#        opt = troptset(opt,'multip',1)

        plotflag = False if test0 is None else True
        if cases > 50:
            print('  ... be patient this may take a while')

        rep = int(ns * cases / maxsize) + 1
        Nstep = int(cases / rep)

        acf = self.tocovdata()
        test1 = []
        for ix in range(rep):
            xs = acf.sim(ns=ns, cases=Nstep)
            for iy in range(1, xs.shape[-1]):
                ts = TimeSeries(xs[:, iy], xs[:, 0].ravel())
                g, _tmp = ts.trdata(method, **opt)
                test1.append(g.dist2gauss())
            if verbose:
                print('finished %d of %d ' % (ix + 1, rep))

        if rep > 1:
            xs = acf.sim(ns=ns, cases=np.remainder(cases, rep))
            for iy in range(1, xs.shape[-1]):
                ts = TimeSeries(xs[:, iy], xs[:, 0].ravel())
                g, _tmp = ts.trdata(method, **opt)
                test1.append(g.dist2gauss())

        if plotflag:
            plotbackend.plot(test1, 'o')
            plotbackend.plot([1, cases], [test0, test0], '--')

            plotbackend.ylabel('e(g(u)-u)')
            plotbackend.xlabel('Simulation number')
        return test1

    def moment(self, nr=2, even=True, j=0):
        ''' Calculates spectral moments from spectrum

        Parameters
        ----------
        nr   : int
            order of moments (recomended maximum 4)
        even : bool
            False for all moments,
            True for only even orders
        j : int
            0 or 1

        Returns
        -------
        m     : list of moments
        mtext : list of strings describing the elements of m, see below

        Details
        -------
        Calculates spectral moments of up to order NR by use of
        Simpson-integration.

                 /                                  /
        mj_t^i = | w^i S(w)^(j+1) dw,  or  mj_x^i = | k^i S(k)^(j+1) dk
                 /                                  /

        where k=w^2/gravity, i=0,1,...,NR

        The strings in output mtext have the same position in the list
        as the corresponding numerical value has in output m
        Notation in mtext: 'm0' is the variance,
                        'm0x' is the first-order moment in x,
                       'm0xx' is the second-order moment in x,
                       'm0t'  is the first-order moment in t,
                             etc.
        For the calculation of moments see Baxevani et al.

        Example:
        >>> import numpy as np
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=3, Tp=7)
        >>> w = np.linspace(0,4,256)
        >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values
        >>> S.moment()
        ([0.5616342024616453, 0.7309966918203602], ['m0', 'm0tt'])

        References
        ----------
        Baxevani A. et al. (2001)
        Velocities for Random Surfaces
        '''
        one_dim_spectra = ['freq', 'enc', 'k1d']
        if self.type not in one_dim_spectra:
            raise ValueError('Unknown spectrum type!')

        f = ravel(self.args)
        S = ravel(self.data)
        if self.freqtype in ['f', 'w']:
            vari = 't'
            if self.freqtype == 'f':
                f = 2. * pi * f
                S = S / (2. * pi)
        else:
            vari = 'x'
        S1 = abs(S) ** (j + 1.)
        m = [simps(S1, x=f)]
        mtxt = 'm%d' % j
        mtext = [mtxt]
        step = mod(even, 2) + 1
        df = f ** step
        for i in range(step, nr + 1, step):
            S1 = S1 * df
            m.append(simps(S1, x=f))
            mtext.append(mtxt + vari * i)
        return m, mtext

    def nyquist_freq(self):
        """
        Return Nyquist frequency

        Example
        -------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=5)
        >>> S = Sj.tospecdata() #Make spectrum ob
        >>> S.nyquist_freq()
        3.0
        """
        return self.args[-1]

    def sampling_period(self):
        ''' Returns sampling interval from Nyquist frequency of spectrum

        Returns
        ---------
        dT : scalar
            sampling interval, unit:
            [m] if wave number spectrum,
            [s] otherwise

        Let wm be maximum frequency/wave number in spectrum, then
            dT=pi/wm
        if angular frequency,
            dT=1/(2*wm)
        if natural frequency (Hz)

        Example
        -------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=5)
        >>> S = Sj.tospecdata() #Make spectrum ob
        >>> S.sampling_period()
        1.0471975511965976

        See also
        '''

        if self.freqtype == 'f':
            wmdt = 0.5  # Nyquist to sampling interval factor
        else:  # ftype == w og ftype == k
            wmdt = pi

        wm = self.args[-1]  # Nyquist frequency
        dt = wmdt / wm  # sampling interval = 1/Fs
        return dt

    def resample(self, dt=None, Nmin=0, Nmax=2 ** 13 + 1, method='stineman'):
        '''
        Interpolate and zero-padd spectrum to change Nyquist freq.

        Parameters
        ----------
        dt : real scalar
            wanted sampling interval (default as given by S, see spec2dt)
            unit: [s] if frequency-spectrum, [m] if wave number spectrum
        Nmin, Nmax : scalar integers
            minimum and maximum number of frequencies, respectively.
        method : string
            interpolation method (options are 'linear', 'cubic' or 'stineman')

        To be used before simulation (e.g. spec2sdat) or evaluation of
        covariance function (spec2cov) to get the wanted sampling interval.
        The input spectrum is interpolated and padded with zeros to reach
        the right max-frequency, w[-1]=pi/dt, f(end)=1/(2*dt), or k[-1]=pi/dt.
        The objective is that output frequency grid should be at least as dense
        as the input grid, have equidistant spacing and length equal to
        2^k+1 (>=Nmin). If the max frequency is changed, the number of points
        in the spectrum is maximized to 2^13+1.

        Note: Also zero-padding down to zero freq, if S does not start there.
        If empty input dt, this is the only effect.

        See also
        --------
        spec2cov, spec2sdat, covinterp, spec2dt
        '''

        ftype = self.freqtype
        w = self.args.ravel()
        n = w.size

        # doInterpolate = 0
        # Nyquist to sampling interval factor
        Cnf2dt = 0.5 if ftype == 'f' else pi  # % ftype == w og ftype == k

        wnOld = w[-1]         # Old Nyquist frequency
        dTold = Cnf2dt / wnOld  # sampling interval=1/Fs
        # dTold = self.sampling_period()

        if dt is None:
            dt = dTold

        # Find how many points that is needed
        nfft = 2 ** nextpow2(max(n - 1, Nmin - 1))
        dttest = dTold * (n - 1) / nfft

        while (dttest > dt) and (nfft < Nmax - 1):
            nfft = nfft * 2
            dttest = dTold * (n - 1) / nfft

        nfft = nfft + 1

        wnNew = Cnf2dt / dt  # % New Nyquist frequency
        dWn = wnNew - wnOld
        doInterpolate = dWn > 0 or w[1] > 0 or (
            nfft != n) or dt != dTold or any(abs(diff(w, axis=0)) > 1.0e-8)

        if doInterpolate > 0:
            S1 = self.data

            dw = min(diff(w))

            if dWn > 0:
                # add a zero just above old max-freq, and a zero at new
                # max-freq to get correct interpolation there
                Nz = 1 + (dWn > dw)  # % Number of zeros to add
                if Nz == 2:
                    w = hstack((w, wnOld + dw, wnNew))
                else:
                    w = hstack((w, wnNew))

                S1 = hstack((S1, zeros(Nz)))

            if w[0] > 0:
                # add a zero at freq 0, and, if there is space, a zero just
                # below min-freq
                Nz = 1 + (w[0] > dw)  # % Number of zeros to add
                if Nz == 2:
                    w = hstack((0, w[0] - dw, w))
                else:
                    w = hstack((0, w))

                S1 = hstack((zeros(Nz), S1))

            # Do a final check on spacing in order to check that the gridding
            # is sufficiently dense:
            # np1 = S1.size
            dwMin = finfo(float).max
            # wnc = min(wnNew,wnOld-1e-5)
            wnc = wnNew
            # specfun = lambda xi : stineman_interp(xi, w, S1)
            specfun = interpolate.interp1d(w, S1, kind='cubic')
            x, unused_y = discretize(specfun, 0, wnc)
            dwMin = minimum(min(diff(x)), dwMin)

            newNfft = 2 ** nextpow2(ceil(wnNew / dwMin)) + 1
            if newNfft > nfft:
                # if (nfft <= 2 ** 15 + 1) and (newNfft > 2 ** 15 + 1):
                #    warnings.warn('Spectrum matrix is very large (>33k). ' +
                #        'Memory problems may occur.')
                nfft = newNfft
            self.args = linspace(0, wnNew, nfft)
            if method == 'stineman':
                self.data = stineman_interp(self.args, w, S1)
            else:
                intfun = interpolate.interp1d(w, S1, kind=method)
                self.data = intfun(self.args)
            self.data = self.data.clip(0)  # clip negative values to 0

    def normalize(self, gravity=9.81):
        '''
        Normalize a spectral density such that m0=m2=1

        Paramter
        --------
        gravity=9.81

        Notes
        -----
        Normalization performed such that
            INT S(freq) dfreq = 1       INT freq^2  S(freq) dfreq = 1
        where integration limits are given by  freq  and  S(freq)  is the
        spectral density; freq can be frequency or wave number.
        The normalization is defined by
            A=sqrt(m0/m2); B=1/A/m0; freq'=freq*A; S(freq')=S(freq)*B

        If S is a directional spectrum then a normalized gravity (.g) is added
        to Sn, such that mxx normalizes to 1, as well as m0 and mtt.
        (See spec2mom for notation of moments)

        If S is complex-valued cross spectral density which has to be
        normalized, then m0, m2 (suitable spectral moments) should be given.

        Example
        -------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=5)
        >>> S = Sj.tospecdata() #Make spectrum ob
        >>> S.moment(2)
        ([1.5614600345079888, 0.95567089481941048], ['m0', 'm0tt'])
        >>> Sn = S.copy(); Sn.normalize()

        Now the moments should be one
        >>> Sn.moment(2)
        ([1.0000000000000004, 0.99999999999999967], ['m0', 'm0tt'])
        '''
        mom, unused_mtext = self.moment(nr=4, even=True)
        m0 = mom[0]
        m2 = mom[1]
        m4 = mom[2]

        SM0 = sqrt(m0)
        SM2 = sqrt(m2)
        A = SM0 / SM2
        B = SM2 / (SM0 * m0)

        if self.freqtype == 'f':
            self.args = self.args * A / 2 / pi
            self.data = self.data * B * 2 * pi
        elif self.freqtype == 'w':
            self.args = self.args * A
            self.data = self.data * B
            m02 = m4 / gravity ** 2
            m20 = m02
            self.g = gravity * sqrt(m0 * m20) / m2
        self.A = A
        self.norm = True
        self.date = now()

    def bandwidth(self, factors=0):
        '''
        Return some spectral bandwidth and irregularity factors

        Parameters
        -----------
        factors : array-like
            Input vector 'factors' correspondence:
            0 alpha=m2/sqrt(m0*m4)                        (irregularity factor)
            1 eps2 = sqrt(m0*m2/m1^2-1)                   (narrowness factor)
            2 eps4 = sqrt(1-m2^2/(m0*m4))=sqrt(1-alpha^2) (broadness factor)
            3 Qp=(2/m0^2)int_0^inf f*S(f)^2 df            (peakedness factor)

        Returns
        --------
        bw : arraylike
            vector of bandwidth factors
            Order of output is the same as order in 'factors'

        Example:
        >>> import numpy as np
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=3, Tp=7)
        >>> w = np.linspace(0,4,256)
        >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values
        >>> S.bandwidth([0,'eps2',2,3])
        array([ 0.73062845,  0.34476034,  0.68277527,  2.90817052])
        '''
        m, unused_mtxt = self.moment(nr=4, even=False)

        fact_dict = dict(alpha=0, eps2=1, eps4=3, qp=3, Qp=3)
        fact = array([fact_dict.get(idx, idx)
                      for idx in list(factors)], dtype=int)

        # fact = atleast_1d(fact)
        alpha = m[2] / sqrt(m[0] * m[4])
        eps2 = sqrt(m[0] * m[2] / m[1] ** 2. - 1.)
        eps4 = sqrt(1. - m[2] ** 2. / m[0] / m[4])
        f = self.args
        S = self.data
        Qp = 2 / m[0] ** 2. * simps(f * S ** 2, x=f)
        bw = array([alpha, eps2, eps4, Qp])
        return bw[fact]

    def characteristic(self, fact='Hm0', T=1200, g=9.81):
        """
        Returns spectral characteristics and their covariance

        Parameters
        ----------
        fact : vector with factor integers or a string or a list of strings
            defining spectral characteristic, see description below.
        T  : scalar
            recording time (sec) (default 1200 sec = 20 min)
        g : scalar
            acceleration of gravity [m/s^2]

        Returns
        -------
        ch : vector
            of spectral characteristics
        R  : matrix
            of the corresponding covariances given T
        chtext : a list of strings
            describing the elements of ch, see example.


        Description
        ------------
        If input spectrum is of wave number type, output are factors for
        corresponding 'k1D', else output are factors for 'freq'.
        Input vector 'factors' correspondence:
        1 Hm0   = 4*sqrt(m0)                         Significant wave height
        2 Tm01  = 2*pi*m0/m1                         Mean wave period
        3 Tm02  = 2*pi*sqrt(m0/m2)                   Mean zero-crossing period
        4 Tm24  = 2*pi*sqrt(m2/m4)                   Mean period between maxima
        5 Tm_10 = 2*pi*m_1/m0                        Energy period
        6 Tp    = 2*pi/{w | max(S(w))}               Peak period
        7 Ss    = 2*pi*Hm0/(g*Tm02^2)                Significant wave steepness
        8 Sp    = 2*pi*Hm0/(g*Tp^2)                  Average wave steepness
        9 Ka    = abs(int S(w)*exp(i*w*Tm02) dw ) /m0  Groupiness parameter
        10 Rs    = (S(0.092)+S(0.12)+S(0.15)/(3*max(S(w)))
                                                     Quality control parameter
        11 Tp1   = 2*pi*int S(w)^4 dw                Peak Period
                  ------------------                 (robust estimate for Tp)
                  int w*S(w)^4 dw

        12 alpha = m2/sqrt(m0*m4)                          Irregularity factor
        13 eps2  = sqrt(m0*m2/m1^2-1)                      Narrowness factor
        14 eps4  = sqrt(1-m2^2/(m0*m4))=sqrt(1-alpha^2)    Broadness factor
        15 Qp    = (2/m0^2)int_0^inf w*S(w)^2 dw           Peakedness factor

        Order of output is same as order in 'factors'
        The covariances are computed with a Taylor expansion technique
        and is currently only available for factors 1, 2, and 3. Variances
        are also available for factors 4,5,7,12,13,14 and 15

        Quality control:
        ----------------
        Critical value for quality control parameter Rs is Rscrit = 0.02
        for surface displacement records and Rscrit=0.0001 for records of
        surface acceleration or slope. If Rs > Rscrit then probably there
        are something wrong with the lower frequency part of S.

        Ss may be used as an indicator of major malfunction, by checking that
        it is in the range of 1/20 to 1/16 which is the usual range for
        locally generated wind seas.

        Examples:
        ---------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=5)
        >>> S = Sj.tospecdata() #Make spectrum ob
        >>> S.characteristic(1)
        (array([ 8.59007646]), array([[ 0.03040216]]), ['Tm01'])

        >>> [ch, R, txt] = S.characteristic([1,2,3])  # fact vector of integers
        >>> S.characteristic('Ss')               # fact a string
        (array([ 0.04963112]), array([[  2.63624782e-06]]), ['Ss'])

        >>> S.characteristic(['Hm0','Tm02'])   # fact a list of strings
        (array([ 4.99833578,  8.03139757]), array([[ 0.05292989,  0.02511371],
               [ 0.02511371,  0.0274645 ]]), ['Hm0', 'Tm02'])

        See also
        ---------
        bandwidth,
        moment

        References
        ----------
        Krogstad, H.E., Wolf, J., Thompson, S.P., and Wyatt, L.R. (1999)
        'Methods for intercomparison of wave measurements'
        Coastal Enginering, Vol. 37, pp. 235--257

        Krogstad, H.E. (1982)
        'On the covariance of the periodogram'
        Journal of time series analysis, Vol. 3, No. 3, pp. 195--207

        Tucker, M.J. (1993)
        'Recommended standard for wave data sampling and near-real-time
        processing'
        Ocean Engineering, Vol.20, No.5, pp. 459--474

        Young, I.R. (1999)
        "Wind generated ocean waves"
        Elsevier Ocean Engineering Book Series, Vol. 2, pp 239
        """

        # TODO: Need more checking on computing the variances for Tm24,alpha,
        #       eps2 and eps4
        # TODO: Covariances between Tm24,alpha, eps2 and eps4 variables are
        #        also needed

        tfact = dict(Hm0=0, Tm01=1, Tm02=2, Tm24=3, Tm_10=4, Tp=5, Ss=6, Sp=7,
                     Ka=8, Rs=9, Tp1=10, Alpha=11, Eps2=12, Eps4=13, Qp=14)
        tfact1 = ('Hm0', 'Tm01', 'Tm02', 'Tm24', 'Tm_10', 'Tp', 'Ss', 'Sp',
                  'Ka', 'Rs', 'Tp1', 'Alpha', 'Eps2', 'Eps4', 'Qp')

        if isinstance(fact, str):
            fact = list((fact,))
        if isinstance(fact, (list, tuple)):
            nfact = []
            for k in fact:
                if isinstance(k, str):
                    nfact.append(tfact.get(k.capitalize(), 15))
                else:
                    nfact.append(k)
        else:
            nfact = fact

        nfact = atleast_1d(nfact)

        if any((nfact > 14) | (nfact < 0)):
            raise ValueError('Factor outside range (0,...,14)')

        # vari = self.freqtype
        f = self.args.ravel()
        S1 = self.data.ravel()
        m, unused_mtxt = self.moment(nr=4, even=False)

        # moments corresponding to freq  in Hz
        for k in range(1, 5):
            m[k] = m[k] / (2 * pi) ** k

        # pi = np.pi
        ind = flatnonzero(f > 0)
        m.append(simps(S1[ind] / f[ind], f[ind]) * 2. * pi)  # = m_1
        m_10 = simps(S1[ind] ** 2 / f[ind], f[ind]) * \
            (2 * pi) ** 2 / T  # = COV(m_1,m0|T=t0)
        m_11 = simps(S1[ind] ** 2. / f[ind] ** 2, f[ind]) * \
            (2 * pi) ** 3 / T  # = COV(m_1,m_1|T=t0)

        # sqrt = np.sqrt
        #     Hm0        Tm01        Tm02             Tm24         Tm_10
        Hm0 = 4. * sqrt(m[0])
        Tm01 = m[0] / m[1]
        Tm02 = sqrt(m[0] / m[2])
        Tm24 = sqrt(m[2] / m[4])
        Tm_10 = m[5] / m[0]

        Tm12 = m[1] / m[2]

        ind = S1.argmax()
        maxS = S1[ind]
        # [maxS ind] = max(S1)
        Tp = 2. * pi / f[ind]  # peak period /length
        Ss = 2. * pi * Hm0 / g / Tm02 ** 2  # Significant wave steepness
        Sp = 2. * pi * Hm0 / g / Tp ** 2  # Average wave steepness
        # groupiness factor
        Ka = abs(simps(S1 * exp(1J * f * Tm02), f)) / m[0]

        # Quality control parameter
        # critical value is approximately 0.02 for surface displacement records
        # If Rs>0.02 then there are something wrong with the lower frequency
        # part of S.
        Rs = np.sum(
            interp(r_[0.0146, 0.0195, 0.0244] * 2 * pi, f, S1)) / 3. / maxS
        Tp2 = 2 * pi * simps(S1 ** 4, f) / simps(f * S1 ** 4, f)

        alpha1 = Tm24 / Tm02  # m(3)/sqrt(m(1)*m(5))
        eps2 = sqrt(Tm01 / Tm12 - 1.)  # sqrt(m(1)*m(3)/m(2)^2-1)
        eps4 = sqrt(1. - alpha1 ** 2)  # sqrt(1-m(3)^2/m(1)/m(5))
        Qp = 2. / m[0] ** 2 * simps(f * S1 ** 2, f)

        ch = r_[Hm0, Tm01, Tm02, Tm24, Tm_10, Tp, Ss,
                Sp, Ka, Rs, Tp2, alpha1, eps2, eps4, Qp]

        # Select the appropriate values
        ch = ch[nfact]
        chtxt = [tfact1[i] for i in nfact]

        # if nargout>1,
        # covariance between the moments:
        # COV(mi,mj |T=t0) = int f^(i+j)*S(f)^2 df/T
        mij, unused_mijtxt = self.moment(nr=8, even=False, j=1)
        for ix, tmp in enumerate(mij):
            mij[ix] = tmp / T / ((2. * pi) ** (ix - 1.0))

        #  and the corresponding variances for
        # {'hm0', 'tm01', 'tm02', 'tm24', 'tm_10','tp','ss', 'sp', 'ka', 'rs',
        #  'tp1','alpha','eps2','eps4','qp'}
        R = r_[4 * mij[0] / m[0],
               mij[0] / m[1] ** 2. - 2. * m[0] * mij[1] /
               m[1] ** 3. + m[0] ** 2. * mij[2] / m[1] ** 4.,
               0.25 * (mij[0] / (m[0] * m[2]) - 2. * mij[2] / m[2] ** 2 +
                       m[0] * mij[4] / m[2] ** 3),
               0.25 * (mij[4] / (m[2] * m[4]) - 2 * mij[6] / m[4] ** 2 +
                       m[2] * mij[8] / m[4] ** 3),
               m_11 / m[0] ** 2 + (m[5] / m[0] ** 2) ** 2 *
               mij[0] - 2 * m[5] / m[0] ** 3 * m_10,
               nan, (8 * pi / g) ** 2 *
               (m[2] ** 2 / (4 * m[0] ** 3) *
                mij[0] + mij[4] / m[0] - m[2] / m[0] ** 2 * mij[2]),
               nan * ones(4),
               m[2] ** 2 * mij[0] / (4 * m[0] ** 3 * m[4]) + mij[4] /
               (m[0] * m[4]) + mij[8] * m[2] ** 2 / (4 * m[0] * m[4] ** 3) -
               m[2] * mij[2] / (m[0] ** 2 * m[4]) + m[2] ** 2 * mij[4] /
               (2 * m[0] ** 2 * m[4] ** 2) - m[2] * mij[6] / m[0] / m[4] ** 2,
               (m[2] ** 2 * mij[0] / 4 + (m[0] * m[2] / m[1]) ** 2 * mij[2] +
                m[0] ** 2 * mij[4] / 4 - m[2] ** 2 * m[0] * mij[1] / m[1] +
                m[0] * m[2] * mij[2] / 2 - m[0] ** 2 * m[2] / m[1] * mij[3]) /
               eps2 ** 2 / m[1] ** 4,
               (m[2] ** 2 * mij[0] / (4 * m[0] ** 2) + mij[4] + m[2] ** 2 *
                mij[8] / (4 * m[4] ** 2) - m[2] * mij[2] / m[0] + m[2] ** 2 *
                mij[4] / (2 * m[0] * m[4]) - m[2] * mij[6] / m[4]) *
               m[2] ** 2 / (m[0] * m[4] * eps4) ** 2,
               nan]

        # and covariances by a taylor expansion technique:
        # Cov(Hm0,Tm01) Cov(Hm0,Tm02) Cov(Tm01,Tm02)
        S0 = r_[2. / (sqrt(m[0]) * m[1]) * (mij[0] - m[0] * mij[1] / m[1]),
                1. / sqrt(m[2]) * (mij[0] / m[0] - mij[2] / m[2]),
                1. / (2 * m[1]) * sqrt(m[0] / m[2]) * (mij[0] / m[0] - mij[2] /
                m[2] - mij[1] / m[1] + m[0] * mij[3] / (m[1] * m[2]))]

        R1 = ones((15, 15))
        R1[:, :] = nan
        for ix, Ri in enumerate(R):
            R1[ix, ix] = Ri

        R1[0, 2:4] = S0[:2]
        R1[1, 2] = S0[2]
        # make lower triangular equal to upper triangular part
        for ix in [0, 1]:
            R1[ix + 1:, ix] = R1[ix, ix + 1:]

        R = R[nfact]
        R1 = R1[nfact, :][:, nfact]

        # Needs further checking:
        # Var(Tm24)= 0.25*(mij[4]/(m[2]*m[4])-
        #                    2*mij[6]/m[4]**2+m[2]*mij[8]/m[4]**3)
        return ch, R1, chtxt

    def setlabels(self):
        ''' Set automatic title, x-,y- and z- labels on SPECDATA object

            based on type, angletype, freqtype
        '''

        N = len(self.type)
        if N == 0:
            raise ValueError(
                'Object does not appear to be initialized, it is empty!')

        labels = ['', '', '']
        if self.type.endswith('dir'):
            title = 'Directional Spectrum'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[2] = r'S($\omega$,$\theta$) $[m^2 s / rad^2]$'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[2] = r'S(f,$\theta$) $[m^2 s / rad]$'

            if self.angletype.startswith('r'):
                labels[1] = 'Wave directions [rad]'
            elif self.angletype.startswith('d'):
                labels[1] = 'Wave directions [deg]'
        elif self.type.endswith('freq'):
            title = 'Spectral density'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[1] = r'S($\omega$) $[m^2 s/ rad]$'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[1] = r'S(f) $[m^2 s]$'
        else:
            title = 'Wave Number Spectrum'
            labels[0] = 'Wave number [rad/m]'
            if self.type.endswith('k1d'):
                labels[1] = r'S(k) $[m^3/ rad]$'
            elif self.type.endswith('k2d'):
                labels[1] = labels[0]
                labels[2] = r'S(k1,k2) $[m^4/ rad^2]$'
            else:
                raise ValueError(
                    'Object does not appear to be initialized, it is empty!')
        if self.norm != 0:
            title = 'Normalized ' + title
            labels[0] = 'Normalized ' + labels[0].split('[')[0]
            if not self.type.endswith('dir'):
                labels[1] = labels[1].split('[')[0]
            labels[2] = labels[2].split('[')[0]

        self.labels.title = title
        self.labels.xlab = labels[0]
        self.labels.ylab = labels[1]
        self.labels.zlab = labels[2]


class SpecData2D(PlotData):

    """ Container class for 2D spectrum data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D, list of vectors for 2D, 3D, ...

    type : string
        spectrum type (default 'freq')
    freqtype : letter
        frequency type (default 'w')
    angletype : string
        angle type of directional spectrum (default 'radians')

    Examples
    --------
    >>> import numpy as np
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=3, Tp=7)
    >>> w = np.linspace(0,4,256)
    >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values

    See also
    --------
    PlotData
    CovData
    """

    def __init__(self, *args, **kwds):

        super(SpecData2D, self).__init__(*args, **kwds)

        self.name = 'WAFO Spectrum Object'
        self.type = 'dir'
        self.freqtype = 'w'
        self.angletype = ''
        self.h = inf
        self.tr = None
        self.phi = 0.
        self.v = 0.
        self.norm = 0
        somekeys = ['angletype', 'phi', 'name', 'h',
                    'tr', 'freqtype', 'v', 'type', 'norm']

        self.__dict__.update(sub_dict_select(kwds, somekeys))

        if self.type.endswith('dir') and self.angletype == '':
            self.angletype = 'radians'

        self.setlabels()

    def toacf(self):
        pass

    def tospecdata(self, type=None):  # @ReservedAssignment
        pass

    def sim(self):
        pass

    def sim_nl(self):
        pass

    def rotate(self, phi=0, rotateGrid=False, method='linear'):
        '''
        Rotate spectrum clockwise around the origin.

        Parameters
        ----------
        phi : real scalar
            rotation angle (default 0)
        rotateGrid : bool
            True if rotate grid of Snew physically (thus Snew.phi=0).
            False if rotate so that only Snew.phi is changed
                        (the grid is not physically rotated)  (default)
        method : string
            interpolation method to use when ROTATEGRID==1, (default 'linear')

        Rotates the spectrum clockwise around the origin.
        This equals a anti-clockwise rotation of the cordinate system (x,y).
        The spectrum can be of any of the two-dimensional types.
        For spectrum in polar representation:
            newtheta = theta-phi, but circulant such that -pi<newtheta<pi
        For spectrum in Cartesian representation:
            If the grid is rotated physically, the size of it is preserved
            (maybe it must be increased such that no nonzero points are
           affected, but this is not implemented yet: i.e. corners are cut off)
        The spectrum is assumed to be zero outside original grid.
        NB! The routine does not change the type of spectrum, use spec2spec
            for this.

        Example
        -------
          S=demospec('dir');
          plotspec(S), hold on
          plotspec(rotspec(S,pi/2),'r'), hold off

        See also
        --------
        spec2spec
        '''
        # TODO: Make physical grid rotation of cartesian coordinates more
        # robust.

        # Snew=S;

        self.phi = mod(self.phi + phi + pi, 2 * pi) - pi
        stype = self.type.lower()[-3::]
        if stype == 'dir':
            # any of the directinal types
            # Make sure theta is from -pi to pi
            theta = self.args[0]
            phi0 = theta[0] + pi
            self.args[0] = theta - phi0

            # make sure -pi<phi<pi
            self.phi = mod(self.phi + phi0 + pi, 2 * pi) - pi
            if (rotateGrid and (self.phi != 0)):
                # Do a physical rotation of spectrum
                # theta = Snew.args[0]
                ntOld = len(theta)
                if (mod(theta[0] - theta[-1], 2 * pi) == 0):
                    nt = ntOld - 1
                else:
                    nt = ntOld

                theta[0:nt] = mod(theta[0:nt] - self.phi + pi, 2 * pi) - pi
                self.phi = 0
                ind = theta.argsort()
                self.data = self.data[ind, :]
                self.args[0] = theta[ind]
                if (nt < ntOld):
                    if (self.args[0][0] == -pi):
                        self.data[ntOld, :] = self.data[0, :]
                    else:
                        # ftype = self.freqtype
                        freq = self.args[1]
                        theta = linspace(-pi, pi, ntOld)
                        # [F, T] = meshgrid(freq, theta)

                        dtheta = self.theta[1] - self.theta[0]
                        self.theta[nt] = self.theta[nt - 1] + dtheta
                        self.data[nt, :] = self.data[0, :]
                        self.data = interp2d(freq,
                                             np.vstack([self.theta[0] - dtheta,
                                                        self.theta]),
                                             np.vstack([self.data[nt, :],
                                                        self.data]),
                                             kind=method)(freq, theta)
                        self.args[0] = theta

        elif stype == 'k2d':
            # any of the 2D wave number types
            # Snew.phi   = mod(Snew.phi+phi+pi,2*pi)-pi
            if (rotateGrid and (self.phi != 0)):
                # Do a physical rotation of spectrum

                [k, k2] = meshgrid(*self.args)
                [th, r] = cart2polar(k, k2)
                [k, k2] = polar2cart(th + self.phi, r)
                ki1, ki2 = self.args
                Sn = interp2(ki1, ki2, self.data, k, k2, method)
                self.data = np.where(np.isnan(Sn), 0, Sn)
                self.phi = 0
        else:
            raise ValueError('Can only rotate two dimensional spectra')
        return

    def moment(self, nr=2, vari='xt'):
        '''
        Calculates spectral moments from spectrum

        Parameters
        ----------
        nr   : int
            order of moments (maximum 4)
        vari : string
            variables in model, optional when two-dim.spectrum,
                   string with 'x' and/or 'y' and/or 't'
        Returns
        -------
        m     : list of moments
        mtext : list of strings describing the elements of m, see below

        Details
        -------
        Calculates spectral moments of up to order four by use of
        Simpson-integration.

           //
        m_jkl=|| k1^j*k2^k*w^l S(w,th) dw dth
           //

        where k1=w^2/gravity*cos(th-phi),  k2=w^2/gravity*sin(th-phi)
        and phi is the angle of the rotation in S.phi. If the spectrum
        has field .g, gravity is replaced by S.g.

        The strings in output mtext have the same position in the cell array
        as the corresponding numerical value has in output m
        Notation in mtext: 'm0' is the variance,
                        'mx' is the first-order moment in x,
                       'mxx' is the second-order moment in x,
                       'mxt' is the second-order cross moment between x and t,
                     'myyyy' is the fourth-order moment in y
                             etc.
        For the calculation of moments see Baxevani et al.

        Example:
        >>> import wafo.spectrum.models as sm
        >>> D = sm.Spreading()
        >>> SD = D.tospecdata2d(sm.Jonswap().tospecdata(),nt=101)
        >>> m,mtext = SD.moment(nr=2,vari='xyt')
        >>> np.round(m,3),mtext
        (array([ 3.061,  0.132, -0.   ,  2.13 ,  0.011,  0.008,  1.677, -0.,
                0.109,  0.109]),
                ['m0', 'mx', 'my', 'mt', 'mxx', 'myy', 'mtt', 'mxy', 'mxt',
                'myt'])

        References
        ----------
        Baxevani A. et al. (2001)
        Velocities for Random Surfaces
        '''

        two_dim_spectra = ['dir', 'encdir', 'k2d']
        if self.type not in two_dim_spectra:
            raise ValueError('Unknown 2D spectrum type!')

        if vari is None and nr <= 1:
            vari = 'x'
        elif vari is None:
            vari = 'xt'
        else:  # secure the mutual order ('xyt')
            vari = ''.join(sorted(vari.lower()))
            Nv = len(vari)

            if vari[0] == 't' and Nv > 1:
                vari = vari[1::] + vari[0]

        Nv = len(vari)

        if not self.type.endswith('dir'):
            S1 = self.tospecdata(self.type[:-2] + 'dir')
        else:
            S1 = self
        w = ravel(S1.args[0])
        theta = S1.args[1] - S1.phi
        S = S1.data
        Sw = simps(S, x=theta, axis=0)
        m = [simps(Sw, x=w)]
        mtext = ['m0']

        if nr > 0:
            vec = []
            g = np.atleast_1d(S1.__dict__.get('g', gravity()))
            # maybe different normalization in x and y => diff. g
            kx = w ** 2 / g[0]
            ky = w ** 2 / g[-1]

            # nw = w.size

            if 'x' in vari:
                ct = np.cos(theta[:, None])
                Sc = simps(S * ct, x=theta, axis=0)
                vec.append(kx * Sc)
                mtext.append('mx')
            if 'y' in vari:
                st = np.sin(theta[:, None])
                Ss = simps(S * st, x=theta, axis=0)
                vec.append(ky * Ss)
                mtext.append('my')
            if 't' in vari:
                vec.append(w * Sw)
                mtext.append('mt')

            if nr > 1:
                if 'x' in vari:
                    Sc2 = simps(S * ct ** 2, x=theta, axis=0)
                    vec.append(kx ** 2 * Sc2)
                    mtext.append('mxx')
                if 'y' in vari:
                    Ss2 = simps(S * st ** 2, x=theta, axis=0)
                    vec.append(ky ** 2 * Ss2)
                    mtext.append('myy')
                if 't' in vari:
                    vec.append(w ** 2 * Sw)
                    mtext.append('mtt')
                if 'x' in vari and 'y' in vari:
                    Scs = simps(S * ct * st, x=theta, axis=0)
                    vec.append(kx * ky * Scs)
                    mtext.append('mxy')
                if 'x' in vari and 't' in vari:
                    vec.append(kx * w * Sc)
                    mtext.append('mxt')
                if 'y' in vari and 't' in vari:
                    vec.append(ky * w * Sc)
                    mtext.append('myt')

                if nr > 3:
                    if 'x' in vari:
                        Sc3 = simps(S * ct ** 3, x=theta, axis=0)
                        Sc4 = simps(S * ct ** 4, x=theta, axis=0)
                        vec.append(kx ** 4 * Sc4)
                        mtext.append('mxxxx')
                    if 'y' in vari:
                        Ss3 = simps(S * st ** 3, x=theta, axis=0)
                        Ss4 = simps(S * st ** 4, x=theta, axis=0)
                        vec.append(ky ** 4 * Ss4)
                        mtext.append('myyyy')
                    if 't' in vari:
                        vec.append(w ** 4 * Sw)
                        mtext.append('mtttt')

                    if 'x' in vari and 'y' in vari:
                        Sc2s = simps(S * ct ** 2 * st, x=theta, axis=0)
                        Sc3s = simps(S * ct ** 3 * st, x=theta, axis=0)
                        Scs2 = simps(S * ct * st ** 2, x=theta, axis=0)
                        Scs3 = simps(S * ct * st ** 3, x=theta, axis=0)
                        Sc2s2 = simps(S * ct ** 2 * st ** 2, x=theta, axis=0)
                        vec.extend((kx ** 3 * ky * Sc3s,
                                    kx ** 2 * ky ** 2 * Sc2s2,
                                    kx * ky ** 3 * Scs3))
                        mtext.extend(('mxxxy', 'mxxyy', 'mxyyy'))
                    if 'x' in vari and 't' in vari:
                        vec.extend((kx ** 3 * w * Sc3,
                                    kx ** 2 * w ** 2 * Sc2, kx * w ** 3 * Sc))
                        mtext.extend(('mxxxt', 'mxxtt', 'mxttt'))
                    if 'y' in vari and 't' in vari:
                        vec.extend((ky ** 3 * w * Ss3, ky ** 2 * w ** 2 * Ss2,
                                    ky * w ** 3 * Ss))
                        mtext.extend(('myyyt', 'myytt', 'myttt'))
                    if 'x' in vari and 'y' in vari and 't' in vari:
                        vec.extend((kx ** 2 * ky * w * Sc2s,
                                    kx * ky ** 2 * w * Scs2,
                                    kx * ky * w ** 2 * Scs))
                        mtext.extend(('mxxyt', 'mxyyt', 'mxytt'))
            # end % if nr>1
            m.extend([simps(vals, x=w) for vals in vec])
        return np.asarray(m), mtext

    def interp(self):
        pass

    def normalize(self):
        pass

    def bandwidth(self):
        pass

    def setlabels(self):
        ''' Set automatic title, x-,y- and z- labels on SPECDATA object

            based on type, angletype, freqtype
        '''

        N = len(self.type)
        if N == 0:
            raise ValueError(
                'Object does not appear to be initialized, it is empty!')

        labels = ['', '', '']
        if self.type.endswith('dir'):
            title = 'Directional Spectrum'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[2] = r'$S(w,\theta) [m**2 s / rad**2]$'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[2] = r'$S(f,\theta) [m**2 s / rad]$'

            if self.angletype.startswith('r'):
                labels[1] = 'Wave directions [rad]'
            elif self.angletype.startswith('d'):
                labels[1] = 'Wave directions [deg]'
        elif self.type.endswith('freq'):
            title = 'Spectral density'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[1] = 'S(w) [m**2 s/ rad]'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[1] = 'S(f) [m**2 s]'
        else:
            title = 'Wave Number Spectrum'
            labels[0] = 'Wave number [rad/m]'
            if self.type.endswith('k1d'):
                labels[1] = 'S(k) [m**3/ rad]'
            elif self.type.endswith('k2d'):
                labels[1] = labels[0]
                labels[2] = 'S(k1,k2) [m**4/ rad**2]'
            else:
                raise ValueError(
                    'Object does not appear to be initialized, it is empty!')
        if self.norm != 0:
            title = 'Normalized ' + title
            labels[0] = 'Normalized ' + labels[0].split('[')[0]
            if not self.type.endswith('dir'):
                labels[1] = labels[1].split('[')[0]
            labels[2] = labels[2].split('[')[0]

        self.labels.title = title
        self.labels.xlab = labels[0]
        self.labels.ylab = labels[1]
        self.labels.zlab = labels[2]


def main():
    import matplotlib
    matplotlib.interactive(True)
    from wafo.spectrum import models as sm

    Sj = sm.Jonswap()
    S = Sj.tospecdata()

    R = S.tocovdata(nr=1)

    Si = R.tospecdata()
    ns = 5000
    dt = .2
    x1 = S.sim_nl(ns=ns, dt=dt)
    x2 = TimeSeries(x1[:, 1], x1[:, 0])
    R = x2.tocovdata(lag=100)
    R.plot()

    S.plot('ro')
    t = S.moment()
    t1 = S.bandwidth([0, 1, 2, 3])
    S1 = S.copy()
    S1.resample(dt=0.3, method='cubic')
    S1.plot('k+')
    x = S1.sim(ns=100)
    import pylab
    pylab.clf()
    pylab.plot(x[:, 0], x[:, 1])
    pylab.show()

    pylab.close('all')
    print('done')


def test_mm_pdf():

    import wafo.spectrum.models as sm
    Sj = sm.Jonswap(Hm0=7, Tp=11)
    w = np.linspace(0, 4, 256)
    S1 = Sj.tospecdata(w)  # Make spectrum object from numerical values
    S = sm.SpecData1D(Sj(w), w)  # Alternatively do it manually
    S0 = S.to_linspec()
    mm = S.to_mm_pdf()


def test_docstrings():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    #test_docstrings()
    test_mm_pdf()
        # main()
