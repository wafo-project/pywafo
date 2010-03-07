from __future__ import division
import warnings
import numpy as np
from numpy import (pi, inf, meshgrid, zeros, ones, where, nonzero, #@UnresolvedImport
           flatnonzero, ceil, sqrt, exp, log, arctan2, #@UnresolvedImport
           tanh, cosh, sinh, random, atleast_1d, maximum, #@UnresolvedImport
           minimum, diff, isnan, any, r_, conj, mod, #@UnresolvedImport
           hstack, vstack, interp, ravel, finfo, linspace, #@UnresolvedImport
           arange, array, nan, newaxis, fliplr, sign) #@UnresolvedImport
from numpy.fft import fft
from scipy.integrate import simps, trapz
from scipy.special import erf
from scipy.linalg import toeplitz
import scipy.interpolate as interpolate
from pylab import stineman_interp

from dispersion_relation import w2k #, k2w
from wafo.wafodata import WafoData, now
from wafo.misc import sub_dict_select, nextpow2, discretize, JITImport
try:
    from wafo.gaussian import Rind
except ImportError:
    
    Rind = None
try:
    from wafo import c_library
except ImportError:
    warnings.warn('Compile the c_libraray.pyd again!')
    c_library = None
    
from wafo.transform import TrData
from wafo.plotbackend import plotbackend


# Trick to avoid error due to circular import
_WAFOCOV = JITImport('wafo.covariance')


__all__ = ['SpecData1D', 'SpecData2D']

def _set_seed(iseed):
    '''Set seed of random generator'''
    if iseed != None:
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

    if h == inf: # go here for faster calculations
        h_s = 0.25 * (abs(k_1) + abs(k_2))
        h_d = -0.25 * abs(abs(k_1) - abs(k_2))
        h_dii = zeros(num_w)
        return h_s, h_d , h_dii

    [w_1, w_2] = meshgrid(w, w)



    w12 = (w_1 * w_2)
    w1p2 = (w_1 + w_2)
    w1m2 = (w_1 - w_2)
    k12 = (k_1 * k_2)
    k1p2 = (k_1 + k_2)
    k1m2 = abs(k_1 - k_2)
    
    if 0: # Langley
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

    else: #  # Marthinsen & Winterstein
        tmp1 = 0.5 * g * k12 / w12
        tmp2 = 0.25 / g * (w_1 ** 2. + w_2 ** 2. + w12)
        h_s = (tmp1 - tmp2 + 0.25 * g * (w_1 * k_2 ** 2. + w_2 * k_1 ** 2) / 
                (w12 * (w1p2))) / (1. - g * (k1p2) / (w1p2) ** 2. * 
                                   tanh((k1p2) * h)) + tmp2 - 0.5 * tmp1 ## OK

        tmp2 = 0.25 / g * (w_1 ** 2 + w_2 ** 2 - w12) # #OK
        h_d = (tmp1 - tmp2 - 0.25 * g * (w_1 * k_2 ** 2 - w_2 * k_1 ** 2) / 
            (w12 * (w1m2))) / (1. - g * (k1m2) / (w1m2) ** 2. * 
                               tanh((k1m2) * h)) + tmp2 - 0.5 * tmp1 # # OK


    ##tmp1 = 0.5*g*k_w./(w.*sqrt(g*h))
    ##tmp2 = 0.25*w.^2/g

# Wave group velocity
    c_g = 0.5 * g * (tanh(k_w * h) + k_w * h * (1.0 - tanh(k_w * h) ** 2)) / w 
    h_dii = (0.5 * (0.5 * g * (k_w / w) ** 2. - 0.5 * w ** 2 / g + 
                    g * k_w / (w * c_g))
            / (1. - g * h / c_g ** 2.) - 0.5 * k_w / sinh(2 * k_w * h))# # OK
    h_d.flat[0::num_w + 1] = h_dii

    ##k    = find(w_1==w_2)
    ##h_d(k) = h_dii

    #% The NaN's occur due to division by zero. => Set the isnans to zero
    
    h_dii = where(isnan(h_dii), 0, h_dii)
    h_d = where(isnan(h_d), 0, h_d)
    h_s = where(isnan(h_s), 0, h_s)

    return h_s, h_d , h_dii

class SpecData1D(WafoData):
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
    >>> S = SpecData1D(Sj(w),w) # Alternatively do it manually

    See also
    --------
    WafoData
    CovData
    """

    def __init__(self, *args, **kwds):
        super(SpecData1D, self).__init__(*args, **kwds)
        self.name = 'WAFO Spectrum Object'
        self.type = 'freq'
        self.freqtype = 'w'
        self.angletype = ''
        self.h = inf
        self.tr = None
        self.phi = 0.0
        self.v = 0.0
        self.norm = False
        somekeys = ['angletype', 'phi', 'name', 'h', 'tr', 'freqtype', 'v',
                    'type', 'norm']

        self.__dict__.update(sub_dict_select(kwds, somekeys))

        self.setlabels()

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
        >>> acfmat[:2,:]
        array([[ 3.06075987,  0.        , -1.67750289,  0.        ],
               [ 3.05246132, -0.16662376, -1.66819445,  0.18634189]])

        See also
        --------
        cov,
        resample,
        objects
        '''

        ftype = self.freqtype # %options are 'f' and 'w' and 'k'
        freq = self.args
        n_f = len(freq)
        dt_old = self.sampling_period()
        if dt is None:
            dt = dt_old
            rate = 1
        else:
            rate = max(round(dt_old * 1. / dt), 1.)


        if nt is None:
            nt = rate * (n_f - 1)
        else: #%check if Nt is ok
            nt = minimum(nt, rate * (n_f - 1))


        checkdt = 1.2 * min(diff(freq)) / 2. / pi
        if ftype in 'k':
            lagtype = 'x'
        else:
            lagtype = 't'
            if ftype in 'f':
                checkdt = checkdt * 2 * pi
        msg1 = 'Step dt = %g in computation of the density is too small.' % dt
        msg2 = 'Step dt = %g is small, and may cause numerical inaccuracies.' % dt

        if (checkdt < 2. ** -16 / dt):
            print(msg1)
            print('The computed covariance (by FFT(2^K)) may differ from the')
            print('theoretical. Solution:')
            raise ValueError('use larger dt or sparser grid for spectrum.')


        # Calculating covariances
        #~~~~~~~~~~~~~~~~~~~~~~~~
        spec = self.copy()
        spec.resample(dt)

        acf = spec.tocovdata(nr, nt, rate=1)
        acfmat = zeros((nt + 1, nr + 1), dtype=float)
        acfmat[:, 0] = acf.data[0:nt + 1]
        fieldname = 'R' + lagtype * nr
        for i in range(1, nr + 1):
            fname = fieldname[:i + 1]
            r_i = getattr(acf, fname)
            acfmat[:, i] = r_i[0:nt + 1]

        eps0 = 0.0001
        if nt + 1 >= 5:
            cc2 = acfmat[0, 0] - acfmat[4, 0] * (acfmat[4, 0] / acfmat[0, 0])
            if (cc2 < eps0):
                warnings.warn(msg1)
        cc1 = acfmat[0, 0] - acfmat[1, 0] * (acfmat[1, 0] / acfmat[0, 0])
        if (cc1 < eps0):
            warnings.warn(msg2)
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

        The input 'rate' gives together with the spectrum
        the time-grid-spacing: dt=pi/(S.w[-1]*rate), S.w[-1] is the Nyquist freq.
        This results in the time-grid: 0:dt:Nt*dt.

        What output is achieved with different S and choices of Nt,Nx and Ny:
        1) S.type='freq' or 'dir', Nt set, Nx,Ny not set: then result R(time) (one-dim)
        2) S.type='k1d' or 'k2d', Nt set, Nx,Ny not set: then result R(x) (one-dim)
        3) Any type, Nt and Nx set =>R(x,time); Nt and Ny set =>R(y,time)
        4) Any type, Nt, Nx and Ny set => R(x,y,time)
        5) Any type, Nt not set, Nx and/or Ny set => Nt set to default, goto 3) or 4)

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
        d_w = abs(diff(freq, n_f=2, axis=0))
        if any(d_w > 1.0e-8):
            txt = '''Not equidistant frequencies/wave numbers in spectrum. 
            Correct it with resample, for example.'''
            raise ValueError(txt)


        if rate is None:
            rate = 1 # %interpolation rate
        elif rate > 16:
            rate = 16
        else: # make sure rate is a power of 2
            rate = 2 ** nextpow2(rate)

        if nt is None:
            nt = rate * (n_f - 1)
        else: #check if Nt is ok
            nt = minimum(nt, rate * (n_f - 1))

        spec = self.copy()

        if self.freqtype in 'k':
            lagtype = 'x'
        else:
            lagtype = 'time'

        d_t = spec.sampling_period()
        #normalize spec so that sum(specn)/(n_f-1)=acf(0)=var(X)
        specn = spec.data * freq[-1]
        if spec.freqtype in 'f':
            w = freq * 2 * pi
        else:
            w = freq

        nfft = rate * 2 ** nextpow2(2 * n_f - 2)

        # periodogram
        rper = r_[specn, zeros(nfft - (2 * n_f) + 2), conj(specn[n_f - 1:0:-1])] 
        time = r_[0:nt + 1] * d_t * (2 * n_f - 2) / nfft

        r = fft(rper, nfft).real / (2 * n_f - 2)
        acf = _WAFOCOV.CovData1D(r[0:nt + 1], time, lagtype=lagtype)
        acf.tr = spec.tr
        acf.h = spec.h
        acf.norm = spec.norm

        if nr > 0:
            w = r_[w , zeros(nfft - 2 * n_f + 2) , -w[n_f - 1:0:-1] ]
            fieldname = 'R' + lagtype * nr 
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
        the perturbation series in the simulation, is to truncate the upper tail
        of the spectrum at FNLIMIT in the calculation of the 2nd order wave 
        components, i.e., in the calculation of sum and difference frequency effects. 
        
        Example:
        --------
        np = 10000;
          iseed = 1;
          pflag = 2;
          S  = jonswap(10);
          fnLimit = inf;  
          [SL,SN] = spec2linspec(S,np,[],[],fnLimit);
          x0 = spec2nlsdat(SL,8*np,[],iseed,[],fnLimit);
          x1 = spec2nlsdat(S,8*np,[],iseed,[],fnLimit); 
          x2 = spec2nlsdat(S,8*np,[],iseed,[],sqrt(2));  
          Se0 = dat2spec(x0);
          Se1 = dat2spec(x1);
          Se2 = dat2spec(x2); 
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
        trace = 1 #% trace the convergence
        max_sim = 30
        tolerance = 5e-4
        
        L = 200 #%maximum lag size of the window function used in
                        #%spectral estimate
        #ftype = self.freqtype #options are 'f' and 'w' and 'k'
#        switch ftype
#         case 'f', 
#          ftype = 'w';
#          S = ttspec(S,ftype);
#        end
        Hm0 = self.characteristic('Hm0')
        Tm02 = self.characteristic('Tm02')
        
       
        if not iseed is None: 
            _set_seed(iseed) #% set the the seed
             
        n = len(self.data)
        if ns is None:
            ns = max(n - 1, 5000)
        if dt is None:
            S = self.interp(dt) # interpolate spectrum  
        else:
            S = self.copy()
                                      
        ns = ns + mod(ns, 2) # make sure np is even    
        
        water_depth = abs(self.h);
        kbar = w2k(2 * pi / Tm02, 0, water_depth)[0]
        
        # Expected maximum amplitude for 1000 waves seastate
        num_waves = 10000  
        Amax = sqrt(2 * log(num_waves)) * Hm0 / 4 
          
        fLimitLo = sqrt(gravity * tanh(kbar * water_depth) * Amax / water_depth ** 3);
        
        
        freq = S.args
        eps = finfo(float).eps
        freq[-1] = freq[-1] - sqrt(eps)
        Hw2 = 0
        
        SL = S
        
        indZero = nonzero(freq < fLimitLo)[0]
        if len(indZero):
            SL.data[indZero] = 0
        
        maxS = max(S.data);
        #Fs = 2*freq(end)+eps; % sampling frequency
        
        for ix in xrange(max_sim):
            [x2, x1] = spec2nlsdat(SL, [np, cases], [], iseed, method, fnLimit)
            #%x2(:,2:end) = x2(:,2:end) -x1(:,2:end);
            S2 = dat2spec(x2, L)   
            S1 = dat2spec(x1, L)
            #%[tf21,fi] = tfe(x2(:,2),x1(:,2),1024,Fs,[],512);
            #%Hw11 = interp1q(fi,tf21.*conj(tf21),freq);
            if True:
                Hw1 = exp(interp1q(S2.args, log(abs(S1.data / S2.data)), freq))  
            else:
                # Geometric mean
                Hw1 = exp((interp1q(S2.args, log(abs(S1.data / S2.data)), freq) + log(Hw2)) / 2)  
                #end
            #Hw1  = (interp1q( S2.w,abs(S1.S./S2.S),freq)+Hw2)/2;
            #plot(freq, abs(Hw11-Hw1),'g')
            #title('diff')
            #pause
            #clf
          
            #d1 = interp1q( S2.w,S2.S,freq);;
          
            SL.data = (Hw1 * S.data)
          
            if len(indZero):
                SL.data[indZero] = 0
                #end
            k = nonzero(SL.data < 0)[0]
            if len(k): # Make sure that the current guess is larger than zero
                #%k
                #Hw1(k)
                Hw1[k] = min(S1.data[k] * 0.9, S.data[k])
                SL.data[k] = max(Hw1[k] * S.data[k], eps)
                #end
            Hw12 = Hw1 - Hw2
            maxHw12 = max(abs(Hw12))
            if trace == 1:
                plotbackend.figure(1),
                plotbackend.semilogy(freq, Hw1, 'r')
                plotbackend.title('Hw')
                plotbackend.figure(2),
                plotbackend.semilogy(freq, abs(Hw12), 'r')
                plotbackend.title('Hw-HwOld')
            
                #pause(3)
                plotbackend.figure(1),
                plotbackend.semilogy(freq, Hw1, 'b')
                plotbackend.title('Hw')
                plotbackend.figure(2),
                plotbackend.semilogy(freq, abs(Hw12), 'b')
                plotbackend.title('Hw-HwOld')
                #figtile
            #end
           
            print('Iteration : %d, Hw12 : %g  Hw12/maxS : %g' % (ix, maxHw12, (maxHw12 / maxS)))
            if (maxHw12 < maxS * tolerance) and (Hw1[-1] < Hw2[-1]) :
                break
            #end
            Hw2 = Hw1
        #end
        
        #%Hw1(end)
        #%maxS*1e-3
        #%if Hw1(end)*S.>maxS*1e-3,
        #%  warning('The Nyquist frequency of the spectrum may be too low')
        #%end
        
        SL.date = now() #datestr(now)
        #if nargout>1
        SN = SL.copy()
        SN.data = S.data - SL.data
        SN.note = SN.note + ' non-linear component (spec2linspec)'
        #end
        SL.note = SL.note + ' linear component (spec2linspec)'
        
        return SL, SN


    def to_t_pdf(self, u=None, pdef='Tc', paramt=None, **options):
        '''
        Density of crest/trough- period or length, version 2. 
       
        Parameters
        ----------
        u : real scalar
            reference level (default the most frequently crossed level).
        pdef : string, 'Tc', Tt', 'Lc' or 'Lt'
            'Tc',    gives half wave period, Tc (default).
            'Tt',    gives half wave period, Tt
            'Lc' and 'Lt' ditto for wave length.
        paramt : [t0, tn, nt] 
            where t0, tn and nt is the first value, last value and the number
            of points, respectively, for which the density will be computed. 
            paramt= [5, 5, 51] implies that the density is computed only for 
            T=5 and using 51 equidistant points in the interval [0,5].
        options : optional parameters
            controlling the performance of the integration. See Rind for details.

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
        if pdef[0] in ('l', 'L'):
            if self.type != 'k1d':
                raise ValueError('Must be spectrum of type: k1d')
        elif pdef[0] in ('t', 'T'):
            if self.type != 'freq':
                raise ValueError('Must be spectrum of type: freq')
        else:
            raise ValueError('pdef must be Tc,Tt or Lc, Lt')
#        if strncmpi('l',def,1)
#          spec=spec2spec(spec,'k1d')
#        elseif strncmpi('t',def,1)
#          spec=spec2spec(spec,'freq')
#        else
#          error('Unknown def')
#        end
        pdef2defnr = dict(tc=1, lc=1, tt= -1, lt= -1)
        defnr = pdef2defnr[pdef.lower()]
         
        S = self.copy()
        S.normalize()
        m, unused_mtxt = self.moment(nr=2, even=True)
        A = sqrt(m[0] / m[1])
        
       
        if self.tr is None:
            y = linspace(-5, 5, 513)
            #g = _wafotransform.
            g = TrData(y, sqrt(m[0]) * y)
        else:
            g = self.tr
        
        
        if u is None:
            u = g.gauss2dat(0) #% most frequently crossed level 
        
        # transform reference level into Gaussian level
        un = g.dat2gauss(u)

        #disp(['The level u for Gaussian process = ', num2str(u)])
        
        if paramt is None:
            #% z2 = u^2/2
            z = -sign(defnr) * un / sqrt(2)
            expectedMaxPeriod = 2 * ceil(2 * pi * A * exp(z) * (0.5 + erf(z) / 2)) 
            paramt = [0, expectedMaxPeriod, 51]
        
        t0 = paramt[0]
        tn = paramt[1]
        Ntime = paramt[2]
        t = linspace(0, tn / A, Ntime) #normalized times
        Nstart = max(round(t0 / tn * (Ntime - 1)), 1)  #% index to starting point to
                                             #% evaluate
                                    
        dt = t[1] - t[0]
        nr = 2
        R = S.tocov_matrix(nr, Ntime - 1, dt)
        #R  = spec2cov2(S,nr,Ntime-1,dt)
        
                        
        xc = vstack((un, un))
        indI = -ones(4, dtype=int)
        Nd = 2
        Nc = 2
        XdInf = 100.e0 * sqrt(-R[0, 2])
        XtInf = 100.e0 * sqrt(R[0, 0])
        
        B_up = hstack([un + XtInf, XdInf, 0])
        B_lo = hstack([un, 0, -XdInf])
        #%INFIN = [1 1 0]
        #BIG   = zeros((Ntime+2,Ntime+2))
        ex = zeros(Ntime + 2, dtype=float)
        #%CC    = 2*pi*sqrt(-R(1,1)/R(1,3))*exp(un^2/(2*R(1,1)))
        #%  XcScale = log(CC)
        opts['xcscale'] = log(2 * pi * sqrt(-R[0, 0] / R[0, 2])) + (un ** 2 / (2 * R[0, 0]))
        
        f = zeros(Ntime, dtype=float)
        err = zeros(Ntime, dtype=float)
        
        rind = Rind(**opts)
        #h11 = fwaitbar(0,[],sprintf('Please wait ...(start at: %s)',datestr(now)))
        for pt in xrange(Nstart, Ntime):
            Nt = pt - Nd + 1
            Ntd = Nt + Nd
            Ntdc = Ntd + Nc
            indI[1] = Nt - 1
            indI[2] = Nt
            indI[3] = Ntd - 1
            
            #% positive wave period  
            BIG = self._covinput(pt, R) 
          
            tmp = rind(BIG, ex[:Ntdc], B_lo, B_up, indI, xc, Nt)
            f[pt], err[pt] = tmp[:2]
            #fwaitbar(pt/Ntime,h11,sprintf('%s Ready: %d of %d',datestr(now),pt,Ntime))
        #end
        #close(h11)
        
        
        titledict = dict(tc='Density of Tc', tt='Density of Tt', lc='Density of Lc', lt='Density of Lt')
        Htxt = titledict.get(pdef.lower())
        
        if pdef[0].lower() == 'l':
            xtxt = 'wave length [m]'
        else:
            xtxt = 'period [s]'
        
        Htxt = '%s_{v =%2.5g}' % (Htxt, u)
        pdf = WafoData(f / A, t * A, title=Htxt, xlab=xtxt)
        pdf.err = err / A
        pdf.u = u
        pdf.options = opts
        return pdf


    def _covinput(self, pt, R):
        """
        Return covariance matrix for Tc or Tt period problems
    
        Parameters
        ----------
        pt : scalar integer
            time
        R : array-like, shape Ntime x 3
            [R0,R1,R2] column vectors with autocovariance and its derivatives, 
            i.e., Ri (i=1:2) are vectors with the 1'st and 2'nd derivatives of R0.  
      
        The order of the variables in the covariance matrix are organized as follows: 
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
        Scd = array([[0, R[pt, 1]], [ -R[pt, 1], 0]])
        
        if pt > 1 :
            #%cov(Xt)
            Stt = toeplitz(R[:pt - 1, 0]) # Cov(X(tn),X(ts))  = r(ts-tn)   = r(|ts-tn|)
            #%cov(Xc,Xt) 
            Sct = R[1:pt, 0]        # Cov(X(tn),X(ts))  = r(ts-tn)   = r(|ts-tn|)
            Sct = vstack((Sct, Sct[::-1]))
            #%Cov(Xd,Xt)
            Sdt = -R[1:pt, 1]         # Cov(X'(t1),X(ts)) = -r'(ts-t1) = r(|s-t|)
            Sdt = vstack((Sdt, -Sdt[::-1]))
            #N   = pt + 3
            big = vstack((hstack((Stt, Sdt.T, Sct.T)),
                             hstack((Sdt, Sdd, Scd.T)),
                             hstack((Sct, Scd, Scc))))
        else:
            #N = 4
            big = vstack((hstack((Sdd, Scd.T)),
                                    hstack((Scd, Scc))))
        return big

    def to_specnorm(self):
        S = self.copy()
        S.normalize()
        return S

    def sim(self, ns=None, cases=1, dt=None, iseed=None, method='random', derivative=False):
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
        applied to the simulated data, the result is a simulation of a transformed
        Gaussian process.

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
        ...     assert(np.abs(m-trueval)<sa)

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

            #nr=0,Nt=None,dt=None
            acf = spec.tocovdata(nr=0)
            T = Nt * d_t
            i = flatnonzero(acf.args > T)

            # Trick to avoid adding high frequency noise to the spectrum
            if i.size > 0:
                acf.data[i[0]::] = 0.0

            return acf.sim(ns=ns, cases=cases, iseed=iseed, derivative=derivative)

        _set_seed(iseed)

        ns = ns + mod(ns, 2) # make sure it is even

        f_i = freq[1:-1]
        s_i = spec.data[1:-1]
        if ftype in ('w', 'k'):
            fact = 2. * pi
            s_i = s_i * fact
            f_i = f_i / fact

        x = zeros((ns, cases + 1))

        d_f = 1 / (ns * d_t)


        # interpolate for freq.  [1:(N/2)-1]*d_f and create 2-sided, uncentered spectra
        f = arange(1, ns / 2.) * d_f

        f_u = hstack((0., f_i, d_f * ns / 2.))
        s_u = hstack((0., abs(s_i) / 2., 0.))


        s_i = interp(f, f_u, s_u)
        s_u = hstack((0., s_i, 0, s_i[(ns / 2) - 2::-1]))
        del(s_i, f_u)

        # Generate standard normal random numbers for the simulations
        randn = random.randn
        z_r = randn((ns / 2) + 1, cases)
        z_i = vstack((zeros((1, cases)), randn((ns / 2) - 1, cases), zeros((1, cases))))

        amp = zeros((ns, cases), dtype=complex)
        amp[0:(ns / 2 + 1), :] = z_r - 1j *z_i
        del(z_r, z_i)
        amp[(ns / 2 + 1):ns, :] = amp[ns / 2 - 1:0:-1, :].conj()
        amp[0, :] = amp[0, :]*sqrt(2.)
        amp[(ns / 2), :] = amp[(ns / 2), :]*sqrt(2.)


        # Make simulated time series
        T = (ns - 1) * d_t
        Ssqr = sqrt(s_u * d_f / 2.)

        # stochastic amplitude
        amp = amp * Ssqr[:, newaxis]


        # Deterministic amplitude
        #amp = sqrt[1]*Ssqr(:,ones(1,cases)).*exp(sqrt(-1)*atan2(imag(amp),real(amp)))
        del(s_u, Ssqr)


        x[:, 1::] = fft(amp, axis=0).real
        x[:, 0] = linspace(0, T, ns) #' %(0:d_t:(np-1)*d_t).'


        if derivative:
            xder = zeros(ns, cases + 1)
            w = 2. * pi * hstack((0, f, 0., -f[-1::-1]))
            amp = -1j * amp * w[:, newaxis]
            xder[:, 1:(cases + 1)] = fft(amp, axis=0).real
            xder[:, 0] = x[:, 0]

        if spec.tr is not None:
            print('   Transforming data.')
            g = spec.tr
            G = fliplr(g) #% the invers of g
            if derivative:
                for i in range(cases):
                    tmp = tranproc(hstack((x[:, i + 1], xder[:, i + 1])), G)
                    x[:, i + 1] = tmp[:, 0]
                    xder[:, i + 1] = tmp[:, 1]

            else:
                for i in range(cases):
                    x[:, i + 1] = tranproc(x[:, i + 1], G)

        if derivative:
            return x, xder
        else:
            return x

# function [x2,x,svec,dvec,amp]=spec2nlsdat(spec,np,dt,iseed,method,truncationLimit)
    def sim_nl(self, ns=None, cases=1, dt=None, iseed=None, method='random',
        fnlimit=1.4142, reltol=1e-3, g=9.81):
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
        the elimination of second order effects from the spectrum, i.e., extract
        the linear components from the spectrum. One way to do this is to use
        SPEC2LINSPEC.

        Example
        --------
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

        ns = ns + mod(ns, 2) # make sure it is even

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

        # interpolate for freq.  [1:(N/2)-1]*df and create 2-sided, uncentered spectra
        f = arange(1, ns / 2.) * df
        f_u = hstack((0., f_i, df * ns / 2.))
        w = 2. * pi * hstack((0., f, df * ns / 2.))
        kw = w2k(w , 0., water_depth, g)[0]
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
        amp[0, :] = amp[0, :]*sqrt(2.)
        amp[(ns / 2), :] = amp[(ns / 2), :]*sqrt(2.)


        # Make simulated time series

        T = (ns - 1) * d_t
        Ssqr = sqrt(s_u * df / 2.)


        if method.startswith('apd') : # apdeterministic
            # Deterministic amplitude and phase
            amp[1:(ns / 2), :] = amp[1, 0]
            amp[(ns / 2 + 1):ns, :] = amp[1, 0].conj()
            amp = sqrt(2) * Ssqr[:, newaxis] * exp(1J * arctan2(amp.imag, amp.real))
        elif method.startswith('ade'): # adeterministic
            # Deterministic amplitude and random phase
            amp = sqrt(2) * Ssqr[:, newaxis] * exp(1J * arctan2(amp.imag, amp.real))
        else:
            # stochastic amplitude
            amp = amp * Ssqr[:, newaxis]
        # Deterministic amplitude
        #amp = sqrt(2)*Ssqr(:,ones(1,cases)).*exp(sqrt(-1)*atan2(imag(amp),real(amp)))
        del(s_u, Ssqr)


        x[:, 1::] = fft(amp, axis=0).real
        x[:, 0] = linspace(0, T, ns) #' %(0:d_t:(np-1)*d_t).'



        x2 = x.copy()

        # If the spectrum does not decay rapidly enough towards zero, the
        # contribution from the wave components at the  upper tail can be very
        # large and unphysical.
        # To ensure convergence of the perturbation series, the upper tail of 
        # the spectrum is truncated in the calculation of sum and difference
        # frequency effects.
        # Find the critical wave frequency to ensure convergence.

        num_waves = 1000. # Typical number of waves in 3 hour seastate
        kbar = w2k(2. * pi / Tm02, 0., water_depth)[0]
        amp_max = sqrt(2 * log(num_waves)) * Hm0 / 4 #% Expected maximum amplitude for 1000 waves seastate

        f_limit_up = fnlimit * sqrt(g * tanh(kbar * water_depth) / amp_max) / (2 * pi)
        f_limit_lo = sqrt(g * tanh(kbar * water_depth) * amp_max / water_depth) / (2 * pi * water_depth)

        nmax = min(flatnonzero(f <= f_limit_up).max(), nmax) + 1
        nmin = max(flatnonzero(f_limit_lo <= f).min(), nmin) + 1

        #if isempty(nmax),nmax = np/2end
        #if isempty(nmin),nmin = 2end % Must always be greater than 1
        f_limit_up = df * nmax
        f_limit_lo = df * nmin

        print('2nd order frequency Limits = %g,%g' % (f_limit_lo, f_limit_up))



##        if nargout>3,
##        %compute the sum and frequency effects separately
##        [svec, dvec] = disufq((amp.'),w,kw,min(h,10^30),g,nmin,nmax)
##        svec = svec.'
##        dvec = dvec.'
##
##        x2s  = fft(svec) % 2'nd order sum frequency component
##        x2d  = fft(dvec) % 2'nd order difference frequency component
##
##        % 1'st order + 2'nd order component.
##        x2(:,2:end) =x(:,2:end)+ real(x2s(1:np,:))+real(x2d(1:np,:))
##        else
        amp = amp.T
        rvec, ivec = c_library.disufq(amp.real, amp.imag, w, kw, water_depth, g, nmin, nmax, cases, ns)

        svec = rvec + 1J * ivec
        svec.shape = (cases, ns)
        x2o = fft(svec, axis=1).T # 2'nd order component


        # 1'st order + 2'nd order component.
        x2[:, 1::] = x[:, 1::] + x2o[0:ns, :].real

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
                   's' = (Fisher's) skew,
                   'k' = (Fisher's) kurtosis.
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
        and E and V is the eigenvalues and eigenvectors, respectively, of the 2'order
        transfer matrix. S is the spectrum and dw is the frequency spacing of S.

        Example:
        --------
        #Simulate a Transformed Gaussian process:
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap()
        >>> S = Sj.tospecdata()
        >>> me, va, sk, ku = S.stats_nl(moments='mvsk')


        Hm0=7;Tp=11
        S = jonswap([],[Hm0 Tp]); [sk, ku, me]=spec2skew(S)
        g=hermitetr([],[Hm0/4 sk ku me]);  g2=[g(:,1), g(:,2)*Hm0/4]
        ys = spec2sdat(S,15000)   % Simulated in the Gaussian world
        xs = gaus2dat(ys,g2)      % Transformed to the real world

        See also
        ---------
        hermitetr, ochitr, lc2tr, dat2tr

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

        #% default options
        if h is None:
            h = self.h

        #S = ttspec(S,'w')
        w = ravel(self.args)
        S = ravel(self.data)
        if self.freqtype in ['f', 'w']:
            vari = 't'
            if self.freqtype == 'f':
                w = 2. * pi * w
                S = S / (2. * pi)
        #m0 = self.moment(nr=0)
        m0 = simps(S, w)
        sa = sqrt(m0)
        Nw = w.size

        Hs, Hd, Hdii = qtf(w, h, g)

        #%return
        #%skew=6/sqrt(m0)^3*simpson(S.w,simpson(S.w,(Hs+Hd).*S1(:,ones(1,Nw))).*S1.')

        Hspd = trapz(trapz((Hs + Hd) * S[newaxis, :], w) * S, w)
        output = []
        if method[0] == 'a': # %approx : Marthinsen, T. and Winterstein, S.R (1992) method
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

##        elif method[0]== 'q': #, #% quasi method
##            Fn = self.nyquist_freq()
##            dw = Fn/Nw
##            tmp1 =sqrt(S[:,newaxis]*S[newaxis,:])*dw
##            Hd = Hd*tmp1
##            Hs = Hs*tmp1
##            k = 6
##            stop = 0
##            while !stop:
##                E = eigs([Hd,Hs;Hs,Hd],[],k)
##                %stop = (length(find(abs(E)<1e-4))>0 | k>1200)
##                %stop = (any(abs(E(:))<1e-4) | k>1200)
##                stop = (any(abs(E(:))<1e-4) | k>=min(2*Nw,1200))
##                k = min(2*k,2*Nw)
##            #end
##
##
##            m02=2*sum(E.^2) % variance of 2'nd order contribution
##
##            %Hstd = 16*trapz(S.w,(Hdii.*S1).^2)
##            %Hstd = trapz(S.w,trapz(S.w,((Hs+Hd)+ 2*Hs.*Hd).*S1(:,ones(1,Nw))).*S1.')
##            ma   = 2*trapz(S.w,Hdii.*S1)
##            %m02  = Hstd-ma^2% variance of second order part
##            sa   = sqrt(m0+m02)
##            skew = 6/sa^3*Hspd
##            kurt = (4*skew/3).^2+3
##        elif method[0]== 'e': #, % Kac and Siegert eigenvalue analysis
##            Fn = self.nyquist_freq()
##            dw = Fn/Nw
##            tmp1 =sqrt(S[:,newaxis]*S[newaxis,:])*dw
##            Hd = Hd*tmp1
##            Hs = Hs*tmp1
##            k = 6
##            stop = 0
##
##
##            while (not stop):
##              [V,D] = eigs([Hd,HsHs,Hd],[],k)
##              E = diag(D)
##              %stop = (length(find(abs(E)<1e-4))>0 | k>=min(2*Nw,1200))
##              stop = (any(abs(E(:))<1e-4) | k>=min(2*Nw,1200))
##              k = min(2*k,2*Nw)
##            #end
##
##
##            h1 = sqrt(S*dw/2)
##            C  = (ctranspose(V)*[h1;h1])
##
##            E2 = E.^2
##            C2 = C.^2
##
##            ma   = sum(E)                     % mean
##            sa   = sqrt(sum(C2)+2*sum(E2))    % standard deviation
##            skew = sum((6*C2+8*E2).*E)/sa^3   % skewness
##            kurt = 3+48*sum((C2+E2).*E2)/sa^4 % kurtosis
        return output

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
        >>> Sj = sm.Jonswap(Hm0=3)
        >>> w = np.linspace(0,4,256)
        >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values
        >>> S.moment()
        ([0.56220770033914191, 0.35433180985851975], ['m0', 'm0tt'])

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

        Let wm be maximum frequency/wave number in spectrum,
        then dT=pi/wm if angular frequency, dT=1/(2*wm) if natural frequency (Hz)

        Example
        -------
        S = jonswap
        dt = spec2dt(S)

        See also
        '''

        if self.freqtype in 'f':
            wmdt = 0.5  # Nyquist to sampling interval factor
        else: # ftype == w og ftype == k
            wmdt = pi

        wm = self.args[-1] #Nyquist frequency
        dt = wmdt / wm #sampling interval = 1/Fs
        return dt

    def resample(self, dt=None, Nmin=0, Nmax=2 ** 13 + 1, method='stineman'):
        ''' 
        Interpolate and zero-padd spectrum to change Nyquist freq.

        Parameters
        ----------
        dt : scalar
            wanted sampling interval (default as given by S, see spec2dt)
            unit: [s] if frequency-spectrum, [m] if wave number spectrum
        Nmin : scalar
            minimum number of frequencies.
        Nmax : scalar
            minimum number of frequencies
        method : string
            interpolation method (options are 'linear', 'cubic' or 'stineman')

        To be used before simulation (e.g. spec2sdat) or evaluation of covariance
        function (spec2cov) to directly get wanted sampling interval.
        The input spectrum is interpolated and padded with zeros to reach
        the right max-frequency, w(end)=pi/dt, f(end)=1/(2*dt), or k(end)=pi/dt.
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

        #%doInterpolate = 0

        if ftype == 'f':
            Cnf2dt = 0.5 # Nyquist to sampling interval factor
        else: #% ftype == w og ftype == k
            Cnf2dt = pi

        wnOld = w[-1]         # Old Nyquist frequency
        dTold = Cnf2dt / wnOld # sampling interval=1/Fs


        if dt is None:
            dt = dTold

        # Find how many points that is needed
        nfft = 2 ** nextpow2(max(n - 1, Nmin - 1))
        dttest = dTold * (n - 1) / nfft

        while (dttest > dt) and (nfft < Nmax - 1):
            nfft = nfft * 2
            dttest = dTold * (n - 1) / nfft

        nfft = nfft + 1

        wnNew = Cnf2dt / dt #% New Nyquist frequency
        dWn = wnNew - wnOld
        doInterpolate = dWn > 0 or w[1] > 0 or (nfft != n) or dt != dTold or any(abs(diff(w, axis=0)) > 1.0e-8)

        if doInterpolate > 0:
            S1 = self.data

            dw = min(diff(w))

            if dWn > 0:
                #% add a zero just above old max-freq, and a zero at new max-freq
                #% to get correct interpolation there
                Nz = 1 + (dWn > dw) # % Number of zeros to add
                if Nz == 2:
                    w = hstack((w, wnOld + dw, wnNew))
                else:
                    w = hstack((w, wnNew))

                S1 = hstack((S1, zeros(Nz)))

            if w[0] > 0:
                #% add a zero at freq 0, and, if there is space, a zero just below min-freq
                Nz = 1 + (w[0] > dw) #% Number of zeros to add
                if Nz == 2:
                    w = hstack((0, w[0] - dw, w))
                else:
                    w = hstack((0, w))

                S1 = hstack((zeros(Nz), S1))


            #% Do a final check on spacing in order to check that the gridding is
            #% sufficiently dense:
            #np1 = S1.size
            dwMin = finfo(float).max
            #%wnc = min(wnNew,wnOld-1e-5)
            wnc = wnNew
            specfun = lambda xi : stineman_interp(xi, w, S1)

            x, unused_y = discretize(specfun, 0, wnc)
            dwMin = minimum(min(diff(x)), dwMin)

            newNfft = 2 ** nextpow2(ceil(wnNew / dwMin)) + 1
            if newNfft > nfft:
                if (nfft <= 2 ** 15 + 1) and (newNfft > 2 ** 15 + 1):
                    warnings.warn('Spectrum matrix is very large (>33k). Memory problems may occur.')

                nfft = newNfft
            self.args = linspace(0, wnNew, nfft)
            if method == 'stineman':
                self.data = stineman_interp(self.args, w, S1)
            else:
                intfun = interpolate.interp1d(w, S1, kind=method)
                self.data = intfun(self.args)
            self.data = self.data.clip(0) # clip negative values to 0

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
    
        Example:
        ------- 
        S = jonswap
        [Sn,mn4] = specnorm(S)
        mts = spec2mom(S,2)     % Should be equal to one!
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
        elif self.freqtype == 'w' :
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
        >>> Sj = sm.Jonswap(Hm0=3)
        >>> w = np.linspace(0,4,256)
        >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values
        >>> S.bandwidth([0,1,2,3])
        array([ 0.65354446,  0.3975428 ,  0.75688813,  2.00207912])
        '''

#        if self.freqtype in 'k':
#            vari = 'k'
#        else:
#            vari = 'w'

        m, unused_mtxt = self.moment(nr=4, even=False)
        
        fact = atleast_1d(factors)
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
        1 Hm0   = 4*sqrt(m0)                              Significant wave height
        2 Tm01  = 2*pi*m0/m1                              Mean wave period
        3 Tm02  = 2*pi*sqrt(m0/m2)                        Mean zero-crossing period
        4 Tm24  = 2*pi*sqrt(m2/m4)                        Mean period between maxima
        5 Tm_10 = 2*pi*m_1/m0                             Energy period
        6 Tp    = 2*pi/{w | max(S(w))}                    Peak period
        7 Ss    = 2*pi*Hm0/(g*Tm02^2)                     Significant wave steepness
        8 Sp    = 2*pi*Hm0/(g*Tp^2)                       Average wave steepness
        9 Ka    = abs(int S(w)*exp(i*w*Tm02) dw ) /m0     Groupiness parameter
        10 Rs    = (S(0.092)+S(0.12)+S(0.15)/(3*max(S(w))) Quality control parameter
        11 Tp1   = 2*pi*int S(w)^4 dw                      Peak Period (robust estimate for Tp)
                  ------------------
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
        >>> import numpy as np
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=5)
        >>> S = Sj.tospecdata() #Make spectrum ob
        >>> S.characteristic(1)
        (array([ 8.59007646]), array([[ 0.03040216]]), ['Tm01'])

        >>> [ch, R, txt] = S.characteristic([1,2,3])  # fact a vector of integers
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
        'Recommended standard for wave data sampling and near-real-time processing'
        Ocean Engineering, Vol.20, No.5, pp. 459--474

        Young, I.R. (1999)
        "Wind generated ocean waves"
        Elsevier Ocean Engineering Book Series, Vol. 2, pp 239
        """

        #% TODO % Need more checking on computing the variances for Tm24,alpha, eps2 and eps4
        #% TODO % Covariances between Tm24,alpha, eps2 and eps4 variables are also needed
        
        tfact = dict(Hm0=0, Tm01=1, Tm02=2, Tm24=3, Tm_10=4, Tp=5, Ss=6, Sp=7, Ka=8,
              Rs=9, Tp1=10, Alpha=11, Eps2=12, Eps4=13, Qp=14)
        tfact1 = ('Hm0', 'Tm01', 'Tm02', 'Tm24', 'Tm_10', 'Tp', 'Ss', 'Sp', 'Ka',
              'Rs', 'Tp1', 'Alpha', 'Eps2', 'Eps4', 'Qp')

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

        #vari = self.freqtype

        f = self.args.ravel()
        S1 = self.data.ravel()
        m, unused_mtxt = self.moment(nr=4, even=False)

        #% moments corresponding to freq  in Hz
        for k in range(1, 5):
            m[k] = m[k] / (2 * pi) ** k

        #pi = np.pi
        ind = flatnonzero(f > 0)
        m.append(simps(S1[ind] / f[ind], f[ind]) * 2. * pi) #  % = m_1
        m_10 = simps(S1[ind] ** 2 / f[ind], f[ind]) * (2 * pi) ** 2 / T #    % = COV(m_1,m0|T=t0)
        m_11 = simps(S1[ind] ** 2. / f[ind] ** 2, f[ind]) * (2 * pi) ** 3 / T  #% = COV(m_1,m_1|T=t0)

        #sqrt = np.sqrt
        #%      Hm0        Tm01        Tm02             Tm24         Tm_10
        Hm0 = 4. * sqrt(m[0])
        Tm01 = m[0] / m[1]
        Tm02 = sqrt(m[0] / m[2])
        Tm24 = sqrt(m[2] / m[4])
        Tm_10 = m[5] / m[0]

        Tm12 = m[1] / m[2]

        ind = S1.argmax()
        maxS = S1[ind]
        #[maxS ind] = max(S1)
        Tp = 2. * pi / f[ind] #                                   % peak period /length
        Ss = 2. * pi * Hm0 / g / Tm02 ** 2 #                             % Significant wave steepness
        Sp = 2. * pi * Hm0 / g / Tp ** 2 #                               % Average wave steepness
        Ka = abs(simps(S1 * exp(1J * f * Tm02), f)) / m[0] #% groupiness factor

        #% Quality control parameter
        #% critical value is approximately 0.02 for surface displacement records
        #% If Rs>0.02 then there are something wrong with the lower frequency part
        #% of S.
        Rs = np.sum(interp(r_[0.0146, 0.0195, 0.0244] * 2 * pi, f, S1)) / 3. / maxS
        Tp2 = 2 * pi * simps(S1 ** 4, f) / simps(f * S1 ** 4, f)


        alpha1 = Tm24 / Tm02 #                 % m(3)/sqrt(m(1)*m(5))
        eps2 = sqrt(Tm01 / Tm12 - 1.)#         % sqrt(m(1)*m(3)/m(2)^2-1)
        eps4 = sqrt(1. - alpha1 ** 2) #          % sqrt(1-m(3)^2/m(1)/m(5))
        Qp = 2. / m[0] ** 2 * simps(f * S1 ** 2, f)

        ch = r_[Hm0, Tm01, Tm02, Tm24, Tm_10, Tp, Ss, Sp, Ka, Rs, Tp2, alpha1, eps2, eps4, Qp]

        #% Select the appropriate values
        ch = ch[nfact]
        chtxt = [tfact1[i] for i in nfact]

        #if nargout>1,
        #% covariance between the moments:
        #%COV(mi,mj |T=t0) = int f^(i+j)*S(f)^2 df/T
        mij, unused_mijtxt = self.moment(nr=8, even=False, j=1)
        for ix, tmp in enumerate(mij):
            mij[ix] = tmp / T / ((2. * pi) ** (ix - 1.0))


        #% and the corresponding variances for
        #%{'hm0', 'tm01', 'tm02', 'tm24', 'tm_10','tp','ss', 'sp', 'ka', 'rs', 'tp1','alpha','eps2','eps4','qp'}
        R = r_[4 * mij[0] / m[0],
               mij[0] / m[1] ** 2. - 2. * m[0] * mij[1] / m[1] ** 3. + m[0] ** 2. * mij[2] / m[1] ** 4.,
            0.25 * (mij[0] / (m[0] * m[2]) - 2. * mij[2] / m[2] ** 2 + m[0] * mij[4] / m[2] ** 3),
            0.25 * (mij[4] / (m[2] * m[4]) - 2 * mij[6] / m[4] ** 2 + m[2] * mij[8] / m[4] ** 3) ,
            m_11 / m[0] ** 2 + (m[5] / m[0] ** 2) ** 2 * mij[0] - 2 * m[5] / m[0] ** 3 * m_10,
            nan,
            (8 * pi / g) ** 2 * (m[2] ** 2 / (4 * m[0] ** 3) * mij[0] + mij[4] / m[0] - m[2] / m[0] ** 2 * mij[2]),
            nan * ones(4),
            m[2] ** 2 * mij[0] / (4 * m[0] ** 3 * m[4]) + mij[4] / (m[0] * m[4]) + mij[8] * m[2] ** 2 / (4 * m[0] * m[4] ** 3) - 
            m[2] * mij[2] / (m[0] ** 2 * m[4]) + m[2] ** 2 * mij[4] / (2 * m[0] ** 2 * m[4] ** 2) - m[2] * mij[6] / m[0] / m[4] ** 2,
            (m[2] ** 2 * mij[0] / 4 + (m[0] * m[2] / m[1]) ** 2 * mij[2] + m[0] ** 2 * mij[4] / 4 - m[2] ** 2 * m[0] * mij[1] / m[1] + 
                m[0] * m[2] * mij[2] / 2 - m[0] ** 2 * m[2] / m[1] * mij[3]) / eps2 ** 2 / m[1] ** 4,
            (m[2] ** 2 * mij[0] / (4 * m[0] ** 2) + mij[4] + m[2] ** 2 * mij[8] / (4 * m[4] ** 2) - m[2] * mij[2] / m[0] + 
            m[2] ** 2 * mij[4] / (2 * m[0] * m[4]) - m[2] * mij[6] / m[4]) * m[2] ** 2 / (m[0] * m[4] * eps4) ** 2,
            nan]

        #% and covariances by a taylor expansion technique:
        #% Cov(Hm0,Tm01) Cov(Hm0,Tm02) Cov(Tm01,Tm02)
        S0 = r_[ 2. / (sqrt(m[0]) * m[1]) * (mij[0] - m[0] * mij[1] / m[1]),
            1. / sqrt(m[2]) * (mij[0] / m[0] - mij[2] / m[2]),
            1. / (2 * m[1]) * sqrt(m[0] / m[2]) * (mij[0] / m[0] - mij[2] / m[2] - mij[1] / m[1] + m[0] * mij[3] / (m[1] * m[2]))]

        R1 = ones((15, 15))
        R1[:, :] = nan
        for ix, Ri in enumerate(R):
            R1[ix, ix] = Ri



        R1[0, 2:4] = S0[:2]
        R1[1, 2] = S0[2]
        for ix in [0, 1]: #%make lower triangular equal to upper triangular part
            R1[ix + 1:, ix] = R1[ix, ix + 1:]


        R = R[nfact]
        R1 = R1[nfact, :][:, nfact]


        #% Needs further checking:
        #% Var(Tm24)= 0.25*(mij[4]/(m[2]*m[4])-2*mij[6]/m[4]**2+m[2]*mij[8]/m[4]**3) ...
        return ch, R1, chtxt

    def setlabels(self):
        ''' Set automatic title, x-,y- and z- labels on SPECDATA object

            based on type, angletype, freqtype
        '''

        N = len(self.type)
        if N == 0:
            raise ValueError('Object does not appear to be initialized, it is empty!')

        labels = ['', '', '']
        if self.type.endswith('dir'):
            title = 'Directional Spectrum'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[2] = 'S(w,\theta) [m^2 s / rad^2]'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[2] = 'S(f,\theta) [m^2 s / rad]'

            if self.angletype.startswith('r'):
                labels[1] = 'Wave directions [rad]'
            elif self.angletype.startswith('d'):
                labels[1] = 'Wave directions [deg]'
        elif self.type.endswith('freq'):
            title = 'Spectral density'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[1] = 'S(w) [m^2 s/ rad]'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[1] = 'S(f) [m^2 s]'
        else:
            title = 'Wave Number Spectrum'
            labels[0] = 'Wave number [rad/m]'
            if self.type.endswith('k1d'):
                labels[1] = 'S(k) [m^3/ rad]'
            elif self.type.endswith('k2d'):
                labels[1] = labels[0]
                labels[2] = 'S(k1,k2) [m^4/ rad^2]'
            else:
                raise ValueError('Object does not appear to be initialized, it is empty!')
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
        
class SpecData2D(WafoData):
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
    >>> Sj = sm.Jonswap(Hm0=3)
    >>> w = np.linspace(0,4,256)
    >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values

    See also
    --------
    WafoData
    CovData
    """

    def __init__(self, *args, **kwds):
        super(SpecData2D, self).__init__(*args, **kwds)

        self.name = 'WAFO Spectrum Object'
        self.type = 'freq'
        self.freqtype = 'w'
        self.angletype = ''
        self.h = inf
        self.tr = None
        self.phi = 0.
        self.v = 0.
        self.norm = 0
        somekeys = ['angletype', 'phi', 'name', 'h', 'tr', 'freqtype', 'v', 'type', 'norm']

        self.__dict__.update(sub_dict_select(kwds, somekeys))

        if self.type.endswith('dir') and self.angletype == '':
            self.angletype = 'radians'

        self.setlabels()

    def toacf(self):
        pass
    def sim(self):
        pass
    def sim_nl(self):
        pass
    def rotate(self):
        pass
    def moment(self, nr=2, vari='xt', even=True):
        ''' 
        Calculates spectral moments from spectrum

        Parameters
        ----------
        nr   : int
            order of moments (maximum 4)
        vari : string
            variables in model, optional when two-dim.spectrum,
                   string with 'x' and/or 'y' and/or 't'
        even : bool
            False for all moments,
            True for only even orders

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
          S=demospec('dir')
          [m,mtext]=spec2mom(S,2,'xyt')

        References
        ----------
        Baxevani A. et al. (2001)
        Velocities for Random Surfaces
        '''

##% Tested on: Matlab 6.0
##% Tested on: Matlab 5.3
##% History:
##% Revised by I.R. 04.04.2001: Introducing the rotation angle phi.
##% Revised by A.B. 23.05.2001: Correcting 'mxxyy' and introducing
##% 'mxxyt','mxyyt' and 'mxytt'.
##% Revised by A.B. 21.10.2001: Correcting 'mxxyt'.
##% Revised by A.B. 21.10.2001: Adding odd-order moments.
##% By es 27.08.1999


        pi = pi
        two_dim_spectra = ['dir', 'encdir', 'k2d']
        if self.type not in two_dim_spectra:
            raise ValueError('Unknown 2D spectrum type!')

##        if (vari==None and nr<=1:
##            vari='x'
##        elif vari==None:
##            vari='xt'
##        else #% secure the mutual order ('xyt')
##            vari=''.join(sorted(vari.lower()))
##            Nv=len(vari)
##
##            if vari[0]=='t' and Nv>1:
##                vari = vari[1::]+ vari[0]
##
##        Nv = len(vari)
##
##        if not self.type.endswith('dir'):
##            S1 = self.tospecdata(self.type[:-2]+'dir')
##        else:
##            S1 = self
##        w = ravel(S1.args[0])
##        theta = S1.args[1]-S1.phi
##        S = S1.data
##        Sw = simps(S,x=theta)
##        m = [simps(Sw,x=w)]
##        mtext=['m0']
##
##        if nr>0:
##
##          nw=w.size
##          if strcmpi(vari(1),'x')
##            Sc=simpson(th,S1.S.*(cos(th)*ones(1,nw))).'
##            % integral S*cos(th) dth
##          end
##          if strcmpi(vari(1),'y')
##            Ss=simpson(th,S1.S.*(sin(th)*ones(1,nw))).'
##            % integral S*sin(th) dth
##            if strcmpi(vari(1),'x')
##            Sc=simpson(th,S1.S.*(cos(th)*ones(1,nw))).'
##            end
##          end
##          if ~isfield(S1,'g')
##            S1.g=gravity
##          end
##          kx=w.^2/S1.g(1) % maybe different normalization in x and y => diff. g
##          ky=w.^2/S1.g(end)
##
##          if Nv>=1
##            switch vari
##              case 'x'
##                vec = kx.*Sc
##                mtext(end+1)={'mx'}
##              case 'y'
##                vec = ky.*Ss
##                mtext(end+1)={'my'}
##              case 't'
##                vec = w.*Sw
##               mtext(end+1)={'mt'}
##            end
##          else
##            vec = [kx.*Sc ky.*Ss w*Sw]
##            mtext(end+(1:3))={'mx', 'my', 'mt'}
##          end
##          if nr>1
##          if strcmpi(vari(1),'x')
##            Sc=simpson(th,S1.S.*(cos(th)*ones(1,nw))).'
##            % integral S*cos(th) dth
##            Sc2=simpson(th,S1.S.*(cos(th).^2*ones(1,nw))).'
##            % integral S*cos(th)^2 dth
##          end
##          if strcmpi(vari(1),'y')||strcmpi(vari(2),'y')
##            Ss=simpson(th,S1.S.*(sin(th)*ones(1,nw))).'
##            % integral S*sin(th) dth
##            Ss2=simpson(th,S1.S.*(sin(th).^2*ones(1,nw))).'
##            % integral S*sin(th)^2 dth
##            if strcmpi(vari(1),'x')
##              Scs=simpson(th,S1.S.*((cos(th).*sin(th))*ones(1,nw))).'
##              % integral S*cos(th)*sin(th) dth
##            end
##          end
##          if ~isfield(S1,'g')
##            S1.g=gravity
##          end
##
##          if Nv==2
##            switch vari
##              case 'xy'
##                vec=[kx.*Sc ky.*Ss kx.^2.*Sc2 ky.^2.*Ss2 kx.*ky.*Scs]
##                mtext(end+(1:5))={'mx','my','mxx', 'myy', 'mxy'}
##              case 'xt'
##                vec=[kx.*Sc w.*Sw kx.^2.*Sc2 w.^2.*Sw kx.*w.*Sc]
##                mtext(end+(1:5))={'mx','mt','mxx', 'mtt', 'mxt'}
##              case 'yt'
##                vec=[ky.*Ss w.*Sw ky.^2.*Ss2 w.^2.*Sw ky.*w.*Ss]
##                mtext(end+(1:5))={'my','mt','myy', 'mtt', 'myt'}
##            end
##          else
##            vec=[kx.*Sc ky.*Ss w.*Sw kx.^2.*Sc2 ky.^2.*Ss2  w.^2.*Sw kx.*ky.*Scs kx.*w.*Sc ky.*w.*Ss]
##            mtext(end+(1:9))={'mx','my','mt','mxx', 'myy', 'mtt', 'mxy', 'mxt', 'myt'}
##          end
##          if nr>3
##            if strcmpi(vari(1),'x')
##              Sc3=simpson(th,S1.S.*(cos(th).^3*ones(1,nw))).'
##              % integral S*cos(th)^3 dth
##              Sc4=simpson(th,S1.S.*(cos(th).^4*ones(1,nw))).'
##              % integral S*cos(th)^4 dth
##            end
##            if strcmpi(vari(1),'y')||strcmpi(vari(2),'y')
##              Ss3=simpson(th,S1.S.*(sin(th).^3*ones(1,nw))).'
##              % integral S*sin(th)^3 dth
##              Ss4=simpson(th,S1.S.*(sin(th).^4*ones(1,nw))).'
##              % integral S*sin(th)^4 dth
##              if strcmpi(vari(1),'x')  %both x and y
##                Sc2s=simpson(th,S1.S.*((cos(th).^2.*sin(th))*ones(1,nw))).'
##                % integral S*cos(th)^2*sin(th) dth
##                Sc3s=simpson(th,S1.S.*((cos(th).^3.*sin(th))*ones(1,nw))).'
##                % integral S*cos(th)^3*sin(th) dth
##                Scs2=simpson(th,S1.S.*((cos(th).*sin(th).^2)*ones(1,nw))).'
##                % integral S*cos(th)*sin(th)^2 dth
##                Scs3=simpson(th,S1.S.*((cos(th).*sin(th).^3)*ones(1,nw))).'
##                % integral S*cos(th)*sin(th)^3 dth
##                Sc2s2=simpson(th,S1.S.*((cos(th).^2.*sin(th).^2)*ones(1,nw))).'
##                % integral S*cos(th)^2*sin(th)^2 dth
##              end
##            end
##            if Nv==2
##              switch vari
##                case 'xy'
##                  vec=[vec kx.^4.*Sc4 ky.^4.*Ss4 kx.^3.*ky.*Sc3s ...
##                        kx.^2.*ky.^2.*Sc2s2 kx.*ky.^3.*Scs3]
##                  mtext(end+(1:5))={'mxxxx','myyyy','mxxxy','mxxyy','mxyyy'}
##                case 'xt'
##                  vec=[vec kx.^4.*Sc4 w.^4.*Sw kx.^3.*w.*Sc3 ...
##                        kx.^2.*w.^2.*Sc2 kx.*w.^3.*Sc]
##                  mtext(end+(1:5))={'mxxxx','mtttt','mxxxt','mxxtt','mxttt'}
##                case 'yt'
##                  vec=[vec ky.^4.*Ss4 w.^4.*Sw ky.^3.*w.*Ss3 ...
##                        ky.^2.*w.^2.*Ss2 ky.*w.^3.*Ss]
##                  mtext(end+(1:5))={'myyyy','mtttt','myyyt','myytt','myttt'}
##              end
##            else
##              vec=[vec kx.^4.*Sc4 ky.^4.*Ss4 w.^4.*Sw kx.^3.*ky.*Sc3s ...
##                   kx.^2.*ky.^2.*Sc2s2 kx.*ky.^3.*Scs3 kx.^3.*w.*Sc3 ...
##                   kx.^2.*w.^2.*Sc2 kx.*w.^3.*Sc ky.^3.*w.*Ss3 ...
##                   ky.^2.*w.^2.*Ss2 ky.*w.^3.*Ss kx.^2.*ky.*w.*Sc2s ...
##                   kx.*ky.^2.*w.*Scs2 kx.*ky.*w.^2.*Scs]
##              mtext(end+(1:15))={'mxxxx','myyyy','mtttt','mxxxy','mxxyy',...
##              'mxyyy','mxxxt','mxxtt','mxttt','myyyt','myytt','myttt','mxxyt','mxyyt','mxytt'}
##
##            end % if Nv==2 ... else ...
##          end % if nr>3
##          end % if nr>1
##          m=[m simpson(w,vec)]
##        end % if nr>0
##      %  end %%if Nv==1... else...    to be removed
##    end % ... else two-dim spectrum



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
            raise ValueError('Object does not appear to be initialized, it is empty!')

        labels = ['', '', '']
        if self.type.endswith('dir'):
            title = 'Directional Spectrum'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[2] = 'S(w,\theta) [m**2 s / rad**2]'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[2] = 'S(f,\theta) [m**2 s / rad]'

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
                raise ValueError('Object does not appear to be initialized, it is empty!')
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
        
def test_specdata():
    import wafo.spectrum.models as sm
    Sj = sm.Jonswap()
    S = Sj.tospecdata()
    me, va, sk, ku = S.stats_nl(moments='mvsk')
        
def main():
    import matplotlib
    matplotlib.interactive(True)
    from wafo.spectrum import models as sm
    
    w = linspace(0, 3, 100)
    Sj = sm.Jonswap()
    S = Sj.tospecdata()
    
    f = S.to_t_pdf(pdef='Tc', paramt=(0, 10, 51), speed=7)
    f.err
    f.plot()
    f.show()
    #pdfplot(f)
    #hold on, 
    #plot(f.x{:}, f.f+f.err,'r',f.x{:}, f.f-f.err)  estimated error bounds
    #hold off  
    #S = SpecData1D(Sj(w),w)
    R = S.tocovdata(nr=1)
    S1 = S.copy()
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

if __name__ == '__main__':
    if  True: #False : #  
        import doctest
        doctest.testmod()
    else:
        main()
