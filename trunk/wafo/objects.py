
#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      pab
#
# Created:     16.09.2008
# Copyright:   (c) pab 2008
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python


from __future__ import division
import warnings
import numpy as np

from numpy import (inf, pi, zeros, ones, sqrt, where, log, exp, sin, arcsin, mod, #@UnresolvedImport
                   newaxis, linspace, arange, sort, all, abs, linspace, vstack, hstack, atleast_1d, #@UnresolvedImport
                   polyfit, r_, nonzero, cumsum, ravel, size, isnan,  nan, floor, ceil, diff, array) #@UnresolvedImport
from numpy.fft import fft
from numpy.random import randn
from scipy.integrate import trapz
from pylab import stineman_interp
from matplotlib.mlab import psd
import scipy.signal

from scipy.special import erf

from wafo.misc import (nextpow2, findtp, findtc, findcross, sub_dict_select, 
                       ecross, JITImport)
from wafodata import WafoData
from plotbackend import plotbackend
import matplotlib
matplotlib.interactive(True)
_wafocov = JITImport('wafo.covariance')
_wafospec = JITImport('wafo.spectrum')

__all__ = ['TimeSeries','LevelCrossings','CyclePairs','TurningPoints',
    'sensortypeid','sensortype']



class LevelCrossings(WafoData):
    '''
    Container class for Level crossing data objects in WAFO

    Member variables
    ----------------
    data : array-like
        number of upcrossings
    args : array-like
        crossing levels

    '''
    def __init__(self,*args,**kwds):
        super(LevelCrossings, self).__init__(*args,**kwds)
        self.labels.title = 'Level crossing spectrum'
        self.labels.xlab = 'Levels'
        self.labels.ylab = 'Count'
        self.stdev = kwds.get('stdev',None)
        self.mean = kwds.get('mean',None)
        self.setplotter(plotmethod='step')

        icmax = self.data.argmax()
        if self.data != None:
            if self.stdev is None or self.mean is None:
                logcros = where(self.data==0.0, inf, -log(self.data))
                logcmin = logcros[icmax]
                logcros = sqrt(2*abs(logcros-logcmin))
                logcros[0:icmax+1] = 2*logcros[icmax]-logcros[0:icmax+1]
                p = polyfit(self.args[10:-9], logcros[10:-9],1) #least square fit
                if self.stdev is None:
                    self.stdev = 1.0/p[0] #estimated standard deviation of x
                if self.mean is None:
                    self.mean = -p[1]/p[0] #self.args[icmax]
            cmax = self.data[icmax]
            x = (self.args-self.mean)/self.stdev
            y = cmax*exp(-x**2/2.0)
            self.children = [WafoData(y,self.args)]

    def sim(self,ns,alpha):
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
        >>> Sj = sm.Jonswap(Hm0=7)
        >>> S = Sj.tospecdata()   #Make spectrum object from numerical values
        >>> alpha = S.characteristic('alpha')[0]
        >>> n = 10000
        >>> xs = S.sim(ns=n)
        >>> ts = mat2timeseries(xs)
        >>> tp = ts.turning_points()
        >>> mm = tp.cycle_pairs()
        >>> lc = mm.level_crossings()  
        
        xs2 = lc.sim(n,alpha)
        ts2 = mat2timeseries(xs2)
        Se  = ts2.tospecdata()
        
        S.plot('b')
        Se.plot('r')
        alpha2 = Se.characteristic('alpha')[0]
        alpha-alpha2
        
        spec2char(Se,'alpha')
        lc2  = dat2lc(xs2)
        figure(gcf+1)
        subplot(211)
        lcplot(lc2)
        subplot(212)
        lcplot(lc)
        """

        # TODO % add a good example
        f = linspace(0,0.49999,1000)
        rho_st = 2.*sin(f*pi)**2-1.
        tmp = alpha*arcsin(sqrt((1.+rho_st)/2))
        tmp = sin(tmp)**2
        a2  = (tmp-rho_st)/(1-tmp)
        y   = vstack((a2+rho_st,1-a2)).min(axis=0)
        maxidx = y.argmax()
        #[maximum,maxidx]=max(y)

        rho_st = rho_st[maxidx]
        a2 = a2[maxidx]
        a1 = 2.*rho_st+a2-1.
        r0 = 1.
        r1 = -a1/(1.+a2)
        r2 = (a1**2-a2-a2**2)/(1+a2)
        sigma2 = r0+a1*r1+a2*r2
        #randn = np.random.randn
        e = randn(ns)*sqrt(sigma2)
        e[:1] = 0.0
        L0 = randn(1)
        L0 = vstack((L0,r1*L0+sqrt(1-r2**2)*randn(1)))
        #%Simulate the process, starting in L0
        lfilter = scipy.signal.lfilter
        L = lfilter(1,[1, a1, a2],e,lfilter([1, a1, a2],1,L0))

        epsilon = 1.01
        min_L   = min(L)
        max_L   = max(L)
        maxi    = max(abs(r_[min_L, max_L]))*epsilon
        mini    = -maxi

        u = linspace(mini,maxi,101)
        G = (1+erf(u/sqrt(2)))/2
        G = G*(1-G)

        x = linspace(0,r1,100)
        factor1 = 1./sqrt(1-x**2)
        factor2 = 1./(1+x)
        integral = zeros(u.shape, dtype=float)
        for i in range(len(integral)):
            y = factor1*exp(-u[i]*u[i]*factor2)
            integral[i] = trapz(x,y)
        #end
        G = G-integral/(2*pi)
        G = G/max(G)

        Z = ((u>=0)*2-1)*sqrt(-2*log(G))

##        sumcr   = trapz(lc(:,1),lc(:,2))
##        lc(:,2) = lc(:,2)/sumcr
##        mcr     = trapz(lc(:,1),lc(:,1).*lc(:,2))
##        scr     = trapz(lc(:,1),lc(:,1).^2.*lc(:,2))
##        scr     = sqrt(scr-mcr^2)
##        g       = lc2tr(lc,mcr,scr)
##
##        f = [u u]
##        f(:,2) = tranproc(Z,fliplr(g))
##
##        process = tranproc(L,f)
##        process = [(1:length(process)) process]
##
##
##        %Check the result without reference to getrfc:
##        LCe = dat2lc(process)
##        max(lc(:,2))
##        max(LCe(:,2))
##
##        clf
##        plot(lc(:,1),lc(:,2)/max(lc(:,2)))
##        hold on
##        plot(LCe(:,1),LCe(:,2)/max(LCe(:,2)),'-.')
##        title('Relative crossing intensity')
##
##        %% Plot made by the function funplot_4, JE 970707
##        %param = [min(process(:,2)) max(process(:,2)) 100]
##        %plot(lc(:,1),lc(:,2)/max(lc(:,2)))
##        %hold on
##        %plot(levels(param),mu/max(mu),'--')
##        %hold off
##        %title('Crossing intensity')
##        %watstamp
##
##        % Temporarily
##        %funplot_4(lc,param,mu)


    def trdata(self, mean=None, sigma=None, **options):
        '''
        Estimate transformation, g, from observed crossing intensity, version2.

        Assumption: a Gaussian process, Y, is related to the
                    non-Gaussian process, X, by Y = g(X). 

        Parameters
        ----------        
           options = structure with the fields:
          csm, gsm  - defines the smoothing of the crossing intensity and the
                     transformation g. Valid values must be
                    0<=csm,gsm<=1. (default csm = 0.9 gsm=0.05)
                     Smaller values gives smoother functions.
             param - vector which defines the region of variation of the data X.
                     (default [-5 5 513]). 
           
          monitor            monitor development of estimation
         linextrap - 0 use a regular smoothing spline 
                     1 use a smoothing spline with a constraint on the ends to 
                       ensure linear extrapolation outside the range of the data.
                       (default)
         cvar      - Variances for the crossing intensity. (default 1)
         gvar      - Variances for the empirical transformation, g. (default  1) 
         ne        - Number of extremes (maxima & minima) to remove from the
                      estimation of the transformation. This makes the
                      estimation more robust against outliers. (default 7)
         Ntr        - Maximum length of empirical crossing intensity.
                      The empirical crossing intensity is interpolated
                      linearly  before smoothing if the length exceeds Ntr.
                      A reasonable NTR will significantly speed up the
                      estimation for long time series without loosing any
                      accuracy. NTR should be chosen greater than
                      PARAM(3). (default 1000)
        Returns
        -------
        gs, ge : TrData objects 
            smoothed and empirical estimate of the transformation g.     
             ma,sa = mean and standard deviation of the process
        
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
        Hm0 = 7
        S = jonswap([],Hm0); g=ochitr([],[Hm0/4]); 
         S.tr = g; S.tr(:,2)=g(:,2)*Hm0/4;
         xs = spec2sdat(S,2^13);
         lc = dat2lc(xs)
         g0 = lc2tr2(lc,0,Hm0/4,'plot','iter');         % Monitor the development
         g1 = lc2tr2(lc,0,Hm0/4,troptset('gvar', .5 )); % Equal weight on all points
         g2 = lc2tr2(lc,0,Hm0/4,'gvar', [3.5 .5 3.5]);  % Less weight on the ends
         hold on, trplot(g1,g)                          % Check the fit
         trplot(g2)
        
         See also  troptset, dat2tr, trplot, findcross, smooth
        
         NB! the transformated data will be N(0,1)
        
        Reference
        --------- 
        Rychlik , I., Johannesson, P., and Leadbetter, M.R. (1997)
        "Modelling and statistical analysis of ocean wavedata 
        using a transformed Gaussian process",
        Marine structures, Design, Construction and Safety, 
        Vol 10, pp 13--47
        ''' 
        
        
        # Tested on: Matlab 5.3, 5.2, 5.1
        # History:
        # by pab 29.12.2000
        # based on lc2tr, but the inversion is faster.
        # by IR and PJ
        pass
        
#        opt = troptset('chkder','on','plotflag','off','csm',.9,'gsm',.05,....
#            'param',[-5 5 513],'delay',2,'linextrap','on','ntr',1000,'ne',7,'cvar',1,'gvar',1);
#        # If just 'defaults' passed in, return the default options in g
#        if nargin==1 && nargout <= 1 && isequal(cross,'defaults')
#          g = opt; 
#          return
#        end
#        error(nargchk(3,inf,nargin)) 
#        if nargin>=4 ,  opt  = troptset(opt,varargin{:}); end
#        csm2 = opt.gsm;
#        param = opt.param;
#        ptime = opt.delay;
#        Ne  = opt.ne;
#        switch opt.chkder;
#          case 'off', chkder = 0;
#          case 'on',  chkder = 1;
#          otherwise,  chkder = opt.chkder;
#        end
#        switch opt.linextrap;
#          case 'off', def = 0;
#          case 'on',  def = 1;
#          otherwise,  def = opt.linextrap;
#        end
#        switch opt.plotflag
#          case {'none','off'},   plotflag = 0;
#          case 'final', plotflag = 1;
#          case 'iter',  plotflag = 2;
#          otherwise,    plotflag = opt.plotflag;
#        end
#        ncr = length(cross);
#        if ncr>opt.ntr && opt.ntr>0,
#           x0 = linspace(cross(1+Ne,1),cross(end-Ne,1),opt.ntr)';
#           cros = [ x0,interp1q(cross(:,1),cross(:,2),x0)];
#           Ne = 0;
#           Ner = opt.ne;
#           ncr = opt.ntr;
#         else
#           Ner = 0;
#          cros=cross;
#        end
#        
#        ng = length(opt.gvar);
#        if ng==1
#          gvar = opt.gvar(ones(ncr,1));
#        else
#          gvar = interp1(linspace(0,1,ng)',opt.gvar(:),linspace(0,1,ncr)','*linear');  
#        end
#        ng = length(opt.cvar);
#        if ng==1
#          cvar = opt.cvar(ones(ncr,1));
#        else
#          cvar = interp1(linspace(0,1,ng)',opt.cvar(:),linspace(0,1,ncr)','*linear');  
#        end
#        
#        g = zeros(param(3),2);
#        
#        uu = levels(param);
#        
#        g(:,1) = sa*uu' + ma;
#        
#        g2   = cros;
#        
#        if Ner>0, # Compute correction factors
#         cor1 = trapz(cross(1:Ner+1,1),cross(1:Ner+1,2));
#         cor2 = trapz(cross(end-Ner-1:end,1),cross(end-Ner-1:end,2));
#        else
#          cor1 = 0;
#          cor2 = 0;
#        end
#        cros(:,2) = cumtrapz(cros(:,1),cros(:,2))+cor1;
#        cros(:,2) = (cros(:,2)+.5)/(cros(end,2) + cor2 +1);
#        cros(:,1) = (cros(:,1)-ma)/sa;
#        
#        # find the mode
#        [tmp,imin]= min(abs(cros(:,2)-.15));
#        [tmp,imax]= min(abs(cros(:,2)-.85));
#        inde = imin:imax;
#        tmp =  smooth(cros(inde,1),g2(inde,2),opt.csm,cros(inde,1),def,cvar(inde));
#        
#        [tmp imax] = max(tmp);
#        u0 = cros(inde(imax),1);
#        #u0 = interp1q(cros(:,2),cros(:,1),.5)
#        
#        
#        cros(:,2) = invnorm(cros(:,2),-u0,1);
#        
#        g2(:,2)   = cros(:,2);
#        # NB! the smooth function does not always extrapolate well outside the edges
#        # causing poor estimate of g  
#        # We may alleviate this problem by: forcing the extrapolation
#        # to be linear outside the edges or choosing a lower value for csm2.
#        
#        inds = 1+Ne:ncr-Ne;# indices to points we are smoothing over
#        scros2 = smooth(cros(inds,1),cros(inds,2),csm2,uu,def,gvar(inds));
#        
#        g(:,2) = scros2';#*sa; #multiply with stdev 
#        
#        if chkder~=0
#           for ix = 1:5
#            dy = diff(g(:,2));
#            if any(dy<=0)
#              warning('WAFO:LCTR2','The empirical crossing spectrum is not sufficiently smoothed.')
#              disp('        The estimated transfer function, g, is not ')
#              disp('        a strictly increasing function.')
#              dy(dy>0)=eps;
#              gvar = -([dy;0]+[0;dy])/2+eps;
#              g(:,2) = smooth(g(:,1),g(:,2),1,g(:,1),def,ix*gvar);
#            else 
#              break
#            end
#          end
#        end
#        if 0,  #either
#          test = sqrt((param(2)-param(1))/(param(3)-1)*sum((uu-scros2).^2));
#        else # or
#          #test=sqrt(simpson(uu,(uu-scros2).^2));
#        # or
#          test=sqrt(trapz(uu,(uu-scros2).^2));
#        end 
#        
#        
#        if plotflag>0, 
#          trplot(g ,g2,ma,sa)
#          #legend(['Smoothed (T='  num2str(test) ')'],'g(u)=u','Not smoothed',0)
#          #ylabel('Transfer function g(u)')
#          #xlabel('Crossing level u')
#          
#          if plotflag>1,pause(ptime),end
#        end


class CyclePairs(WafoData):
    '''
    Container class for Cycle Pairs data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D


    '''
    def __init__(self, *args, **kwds):
        super(CyclePairs, self).__init__(*args, **kwds)
        self.type_ = kwds.get('type_', 'max2min')
        self.stdev = kwds.get('stdev', None)
        self.mean = kwds.get('mean', None)

        self.labels.title = self.type_+ ' cycle pairs'
        self.labels.xlab = 'min'
        self.labels.ylab = 'max'

    def amplitudes(self):
        return (self.data-self.args)/2.
    
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
        >>> h = mm.plot('.')
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
        return atleast_1d([K*np.sum(amp**betai) for betai in beta])

    def level_crossings(self, type_='uM'):
        """ Return number of upcrossings from a cycle count.

        Parameters
        ----------
        type_ : int or string
            defining crossing type, options are
            0,'u'  : only upcrossings.
            1,'uM' : upcrossings and maxima (default).
            2,'umM': upcrossings, minima, and maxima.
            3,'um' :upcrossings and minima.
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
        >>> h = mm.plot('.')
        >>> lc = mm.level_crossings()
        >>> h2 = lc.plot()

        See also
        --------
        TurningPoints
        LevelCrossings
        """

        if isinstance(type_, str):
            t = dict(u=0, uM=1, umM=2, um=3)
            defnr = t.get(type_, 1)
        else:
            defnr = type_

        if ((defnr<0) or (defnr>3)):
            raise ValueError('type_ must be one of (1,2,3,4).')

        index, = nonzero(self.args <= self.data)
        if index.size == 0:
            index, = nonzero(self.args >= self.data)
            M = self.args[index]
            m = self.data[index]
        else:
            m = self.args[index]
            M = self.data[index]

#if isempty(index)
#  error('Error in input cc.')
#end
        ncc = len(m)
        #ones = np.ones
        #zeros = np.zeros
        #cumsum = np.cumsum
        minima = vstack((m, ones(ncc), zeros(ncc), ones(ncc)))
        maxima = vstack((M, -ones(ncc), ones(ncc), zeros(ncc)))

        extremes = hstack((maxima, minima))
        index = extremes[0].argsort()
        extremes = extremes[:, index]

        ii = 0
        n = extremes.shape[1]
        extr = zeros((4, n))
        extr[:, 0] = extremes[:, 0]
        for i in xrange(1, n):
            if extremes[0, i] == extr[0, ii]:
                extr[1:4, ii] = extr[1:4, ii] + extremes[1:4, i]
            else:
                ii += 1
                extr[:, ii] = extremes[:, i]

        #[xx nx]=max(extr(:,1))
        nx = extr[0].argmax()+1
        levels = extr[0, 0:nx]
        if defnr == 2: ## This are upcrossings + maxima
            dcount = cumsum(extr[1, 0:nx]) + extr[2, 0:nx]-extr[3, 0:nx]
        elif defnr == 4: # # This are upcrossings + minima
            dcount = cumsum(extr[1, 0:nx])
            dcount[nx-1] = dcount[nx-2]
        elif defnr == 1: ## This are only upcrossings
            dcount = cumsum(extr[1, 0:nx]) - extr[3, 0:nx]
        elif defnr == 3: ## This are upcrossings + minima + maxima
            dcount = cumsum(extr[1, 0:nx]) + extr[2, 0:nx]
        return LevelCrossings(dcount, levels, stdev=self.stdev)

class TurningPoints(WafoData):
    '''
    Container class for Turning Points data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D


    '''
    def __init__(self, *args, **kwds):
        super(TurningPoints, self).__init__(*args, **kwds)
        self.name='WAFO TurningPoints Object'
        somekeys = ['name']
        self.__dict__.update(sub_dict_select(kwds, somekeys))

        #self.setlabels()
        if not any(self.args):
            n = len(self.data)
            self.args = range(0, n)
        else:
            self.args = ravel(self.args)
        self.data= ravel(self.data)

    def cycle_pairs(self, type_='min2max'):
        """ Return min2Max or Max2min cycle pairs from turning points

        Parameters
        ----------
        type_ : string
            type of cycles to return options are 'min2max' or 'max2min'

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
        >>> h = mM.plot('.')


        See also
        --------
        TurningPoints
        SurvivalCycleCount
        """
        if self.data[0]>self.data[1]:
            im = 1
            iM = 0
        else:
            im = 0
            iM = 1

        # Extract min-max and max-min cycle pairs
        #n = len(self.data)
        if type_.lower().startswith('min2max'):
            m = self.data[im:-1:2]
            M = self.data[im+1::2]
        else:
            type_ = 'max2min'
            M = self.data[iM:-1:2]
            m = self.data[iM+1::2]

        return CyclePairs(M, m, type=type_)

def mat2timeseries(x):
    """
    Convert 2D arrays to TimeSeries object
        assuming 1st column is time and the remaining columns contain data.
    """
    return TimeSeries(x[:, 1::], x[:, 0].ravel())

class TimeSeries(WafoData):
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
    >>> x = wafo.data.sea()
    >>> ts = mat2timeseries(x)
    >>> rf = ts.tocovdata(lag=150)
    >>> h = rf.plot()

    '''
    def __init__(self, *args, **kwds):
        super(TimeSeries, self).__init__(*args, **kwds)
        self.name = 'WAFO TimeSeries Object'
        self.sensortypes = ['n', ]
        self.position = zeros(3)
        somekeys = ['sensortypes', 'position']
        self.__dict__.update(sub_dict_select(kwds, somekeys))

        #self.setlabels()
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
        dt1 = self.args[1]-self.args[0]
        n = size(self.args)-1
        t = self.args[-1]-self.args[0]
        dt = t/n
        if abs(dt-dt1) > 1e-10:
            warnings.warn('Data is not uniformly sampled!')
        return dt

    def tocovdata(self, lag=None, flag='biased', norm=False, dt = None):
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
            stdev : estimated large lag standard deviation of the estimate
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
         >>> x = wafo.data.sea()
         >>> ts = mat2timeseries(x)
         >>> acf = ts.tocovdata(150)
         >>> h = acf.plot()
        '''
        n = len(self.data)
        if not lag:
            lag = n-1

        x = self.data.flatten()
        indnan = isnan(x)
        if any(indnan):
            x = x - x[1-indnan].mean() # remove the mean pab 09.10.2000
            #indnan = find(indnan)
            Ncens = n - sum(indnan)
            x[indnan] = 0. # pab 09.10.2000 much faster for censored samples
        else:
            indnan = None
            Ncens = n
            x = x - x.mean()

        #fft = np.fft.fft
        nfft = 2**nextpow2(n)
        Rper = abs(fft(x,nfft))**2/Ncens # Raw periodogram

        R = np.real(fft(Rper))/nfft # %ifft=fft/nfft since Rper is real!
        lags = range(0,lag+1)
        if flag.startswith('unbiased'):
            # unbiased result, i.e. divide by n-abs(lag)
            R = R[lags]*Ncens/arange(Ncens, Ncens-lag, -1)
        #else  % biased result, i.e. divide by n
        #  r=r(1:L+1)*Ncens/Ncens

        c0 = R[0]
        if norm:
            R = R/c0
        if dt is None:
            dt = self.sampling_period()
        t = linspace(0,lag*dt,lag+1)
        #cumsum = np.cumsum
        acf = _wafocov.CovData1D(R[lags],t)
        acf.stdev=sqrt(r_[ 0, 1 ,1+2*cumsum(R[1:]**2)]/Ncens)
        acf.children = [WafoData(-2.*acf.stdev[lags],t),WafoData(2.*acf.stdev[lags],t)]
        acf.norm = norm
        return acf

    def tospecdata(self,*args,**kwargs):
        """ 
        Return power spectral density by Welches average periodogram method.

        Parameters
        ----------
        NFFT : int, scalar
            if len(data) < NFFT, it will be zero padded to `NFFT`
            before estimation. Must be even; a power 2 is most efficient.
        detrend : function
        Fs : real, scalar
            sampling frequency (samples per time unit).

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
        fs = 1./(2*self.sampling_period())
        S, f = psd(self.data.ravel(), Fs=fs, *args, **kwargs)
        fact = 2.0*pi
        w = fact*f
        return _wafospec.SpecData1D(S/fact, w)

    def turning_points(self,h=0.0,wavetype=None):
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
            'mw' 'Mw' or 'none'.
            If None all rainflow filtered min and max
            will be returned, otherwise only the rainflow filtered
            min and max, which define a wave according to the
            wave definition, will be returned.

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
        ind = findtp(self.data, max(h,0.0), wavetype)
        try:
            t = self.args[ind]
        except:
            t = ind
        return TurningPoints(self.data[ind],t)

    def trough_crest(self,v=None,wavetype=None):
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
        return TurningPoints(self.data[ind], t)
    
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
        >>> import wafo
        >>> x = wafo.data.sea()
        >>> ts = wafo.objects.mat2timeseries(x[0:400,:])
        >>> T = ts.wave_periods(vh=0.0,pdef='c2c')

        T = dat2wa(x1,0,'c2c') #% Returns crest2crest waveperiods
        subplot(121), waveplot(x1,'-',1,1),subplot(122),histgrm(T)

        See also:
        --------
        findtp,
        findtc,
        findcross, perioddef
        """

##% This is a more flexible version than the dat2hwa or tp2wa routines.
##% There is a secret option: if pdef='all' the function returns
##% all the waveperiods 'd2t', 't2u', 'u2c' and 'c2d' in sequence.
##% It is up to the user to extract the right waveperiods.
##% If the first is a down-crossing then the first is a 'd2t' waveperiod.
##% If the first is a up-crossing then the first is a 'u2c' waveperiod.
##%
##%	Example:
##%		[T ind]=dat2wa(x,0,'all') %returns all waveperiods
##%		nn = length(T)
##%		% want to extract all t2u waveperiods
##%		if x(ind(1),2)>0 % if first is down-crossing
##%			Tt2u=T(2:4:nn)
##%		else 		% first is up-crossing
##%			Tt2u=T(4:4:nn)
##%		end

        if rate>1: #% interpolate with spline
            n = ceil(self.data.size*rate)
            ti = linspace(self.args[0], self.args[-1], n)
            x = stineman_interp(ti, self.args, self.data)
        else:
            x = self.data
            ti = self.args


        if vh is None:
            if pdef[0] in ('m','M'):
                vh = 0
                print('   The minimum rfc height, h,  is set to: %g' % vh)
            else:
                vh = x.mean()
                print('   The level l is set to: %g' % vh)


        if index is None:
            if pdef in ('m2m', 'm2M', 'M2m','M2M'):
                index = findtp(x, vh, wdef)
            elif pdef in ('u2u','u2d','d2u', 'd2d'):
                index = findcross(x, vh, wdef)
            elif pdef in ('t2t','t2c','c2t', 'c2c'):
                index = findtc(x,vh,wdef)[0]
            elif pdef in ('d2t','t2u', 'u2c', 'c2d','all'):
                index, v_ind = findtc(x, vh, wdef)
                index = sort(r_[index, v_ind]) #% sorting crossings and tp in sequence
            else:
                raise ValueError('Unknown pdef option!')

        if (x[index[0]]>x[index[1]]): #% if first is down-crossing or max
            if pdef in  ('d2t', 'M2m', 'c2t', 'd2u' , 'M2M', 'c2c', 'd2d', 'all'):
                start = 1
            elif pdef in ('t2u', 'm2M', 't2c', 'u2d' ,'m2m', 't2t', 'u2u'):
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

        # determine the steps between wanted periods
        if pdef in ('d2t', 't2u', 'u2c', 'c2d' ):
            step = 4
        elif pdef in ('all'):
            step = 1 #% secret option!
        else:
            step = 2

        #% determine the distance between min2min, t2t etc..
        if pdef in ('m2m', 't2t', 'u2u', 'M2M', 'c2c', 'd2d'):
            dist = 2
        else:
            dist = 1

        nn = len(index)
        #% New call: (pab 28.06.2001)
        if pdef[0] in ('u', 'd'):
            t0 = ecross(ti, x, index[start:(nn-dist):step], vh)
        else: # % min, Max, trough, crest or all crossings wanted
            t0 = x[index[start:(nn-dist):step]]

        if pdef[2] in ('u','d'):
            t1 = ecross(ti, x, index[(start+dist):nn:step], vh)
        else: # % min, Max, trough, crest or all crossings wanted
            t1 = x[index[(start+dist):nn:step]]

        T = t1 - t0
##        if False: #% Secret option: indices to the actual crossings used.
##            index=index.ravel()
##            ind = [index(start:(nn-dist):step) index((start+dist):nn:step)].'
##            ind = ind(:)


        return T, index

        #% Old call: kept just in case
        #%T  = x(index((start+dist):step:nn),1)-x(index(start:step:(nn-dist)),1)



    def reconstruct(self):
        pass
    def plot_wave(self, sym1='k.', ts=None, sym2='k+', nfig=None, nsub=None, 
                  stdev=None, vfact=3):
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
        stdev : real scalar
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
        # TODO: finish reconstruct
        nw = 20
        tn = self.args
        xn = self.data.ravel()
        indmiss = isnan(xn) # indices to missing points
        indg = where(1-indmiss)[0]
        if ts is None:
            tc_ix = findtc(xn[indg],0,'tw')[0]
            xn2 = xn[tc_ix]
            tn2 = tn[tc_ix] 
        else:
            xn2 = ts.data
            tn2 = ts.args 
        
        if stdev is None:
            stdev = xn[indg].std()
            
        if nsub is None:
            nsub = int(floor(len(xn2)/(2*nw)))+1 # about Nw mdc waves in each plot
        if nfig is None:
            nfig = int(ceil(nsub/6)) 
            nsub = min(6,int(ceil(nsub/nfig)))
        
        n = len(xn)
        Ns = int(floor(n/(nfig*nsub)))
        ind = r_[0:Ns]
        if all(xn>=0):
            vscale = [0, 2*stdev*vfact]
        else:
            vscale = array([-1, 1])*vfact*stdev
        
        
        XlblTxt = 'Time [sec]'
        dT = 1
        timespan = tn[ind[-1]]-tn[ind[0]] 
        if abs(timespan)>18000: # more than 5 hours
            dT = 1/(60*60)
            XlblTxt = 'Time (hours)'
        elif abs(timespan)>300:# more than 5 minutes
            dT = 1/60
            XlblTxt = 'Time (minutes)'   
         
        if np.max(abs(xn[indg]))>5*stdev:
            XlblTxt = XlblTxt +' (Spurious data since max > 5 std.)'
        
        plot = plotbackend.plot
        subplot = plotbackend.subplot
        figs = []
        for iz in xrange(nfig):
            figs.append(plotbackend.figure())
            plotbackend.title('Surface elevation from mean water level (MWL).')
            for ix in xrange(nsub):
                if nsub>1:
                    subplot(nsub,1,ix)
                
                h_scale = array([tn[ind[0]], tn[ind[-1]]])
                ind2 = where((h_scale[0]<=tn2) & (tn2<=h_scale[1]))[0]
                plot(tn[ind]*dT, xn[ind], sym1)
                if len(ind2)>0: 
                    plot(tn2[ind2]*dT,xn2[ind2],sym2) 
                plot(h_scale*dT, [0, 0], 'k-')
                #plotbackend.axis([h_scale*dT, v_scale])
          
                for iy in [-2, 2]:
                    plot(h_scale*dT, iy*stdev*ones(2), ':')
              
                ind = ind + Ns
            #end
            plotbackend.xlabel(XlblTxt)
        
        return figs
        

    def plot_sp_wave(self, wave_idx_, tz_idx=None, *args, **kwds):
        """
        Plot specified wave(s) from timeseries

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
        if tz_idx is None:
            tc_ind, tz_idx = findtc(self.data,0,'tw') # finding trough to trough waves

        dw = nonzero(abs(diff(wave_idx))>1)[0]
        Nsub = dw.size+1
        Nwp = zeros(Nsub, dtype=int)
        if Nsub>1:
            dw = dw + 1
            Nwp[Nsub-1] = wave_idx[-1]-wave_idx[dw[-1]]+1
            wave_idx[dw[-1]+1:] = -2
            for ix in range(Nsub-2,1,-2):
                Nwp[ix] = wave_idx[dw[ix]-1] - wave_idx[dw[ix-1]]+1 # # of waves pr subplot
                wave_idx[dw[ix-1]+1:dw[ix]] = -2

            Nwp[0] = wave_idx[dw[0]-1] - wave_idx[0]+1
            wave_idx[1:dw[0]] = -2
            wave_idx = wave_idx[wave_idx>-1]
        else:
            Nwp[0] = wave_idx[-1]-wave_idx[0]+1
        #end

        Nsub = min(6,Nsub)
        Nfig = int(ceil(Nsub/6))
        Nsub = min(6,int(ceil(Nsub/Nfig)))
        figs = []
        for iy in range(Nfig):
            figs.append(plotbackend.figure())
            for ix in range(Nsub):
                plotbackend.subplot(Nsub,1,mod(ix,Nsub)+1)
                ind = r_[tz_idx[2*wave_idx[ix]-1]:tz_idx[2*wave_idx[ix]+2*Nwp[ix]-1]]
                ## indices to wave
                plotbackend.plot(self.args[ind],self.data[ind],*args,**kwds)
                plotbackend.hold('on')
                xi = [self.args[ind[0]],self.args[ind[-1]]]
                plotbackend.plot(xi,[0, 0])

                if Nwp[ix]==1:
                    plotbackend.ylabel('Wave %d' % wave_idx[ix])
                else:
                    plotbackend.ylabel('Wave %d - %d' % (wave_idx[ix], wave_idx[ix]+Nwp[ix]-1))

            plotbackend.xlabel('Time [sec]')
            #wafostamp
        return figs

def sensortypeid(*sensortypes):
    ''' Return ID for sensortype name

    Parameter
    ---------
    sensortypes : list of strings defining the sensortype

    Returns
    -------
    sensorids : list of integers defining the sensortype

    Valid senor-ids and -types for time series are as follows:
        0,  'n'    : Surface elevation              (n=Eta)
        1,  'n_t'  : Vertical surface velocity
        2,  'n_tt' : Vertical surface acceleration
        3,  'n_x'  : Surface slope in x-direction
        4,  'n_y'  : Surface slope in y-direction
        5,  'n_xx' : Surface curvature in x-direction
        6,  'n_yy' : Surface curvature in y-direction
        7,  'n_xy' : Surface curvature in xy-direction
        8,  'P'    : Pressure fluctuation about static MWL pressure
        9,  'U'    : Water particle velocity in x-direction
        10, 'V'    : Water particle velocity in y-direction
        11, 'W'    : Water particle velocity in z-direction
        12, 'U_t'  : Water particle acceleration in x-direction
        13, 'V_t'  : Water particle acceleration in y-direction
        14, 'W_t'  : Water particle acceleration in z-direction
        15, 'X_p'  : Water particle displacement in x-direction from its mean position
        16, 'Y_p'  : Water particle displacement in y-direction from its mean position
        17, 'Z_p'  : Water particle displacement in z-direction from its mean position

    Example:
    >>> sensortypeid('W','v')
    [11, 10]
    >>> sensortypeid('rubbish')
    [1.#QNAN]

    See also 
    --------
    sensortype
    '''

    sensorid_table = dict(n=0, n_t=1, n_tt=2, n_x=3, n_y=4, n_xx=5,
        n_yy=6, n_xy=7, p=8, u=9, v=10, w=11, u_t=12,
        v_t=13, w_t=14, x_p=15, y_p=16, z_p=17)
    try:
        return [sensorid_table.get(name.lower(), nan) for name in sensortypes]
    except:
        raise ValueError('Input must be a string!')



def sensortype(*sensorids):
    ''' 
    Return sensortype name

    Parameter
    ---------
    sensorids : vector or list of integers defining the sensortype

    Returns
    -------
    sensornames : tuple of strings defining the sensortype
        Valid senor-ids and -types for time series are as follows:
        0,  'n'    : Surface elevation              (n=Eta)
        1,  'n_t'  : Vertical surface velocity
        2,  'n_tt' : Vertical surface acceleration
        3,  'n_x'  : Surface slope in x-direction
        4,  'n_y'  : Surface slope in y-direction
        5,  'n_xx' : Surface curvature in x-direction
        6,  'n_yy' : Surface curvature in y-direction
        7,  'n_xy' : Surface curvature in xy-direction
        8,  'P'    : Pressure fluctuation about static MWL pressure
        9,  'U'    : Water particle velocity in x-direction
        10, 'V'    : Water particle velocity in y-direction
        11, 'W'    : Water particle velocity in z-direction
        12, 'U_t'  : Water particle acceleration in x-direction
        13, 'V_t'  : Water particle acceleration in y-direction
        14, 'W_t'  : Water particle acceleration in z-direction
        15, 'X_p'  : Water particle displacement in x-direction from its mean position
        16, 'Y_p'  : Water particle displacement in y-direction from its mean position
        17, 'Z_p'  : Water particle displacement in z-direction from its mean position

    Example:
    >>> sensortype(range(3))
    ('n', 'n_t', 'n_tt')

    See also 
    --------
    sensortypeid, tran
    '''
    valid_names = ('n', 'n_t', 'n_tt', 'n_x', 'n_y', 'n_xx', 'n_yy', 'n_xy',
                  'p', 'u', 'v', 'w', 'u_t', 'v_t', 'w_t', 'x_p', 'y_p', 'z_p',
                  nan)
    ids = atleast_1d(*sensorids)
    if isinstance(ids, list):
        ids = hstack(ids)
    n = len(valid_names) - 1
    ids = where(((ids<0) | (n<ids)), n , ids)
    
    #try:
    return tuple(valid_names[i] for i in ids)
    #except:
    #    raise ValueError('Input must be an integer!')

def main0():
    import wafo
    ts = wafo.objects.mat2timeseries(wafo.data.sea())
    tp = ts.turning_points()
    mm = tp.cycle_pairs()
    lc = mm.level_crossings()
    lc.plot()
    T = ts.wave_periods(vh=0.0,pdef='c2c')
    
    
   
    #main()
    import wafo.spectrum.models as sm
    Sj = sm.Jonswap()
    S = Sj.tospecdata()

    R = S.tocovdata()
    x = R.sim(ns=1000,dt=0.2)
    S.characteristic(['hm0','tm02'])
    ns = 1000 
    dt = .2
    x1 = S.sim(ns,dt=dt)

    ts = TimeSeries(x1[:,1],x1[:,0])
    tp = ts.turning_points(0.0)

    x = np.arange(-2,2,0.2)

    # Plot 2 objects in one call
    d2 = WafoData(np.sin(x),x,xlab='x',ylab='sin',title='sinus')


    d0 = d2.copy()
    d0.data = d0.data*0.9
    d1 = d2.copy()
    d1.data = d1.data*1.2
    d1.children = [d0]
    d2.children = [d1]

    d2.plot()
    print 'Done'

def main():
    sensortype(range(21))
    
if __name__ == '__main__':
    if  True: #False : #  
        import doctest
        doctest.testmod()
    else:
        main()
       