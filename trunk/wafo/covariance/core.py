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

from __future__ import division
import warnings
#import numpy as np
from numpy import (zeros, sqrt, dot, newaxis, inf, where, pi, nan, #@UnresolvedImport
                   atleast_1d, hstack, vstack, r_, linspace, flatnonzero, size, #@UnresolvedImport
                   isnan, finfo, diag, ceil, floor, random) #@UnresolvedImport
from numpy.fft import fft
from numpy.random import randn
import scipy.interpolate as interpolate
from scipy.linalg import toeplitz, sqrtm, svd, cholesky, diagsvd, pinv
from scipy import sparse
from pylab import stineman_interp

from wafo.wafodata import WafoData
from wafo.misc import sub_dict_select, nextpow2 #, JITImport
import wafo.spectrum as _wafospec
#_wafospec = JITImport('wafo.spectrum')


__all__ = ['CovData1D']

def _set_seed(iseed):
    if iseed != None:
        try:
            random.set_state(iseed)
        except:
            random.seed(iseed)


#def rndnormnd(cov, mean=0.0, cases=1, method='svd'):
#    '''
#    Random vectors from a multivariate Normal distribution
#    
#    Parameters
#    ----------
#    mean, cov : array-like
#         mean and covariance, respectively.
#    cases : scalar integer
#        number of sample vectors
#    method : string
#        defining squareroot method for covariance
#        'svd' : Singular value decomp.  (stable, quite fast) (default)
#        'chol' : Cholesky decomposition (fast, but unstable) 
#        'sqrtm' :  sqrtm                (stable and slow) 
#     
#    Returns
#    -------
#    r : matrix of random numbers from the multivariate normal
#        distribution with the given mean and covariance matrix.
#        
#    The covariance must be a symmetric, semi-positive definite matrix with shape
#    equal to the size of the mean. METHOD used for calculating the square root 
#    of COV is either svd, cholesky or sqrtm. (cholesky is fastest but least accurate.)
#    When cholesky is chosen and S is not positive definite, the svd-method 
#    is used instead.
#    
#    Example
#    -------
#    mu = [0, 5]
#    S = [[1 0.45], [0.45 0.25]]
#    r = rndnormnd(S, mu, 1)
#    plot(r(:,1),r(:,2),'.')
#    
#       d = 40; rho = 2*rand(1,d)-1;
#       mu = zeros(0,d);
#       S = (rho.'*rho-diag(rho.^2))+eye(d);
#       r = rndnormnd(S,mu,100,'genchol')'; 
#    
#    See also
#    --------
#    chol, svd, sqrtm, genchol
#    '''
#    sa = np.atleast_2d(cov)
#    mu = np.atleast_1d(mean).ravel() 
#    m, n = sa.shape
#    if m != n:
#        raise ValueError('Covariance must be square')
#    def svdfun(sa):
#        u, s, vh = svd(sa, full_matrices=False)
#        sqt = diagsvd(sqrt(s))
#        return dot(u, dot(sqt, vh))
#     
#    sqrtfuns = dict(sqrtm=sqrtm, svd=svdfun, cholesky=cholesky)
#    sqrtfun = sqrtfuns[method] 
#    std = sqrtfun(sa)
#    return dot(std,random.randn(n, cases)) + mu[:,newaxis]


class CovData1D(WafoData):
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
    >>> Sj = sp.models.Jonswap(Hm0=3)
    >>> w = np.linspace(0,4,256)
    >>> S = sp.SpecData1D(Sj(w),w) #Make spectrum object from numerical values

    See also
    --------
    WafoData
    CovData
    """

    def __init__(self,*args,**kwds):
        super(CovData1D, self).__init__(*args,**kwds)

        self.name = 'WAFO Covariance Object'
        self.type = 'time'
        self.lagtype = 't'
        self.h = inf
        self.tr = None
        self.phi = 0.
        self.v = 0.
        self.norm = 0
        somekeys = ['phi', 'name', 'h', 'tr', 'lagtype', 'v', 'type', 'norm']

        self.__dict__.update(sub_dict_select(kwds,somekeys))

        self.setlabels()
    def setlabels(self):
        ''' Set automatic title, x-,y- and z- labels

            based on type,
        '''

        N = len(self.type)
        if N==0:
            raise ValueError('Object does not appear to be initialized, it is empty!')

        labels = ['','ACF','']

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



##    def copy(self):
##        kwds = self.__dict__.copy()
##        wdata = CovData1D(**kwds)
##        return wdata

    def tospecdata(self, rate=None, method='linear', nugget=0.0, trunc=1e-5, fast=True):
        '''
        Computes spectral density from the auto covariance function

        Parameters
        ----------
        rate = scalar, int
            1,2,4,8...2^r, interpolation rate for f (default 1)

        method: string
            interpolation method 'stineman', 'linear', 'cubic'

        nugget = scalar, real
            nugget effect to ensure that round off errors do not result in
            negative spectral estimates. Good choice might be 10^-12.

        trunc : scalar, real
            truncates all spectral values where S/max(S) < trunc
                      0 <= trunc <1   This is to ensure that high frequency
                      noise is not added to the spectrum.  (default 1e-5)
        fast  : bool
             if True : zero-pad to obtain power of 2 length ACF (default)
             otherwise  no zero-padding of ACF, slower but more accurate.

        Returns
        --------
        S = SpecData1D object
            spectral density

         NB! This routine requires that the covariance is evenly spaced
             starting from zero lag. Currently only capable of 1D matrices.

        Example:
        >>> import wafo.spectrum.models as sm
        >>> import numpy as np
        >>> import scipy.signal.signaltools as st
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
        >>> assert(all(abs(S1.data-S.data)<1e-4) ,'COV2SPEC')

        See also
        --------
        spec2cov
        datastructures
        '''

        dT = self.sampling_period()
        # dT = time-step between data points.

        ACF, unused_ti = atleast_1d(self.data, self.args)

        if self.lagtype in 't':
            spectype = 'freq'
            ftype = 'w'
        else:
            spectype = 'k1d'
            ftype = 'k'

        if rate is None:
            rate = 1 ##interpolation rate
        else:
            rate = 2**nextpow2(rate) ##make sure rate is a power of 2


        ## add a nugget effect to ensure that round off errors
        ## do not result in negative spectral estimates
        ACF[0] = ACF[0] +nugget
        n = ACF.size
        # embedding a circulant vector and Fourier transform
        if fast:
            nfft = 2**nextpow2(2*n-2)
        else:
            nfft = 2*n-2

        nf   = nfft/2 ## number of frequencies
        ACF  = r_[ACF,zeros(nfft-2*n+2),ACF[n-1:0:-1]]

        Rper = (fft(ACF,nfft).real).clip(0) ## periodogram
        RperMax = Rper.max()
        Rper = where(Rper<trunc*RperMax,0,Rper)
        pi = pi
        S = abs(Rper[0:(nf+1)])*dT/pi
        w = linspace(0,pi/dT,nf+1)
        So = _wafospec.SpecData1D(S, w, type=spectype, freqtype=ftype)
        So.tr = self.tr
        So.h = self.h
        So.norm = self.norm

        if rate > 1:
            So.args = linspace(0, pi/dT, nf*rate)
            if method=='stineman':
                So.data = stineman_interp(So.args, w, S)
            else:
                intfun = interpolate.interp1d(w, S, kind=method)
                So.data = intfun(So.args)
            So.data = So.data.clip(0) # clip negative values to 0
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
        dt1 = self.args[1]-self.args[0]
        n = size(self.args)-1
        t = self.args[-1]-self.args[0]
        dt = t/n
        if abs(dt-dt1) > 1e-10:
            warnings.warn('Data is not uniformly sampled!')
        return dt

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
        applied to the simulated data, the result is a simulation of a transformed
        Gaussian process.

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
        nugget = 0 # 10**-12

        _set_seed(iseed)

        ACF = self.data.ravel()
        n = ACF.size


        I = ACF.argmax()
        if I != 0:
            raise ValueError('ACF does not have a maximum at zero lag')

        ACF.shape = (n, 1)

        dT = self.sampling_period()

        x = zeros((ns, cases+1))

        if derivative:
            xder = x.copy()

        ## add a nugget effect to ensure that round off errors
        ## do not result in negative spectral estimates
        ACF[0] = ACF[0] + nugget

        ## Fast and exact simulation of simulation of stationary
        ## Gaussian process throug circulant embedding of the
        ## Covariance matrix
        floatinfo = finfo(float)
        if (abs(ACF[-1]) > floatinfo.eps): ## assuming ACF(n+1)==0
            m2 = 2*n-1
            nfft = 2**nextpow2(max(m2, 2*ns))
            ACF = r_[ACF, zeros((nfft-m2,1)), ACF[-1:0:-1,:]]
            #disp('Warning: I am now assuming that ACF(k)=0 ')
            #disp('for k>MAXLAG.')
        else: # # ACF(n)==0
            m2 = 2*n-2
            nfft = 2**nextpow2(max(m2, 2*ns))
            ACF = r_[ACF, zeros((nfft-m2, 1)), ACF[n-1:1:-1, :]]

        ##m2=2*n-2
        S = fft(ACF,nfft,axis=0).real ## periodogram

        I = S.argmax()
        k = flatnonzero(S<0)
        if k.size>0:
            #disp('Warning: Not able to construct a nonnegative circulant ')
            #disp('vector from the ACF. Apply the parzen windowfunction ')
            #disp('to the ACF in order to avoid this.')
            #disp('The returned result is now only an approximation.')

            # truncating negative values to zero to ensure that
            # that this noise is not added to the simulated timeseries

            S[k] = 0.

            ix = flatnonzero(k>2*I)
            if ix.size>0:
##    # truncating all oscillating values above 2 times the peak
##    # frequency to zero to ensure that
##    # that high frequency noise is not added to
##    # the simulated timeseries.
                ix0 = k[ix[0]]
                S[ix0:-ix0] =0.0


       
        trunc = 1e-5
        maxS = S[I]
        k = flatnonzero(S[I:-I]<maxS*trunc)
        if k.size>0:
            S[k+I]=0.
            ## truncating small values to zero to ensure that
            ## that high frequency noise is not added to
            ## the simulated timeseries

        cases1 = floor(cases/2)
        cases2 = ceil(cases/2)
# Generate standard normal random numbers for the simulations

        #randn = np.random.randn
        epsi = randn(nfft,cases2)+1j*randn(nfft,cases2)
        Ssqr = sqrt(S/(nfft)) # #sqrt(S(wn)*dw )
        ephat = epsi*Ssqr #[:,np.newaxis]
        y = fft(ephat,nfft,axis=0)
        x[:, 1:cases+1] = hstack((y[2:ns+2, 0:cases2].real, y[2:ns+2, 0:cases1].imag))

        x[:, 0] = linspace(0,(ns-1)*dT,ns) ##(0:dT:(dT*(np-1)))'

        if derivative:
            Ssqr = Ssqr*r_[0:(nfft/2+1), -(nfft/2-1):0]*2*pi/nfft/dT
            ephat = epsi*Ssqr #[:,newaxis]
            y = fft(ephat,nfft,axis=0)
            xder[:, 1:(cases+1)] = hstack((y[2:ns+2, 0:cases2].imag -y[2:ns+2, 0:cases1].real))
            xder[:, 0] = x[:,0]

        if self.tr is not None:
            print('   Transforming data.')
            g = self.tr
            if derivative:
                for ix in range(cases):
                    tmp = g.gauss2dat(x[:,ix+1], xder[:,ix+1])
                    x[:,ix+1] = tmp[0]
                    xder[:,ix+1] = tmp[1]
            else:
                for ix in range(cases):
                    x[:, ix+1] = g.gauss2dat(x[:, ix+1])

        if derivative:
            return x, xder
        else:
            return x
        
    def simcond(self, xo, cases=1, method='approx', inds=None):
        """ 
        Simulate values conditionally on observed known values
        
        Parameters
        ----------
        x : array-like 
            datavector including missing data. 
            (missing data must be NaN if inds is not given)
            Assumption: The covariance of x is equal to self and have the 
            same sample period.
        cases : scalar integer
            number of cases, i.e., number of columns of sample (default=1)
        method : string
            defining method used in the conditional simulation. Options are:
            'approximate': Condition only on the closest points. Pros: quite fast
            'pseudo': Use pseudo inverse to calculate conditional covariance matrix
            'exact' : Exact simulation. Cons: Slow for large data sets, may not 
                    return any result due to near singularity of the covariance matrix.
        inds : integers 
            indices to spurious or missing data in x
            
        Returns
        -------
        sample : ndarray
            a random sample of the missing values conditioned on the observed data.
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
        Int. Journal of Offshore and Polar Engineering, Vol 11, No. 1, pp 23--32 
        
        Brodtkorb, P, Myrhaug, D, and Rue, H (1999)
        "Joint distribution of wave height and wave crest velocity from
        reconstructed data"
        in Proceedings of 9th ISOPE Conference, Vol III, pp 66-73
        """     
        # TODO does not work yet.
        
        # secret methods:
        #         'dec1-3': different decomposing algorithm's 
        #                   which is only correct for a variables
        #                   having the Markov property 
        #                   Cons:  3 is not correct at all, but seems to give
        #                          a reasonable result 
        #                   Pros: 1 is slow, 2 is quite fast and 3 is very fast
        #                   Note: (mu1oStd is not given for method ='dec3')
        compute_sigma = True
        x = atleast_1d(xo).ravel()
        acf = atleast_1d(self.data).ravel()
        
        N = len(x)
        n = len(acf)
        
        i = acf.argmax()
        if i != 0:
            raise ValueError('This is not a valid ACF!!')
        
        
        if not inds is None:
            x[inds] = nan
        inds = where(isnan(x))[0]  #indices to the unknown observations
        
        Ns = len(inds) # # missing values
        if Ns == 0:
            warnings.warn('No missing data, unable to continue.')
            return xo, zeros(Ns), zeros(Ns)
        #end
        if Ns == N:# simulated surface from the apriori distribution
            txt = '''All data missing,  
            returning sample from the unconditional distribution.'''
            warnings.warn(txt)
            return self.sim(ns=N, cases=cases), zeros(Ns), zeros(Ns)
        
        indg = where(1-isnan(x))[0] #indices to the known observations
        
        #initializing variables
        mu1o = zeros(Ns, 1)
        mu1o_std = mu1o
        sample = zeros((Ns, cases))
        if method[0] == 'd':
            # simulated surface from the apriori distribution
            xs = self.sim(ns=N, cases=cases) 
            mu1os = zeros((Ns, cases))
        
        if method.startswith('dec1'):
            # only correct for variables having the Markov property
            # but still seems to give a reasonable answer. Slow procedure.
            Sigma = sptoeplitz(hstack((acf, zeros(N-n))))
         
            #Soo=Sigma(~inds,~inds); # covariance between known observations
            #S11=Sigma(inds,inds); # covariance between unknown observations
            #S1o=Sigma(inds,~inds);# covariance between known and unknown observations
            #tmp=S1o*pinv(full(Soo)); 
            #tmp=S1o/Soo; # this is time consuming if Soo large
            tmp = 2*Sigma[inds, indg]/(Sigma[indg, indg] + Sigma[indg, indg].T )
            
            if compute_sigma:
                #standard deviation of the expected surface
                #mu1o_std=sqrt(diag(S11-tmp*S1o'));
                mu1o_std = sqrt(diag(Sigma[inds, inds]-tmp*Sigma[indg, inds]))
            
            
            #expected surface conditioned on the known observations from x
            mu1o = tmp*x[indg]
            #expected surface conditioned on the known observations from xs
            mu1os = tmp*(xs[indg,:])
            # sampled surface conditioned on the known observations
            sample = mu1o + xs[inds,:] - mu1os 
            
        elif method.startswith('dec2'):
            # only correct for variables having the Markov property
            # but still seems to give a reasonable answer
            # approximating the expected surfaces conditioned on 
            # the known observations from x and xs by only using the closest points
            Sigma = sptoeplitz(hstack((acf,zeros(n))))
            n2 = int(floor(n/2))
            idx = r_[0:2*n] + max(0,inds[0]-n2) # indices to the points used
            tmpinds = zeros(N,dtype=bool)
            tmpinds[inds] = True # temporary storage of indices to missing points
            tinds = where(tmpinds[idx])[0] # indices to the points used
            tindg = where(1-tmpinds[idx])[0]
            ns = len(tinds); # number of missing data in the interval
            nprev = 0; # number of previously simulated points
            xsinds = xs[inds,:]
            while ns>0:
                tmp=2*Sigma[tinds, tindg]/(Sigma[tindg, tindg]+Sigma[tindg, tindg].T)
                if compute_sigma:
                    #standard deviation of the expected surface
                    #mu1o_std=sqrt(diag(S11-tmp*S1o'));
                    ix = slice(nprev+1,nprev+ns+1)
                    mu1o_std[ix] = max(mu1o_std[ix], 
                            sqrt(diag(Sigma[tinds, tinds]-tmp*Sigma[tindg,tinds])))
                #end
              
                #expected surface conditioned on the closest known observations
                # from x and xs2
                mu1o[(nprev+1):(nprev+ns+1)] = tmp*x[idx[tindg]]
                mu1os[(nprev+1):(nprev+ns+1),:] = tmp*xs[idx[tindg],:]      
               
                if idx[-1]==N-1:# 
                    ns =0  # no more points to simulate
                else:
                    # updating by  putting expected surface into x     
                    x[idx[tinds]] = mu1o[(nprev+1):(nprev+ns+1)]
                    xs[idx[tinds]] = mu1os[(nprev+1):(nprev+ns+1)]
        
                    nw = sum(tmpinds[idx[-n2:]])# # data which we want to simulate once 
                    tmpinds[idx[:-n2]] = False # removing indices to data ..
                    # which has been simulated
                    nprev = nprev+ns-nw # update # points simulated so far
                              
                    if (nw==0) and (nprev<Ns): 
                        idx= r_[0:2*n]+(inds[nprev+1]-n2) # move to the next missing data
                    else:
                        idx = idx+n
                    #end
                    tmp = N-idx[-1]
                    if tmp<0: # checking if tmp exceeds the limits
                        idx = idx+tmp
                    #end
                    # find new interval with missing data
                    tinds = where(tmpinds[idx])[0]
                    tindg = where(1-tmpinds[idx])[0]
                    ns = len(tinds);# # missing data
                #end  
            #end
            # sampled surface conditioned on the known observations
            sample = mu1o+(xsinds-mu1os) 
        elif method.startswith('dec3'): 
            # this is not correct for even for variables having the 
            # Markov property but still seems to give a reasonable answer
            # a quasi approach approximating the expected surfaces conditioned on 
            # the known observations from x and xs with a spline
            
            mu1o = interp1(indg, x[indg],inds,'spline')
            mu1os = interp1(indg, xs[indg,:],inds,'spline')
            # sampled surface conditioned on the known observations
            sample = mu1o + (xs[inds,:]-mu1os) 
        
        elif method.startswith('exac') or method.startswith('pseu'):
            # exact but slow. It also may not return any result
            Sigma = sptoeplitz(hstack((acf,zeros(N-n))))
            #Soo=Sigma(~inds,~inds); # covariance between known observations
            #S11=Sigma(inds,inds); # covariance between unknown observations
            #S1o=Sigma(inds,~inds);# covariance between known and unknown observations
            #tmp=S1o/Soo; # this is time consuming if Soo large
            if method[0]=='e': #exact
                tmp = 2*Sigma[inds,indg]/(Sigma[indg,indg]+Sigma[indg,indg].T);
            else: # approximate the inverse with pseudo inverse
                tmp = dot(Sigma[inds, indg],pinv(Sigma[indg,indg]))
            #end
            #expected surface conditioned on the known observations from x
            mu1o = dot(tmp,x[indg])
            # Covariance conditioned on the known observations
            Sigma1o = Sigma[inds,inds] - tmp*Sigma[indg,inds]
            #sample conditioned on the known observations from x
            sample = random.multivariate_normal(mu1o, Sigma1o, cases)
            #rndnormnd(mu1o,Sigma1o,cases )
            
            if compute_sigma:
                #standard deviation of the expected surface
                mu1o_std=sqrt(diag(Sigma1o));
            #end
 
        elif method.startswith('appr'):
            # approximating by only  condition on 
            # the closest points
            # checking approximately how many lags we need in order to 
            # ensure conditional independence
            # using that the inverse of the circulant covariance matrix has 
            # approximately the same bandstructure as the inverse of the
            # covariance matrix
            
            Nsig = 2*n;
            
            Sigma = sptoeplitz(hstack((ACF,zeros(Nsig-n))))
            n2 = floor(Nsig/4)
            idx = r_[0:Nsig]+max(0,inds[0]-n2) # indices to the points used
            tmpinds = zeros(N,dtype=bool)
            tmpinds[inds] = True # temporary storage of indices to missing points
            tinds = where(tmpinds[idx])[0] # indices to the points used
            tindg = where(1-tmpinds[idx])[0]
            ns = len(tinds) # number of missing data in the interval
            
            nprev = 0  # number of previously simulated points
            x2 = x
            
            while ns>0:
                #make sure MATLAB uses a symmetric matrix solver
                tmp = 2*Sigma[tinds,tindg]/(Sigma[tindg,tindg]+Sigma[tindg,tindg].T)
                Sigma1o = Sigma[tinds,tinds] - tmp*Sigma[tindg,tinds]
                if compute_sigma:
                    #standard deviation of the expected surface
                    #mu1o_std=sqrt(diag(S11-tmp*S1o'));
                    mu1o_std[(nprev+1):(nprev+ns+1)] = max( mu1o_std[(nprev+1):(nprev+ns)] , 
                                                            sqrt(diag(Sigma1o)))
                #end
        
                #expected surface conditioned on the closest known observations from x
                mu1o[(nprev+1):(nprev+ns+1)] = tmp*x2[idx[tindg]]
                #sample conditioned on the known observations from x
                sample[(nprev+1):(nprev+ns+1),:] = rndnormnd(tmp*x[idx[tindg]],Sigma1o, cases)     
                if idx[-1] == N-1: 
                    ns = 0 # no more points to simulate
                else:
                    # updating
                    x2[idx[tinds]] = mu1o[(nprev+1):(nprev+ns+1)] #expected surface
                    x[idx[tinds]] = sample[(nprev+1):(nprev+ns+1)]#sampled surface
                    nw = sum(tmpinds[idx[-n2::]]==True)# # data we want to simulate once more
                    tmpinds[idx[:-n2]] = False # removing indices to data ..
                    # which has been simulated
                    nprev = nprev+ns-nw # update # points simulated so far
            
                    if (nw==0) and (nprev<Ns):
                        idx = r_[0:Nsig]+(inds[nprev+1]-n2) # move to the next missing data
                    else:
                        idx = idx+n
                    #end
                    tmp = N-idx[-1]
                    if tmp<0: # checking if tmp exceeds the limits
                        idx = idx + tmp
                    #end
                    # find new interval with missing data
                    tinds = where(tmpinds[idx])[0]
                    tindg = where(1-tmpinds[idx])[0]
                    ns = len(tinds);# # missing data in the interval
              #end
            #end
        #end
        return sample
#          plot(find(~inds),x(~inds),'.')
#          hold on,
#          ind=find(inds);
#          plot(ind,mu1o   ,'*')
#          plot(ind,sample,'r+')
#          #mu1o_std
#          plot(ind,[mu1o-2*mu1o_std mu1o+2*mu1o_std ] ,'d')
#          #plot(xs),plot(ind,mu1os,'r*')
#          hold off
#          legend('observed values','mu1o','sampled values','2 stdev')
#          #axis([770 850 -1 1])
#          #axis([1300 1325 -1 1])
        
def sptoeplitz(x):
    k = where(x.ravel())[0]
    n = len(x)
    if len(k)>0.3*n:
        return toeplitz(x)
    else:
        spdiags = sparse.dia_matrix
        data = x[k].reshape(-1,1).repeat(n,axis=-1)
        offsets = k
        y = spdiags((data, offsets), shape=(n,n))
        if k[0]==0:
            offsets = k[1::]
            data = data[1::,:]
        return y + spdiags((data, -offsets), shape=(n,n))

def test_covdata():
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
    S = Sj.tospecdata()   #Make spec
    S.plot()
    R = S.tocovdata()
    R.plot()
    #x = R.sim(ns=1000,dt=0.2)


if __name__ == '__main__':
    if  True: #False : #  
        import doctest
        doctest.testmod()
    else:
        main()