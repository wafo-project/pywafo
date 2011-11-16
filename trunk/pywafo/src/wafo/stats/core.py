from __future__ import division
import warnings
from wafo.wafodata import WafoData
from wafo.misc import findextrema
from scipy import special
import numpy as np
from numpy import inf
from numpy import atleast_1d, nan, ndarray, sqrt, vstack, ones, where, zeros
from numpy import arange, floor, linspace, asarray #, reshape, repeat, product
from time import gmtime, strftime


__all__ = ['edf', 'edfcnd','reslife', 'dispersion_idx','decluster','findpot', 
           'declustering_time','interexceedance_times', 'extremal_idx']

arr = asarray

def now():
    '''
    Return current date and time as a string
    '''
    return strftime("%a, %d %b %Y %H:%M:%S", gmtime())

def valarray(shape, value=nan, typecode=None):
    """Return an array of all value.
    """
    #out = reshape(repeat([value], product(shape, axis=0), axis=0), shape)
    out = ones(shape, dtype=bool) * value
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, ndarray):
        out = arr(out)
    return out 
def _cdff(self, x, dfn, dfd):
    return special.fdtr(dfn, dfd, x)
def _cdft(x,df):
    return special.stdtr(df, x)
def _invt(q, df):
    return special.stdtrit(df, q)
def _cdfchi2(x, df):
    return special.chdtr(df, x)
def _invchi2(q, df):
    return special.chdtri(df, q)
def _cdfnorm(x):
    return special.ndtr(x)
def _invnorm(q):
    return special.ndtri(q)

def edf(x, method=2):
    ''' 
    Returns Empirical Distribution Function (EDF).
    
    Parameters
    ----------
    x : array-like
        data vector
    method : integer scalar 
        1. Interpolation so that F(X_(k)) == (k-0.5)/n.
        2. Interpolation so that F(X_(k)) == k/(n+1).    (default)
        3. The empirical distribution. F(X_(k)) = k/n
     
    Example
    -------
    >>> import wafo.stats as ws
    >>> x = np.linspace(0,6,200)
    >>> R = ws.rayleigh.rvs(scale=2,size=100)
    >>> F = ws.edf(R)
    >>> h = F.plot()
      
     See also edf, pdfplot, cumtrapz
    '''
    z = atleast_1d(x)       
    z.sort()
    
    N = len(z)
    if method == 1:
        Fz1 = arange(0.5, N) / N
    elif method == 3:
        Fz1 = arange(1, N + 1) / N
    else:
        Fz1 = arange(1, N + 1) / (N + 1)
     
    F = WafoData(Fz1, z, xlab='x', ylab='F(x)')
    F.setplotter('step')
    return F

def edfcnd(x, c=None, method=2):
    ''' 
    Returns empirical Distribution Function CoNDitioned that X>=c (EDFCND).
    
    Parameters
    ----------
    x : array-like
        data vector
    method : integer scalar 
        1. Interpolation so that F(X_(k)) == (k-0.5)/n.
        2. Interpolation so that F(X_(k)) == k/(n+1).    (default)
        3. The empirical distribution. F(X_(k)) = k/n
     
    Example
    -------
    >>> import wafo.stats as ws
    >>> x = np.linspace(0,6,200)
    >>> R = ws.rayleigh.rvs(scale=2,size=100)
    >>> Fc = ws.edfcnd(R, 1)
    >>> hc = Fc.plot()
    >>> F = ws.edf(R)
    >>> h = F.plot()
    
     See also edf, pdfplot, cumtrapz
    '''
    z = atleast_1d(x)
    if c is None:
        c = floor(min(z.min(), 0))
    
    try:
        F = edf(z[c <= z], method=method)
    except:
        ValueError('No data points above c=%d' % int(c)) 
    
    if - inf < c:
        F.labels.ylab = 'F(x| X>=%g)' % c
    
    return F


def reslife(data, u=None, umin=None, umax=None, nu=None, nmin=3, alpha=0.05, plotflag=False):
    ''' 
    Return Mean Residual Life, i.e., mean excesses vs thresholds
    
    Parameters
    ---------
    data : array_like
        vector of data of length N. 
    u :  array-like
        threshold values (default linspace(umin, umax, nu))
    umin, umax : real scalars
        Minimum and maximum threshold, respectively (default min(data), max(data)).
    nu : scalar integer
        number of threshold values (default min(N-nmin,100))
    nmin : scalar integer
        Minimum number of extremes to include. (Default 3).
    alpha : real scalar
        Confidence coefficient (default 0.05)        
    plotflag: bool
        
    
    Returns
    -------
    mrl : WafoData object
        Mean residual life values, i.e., mean excesses over thresholds, u.
    
    Notes
    -----
    RESLIFE estimate mean excesses over thresholds. The purpose of MRL is
    to determine the threshold where the upper tail of the data can be 
    approximated with the generalized Pareto distribution (GPD). The GPD is
    appropriate for the tail, if the MRL is a linear function of the 
    threshold, u. Theoretically in the GPD model 
    
        E(X-u0|X>u0) = s0/(1+k)
        E(X-u |X>u)  = s/(1+k) = (s0 -k*u)/(1+k)   for u>u0
    
    where k,s is the shape and scale parameter, respectively.
    s0 = scale parameter for threshold u0<u.
    
    Example
    -------
    >>> import wafo
    >>> R = wafo.stats.genpareto.rvs(0.1,2,2,size=100)
    >>> mrl = reslife(R,nu=20)
    >>> h = mrl.plot() 
      
    See also
    ---------
    genpareto
    fitgenparrange, disprsnidx
    '''
    if u is None:
        sd = np.sort(data)
        n = len(data)
        
        nmin = max(nmin, 0)
        if 2 * nmin > n:
            warnings.warn('nmin possibly too large!')
        
        sdmax, sdmin = sd[-nmin], sd[0]
        umax = sdmax if umax is None else min(umax, sdmax)
        umin = sdmin if umin is None else max(umin, sdmin)
        
        if nu is None:
            nu = min(n - nmin, 100)
        
        u = linspace(umin, umax, nu)
    
    
    nu = len(u)
    
    #mrl1 = valarray(nu) 
    #srl = valarray(nu)
    #num = valarray(nu)
    
    mean_and_std = lambda data1 : (data1.mean(), data1.std(), data1.size)
    dat = arr(data)
    tmp = arr([mean_and_std(dat[dat > tresh] - tresh) for tresh in u.tolist()])
    
    mrl, srl, num = tmp.T
    p = 1 - alpha
    alpha2 = alpha / 2
    
    # Approximate P% confidence interval
    #%Za = -invnorm(alpha2);   % known mean
    Za = -_invt(alpha2, num - 1) # unknown mean
    mrlu = mrl + Za * srl / sqrt(num)
    mrll = mrl - Za * srl / sqrt(num)
    
    #options.CI = [mrll,mrlu];
    #options.numdata = num;
    titleTxt = 'Mean residual life with %d%s CI' % (100 * p, '%')
    res = WafoData(mrl, u, xlab='Threshold', ylab='Mean Excess', title=titleTxt)
    res.workspace = dict(numdata=num, umin=umin, umax=umax, nu=nu, nmin=nmin, alpha=alpha)
    res.children = [WafoData(vstack([mrll, mrlu]).T, u, xlab='Threshold', title=titleTxt)]
    res.plot_args_children = [':r']
    if plotflag:
        res.plot()
    return res

def dispersion_idx(data, t=None, u=None, umin=None, umax=None, nu=None, nmin=10, tb=1,
               alpha=0.05, plotflag=False):
    '''Return Dispersion Index vs threshold
    
    Parameters
    ----------
    data, ti : array_like
        data values and sampled times, respectively.
    u :  array-like
        threshold values (default linspace(umin, umax, nu))
    umin, umax : real scalars
        Minimum and maximum threshold, respectively (default min(data), max(data)).
    nu : scalar integer
        number of threshold values (default min(N-nmin,100))
    nmin : scalar integer
        Minimum number of extremes to include. (Default 10).
    tb : Real scalar
        Block period (same unit as the sampled times)  (default 1)
    alpha : real scalar
        Confidence coefficient (default 0.05)        
    plotflag: bool
    
    Returns
    -------
    DI : WafoData object
        Dispersion index
    b_u : real scalar
        threshold where the number of exceedances in a fixed period (Tb) is
        consistent with a Poisson process.
    ok_u : array-like
        all thresholds where the number of exceedances in a fixed period (Tb) is
        consistent with a Poisson process.
    Notes
    ------
    DISPRSNIDX estimate the Dispersion Index (DI) as function of threshold. 
    DI measures the homogenity of data and the purpose of DI is to determine 
    the threshold where the number of exceedances in a fixed period (Tb) is
    consistent with a Poisson process. For a Poisson process the DI is one. 
    Thus the threshold should be so high that DI is not significantly
    different from 1.

    The Poisson hypothesis is not rejected if the estimated DI is between:

    chi2(alpha/2, M-1)/(M-1)< DI < chi^2(1 - alpha/2, M-1 }/(M - 1)

    where M is the total number of fixed periods/blocks -generally
    the total number of years in the sample.
    
    Example
    -------
    >>> import wafo.data
    >>> xn = wafo.data.sea()
    >>> t, data = xn.T
    >>> Ie = findpot(data,t,0,5);
    >>> di, u, ok_u = dispersion_idx(data[Ie],t[Ie],tb=100)
    >>> h = di.plot() # a threshold around 1 seems appropriate.
    
    vline(u)
    
    See also
    --------
    reslife, 
    fitgenparrange, 
    extremal_idx
    
    
    References
    ----------
    Ribatet, M. A.,(2006),
    A User's Guide to the POT Package (Version 1.0)
    month = {August},
    url = {http://cran.r-project.org/}
    
    Cunnane, C. (1979) Note on the poisson assumption in
    partial duration series model. Water Resource Research, 15\bold{(2)}
         :489--494.}
    '''
    
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful, but without any warranty; without even
# the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public
# License for moredetails.
# The GNU General Public License can be obtained from http://www.gnu.org/copyleft/gpl.html. You
# can also obtain it by writing to the Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
#  MA 02111-1307, USA.
    
    
    
    n = len(data)
    if t is None:
        ti = arange(n)
    else:
        ti = arr(t) - min(t)
    
    t1 = np.empty(ti.shape,dtype=int)
    t1[:] = np.floor(ti / tb)
    
    
    if u is None:
        sd = np.sort(data)
       
        
        nmin = max(nmin, 0)
        if 2 * nmin > n:
            warnings.warn('nmin possibly too large!')
        
        sdmax, sdmin = sd[-nmin], sd[0]
        umax = sdmax if umax is None else min(umax, sdmax)
        umin = sdmin if umin is None else max(umin, sdmin)
        
        if nu is None:
            nu = min(n - nmin, 100)
        
        u = linspace(umin, umax, nu)
    
   
    
    nu = len(u)
   
    di = np.zeros(nu)
    
    d = arr(data)
    
    mint = int(min(t1)) #; % mint should be 0.
    maxt = int(max(t1))
    M = maxt - mint + 1;
    occ = np.zeros(M);
    
    for ix, tresh in enumerate(u.tolist()):
        excess = (d > tresh)
        lambda_ = excess.sum() / M
        for block in  range(M):
            occ[block] = sum(excess[t1 == block])
      
        di[ix] = occ.var() / lambda_
    
    p = 1 - alpha
   
    diUp = _invchi2(1 - alpha / 2, M - 1) / (M - 1)
    diLo = _invchi2(alpha / 2, M - 1) / (M - 1)
    
    # Find appropriate threshold
    k1, = np.where((diLo < di) & (di < diUp))
    if len(k1) > 0:
        ok_u = u[k1]
        b_di = (di[k1].mean() < di[k1])
        k = b_di.argmax()
        b_u = ok_u[k]
    else:
        b_u = ok_u = None
      
    CItxt = '%d%s CI' % (100 * p, '%')
    titleTxt = 'Dispersion Index plot';
    
    res = WafoData(di, u, title=titleTxt, labx='Threshold', laby='Dispersion Index')
        #'caption',CItxt);
    res.workspace = dict(umin=umin, umax=umax, nu=nu, nmin=nmin, alpha=alpha)
    res.children = [WafoData(vstack([diLo * ones(nu), diUp * ones(nu)]).T, u, xlab='Threshold', title=CItxt)]
    res.plot_args_children = ['--r']
    if plotflag:
        res.plot(di)
    return res, b_u, ok_u

def decluster(data, t=None, thresh=None, tmin=1):   
    '''
    Return declustered peaks over threshold values
    
    Parameters
    ----------
    data, t : array-like     
        data-values and sampling-times, respectively.
    thresh : real scalar
        minimum threshold for levels in data.
    tmin : real scalar
        minimum distance to another peak [same unit as t] (default 1)
    
    Returns
    -------
    ev, te : ndarray
        extreme values and its corresponding sampling times, respectively, i.e., 
        all data > thresh which are at least tmin distance apart.
      
    Example
    -------
    >>> import pylab
    >>> import wafo.data
    >>> from wafo.misc import findtc
    >>> x  = wafo.data.sea()
    >>> t, data = x[:400,:].T
    >>> itc, iv = findtc(data,0,'dw')
    >>> ytc, ttc = data[itc], t[itc]
    >>> ymin = 2*data.std()
    >>> tmin = 10 # sec 
    >>> [ye, te] = decluster(ytc,ttc, ymin,tmin);
    >>> h = pylab.plot(t,data,ttc,ytc,'ro',t,zeros(len(t)),':',te,ye,'k.')
    
    See also
    --------
    fitgenpar, findpot, extremalidx
    '''
    if t is None:
        t = np.arange(len(data))
    i = findpot(data, t, thresh, tmin)
    return data[i], t[i]

def findpot(data, t=None, thresh=None, tmin=1):
    '''
    Retrun indices to Peaks over threshold values
    
    Parameters
    ----------
    data, t : array-like     
        data-values and sampling-times, respectively.
    thresh : real scalar
        minimum threshold for levels in data.
    tmin : real scalar
        minimum distance to another peak [same unit as t] (default 1)
        
    Returns
    -------
    Ie : ndarray
        indices to extreme values, i.e., all data > tresh which are at least 
        tmin distance apart.
    
    Example
    -------
    >>> import pylab
    >>> import wafo.data
    >>> from wafo.misc import findtc
    >>> x  = wafo.data.sea()
    >>> t, data = x.T
    >>> itc, iv = findtc(data,0,'dw')
    >>> ytc, ttc = data[itc], t[itc]
    >>> ymin = 2*data.std()
    >>> tmin = 10 # sec
    >>> I = findpot(data, t, ymin, tmin)
    >>> yp, tp = data[I], t[I]
    >>> Ie = findpot(yp, tp, ymin,tmin)
    >>> ye, te = yp[Ie], tp[Ie]
    >>> h = pylab.plot(t,data,ttc,ytc,'ro',t,zeros(len(t)),':',te, ye,'k.',tp,yp,'+')
    
    See also
    --------
    fitgenpar, decluster, extremalidx
    '''
    Data = arr(data)
    if t is None:
        ti = np.arange(len(Data))
    else:
        ti = arr(t)
        
    Ie, = where(Data > thresh);
    Ye = Data[Ie]
    Te = ti[Ie]
    if len(Ye) <= 1:
        return Ie
    
    dT = np.diff(Te)
    notSorted = np.any(dT < 0);
    if notSorted:
        I = np.argsort(Te)
        Te = Te[I] 
        Ie = Ie[I]
        Ye = Ye[I]
        dT = np.diff(Te)
    
    isTooSmall = (dT <= tmin)
    
    if np.any(isTooSmall):
        isTooClose = np.hstack((isTooSmall[0], isTooSmall[:-1] | isTooSmall[1:], isTooSmall[-1]))
     
        #Find opening (NO) and closing (NC) index for data beeing to close:
        iy = findextrema(np.hstack([0, 0, isTooSmall, 0]))
    
        NO = iy[::2] - 1 
        NC = iy[1::2] 
      
        for no, nc in zip(NO, NC):
            iz = slice(no, nc)
            iOK = _find_ok_peaks(Ye[iz], Te[iz], tmin)
            if len(iOK):
                isTooClose[no + iOK] = 0
        # Remove data which is too close to other data.        
        if isTooClose.any():
            #len(tooClose)>0:
            iOK, = where(1 - isTooClose)
            Ie = Ie[iOK]

    return Ie  
    
    
def _find_ok_peaks(Ye, Te, Tmin):
    '''
    Return indices to the largest maxima that are at least Tmin
    distance apart.
    '''
    Ny = len(Ye)
    
    I = np.argsort(-Ye) #  sort in descending order
    
    Te1 = Te[I]
    oOrder = zeros(Ny, dtype=int)
    oOrder[I] = range(Ny) #indices to the variables original location
    
    isTooClose = zeros(Ny, dtype=bool)
      
    pool = zeros((Ny, 2))
    T_range = np.hstack([-Tmin, Tmin])
    K = 0
    for i, ti in enumerate(Te1):
        isTooClose[i] = np.any((pool[:K, 0] <= ti) & (ti <= pool[:K, 1]))
        if not isTooClose[i]:
            pool[K] = ti + T_range
            K += 1
      
    iOK, = where(1 - isTooClose[oOrder])
    return iOK

def declustering_time(t):
    '''
    Returns minimum distance between clusters.
    
    Parameters
    ----------
    t : array-like     
        sampling times for data.

    Returns
    -------
    tc : real scalar
        minimum distance between clusters.
    
    Example
    -------
    >>> import wafo.data
    >>> x  = wafo.data.sea()
    >>> t, data = x[:400,:].T
    >>> Ie = findpot(data,t,0,5);
    >>> tc = declustering_time(Ie) 
    >>> tc
    21
    
    '''   
    t0 = arr(t)
    nt = len(t0)
    if nt<2:
        return arr([])
    ti = interexceedance_times(t0)
    ei = extremal_idx(ti)
    if ei==1:
        tc = ti.min()
    else:
        i = int(np.floor(nt*ei))
        sti = -np.sort(-ti)
        tc = sti[min(i, nt-2)] #% declustering time
    return tc
    
    
def interexceedance_times(t):
    '''
    Returns interexceedance times of data
    
    Parameters
    ----------
    t : array-like     
        sampling times for data.
    Returns
    -------
    ti : ndarray
        interexceedance times
        
    Example
    -------
    >>> t = [1,2,5,10]
    >>> interexceedance_times(t)
    array([1, 3, 5])
    
    '''  
    return np.diff(np.sort(t))  

def extremal_idx(ti):
    '''
    Returns Extremal Index measuring the dependence of data
    
    Parameters
    ----------
    ti : array-like     
        interexceedance times for data.

    Returns
    -------
    ei : real scalar
        Extremal index.
   
    Notes
    ----- 
    The Extremal Index (EI) is  one if the data are independent and less than 
    one if there are some dependence. The extremal index can also be intepreted 
    as the reciprocal of the mean cluster size.
    
    Example
    -------
    >>> import wafo.data
    >>> x  = wafo.data.sea()
    >>> t, data = x[:400,:].T
    >>> Ie = findpot(data,t,0,5);
    >>> ti = interexceedance_times(Ie)
    >>> ei = extremal_idx(ti) 
    >>> ei
    1
    
    See also 
    --------
    reslife, fitgenparrange, disprsnidx, findpot, decluster
    
    
    Reference
    ---------
    Christopher A. T. Ferro, Johan Segers (2003)
    Inference for clusters of extreme values
    Journal of the Royal Statistical society: Series B (Statistical Methodology) 54 (2), 545-556
    doi:10.1111/1467-9868.00401
    '''
    t = arr(ti)
    tmax = t.max() 
    if tmax<=1:
        ei = 0
    elif tmax<=2:
        ei = min(1, 2*t.mean()**2/((t**2).mean()))
    else:
        ei = min(1, 2*np.mean(t-1)**2/np.mean((t-1)*(t-2)))
    return ei

def _logit(p):
    return np.log(p)-np.log1p(-p) 
def _logitinv(x):
    return 1.0/(np.exp(-x)+1)    
    
class RegLogit(object):
    '''
    REGLOGIT Fit ordinal logistic regression model.
    
      CALL model = reglogit (options)
    
        model = fitted model object with methods
          .compare() : Compare small LOGIT object versus large one
          .predict() : Predict from a fitted LOGIT object
          .summary() : Display summary of fitted LOGIT object.
    
           y = vector of K ordered categories
           x = column vectors of covariates 
     options = struct defining performance of REGLOGIT
          .maxiter    : maximum number of iterations. 
          .accuracy   : accuracy in convergence.
          .betastart  : Start value for BETA           (default 0)
          .thetastart : Start value for THETA          (default depends on Y)
          .alpha      : Confidence coefficent          (default 0.05)
          .verbose    : 1 display summary info about fitted model
                        2 display convergence info in each iteration
                          otherwise no action
          .deletecolinear : If true delete colinear covarites (default)
          

     Methods
      .predict   : Predict from a fitted LOGIT object
      .summary   : Display summary of fitted LOGIT object.
      .compare   : Compare small LOGIT versus large one
 
 

    
     Suppose Y takes values in K ordered categories, and let
     gamma_i (x) be the cumulative probability that Y
     falls in one of the first i categories given the covariate
     X.  The ordinal logistic regression model is
    
     logit (mu_i (x)) = theta_i + beta' * x,   i = 1...k-1
    
     The number of ordinal categories, K, is taken to be the number
     of distinct values of round (Y).  If K equals 2,
     Y is binary and the model is ordinary logistic regression.  The
     matrix X is assumed to have full column rank.
    
     Given Y only, theta = REGLOGIT(Y) fits the model with baseline logit odds
     only. 
     
     Example
      y=[1 1 2 1 3 2 3 2 3 3]'
      x = (1:10)'
      b = reglogit(y,x)
      b.display() % members and methods
      b.get()     % return members
      b.summary()
      [mu,plo,pup] = b.predict();
      plot(x,mu,'g',x,plo,'r:',x,pup,'r:')
    
      y2 = [zeros(5,1);ones(5,1)];
      x1 = [29,30,31,31,32,29,30,31,32,33];
      x2 = [62,83,74,88,68,41,44,21,50,33];
      X = [x1;x2].';
      b2 = reglogit(y2,X);
      b2.summary();
      b21 = reglogit(y2,X(:,1));
      b21.compare(b2)
    
     See also regglm, reglm, regnonlm
    '''
    
    #% Copyright (C) 1995, 1996, 1997, 1998, 1999, 2000, 2002, 2005, 2007
    #%               Kurt Hornik
    #%
    #% Reglogit is free software; you can redistribute it and/or modify it
    #% under the terms of the GNU General Public License as published by
    #% the Free Software Foundation; either version 3 of the License, or (at
    #% your option) any later version.
    #%
    #% Reglogit is distributed in the hope that it will be useful, but
    #% WITHOUT ANY WARRANTY; without even the implied warranty of
    #% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    #% General Public License for more details.
    #%
    #% You should have received a copy of the GNU General Public License
    #% along with Reglogit; see the file COPYING.  If not, see
    #% <http://www.gnu.org/licenses/>.
    
    
    
    #% Original for MATLAB written by Gordon K Smyth <gks@maths.uq.oz.au>,
    #% U of Queensland, Australia, on Nov 19, 1990.  Last revision Aug 3,
    #% 1992.
    #
    #% Author: Gordon K Smyth <gks@maths.uq.oz.au>,
    #% Adapted-By: KH <Kurt.Hornik@wu-wien.ac.at>
    #% Revised by: pab
    #%  -renamed from  logistic_regression to reglogit
    #%  -added predict, summary and compare
    #% Description: Ordinal logistic regression
    #
    #% Uses the auxiliary functions logistic_regression_derivatives and
    #% logistic_regression_likelihood.
    
    def __init__(self, maxiter=500,accuracy=1e-6, alpha=0.05, deletecolinear=True, verbose=False):
         
        self.maxiter =maxiter
        self.accuracy = accuracy
        self.alpha = alpha
        self.deletcolinear =  deletecolinear
        self.verbose = False
        self.family = None
        self.link = None
        self.numvar = None
        self.numobs = None 
        self.numk = None
        self.df = None
        self.df_null = None
        self.params = None
        self.params_ci = None
        self.params_cov = None
        self.params_std = None
        self.params_corr = None
        self.params_tstat = None
        self.params_pvalue = None
        self.mu = None
        self.eta = None
        self.X = None
        self.Y = None
        self.theta = None
        self.beta = None
        self.residual = None
        self.residual1d = None
        self.deviance = None
        self.deviance_null = None
        self.d2L = None
        self.dL = None
        self.dispersionfit = None
        self.dispersion = 1
        self.R2 = None
        self.R2adj = None
        self.numiter = None
        self.converged = None
        self.note = ''
        self.date = now()
      
    def check_xy(self, y, X):    
        y = np.round(np.atleast_2d(y))
        my = y.shape[0] 
        if X is None: 
            X = np.zeros((my, 0))
        elif self.deletecolinear:
            X = np.atleast_2d(X)
            # Make sure X is full rank
            s = np.linalg.svd(X)[1]
            tol = max(X.shape) * np.finfo(s.max()).eps
            ix = np.flatnonzero(s>tol)
            iy = np.flatnonzero(s<=tol)
            if len(ix): 
                X = X[:, ix]
                txt = [' %d,' % i for i in iy]
                txt[-1] = ' %d' % iy[-1]
                warnings.warn('Covariate matrix is singular. Removing column(s):%s',txt)
        mx = X.shape[0]
        if (mx != my):
            raise ValueError('x and y must have the same number of observations');
        return y, X

    
    def fit(self, y, X=None, theta0=None, beta0=None):
        '''
        Member variables
      .df           : degrees of freedom for error.
      .params       : estimated model parameters
      .params_ci    : 100(1-alpha)% confidence interval for model parameters
      .params_tstat : t statistics for model's estimated parameters.
      .params_pvalue: p value for model's estimated parameters.
      .params_std   : standard errors for estimated parameters
      .params_corr  : correlation matrix for estimated parameters.
      .mu           : fitted values for the model.
      .eta          : linear predictor for the model.
      .residual     : residual for the model (Y-E(Y|X)).
      .dispersnfit  : The estimated error variance
      .deviance     : deviance for the model equal minus twice the log-likelihood.
      .d2L          : Hessian matrix (double derivative of log-likelihood)
      .dL           : First derivative of loglikelihood w.r.t. THETA and BETA.

        '''
        self.family = 'multinomial';
        self.link = 'logit';
        y, X = self.check_xy(y, X)
    
     
        # initial calculations
        tol = self.accuracy 
        incr = 10 
        decr = 2
        ymin = y.min() 
        ymax = y.max() 
        yrange = ymax - ymin
        z  = (y * ones((1, yrange))) == ((y * 0 + 1) * np.arange(ymin, ymax))
        z1 = (y * ones((1, yrange))) == ((y * 0 + 1) * np.arange(ymin + 1, ymax+1))
        z  = z[:, np.flatnonzero(z.any(axis=0))];
        z1 = z1[:, np.flatnonzero(z1.any(axis=0))]
        [mz, nz] = z.shape 
        [mx, nx] = X.shape 
        [my, ny] = y.shape 
      
        g = z.sum(axis=0).cumsum() / my
        theta00 = np.log(g / (1 - g))
        beta00 = np.zeros((nx, 1))
        # starting values 
        if theta0 is None:
            theta0 = theta00
        
        if beta0 is None: 
            beta0  = beta00
      
        tb = np.vstack((theta0, beta0))
    
        # likelihood and derivatives at starting values
        [dev, dl, d2l] = self.loglike(tb, y, X, z, z1)
        
        epsilon = np.std(d2l) / 1000;
        if np.any(beta0) or np.any(theta00!=theta0):
            tb0 = np.vstack((theta00,beta00))
            nulldev = self.loglike (tb0, y, X, z, z1)[0]
        else:
            nulldev = dev
      
      
        # maximize likelihood using Levenberg modified Newton's method
        iter = 0;
        stop = False
        while not stop:
            iter += 1
            tbold = tb;
            devold = dev;
            tb = tbold - np.linalg.lstsq(d2l, dl)
            [dev,p,dl,d2l] = self.loglike(tb, y, X, z, z1)
            if ((dev - devold) / (dl.T * (tb - tbold)) < 0):
                epsilon = epsilon / decr
            else:
                while ((dev - devold) / (dl.T * (tb - tbold)) > 0):
                    epsilon = epsilon * incr;
                    if (epsilon > 1e+15):
                        raise ValueError('epsilon too large');
              
                    tb = tbold - np.linalg.lstsq(d2l - epsilon * np.eye(d2l.shape), dl);
                    [dev,p,dl,d2l] = self.loglike(tb, y, X, z, z1);
                    print('epsilon %g' % epsilon) 
                    #end %while
                    #end else
            #[dl, d2l] = logistic_regression_derivatives (X, z, z1, g, g1, p);
            if (self.verbose>1):
                 
                print('Iter: %d,  Deviance: %8.6f',iter,dev)
                print('First derivative'); 
                print(dl.T);
                print('Eigenvalues of second derivative'); 
                print(np.linalg.eig(d2l)[0].T);
                #end
                #end
            stop = abs (dl.T * np.linalg.lstsq(d2l, dl) / len(dl)) <= tol or iter>self.maxiter
            #end %while
        
        #% tidy up output
        
        theta = tb[:nz, 0]
        beta  = tb[nz:(nz + nx), 1]
        pcov = np.linalg.pinv(-d2l)
        se = sqrt(np.diag (pcov))
        
        
        if (nx > 0):
            eta = ((X * beta) * ones (1, nz)) + ((y * 0 + 1) * theta.T);
        else:
            eta = (y * 0 + 1) * theta.T;
            #end
        gammai = np.diff(np.hstack(((y * 0), self.logitinv(eta), (y * 0 + 1))),1,2)
        k0 = min(y)
        mu = (k0-1)+np.dot(gammai,np.arange(1,nz+2).T); #% E(Y|X)
        r  = np.corrcoef(np.hstack((y,mu)))
        R2 = r[0,1]**2; #coefficient of determination
        R2adj = max(1 - (1-R2)* (my-1)/(my-nx-nz-1),0); # adjusted coefficient of determination
        
        res = y-mu
         
        if nz==1:
            self.family = 'binomial';
        else:
            self.family = 'multinomial';
        
        self.link   = 'logit';
        
        self.numvar = nx+nz;
        self.numobs = my;
        self.numk   = nz+1;
        self.df     = max(my-nx-nz,0);
        self.df_null = my-nz; #nulldf;  nulldf =  n - nz;
        self.params = tb[:(nz + nx),0].T;
        self.params_ci = 1;
        self.params_std = se.T;  
        self.params_cov = pcov
        self.params_tstat = (self.params/self.params_std);
        if False: # % options.estdispersn %dispersion_parameter=='mean_deviance'
            self.params_pvalue=2.*_cdft(-abs(self.params_tstat),self.df);
            bcrit = -se.T*_invt(self.alpha/2,self.df);
        else:
            self.params_pvalue=2.*_cdfnorm(-abs(self.params_tstat));
            bcrit = -se.T*_invnorm(self.alpha/2);
        #end
        self.params_ci = np.vstack((self.params+bcrit,self.params-bcrit))
        
        self.mu = gammai;
        self.eta = self.logit(gammai);
        self.X = X;
        
        self.theta = theta;
        self.beta  = beta;
        self.gamma = gammai;
        self.residual  = res.T; 
        self.residualD = np.sign(self.residual)*sqrt(-2*np.log(p)).T;
        self.deviance  = dev;
        self.deviance_null = nulldev;
        self.d2L = d2l;
        self.dL = dl.T;
        self.dispersnfit=1;
        self.dispersn = 1;
        self.R2 = R2;
        self.R2adj = R2adj;
        self.numiter = iter;
        self.converged = iter<self.maxiter;
        self.note = '';
        self.date = now()
        
        if (self.verbose):
            self.summary()


    def compare(self, object2):
        ''' Compare  small LOGIT versus large one

        CALL     [pvalue] = compare(object2)

        The standard hypothesis test of a larger linear regression 
        model against a smaller one. The standard Chi2-test is used.
        The output is the p-value, the residuals from the smaller 
        model, and the residuals from the larger model.

        See also fitls  
        '''
  
        try:
            if self.numvar>object2.numvar:
                devL = self.deviance;
                nL   = self.numvar;
                dfL = self.df;
                Al = self.X;
                disprsn = self.dispersionfit;
                devs = object2.deviance;
                ns   = object2.numvar;
                dfs = object2.df;
                As = object2.X;
            else:
                devL = object2.deviance;
                nL   = object2.numvar;
                dfL = object2.df;
                Al = object2.X;
                disprsn = object2.dispersionfit;
                devs = self.deviance;
                ns   = self.numvar;
                dfs = self.df;
                As = self.X;
            #end
            
            if  (((As-np.dot(Al*np.linalg.lstsq(Al,As)))>500*np.finfo(float).eps).any() or
                 object2.family!=self.family or object2.link!=self.link):
                warnings.warn('Small model not included in large model, result is rubbish!')
            
            
        except:
            raise ValueError('Apparently not a valid regression object')
    
        
        
        pmq = np.abs(nL-ns);
        print(' ')
        print('                       Analysis of Deviance')
        if False: # %options.estdispersn   
            localstat = abs(devL-devs)/disprsn/pmq;
#            localpvalue = 1-cdff(localstat, pmq, dfL)
#            print('Model    DF      Residual deviance      F-stat        Pr(>F)')
        else:
            localstat = abs(devL-devs)/disprsn;
            localpvalue = 1-_cdfchi2(localstat,pmq)
            print('Model    DF      Residual deviance      Chi2-stat        Pr(>Chi2)')
        #end
        
        
        print('Small    %d       %12.4f       %12.4f    %12.4f' % (dfs,devs,localstat,localpvalue))
        print('Full     %d       %12.4f' % (dfL,devL))
        print(' ')
        
        return localpvalue

    def anode(self):
        print(' ')
        print('                       Analysis of Deviance')
        if False: # %options.estdispersn
            localstat = abs(self.deviance_null-self.deviance)/self.dispersnfit/(self.numvar-1);
            localpvalue = 1-_cdff(localstat,self.numvar-1,self.df);
            print('Model    DF      Residual deviance      F-stat        Pr(>F)')
        else:
            localstat = abs(self.deviance_null-self.deviance)/self.dispersnfit;
            localpvalue = 1-_cdfchi2(localstat,self.numvar-1);
            print('Model    DF      Residual deviance      Chi2-stat        Pr(>Chi2)')
        #end
        
       
        print('Null     %d       %12.4f       %12.4f    %12.4f' % (self.df_null,self.deviance_null,localstat,localpvalue))
        print('Full     %d       %12.4f' % (self.df,self.deviance))
        print(' ')
    
        print(' R2 =  %2.4f,     R2adj = %2.4f' % (self.R2,self.R2adj))
        print(' ')
        return localpvalue
    def summary(self):
        txtlink = self.link;
         
        print('Call:')
        print('reglogit(formula = %s(Pr(grp(y)<=i)) ~ theta_i+beta*x, family = %s)' %(txtlink,self.family))
        print(' ')
        print('Deviance Residuals:')
        m,q1,me, q3,M = np.percentile(self.residualD,q=[0, 25, 50, 75, 100])
        print('    Min       1Q         Median       3Q        Max  ')
        print('%2.4f     %2.4f     %2.4f     %2.4f     %2.4f' % (m, q1, me, q3, M))
        print(' ')
        print(' Coefficients:')
        if False: # %options.estdispersn
            print('            Estimate      Std. Error     t value       Pr(>|t|)')
        else:
            print('            Estimate      Std. Error     z value       Pr(>|z|)')
        #end
        e, s, z, p = self.params, self.params_std, self.params_tstat, self.params_pvalue
        for i in range(self.numk):
            print('theta_%d         %2.4f        %2.4f        %2.4f        %2.4f' % (i,e[i],s[i], z[i], p[i]))
        
        for i in range(self.numk, self.numvar):
            print(' beta_%d         %2.4f        %2.4f        %2.4f        %2.4f\n' % (i-self.numk,e[i],s[i], z[i], p[i]))
        
        print(' ')
        print('(Dispersion parameter for %s family taken to be %2.2f)' % (self.family,self.dispersionfit))
        print(' ')
        if True: #%options.constant
            print('    Null deviance: %2.4f  on %d  degrees of freedom' % (self.deviance_null,self.df_null))
        #end
        print('Residual deviance: %2.4f  on %d  degrees of freedom' % (self.deviance,self.df))
        
        self.anode()
        
        #end % summary
  
    def predict(self, Xnew=None,alpha=0.05, fulloutput=False):
        '''LOGIT/PREDICT Predict from a fitted LOGIT object
        
         CALL [y,ylo,yup] = predict(Xnew,options)
        
         y        = predicted value
         ylo,yup  = 100(1-alpha)% confidence interval for y
         
         Xnew     =  new covariate
         options  = options struct defining the calculation
                .alpha : confidence coefficient (default 0.05)
                .size  : size if binomial family (default 1).    
        '''
        
        [mx, nx] = self.X.shape
        if Xnew is None:
            Xnew = self.X;
        else:
            Xnew = np.atleast_2d(Xnew)
            notnans = np.flatnonzero(1-(1-np.isfinite(Xnew)).any(axis=1))
            Xnew = Xnew[notnans,:]
        
        [n,p] = Xnew.shape
        
         
        if p != nx:
            raise ValueError('Number of covariates must match the number of regression coefficients')
        
        
        nz = self.numk-1;
        one = ones((n,1))
        if (nx > 0):
            eta = np.dot(Xnew * self.beta)  + self.theta.T;
        else:
            eta = one * self.theta.T
        #end
        y = np.diff(np.hstack((zeros((n,1)), self.logitinv(eta), one)),1,2)
        if fulloutput:
            eps = np.finfo(float).eps
            pcov = self.params_cov;
            if (nx > 0):
                np1 = pcov.shape[0]
        
                [U, S, V]= np.linalg.svd(pcov,0);
                R = np.dot(U,np.dot(sqrt(S),V.T)); #%squareroot of pcov
                ib = np.r_[0,nz:np1]
        
                #% Var(eta_i) = var(theta_i+Xnew*b)
                vareta = zeros((n,nz));
                u = np.hstack((one,Xnew))
                for i in range(nz):
                    ib[0] = i
                    vareta[:,i] = np.maximum(sum((np.dot(u,R[ib,ib]))**2,axis=1),eps)
                    #end
            else:
                vareta = np.diag(pcov)
                #end
            crit = -_invnorm(alpha/2);
        
         
            ecrit = crit * sqrt(vareta);
            mulo = _logitinv(eta-ecrit);
            muup = _logitinv(eta+ecrit);
            ylo1 = np.diff(np.hstack((zeros((n,1)), mulo , one)),1,2);
            yup1 = np.diff(np.hstack((zeros((n,1)), muup , one)),1,2);
         
            ylo = np.minimum(ylo1,yup1);
            yup = np.maximum(ylo1,yup1);
        
            for i in range(1, nz): #= 2:self.numk-1
                yup[:,i]  = np.hstack((yup[:,i],muup[:,i]-mulo[:,i-1])).max(axis=1)
                #end
            return y,ylo,yup
        return y

    def loglike(self, beta, y, x, z, z1):
        '''
        [dev, p g, g1] = loglike( y ,x,beta,z,z1)
        Calculates likelihood for the ordinal logistic regression model.
        '''
        # Author: Gordon K. Smyth <gks@maths.uq.oz.au>
        zx = np.hstack((z, x))
        z1x = np.hstack((z1, x))
        g = _logitinv(np.dot(zx, beta))
        g1 = _logitinv(np.dot(z1x, beta))
        g = np.maximum(y == y.max(), g)
        g1 = np.minimum(y > y.min(), g1)

        
        p = g - g1
        dev = -2 * sum (np.log(p));
        #return dev, p, g, g1
##end %function
#
#
#    def logistic_regression_derivatives(self, x, z, z1, g, g1, p):
        '''% [dl, d2l] = logistic_regression_derivatives(x, z, z1, g, g1, p)
        % Called by logistic_regression.  Calculates derivates of the
        % log-likelihood for ordinal logistic regression model.
        '''
        # Author: Gordon K. Smyth <gks@maths.uq.oz.au>
        # Description: Derivates of log-likelihood in logistic regression
        
        
        # first derivative
        v = g * (1 - g) / p; 
        v1 = g1 * (1 - g1) / p;
        dlogp = np.hstack(((dmult(v, z) - dmult(v1, z1)), (dmult(v - v1, x))))
        dl = np.sum(dlogp, axis=0).T
        
        # second derivative
        w = v * (1 - 2 * g)
        w1 = v1 * (1 - 2 * g1)
        d2l = zx.T * dmult (w, zx) - z1x.T * dmult(w1, z1x) - dlogp.T * dlogp;
        
        return dev, p, dl, d2l
        #end %function


def dmult(A,B):
    ''' Return the product of diag(A) and B
    
     USAGE:     m = dmult(a,b)
      where:    a = a matrix
                b = a matrix
     -----------------------------------------------------
     RETURNS:  m = diag(A) times B
     -----------------------------------------------------             
     NOTE: a Gauss compatability function
     -----------------------------------------------------
    '''
    #% written by:
    #%  Gordon K Smyth, U of Queensland, Australia, gks@maths.uq.oz.au
    #% Nov 19, 1990.  Last revision Aug 29, 1995.
    
    return A[:,None]*B;



def _test_dispersion_idx():
    import wafo.data
    xn = wafo.data.sea()
    t, data = xn.T
    Ie = findpot(data,t,0,5);
    di, u, ok_u = dispersion_idx(data[Ie],t[Ie],tb=100)
    di.plot() # a threshold around 1 seems appropriate.
    di.show()
    pass

def _test_findpot():
    import pylab
    import wafo.data
    from wafo.misc import findtc
    x = wafo.data.sea()
    t, data = x[:, :].T
    itc, iv = findtc(data, 0, 'dw')
    ytc, ttc = data[itc], t[itc]
    ymin = 2 * data.std()
    tmin = 10 # sec
    I = findpot(data, t, ymin, tmin)
    yp, tp = data[I], t[I]
    Ie = findpot(yp, tp, ymin, tmin)
    ye, te = yp[Ie], tp[Ie]
    h = pylab.plot(t, data, ttc,ytc,'ro', t, zeros(len(t)), ':', te, ye, 'kx', tp, yp, '+')
    pylab.show() #
    pass

def _test_reslife():
    import wafo
    R = wafo.stats.genpareto.rvs(0.1, 2, 2, size=100)
    mrl = reslife(R, nu=20)
    mrl.plot()
    
def test_reglogit():
    y=np.array([1, 1, 2, 1, 3, 2, 3, 2, 3, 3])
    x = np.arange(10).T
    b = reglogit(y,x)
    b.display() % members and methods
    
    b.summary()
    [mu,plo,pup] = b.predict();
    plot(x,mu,'g',x,plo,'r:',x,pup,'r:')
def main():
    #_test_dispersion_idx() 
    import doctest
    doctest.testmod()
    
    
if __name__ == '__main__':
    pass
    main()
    
