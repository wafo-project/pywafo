from __future__ import division
import warnings
from wafo.wafodata import WafoData
from wafo.misc import findextrema
from scipy import special
import numpy as np
from numpy import inf
from numpy import atleast_1d, nan, ndarray, sqrt, vstack, ones, where, zeros
from numpy import arange, floor, linspace, asarray, reshape, repeat, product


__all__ = ['edf', 'edfcnd','reslife', 'dispersion_idx','decluster','findpot', 
           'declustering_time','extremal_idx']

arr = asarray

def valarray(shape, value=nan, typecode=None):
    """Return an array of all value.
    """
    out = reshape(repeat([value], product(shape, axis=0), axis=0), shape)
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, ndarray):
        out = arr(out)
    return out 

def _invt(q, df):
    return special.stdtrit(df, q)

def _invchi2(q, df):
    return special.chdtri(df, q)


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
    
def main():
    #_test_dispersion_idx() 
    import doctest
    doctest.testmod()
    
    
if __name__ == '__main__':
    main()
