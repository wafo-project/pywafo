from __future__ import division
from wafo.wafodata import WafoData
import numpy as np
from numpy import inf
from numpy import atleast_1d
from numpy import arange, floor
__all__ = ['edf']

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
    >>> F.plot()
      
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
    >>> Fc = ws.edfcd(R, 1)
    >>> Fc.plot()
    >>> F = ws.edf(R)
    >>> F.plot()
    
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
