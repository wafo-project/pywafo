# -*- coding:utf-8 -*-
"""
Created on 19. nov. 2010

@author: pab
"""
from numpy import array, log
import wafo.stats as ws
from wafo.stats.estimation import FitDistribution, Profile
def test_profile():
    '''
    # MLE 
    import wafo.stats as ws
    R = ws.weibull_min.rvs(1,size=20)
    >>> R = array([ 0.08018795,  1.09015299,  2.08591271,  0.51542081,  0.75692042,
    ...    0.57837017,  0.45419753,  1.1559131 ,  0.26582267,  0.51517273,
    ...    0.75008492,  0.59404957,  1.33748264,  0.14472142,  0.77459603,
    ...    1.77312556,  1.06347991,  0.42007769,  0.71094628,  0.02366977])
    
    >>> phat = FitDistribution(ws.weibull_min, R, 1, scale=1, floc=0.0)
    >>> phat.par
    array([ 1.37836487,  0.        ,  0.82085633])
    
    # Better CI for phat.par[i=0]
    >>> Lp = Profile(phat, i=0)
    >>> Lp.get_bounds(alpha=0.1)
    array([ 1.00225064,  1.8159036 ])
    
    >>> SF = 1./990
    >>> x = phat.isf(SF)
    >>> x
    3.3323076459875312
    
    # CI for x
    >>> Lx = phat.profile(i=0, x=x, link=phat.dist.link)
    >>> Lx.get_bounds(alpha=0.2)
    array([ 2.52211661,  4.98664787])
    
    # CI for logSF=log(SF)
    >>> logSF = log(SF)
    >>> Lsf = phat.profile(i=0, logSF=logSF, link=phat.dist.link, pmin=logSF-10,pmax=logSF+5)
    >>> Lsf.get_bounds(alpha=0.2)
    array([-10.87488318,  -4.36225468])
    '''

if __name__ == '__main__':
    import doctest
    doctest.testmod()