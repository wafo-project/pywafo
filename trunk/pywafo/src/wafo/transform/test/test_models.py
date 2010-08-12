from wafo.transform.models import TrHermite, TrOchi, TrLinear

def test_trhermite():
    '''
    >>> std = 7./4
    >>> g = TrHermite(sigma=std, ysigma=std)
    >>> g.dist2gauss()
    3.9858776379926808
    
    >>> g.mean
    0.0
    >>> g.sigma
    1.75
    >>> g.dat2gauss([0,1,2,3])
    array([ 0.04654321,  1.03176393,  1.98871279,  2.91930895])
    
    '''
def test_trochi():
    '''
    >>> std = 7./4
    >>> g = TrOchi(sigma=std, ysigma=std)
    >>> g.dist2gauss()
    5.9322684525265501
    >>> g.mean
    0.0
    >>> g.sigma
    1.75
    >>> g.dat2gauss([0,1,2,3])
    array([  6.21927960e-04,   9.90237621e-01,   1.96075606e+00,
             2.91254576e+00])
    '''
def test_trlinear():
    '''
    >>> std = 7./4
    >>> g = TrLinear(sigma=std, ysigma=std)
    >>> g.dist2gauss()
    0.0
    >>> g.mean
    0.0
    >>> g.sigma
    1.75
    >>> g.dat2gauss([0,1,2,3])
    array([ 0.,  1.,  2.,  3.])
    '''
if __name__=='__main__':
    import doctest
    doctest.testmod()