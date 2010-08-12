from wafo.transform import TrData

def test_trdata():
    '''
    Construct a linear transformation model
    >>> import numpy as np
    >>> sigma = 5; mean = 1
    >>> u = np.linspace(-5,5); x = sigma*u+mean; y = u
    >>> g = TrData(y,x)
    >>> g.mean
    array([ 1.])
    >>> g.sigma
    array([ 5.])
    
    >>> g = TrData(y,x,mean=1,sigma=5)
    >>> g.mean
    1
    >>> g.sigma
    5
    >>> g.dat2gauss(1,2,3)
    [array([ 0.]), array([ 0.4]), array([ 0.6])]
    >>> g.dat2gauss([0,1,2,3])
    array([-0.2,  0. ,  0.2,  0.4])
    
    Check that the departure from a Gaussian model is zero
    >>> g.dist2gauss() < 1e-16
    True
    '''
    
if __name__=='__main__':
    import doctest
    doctest.testmod()