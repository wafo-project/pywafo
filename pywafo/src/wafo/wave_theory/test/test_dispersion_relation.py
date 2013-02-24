'''
Created on 19. juli 2010

@author: pab
'''

from wafo.wave_theory.dispersion_relation import w2k,k2w #@UnusedImport

def test_k2w():
    '''
    >>> from numpy import arange
    >>> k2w(arange(0.01,.5,0.2))[0]
    array([ 0.3132092 ,  1.43530485,  2.00551739])
    >>> k2w(arange(0.01,.5,0.2),h=20)[0]
    array([ 0.13914927,  1.43498213,  2.00551724])
    '''


def test_w2k():
    '''
    >>> w2k(range(4))[0]
    array([ 0.        ,  0.1019368 ,  0.4077472 ,  0.91743119])
    >>> w2k(range(4),h=20)[0]
    array([ 0.        ,  0.10503601,  0.40774726,  0.91743119])
    '''
    
def test_doctstrings():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    test_doctstrings()