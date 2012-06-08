'''
Created on 20. jan. 2011

@author: pab
'''
import numpy as np
from numpy import exp
from wafo.misc import meshgrid
__all__ = ['peaks', 'humps', 'magic']

def magic(n):
    '''
    Return magic square  for n of odd orders.
    
    A magic square has the property that the sum of every row and column, 
    as well as both diagonals, is the same number.
    
    
    '''
    if np.mod(n,1)==1: # odd order
        ix = np.arange(n)+1
        J, I = np.meshgrid(ix,ix)
        A = np.mod(I+J-(n+3)/2,n)
        B = np.mod(I+2*J-2,n)
        M = n*A + B + 1
        
    return M

def peaks(x=None, y=None, n=51):
    '''
    Return the "well" known MatLab (R) peaks function
    evaluated in the [-3,3] x,y range
    
    Example
    -------
    >>> import pylab as plt
    >>> x,y,z = peaks()
    >>> h = plt.contourf(x,y,z)
    
    '''
    if x is None:
        x = np.linspace(-3, 3, n)
    if y is None:
        y = np.linspace(-3, 3, n)
        
    [x1, y1] = meshgrid(x, y)

    z = (3 * (1 - x1) ** 2 * exp(-(x1 ** 2) - (y1 + 1) ** 2) 
         - 10 * (x1 / 5 - x1 ** 3 - y1 ** 5) * exp(-x1 ** 2 - y1 ** 2) 
         - 1. / 3 * exp(-(x1 + 1) ** 2 - y1 ** 2))

    return x1, y1, z

def humps(x=None):
    '''
    Computes a function that has three roots, and some humps.
    '''
    if x is None:
        y = np.linspace(0, 1)
    else:
        y = np.asarray(x)

    return 1.0 / ((y - 0.3) ** 2 + 0.01) + 1.0 / ((y - 0.9) ** 2 + 0.04) + 2 * y - 5.2

if __name__ == '__main__':
    pass