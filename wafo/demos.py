'''
Created on 20. jan. 2011

@author: pab
'''
import numpy as np
from numpy import exp, meshgrid
__all__ = ['peaks', 'humps', 'magic']


def magic(n):
    '''
    Return magic square  for n of any orders > 2.

    A magic square has the property that the sum of every row and column,
    as well as both diagonals, is the same number.

    Examples
    --------
    >>> magic(3)
    array([[8, 1, 6],
           [3, 5, 7],
           [4, 9, 2]])

    >>> magic(4)
    array([[16,  2,  3, 13],
           [ 5, 11, 10,  8],
           [ 9,  7,  6, 12],
           [ 4, 14, 15,  1]])

    >>> magic(6)
    array([[35,  1,  6, 26, 19, 24],
           [ 3, 32,  7, 21, 23, 25],
           [31,  9,  2, 22, 27, 20],
           [ 8, 28, 33, 17, 10, 15],
           [30,  5, 34, 12, 14, 16],
           [ 4, 36, 29, 13, 18, 11]])
    '''
    if (n < 3):
        raise ValueError('n must be greater than 2.')

    if np.mod(n, 2) == 1:  # odd order
        ix = np.arange(n) + 1
        J, I = np.meshgrid(ix, ix)
        A = np.mod(I + J - (n + 3) / 2, n)
        B = np.mod(I + 2 * J - 2, n)
        M = n * A + B + 1
    elif np.mod(n, 4) == 0:  # doubly even order
        M = np.arange(1, n * n + 1).reshape(n, n)
        ix = np.mod(np.arange(n) + 1, 4) // 2
        J, I = np.meshgrid(ix, ix)
        iz = np.flatnonzero(I == J)
        M.put(iz, n * n + 1 - M.flat[iz])
    else:  # singly even order
        p = n // 2
        M0 = magic(p)

        M = np.hstack((np.vstack((M0, M0 + 3 * p * p)),
                       np.vstack((M0 + 2 * p * p, M0 + p * p))))

        if n > 2:
            k = (n - 2) // 4
            Jvec = np.hstack((np.arange(k), np.arange(n - k + 1, n)))
            for i in range(p):
                for j in Jvec:
                    temp = M[i][j]
                    M[i][j] = M[i + p][j]
                    M[i + p][j] = temp

            i = k
            j = 0
            temp = M[i][j]
            M[i][j] = M[i + p][j]
            M[i + p][j] = temp

            j = i
            temp = M[i + p][j]
            M[i + p][j] = M[i][j]
            M[i][j] = temp

    return M


def peaks(x=None, y=None, n=51):
    '''
    Return the "well" known MatLab (R) peaks function
    evaluated in the [-3,3] x,y range

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> x,y,z = peaks()

    h = plt.contourf(x,y,z)

    '''
    if x is None:
        x = np.linspace(-3, 3, n)
    if y is None:
        y = np.linspace(-3, 3, n)

    [x1, y1] = meshgrid(x, y)

    z = (3 * (1 - x1) ** 2 * exp(-(x1 ** 2) - (y1 + 1) ** 2) -
         10 * (x1 / 5 - x1 ** 3 - y1 ** 5) * exp(-x1 ** 2 - y1 ** 2) -
         1. / 3 * exp(-(x1 + 1) ** 2 - y1 ** 2))

    return x1, y1, z


def humps(x=None):
    '''
    Computes a function that has three roots, and some humps.

     Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0,1)
    >>> y = humps(x)

    h = plt.plot(x,y)
    '''
    if x is None:
        y = np.linspace(0, 1)
    else:
        y = np.asarray(x)

    return 1.0 / ((y - 0.3) ** 2 + 0.01) + 1.0 / ((y - 0.9) ** 2 + 0.04) + \
        2 * y - 5.2


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

if __name__ == '__main__':
    test_docstrings()
