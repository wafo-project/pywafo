'''
Created on 15. des. 2016

@author: pab
'''
from __future__ import division
from scipy import sparse
import numpy as np
from wafo.testing import test_docstrings
from itertools import product

__all__ = ['accum',  'gridcount']


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


def bitget(int_type, offset):
    """Returns the value of the bit at the offset position in int_type.

    Example
    -------
    >>> bitget(5, np.r_[0:4])
    array([1, 0, 1, 0])

    """
    return np.bitwise_and(int_type, 1 << offset) >> offset


def accumsum_sparse(accmap, a, shape=None, dtype=None):
    """
    Example
    -------
    >>> from numpy import array
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accumsum_sparse(accmap, a, (3,))
    >>> np.allclose(s.toarray(), [ 9,  7, 15])
    True

     # group vals by idx and sum
    >>> vals = array([12.0, 3.2, -15, 88, 12.9])
    >>> idx = array([1,    0,    1,  4,   1  ])
    >>> np.allclose(accumsum_sparse(idx, vals).toarray(),
    ...   [3.2, 9.9, 0, 0, 88.])
    True
    """
    if dtype is None:
        dtype = a.dtype
    if shape is None:
        shape = 1 + np.squeeze(np.apply_over_axes(np.max, accmap,
                                                  axes=tuple(range(a.ndim))))
    shape = np.atleast_1d(shape)
    if len(shape) > 1:
        binx = accmap[:, 0]
        biny = accmap[:, 1]
        out = sparse.coo_matrix(
            (a.ravel(), (binx, biny)), shape=shape, dtype=dtype).tocsr()
    else:
        binx = accmap.ravel()
        zero = np.zeros(len(binx))
        out = sparse.coo_matrix(
            (a.ravel(), (binx, zero)), shape=(shape, 1), dtype=dtype).tocsr().T
    return out


def accumsum(accmap, a, shape=None):
    """

    Example
    -------
    >>> from numpy import array
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])

    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])  # Sum the diagonals.
    >>> np.allclose(accumsum(accmap, a, (3,)), [ 9,  7, 15])
    True
    >>> accmap = array([[0,1,2],[0,1,2],[0,1,2]])  # Sum the columns.
    >>> np.allclose(accumsum(accmap, a, (3,)), [4, 9, 18])
    True

    # group vals by idx and sum
    >>> vals = array([12.0, 3.2, -15, 88, 12.9])
    >>> idx = array([1,    0,    1,  4,   1  ])
    >>> np.allclose(accumsum(idx, vals), [3.2, 9.9, 0, 0, 88.])
    True
    """
    if shape is None:
        shape = 1 + np.squeeze(np.apply_over_axes(np.max, accmap,
                                                  axes=tuple(range(a.ndim))))
    return np.bincount(accmap.ravel(), a.ravel(), np.array(shape).max())


def accum(accmap, a, func=None, shape=None, fill_value=0, dtype=None):
    """An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    shape : ndarray or None
        The shape of the output array.  If None, the shape will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `shape` if `shape` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])  # Sum the diagonals.
    >>> s = accum(accmap, a)
    >>> s
    array([ 9,  7, 15])

    # A 2D output, from sub-arrays with shapes and positions like this:
    # [ (2,2) (2,1)]
    # [ (1,2) (1,1)]
    >>> accmap = array([
    ...        [[0,0],[0,0],[0,1]],
    ...        [[0,0],[0,0],[0,1]],
    ...        [[1,0],[1,0],[1,1]]])

    # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> accum(accmap, a, dtype=float)
    array([[ 6.,  9.],
           [ 7.,  9.]])

    # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)

    """

    def create_array_of_python_lists(accmap, a, shape):
        vals = np.empty(shape, dtype='O')
        for s in product(*[range(k) for k in shape]):
            vals[s] = []

        for s in product(*[range(k) for k in a.shape]):
            indx = tuple(accmap[s])
            val = a[s]
            vals[indx].append(val)

        return vals

    def _initialize(accmap, a, func, shape, dtype):
        """Check for bad arguments and handle the defaults."""

        _assert(accmap.shape[:a.ndim] == a.shape,
                "The initial dimensions of accmap must be the same as a.shape")
        if accmap.shape == a.shape:
            accmap = np.expand_dims(accmap, -1)
        if func is None:
            func = np.sum
        if dtype is None:
            dtype = a.dtype
        adims = tuple(range(a.ndim))
        if shape is None:
            shape = 1 + np.squeeze(np.apply_over_axes(np.max, accmap,
                                                      axes=adims))
        shape = np.atleast_1d(shape)
        return accmap, shape, dtype, func

    accmap, shape, dtype, func = _initialize(accmap, a, func, shape, dtype)

    # Create an array of python lists of values.
    vals = create_array_of_python_lists(accmap, a, shape)

    # Create the output array.
    out = np.empty(shape, dtype=dtype)
    for s in product(*[range(k) for k in shape]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])
    return out


def _gridcount_nd(acfun, data, x, y, w, binx):
    d, inc = x.shape
    c_shape = np.repeat(inc, d)
    nc = c_shape.prod()  # inc ** d

    c = np.zeros((nc, ))
    fact2 = np.asarray(np.reshape(inc * np.arange(d), (d, -1)), dtype=int)
    fact1 = np.asarray(np.reshape(c_shape.cumprod() / inc, (d, -1)), dtype=int)
    bt0 = [0, 0]
    x1 = x.ravel()
    for ir in range(2 ** (d - 1)):
        bt0[0] = np.reshape(bitget(ir, np.arange(d)), (d, -1))
        bt0[1] = 1 - bt0[0]
        for ix in range(2):
            one = np.mod(ix, 2)
            two = np.mod(ix + 1, 2)
            # Convert to linear index
            # linear index to c
            b1 = np.sum((binx + bt0[one]) * fact1, axis=0)
            bt2 = bt0[two] + fact2
            b2 = binx + bt2   # linear index to X
            c += acfun(b1, np.abs(np.prod(x1[b2] - data, axis=0)) * y,
                       shape=(nc, ))

    c = np.reshape(c / w, c_shape, order='F')

    T = np.arange(d).tolist()
    T[1], T[0] = T[0], T[1]
    # make sure c is stored in the same way as meshgrid
    c = c.transpose(*T)
    return c


def _gridcount_1d(acfun, data, x, y, w, binx, inc):
    x.shape = -1,
    c = (acfun(binx, (x[binx + 1] - data) * y, shape=(inc, )) +
         acfun(binx + 1, (data - x[binx]) * y, shape=(inc, ))).ravel()
    return c / w


def gridcount(data, X, y=1):
    '''
    Returns D-dimensional histogram using linear binning.

    Parameters
    ----------
    data : array-like
        column vectors with D-dimensional data, shape D x Nd
    X : array-like
        row vectors defining discretization in each dimension, shape D x N.
        The discretization must include the range of the data.
    y : array-like
        response data. Scalar or vector of size Nd.

    Returns
    -------
    c : ndarray
        gridcount,  shape N x N x ... x N

    GRIDCOUNT obtains the grid counts using linear binning.
    There are 2 strategies: simple- or linear- binning.
    Suppose that an observation occurs at x and that the nearest point
    below and above is y and z, respectively. Then simple binning strategy
    assigns a unit weight to either y or z, whichever is closer. Linear
    binning, on the other hand, assigns the grid point at y with the weight
    of (z-x)/(z-y) and the gridpoint at z a weight of (y-x)/(z-y).

    In terms of approximation error of using gridcounts as pdf-estimate,
    linear binning is significantly more accurate than simple binning.

     NOTE: The interval [min(X);max(X)] must include the range of the data.
           The order of C is permuted in the same order as
           meshgrid for D==2 or D==3.

    Example
    -------
    >>> import numpy as np
    >>> import wafo.kdetools as wk
    >>> N = 20
    >>> data  = np.random.rayleigh(1,N)
    >>> data = np.array(
    ...    [ 1.07855907,  1.51199717,  1.54382893,  1.54774808,  1.51913566,
    ...     1.11386486,  1.49146216,  1.51127214,  2.61287913,  0.94793051,
    ...     2.08532731,  1.35510641,  0.56759888,  1.55766981,  0.77883602,
    ...     0.9135759 ,  0.81177855,  1.02111483,  1.76334202,  0.07571454])
    >>> x = np.linspace(0,max(data)+1,50)
    >>> dx = x[1]-x[0]

    >>> c = wk.gridcount(data, x)
    >>> np.allclose(c[:5], [ 0.,  0.9731147,  0.0268853,  0.,  0.])
    True

    >>> pdf = c/dx/N
    >>> np.allclose(np.trapz(pdf, x), 1)
    True

    import matplotlib.pyplot as plt
    h = plt.plot(x,c,'.')   # 1D histogram
    h1 = plt.plot(x, pdf) #  1D probability density plot

    See also
    --------
    bincount, accum, kde

    Reference
    ----------
    Wand,M.P. and Jones, M.C. (1995)
    'Kernel smoothing'
    Chapman and Hall, pp 182-192
    '''
    dataset, x = np.atleast_2d(data, X)
    y = np.atleast_1d(y).ravel()
    d, inc = x.shape

    _assert(d == dataset.shape[0], 'Dimension 0 of data and X do not match.')

    xlo, xup = x[:, 0], x[:, -1]

    datlo, datup = dataset.min(axis=1), dataset.max(axis=1)
    _assert(not ((datlo < xlo) | (xup < datup)).any(),
            'X does not include whole range of the data!')

    #  acfun = accumsum_sparse  # faster than accum
    acfun = accumsum  # faster than accumsum_sparse

    dx = np.diff(x[:, :2], axis=1)
    binx = np.asarray(np.floor((dataset - xlo[:, np.newaxis]) / dx), dtype=int)
    w = dx.prod()
    if d == 1:
        return _gridcount_1d(acfun, dataset, x, y, w, binx, inc)
    return _gridcount_nd(acfun, dataset, x, y, w, binx)


if __name__ == '__main__':
    test_docstrings(__file__)
