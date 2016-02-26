import numpy as np
from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct
import os
path = os.path.dirname(os.path.realpath(__file__))

__all__ = ['dct', 'idct', 'dctn', 'idctn']


def dct(x, type=2, n=None, axis=-1, norm='ortho'):  # @ReservedAssignment
    '''
    Return the Discrete Cosine Transform of arbitrary type sequence x.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.
    axis : int, optional
        Axis over which to compute the transform.
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is 'ortho'.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idct

    Notes
    -----
    For a single dimension array ``x``, ``dct(x, norm='ortho')`` is equal to
    MATLAB ``dct(x)``.

    There are theoretically 8 types of the DCT, only the first 3 types are
    implemented in scipy. 'The' DCT generally refers to DCT type 2, and 'the'
    Inverse DCT generally refers to DCT type 3.

    type I
    ~~~~~~
    There are several definitions of the DCT-I; we use the following
    (for ``norm=None``)::

                                         N-2
      y[k] = x[0] + (-1)**k x[N-1] + 2 * sum x[n]*cos(pi*k*n/(N-1))
                                         n=1

    Only None is supported as normalization mode for DCT-I. Note also that the
    DCT-I is only supported for input size > 1

    type II
    ~~~~~~~
    There are several definitions of the DCT-II; we use the following
    (for ``norm=None``)::

                N-1
      y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                n=0

    If ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor `f`::

      f = sqrt(1/(4*N)) if k = 0,
      f = sqrt(1/(2*N)) otherwise.

    Which makes the corresponding matrix of coefficients orthonormal
    (``OO' = Id``).

    type III
    ~~~~~~~~

    There are several definitions, we use the following
    (for ``norm=None``)::

                        N-1
      y[k] = x[0] + 2 * sum x[n]*cos(pi*(k+0.5)*n/N), 0 <= k < N.
                        n=1

    or, for ``norm='ortho'`` and 0 <= k < N::

                                          N-1
      y[k] = x[0] / sqrt(N) + sqrt(1/N) * sum x[n]*cos(pi*(k+0.5)*n/N)
                                          n=1

    The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up
    to a factor `2N`. The orthonormalized DCT-III is exactly the inverse of
    the orthonormalized DCT-II.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.abs(x-idct(dct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(x-dct(idct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)

    References
    ----------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/

    'A Fast Cosine Transform in One and Two Dimensions', by J. Makhoul, `IEEE
    Transactions on acoustics, speech and signal processing` vol. 28(1),
    pp. 27-34, http://dx.doi.org/10.1109/TASSP.1980.1163351 (1980).
    '''
    return _dct(x, type, n, axis, norm)


def idct(x, type=2, n=None, axis=-1, norm='ortho'):  # @ReservedAssignment
    '''
    Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.
    axis : int, optional
        Axis over which to compute the transform.
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is 'ortho'.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    dct

    Notes
    -----
    For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to
    matlab ``idct(x)``.

    'The' IDCT is the IDCT of type 2, which is the same as DCT of type 3.

    IDCT of type 1 is the DCT of type 1, IDCT of type 2 is the DCT of type 3,
    and IDCT of type 3 is the DCT of type 2. For the definition of these types,
    see `dct`.
    '''
    return _idct(x, type, n, axis, norm)


def _get_shape(y, shape, axes):
    if shape is None:
        if axes is None:
            shape = y.shape
        else:
            shape = np.take(y.shape, axes)
    shape = tuple(shape)
    return shape


def _get_axes(y, shape, axes):
    if axes is None:
        axes = range(y.ndim)
    if len(axes) != len(shape):
        raise ValueError("when given, axes and shape arguments "
                         "have to be of the same length")
    return list(axes)


def _raw_dctn(y, type, shape, axes, norm, fun):  # @ReservedAssignment
    shape = _get_shape(y, shape, axes)
    axes = _get_axes(y, shape, axes)
    shape0 = list(y.shape)
    ndim = y.ndim
    isvector = max(shape0) == y.size
    if isvector and ndim == 1:
        y = np.atleast_2d(y)
        y = y.T
    for dim in range(ndim):
        y = shiftdim(y, 1)
        if dim not in axes:
            continue
        n = shape[axes.index(dim)]
        shape0[dim] = n
        y = fun(y, type, n, norm=norm)

    return y.reshape(shape0)


def dctn(x, type=2, shape=None, axes=None,  # @ReservedAssignment
         norm='ortho'):
    '''
    Return the N-D Discrete Cosine Transform of array x.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3}, optional
        Type of the DCT (see Notes). Default type is 2.
    shape : tuple of ints, optional
        The shape of the result.  If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
        length ``shape[i]``.
    axes : array_like of ints, optional
        The axes of `x` (`y` if `shape` is not None) along which the
        transform is applied.
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes in dct). Default is 'ortho'.

    Returns
    -------
    y : ndarray of real
        The transformed input array.


    Notes
    -----
    Y = DCTN(X) returns the discrete cosine transform of X. The array Y is
    the same size as X and contains the discrete cosine transform
    coefficients. This transform can be inverted using IDCTN.

    Input array can be numeric or logical. The returned array is of class
    double.

    Reference
    ---------
    Narasimha M. et al, On the computation of the discrete cosine
    transform, IEEE Trans Comm, 26, 6, 1978, pp 934-936.

    Example
    -------
    >>> import os
    >>> import numpy as np
    >>> import scipy.ndimage as sn
    >>> import matplotlib.pyplot as plt
    >>> name = os.path.join(path, 'autumn.gif')
    >>> rgb = sn.imread(name)
    >>> J = dctn(rgb)
    >>> (np.abs(rgb-idctn(J))<1e-7).all()
    True

    plt.imshow(np.log(np.abs(J)))
    plt.colorbar() #colormap(jet), colorbar

    The commands below set values less than magnitude 10 in the DCT matrix
    to zero, then reconstruct the image using the inverse DCT.

    J[np.abs(J)<10] = 0
    K = idctn(J)
    plt.figure(0)
    plt.imshow(rgb)
    plt.figure(1)
    plt.imshow(K,[0 255])

    See also
    --------
    idctn, dct, idct
    '''
    y = np.atleast_1d(x)
    return _raw_dctn(y, type, shape, axes, norm, dct)


def idctn(x, type=2, shape=None, axes=None,  # @ReservedAssignment
          norm='ortho'):
    '''Return inverse N-D Discrete Cosine Transform of array x.

    For description of parameters see `dctn`.

    See Also
    --------
    dctn : for detailed information.
    '''
    y = np.atleast_1d(x)
    return _raw_dctn(y, type, shape, axes, norm, idct)


def num_leading_ones(x):
    first = 0
    for i, xi in enumerate(x):
        if xi != 1:
            first = i
            break
    return first


def no_leading_ones(x):
    first = num_leading_ones(x)
    return x[first:]


def shiftdim(x, n=None):
    '''
    Shift dimensions

    Parameters
    ----------
    x : array
    n : int

    Notes
    -----
    Shiftdim is handy for functions that intend to work along the first
    non-singleton dimension of the array.

    If n is None returns the array with the same number of elements as X but
    with any leading singleton dimensions removed.

    When n is positive, shiftdim shifts the dimensions to the left and wraps
    the n leading dimensions to the end.

    When n is negative, shiftdim shifts the dimensions to the right and pads
    with singletons.

    See also
    --------
    reshape, squeeze
    '''
    if n is None:
        return x.reshape(no_leading_ones(x.shape))
    elif n >= 0:
        return x.transpose(np.roll(range(x.ndim), -n))
    else:
        return x.reshape((1,) * -n + x.shape)


def test_shiftdim():
    a = np.arange(6).reshape((1, 1, 3, 1, 2))

    print(a.shape)
    print(a.ndim)

    print(range(a.ndim))
    # move items 2 places to the left so that x0 <- x2, x1 <- x3, etc
    print(np.roll(range(a.ndim), -2))
    print(a.transpose(np.roll(range(a.ndim), -2)))  # transposition of the axes
    # with a matrix 2x2, A.transpose((1,0)) would be the transpose of A
    b = shiftdim(a)
    print(b.shape)

    c = shiftdim(b, -2)
    print(c.shape)

    print(c == a)


def example_dct2():
    import scipy.ndimage as sn
    import matplotlib.pyplot as plt
    name = os.path.join(path, 'autumn.gif')
    rgb = np.asarray(sn.imread(name), dtype=np.float16)
    # np.fft.fft(rgb)
    print(np.max(rgb), np.min(rgb))
    plt.set_cmap('jet')
    J = dctn(rgb)
    irgb = idctn(J)
    print(np.abs(rgb-irgb).max())
    plt.imshow(np.log(np.abs(J)))
    # plt.colorbar() #colormap(jet), colorbar
    plt.show('hold')
    v = np.percentile(np.abs(J.ravel()), 10.0)
    print(len(np.flatnonzero(J)), v)
    J[np.abs(J) < v] = 0
    print(len(np.flatnonzero(J)))
    plt.imshow(np.log(np.abs(J)))
    plt.show('hold')
    K = idctn(J)
    print(np.abs(rgb-K).max())
    plt.figure(1)
    plt.imshow(rgb)
    plt.figure(2)
    plt.imshow(K, vmin=0, vmax=255)
    plt.show('hold')


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    test_docstrings()
