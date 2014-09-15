import numpy as np
from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct
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

    References
    ----------

    http://en.wikipedia.org/wiki/Discrete_cosine_transform

    'A Fast Cosine Transform in One and Two Dimensions', by J. Makhoul, `IEEE
    Transactions on acoustics, speech and signal processing` vol. 28(1),
    pp. 27-34, http://dx.doi.org/10.1109/TASSP.1980.1163351 (1980).
    '''
    farr = np.asfarray
    if np.iscomplex(x).any():
        return _dct(farr(x.real), type, n, axis, norm) + \
            1j * _dct(farr(x.imag), type, n, axis, norm)
    else:
        return _dct(farr(x), type, n, axis, norm)


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
    farr = np.asarray
    if np.iscomplex(x).any():
        return _idct(farr(x.real), type, n, axis, norm) + \
            1j * _idct(farr(x.imag), type, n, axis, norm)
    else:
        return _idct(farr(x), type, n, axis, norm)


def dctn(x, type=2, axis=None, norm='ortho'):  # @ReservedAssignment
    '''
    DCTN N-D discrete cosine transform.

    Y = DCTN(X) returns the discrete cosine transform of X. The array Y is
    the same size as X and contains the discrete cosine transform
    coefficients. This transform can be inverted using IDCTN.

    DCTN(X,axis) applies the DCTN operation across the dimension axis.

    Class Support
    -------------
    Input array can be numeric or logical. The returned array is of class
    double.

    Reference
    ---------
    Narasimha M. et al, On the computation of the discrete cosine
    transform, IEEE Trans Comm, 26, 6, 1978, pp 934-936.

    Example
    -------
    RGB = imread('autumn.tif');
    I = rgb2gray(RGB);
    J = dctn(I);
    imshow(log(abs(J)),[]), colormap(jet), colorbar

    The commands below set values less than magnitude 10 in the DCT matrix
    to zero, then reconstruct the image using the inverse DCT.

        J(abs(J)<10) = 0;
        K = idctn(J);
        figure, imshow(I)
        figure, imshow(K,[0 255])

    See also
    --------
    idctn, dct, idct
    '''

    y = np.atleast_1d(x)
    shape0 = y.shape
    if axis is None:
        y = y.squeeze()  # Working across singleton dimensions is useless
    ndim = y.ndim
    isvector = max(shape0) == y.size
    if isvector:
        if ndim == 1:
            y = np.atleast_2d(y)
            y = y.T
        elif y.shape[0] == 1:
            if axis == 0:
                return x
            elif axis == 1:
                axis = 0
            y = y.T
        elif axis == 1:
            return y

    if np.iscomplex(y).any():
        y = dctn(y.real, type, axis, norm) + 1j * \
            dctn(y.imag, type, axis, norm)
    else:
        y = np.asfarray(y)
        for dim in range(ndim):
            y = y.transpose(np.roll(range(y.ndim), -1))
            #y = shiftdim(y,1)
            if axis is not None and dim != axis:
                continue
            y = _dct(y, type, norm=norm)
    return y.reshape(shape0)


def idctn(x, type=2, axis=None, norm='ortho'):  # @ReservedAssignment
    y = np.atleast_1d(x)
    shape0 = y.shape
    if axis is None:
        y = y.squeeze()  # Working across singleton dimensions is useless
    ndim = y.ndim
    isvector = max(shape0) == y.size
    if isvector:
        if ndim == 1:
            y = np.atleast_2d(y)
            y = y.T
        elif y.shape[0] == 1:
            if axis == 0:
                return x
            elif axis == 1:
                axis = 0
            y = y.T
        elif axis == 1:
            return y

    if np.iscomplex(y).any():
        y = idctn(y.real, type, axis, norm) + 1j * \
            idctn(y.imag, type, axis, norm)
    else:
        y = np.asfarray(y)
        for dim in range(ndim):
            y = y.transpose(np.roll(range(y.ndim), -1))
            #y = shiftdim(y,1)
            if axis is not None and dim != axis:
                continue
            y = _idct(y, type, norm=norm)
    return y.reshape(shape0)

# def dct(x, n=None):
#    """
#    Discrete Cosine Transform
#
#                      N-1
#           y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
#                      n=0
#
#    Examples
#    --------
#    >>> import numpy as np
#    >>> x = np.arange(5)
#    >>> np.abs(x-idct(dct(x)))<1e-14
#    array([ True,  True,  True,  True,  True], dtype=bool)
#    >>> np.abs(x-dct(idct(x)))<1e-14
#    array([ True,  True,  True,  True,  True], dtype=bool)
#
#    Reference
#    ---------
#    http://en.wikipedia.org/wiki/Discrete_cosine_transform
#    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
#    """
#    fft = np.fft.fft
#    x = np.atleast_1d(x)
#
#    if n is None:
#        n = x.shape[-1]
#
#    if x.shape[-1] < n:
#        n_shape = x.shape[:-1] + (n - x.shape[-1],)
#        xx = np.hstack((x, np.zeros(n_shape)))
#    else:
#        xx = x[..., :n]
#
#    real_x = np.all(np.isreal(xx))
#    if (real_x and (np.remainder(n, 2) == 0)):
#        xp = 2 * fft(np.hstack((xx[..., ::2], xx[..., ::-2])))
#    else:
#        xp = fft(np.hstack((xx, xx[..., ::-1])))
#        xp = xp[..., :n]
#
#    w = np.exp(-1j * np.arange(n) * np.pi / (2 * n))
#
#    y = xp * w
#
#    if real_x:
#        return y.real
#    else:
#        return y
#
# def idct(x, n=None):
#    """
#    Inverse Discrete Cosine Transform
#
#                N-1
#    x[k] = 1/N sum w[n]*y[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
#               n=0
#
#    w(0) = 1/2
#    w(n) = 1 for n>0
#
#    Examples
#    --------
#    >>> import numpy as np
#    >>> x = np.arange(5)
#    >>> np.abs(x-idct(dct(x)))<1e-14
#    array([ True,  True,  True,  True,  True], dtype=bool)
#    >>> np.abs(x-dct(idct(x)))<1e-14
#    array([ True,  True,  True,  True,  True], dtype=bool)
#
#    Reference
#    ---------
#    http://en.wikipedia.org/wiki/Discrete_cosine_transform
#    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
#    """
#
#    ifft = np.fft.ifft
#    x = np.atleast_1d(x)
#
#    if n is None:
#        n = x.shape[-1]
#
#    w = np.exp(1j * np.arange(n) * np.pi / (2 * n))
#
#    if x.shape[-1] < n:
#        n_shape = x.shape[:-1] + (n - x.shape[-1],)
#        xx = np.hstack((x, np.zeros(n_shape))) * w
#    else:
#        xx = x[..., :n] * w
#
#    real_x = np.all(np.isreal(x))
#    if (real_x and (np.remainder(n, 2) == 0)):
#        xx[..., 0] = xx[..., 0] * 0.5
#        yp = ifft(xx)
#        y = np.zeros(xx.shape, dtype=complex)
#        y[..., ::2] = yp[..., :n / 2]
#        y[..., ::-2] = yp[..., n / 2::]
#    else:
#        yp = ifft(np.hstack((xx, np.zeros_like(xx[..., 0]),
#                                        np.conj(xx[..., :0:-1]))))
#        y = yp[..., :n]
#
#    if real_x:
#        return y.real
#    else:
#        return y
#
# def dctn(y, axis=None, w=None):
#    '''
#    DCTN N-D discrete cosine transform.
#
#    Y = DCTN(X) returns the discrete cosine transform of X. The array Y is
#    the same size as X and contains the discrete cosine transform
#    coefficients. This transform can be inverted using IDCTN.
#
#    DCTN(X,axis) applies the DCTN operation across the dimension axis.
#
#    Class Support
#    -------------
#    Input array can be numeric or logical. The returned array is of class
#    double.
#
#    Reference
#    ---------
#    Narasimha M. et al, On the computation of the discrete cosine
#    transform, IEEE Trans Comm, 26, 6, 1978, pp 934-936.
#
#    Example
#    -------
#    RGB = imread('autumn.tif');
#    I = rgb2gray(RGB);
#    J = dctn(I);
#    imshow(log(abs(J)),[]), colormap(jet), colorbar
#
#    The commands below set values less than magnitude 10 in the DCT matrix
#    to zero, then reconstruct the image using the inverse DCT.
#
#        J(abs(J)<10) = 0;
#        K = idctn(J);
#        figure, imshow(I)
#        figure, imshow(K,[0 255])
#
#    See also
#    --------
#    idctn, dct, idct
#    '''
#
#    y = np.atleast_1d(y)
#    shape0 = y.shape
#
#
#    if axis is None:
# y = y.squeeze() # Working across singleton dimensions is useless
#    dimy = y.ndim
#    if dimy==1:
#        y = np.atleast_2d(y)
#        y = y.T
# Some modifications are required if Y is a vector
# if isvector(y):
# if y.shape[0]==1:
# if axis==0:
# return y, None
# elif axis==1:
# axis=0
##            y = y.T
# elif axis==1:
# return y, None
#
#    if w is None:
#        w = [0,] * dimy
#        for dim in range(dimy):
#            if axis is not None and dim!=axis:
#                continue
#            n = (dimy==1)*y.size + (dimy>1)*shape0[dim]
# w{dim} = exp(1i*(0:n-1)'*pi/2/n);
#            w[dim] = np.exp(1j * np.arange(n) * np.pi / (2 * n))
#
# --- DCT algorithm ---
#    if np.iscomplex(y).any():
#        y = dctn(np.real(y),axis,w) + 1j*dctn(np.imag(y),axis,w)
#    else:
#        for dim in range(dimy):
#            y = shiftdim(y,1)
#            if axis is not None and dim!=axis:
# y = shiftdim(y, 1)
#                continue
#            siz = y.shape
#            n = siz[-1]
#            y = y[...,np.r_[0:n:2, 2*int(n//2)-1:0:-2]]
#            y = y.reshape((-1,n))
#            y = y*np.sqrt(2*n);
#            y = (np.fft.ifft(y, n=n, axis=1) * w[dim]).real
#            y[:,0] = y[:,0]/np.sqrt(2)
#            y = y.reshape(siz)
#
# end
# end
#
#    return y.reshape(shape0), w
#
# def idctn(y, axis=None, w=None):
#    '''
#    IDCTN N-D inverse discrete cosine transform.
#       X = IDCTN(Y) inverts the N-D DCT transform, returning the original
#       array if Y was obtained using Y = DCTN(X).
#
#       IDCTN(X,DIM) applies the IDCTN operation across the dimension DIM.
#
#       Class Support
#       -------------
#       Input array can be numeric or logical. The returned array is of class
#       double.
#
#       Reference
#       ---------
#       Narasimha M. et al, On the computation of the discrete cosine
#       transform, IEEE Trans Comm, 26, 6, 1978, pp 934-936.
#
#       Example
#       -------
#           RGB = imread('autumn.tif');
#           I = rgb2gray(RGB);
#           J = dctn(I);
#           imshow(log(abs(J)),[]), colormap(jet), colorbar
#
#       The commands below set values less than magnitude 10 in the DCT matrix
#       to zero, then reconstruct the image using the inverse DCT.
#
#           J(abs(J)<10) = 0;
#           K = idctn(J);
#           figure, imshow(I)
#           figure, imshow(K,[0 255])
#
#       See also
#       --------
#       dctn, idct, dct
#
#       -- Damien Garcia -- 2009/04, revised 2009/11
#       website: <a
#       href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>
#
#     ----------
#       [Y,W] = IDCTN(X,DIM,W) uses and returns the weights which are used by
#       the program. If IDCTN is required for several large arrays of same
#       size, the weights can be reused to make the algorithm faster. A typical
#       syntax is the following:
#          w = [];
#          for k = 1:10
#              [y{k},w] = idctn(x{k},[],w);
#          end
#       The weights (w) are calculated during the first call of IDCTN then
#       reused in the next calls.
#    '''
#
#    y = np.atleast_1d(y)
#    shape0 = y.shape
#
#    if axis is None:
# y = y.squeeze() # Working across singleton dimensions is useless
#
#    dimy = y.ndim
#    if dimy==1:
#        y = np.atleast_2d(y)
#        y = y.T
# Some modifications are required if Y is a vector
# if isvector(y):
# if y.shape[0]==1:
# if axis==0:
# return y, None
# elif axis==1:
# axis=0
##            y = y.T
# elif axis==1:
# return y, None
##
#
#
#    if w is None:
#        w = [0,] * dimy
#        for dim in range(dimy):
#            if axis is not None and dim!=axis:
#                continue
#            n = (dimy==1)*y.size + (dimy>1)*shape0[dim]
# w{dim} = exp(1i*(0:n-1)'*pi/2/n);
#            w[dim] = np.exp(1j * np.arange(n) * np.pi / (2 * n))
# --- IDCT algorithm ---
#    if np.iscomplex(y).any():
#        y = np.complex(idctn(np.real(y),axis,w),idctn(np.imag(y),axis,w))
#    else:
#        for dim in range(dimy):
#            y = shiftdim(y,1)
#            if axis is not None and dim!=axis:
# y = shiftdim(y, 1)
#                continue
#            siz = y.shape
#            n = siz[-1]
#
#            y = y.reshape((-1,n)) * w[dim]
#            y[:,0] = y[:,0]/np.sqrt(2)
#            y = (np.fft.ifft(y, n=n, axis=1)).real
#            y = y * np.sqrt(2*n)
#
#            I = np.empty(n,dtype=int)
#            I.put(np.r_[0:n:2],np.r_[0:int(n//2)+np.remainder(n,2)])
#            I.put(np.r_[1:n:2],np.r_[n-1:int(n//2)-1:-1])
#            y = y[:,I]
#
#            y = y.reshape(siz)
#
#
#    y = y.reshape(shape0);
#    return y, w


def no_leading_ones(x):
    first = 0
    for i, xi in enumerate(x):
        if xi != 1:
            first = i
            break
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


def test_dctn():
    a = np.arange(12)  # .reshape((3,-1))
    print('a = ', a)
    print(' ')
    y = dct(a)
    x = idct(y)
    print('y = dct(a)')
    print(y)
    print('x = idct(y)')
    print(x)
    print(' ')

#    y1 = dct1(a)
#    x1 = idct1(y)
#    print('y1 = dct1(a)')
#    print(y1)
#    print('x1 = idct1(y)')
#    print(x1)
#    print(' ')

    yn = dctn(a)
    xn = idctn(yn)
    print('yn = dctn(a)')
    print(yn)
    print('xn = idctn(yn)')
    print(xn)
    print(' ')

#    yn1 = dctn1(a)
#    xn1 = idctn1(yn1)
#    print('yn1 = dctn1(a)')
#    print(yn1)
#    print('xn1 = idctn1(yn)')
#    print(xn1)


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    test_docstrings()
    # test_dctn()
