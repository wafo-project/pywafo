import numpy as np
__all__ = ['dct', 'idct']
def dct(x, n=None):
    """
    Discrete Cosine Transform

                      N-1
           y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                      n=0

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.abs(x-idct(dct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(x-dct(idct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)

    Reference
    ---------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
    """
    fft = np.fft.fft
    x = np.atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    if x.shape[-1] < n:
        n_shape = x.shape[:-1] + (n - x.shape[-1],)
        xx = np.hstack((x, np.zeros(n_shape)))
    else:
        xx = x[..., :n]

    real_x = np.all(np.isreal(xx))
    if (real_x and (np.remainder(n, 2) == 0)):
        xp = 2 * fft(np.hstack((xx[..., ::2], xx[..., ::-2])))
    else:
        xp = fft(np.hstack((xx, xx[..., ::-1])))
        xp = xp[..., :n]

    w = np.exp(-1j * np.arange(n) * np.pi / (2 * n))

    y = xp * w

    if real_x:
        return y.real
    else:
        return y

def idct(x, n=None):
    """
    Inverse Discrete Cosine Transform

                       N-1
           x[k] = 1/N sum w[n]*y[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                       n=0

           w(0) = 1/2
           w(n) = 1 for n>0

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.abs(x-idct(dct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(x-dct(idct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)

    Reference
    ---------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
    """

    ifft = np.fft.ifft
    x = np.atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    w = np.exp(1j * np.arange(n) * np.pi / (2 * n))

    if x.shape[-1] < n:
        n_shape = x.shape[:-1] + (n - x.shape[-1],)
        xx = np.hstack((x, np.zeros(n_shape))) * w
    else:
        xx = x[..., :n] * w

    real_x = np.all(np.isreal(x))
    if (real_x and (np.remainder(n, 2) == 0)):
        xx[..., 0] = xx[..., 0] * 0.5
        yp = ifft(xx)
        y = np.zeros(xx.shape, dtype=complex)
        y[..., ::2] = yp[..., :n / 2]
        y[..., ::-2] = yp[..., n / 2::]
    else:
        yp = ifft(np.hstack((xx, np.zeros_like(xx[..., 0]), np.conj(xx[..., :0:-1]))))
        y = yp[..., :n]

    if real_x:
        return y.real
    else:
        return y
