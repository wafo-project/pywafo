import numpy as np
__all__ = ['dct', 'idct', 'dctn', 'idctn']
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

def dctn(y, axis=None, w=None):
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

    y = np.atleast_1d(y)
    shape0 = y.shape
    
    
    if axis is None:
        y = y.squeeze() # Working across singleton dimensions is useless
    dimy = y.ndim
    if dimy==1:
        y = np.atleast_2d(y)
        y = y.T
    # Some modifications are required if Y is a vector
#    if isvector(y):
#        if y.shape[0]==1:
#            if axis==0: 
#                return y, None
#            elif axis==1: 
#                axis=0    
#            y = y.T
#        elif axis==1: 
#            return y, None 

    if w is None:
        w = [0,] * dimy
        for dim in range(dimy):
            if axis is not None and dim!=axis:
                continue
            n = (dimy==1)*y.size + (dimy>1)*shape0[dim]
            #w{dim} = exp(1i*(0:n-1)'*pi/2/n);
            w[dim] = np.exp(1j * np.arange(n) * np.pi / (2 * n))
    
    # --- DCT algorithm ---
    if np.iscomplex(y).any():
        y = np.complex(dctn(np.real(y),axis,w),dctn(np.imag(y),axis,w))
    else:
        for dim in range(dimy):
            y = shiftdim(y,1)
            if axis is not None and dim!=axis:
                y = shiftdim(y, 1)
                continue
            siz = y.shape 
            n = siz[-1]
            y = y[...,np.r_[0:n:2, 2*int(n//2)-1:0:-2]]
            y = y.reshape((-1,n))
            y = y*np.sqrt(2*n);
            y = (np.fft.ifft(y, n=n, axis=1) * w[dim]).real
            y[:,0] = y[:,0]/np.sqrt(2)
            y = y.reshape(siz)
            
        #end
    #end
            
    return y.reshape(shape0), w

def idctn(y, axis=None, w=None):
    '''
    IDCTN N-D inverse discrete cosine transform.
       X = IDCTN(Y) inverts the N-D DCT transform, returning the original
       array if Y was obtained using Y = DCTN(X).
    
       IDCTN(X,DIM) applies the IDCTN operation across the dimension DIM.
    
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
       dctn, idct, dct 
    
       -- Damien Garcia -- 2009/04, revised 2009/11
       website: <a
       href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>
    
     ----------
       [Y,W] = IDCTN(X,DIM,W) uses and returns the weights which are used by
       the program. If IDCTN is required for several large arrays of same
       size, the weights can be reused to make the algorithm faster. A typical
       syntax is the following:
          w = [];
          for k = 1:10
              [y{k},w] = idctn(x{k},[],w);
          end
       The weights (w) are calculated during the first call of IDCTN then
       reused in the next calls.
    '''

    y = np.atleast_1d(y)
    shape0 = y.shape
    
    if axis is None:
        y = y.squeeze() # Working across singleton dimensions is useless
    
    dimy = y.ndim
    if dimy==1:
        y = np.atleast_2d(y)
        y = y.T
    # Some modifications are required if Y is a vector
#    if isvector(y):
#        if y.shape[0]==1:
#            if axis==0: 
#                return y, None
#            elif axis==1: 
#                axis=0    
#            y = y.T
#        elif axis==1: 
#            return y, None 
#        
    
    
    if w is None:
        w = [0,] * dimy
        for dim in range(dimy):
            if axis is not None and dim!=axis:
                continue
            n = (dimy==1)*y.size + (dimy>1)*shape0[dim]
            #w{dim} = exp(1i*(0:n-1)'*pi/2/n);
            w[dim] = np.exp(1j * np.arange(n) * np.pi / (2 * n))
    # --- IDCT algorithm ---
    if np.iscomplex(y).any():
        y = np.complex(idctn(np.real(y),axis,w),idctn(np.imag(y),axis,w))
    else:
        for dim in range(dimy):
            y = shiftdim(y,1)
            if axis is not None and dim!=axis:
                #y = shiftdim(y, 1)
                continue
            siz = y.shape 
            n = siz[-1]
            
            y = y.reshape((-1,n)) * w[dim]
            y[:,0] = y[:,0]/np.sqrt(2)
            y = (np.fft.ifft(y, n=n, axis=1)).real
            y = y * np.sqrt(2*n)
            
            I = np.empty(n,dtype=int)
            I.put(np.r_[0:n:2],np.r_[0:int(n//2)+np.remainder(n,2)])
            I.put(np.r_[1:n:2],np.r_[n-1:int(n//2)-1:-1])
            y = y[:,I]
            
            y = y.reshape(siz)
            
        
    y = y.reshape(shape0);
    return y, w



def no_leading_ones(x):
    first = 0
    for i, xi in enumerate(x):
        if xi != 1:
            first = i
            break
    return x[first:]
    
def shiftdim(x, n=None):  
    if n is None:  
        # returns the array B with the same number of  
        # elements as X but with any leading singleton  
        # dimensions removed.   
        return x.reshape(no_leading_ones(x.shape))  
    elif n>=0:  
        # When n is positive, shiftdim shifts the dimensions  
        # to the left and wraps the n leading dimensions to the end.  
        return x.transpose(np.roll(range(x.ndim), -n))  
    else:  
        # When n is negative, shiftdim shifts the dimensions  
        # to the right and pads with singletons.  
        return x.reshape((1,)*-n+x.shape)  

def test_dctn():
    a = np.arange(12).reshape((3,-1))
    #y = dct(a)
    #x = idct(y)
    #print(y)
    #print(x)
    
    print(a)
    yn = dctn(a)[0]
    xn = idctn(yn)[0]
    
    print(yn)
    print(xn)
    
    
          
def test_docstrings():
    import doctest
    doctest.testmod()
    
if __name__ == '__main__':
    #test_docstrings()
    test_dctn() 