import numpy as np
#from math import pow
#from numpy import zeros,dot
from numpy import abs, size, convolve, linalg, concatenate #@UnresolvedImport

__all__ = ['calc_coeff', 'smooth', 'smooth_last']


def calc_coeff(n, degree, diff_order=0):
    """ calculates filter coefficients for symmetric savitzky-golay filter.
        see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

        n   means that 2*n+1 values contribute to the
                     smoother.

        degree   is degree of fitting polynomial

        diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first
                                                 derivative of function.
                     and so on ...

    """
    order_range = np.arange(degree + 1) 
    k_range = np.arange(-n, n + 1, dtype=float).reshape(-1, 1)
    b = np.mat(k_range ** order_range)   
    #b = np.mat([[float(k)**i for i in order_range] for k in range(-n,n+1)])
    coeff = linalg.pinv(b).A[diff_order]
    return coeff

def smooth_last(signal, coeff, k=0):
    n = size(coeff - 1) // 2
    y = np.squeeze(signal)
    if y.ndim > 1:
        coeff.shape = (-1, 1)
    first_vals = y[0] - abs(y[n:0:-1] - y[0])
    last_vals = y[-1] + abs(y[-2:-n - 2:-1] - y[-1]) 
    y = concatenate((first_vals, y, last_vals))
    return (y[-2 * n - 1 - k:-k] * coeff).sum(axis=0)
    
         
def smooth(signal, coeff, pad=True):

    """ applies coefficients calculated by calc_coeff()
        to signal """

    n = size(coeff - 1) // 2
    y = np.squeeze(signal)
    if pad:
        first_vals = y[0] - abs(y[n:0:-1] - y[0])
        last_vals = y[-1] + abs(y[-2:-n - 2:-1] - y[-1]) 
        y = concatenate((first_vals, y, last_vals))
        n *= 2 
    d = y.ndim
    if d > 1:
        y1 = y.reshape(y.shape[0], -1)
        res = []
        for i in range(y1.shape[1]):
            res.append(convolve(y1[:, i], coeff)[n:-n])
        res = np.asarray(res).T
    else:
        res = convolve(y, coeff)[n:-n]
    return res

class SavitzkyGolay(object):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    
    Parameters
    ----------
    n : int
        the size of the smoothing window is 2*n+1. 
    degree : int
        the order of the polynomial used in the filtering.
        Must be less than `window_size` - 1, i.e, less than 2*n.
    diff_order : int
        the order of the derivative to compute (default = 0 means only smoothing)
        0 means that filter results in smoothing of function
        1 means that filter results in smoothing the first derivative of function.
        and so on ...
        
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    
    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape) 
    >>> ysg = SavitzkyGolay(n=15, degree=4).smooth(y)
    >>> import matplotlib.pyplot as plt
    >>> hy = plt.plot(t, y, label='Noisy signal')
    >>> h = plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> h = plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> h = plt.legend()
    >>> plt.show()
    
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    def __init__(self, n, degree=1, diff_order=0):
        self.n = n
        self.degree = degree
        self.diff_order = diff_order
        self.calc_coeff()
        
    def calc_coeff(self):
        """ calculates filter coefficients for symmetric savitzky-golay filter.
        """
        n = self.n
        order_range = np.arange(self.degree + 1) 
        k_range = np.arange(-n, n + 1, dtype=float).reshape(-1, 1)
        b = np.mat(k_range ** order_range)   
        #b = np.mat([[float(k)**i for i in order_range] for k in range(-n,n+1)])
        self._coeff = linalg.pinv(b).A[self.diff_order]
    def smooth_last(self, signal, k=0):
        coeff = self._coeff
        n = size(coeff - 1) // 2
        y = np.squeeze(signal)
        if y.ndim > 1:
            coeff.shape = (-1, 1)
        first_vals = y[0] - abs(y[n:0:-1] - y[0])
        last_vals = y[-1] + abs(y[-2:-n - 2:-1] - y[-1]) 
        y = concatenate((first_vals, y, last_vals))
        return (y[-2 * n - 1 - k:-k] * coeff).sum(axis=0)
    
         
    def smooth(self, signal, pad=True):
        """
        Returns smoothed signal (or it's n-th derivative).
        
        Parameters
        ---------- 
        y : array_like, shape (N,)
            the values of the time history of the signal.
        pad : bool
           pad first and last values to lessen the end effects.
            
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        """
        coeff = self._coeff
        n = size(coeff - 1) // 2
        y = np.squeeze(signal)
        if pad:
            first_vals = y[0] - abs(y[n:0:-1] - y[0])
            last_vals = y[-1] + abs(y[-2:-n - 2:-1] - y[-1]) 
            y = concatenate((first_vals, y, last_vals))
            n *= 2 
        d = y.ndim
        if d > 1:
            y1 = y.reshape(y.shape[0], -1)
            res = []
            for i in range(y1.shape[1]):
                res.append(convolve(y1[:, i], coeff)[n:-n])
            res = np.asarray(res).T
        else:
            res = convolve(y, coeff)[n:-n]
        return res

class Kalman(object): 
    '''                   
    Kalman filter object - updates a system state vector estimate based upon an
              observation, using a discrete Kalman filter.
    
    The Kalman filter is "optimal" under a variety of
    circumstances.  An excellent paper on Kalman filtering at 
    the introductory level, without detailing the mathematical 
    underpinnings, is:
    
    "An Introduction to the Kalman Filter"
    Greg Welch and Gary Bishop, University of North Carolina
    http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
    
    PURPOSE:
    The purpose of each iteration of a Kalman filter is to update
    the estimate of the state vector of a system (and the covariance
    of that vector) based upon the information in a new observation.
    The version of the Kalman filter in this function assumes that
    observations occur at fixed discrete time intervals. Also, this
    function assumes a linear system, meaning that the time evolution
    of the state vector can be calculated by means of a state transition
    matrix.
    
    USAGE:
    filt = Kalman(R, x, P, A, B=0, u=0, Q, H)
    x = filt(z)
    
    filt is a "system" object containing various fields used as input
    and output. The state estimate "x" and its covariance "P" are
    updated by the function. The other fields describe the mechanics
    of the system and are left unchanged. A calling routine may change
    these other fields as needed if state dynamics are time-dependent;
    otherwise, they should be left alone after initial values are set.
    The exceptions are the observation vector "z" and the input control
    (or forcing function) "u." If there is an input function, then
    "u" should be set to some nonzero value by the calling routine.
    
    System dynamics
    ---------------
    
    The system evolves according to the following difference equations,
    where quantities are further defined below:
    
    x = Ax + Bu + w  meaning the state vector x evolves during one time
                     step by premultiplying by the "state transition
                     matrix" A. There is optionally (if nonzero) an input
                     vector u which affects the state linearly, and this
                     linear effect on the state is represented by
                     premultiplying by the "input matrix" B. There is also
                     gaussian process noise w.
    z = Hx + v       meaning the observation vector z is a linear function
                     of the state vector, and this linear relationship is
                     represented by premultiplication by "observation
                     matrix" H. There is also gaussian measurement
                     noise v.
    where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
          v ~ N(0,R) meaning v is gaussian noise with covariance R
    
    VECTOR VARIABLES:
    
    s.x = state vector estimate. In the input struct, this is the
          "a priori" state estimate (prior to the addition of the
          information from the new observation). In the output struct,
          this is the "a posteriori" state estimate (after the new
          measurement information is included).
    s.z = observation vector
    s.u = input control vector, optional (defaults to zero).
    
    MATRIX VARIABLES:
    
    s.A = state transition matrix (defaults to identity).
    s.P = covariance of the state vector estimate. In the input struct,
          this is "a priori," and in the output it is "a posteriori."
          (required unless autoinitializing as described below).
    s.B = input matrix, optional (defaults to zero).
    s.Q = process noise covariance (defaults to zero).
    s.R = measurement noise covariance (required).
    s.H = observation matrix (defaults to identity).
    
    NORMAL OPERATION:
    
    (1) define all state definition fields: A,B,H,Q,R
    (2) define intial state estimate: x,P
    (3) obtain observation and control vectors: z,u
    (4) call the filter to obtain updated state estimate: x,P
    (5) return to step (3) and repeat
    
    INITIALIZATION:
    
    If an initial state estimate is unavailable, it can be obtained
    from the first observation as follows, provided that there are the
    same number of observable variables as state variables. This "auto-
    intitialization" is done automatically if s.x is absent or NaN.
    
    x = inv(H)*z
    P = inv(H)*R*inv(H')
    
    This is mathematically equivalent to setting the initial state estimate
    covariance to infinity.
    
    Example (Automobile Voltimeter):
    -------
    # Define the system as a constant of 12 volts:
    >>> V0 = 12
    >>> h = 1      # voltimeter measure the voltage itself
    >>> q = 1e-5   # variance of process noise s the car operates
    >>> r = 0.1**2 # variance of measurement error
    >>> b = 0      # no system input
    >>> u = 0      # no system input
    >>> filt = Kalman(R=r, A=1, Q=q, H=h, B=b, u=u)
    
    # Generate random voltages and watch the filter operate.
    >>> n = 50
    >>> truth = np.random.randn(n)*np.sqrt(q) + V0
    >>> z = truth + np.random.randn(n)*np.sqrt(r) # measurement
    >>> x = np.zeros(n)
    
    >>> for i, zi in enumerate(z):
    ...    x[i] = filt(zi) #  perform a Kalman filter iteration
    
    >>> import matplotlib.pyplot as plt
    >>> hz = plt.plot(z,'r.', label='observations')
    >>> hx = plt.plot(x,'b-', label='Kalman output')   # a-posteriori state estimates:
    >>> ht = plt.plot(truth,'g-', label='true voltage')
    >>> h = plt.legend() 
    >>> h = plt.title('Automobile Voltimeter Example')
    
    '''

    def __init__(self, R, x=None, P=None, A=None, B=0, u=0, Q=None, H=None):
        self.R = R
        self.x = x
        self.P = P
        self.u = u
        self.A = A
        self.B = B
        self.Q = Q
        self.H = H
        self.reset()
        
    def reset(self):
        self._filter = self._filter_first
    
    def _filter_first(self, z):
        
        self._filter = self._filter_main
        
        auto_init = self.x is None
        if auto_init: 
            n = np.size(z)
        else:
            n = np.size(self.x)
        if self.A is None:
            self.A = np.eye(n, n)
        self.A = np.atleast_2d(self.A)
        if self.Q is None:
            self.Q = np.zeros((n, n))
        self.Q = np.atleast_2d(self.Q)
        if self.H is None:
            self.H = np.eye(n, n)
        self.H = np.atleast_2d(self.H)
#        if np.diff(np.shape(self.H)):
#            raise ValueError('Observation matrix must be square and invertible for state autointialization.')
        HI = np.linalg.inv(self.H)
        if self.P is None:
            self.P = np.dot(np.dot(HI, self.R), HI.T) 
        self.P = np.atleast_2d(self.P)
        if auto_init:
            #initialize state estimate from first observation
            self.x = np.dot(HI, z)
            return self.x
        else:
            return self._filter_main(z)

   
    def _filter_main(self, z):
        ''' This is the code which implements the discrete Kalman filter:
        '''
        A = self.A
        H = self.H
        P = self.P
 
        # Prediction for state vector and covariance:
        x = np.dot(A, self.x) + np.dot(self.B, self.u)
        P = np.dot(np.dot(A, P), A.T) + self.Q
    
        # Compute Kalman gain factor:
        PHT = np.dot(P, H.T)
        K = np.dot(PHT, np.linalg.inv(np.dot(H, PHT) + self.R))
    
        # Correction based on observation:
        self.x = x + np.dot(K, z - np.dot(H, x))
        self.P = P - np.dot(K, np.dot(H, P))
       
        # Note that the desired result, which is an improved estimate
        # of the system state vector x and its covariance P, was obtained
        # in only five lines of code, once the system was defined. (That's
        # how simple the discrete Kalman filter is to use.) Later,
        # we'll discuss how to deal with nonlinear systems.

        
        return self.x
    def __call__(self, z):
        return self._filter(z)
    
def test_kalman():
    V0 = 12
    h = np.atleast_2d(1) # voltimeter measure the voltage itself
    q = 1e-9 # variance of process noise as the car operates
    r = 0.05 ** 2 # variance of measurement error
    b = 0 # no system input
    u = 0 # no system input
    filt = Kalman(R=r, A=1, Q=q, H=h, B=b, u=u)
    
    # Generate random voltages and watch the filter operate.
    n = 50
    truth = np.random.randn(n) * np.sqrt(q) + V0
    z = truth + np.random.randn(n) * np.sqrt(r) # measurement
    x = np.zeros(n)
    
    for i, zi in enumerate(z):
        x[i] = filt(zi) #  perform a Kalman filter iteration
    
    import matplotlib.pyplot as plt
    _hz = plt.plot(z, 'r.', label='observations')
    _hx = plt.plot(x, 'b-', label='Kalman output')   # a-posteriori state estimates:
    _ht = plt.plot(truth, 'g-', label='true voltage')
    plt.legend() 
    plt.title('Automobile Voltimeter Example')
    plt.show()


def test_smooth():
    import matplotlib.pyplot as plt
    t = np.linspace(-4, 4, 500)
    y = np.exp(-t ** 2) + np.random.normal(0, 0.05, t.shape)
    coeff = calc_coeff(num_points=3, degree=2, diff_order=0)
    ysg = smooth(y, coeff, pad=True)
    
    plt.plot(t, y, t, ysg, '--')
    plt.show()
    
def test_docstrings():
    import doctest
    doctest.testmod()
if __name__ == '__main__':
    test_docstrings()
    #test_kalman()
    #test_smooth()

