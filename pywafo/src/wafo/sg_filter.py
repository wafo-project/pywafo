import numpy as np
#from math import pow
#from numpy import zeros,dot
from numpy import abs, size, convolve, linalg, concatenate  # @UnresolvedImport
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, expm
from scipy.signal import medfilt

__all__ = ['calc_coeff', 'smooth', 'smooth_last',
           'SavitzkyGolay', 'Kalman', 'HodrickPrescott']


def calc_coeff(n, degree, diff_order=0):
    """ calculates filter coefficients for symmetric savitzky-golay filter.
        see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

        n   means that 2*n+1 values contribute to the smoother.

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
    """applies coefficients calculated by calc_coeff() to signal."""

    n = size(coeff - 1) // 2
    y = np.squeeze(signal)
    if n == 0:
        return y
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
    r"""Smooth and optionally differentiate data with a Savitzky-Golay filter.

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
        order of the derivative to compute (default = 0 means only smoothing)
        0 means that filter results in smoothing of function
        1 means that filter results in smoothing the first derivative of the
          function and so on ...

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
    >>> ysg = SavitzkyGolay(n=20, degree=2).smooth(y)
    >>> import matplotlib.pyplot as plt
    >>> h = plt.plot(t, y, label='Noisy signal')
    >>> h1 = plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> h2 = plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> h3 = plt.legend()
    >>> h4 = plt.title('Savitzky-Golay')
    plt.show()

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
        #b =np.mat([[float(k)**i for i in order_range] for k in range(-n,n+1)])
        self._coeff = linalg.pinv(b).A[self.diff_order]

    def smooth_last(self, signal, k=0):
        coeff = self._coeff
        n = size(coeff - 1) // 2
        y = np.squeeze(signal)
        if n == 0:
            return y
        if y.ndim > 1:
            coeff.shape = (-1, 1)
        first_vals = y[0] - abs(y[n:0:-1] - y[0])
        last_vals = y[-1] + abs(y[-2:-n - 2:-1] - y[-1])
        y = concatenate((first_vals, y, last_vals))
        return (y[-2 * n - 1 - k:-k] * coeff).sum(axis=0)

    def __call__(self, signal):
        return self.smooth(signal)

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
        if n == 0:
            return y
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


class HodrickPrescott(object):

    '''Smooth data with a Hodrick-Prescott filter.

    The Hodrick-Prescott filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameter
    ---------
    w : real scalar
        smooting parameter. Larger w means more smoothing. Values usually
        in the [100, 20000] interval. As w approach infinity H-P will approach
        a line.

    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = HodrickPrescott(w=10000)(y)
    >>> import matplotlib.pyplot as plt
    >>> h = plt.plot(t, y, label='Noisy signal')
    >>> h1 = plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> h2 = plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> h3 = plt.legend()
    >>> h4 = plt.title('Hodrick-Prescott')
    >>> plt.show()

    References
    ----------
    .. [1] E. T. Whittaker, On a new method of graduation. In proceedings of
        the Edinburgh Mathematical association., 1923, 78, pp 88-89.
    .. [2] R. Hodrick and E. Prescott, Postwar U.S. business cycles: an
        empirical investigation,
        Journal of money, credit and banking, 1997, 29 (1), pp 1-16.
    .. [3] Kim Hyeongwoo, Hodrick-Prescott filter,
        2004, www.auburn.edu/~hzk0001/hpfilter.pdf
    '''

    def __init__(self, w=100):
        self.w = w

    def _get_matrix(self, n):
        w = self.w
        diag_matrix = np.repeat(
            np.atleast_2d([w, -4 * w, 6 * w + 1, -4 * w, w]).T, n, axis=1)
        A = spdiags(diag_matrix, np.arange(-2, 2 + 1), n, n).tocsr()
        A[0, 0] = A[-1, -1] = 1 + w
        A[1, 1] = A[-2, -2] = 1 + 5 * w
        A[0, 1] = A[1, 0] = A[-2, -1] = A[-1, -2] = -2 * w
        return A

    def __call__(self, x):
        x = np.atleast_1d(x).flatten()
        n = len(x)
        if n < 4:
            return x.copy()

        A = self._get_matrix(n)
        return spsolve(A, x)


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
    filt = Kalman(R, x, P, A, B=0, Q, H)
    x = filt(z, u=0)

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
    z = observation vector
    u = input control vector, optional (defaults to zero).

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
    >>> filt = Kalman(R=r, A=1, Q=q, H=h, B=b)

    # Generate random voltages and watch the filter operate.
    >>> n = 50
    >>> truth = np.random.randn(n)*np.sqrt(q) + V0
    >>> z = truth + np.random.randn(n)*np.sqrt(r) # measurement
    >>> x = np.zeros(n)

    >>> for i, zi in enumerate(z):
    ...     x[i] = filt(zi, u) #  perform a Kalman filter iteration

    >>> import matplotlib.pyplot as plt
    >>> hz = plt.plot(z,'r.', label='observations')

    # a-posteriori state estimates:
    >>> hx = plt.plot(x,'b-', label='Kalman output')
    >>> ht = plt.plot(truth,'g-', label='true voltage')
    >>> h = plt.legend()
    >>> h1 = plt.title('Automobile Voltimeter Example')
    >>> plt.show()

    '''

    def __init__(self, R, x=None, P=None, A=None, B=0, Q=None, H=None):
        self.R = R  # Estimated error in measurements.
        self.x = x  # Initial state estimate.
        self.P = P  # Initial covariance estimate.
        self.A = A  # State transition matrix.
        self.B = B   # Control matrix.
        self.Q = Q  # Estimated error in process.
        self.H = H  # Observation matrix.
        self.reset()

    def reset(self):
        self._filter = self._filter_first

    def _filter_first(self, z, u):

        self._filter = self._filter_main

        auto_init = self.x is None
        if auto_init:
            n = np.size(z)
        else:
            n = np.size(self.x)
        if self.A is None:
            self.A = np.eye(n)
        self.A = np.atleast_2d(self.A)
        if self.Q is None:
            self.Q = np.zeros((n, n))
        self.Q = np.atleast_2d(self.Q)
        if self.H is None:
            self.H = np.eye(n)
        self.H = np.atleast_2d(self.H)
        try:
            HI = np.linalg.inv(self.H)
        except:
            HI = np.eye(n)
        if self.P is None:
            self.P = np.dot(np.dot(HI, self.R), HI.T)

        self.P = np.atleast_2d(self.P)
        if auto_init:
            # initialize state estimate from first observation
            self.x = np.dot(HI, z)
            return self.x
        else:
            return self._filter_main(z, u)

    def _predict_state(self, x, u):
        return np.dot(self.A, x) + np.dot(self.B, u)

    def _predict_covariance(self, P):
        A = self.A
        return np.dot(np.dot(A, P), A.T) + self.Q

    def _compute_gain(self, P):
        """Kalman gain factor."""
        H = self.H
        PHT = np.dot(P, H.T)
        innovation_covariance = np.dot(H, PHT) + self.R
        #return np.linalg.solve(PHT, innovation_covariance)
        return np.dot(PHT, np.linalg.inv(innovation_covariance))

    def _update_state_from_observation(self, x, z, K):
        innovation = z - np.dot(self.H, x)
        return x + np.dot(K, innovation)

    def _update_covariance(self, P, K):
        return P - np.dot(K, np.dot(self.H, P))
        return np.dot(np.eye(len(P)) - K * self.H, P)

    def _filter_main(self, z, u):
        ''' This is the code which implements the discrete Kalman filter:
        '''
        P = self._predict_covariance(self.P)
        x = self._predict_state(self.x, u)

        K = self._compute_gain(P)

        self.P = self._update_covariance(P, K)
        self.x = self._update_state_from_observation(x, z, K)

        return self.x

    def __call__(self, z, u=0):
        return self._filter(z, u)


def test_kalman():
    V0 = 12
    h = np.atleast_2d(1)  # voltimeter measure the voltage itself
    q = 1e-9  # variance of process noise as the car operates
    r = 0.05 ** 2  # variance of measurement error
    b = 0  # no system input
    u = 0  # no system input
    filt = Kalman(R=r, A=1, Q=q, H=h, B=b)

    # Generate random voltages and watch the filter operate.
    n = 50
    truth = np.random.randn(n) * np.sqrt(q) + V0
    z = truth + np.random.randn(n) * np.sqrt(r)  # measurement
    x = np.zeros(n)

    for i, zi in enumerate(z):
        x[i] = filt(zi, u)  # perform a Kalman filter iteration

    import matplotlib.pyplot as plt
    _hz = plt.plot(z, 'r.', label='observations')
    # a-posteriori state estimates:
    _hx = plt.plot(x, 'b-', label='Kalman output')
    _ht = plt.plot(truth, 'g-', label='true voltage')
    plt.legend()
    plt.title('Automobile Voltimeter Example')
    plt.show('hold')


def lti_disc(F, L=None, Q=None, dt=1):
    '''
    LTI_DISC  Discretize LTI ODE with Gaussian Noise

    Syntax:
     [A,Q] = lti_disc(F,L,Qc,dt)

    In:
     F  - NxN Feedback matrix
     L  - NxL Noise effect matrix        (optional, default identity)
     Qc - LxL Diagonal Spectral Density  (optional, default zeros)
     dt - Time Step                      (optional, default 1)

    Out:
     A - Transition matrix
     Q - Discrete Process Covariance

    Description:
     Discretize LTI ODE with Gaussian Noise. The original
     ODE model is in form

       dx/dt = F x + L w,  w ~ N(0,Qc)

     Result of discretization is the model

       x[k] = A x[k-1] + q, q ~ N(0,Q)

     Which can be used for integrating the model
     exactly over time steps, which are multiples
     of dt.
     '''
    n = np.shape(F)[0]
    if L is None:
        L = np.eye(n)

    if Q is None:
        Q = np.zeros((n, n))
    # Closed form integration of transition matrix
    A = expm(F * dt)

    # Closed form integration of covariance
    # by matrix fraction decomposition

    Phi = np.vstack((np.hstack((F, np.dot(np.dot(L, Q), L.T))),
                     np.hstack((np.zeros((n, n)), -F.T))))
    AB = np.dot(expm(Phi * dt), np.vstack((np.zeros((n, n)), np.eye(n))))
    #Q = AB[:n, :] / AB[n:(2 * n), :]
    Q = np.linalg.solve(AB[n:(2 * n), :].T, AB[:n, :].T)
    return A, Q


def test_kalman_sine():
    '''Kalman Filter demonstration with sine signal.'''
    sd = 1.
    dt = 0.1
    w = 1
    T = np.arange(0, 30 + dt / 2, dt)
    n = len(T)
    X = np.sin(w * T)
    Y = X + sd * np.random.randn(n)

    ''' Initialize KF to values
       x = 0
       dx/dt = 0
     with great uncertainty in derivative
    '''
    M = np.zeros((2, 1))
    P = np.diag([0.1, 2])
    R = sd ** 2
    H = np.atleast_2d([1, 0])
    q = 0.1
    F = np.atleast_2d([[0, 1],
                       [0, 0]])
    A, Q = lti_disc(F, L=None, Q=np.diag([0, q]), dt=dt)

    # Track and animate
    m = M.shape[0]
    MM = np.zeros((m, n))
    PP = np.zeros((m, m, n))
    '''In this demonstration we estimate a stationary sine signal from noisy
    measurements by using the classical Kalman filter.'
    '''
    filt = Kalman(R=R, x=M, P=P, A=A, Q=Q, H=H, B=0)

    # Generate random voltages and watch the filter operate.
    #n = 50
    #truth = np.random.randn(n) * np.sqrt(q) + V0
    #z = truth + np.random.randn(n) * np.sqrt(r)  # measurement
    truth = X
    z = Y
    x = np.zeros((n, m))

    for i, zi in enumerate(z):
        x[i] = filt(zi, u=0).ravel()

    import matplotlib.pyplot as plt
    _hz = plt.plot(z, 'r.', label='observations')
    # a-posteriori state estimates:
    _hx = plt.plot(x[:, 0], 'b-', label='Kalman output')
    _ht = plt.plot(truth, 'g-', label='true voltage')
    plt.legend()
    plt.title('Automobile Voltimeter Example')
    plt.show()

#     for k in range(m):
#         [M,P] = kf_predict(M,P,A,Q);
#         [M,P] = kf_update(M,P,Y(k),H,R);
#
#         MM(:,k) = M;
#         PP(:,:,k) = P;
#
#       %
#       % Animate
#       %
#       if rem(k,10)==1
#         plot(T,X,'b--',...
#              T,Y,'ro',...
#              T(k),M(1),'k*',...
#              T(1:k),MM(1,1:k),'k-');
#         legend('Real signal','Measurements','Latest estimate','Filtered estimate')
#         title('Estimating a noisy sine signal with Kalman filter.');
#         drawnow;
#
#         pause;
#       end
#     end
#
#     clc;
#     disp('In this demonstration we estimate a stationary sine signal from noisy measurements by using the classical Kalman filter.');
#     disp(' ');
#     disp('The filtering results are now displayed sequantially for 10 time step at a time.');
#     disp(' ');
#     disp('<push any key to see the filtered and smoothed results together>')
#     pause;
#     %
#     % Apply Kalman smoother
#     %
#     SM = rts_smooth(MM,PP,A,Q);
#     plot(T,X,'b--',...
#          T,MM(1,:),'k-',...
#          T,SM(1,:),'r-');
#     legend('Real signal','Filtered estimate','Smoothed estimate') 
#     title('Filtered and smoothed estimate of the original signal');
#     
#     clc;
#     disp('The filtered and smoothed estimates of the signal are now displayed.')
#     disp(' ');
#     disp('RMS errors:');
#     %
#     % Errors
#     %
#     fprintf('KF = %.3f\nRTS = %.3f\n',...
#             sqrt(mean((MM(1,:)-X(1,:)).^2)),...
#             sqrt(mean((SM(1,:)-X(1,:)).^2)));


class HampelFilter(object):
    '''
    Hampel Filter.

    HAMPEL(X,Y,DX,T,varargin) returns the Hampel filtered values of the
    elements in Y. It was developed to detect outliers in a time series,
    but it can also be used as an alternative to the standard median
    filter.

    X,Y are row or column vectors with an equal number of elements.
    The elements in Y should be Gaussian distributed.

    Parameters
    ----------
    dx : positive scalar (default 3 * median(diff(X))
        which defines the half width of the filter window. Dx should be
        dimensionally equivalent to the values in X.
    t : positive scalar (default 3)
        which defines the threshold value used in the equation
        |Y - Y0| > T * S0.
    adaptive: real scalar
        if greater than 0 it uses an experimental adaptive Hampel filter.
        If none it uses a standard Hampel filter
    fulloutput: bool
        if True also the vectors: outliers, Y0,LB,UB,ADX, which corresponds to
        the mask of the replaced values, nominal data, lower and upper bounds
        on the Hampel filter and the relative half size of the local window,
        respectively. outliers.sum() gives the number of outliers detected.

    Examples
    ---------
    Hampel filter removal of outliers
    >>> import numpy as np
    >>> randint = np.random.randint
    >>> Y = 5000 + np.random.randn(1000)
    >>> outliers = randint(0,1000, size=(10,))
    >>> Y[outliers] = Y[outliers] + randint(1000, size=(10,))
    >>> YY, res = HampelFilter(fulloutput=True)(Y)
    >>> YY1, res1 = HampelFilter(dx=1, t=3, adaptive=0.1, fulloutput=True)(Y)
    >>> YY2, res2 = HampelFilter(dx=3, t=0, fulloutput=True)(Y)  # Y0 = median

    X = np.arange(len(YY))
    plt.plot(X, Y, 'b.')  # Original Data
    plt.plot(X, YY, 'r')  # Hampel Filtered Data
    plt.plot(X, res['Y0'], 'b--')  # Nominal Data
    plt.plot(X, res['LB'], 'r--')  # Lower Bounds on Hampel Filter
    plt.plot(X, res['UB'], 'r--')  # Upper Bounds on Hampel Filter
    i = res['outliers']
    plt.plot(X[i], Y[i], 'ks')  # Identified Outliers
    plt.show('hold')

     References
    ----------
    Chapters 1.4.2, 3.2.2 and 4.3.4 in Mining Imperfect Data: Dealing with
    Contamination and Incomplete Records by Ronald K. Pearson.

    Acknowledgements
    I would like to thank Ronald K. Pearson for the introduction to moving
    window filters. Please visit his blog at:
    http://exploringdatablog.blogspot.com/2012/01/moving-window-filters-and
    -pracma.html
    '''
    def __init__(self, dx=None, t=3, adaptive=None, fulloutput=False):
        self.dx = dx
        self.t = t
        self.adaptive = adaptive
        self.fulloutput = fulloutput

    def __call__(self, y, x=None):
        Y = np.atleast_1d(y).ravel()
        if x is None:
            x = range(len(Y))
        X = np.atleast_1d(x).ravel()

        dx = self.dx
        if dx is None:
            dx = 3 * np.median(np.diff(X))
        if not np.isscalar(dx):
            raise ValueError('DX must be a scalar.')
        elif dx < 0:
            raise ValueError('DX must be larger than zero.')

        YY = Y
        S0 = np.nan * np.zeros(YY.shape)
        Y0 = np.nan * np.zeros(YY.shape)
        ADX = dx * np.ones(Y.shape)

        def localwindow(X, Y, DX, i):
            mask = (X[i] - DX <= X) & (X <= X[i] + DX)
            Y0 = np.median(Y[mask])
            # Calculate Local Scale of Natural Variation
            S0 = 1.4826 * np.median(np.abs(Y[mask] - Y0))
            return Y0, S0

        def smgauss(X, V, DX):
            Xj = X
            Xk = np.atleast_2d(X).T
            Wjk = np.exp(-((Xj - Xk) / (2 * DX)) ** 2)
            G = np.dot(Wjk, V) / np.sum(Wjk, axis=0)
            return G

        if len(X) > 1:
            if self.adaptive is None:
                for i in range(len(Y)):
                    Y0[i], S0[i] = localwindow(X, Y, dx, i)
            else:  # 'adaptive'

                Y0Tmp = np.nan * np.zeros(YY.shape)
                S0Tmp = np.nan * np.zeros(YY.shape)
                DXTmp = np.arange(1, len(S0) + 1) * dx
                # Integer variation of Window Half Size

                # Calculate Initial Guess of Optimal Parameters Y0, S0, ADX
                for i in range(len(Y)):
                    j = 0
                    S0Rel = np.inf
                    while S0Rel > self.adaptive:
                        Y0Tmp[j], S0Tmp[j] = localwindow(X, Y, DXTmp[j], i)
                        if j > 0:
                            S0Rel = abs((S0Tmp[j - 1] - S0Tmp[j]) /
                                        (S0Tmp[j - 1] + S0Tmp[j]) / 2)
                        j += 1
                    Y0[i] = Y0Tmp[j - 2]
                    S0[i] = S0Tmp[j - 2]
                    ADX[i] = DXTmp[j - 2] / dx

                # Gaussian smoothing of relevant parameters
                DX = 2 * np.median(np.diff(X))
                ADX = smgauss(X, ADX, DX)
                S0 = smgauss(X, S0, DX)
                Y0 = smgauss(X, Y0, DX)

        T = self.t
        ## Prepare Output
        self.UB = Y0 + T * S0
        self.LB = Y0 - T * S0
        outliers = np.abs(Y - Y0) > T * S0  # possible outliers
        YY[outliers] = Y0[outliers]
        self.outliers = outliers
        self.num_outliers = outliers.sum()
        self.ADX = ADX
        self.Y0 = Y0
        if self.fulloutput:
            return YY, dict(outliers=outliers, Y0=Y0,
                            LB=self.LB, UB=self.UB, ADX=ADX)
        return YY


def test_hampel():
    import matplotlib.pyplot as plt
    randint = np.random.randint
    Y = 5000 + np.random.randn(1000)
    outliers = randint(0, 1000, size=(10,))
    Y[outliers] = Y[outliers] + randint(1000, size=(10,))
    YY, res = HampelFilter(dx=3, t=3, fulloutput=True)(Y)
    YY1, res1 = HampelFilter(dx=1, t=3, adaptive=0.1, fulloutput=True)(Y)
    YY2, res2 = HampelFilter(dx=3, t=0, fulloutput=True)(Y)  # median
    plt.figure(1)
    plot_hampel(Y, YY, res)
    plt.title('Standard HampelFilter')
    plt.figure(2)
    plot_hampel(Y, YY1, res1)
    plt.title('Adaptive HampelFilter')
    plt.figure(3)
    plot_hampel(Y, YY2, res2)
    plt.title('Median filter')
    plt.show('hold')


def plot_hampel(Y, YY, res):
    import matplotlib.pyplot as plt
    X = np.arange(len(YY))
    plt.plot(X, Y, 'b.')  # Original Data
    plt.plot(X, YY, 'r')  # Hampel Filtered Data
    plt.plot(X, res['Y0'], 'b--')  # Nominal Data
    plt.plot(X, res['LB'], 'r--')  # Lower Bounds on Hampel Filter
    plt.plot(X, res['UB'], 'r--')  # Upper Bounds on Hampel Filter
    i = res['outliers']
    plt.plot(X[i], Y[i], 'ks')  # Identified Outliers
    #plt.show('hold')


def test_tide_filter():
#     import statsmodels.api as sa
    import wafo.spectrum.models as sm
    sd = 10
    Sj = sm.Jonswap(Hm0=4.* sd)
    S = Sj.tospecdata()

    q = (0.1 * sd) ** 2   # variance of process noise s the car operates
    r = (100 * sd) ** 2  # variance of measurement error
    b = 0  # no system input
    u = 0  # no system input

    from scipy.signal import butter, lfilter, filtfilt, lfilter_zi
    freq_tide = 1. / (12 * 60 * 60)
    freq_wave = 1. / 10
    freq_filt = freq_wave / 10
    dt = 1.
    freq = 1. / dt
    fn = (freq / 2)

    P = 10* np.diag([1, 0.01])
    R = r
    H = np.atleast_2d([1, 0])

    F = np.atleast_2d([[0, 1],
                       [0, 0]])
    A, Q = lti_disc(F, L=None, Q=np.diag([0, q]), dt=dt)

    t = np.arange(0, 60 * 12, 1. / freq)
    w = 2 * np.pi * freq  # 1 Hz
    tide = 100 * np.sin(freq_tide * w * t + 2 * np.pi / 4) + 100
    y = tide + S.sim(len(t), dt=1. / freq)[:, 1].ravel()
#     lowess = sa.nonparametric.lowess
#     y2 = lowess(y, t, frac=0.5)[:,1]

    filt = Kalman(R=R, x=np.array([[tide[0]], [0]]), P=P, A=A, Q=Q, H=H, B=b)
    filt2 = Kalman(R=R, x=np.array([[tide[0]], [0]]), P=P, A=A, Q=Q, H=H, B=b)
    #y = tide + 0.5 * np.sin(freq_wave * w * t)
    # Butterworth filter
    b, a = butter(9, (freq_filt / fn), btype='low')
    #y2 = [lowess(y[max(i-60,0):i + 1], t[max(i-60,0):i + 1], frac=.3)[-1,1] for i in range(len(y))]
    #y2 = [lfilter(b, a, y[:i + 1])[i] for i in range(len(y))]
    #y3 = filtfilt(b, a, y[:16]).tolist() + [filtfilt(b, a, y[:i + 1])[i] for i in range(16, len(y))]
    #y0 = medfilt(y, 41)
    zi = lfilter_zi(b, a)
    #y2 = lfilter(b, a, y)#, zi=y[0]*zi)  # standard filter
    y3 = filtfilt(b, a, y)  # filter with phase shift correction
    y4 =[]
    y5 = []
    for i, j in enumerate(y):
        tmp = filt(j, u=u).ravel()
        tmp = filt2(tmp[0], u=u).ravel()
#         if i==0:
#             print(filt.x)
#             print(filt2.x)
        y4.append(tmp[0])
        y5.append(tmp[1])
    y0 = medfilt(y4, 41)
    print(filt.P)
    # plot
    import matplotlib.pyplot as plt
    plt.plot(t, y, 'r.-', linewidth=2, label='raw data')
    #plt.plot(t, y2, 'b.-', linewidth=2, label='lowess @ %g Hz' % freq_filt)
    #plt.plot(t, y2, 'b.-', linewidth=2, label='filter @ %g Hz' % freq_filt)
    plt.plot(t, y3, 'g.-', linewidth=2, label='filtfilt @ %g Hz' % freq_filt)
    plt.plot(t, y4, 'k.-', linewidth=2, label='kalman')
    #plt.plot(t, y5, 'k.', linewidth=2, label='kalman2')
    plt.plot(t, tide, 'y-', linewidth=2, label='True tide')
    plt.legend(frameon=False, fontsize=14)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show('hold')


def test_smooth():
    import matplotlib.pyplot as plt
    t = np.linspace(-4, 4, 500)
    y = np.exp(-t ** 2) + np.random.normal(0, 0.05, t.shape)
    coeff = calc_coeff(n=0, degree=0, diff_order=0)
    ysg = smooth(y, coeff, pad=True)

    plt.plot(t, y, t, ysg, '--')
    plt.show()


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

if __name__ == '__main__':
    #test_kalman_sine()
    test_tide_filter()
    #test_docstrings()
    #test_hampel()
    #test_kalman()
    # test_smooth()
