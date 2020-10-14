import numpy as np
from scipy.sparse.linalg import expm
from scipy.signal import medfilt
from wafo.plotbackend import plotbackend as plt
from wafo.sg_filter._core import (SavitzkyGolay, smoothn, Kalman,
                                  HodrickPrescott, HampelFilter)


def demo_savitzky_on_noisy_chirp():
    """
    Examples
    --------
    >>> demo_savitzky_on_noisy_chirp()

    >>> plt.close()
    """
    plt.figure(figsize=(7, 12))

    # generate chirp signal
    tvec = np.arange(0, 6.28, .02)
    true_signal = np.sin(tvec * (2.0 + tvec))
    true_d_signal = (2 + tvec) * np.cos(tvec * (2.0 + tvec))

    # add noise to signal
    noise = np.random.normal(size=true_signal.shape)
    signal = true_signal + .15 * noise

    # plot signal
    plt.subplot(311)
    plt.plot(signal)
    plt.title('signal')

    # smooth and plot signal
    plt.subplot(312)
    savgol = SavitzkyGolay(n=8, degree=4)
    s_signal = savgol.smooth(signal)
    s2 = smoothn(signal, robust=True)
    plt.plot(s_signal)
    plt.plot(s2)
    plt.plot(true_signal, 'r--')
    plt.title('smoothed signal')

    # smooth derivative of signal and plot it
    plt.subplot(313)
    savgol1 = SavitzkyGolay(n=8, degree=1, diff_order=1)

    dt = tvec[1] - tvec[0]
    d_signal = savgol1.smooth(signal) / dt

    plt.plot(d_signal)
    plt.plot(true_d_signal, 'r--')
    plt.title('smoothed derivative of signal')


def demo_kalman_voltimeter():
    """
    Examples
    --------
    >>> demo_kalman_voltimeter()

    >>> plt.close()
    """
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

    _hz = plt.plot(z, 'r.', label='observations')
    # a-posteriori state estimates:
    _hx = plt.plot(x, 'b-', label='Kalman output')
    _ht = plt.plot(truth, 'g-', label='true voltage')
    plt.legend()
    plt.title('Automobile Voltimeter Example')


def lti_disc(F, L=None, Q=None, dt=1):
    """LTI_DISC  Discretize LTI ODE with Gaussian Noise.

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

    """
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
    # Q = AB[:n, :] / AB[n:(2 * n), :]
    Q = np.linalg.solve(AB[n:(2 * n), :].T, AB[:n, :].T)
    return A, Q


def demo_kalman_sine():
    """Kalman Filter demonstration with sine signal.

    Examples
    --------
    >>> demo_kalman_sine()

    >>> plt.close()
    """
    sd = 0.5
    dt = 0.1
    w = 1
    T = np.arange(0, 30 + dt / 2, dt)
    n = len(T)
    X = 3 * np.sin(w * T)
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
    _MM = np.zeros((m, n))
    _PP = np.zeros((m, m, n))
    '''In this demonstration we estimate a stationary sine signal from noisy
    measurements by using the classical Kalman filter.'
    '''
    filt = Kalman(R=R, x=M, P=P, A=A, Q=Q, H=H, B=0)

    # Generate random voltages and watch the filter operate.
    # n = 50
    # truth = np.random.randn(n) * np.sqrt(q) + V0
    # z = truth + np.random.randn(n) * np.sqrt(r)  # measurement
    truth = X
    z = Y
    x = np.zeros((n, m))

    for i, zi in enumerate(z):
        x[i] = np.ravel(filt(zi, u=0))

    _hz = plt.plot(z, 'r.', label='observations')
    # a-posteriori state estimates:
    _hx = plt.plot(x[:, 0], 'b-', label='Kalman output')
    _ht = plt.plot(truth, 'g-', label='true voltage')
    plt.legend()
    plt.title('Automobile Voltimeter Example')


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
#         legend('Real signal','Measurements','Latest estimate',
#                'Filtered estimate')
#         title('Estimating a noisy sine signal with Kalman filter.');
#         drawnow;
#
#         pause;
#       end
#     end
#
#     clc;
#     disp('In this demonstration we estimate a stationary sine signal '
#        'from noisy measurements by using the classical Kalman filter.');
#     disp(' ');
#     disp('The filtering results are now displayed sequantially for 10 time '
#            'step at a time.');
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
#     disp('The filtered and smoothed estimates of the signal are now '
#        'displayed.')
#     disp(' ');
#     disp('RMS errors:');
#     %
#     % Errors
#     %
#     fprintf('KF = %.3f\nRTS = %.3f\n',...
#             sqrt(mean((MM(1,:)-X(1,:)).^2)),...
#             sqrt(mean((SM(1,:)-X(1,:)).^2)));


def demo_hampel():
    """
    Examples
    --------
    >>> demo_hampel()

    >>> plt.close()
    """
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


def plot_hampel(Y, YY, res):
    X = np.arange(len(YY))
    plt.plot(X, Y, 'b.')  # Original Data
    plt.plot(X, YY, 'r')  # Hampel Filtered Data
    plt.plot(X, res['Y0'], 'b--')  # Nominal Data
    plt.plot(X, res['LB'], 'r--')  # Lower Bounds on Hampel Filter
    plt.plot(X, res['UB'], 'r--')  # Upper Bounds on Hampel Filter
    i = res['outliers']
    plt.plot(X[i], Y[i], 'ks')  # Identified Outliers


def demo_tide_filter():
    """
    Examples
    --------
    >>> demo_tide_filter()

    >>> plt.close()
    """
    # import statsmodels.api as sa
    import wafo.spectrum.models as sm
    sd = 10
    Sj = sm.Jonswap(Hm0=4. * sd)
    S = Sj.tospecdata()

    q = (0.1 * sd) ** 2   # variance of process noise s the car operates
    r = (100 * sd) ** 2  # variance of measurement error
    b = 0  # no system input
    u = 0  # no system input

    from scipy.signal import butter, filtfilt, lfilter_zi  # lfilter,
    freq_tide = 1. / (12 * 60 * 60)
    freq_wave = 1. / 10
    freq_filt = freq_wave / 10
    dt = 1.
    freq = 1. / dt
    fn = (freq / 2)

    P = 10 * np.diag([1, 0.01])
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
    # y = tide + 0.5 * np.sin(freq_wave * w * t)
    # Butterworth filter
    b, a = butter(9, (freq_filt / fn), btype='low')
    # y2 = [lowess(y[max(i-60,0):i + 1], t[max(i-60,0):i + 1], frac=.3)[-1,1]
    #    for i in range(len(y))]
    # y2 = [lfilter(b, a, y[:i + 1])[i] for i in range(len(y))]
    # y3 = filtfilt(b, a, y[:16]).tolist() + [filtfilt(b, a, y[:i + 1])[i]
    #    for i in range(16, len(y))]
    # y0 = medfilt(y, 41)
    _zi = lfilter_zi(b, a)
    # y2 = lfilter(b, a, y)#, zi=y[0]*zi)  # standard filter
    y3 = filtfilt(b, a, y)  # filter with phase shift correction
    y4 = []
    y5 = []
    for _i, j in enumerate(y):
        tmp = np.ravel(filt(j, u=u))
        tmp = np.ravel(filt2(tmp[0], u=u))
#         if i==0:
#             print(filt.x)
#             print(filt2.x)
        y4.append(tmp[0])
        y5.append(tmp[1])
    _y0 = medfilt(y4, 41)
    # print(filt.P)
    # plot

    plt.plot(t, y, 'r.-', linewidth=2, label='raw data')
    # plt.plot(t, y2, 'b.-', linewidth=2, label='lowess @ %g Hz' % freq_filt)
    # plt.plot(t, y2, 'b.-', linewidth=2, label='filter @ %g Hz' % freq_filt)
    plt.plot(t, y3, 'g.-', linewidth=2, label='filtfilt @ %g Hz' % freq_filt)
    plt.plot(t, y4, 'k.-', linewidth=2, label='kalman')
    # plt.plot(t, y5, 'k.', linewidth=2, label='kalman2')
    plt.plot(t, tide, 'y-', linewidth=2, label='True tide')
    plt.legend(frameon=False, fontsize=14)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")


def demo_savitzky_on_exponential():
    """
    Examples
    --------
    >>> demo_savitzky_on_exponential()

    >>> plt.close()
    """
    t = np.linspace(-4, 4, 500)
    y = np.exp(-t ** 2) + np.random.normal(0, 0.05, np.shape(t))
    n = 11
    ysg = SavitzkyGolay(n, degree=1, diff_order=0)(y)
    plt.plot(t, y, t, ysg, '--')


def demo_smoothn_on_1d_cos():
    """
    Examples
    --------
    >>> demo_smoothn_on_1d_cos()

    >>> plt.close()
    """
    x = np.linspace(0, 100, 2 ** 8)
    y = np.cos(x / 10) + (x / 50) ** 2 + np.random.randn(np.size(x)) / 10
    y[np.r_[70, 75, 80]] = np.array([5.5, 5, 6])
    z = smoothn(y)  # Regular smoothing
    zr = smoothn(y, robust=True)  # Robust smoothing
    _h0 = plt.subplot(121),
    _h = plt.plot(x, y, 'r.', x, z, 'k', linewidth=2)
    plt.title('Regular smoothing')
    plt.subplot(122)
    plt.plot(x, y, 'r.', x, zr, 'k', linewidth=2)
    plt.title('Robust smoothing')


def demo_smoothn_on_2d_exp_sin():
    """
    Examples
    --------
    >>> demo_smoothn_on_2d_exp_sin()

    >>> plt.close()
    """
    xp = np.arange(0, 1, 0.02)  # np.r_[0:1:0.02]
    [x, y] = np.meshgrid(xp, xp)
    f = np.exp(x + y) + np.sin((x - 2 * y) * 3)
    fn = f + np.random.randn(*f.shape) * 0.5
    _fs, s = smoothn(fn, fulloutput=True)
    fs2 = smoothn(fn, s=2 * s)
    _h = plt.subplot(131),
    _h = plt.contourf(xp, xp, fn)
    _h = plt.subplot(132),
    _h = plt.contourf(xp, xp, fs2)
    _h = plt.subplot(133),
    _h = plt.contourf(xp, xp, f)


def _cardioid(n=1000):
    t = np.linspace(0, 2 * np.pi, n)
    x0 = 2 * np.cos(t) * (1 - np.cos(t))
    y0 = 2 * np.sin(t) * (1 - np.cos(t))
    x = x0 + np.random.randn(x0.size) * 0.1
    y = y0 + np.random.randn(y0.size) * 0.1
    return x, y, x0, y0


def demo_smoothn_on_cardioid():
    """
    Examples
    --------
    >>> demo_smoothn_on_cardioid()

    >>> plt.close()
    """
    x, y, x0, y0 = _cardioid()
    z = smoothn(x + 1j * y, robust=False)
    plt.plot(x0, y0, 'y',
             x, y, 'r.',
             np.real(z), np.imag(z), 'k', linewidth=2)


def demo_hodrick_on_cardioid():
    """
    Examples
    --------
    >>> demo_hodrick_on_cardioid()

    >>> plt.close()
    """
    x, y, x0, y0 = _cardioid()

    smooth = HodrickPrescott(w=20000)
    # smooth = HampelFilter(adaptive=50)
    xs, ys = smooth(x), smooth(y)
    plt.plot(x0, y0, 'y',
             x, y, 'r.',
             xs, ys, 'k', linewidth=2)


if __name__ == '__main__':
    from wafo.testing import test_docstrings
    test_docstrings(__file__)
    # demo_savitzky_on_noisy_chirp()
    # plt.show('hold')  # show plot
    # demo_kalman_sine()
    # demo_tide_filter()
    # demo_hampel()
    # demo_kalman_voltimeter()
    # demo_savitzky_on_exponential()
#     plt.figure(1)
#     demo_hodrick_on_cardioid()
#     plt.figure(2)
#     # demo_smoothn_on_1d_cos()
#     demo_smoothn_on_cardioid()
#     plt.show('hold')
