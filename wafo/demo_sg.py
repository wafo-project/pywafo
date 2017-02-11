import matplotlib.pyplot as plt
import numpy as np
from wafo.sg_filter import SavitzkyGolay, smoothn  # calc_coeff, smooth


def example_reconstruct_noisy_chirp():
    """
    Example
    -------
    >>> example_reconstruct_noisy_chirp()
    """
    plt.figure(figsize=(7, 12))

    # generate chirp signal
    tvec = np.arange(0, 6.28, .02)
    true_signal = np.sin(tvec * (2.0 + tvec))
    true_d_signal = (2+tvec) * np.cos(tvec * (2.0 + tvec))

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

    dt = tvec[1]-tvec[0]
    d_signal = savgol1.smooth(signal) / dt

    plt.plot(d_signal)
    plt.plot(true_d_signal, 'r--')
    plt.title('smoothed derivative of signal')


if __name__ == '__main__':
    from wafo.testing import test_docstrings
    test_docstrings(__file__)
    # example_reconstruct_noisy_chirp()
    # plt.show('hold')  # show plot
