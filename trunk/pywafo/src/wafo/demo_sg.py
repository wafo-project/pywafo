import matplotlib.pyplot as plt
from pylab import subplot, plot, title, figure
from numpy import random, arange, sin
from sg_filter import SavitzkyGolay, smoothn  # calc_coeff, smooth


figure(figsize=(7, 12))


# generate chirp signal
tvec = arange(0, 6.28, .02)
true_signal = sin(tvec * (2.0 + tvec))

# add noise to signal
noise = random.normal(size=true_signal.shape)
signal = true_signal + .15 * noise

# plot signal
subplot(311)
plot(signal)
title('signal')

# smooth and plot signal
subplot(312)
savgol = SavitzkyGolay(n=8, degree=4)
s_signal = savgol.smooth(signal)
s2 = smoothn(signal, robust=True)
plot(s_signal)
plot(s2)
plot(true_signal, 'r--')
title('smoothed signal')

# smooth derivative of signal and plot it
subplot(313)
savgol1 = SavitzkyGolay(n=8, degree=1, diff_order=1)

d_signal = savgol1.smooth(signal)

plot(d_signal)
title('smoothed derivative of signal')

plt.show('hold')  # show plot
