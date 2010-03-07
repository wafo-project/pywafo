from pylab import subplot, plot, title, savefig, figure, arange, sin, random #@UnresolvedImport
from sg_filter import calc_coeff, smooth


figure(figsize=(7,12))


# generate chirp signal
tvec = arange(0, 6.28, .02)
signal = sin(tvec*(2.0+tvec))

# add noise to signal
noise = random.normal(size=signal.shape)
signal += (2000.+.15 * noise)

# plot signal
subplot(311)
plot(signal)
title('signal')

# smooth and plot signal
subplot(312)
coeff = calc_coeff(8, 4)
s_signal = smooth(signal, coeff) 

plot(s_signal)
title('smoothed signal')

# smooth derivative of signal and plot it
subplot(313)
coeff = calc_coeff(8, 1, 1)
d_signal = smooth(signal, coeff)

plot(d_signal)
title('smoothed derivative of signal')

# show plot
savefig("savitzky.png")





