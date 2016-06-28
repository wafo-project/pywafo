from tutor_init import *
import itertools
# import sys
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

MARKERS = ('o', 'x', '+', '.', '<', '>', '^', 'v')


def plot_varying_symbols(x, y, color='red', size=5):
    """
    Create a plot with varying symbols
    Parameters
    ----------
    x : numpy array with x data of the points
    y : numpy array with y data of the points
    color : color of the symbols

    Returns
    -------

    """
    markers = itertools.cycle(MARKERS)
    for q, p in zip(x, y):
        plt.plot(q, p, marker=markers.next(), linestyle='', color=color,
                 markersize=size)


def damage_vs_S(S, beta, K):
    """
    calculate the damage 1/N for a given stress S
    Parameters
    ----------
    S       : Stress [Pa]
    beta    : coefficient, typically 3
    K       : constant

    Returns
    -------

    """
    return K * np.power(S, beta)

# Section 4.3.1 Crossing intensity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import wafo.data as wd
import wafo.objects as wo
import wafo.misc as wm

xx_sea = wd.sea()

Tlength = xx_sea[-1, 0] - xx_sea[0, 0]
beta = 3
K1 = 6.5e-31
Np = 200
Tp = Tlength / Np
A = 100e6
log.info("setting sin wave with Tp={} and T={}".format(Tp, Tlength))
Nc = 1.0 / damage_vs_S(A, beta, K1)
damage = float(Np) / float(Nc)
log.info("budget at S={} N={}: damage = {} ".format(A, Nc, damage))
#xx_sea[:, 1] = A * np.cos(2 * np.pi * xx_sea[:, 0]/Tp)
xx_sea[:, 1] *= 500e6

log.info("loaded sea time series {}".format(xx_sea.shape))
ts = wo.mat2timeseries(xx_sea)

tp = ts.turning_points()
mM = tp.cycle_pairs(kind='min2max')
Mm = tp.cycle_pairs(kind='max2min')
lc = mM.level_crossings(intensity=True)
T_sea = ts.args[-1] - ts.args[0]

# for i in dir(mM):
#    print(i)


ts1 = wo.mat2timeseries(xx_sea[:, :])
tp1 = ts1.turning_points()
sig_tp = ts.turning_points(h=0, wavetype='astm')
try:
    sig_cp = sig_tp.cycle_astm()
    log.info("Successfully used cycle_astm")
except AttributeError:
    log.warning("Could use cycle_astm")
    sig_cp = None
tp1 = ts1.turning_points()
tp2 = ts1.turning_points(wavetype='Mw')
mM1 = tp1.cycle_pairs(kind='min2max')
Mm1 = tp1.cycle_pairs(kind='max2min')

tp_rfc = tp1.rainflow_filter(h=100e6)
mM_rfc = tp_rfc.cycle_pairs()
try:
    mM_rfc_a = tp1.cycle_astm()
except AttributeError:
    mM_rfc_a = None
tc1 = ts1.trough_crest()
min_to_max = True
rfc_plot = True
if min_to_max:
    m1, M1 = mM1.get_minima_and_maxima()
    i_min_start = 0
else:
    m1, M1 = Mm1.get_minima_and_maxima()
    i_min_start = 2

m_rfc, M_rfc = mM_rfc.get_minima_and_maxima()
# m_rfc_a, M_rfc_a = mM_rfc_a.get_minima_and_maxima()
ts1.plot('b-')
if rfc_plot:
    plot_varying_symbols(tp_rfc.args[0::2], m_rfc, color='red', size=10)
    plot_varying_symbols(tp_rfc.args[1::2], M_rfc, color='green', size=10)
else:
    plot_varying_symbols(tp.args[i_min_start::2], m1, color='red', size=10)
    plot_varying_symbols(tp.args[1::2], M1, color='green', size=10)

set_windows_title("Sea time series", log)

plt.figure()
plt.subplot(122),
mM.plot()
plt.title('min-max cycle pairs')
plt.subplot(121),
mM_rfc.plot()

title = 'Rainflow filtered cycles'
plt.title(title)
set_windows_title(title)


# Min-max and rainflow cycle distributions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import wafo.misc as wm
ampmM_sea = mM.amplitudes()
ampRFC_sea = mM_rfc.amplitudes()
plt.figure()
title = "s_n_curve"
set_windows_title(title)
S = np.linspace(1e6, 1000e6)
plt.loglog(S, damage_vs_S(S, beta, K1))
plt.figure()
plt.subplot(121)
stress_range = (1, 1e9)
n_bins = 100
wm.plot_histgrm(ampmM_sea, bins=n_bins, range=stress_range)
plt.xlim(stress_range)
ylim = plt.gca().get_ylim()
plt.title('min-max amplitude distribution')
plt.subplot(122)
if sig_cp is not None:
    wm.plot_histgrm(sig_cp[:, 0], bins=n_bins, range=stress_range)
    plt.gca().set_ylim(ylim)
    title = 'Rainflow amplitude distribution'
    plt.title(title)
    plt.semilogy
    set_windows_title(title)

    hist, bin_edges = np.histogram(
        sig_cp[
            :, 0], bins=n_bins, range=stress_range)

    plt.figure()
    title = "my_bins"
    plt.title(title)
    plt.title(title)
    set_windows_title(title)
    plt.semilogy
    plt.bar(bin_edges[:-1], hist, width=stress_range[1] / n_bins)

    print("damage min/max : {}".format(mM_rfc.damage([beta], K1)))

    damage_rfc = K1 * np.sum(sig_cp[:, 0] ** beta)
    print("damage rfc : {}".format(damage_rfc))
plt.show('hold')
