'''
Created on 2. jan. 2017

@author: pab
'''
from __future__ import absolute_import, division
import scipy.stats
import numpy as np
import warnings
from wafo.plotbackend import plotbackend as plt
from wafo.kdetools import Kernel, TKDE, KDE, KRegression, BKRegression
try:
    from wafo import fig
except ImportError:
    warnings.warn('fig import only supported on Windows')

__all__ = ['kde_demo1', 'kde_demo2', 'kde_demo3', 'kde_demo4', 'kde_demo5',
           'kreg_demo1', ]


def kde_demo1():
    """KDEDEMO1 Demonstrate the smoothing parameter impact on KDE.

    KDEDEMO1 shows the true density (dotted) compared to KDE based on 7
    observations (solid) and their individual kernels (dashed) for 3
    different values of the smoothing parameter, hs.

    """
    st = scipy.stats
    x = np.linspace(-4, 4, 101)
    x0 = x / 2.0
    data = np.random.normal(loc=0, scale=1.0, size=7)
    kernel = Kernel('gauss')
    hs = kernel.hns(data)
    h_vec = [hs / 2, hs, 2 * hs]

    for ix, h in enumerate(h_vec):
        plt.figure(ix)
        kde = KDE(data, hs=h, kernel=kernel)
        f2 = kde(x, output='plot', title='h_s = {0:2.2f}'.format(float(h)),
                 ylab='Density')
        f2.plot('k-')

        plt.plot(x, st.norm.pdf(x, 0, 1), 'k:')
        n = len(data)
        plt.plot(data, np.zeros(data.shape), 'bx')
        y = kernel(x0) / (n * h * kernel.norm_factor(d=1, n=n))
        for i in range(n):
            plt.plot(data[i] + x0 * h, y, 'b--')
            plt.plot([data[i], data[i]], [0, np.max(y)], 'b')

        plt.axis([min(x), max(x), 0, 0.5])


def kde_demo2():
    '''Demonstrate the difference between transformation- and ordinary-KDE.

    KDEDEMO2 shows that the transformation KDE is a better estimate for
    Rayleigh distributed data around 0 than the ordinary KDE.
    '''
    st = scipy.stats
    data = st.rayleigh.rvs(scale=1, size=300)

    x = np.linspace(1.5e-2, 5, 55)

    kde = KDE(data)
    f = kde(output='plot', title='Ordinary KDE (hs={0:})'.format(kde.hs))
    plt.figure(0)
    f.plot()

    plt.plot(x, st.rayleigh.pdf(x, scale=1), ':')

    # plotnorm((data).^(L2))  # gives a straight line => L2 = 0.5 reasonable
    hs = Kernel('gauss').get_smoothing(data**0.5)
    tkde = TKDE(data, hs=hs, L2=0.5)
    ft = tkde(x, output='plot',
              title='Transformation KDE (hs={0:})'.format(tkde.tkde.hs))
    plt.figure(1)
    ft.plot()

    plt.plot(x, st.rayleigh.pdf(x, scale=1), ':')

    plt.figure(0)


def kde_demo3():
    '''Demonstrate the difference between transformation and ordinary-KDE in 2D

    KDEDEMO3 shows that the transformation KDE is a better estimate for
    Rayleigh distributed data around 0 than the ordinary KDE.
    '''
    st = scipy.stats
    data = st.rayleigh.rvs(scale=1, size=(2, 300))

    # x = np.linspace(1.5e-3, 5, 55)

    kde = KDE(data)
    f = kde(output='plot', title='Ordinary KDE', plotflag=1)
    plt.figure(0)
    f.plot()

    plt.plot(data[0], data[1], '.')

    # plotnorm((data).^(L2)) % gives a straight line => L2 = 0.5 reasonable
    hs = Kernel('gauss').get_smoothing(data**0.5)
    tkde = TKDE(data, hs=hs, L2=0.5)
    ft = tkde.eval_grid_fast(
        output='plot', title='Transformation KDE', plotflag=1)

    plt.figure(1)
    ft.plot()

    plt.plot(data[0], data[1], '.')

    plt.figure(0)


def kde_demo4(N=50):
    '''Demonstrate that the improved Sheather-Jones plug-in (hisj) is superior
       for 1D multimodal distributions

    KDEDEMO4 shows that the improved Sheather-Jones plug-in smoothing is a
    better compared to normal reference rules (in this case the hns)
    '''
    st = scipy.stats

    data = np.hstack((st.norm.rvs(loc=5, scale=1, size=(N,)),
                      st.norm.rvs(loc=-5, scale=1, size=(N,))))

    # x = np.linspace(1.5e-3, 5, 55)

    kde = KDE(data, kernel=Kernel('gauss', 'hns'))
    f = kde(output='plot', title='Ordinary KDE', plotflag=1)

    kde1 = KDE(data, kernel=Kernel('gauss', 'hisj'))
    f1 = kde1(output='plot', label='Ordinary KDE', plotflag=1)

    plt.figure(0)
    f.plot('r', label='hns={0}'.format(kde.hs))
    # plt.figure(2)
    f1.plot('b', label='hisj={0}'.format(kde1.hs))
    x = np.linspace(-9, 9)
    plt.plot(x, (st.norm.pdf(x, loc=-5, scale=1) +
                 st.norm.pdf(x, loc=5, scale=1)) / 2, 'k:',
             label='True density')
    plt.legend()


def kde_demo5(N=500):
    '''Demonstrate that the improved Sheather-Jones plug-in (hisj) is superior
       for 2D multimodal distributions

    KDEDEMO5 shows that the improved Sheather-Jones plug-in smoothing is better
    compared to normal reference rules (in this case the hns)
    '''
    st = scipy.stats

    data = np.hstack((st.norm.rvs(loc=5, scale=1, size=(2, N,)),
                      st.norm.rvs(loc=-5, scale=1, size=(2, N,))))
    kde = KDE(data, kernel=Kernel('gauss', 'hns'))
    f = kde(output='plot', plotflag=1,
            title='Ordinary KDE, hns={0:s}'.format(str(list(kde.hs))))

    kde1 = KDE(data, kernel=Kernel('gauss', 'hisj'))
    f1 = kde1(output='plot', plotflag=1,
              title='Ordinary KDE, hisj={0:s}'.format(str(list(kde1.hs))))

    plt.figure(0)
    plt.clf()
    f.plot()
    plt.plot(data[0], data[1], '.')
    plt.figure(1)
    plt.clf()
    f1.plot()
    plt.plot(data[0], data[1], '.')


def kreg_demo1(hs=None, fast=False, fun='hisj'):
    """Compare KRegression to KernelReg from statsmodels.nonparametric
    """
    N = 100
    # ei = np.random.normal(loc=0, scale=0.075, size=(N,))
    ei = np.array([
        -0.08508516, 0.10462496, 0.07694448, -0.03080661, 0.05777525,
        0.06096313, -0.16572389, 0.01838912, -0.06251845, -0.09186784,
        -0.04304887, -0.13365788, -0.0185279, -0.07289167, 0.02319097,
        0.06887854, -0.08938374, -0.15181813, 0.03307712, 0.08523183,
        -0.0378058, -0.06312874, 0.01485772, 0.06307944, -0.0632959,
        0.18963205, 0.0369126, -0.01485447, 0.04037722, 0.0085057,
        -0.06912903, 0.02073998, 0.1174351, 0.17599277, -0.06842139,
        0.12587608, 0.07698113, -0.0032394, -0.12045792, -0.03132877,
        0.05047314, 0.02013453, 0.04080741, 0.00158392, 0.10237899,
        -0.09069682, 0.09242174, -0.15445323, 0.09190278, 0.07138498,
        0.03002497, 0.02495252, 0.01286942, 0.06449978, 0.03031802,
        0.11754861, -0.02322272, 0.00455867, -0.02132251, 0.09119446,
        -0.03210086, -0.06509545, 0.07306443, 0.04330647, 0.078111,
        -0.04146907, 0.05705476, 0.02492201, -0.03200572, -0.02859788,
        -0.05893749, 0.00089538, 0.0432551, 0.04001474, 0.04888828,
        -0.17708392, 0.16478644, 0.1171006, 0.11664846, 0.01410477,
        -0.12458953, -0.11692081, 0.0413047, -0.09292439, -0.07042327,
        0.14119701, -0.05114335, 0.04994696, -0.09520663, 0.04829406,
        -0.01603065, -0.1933216, 0.19352763, 0.11819496, 0.04567619,
        -0.08348306, 0.00812816, -0.00908206, 0.14528945, 0.02901065])
    x = np.linspace(0, 1, N)

    va_1 = 0.3 ** 2
    va_2 = 0.7 ** 2
    y0 = np.exp(-x ** 2 / (2 * va_1)) + 1.3*np.exp(-(x - 1) ** 2 / (2 * va_2))
    y = y0 + ei
    kernel = Kernel('gauss', fun=fun)
    hopt = kernel.hisj(x)
    kreg = KRegression(
        x, y, p=0, hs=hs, kernel=kernel, xmin=-2 * hopt, xmax=1 + 2 * hopt)
    if fast:
        kreg.__call__ = kreg.eval_grid_fast

    f = kreg(x, output='plot', title='Kernel regression', plotflag=1)
    plt.figure(0)
    f.plot(label='p=0')

    kreg.p = 1
    f1 = kreg(x, output='plot', title='Kernel regression', plotflag=1)
    f1.plot(label='p=1')
    # print(f1.data)
    plt.plot(x, y, '.', label='data')
    plt.plot(x, y0, 'k', label='True model')
    from statsmodels.nonparametric.kernel_regression import KernelReg
    kreg2 = KernelReg(y, x, ('c'))
    y2 = kreg2.fit(x)
    plt.plot(x, y2[0], 'm', label='statsmodel')

    plt.legend()
    plt.show()

    print(kreg.tkde.tkde._inv_hs)
    print(kreg.tkde.tkde.hs)


def _get_data(n=100, symmetric=False, loc1=1.1, scale1=0.6, scale2=1.0):
    """
    Return test data for binomial regression demo.
    """
    st = scipy.stats
    dist = st.norm

    norm1 = scale2 * (dist.pdf(-loc1, loc=-loc1, scale=scale1) +
                      dist.pdf(-loc1, loc=loc1, scale=scale1))

    def fun1(x):
        return ((dist.pdf(x, loc=-loc1, scale=scale1) +
                 dist.pdf(x, loc=loc1, scale=scale1)) / norm1).clip(max=1.0)

    x = np.sort(6 * np.random.rand(n, 1) - 3, axis=0)

    y = (fun1(x) > np.random.rand(n, 1)).ravel()
    # y = (np.cos(x)>2*np.random.rand(n, 1)-1).ravel()
    x = x.ravel()

    if symmetric:
        xi = np.hstack((x.ravel(), -x.ravel()))
        yi = np.hstack((y, y))
        i = np.argsort(xi)
        x = xi[i]
        y = yi[i]
    return x, y, fun1


def check_bkregression():
    """
    Check binomial regression
    """
    plt.ion()
    k = 0
    for _i, n in enumerate([50, 100, 300, 600]):
        x, y, fun1 = _get_data(n, symmetric=True, loc1=0.1,
                               scale1=0.6, scale2=0.75)
        bkreg = BKRegression(x, y, a=0.05, b=0.05)
        fbest = bkreg.prb_search_best(
            hsfun='hste', alpha=0.05, color='g', label='Transit_D')

        figk = plt.figure(k)
        ax = figk.gca()
        k += 1
#        fbest.score.plot(axis=ax)
#        axsize = ax.axis()
#        ax.vlines(fbest.hs,axsize[2]+1,axsize[3])
#        ax.set(yscale='log')
        fbest.labels.title = 'N = {:d}'.format(n)
        fbest.plot(axis=ax)
        ax.plot(x, fun1(x), 'r')
        ax.legend(frameon=False, markerscale=4)
        # ax = plt.gca()
        ax.set_yticklabels(ax.get_yticks() * 100.0)
        ax.grid(True)

    fig.tile(range(0, k))
    plt.ioff()
    plt.show('hold')


if __name__ == '__main__':
    # kde_demo5()
    # check_bkregression()
    kreg_demo1(hs=0.04, fast=True)
    plt.show('hold')
