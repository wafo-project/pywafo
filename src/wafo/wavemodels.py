'''
Created on 13. mar. 2018

@author: pab
'''
import numpy as np
from numpy import pi, sqrt
import wafo.transform.estimation as te
import wafo.transform as wt
from wafo.containers import PlotData
from wafo.kdetools.kernels import qlevels
from wafo.misc import tranproc
import warnings


def _set_default_t_h_g(t, h, g, m0, m2):
    if g is None:
        y = np.linspace(-5, 5)
        x = sqrt(m0) * y + 0
        g = wt.TrData(y, x)
    if t is None:
        tt1 = 2 * pi * sqrt(m0 / m2)
        t = np.linspace(0, 1.7 * tt1, 51)
    if h is None:
        px = g.gauss2dat([0, 4.])
        px = abs(px[1] - px[0])
        h = np.linspace(0, 1.3 * px, 41)
    return h, t, g


def lh83pdf(t=None, h=None, mom=None, g=None):
    """
    LH83PDF Longuet-Higgins (1983) approximation of the density (Tc,Ac)
             in a stationary Gaussian transform process X(t) where
             Y(t) = g(X(t)) (Y zero-mean Gaussian, X  non-Gaussian).

      CALL:  f   = lh83pdf(t,h,[m0,m1,m2],g);

            f    = density of wave characteristics of half-wavelength
                   in a stationary Gaussian transformed process X(t),
                   where Y(t) = g(X(t)) (Y zero-mean Gaussian)
           t,h   = vectors of periods and amplitudes, respectively.
                   default depending on the spectral moments
        m0,m1,m2 = the 0'th,1'st and 2'nd moment of the spectral density
                   with angular  frequency.
            g    = space transformation, Y(t)=g(X(t)), default: g is identity
                   transformation, i.e. X(t) = Y(t)  is Gaussian,
                   The transformation, g, can be estimated using lc2tr
                   or dat2tr or given apriori by ochi.

    Examples
    --------
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap()
    >>> w = np.linspace(0,4,256)
    >>> S = Sj.tospecdata(w)   #Make spectrum object from numerical values
    >>> S = sm.SpecData1D(Sj(w),w) # Alternatively do it manually
    >>> mom, mom_txt = S.moment(nr=2, even=False)
    >>> f = lh83pdf(mom=mom)
    >>> f.plot()

    See also
    --------
    cav76pdf,  lc2tr, dat2tr

    References
    ----------
    Longuet-Higgins,  M.S. (1983)
    "On the joint distribution wave periods and amplitudes in a
     random wave field", Proc. R. Soc. A389, pp 24--258

    Longuet-Higgins,  M.S. (1975)
    "On the joint distribution wave periods and amplitudes of sea waves",
    J. geophys. Res. 80, pp 2688--2694
    """

    # tested on: matlab 5.3
    # History:
    # Revised pab 01.04.2001
    # - Added example
    # - Better automatic scaling for h,t
    # revised by IR 18.06.2000, fixing transformation and transposing t and h to fit simpson req.
    # revised by pab 28.09.1999
    #   made more efficient calculation of f
    # by Igor Rychlik

    m0, m1, m2 = mom
    h, t, g = _set_default_t_h_g(t, h, g, m0, m2)

    L0 = m0
    L1 = m1 / (2 * pi)
    L2 = m2 / (2 * pi)**2
    eps2 = sqrt((L2 * L0) / (L1**2) - 1)

    if np.any(~np.isreal(eps2)):
        raise ValueError('input moments are not correct')

    const = 4 / sqrt(pi) / eps2 / (1 + 1 / sqrt(1 + eps2**2))

    a = len(h)
    b = len(t)
    der = np.ones((a, 1))

    h_lh = g.dat2gauss(h.ravel(), der.ravel())

    der = abs(h_lh[1])  # abs(h_lh[:, 1])
    h_lh = h_lh[0]

    # Normalization + transformation of t and h ???????
    # Without any transformation

    t_lh = t / (L0 / L1)
    # h_lh = h_lh/sqrt(2*L0)
    h_lh = h_lh / sqrt(2)
    t_lh = 2 * t_lh

    # Computation of the distribution
    T, H = np.meshgrid(t_lh[1:b], h_lh)
    f_th = np.zeros((a, b))
    tmp = const * der[:, None] * (H / T)**2 * np.exp(-H**2. *
                                                     (1 + ((1 - 1. / T) / eps2)**2)) / ((L0 / L1) * sqrt(2) / 2)
    f_th[:, 1:b] = tmp

    f = PlotData(f_th, (t, h),
                 xlab='Tc', ylab='Ac',
                 title='Joint density of (Tc,Ac) - Longuet-Higgins (1983)',
                 plot_kwds=dict(plotflag=1))

    return _add_contour_levels(f)


def cav76pdf(t=None, h=None, mom=None, g=None):
    """
    CAV76PDF Cavanie et al. (1976) approximation of the density  (Tc,Ac)
             in a stationary Gaussian transform process X(t) where
             Y(t) = g(X(t)) (Y zero-mean Gaussian, X  non-Gaussian).

     CALL:  f = cav76pdf(t,h,[m0,m2,m4],g);

            f    = density of wave characteristics of half-wavelength
                   in a stationary Gaussian transformed process X(t),
                   where Y(t) = g(X(t)) (Y zero-mean Gaussian)
           t,h   = vectors of periods and amplitudes, respectively.
                   default depending on the spectral moments
     m0,m2,m4    = the 0'th, 2'nd and 4'th moment of the spectral density
                   with angular frequency.
            g    = space transformation, Y(t)=g(X(t)), default: g is identity
                   transformation, i.e. X(t) = Y(t)  is Gaussian,
                   The transformation, g, can be estimated using lc2tr
                   or dat2tr or given a priori by ochi.
           []    = default values are used.

     Examples
     --------
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap()
    >>> w = np.linspace(0,4,256)
    >>> S = Sj.tospecdata(w)   #Make spectrum object from numerical values
    >>> S = sm.SpecData1D(Sj(w),w) # Alternatively do it manually
    >>> mom, mom_txt = S.moment(nr=4, even=True)
    >>> f = cav76pdf(mom=mom)
    >>> f.plot()

     See also
     --------
     lh83pdf, lc2tr, dat2tr

     References
     ----------
     Cavanie, A., Arhan, M. and Ezraty, R. (1976)
     "A statistical relationship between individual heights and periods of
      storm waves".
     In Proceedings Conference on Behaviour of Offshore Structures,
     Trondheim, pp. 354--360
     Norwegian Institute of Technology, Trondheim, Norway

     Lindgren, G. and Rychlik, I. (1982)
     Wave Characteristics Distributions for Gaussian Waves --
     Wave-lenght, Amplitude and Steepness, Ocean Engng vol 9, pp. 411-432.
    """
    # tested on: matlab 5.3 NB! note
    # History:
    # revised pab 04.11.2000
    # - fixed xlabels i.e. f.labx={'Tc','Ac'}
    # revised by IR 4 X 2000. fixed transform and normalisation
    # using Lindgren & Rychlik (1982) paper.
    # At the end of the function there is a text with derivation of the density.
    #
    # revised by jr 21.02.2000
    # - Introduced cell array for f.x for use with pdfplot
    # by pab 28.09.1999

    m0, m2, m4 = mom
    h, t, g = _set_default_t_h_g(t, h, g, m0, m2)

    eps4 = 1.0 - m2**2 / (m0 * m4)
    alfa = m2 / sqrt(m0 * m4)
    if np.any(~np.isreal(eps4)):
        raise ValueError('input moments are not correct')

    a = len(h)
    b = len(t)
    der = np.ones((a, 1))

    h_lh = g.dat2gauss(h.ravel(), der.ravel())
    der = abs(h_lh[1])
    h_lh = h_lh[0]

    # Normalization + transformation of t and h

    pos = 2 / (1 + alfa)    # inverse of a fraction of positive maxima
    cons = 2 * pi**4 * pos / sqrt(2 * pi) / m4 / sqrt((1 - alfa**2))
    # Tm=2*pi*sqrt(m0/m2)/alpha; #mean period between positive maxima

    t_lh = t
    h_lh = sqrt(m0) * h_lh

    # Computation of the distribution
    T, H = np.meshgrid(t_lh[1:b], h_lh)
    f_th = np.zeros((a, b))

    f_th[:, 1:b] = cons * der[:, None] * (H**2 / (T**5)) * np.exp(-0.5 * (
        H / T**2)**2. * ((T**2 - pi**2 * m2 / m4)**2 / (m0 * (1 - alfa**2)) + pi**4 / m4))
    f = PlotData(f_th, (t, h),
                 xlab='Tc', ylab='Ac',
                 title='Joint density of (Tc,Ac) - Cavanie et al. (1976)',
                 plot_kwds=dict(plotflag=1))
    return _add_contour_levels(f)


def _add_contour_levels(f):
    p_levels = np.r_[10:90:20, 95, 99, 99.9]
    try:
        c_levels = qlevels(f.data, p=p_levels)
        f.clevels = c_levels
        f.plevels = p_levels
    except ValueError as e:
        msg = "Could not calculate contour levels!. ({})".format(str(e))
        warnings.warn(msg)
    return f

# Let U,Z be the height and second derivative (curvature) at a local maximum in a Gaussian proces
# with spectral moments m0,m2,m4. The conditional density ($U>0$) has the following form
# $$
# f(z,u)=c \frac{1}{\sqrt{2\pi}}\frac{1}{\sqrt{m0(1-\alpha^2)}}\exp(-0.5\left(\frac{u-z(m2/m4)}
# {\sqrt{m0(1-\alpha^2)}}\right)^2)\frac{|z|}{m4}\exp(-0.5z^2/m4), \quad z<0,
# $$
# where $c=2/(1+\alpha)$, $\alpha=m2/\sqrt{m0\cdot m4}$.
#
# The cavanie approximation is based on the model $X(t)=U \cos(\pi t/T)$, consequently
# we have $U=H$ and by twice differentiation $Z=-U(\pi^2/T)^2\cos(0)$. The variable change has
# Jacobian $2\pi^2 H/T^3$ giving the final formula for the density of $T,H$
# $$
# f(t,h)=c \frac{2\pi^4}{\sqrt{2\pi}}\frac{1}{m4\sqrt{m0(1-\alpha^2)}}\frac{h^2}{t^5}
#       \exp(-0.5\frac{h^2}{t^4}\left(\left(\frac{t^2-\pi^2(m2/m4)}
# {\sqrt{m0(1-\alpha^2)}}\right)^2+\frac{\pi^4}{m4}\right)).
# $$
#


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    test_docstrings()
