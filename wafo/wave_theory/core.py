'''
Created on 3. juni 2011

@author: pab
'''
from __future__ import absolute_import
import numpy as np
from numpy import exp, expm1, inf, nan, pi, hstack, where, atleast_1d, cos, sin
from .dispersion_relation import w2k, k2w  # @UnusedImport
from ..misc import JITImport
__all__ = ['w2k', 'k2w', 'sensor_typeid', 'sensor_type', 'TransferFunction']

_MODELS = JITImport('wafo.spectrum.models')


def hyperbolic_ratio(a, b, sa, sb):
    '''
    Return ratio of hyperbolic functions
          to allow extreme variations of arguments.

    Parameters
    ----------
    a, b : array-like
        arguments vectors of the same size
    sa, sb : scalar integers
        defining the hyperbolic function used, i.e.,
        f(x,1)=cosh(x), f(x,-1)=sinh(x)

    Returns
    -------
    r : ndarray
        f(a,sa)/f(b,sb), ratio of hyperbolic functions of same
                    size as a and b
     Examples
     --------
     >>> x = [-2,0,2]
     >>> hyperbolic_ratio(x,1,1,1)   # gives r=cosh(x)/cosh(1)
     array([ 2.438107  ,  0.64805427,  2.438107  ])
     >>> hyperbolic_ratio(x,1,1,-1)  # gives r=cosh(x)/sinh(1)
     array([ 3.20132052,  0.85091813,  3.20132052])
     >>> hyperbolic_ratio(x,1,-1,1)  # gives r=sinh(x)/cosh(1)
     array([-2.35040239,  0.        ,  2.35040239])
     >>> hyperbolic_ratio(x,1,-1,-1) # gives r=sinh(x)/sinh(1)
     array([-3.08616127,  0.        ,  3.08616127])
     >>> hyperbolic_ratio(1,x,1,1)   # gives r=cosh(1)/cosh(x)
     array([ 0.41015427,  1.54308063,  0.41015427])
     >>> hyperbolic_ratio(1,x,1,-1)  # gives r=cosh(1)/sinh(x)
     array([-0.42545906,         inf,  0.42545906])
     >>> hyperbolic_ratio(1,x,-1,1)  # gives r=sinh(1)/cosh(x)
     array([ 0.3123711 ,  1.17520119,  0.3123711 ])
     >>> hyperbolic_ratio(1,x,-1,-1) # gives r=sinh(1)/sinh(x)
     array([-0.32402714,         inf,  0.32402714])

     See also
     --------
     tran
    '''

    ak, bk, sak, sbk = np.atleast_1d(a, b, np.sign(sa), np.sign(sb))
    # old call
    # return exp(ak-bk)*(1+sak*exp(-2*ak))/(1+sbk*exp(-2*bk))
    # TODO: Does not always handle division by zero correctly

    signRatio = np.where(sak * ak < 0, sak, 1)
    signRatio = np.where(sbk * bk < 0, sbk * signRatio, signRatio)

    bk = np.abs(bk)
    ak = np.abs(ak)

    num = np.where(sak < 0, expm1(-2 * ak), 1 + exp(-2 * ak))
    den = np.where(sbk < 0, expm1(-2 * bk), 1 + exp(-2 * bk))
    iden = np.ones(den.shape) * inf
    ind = np.flatnonzero(den != 0)
    iden.flat[ind] = 1.0 / den[ind]
    val = np.where(num == den, 1, num * iden)
    # ((sak+exp(-2*ak))/(sbk+exp(-2*bk)))
    return signRatio * exp(ak - bk) * val


def sensor_typeid(*sensortypes):
    ''' Return ID for sensortype name

    Parameter
    ---------
    sensortypes : list of strings defining the sensortype

    Returns
    -------
    sensorids : list of integers defining the sensortype

    Valid senor-ids and -types for time series are as follows:
        0,  'n'    : Surface elevation              (n=Eta)
        1,  'n_t'  : Vertical surface velocity
        2,  'n_tt' : Vertical surface acceleration
        3,  'n_x'  : Surface slope in x-direction
        4,  'n_y'  : Surface slope in y-direction
        5,  'n_xx' : Surface curvature in x-direction
        6,  'n_yy' : Surface curvature in y-direction
        7,  'n_xy' : Surface curvature in xy-direction
        8,  'P'    : Pressure fluctuation about static MWL pressure
        9,  'U'    : Water particle velocity in x-direction
        10, 'V'    : Water particle velocity in y-direction
        11, 'W'    : Water particle velocity in z-direction
        12, 'U_t'  : Water particle acceleration in x-direction
        13, 'V_t'  : Water particle acceleration in y-direction
        14, 'W_t'  : Water particle acceleration in z-direction
        15, 'X_p'  : Water particle displacement in x-direction from mean pos.
        16, 'Y_p'  : Water particle displacement in y-direction from mean pos.
        17, 'Z_p'  : Water particle displacement in z-direction from mean pos.

    Example:
    >>> sensor_typeid('W','v')
    [11, 10]
    >>> sensor_typeid('rubbish')
    [nan]

    See also
    --------
    sensor_type
    '''

    sensorid_table = dict(n=0, n_t=1, n_tt=2, n_x=3, n_y=4, n_xx=5,
                          n_yy=6, n_xy=7, p=8, u=9, v=10, w=11, u_t=12,
                          v_t=13, w_t=14, x_p=15, y_p=16, z_p=17)
    try:
        return [sensorid_table.get(name.lower(), nan) for name in sensortypes]
    except:
        raise ValueError('Input must be a string!')


def sensor_type(*sensorids):
    '''
    Return sensortype name

    Parameter
    ---------
    sensorids : vector or list of integers defining the sensortype

    Returns
    -------
    sensornames : tuple of strings defining the sensortype
        Valid senor-ids and -types for time series are as follows:
        0,  'n'    : Surface elevation              (n=Eta)
        1,  'n_t'  : Vertical surface velocity
        2,  'n_tt' : Vertical surface acceleration
        3,  'n_x'  : Surface slope in x-direction
        4,  'n_y'  : Surface slope in y-direction
        5,  'n_xx' : Surface curvature in x-direction
        6,  'n_yy' : Surface curvature in y-direction
        7,  'n_xy' : Surface curvature in xy-direction
        8,  'P'    : Pressure fluctuation about static MWL pressure
        9,  'U'    : Water particle velocity in x-direction
        10, 'V'    : Water particle velocity in y-direction
        11, 'W'    : Water particle velocity in z-direction
        12, 'U_t'  : Water particle acceleration in x-direction
        13, 'V_t'  : Water particle acceleration in y-direction
        14, 'W_t'  : Water particle acceleration in z-direction
        15, 'X_p'  : Water particle displacement in x-direction from mean pos.
        16, 'Y_p'  : Water particle displacement in y-direction from mean pos.
        17, 'Z_p'  : Water particle displacement in z-direction from mean pos.

    Example:
    >>> sensor_type(range(3))
    ('n', 'n_t', 'n_tt')

    See also
    --------
    sensor_typeid, tran
    '''
    valid_names = ('n', 'n_t', 'n_tt', 'n_x', 'n_y', 'n_xx', 'n_yy', 'n_xy',
                   'p', 'u', 'v', 'w', 'u_t', 'v_t', 'w_t', 'x_p', 'y_p',
                   'z_p', nan)
    ids = atleast_1d(*sensorids)
    if isinstance(ids, list):
        ids = hstack(ids)
    n = len(valid_names) - 1
    ids = where(((ids < 0) | (n < ids)), n, ids)
    return tuple(valid_names[i] for i in ids)


class TransferFunction(object):

    '''
    Class for computing transfer functions based on linear wave theory
        of the system with input surface elevation,
        eta(x0,y0,t) = exp(i*(kx*x0+ky*y0-w*t)),
        and output Y determined by sensortype and position of sensor.

    Member methods
    --------------
    tran(w, theta, kw)

    Hw  = a function of frequency only (not direction)   size  1 x Nf
    Gwt = a function of frequency and direction          size Nt x Nf
    w = vector of angular frequencies in Rad/sec. Length Nf
    theta = vector of directions in radians           Length Nt   (default 0)
            ( theta = 0 -> positive x axis theta = pi/2 -> positive y axis)
    Member variables
    ----------------
    pos : [x,y,z],  (default [0,0,0])
        vector giving coordinate position relative to [x0 y0 z0]
    sensortype = string
        defining the sensortype or transfer function in output.
        0,  'n'    : Surface elevation              (n=Eta)     (default)
        1,  'n_t'  : Vertical surface velocity
        2,  'n_tt' : Vertical surface acceleration
        3,  'n_x'  : Surface slope in x-direction
        4,  'n_y'  : Surface slope in y-direction
        5,  'n_xx' : Surface curvature in x-direction
        6,  'n_yy' : Surface curvature in y-direction
        7,  'n_xy' : Surface curvature in xy-direction
        8,  'P'    : Pressure fluctuation about static MWL pressure
        9,  'U'    : Water particle velocity in x-direction
        10, 'V'    : Water particle velocity in y-direction
        11, 'W'    : Water particle velocity in z-direction
        12, 'U_t'  : Water particle acceleration in x-direction
        13, 'V_t'  : Water particle acceleration in y-direction
        14, 'W_t'  : Water particle acceleration in z-direction
        15, 'X_p'  : Water particle displacement in x-direction from mean pos.
        16, 'Y_p'  : Water particle displacement in y-direction from mean pos.
        17, 'Z_p'  : Water particle displacement in z-direction from mean pos.
    h : real scalar
        water depth      (default inf)
    g : real scalar
        acceleration of gravity (default 9.81 m/s**2)
    rho : real scalar
        water density    (default 1028 kg/m**3)
    bet : 1 or -1 (default 1)
        1, theta given in terms of directions toward which waves travel
        -1, theta given in terms of directions from which waves come
    igam : 1,2 or 3
        1, if z is measured positive upward from mean water level (default)
        2, if z is measured positive downward from mean water level
        3, if z is measured positive upward from sea floor
    thetax, thetay : real scalars
        angle in degrees clockwise from true north to positive x-axis and
        positive y-axis, respectively. (default theatx=90, thetay=0)

    Example
    -------
    >>> import pylab as plt
    >>> N=50; f0=0.1; th0=0; h=50; w0 = 2*pi*f0
    >>> t = np.linspace(0,15,N)
    >>> eta0 = np.exp(-1j*w0*t)
    >>> stypes = ['n', 'n_x', 'n_y'];
    >>> tf = TransferFunction(pos=(0, 0, 0), h=50)
    >>> vals = []
    >>> fh = plt.plot(t, eta0.real, 'r.')
    >>> plt.hold(True)
    >>> for i,stype in enumerate(stypes):
    ...    tf.sensortype = stype
    ...    Hw, Gwt = tf.tran(w0,th0)
    ...    vals.append((Hw*Gwt*eta0).real.ravel())

    fh = plt.plot(t, vals[i])
    plt.show()


    See also
    --------
    dat2dspec, sensor_type, sensor_typeid

    Reference
    ---------
    Young I.R. (1994)
    "On the measurement of directional spectra",
    Applied Ocean Research, Vol 16, pp 283-294
    '''

    def __init__(self, pos=(0, 0, 0), sensortype='n', h=inf, g=9.81, rho=1028,
                 bet=1, igam=1, thetax=90, thetay=0):
        self.pos = pos
        self.sensortype = sensortype if isinstance(
            sensortype, str) else sensor_type(sensortype)
        self.h = h
        self.g = g
        self.rho = rho
        self.bet = bet
        self.igam = igam
        self.thetax = thetax
        self.thetay = thetay
        self._tran_dict = dict(n=self._n, n_t=self._n_t, n_tt=self._n_tt,
                               n_x=self._n_x, n_y=self._n_y, n_xx=self._n_xx,
                               n_yy=self._n_yy, n_xy=self._n_xy,
                               P=self._p, p=self._p,
                               U=self._u, u=self._u,
                               V=self._v, v=self._v,
                               W=self._w, w=self._w,
                               U_t=self._u_t, u_t=self._u_t,
                               V_t=self._v_t, v_t=self._v_t,
                               W_t=self._w_t, w_t=self._w_t,
                               X_p=self._x_p, x_p=self._x_p,
                               Y_p=self._y_p, y_p=self._y_p,
                               Z_p=self._z_p, z_p=self._z_p)

    def tran(self, w, theta=0, kw=None):
        '''
        Return transfer functions based on linear wave theory
        of the system with input surface elevation,
        eta(x0,y0,t) = exp(i*(kx*x0+ky*y0-w*t)),
        and output,
        Y  = Hw*Gwt*eta, determined by sensortype and position of sensor.

        Parameters
        ----------
        w : array-like
            vector of angular frequencies in Rad/sec. Length Nf
        theta : array-like
            vector of directions in radians           Length Nt   (default 0)
            ( theta = 0 -> positive x axis theta = pi/2 -> positive y axis)
        kw : array-like
            vector of wave numbers corresponding to angular frequencies, w.
            Length Nf (default calculated with w2k)

        Returns
        -------
        Hw  = transfer function of frequency only (not direction) size  1 x Nf
        Gwt = transfer function of frequency and direction        size Nt x Nf

        The complete transfer function Hwt = Hw*Gwt is a function of
        w (columns) and theta (rows)                   size Nt x Nf
        '''
        if kw is None:
            # wave number as function of angular frequency
            kw, unusedkw2 = w2k(w, 0, self.h)

        w, theta, kw = np.atleast_1d(w, theta, kw)
        # make sure they have the correct orientation
        theta.shape = (-1, 1)
        kw.shape = (-1,)
        w.shape = (-1,)

        tran_fun = self._tran_dict[self.sensortype]
        Hw, Gwt = tran_fun(w, theta, kw)

        # New call to avoid singularities. pab 07.11.2000
        # Set Hw to 0 for expressions w*hyperbolic_ratio(z*k,h*k,1,-1)= 0*inf
        ind = np.flatnonzero(1 - np.isfinite(Hw))
        Hw.flat[ind] = 0

        sgn = np.sign(Hw)
        k0 = np.flatnonzero(sgn < 0)
        if len(k0):  # make sure Hw>=0 ie. transfer negative signs to Gwt
            Gwt[:, k0] = -Gwt[:, k0]
            Hw[:, k0] = -Hw[:, k0]

        if self.igam == 2:
            # pab 09 Oct.2002: bug fix
            # Changing igam by 2 should affect the directional result in the
            # same way that changing eta by -eta!
            Gwt = -Gwt
        return Hw, Gwt
    __call__ = tran

# --- Private member methods ---

    def _get_ee_cthxy(self, theta, kw):
        # convert from angle in degrees to radians
        bet = self.bet
        thxr = self.thetax * pi / 180
        thyr = self.thetay * pi / 180

        cthx = bet * cos(theta - thxr + pi / 2)
        # cthy = cos(theta-thyr-pi/2)
        cthy = bet * sin(theta - thyr)

        # Compute location complex exponential
        x, y, unused_z = list(self.pos)
        # exp(i*k(w)*(x*cos(theta)+y*sin(theta)) size Nt X Nf
        ee = exp((1j * (x * cthx + y * cthy)) * kw)
        return ee, cthx, cthy

    def _get_zk(self, kw):
        h = self.h
        z = self.pos[2]
        if self.igam == 1:
            # z measured positive upward from mean water level (default)
            zk = kw * (h + z)
        elif self.igam == 2:
            # z measured positive downward from mean water level
            zk = kw * (h - z)
        else:
            zk = kw * z  # z measured positive upward from sea floor
        return zk

    # --- Surface elevation ---
    def _n(self, w, theta, kw):
        '''n = Eta = wave profile
        '''
        ee, unused_cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        return np.ones_like(w), ee

    # --- Vertical surface velocity and acceleration ---
    def _n_t(self, w, theta, kw):
        ''' n_t = Eta_t '''
        ee, unused_cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        return w, -1j * ee

    def _n_tt(self, w, theta, kw):
        '''n_tt = Eta_tt'''
        ee, unused_cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        return w ** 2, -ee

    # --- Surface slopes ---
    def _n_x(self, w, theta, kw):
        ''' n_x = Eta_x = x-slope'''
        ee, cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        return kw, 1j * cthx * ee

    def _n_y(self, w, theta, kw):
        ''' n_y = Eta_y = y-slope'''
        ee, unused_cthx, cthy = self._get_ee_cthxy(theta, kw)
        return kw, 1j * cthy * ee

    # --- Surface curvatures ---
    def _n_xx(self, w, theta, kw):
        ''' n_xx = Eta_xx = Surface curvature (x-dir)'''
        ee, cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        return kw ** 2, -(cthx ** 2) * ee

    def _n_yy(self, w, theta, kw):
        ''' n_yy = Eta_yy = Surface curvature (y-dir)'''
        ee, unused_cthx, cthy = self._get_ee_cthxy(theta, kw)
        return kw ** 2, -cthy ** 2 * ee

    def _n_xy(self, w, theta, kw):
        ''' n_xy = Eta_xy = Surface curvature (xy-dir)'''
        ee, cthx, cthy = self._get_ee_cthxy(theta, kw)
        return kw ** 2, -cthx * cthy * ee

    # --- Pressure---
    def _p(self, w, theta, kw):
        ''' pressure fluctuations'''
        ee, unused_cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        # hyperbolic_ratio =  cosh(zk)/cosh(hk)
        return self.rho * self.g * hyperbolic_ratio(zk, hk, 1, 1), ee

    # --- Water particle velocities ---
    def _u(self, w, theta, kw):
        ''' U = x-velocity'''
        ee, cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        # w*cosh(zk)/sinh(hk), cos(theta)*ee
        return w * hyperbolic_ratio(zk, hk, 1, -1), cthx * ee

    def _v(self, w, theta, kw):
        '''V = y-velocity'''
        ee, unused_cthx, cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        # w*cosh(zk)/sinh(hk), sin(theta)*ee
        return w * hyperbolic_ratio(zk, hk, 1, -1), cthy * ee

    def _w(self, w, theta, kw):
        ''' W = z-velocity'''
        ee, unused_cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        # w*sinh(zk)/sinh(hk), -?
        return w * hyperbolic_ratio(zk, hk, -1, -1), -1j * ee

    # --- Water particle acceleration ---
    def _u_t(self, w, theta, kw):
        ''' U_t = x-acceleration'''
        ee, cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        # w^2*cosh(zk)/sinh(hk), ?
        return (w ** 2) * hyperbolic_ratio(zk, hk, 1, -1), -1j * cthx * ee

    def _v_t(self, w, theta, kw):
        ''' V_t = y-acceleration'''
        ee, unused_cthx, cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        # w^2*cosh(zk)/sinh(hk), ?
        return (w ** 2) * hyperbolic_ratio(zk, hk, 1, -1), -1j * cthy * ee

    def _w_t(self, w, theta, kw):
        ''' W_t = z-acceleration'''
        ee, unused_cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        # w*sinh(zk)/sinh(hk), ?
        return (w ** 2) * hyperbolic_ratio(zk, hk, -1, -1), -ee

    # --- Water particle displacement ---
    def _x_p(self, w, theta, kw):
        ''' X_p = x-displacement'''
        ee, cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        # cosh(zk)./sinh(hk), ?
        return hyperbolic_ratio(zk, hk, 1, -1), 1j * cthx * ee

    def _y_p(self, w, theta, kw):
        ''' Y_p = y-displacement'''
        ee, unused_cthx, cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        # cosh(zk)./sinh(hk), ?
        return hyperbolic_ratio(zk, hk, 1, -1), 1j * cthy * ee

    def _z_p(self, w, theta, kw):
        ''' Z_p = z-displacement'''
        ee, unused_cthx, unused_cthy = self._get_ee_cthxy(theta, kw)
        hk = kw * self.h
        zk = self._get_zk(kw)
        return hyperbolic_ratio(zk, hk, -1, -1), ee  # sinh(zk)./sinh(hk), ee


def wave_pressure(z, Hm0, h=10000, g=9.81, rho=1028):
    '''
    Calculate pressure amplitude due to water waves.

    Parameters
    ----------
    z : array-like
        depth where pressure is calculated [m]
    Hm0 : array-like
        significant wave height (same as the average of the 1/3'rd highest
        waves in a seastate. [m]
    h : real scalar
        waterdepth (default 10000 [m])
    g : real scalar
        acceleration of gravity (default 9.81 m/s**2)
    rho : real scalar
        water density    (default 1028 kg/m**3)


    Returns
    -------
    p : ndarray
        pressure amplitude due to water waves at water depth z. [Pa]

    PRESSURE calculate pressure amplitude due to water waves according to
    linear theory.

    Example
    -----
    >>> import pylab as plt
    >>> z = -np.linspace(10,20)
    >>> fh = plt.plot(z, wave_pressure(z, Hm0=1, h=20))
    >>> plt.show()

    See also
    --------
    w2k

    '''

    # Assume seastate with jonswap spectrum:
    Tp = 4 * np.sqrt(Hm0)
    gam = _MODELS.jonswap_peakfact(Hm0, Tp)
    Tm02 = Tp / (1.30301 - 0.01698 * gam + 0.12102 / gam)
    w = 2 * np.pi / Tm02
    kw, unused_kw2 = w2k(w, 0, h)

    hk = kw * h
    zk1 = kw * z
    zk = hk + zk1  # z measured positive upward from mean water level (default)
    #  zk = hk-zk1  # z measured positive downward from mean water level
    #  zk1 = -zk1
    #  zk = zk1  # z measured positive upward from sea floor

    #  cosh(zk)/cosh(hk) approx exp(zk) for large h
    #  hyperbolic_ratio(zk,hk,1,1) = cosh(zk)/cosh(hk)
    # pr = np.where(np.pi < hk, np.exp(zk1), hyperbolic_ratio(zk, hk, 1, 1))
    pr = hyperbolic_ratio(zk, hk, 1, 1)
    pressure = (rho * g * Hm0 / 2) * pr

    #    pos = [np.zeros_like(z),np.zeros_like(z),z]
    #    tf = TransferFunction(pos=pos, sensortype='p', h=h, rho=rho, g=g)
    #    Hw, Gwt = tf.tran(w,0)
    #    pressure2 = np.abs(Hw) * Hm0 / 2

    return pressure


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


def main():
    sensor_type(range(21))
if __name__ == '__main__':
    test_docstrings()
