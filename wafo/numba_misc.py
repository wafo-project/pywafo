'''
Created on 6. okt. 2016

@author: pab
'''
from __future__ import absolute_import, division
from numba import jit, float64, int64, int32, int8, void
import numpy as np


@jit(int64(int64[:], int8[:]))
def _findcross(ind, y):
    ix, dcross, start, v = 0, 0, 0, 0
    n = len(y)
    if y[0] < v:
        dcross = -1  # first is a up-crossing
    elif y[0] > v:
        dcross = 1  # first is a down-crossing
    elif y[0] == v:
        # Find out what type of crossing we have next time..
        for i in range(1, n):
            start = i
            if y[i] < v:
                ind[ix] = i - 1  # first crossing is a down crossing
                ix += 1
                dcross = -1  # The next crossing is a up-crossing
                break
            elif y[i] > v:
                ind[ix] = i - 1  # first crossing is a up-crossing
                ix += 1
                dcross = 1  # The next crossing is a down-crossing
                break

    for i in range(start, n - 1):
        if ((dcross == -1 and y[i] <= v and v < y[i + 1]) or
                (dcross == 1 and v <= y[i] and y[i + 1] < v)):

            ind[ix] = i
            ix += 1
            dcross = -dcross
    return ix


def findcross(xn):
    '''Return indices to zero up and downcrossings of a vector
    '''
    ind = np.empty(len(xn), dtype=np.int64)
    m = _findcross(ind, xn)
    return ind[:m]


def _make_findrfc(cmp1, cmp2):

    @jit(int64(int64[:], float64[:], float64), nopython=True)
    def findrfc2(t, y, h):
        # cmp1, cmp2 = (a_le_b, a_lt_b) if method==0 else (a_lt_b, a_le_b)

        n = len(y)
        j, t0, z0 = 0, 0, 0
        y0 = y[t0]
        # The rainflow filter
        for ti in range(1, n):
            fpi = y0 + h
            fmi = y0 - h
            yi = y[ti]

            if z0 == 0:
                if cmp1(yi, fmi):
                    z1 = -1
                elif cmp1(fpi, yi):
                    z1 = +1
                else:
                    z1 = 0
                t1, y1 = (t0, y0) if z1 == 0 else (ti, yi)
            else:
                if (((z0 == +1) and cmp1(yi, fmi)) or
                        ((z0 == -1) and cmp2(yi, fpi))):
                    z1 = -1
                elif (((z0 == +1) and cmp2(fmi, yi)) or
                        ((z0 == -1) and cmp1(fpi, yi))):
                    z1 = +1
                else:
                    raise ValueError
                #     warnings.warn('Something wrong, i={}'.format(tim1))

                # Update y1
                if z1 != z0:
                    t1, y1 = ti, yi
                elif z1 == -1:
                    t1, y1 = (t0, y0) if y0 < yi else (ti, yi)
                elif z1 == +1:
                    t1, y1 = (t0, y0) if y0 > yi else (ti, yi)

            # Update y if y0 is a turning point
            if abs(z0 - z1) == 2:
                j += 1
                t[j] = t0

            # Update t0, y0, z0
            t0, y0, z0 = t1, y1, z1
        # end

        # Update y if last y0 is greater than (or equal) threshold
        if cmp2(h, abs(y0 - y[t[j]])):
            j += 1
            t[j] = t0
        return j + 1
    return findrfc2


@jit(int32(float64, float64), nopython=True)
def a_le_b(a, b):
    return a <= b


@jit(int32(float64, float64), nopython=True)
def a_lt_b(a, b):
    return a < b


_findrfc_le = _make_findrfc(a_le_b, a_lt_b)
_findrfc_lt = _make_findrfc(a_lt_b, a_le_b)


@jit(int64(int64[:], float64[:], float64), nopython=True)
def _findrfc(ind, y, h):
    n = len(y)
    t_start = 0
    NC = n // 2 - 1
    ix = 0
    for i in range(NC):
        Tmi = t_start + 2 * i
        Tpl = t_start + 2 * i + 2
        xminus = y[2 * i]
        xplus = y[2 * i + 2]

        if(i != 0):
            j = i - 1
            while ((j >= 0) and (y[2 * j + 1] <= y[2 * i + 1])):
                if (y[2 * j] < xminus):
                    xminus = y[2 * j]
                    Tmi = t_start + 2 * j
                j -= 1
        if (xminus >= xplus):
            if (y[2 * i + 1] - xminus >= h):
                ind[ix] = Tmi
                ix += 1
                ind[ix] = (t_start + 2 * i + 1)
                ix += 1
            # goto L180 continue
        else:
            j = i + 1
            while (j < NC):
                if (y[2 * j + 1] >= y[2 * i + 1]):
                    break  # goto L170
                if((y[2 * j + 2] <= xplus)):
                    xplus = y[2 * j + 2]
                    Tpl = (t_start + 2 * j + 2)
                j += 1
            else:
                if ((y[2 * i + 1] - xminus) >= h):
                    ind[ix] = Tmi
                    ix += 1
                    ind[ix] = (t_start + 2 * i + 1)
                    ix += 1
                # iy = i
                continue

            # goto L180
            # L170:
            if (xplus <= xminus):
                if ((y[2 * i + 1] - xminus) >= h):
                    ind[ix] = Tmi
                    ix += 1
                    ind[ix] = (t_start + 2 * i + 1)
                    ix += 1
            elif ((y[2 * i + 1] - xplus) >= h):
                ind[ix] = (t_start + 2 * i + 1)
                ix += 1
                ind[ix] = Tpl
                ix += 1

        # L180:
        # iy=i
    #  /* for i */
    return ix


def findrfc(y, h, method=0):
    n = len(y)
    t = np.zeros(n, dtype=np.int64)
    findrfc_ = [_findrfc_le, _findrfc_lt, _findrfc][method]
    m = findrfc_(t, y, h)
    return t[:m]


@jit(void(float64[:], float64[:], float64[:], float64[:],
          float64[:], float64[:], float64, float64,
          int32, int32, int32, int32), nopython=True)
def _finite_water_disufq(rvec, ivec, rA, iA, w, kw, h, g, nmin, nmax, m, n):
    # kfact is set to 2 in order to exploit the symmetry.
    # If you set kfact to 1, you must uncomment all statements
    # including the expressions: rvec[iz2], rvec[iv2], ivec[iz2] and ivec[iv2].

    kfact = 2.0
    for ix in range(nmin - 1, nmax):
        # for (ix = nmin-1;ix<nmax;ix++) {
        kw1 = kw[ix]
        w1 = w[ix]
        tmp1 = np.tanh(kw1 * h)
        # Cg, wave group velocity
        Cg = 0.5 * g * (tmp1 + kw1 * h * (1.0 - tmp1 * tmp1)) / w1  # OK
        tmp1 = 0.5 * g * (kw1 / w1) * (kw1 / w1)
        tmp2 = 0.5 * w1 * w1 / g
        tmp3 = g * kw1 / (w1 * Cg)
        tmp4 = kw1 / np.sinh(2.0 * kw1 * h) if kw1 * h < 300.0 else 0.0

        # Difference frequency effects finite water depth
        Edij = (tmp1 - tmp2 + tmp3) / (1.0 - g * h / (Cg * Cg)) - tmp4  # OK

        # Sum frequency effects finite water depth
        Epij = (3.0 * (tmp1 - tmp2) /
                (1.0 - tmp1 / kw1 * np.tanh(2.0 * kw1 * h)) +
                3.0 * tmp2 - tmp1)  # OK
        # printf("Edij = %f Epij = %f \n", Edij,Epij);

        ixi = ix * m
        iz1 = 2 * ixi
        # iz2 = n*m-ixi;
        for i in range(m):
            rrA = rA[ixi] * rA[ixi]
            iiA = iA[ixi] * iA[ixi]
            riA = rA[ixi] * iA[ixi]

            # Sum frequency effects along the diagonal
            rvec[iz1] += kfact * (rrA - iiA) * Epij
            ivec[iz1] += kfact * 2.0 * riA * Epij
            # rvec[iz2] +=  kfact*(rrA-iiA)*Epij;
            # ivec[iz2] -=  kfact*2.0*riA*Epij;
            # iz2++;

            # Difference frequency effects along the diagonal
            # are only contributing to the mean
            rvec[i] += 2.0 * (rrA + iiA) * Edij
            ixi += 1
            iz1 += 1
            # }
        for jy in range(ix + 1, nmax):
            # w1  = w[ix];
            # kw1 = kw[ix];
            w2 = w[jy]
            kw2 = kw[jy]
            tmp1 = g * (kw1 / w1) * (kw2 / w2)
            tmp2 = 0.5 / g * (w1 * w1 + w2 * w2 + w1 * w2)
            tmp3 = 0.5 * g * \
                (w1 * kw2 * kw2 + w2 * kw1 * kw1) / (w1 * w2 * (w1 + w2))
            tmp4 = (1 - g * (kw1 + kw2) / (w1 + w2) / (w1 + w2) *
                    np.tanh((kw1 + kw2) * h))
            Epij = (tmp1 - tmp2 + tmp3) / tmp4 + tmp2 - 0.5 * tmp1  # OK */

            tmp2 = 0.5 / g * (w1 * w1 + w2 * w2 - w1 * w2)  # OK*/
            tmp3 = -0.5 * g * \
                (w1 * kw2 * kw2 - w2 * kw1 * kw1) / (w1 * w2 * (w1 - w2))
            tmp4 = (1.0 - g * (kw1 - kw2) / (w1 - w2) /
                    (w1 - w2) * np.tanh((kw1 - kw2) * h))
            Edij = (tmp1 - tmp2 + tmp3) / tmp4 + tmp2 - 0.5 * tmp1  # OK */
            # printf("Edij = %f Epij = %f \n", Edij,Epij);

            ixi = ix * m
            jyi = jy * m
            iz1 = ixi + jyi
            iv1 = jyi - ixi
            # iz2 = (n*m-iz1)
            # iv2 = n*m-iv1
            for i in range(m):
                # for (i=0;i<m;i++,ixi++,jyi++,iz1++,iv1++) {
                rrA = rA[ixi] * rA[jyi]  # rrA = rA[i][ix]*rA[i][jy];
                iiA = iA[ixi] * iA[jyi]  # iiA = iA[i][ix]*iA[i][jy];
                riA = rA[ixi] * iA[jyi]  # riA = rA[i][ix]*iA[i][jy];
                irA = iA[ixi] * rA[jyi]  # irA = iA[i][ix]*rA[i][jy];

                # Sum frequency effects */
                tmp1 = kfact * 2.0 * (rrA - iiA) * Epij
                tmp2 = kfact * 2.0 * (riA + irA) * Epij
                rvec[iz1] += tmp1  # rvec[i][jy+ix] += tmp1
                ivec[iz1] += tmp2  # ivec[i][jy+ix] += tmp2
                # rvec[iz2] += tmp1 # rvec[i][n*m-(jy+ix)] += tmp1
                # ivec[iz2] -= tmp2 # ivec[i][n*m-(jy+ix)] -= tmp2
                # iz2++

                # Difference frequency effects */
                tmp1 = kfact * 2.0 * (rrA + iiA) * Edij
                tmp2 = kfact * 2.0 * (riA - irA) * Edij
                rvec[iv1] += tmp1  # rvec[i][jy-ix] += tmp1
                ivec[iv1] += tmp2  # ivec[i][jy-ix] -= tmp2

                # rvec[iv2] += tmp1
                # ivec[iv2] -= tmp2
                # iv2 += 1
                ixi += 1
                jyi += 1
                iz1 += 1
                iv1 += 1


@jit(void(float64[:], float64[:], float64[:], float64[:],
          float64[:], float64[:], float64, float64,
          int32, int32, int32, int32), nopython=True)
def _deep_water_disufq(rvec, ivec, rA, iA, w, kw, h, g, nmin, nmax, m, n):
    # kfact is set to 2 in order to exploit the symmetry.
    # If you set kfact to 1, you must uncomment all statements
    # including the expressions: rvec[iz2], rvec[iv2], ivec[iz2] and ivec[iv2].

    kfact = 2.0
    for ix in range(nmin - 1, nmax):
        ixi = ix * m
        iz1 = 2 * ixi
        iz2 = n * m - ixi
        kw1 = kw[ix]
        Epij = kw1
        for _i in range(m):
            rrA = rA[ixi] * rA[ixi]
            iiA = iA[ixi] * iA[ixi]
            riA = rA[ixi] * iA[ixi]

            # Sum frequency effects along the diagonal
            tmp1 = kfact * (rrA - iiA) * Epij
            tmp2 = kfact * 2.0 * riA * Epij
            rvec[iz1] += tmp1
            ivec[iz1] += tmp2
            ixi += 1
            iz1 += 1
            # rvec[iz2] += tmp1
            # ivec[iz2] -= tmp2
            iz2 += 1

            # Difference frequency effects are zero along the diagonal
            # and are thus not contributing to the mean.
        for jy in range(ix + 1, nmax):
            kw2 = kw[jy]
            Epij = 0.5 * (kw2 + kw1)
            Edij = -0.5 * (kw2 - kw1)
            # printf("Edij = %f Epij = %f \n", Edij,Epij);

            ixi = ix * m
            jyi = jy * m
            iz1 = ixi + jyi
            iv1 = jyi - ixi
            iz2 = (n*m-iz1)
            iv2 = (n*m-iv1)
            for _i in range(m):
                rrA = rA[ixi] * rA[jyi]  # rrA = rA[i][ix]*rA[i][jy]
                iiA = iA[ixi] * iA[jyi]  # iiA = iA[i][ix]*iA[i][jy]
                riA = rA[ixi] * iA[jyi]  # riA = rA[i][ix]*iA[i][jy]
                irA = iA[ixi] * rA[jyi]  # irA = iA[i][ix]*rA[i][jy]

                # Sum frequency effects
                tmp1 = kfact * 2.0 * (rrA - iiA) * Epij
                tmp2 = kfact * 2.0 * (riA + irA) * Epij
                rvec[iz1] += tmp1  # rvec[i][ix+jy] += tmp1
                ivec[iz1] += tmp2  # ivec[i][ix+jy] += tmp2
                # rvec[iz2] += tmp1  # rvec[i][n*m-(ix+jy)] += tmp1
                # ivec[iz2] -= tmp2  # ivec[i][n*m-(ix+jy)] -= tmp2
                iz2 += 1

                # Difference frequency effects */
                tmp1 = kfact * 2.0 * (rrA + iiA) * Edij
                tmp2 = kfact * 2.0 * (riA - irA) * Edij

                rvec[iv1] += tmp1  # rvec[i][jy-ix] += tmp1
                ivec[iv1] += tmp2  # ivec[i][jy-ix] += tmp2

                # rvec[iv2] += tmp1  # rvec[i][n*m-(jy-ix)] += tmp1
                # ivec[iv2] -= tmp2  # ivec[i][n*m-(jy-ix)] -= tmp2
                iv2 += 1
                ixi += 1
                jyi += 1
                iz1 += 1
                iv1 += 1


def disufq(rA, iA, w, kw, h, g, nmin, nmax, m, n):
    """
    DISUFQ  Is an internal function to spec2nlsdat

    Parameters
    ----------
    rA, iA     = real and imaginary parts of the amplitudes (size m X n).
    w          = vector with angular frequencies (w>=0)
    kw         = vector with wavenumbers (kw>=0)
    h          = water depth             (h >=0)
    g          = constant acceleration of gravity
    nmin       = minimum index where rA(:,nmin) and iA(:,nmin) is
                 greater than zero.
    nmax       = maximum index where rA(:,nmax) and iA(:,nmax) is
                 greater than zero.
    m          = size(rA,1),size(iA,1)
    n          = size(rA,2),size(iA,2), or size(rvec,2),size(ivec,2)

    returns
    -------
    rvec, ivec = real and imaginary parts of the resultant  (size m X n).

    DISUFQ returns the summation of difference frequency and sum
    frequency effects in the vector vec = rvec +sqrt(-1)*ivec.
    The 2'nd order contribution to the Stokes wave is then calculated by
    a simple 1D Fourier transform, real(FFT(vec)).

    """
    rvec = np.zeros(n * m)
    ivec = np.zeros(n * m)

    if h > 10000:  # { /* deep water /Inifinite water depth */
        _deep_water_disufq(rvec, ivec, rA, iA, w, kw, h, g, nmin, nmax, m, n)
    else:
        _finite_water_disufq(rvec, ivec, rA, iA, w, kw, h, g, nmin, nmax, m, n)
    return rvec, ivec


@jit(int32[:](float64[:], float64[:], float64[:, :]))
def _findrfc3_astm(array_ext, a, array_out):
    """
    Rain flow without time analysis

    Return [ampl ampl_mean nr_of_cycle]

    By Adam Nieslony
    Visit the MATLAB Central File Exchange for latest version
    http://www.mathworks.com/matlabcentral/fileexchange/3026
    """
    n = len(array_ext)
    po = 0
    # The original rainflow counting by Nieslony, unchanged
    j = -1
    c_nr1 = 1
    for i in range(n):
        j += 1
        a[j] = array_ext[i]
        while j >= 2 and abs(a[j - 1] - a[j - 2]) <= abs(a[j] - a[j - 1]):
            ampl = abs((a[j - 1] - a[j - 2]) / 2)
            mean = (a[j - 1] + a[j - 2]) / 2
            if j == 2:
                a[0] = a[1]
                a[1] = a[2]
                j = 1
                if (ampl > 0):
                    array_out[po, :] = (ampl, mean, 0.5)
                    po += 1
            else:
                a[j - 2] = a[j]
                j = j - 2
                if (ampl > 0):
                    array_out[po, :] = (ampl, mean, 1.0)
                    po += 1
                    c_nr1 += 1

    c_nr2 = 1
    for i in range(j):
        ampl = abs(a[i] - a[i + 1]) / 2
        mean = (a[i] + a[i + 1]) / 2
        if (ampl > 0):
            array_out[po, :] = (ampl, mean, 0.5)
            po += 1
            c_nr2 += 1
    return c_nr1, c_nr2


@jit(int32[:](float64[:], float64[:], float64[:], float64[:], float64[:, :]))
def _findrfc5_astm(array_ext, array_t, a, t, array_out):
    """
    Rain flow with time analysis

    returns
    [ampl ampl_mean nr_of_cycle cycle_begin_time cycle_period_time]

    By Adam Nieslony
    Visit the MATLAB Central File Exchange for latest version
    http://www.mathworks.com/matlabcentral/fileexchange/3026
    """
    n = len(array_ext)
    po = 0
    # The original rainflow counting by Nieslony, unchanged
    j = -1
    c_nr1 = 1
    for i in range(n):
        j += 1
        a[j] = array_ext[i]
        t[j] = array_t[i]
        while (j >= 2) and (abs(a[j - 1] - a[j - 2]) <= abs(a[j] - a[j - 1])):
            ampl = abs((a[j - 1] - a[j - 2]) / 2)
            mean = (a[j - 1] + a[j - 2]) / 2
            period = (t[j - 1] - t[j - 2]) * 2
            atime = t[j - 2]
            if j == 2:
                a[0] = a[1]
                a[1] = a[2]
                t[0] = t[1]
                t[1] = t[2]
                j = 1
                if (ampl > 0):
                    array_out[po, :] = (ampl, mean, 0.5, atime, period)
                    po += 1
            else:
                a[j - 2] = a[j]
                t[j - 2] = t[j]
                j = j - 2
                if (ampl > 0):
                    array_out[po, :] = (ampl, mean, 1.0, atime, period)
                    po += 1
                    c_nr1 += 1

    c_nr2 = 1
    for i in range(j):
        # for (i=0; i<j; i++) {
        ampl = abs(a[i] - a[i + 1]) / 2
        mean = (a[i] + a[i + 1]) / 2
        period = (t[i + 1] - t[i]) * 2
        atime = t[i]
        if (ampl > 0):
            array_out[po, :] = (ampl, mean, 0.5, atime, period)
            po += 1
            c_nr2 += 1
    return c_nr1, c_nr2


def findrfc_astm(tp, t=None):
    """
    Return rainflow counted cycles

    Nieslony's Matlab implementation of the ASTM standard practice for rainflow
    counting ported to a Python C module.

    Parameters
    ----------
    tp : array-like
        vector of turningpoints (NB! Only values, not sampled times)
    t : array-like
        vector of sampled times

    Returns
    -------
    sig_rfc : array-like
        array of shape (n,3) or (n, 5) with:
        sig_rfc[:,0] Cycles amplitude
        sig_rfc[:,1] Cycles mean value
        sig_rfc[:,2] Cycle type, half (=0.5) or full (=1.0)
        sig_rfc[:,3] cycle_begin_time (only if t is given)
        sig_rfc[:,4] cycle_period_time (only if t is given)
    """

    y1 = np.atleast_1d(tp).ravel()
    n = len(y1)
    a = np.zeros(n)
    if t is None:
        sig_rfc = np.zeros((n, 3))
        cnr = _findrfc3_astm(y1, a, sig_rfc)
    else:
        t1 = np.atleast_1d(t).ravel()
        sig_rfc = np.zeros((n, 5))
        t2 = np.zeros(n)
        cnr = _findrfc5_astm(y1, t1, a, t2, sig_rfc)
    # the sig_rfc was constructed too big in rainflow.rf3, so
    # reduce the sig_rfc array as done originally by a matlab mex c function
    # n = len(sig_rfc)
    return sig_rfc[:n - cnr[0]]

if __name__ == '__main__':
    pass
