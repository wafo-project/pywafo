import numpy as np


def nt2fr(nt, kind=0):
    """
    NT2FR  Calculates the frequency matrix given the counting distribution matrix.

    Parameters
    ----------
    nt  = the counting distribution matrix,
    kind = 0,1

    Returns
    -------
    fr  = the frequency matrix,

      If kind=0 function computes the inverse to

         N_T(u,v) = #{ (M_i,m_i); M_i>u, m_i<v }

      and if def=1 the inverse to

         N_T(u,v) = #{ (M_i,m_i); M_i>=u, m_i=<v }

      where  (M_i,m_i)  are  cycles and v,u are in the discretization grid.
    """

    n = len(nt)
    fr_raw = (nt[:n - 1, :][:, n - 1] + nt[1:, :][:, 1:] -
              nt[:n - 1, :][:, 1:] - nt[1:, :][:, :n - 1])

    fr = np.zeros(np.shape(nt))
    if kind == 0:
        fr[:n - 1, :n - 1] = fr_raw
    else:
        fr[1:, 1:] = fr_raw
    return fr


def fr2nt(fr):
    """
    FR2NT  Calculates the counting distribution given the frequency matrix.

    Parameters
    ----------
    fr = a square frequency matrix for a cycle count.

    Returns
    ---------
    nt = a square counting distribution matrix for a cycle count,

    Copyright 1993, Mats Frendahl, Dept. of Math. Stat., University of Lund.
    """

    n, m = np.shape(fr)
    if n < 3:
        raise ValueError('The dimension must be 3 or larger!')
    if n != m:
        raise ValueError('The matrix is not square!')

    m1 = np.cumsum(np.cumsum(fr, axis=1) - fr)
    m2 = np.zeros((n, n))
    m2[1:, 1:] = m1[:n - 2, :][:, 1:]
    nt = np.fliplr(np.triu(np.fliplr(m2), 0))

    return nt


def iter_(frfc, fmM_0=None, k=1, epsilon=1e-5):
    """
    ITER  Calculates a Markov matrix given a rainflow matrix

     CALL: [fmM_k frfc_k] = iter_(frfc, fmM_0, k, eps)

    Parameters
    ----------
    frfc   = the rainflow matrix to be inverted,
    fmM_0  = the first approximation to the Markov matrix, if not
             specified  fmM_0=frfc,
    k      = number of iterations, if not specified, k=1.
    eps    = a convergence treshold, default value; eps=0.00001

    Return
    ------
    fmM_k  = the solution to the equation frfc = fmM + F(fmM),
    frfc_k = the rainflow matrix; frfc_k = fmM_k + F(fmM_k).


    See also
    --------
    iter_mc, spec2cmat, mctp2rfm, mc2rfm

    References
    ----------
    Rychlik, I. (1996)
    'Simulation of load sequences from Rainflow matrices: Markov method'
    Int. J. Fatigue, Vol 18, pp 429-438
    """
    return _raw_iter(mctp2rfc, frfc, fmM_0, k, epsilon)


def _raw_iter(fun2rfc, frfc, fmM_0=None, k=1, epsilon=1e-5):
    if fmM_0 is None:
        fmM_0 = frfc

    frfc0 = np.fliplr(frfc)
    fmM = np.fliplr(fmM_0)
    for _i in range(k):
        fmM_old = fmM
        frfc = fun2rfc(fmM)
        fmM = fmM_old + (frfc0 - frfc)
        fmM = np.maximum(0, fmM)
        # check = [k-i+1,  sum(sum(abs(fmM_old-fmM))), sum(sum(frfc0))/sum(sum(fmM))];
        # disp(['iteration step, accuracy  ', num2str(check(1:2))])
        converged = not np.sum(np.abs(fmM_old - fmM)) > epsilon
        if converged:
            break

    F = np.fliplr(fmM)
    frfc = np.fliplr(fun2rfc(fmM))

    return F, frfc


def iter_mc(frfc, fmM_0=None, k=1, epsilon=1e-5):
    """
    ITER_MC  Calculates a kernel of a MC given a rainflow matrix

        Solves  f_rfc = f_xy + F_mc(f_xy) for f_xy.

      Call: [fmM_k frfc_k]=iter_mc(frfc,fmM_0,k,eps)

       fmM_k  = the solution to the equation frfc = fmM + F(fmM),
       frfc_k = the rainflow matrix; frfc_k = fmM_k + F(fmM_k).


       frfc   = the rainflow matrix to be inverted,
       fmM_0  = the first approximation to the Markov matrix, if not
                specified  fmM_0=frfc,
       k      = number of iterations, if not specified, k=1.
       eps    = a convergence treshold, default value; eps=0.00001

     See also
     --------
     iter_, spec2cmat, mctp2rfm, mc2rfm

     References
     ----------
     Rychlik, I. (1996)
     'Simulation of load sequences from Rainflow matrices: Markov method'
     Int. J. Fatigue, Vol 18, pp 429-438
    """
    return _raw_iter(mc2rfc, frfc, fmM_0, k, epsilon)


def _raise_kind_error(kind):
    if kind in (-1, 0):
        raise NotImplementedError('kind = {} not yet implemented'.format(kind))
    else:
        raise ValueError('kind = {}: not a valid value of kind'.format(kind))


def nt2cmat(nt, kind=1):
    """
    Return cycle matrix from a counting distribution.

    Parameters
    ----------
    NT: 2D array
        Counting distribution. [nxn]
    kind     =  1: causes peaks to be projected upwards and troughs
                   downwards to the closest discrete level (default).
             =  0: causes peaks and troughs to be projected to
                   the closest discrete level.
             = -1: causes peaks to be projected downwards and the
                   troughs upwards to the closest discrete level.

    Returns
    -------
    cmat = Cycle matrix. [nxn]

    Examples
    --------
    >>> import numpy as np
    >>> cmat0 = np.round(np.triu(np.random.rand(4, 4), 1)*10)
    >>> cmat0 = np.array([[ 0.,  5.,  6.,  9.],
    ...                   [ 0.,  0.,  1.,  7.],
    ...                   [ 0.,  0.,  0.,  4.],
    ...                   [ 0.,  0.,  0.,  0.]])

    >>> nt = cmat2nt(cmat0)
    >>> np.allclose(nt,
    ...    [[  0.,   0.,   0.,   0.],
    ...    [ 20.,  15.,   9.,   0.],
    ...    [ 28.,  23.,  16.,   0.],
    ...    [ 32.,  27.,  20.,   0.]])
    True
    >>> cmat = nt2cmat(nt)
    >>> np.allclose(cmat, [[ 0.,  5.,  6.,  9.],
    ...                    [ 0.,  0.,  1.,  7.],
    ...                    [ 0.,  0.,  0.,  4.],
    ...                    [ 0.,  0.,  0.,  0.]])
    True

    See also
    --------
    cmat2nt
    """
    n = len(nt)  # Number of discrete levels
    if kind == 1:
        I = np.r_[0:n - 1]
        J = np.r_[1:n]
        c = nt[I + 1][:, J - 1] - nt[I][:, J - 1] - nt[I + 1][:, J] + nt[I][:, J]
        c2 = np.vstack((c, np.zeros((n - 1))))
        cmat = np.hstack((np.zeros((n, 1)), c2))
    elif kind == 11:  # same as def=1 but using for-loop
        cmat = np.zeros((n, n))
        j = np.r_[1:n]
        for i in range(n - 1):
            cmat[i, j] = nt[i + 1, j - 1] - nt[i, j - 1] - nt[i + 1, j] + nt[i, j]
    else:
        _raise_kind_error(kind)
    return cmat


def cmat2nt(cmat, kind=1):
    """
    CMAT2NT Calculates a counting distribution from a cycle matrix.

    Parameters
    ----------
     cmat     = Cycle matrix. [nxn]
     kind     =  1: causes peaks to be projected upwards and troughs
                    downwards to the closest discrete level (default).
              =  0: causes peaks and troughs to be projected to
                    the closest discrete level.
              = -1: causes peaks to be projected downwards and the
                    troughs upwards to the closest discrete level.
    Returns
    -------
    NT: n x n array
        Counting distribution.

    Examples
    --------
    >>> import numpy as np
    >>> cmat0 = np.round(np.triu(np.random.rand(4, 4), 1)*10)
    >>> cmat0 = np.array([[ 0.,  5.,  6.,  9.],
    ...                   [ 0.,  0.,  1.,  7.],
    ...                   [ 0.,  0.,  0.,  4.],
    ...                   [ 0.,  0.,  0.,  0.]])

    >>> nt = cmat2nt(cmat0, kind=11)
    >>> np.allclose(nt,
    ...    [[  0.,   0.,   0.,   0.],
    ...    [ 20.,  15.,   9.,   0.],
    ...    [ 28.,  23.,  16.,   0.],
    ...    [ 32.,  27.,  20.,   0.]])
    True

    >>> cmat = nt2cmat(nt, kind=11)
    >>> np.allclose(cmat, [[ 0.,  5.,  6.,  9.],
    ...                    [ 0.,  0.,  1.,  7.],
    ...                    [ 0.,  0.,  0.,  4.],
    ...                    [ 0.,  0.,  0.,  0.]])
    True

    See also
    --------
    nt2cmat
    """
    n = len(cmat)  # Number of discrete levels
    nt = np.zeros((n, n))

    if kind == 1:
        csum = np.cumsum
        flip = np.fliplr
        nt[1:n, :n - 1] = flip(csum(flip(csum(cmat[:-1, 1:], axis=0)), axis=1))
    elif kind == 11:  # same as def=1 but using for-loop
        # j = np.r_[1:n]
        for i in range(1, n):
            for j in range(n - 1):
                nt[i, j] = np.sum(cmat[:i, j + 1:n])
    else:
        _raise_kind_error(kind)
    return nt


def mctp2tc(f_Mm, utc, param, f_mM=None):
    """
    MCTP2TC  Calculates frequencies for the  upcrossing troughs and crests
    using Markov chain of turning points.

    Parameters
    ----------
    f_Mm  = the frequency matrix for the Max2min cycles,
    utc   = the reference level,
    param = a vector defining the discretization used to compute f_Mm,
            note that f_mM has to be computed on the same grid as f_mM.
    f_mM  = the frequency matrix for the min2Max cycles.

    Returns
    -------
    f_tc  = the matrix with frequences of upcrossing troughs and crests,

    Examples
    --------
    >>> fmM = np.array([[ 0.0183,    0.0160,    0.0002,    0.0000,         0],
    ...            [0.0178,    0.5405,    0.0952,         0,         0],
    ...            [0.0002,    0.0813,         0,         0,         0],
    ...            [0.0000,         0,         0,         0,         0],
    ...            [     0,         0,         0,         0,         0]])
    >>> param = (-1, 1, len(fmM))
    >>> utc = 0
    >>> f_tc = mctp2tc(fmM, utc, param)
    >>> np.allclose(f_tc,
    ...     [[ 0.0,   1.59878359e-02,  -1.87345256e-04,   0.0,   0.0],
    ...      [ 0.0,   5.40312726e-01,   3.86782958e-04,   0.0,   0.0],
    ...      [ 0.0,   0.0,   0.0,   0.0,   0.0],
    ...      [ 0.0,   0.0,   0.0,   0.0,   0.0],
    ...      [ 0.0,   0.0,   0.0,   0.0,   0.0]])
    True
    """
    def _check_ntc(ntc, n):
        if ntc > n - 1:
            raise IndexError('index for mean-level out of range, stop')

    def _check_discretization(param, ntc):
        if not (1 < ntc < param[2]):
            raise ValueError('the reference level out of range, stop')

    def _normalize_rows(arr):
        n = len(arr)
        for i in range(n):
            rowsum = np.sum(arr[i])
            if rowsum != 0:
                arr[i] = arr[i] / rowsum
        return arr

    def _make_tempp(P, Ph, i, ntc):
        Ap = P[i:ntc - 1, i + 1:ntc]
        Bp = Ph[i + 1:ntc, i:ntc - 1]
        dim_p = ntc - 1 - i
        tempp = np.zeros((dim_p, 1))
        I = np.eye(len(Ap))
        if i == 1:
            e = Ph[i + 1:ntc, 0]
        else:
            e = np.sum(Ph[i + 1:ntc, :i - 1], axis=1)

        if max(abs(e)) > 1e-10:
            if dim_p == 1:
                tempp[0] = (Ap / (1 - Bp * Ap) * e)
            else:
                rh = I - np.dot(Bp, Ap)
                tempp = np.dot(Ap, np.linalg.solve(rh, e))

            # end
        # end
        return tempp

    def _make_tempm(P, Ph, j, ntc, n):
        Am = P[ntc - 1:j, ntc:j + 1]
        Bm = Ph[ntc:j + 1, ntc - 1:j]
        dim_m = j - ntc + 1
        tempm = np.zeros((dim_m, 1))
        Im = np.eye(len(Am))
        if j == n - 1:
            em = P[ntc - 1:j, n]
        else:
            em = np.sum(P[ntc - 1:j, j + 1:n], axis=1)
        # end
        if max(abs(em)) > 1e-10:
            if dim_m == 1:
                tempm[0] = (Bm / (1 - Am * Bm) * em)
            else:
                rh = Im - np.dot(Am, Bm)
                tempm = np.dot(Bm, np.linalg.lstsq(rh, em)[0])
            # end
        # end
        return tempm

    if f_mM is None:
        f_mM = np.copy(f_Mm) * 1.0

    u = np.linspace(*param)
    udisc = u[::-1]  # np.fliplr(u)
    ntc = np.sum(udisc >= utc)
    n = len(f_Mm)
    _check_ntc(ntc, n)
    _check_discretization(param, ntc)

    # normalization of frequency matrices
    f_Mm = _normalize_rows(f_Mm)
    P = np.fliplr(f_Mm)
    Ph = np.rot90(np.fliplr(f_mM * 1.0), -1)
    Ph = _normalize_rows(Ph)
    Ph = np.fliplr(Ph)

    F = np.zeros((n, n))
    F[:ntc - 1, :(n - ntc)] = f_mM[:ntc - 1, :(n - ntc)]
    F = cmat2nt(F)

    for i in range(1, ntc):
        for j in range(ntc - 1, n - 1):
            if i < ntc - 1:
                tempp = _make_tempp(P, Ph, i, ntc)
                b = np.dot(np.dot(tempp.T, f_mM[i:ntc - 1, n - j - 2::-1]),
                           np.ones((n - j - 1, 1)))
            # end
            if j > ntc - 1:
                tempm = _make_tempm(P, Ph, j, ntc, n)
                c = np.dot(np.dot(np.ones((1, i)),
                                  f_mM[:i, n - ntc - 1:n - j - 2:-1]),
                           tempm)
            # end
            if (j > ntc - 1) and (i < ntc - 1):
                a = np.dot(np.dot(tempp.T,
                                  f_mM[i:ntc - 1, n - ntc - 1:-1:n - j + 1]),
                           tempm)
                F[i, n - j - 1] = F[i, n - j - 1] + a + b + c
            # end
            if (j == ntc - 1) and (i < ntc - 1):
                F[i, n - ntc] = F[i, n - ntc] + b
                for k in range(ntc):
                    F[i, n - k - 1] = F[i, n - ntc]
                # end
            # end
            if (j > ntc - 1) and (i == ntc - 1):
                F[i, n - j - 1] = F[i, n - j - 1] + c
                for k in range(ntc - 1, n):
                    F[k, n - j - 1] = F[ntc - 1, n - j - 1]
                # end
            # end
        # end
    # end

    # fmax=max(max(F));
    #  contour (u,u,flipud(F),...
    # fmax*[0.005 0.01 0.02 0.05 0.1 0.2 0.4 0.6 0.8])
    #  axis([param(1) param(2) param(1) param(2)])
    #  title('Crest-trough density')
    #  ylabel('crest'), xlabel('trough')
    #  axis('square')
    # if mlver>1, commers, end
    return nt2cmat(F)


def mctp2rfc(fmM, fMm=None):
    """
    Return Rainflow matrix given a Markov chain of turning points

    computes f_rfc = f_mM + F_mct(f_mM).

    Parameters
    ----------
    fmM =  the min2max Markov matrix,
    fMm = the max2min Markov matrix,

    Returns
    -------
    f_rfc = the rainflow matrix,

    Examples
    --------
    >>> fmM = np.array([[ 0.0183,    0.0160,    0.0002,    0.0000,         0],
    ...            [0.0178,    0.5405,    0.0952,         0,         0],
    ...            [0.0002,    0.0813,         0,         0,         0],
    ...            [0.0000,         0,         0,         0,         0],
    ...            [     0,         0,         0,         0,         0]])

    >>> np.allclose(mctp2rfc(fmM),
    ...    [[  2.669981e-02,   7.799700e-03,   4.906077e-07, 0.0, 0.0],
    ...     [  9.599629e-03,   5.485009e-01,   9.539951e-02, 0.0, 0.0],
    ...     [  5.622974e-07,   8.149944e-02,   0.0,          0.0, 0.0],
    ...     [  0.0,   0.0,   0.0,   0.0, 0.0],
    ...     [  0.0,   0.0,   0.0,   0.0, 0.0]], 1.e-7)
    True
    """
    def _get_PMm(AA1, MA, nA):
        PMm = AA1.copy()
        for j in range(nA):
            norm = MA[j]
            if norm != 0:
                PMm[j, :] = PMm[j, :] / norm
        PMm = np.fliplr(PMm)
        return PMm

    if fMm is None:
        fmM = np.atleast_1d(fmM)
        fMm = fmM
    else:
        fmM, fMm = np.atleast_1d(fmM, fMm)
    f_mM, f_Mm = fmM.copy(), fMm.copy()
    N = max(f_mM.shape)
    f_max = np.sum(f_mM, axis=1)
    f_min = np.sum(f_mM, axis=0)
    f_rfc = np.zeros((N, N))
    f_rfc[N - 2, 0] = f_max[N - 2]
    f_rfc[0, N - 2] = f_min[N - 2]
    for k in range(2, N - 1):
        for i in range(1, k):
            AA = f_mM[N - 1 - k:N - 1 - k + i, k - i:k]
            AA1 = f_Mm[N - 1 - k:N - 1 - k + i, k - i:k]
            RAA = f_rfc[N - 1 - k:N - 1 - k + i, k - i:k]
            nA = max(AA.shape)
            MA = f_max[N - 1 - k:N - 1 - k + i]
            mA = f_min[k - i:k]
            SA = AA.sum()
            SRA = RAA.sum()

            DRFC = SA - SRA
            NT = min(mA[0] - sum(RAA[:, 0]), MA[0] - sum(RAA[0, :]))  # check!
            NT = max(NT, 0)  # ??check

            if NT > 1e-6 * max(MA[0], mA[0]):
                NN = MA - np.sum(AA, axis=1)  # T
                e = (mA - np.sum(AA, axis=0))  # T
                e = np.flipud(e)
                PmM = np.rot90(AA.copy())
                for j in range(nA):
                    norm = mA[nA - 1 - j]
                    if norm != 0:
                        PmM[j, :] = PmM[j, :] / norm
                        e[j] = e[j] / norm
                    # end
                # end
                fx = 0.0
                if (max(np.abs(e)) > 1e-6 and
                        max(np.abs(NN)) > 1e-6 * max(MA[0], mA[0])):
                    PMm = _get_PMm(AA1, MA, nA)

                    A = PMm
                    B = PmM

                    if nA == 1:
                        fx = NN * (A / (1 - B * A) * e)
                    else:
                        rh = np.eye(A.shape[0]) - np.dot(B, A)
                        # least squares
                        fx = np.dot(NN, np.dot(A, np.linalg.solve(rh, e)))
                    # end
                # end
                f_rfc[N - 1 - k, k - i] = fx + DRFC

                #  check2=[ DRFC  fx]
                # pause
            else:
                f_rfc[N - 1 - k, k - i] = 0.0
            # end
        # end
        m0 = max(0, f_min[0] - np.sum(f_rfc[N - k + 1:N, 0]))
        M0 = max(0, f_max[N - 1 - k] - np.sum(f_rfc[N - 1 - k, 1:k]))
        f_rfc[N - 1 - k, 0] = min(m0, M0)
        #  n_loops_left=N-k+1
    # end

    for k in range(1, N):
        M0 = max(0, f_max[0] - np.sum(f_rfc[0, N - k:N]))
        m0 = max(0, f_min[N - 1 - k] - np.sum(f_rfc[1:k + 1, N - 1 - k]))
        f_rfc[0, N - 1 - k] = min(m0, M0)
    # end

#    clf
#    subplot(1,2,2)
#    pcolor(levels(paramm),levels(paramM),flipud(f_mM))
#      title('Markov matrix')
#      ylabel('max'), xlabel('min')
#    axis([paramm(1) paramm(2) paramM(1) paramM(2)])
#    axis('square')
#
#    subplot(1,2,1)
#    pcolor(levels(paramm),levels(paramM),flipud(f_rfc))
#      title('Rainflow matrix')
#      ylabel('max'), xlabel('rfc-min')
#    axis([paramm(1) paramm(2) paramM(1) paramM(2)])
#    axis('square')

    return f_rfc


def mc2rfc(f_xy, paramv=None, paramu=None):
    """
    MC2RFC  Calculates a rainflow matrix given a Markov chain with kernel f_xy;
           f_rfc = f_xy + F_mc(f_xy).

      CALL: f_rfc = mc2rfc(f_xy);

      where

            f_rfc = the rainflow matrix,
            f_xy =  the frequency matrix of Markov chain (X0,X1)
                    but only the triangular part for X1>X0.

      Further optional input arguments;

      CALL:  f_rfc = mc2rfc(f_xy,paramx,paramy);

           paramx = the parameter matrix defining discretization of x-values,
           paramy = the parameter matrix defining discretization of y-values,
    """
    N = len(f_xy)
    if paramv is None:
        paramv = (-1, 1, N)

    if paramu is None:
        paramu = paramv

    dd = np.diag(np.rot90(f_xy))
    Splus = np.sum(f_xy, axis=1).T
    Sminus = np.fliplr(sum(f_xy))
    Max_rfc = np.zeros((N, 1))
    Min_rfc = np.zeros((N, 1))
    norm = np.zeros((N, 1))
    for i in range(N):
        Spm = Sminus[i] + Splus[i] - dd[i]
        if Spm > 0:
            Max_rfc[i] = (Splus[i] - dd[i]) * \
                (Splus[i] - dd[i]) / (1 - dd[i] / Spm) / Spm
            Min_rfc[i] = (Sminus[i] - dd[i]) * \
                (Sminus[i] - dd[i]) / (1 - dd[i] / Spm) / Spm
            norm[i] = Spm
        # end if
    # end for

    # cross=zeros(N,1)
    # for i=2:N
    #   cross(N-i+1)=cross(N-i+2)+Sminus(N-i+2)-Splus(N-i+2)
    # end

    f_rfc = np.zeros((N, N))
    f_rfc[N - 1, 1] = Max_rfc[N - 1]
    f_rfc[1, N - 1] = Min_rfc[2]

    for k in range(2, N - 1):
        for i in range(1, k - 1):

            #       AAe= f_xy(1:N-k,1:k-i)
            #       SAAe=sum(sum(AAe))
            AA = f_xy[N - k:N - k + i, :][:, k - i:k]
            # RAA = f_rfc(N-k+1:N-k+i,k-i+1:k)
            RAA = f_rfc[N - k:N - k + i, :][:, k - i:k]
            nA = len(AA)
            # MA = Splus(N-k+1:N-k+i)
            MA = Splus[N - k:N - k + i]
            mA = Sminus[N - k:N - k + i]
            normA = norm[N - k:N - k + i]
            MA_rfc = Max_rfc[N - k:N - k + i]
            # mA_rfc=Min_rfc(k-i+1:k)
            SA = np.sum(AA)
            SRA = np.sum(RAA)
            SMA_rfc = sum(MA_rfc)
            SMA = np.sum(MA)
            DRFC = SA - SMA - SRA + SMA_rfc

            NT = MA_rfc[0] - np.sum(RAA[0, :])

            #       if k==35
            #          check=[MA_rfc(1) sum(RAA(1,:))]
            #          pause
            #       end

            NT = np.maximum(NT, 0)

            if NT > 1e-6 * MA_rfc[0]:

                NN = MA - np.sum(AA, axis=1).T
                e = (np.fliplr(mA) - np.sum(AA)).T
                e = np.flipud(e)
                AA = AA + np.flipud(np.rot90(AA, -1))
                AA = np.rot90(AA)
                AA = AA - 0.5 * np.diag(np.diag(AA))

                for j in range(nA):
                    if normA[j] != 0:
                        AA[j, :] = AA[j, :] / normA[j]
                        e[j] = e[j] / normA[j]
                    # end if
                # end for
                fx = 0.

                if np.max(
                        np.abs(e)) > 1e-7 and np.max(np.abs(NN)) > 1e-7 * MA_rfc[0]:
                    I = np.eye(np.shape(AA))

                    if nA == 1:
                        fx = NN / (1 - AA) * e
                    else:
                        # TODO CHECK this
                        fx = NN * np.linalg.solve((I - AA), e)[0]  # (I-AA)\e
                    # end
                # end

                f_rfc[N - k, k - i] = DRFC + fx
            # end
        # end
        m0 = np.maximum(0, Min_rfc[N] - sum(f_rfc[N - k + 1:N, 0]))
        M0 = np.maximum(0, Max_rfc[N - k] - sum(f_rfc[N - k, 1:k]))
        f_rfc[N - k, 0] = min(m0, M0)
        #  n_loops_left=N-k+1
    # end for

    for k in range(1, N):
        M0 = max(0, Max_rfc[0] - sum(f_rfc[0, N - k + 1:N]))
        m0 = max(0, Min_rfc[k] - sum(f_rfc[1:k, N - k]))
        f_rfc[0, N - k] = min(m0, M0)
    # end for
    f_rfc = f_rfc + np.rot90(np.diag(dd), -1)

#     clf
#     subplot(1,2,2)
#     pcolor(levels(paramv),levels(paramu),flipud(f_xy+flipud(rot90(f_xy,-1))))
#       axis([paramv(1), paramv(2), paramu(1), paramu(2)])
#       title('MC-kernel  f(x,y)')
#       ylabel('y'), xlabel('x')
#     axis('square')
#
#     subplot(1,2,1)
#     pcolor(levels(paramv),levels(paramu),flipud(f_rfc))
#       axis([paramv(1), paramv(2), paramu(1), paramu(2)])
#       title('Rainflow matrix')
#       ylabel('max'), xlabel('rfc-min')
#     axis('square')

    return f_rfc


def mktestmat(param=(-1, 1, 32), x0=None, s=None, lam=1, numsubzero=0):
    """

    MKTESTMAT   Makes test matrices for min-max (and max-min) matrices.

    Parameters
    ----------
    param  = Parameter vector, [a b n], defines discretization.
    x0     = Center of ellipse. [min Max]             [1x2]
    s      = Standard deviation. (0<s<infinity)       [1x1]
    lam    = Orientation of ellipse. (0<lam<infinity) [1x1]
             lam=1 gives circles.
    numsubzero = Number of subdiagonals that are set to zero
                (-Inf: no subdiagonals that are set to zero)
                (Optional, Default = 0, only the diagonal is zero)

    Returns
    -------
    F      = min-max matrix.                          [nxn]
    Fh     = max-min matrix.                          [nxn]

     Makes a Normal kernel (Iso-lines are ellipses).
     Each element in F =(F(i,j)) is
       F[i,j] = exp(-1/2*(x-x0)*inv(S)*(x-x0));
     where
       S = 1/2*s**2*[lam**2+1 lam**2-1; lam**2-1 lam**2+1]

     The matrix Fh is obtained by assuming a time-reversible process.
     These matrices can be used for testing.

    Examples
    --------
    >>> import numpy as np
    >>> param = [-1, 1, 32]
    >>> F, Fh = mktestmat(param, x0=[-0.2, 0.2], s=0.25, lam=0.5)
    >>> np.allclose(F[0,:4],
    ...    [  8.45677332e-29,   5.94193456e-27,   3.53462989e-25, 1.78013497e-23])
    True

    >>> u = np.linspace(*param)

    cmatplot(u,u,F,3)
    axis('square')
    >>> F, Fh = mktestmat(param, x0=[-0.2, 0.2], s=0.25, lam=0.5, numsubzero=-np.inf)
    >>> np.allclose(F[0,:4],
    ...            [  8.45677332e-29,   5.94193456e-27,   3.53462989e-25, 1.78013497e-23])
    True

       cmatplot(u,u,F,3); axis('square');

       close all;
    """
#      History:
#      Revised by PJ  23-Nov-1999
#        updated for WAFO
#      Created by PJ (Paer Johannesson) 1997
#        Copyright (c) 1997 by Paer Johannesson
#        Toolbox: Rainflow Cycles for Switching Processes V.1.0, 2-Oct-1997

    if x0 is None:
        x0 = np.ones((2,)) * (param[1] + param[0]) / 2
    if s is None:
        s = (param[1] - param[0]) / 4

    if np.isinf(numsubzero):
        numsubzero = -(param[2] + 1)

    u = np.linspace(*param)
    n = param[2]

    # F - min-Max matrix

    F = np.zeros((n, n))
    S = 1 / 2 * s**2 * np.array([[lam**2 + 1, lam**2 - 1],
                                 [lam**2 - 1, lam**2 + 1]])
    Sm1 = np.linalg.pinv(S)
    for i in range(min(n - 1 - numsubzero, n)):
        for j in range(max(i + numsubzero, 0), n):
            dx = np.array([u[i], u[j]]) - x0
            F[i, j] = np.exp(-1 / 2 * np.dot(np.dot(dx, Sm1), dx))

    Fh = F.T  # Time-reversible Max-min matrix
    return F, Fh


def cmatplot(cmat, ux=None, uy=None, method=1, clevels=None):
    """
    CMATPLOT Plots a cycle matrix, e.g. a rainflow matrix.

    CALL:  cmatplot(F)
        cmatplot(F,method)
        cmatplot(ux,uy,F)
        cmatplot(ux,uy,F,method)

    F      = Cycle matrix (e.g. rainflow matrix) [nxm]
    method = 1: mesh-plot (default)
          2: surf-plot
          3: pcolor-plot    [axis('square')]
          4: contour-plot   [axis('square')]
          5: TechMath-plot  [axis('square')]
          11: From-To, mesh-plot
          12: From-To, surf-plot
          13: From-To, pcolor-plot   [axis('square')]
          14: From-To, contour-plot  [axis('square')]
          15: From-To, TechMath-plot [axis('square')]
    ux     = x-axis (default: 1:m)
    uy     = y-axis (default: 1:n)

    Examples:
    param = [-1 1 64]; u=levels(param);
    F = mktestmat(param,[-0.2 0.2],0.25,1/2);
    cmatplot(F,method=1);
    cmatplot(u,u,F,method=2); colorbar;
    cmatplot(u,u,F,method=3); colorbar;
    cmatplot(u,u,F,method=4);

    close all;

    See also
    --------
    cocc, plotcc
    """

    F = cmat
    shape = np.shape(F)
    if ux is None:
        ux = np.arange(shape[1])  # Antalet kolumner

    if uy is None:
        uy = np.arange(shape[0])  # Antalet rader

    if clevels is None:
        Fmax = np.max(F)
        if method in [5, 15]:
            clevels = Fmax * np.r_[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        else:  # 4, 14
            clevels = Fmax * np.r_[0.005,
                                   0.01,
                                   0.02,
                                   0.05,
                                   0.1,
                                   0.2,
                                   0.4,
                                   0.6,
                                   0.8]

    # Make sure ux and uy are row vectors
    ux = ux.ravel()
    uy = uy.ravel()

    n = len(F)

    from matplotlib import pyplot as plt
    if method == 1:      # mesh
        F = np.flipud(F.T)  # Vrid cykelmatrisen for att plotta rett
        plt.mesh(ux, np.fliplr(uy), F)
        plt.xlabel('min')
        plt.ylabel('Max')
        # view(-37.5-90,30);
        # v = axis;
        # plt.axis([min(ux) max(ux) min(uy) max(uy) v[5:6]]);
    elif method == 2:  # surf
        F = np.flipud(F.T)  # Vrid cykelmatrisen for att plotta rett
        plt.surf(ux, np.fliplr(uy), F)
        plt.xlabel('min')
        plt.ylabel('Max')
        # view(-37.5-90,30);
        # v = axis;
        # plt.axis([min(ux) max(ux) min(uy) max(uy) v(5:6)]);
#     elseif method == 3  # pcolor
#       F = flipud(F');
#       F1 = [F zeros(length(uy),1); zeros(1,length(ux)+1)];
#       F2 = F1; F2(F2==0)=NaN;
#       F1 = F2;
#       dx=ux(2)-ux(1); dy=uy(2)-uy(1);
#       ux1 = [ux ux(length(ux))+dx] - dx/2;
#       uy1 = [uy uy(length(uy))+dy] - dy/2;
#       pcolor(ux1,fliplr(uy1),F1);
#       xlabel('min');
#       ylabel('Max');
#       v = axis; axis([min(ux1) max(ux1) min(uy1) max(uy1)]);
#       axis('square')
#     elseif method == 4  # contour
#       F = flipud(F');
#       if isempty(clevels)
#         Fmax=max(max(F));
#         clevels=Fmax*[0.005 0.01 0.02 0.05 0.1 0.2 0.4 0.6 0.8];
#       end
#       contour(ux,fliplr(uy),F,clevels);
#       xlabel('min');
#       ylabel('Max');
#       v = axis; axis([min(ux) max(ux) min(uy) max(uy)]);
#       axis('square');
#
#     #  Cstr=num2str(clevels(1),4);
#     #  for i=2:length(clevels)
#     #    Cstr=[Cstr ',' num2str(clevels(i),4)];
#     #  end
#     #  title(['ISO-lines: ' Cstr])
#
#     if 1==2
#       clevels=sort(clevels);
#       n_clevels=length(clevels);
#       if n_clevels>12
#         disp('   Only the first 12 levels will be listed in table.')
#         n_clevels=12;
#       end
#
#       textstart_x=0.65;
#       textstart_y=0.45;
#       delta_y=1/33;
#       h=figtext(textstart_x,textstart_y,'Level curves at:','normalized');
#       set(h,'FontWeight','Bold')
#
#       textstart_y=textstart_y-delta_y;
#
#       for i=1:n_clevels
#         textstart_y=textstart_y-delta_y;
#         figtext(textstart_x,textstart_y,num2str(clevels(i)),'normalized')
#       end
#     end # 1==2
#
#     elseif method == 5 |  method == 15 # TechMath-typ
#
#       if isempty(clevels)
#         Fmax=max(max(F));
#         clevels=Fmax*[0.001 0.005 0.01 0.05 0.1 0.5 1.0];
#       end
#       v=clevels;
#     #  axis('ij');
#       sym = '...x+***';
#       sz = [6 20 24 8 8 8 12 16];
#
#     #  plot(-1,-1,sym(1),'markersize',1),hold on
#       for i = 1:length(v)
#         plot(-1,-1,sym(i),'markersize',sz(i));hold on;
#       end
#
#       for i = 1:length(v)-1
#         Ind = (F>v(i)) & (F<=v(i+1));
#         [I,J] = find(Ind);
#     #    axis('ij');
#         plot(I,J,sym(i),'markersize',sz(i));hold on;
#       end
#       plot([1 n],[1 n],'--'); grid;
#       hold off;
#
#       axis([0.5 n+0.5 0.5 n+0.5]);
#
#       #legendText = sprintf('#6g < f <= %6g\n',[v(1:nv-1); v(2:nv)])
#       #legendText = sprintf('<= %g\n',v(2:end))
#
#       legendText=num2str(v(1:end)');
#
#       legend(legendText,-1);
#
#       title('From-To plot');
#       xlabel('To / Standing');
#       ylabel('From / Hanging');
#
#       if method == 15
#         axis('ij');
#       end
#
#     elseif method == 11  # mesh
#
#       mesh(ux,uy,F);
#       axis('ij');
#       xlabel('To');
#       ylabel('From');
#       view(-37.5-90,30);
#       v = axis; axis([min(ux) max(ux) min(uy) max(uy) v(5:6)]);
#
#     elseif method == 12  # surf
#
#       surf(ux,uy,F);
#       axis('ij');
#       xlabel('To');
#       ylabel('From');
#       view(-37.5-90,30);
#       v = axis; axis([min(ux) max(ux) min(uy) max(uy) v(5:6)]);
#
#     elseif method == 13  # From-To-Matrix - pcolor
#
#       F1 = [F zeros(length(uy),1); zeros(1,length(ux)+1)];
#       F2 = F1; F2(F2==0)=NaN;
#       F1 = F2;
#       dx=ux(2)-ux(1); dy=uy(2)-uy(1);
#       ux1 = [ux ux(length(ux))+dx] - dx/2;
#       uy1 = [uy uy(length(uy))+dy] - dy/2;
#       axis('ij');
#       pcolor(ux1,uy1,F1)
#       axis('ij');
#       xlabel('To');
#       ylabel('From');
#       v = axis; axis([min(ux1) max(ux1) min(uy1) max(uy1)]);
#       axis('square')
#
#     elseif method == 14  # contour
#       if isempty(clevels)
#         Fmax=max(max(F));
#         clevels=Fmax*[0.005 0.01 0.02 0.05 0.1 0.2 0.4 0.6 0.8];
#       end
#       contour(ux,uy,F,clevels);
#       axis('ij');
#       xlabel('To');
#       ylabel('From');
#       v = axis; axis([min(ux) max(ux) min(uy) max(uy)]);
#       axis('square');

#     #  Cstr=num2str(clevels(1),4);
#     #  for i=2:length(clevels)
#     #    Cstr=[Cstr ',' num2str(clevels(i),4)];
#     #  end
#     #  title(['ISO-lines: ' Cstr])
#
#       if 1==1
#         clevels=sort(clevels);
#         n_clevels=length(clevels);
#         if n_clevels>12
#           disp('   Only the first 12 levels will be listed in table.');
#           n_clevels=12;
#         end
#
#         textstart_x=0.10;
#         textstart_y=0.45;
#         delta_y=1/33;
#         h=figtext(textstart_x,textstart_y,'Level curves at:','normalized');
#         set(h,'FontWeight','Bold')
#
#         textstart_y=textstart_y-delta_y;
#
#         for i=1:n_clevels
#           textstart_y=textstart_y-delta_y;
#           figtext(textstart_x,textstart_y,num2str(clevels(i)),'normalized')
#         end
#       end
#
#     elseif method == 15  # TechMath-typ
#     # See: 'method == 5'
#
#     #  if isempty(clevels)
#     #    Fmax=max(max(F));
#     #    clevels=Fmax*[0.005 0.01 0.05 0.1 0.4 0.8];
#     #  end
#     #  v=clevels;
#     #  axis('ij');
#     #  sym = '...***';
#     #  sz = [8 12 16 8 12 16]
#     #  for i = 1:length(v)-1
#     #    Ind = (F>v(i)) & (F<=v(i+1));
#     #    [I,J] = find(Ind);
#     #    axis('ij');
#     #    plot(J,I,sym(i),'markersize',sz(i)),hold on
#     #  end
#     #  hold off
#     #
#     #  axis([0.5 n+0.5 0.5 n+0.5])
#     #
#     #  %legendText = sprintf('%6g < f <= %6g\n',[v(1:nv-1); v(2:nv)])
#     #  legendText = sprintf('<= %g\n',v(2:nv))
#     #
#     #  %legendText=num2str(v(2:nv)')
#     #
#     #  legend(legendText,-1)
#
#     end
#
#     end % _cmatplot


def arfm2mctp(Frfc):
    """
    ARFM2MCTP  Calculates the markov matrix given an asymmetric rainflow matrix.

    CALL:  F = arfm2mctp(Frfc);

    F      = Markov matrix (from-to-matrix)             [n,n]
    Frfc   = Rainflow Matrix                            [n,n]

    Examples
    --------
    param = [-1 1 32];
    u = levels(param);
    F = mktestmat(param,[-0.2 0.2],0.15,2);
    F = F/sum(sum(F));
    Farfc = mctp2arfm({F []});
    F1 = arfm2mctp(Farfc);
    cmatplot(u,u,{F+F' F1},3);
    assert(F1(20,21:25), [0.00209800691364310, 0.00266223402503216,...
                          0.00300934711658560, 0.00303029619424592,...
                          0.00271822008031848], 1e-10);
    assert(sum(sum(abs(F1-(F+F')))), 0, 1e-10)  should be zero

    close all;

    See also
    --------
    rfm2mctp, mctp2arfm, smctp2arfm, cmatplot

    References
    ----------
    P. Johannesson (1999):
    Rainflow Analysis of Switching Markov Loads.
    PhD thesis, Mathematical Statistics, Centre for Mathematical Sciences,
    Lund Institute of Technology.
    """
    # Tested  on Matlab  5.3
    #
    # History:
    # Revised by PJ  09-Apr-2001
    #   updated for WAFO

    # Copyright (c) 1997-1998 by Pear Johannesson
    # Toolbox: Rainflow Cycles for Switching Processes V.1.1, 22-Jan-1998

    # Recursive formulation a'la Igor
    #
    # This program used the formulation where the probabilities
    # of the events are calculated using "elementary" events for
    # the MCTP.
    #
    # Standing
    #    pS = Max*pS1*pS2*pS3;
    #    F_rfc(i,j) = pS;
    # Hanging
    #    pH = Min*pH1*pH2*pH3;
    #    F_rfc(j,i) = pH;
    #
    # The cond. prob. pS1, pS2, pS3, pH1, pH2, pH3 are calculated using
    # the elementary cond. prob. C, E, R, D, E3, Ch, Eh, Rh, Dh, E3h.

    # T(1,:)=clock;

    N = np.sum(Frfc)
    Frfc = Frfc / N

    n = len(Frfc)  # Number of levels

    # T(7,:)=clock;
    # Transition matrices for MC

    Q = np.zeros((n, n))
    Qh = np.zeros((n, n))

    # Transition matrices for time-reversed MC

    Qr = np.zeros((n, n))
    Qrh = np.zeros((n, n))

    # Probability of minimum and of maximun

    MIN = np.sum(np.triu(Frfc).T, axis=0) + np.sum(np.tril(Frfc), axis=0)
    MAX = np.sum(np.triu(Frfc), axis=0) + np.sum(np.tril(Frfc).T, axis=0)

    # Calculate rainflow matrix

    F = np.zeros((n, n))
    EYE = np.eye((n, n))

    # fprintf(1,'Calculating row ');
    for k in range(n - 1):  # k = subdiagonal
        #  fprintf(1,'-%1d',i);

        for i in range(n - k):  # i = minimum

            j = i + k + 1  # maximum;

#         pS = Frfc(i,j);  # Standing cycle
#         pH = Frfc(j,i);  # Hanging cycle
#
#         Min = MIN[i]
#         Max = MAX[j]
#
#     #   fprintf(1,'Min=%f, Max=%f\n',Min,Max);
#
#
#         if j - i == 2:  # Second subdiagonal
#
#             # For Part 1 & 2 of cycle
#
#             #C   = y/Min;
#             c0  = 0;
#             c1  = 1/Min;
#             #Ch  = x/Max;
#             c0h = 0;
#             c1h = 1/Max;
#             d1  = Qr(i,i+1)*(1-Qrh(i+1,i));
#             D   = d1;
#             d1h = Qrh(j,j-1)*(1-Qr(j-1,j));
#             Dh  = d1h;
#             d0  = sum(Qr(i,i+1:j-1));
#             #E   = 1-d0-y/Min;
#             e0  = 1-d0;
#             e1  = -1/Min;
#             d0h = sum(Qrh(j,i+1:j-1));
#             #Eh  = 1-d0h-x/Max;
#             e0h = 1-d0h;
#             e1h = -1/Max;
#             r1  = Qr(i,i+1)*Qrh(i+1,i);
#             R   = r1;
#             r1h = Qrh(j,j-1)*Qr(j-1,j);
#             Rh  = r1h;
#
#             # For Part 3 of cycle
#
#             d3h = sum(Qh(j,i+1:j-1));
#             E3h = 1-d3h;
#             d3  = sum(Q(i,i+1:j-1));
#             E3 = 1-d3;
#
#             # Define coeficients for equation system
#             a0 = -pS+2*pS*Rh-pS*Rh^2+pS*R-2*pS*Rh*R+pS*Rh^2*R;
#             a1 = -E3h*Max*c1h*e0*Rh+E3h*Max*c1h*e0;
#             a3 = -E3h*Max*c1h*e1*Rh+E3h*Max*c1h*Dh*c1+E3h*Max*c1h*e1+pS*c1h*c1-pS*c1h*c1*Rh;
#
#             b0 = -pH+2*pH*R+pH*Rh-2*pH*Rh*R-pH*R^2+pH*Rh*R^2;
#             b2 = -Min*E3*e0h*R*c1+Min*E3*e0h*c1;
#             b3 = Min*E3*e1h*c1+Min*E3*D*c1h*c1-pH*c1h*c1*R-Min*E3*e1h*R*c1+pH*c1h*c1;
#
#             C2 = a3*b2;
#             C1 = (-a0*b3+a1*b2+a3*b0);
#             C0 = a1*b0;
#             # Solve: C2*z^2 + C1*z + C0 = 0
#             z1 = -C1/2/C2 + sqrt((C1/2/C2)^2-C0/C2);
#             z2 = -C1/2/C2 - sqrt((C1/2/C2)^2-C0/C2);
#
#             # Solution 1
#             x1 = -(b0+b2*z1)/(b3*z1);
#             y1 = z1;
#             # Solution 2
#             x2 = -(b0+b2*z2)/(b3*z2);
#             y2 = z2;
#
#             x = x2;
#             y = y2;
#
#             # fprintf(1,'2nd: i=%d, j=%d: x1=%f, y1=%f, x2=%f, y2=%f\n',i,j,x1,y1,x2,y2);
#
#             # Test Standing cycle: assume x=y
#
#             C0 = a0; C1 = a1; C2 = a3;
#             z1S = -C1/2/C2 + sqrt((C1/2/C2)^2-C0/C2);
#             z2S = -C1/2/C2 - sqrt((C1/2/C2)^2-C0/C2);
#
#             # Test Hanging cycle: assume x=y
#
#             C0 = b0; C1 = b2; C2 = b3;
#             z1H = -C1/2/C2 + sqrt((C1/2/C2)^2-C0/C2);
#             z2H = -C1/2/C2 - sqrt((C1/2/C2)^2-C0/C2);
#
#             # fprintf(1,'2nd: i=%d, j=%d: z1S=%f,: z2S=%f, z1H=%f, z2H=%f\n',i,j,z1S,z2S,z1H,z2H)
#
#         else
#
#           Eye = EYE(1:j-i-2,1:j-i-2);
#
#           # For Part 1 & 2 of cycle
#
#           I  = i+1:j-2;
#           J  = i+2:j-1;
#           A  = Qr(I,J);
#           Ah = Qrh(J,I);
#           a  = Qr(i,J);
#           ah = Qrh(j,I);
#           b  = Qr(I,j);
#           bh = Qrh(J,i);
#
#           e  = 1 - sum(Qr(I,i+2:j),2);
#           eh = 1 - sum(Qrh(J,i:j-2),2);
#
#           Inv = inv(Eye-A*Ah);
#           #C   = y/Min + a*Ah*Inv*b;
#           c0  = a*Ah*Inv*b;
#           c1  = 1/Min;
#           #Ch  = x/Max + ah*Inv*A*bh;
#           c0h = ah*Inv*A*bh;
#           c1h = 1/Max;
#           d1  = Qr(i,i+1)*(1-Qrh(i+1,i));
#           D   = d1+a*eh+a*Ah*Inv*A*eh;
#           d1h = Qrh(j,j-1)*(1-Qr(j-1,j));
#           Dh  = d1h+ah*Inv*e;
#           d0  = sum(Qr(i,i+1:j-1));
#           #E   = 1-d0-y/Min+a*Ah*Inv*e;
#           e0  = 1-d0+a*Ah*Inv*e;
#           e1  = -1/Min;
#           d0h = sum(Qrh(j,i+1:j-1));
#           #Eh  = 1-d0h-x/Max+ah*Inv*A*eh;
#           e0h = 1-d0h+ah*Inv*A*eh;
#           e1h = -1/Max;
#           r1  = Qr(i,i+1)*Qrh(i+1,i);
#           R   = r1+a*bh+a*Ah*Inv*A*bh;
#           r1h = Qrh(j,j-1)*Qr(j-1,j);
#           Rh  = r1h+ah*Inv*b;
#
#           # For Part 3 of cycle
#
#           A3  = Q(I,J);
#           A3h = Qh(J,I);
#           Inv3 = inv(Eye-A3*A3h);
#
#           # For Standing cycle
#           d3h = sum(Qh(j,i+1:j-1));
#           c3h = Qh(j,I);
#           e3h = 1 - sum(Qh(J,i+1:j-2),2);
#           E3h = 1-d3h + c3h*Inv3*A3*e3h;
#
#           # For Hanging cycle
#           d3  = sum(Q(i,i+1:j-1));
#           c3  = Q(i,J);
#           e3  = 1 - sum(Q(I,i+2:j-1),2);
#           E3  = 1-d3 + c3*A3h*Inv3*e3;
#
#         end
#
#         if j-i == 1  # First subdiagonal
#
#             if i == 1
#             x = Max;
#             y = Max;
#           elseif j == n
#             x = Min;
#             y = Min;
#           else
#             if pS == 0
#               x = 0;
#               y = pH;
#             elseif pH == 0
#               x = pS;
#               y = 0;
#             else
#               x = Min*pS/(Min-pH);
#               y = Max*pH/(Max-pS);
#             end
#           end
#
#         elseif j-i >= 2
#           if i == 1
#             x = Max*(1-sum(Qh(j,2:j-1)));
#             y = Max*(1-sum(Qrh(j,2:j-1)));
#           elseif j == n
#             x = Min*(1-sum(Q(i,i+1:n-1)));
#             y = Min*(1-sum(Qr(i,i+1:n-1)));
#           else
#             if pS == 0
#               x = 0;
#               y = pH;
#             elseif pH == 0
#               x = pS;
#               y = 0;
#             else
#               # Define coeficients for equation system
#               a0 = (pS*c0h*c0+pS*Rh^2*R-2*pS*Rh*R-E3h*Max*c0h*e0*Rh+E3h*Max*c0h*e0+2*pS*Rh
#                     +pS*R-pS*c0h*c0*Rh-pS-pS*Rh^2+E3h*Max*c0h*Dh*c0)
#               a1 = pS*c1h*c0+E3h*Max*c1h*Dh*c0-E3h*Max*c1h*e0*Rh-pS*c1h*c0*Rh+E3h*Max*c1h*e0;
#               a2 = pS*c0h*c1+E3h*Max*c0h*e1-pS*c0h*c1*Rh+E3h*Max*c0h*Dh*c1-E3h*Max*c0h*e1*Rh;
#               a3 = -E3h*Max*c1h*e1*Rh+E3h*Max*c1h*Dh*c1+E3h*Max*c1h*e1+pS*c1h*c1-pS*c1h*c1*Rh;
#
#               b0 = (pH*c0h*c0+pH*Rh*R^2-pH+pH*Rh-2*pH*Rh*R-pH*c0h*c0*R+Min*E3*e0h*c0
#                    -Min*E3*e0h*R*c0+Min*E3*D*c0h*c0+2*pH*R-pH*R^2)
#               b1 = Min*E3*D*c1h*c0+Min*E3*e1h*c0+pH*c1h*c0-Min*E3*e1h*R*c0-pH*c1h*c0*R;
#               b2 = -pH*c0h*c1*R-Min*E3*e0h*R*c1+Min*E3*D*c0h*c1+Min*E3*e0h*c1+pH*c0h*c1;
#               b3 = Min*E3*e1h*c1+Min*E3*D*c1h*c1-pH*c1h*c1*R-Min*E3*e1h*R*c1+pH*c1h*c1;
#
#               C2 = a2*b3-a3*b2;
#               C1 = a0*b3-a1*b2+a2*b1-a3*b0;
#               C0 = a0*b1-a1*b0;
#     #fprintf(1,'i=%d, j=%d, C0/C2=%f,C1/C2=%f,C2=%f\n',i,j,C0/C2,C1/C2,C2);
#               # Solve: C2*z^2 + C1*z + C0 = 0
#               z1 = -C1/2/C2 + sqrt((C1/2/C2)^2-C0/C2);
#               z2 = -C1/2/C2 - sqrt((C1/2/C2)^2-C0/C2);
#
#               # Solution 1
#               x1 = -(b0+b2*z1)/(b1+b3*z1);
#               y1 = z1;
#               # Solution 2
#               x2 = -(b0+b2*z2)/(b1+b3*z2);
#               y2 = z2;
#
#               x = x2;
#               y = y2;
#
#     #          fprintf(1,'End: i=%d, j=%d: x1=%f, y1=%f, x2=%f, y2=%f\n',i,j,x1,y1,x2,y2);
#             end
#           end
#         end
#
#     #    fprintf(1,'i=%d, j=%d: x=%f, y=%f\n',i,j,x,y);
#
#         # min-max
#         F(i,j) = x;
#
#         # max-min
#         F(j,i) = y;
#
#         # Fill the transitions matrices
#         Q(i,j)   = x/Min;
#         Qh(j,i)  = y/Max;
#         Qr(i,j)  = y/Min;
#         Qrh(j,i) = x/Max;
#
#       end
#     end
#     #fprintf(1,'\n');
#
#
#     T(8,:)=clock;
#
#     return F,T


if __name__ == "__main__":
    from wafo.testing import test_docstrings
    test_docstrings(__file__)
