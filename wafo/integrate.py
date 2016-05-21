from __future__ import absolute_import, division
import warnings
import numpy as np
from numpy import pi, sqrt, ones, zeros  # @UnresolvedImport
from scipy import integrate as intg
import scipy.special.orthogonal as ort
from scipy import special as sp
from .plotbackend import plotbackend as plt
from scipy.integrate import simps, trapz
from .demos import humps
from .misc import dea3
from .dctpack import dct
# from pychebfun import Chebfun

_EPS = np.finfo(float).eps
_POINTS_AND_WEIGHTS = {}

__all__ = ['dea3', 'clencurt', 'romberg',
           'h_roots', 'j_roots', 'la_roots', 'p_roots', 'qrule',
           'gaussq', 'richardson', 'quadgr', 'qdemo']


def clencurt(fun, a, b, n0=5, trace=False, args=()):
    '''
    Numerical evaluation of an integral, Clenshaw-Curtis method.

    Parameters
    ----------
    fun : callable
    a, b : array-like
        Lower and upper integration limit, respectively.
    n : integer
        defines number of evaluation points (default 5)

    Returns
    -------
    Q     = evaluated integral
    tol   = Estimate of the approximation error

    Notes
    -----
    CLENCURT approximates the integral of f(x) from a to b
    using an 2*n+1 points Clenshaw-Curtis formula.
    The error estimate is usually a conservative estimate of the
    approximation error.
    The integral is exact for polynomials of degree 2*n or less.

    Example
    -------
    >>> import numpy as np
    >>> val,err = clencurt(np.exp,0,2)
    >>> abs(val-np.expm1(2))< err, err<1e-10
    (array([ True], dtype=bool), array([ True], dtype=bool))


    See also
    --------
    simpson,
    gaussq

    References
    ----------
    [1] Goodwin, E.T. (1961),
    "Modern Computing Methods",
    2nd edition, New yourk: Philosophical Library, pp. 78--79

    [2] Clenshaw, C.W. and Curtis, A.R. (1960),
    Numerische Matematik, Vol. 2, pp. 197--205
    '''

    # make sure n is even
    n = 2 * n0
    a, b = np.atleast_1d(a, b)
    a_shape = a.shape
    af = a.ravel()
    bf = b.ravel()

    Na = np.prod(a_shape)

    s = np.r_[0:n + 1]
    s2 = np.r_[0:n + 1:2]
    s2.shape = (-1, 1)
    x1 = np.cos(np.pi * s / n)
    x1.shape = (-1, 1)
    x = x1 * (bf - af) / 2. + (bf + af) / 2

    if hasattr(fun, '__call__'):
        f = fun(x)
    else:
        x0 = np.flipud(fun[:, 0])
        n = len(x0) - 1
        if abs(x - x0) > 1e-8:
            raise ValueError(
                'Input vector x must equal cos(pi*s/n)*(b-a)/2+(b+a)/2')

        f = np.flipud(fun[:, 1::])

    if trace:
        plt.plot(x, f, '+')

    # using a Gauss-Lobatto variant, i.e., first and last
    # term f(a) and f(b) is multiplied with 0.5
    f[0, :] = f[0, :] / 2
    f[n, :] = f[n, :] / 2

# % x = cos(pi*0:n/n)
# % f = f(x)
# %
# %               N+1
# %  c(k) = (2/N) sum  f''(n)*cos(pi*(2*k-2)*(n-1)/N), 1 <= k <= N/2+1.
# %               n=1
    fft = np.fft.fft
    tmp = np.real(fft(f[:n, :], axis=0))
    c = 2 / n * (tmp[0:n / 2 + 1, :] + np.cos(np.pi * s2) * f[n, :])
    c[0, :] = c[0, :] / 2
    c[n / 2, :] = c[n / 2, :] / 2

# % alternative call
    # c2 = dct(f)

    c = c[0:n / 2 + 1, :] / ((s2 - 1) * (s2 + 1))
    Q = (af - bf) * np.sum(c, axis=0)
    # Q = (a-b).*sum( c(1:n/2+1,:)./repmat((s2-1).*(s2+1),1,Na))

    abserr = (bf - af) * np.abs(c[n / 2, :])

    if Na > 1:
        abserr = np.reshape(abserr, a_shape)
        Q = np.reshape(Q, a_shape)
    return Q, abserr


def romberg(fun, a, b, releps=1e-3, abseps=1e-3):
    '''
    Numerical integration with the Romberg method

    Parameters
    ----------
    fun : callable
        function to integrate
    a, b : real scalars
        lower and upper integration limits,  respectively.
    releps, abseps : scalar, optional
        requested relative and absolute error, respectively.

    Returns
    -------
    Q : scalar
        value of integral
    abserr : scalar
        estimated absolute error of integral

    ROMBERG approximates the integral of F(X) from A to B
    using Romberg's method of integration.  The function F
    must return a vector of output values if a vector of input values is given.


    Example
    -------
    >>> import numpy as np
    >>> [q,err] = romberg(np.sqrt,0,10,0,1e-4)
    >>> q,err
    (array([ 21.0818511]), array([  6.6163547e-05]))
    '''
    h = b - a
    hMin = 1.0e-9
    # Max size of extrapolation table
    tableLimit = max(min(np.round(np.log2(h / hMin)), 30), 3)

    rom = zeros((2, tableLimit))

    rom[0, 0] = h * (fun(a) + fun(b)) / 2
    ipower = 1
    fp = ones(tableLimit) * 4

    # Ih1 = 0
    Ih2 = 0.
    Ih4 = rom[0, 0]
    abserr = Ih4
    # epstab = zeros(1,decdigs+7)
    # newflg = 1
    # [res,abserr,epstab,newflg] = dea(newflg,Ih4,abserr,epstab)
    two = 1
    one = 0
    for i in range(1, tableLimit):
        h *= 0.5
        Un5 = np.sum(fun(a + np.arange(1, 2 * ipower, 2) * h)) * h

        #     trapezoidal approximations
        # T2n = 0.5 * (Tn + Un) = 0.5*Tn + Un5
        rom[two, 0] = 0.5 * rom[one, 0] + Un5

        fp[i] = 4 * fp[i - 1]
        #   Richardson extrapolation
        for k in range(i):
            rom[two, k + 1] = (rom[two, k] +
                               (rom[two, k] - rom[one, k]) / (fp[k] - 1))

        Ih1 = Ih2
        Ih2 = Ih4
        Ih4 = rom[two, i]

        if (2 <= i):
            res, abserr = dea3(Ih1, Ih2, Ih4)
            # Ih4 = res
            if (abserr <= max(abseps, releps * abs(res))):
                break

        # rom(1,1:i) = rom(2,1:i)
        two = one
        one = (one + 1) % 2
        ipower *= 2
    return res, abserr


def h_roots(n, method='newton'):
    '''
    Returns the roots (x) of the nth order Hermite polynomial,
    H_n(x), and weights (w) to use in Gaussian Quadrature over
    [-inf,inf] with weighting function exp(-x**2).

    Parameters
    ----------
    n : integer
        number of roots
    method : 'newton' or 'eigenvalue'
        uses Newton Raphson to find zeros of the Hermite polynomial (Fast)
        or eigenvalue of the jacobi matrix (Slow) to obtain the nodes and
        weights, respectively.

    Returns
    -------
    x : ndarray
        roots
    w : ndarray
        weights

    Example
    -------
    >>> import numpy as np
    >>> [x,w] = h_roots(10)
    >>> np.sum(x*w)
    -5.2516042729766621e-19

    See also
    --------
    qrule, gaussq

    References
    ----------
    [1]  Golub, G. H. and Welsch, J. H. (1969)
    'Calculation of Gaussian Quadrature Rules'
    Mathematics of Computation, vol 23,page 221-230,

    [2]. Stroud and Secrest (1966), 'gaussian quadrature formulas',
      prentice-hall, Englewood cliffs, n.j.
    '''

    if not method.startswith('n'):
        return ort.h_roots(n)
    else:
        sqrt = np.sqrt
        max_iter = 10
        releps = 3e-14
        C = [9.084064e-01, 5.214976e-02, 2.579930e-03, 3.986126e-03]
        # PIM4=0.7511255444649425
        PIM4 = np.pi ** (-1. / 4)

        # The roots are symmetric about the origin, so we have to
        # find only half of them.
        m = int(np.fix((n + 1) / 2))

        # Initial approximations to the roots go into z.
        anu = 2.0 * n + 1
        rhs = np.arange(3, 4 * m, 4) * np.pi / anu
        r3 = rhs ** (1. / 3)
        r2 = r3 ** 2
        theta = r3 * (C[0] + r2 * (C[1] + r2 * (C[2] + r2 * C[3])))
        z = sqrt(anu) * np.cos(theta)

        L = zeros((3, len(z)))
        k0 = 0
        kp1 = 1
        for _its in range(max_iter):
            # Newtons method carried out simultaneously on the roots.
            L[k0, :] = 0
            L[kp1, :] = PIM4

            for j in range(1, n + 1):
                # Loop up the recurrence relation to get the Hermite
                # polynomials evaluated at z.
                km1 = k0
                k0 = kp1
                kp1 = np.mod(kp1 + 1, 3)

                L[kp1, :] = (z * sqrt(2 / j) * L[k0, :] -
                             np.sqrt((j - 1) / j) * L[km1, :])

            # L now contains the desired Hermite polynomials.
            # We next compute pp, the derivatives,
            # by the relation (4.5.21) using p2, the polynomials
            # of one lower order.

            pp = sqrt(2 * n) * L[k0, :]
            dz = L[kp1, :] / pp

            z = z - dz  # Newtons formula.

            if not np.any(abs(dz) > releps):
                break
        else:
            warnings.warn('too many iterations!')

        x = np.empty(n)
        w = np.empty(n)
        x[0:m] = z      # Store the root
        x[n - 1:n - m - 1:-1] = -z     # and its symmetric counterpart.
        w[0:m] = 2. / pp ** 2    # Compute the weight
        w[n - 1:n - m - 1:-1] = w[0:m]  # and its symmetric counterpart.
        return x, w


def j_roots(n, alpha, beta, method='newton'):
    '''
    Returns the roots of the nth order Jacobi polynomial, P^(alpha,beta)_n(x)
    and weights (w) to use in Gaussian Quadrature over [-1,1] with weighting
    function (1-x)**alpha (1+x)**beta with alpha,beta > -1.

    Parameters
    ----------
    n : integer
        number of roots
    alpha,beta : scalars
        defining shape of Jacobi polynomial
    method : 'newton' or 'eigenvalue'
        uses Newton Raphson to find zeros of the Hermite polynomial (Fast)
        or eigenvalue of the jacobi matrix (Slow) to obtain the nodes and
        weights, respectively.

    Returns
    -------
    x : ndarray
        roots
    w : ndarray
        weights


    Example
    --------
    >>> [x,w]= j_roots(10,0,0)
    >>> sum(x*w)
    2.7755575615628914e-16

    See also
    --------
    qrule, gaussq


    Reference
    ---------
    [1]  Golub, G. H. and Welsch, J. H. (1969)
     'Calculation of Gaussian Quadrature Rules'
      Mathematics of Computation, vol 23,page 221-230,

    [2]. Stroud and Secrest (1966), 'gaussian quadrature formulas',
          prentice-hall, Englewood cliffs, n.j.
    '''

    if not method.startswith('n'):
        [x, w] = ort.j_roots(n, alpha, beta)
    else:
        max_iter = 10
        releps = 3e-14

        # Initial approximations to the roots go into z.
        alfbet = alpha + beta

        z = np.cos(np.pi * (np.arange(1, n + 1) - 0.25 + 0.5 * alpha) /
                   (n + 0.5 * (alfbet + 1)))

        L = zeros((3, len(z)))
        k0 = 0
        kp1 = 1
        for _its in range(max_iter):
            # Newton's method carried out simultaneously on the roots.
            tmp = 2 + alfbet
            L[k0, :] = 1
            L[kp1, :] = (alpha - beta + tmp * z) / 2

            for j in range(2, n + 1):
                # Loop up the recurrence relation to get the Jacobi
                # polynomials evaluated at z.
                km1 = k0
                k0 = kp1
                kp1 = np.mod(kp1 + 1, 3)

                a = 2. * j * (j + alfbet) * tmp
                tmp = tmp + 2
                c = 2 * (j - 1 + alpha) * (j - 1 + beta) * tmp
                b = (tmp - 1) * (alpha ** 2 - beta ** 2 + tmp * (tmp - 2) * z)

                L[kp1, :] = (b * L[k0, :] - c * L[km1, :]) / a

            # L now contains the desired Jacobi polynomials.
            # We next compute pp, the derivatives with a standard
            # relation involving the polynomials of one lower order.

            pp = ((n * (alpha - beta - tmp * z) * L[kp1, :] +
                  2 * (n + alpha) * (n + beta) * L[k0, :]) /
                  (tmp * (1 - z ** 2)))
            dz = L[kp1, :] / pp
            z = z - dz  # Newton's formula.

            if not any(abs(dz) > releps * abs(z)):
                break
        else:
            warnings.warn('too many iterations in jrule')

        x = z  # %Store the root and the weight.
        f = (sp.gammaln(alpha + n) + sp.gammaln(beta + n) -
             sp.gammaln(n + 1) - sp.gammaln(alpha + beta + n + 1))
        w = (np.exp(f) * tmp * 2 ** alfbet / (pp * L[k0, :]))

    return x, w


def la_roots(n, alpha=0, method='newton'):
    '''
    Returns the roots (x) of the nth order generalized (associated) Laguerre
    polynomial, L^(alpha)_n(x), and weights (w) to use in Gaussian quadrature
    over [0,inf] with weighting function exp(-x) x**alpha with alpha > -1.

    Parameters
    ----------
    n : integer
        number of roots
    method : 'newton' or 'eigenvalue'
        uses Newton Raphson to find zeros of the Laguerre polynomial (Fast)
        or eigenvalue of the jacobi matrix (Slow) to obtain the nodes and
        weights, respectively.

    Returns
    -------
    x : ndarray
        roots
    w : ndarray
        weights

    Example
    -------
    >>> import numpy as np
    >>> [x,w] = h_roots(10)
    >>> np.sum(x*w)
    -5.2516042729766621e-19

    See also
    --------
    qrule, gaussq

    References
    ----------
    [1]  Golub, G. H. and Welsch, J. H. (1969)
    'Calculation of Gaussian Quadrature Rules'
    Mathematics of Computation, vol 23,page 221-230,

    [2]. Stroud and Secrest (1966), 'gaussian quadrature formulas',
      prentice-hall, Englewood cliffs, n.j.
    '''

    if alpha <= -1:
        raise ValueError('alpha must be greater than -1')

    if not method.startswith('n'):
        return ort.la_roots(n, alpha)
    else:
        max_iter = 10
        releps = 3e-14
        C = [9.084064e-01, 5.214976e-02, 2.579930e-03, 3.986126e-03]

        # Initial approximations to the roots go into z.
        anu = 4.0 * n + 2.0 * alpha + 2.0
        rhs = np.arange(4 * n - 1, 2, -4) * np.pi / anu
        r3 = rhs ** (1. / 3)
        r2 = r3 ** 2
        theta = r3 * (C[0] + r2 * (C[1] + r2 * (C[2] + r2 * C[3])))
        z = anu * np.cos(theta) ** 2

        dz = zeros(len(z))
        L = zeros((3, len(z)))
        Lp = zeros((1, len(z)))
        pp = zeros((1, len(z)))
        k0 = 0
        kp1 = 1
        k = slice(len(z))
        for _its in range(max_iter):
            # Newton's method carried out simultaneously on the roots.
            L[k0, k] = 0.
            L[kp1, k] = 1.

            for jj in range(1, n + 1):
                # Loop up the recurrence relation to get the Laguerre
                # polynomials evaluated at z.
                km1 = k0
                k0 = kp1
                kp1 = np.mod(kp1 + 1, 3)

                L[kp1, k] = ((2 * jj - 1 + alpha - z[k]) * L[
                             k0, k] - (jj - 1 + alpha) * L[km1, k]) / jj
            # end
            # L now contains the desired Laguerre polynomials.
            # We next compute pp, the derivatives with a standard
            #  relation involving the polynomials of one lower order.

            Lp[k] = L[k0, k]
            pp[k] = (n * L[kp1, k] - (n + alpha) * Lp[k]) / z[k]

            dz[k] = L[kp1, k] / pp[k]
            z[k] = z[k] - dz[k]  # % Newton?s formula.
            # k = find((abs(dz) > releps.*z))

            if not np.any(abs(dz) > releps):
                break
        else:
            warnings.warn('too many iterations!')

        x = z
        w = -np.exp(sp.gammaln(alpha + n) - sp.gammaln(n)) / (pp * n * Lp)
        return x, w


def p_roots(n, method='newton', a=-1, b=1):
    '''
    Returns the roots (x) of the nth order Legendre polynomial, P_n(x),
    and weights (w) to use in Gaussian Quadrature over [-1,1] with weighting
    function 1.

    Parameters
    ----------
    n : integer
        number of roots
    method : 'newton' or 'eigenvalue'
        uses Newton Raphson to find zeros of the Hermite polynomial (Fast)
        or eigenvalue of the jacobi matrix (Slow) to obtain the nodes and
        weights, respectively.

    Returns
    -------
    x : ndarray
        roots
    w : ndarray
        weights


    Example
    -------
    Integral of exp(x) from a = 0 to b = 3 is: exp(3)-exp(0)=
    >>> import numpy as np
    >>> [x,w] = p_roots(11,a=0,b=3)
    >>> np.sum(np.exp(x)*w)
    19.085536923187668

    See also
    --------
    quadg.


    References
    ----------
    [1] Davis and Rabinowitz (1975) 'Methods of Numerical Integration',
        page 365, Academic Press.

    [2]  Golub, G. H. and Welsch, J. H. (1969)
        'Calculation of Gaussian Quadrature Rules'
        Mathematics of Computation, vol 23,page 221-230,

    [3] Stroud and Secrest (1966), 'gaussian quadrature formulas',
        prentice-hall, Englewood cliffs, n.j.
    '''

    if not method.startswith('n'):
        x, w = ort.p_roots(n)
    else:

        m = int(np.fix((n + 1) / 2))

        mm = 4 * m - 1
        t = (np.pi / (4 * n + 2)) * np.arange(3, mm + 1, 4)
        nn = (1 - (1 - 1 / n) / (8 * n * n))
        xo = nn * np.cos(t)

        if method.endswith('1'):

            # Compute the zeros of the N+1 Legendre Polynomial
            # using the recursion relation and the Newton-Raphson method

            # Legendre-Gauss Polynomials
            L = zeros((3, m))

            # Derivative of LGP
            Lp = zeros((m,))
            dx = zeros((m,))

            releps = 1e-15
            max_iter = 100
            # Compute the zeros of the N+1 Legendre Polynomial
            # using the recursion relation and the Newton-Raphson method

            # Iterate until new points are uniformly within epsilon of old
            # points
            k = slice(m)
            k0 = 0
            kp1 = 1
            for _ix in range(max_iter):
                L[k0, k] = 1
                L[kp1, k] = xo[k]

                for jj in range(2, n + 1):
                    km1 = k0
                    k0 = kp1
                    kp1 = np.mod(k0 + 1, 3)
                    L[kp1, k] = ((2 * jj - 1) * xo[k] * L[
                                 k0, k] - (jj - 1) * L[km1, k]) / jj

                Lp[k] = n * (L[k0, k] - xo[k] * L[kp1, k]) / (1 - xo[k] ** 2)

                dx[k] = L[kp1, k] / Lp[k]
                xo[k] = xo[k] - dx[k]
                k, = np.nonzero((abs(dx) > releps * np.abs(xo)))
                if len(k) == 0:
                    break
            else:
                warnings.warn('Too many iterations!')

            x = -xo
            w = 2. / ((1 - x ** 2) * (Lp ** 2))
        else:
            # Algorithm given by Davis and Rabinowitz in 'Methods
            # of Numerical Integration', page 365, Academic Press, 1975.

            e1 = n * (n + 1)

            for _j in range(2):
                pkm1 = 1
                pk = xo
                for k in range(2, n + 1):
                    t1 = xo * pk
                    pkp1 = t1 - pkm1 - (t1 - pkm1) / k + t1
                    pkm1 = pk
                    pk = pkp1

                den = 1. - xo * xo
                d1 = n * (pkm1 - xo * pk)
                dpn = d1 / den
                d2pn = (2. * xo * dpn - e1 * pk) / den
                d3pn = (4. * xo * d2pn + (2 - e1) * dpn) / den
                d4pn = (6. * xo * d3pn + (6 - e1) * d2pn) / den
                u = pk / dpn
                v = d2pn / dpn
                h = (-u * (1 + (.5 * u) * (v + u *
                                           (v * v - u * d3pn / (3 * dpn)))))
                p = (pk + h * (dpn + (.5 * h) * (d2pn + (h / 3) *
                                                 (d3pn + .25 * h * d4pn))))
                dp = dpn + h * (d2pn + (.5 * h) * (d3pn + h * d4pn / 3))
                h = h - p / dp
                xo = xo + h

            x = -xo - h
            fx = (d1 - h * e1 * (pk + (h / 2) *
                                 (dpn + (h / 3) * (d2pn + (h / 4) *
                                                   (d3pn + (.2 * h) * d4pn)))))
            w = 2 * (1 - x ** 2) / (fx ** 2)

        if (m + m) > n:
            x[m - 1] = 0.0

        if not ((m + m) == n):
            m = m - 1

        x = np.hstack((x, -x[m - 1::-1]))
        w = np.hstack((w, w[m - 1::-1]))

    if (a != -1) | (b != 1):
        # Linear map from[-1,1] to [a,b]
        dh = (b - a) / 2
        x = dh * (x + 1) + a
        w = w * dh

    return x, w


def qrule(n, wfun=1, alpha=0, beta=0):
    '''
    Return nodes and weights for Gaussian quadratures.

    Parameters
    ----------
    n : integer
        number of base points
    wfun : integer
        defining the weight function, p(x). (default wfun = 1)
        1,11,21: p(x) = 1                       a =-1,   b = 1  Gauss-Legendre
        2,12   : p(x) = exp(-x^2)               a =-inf, b = inf Hermite
        3,13   : p(x) = x^alpha*exp(-x)         a = 0,   b = inf Laguerre
        4,14   : p(x) = (x-a)^alpha*(b-x)^beta  a =-1,   b = 1 Jacobi
        5      : p(x) = 1/sqrt((x-a)*(b-x)), a =-1,   b = 1 Chebyshev 1'st kind
        6      : p(x) = sqrt((x-a)*(b-x)),   a =-1,   b = 1 Chebyshev 2'nd kind
        7      : p(x) = sqrt((x-a)/(b-x)),   a = 0,   b = 1
        8      : p(x) = 1/sqrt(b-x),         a = 0,   b = 1
        9      : p(x) = sqrt(b-x),           a = 0,   b = 1

    Returns
    -------
    bp = base points (abscissas)
    wf = weight factors

    The Gaussian Quadrature integrates a (2n-1)th order
    polynomial exactly and the integral is of the form
               b                         n
              Int ( p(x)* F(x) ) dx  =  Sum ( wf_j* F( bp_j ) )
               a                        j=1
    where p(x) is the weight function.
    For Jacobi and Laguerre: alpha, beta >-1 (default alpha=beta=0)

    Examples:
    ---------
    >>> [bp,wf] = qrule(10)
    >>> sum(bp**2*wf)  # integral of x^2 from a = -1 to b = 1
    0.66666666666666641
    >>> [bp,wf] = qrule(10,2)
    >>> sum(bp**2*wf)  # integral of exp(-x.^2)*x.^2 from a = -inf to b = inf
    0.88622692545275772
    >>> [bp,wf] = qrule(10,4,1,2)
    >>> (bp*wf).sum()     # integral of (x+1)*(1-x)^2 from  a = -1 to b = 1
    0.26666666666666755

    See also
    --------
    gaussq

    Reference
    ---------
    Abromowitz and Stegun (1954)
    (for method 5 to 9)
    '''

    if (alpha <= -1) | (beta <= -1):
        raise ValueError('alpha and beta must be greater than -1')

    if wfun == 1:  # Gauss-Legendre
        [bp, wf] = p_roots(n)
    elif wfun == 2:  # Hermite
        [bp, wf] = h_roots(n)
    elif wfun == 3:  # Generalized Laguerre
        [bp, wf] = la_roots(n, alpha)
    elif wfun == 4:  # Gauss-Jacobi
        [bp, wf] = j_roots(n, alpha, beta)
    elif wfun == 5:  # p(x)=1/sqrt((x-a)*(b-x)), a=-1 and b=1 (default)
        jj = np.arange(1, n + 1)
        wf = ones(n) * np.pi / n
        bp = np.cos((2 * jj - 1) * np.pi / (2 * n))

    elif wfun == 6:  # p(x)=sqrt((x-a)*(b-x)),   a=-1 and b=1
        jj = np.arange(1, n + 1)
        xj = jj * np.pi / (n + 1)
        wf = np.pi / (n + 1) * np.sin(xj) ** 2
        bp = np.cos(xj)

    elif wfun == 7:  # p(x)=sqrt((x-a)/(b-x)),   a=0 and b=1
        jj = np.arange(1, n + 1)
        xj = (jj - 0.5) * pi / (2 * n + 1)
        bp = np.cos(xj) ** 2
        wf = 2 * np.pi * bp / (2 * n + 1)

    elif wfun == 8:  # p(x)=1/sqrt(b-x),         a=0 and b=1
        [bp1, wf1] = p_roots(2 * n)
        k, = np.where(0 <= bp1)
        wf = 2 * wf1[k]
        bp = 1 - bp1[k] ** 2

    elif wfun == 9:  # p(x)=np.sqrt(b-x),           a=0 and b=1
        [bp1, wf1] = p_roots(2 * n + 1)
        k, = np.where(0 < bp1)
        wf = 2 * bp1[k] ** 2 * wf1[k]
        bp = 1 - bp1[k] ** 2
    else:
        raise ValueError('unknown weight function')
    return bp, wf


class _Gaussq(object):
    '''
    Numerically evaluate integral, Gauss quadrature.

    Parameters
    ----------
    fun : callable
    a,b : array-like
        lower and upper integration limits, respectively.
    releps, abseps : real scalars, optional
        relative and absolute tolerance, respectively.
        (default releps=abseps=1e-3).
    wfun : scalar integer, optional
        defining the weight function, p(x). (default wfun = 1)
        1 : p(x) = 1                       a =-1,   b = 1   Gauss-Legendre
        2 : p(x) = exp(-x^2)               a =-inf, b = inf Hermite
        3 : p(x) = x^alpha*exp(-x)         a = 0,   b = inf Laguerre
        4 : p(x) = (x-a)^alpha*(b-x)^beta  a =-1,   b = 1 Jacobi
        5 : p(x) = 1/sqrt((x-a)*(b-x)),    a =-1,   b = 1 Chebyshev 1'st kind
        6 : p(x) = sqrt((x-a)*(b-x)),      a =-1,   b = 1 Chebyshev 2'nd kind
        7 : p(x) = sqrt((x-a)/(b-x)),      a = 0,   b = 1
        8 : p(x) = 1/sqrt(b-x),            a = 0,   b = 1
        9 : p(x) = sqrt(b-x),              a = 0,   b = 1
    trace : bool, optional
        If non-zero a point plot of the integrand (default False).
    gn : scalar integer
        number of base points to start the integration with (default 2).
    alpha, beta : real scalars, optional
        Shape parameters of Laguerre or Jacobi weight function
        (alpha,beta>-1) (default alpha=beta=0)

    Returns
    -------
    val : ndarray
        evaluated integral
    err : ndarray
        error estimate, absolute tolerance abs(int-intold)

    Notes
    -----
    GAUSSQ numerically evaluate integral using a Gauss quadrature.
    The Quadrature integrates a (2m-1)th order polynomial exactly and the
    integral is of the form
             b
             Int (p(x)* Fun(x)) dx
              a
    GAUSSQ is vectorized to accept integration limits A, B and
    coefficients P1,P2,...Pn, as matrices or scalars and the
    result is the common size of A, B and P1,P2,...,Pn.

    Examples
    ---------
    integration of x**2        from 0 to 2 and from 1 to 4

    >>> from scitools import numpyutils as npt
    >>> A = [0, 1]; B = [2,4]
    >>> fun = npt.wrap2callable('x**2')
    >>> [val1,err1] = gaussq(fun,A,B)
    >>> val1
    array([  2.6666667,  21.       ])
    >>> err1
    array([  1.7763568e-15,   1.0658141e-14])

    Integration of x^2*exp(-x) from zero to infinity:
    >>> fun2 = npt.wrap2callable('1')
    >>> val2, err2 = gaussq(fun2, 0, npt.inf, wfun=3, alpha=2)
    >>> val3, err3 = gaussq(lambda x: x**2,0, npt.inf, wfun=3, alpha=0)
    >>> val2, err2
    (array([ 2.]), array([  6.6613381e-15]))
    >>> val3, err3
    (array([ 2.]), array([  1.7763568e-15]))

    Integrate humps from 0 to 2 and from 1 to 4
    >>> val4, err4 = gaussq(humps,A,B)

    See also
    --------
    qrule
    gaussq2d
    '''
    def _get_dx(self, wfun, jacob, alpha, beta):
        if wfun in [1, 2, 3, 7]:
            dx = jacob
        elif wfun == 4:
            dx = jacob ** (alpha + beta + 1)
        elif wfun == 5:
            dx = ones((np.size(jacob), 1))
        elif wfun == 6:
            dx = jacob ** 2
        elif wfun == 8:
            dx = sqrt(jacob)
        elif wfun == 9:
            dx = sqrt(jacob) ** 3
        else:
            raise ValueError('unknown option')
        return dx.ravel()

    def _points_and_weights(self, gn, wfun, alpha, beta):
        global _POINTS_AND_WEIGHTS
        name = 'wfun%d_%d_%g_%g' % (wfun, gn, alpha, beta)
        x_and_w = _POINTS_AND_WEIGHTS.setdefault(name, [])
        if len(x_and_w) == 0:
            x_and_w.extend(qrule(gn, wfun, alpha, beta))
        xn, w = x_and_w
        return xn, w

    def _initialize_trace(self, max_iter):
        if self.trace:
            self.x_trace = [0] * max_iter
            self.y_trace = [0] * max_iter

    def _plot_trace(self, x, y):
        if self.trace:
            self.x_trace.append(x.ravel())
            self.y_trace.append(y.ravel())
            hfig = plt.plot(x, y, 'r.')
            plt.setp(hfig, 'color', 'b')

    def _plot_final_trace(self):
        if self.trace > 0:
            plt.clf()
            plt.plot(np.hstack(self.x_trace), np.hstack(self.y_trace), '+')

    def _get_jacob(self, wfun, A, B):
        if wfun in [2, 3]:
            nk = np.size(A)
            jacob = ones((nk, 1))
        else:
            jacob = (B - A) * 0.5
            if wfun in [7, 8, 9]:
                jacob = jacob * 2
        return jacob

    def _warn(self, k, a_shape):
        nk = len(k)
        if nk > 1:
            if (nk == np.prod(a_shape)):
                tmptxt = 'All integrals did not converge'
            else:
                tmptxt = '%d integrals did not converge' % (nk, )
            tmptxt = tmptxt + '--singularities likely!'
        else:
            tmptxt = 'Integral did not converge--singularity likely!'
        warnings.warn(tmptxt)

    def _initialize(self, wfun, a, b, args):
        args = np.broadcast_arrays(*np.atleast_1d(a, b, *args))
        a_shape = args[0].shape
        args = [np.reshape(x, (-1, 1)) for x in args]
        A, B = args[:2]
        args = args[2:]
        if wfun in [2, 3]:
            A = zeros((A.size, 1))
        return A, B, args, a_shape

    def __call__(self, fun, a, b, releps=1e-3, abseps=1e-3, alpha=0, beta=0,
                 wfun=1, trace=False, args=(), max_iter=11):
        self.trace = trace
        gn = 2

        A, B, args, a_shape = self._initialize(wfun, a, b, args)

        jacob = self._get_jacob(wfun, A, B)
        shift = int(wfun in [1, 4, 5, 6])
        dx = self._get_dx(wfun, jacob, alpha, beta)

        self._initialize_trace(max_iter)

        # Break out of the iteration loop for three reasons:
        #  1) the last update is very small (compared to int and to releps)
        #  2) There are more than 11 iterations. This should NEVER happen.
        dtype = np.result_type(fun((A+B)*0.5, *args))
        nk = np.prod(a_shape)  # # of integrals we have to compute
        k = np.arange(nk)
        opts = (nk, dtype)
        val, val_old, abserr = zeros(*opts), ones(*opts), zeros(*opts)
        for i in range(max_iter):
            xn, w = self._points_and_weights(gn, wfun, alpha, beta)
            x = (xn + shift) * jacob[k, :] + A[k, :]

            pi = [xi[k, :] for xi in args]
            y = fun(x, *pi)
            self._plot_trace(x, y)
            val[k] = np.sum(w * y, axis=1) * dx[k]  # do the integration
            if any(np.isnan(val)):
                val[np.isnan(val)] = val_old[np.isnan(val)]
            if 1 < i:
                abserr[k] = abs(val_old[k] - val[k])  # absolute tolerance
                k, = np.where(abserr > np.maximum(abs(releps * val), abseps))
                nk = len(k)  # of integrals we have to compute again
                if nk == 0:
                    break
            val_old[k] = val[k]
            gn *= 2  # double the # of basepoints and weights
        else:
            self._warn(k, a_shape)

        # make sure int is the same size as the integration  limits
        val.shape = a_shape
        abserr.shape = a_shape

        self._plot_final_trace()
        return val, abserr
gaussq = _Gaussq()


def richardson(Q, k):
    # license BSD
    # Richardson extrapolation with parameter estimation
    c = np.real((Q[k - 1] - Q[k - 2]) / (Q[k] - Q[k - 1])) - 1.
    # The lower bound 0.07 admits the singularity x.^-0.9
    c = max(c, 0.07)
    R = Q[k] + (Q[k] - Q[k - 1]) / c
    return R


class _Quadgr(object):

    def __call__(self, fun, a, b, abseps=1e-5, max_iter=17):
        '''
        Gauss-Legendre quadrature with Richardson extrapolation.

        [Q,ERR] = QUADGR(FUN,A,B,TOL) approximates the integral of a function
        FUN from A to B with an absolute error tolerance TOL. FUN is a function
        handle and must accept vector arguments. TOL is 1e-6 by default. Q is
        the integral approximation and ERR is an estimate of the absolute
        error.

        QUADGR uses a 12-point Gauss-Legendre quadrature. The error estimate is
        based on successive interval bisection. Richardson extrapolation
        accelerates the convergence for some integrals, especially integrals
        with endpoint singularities.

        Examples
        --------
        >>> import numpy as np
        >>> Q, err = quadgr(np.log,0,1)
        >>> quadgr(np.exp,0,9999*1j*np.pi)
        (-2.0000000000122662, 2.1933237448479304e-09)

        >>> quadgr(lambda x: np.sqrt(4-x**2),0,2,1e-12)
        (3.1415926535897811, 1.5809575870662229e-13)

        >>> quadgr(lambda x: x**-0.75,0,1)
        (4.0000000000000266, 5.6843418860808015e-14)

        >>> quadgr(lambda x: 1./np.sqrt(1-x**2),-1,1)
        (3.141596056985029, 6.2146261559092864e-06)

        >>> quadgr(lambda x: np.exp(-x**2),-np.inf,np.inf,1e-9) #% sqrt(pi)
        (1.7724538509055152, 1.9722334876348668e-11)

        >>> quadgr(lambda x: np.cos(x)*np.exp(-x),0,np.inf,1e-9)
        (0.50000000000000044, 7.3296813063450372e-11)

        See also
        --------
        QUAD,
        QUADGK
        '''
        # Author: jonas.lundgren@saabgroup.com, 2009. license BSD
        # Order limits (required if infinite limits)
        a = np.asarray(a)
        b = np.asarray(b)
        if a == b:
            Q = b - a
            err = b - a
            return Q, err
        elif np.real(a) > np.real(b):
            reverse = True
            a, b = b, a
        else:
            reverse = False

        # Infinite limits
        if np.isinf(a) | np.isinf(b):
            # Check real limits
            if ~ np.isreal(a) | ~np.isreal(b) | np.isnan(a) | np.isnan(b):
                raise ValueError('Infinite intervals must be real.')

            # Change of variable
            if np.isfinite(a) & np.isinf(b):
                # a to inf
                Q, err = quadgr(lambda t: fun(a + t / (1 - t)) / (1 - t) ** 2,
                                0, 1, abseps)
            elif np.isinf(a) & np.isfinite(b):
                # -inf to b
                Q, err = quadgr(lambda t: fun(b + t / (1 + t)) / (1 + t) ** 2,
                                -1, 0, abseps)
            else:  # -inf to inf
                Q1, err1 = quadgr(lambda t: fun(t / (1 - t)) / (1 - t) ** 2,
                                  0, 1, abseps / 2)
                Q2, err2 = quadgr(lambda t: fun(t / (1 + t)) / (1 + t) ** 2,
                                  -1, 0, abseps / 2)
                Q = Q1 + Q2
                err = err1 + err2

            # Reverse direction
            if reverse:
                Q = -Q
            return Q, err

        # Gauss-Legendre quadrature (12-point)
        xq = np.asarray(
            [0.12523340851146894, 0.36783149899818018, 0.58731795428661748,
             0.76990267419430469, 0.9041172563704748, 0.98156063424671924])
        wq = np.asarray(
            [0.24914704581340288, 0.23349253653835478, 0.20316742672306584,
             0.16007832854334636, 0.10693932599531818, 0.047175336386511842])
        xq = np.hstack((xq, -xq))
        wq = np.hstack((wq, wq))
        nq = len(xq)
        dtype = np.result_type(fun(a), fun(b))

        # Initiate vectors
        Q0 = zeros(max_iter, dtype=dtype)  # Quadrature
        Q1 = zeros(max_iter, dtype=dtype)  # First Richardson extrapolation
        Q2 = zeros(max_iter, dtype=dtype)  # Second Richardson extrapolation

        # One interval
        hh = (b - a) / 2             # Half interval length
        x = (a + b) / 2 + hh * xq      # Nodes
        # Quadrature
        Q0[0] = hh * np.sum(wq * fun(x), axis=0)

        # Successive bisection of intervals
        for k in range(1, max_iter):

            # Interval bisection
            hh = hh / 2
            x = np.hstack([x + a, x + b]) / 2
            # Quadrature
            Q0[k] = hh * np.sum(wq * np.sum(np.reshape(fun(x), (-1, nq)),
                                            axis=0),
                                axis=0)

            # Richardson extrapolation
            if k >= 5:
                Q1[k] = richardson(Q0, k)
                Q2[k] = richardson(Q1, k)
            elif k >= 3:
                Q1[k] = richardson(Q0, k)

            # Estimate absolute error
            if k >= 6:
                Qv = np.hstack((Q0[k], Q1[k], Q2[k]))
                Qw = np.hstack((Q0[k - 1], Q1[k - 1], Q2[k - 1]))
            elif k >= 4:
                Qv = np.hstack((Q0[k], Q1[k]))
                Qw = np.hstack((Q0[k - 1], Q1[k - 1]))
            else:
                Qv = np.atleast_1d(Q0[k])
                Qw = Q0[k - 1]

            errors = np.atleast_1d(abs(Qv - Qw))
            j = errors.argmin()
            err = errors[j]
            Q = Qv[j]
            if k >= 2:  # and not iscomplex:
                _val, err1 = dea3(Q0[k - 2], Q0[k - 1], Q0[k])

            # Convergence
            if (err < abseps) | ~np.isfinite(Q):
                break
        else:
            warnings.warn('Max number of iterations reached without ' +
                          'convergence.')

        if ~ np.isfinite(Q):
            warnings.warn('Integral approximation is Infinite or NaN.')

        # The error estimate should not be zero
        err = err + 2 * np.finfo(Q).eps
        # Reverse direction
        if reverse:
            Q = -Q

        return Q, err

quadgr = _Quadgr()


def boole(y, x):
    a, b = x[0], x[-1]
    n = len(x)
    h = (b - a) / (n - 1)
    return (2 * h / 45) * (7 * (y[0] + y[-1]) + 12 * np.sum(y[2:n - 1:4]) +
                           32 * np.sum(y[1:n - 1:2]) +
                           14 * np.sum(y[4:n - 3:4]))


def qdemo(f, a, b, kmax=9, plot_error=False):
    '''
    Compares different quadrature rules.

    Parameters
    ----------
    f : callable
        function
    a,b : scalars
        lower and upper integration limits

    Details
    -------
    qdemo(f,a,b) computes and compares various approximations to
    the integral of f from a to b.  Three approximations are used,
    the composite trapezoid, Simpson's, and Boole's rules, all with
    equal length subintervals.
    In a case like qdemo(exp,0,3) one can see the expected
    convergence rates for each of the three methods.
    In a case like qdemo(sqrt,0,3), the convergence rate is limited
    not by the method, but by the singularity of the integrand.

    Example
    -------
    >>> import numpy as np
    >>> qdemo(np.exp,0,3)
    true value =  19.08553692
     ftn,                Boole,            Chebychev
    evals       approx        error       approx        error
       3, 19.4008539142, 0.3153169910, 19.5061466023, 0.4206096791
       5, 19.0910191534, 0.0054822302, 19.0910191534, 0.0054822302
       9, 19.0856414320, 0.0001045088, 19.0855374134, 0.0000004902
      17, 19.0855386464, 0.0000017232, 19.0855369232, 0.0000000000
      33, 19.0855369505, 0.0000000273, 19.0855369232, 0.0000000000
      65, 19.0855369236, 0.0000000004, 19.0855369232, 0.0000000000
     129, 19.0855369232, 0.0000000000, 19.0855369232, 0.0000000000
     257, 19.0855369232, 0.0000000000, 19.0855369232, 0.0000000000
     513, 19.0855369232, 0.0000000000, 19.0855369232, 0.0000000000
     ftn,      Clenshaw-Curtis,       Gauss-Legendre
    evals       approx        error       approx        error
       3, 19.5061466023, 0.4206096791, 19.0803304585, 0.0052064647
       5, 19.0834145766, 0.0021223465, 19.0855365951, 0.0000003281
       9, 19.0855369150, 0.0000000082, 19.0855369232, 0.0000000000
      17, 19.0855369232, 0.0000000000, 19.0855369232, 0.0000000000
      33, 19.0855369232, 0.0000000000, 19.0855369232, 0.0000000000
      65, 19.0855369232, 0.0000000000, 19.0855369232, 0.0000000000
     129, 19.0855369232, 0.0000000000, 19.0855369232, 0.0000000000
     257, 19.0855369232, 0.0000000000, 19.0855369232, 0.0000000000
     513, 19.0855369232, 0.0000000000, 19.0855369232, 0.0000000000
     ftn,                Simps,                Trapz
    evals       approx        error       approx        error
       3, 19.5061466023, 0.4206096791, 22.5366862979, 3.4511493747
       5, 19.1169646189, 0.0314276957, 19.9718950387, 0.8863581155
       9, 19.0875991312, 0.0020622080, 19.3086731081, 0.2231361849
      17, 19.0856674267, 0.0001305035, 19.1414188470, 0.0558819239
      33, 19.0855451052, 0.0000081821, 19.0995135407, 0.0139766175
      65, 19.0855374350, 0.0000005118, 19.0890314614, 0.0034945382
     129, 19.0855369552, 0.0000000320, 19.0864105817, 0.0008736585
     257, 19.0855369252, 0.0000000020, 19.0857553393, 0.0002184161
     513, 19.0855369233, 0.0000000001, 19.0855915273, 0.0000546041
    '''
    true_val, _tol = intg.quad(f, a, b)
    print('true value = %12.8f' % (true_val,))
    neval = zeros(kmax, dtype=int)
    vals_dic = {}
    err_dic = {}

    # try various approximations
    methods = [trapz, simps, boole, ]

    for k in range(kmax):
        n = 2 ** (k + 1) + 1
        neval[k] = n
        x = np.linspace(a, b, n)
        y = f(x)
        for method in methods:
            name = method.__name__.title()
            q = method(y, x)
            vals_dic.setdefault(name, []).append(q)
            err_dic.setdefault(name, []).append(abs(q - true_val))

        name = 'Clenshaw-Curtis'
        q, _ec3 = clencurt(f, a, b, (n - 1) / 2)
        vals_dic.setdefault(name, []).append(q[0])
        err_dic.setdefault(name, []).append(abs(q[0] - true_val))

        name = 'Chebychev'
        ck = np.polynomial.chebyshev.chebfit(x, y, deg=min(n-1, 36))
        cki = np.polynomial.chebyshev.chebint(ck)
        q = np.polynomial.chebyshev.chebval(x[-1], cki)
        vals_dic.setdefault(name, []).append(q)
        err_dic.setdefault(name, []).append(abs(q - true_val))
        # ck = chebfit(f,n,a,b)
        # q  = chebval(b,chebint(ck,a,b),a,b)
        # qc2[k] = q; ec2[k] = abs(q - true)

        name = 'Gauss-Legendre'  # quadrature
        q = intg.fixed_quad(f, a, b, n=n)[0]
        # [x, w]=qrule(n,1)
        # x = (b-a)/2*x + (a+b)/2     % Transform base points X.
        # w = (b-a)/2*w               % Adjust weigths.
        # q = sum(feval(f,x)*w)
        vals_dic.setdefault(name, []).append(q)
        err_dic.setdefault(name, []).append(abs(q - true_val))

    # display results
    names = sorted(vals_dic.keys())
    num_cols = 2
    formats = ['%4.0f, ', ] + ['%10.10f, ', ] * num_cols * 2
    formats[-1] = formats[-1].split(',')[0]
    formats_h = ['%4s, ', ] + ['%20s, ', ] * num_cols
    formats_h[-1] = formats_h[-1].split(',')[0]
    headers = ['evals'] + ['%12s %12s' % ('approx', 'error')] * num_cols
    while len(names) > 0:
        print(''.join(fi % t for fi, t in zip(formats_h,
                                              ['ftn'] + names[:num_cols])))
        print(' '.join(headers))

        data = [neval]
        for name in names[:num_cols]:
            data.append(vals_dic[name])
            data.append(err_dic[name])
        data = np.vstack(tuple(data)).T
        for k in range(kmax):
            tmp = data[k].tolist()
            print(''.join(fi % t for fi, t in zip(formats, tmp)))
        if plot_error:
            plt.figure(0)
            for name in names[:num_cols]:
                plt.loglog(neval, err_dic[name], label=name)

        names = names[num_cols:]
    if plot_error:
        plt.xlabel('number of function evaluations')
        plt.ylabel('error')
        plt.legend()
        plt.show('hold')


def main():
    #    val, err = clencurt(np.exp, 0, 2)
    #    valt = np.exp(2) - np.exp(0)
    #    [Q, err] = quadgr(lambda x: x ** 2, 1, 4, 1e-9)
    #    [Q, err] = quadgr(humps, 1, 4, 1e-9)
    #
    #    [x, w] = h_roots(11, 'newton')
    #    sum(w)
    #    [x2, w2] = la_roots(11, 1, 't')
    #
    # from scitools import numpyutils as npu #@UnresolvedImport
    #    fun = npu.wrap2callable('x**2')
    #    p0 = fun(0)
    #    A = [0, 1, 1]; B = [2, 4, 3]
    #    area, err = gaussq(fun, A, B)
    #
    #    fun = npu.wrap2callable('x**2')
    #    [val1, err1] = gaussq(fun, A, B)
    #
    #
    # Integration of x^2*exp(-x) from zero to infinity:
    #    fun2 = npu.wrap2callable('1')
    #    [val2, err2] = gaussq(fun2, 0, np.inf, wfun=3, alpha=2)
    #    [val2, err2] = gaussq(lambda x: x ** 2, 0, np.inf, wfun=3, alpha=0)
    #
    # Integrate humps from 0 to 2 and from 1 to 4
    #    [val3, err3] = gaussq(humps, A, B)
    #
    #    [x, w] = p_roots(11, 'newton', 1, 3)
    #    y = np.sum(x ** 2 * w)

    x = np.linspace(0, np.pi / 2)
    _q0 = np.trapz(humps(x), x)
    [q, err] = romberg(humps, 0, np.pi / 2, 1e-4)
    print(q, err)


def test_docstrings():
    np.set_printoptions(precision=7)
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    test_docstrings()
    # qdemo(np.exp, 0, 3, plot_error=True)
    # plt.show('hold')
    # main()
