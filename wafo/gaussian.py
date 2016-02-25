from __future__ import absolute_import
from numpy import (r_, minimum, maximum, atleast_1d, atleast_2d, mod, ones,
                   floor, random, eye, nonzero, where, repeat, sqrt, exp, inf,
                   diag, zeros, sin, arcsin, nan)
from numpy import triu
from scipy.special import ndtr as cdfnorm, ndtri as invnorm
from scipy.special import erfc
import warnings
import numpy as np

try:
    from . import mvn  # @UnresolvedImport
except ImportError:
    warnings.warn('mvn not found. Check its compilation.')
    mvn = None
try:
    from . import mvnprdmod  # @UnresolvedImport
except ImportError:
    warnings.warn('mvnprdmod not found. Check its compilation.')
    mvnprdmod = None
try:
    from . import rindmod  # @UnresolvedImport
except ImportError:
    warnings.warn('rindmod not found. Check its compilation.')
    rindmod = None


__all__ = ['Rind', 'rindmod', 'mvnprdmod', 'mvn', 'cdflomax', 'prbnormtndpc',
           'prbnormndpc', 'prbnormnd', 'cdfnorm2d', 'prbnorm2d', 'cdfnorm',
           'invnorm', 'test_docstring']


class Rind(object):

    '''
    RIND Computes multivariate normal expectations

    Parameters
    ----------
    S : array-like, shape Ntdc x Ntdc
        Covariance matrix of X=[Xt,Xd,Xc]  (Ntdc = Nt+Nd+Nc)
    m : array-like, size Ntdc
        expectation of X=[Xt,Xd,Xc]
    Blo, Bup : array-like, shape Mb x Nb
        Lower and upper barriers used to compute the integration limits,
        Hlo and Hup, respectively.
    indI : array-like, length Ni
        vector of indices to the different barriers in the indicator function.
        (NB! restriction  indI(1)=-1, indI(NI)=Nt+Nd, Ni = Nb+1)
        (default indI = 0:Nt+Nd)
    xc : values to condition on (default xc = zeros(0,1)), size Nc x Nx
    Nt : size of Xt             (default Nt = Ntdc - Nc)

    Returns
    -------
    val: ndarray, size Nx
        expectation/density as explained below
    err, terr : ndarray, size Nx
        estimated sampling error and estimated truncation error, respectively.
        (err is with 99 confidence level)

    Notes
    -----
    RIND computes multivariate normal expectations, i.e.,
        E[Jacobian*Indicator|Condition ]*f_{Xc}(xc(:,ix))
    where
        "Indicator" = I{ Hlo(i) < X(i) < Hup(i), i = 1:N_t+N_d }
        "Jacobian"  = J(X(Nt+1),...,X(Nt+Nd+Nc)), special case is
        "Jacobian"  = |X(Nt+1)*...*X(Nt+Nd)|=|Xd(1)*Xd(2)..Xd(Nd)|
        "condition" = Xc=xc(:,ix),  ix=1,...,Nx.
        X = [Xt, Xd, Xc], a stochastic vector of Multivariate Gaussian
        variables where Xt,Xd and Xc have the length Nt,Nd and Nc, respectively
        (Recommended limitations Nx,Nt<=100, Nd<=6 and Nc<=10)

    Multivariate probability is computed if Nd = 0.

    If  Mb<Nc+1 then Blo[Mb:Nc+1,:] is assumed to be zero.
    The relation to the integration limits Hlo and Hup are as follows

        Hlo(i) = Blo(1,j)+Blo(2:Mb,j).T*xc(1:Mb-1,ix),
        Hup(i) = Bup(1,j)+Bup(2:Mb,j).T*xc(1:Mb-1,ix),

    where i=indI(j-1)+1:indI(j), j=2:NI, ix=1:Nx

    NOTE : RIND is only using upper triangular part of covariance matrix, S
    (except for method=0).

    Examples
    --------
    Compute Prob{Xi<-1.2} for i=1:5 where Xi are zero mean Gaussian with
            Cov(Xi,Xj) = 0.3 for i~=j and
            Cov(Xi,Xi) = 1   otherwise
    >>> import wafo.gaussian as wg
    >>> n = 5
    >>> Blo =-np.inf; Bup=-1.2; indI=[-1, n-1]  # Barriers
    >>> m = np.zeros(n); rho = 0.3;
    >>> Sc =(np.ones((n,n))-np.eye(n))*rho+np.eye(n)
    >>> rind = wg.Rind()
    >>> E0, err0, terr0 = rind(Sc,m,Blo,Bup,indI)  #  exact prob. 0.001946

    >>> A = np.repeat(Blo,n); B = np.repeat(Bup,n)  # Integration limits
    >>> E1  = rind(np.triu(Sc),m,A,B)   #same as E0

    Compute expectation E( abs(X1*X2*...*X5) )
    >>> xc = np.zeros((0,1))
    >>> infinity = 37
    >>> dev = np.sqrt(np.diag(Sc))  # std
    >>> ind = np.nonzero(indI[1:])[0]
    >>> Bup, Blo = np.atleast_2d(Bup,Blo)
    >>> Bup[0,ind] = np.minimum(Bup[0,ind] , infinity*dev[indI[ind+1]])
    >>> Blo[0,ind] = np.maximum(Blo[0,ind] ,-infinity*dev[indI[ind+1]])
    >>> np.allclose(rind(Sc,m,Blo,Bup,indI, xc, nt=0),
    ...   ([0.05494076], [ 0.00083066], [  1.00000000e-10]), rtol=1e-3)
    True

    Compute expectation E( X1^{+}*X2^{+} ) with random
    correlation coefficient,Cov(X1,X2) = rho2.
    >>> m2  = [0, 0]; rho2 = np.random.rand(1)
    >>> Sc2 = [[1, rho2], [rho2 ,1]]
    >>> Blo2 = 0; Bup2 = np.inf; indI2 = [-1, 1]
    >>> rind2 = wg.Rind(method=1)
    >>> def g2(x):
    ...     return (x*(np.pi/2+np.arcsin(x))+np.sqrt(1-x**2))/(2*np.pi)
    >>> E2 = g2(rho2)   # exact value
    >>> E3 = rind(Sc2,m2,Blo2,Bup2,indI2,nt=0)
    >>> E4 = rind2(Sc2,m2,Blo2,Bup2,indI2,nt=0)
    >>> E5 = rind2(Sc2,m2,Blo2,Bup2,indI2,nt=0,abseps=1e-4)

    See also
    --------
    prbnormnd, prbnormndpc

    References
    ----------
    Podgorski et al. (2000)
    "Exact distributions for apparent waves in irregular seas"
     Ocean Engineering,  Vol 27, no 1, pp979-1016.

    P. A. Brodtkorb (2004),
    Numerical evaluation of multinormal expectations
    In Lund university report series
    and in the Dr.Ing thesis:
    The probability of Occurrence of dangerous Wave Situations at Sea.
    Dr.Ing thesis, Norwegian University of Science and Technolgy, NTNU,
    Trondheim, Norway.

    Per A. Brodtkorb (2006)
    "Evaluating Nearly Singular Multinormal Expectations with Application to
    Wave Distributions",
    Methodology And Computing In Applied Probability, Volume 8, Number 1,
    pp. 65-91(27)
    '''

    def __init__(self, **kwds):
        '''
        Parameters
        ----------
        method : integer, optional
            defining the integration method
            0 Integrate by Gauss-Legendre quadrature  (Podgorski et al. 1999)
            1 Integrate by SADAPT for Ndim<9 and by KRBVRC otherwise
            2 Integrate by SADAPT for Ndim<20 and by KRBVRC otherwise
            3 Integrate by KRBVRC by Genz (1993) (Fast Ndim<101) (default)
            4 Integrate by KROBOV by Genz (1992) (Fast Ndim<101)
            5 Integrate by RCRUDE by Genz (1992) (Slow Ndim<1001)
            6 Integrate by SOBNIED               (Fast Ndim<1041)
            7 Integrate by DKBVRC by Genz (2003) (Fast Ndim<1001)

        xcscale : real scalar, optional
            scales the conditinal probability density, i.e.,
            f_{Xc} = exp(-0.5*Xc*inv(Sxc)*Xc + XcScale) (default XcScale=0)
        abseps, releps : real scalars, optional
            absolute and relative error tolerance.
            (default abseps=0, releps=1e-3)
        coveps : real scalar, optional
            error tolerance in Cholesky factorization (default 1e-13)
        maxpts, minpts : scalar integers, optional
            maximum and minimum number of function values allowed. The
            parameter, maxpts, can be used to limit the time. A sensible
            strategy is to start with MAXPTS = 1000*N, and then increase MAXPTS
            if ERROR is too large.
                (Only for METHOD~=0) (default maxpts=40000, minpts=0)
        seed : scalar integer, optional
            seed to the random generator used in the integrations
                (Only for METHOD~=0)(default floor(rand*1e9))
        nit : scalar integer, optional
            maximum number of Xt variables to integrate. This parameter can be
            used to limit the time. If NIT is less than the rank of the
            covariance matrix, the returned result is a upper bound for the
            true value of the integral.  (default 1000)
        xcutoff : real scalar, optional
            cut off value where the marginal normal distribution is truncated.
            (Depends on requested accuracy. A value between 4 and 5 is
            reasonable.)
        xsplit : real scalar
            parameter controlling performance of quadrature integration:
            if Hup>=xCutOff AND Hlo<-XSPLIT OR
                Hup>=XSPLIT AND Hlo<=-xCutOff then
                    do a different integration to increase speed
                     in rind2 and rindnit. This give slightly different results
            if XSPILT>=xCutOff => do the same integration always
                (Only for METHOD==0)(default XSPLIT = 1.5)
        quadno : scalar integer
            Quadrature formulae number used in integration of Xd variables.
            This number implicitly determines number of nodes
                used.  (Only for METHOD==0)
        speed : scalar integer
            defines accuracy of calculations by choosing different parameters,
            possible values: 1,2...,9 (9 fastest,  default []).
            If not speed is None the parameters, ABSEPS, RELEPS, COVEPS,
                XCUTOFF, MAXPTS and QUADNO will be set according to
                INITOPTIONS.
        nc1c2 : scalar integer, optional
            number of times to use the regression equation to restrict
            integration area. Nc1c2 = 1,2 is recommended. (default 2)
            (note: works only for method >0)
        '''
        self.method = 3
        self.xcscale = 0
        self.abseps = 0
        self.releps = 1e-3,
        self.coveps = 1e-10
        self.maxpts = 40000
        self.minpts = 0
        self.seed = None
        self.nit = 1000,
        self.xcutoff = None
        self.xsplit = 1.5
        self.quadno = 2
        self.speed = None
        self.nc1c2 = 2

        self.__dict__.update(**kwds)
        self.initialize(self.speed)
        self.set_constants()

    def initialize(self, speed=None):
        '''
        Initializes member variables according to speed.

        Parameter
        ---------
        speed : scalar integer
            defining accuracy of calculations.
            Valid numbers:  1,2,...,10
            (1=slowest and most accurate,10=fastest, but less accuracy)


        Member variables initialized according to speed:
        -----------------------------------------------
        speed : Integer defining accuracy of calculations.
        abseps : Absolute error tolerance.
        releps : Relative error tolerance.
        covep : Error tolerance in Cholesky factorization.
        xcutoff : Truncation limit of the normal CDF
        maxpts : Maximum number of function values allowed.
        quadno : Quadrature formulae used in integration of Xd(i)
                implicitly determining # nodes
        '''
        if speed is None:
            return
        self.speed = min(max(speed, 1), 13)

        self.maxpts = 10000
        self.quadno = r_[1:4] + (10 - min(speed, 9)) + (speed == 1)
        if speed in (11, 12, 13):
            self.abseps = 1e-1
        elif speed == 10:
            self.abseps = 1e-2
        elif speed in (7, 8, 9):
            self.abseps = 1e-2
        elif speed in (4, 5, 6):
            self.maxpts = 20000
            self.abseps = 1e-3
        elif speed in (1, 2, 3):
            self.maxpts = 30000
            self.abseps = 1e-4

        if speed < 12:
            tmp = max(abs(11 - abs(speed)), 1)
            expon = mod(tmp + 1, 3) + 1
            self.coveps = self.abseps * ((1.0e-1) ** expon)
        elif speed < 13:
            self.coveps = 0.1
        else:
            self.coveps = 0.5

        self.releps = min(self.abseps, 1.0e-2)

        if self.method == 0:
            # This gives approximately the same accuracy as when using
            # RINDDND and RINDNIT
            #    xCutOff= MIN(MAX(xCutOff+0.5d0,4.d0),5.d0)
            self.abseps = self.abseps * 1.0e-1
        trunc_error = 0.05 * max(0, self.abseps)
        self.xcutoff = max(min(abs(invnorm(trunc_error)), 7), 1.2)
        self.abseps = max(self.abseps - trunc_error, 0)

    def set_constants(self):
        if self.xcutoff is None:
            trunc_error = 0.1 * self.abseps
            self.nc1c2 = max(1, self.nc1c2)
            xcut = abs(invnorm(trunc_error / (self.nc1c2 * 2)))
            self.xcutoff = max(min(xcut, 8.5), 1.2)
            # self.abseps  = max(self.abseps- truncError,0);
            # self.releps  = max(self.releps- truncError,0);

        if self.method > 0:
            names = ['method', 'xcscale', 'abseps', 'releps', 'coveps',
                     'maxpts', 'minpts', 'nit', 'xcutoff', 'nc1c2', 'quadno',
                     'xsplit']

            constants = [getattr(self, name) for name in names]
            constants[0] = mod(constants[0], 10)
            rindmod.set_constants(*constants)  # @UndefinedVariable

    def __call__(self, cov, m, ab, bb, indI=None, xc=None, nt=None, **kwds):
        if any(kwds):
            self.__dict__.update(**kwds)
            self.set_constants()
        if xc is None:
            xc = zeros((0, 1))

        BIG, Blo, Bup, xc = atleast_2d(cov, ab, bb, xc)
        Blo = Blo.copy()
        Bup = Bup.copy()

        Ntdc = BIG.shape[0]
        Nc = xc.shape[0]
        if nt is None:
            nt = Ntdc - Nc

        unused_Mb, Nb = Blo.shape
        Nd = Ntdc - nt - Nc
        Ntd = nt + Nd

        if indI is None:
            if Nb != Ntd:
                raise ValueError('Inconsistent size of Blo and Bup')
            indI = r_[-1:Ntd]

        Ex, indI = atleast_1d(m, indI)
        if self.seed is None:
            seed = int(floor(random.rand(1) * 1e10))  # @UndefinedVariable
        else:
            seed = int(self.seed)

        #   INFIN  = INTEGER, array of integration limits flags:  size 1 x Nb
        #            if INFIN(I) < 0, Ith limits are (-infinity, infinity);
        #            if INFIN(I) = 0, Ith limits are (-infinity, Hup(I)];
        #            if INFIN(I) = 1, Ith limits are [Hlo(I), infinity);
        #            if INFIN(I) = 2, Ith limits are [Hlo(I), Hup(I)].
        infinity = 37
        dev = sqrt(diag(BIG))  # std
        ind = nonzero(indI[1:] > -1)[0]
        infin = repeat(2, len(indI) - 1)
        infin[ind] = (2 - (Bup[0, ind] > infinity * dev[indI[ind + 1]]) -
                      2 * (Blo[0, ind] < -infinity * dev[indI[ind + 1]]))

        Bup[0, ind] = minimum(Bup[0, ind], infinity * dev[indI[ind + 1]])
        Blo[0, ind] = maximum(Blo[0, ind], -infinity * dev[indI[ind + 1]])
        ind2 = indI + 1

        return rindmod.rind(BIG, Ex, xc, nt, ind2, Blo, Bup, infin, seed)  # @UndefinedVariable @IgnorePep8


def test_rind():
    ''' Small test function
    '''
    n = 5
    Blo = -inf
    Bup = -1.2
    indI = [-1, n - 1]  # Barriers
    # A = np.repeat(Blo, n)
    # B = np.repeat(Bup, n)  # Integration limits
    m = zeros(n)
    rho = 0.3
    Sc = (ones((n, n)) - eye(n)) * rho + eye(n)
    rind = Rind()
    E0 = rind(Sc, m, Blo, Bup, indI)  # exact prob. 0.001946  A)
    print(E0)

    A = repeat(Blo, n)
    B = repeat(Bup, n)  # Integration limits
    _E1 = rind(triu(Sc), m, A, B)  # same as E0

    xc = zeros((0, 1))
    infinity = 37
    dev = sqrt(diag(Sc))  # std
    ind = nonzero(indI[1:])[0]
    Bup, Blo = atleast_2d(Bup, Blo)
    Bup[0, ind] = minimum(Bup[0, ind], infinity * dev[indI[ind + 1]])
    Blo[0, ind] = maximum(Blo[0, ind], -infinity * dev[indI[ind + 1]])
    _E3 = rind(Sc, m, Blo, Bup, indI, xc, nt=1)


def cdflomax(x, alpha, m0):
    '''
    Return CDF for local maxima for a zero-mean Gaussian process

    Parameters
    ----------
    x : array-like
        evaluation points
    alpha, m0 : real scalars
        irregularity factor and zero-order spectral moment (variance of the
        process), respectively.

    Returns
    -------
    prb : ndarray
        distribution function evaluated at x

    Notes
    -----
    The cdf is calculated from an explicit expression involving the
    standard-normal cdf. This relation is sometimes written as a convolution

           M = sqrt(m0)*( sqrt(1-a^2)*Z + a*R )

    where  M  denotes local maximum, Z  is a standard normal r.v.,
    R  is a standard Rayleigh r.v., and "=" means equality in distribution.

    Note that all local maxima of the process are considered, not
    only crests of waves.

    Example
    -------
    >>> import pylab
    >>> import wafo.gaussian as wg
    >>> import wafo.spectrum.models as wsm
    >>> import wafo.objects as wo
    >>> import wafo.stats as ws
    >>> S = wsm.Jonswap(Hm0=10).tospecdata();
    >>> xs = S.sim(10000)
    >>> ts = wo.mat2timeseries(xs)
    >>> tp = ts.turning_points()
    >>> mM = tp.cycle_pairs()
    >>> m0 = S.moment(1)[0]
    >>> alpha = S.characteristic('alpha')[0]
    >>> x = np.linspace(-10,10,200);
    >>> mcdf = ws.edf(mM.data)
    >>> h = mcdf.plot(), pylab.plot(x,wg.cdflomax(x,alpha,m0))

    See also
    --------
    spec2mom, spec2bw
    '''
    c1 = 1.0 / (sqrt(1 - alpha ** 2)) * x / sqrt(m0)
    c2 = alpha * c1
    return cdfnorm(c1) - alpha * exp(-x ** 2 / 2 / m0) * cdfnorm(c2)


def prbnormtndpc(rho, a, b, D=None, df=0, abseps=1e-4, IERC=0, HNC=0.24):
    '''
    Return Multivariate normal or T probability with product correlation.

    Parameters
    ----------
    rho : array-like
        vector of coefficients defining the correlation coefficient by:
            correlation(I,J) =  rho[i]*rho[j]) for J!=I
        where -1 < rho[i] < 1
    a,b : array-like
        vector of lower and upper integration limits, respectively.
        Note: any values greater the 37 in magnitude, are considered as
        infinite values.
    D : array-like
        vector of means (default zeros(size(rho)))
    df = Degrees of freedom, NDF<=0 gives normal probabilities (default)
    abseps = absolute error tolerance. (default 1e-4)
    IERC   = 1 if strict error control based on fourth derivative
             0 if error control based on halving the intervals (default)
    HNC   = start interval width of simpson rule (default 0.24)

    Returns
    -------
    value  = estimated value for the integral
    bound  = bound on the error of the approximation
    inform = INTEGER, termination status parameter:
        0, if normal completion with ERROR < EPS;
        1, if N > 1000 or N < 1.
        2, IF  any abs(rho)>=1
        4, if  ANY(b(I)<=A(i))
        5, if number of terms exceeds maximum number of evaluation points
        6, if fault accurs in normal subroutines
        7, if subintervals are too narrow or too many
        8, if bounds exceeds abseps

     PRBNORMTNDPC calculates multivariate normal or student T probability
     with product correlation structure for rectangular regions.
     The accuracy is as best around single precision, i.e., about 1e-7.

    Example:
    --------
    >>> import wafo.gaussian as wg
    >>> rho2 = np.random.rand(2)
    >>> a2   = np.zeros(2)
    >>> b2   = np.repeat(np.inf,2)
    >>> [val2,err2, ift2] = wg.prbnormtndpc(rho2,a2,b2)
    >>> def g2(x):
    ...     return 0.25+np.arcsin(x[0]*x[1])/(2*np.pi)
    >>> E2 = g2(rho2)  # exact value
    >>> np.abs(E2-val2)<err2
    True

    >>> rho3 = np.random.rand(3)
    >>> a3 = np.zeros(3)
    >>> b3 = np.repeat(inf,3)
    >>> [val3, err3, ift3] = wg.prbnormtndpc(rho3,a3,b3)
    >>> def g3(x):
    ...     return 0.5-sum(np.sort(np.arccos([x[0]*x[1],
    ...                                       x[0]*x[2],x[1]*x[2]])))/(4*np.pi)
    >>> E3 = g3(rho3)   #  Exact value
    >>> np.abs(E3-val3) < 5 * err2
    True


    See also
    --------
    prbnormndpc, prbnormnd, Rind

    Reference
    ---------
    Charles Dunnett (1989)
    "Multivariate normal probability integrals with product correlation
    structure", Applied statistics, Vol 38,No3, (Algorithm AS 251)
    '''

    if D is None:
        D = zeros(len(rho))
    # Make sure integration limits are finite
    A = np.clip(a - D, -100, 100)
    B = np.clip(b - D, -100, 100)

    return mvnprdmod.prbnormtndpc(rho, A, B, df, abseps, IERC, HNC)  # @UndefinedVariable @IgnorePep8


def prbnormndpc(rho, a, b, abserr=1e-4, relerr=1e-4, usesimpson=True,
                usebreakpoints=False):
    '''
    Return Multivariate Normal probabilities with product correlation

    Parameters
    ----------
      rho  = vector defining the correlation structure, i.e.,
              corr(Xi,Xj) = rho(i)*rho(j) for i~=j
                          = 1             for i==j
                 -1 <= rho <= 1
      a,b   = lower and upper integration limits respectively.
      tol   = requested absolute tolerance

    Returns
    -------
    value = value of integral
    error = estimated absolute error

    PRBNORMNDPC calculates multivariate normal probability
    with product correlation structure for rectangular regions.
    The accuracy is up to almost double precision, i.e., about 1e-14.

    Example:
    -------
    >>> import wafo.gaussian as wg
    >>> rho2 = np.random.rand(2)
    >>> a2   = np.zeros(2)
    >>> b2   = np.repeat(np.inf,2)
    >>> [val2,err2, ift2] = wg.prbnormndpc(rho2,a2,b2)
    >>> g2 = lambda x : 0.25+np.arcsin(x[0]*x[1])/(2*np.pi)
    >>> E2 = g2(rho2)  #% exact value
    >>> np.abs(E2-val2)<err2
    True

    >>> rho3 = np.random.rand(3)
    >>> a3   = np.zeros(3)
    >>> b3   = np.repeat(inf,3)
    >>> [val3,err3, ift3] = wg.prbnormndpc(rho3,a3,b3)
    >>> def g3(x):
    ...     return 0.5-sum(np.sort(np.arccos([x[0]*x[1],
    ...                                       x[0]*x[2],x[1]*x[2]])))/(4*np.pi)
    >>> E3 = g3(rho3)   #  Exact value
    >>> np.abs(E3-val3)<err2
    True

    See also
    --------
    prbnormtndpc, prbnormnd, Rind

    Reference
    ---------
    P. A. Brodtkorb (2004),
    "Evaluating multinormal probabilites with product correlation structure."
    In Lund university report series
    and in the Dr.Ing thesis:
    "The probability of Occurrence of dangerous Wave Situations at Sea."
    Dr.Ing thesis, Norwegian University of Science and Technolgy, NTNU,
    Trondheim, Norway.

    '''
    # Call fortran implementation
    val, err, ier = mvnprdmod.prbnormndpc(rho, a, b, abserr, relerr, usebreakpoints, usesimpson)  # @UndefinedVariable @IgnorePep8

    if ier > 0:
        warnings.warn('Abnormal termination ier = %d\n\n%s' %
                      (ier, _ERRORMESSAGE[ier]))
    return val, err, ier

_ERRORMESSAGE = {}
_ERRORMESSAGE[0] = ''
_ERRORMESSAGE[1] = '''
    Maximum number of subdivisions allowed has been achieved. one can allow
    more subdivisions by increasing the value of limit (and taking the
    according dimension adjustments into account). however, if this yields
    no improvement it is advised to analyze the integrand in order to
    determine the integration difficulties. if the position of a local
    difficulty can be determined (i.e. singularity discontinuity within
    the interval), it should be supplied to the routine as an element of
    the vector points. If necessary an appropriate special-purpose
    integrator must be used, which is designed for handling the type of
    difficulty involved.
    '''
_ERRORMESSAGE[2] = '''
    the occurrence of roundoff error is detected, which prevents the requested
    tolerance from being achieved. The error may be under-estimated.'''

_ERRORMESSAGE[3] = '''
    Extremely bad integrand behaviour occurs at some points of the integration
    interval.'''
_ERRORMESSAGE[4] = '''
    The algorithm does not converge. Roundoff error is detected in the
    extrapolation table. It is presumed that the requested tolerance cannot be
    achieved, and that the returned result is the best which can be obtained.
    '''
_ERRORMESSAGE[5] = '''
     The integral is probably divergent, or slowly convergent.
     It must be noted that divergence can occur with any other value of ier>0.
     '''
_ERRORMESSAGE[6] = '''the input is invalid because:
        1) npts2 < 2
        2) break points are specified outside the integration range
        3) (epsabs<=0 and epsrel<max(50*rel.mach.acc.,0.5d-28))
        4) limit < npts2.'''


def prbnormnd(correl, a, b, abseps=1e-4, releps=1e-3, maxpts=None, method=0):
    '''
    Multivariate Normal probability by Genz' algorithm.


    Parameters
    CORREL = Positive semidefinite correlation matrix
    A      = vector of lower integration limits.
    B      = vector of upper integration limits.
    ABSEPS = absolute error tolerance.
    RELEPS = relative error tolerance.
    MAXPTS = maximum number of function values allowed. This
        parameter can be used to limit the time. A sensible strategy is to
        start with MAXPTS = 1000*N, and then increase MAXPTS if ERROR is too
        large.
    METHOD = integer defining the integration method
        -1 KRBVRC randomized Korobov rules for the first 20 variables,
            randomized Richtmeyer rules for the rest, NMAX = 500
         0 KRBVRC, NMAX = 100 (default)
         1 SADAPT Subregion Adaptive integration method, NMAX = 20
         2 KROBOV Randomized KOROBOV rules,              NMAX = 100
         3 RCRUDE Crude Monte-Carlo Algorithm with simple
           antithetic variates and weighted results on restart
      4 SPHMVN Monte-Carlo algorithm by Deak (1980),  NMAX = 100
    Returns
    -------
    VALUE  REAL estimated value for the integral
    ERROR  REAL estimated absolute error, with 99% confidence level.
    INFORM INTEGER, termination status parameter:
                if INFORM = 0, normal completion with ERROR < EPS;
                if INFORM = 1, completion with ERROR > EPS and MAXPTS
                               function vaules used; increase MAXPTS to
                               decrease ERROR;
                if INFORM = 2, N > NMAX or N < 1. where NMAX depends on the
                               integration method
    Example
    -------
    Compute the probability that X1<0,X2<0,X3<0,X4<0,X5<0,
    Xi are zero-mean Gaussian variables with variances one
    and correlations Cov(X(i),X(j))=0.3:
    indI=[0 5], and barriers B_lo=[-inf 0], B_lo=[0  inf]
    gives H_lo = [-inf -inf -inf -inf -inf]  H_lo = [0 0 0 0 0]

    >>> Et = 0.001946 # #  exact prob.
    >>> n = 5; nt = n
    >>> Blo =-np.inf; Bup=0; indI=[-1, n-1]  # Barriers
    >>> m = 1.2*np.ones(n); rho = 0.3;
    >>> Sc =(np.ones((n,n))-np.eye(n))*rho+np.eye(n)
    >>> rind = Rind()
    >>> E0, err0, terr0 = rind(Sc,m,Blo,Bup,indI, nt=nt)

    >>> A = np.repeat(Blo,n)
    >>> B = np.repeat(Bup,n)-m
    >>> [val,err,inform] = prbnormnd(Sc,A,B);[val, err, inform]
    [0.0019456719705212067, 1.0059406844578488e-05, 0]

    >>> np.abs(val-Et)< err0+terr0
    array([ True], dtype=bool)
    >>> 'val = %2.5f' % val
    'val = 0.00195'

    See also
    --------
    prbnormndpc, Rind
    '''

    m, n = correl.shape
    Na = len(a)
    Nb = len(b)
    if (m != n or m != Na or m != Nb):
        raise ValueError('Size of input is inconsistent!')

    if maxpts is None:
        maxpts = 1000 * n

    maxpts = max(round(maxpts), 10 * n)

#    %            array of correlation coefficients; the correlation
#    %            coefficient in row I column J of the correlation matrix
#    %            should be stored in CORREL( J + ((I-2)*(I-1))/2 ), for J < I.
#    %            The correlation matrix must be positive semidefinite.

    D = np.diag(correl)
    if (any(D != 1)):
        raise ValueError('This is not a correlation matrix')

    # Make sure integration limits are finite
    A = np.clip(a, -100, 100)
    B = np.clip(b, -100, 100)
    ix = np.where(np.triu(np.ones((m, m)), 1) != 0)
    L = correl[ix].ravel()  # % return only off diagonal elements

    infinity = 37
    infin = np.repeat(2, n) - (B > infinity) - 2 * (A < -infinity)

    err, val, inform = mvn.mvndst(A, B, infin, L, maxpts, abseps, releps)  # @UndefinedVariable @IgnorePep8

    return val, err, inform

    # CALL the mexroutine
#    t0 = clock;
#    if ((method==0) && (n<=100)),
#      %NMAX = 100
#      [value, err,inform] = mexmvnprb(L,A,B,abseps,releps,maxpts);
#    elseif ( (method<0) || ((method<=0) && (n>100)) ),
#      % NMAX = 500
#      [value, err,inform] = mexmvnprb2(L,A,B,abseps,releps,maxpts);
#    else
#      [value, err,inform] = mexGenzMvnPrb(L,A,B,abseps,releps,maxpts,method);
#    end
#    exTime = etime(clock,t0);
#  '

#     gauss legendre points and weights, n = 6
_W6 = [0.1713244923791705e+00, 0.3607615730481384e+00, 0.4679139345726904e+00]
_X6 = [-0.9324695142031522e+00, -
       0.6612093864662647e+00, -0.2386191860831970e+00]
#     gauss legendre points and weights, n = 12
_W12 = [0.4717533638651177e-01, 0.1069393259953183e+00, 0.1600783285433464e+00,
        0.2031674267230659e+00, 0.2334925365383547e+00, 0.2491470458134029e+00]
_X12 = [-0.9815606342467191e+00, -0.9041172563704750e+00,
        -0.7699026741943050e+00,
        - 0.5873179542866171e+00, -0.3678314989981802e+00,
        -0.1252334085114692e+00]
#     gauss legendre points and weights, n = 20
_W20 = [0.1761400713915212e-01, 0.4060142980038694e-01,
        0.6267204833410906e-01, 0.8327674157670475e-01,
        0.1019301198172404e+00, 0.1181945319615184e+00,
        0.1316886384491766e+00, 0.1420961093183821e+00,
        0.1491729864726037e+00, 0.1527533871307259e+00]
_X20 = [-0.9931285991850949e+00, -0.9639719272779138e+00,
        - 0.9122344282513259e+00, -0.8391169718222188e+00,
        - 0.7463319064601508e+00, -0.6360536807265150e+00,
        - 0.5108670019508271e+00, -0.3737060887154196e+00,
        - 0.2277858511416451e+00, -0.7652652113349733e-01]


def cdfnorm2d(b1, b2, r):
    '''
    Returnc Bivariate Normal cumulative distribution function

    Parameters
    ----------

    b1, b2 : array-like
        upper integration limits
    r : real scalar
        correlation coefficient  (-1 <= r <= 1).

    Returns
    -------
    bvn : ndarray
        distribution function evaluated at b1, b2.

    Notes
    -----
    CDFNORM2D computes bivariate normal probabilities, i.e., the probability
    Prob(X1 <= B1 and X2 <= B2) with an absolute error less than 1e-15.

    This function is based on the method described by Drezner, z and
    G.O. Wesolowsky, (1989), with major modifications for double precision,
    and for |r| close to 1.

    Example
    -------
    >>> import wafo.gaussian as wg
    >>> x = np.linspace(-5,5,20)
    >>> [B1,B2] = np.meshgrid(x, x)
    >>> r  = 0.3;
    >>> F = wg.cdfnorm2d(B1,B2,r)

    surf(x,x,F)

    See also
    --------
    cdfnorm

    Reference
    ---------
    Drezner, z and g.o. Wesolowsky, (1989),
    "On the computation of the bivariate normal integral",
    Journal of statist. comput. simul. 35, pp. 101-107,
    '''
    # Translated into Python
    # Per A. Brodtkorb
    #
    # Original code
    # by  alan genz
    #     department of mathematics
    #     washington state university
    #     pullman, wa 99164-3113
    #     email : alangenz@wsu.edu

    b1, b2, r = np.broadcast_arrays(b1, b2, r)
    cshape = b1.shape
    h, k, r = -b1.ravel(), -b2.ravel(), r.ravel()

    bvn = where(abs(r) > 1, nan, 0.0)

    two = 2.e0
    twopi = 6.283185307179586e0

    hk = h * k

    k0, = nonzero(abs(r) < 0.925e0)
    if len(k0) > 0:
        hs = (h[k0] ** 2 + k[k0] ** 2) / two
        asr = arcsin(r[k0])
        k1, = nonzero(r[k0] >= 0.75)
        if len(k1) > 0:
            k01 = k0[k1]
            for i in range(10):
                for sign in - 1, 1:
                    sn = sin(asr[k1] * (sign * _X20[i] + 1) / 2)
                    bvn[k01] = bvn[k01] + _W20[i] * \
                        exp((sn * hk[k01] - hs[k1]) / (1 - sn * sn))

        k1, = nonzero((0.3 <= r[k0]) & (r[k0] < 0.75))
        if len(k1) > 0:
            k01 = k0[k1]
            for i in range(6):
                for sign in - 1, 1:
                    sn = sin(asr[k1] * (sign * _X12[i] + 1) / 2)
                    bvn[k01] = bvn[k01] + _W12[i] * \
                        exp((sn * hk[k01] - hs[k1]) / (1 - sn * sn))

        k1, = nonzero(r[k0] < 0.3)
        if len(k1) > 0:
            k01 = k0[k1]
            for i in range(3):
                for sign in - 1, 1:
                    sn = sin(asr[k1] * (sign * _X6[i] + 1) / 2)
                    bvn[k01] = bvn[k01] + _W6[i] * \
                        exp((sn * hk[k01] - hs[k1]) / (1 - sn * sn))

        bvn[k0] *= asr / (two * twopi)
        bvn[k0] += fi(-h[k0]) * fi(-k[k0])

    k1, = nonzero((0.925 <= abs(r)) & (abs(r) <= 1))
    if len(k1) > 0:
        k2, = nonzero(r[k1] < 0)
        if len(k2) > 0:
            k12 = k1[k2]
            k[k12] = -k[k12]
            hk[k12] = -hk[k12]

        k3, = nonzero(abs(r[k1]) < 1)
        if len(k3) > 0:
            k13 = k1[k3]
            a2 = (1 - r[k13]) * (1 + r[k13])
            a = sqrt(a2)
            b = abs(h[k13] - k[k13])
            bs = b * b
            c = (4.e0 - hk[k13]) / 8.e0
            d = (12.e0 - hk[k13]) / 16.e0
            asr = -(bs / a2 + hk[k13]) / 2.e0
            k4, = nonzero(asr > -100.e0)
            if len(k4) > 0:
                bvn[k13[k4]] = (a[k4] * exp(asr[k4]) *
                                (1 - c[k4] * (bs[k4] - a2[k4]) *
                                 (1 - d[k4] * bs[k4] / 5) / 3 +
                                 c[k4] * d[k4] * a2[k4] ** 2 / 5))

            k5, = nonzero(hk[k13] < 100.e0)
            if len(k5) > 0:
                #               b = sqrt(bs);
                k135 = k13[k5]
                bvn[k135] = bvn[k135] - exp(-hk[k135] / 2) * sqrt(twopi) * fi(-b[k5] / a[k5]) * \
                    b[k5] * (1 - c[k5] * bs[k5] * (1 - d[k5] * bs[k5] / 5) / 3)

            a /= two
            for i in range(10):
                for sign in - 1, 1:
                    xs = (a * (sign * _X20[i] + 1)) ** 2
                    rs = sqrt(1 - xs)
                    asr = -(bs / xs + hk[k13]) / 2
                    k6, = nonzero(asr > -100.e0)
                    if len(k6) > 0:
                        k136 = k13[k6]
                        bvn[k136] += (a[k6] * _W20[i] * exp(asr[k6]) *
                                      (exp(-hk[k136] * (1 - rs[k6]) /
                                           (2 * (1 + rs[k6]))) / rs[k6] -
                                       (1 + c[k6] * xs[k6] *
                                        (1 + d[k6] * xs[k6]))))

            bvn[k3] = -bvn[k3] / twopi

        k7, = nonzero(r[k1] > 0)
        if len(k7):
            k17 = k1[k7]
            bvn[k17] += fi(-np.maximum(h[k17], k[k17]))

        k8, = nonzero(r[k1] < 0)
        if len(k8) > 0:
            k18 = k1[k8]
            bvn[k18] = -bvn[k18] + np.maximum(0, fi(-h[k18]) - fi(-k[k18]))

    bvn.shape = cshape
    return bvn


def fi(x):
    return 0.5 * (erfc((-x) / sqrt(2)))


def prbnorm2d(a, b, r):
    '''
    Returns Bivariate Normal probability

    Parameters
    ---------
    a, b : array-like, size==2
        vector with lower and upper integration limits, respectively.
    r : real scalar
        correlation coefficient

    Returns
    -------
    prb : real scalar
        computed probability Prob(A[0] <= X1 <= B[0] and A[1] <= X2 <= B[1])
        with an absolute error less than 1e-15.

    Example
    -------
    >>> import wafo.gaussian as wg
    >>> a = [-1, -2]
    >>> b = [1, 1]
    >>> r = 0.3
    >>> wg.prbnorm2d(a,b,r)
    array([ 0.56659121])

    See also
    --------
    cdfnorm2d,
    cdfnorm,
    prbnormndpc
    '''
    infinity = 37
    lower = np.asarray(a)
    upper = np.asarray(b)
    if np.all((lower <= -infinity) & (infinity <= upper)):
        return 1.0
    if (lower >= upper).any():
        return 0.0
    correl = r
    infin = np.repeat(2, 2) - (upper > infinity) - 2 * (lower < -infinity)

    if np.all(infin == 2):
        return (bvd(lower[0], lower[1], correl) -
                bvd(upper[0], lower[1], correl) -
                bvd(lower[0], upper[1], correl) +
                bvd(upper[0], upper[1], correl))
    elif (infin[0] == 2 and infin[1] == 1):
        return (bvd(lower[0], lower[1], correl) -
                bvd(upper[0], lower[1], correl))
    elif (infin[0] == 1 and infin[1] == 2):
        return (bvd(lower[0], lower[1], correl) -
                bvd(lower[0], upper[1], correl))
    elif (infin[0] == 2 and infin[1] == 0):
        return (bvd(-upper[0], -upper[1], correl) -
                bvd(-lower[0], -upper[1], correl))
    elif (infin[0] == 0 and infin[1] == 2):
        return (bvd(-upper[0], -upper[1], correl) -
                bvd(-upper[0], -lower[1], correl))
    elif (infin[0] == 1 and infin[1] == 0):
        return bvd(lower[0], -upper[1], -correl)
    elif (infin[0] == 0 and infin[1] == 1):
        return bvd(-upper[0], lower[1], -correl)
    elif (infin[0] == 1 and infin[1] == 1):
        return bvd(lower[0], lower[1], correl)
    elif (infin[0] == 0 and infin[1] == 0):
        return bvd(-upper[0], -upper[1], correl)
    return 1


def bvd(lo, up, r):
    return cdfnorm2d(-lo, -up, r)


def test_docstrings():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    test_docstrings()
# if __name__ == '__main__':
#    if False: #True: #
#        test_rind()
#    else:
#        import doctest
#        doctest.testmod()
