import numpy as np
from numpy import (r_, minimum, maximum, atleast_1d, atleast_2d, mod, zeros, #@UnresolvedImport
        ones, floor, random, eye, nonzero, repeat, sqrt, inf, diag, triu) #@UnresolvedImport
from scipy.special import ndtri as invnorm
import rindmod


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
        Lower and upper barriers used to compute the integration limits, Hlo and Hup, respectively. 
    indI : array-like, length Ni
        vector of indices to the different barriers in the indicator function.
        (NB! restriction  indI(1)=0, indI(NI)=Nt+Nd, Ni = Nb+1)
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
        variables where Xt,Xd and Xc have the length Nt,Nd and Nc, respectively.
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
    >>> n = 5
    >>> Blo =-np.inf; Bup=-1.2; indI=[-1, n-1]  # Barriers
    >>> m = np.zeros(n); rho = 0.3;
    >>> Sc =(np.ones((n,n))-np.eye(n))*rho+np.eye(n)
    >>> rind = Rind()
    >>> E0 = rind(Sc,m,Blo,Bup,indI)  #  exact prob. 0.001946

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
    >>> rind(Sc,m,Blo,Bup,indI, xc, nt=0)
    (array([ 0.05494076]), array([ 0.00083066]), array([  1.00000000e-10]))
    
    Compute expectation E( X1^{+}*X2^{+} ) with random
    correlation coefficient,Cov(X1,X2) = rho2.
    >>> m2  = [0, 0]; rho2 = np.random.rand(1)
    >>> Sc2 = [[1, rho2], [rho2 ,1]]
    >>> Blo2 = 0; Bup2 = np.inf; indI2 = [-1, 1]
    >>> rind2 = Rind(method=1)
    >>> g2 = lambda x : (x*(np.pi/2+np.arcsin(x))+np.sqrt(1-x**2))/(2*np.pi)
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
    Methodology And Computing In Applied Probability, Volume 8, Number 1, pp. 65-91(27)
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
            absolute and relative error tolerance. (default abseps=0, releps=1e-3)
        coveps : real scalar, optional
            error tolerance in Cholesky factorization (default 1e-13)
        maxpts, minpts : scalar integers, optional
            maximum and minimum number of function values allowed. The parameter, 
            maxpts, can be used to limit the time. A sensible strategy is to start 
            with MAXPTS = 1000*N, and then increase MAXPTS if ERROR is too large.    
                (Only for METHOD~=0) (default maxpts=40000, minpts=0) 
        seed : scalar integer, optional
            seed to the random generator used in the integrations
                (Only for METHOD~=0)(default floor(rand*1e9))
        nit : scalar integer, optional 
            maximum number of Xt variables to integrate. This parameter can be used 
            to limit the time. If NIT is less than the rank of the covariance matrix,
            the returned result is a upper bound for the true value of the integral.  
            (default 1000)
        xcutoff : real scalar, optional
            cut off value where the marginal normal distribution is truncated. 
            (Depends on requested accuracy. A value between 4 and 5 is reasonable.)
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
            number of times to use the regression equation to restrict integration 
            area. Nc1c2 = 1,2 is recommended. (default 2) 
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
            
        if self.method == 0 : 
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
            #self.abseps  = max(self.abseps- truncError,0);
            #self.releps  = max(self.releps- truncError,0);

        if self.method > 0:
            names = ['method', 'xcscale', 'abseps', 'releps', 'coveps',
                    'maxpts', 'minpts', 'nit', 'xcutoff', 'nc1c2', 'quadno',
                    'xsplit']

            constants = [getattr(self, name) for name in names]
            constants[0] = mod(constants[0], 10)
            rindmod.set_constants(*constants) #@UndefinedVariable
        
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
            seed = int(floor(random.rand(1) * 1e10)) #@UndefinedVariable
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
        infin[ind] = (2 - (Bup[0, ind] > infinity * dev[indI[ind + 1]]) 
                      - 2 * (Blo[0, ind] < -infinity * dev[indI[ind + 1]]))

        Bup[0, ind] = minimum(Bup[0, ind], infinity * dev[indI[ind + 1]])
        Blo[0, ind] = maximum(Blo[0, ind], -infinity * dev[indI[ind + 1]])
        ind2 = indI + 1
        return rindmod.rind(BIG, Ex, xc, nt, ind2, Blo, Bup, infin, seed) #@UndefinedVariable
              
def test_rind():
    ''' Small test function
    '''
    n = 5
    Blo = -inf
    Bup = -1.2
    indI = [-1, n - 1]  # Barriers
#    A = np.repeat(Blo, n)
#    B = np.repeat(Bup, n)  # Integration limits
    m = zeros(n)
    rho = 0.3
    Sc = (ones((n, n)) - eye(n)) * rho + eye(n)
    rind = Rind()
    E0 = rind(Sc, m, Blo, Bup, indI)  #  exact prob. 0.001946  A)    
    print(E0)
    
    A = repeat(Blo, n) 
    B = repeat(Bup, n)  # Integration limits
    E1 = rind(triu(Sc), m, A, B)   #same as E0
    
    xc = zeros((0, 1))
    infinity = 37
    dev = sqrt(diag(Sc))  # std
    ind = nonzero(indI[1:])[0]
    Bup, Blo = atleast_2d(Bup, Blo)
    Bup[0, ind] = minimum(Bup[0, ind], infinity * dev[indI[ind + 1]])
    Blo[0, ind] = maximum(Blo[0, ind], -infinity * dev[indI[ind + 1]])
    E3 = rind(Sc, m, Blo, Bup, indI, xc, nt=1)

       
if __name__ == '__main__':
    if False: #True: #  
        test_rind()
    else:
        import doctest
        doctest.testmod()
