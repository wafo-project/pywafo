#from math import *
from numpy import zeros, convolve, dot, linalg, size #@UnresolvedImport

all = ['calc_coeff','smooth']

def _resub(D, rhs):
    """ solves D D^T = rhs by resubstituion.
        D is lower triangle-matrix from cholesky-decomposition """

    M = D.shape[0]
    x1= zeros((M,),float)
    x2= zeros((M,),float)

    # resub step 1
    for l in range(M):
        sum = rhs[l]
        for n in range(l):
            sum -= D[l,n]*x1[n]
        x1[l] = sum/D[l,l]

    # resub step 2
    for l in range(M-1,-1,-1):
        sum = x1[l]
        for n in range(l+1,M):
            sum -= D[n,l]*x2[n]
        x2[l] = sum/D[l,l]

    return x2


def calc_coeff(num_points, pol_degree, diff_order=0):

    """
    Calculates filter coefficients for symmetric savitzky-golay filter.
    see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

    Parameters
    ----------
    num_points : scalar, integer
        means that 2*num_points+1 values contribute to the smoother.
    pol_degree : scalar, integer
        is degree of fitting polynomial
    diff_order : scalar, integer
        is degree of implicit differentiation.
        0 means that filter results in smoothing of function
        1 means that filter results in smoothing the first
                                    derivative of function.
        and so on ...

    """

    # setup normal matrix
    A = zeros((2*num_points+1, pol_degree+1), float)
    for i in range(2*num_points+1):
        for j in range(pol_degree+1):
            A[i,j] = pow(i-num_points, j)

    # calculate diff_order-th row of inv(A^T A)
    ATA = dot(A.transpose(), A)
    rhs = zeros((pol_degree+1,), float)
    rhs[diff_order] = 1
    D = linalg.cholesky(ATA)
    wvec = _resub(D, rhs)

    # calculate filter-coefficients
    coeff = zeros((2*num_points+1,), float)
    for n in range(-num_points, num_points+1):
        x = 0.0
        for m in range(pol_degree+1):
            x += wvec[m]*pow(n, m)
        coeff[n+num_points] = x
    return coeff

def smooth(signal, coeff):
    """
    applies coefficients calculated by calc_coeff()
        to signal
    """

    N = size(coeff-1)/2
    res = convolve(signal, coeff)
    return res[N:-N]


