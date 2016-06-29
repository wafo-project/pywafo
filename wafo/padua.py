'''

All the software  contained in this library  is protected by copyright.
Permission  to use, copy, modify, and  distribute this software for any
purpose without fee is hereby granted, provided that this entire notice
is included  in all copies  of any software which is or includes a copy
or modification  of this software  and in all copies  of the supporting
documentation for such software.

THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
WARRANTY. IN NO EVENT, NEITHER  THE AUTHORS, NOR THE PUBLISHER, NOR ANY
MEMBER  OF THE EDITORIAL BOARD OF  THE JOURNAL  "NUMERICAL ALGORITHMS",
NOR ITS EDITOR-IN-CHIEF, BE  LIABLE FOR ANY ERROR  IN THE SOFTWARE, ANY
MISUSE  OF IT  OR ANY DAMAGE ARISING OUT OF ITS USE. THE ENTIRE RISK OF
USING THE SOFTWARE LIES WITH THE PARTY DOING SO.

ANY USE OF THE SOFTWARE  CONSTITUTES  ACCEPTANCE  OF THE TERMS  OF THE
ABOVE STATEMENT.


AUTHORS:
Per A Brodtkorb
Python code Based on matlab code written by:

Marco Caliari
University of Verona, Italy
E-mail: marco.caliari@univr.it

Stefano de Marchi, Alvise Sommariva, Marco Vianello
University of Padua, Italy
E-mail: demarchi@math.unipd.it, alvise@math.unipd.it,
        marcov@math.unipd.it

Reference
---------
Padua2DM: fast interpolation and cubature at the Padua points in Matlab/Octave
NUMERICAL ALGORITHMS, 56 (2011), PP. 45-60


Padua module
------------
In polynomial interpolation of two variables, the Padua points are the first
known example (and up to now the only one) of a unisolvent point set
(that is, the interpolating polynomial is unique) with minimal growth of their
Lebesgue constant, proven to be O(log2 n).
This module provides all the functions needed to perform interpolation and
cubature at the Padua points, together with the functions and the demos used
in the paper.

pdint.m                 : main function for interpolation at the Padua points
pdcub.m                 : main function for cubature at the Padua points
pdpts.m                 : function for the computation of the Padua points
padua_fit.m              : function for the computation of the interpolation
                          coefficients by FFT (recommended)
pdcfsMM.m               : function for the computation of the interpolation
                          coefficients by matrix multiplications
pdval.m                 : function for the evaluation of the interpolation
                          polynomial
pdwtsFFT.m              : function for the computation of the cubature
                          weights by FFT
pdwtsMM.m               : function for the computation of the cubature
                          weights by matrix multiplications (recommended)
funct.m                 : function containing some test functions
demo_pdint.m            : demo script for pdint
demo_cputime_pdint.m    : demo script for the computation of CPU time for
                          interpolation
demo_errors_pdint.m     : demo script for the comparison of interpolation with
                          coefficients computed by FFT or by matrix
                          multiplications
demo_pdcub              : demo script for pdcub
demo_cputime_pdcub.m    : demo script for the computation of CPU time for
                          cubature
demo_errors_pdcub.m     : demo script for the comparison of cubature with
                          weights computed by FFT or by matrix multiplications
demo_errors_pdcub_gl.m  : demo script for the comparison of different cubature
                          formulas
cubature_square.m       : function for the computation of some cubature
                          formulas for the square
omelyan_solovyan_rule.m : function for the computation of Omelyan-Solovyan
                          cubature points and weights
Contents.m              : Contents file for Matlab


'''
from __future__ import absolute_import, division
import numpy as np
from numpy.fft import fft
from .dctpack import dct
# from scipy.fftpack.realtransforms import dct


class _ExampleFunctions(object):
    '''
    Computes test function in the points (x, y)

    Parameters
    ----------
    x,y : array-like
        evaluate the function in the points (x,y)
    id : scalar int (default 0)
        id defining which test function to use. Options are
        0: franke
        1: half_sphere
        2: poly_degree20
        3: exp_fun1
        4: exp_fun100
        5: cos30
        6: constant
        7: exp_xy
        8: runge
        9: abs_cubed
        10: gauss
        11: exp_inv

    Returns
    -------
    z : array-like
        value of the function in the points (x,y)
    '''
    @staticmethod
    def franke(x, y):
        '''Franke function.

        The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 500, is
        2.1547794245591083e+000 with an estimated absolute error of 8.88e-016.

        The value of the definite integral on the square [0,1] x [0,1],
        obtained using a Padua Points cubature formula of degree 500, is
        4.06969589491556e-01 with an estimated absolute error of 8.88e-016.

        Maple: 0.40696958949155611906
        '''
        exp = np.exp
        return (3. / 4 * exp(-((9. * x - 2)**2 + (9. * y - 2)**2) / 4) +
                3. / 4 * exp(-(9. * x + 1)**2 / 49 - (9. * y + 1) / 10) +
                1. / 2 * exp(-((9. * x - 7)**2 + (9. * y - 3)**2) / 4) -
                1. / 5 * exp(-(9. * x - 4)**2 - (9. * y - 7)**2))

    @staticmethod
    def half_sphere(x, y):
        '''The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 2000, is
        3.9129044444568244e+000 with an estimated absolute error of 3.22e-010.
        '''
        return ((x - 0.5)**2 + (y - 0.5)**2)**(1. / 2)

    @staticmethod
    def poly_degree20(x, y):
        ''''Bivariate polynomial having moderate degree.
        The value of the definite integral on the square [-1,1] x
        [-1,1], up to machine precision, is 18157.16017316017 (see ref. 6).
        The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 500,
        is 1.8157160173160162e+004.

        2D modification of an example by L.N.Trefethen (see ref. 7), f(x)=x^20.
        '''
        return (x + y)**20

    @staticmethod
    def exp_fun1(x, y):
        ''' The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 2000,
        is 2.1234596326670683e+001 with an estimated absolute error of
        7.11e-015.
        '''
        return np.exp((x - 0.5)**2 + (y - 0.5)**2)

    @staticmethod
    def exp_fun100(x, y):
        '''The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 2000,
        is 3.1415926535849605e-002 with an estimated absolute error of
        3.47e-017.
        '''
        return np.exp(-100 * ((x - 0.5)**2 + (y - 0.5)**2))

    @staticmethod
    def cos30(x, y):
        ''' The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 500,
        is 4.3386955120336568e-003 with an estimated absolute error of
        2.95e-017.
        '''
        return np.cos(30 * (x + y))

    @staticmethod
    def constant(x, y):
        '''Constant.
        To test interpolation and cubature at degree 0.
        The value of the definite integral on the square [-1,1] x [-1,1]
        is 4.
        '''
        return np.ones(np.shape(x))

    @staticmethod
    def exp_xy(x, y):
        '''The value of the definite integral on the square [-1,1] x [-1,1]
        is up to machine precision is 5.524391382167263 (see ref. 6).
        2. The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 500,
        is 5.5243913821672628e+000 with an estimated absolute error of
        0.00e+000.
        2D modification of an example by L.N.Trefethen (see ref. 7),
        f(x)=exp(x).
        '''
        return np.exp(x + y)

    @staticmethod
    def runge(x, y):
        ''' Bivariate Runge function: as 1D complex function is analytic
        in a neighborhood of [-1; 1] but not throughout the complex plane.

        The value of the definite integral on the square [-1,1] x [-1,1],
        up to machine precision, is 0.597388947274307 (see ref. 6).
        The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 500,
        is 5.9738894727430725e-001 with an estimated absolute error of
        0.00e+000.

        2D modification of an example by L.N.Trefethen (see ref. 7),
        f(x)=1/(1+16*x^2).
        '''
        return 1. / (1 + 16 * (x**2 + y**2))

    @staticmethod
    def abs_cubed(x, y):
        '''Low regular function.
        The value of the definite integral on the square [-1,1] x [-1,1],
        up to machine precision, is 2.508723139534059 (see ref. 6).
        The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 500,
        is 2.5087231395340579e+000 with an estimated absolute error of
        0.00e+000.

        2D modification of an example by L.N.Trefethen (see ref. 7),
        f(x)=abs(x)^3.
        '''
        return (x**2 + y**2)**(3 / 2)

    @staticmethod
    def gauss(x, y):
        '''Bivariate gaussian: smooth function.
        The value of the definite integral on the square [-1,1] x [-1,1],
        up to machine precision, is 2.230985141404135 (see ref. 6).
        The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 500,
        is 2.2309851414041333e+000 with an estimated absolute error of
        2.66e-015.

        2D modification of an example by L.N.Trefethen (see ref. 7),
        f(x)=exp(-x^2).
        '''
        return np.exp(-x**2 - y**2)

    @staticmethod
    def exp_inv(x, y):
        '''Bivariate example stemming from a 1D C-infinity function.
        The value of the definite integral on the square [-1,1] x [-1,1],
        up to machine precision, is 0.853358758654305 (see ref. 6).
        The value of the definite integral on the square [-1,1] x [-1,1],
        obtained using a Padua Points cubature formula of degree 2000,
        is 8.5335875865430544e-001 with an estimated absolute error of
        3.11e-015.

        2D modification of an example by L.N.Trefethen (see ref. 7),
        f(x)=exp(-1/x^2).
        '''
        arg_z = (x**2 + y**2)
        # Avoid cases in which "arg_z=0", setting only in those instances
        # "arg_z=eps".
        arg_z = arg_z + (1 - np.abs(np.sign(arg_z))) * 1.e-100
        arg_z = 1. / arg_z
        return np.exp(-arg_z)

    def __call__(self, x, y, id=0):  # @ReservedAssignment
        s = self
        test_function = [s.franke, s.half_sphere, s.poly_degree20, s.exp_fun1,
                         s.exp_fun100, s.cos30, s.constant, s.exp_xy, s.runge,
                         s.abs_cubed, s.gauss, s.exp_inv]
        return test_function[id](x, y)
example_functions = _ExampleFunctions()


def _find_m(n):
    ix = np.r_[1:(n + 1) * (n + 2):2]
    if np.mod(n, 2) == 0:
        n2 = n // 2
        offset = np.array([[0, 1] * n2 + [0, ]] * (n2 + 1))
        ix = ix - offset.ravel(order='F')
    return ix


def padua_points(n, domain=(-1, 1, -1, 1)):
    ''' Return Padua points

    Parameters
    ----------
    n : scalar integer
         interpolation degree
    domain : vector [a,b,c,d]
        defining the rectangle [a,b] x [c,d]. (default domain = (-1,1,-1,1))

    Returns
    -------
    pad : array of shape (2 x (n+1)*(n+2)/2) such that
        (pad[0,:], pad[1,: ]) defines the Padua points in the domain
        rectangle [a,b] x [c,d].
    or
     X1,Y1,X2,Y2 : arrays
         Two subgrids X1,Y1 and X2,Y2 defining the Padua points
    -------------------------------------------------------------------------------
    '''
    a, b, c, d = domain
    t0 = [np.pi] if n == 0 else np.linspace(0, np.pi, n + 1)
    t1 = np.linspace(0, np.pi, n + 2)
    zn = (a + b + (b - a) * np.cos(t0)) / 2
    zn1 = (c + d + (d - c) * np.cos(t1)) / 2

    Pad1, Pad2 = np.meshgrid(zn, zn1)
    ix = _find_m(n)
    return np.vstack((Pad1.ravel(order='F')[ix],
                      Pad2.ravel(order='F')[ix]))


def error_estimate(C0f):
    ''' Return interpolation error estimate from Padua coefficients
    '''
    n = C0f.shape[1]
    C0f2 = np.fliplr(C0f)
    errest = sum(np.abs(np.diag(C0f2)))
    if (n >= 1):
        errest = errest + sum(np.abs(np.diag(C0f2, -1)))
        if (n >= 2):
            errest = errest + sum(np.abs(np.diag(C0f2, -2)))
    return 2 * errest


def padua_fit(Pad, fun, *args):
    '''
    Computes the Chebyshevs coefficients

    so that f(x, y) can be approximated by:

           f(x, y) = sum cjk*Tjk(x, y)

    Parameters
    ----------
    Pad : array-like
        Padua points, as computed  with padua_points function.
    fun : function to be interpolated in the form
        fun(x, y, *args), where *args are optional arguments for fun.

    Returns
    -------
    coefficents: coefficient matrix
    abs_err   : interpolation error estimate

    '''

    N = np.shape(Pad)[1]
    # recover the degree n from N = (n+1)(n+2)/2
    n = int(round(-3 + np.sqrt(1 + 8 * N)) / 2)
    C0f = fun(Pad[0], Pad[1], *args)
    if (n > 0):
        ix = _find_m(n)
        GfT = np.zeros((n + 2) * (n + 1))
        GfT[ix] = C0f * 2 / (n * (n + 1))
        GfT = GfT.reshape(n + 1, n + 2)
        GfT = GfT.T
        GfT[0] = GfT[0] / 2
        GfT[n + 1] = GfT[n + 1] / 2
        GfT[:, 0] = GfT[:, 0] / 2
        GfT[:, n] = GfT[:, n] / 2
        Gf = GfT.T
        # compute the interpolation coefficient matrix C0f by FFT
        Gfhat = np.real(fft(Gf, 2 * n, axis=0))
        Gfhathat = np.real(fft(Gfhat[:n + 1, :], 2 * (n + 1), axis=1))
        C0f = 2 * Gfhathat[:, 0:n + 1]
        C0f[0] = C0f[0, :] / np.sqrt(2)
        C0f[:, 0] = C0f[:, 0] / np.sqrt(2)
        C0f = np.fliplr(np.triu(np.fliplr(C0f)))
        C0f[n] = C0f[n] / 2

    return C0f, error_estimate(C0f)


def paduavals2coefs(f):
    useFFTwhenNisMoreThan = 100
    m = len(f)
    n = int(round(-1.5 + np.sqrt(.25 + 2 * m)))
    x = padua_points(n)
    idx = _find_m(n)
    w = 0 * x[0] + 1. / (n * (n + 1))
    idx1 = np.all(np.abs(x) == 1, axis=0)
    w[idx1] = .5 * w[idx1]
    idx2 = np.all(np.abs(x) != 1, axis=0)
    w[idx2] = 2 * w[idx2]

    G = np.zeros(idx.max() + 1)
    G[idx] = 4 * w * f

    if (n < useFFTwhenNisMoreThan):
        t1 = np.r_[0:n + 1].reshape(-1, 1)
        Tn1 = np.cos(t1 * t1.T * np.pi / n)
        t2 = np.r_[0:n + 2].reshape(-1, 1)
        Tn2 = np.cos(t2 * t2.T * np.pi / (n + 1))
        C = np.dot(Tn2, np.dot(G, Tn1))
    else:

        # dct = @(c) chebtech2.coeffs2vals(c);
        C = np.rot90(dct(dct(G.T).T))  # , axis=1)

    C[0] = .5 * C[0]
    C[:, 1] = .5 * C[:, 1]
    C[0, -1] = .5 * C[0, -1]
    del C[-1]

    # Take upper-left triangular part:
    return np.fliplr(np.triu(np.fliplr(C)))
    # C = triu(C(:,end:-1:1));
    # C = C(:,end:-1:1);


# TODO: padua_fit2 does not work correctly yet.
def padua_fit2(Pad, fun, *args):
    N = np.shape(Pad)[1]
    # recover the degree n from N = (n+1)(n+2)/2
    _n = int(round(-3 + np.sqrt(1 + 8 * N)) / 2)
    C0f = fun(Pad[0], Pad[1], *args)
    return paduavals2coefs(C0f)


def _compute_moments(n):
    k = np.r_[0:n:2]
    mom = 2 * np.sqrt(2) / (1 - k ** 2)
    mom[0] = 2
    return mom


def padua_cubature(coefficients, domain=(-1, 1, -1, 1)):
    '''
    Compute the integral through the coefficient matrix.
    '''
    n = coefficients.shape[1]
    mom = _compute_moments(n)
    M1, M2 = np.meshgrid(mom, mom)
    M = M1 * M2
    C0fM = coefficients[0:n:2, 0:n:2] * M
    a, b, c, d = domain
    integral = (b - a) * (d - c) * C0fM.sum() / 4
    return integral


def padua_val(X, Y, coefficients, domain=(-1, 1, -1, 1), use_meshgrid=False):
    '''
    Evaluate polynomial in padua form at X, Y.

     Evaluate the interpolation polynomial defined through its coefficient
     matrix coefficients at the target points X(:,1),X(:,2) or at the
     meshgrid(X1,X2)

    Parameters
    ----------
    X, Y: array-like
        evaluation points.
    coefficients : array-like
         coefficient matrix
    domain : a vector [a,b,c,d]
         defining the rectangle [a,b] x [c,d]
    use_meshgrid: bool
        If True interpolate at the points meshgrid(X, Y)

    Returns
    -------
     fxy : array-like
         evaluation of the interpolation polynomial at the target points
    '''
    X, Y = np.atleast_1d(X, Y)
    original_shape = X.shape
    min, max = np.minimum, np.maximum  # @ReservedAssignment
    a, b, c, d = domain
    n = np.shape(coefficients)[1]

    X1 = min(max(2 * (X.ravel() - a) / (b - a) - 1, -1), 1).reshape(1, -1)
    X2 = min(max(2 * (Y.ravel() - c) / (d - c) - 1, -1), 1).reshape(1, -1)
    tn = np.r_[0:n][:, None]
    TX1 = np.cos(tn * np.arccos(X1))
    TX2 = np.cos(tn * np.arccos(X2))
    TX1[1:n + 1] = TX1[1:n + 1] * np.sqrt(2)
    TX2[1:n + 1] = TX2[1:n + 1] * np.sqrt(2)
    if use_meshgrid:  # eval on meshgrid points
        return np.dot(TX1.T, np.dot(coefficients, TX2)).T
    # scattered points
    val = np.sum(np.dot(TX1.T, coefficients) * TX2.T, axis=1)
    return val.reshape(original_shape)
