#-------------------------------------------------------------------------------
# Name:        kdetools
# Purpose:
#
# Author:      pab
#
# Created:     01.11.2008
# Copyright:   (c) pab 2008
# Licence:     LGPL
#-------------------------------------------------------------------------------
#!/usr/bin/env python
from __future__ import division
import warnings
import numpy as np
from numpy import pi, sqrt, atleast_2d, exp, newaxis, array #@UnresolvedImport
import scipy
from scipy import linalg
from scipy.special import gamma
from misc import tranproc, trangood
from itertools import product
from wafo.misc import meshgrid

_stats_epan = (1. / 5, 3. / 5, np.inf)
_stats_biwe = (1. / 7, 5. / 7, 45. / 2)
_stats_triw = (1. / 9, 350. / 429, np.inf)
_stats_rect = (1. / 3, 1. / 2, np.inf)
_stats_tria = (1. / 6, 2. / 3, np.inf)
_stats_lapl = (2, 1. / 4, np.inf)
_stats_logi = (pi ** 2 / 3, 1. / 6, 1 / 42)
_stats_gaus = (1, 1. / (2 * sqrt(pi)), 3. / (8 * sqrt(pi)))
              


def sphere_volume(d, r=1.0):
    """
    Returns volume of  d-dimensional sphere with radius r

    Parameters
    ----------
    d : scalar or array_like
        dimension of sphere
    r : scalar or array_like
        radius of sphere (default 1)
        
    Example
    -------
    >>> sphere_volume(2., r=2.)
    12.566370614359172
    >>> sphere_volume(2., r=1.)
    3.1415926535897931

    Reference
    ---------
    Wand,M.P. and Jones, M.C. (1995)
    'Kernel smoothing'
    Chapman and Hall, pp 105
    """
    return (r ** d) * 2. * pi ** (d / 2.) / (d * gamma(d / 2.))

class TKDE(object):
    """ Transformation Kernel-Density Estimator.

    Parameters
    ----------
    dataset : (# of dims, # of data)-array
        datapoints to estimate from
    hs : array-like (optional) 
        smooting parameter vector/matrix.
        (default compute from data using kernel.get_smoothing function)
    kernel :  kernel function object.
        kernel must have get_smoothing method
    alpha : real scalar (optional)
        sensitivity parameter               (default 0 regular KDE)
        A good choice might be alpha = 0.5 ( or 1/D)
        alpha = 0      Regular  KDE (hs is constant)
        0 < alpha <= 1 Adaptive KDE (Make hs change)  
    L2 : array-like 
        vector of transformation parameters (default 1 no transformation)
        t(xi;L2) = xi^L2*sign(L2)   for L2(i) ~= 0
        t(xi;L2) = log(xi)          for L2(i) == 0 
        If single value of L2 is given then the transformation is the same in all directions.
        
    Members
    -------
    d : int
        number of dimensions
    n : int
        number of datapoints

    Methods
    -------
    kde.evaluate(points) : array
        evaluate the estimated pdf on a provided set of points
    kde(points) : array
        same as kde.evaluate(points)
   
    
    Example
    -------
    N = 20
    data = np.random.rayleigh(1, size=(N,))
    >>> data = array([ 0.75355792,  0.72779194,  0.94149169,  0.07841119,  2.32291887,
    ...        1.10419995,  0.77055114,  0.60288273,  1.36883635,  1.74754326,
    ...        1.09547561,  1.01671133,  0.73211143,  0.61891719,  0.75903487,
    ...        1.8919469 ,  0.72433808,  1.92973094,  0.44749838,  1.36508452])

    >>> import wafo.kdetools as wk
    >>> x = np.linspace(0.01, max(data.ravel()) + 1, 10)  
    >>> kde = wk.TKDE(data, hs=0.5, L2=0.5)
    >>> f = kde(x)
    >>> f
    array([ 1.03982714,  0.45839018,  0.39514782,  0.32860602,  0.26433318,
            0.20717946,  0.15907684,  0.1201074 ,  0.08941027,  0.06574882])
    
    import pylab as plb          
    h1 = plb.plot(x, f) #  1D probability density plot
    t = np.trapz(f, x)   
    """

    def __init__(self, dataset, hs=None, kernel=None, alpha=0.0, L2=None):
        self.dataset = atleast_2d(dataset)
        self.hs = hs
        self.kernel = kernel
        self.alpha = alpha
        self.L2 = L2
        self.d, self.n = self.dataset.shape
        self.initialize()
    
    def initialize(self):
        tdataset = self._dat2gaus(self.dataset)
        self.kde = KDE(tdataset, self.hs, self.kernel, self.alpha)
    
    def _check_shape(self, points):
        points = atleast_2d(points)
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)
        return points   
    
    def _dat2gaus(self, points):
        if self.L2 is None:
            return points # default no transformation
        
        L2 = np.atleast_1d(self.L2) * np.ones(self.d) # default no transformation
        
        tpoints = points.copy()
        for i, v2 in enumerate(L2.tolist()):
            tpoints[i] = np.where(v2 == 0, np.log(points[i]), points[i] ** v2)
        return tpoints
        
    def _scale_pdf(self, pdf, points):
        if self.L2 is None:
            return pdf
        L2 = np.atleast_1d(self.L2) * np.ones(self.d) # default no transformation
        for i, v2 in enumerate(L2.tolist()):
            factor =  v2 * np.sign(v2) if v2 else 1
            pdf *= np.where(v2 == 1, 1, points[i] ** (v2 - 1) * factor)
        if (np.abs(np.diff(pdf)).max() > 10).any():
            msg = ''' Numerical problems may have occured due to the power
                    transformation. Check the KDE for spurious spikes'''
            warnings.warn(msg)
        return pdf
    
    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError if the dimensionality of the input points is different than
        the dimensionality of the KDE.
        """
        if self.L2 is None:
            return self.kde(points)
        points = self._check_shape(points)
        tpoints = self._dat2gaus(points)
        tf = self.kde(tpoints)
        f = self._scale_pdf(tf, points)
        return f
    
    __call__ = evaluate
    
class KDE(object):
    """ Kernel-Density Estimator.

    Parameters
    ----------
    dataset : (# of dims, # of data)-array
        datapoints to estimate from
    hs : array-like (optional) 
        smooting parameter vector/matrix.
        (default compute from data using kernel.get_smoothing function)
    kernel :  kernel function object.
        kernel must have get_smoothing method
    alpha : real scalar (optional)
        sensitivity parameter               (default 0 regular KDE)
        A good choice might be alpha = 0.5 ( or 1/D)
        alpha = 0      Regular  KDE (hs is constant)
        0 < alpha <= 1 Adaptive KDE (Make hs change)  


    Members
    -------
    d : int
        number of dimensions
    n : int
        number of datapoints

    Methods
    -------
    kde.evaluate(points) : array
        evaluate the estimated pdf on a provided set of points
    kde(points) : array
        same as kde.evaluate(points)
   
    
    Example
    -------
    N = 20
    data = np.random.rayleigh(1, size=(N,))
    >>> data = array([ 0.75355792,  0.72779194,  0.94149169,  0.07841119,  2.32291887,
    ...        1.10419995,  0.77055114,  0.60288273,  1.36883635,  1.74754326,
    ...        1.09547561,  1.01671133,  0.73211143,  0.61891719,  0.75903487,
    ...        1.8919469 ,  0.72433808,  1.92973094,  0.44749838,  1.36508452])

    >>> x = np.linspace(0, max(data.ravel()) + 1, 10) 
    >>> import wafo.kdetools as wk 
    >>> kde = wk.KDE(data, hs=0.5, alpha=0.5)
    >>> f = kde(x)
    >>> f
    array([ 0.17252055,  0.41014271,  0.61349072,  0.57023834,  0.37198073,
            0.21409279,  0.12738463,  0.07460326,  0.03956191,  0.01887164])
    
    import pylab as plb          
    h1 = plb.plot(x, f) #  1D probability density plot
    t = np.trapz(f, x)   
    """

    def __init__(self, dataset, hs=None, kernel=None, alpha=0.0):
        self.kernel = kernel if kernel else Kernel('gauss')
        self.hs = hs
        self.alpha = alpha
        
        self.dataset = atleast_2d(dataset)
        self.d, self.n = self.dataset.shape
        self.initialize()

    def initialize(self):
        self._compute_smoothing()
        if self.alpha > 0:
            pilot = KDE(self.dataset, hs=self.hs, kernel=self.kernel, alpha=0)
            f = pilot(self.dataset) # get a pilot estimate by regular KDE (alpha=0)
            g = np.exp(np.mean(np.log(f)))
            self._lambda = (f / g) ** (-self.alpha)
        else:
            self._lambda = np.ones(self.n)

    def _compute_smoothing(self):
        """Computes the smoothing matrix
        """
        get_smoothing = self.kernel.get_smoothing
        h = self.hs
        if h is None:
            h = get_smoothing(self.dataset)
        h = np.atleast_1d(h)
        hsiz = h.shape
    
        if (min(hsiz) == 1) or (self.d == 1):
            if max(hsiz) == 1:
                h = h * np.ones(self.d)
            else:
                h.shape = (self.d,) # make sure it has the correct dimension
          
            # If h negative calculate automatic values
            ind, = np.where(h <= 0)
            for i in ind.tolist(): # 
                h[i] = get_smoothing(self.dataset[i])
            deth = h.prod()
            self.inv_hs = linalg.diag(1.0 / h)
        else: #fully general smoothing matrix
            deth = linalg.det(h)
            if deth <= 0:
                raise ValueError('bandwidth matrix h must be positive definit!')
            self.inv_hs = linalg.inv(h)
        self.hs = h
        self._norm_factor = deth * self.n
    
    def eval_grid(self, *args):
        grd = meshgrid(*args)
        shape0 = grd[0].shape
        d = len(grd)
        for i in range(d):
            grd[i] = grd[i].ravel()
        f = self.evaluate(np.vstack(grd))
        return f.reshape(shape0)
    
    def _check_shape(self, points):
        points = atleast_2d(points)
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)
        return points   
    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError if the dimensionality of the input points is different than
        the dimensionality of the KDE.
        """

        points = self._check_shape(points)
        d, m = points.shape
       
        result = np.zeros((m,))

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_hs / self._lambda[i], diff)
                result += self.kernel(tdiff) / self._lambda[i] ** d
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_hs, diff / self._lambda[np.newaxis, :])
                tmp = self.kernel(tdiff) / self._lambda ** d
                result[i] = tmp.sum(axis= -1)

        result /= (self._norm_factor * self.kernel.norm_factor(d, self.n))

        return result

    __call__ = evaluate

class KDEBIN(KDE):
    def __init__(self, dataset, hs=None, kernel=None, alpha=0.0, inc=128):
        KDE.__init__(self, dataset, hs, kernel, alpha)
        self.inc = inc
    def evaluate(self, *args):
        pass
class _Kernel(object):
    def __init__(self, r=1.0, stats=None):
        self.r = r # radius of kernel
        self.stats = stats
    def norm_factor(self, d=1, n=None):
        return 1.0
    def norm_kernel(self, x):
        X = np.atleast_2d(x)
        return self._kernel(X) / self.norm_factor(*X.shape)
    def kernel(self, x):
        return self._kernel(np.atleast_2d(x))
    def deriv4_6_8_10(self, t, numout=4):
        raise Exception('Method not implemented for this kernel!')
    __call__ = kernel 
    
class _KernelMulti(_Kernel):
    # p=0;  %Sphere = rect for 1D
    # p=1;  %Multivariate Epanechnikov kernel.
    # p=2;  %Multivariate Bi-weight Kernel
    # p=3;  %Multi variate Tri-weight Kernel 
    # p=4;  %Multi variate Four-weight Kernel
    def __init__(self, r=1.0, p=1, stats=None):
        self.r = r
        self.p = p
        self.stats = stats
    def norm_factor(self, d=1, n=None):
        r = self.r
        p = self.p
        c = 2 ** p * np.prod(np.r_[1:p + 1]) * sphere_volume(d, r) / np.prod(np.r_[(d + 2):(2 * p + d + 1):2])# normalizing constant
        return c
    def _kernel(self, x):
        r = self.r
        p = self.p
        x2 = x ** 2
        return ((1.0 - x2.sum(axis=0) / r ** 2).clip(min=0.0)) ** p

mkernel_epanechnikov = _KernelMulti(p=1, stats=_stats_epan)
mkernel_biweight = _KernelMulti(p=2, stats=_stats_biwe)
mkernel_triweight = _KernelMulti(p=3, stats=_stats_triw)

class _KernelProduct(_KernelMulti):
    # p=0;  %rectangular
    # p=1;  %1D product Epanechnikov kernel.
    # p=2;  %1D product Bi-weight Kernel
    # p=3;  %1D product Tri-weight Kernel 
    # p=4;  %1D product Four-weight Kernel
    
    def norm_factor(self, d=1, n=None):
        r = self.r
        p = self.p
        c = 2 ** p * np.prod(np.r_[1:p + 1]) * sphere_volume(1, r) / np.prod(np.r_[(1 + 2):(2 * p + 2):2])# normalizing constant
        return c ** d
    def _kernel(self, x):
        r = self.r # radius
        pdf = (1 - (x / r) ** 2).clip(min=0.0)
        return pdf.prod(axis=0)

mkernel_p1epanechnikov = _KernelProduct(p=1, stats=_stats_epan)
mkernel_p1biweight = _KernelProduct(p=2, stats=_stats_biwe)
mkernel_p1triweight = _KernelProduct(p=3, stats=_stats_triw)


class _KernelRectangular(_Kernel):
    def _kernel(self, x):
        return np.where(np.all(np.abs(x) <= self.r, axis=0), 1, 0.0)
    def norm_factor(self, d=1, n=None):
        r = self.r
        return (2 * r) ** d
mkernel_rectangular = _KernelRectangular(stats=_stats_rect)

class _KernelTriangular(_Kernel):
    def _kernel(self, x):
        pdf = (1 - np.abs(x)).clip(min=0.0)
        return pdf.prod(axis=0)
mkernel_triangular = _KernelTriangular(stats=_stats_tria)
    
class _KernelGaussian(_Kernel):
    def _kernel(self, x):
        x2 = x ** 2
        return exp(-0.5 * x2.sum(axis=0))       
    def norm_factor(self, d=1, n=None):
        return (2 * pi) ** (d / 2.0) 
    def deriv4_6_8_10(self, t, numout=4):
        '''
        Returns 4th, 6th, 8th and 10th derivatives of the kernel function.
        '''
        phi0 = exp(-0.5*t**2)/sqrt(2*pi)
        p4 = [1, 0, -6, 0, +3]
        p4val = np.polyval(p4,t)*phi0
        if numout==1:
            return p4val
        out = [p4val]
        pn = p4
        for ix in range(numout-1):
            pnp1 = np.polyadd(-np.r_[pn, 0], np.polyder(pn))
            pnp2 = np.polyadd(-np.r_[pnp1, 0], np.polyder(pnp1))
            out.append(np.polyval(pnp2, t)*phi0)
            pn = pnp2
        return out
    
mkernel_gaussian = _KernelGaussian(stats=_stats_gaus)

#def mkernel_gaussian(X):
#    x2 = X ** 2
#    d = X.shape[0]
#    return (2 * pi) ** (-d / 2) * exp(-0.5 * x2.sum(axis=0))       

class _KernelLaplace(_Kernel):
    def _kernel(self, x):
        absX = np.abs(x)    
        return exp(-absX.sum(axis=0))
    def norm_factor(self, d=1, n=None):
        return 2 ** d    
mkernel_laplace = _KernelLaplace(stats=_stats_lapl)

class _KernelLogistic(_Kernel):
    def _kernel(self, x):
        s = exp(x)
        return np.prod(s / (s + 1) ** 2, axis=0)
mkernel_logistic = _KernelLogistic(stats=_stats_logi)

_MKERNEL_DICT = dict(
                     epan=mkernel_epanechnikov,
                     biwe=mkernel_biweight,
                     triw=mkernel_triweight,
                     p1ep=mkernel_p1epanechnikov,
                     p1bi=mkernel_p1biweight,
                     p1tr=mkernel_p1triweight,
                     rect=mkernel_rectangular,
                     tria=mkernel_triangular,
                     lapl=mkernel_laplace,
                     logi=mkernel_logistic,
                     gaus=mkernel_gaussian
                     )
_KERNEL_EXPONENT_DICT = dict(re=0, sp=0, ep=1, bi=2, tr=3, fo=4, fi=5, si=6, se=7)

class Kernel(object):
    '''
    Multivariate kernel
    
    Parameters
    ----------
    name : string
        defining the kernel. Valid options are:
        'epanechnikov'  - Epanechnikov kernel. 
        'biweight'      - Bi-weight kernel.
        'triweight'     - Tri-weight kernel.
        'p1epanechnikov' - product of 1D Epanechnikov kernel. 
        'p1biweight'    - product of 1D Bi-weight kernel.
        'p1triweight'   - product of 1D Tri-weight kernel.
        'triangular'    - Triangular kernel.
        'gaussian'      - Gaussian kernel
        'rectangular'   - Rectangular kernel. 
        'laplace'       - Laplace kernel.
        'logistic'      - Logistic kernel.
    Note that only the first 4 letters of the kernel name is needed.
    
    Examples
    --------
     N = 20
    data = np.random.rayleigh(1, size=(N,))
    >>> data = array([ 0.75355792,  0.72779194,  0.94149169,  0.07841119,  2.32291887,
    ...        1.10419995,  0.77055114,  0.60288273,  1.36883635,  1.74754326,
    ...        1.09547561,  1.01671133,  0.73211143,  0.61891719,  0.75903487,
    ...        1.8919469 ,  0.72433808,  1.92973094,  0.44749838,  1.36508452])

    >>> Kernel('gaussian').stats()  
    (1, 0.28209479177387814, 0.21157109383040862)
    >>> Kernel('laplace').stats()
    (2, 0.25, inf)
    
    >>> triweight = Kernel('triweight'); triweight.stats()
    (0.1111111111111111, 0.81585081585081587, inf)
    
    >>> triweight(np.linspace(-1,1,11))
    array([ 0.      ,  0.046656,  0.262144,  0.592704,  0.884736,  1.      ,
            0.884736,  0.592704,  0.262144,  0.046656,  0.      ])
    >>> triweight.hns(data)
    array([ 0.82087056])
    >>> triweight.hos(data)
    array([ 0.88265652])
    >>> triweight.hste(data)
    array([ 0.56570278])
    
    See also
    --------
    mkernel
    
    References
    ---------- 
    B. W. Silverman (1986) 
    'Density estimation for statistics and data analysis'  
     Chapman and Hall, pp. 43, 76 
     
    Wand, M. P. and Jones, M. C. (1995) 
    'Density estimation for statistics and data analysis'  
     Chapman and Hall, pp 31, 103,  175  
    '''
    def __init__(self, name, fun='hns'):
        self.kernel = _MKERNEL_DICT[name[:4]]
        #self.name = self.kernel.__name__.replace('mkernel_', '').title()
        try:
            self.get_smoothing = getattr(self, fun) 
        except:
            self.get_smoothing = self.hns
        
    def stats(self):
        ''' Return some 1D statistics of the kernel.
      
        Returns
        ------- 
        mu2 : real scalar 
            2'nd order moment, i.e.,int(x^2*kernel(x))
        R : real scalar
            integral of squared kernel, i.e., int(kernel(x)^2)
        Rdd  : real scalar
            integral of squared double derivative of kernel, i.e., int( (kernel''(x))^2 ).
                  
        Reference
        --------- 
        Wand,M.P. and Jones, M.C. (1995) 
        'Kernel smoothing'
        Chapman and Hall, pp 176.
        '''  
        return self.kernel.stats
        #name = self.name[2:6] if self.name[:2].lower() == 'p1' else self.name[:4] 
        #return _KERNEL_STATS_DICT[name.lower()]
    def deriv4_6_8_10(self, t, numout=4):
        return self.kernel.deriv4_6_8_10(t, numout)
    
    def hns(self, data):
        '''
        Returns Normal Scale Estimate of Smoothing Parameter.
        
        Parameter
        ---------
        data : 2D array
            shape d x n (d = # dimensions )
        
        Returns
        -------
        h : array-like
            one dimensional optimal value for smoothing parameter
            given the data and kernel.  size D
         
        HNS only gives an optimal value with respect to mean integrated 
        square error, when the true underlying distribution 
        is Gaussian. This works reasonably well if the data resembles a
        Gaussian distribution. However if the distribution is asymmetric,
        multimodal or have long tails then HNS may  return a to large
        smoothing parameter, i.e., the KDE may be oversmoothed and mask
        important features of the data. (=> large bias).
        One way to remedy this is to reduce H by multiplying with a constant 
        factor, e.g., 0.85. Another is to try different values for H and make a 
        visual check by eye.
        
        Example: 
          data = rndnorm(0, 1,20,1)
          h = hns(data,'epan');
        
        See also:
        ---------  
        hste, hbcv, hboot, hos, hldpi, hlscv, hscv, hstt, kde
        
        Reference:  
        ---------
        B. W. Silverman (1986) 
        'Density estimation for statistics and data analysis'  
        Chapman and Hall, pp 43-48 
        Wand,M.P. and Jones, M.C. (1995) 
        'Kernel smoothing'
        Chapman and Hall, pp 60--63
        '''
        
        A = np.atleast_2d(data)
        n = A.shape[1]
        
        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, Rdd = self.stats()
        AMISEconstant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        iqr = np.abs(np.percentile(A, 75, axis=1) - np.percentile(A, 25, axis=1))# interquartile range
        stdA = np.std(A, axis=1, ddof=1)
        #  % use of interquartile range guards against outliers.
        #  % the use of interquartile range is better if 
        #  % the distribution is skew or have heavy tails
        #  % This lessen the chance of oversmoothing.
        return np.where(iqr > 0, np.minimum(stdA, iqr / 1.349), stdA) * AMISEconstant
    
    def hos(self, data):
        ''' Returns Oversmoothing Parameter.

        
        
           h      = one dimensional maximum smoothing value for smoothing parameter
                    given the data and kernel.  size 1 x D
           data   = data matrix, size N x D (D = # dimensions )
         
         The oversmoothing or maximal smoothing principle relies on the fact
         that there is a simple upper bound for the AMISE-optimal bandwidth for
         estimation of densities with a fixed value of a particular scale
         measure. While HOS will give too large bandwidth for optimal estimation 
         of a general density it provides an excellent starting point for
         subjective choice of bandwidth. A sensible strategy is to plot an
         estimate with bandwidth HOS and then sucessively look at plots based on 
         convenient fractions of HOS to see what features are present in the
         data for various amount of smoothing. The relation to HNS is given by:
         
                   HOS = HNS/0.93
        
          Example: 
          data = rndnorm(0, 1,20,1)
          h = hos(data,'epan');
          
         See also  hste, hbcv, hboot, hldpi, hlscv, hscv, hstt, kde, kdefun
        
         Reference
         --------- 
          B. W. Silverman (1986) 
         'Density estimation for statistics and data analysis'  
          Chapman and Hall, pp 43-48 
        
          Wand,M.P. and Jones, M.C. (1986) 
         'Kernel smoothing'
          Chapman and Hall, pp 60--63
        '''    
        return self.hns(data) / 0.93
    def hmns(self, data):
        '''
        Returns Multivariate Normal Scale Estimate of Smoothing Parameter.
        
         CALL:  h = hmns(data,kernel)
        
           h      = M dimensional optimal value for smoothing parameter
                    given the data and kernel.  size D x D
           data   = data matrix, size D x N (D = # dimensions )
           kernel = 'epanechnikov'  - Epanechnikov kernel.
                    'biweight'      - Bi-weight kernel.
                    'triweight'     - Tri-weight kernel.  
                    'gaussian'      - Gaussian kernel
          
          Note that only the first 4 letters of the kernel name is needed.
         
         HMNS  only gives  a optimal value with respect to mean integrated 
         square error, when the true underlying distribution  is
         Multivariate Gaussian. This works reasonably well if the data resembles a
         Multivariate Gaussian distribution. However if the distribution is 
         asymmetric, multimodal or have long tails then HNS is maybe more 
         appropriate.
        
          Example: 
            data = rndnorm(0, 1,20,2)
            h = hmns(data,'epan');
          
         See also 
         --------
          
        hns, hste, hbcv, hboot, hos, hldpi, hlscv, hscv, hstt
        
         Reference
         ----------  
          B. W. Silverman (1986) 
         'Density estimation for statistics and data analysis'  
          Chapman and Hall, pp 43-48, 87 
        
          Wand,M.P. and Jones, M.C. (1995) 
         'Kernel smoothing'
          Chapman and Hall, pp 60--63, 86--88
        '''
        # TODO: implement more kernels  
          
        A = np.atleast_2d(data)
        d, n = A.shape
        
        if d == 1:
            return self.hns(data)
        name = self.name[:4].lower()
        if name == 'epan':        # Epanechnikov kernel
            a = (8.0 * (d + 4.0) * (2 * sqrt(pi)) ** d / sphere_volume(d)) ** (1. / (4.0 + d))
        elif name == 'biwe': # Bi-weight kernel
            a = 2.7779;
            if d > 2:
                raise ValueError('not implemented for d>2')
        elif name == 'triw': # Triweight
            a = 3.12;
            if d > 2:
                raise ValueError('not implemented for d>2')
        elif name == 'gaus': # Gaussian kernel
            a = (4.0 / (d + 2.0)) ** (1. / (d + 4.0))
        else:
            raise ValueError('Unknown kernel.')
         
        covA = scipy.cov(A)
        
        return a * linalg.sqrtm(covA) * n * (-1. / (d + 4))
    def hste(self, data, h0=None, inc=128, maxit=100, releps=0.01, abseps=0.0):
        '''HSTE 2-Stage Solve the Equation estimate of smoothing parameter.
        
         CALL:  hs = hste(data,kernel,h0)
         
               hs = one dimensional value for smoothing parameter
                    given the data and kernel.  size 1 x D
           data   = data matrix, size N x D (D = # dimensions )
           kernel = 'gaussian'  - Gaussian kernel (default)
                     ( currently the only supported kernel)
               h0 = initial starting guess for hs (default h0=hns(A,kernel))
        
          Example: 
           x  = rndnorm(0,1,50,1);
           hs = hste(x,'gauss');
        
         See also  hbcv, hboot, hos, hldpi, hlscv, hscv, hstt, kde, kdefun
        
         Reference:  
          B. W. Silverman (1986) 
         'Density estimation for statistics and data analysis'  
          Chapman and Hall, pp 57--61
        
          Wand,M.P. and Jones, M.C. (1986) 
         'Kernel smoothing'
          Chapman and Hall, pp 74--75
        '''  
        # TODO: NB: this routine can be made faster:
        # TODO: replace the iteration in the end with a Newton Raphson scheme
        
        A = np.atleast_2d(data)
        d, n= A.shape
        
        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, Rdd = self.stats()
        
        AMISEconstant = (8 * sqrt(pi) * R / (3 * mu2 ** 2 * n)) ** (1. / 5)
        STEconstant = R /(mu2**(2)*n)
        
        sigmaA = self.hns(A)/AMISEconstant
        if h0 is None:
            h0 = sigmaA*AMISEconstant
        
        h = np.asarray(h0, dtype=float)
       
        nfft = inc*2 
        amin   = A.min(axis=1) # Find the minimum value of A.
        amax   = A.max(axis=1) #Find the maximum value of A.
        arange = amax-amin # Find the range of A.
        
        #% xa holds the x 'axis' vector, defining a grid of x values where 
        #% the k.d. function will be evaluated.
        
        ax1 = amin-arange/8.0
        bx1 = amax+arange/8.0
        
        kernel2 = Kernel('gaus') 
        mu2,R,Rdd = kernel2.stats()
        STEconstant2 = R /(mu2**(2)*n)
        fft = np.fft.fft
        ifft = np.fft.ifft
        
        for dim in range(d):
            s = sigmaA[dim]
            ax = ax1[dim]
            bx = bx1[dim]
          
            xa = np.linspace(ax,bx,inc) 
            xn = np.linspace(0,bx-ax,inc)
          
            c = gridcount(A[dim],xa)
       
        
            # Step 1
            psi6NS = -15/(16*sqrt(pi)*s**7)
            psi8NS = 105/(32*sqrt(pi)*s**9)
        
            # Step 2
            k40, k60 = kernel2.deriv4_6_8_10(0, numout=2)
            g1 = (-2*k40/(mu2*psi6NS*n))**(1.0/7)
            g2 = (-2*k60/(mu2*psi8NS*n))**(1.0/9)
        
            # Estimate psi6 given g2.
            kw4, kw6 = kernel2.deriv4_6_8_10(xn/g2, numout=2) # kernel weights.
            kw = np.r_[kw6,0,kw6[-1:0:-1]]             # Apply fftshift to kw.
            z = np.real(ifft(fft(c,nfft)*fft(kw)))     # convolution.
            psi6 = np.sum(c*z[:inc])/(n*(n-1)*g2**7)
        
            # Estimate psi4 given g1.
            kw4  = kernel2.deriv4_6_8_10(xn/g1, numout=1) # kernel weights.
            kw   = np.r_[kw4,0,kw4[-1:0:-1]]  #Apply 'fftshift' to kw.
            z    = np.real(ifft(fft(c,nfft)*fft(kw))) # convolution.
            psi4 = np.sum(c*z[:inc])/(n*(n-1)*g1**5)
        
            
            
            h1    = h[dim]
            h_old = 0
            count = 0
          
            while ((abs(h_old-h1)>max(releps*h1,abseps)) and (count < maxit)):
                count += 1
                h_old = h1
          
                # Step 3
                gamma=((2*k40*mu2*psi4*h1**5)/(-psi6*R))**(1.0/7)
        
                # Now estimate psi4 given gamma.
                kw4 = kernel2.deriv4_6_8_10(xn/gamma, numout=1) #kernel weights. 
                kw  = np.r_[kw4,0,kw4[-1:0:-1]] # Apply 'fftshift' to kw.
                z   = np.real(ifft(fft(c,nfft)*fft(kw))) # convolution.
        
                psi4Gamma  = np.sum(c*z[:inc])/(n*(n-1)*gamma**5)
          
                # Step 4
                h1 = (STEconstant2/psi4Gamma)**(1.0/5)
            
            # Kernel other than Gaussian scale bandwidth
            h1  = h1*(STEconstant/STEconstant2)**(1.0/5)
          
        
            if count>= maxit:
                warnings.warn('The obtained value did not converge.')
          
            h[dim] = h1
        #end % for dim loop
        return h
    
    def norm_factor(self, d=1, n=None):
        return  self.kernel.norm_factor(d, n)    
    def evaluate(self, X):
        return self.kernel(np.atleast_2d(X))
    __call__ = evaluate
    
def mkernel(X, kernel):
    '''
    MKERNEL Multivariate Kernel Function.
     
    Paramaters
    ---------  
    X : array-like  
        matrix  size d x n (d = # dimensions, n = # evaluation points)
    kernel : string
        defining kernel
        'epanechnikov'  - Epanechnikov kernel. 
        'biweight'      - Bi-weight kernel.
        'triweight'     - Tri-weight kernel.
        'p1epanechnikov' - product of 1D Epanechnikov kernel. 
        'p1biweight'    - product of 1D Bi-weight kernel.
        'p1triweight'   - product of 1D Tri-weight kernel.
        'triangular'    - Triangular kernel.
        'gaussian'      - Gaussian kernel
        'rectangular'   - Rectangular kernel. 
        'laplace'       - Laplace kernel.
        'logistic'      - Logistic kernel.
    Note that only the first 4 letters of the kernel name is needed.  
    Returns
    -------         
    z : ndarray
        kernel function values evaluated at X
      
    
    See also  
    --------
    kde, kdefun, kdebin
     
    References
    ---------- 
    B. W. Silverman (1986) 
    'Density estimation for statistics and data analysis'  
     Chapman and Hall, pp. 43, 76 
     
    Wand, M. P. and Jones, M. C. (1995) 
    'Density estimation for statistics and data analysis'  
     Chapman and Hall, pp 31, 103,  175  
    '''
    fun = _MKERNEL_DICT[kernel[:4]]
    return fun(np.atleast_2d(X))


def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    """
    An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1) 
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array. 
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    >>> s
    array([ 9,  7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
    ...        [[0,0],[0,0],[0,1]],
    ...        [[0,0],[0,0],[0,1]],
    ...        [[1,0],[1,0],[1,1]]])
    >>> # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)
    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out

def bitget(int_type, offset):
    '''
    Returns the value of the bit at the offset position in int_type.
    
    Example
    -------
    >>> bitget(5, np.r_[0:4])
    array([1, 0, 1, 0])
    '''
    mask = (1 << offset)
    return (int_type & mask) != 0

def gridcount(data, X):
    '''
    Returns D-dimensional histogram using linear binning.
      
    Parameters
    ----------
    data = column vectors with D-dimensional data, size D x Nd 
    X    = row vectors defining discretization, size D x N
            Must include the range of the data.
    
    Returns
    -------
    c    = gridcount,  size N x N x ... x N
             
    GRIDCOUNT obtains the grid counts using linear binning.
    There are 2 strategies: simple- or linear- binning.
    Suppose that an observation occurs at x and that the nearest point
    below and above is y and z, respectively. Then simple binning strategy
    assigns a unit weight to either y or z, whichever is closer. Linear
    binning, on the other hand, assigns the grid point at y with the weight
    of (z-x)/(z-y) and the gridpoint at z a weight of (y-x)/(z-y).
      
    In terms of approximation error of using gridcounts as pdf-estimate,
    linear binning is significantly more accurate than simple binning.  
    
     NOTE: The interval [min(X);max(X)] must include the range of the data.
           The order of C is permuted in the same order as 
           meshgrid for D==2 or D==3.  
       
    Example
    -------
    >>> import numpy as np
    >>> import wafo.kdetools as wk
    >>> import pylab as plb
    >>> N     = 20;
    >>> data  = np.random.rayleigh(1,N)
    >>> x = np.linspace(0,max(data)+1,50)  
    >>> dx = x[1]-x[0]  
    
    >>> c = wk.gridcount(data,x)
    
    >>> h = plb.plot(x,c,'.')   # 1D histogram
    >>> pdf = c/dx/N
    >>> h1 = plb.plot(x, pdf) #  1D probability density plot
    >>> np.trapz(pdf, x)   
    0.99999999999999956
    
    See also
    --------
    bincount, accum, kdebin
      
    Reference
    ----------
    Wand,M.P. and Jones, M.C. (1995) 
    'Kernel smoothing'
    Chapman and Hall, pp 182-192  
    '''  
    dat = np.atleast_2d(data)
    x = np.atleast_2d(X)
    d, n = dat.shape
    d1, inc = x.shape
    
    if d != d1:
        raise ValueError('Dimension 0 of data and X do not match.')
    
    dx = np.diff(x[:, :2], axis=1)
    xlo = x[:, 0]
    xup = x[:, -1]
    
    datlo = dat.min(axis=1)
    datup = dat.max(axis=1)
    if ((datlo < xlo) | (xup < datup)).any():
        raise ValueError('X does not include whole range of the data!')
    
    csiz = np.repeat(inc, d)
    
      
    binx = np.asarray(np.floor((dat - xlo[:, newaxis]) / dx), dtype=int)
    w = dx.prod()
    abs = np.abs
    if  d == 1:
        x.shape = (-1,)
        c = (accum(binx, (x[binx + 1] - dat), size=[inc, ]) + 
             accum(binx, (dat - x[binx]), size=[inc, ])) / w
    elif d == 2:
        b2 = binx[1]
        b1 = binx[0]
        c_ = np.c_
        stk = np.vstack
        c = (accum(c_[b1, b2] , abs(np.prod(stk([X[0, b1 + 1], X[1, b2 + 1]]) - dat, axis=0)), size=[inc, inc]) + 
          accum(c_[b1 + 1, b2]  , abs(np.prod(stk([X[0, b1], X[1, b2 + 1]]) - dat, axis=0)), size=[inc, inc]) + 
          accum(c_[b1  , b2 + 1], abs(np.prod(stk([X[0, b1 + 1], X[1, b2]]) - dat, axis=0)), size=[inc, inc]) + 
          accum(c_[b1 + 1, b2 + 1], abs(np.prod(stk([X[0, b1], X[1, b2]]) - dat, axis=0)), size=[inc, inc])) / w
      
    else: # % d>2
       
        Nc = csiz.prod()
        c = np.zeros((Nc,))
     
        fact2 = np.asarray(np.reshape(inc * np.arange(d), (d, -1)), dtype=int)
        fact1 = np.asarray(np.reshape(csiz.cumprod() / inc, (d, -1)), dtype=int)
        #fact1 = fact1(ones(n,1),:);
        bt0 = [0, 0]
        X1 = X.ravel()
        for ir in xrange(2 ** (d - 1)):
            bt0[0] = np.reshape(bitget(ir, np.arange(d)), (d, -1))
            bt0[1] = 1 - bt0[0]
            for ix in xrange(2):
                one = np.mod(ix, 2)
                two = np.mod(ix + 1, 2)
                # Convert to linear index 
                b1 = np.sum((binx + bt0[one]) * fact1, axis=0) #linear index to c
                bt2 = bt0[two] + fact2
                b2 = binx + bt2                     # linear index to X
                c += accum(b1, abs(np.prod(X1[b2] - dat, axis=0)), size=(Nc,))
                
        c = np.reshape(c / w, csiz, order='C')
        # TODO: check that the flipping of axis is correct
        T = range(d); T[-2], T[-1] = T[-1], T[-2]
        c = c.transpose(*T)

    if d == 2: # make sure c is stored in the same way as meshgrid
        c = c.T
    elif d == 3:
        c = c.transpose(1, 0, 2)
    
    return c

    
def main():
    import doctest
    doctest.testmod()
    
if __name__ == '__main__':
    main()
    
