#-------------------------------------------------------------------------------
# Name:        kdetools
# Purpose:
#
# Author:      pab
#
# Created:     01.11.2008
# Copyright:   (c) pab2 2008
# Licence:     LGPL
#-------------------------------------------------------------------------------
#!/usr/bin/env python
from __future__ import division
import numpy as np
from numpy import pi, sqrt, atleast_2d, exp, newaxis #@UnresolvedImport
import scipy
from scipy.linalg import sqrtm
from scipy.special import gamma
from misc import tranproc, trangood
from itertools import product


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



class kde(object):
    """ Representation of a kernel-density estimate using Gaussian kernels.

    Parameters
    ----------
    dataset : (# of dims, # of data)-array
        datapoints to estimate from

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
    kde.integrate_gaussian(mean, cov) : float
        multiply pdf with a specified Gaussian and integrate over the whole domain
    kde.integrate_box_1d(low, high) : float
        integrate pdf (1D only) between two bounds
    kde.integrate_box(low_bounds, high_bounds) : float
        integrate pdf over a rectangular space between low_bounds and high_bounds
    kde.integrate_kde(other_kde) : float
        integrate two kernel density estimates multiplied together

   Internal Methods
   ----------------
    kde.covariance_factor() : float
        computes the coefficient that multiplies the data covariance matrix to
        obtain the kernel covariance matrix. Set this method to
        kde.scotts_factor or kde.silverman_factor (or subclass to provide your
        own). The default is scotts_factor.
    """

    def __init__(self, dataset, **kwds):
        self.kernel = 'gauss'
        self.hs = None
        self.hsmethod = None
        self.L2 = None
        self.alpha = 0
        self.__dict__.update(kwds)

        self.dataset = atleast_2d(dataset)
        self.d, self.n = self.dataset.shape


        self._compute_covariance()


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

        points = atleast_2d(points).astype(self.dataset.dtype)

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

        result = np.zeros((m,), points.dtype)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result += np.exp(-energy)
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_cov, diff)
                energy = sum(diff * tdiff, axis=0) / 2.0
                result[i] = np.sum(np.exp(-energy), axis=0)

        result /= self._norm_factor

        return result

    __call__ = evaluate

#function [f, hs,lambda]= kdefun(A,options,varargin)
#%KDEFUN  Kernel Density Estimator.
#%
#% CALL:  [f, hs] = kdefun(data,options,x1,x2,...,xd)
#%
#%   f      = kernel density estimate evaluated at x1,x2,...,xd.
#%   data   = data matrix, size N x D (D = # dimensions)
#%  options = kdeoptions-structure or cellvector of named parameters with
#%            corresponding values, see kdeoptset for details.
#%   x1,x2..= vectors/matrices defining the points to evaluate the density
#%
#%  KDEFUN gives a slow, but exact kernel density estimate evaluated at x1,x2,...,xd.
#%  Notice that densities close to normality appear to be the easiest for the kernel
#%  estimator to estimate and that the degree of estimation difficulty increases with
#%  skewness, kurtosis and multimodality.
#%
#%  If D > 1 KDE calculates quantile levels by integration. An
#%  alternative is to calculate them by ranking the kernel density
#%  estimate obtained at the points DATA  i.e. use the commands
#%
#%      f    = kde(data);
#%      r    = kdefun(data,[],num2cell(data,1));
#%      f.cl = qlevels2(r,f.PL);
#%
#%  The first is probably best when estimating the pdf and the latter is the
#%  easiest and most robust for multidimensional data when only a visualization
#%  of the data is needed.
#%
#%  For faster estimates try kdebin.
#%
#% Examples:
#%     data = rndray(1,500,1);
#%     x = linspace(sqrt(eps),5,55);
#%     plotnorm((data).^(.5)) % gives a straight line => L2 = 0.5 reasonable
#%     f = kdefun(data,{'L2',.5},x);
#%     plot(x,f,x,pdfray(x,1),'r')
#%
#% See also  kde, mkernel, kdebin
#
#% Reference:
#%  B. W. Silverman (1986)
#% 'Density estimation for statistics and data analysis'
#%  Chapman and Hall , pp 100-110
#%
#%  Wand, M.P. and Jones, M.C. (1995)
#% 'Kernel smoothing'
#%  Chapman and Hall, pp 43--45
#
#
#
#
#%Tested on: matlab 5.2
#% History:
#% revised pab Feb2004
#%  -options moved into a structure
#% revised pab Dec2003
#% -removed some code
#% revised pab 27.04.2001
#% - changed call from mkernel to mkernel2 (increased speed by 10%)
#% revised pab 01.01.2001
#% - added the possibility that L2 is a cellarray of parametric
#%   or non-parametric transformations (secret option)
#% revised pab 14.12.1999
#%  - fixed a small error in example in help header
#% revised pab 28.10.1999
#%  - added L2
#% revised pab 21.10.99
#%  - added alpha to input arguments
#%  - made it fully general for d dimensions
#%  - HS may be a smoothing matrix
#% revised pab 21.09.99
#%  - adapted from kdetools by Christian Beardah
#
#  defaultoptions = kdeoptset;
#% If just 'defaults' passed in, return the default options in g
#if ((nargin==1) && (nargout <= 1) &&  isequal(A,'defaults')),
#  f = defaultoptions;
#  return
#end
#error(nargchk(1,inf, nargin))
#
#[n, d]=size(A); % Find dimensions of A,
#               % n=number of data points,
#               % d=dimension of the data.
#if (nargin<2 || isempty(options))
#  options  = defaultoptions;
#else
#  switch lower(class(options))
#   case {'char','struct'},
#    options = kdeoptset(defaultoptions,options);
#   case {'cell'}
#
#      options = kdeoptset(defaultoptions,options{:});
#   otherwise
#    error('Invalid options')
#  end
#end
#kernel   = options.kernel;
#h        = options.hs;
#alpha    = options.alpha;
#L2       = options.L2;
#hsMethod = options.hsMethod;
#
#if isempty(h)
#  h=zeros(1,d);
#end
#
#L22 = cell(1,d);
#k3=[];
#if isempty(L2)
#  L2=ones(1,d); % default no transformation
#elseif iscell(L2)   % cellarray of non-parametric and parametric transformations
#  Nl2 = length(L2);
#  if ~(Nl2==1||Nl2==d), error('Wrong size of L2'), end
#  [L22{1:d}] = deal(L2{1:min(Nl2,d)});
#  L2 = ones(1,d); % default no transformation
#  for ix=1:d,
#    if length(L22{ix})>1,
#      k3=[k3 ix];       % Non-parametric transformation
#    else
#     L2(ix) = L22{ix};  % Parameter to the Box-Cox transformation
#    end
#  end
#elseif length(L2)==1
#  L2=L2(:,ones(1,d));
#end
#
#amin=min(A);
#if any((amin(L2~=1)<=0))  ,
#  error('DATA cannot be negative or zero when L2~=1')
#end
#
#
#nv=length(varargin);
#if nv<d,
#  error('some or all of the evaluation points x1,x2,...,xd is missing')
#end
#
#xsiz = size(varargin{1}); % remember size of input
#Nx   = prod(xsiz);
#X    = zeros(Nx,d);
#for ix=1:min(nv,d),
#  if (any(varargin{ix}(:)<=0) && (L2(ix)~=1)),
#    error('xi cannot be negative or zero when L2~=1')
#  end
#  X(:,ix)=varargin{ix}(:); % make sure it is a column vector
#end
#
#
#%new call
#lX = X; %zeros(Nx,d);
#lA = A; %zeros(size(A));
#
#k1 = find(L2==0); % logaritmic transformation
#if any(k1)
#  lA(:,k1)=log(A(:,k1));
#  lX(:,k1)=log(X(:,k1));
#end
#k2=find(L2~=0 & L2~=1); % power transformation
#if any(k2)
#  lA(:,k2)=sign(L2(ones(n,1),k2)).*A(:,k2).^L2(ones(n,1),k2);
#  lX(:,k2)=sign(L2(ones(Nx,1),k2)).*X(:,k2).^L2(ones(Nx,1),k2);
#end
#% Non-parametric transformation
#for ix = k3,
#  lA(:,ix) = tranproc(A(:,ix),L22{ix});
#  lX(:,ix) = tranproc(X(:,ix),L22{ix});
#end
#
#
#hsiz=size(h);
#if (min(hsiz)==1)||(d==1)
#  if max(hsiz)==1,
#    h=h*ones(1,d);
#  else
#    h=reshape(h,[1,d]); % make sure it has the correct dimension
#  end;
#  ind=find(h<=0);
#  if any(ind)    % If no value of h has been specified by the user then
#    h(ind)=feval(hsMethod,lA(:,ind),kernel); % calculate automatic values.
#  end
#  deth = prod(h);
#else  % fully general smoothing matrix
#  deth = det(h);
#  if deth<=0
#    error('bandwidth matrix h must be positive definit')
#  end
#end
#
#if alpha>0
#  Xn   = num2cell(lA,1);
#  opt1 = kdeoptset('kernel',kernel,'hs',h,'alpha',0,'L2',1);
#  f2   = kdefun(lA,opt1,Xn{:}); % get a pilot estimate by regular KDE (alpha=0)
#  g    = exp(sum(log(f2))/n);
#
#  lambda=(f2(:)/g).^(-alpha);
#else
#  lambda=ones(n,1);
#end
#
#
#
#
#
#f=zeros(Nx,1);
#if (min(hsiz)==1)||(d==1)
#  for ix=1:n,     % Sum over all data points
#    Avec=lA(ix,:);
#    Xnn=(lX-Avec(ones(Nx,1),:))./(h(ones(Nx,1),:) *lambda(ix));
#    f = f + mkernel2(Xnn,kernel)/lambda(ix)^d;
#  end
#else % fully general
#  h1=inv(h);
#  for ix=1:n,     % Sum over all data points
#    Avec=lA(ix,:);
#    Xnn=(lX-Avec(ones(Nx,1),:))*(h1/lambda(ix));
#    f = f + mkernel2(Xnn,kernel)/lambda(ix)^d;
#  end
#end
#f=f/(n*deth);
#
#% transforming back
#if any(k1), % L2=0 i.e. logaritmic transformation
#  for ix=k1
#    f=f./X(:,ix);
#  end
#  if any(max(abs(diff(f)))>10)
#    disp('Warning: Numerical problems may have occured due to the logaritmic')
#    disp('transformation. Check the KDE for spurious spikes')
#  end
#end
#if any(k2) % L2~=0 i.e. power transformation
#  for ix=k2
#    f=f.*(X(:,ix).^(L2(ix)-1))*L2(ix)*sign(L2(ix));
#  end
#  if any(max(abs(diff(f)))>10)
#    disp('Warning: Numerical problems may have occured due to the power')
#    disp('transformation. Check the KDE for spurious spikes')
#  end
#end
#if any(k3), % non-parametric transformation
#  oneC = ones(Nx,1);
#  for ix=k3
#    gn  = L22{ix};
#    %Gn  = fliplr(L22{ix});
#    %x0  = tranproc(lX(:,ix),Gn);
#    if any(isnan(X(:,ix))),
#      error('The transformation does not have a strictly positive derivative.')
#    end
#    hg1  = tranproc([X(:,ix) oneC],gn);
#    der1 = abs(hg1(:,2)); % dg(X)/dX = 1/(dG(Y)/dY)
#    % alternative 2
#    %pp  = smooth(Gn(:,1),Gn(:,2),1,[],1);
#    %dpp = diffpp(pp);
#    %der1 = 1./abs(ppval(dpp,f.x{ix}));
#    % Alternative 3
#    %pp  = smooth(gn(:,1),gn(:,2),1,[],1);
#    %dpp = diffpp(pp);
#    %%plot(hg1(:,1),der1-abs(ppval(dpp,x0)))
#    %der1 = abs(ppval(dpp,x0));
#    if any(der1<=0),
#      error('The transformation must have a strictly positive derivative')
#    end
#    f = f.*der1;
#  end
#  if any(max(abs(diff(f)))>10)
#    disp('Warning: Numerical problems may have occured due to the power')
#    disp('transformation. Check the KDE for spurious spikes')
#  end
#end
#
#f=reshape(f,xsiz); % restore original shape
#if nargout>1
#  hs=h;
#end


def mkernel_multi(X, p=None):
    # p=0;  %Sphere = rect for 1D
    # p=1;  %Multivariate Epanechnikov kernel.
    # p=2;  %Multivariate Bi-weight Kernel
    # p=3;  %Multi variate Tri-weight Kernel 
    # p=4;  %Multi variate Four-weight Kernel
    d, n = X.shape
    r = 1    # radius of the kernel
    c = 2 ** p * np.prod(np.r_[1:p + 1]) * sphere_volume(d, r) / np.prod(np.r_[(d + 2):(2 * p + d + 1):2])# normalizing constant
    # c = beta(r+1,r+1)*vsph(d,b)*(2^(2*r)); % Wand and Jones pp 31
    # the commented c above does note yield the right scaling for d>1   
    x2 = X ** 2
    return ((1.0 - x2.sum(axis=0) / r ** 2).clip(min=0.0)) ** p / c

def mkernel_epanechnikov(X):
    return mkernel_multi(X, p=1)
def mkernel_biweight(X):
    return mkernel_multi(X, p=2)
def mkernel_triweight(X):
    return mkernel_multi(X, p=3)
 
def mkernel_product(X, p):
    # p=0;  %rectangular
    # p=1;  %1D product Epanechnikov kernel.
    # p=2;  %1D product Bi-weight Kernel
    # p=3;  %1D product Tri-weight Kernel 
    # p=4;  %1D product Four-weight Kernel
    d, n = X.shape  
    r = 1.0 # radius
    pdf = (1 - (X / r) ** 2).clip(min=0.0)
    c = 2 ** p * np.prod(np.r_[1:p + 1]) * sphere_volume(1, r) / np.prod(np.r_[(1 + 2):(2 * p + 2):2])# normalizing constant
    return np.prod(pdf, axis=0) / c ** d

def mkernel_p1epanechnikov(X):
    return mkernel_product(X, p=1)
def mkernel_p1biweight(X):
    return mkernel_product(X, p=2)
def mkernel_p1triweight(X):
    return mkernel_product(X, p=3)   


def mkernel_rectangular(X):
    d, n = X.shape
    return np.where(np.all(np.abs(X) <= 1, axis=0), 0.5 ** d, 0.0)
    
def mkernel_triangular(X):
    pdf = (1 - np.abs(X)).clip(min=0.0)
    return np.prod(pdf, axis=0)
       
def mkernel_gaussian(X):
    x2 = X ** 2
    d, n = X.shape
    return (2 * pi) ** (-d / 2) * exp(-0.5 * x2.sum(axis=0))       
    
def mkernel_laplace(X):
    ''' Return multivariate laplace kernel'''
    d, n = X.shape
    absX = np.abs(X)
    return 0.5 ** d * exp(-absX.sum(axis=0))

def mkernel_logistic(X):
    ''' Return multivariate logistic kernel'''
    s = exp(X)
    return np.prod(s / (s + 1) ** 2, axis=0)    

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
_KERNEL_STATS_DICT = dict(epan=(1. / 5, 3. / 5, np.inf),
                          biwe=(1. / 7, 5. / 7, 45. / 2),
                          triw=(1. / 9, 350. / 429, np.inf),
                          rect=(1. / 3, 1. / 2, np.inf),
                          tria=(1. / 6, 2. / 3, np.inf),
                          lapl=(2, 1. / 4, np.inf),
                          logi=(pi ** 2 / 3, 1. / 6, 1 / 42),
                          gaus=(1, 1. / (2 * sqrt(pi)), 3. / (8 * sqrt(pi)))
                          )


  
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
    >>> Kernel('gaussian').stats()  
    (1, 0.28209479177387814, 0.21157109383040862)
    >>> Kernel('laplace').stats()
    (2, 0.25, inf)
    
    >>> triweight = Kernel('triweight'); triweight.stats()
    (0.1111111111111111, 0.81585081585081587, inf)
    
    >>> triweight(np.linspace(-1,1,11))
    array([ 0.     ,  0.05103,  0.28672,  0.64827,  0.96768,  1.09375,
            0.96768,  0.64827,  0.28672,  0.05103,  0.     ])
    >>> triweight.hns(np.random.normal(size=100))
    
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
    def __init__(self, name):
        self.kernel = _MKERNEL_DICT[name[:4]]
        self.name = self.kernel.__name__.replace('mkernel_','').title()
        
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
        name = self.name[2:6] if self.name[:2].lower()=='p1' else self.name[:4] 
        return _KERNEL_STATS_DICT[name.lower()]
    
    def hns(self, data):
        '''
        HNS Normal Scale Estimate of Smoothing Parameter.
        
         CALL:  h = hns(data,kernel)
        
           h      = one dimensional optimal value for smoothing parameter
                    given the data and kernel.  size 1 x D
           data   = data matrix, size N x D (D = # dimensions )
         
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
          
         See also  hste, hbcv, hboot, hos, hldpi, hlscv, hscv, hstt, kde
        
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
        d,n = A.shape
        
        # R= int(mkernel(x)^2),  mu2= int(x^2*mkernel(x))
        mu2, R, Rdd = self.stats()
        AMISEconstant = (8*sqrt(pi)*R/(3*mu2**2*n))**(1./5)
        iqr = np.abs(np.percentile(A, 75, axis=1)-np.percentile(A, 25, axis=1))# interquartile range
        stdA = np.std(A, axis=1)
        #  % use of interquartile range guards against outliers.
        #  % the use of interquartile range is better if 
        #  % the distribution is skew or have heavy tails
        #  % This lessen the chance of oversmoothing.
        return np.where(iqr>0, np.minimum(stdA, iqr/1.349), stdA)*AMISEconstant
    def hos(self, data):
        ''' Return Oversmoothing Parameter.

        
        
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
        
         Reference:  
          B. W. Silverman (1986) 
         'Density estimation for statistics and data analysis'  
          Chapman and Hall, pp 43-48 
        
          Wand,M.P. and Jones, M.C. (1986) 
         'Kernel smoothing'
          Chapman and Hall, pp 60--63
        '''    
        return self.hns(data)/0.93
    def hmns(self, data):
        '''
        Returns Multivariate Normal Scale Estimate of Smoothing Parameter.
        
         CALL:  h = hmns(data,kernel)
        
           h      = M dimensional optimal value for smoothing parameter
                    given the data and kernel.  size D x D
           data   = data matrix, size N x D (D = # dimensions )
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
          
         See also  hns, hste, hbcv, hboot, hos, hldpi, hlscv, hscv, hstt
         Reference:  
          B. W. Silverman (1986) 
         'Density estimation for statistics and data analysis'  
          Chapman and Hall, pp 43-48, 87 
        
          Wand,M.P. and Jones, M.C. (1995) 
         'Kernel smoothing'
          Chapman and Hall, pp 60--63, 86--88
        '''
        # TODO implement more kernels  
          
        A = np.atleast_2d(data)
        d,n = A.shape
        
        if d==1:
            return self.hns(data)
        name = self.name[:4].lower()
        if name=='epan':        # Epanechnikov kernel
            a=(8*(d+4)*(2*sqrt(pi))**d/sphere_volume(d))**(1./(4+d))
        elif name=='biwe': # Bi-weight kernel
            a=2.7779;
            if d>2:
                raise ValueError('not implemented for d>2')
        elif name=='triw': # Triweight
            a=3.12;
            if d>2:
                raise ValueError('not implemented for d>2')
        elif name=='gaus': # Gaussian kernel
            a = (4/(d+2))**(1./(d+4))
        else:
            raise ValueError('Unknown kernel.')
         
        covA = scipy.cov(A)
        
        return a*sqrtm(covA)*n*(-1./(d+4))
        
        


        
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
    array([9, 7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
            [[0,0],[0,0],[0,1]],
            [[0,0],[0,0],[0,1]],
            [[1,0],[1,0],[1,1]],
        ])
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
def gridcount(data,X):
    '''
    GRIDCOUNT D-dimensional histogram using linear binning.
      
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
      N     = 500;
      data  = rndray(1,N,1);
      x = linspace(0,max(data)+1,50).';  
      dx = x(2)-x(1);  
      c = gridcount(data,x);
      plot(x,c,'.')    1D histogram
      plot(x,c/dx/N)   1D probability density plot
      trapz(x,c/dx/N)   
    
     See also  bincount, kdebin
      
    Reference
    ----------
    Wand,M.P. and Jones, M.C. (1995) 
    'Kernel smoothing'
    Chapman and Hall, pp 182-192  
    '''  
    dat = np.atleast_2d(data)
    x = np.atleast_2d(X)
    d,n = dat.shape
    d1, inc = x.shape
    
    if d!=d1:
        raise ValueError('Dimension 0 of data and X do not match.')
    
    
    dx  = np.diff(x[:,:2],axis=1)
    xlo = x[:,0]
    xup = x[:,-1]
    
    datlo = dat.min(axis=1)
    datup = dat.max(axis=1)
    if ((datlo<xlo) or (xup<datup)).any():
        raise ValueError('X does not include whole range of the data!')
    
    csiz = np.repeat(inc, d)
    
      
    binx = np.assarray(np.floor((dat-xlo[:, newaxis])/dx[:, newaxis]),dtype=int)
    w  = dx.prod()
    if  d==1:
        c = (accum(binx,(x[binx+1]-data),size=[inc,]) +
             accum(binx,(data-x[binx]),size=[inc,]))/w
    elif d==2:
        b2 = binx[1]
        b1 = binx[0]
        c = (accum([b1,b2] ,abs(prod(([X(b1+1,1) X(b2+1,2)]-data),2)),size=[inc,inc])+...
          accum([b1+1,b2]  ,abs(prod(([X(b1,1) X(b2+1,2)]-data),2)),size=[inc,inc])+...
          accum([b1  ,b2+1],abs(prod(([X(b1+1,1) X(b2,2)]-data),2)),size=[inc,inc])+...
          accum([b1+1,b2+1],abs(prod(([X(b1,1) X(b2,2)]-data),2)),size=[inc,inc]))/w;
     
      
    else: # % d>2
     useSparse = 0;
      Nc = prod(csiz);
      c  = zeros(Nc,1);
     
      fact2 = [0 inc*(1:d-1)];
      fact1 = [1 cumprod(csiz(1:d-1))];
      fact1 = fact1(ones(n,1),:);
      for ir=0:2^(d-1)-1,
        bt0(:,:,1) = bitget(ir,1:d);
        bt0(:,:,2) = ~bt0(:,:,1);
        
        for ix = 0:1
          one = mod(ix,2)+1;
          two = mod(ix+1,2)+1;
          % Convert to linear index (faster than sub2ind)
          b1  = sum((binx + bt0(ones(n,1),:,one)-1).*fact1,2)+1; %linear index to c
          bt2 = bt0(:,:,two) + fact2;
          b2  = binx + bt2(ones(n,1),:);                     % linear index to X
          try
            if useSparse
              % Fast gridding using sparse
              c = c + sparse(b1,1,abs(prod(X(b2)-data,2)),Nc,1);
            else
              c = c + accum(b1,abs(prod(X(b2)-data,2)),[Nc,1]);
              %c = c + accum([b1,ones(n,1)],abs(prod(X(b2)-data,2)),[Nc,1]);
              %[len,bin,val] = bincount(b1,abs(prod(X(b2)-data,2)));
              %c(bin)        = c(bin)+val;
            end
          catch
             c = c + sparse(b1,1,abs(prod(X(b2)-data,2)),Nc,1);
          end
        end
      end
      c = reshape(c/w,csiz);
    end
    switch d % make sure c is stored in the same way as meshgrid
     case 2,  c = c.';
     case 3,  c = permute(c,[2 1 3]);
    end
    return

def main():
    import doctest
    doctest.testmod()
    
if __name__ == '__main__':
    main()
