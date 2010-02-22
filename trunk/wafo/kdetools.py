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
#import numpy as np
from scipy.special import gamma
from numpy import pi, atleast_2d #@UnresolvedImport
from misc import tranproc, trangood
def sphere_volume(d, r=1.0):
    """
     Returns volume of  d-dimensional sphere with radius r

    Parameters
    ----------
    d : scalar or array_like
        dimension of sphere
    r : scalar or array_like
        radius of sphere (default 1)

    Reference
    ---------
    Wand,M.P. and Jones, M.C. (1995)
    'Kernel smoothing'
    Chapman and Hall, pp 105
    """
    return (r**d)* 2.*pi**(d/2.)/(d*gamma(d/2.))


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

    def __init__(self, dataset,**kwds):
        self.kernel='gauss'
        self.hs = None
        self.hsmethod=None
        self.L2 = None
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
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        result = zeros((m,), points.dtype)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:,i,newaxis] - points
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff*tdiff,axis=0)/2.0
                result += exp(-energy)
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:,i,newaxis]
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff*tdiff,axis=0)/2.0
                result[i] = sum(exp(-energy),axis=0)

        result /= self._norm_factor

        return result

    __call__ = evaluate

##function [f, hs,lambda]= kdefun(A,options,varargin)
##%KDEFUN  Kernel Density Estimator.
##%
##% CALL:  [f, hs] = kdefun(data,options,x1,x2,...,xd)
##%
##%   f      = kernel density estimate evaluated at x1,x2,...,xd.
##%   data   = data matrix, size N x D (D = # dimensions)
##%  options = kdeoptions-structure or cellvector of named parameters with
##%            corresponding values, see kdeoptset for details.
##%   x1,x2..= vectors/matrices defining the points to evaluate the density
##%
##%  KDEFUN gives a slow, but exact kernel density estimate evaluated at x1,x2,...,xd.
##%  Notice that densities close to normality appear to be the easiest for the kernel
##%  estimator to estimate and that the degree of estimation difficulty increases with
##%  skewness, kurtosis and multimodality.
##%
##%  If D > 1 KDE calculates quantile levels by integration. An
##%  alternative is to calculate them by ranking the kernel density
##%  estimate obtained at the points DATA  i.e. use the commands
##%
##%      f    = kde(data);
##%      r    = kdefun(data,[],num2cell(data,1));
##%      f.cl = qlevels2(r,f.PL);
##%
##%  The first is probably best when estimating the pdf and the latter is the
##%  easiest and most robust for multidimensional data when only a visualization
##%  of the data is needed.
##%
##%  For faster estimates try kdebin.
##%
##% Examples:
##%     data = rndray(1,500,1);
##%     x = linspace(sqrt(eps),5,55);
##%     plotnorm((data).^(.5)) % gives a straight line => L2 = 0.5 reasonable
##%     f = kdefun(data,{'L2',.5},x);
##%     plot(x,f,x,pdfray(x,1),'r')
##%
##% See also  kde, mkernel, kdebin
##
##% Reference:
##%  B. W. Silverman (1986)
##% 'Density estimation for statistics and data analysis'
##%  Chapman and Hall , pp 100-110
##%
##%  Wand, M.P. and Jones, M.C. (1995)
##% 'Kernel smoothing'
##%  Chapman and Hall, pp 43--45
##
##
##
##
##%Tested on: matlab 5.2
##% History:
##% revised pab Feb2004
##%  -options moved into a structure
##% revised pab Dec2003
##% -removed some code
##% revised pab 27.04.2001
##% - changed call from mkernel to mkernel2 (increased speed by 10%)
##% revised pab 01.01.2001
##% - added the possibility that L2 is a cellarray of parametric
##%   or non-parametric transformations (secret option)
##% revised pab 14.12.1999
##%  - fixed a small error in example in help header
##% revised pab 28.10.1999
##%  - added L2
##% revised pab 21.10.99
##%  - added alpha to input arguments
##%  - made it fully general for d dimensions
##%  - HS may be a smoothing matrix
##% revised pab 21.09.99
##%  - adapted from kdetools by Christian Beardah
##
##  defaultoptions = kdeoptset;
##% If just 'defaults' passed in, return the default options in g
##if ((nargin==1) && (nargout <= 1) &&  isequal(A,'defaults')),
##  f = defaultoptions;
##  return
##end
##error(nargchk(1,inf, nargin))
##
##[n, d]=size(A); % Find dimensions of A,
##               % n=number of data points,
##               % d=dimension of the data.
##if (nargin<2 || isempty(options))
##  options  = defaultoptions;
##else
##  switch lower(class(options))
##   case {'char','struct'},
##    options = kdeoptset(defaultoptions,options);
##   case {'cell'}
##
##      options = kdeoptset(defaultoptions,options{:});
##   otherwise
##    error('Invalid options')
##  end
##end
##kernel   = options.kernel;
##h        = options.hs;
##alpha    = options.alpha;
##L2       = options.L2;
##hsMethod = options.hsMethod;
##
##if isempty(h)
##  h=zeros(1,d);
##end
##
##L22 = cell(1,d);
##k3=[];
##if isempty(L2)
##  L2=ones(1,d); % default no transformation
##elseif iscell(L2)   % cellarray of non-parametric and parametric transformations
##  Nl2 = length(L2);
##  if ~(Nl2==1||Nl2==d), error('Wrong size of L2'), end
##  [L22{1:d}] = deal(L2{1:min(Nl2,d)});
##  L2 = ones(1,d); % default no transformation
##  for ix=1:d,
##    if length(L22{ix})>1,
##      k3=[k3 ix];       % Non-parametric transformation
##    else
##     L2(ix) = L22{ix};  % Parameter to the Box-Cox transformation
##    end
##  end
##elseif length(L2)==1
##  L2=L2(:,ones(1,d));
##end
##
##amin=min(A);
##if any((amin(L2~=1)<=0))  ,
##  error('DATA cannot be negative or zero when L2~=1')
##end
##
##
##nv=length(varargin);
##if nv<d,
##  error('some or all of the evaluation points x1,x2,...,xd is missing')
##end
##
##xsiz = size(varargin{1}); % remember size of input
##Nx   = prod(xsiz);
##X    = zeros(Nx,d);
##for ix=1:min(nv,d),
##  if (any(varargin{ix}(:)<=0) && (L2(ix)~=1)),
##    error('xi cannot be negative or zero when L2~=1')
##  end
##  X(:,ix)=varargin{ix}(:); % make sure it is a column vector
##end
##
##
##%new call
##lX = X; %zeros(Nx,d);
##lA = A; %zeros(size(A));
##
##k1 = find(L2==0); % logaritmic transformation
##if any(k1)
##  lA(:,k1)=log(A(:,k1));
##  lX(:,k1)=log(X(:,k1));
##end
##k2=find(L2~=0 & L2~=1); % power transformation
##if any(k2)
##  lA(:,k2)=sign(L2(ones(n,1),k2)).*A(:,k2).^L2(ones(n,1),k2);
##  lX(:,k2)=sign(L2(ones(Nx,1),k2)).*X(:,k2).^L2(ones(Nx,1),k2);
##end
##% Non-parametric transformation
##for ix = k3,
##  lA(:,ix) = tranproc(A(:,ix),L22{ix});
##  lX(:,ix) = tranproc(X(:,ix),L22{ix});
##end
##
##
##hsiz=size(h);
##if (min(hsiz)==1)||(d==1)
##  if max(hsiz)==1,
##    h=h*ones(1,d);
##  else
##    h=reshape(h,[1,d]); % make sure it has the correct dimension
##  end;
##  ind=find(h<=0);
##  if any(ind)    % If no value of h has been specified by the user then
##    h(ind)=feval(hsMethod,lA(:,ind),kernel); % calculate automatic values.
##  end
##  deth = prod(h);
##else  % fully general smoothing matrix
##  deth = det(h);
##  if deth<=0
##    error('bandwidth matrix h must be positive definit')
##  end
##end
##
##if alpha>0
##  Xn   = num2cell(lA,1);
##  opt1 = kdeoptset('kernel',kernel,'hs',h,'alpha',0,'L2',1);
##  f2   = kdefun(lA,opt1,Xn{:}); % get a pilot estimate by regular KDE (alpha=0)
##  g    = exp(sum(log(f2))/n);
##
##  lambda=(f2(:)/g).^(-alpha);
##else
##  lambda=ones(n,1);
##end
##
##
##
##
##
##f=zeros(Nx,1);
##if (min(hsiz)==1)||(d==1)
##  for ix=1:n,     % Sum over all data points
##    Avec=lA(ix,:);
##    Xnn=(lX-Avec(ones(Nx,1),:))./(h(ones(Nx,1),:) *lambda(ix));
##    f = f + mkernel2(Xnn,kernel)/lambda(ix)^d;
##  end
##else % fully general
##  h1=inv(h);
##  for ix=1:n,     % Sum over all data points
##    Avec=lA(ix,:);
##    Xnn=(lX-Avec(ones(Nx,1),:))*(h1/lambda(ix));
##    f = f + mkernel2(Xnn,kernel)/lambda(ix)^d;
##  end
##end
##f=f/(n*deth);
##
##% transforming back
##if any(k1), % L2=0 i.e. logaritmic transformation
##  for ix=k1
##    f=f./X(:,ix);
##  end
##  if any(max(abs(diff(f)))>10)
##    disp('Warning: Numerical problems may have occured due to the logaritmic')
##    disp('transformation. Check the KDE for spurious spikes')
##  end
##end
##if any(k2) % L2~=0 i.e. power transformation
##  for ix=k2
##    f=f.*(X(:,ix).^(L2(ix)-1))*L2(ix)*sign(L2(ix));
##  end
##  if any(max(abs(diff(f)))>10)
##    disp('Warning: Numerical problems may have occured due to the power')
##    disp('transformation. Check the KDE for spurious spikes')
##  end
##end
##if any(k3), % non-parametric transformation
##  oneC = ones(Nx,1);
##  for ix=k3
##    gn  = L22{ix};
##    %Gn  = fliplr(L22{ix});
##    %x0  = tranproc(lX(:,ix),Gn);
##    if any(isnan(X(:,ix))),
##      error('The transformation does not have a strictly positive derivative.')
##    end
##    hg1  = tranproc([X(:,ix) oneC],gn);
##    der1 = abs(hg1(:,2)); % dg(X)/dX = 1/(dG(Y)/dY)
##    % alternative 2
##    %pp  = smooth(Gn(:,1),Gn(:,2),1,[],1);
##    %dpp = diffpp(pp);
##    %der1 = 1./abs(ppval(dpp,f.x{ix}));
##    % Alternative 3
##    %pp  = smooth(gn(:,1),gn(:,2),1,[],1);
##    %dpp = diffpp(pp);
##    %%plot(hg1(:,1),der1-abs(ppval(dpp,x0)))
##    %der1 = abs(ppval(dpp,x0));
##    if any(der1<=0),
##      error('The transformation must have a strictly positive derivative')
##    end
##    f = f.*der1;
##  end
##  if any(max(abs(diff(f)))>10)
##    disp('Warning: Numerical problems may have occured due to the power')
##    disp('transformation. Check the KDE for spurious spikes')
##  end
##end
##
##f=reshape(f,xsiz); % restore original shape
##if nargout>1
##  hs=h;
##end
##
##
##
##
##
##
##
##
##
##
##function [z,c]=mkernel(varargin)
##%MKERNEL Multivariate Kernel Function.
##%
##% CALL:  z = mkernel(x1,x2,...,xd,kernel);
##%        z = mkernel(X,kernel);
##%
##%
##%   z      = kernel function values evaluated at x1,x2,...,xd
##%   x1,x2..= input arguments, vectors or matrices with common size
##% or
##%   X      = cellarray of vector/matrices with common size
##%            (i.e. X{1}=x1, X{2}=x2....)
##%
##%   kernel = 'epanechnikov'  - Epanechnikov kernel.
##%            'epa1'          - product of 1D Epanechnikov kernel.
##%            'biweight'      - Bi-weight kernel.
##%            'biw1'          - product of 1D Bi-weight kernel.
##%            'triweight'     - Tri-weight kernel.
##%            'triangular'    - Triangular kernel.
##%            'gaussian'      - Gaussian kernel
##%            'rectangular'   - Rectangular kernel.
##%            'laplace'       - Laplace kernel.
##%            'logistic'      - Logistic kernel.
##%
##%  Note that only the first 4 letters of the kernel name is needed.
##%
##% See also  kde, kdefun, kdebin
##
##%  Reference:
##%  B. W. Silverman (1986)
##% 'Density estimation for statistics and data analysis'
##%  Chapman and Hall, pp. 43, 76
##%
##%  Wand, M. P. and Jones, M. C. (1995)
##% 'Density estimation for statistics and data analysis'
##%  Chapman and Hall, pp 31, 103,  175
##
##%Tested on: matlab 5.3
##% History:
##% Revised pab sep2005
##% -replaced reference to kdefft with kdebin
##% revised pab aug2005
##% -Fixed some bugs
##% revised pab Dec2003
##% removed some old code
##% revised pab 27.04.2001
##% - removed some old calls
##%  revised pab 01.01.2001
##%  - speeded up tri3
##%  revised pab 01.12.1999
##%   - added four weight, sphere
##%   - made comparison smarter => faster execution for d>1
##%  revised pab 26.10.1999
##%   fixed normalization fault in epan
##% by pab 21.09.99
##%  added multivariate epan, biweight and triweight
##%
##% collected all knorm,kepan ... into this file
##% adapted from kdetools CB
##
##d=length(varargin)-1;
##kstr=varargin{d+1}; % kernel string
##if iscell(varargin{1})
##  X=varargin{1};
##  d=numel(X);
##else
##  X=varargin;
##end
##
##switch lower(kstr(1:4))
##  case {'sphe','epan','biwe','triw','four'}
##    switch lower(kstr(1:4))
##      case 'sphe', r=0;  %Sphere = rect for 1D
##      case 'epan', r=1;  %Multivariate Epanechnikov kernel.
##      case 'biwe', r=2;  %Multivariate Bi-weight Kernel
##      case 'triw', r=3;  %Multi variate Tri-weight Kernel
##      case 'four', r=4;  %Multi variate Four-weight Kernel
##        % as r -> infty, b -> infty => kernel -> Gaussian distribution
##    end
##    b=1;% radius of the kernel
##    b2=b^2;
##    s=X{1}.^2;
##    k=find(s<=b2);
##    z=zeros(size(s));
##    ix=2;
##    while (any(k) && (ix<=d)),
##      s(k)=s(k)+X{ix}(k).^2;
##      k1=(s(k)<=b2);
##      k=k(k1);
##      ix=ix+1;
##    end;
##    if any(k)
##      c=2^r*prod(1:r)*vsph(d,b)/prod((d+2):2:(d+2*r)); % normalizing constant
##      %c=beta(r+1,r+1)*vsph(d,b)*(2^(2*r)); % Wand and Jones pp 31
##      % the commented c above does note yield the right scaling
##      % for d>1
##      z(k)=((1-s(k)/b2).^r)/c;
##    end
##
##  case 'rect', % 1D product Rectangular Kernel
##    z=zeros(size(X{1}));
##    k=find(abs(X{1})<=1);
##    ix=2;
##    while (any(k) && (ix<=d)),
##      k1 =(abs(X{ix}(k))<=1);
##      k=k(k1);
##      ix=ix+1;
##    end
##    if any(k)
##      z(k)=(0.5^d);
##    end
##  case {'epa1','biw1','triw1','fou1'}
##    switch lower(kstr(1:4))
##      %case 'rect', r=0;  %rectangular
##      case 'epa1', r=1;  %1D product Epanechnikov kernel.
##      case 'biw1', r=2;  %1D product Bi-weight Kernel
##      case 'tri1', r=3;  %1D product Tri-weight Kernel
##      case 'fou1', r=4;  %1D product Four-weight Kernel
##    end
##    b=1;
##    b2=b^2;
##    b21=1/b2;
##    z=zeros(size(X{1}));
##    k=find(abs(X{1})<=b);
##    ix=2;
##    while (any(k) && (ix<=d)),
##      %for ix=2:d
##      k1 =(abs(X{ix}(k))<=b);
##      k  = k(k1);
##      ix=ix+1;
##    end
##    if any(k)
##      c=2^r*prod(1:r)*vsph(1,b)/prod((1+2):2:(1+2*r)); % normalizing constant
##      z(k) = (1-X{1}(k).^2*b21).^r;
##      for ix=2:d
##        z(k)=z(k).*(1-X{ix}(k).^2*b21).^r;
##      end;
##      z(k)=z(k)/c^d;
##    end
##  case 'tria',% 1D product Triangular Kernel
##    z=zeros(size(X{1}));
##    k=find(abs(X{1})<1);
##    ix=2;
##    while (any(k) && (ix<=d)),
##      %for ix=2:d
##      k1 =(abs(X{ix}(k))<1);
##      k  = k(k1);
##      ix=ix+1;
##    end
##    if any(k)
##      z(k) = (1-abs(X{1}(k)));
##      for ix=2:d
##        z(k)=z(k).*(1-abs(X{ix}(k)));
##      end
##    end
##   case {'norm','gaus'},% multivariate gaussian  Density Function.
##     s=X{1}.^2;
##     for ix=2:d
##       s=s+X{ix}.^2;
##     end;
##     z=(2*pi)^(-d/2)*exp(-0.5*s);
##   case 'lapl' % Laplace Kernel
##     z=0.5*exp(-abs(X{1}));
##     for ix=2:d
##       z=z.*0.5*exp(-abs(X{ix}));
##     end
##   case 'logi', % Logistic Kernel
##     z1=exp(X{1});
##     z=z1./(z1+1).^2;
##     for ix=2:d
##       z1=exp(X{ix});
##       z=z.*z1./(z1+1).^2;
##     end
##
##   otherwise, error('unknown kernel')
## end
##
##


