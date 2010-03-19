from numpy import asarray, ndarray, reshape, repeat, nan, product

def valarray(shape,value=nan,typecode=None):
    """Return an array of all value.
    """
    out = reshape(repeat([value],product(shape,axis=0),axis=0),shape)
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, ndarray):
        out = asarray(out)
    return out

class rv_frozen(object):
    ''' Frozen continous or discrete 1D Random Variable object (RV)

    Methods
    -------
    RV.rvs(size=1)
        - random variates

    RV.pdf(x)
        - probability density function (continous case)

    RV.pmf(x)
        - probability mass function (discrete case)

    RV.cdf(x)
        - cumulative density function

    RV.sf(x)
        - survival function (1-cdf --- sometimes more accurate)

    RV.ppf(q)
        - percent point function (inverse of cdf --- percentiles)

    RV.isf(q)
        - inverse survival function (inverse of sf)

    RV.stats(moments='mv')
        - mean('m'), variance('v'), skew('s'), and/or kurtosis('k')

    RV.entropy()
        - (differential) entropy of the RV.

    Parameters
    ----------
    x : array-like
        quantiles
    q : array-like
        lower or upper tail probability
    size : int or tuple of ints, optional, keyword
        shape of random variates
    moments : string, optional, keyword
        one or more of 'm' mean, 'v' variance, 's' skewness, 'k' kurtosis
    '''
    def __init__(self, dist, *args, **kwds):
        self.dist = dist
        loc0, scale0 = map(kwds.get, ['loc', 'scale'])
        if isinstance(dist,rv_continuous):
            args, loc0, scale0 = dist.fix_loc_scale(args, loc0, scale0)
            self.par = args + (loc0, scale0)
        else: # rv_discrete
            args, loc0 = dist.fix_loc(args, loc0)
            self.par = args + (loc0,)


    def pdf(self,x):
        ''' Probability density function at x of the given RV.'''
        return self.dist.pdf(x,*self.par)
    def cdf(self,x):
        '''Cumulative distribution function at x of the given RV.'''
        return self.dist.cdf(x,*self.par)
    def ppf(self,q):
        '''Percent point function (inverse of cdf) at q of the given RV.'''
        return self.dist.ppf(q,*self.par)
    def isf(self,q):
        '''Inverse survival function at q of the given RV.'''
        return self.dist.isf(q,*self.par)
    def rvs(self, size=None):
        '''Random variates of given type.'''
        kwds = dict(size=size)
        return self.dist.rvs(*self.par,**kwds)
    def sf(self,x):
        '''Survival function (1-cdf) at x of the given RV.'''
        return self.dist.sf(x,*self.par)
    def stats(self,moments='mv'):
        ''' Some statistics of the given RV'''
        kwds = dict(moments=moments)
        return self.dist.stats(*self.par,**kwds)
    def moment(self,n):
        par1 = self.par[:self.dist.numargs]
        return self.dist.moment(n,*par1)
    def entropy(self):
        return self.dist.entropy(*self.par)
    def pmf(self,k):
        '''Probability mass function at k of the given RV'''
        return self.dist.pmf(k,*self.par)