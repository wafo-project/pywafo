'''
'''
from __future__ import division
from numpy import trapz, sqrt, linspace  # @UnresolvedImport

from wafo.containers import PlotData
from wafo.misc import tranproc  # , trangood

__all__ = ['TrData', 'TrCommon']


class TrCommon(object):

    """
    <generic> transformation model, g.

    Information about the moments of the process can be obtained by site
    specific data, laboratory measurements or by resort to theoretical models.

    Assumption
    ----------
    The Gaussian process, Y, distributed N(0,1) is related to the
    non-Gaussian process, X, by Y = g(X).

    Methods
    -------
    dist2gauss : Returns a measure of departure from the Gaussian model, i.e.,
                int (g(x)-xn)**2 dx  where int. limits are given by X.
    dat2gauss : Transform non-linear data to Gaussian scale
    gauss2dat : Transform Gaussian data to non-linear scale

    Member variables
    ----------------
    mean, sigma, skew, kurt : real, scalar
        mean, standard-deviation, skewness and kurtosis, respectively, of the
        non-Gaussian process. Default mean=0, sigma=1, skew=0.16, kurt=3.04.
        skew=kurt-3=0 for a Gaussian process.
    """

    def __init__(self, mean=0.0, var=1.0, skew=0.16, kurt=3.04, *args, **kwds):
        sigma = kwds.get('sigma', None)
        if sigma is None:
            sigma = sqrt(var)
        self.mean = mean
        self.sigma = sigma
        self.skew = skew
        self.kurt = kurt
        # Mean and std in the Gaussian world:
        self.ymean = kwds.get('ymean', 0e0)
        self.ysigma = kwds.get('ysigma', 1e0)

    def __call__(self, x, *xi):
        return self._dat2gauss(x, *xi)

    def dist2gauss(self, x=None, xnmin=-5, xnmax=5, n=513):
        """
        Return a measure of departure from the Gaussian model.

        Parameters
        ----------
        x : vector  (default sigma*linspace(xnmin,xnmax,n)+mean)
        xnmin : real, scalar
            minimum on normalized scale
        xnmax : real, scalar
            maximum on normalized scale
        n : integer, scalar
            number of evaluation points


        Returns
        -------
        t0 : real, scalar
            a measure of departure from the Gaussian model calculated as
            trapz((xn-g(x))**2., xn) where int. limits is given by X.
        """
        if x is None:
            xn = linspace(xnmin, xnmax, n)
            x = self.sigma * xn + self.mean
        else:
            xn = (x - self.mean) / self.sigma

        yn = (self._dat2gauss(x) - self.ymean) / self.ysigma
        t0 = trapz((xn - yn) ** 2., xn)
        return t0

    def gauss2dat(self, y, *yi):
        """
        Transforms Gaussian data, y, to non-linear scale.

        Parameters
        ----------
        y, y1,..., yn : array-like
            input vectors with Gaussian data values, where yi is the i'th time
            derivative of y. (n<=4)
        Returns
        -------
        x, x1,...,xn : array-like
            transformed data to a non-linear scale

        See also
        --------
        dat2gauss
        tranproc
        """
        return self._gauss2dat(y, *yi)

    def _gauss2dat(self, y, *yi):
        pass

    def dat2gauss(self, x, *xi):
        """
        Transforms non-linear data, x, to Gaussian scale.

        Parameters
        ----------
        x, x1,...,xn : array-like
            input vectors with non-linear data values, where xi is the i'th
            time derivative of x. (n<=4)
        Returns
        -------
        y, y1,...,yn : array-like
            transformed data to a Gaussian scale

        See also
        --------
        gauss2dat
        tranproc.
        """
        return self._dat2gauss(x, *xi)

    def _dat2gauss(self, x, *xi):
        pass


class TrData(PlotData, TrCommon):
    __doc__ = TrCommon.__doc__.split('mean')[0].replace('<generic>',
                                                        'Data') + """
    data : array-like
        Gaussian values, Y
    args : array-like
        non-Gaussian values, X
    ymean, ysigma : real, scalars (default ymean=0, ysigma=1)
        mean and standard-deviation, respectively, of the process in Gaussian
        world.
    mean, sigma : real, scalars
        mean and standard-deviation, respectively, of the non-Gaussian process.
        Default:
        mean = self.gauss2dat(ymean),
        sigma = (self.gauss2dat(ysigma)-self.gauss2dat(-ysigma))/2

    Example
    -------
    Construct a linear transformation model
    >>> import numpy as np
    >>> import wafo.transform as wt
    >>> sigma = 5; mean = 1
    >>> u = np.linspace(-5,5); x = sigma*u+mean; y = u
    >>> g = wt.TrData(y,x)
    >>> g.mean
    array([ 1.])
    >>> g.sigma
    array([ 5.])

    >>> g = wt.TrData(y,x,mean=1,sigma=5)
    >>> g.mean
    1
    >>> g.sigma
    5
    >>> g.dat2gauss(1,2,3)
    [array([ 0.]), array([ 0.4]), array([ 0.6])]

    Check that the departure from a Gaussian model is zero
    >>> g.dist2gauss() < 1e-16
    True
    """

    def __init__(self, *args, **kwds):
        options = dict(title='Transform',
                       xlab='x', ylab='g(x)',
                       plot_args=['r'],
                       plot_args_children=['g--'],)
        options.update(**kwds)
        super(TrData, self).__init__(*args, **options)
        self.ymean = kwds.get('ymean', 0e0)
        self.ysigma = kwds.get('ysigma', 1e0)
        self.mean = kwds.get('mean', None)
        self.sigma = kwds.get('sigma', None)

        if self.mean is None:
            # self.mean = np.mean(self.args)
            self.mean = self.gauss2dat(self.ymean)
        if self.sigma is None:
            yp = self.ymean + self.ysigma
            ym = self.ymean - self.ysigma
            self.sigma = (self.gauss2dat(yp) - self.gauss2dat(ym)) / 2.

        self.children = [
            PlotData((self.args - self.mean) / self.sigma, self.args)]

    def trdata(self):
        return self

    def _gauss2dat(self, y, *yi):
        return tranproc(self.data, self.args, y, *yi)

    def _dat2gauss(self, x, *xi):
        return tranproc(self.args, self.data, x, *xi)


class EstimateTransform(object):
    pass


def main():
    pass

if __name__ == '__main__':
    if True:  # False : #
        import doctest
        doctest.testmod()
    else:
        main()
