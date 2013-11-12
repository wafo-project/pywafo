"""
Commentary
----------

Most of the work is done by the scipy.stats.distributions module.

This provides a plethora of continuous distributions to play with.

Each distribution has functions to generate random deviates, pdf's,
cdf's etc. as well as a function to fit the distribution to some given
data.

The fitting uses scipy.optimize.fmin to minimise the log odds of the
data given the distribution.

There are a couple of problems with this approach.  First it is
sensitive to the initial guess at the parameters.  Second it can be a
little slow.

Two key parameters are the 'loc' and 'scale' parameters.  Data is
shifted by 'loc' and scaled by scale prior to fitting.  Supplying
appropriate values for these parameters is important to getting a good
fit.

See the factory() function which picks from a handful of common
approaches for each distribution.

For some distributions (eg normal) it really makes sense just to
calculate the parameters directly from the data.

The code in the __ifmain__ should be a good guide how to use this.

Simply:
      get a QuickFit object
      add the distributions you want to try to fit
      call fit() with your data
      call fit_stats() to generate some stats on the fit.
      call plot() if you want to see a plot.


Named after Mrs Twolumps, minister's secretary in the silly walks
sketch, who brings in coffee with a full silly walk.

Tenuous link with curve fitting is that you generally see "two lumps"
one in your data and the other in the curve that is being fitted.

Or alternately, if your data is not too silly then you can fit a
curve to it.

License is GNU LGPL v3, see https://launchpad.net/twolumps
"""
import inspect
from itertools import izip

import numpy
from wafo import stats
from scipy import mean, std

def factory(name):
    """ Factory to return appropriate objects for each distro. """
    fitters = dict(

        beta=ZeroOneScipyDistribution,
        alpha=ZeroOneScipyDistribution,
        ncf=ZeroOneScipyDistribution,
        triang=ZeroOneScipyDistribution,
        uniform=ZeroOneScipyDistribution,
        powerlaw=ZeroOneScipyDistribution,

        pareto=MinLocScipyDistribution,
        expon=MinLocScipyDistribution,
        gamma=MinLocScipyDistribution,
        lognorm=MinLocScipyDistribution,
        maxwell=MinLocScipyDistribution,
        weibull_min=MinLocScipyDistribution,

        weibull_max=MaxLocScipyDistribution)

    return fitters.get(name, ScipyDistribution)(name)
  

def get_continuous_distros():
    """ Find all attributes of stats that are continuous distributions. """
    
    fitters = []
    skip = set()
    for name, item in inspect.getmembers(stats):
        if name in skip: continue
        if item is stats.rv_continuous: continue
        if isinstance(item, stats.rv_continuous):
            fitters.append([name, factory(name)])

    return fitters


class ScipyDistribution(object):

    def __init__(self, name):

        self.name = name
        self.distro = self.get_distro()
        self.fitted = None

    def __getattr__(self, attr):
        """ Try delegating to the distro object """
        return getattr(self.distro, attr)

    def get_distro(self):

        return getattr(stats, self.name)
    
    def set_distro(self, parms):
        
        self.distro = getattr(stats, self.name)(*parms)

        return self.distro

    def calculate_loc_and_scale(self, data):
        """ Calculate loc and scale parameters for fit.

        Depending on the distribution, these need to be approximately
        right to get a good fit.
        """
        return mean(data), std(data)
        
    def fit(self, data, *args, **kwargs):
        """ This needs some work.

        Seems the various scipy distributions do a reasonable job if given a good hint.

        Need to get distro specific hints.
        """

        fits = []

        # try with and without providing loc and scale hints
        # increases chance of a fit without an exception being
        # generated.
        for (loc, scale) in ((0.0, 1.0),
                             self.calculate_loc_and_scale(data)):

            try:
                parms = self.get_distro().fit(data, loc=loc, scale=scale)
                    
                self.set_distro(list(parms))
                expected = self.expected(data)
                rss = ((expected-data)**2).sum()
                fits.append([rss, list(parms)])
                
                parms = self.get_distro().fit(data, floc=loc, scale=scale)
                    
                self.set_distro(list(parms))
                expected = self.expected(data)
                rss = ((expected-data)**2).sum()
                fits.append([rss, list(parms)])
            except:
                pass

        # no fits means all tries raised exceptions
        if not fits:
            raise Exception("Exception in fit()")

        # pick the one with the smallest rss
        fits.sort()
        self.parms = fits[0][1]
        print self.parms
    
        return self.set_distro(list(self.parms))

    def expected(self, data):
        """ Calculate expected values at each data point """
        if self.fitted is not None:
            return self.fitted

        n = len(data)
        xx = numpy.linspace(0, 1, n + 2)[1:-1]
        self.fitted = self.ppf(xx)
        #self.fitted = [self.ppf(x) for x in xx]

        return self.fitted
    
    def fit_stats(self, data):
        """ Return stats on the fits

        data assumed to be sorted.
        """
        n = len(data)

        dvar = numpy.var(data)
        expected = self.expected(data)
        evar = numpy.var(expected)

        rss = 0.0
        for expect, obs in izip(expected, data):
            rss += (obs-expect) ** 2.0

        self.rss = rss
        self.dss = dvar * n
        self.fss = evar * n
        
    def residuals(self, data):
        """ Return residuals """
        expected = self.expected(data)

        return numpy.array(data) - numpy.array(expected)
        


class MinLocScipyDistribution(ScipyDistribution):

    def calculate_loc_and_scale(self, data):
        """ Set loc to min value in the data.

        Useful for weibull_min
        """
        return min(data), std(data)

class MaxLocScipyDistribution(ScipyDistribution):

    def calculate_loc_and_scale(self, data):
        """ Set loc to max value in the data.

        Useful for weibull_max
        """
        return max(data), std(data)

class ZeroOneScipyDistribution(ScipyDistribution):

    def calculate_loc_and_scale(self, data):
        """ Set loc and scale to move to [0, 1] interval.

        Useful for beta distribution
        """
        return min(data), max(data)-min(data)

class QuickFit(object):
    """ Fit a family of distributions.

    Calculates stats on each fit.

    Option to create plots.
    """

    def __init__(self):

        self.distributions = []

    def add_distribution(self, distribution):
        """ Add a ready-prepared ScipyDistribution """
        self.distributions.append(distribution)

    def add(self, name):
        """ Add a distribution by name. """

        self.distributions.append(factory(name))

    def fit(self, data):
        """ Fit all of the distros we have """
        fitted = []
        for distro in self.distributions:
            print 'fitting distro', distro.name
            try:
                distro.fit(data)
            except:
                continue
            fitted.append(distro)
        self.distributions = fitted
            
        print 'finished fitting'

    def stats(self, data):
        """ Return stats on the fits """
        for dd in self.distributions:
            dd.fit_stats(data)

    def get_topn(self, n):
        """ Return top-n best fits. """
        data = [[x.rss, x] for x in self.distributions if numpy.isfinite(x.rss)]
        data.sort()

        if not n:
            n = len(data)

        return [x[1] for x in data[:n]]

    def fit_plot(self, data, topn=0, bins=20):
        """ Create a plot. """
        from matplotlib import pylab as pl

        distros = self.get_topn(topn)

        xx = numpy.linspace(data.min(), data.max(), 300)

        table = []
        nparms = max(len(x.parms) for x in distros)
        tcolours = []
        for dd in distros:
            patch = pl.plot(xx, [dd.pdf(p) for p in xx], label='%10.2f%% %s' % (100.0*dd.rss/dd.dss, dd.name))
            row = ['', dd.name, '%10.2f%%' % (100.0*dd.rss/dd.dss,)] + ['%0.2f' % x for x in dd.parms]
            while len(row) < 3 + nparms:
                row.append('')
            table.append(row)
            tcolours.append([patch[0].get_markerfacecolor()] + ['w'] * (2+nparms))

        # add a historgram with the data
        pl.hist(data, bins=bins, normed=True)
        tab = pl.table(cellText=table, cellColours=tcolours,
                       colLabels=['', 'Distribution', 'Res. SS/Data SS'] + ['P%d' % (x + 1,) for x in range(nparms)],
                       bbox=(0.0, 1.0, 1.0, 0.3))
                 #loc='top'))
        #pl.legend(loc=0)
        tab.auto_set_font_size(False)
        tab.set_fontsize(10.)

    def residual_plot(self, data, topn=0):
        """ Create a residual plot. """
        from matplotlib import pylab as pl

        distros = self.get_topn(topn)


        n = len(data)
        xx = numpy.linspace(0, 1, n + 2)[1:-1]
        for dd in distros:

            pl.plot(xx, dd.residuals(data), label='%10.2f%% %s' % (100.0*dd.rss/dd.dss, dd.name))
        pl.grid(True)

    def plot(self, data, topn):
        """ Plot data fit and residuals """
        from matplotlib import pylab as pl
        pl.axes([0.1, 0.4, 0.8, 0.4])   # leave room above the axes for the table
        self.fit_plot(data, topn=topn)

        pl.axes([0.1, 0.05, 0.8, 0.3]) 
        self.residual_plot(data, topn=topn)


def read_data(infile, field):
    """ Simple utility to extract a field out of a csv file. """
    import csv

    reader = csv.reader(infile)
    header = reader.next()
    field = header.index(field)
    data = []
    for row in reader:
        data.append(float(row[field]))

    return data
        
if __name__ == '__main__':

    import sys
    import optparse

    from matplotlib import pylab as pl
    
    parser = optparse.OptionParser()
    parser.add_option('-d', '--distro', action='append', default=[])
    parser.add_option('-l', '--list', action='store_true',
                      help='List available distros')

    parser.add_option('-i', '--infile')
    parser.add_option('-f', '--field', default='P/L')

    parser.add_option('-n', '--topn', type='int', default=0)

    parser.add_option('-s', '--sample', default='normal',
                      help='generate a sample from this distro as a test')
    parser.add_option('--size', type='int', default=1000,
                      help='Size of sample to generate')

    
    opts, args = parser.parse_args()
    
    if opts.list:
        for name, distro in get_continuous_distros():
            print name
        sys.exit()
    opts.distro = ['weibull_min', 'norm']
    if not opts.distro:
        opts.distro = [x[0] for x in get_continuous_distros()]

    quickfit = QuickFit()
    for distro in opts.distro:
        quickfit.add(distro)

    if opts.sample:
        data = getattr(numpy.random, opts.sample)(size=opts.size)
    else:
        data = numpy.array(read_data(open(opts.infile), opts.field))
        
    data.sort()

    quickfit.fit(data)
    print 'doing stats'
    quickfit.stats(data)

    print 'doing plot'
    quickfit.plot(data, topn=opts.topn)
    pl.show()
    
        

    

    

        

    
