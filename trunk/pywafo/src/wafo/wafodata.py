import warnings
from graphutil import cltext
from plotbackend import plotbackend
from time import gmtime, strftime
import numpy as np
from scipy.integrate.quadrature import cumtrapz  # @UnresolvedImport
from scipy.interpolate import griddata
from scipy import integrate

__all__ = ['PlotData', 'AxisLabels']


def empty_copy(obj):
    class Empty(obj.__class__):

        def __init__(self):
            pass
    newcopy = Empty()
    newcopy.__class__ = obj.__class__
    return newcopy


def _set_seed(iseed):
    if iseed != None:
        try:
            np.random.set_state(iseed)
        except:
            np.random.seed(iseed)


def now():
    '''
    Return current date and time as a string
    '''
    return strftime("%a, %d %b %Y %H:%M:%S", gmtime())


class PlotData(object):

    '''
    Container class for data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D, list of vectors for 2D, 3D, ...
    labels : AxisLabels
    children : list of PlotData objects

    Member methods
    --------------
    plot :
    copy :


    Example
    -------
    >>> import numpy as np
    >>> x = np.arange(-2, 2, 0.2)

    # Plot 2 objects in one call
    >>> d2 = PlotData(np.sin(x), x, xlab='x', ylab='sin', title='sinus')
    >>> h = d2.plot()

    Plot with confidence interval
    >>> d3 = PlotData(np.sin(x),x)
    >>> d3.children = [PlotData(np.vstack([np.sin(x)*0.9, np.sin(x)*1.2]).T,x)]
    >>> d3.plot_args_children=[':r']
    >>> h = d3.plot()

    See also
    --------
    specdata,
    covdata
    '''

    def __init__(self, data=None, args=None, *args2, **kwds):
        self.data = data
        self.args = args
        self.date = now()
        self.plotter = kwds.pop('plotter', None)
        self.children = None
        self.plot_args_children = kwds.pop('plot_args_children', [])
        self.plot_kwds_children = kwds.pop('plot_kwds_children', {})
        self.plot_args = kwds.pop('plot_args', [])
        self.plot_kwds = kwds.pop('plot_kwds', {})

        self.labels = AxisLabels(**kwds)
        if not self.plotter:
            self.setplotter(kwds.get('plotmethod', None))

    def plot(self, *args, **kwds):
        axis = kwds.pop('axis', None)
        if axis is None:
            axis = plotbackend.gca()
        tmp = None
        plotflag = kwds.get('plotflag', None)
        if not plotflag and self.children != None:
            plotbackend.hold('on')
            tmp = []
            child_args = kwds.pop(
                'plot_args_children', tuple(self.plot_args_children))
            child_kwds = dict(self.plot_kwds_children).copy()
            child_kwds.update(kwds.pop('plot_kwds_children', {}))
            child_kwds['axis'] = axis
            for child in self.children:
                tmp1 = child.plot(*child_args, **child_kwds)
                if tmp1 != None:
                    tmp.append(tmp1)
            if len(tmp) == 0:
                tmp = None
        main_args = args if len(args) else tuple(self.plot_args)
        main_kwds = dict(self.plot_kwds).copy()
        main_kwds.update(kwds)
        main_kwds['axis'] = axis
        tmp2 = self.plotter.plot(self, *main_args, **main_kwds)
        return tmp2, tmp

    def eval_points(self, *args, **kwds):
        '''
        >>> x = np.linspace(0,5,20)
        >>> d = PlotData(np.sin(x),x)
        >>> xi = np.linspace(0,5,60)
        >>> di = PlotData(d.eval_points(xi, method='cubic'),xi)
        >>> h = d.plot('.')
        >>> hi = di.plot()

        '''
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            ndim = len(self.args)
            if ndim < 2:
                msg = '''Unable to determine plotter-type, because len(self.args)<2.
                If the data is 1D, then self.args should be a vector!
                If the data is 2D, then length(self.args) should be 2.
                If the data is 3D, then length(self.args) should be 3.
                Unless you fix this, the plot methods will not work!'''
                warnings.warn(msg)
            else:
                return griddata(self.args, self.data.ravel(), *args, **kwds)
        else:  # One dimensional data
            return griddata((self.args,), self.data, *args, **kwds)

    def integrate(self, a, b, **kwds):
        '''
        >>> x = np.linspace(0,5,60)
        >>> d = PlotData(np.sin(x), x)
        >>> d.integrate(0,np.pi/2)
        0.99940054759302188

        '''
        method = kwds.pop('method', 'trapz')
        fun = getattr(integrate, method)
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            ndim = len(self.args)
            if ndim < 2:
                msg = '''Unable to determine plotter-type, because len(self.args)<2.
                If the data is 1D, then self.args should be a vector!
                If the data is 2D, then length(self.args) should be 2.
                If the data is 3D, then length(self.args) should be 3.
                Unless you fix this, the plot methods will not work!'''
                warnings.warn(msg)
            else:
                return griddata(self.args, self.data.ravel(), **kwds)
        else:  # One dimensional data

            x = self.args
            ix = np.flatnonzero((a < x) & (x < b))
            xi = np.hstack((a, x.take(ix), b))
            fi = np.hstack(
                (self.eval_points(a), self.data.take(ix), self.eval_points(b)))
            return fun(fi, xi, **kwds)

    def show(self):
        self.plotter.show()

    def copy(self):
        newcopy = empty_copy(self)
        newcopy.__dict__.update(self.__dict__)
        return newcopy

    def setplotter(self, plotmethod=None):
        '''
            Set plotter based on the data type data_1d, data_2d, data_3d or data_nd
        '''
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            ndim = len(self.args)
            if ndim < 2:
                msg = '''Unable to determine plotter-type, because len(self.args)<2.
                If the data is 1D, then self.args should be a vector!
                If the data is 2D, then length(self.args) should be 2.
                If the data is 3D, then length(self.args) should be 3.
                Unless you fix this, the plot methods will not work!'''
                warnings.warn(msg)
            elif ndim == 2:
                self.plotter = Plotter_2d(plotmethod)
            else:
                warnings.warn('Plotter method not implemented for ndim>2')

        else:  # One dimensional data
            self.plotter = Plotter_1d(plotmethod)


class AxisLabels:

    def __init__(self, title='', xlab='', ylab='', zlab='', **kwds):
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self.zlab = zlab

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '%s\n%s\n%s\n%s\n' % (self.title, self.xlab, self.ylab, self.zlab)

    def copy(self):
        newcopy = empty_copy(self)
        newcopy.__dict__.update(self.__dict__)
        return newcopy

    def labelfig(self, axis=None):
        if axis is None:
            axis = plotbackend.gca()
        try:
            h1 = axis.set_title(self.title)
            h2 = axis.set_xlabel(self.xlab)
            h3 = axis.set_ylabel(self.ylab)
            #h4 = plotbackend.zlabel(self.zlab)
            return h1, h2, h3
        except:
            pass


class Plotter_1d(object):

    """

    Parameters
    ----------
    plotmethod : string
        defining type of plot. Options are:
        bar : bar plot with rectangles
        barh : horizontal bar plot with rectangles
        loglog : plot with log scaling on the *x* and *y* axis
        semilogx :  plot with log scaling on the *x* axis
        semilogy :  plot with log scaling on the *y* axis
        plot : Plot lines and/or markers (default)
        stem : Stem plot
        step : stair-step plot
        scatter : scatter plot
    """

    def __init__(self, plotmethod='plot'):
        self.plotfun = None
        if plotmethod is None:
            plotmethod = 'plot'
        self.plotmethod = plotmethod
        self.plotbackend = plotbackend
#        try:
#            self.plotfun = getattr(plotbackend, plotmethod)
#        except:
#            pass

    def show(self):
        plotbackend.show()

    def plot(self, wdata, *args, **kwds):
        axis = kwds.pop('axis', None)
        if axis is None:
            axis = plotbackend.gca()
        plotflag = kwds.pop('plotflag', False)
        if plotflag:
            h1 = self._plot(axis, plotflag, wdata, *args, **kwds)
        else:
            if isinstance(wdata.data, (list, tuple)):
                vals = tuple(wdata.data)
            else:
                vals = (wdata.data,)
            if isinstance(wdata.args, (list, tuple)):
                args1 = tuple((wdata.args)) + vals + args
            else:
                args1 = tuple((wdata.args,)) + vals + args
            plotfun = getattr(axis, self.plotmethod)
            h1 = plotfun(*args1, **kwds)
        h2 = wdata.labels.labelfig(axis)
        return h1, h2

    def _plot(self, axis, plotflag, wdata, *args, **kwds):
        x = wdata.args
        data = transformdata(x, wdata.data, plotflag)
        dataCI = getattr(wdata, 'dataCI', ())
        h1 = plot1d(axis, x, data, dataCI, plotflag, *args, **kwds)
        return h1


def plot1d(axis, args, data, dataCI, plotflag, *varargin, **kwds):

    plottype = np.mod(plotflag, 10)
    if plottype == 0:  # %  No plotting
        return []
    elif plottype == 1:
        H = axis.plot(args, data, *varargin, **kwds)
    elif plottype == 2:
        H = axis.step(args, data, *varargin, **kwds)
    elif plottype == 3:
        H = axis.stem(args, data, *varargin, **kwds)
    elif plottype == 4:
        H = axis.errorbar(
            args, data, yerr=[dataCI[:, 0] - data, dataCI[:, 1] - data], *varargin, **kwds)
    elif plottype == 5:
        H = axis.bar(args, data, *varargin, **kwds)
    elif plottype == 6:
        level = 0
        if np.isfinite(level):
            H = axis.fill_between(args, data, level, *varargin, **kwds)
        else:
            H = axis.fill_between(args, data, *varargin, **kwds)
    elif plottype == 7:
        H = axis.plot(args, data, *varargin, **kwds)
        H = axis.fill_between(
            args, dataCI[:, 0], dataCI[:, 1], alpha=0.2, color='r')

    scale = plotscale(plotflag)
    logXscale = 'x' in scale
    logYscale = 'y' in scale
    logZscale = 'z' in scale

    if logXscale:
        axis.set(xscale='log')
    if logYscale:
        axis.set(yscale='log')
    if logZscale:
        axis.set(zscale='log')

    transFlag = np.mod(plotflag // 10, 10)
    logScale = logXscale or logYscale or logZscale
    if logScale or (transFlag == 5 and not logScale):
        ax = list(axis.axis())
        fmax1 = data.max()
        if transFlag == 5 and not logScale:
            ax[3] = 11 * np.log10(fmax1)
            ax[2] = ax[3] - 40
        else:
            ax[3] = 1.15 * fmax1
            ax[2] = ax[3] * 1e-4

        axis.axis(ax)

    if np.any(dataCI) and plottype < 3:
        axis.hold(True)
        plot1d(axis, args, dataCI, (), plotflag, 'r--')
    return H


def plotscale(plotflag):
    '''
    Return plotscale from plotflag

     CALL scale = plotscale(plotflag)

     plotflag = integer defining plotscale.
       Let scaleId = floor(plotflag/100). 
       If scaleId < 8 then:
          0 'linear' : Linear scale on all axes.
          1 'xlog'   : Log scale on x-axis.
          2 'ylog'   : Log scale on y-axis.
          3 'xylog'  : Log scale on xy-axis.
          4 'zlog'   : Log scale on z-axis.
          5 'xzlog'  : Log scale on xz-axis.
          6 'yzlog'  : Log scale on yz-axis.
          7 'xyzlog' : Log scale on xyz-axis.
      otherwise
       if (mod(scaleId,10)>0)            : Log scale on x-axis.
       if (mod(floor(scaleId/10),10)>0)  : Log scale on y-axis.
       if (mod(floor(scaleId/100),10)>0) : Log scale on z-axis.

     scale    = string defining plotscale valid options are:
           'linear', 'xlog', 'ylog', 'xylog', 'zlog', 'xzlog',
           'yzlog',  'xyzlog' 

    Example
    >>> for id in range(100,701,100):
    ...    plotscale(id)  
    'xlog'
    'ylog'
    'xylog'
    'zlog'
    'xzlog'
    'yzlog'
    'xyzlog'

    >>> plotscale(200)  
    'ylog'
    >>> plotscale(300) 
    'xylog'
    >>> plotscale(300) 
    'xylog'

    See also 
    --------
    transformdata
    '''
    scaleId = plotflag // 100
    if scaleId > 7:
        logXscaleId = np.mod(scaleId, 10) > 0
        logYscaleId = (np.mod(scaleId // 10, 10) > 0) * 2
        logZscaleId = (np.mod(scaleId // 100, 10) > 0) * 4
        scaleId = logYscaleId + logXscaleId + logZscaleId

    scales = ['linear', 'xlog', 'ylog', 'xylog',
              'zlog', 'xzlog', 'yzlog', 'xyzlog']

    return scales[scaleId]


def transformdata(x, f, plotflag):
    transFlag = np.mod(plotflag // 10, 10)
    if transFlag == 0:
        data = f
    elif transFlag == 1:
        data = 1 - f
    elif transFlag == 2:
        data = cumtrapz(f, x)
    elif transFlag == 3:
        data = 1 - cumtrapz(f, x)
    if transFlag in (4, 5):
        if transFlag == 4:
            data = -np.log1p(-cumtrapz(f, x))
        else:
            if any(f < 0):
                raise ValueError(
                    'Invalid plotflag: Data or dataCI is negative, but must be positive')
            data = 10 * np.log10(f)
    return data


class Plotter_2d(Plotter_1d):

    """
    Parameters
    ----------
    plotmethod : string
        defining type of plot. Options are:
        contour (default)
        contourf
        mesh
        surf
    """

    def __init__(self, plotmethod='contour'):
        if plotmethod is None:
            plotmethod = 'contour'
        super(Plotter_2d, self).__init__(plotmethod)

    def _plot(self, axis, plotflag, wdata, *args, **kwds):
        h1 = plot2d(axis, wdata, plotflag, *args, **kwds)
        return h1


def plot2d(axis, wdata, plotflag, *args, **kwds):
    f = wdata
    if isinstance(wdata.args, (list, tuple)):
        args1 = tuple((wdata.args)) + (wdata.data,) + args
    else:
        args1 = tuple((wdata.args,)) + (wdata.data,) + args
    if plotflag in (1, 6, 7, 8, 9):
        isPL = False
        # check if contour levels is submitted
        if hasattr(f, 'clevels') and len(f.clevels) > 0:
            CL = f.clevels
            isPL = hasattr(f, 'plevels') and f.plevels is not None
            if isPL:
                PL = f.plevels  # levels defines quantile levels? 0=no 1=yes
        else:
            dmax = np.max(f.data)
            dmin = np.min(f.data)
            CL = dmax - (dmax - dmin) * \
                (1 - np.r_[0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.75])
        clvec = np.sort(CL)

        if plotflag in [1, 8, 9]:
            h = axis.contour(*args1, levels=CL, **kwds)
        # else:
        #  [cs hcs] = contour3(f.x{:},f.f,CL,sym);

        if plotflag in (1, 6):
            ncl = len(clvec)
            if ncl > 12:
                ncl = 12
                warnings.warn(
                    'Only the first 12 levels will be listed in table.')

            clvals = PL[:ncl] if isPL else clvec[:ncl]
            # print contour level text
            unused_axcl = cltext(clvals, percent=isPL)
        elif any(plotflag == [7, 9]):
            axis.clabel(h)
        else:
            axis.clabel(h)
    elif plotflag == 2:
        h = axis.mesh(*args1, **kwds)
    elif plotflag == 3:
        # shading interp % flat, faceted       % surfc
        h = axis.surf(*args1, **kwds)
    elif plotflag == 4:
        h = axis.waterfall(*args1, **kwds)
    elif plotflag == 5:
        h = axis.pcolor(*args1, **kwds)  # %shading interp % flat, faceted
    elif plotflag == 10:
        h = axis.contourf(*args1, **kwds)
        axis.clabel(h)
        plotbackend.colorbar(h)
    else:
        raise ValueError('unknown option for plotflag')
    # if any(plotflag==(2:5))
    #   shading(shad);
    # end
    #    pass


def test_eval_points():
    plotbackend.ioff()
    x = np.linspace(0, 5, 21)
    d = PlotData(np.sin(x), x)
    xi = np.linspace(0, 5, 61)
    di = PlotData(d.eval_points(xi, method='cubic'), xi)
    d.plot('.')
    di.plot()
    di.show()


def test_integrate():
    x = np.linspace(0, 5, 60)
    d = PlotData(np.sin(x), x)
    print(d.integrate(0, np.pi / 2, method='simps'))


def test_docstrings():
    import doctest
    doctest.testmod()


def main():
    pass

if __name__ == '__main__':

    # test_integrate()
    # test_eval_points()
    test_docstrings()
    # main()
