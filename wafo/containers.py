from __future__ import absolute_import
import warnings
from wafo.graphutil import cltext
from wafo.plotbackend import plotbackend as plt
from time import gmtime, strftime
import numpy as np
from scipy.integrate.quadrature import cumtrapz  # @UnresolvedImport
from scipy import interpolate
from scipy import integrate
from _warnings import warn
__all__ = ['PlotData', 'AxisLabels']


def empty_copy(obj):
    class Empty(obj.__class__):

        def __init__(self):
            pass
    newcopy = Empty()
    newcopy.__class__ = obj.__class__
    return newcopy


def now():
    '''
    Return current date and time as a string
    '''
    return strftime("%a, %d %b %Y %H:%M:%S", gmtime())


class PlotData(object):

    '''
    Container class for data with interpolation and plotting methods

    Member variables
    ----------------
    data : array_like
    args : vector for 1D, list of vectors for 2D, 3D, ...
    labels : AxisLabels
    children : list of PlotData objects
    plot_args_children : list of arguments to the children plots
    plot_kwds_children : dict of keyword arguments to the children plots
    plot_args : list of arguments to the main plot
    plot_kwds : dict of keyword arguments to the main plot

    Member methods
    --------------
    copy : return a copy of object
    eval_points : interpolate data at given points and return the result
    plot : plot data on given axis and the object handles

    Example
    -------
    >>> import numpy as np
    >>> x = np.linspace(0, np.pi, 9)

    # Plot 2 objects in one call
    >>> d2 = PlotData(np.sin(x), x, xlab='x', ylab='sin', title='sinus',
    ...               plot_args=['r.'])

    >>> h = d2.plot()
    >>> h1 = d2()

    # Plot with confidence interval
    >>> ci = PlotData(np.vstack([np.sin(x)*0.9, np.sin(x)*1.2]).T, x,
    ...               plot_args=[':r'])
    >>> d3 = PlotData(np.sin(x), x, children=[ci])
    >>> h = d3.plot() # plot data, CI red dotted line
    >>> h = d3.plot(plot_args_children=['b--']) # CI with blue dashed line

    '''

    def __init__(self, data=None, args=None, *args2, **kwds):
        self.data = data
        self.args = args
        self.date = now()
        self.plotter = kwds.pop('plotter', None)
        self.children = kwds.pop('children', None)
        self.plot_args_children = kwds.pop('plot_args_children', [])
        self.plot_kwds_children = kwds.pop('plot_kwds_children', {})
        self.plot_args = kwds.pop('plot_args', [])
        self.plot_kwds = kwds.pop('plot_kwds', {})

        self.labels = AxisLabels(**kwds)
        if not self.plotter:
            self.setplotter(kwds.get('plotmethod'))

    def copy(self):
        newcopy = empty_copy(self)
        newcopy.__dict__.update(self.__dict__)
        return newcopy

    def eval_points(self, *points, **kwds):
        '''
        Interpolate data at points

        Parameters
        ----------
        points :  ndarray of float, shape (..., ndim)
            Points where to interpolate data at.
              method : {'linear', 'nearest', 'cubic'}
        method : {'linear', 'nearest', 'cubic'}
            Method of interpolation. One of
            - ``nearest``: return the value at the data point closest to
              the point of interpolation.
            - ``linear``: tesselate the input point set to n-dimensional
              simplices, and interpolate linearly on each simplex.
            - ``cubic`` (1-D): return the value detemined from a cubic
              spline.
            - ``cubic`` (2-D): return the value determined from a
              piecewise cubic, continuously differentiable (C1), and
              approximately curvature-minimizing polynomial surface.
        fill_value : float, optional
            Value used to fill in for requested points outside of the
            convex hull of the input points.  If not provided, then the
            default is ``nan``. This option has no effect for the
            'nearest' method.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.arange(-2, 2, 0.4)
        >>> xi = np.arange(-2, 2, 0.1)

        >>> d = PlotData(np.sin(x), x, xlab='x', ylab='sin', title='sinus',
        ...                plot_args=['r.'])
        >>> di = PlotData(d.eval_points(xi), xi)

        >>> hi = di.plot()
        >>> h = d.plot()
        >>> dicdf = di.to_cdf()
        >>> h = dicdf.plot()


        See also
        --------
        scipy.interpolate.griddata
        '''
        options = dict(method='linear')
        options.update(**kwds)
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            ndim = len(self.args)
            if ndim < 2:
                msg = '''
                Unable to determine plotter-type, because len(self.args)<2.
                If the data is 1D, then self.args should be a vector!
                If the data is 2D, then length(self.args) should be 2.
                If the data is 3D, then length(self.args) should be 3.
                Unless you fix this, the interpolation will not work!'''
                warnings.warn(msg)
            else:
                xi = np.meshgrid(*self.args)
                return interpolate.griddata(xi, self.data.ravel(), points,
                                            **options)
        # One dimensional data
        return interpolate.griddata(self.args, self.data, points, **options)

    def to_cdf(self):
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            raise NotImplementedError('integration for ndim>1 not implemented')
        cdf = np.hstack((0, cumtrapz(self.data, self.args)))
        return PlotData(cdf, np.copy(self.args), xlab='x', ylab='F(x)')

    def _get_fi_xi(self, a, b):
        x = self.args
        if a is None:
            a = x[0]
        if b is None:
            b = x[-1]
        ix = np.flatnonzero((a < x) & (x < b))
        xi = np.hstack((a, x.take(ix), b))

        if self.data.ndim > 1:
            fi = np.vstack((self.eval_points(a),
                            self.data[ix, :],
                            self.eval_points(b))).T
        else:
            fi = np.hstack((self.eval_points(a), self.data.take(ix),
                            self.eval_points(b)))
        return fi, xi

    def integrate(self, a=None, b=None, **kwds):
        '''
        >>> x = np.linspace(0,5,60)
        >>> y = np.sin(x)
        >>> ci = PlotData(np.vstack((y*.9, y*1.1)).T, x)
        >>> d = PlotData(y, x, children=[ci])
        >>> d.integrate(0, np.pi/2, return_ci=True)
        array([ 0.99940055,  0.89946049,  1.0993406 ])
        >>> np.allclose(d.integrate(0, 5, return_ci=True),
        ...    d.integrate(return_ci=True))
        True

        '''
        method = kwds.pop('method', 'trapz')
        fun = getattr(integrate, method)
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            raise NotImplementedError('integration for ndim>1 not implemented')
        # One dimensional data
        return_ci = kwds.pop('return_ci', False)
        fi, xi = self._get_fi_xi(a, b)
        res = fun(fi, xi, **kwds)
        if return_ci:
            res_ci = [child.integrate(a, b, method=method)
                      for child in self.children]
            return np.hstack((res, np.ravel(res_ci)))
        return res

    def _plot_children(self, axis, plotflag, kwds):
        axis.hold('on')
        tmp = []
        child_args = kwds.pop('plot_args_children',
                              tuple(self.plot_args_children))
        child_kwds = dict(self.plot_kwds_children).copy()
        child_kwds.update(kwds.pop('plot_kwds_children', {}))
        child_kwds['axis'] = axis
        for child in self.children:
            tmp1 = child.plot(*child_args, **child_kwds)
            if tmp1 is not None:
                tmp.append(tmp1)
        if tmp:
            return tmp
        return None

    def plot(self, *args, **kwargs):
        kwds = kwargs.copy()
        axis = kwds.pop('axis', None)
        if axis is None:
            axis = plt.gca()
        default_plotflag = self.plot_kwds.get('plotflag')
        plotflag = kwds.get('plotflag', default_plotflag)
        tmp = None
        if not plotflag and self.children is not None:
            tmp = self._plot_children(axis, plotflag, kwds)
        main_args = args if len(args) else tuple(self.plot_args)
        main_kwds = dict(self.plot_kwds).copy()
        main_kwds.update(kwds)
        main_kwds['axis'] = axis
        tmp2 = self.plotter.plot(self, *main_args, **main_kwds)
        return tmp2, tmp

    def setplotter(self, plotmethod=None):
        '''
            Set plotter based on the data type:
                data_1d, data_2d, data_3d or data_nd
        '''
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            ndim = len(self.args)
            if ndim < 2:
                msg = '''
                Unable to determine plotter-type, because len(self.args)<2.
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

    def show(self, *args, **kwds):
        self.plotter.show(*args, **kwds)

    __call__ = plot
    interpolate = eval_points


class AxisLabels:

    def __init__(self, title='', xlab='', ylab='', zlab='', **kwds):
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self.zlab = zlab

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        txt = 'AxisLabels(title={}, xlab={}, ylab={}, zlab={})'
        return txt.format(self.title, self.xlab, self.ylab, self.zlab)

    def copy(self):
        newcopy = empty_copy(self)
        newcopy.__dict__.update(self.__dict__)
        return newcopy

    def _add_title_if_fun_is_set_title(self, txt, title0, fun):
        if fun.startswith('set_title'):
            if title0.lower().strip() != txt.lower().strip():
                txt = title0 + '\n' + txt
        return txt

    def _labelfig(self, axis):
        h = []
        title0 = axis.get_title()
        for fun, txt in zip(
                ('set_title', 'set_xlabel', 'set_ylabel', 'set_ylabel'),
                (self.title, self.xlab, self.ylab, self.zlab)):
            if txt:
                txt = self._add_title_if_fun_is_set_title(txt, title0, fun)
                h.append(getattr(axis, fun)(txt))
        return h

    def labelfig(self, axis=None):
        if axis is None:
            axis = plt.gca()

        try:
            return self._labelfig(axis)
        except Exception as err:
            warnings.warn(str(err))


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

    def show(self, *args, **kwds):
        plt.show(*args, **kwds)

    def plot(self, wdata, *args, **kwds):
        axis = kwds.pop('axis', None)
        if axis is None:
            axis = plt.gca()
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
        data = transformdata_1d(x, wdata.data, plotflag)
        dataCI = getattr(wdata, 'dataCI', ())
        if dataCI:
            dataCI = transformdata_1d(x, dataCI, plotflag)
        h1 = plot1d(axis, x, data, dataCI, plotflag, *args, **kwds)
        return h1
    __call__ = plot


def set_axis(axis, f_max, trans_flag, log_scale):
    if log_scale or (trans_flag == 5 and not log_scale):
        ax = list(axis.axis())
        if trans_flag == 8 and not log_scale:
            ax[3] = 11 * np.log10(f_max)
            ax[2] = ax[3] - 40
        else:
            ax[3] = 1.15 * f_max
            ax[2] = ax[3] * 1e-4
        axis.axis(ax)


def set_plot_scale(axis, f_max, plotflag):
    scale = plotscale(plotflag)
    log_scale = False
    for dim in ['x', 'y', 'z']:
        if dim in scale:
            log_scale = True
            opt = {'{}scale'.format(dim): 'log'}
            axis.set(**opt)

    trans_flag = np.mod(plotflag // 10, 10)
    set_axis(axis, f_max, trans_flag, log_scale)


def plot1d(axis, args, data, dataCI, plotflag, *varargin, **kwds):
    h = []
    plottype = np.mod(plotflag, 10)
    if plottype == 0:  # No plotting
        return h
    fun = {1: 'plot', 2: 'step', 3: 'stem', 5: 'bar'}.get(plottype)
    if fun is not None:
        plotfun = getattr(axis, fun)
        h.extend(plotfun(args, data, *varargin, **kwds))
        if np.any(dataCI) and plottype < 3:
            axis.hold(True)
            h.extend(plotfun(args, dataCI, 'r--'))
    elif plottype == 4:
        h = axis.errorbar(args, data,
                          yerr=[dataCI[:, 0] - data,
                                dataCI[:, 1] - data],
                          *varargin, **kwds)
    elif plottype == 6:
        h = axis.fill_between(args, data, 0, *varargin, **kwds)
    elif plottype == 7:
        h = axis.plot(args, data, *varargin, **kwds)
        h.extend(axis.fill_between(args, dataCI[:, 0], dataCI[:, 1],
                                   alpha=0.2, color='r'))
    fmax1 = data.max()
    set_plot_scale(axis, fmax1, plotflag)
    return h


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

     scale = string defining plotscale valid options are:
           'linear', 'xlog', 'ylog', 'xylog', 'zlog', 'xzlog',
           'yzlog',  'xyzlog'

    Examples
    --------
    >>> for i in range(7):
    ...    plotscale(i*100)
    'linear'
    'xlog'
    'ylog'
    'xylog'
    'zlog'
    'xzlog'
    'yzlog'

    >>> plotscale(100)
    'xlog'
    >>> plotscale(1000)
    'ylog'
    >>> plotscale(10000)
    'zlog'
    >>> plotscale(1100)
    'xylog'
    >>> plotscale(11100)
    'xyzlog'

     See also
     ---------
     plotscale
    '''
    scaleId = plotflag // 100
    if scaleId > 7:
        logXscaleId = np.mod(scaleId, 10) > 0
        logYscaleId = (np.mod(scaleId // 10, 10) > 0) * 2
        logZscaleId = (np.mod(scaleId // 100, 10) > 0) * 4
        scaleId = logYscaleId + logXscaleId + logZscaleId

    scales = ['linear', 'xlog', 'ylog', 'xylog', 'zlog', 'xzlog',
              'yzlog', 'xyzlog']

    return scales[scaleId]


def plotflag2plottype_1d(plotflag):
    plottype = np.mod(plotflag, 10)
    return ['', 'plot', 'step', 'stem', 'errorbar', 'bar'][plottype]


def plotflag2transform_id(plotflag):
    transform_id = np.mod(plotflag // 10, 10)
    return ['f', '1-f',
            'cumtrapz(f)', '1-cumtrapz(f)',
            'log(f)', 'log(1-f)'
            'log(cumtrapz(f))', 'log(cumtrapz(f))',
            'log10(f)'][transform_id]


def transform_id2plotflag2(transform_id):
    return {'': 0, 'None': 0, 'f': 0, '1-f': 1,
            'cumtrapz(f)': 2, '1-cumtrapz(f)': 3,
            'log(f)': 4, 'log(1-f)': 5,
            'log(cumtrapz(f))': 6, 'log(1-cumtrapz(f))': 7,
            '10log10(f)': 8}[transform_id] * 10


def transformdata_1d(x, f, plotflag):
    transform_id = np.mod(plotflag // 10, 10)
    transform = [lambda f, x: f,
                 lambda f, x: 1 - f,
                 lambda f, x: cumtrapz(f, x),
                 lambda f, x: 1 - cumtrapz(f, x),
                 lambda f, x: np.log(f),
                 lambda f, x: np.log1p(-f),
                 lambda f, x: np.log(cumtrapz(f, x)),
                 lambda f, x: np.log1p(-cumtrapz(f, x)),
                 lambda f, x: 10*np.log10(f)
                 ][transform_id]
    return transform(f, x)


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


def _get_contour_levels(f):
    isPL = False
    PL = None
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
    return clvec, PL


def plot2d(axis, wdata, plotflag, *args, **kwds):
    f = wdata
    if isinstance(wdata.args, (list, tuple)):
        args1 = tuple((wdata.args)) + (wdata.data,) + args
    else:
        args1 = tuple((wdata.args,)) + (wdata.data,) + args

    pltfun = [None, axis.contour, axis.mesh, axis.surf, axis.waterfal,
              axis.pcolor, axis.contour, axis.contour, axis.contour,
              axis.contour, axis.contourf][plotflag]

    if plotflag in (1, 6, 7, 8, 9):
        clvec, PL = _get_contour_levels(f)

        if plotflag in [1, 8, 9]:
            h = pltfun(*args1, levels=clvec, **kwds)
        else:
            h = pltfun(*args1, **kwds)
        #  [cs hcs] = contour3(f.x{:},f.f,CL,sym);

        if plotflag in (1, 6):
            ncl = len(clvec)
            if ncl > 12:
                ncl = 12
                warnings.warn(
                    'Only the first 12 levels will be listed in table.')

            isPL = PL is not None
            clvals = PL[:ncl] if isPL else clvec[:ncl]
            unused_axcl = cltext(clvals, percent=isPL)
        else:
            axis.clabel(h)
    else:
        h = pltfun(*args1, **kwds)
        if plotflag == 10:
            axis.clabel(h)
            plt.colorbar(h)


def test_plotdata():
    plt.ioff()
    x = np.linspace(0, np.pi, 9)
    xi = np.linspace(0, np.pi, 4*9)

    d = PlotData(np.sin(x)/2, x, dataCI=[], xlab='x', ylab='sin',
                 title='sinus', plot_args=['r.'])
    di = PlotData(d.eval_points(xi, method='cubic'), xi)
    unused_hi = di.plot()
    unused_h = d.plot()
    f = di.to_cdf()

    for i in range(4):
        _ = di.plot(plotflag=i)
    d.show('hold')


if __name__ == '__main__':
    from wafo.testing import test_docstrings
    test_docstrings(__file__)
    # test_plotdata()
