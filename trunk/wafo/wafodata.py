import warnings
from plotbackend import plotbackend
from time import gmtime, strftime
import numpy as np

__all__ = ['WafoData', 'AxisLabels']

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

class WafoData(object):
    '''
    Container class for data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D, list of vectors for 2D, 3D, ...
    labels : AxisLabels
    children : list of WafoData objects

    Member methods
    --------------
    plot :
    copy :


    Example
    -------
    >>> import numpy as np
    >>> x = np.arange(-2, 2, 0.2)

    # Plot 2 objects in one call
    >>> d2 = WafoData(np.sin(x), x, xlab='x', ylab='sin', title='sinus')
    >>> h = d2.plot()

    Plot with confidence interval
    d3 = wdata(sin(x),x)
    d3 = set(d3,'dataCI',[sin(x(:))*0.9 sin(x(:))*1.2])
    plot(d3)

    See also
    --------
    wdata/plot,
    specdata,
    covdata
    '''
    def __init__(self, data=None, args=None,**kwds):
        self.data = data
        self.args = args
        self.date = now()
        self.plotter = None
        self.children = None
        self.labels = AxisLabels(**kwds)
        self.setplotter()

    def plot(self,*args,**kwds):
        tmp = None
        if self.children!=None:
            plotbackend.hold('on')
            tmp = []
            for child in self.children:
                tmp1 = child.plot(*args, **kwds)
                if tmp1 !=None:
                    tmp.append(tmp1)
            if len(tmp)==0:
                tmp = None

        tmp2 =  self.plotter.plot(self,*args,**kwds)
        return tmp2,tmp
    
    def show(self):
        self.plotter.show()

    def copy(self):
        newcopy = empty_copy(self)
        newcopy.__dict__.update(self.__dict__)
        return newcopy


    def setplotter(self,plotmethod=None):
        '''
            Set plotter based on the data type data_1d, data_2d, data_3d or data_nd
        '''

        if isinstance(self.args,(list,tuple)): # Multidimensional data
            ndim = len(self.args)
            if ndim<2:
                warnings.warn('Unable to determine plotter-type, because len(self.args)<2.')
                print('If the data is 1D, then self.args should be a vector!')
                print('If the data is 2D, then length(self.args) should be 2.')
                print('If the data is 3D, then length(self.args) should be 3.')
                print('Unless you fix this, the plot methods will not work!')
            elif ndim==2:
                self.plotter = Plotter_2d(plotmethod)
            else:
                warnings.warn('Plotter method not implemented for ndim>2')

        else: #One dimensional data
            self.plotter = Plotter_1d(plotmethod)


class AxisLabels:
    def __init__(self,title='',xlab='',ylab='',zlab='',**kwds):
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self.zlab = zlab
    def copy(self):
        newcopy = empty_copy(self)
        newcopy.__dict__.update(self.__dict__)
        return newcopy
        #lbkwds = self.labels.__dict__.copy()
        #labels = AxisLabels(**lbkwds)
        #return labels

    def labelfig(self):
        try:
            h1 = plotbackend.title(self.title)
            h2 = plotbackend.xlabel(self.xlab)
            h3 = plotbackend.ylabel(self.ylab)
            #h4 = plotbackend.zlabel(self.zlab)
            return h1,h2,h3
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

    def __init__(self,plotmethod='plot'):
        self.plotfun = None
        if plotmethod is None:
            plotmethod = 'plot'
        self.plotbackend = plotbackend
        try:
            #self.plotfun = plotbackend.__dict__[plotmethod]
            self.plotfun = getattr(plotbackend, plotmethod)
        except:
            pass
    def show(self):
        plotbackend.show()

    def plot(self,wdata,*args,**kwds):
        if isinstance(wdata.args,(list,tuple)):
            args1 = tuple((wdata.args))+(wdata.data,)+args
        else:
            args1 = tuple((wdata.args,))+(wdata.data,)+args
        h1 = self.plotfun(*args1,**kwds)
        h2 = wdata.labels.labelfig()
        return h1,h2

class Plotter_2d(Plotter_1d):
    """
    Parameters
    ----------
    plotmethod : string
        defining type of plot. Options are:
        contour (default)
        mesh
        surf
    """

    def __init__(self,plotmethod='contour'):
        if plotmethod is None:
            plotmethod = 'contour'
        super(Plotter_2d,self).__init__(plotmethod)
        #self.plotfun = plotbackend.__dict__[plotmethod]

       
def main():
    pass

if __name__ == '__main__':
    if  True: #False : #  
        import doctest
        doctest.testmod()
    else:
        main()