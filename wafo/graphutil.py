'''
Created on 20. jan. 2011

@author: pab

license BSD
'''
from __future__ import absolute_import, division
import warnings
import numpy as np
from .plotbackend import plotbackend
from matplotlib import mlab
__all__ = ['cltext', 'tallibing', 'test_docstrings']

_TALLIBING_GID = 'TALLIBING'
_CLTEXT_GID = 'CLTEXT'


def _matchfun(x, gidtxt):
    if hasattr(x, 'get_gid'):
        return x.get_gid() == gidtxt
    return False


def delete_text_object(gidtxt, figure=None, axis=None, verbose=False):
    '''
    Delete all text objects matching the gidtxt if it exists

    Parameters
    ----------
    gidtxt : string

    figure, axis : objects
        current figure and current axis, respectively.
    verbose : bool
        If true print warnings when trying to delete non-existent objects
    '''
    if figure is None:
        figure = plotbackend.gcf()
    if axis is None:
        axis = figure.gca()

    def lmatchfun(x):
        return _matchfun(x, gidtxt)

    def _delete_gid_objects(handle, gidtxt, verbose):
        objs = handle.findobj(lmatchfun)
        name = handle.__name__
        msg = "Tried to delete a non-existing {0} from {1}".format(gidtxt,
                                                                   name)
        for obj in objs:
            try:
                handle.texts.remove(obj)
            except:
                if verbose:
                    warnings.warn(msg)

    for handle in [axis, figure]:
        _delete_gid_objects(handle, gidtxt, verbose)


def cltext(levels, percent=False, n=4, xs=0.036, ys=0.94, zs=0, figure=None,
           axis=None):
    '''
    Places contour level text in the current window

    Parameters
    ----------
    levels :  vector
        contour levels or the corresponding percent which the
        contour line encloses
    percent : bool
        False if levels are the actual contour levels (default)
        True  if levels are the corresponding percent which the
            contour line encloses
    n : integer
        maximum N digits of precision (default 4)
    figure, axis : objects
        current figure and current axis, respectively.
        default figure = plotbackend.gcf(),
                axis = plotbackend.gca()

    Returns
    -------
    h       = handles to the text objects.


    Notes
    -----
    CLTEXT creates text objects in the current figure and prints
          "Level curves at:"        if percent is False and
          "Level curves enclosing:" otherwise
    and the contour levels or percent.

    The handles to the lines of text may also be found by
          h  = findobj(gcf,'gid','CLTEXT','type','text');
          h  = findobj(gca,'gid','CLTEXT','type','text');
    To make the text objects follow the data in the axes set the units
    for the text objects 'data' by
          set(h,'unit','data')

    Examples
    --------
    >>> import wafo.graphutil as wg
    >>> import wafo.demos as wd
    >>> import pylab as plt
    >>> x,y,z  = wd.peaks();
    >>> h = plt.contour(x,y,z)
    >>> h = wg.cltext(h.levels)
    >>> plt.show()
    '''
    # TODO : Make it work like legend does (but without the box): include
    # position options etc...
    if figure is None:
        figure = plotbackend.gcf()
    if axis is None:
        axis = figure.gca()

    clevels = np.atleast_1d(levels)

    axpos = axis.get_position()
    xint = axpos.intervalx
    yint = axpos.intervaly

    xss = xint[0] + xs * (xint[1] - xint[0])
    yss = yint[0] + ys * (yint[1] - yint[0])

    # delete cltext object if it exists
    delete_text_object(_CLTEXT_GID, axis=axis)

    charHeight = 1.0 / 33.0
    delta_y = charHeight

    if percent:
        titletxt = 'Level curves enclosing:'
    else:
        titletxt = 'Level curves at:'

    format_ = '%0.' + ('%d' % n) + 'g\n'

    cltxt = ''.join([format_ % level for level in clevels.tolist()])

    titleProp = dict(gid=_CLTEXT_GID, horizontalalignment='left',
                     verticalalignment='center', fontweight='bold', axes=axis)

    ha1 = figure.text(xss, yss, titletxt, **titleProp)

    yss -= delta_y
    txtProp = dict(gid=_CLTEXT_GID, horizontalalignment='left',
                   verticalalignment='top', axes=axis)

    ha2 = figure.text(xss, yss, cltxt, **txtProp)
    plotbackend.draw_if_interactive()
    return ha1, ha2


def tallibing(*args, **kwds):
    '''
    TALLIBING  Display numbers on field-plot

    CALL h=tallibing(x,y,n,size,color)

    Parameters
    ----------
    x, y : array
        position matrices
    n : array
        corresponding matrix of the values to be written
             (non-integers are rounded)
    mid_points : bool (default True)
        data-point-positions are in the middle of bins instead of the corners
    size : int, (default=8)
        font size (optional)
    color : str, (default='white')
        color of text (optional)

    Returns
    -------
    h : list
        handles to TEXT objects

    TALLIBING writes the numbers in a 2D array as text at the positions
    given by the x and y coordinate matrices.
    When plotting binned results, the number of datapoints in each
    bin can be written on the bins in the plot.

    Example
    -------
    >>> import wafo.graphutil as wg
    >>> import wafo.demos as wd
    >>> [x,y,z] = wd.peaks(n=20)
    >>> h0 = wg.pcolor(x,y,z)
    >>> h1 = wg.tallibing(x,y,z)

    See also
    --------
    text
    '''

    axis = kwds.pop('axis', None)
    if axis is None:
        axis = plotbackend.gca()

    x, y, n = _parse_data(*args, **kwds)
    if mlab.isvector(x) or mlab.isvector(y):
        x, y = np.meshgrid(x, y)

    n = np.round(n)

    # delete tallibing object if it exists
    delete_text_object(_TALLIBING_GID, axis=axis)

    txtProp = dict(gid=_TALLIBING_GID, size=8, color='w',
                   horizontalalignment='center',
                   verticalalignment='center', fontweight='demi', axes=axis)

    txtProp.update(**kwds)
    h = []
    for xi, yi, ni in zip(x.ravel(), y.ravel(), n.ravel()):
        if ni:
            h.append(axis.text(xi, yi, str(ni), **txtProp))
    plotbackend.draw_if_interactive()
    return h


def _parse_data(*args, **kwds):
    nargin = len(args)
    data = np.atleast_2d(args[-1]).copy()
    M, N = data.shape
    if nargin == 1:
        x = np.arange(N)
        y = np.arange(M)
    elif nargin == 3:
        x, y = np.atleast_1d(*args[:-1])
        if min(x.shape) != 1:
            x = x[0]
        if min(y.shape) != 1:
            y = y[:, 0]
    else:
        raise ValueError(
            'Requires 3 or 1 in arguments! (x,y,data) or (data)')
    if kwds.pop('mid_point', True):
        xx = _find_mid_points(x)
        yy = _find_mid_points(y)
        return xx, yy, data
    return x, y, data

pcolor = getattr(plotbackend, 'pcolor', None)
pcolormesh = getattr(plotbackend, 'pcolormesh', None)


def _find_mid_points(x):
    ''' Return points half way between all values of X and outside the
     endpoints. The outer limits have same distance from X's endpoints as
     the limits just inside.
    '''
    dx = np.diff(x) * 0.5
    dx = np.hstack((dx, dx[-1]))
    return x + dx


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

if __name__ == '__main__':
    test_docstrings()
