'''
Created on 20. jan. 2011

@author: pab

license BSD
'''
from __future__ import division
import warnings
import numpy as np
from wafo.plotbackend import plotbackend
from matplotlib import mlab
__all__ = ['cltext', 'epcolor', 'tallibing', 'test_docstrings']

_TALLIBING_GID = 'TALLIBING'
_CLTEXT_GID = 'CLTEXT'

def _matchfun(x, gidtxt): 
    if hasattr(x, 'get_gid'):
        return x.get_gid() == gidtxt
    return False

def delete_object(gidtxt, cf=None, ca=None, verbose=False):
    '''
    Delete all objects matching the gidtxt if it exists
    '''
    if cf is None:
        cf = plotbackend.gcf()
    if ca is None:
        ca = plotbackend.gca()
    lmatchfun = lambda x : _matchfun(x, gidtxt) 
    objs = plotbackend.findobj(cf, lmatchfun)
    if len(objs):
        for i in objs:
            try:
                ca.texts.remove(i)
            except:
                if verbose:
                    warnings.warn('Tried to delete a non-existing %s from axes' % gidtxt)
            try:
                cf.texts.remove(i)
            except:
                if verbose:
                    warnings.warn('Tried to delete a non-existing %s from figure' % gidtxt)
                
def cltext(levels, percent=False, n=4, xs=0.036, ys=0.94, zs=0):
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
    # TODO : Make it work like legend does (but without the box): include position options etc...
    clevels = np.atleast_1d(levels)

    cax = plotbackend.gca()
    axpos = cax.get_position()
    xint = axpos.intervalx
    yint = axpos.intervaly
    
    xss = xint[0] + xs * (xint[1] - xint[0])
    yss = yint[0] + ys * (yint[1] - yint[0])
    
    # delete cltext object if it exists 
    delete_object(_CLTEXT_GID, ca=cax)
     
    charHeight = 1.0 / 33.0
    delta_y = charHeight
    
    if percent:
        titletxt = 'Level curves enclosing:';
    else:
        titletxt = 'Level curves at:';

    format_ = '%0.' + ('%d' % n) + 'g\n'
     
    cltxt = ''.join([format_ % level for level in clevels.tolist()])
    
    titleProp = dict(gid=_CLTEXT_GID, horizontalalignment='left',
                     verticalalignment='center', fontweight='bold', axes=cax) # 
    ha1 = plotbackend.figtext(xss, yss, titletxt, **titleProp)
    
    yss -= delta_y;
    txtProp = dict(gid=_CLTEXT_GID, horizontalalignment='left',
                     verticalalignment='top', axes=cax)
    
    ha2 = plotbackend.figtext(xss, yss, cltxt, **txtProp)
        
    return ha1, ha2

def tallibing(x, y, n, **kwds):
    '''
    TALLIBING  Display numbers on field-plot
    
    CALL h=tallibing(x,y,n,size,color)
    
    x,y    = position matrices
    n      = the corresponding matrix of the values to be written
             (non-integers are rounded)
    size   = font size (optional) (default=8)
    color  = color of text (optional) (default='white')
    h      = column-vector of handles to TEXT objects
    
    TALLIBING writes the numbers in a 2D array as text at the positions 
    given by the x and y coordinate matrices.
    When plotting binned results, the number of datapoints in each
    bin can be written on the bins in the plot.
    
    Example
    ------- 
    >>> import wafo.graphutil as wg
    >>> import wafo.demos as wd
    >>> [x,y,z] = wd.peaks(n=20) 
    >>> wg.epcolor(x,y,z)
    >>> wg.tallibing(x,y,z)
      
    pcolor(x,y,z); shading interp; 
    
    See also
    --------
    text
    '''
    x, y, n = np.atleast_1d(x, y, n)
    if mlab.isvector(x) or mlab.isvector(y): 
        x, y = np.meshgrid(x,y)
    
    cax = plotbackend.gca()
    
    x = x.ravel()
    y = y.ravel()
    n = n.ravel()
    n = np.round(n)
    
    # delete tallibing object if it exists 
    delete_object(_TALLIBING_GID, ca=cax)
     
    txtProp = dict(gid=_TALLIBING_GID, size=8, color='w', horizontalalignment='center',
                     verticalalignment='center', fontweight='demi', axes=cax)
     
    txtProp.update(**kwds)
    h = []
    for xi,yi, ni in zip(x,y,n):
        if ni:
            h.append(plotbackend.text(xi, yi, str(ni), **txtProp))
    return h

def epcolor(*args, **kwds):    
    '''
    Pseudocolor (checkerboard) plot with mid-bin positioning.
    
     h = epcolor(x,y,data)
     
     
     [x,y]= the axes corresponding to the data-positions. Vectors or
            matrices. If omitted, giving only data-matrix as inargument, the
            matrix-indices are used as axes.
     data = data-matrix
    
     EPCOLOR make a checkerboard plot where the data-point-positions are in
     the middle of the bins instead of in the corners, and the last column
     and row of data are used.
    
  
     Example:      
     >>> import wafo.demos as wd
     >>> import wafo.graphutil as wg
     >>> x, y, z = wd.peaks(n=20)
     >>> wg.epcolor(x,y,z)
    
     See also 
     --------
     pylab.pcolor
    '''
    midbin = kwds.pop('midbin', True)
    if not midbin:
        return plotbackend.pcolor(*args,**kwds)

    nargin = len(args)
    data = np.atleast_2d(args[-1]).copy()
    M, N = data.shape
    if nargin==1:
        x = np.arange(N)
        y = np.arange(M)
    elif nargin==3:
        x, y = np.atleast_1d(*args[:-1])
        if min(x.shape)!=1: 
            x = x[0]
        if min(y.shape)!=1:
            y = y[:,0]
    else:
        raise ValueError('pcolor takes 3 or 1 inarguments! (x,y,data) or (data)')

    xx = _findbins(x)
    yy = _findbins(y)
    return plotbackend.pcolor(xx, yy, data, **kwds)
 
def _findbins(x):
    ''' Return points half way between all values of X _and_ outside the
     endpoints. The outer limits have same distance from X's endpoints as
     the limits just inside.
    '''
    dx = np.diff(x) * 0.5
    dx = np.hstack((dx, dx[-1])) 
    return np.hstack((x[0] - dx[0], x + dx))

 
def test_docstrings():
    import doctest
    doctest.testmod()
    
if __name__ == '__main__':
    test_docstrings()