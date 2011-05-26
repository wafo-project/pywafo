'''
Created on 20. jan. 2011

@author: pab

license BSD
'''
from __future__ import division
import warnings
import numpy as np
from wafo.plotbackend import plotbackend

__all__ = ['cltext']

def cltext(levels, percent=False, n=4, xs=0.036, ys=0.94, zs=0):
    '''
    Places contour level text in the current window
              
    Parameters
    ----------
    levels  = vector of contour levels or the corresponding percent which the
              contour line encloses
    percent = 0 if levels are the actual contour levels (default)
              1 if levels are the corresponding percent which the
                contour line encloses
    n       = maximum N digits of precision (default 4)
    Returns
    h       = handles to the text objects.
    CLTEXT creates text objects in the current figure and prints 
          "Level curves at:"        if percent is False and
          "Level curves enclosing:" otherwise
    and the contour levels or percent.
    
    NOTE: 
    -The handles to the lines of text may also be found by 
          h  = findobj(gcf,'gid','CLTEXT','type','text');
          h  = findobj(gca,'gid','CLTEXT','type','text');
    -To make the text objects follow the data in the axes set the units 
    for the text objects 'data' by    
          set(h,'unit','data')
    
    Examples:
    >>> from wafo.integrate import peaks
    >>> import pylab as plt
    >>> x,y,z  = peaks();
    >>> h = plt.contour(x,y,z)
    >>> h = cltext(h.levels)
    >>> plt.show()
    
    data = rndray(1,2000,2); f = kdebin(data,{'kernel','epan','L2',.5,'inc',128});
    contour(f.x{:},f.f,f.cl),cltext(f.pl,1)
    
    See also
    pdfplot
    '''
    # TODO : Make it work like legend does (but without the box): include position options etc...
    clevels = np.atleast_1d(levels)
    _CLTEXT_TAG = 'CLTEXT'
    cax = plotbackend.gca()
    axpos = cax.get_position()
    xint = axpos.intervalx
    yint = axpos.intervaly
    
    xss = xint[0] + xs * (xint[1] - xint[0])
    yss = yint[0] + ys * (yint[1] - yint[0])
    
    cf = plotbackend.gcf() # get current figure
    #% delete cltext object if it exists 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def matchfun(x): 
        if hasattr(x, 'get_gid'):
            return x.get_gid() == _CLTEXT_TAG
        return False
    h_cltxts = plotbackend.findobj(cf, matchfun); 
    if len(h_cltxts):
        for i in h_cltxts:
            try:
                cax.texts.remove(i)
            except:
                warnings.warn('Tried to delete a non-existing CL-text')
     
            try:
                cf.texts.remove(i)
            except:
                warnings.warn('Tried to delete a non-existing CL-text')
     
    charHeight = 1.0 / 33.0
    delta_y = charHeight
    
    if percent:
        titletxt = 'Level curves enclosing:';
    else:
        titletxt = 'Level curves at:';

    format = '%0.' + ('%d' % n) + 'g\n'
     
    cltxt = ''.join([format % level for level in clevels.tolist()])
    
    titleProp = dict(gid=_CLTEXT_TAG, horizontalalignment='left',
                     verticalalignment='center', fontweight='bold', axes=cax) # 
    ha1 = plotbackend.figtext(xss, yss, titletxt, **titleProp)
    
    yss -= delta_y;
    txtProp = dict(gid=_CLTEXT_TAG, horizontalalignment='left',
                     verticalalignment='top', axes=cax)
    
    ha2 = plotbackend.figtext(xss, yss, cltxt, **txtProp)
        
    return ha1, ha2
if __name__ == '__main__':
    pass