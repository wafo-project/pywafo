'''
Module FIG
------------
Module for manipulating windows/figures created using
pylab or enthought.mayavi.mlab on the windows platform.

Figure manipulation involves
maximization, minimization, hiding, closing, stacking or tiling.

This module assumes that the figures are uniquely numbered in the following way:
Figure 1
Figure 2
....
or
TVTK scene 1
TVTK scene 2
TVTK scene 3
...

Example
-------
>>> import pylab as p
>>> import wafo.fig as fig
>>> for ix in range(6):
...     f = p.figure(ix)
>>> fig.stack('all')
>>> fig.stack(1,2)
>>> fig.hide(1)
>>> fig.restore(1)
>>> fig.tile()
>>> fig.pile()
>>> fig.maximize(4)
>>> fig.close('all')
'''

#!/usr/bin/env python
from __future__ import division
# import win32api
import win32gui
import win32con
import msvcrt
import numpy

__all__ = ['close', 'cycle', 'hide', 'keep', 'maximize', 'minimize', 'pile',
           'restore', 'stack', 'tile']

# Figure format strings to recognize in window title
_FIG_FORMATS = ('Figure', 'TVTK Scene', 'Chaco Plot Window: Figure')
_SCREENSIZE = None


def _getScreenSize(wnds):
    ''' Return screen size X,Y,W,H

    Returns
    --------
    X Specifies the new position of the left side of the screen.
    Y Specifies the new position of the top of the screen.
    W Specifies the new width of the screen.
    H Specifies the new height of the screen.

    Parameters
    ----------
    wnds: list of handles to open figures
      (Note: only needed the first time)

    '''

    global _SCREENSIZE
    if _SCREENSIZE is None:
        MoveWindow = win32gui.MoveWindow
        GetWindowRect = win32gui.GetWindowRect
        SW_MAXIMIZE = win32con.SW_SHOWMAXIMIZED
        hwnd = [wnds[0]]
        pos = list(GetWindowRect(hwnd[0]))
        pos[3] -= pos[1]
        pos[2] -= pos[0]
        _show_windows(hwnd, SW_MAXIMIZE)
        _SCREENSIZE = list(GetWindowRect(hwnd[0]))  # Screen size
        _SCREENSIZE[3] -= _SCREENSIZE[1]
        _SCREENSIZE[2] -= _SCREENSIZE[0]
        _SCREENSIZE = tuple(_SCREENSIZE)

        MoveWindow(hwnd[0], pos[0], pos[1], pos[2], pos[3], 1)

    return list(_SCREENSIZE)


def _windowEnumerationHandler(hwnd, resultList):
    '''Pass to win32gui.EnumWindows() to generate list of window handle,
    window text tuples.
    '''

    resultList.append((hwnd, win32gui.GetWindowText(hwnd)))


def _findTopWindows(wantedTitle=None):
    ''' Return list of window handle and window title tuples

    @param wantedTitle:
    '''
    topWindows = []
    win32gui.EnumWindows(_windowEnumerationHandler, topWindows)
    if wantedTitle is None:
        return topWindows
    else:
        return [(hwnd, windowTxt) for hwnd, windowTxt in topWindows
                if windowTxt.startswith(wantedTitle)]


def findallfigs():
    '''
    Return list of all figure numbers
    '''
    figs = []
    global _FIG_FORMATS
    for wantedTitle in _FIG_FORMATS:
        windowList = _findTopWindows(wantedTitle)
        for unused_hwnd, wndTitle in windowList:
            try:
                fig = int(wndTitle.split()[-1])
                figs.append(fig)
            except:
                pass
    figs.sort()
    return figs


def _figparse(*args):
    figs = []
    for arg in args:
        if isinstance(arg, (list, tuple, set)):
            for val in arg:
                figs.append(int(val))
        elif isinstance(arg, int):
            figs.append(arg)
        elif arg == 'all':
            figs = 'all'
            break
        else:
            raise TypeError('Only integers arguments accepted!')
            # raise TypeError('Unrecognized argument type (%s)!'%type(arg))

    if len(figs) == 0 or figs == 'all':
        figs = findallfigs()
    return figs


def _fig2wnd(figs):
    ''' Find figure handle from figure number
    '''
    FindWindow = win32gui.FindWindow
    wnd_handles = []
    global _FIG_FORMATS
    for fig in figs:
        for format_ in _FIG_FORMATS:
            winTitle = format_ + ' %d' % fig
            hwnd = FindWindow(None, winTitle)
            if not hwnd == 0:
                wnd_handles.append(hwnd)
    return wnd_handles


def _show_figure(figs, cmdshow):
    ''' sets the specified figure's show state.

    @param figs: vector for figure numbers
    @param cmdshow: one of following commands:
    SW_FORCEMINIMIZE:   Minimizes a window, even if the thread that owns the
                        window is not responding. This flag should only be used
                        when minimizing windows from a different thread.
    SW_HIDE:            Hides the window and activates another window.
    SW_MAXIMIZE:    Maximizes the specified window.
    SW_MINIMIZE:    Minimizes the specified window and activates the next
                    top-level window in the Z order.
    SW_RESTORE:     Activates and displays the window. If the window is
                    minimized or maximized, the system restores it to its
                    original size and position. An application should
                    specify this flag when restoring a minimized window.
    SW_SHOW:        Activates the window and displays it in its current size
                    and position.
    SW_SHOWDEFAULT: Sets the show state based on the SW_ value specified in the
                    STARTUPINFO structure passed to the CreateProcess function
                    by the program that started the application.
    SW_SHOWMAXIMIZED: Activates the window and displays it as a maximized
                    window.
    SW_SHOWMINIMIZED: Activates the window and displays it as a minimized
                    window.
    SW_SHOWMINNOACTIVE:  Displays the window as a minimized window. This value
                    is similar to SW_SHOWMINIMIZED, except the window is not
                    activated.
    SW_SHOWNA:  Displays the window in its current size and position. This
            value is similar to SW_SHOW, except the window is not activated.
    SW_SHOWNOACTIVATE: Displays a window in its most recent size and position.
                    This value is similar to SW_SHOWNORMAL, except the window
                    is not actived.
    SW_SHOWNORMAL: Activates and displays a window. If the window is minimized
                or maximized, the system restores it to its original size and
                position. An application should specify this flag when
                displaying the window for the first time.
    '''
    BringWindowToTop = win32gui.BringWindowToTop
    FindWindow = win32gui.FindWindow
    ShowWindow = win32gui.ShowWindow
    global _FIG_FORMATS
    for fig in figs:
        for format_ in _FIG_FORMATS:
            winTitle = format_ + ' %d' % fig
            hwnd = FindWindow(None, winTitle)
            if not hwnd == 0:
                # ShowWindow(hwnd,cmdshow)
                BringWindowToTop(hwnd)
                ShowWindow(hwnd, cmdshow)


def _show_windows(wnds, cmdshow):
    ''' sets the specified window's show state.

    @param wnds: list of window handles numbers
    @param cmdshow: one of following commands:
    SW_FORCEMINIMIZE:   Minimizes a window, even if the thread that owns the
                        window is not responding. This flag should only be used
                        when minimizing windows from a different thread.
    SW_HIDE:            Hides the window and activates another window.
    SW_MAXIMIZE:    Maximizes the specified window.
    SW_MINIMIZE:    Minimizes the specified window and activates the next
                    top-level window in the Z order.
    SW_RESTORE:    Activates and displays the window. If the window is
                minimized or maximized, the system restores it to its original
                size and position. An application should specify this flag when
                restoring a minimized window.
    SW_SHOW:    Activates the window and displays it in its current size and
                position.
    SW_SHOWDEFAULT:    Sets the show state based on the SW_ value specified in
                        the STARTUPINFO structure passed to the CreateProcess
                        function by the program that started the application.
    SW_SHOWMAXIMIZED: Activates the window and displays it as a maximized
                        window.
    SW_SHOWMINIMIZED:    Activates the window and displays it as a minimized
                        window.
    SW_SHOWMINNOACTIVE:    Displays the window as a minimized window. This
                        value is similar to SW_SHOWMINIMIZED, except the window
                        is not activated.
    SW_SHOWNA:    Displays the window in its current size and position. This
            value is similar to SW_SHOW, except the window is not activated.
    SW_SHOWNOACTIVATE:  Displays a window in its most recent size and position.
                        This value is similar to SW_SHOWNORMAL, except the
                        window is not actived.
    SW_SHOWNORMAL: Activates and displays a window. If the window is minimized
                or maximized, the system restores it to its original size and
                position. An application should specify this flag when
                displaying the window for the first time.
    '''

    ShowWindow = win32gui.ShowWindow
    BringWindowToTop = win32gui.BringWindowToTop
    for hwnd in wnds:
        if not hwnd == 0:
            # ShowWindow(hwnd,cmdshow)
            BringWindowToTop(hwnd)
            ShowWindow(hwnd, cmdshow)


def keep(*figs):
    ''' Keeps figure windows of your choice and closes the rest.

    Parameters
    ----------
    figs : list of integers specifying which figures to keep.

    Example:
    --------
    # keep only figures 1,2,3,5 and 7
    >>> import pylab as p
    >>> import wafo.fig as fig
    >>> for ix in range(10):    f = p.figure(ix)
    >>> fig.keep( range(1,4),  5, 7)

    or
        fig.keep([range(1,4),  5, 7])
    >>> fig.close()

    See also
    --------
    pyfig.close
    '''
    figs2keep = []
    for fig in figs:
        if isinstance(fig, (list, tuple, set)):
            for val in fig:
                figs2keep.append(int(val))
        elif isinstance(fig, int):
            figs2keep.append(fig)
        else:
            raise TypeError('Only integers arguments accepted!')

    if len(figs2keep) > 0:
        allfigs = set(findallfigs())

# Remove figure handles in the "keep" list
        figs2delete = allfigs.difference(figs2keep)
        close(figs2delete)
        # for fig in figs2delete:
        #    close(fig)


def close(*figs):
    """  Close figure window(s)

    Parameters
    ----------
    figs : list of integers or string
        specifying which figures to close (default 'all').

    Examples
    --------
    >>> import pylab as p
    >>> import wafo.fig as fig
    >>> for ix in range(5):    f = p.figure(ix)
    >>> fig.close(3,4)   # close figure 3 and 4
    >>> fig.close('all') # close all remaining figures

    or even simpler
    fig.close() # close all remaining figures

    See also
    --------
    pyfig.keep
    """
    figlist = _figparse(*figs)
    wnds = _fig2wnd(figlist)
    for wnd in wnds:
        win32gui.SendMessage(wnd, win32con.WM_CLOSE, 0, 0)


def restore(*figs):
    '''Restore figures window size and position to its default value.

    Parameters
    ----------
    figs : list of integers or string
        specifying which figures to restor (default 'all').

    Description
    -----------
    RESTORE Activates and displays the window. If the window is minimized
    or maximized, the system restores it to its original size and position.

    Examples
    ---------
      >>> import pylab as p
      >>> import wafo.fig as fig
      >>> for ix in range(5):    f = p.figure(ix)
      >>> fig.restore('all')   #Restores all figures
      >>> fig.restore()        #same as restore('all')
      >>> fig.restore(p.gcf().number)   #Restores the current figure
      >>> fig.restore(3)       #Restores figure 3
      >>> fig.restore([2, 4])   #Restores figures 2 and 4

       or alternatively
          fig.restore(2, 4)
      >>> fig.close()

    See also
    --------
    fig.close,
    fig.keep
    '''

    figlist = _figparse(*figs)

    SW_RESTORE = win32con.SW_SHOWNORMAL  # SW_RESTORE
    _show_figure(figlist, SW_RESTORE)


def hide(*figs):
    '''hide figure(s) window size

    Parameters
    ----------
    figs : list of integers or string
        specifying which figures to hide (default 'all').

    Examples:
    --------
     >>> import wafo.fig as fig
     >>> import pylab as p
     >>> for ix in range(5):    f = p.figure(ix)
     >>> fig.hide('all')   #hides all unhidden figures
     >>> fig.hide()        #same as hide('all')
     >>> fig.hide(p.gcf().number)   #hides the current figure
     >>> fig.hide(3)       #hides figure 3
     >>> fig.hide([2, 4])   #hides figures 2 and 4

     or alternatively
         fig.hide(2, 4)
     >>> fig.close()

     See also
     --------
     pyfig.cycle,
     pyfig.keep,
     pyfig.restore
    '''

    figlist = _figparse(*figs)
    SW_HIDE = win32con.SW_HIDE
    # SW_hide = win32con.SW_hide
    _show_figure(figlist, SW_HIDE)


def minimize(*figs):
    '''Minimize figure(s) window size

    Parameters
    ----------
    figs : list of integers or string
        specifying which figures to minimize (default 'all').

    Examples:
    ---------
     >>> import wafo.fig as fig
     >>> import pylab as p
     >>> for ix in range(5):    f = p.figure(ix)
     >>> fig.minimize('all')   #Minimizes all unhidden figures
     >>> fig.minimize()        #same as minimize('all')
     >>> fig.minimize(p.gcf().number)   #Minimizes the current figure
     >>> fig.minimize(3)       #Minimizes figure 3
     >>> fig.minimize([2, 4])   #Minimizes figures 2 and 4

     or alternatively
         fig.minimize(2, 4)
     >>> fig.close()

     See also
     --------
     pyfig.cycle,
     pyfig.keep,
     pyfig.restore
    '''

    figlist = _figparse(*figs)
    SW_MINIMIZE = win32con.SW_SHOWMINIMIZED
    _show_figure(figlist, SW_MINIMIZE)


def maximize(*figs):
    '''Maximize figure(s) window size

    Parameters
    ----------
    figs : list of integers or string
        specifying which figures to maximize (default 'all').

    Examples:
    ---------
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(5):    f = p.figure(ix)
     >>> fig.maximize('all')   #Maximizes all unhidden figures
     >>> fig.maximize()        #same as maximize('all')
     >>> fig.maximize(p.gcf().number)   #Maximizes the current figure
     >>> fig.maximize(3)       #Maximizes figure 3
     >>> fig.maximize([2, 4])   #Maximizes figures 2 and 4

     or alternatively
         fig.maximize(2, 4)
     >>> fig.close()

     See also
     --------
     pyfig.cycle,
     pyfig.keep,
     pyfig.restore
    '''

    figlist = _figparse(*figs)
    SW_MAXIMIZE = win32con.SW_SHOWMAXIMIZED
    # SW_MAXIMIZE = win32con.SW_MAXIMIZE
    _show_figure(figlist, SW_MAXIMIZE)


def pile(*figs):
    ''' Pile figure windows

    Parameters
    ----------
    figs : list of integers or string
        specifying which figures to pile (default 'all').

    Description
    -------------
       PILE piles all open figure windows on top of eachother
       with complete overlap. PILE(FIGS) can be used to specify which
       figures that should be piled. Figures are not sorted when specified.

     Example:
     --------
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(5): f = p.figure(ix)
     >>> fig.pile()                # pile all open figures
     >>> fig.pile(range(1,4), 5, 7)  # pile figure 1,2,3,5 and 7
     >>> fig.close()

     See also
     --------
     pyfig.cycle, pyfig.keep, pyfig.maximize, pyfig.restore,
             pyfig.stack, pyfig.tile
    '''

    figlist = _figparse(*figs)
    wnds = _fig2wnd(figlist)
    numfigs = len(wnds)
    if numfigs > 0:
        pos = _getScreenSize(wnds)
        pos[3] = int(pos[3] / 2)
        pos[2] = int(pos[2] / 2.5)
        pos[1] = int(pos[3] / 2)
        pos[0] = int(pos[2] / 2)
        BringWindowToTop = win32gui.BringWindowToTop
        MoveWindow = win32gui.MoveWindow
        for wnd in wnds:
            MoveWindow(wnd, pos[0], pos[1], pos[2], pos[3], 1)
            BringWindowToTop(wnd)


def stack(*figs):
    ''' Stack figure windows

    Parameters
    ----------
    figs : list of integers or string
        specifying which figures to stack (default 'all').

    Description
    -----------
       STACK stacks all open figure windows on top of eachother
       with maximum overlap. STACK(FIGS) can be used to specify which
       figures that should be stacked. Figures are not sorted when specified.

     Example:
     --------
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(5):  f = p.figure(ix)
     >>> fig.stack()                # stack all open figures
     >>> fig.stack(range(1,4), 5, 7)  # stack figure 1,2,3,5 and 7
     >>> fig.close()

     See also
     --------
      pyfig.cycle, pyfig.keep, pyfig.maximize, pyfig.restore,
             pyfig.pile, pyfig.tile
    '''

    figlist = _figparse(*figs)
    wnds = _fig2wnd(figlist)
    numfigs = len(wnds)
    if numfigs > 0:
        screenpos = _getScreenSize(wnds)

        maxfigs = numpy.fix(screenpos[3] / 20)

        if (numfigs > maxfigs):            # figure limit check
            print(' More than %d requested ' % maxfigs)
            return
        BringWindowToTop = win32gui.BringWindowToTop
        MoveWindow = win32gui.MoveWindow
        GetWindowRect = win32gui.GetWindowRect
#
#
# tile figures by postiion
# Location (1,1) is at bottom left corner
#
        # print('Screensz = ',screenpos)
        for iy in range(numfigs):
            pos = list(GetWindowRect(wnds[iy]))
            pos[3] -= pos[1]
            pos[2] -= pos[0]
            # print('[x, y, w, h] = ', pos)
            ypos = screenpos[1] + iy * 20
            # int(screenpos[3] - iy*20 -pos[3] -70) # figure location (row)
            xpos = int(iy * 5 + 15 + screenpos[0])  # figure location (column)
            MoveWindow(wnds[iy], xpos, ypos, pos[2], pos[3], 1)
            BringWindowToTop(wnds[iy])


def tile(*figs, **kwds):
    ''' Tile figure windows.

    Parameters
    ----------
    figs : list of integers or string
        specifying which figures to tile (default 'all').
    kwds : dict with key pairs
        specifying how many pairs of figures that are tiled at a time

    Description
    -----------
       TILE places all open figure windows around on the screen with no
       overlap. TILE(FIGS) can be used to specify which figures that
       should be tiled. Figures are not sorted when specified.

     Example:
     --------
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(5):  f = p.figure(ix)
     >>> fig.tile()             # tile all open figures
     >>> fig.tile( range(1,4), 5, 7)    # tile figure 1,2,3,5 and 7
     >>> fig.tile(range(1,11), pairs=2) # tile figure 1 to 10 two at a time
     >>> fig.tile(range(1,11), pairs=3) # tile figure 1 to 10 three at a time
     >>> fig.close()

     See also
     --------
     pyfig.cycle, pyfig.keep, pyfig.maximize, pyfig.minimize
     pyfig.restore, pyfig.pile, pyfig.stack

    '''
    figlist = _figparse(*figs)
    wnds = _fig2wnd(figlist)

    nfigs = len(wnds)  # Number of windows.

    if nfigs > 0:
        nfigspertile = kwds.get('pairs', nfigs)

        ceil = numpy.ceil
        sqrt = numpy.sqrt
        maximum = numpy.maximum

        nlayers = int(ceil(nfigs / nfigspertile))

        nh = int(ceil(sqrt(nfigspertile)))  # Number of figures horisontally.
        nv = int(ceil(nfigspertile / nh))  # Number of figures vertically.

        nh = maximum(nh, 2)
        nv = maximum(nv, 2)

# Get the screen size.
# --------------------

        BringWindowToTop = win32gui.BringWindowToTop
        MoveWindow = win32gui.MoveWindow
        screenpos = _getScreenSize(wnds)
        # scrdim = win32gui.GetWindowPlacement(h)[4]

        scrwid = screenpos[2]               # Screen width.
        scrhgt = screenpos[3]               # Screen height.

#
# The elements in the vector specifying the position.
# 1 - Window left position
# 2 - Window top position
# 3 - Window horizontal size
# 4 - Window vertical size
#  ------------------------------------------
        hspc = 10            # Horisontal space.
        topspc = 20            # Space above top figure.
        medspc = 10            # Space between figures.
        botspc = 20            # Space below bottom figure.

        # print('scrwid = %d' % scrwid)
        figwid = (scrwid - (nh + 1) * hspc) / nh
        # print('figwid = %d' % figwid)
        fighgt = (scrhgt - (topspc + botspc) - (nv - 1) * medspc) / nv

        figwid = int(numpy.round(figwid))
        fighgt = int(numpy.round(fighgt))

#
# Put the figures where they belong.
# -----------------------------------
        idx = 0
        for unused_ix in range(nlayers):
            for row in range(nv):
                for col in range(nh):
                    if (row) * nh + col < nfigspertile:
                        if idx < nfigs:
                            figlft = int(screenpos[0] + (col + 1) * hspc +
                                         col * figwid)
                            figtop = int(screenpos[1] + topspc +
                                         row * (fighgt + medspc))
            # figpos = [ figlft figtop figwid fighgt ];    # Figure position.
            # fighnd = FindWindow(0,'Figure %d' % figs[idx]) # Figure handle.
                            fighnd = wnds[idx]
                            MoveWindow(fighnd, figlft, figtop, figwid, fighgt,
                                       1)     # Set position.
                            BringWindowToTop(fighnd)
                            # figure(figs[idx])     # Raise figure.
                        idx += 1


def cycle(*figs, **kwds):
    ''' Cycle through figure windows.

    Parameters
    ----------
    figs : list of integers or string
        specifying which figures to cycle through (default 'all').
    kwds : dict with the following keys
        pairs : number of figures to cycle in pairs (default 1)
        maximize: If True maximize figure when viewing (default False)

    Description
    -----------
       CYCLE brings up all open figure in ascending order and pauses after
       each figure. Press escape to quit cycling, backspace to display previous
       figure(s) and press any other key to display next figure(s)
       When done, the figures are sorted in ascending order.

       CYCLE(maximize=True) does the same thing, except that figures are
                           maximized.
       CYCLE(pairs=2)   cycle through all figures in pairs of 2.

     Examples:
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(4): f = p.figure(ix)

     fig.cycle(range(3))               #Cycle trough figure 0 to 2
     fig.cycle(range(3) maximize=True) #Cycle trough figure 1 to 3 with figs
                                       # maximized
     fig.cycle()                       #Cycle through all figures one at a time
     fig.tile(pairs=2)
     fig.cycle(pairs=2)                #Cycle through all figures two at a time
     >>> fig.close()

    See also
    --------
     pyfig.keep, pyfig.maximize, pyfig.restore, pyfig.pile,
            pyfig.stack, pyfig.tile
    '''
    # TODO : display is not updated for each cycle => function is useless
    figlist = _figparse(*figs)
    wnds = _fig2wnd(figlist)

    numfigs = len(wnds)
    if numfigs > 0:
        maximize = kwds.get('maximize', False)
        pairs = kwds.get('pairs', 1)

        if maximize or pairs is None:
            nfigspercycle = 1
        else:
            nfigspercycle = pairs

        # n    = length(figs);
        # nlayers = ceil(n/nfigspercycle);

        # Bring one figure up at a time.
        i = 0
        escape_key = chr(27)
        backspace_key = chr(8)
        while 0 <= i and i < numfigs:

            if maximize:
                cmdshow = win32con.SW_SHOWMAXIMIZED
            else:
                cmdshow = win32con.SW_SHOWNORMAL

            iu = min(i + nfigspercycle, numfigs)
            wnd = wnds[i:iu]
            _show_windows(wnd, cmdshow)

            if i + nfigspercycle - 1 < numfigs:
                print('Press escape to quit, backspace to display previous ' +
                      'figure(s) and any other key to display next figure(s)')

            # time.sleep(0.5)

            B = msvcrt.getch()

            if maximize:  # restore window position
                _show_windows(wnd, win32con.SW_RESTORE)

            if B == backspace_key:  # Back space
                i -= nfigspercycle
            elif B == escape_key:
                break
            else:
                i += nfigspercycle

        # Sort the figures.
        wnds.reverse()
        _show_windows(wnds, win32con.SW_SHOWNORMAL)


def test_docstrings():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    test_docstrings()


# def _errcheck(result, func, args):
#    if not result:
#        raise WinError()
#    return args
#
# def bring_window2top():
# #WINUSERAPI BOOL WINAPI
# #GetWindowRect(
# #     HWND hWnd,
# #     LPRECT lpRect);
# #
# #Here is the wrapping with ctypes:
#    NULL = 0
#    SW_HIDE = 0
#    SW_NORMAL = 1
#    SW_SHOWNORMAL = 1
#    SW_SHOWMINIMIZED = 2
#    SW_MAXIMIZE = 3
#    SW_SHOWMAXIMIZED = 3
#    SW_SHOWNOACTIVATE = 4
#    SW_SHOW= 5
#    SW_MINIMIZE = 6
#    SW_SHOWMINNOACTIVE= 7
#    SW_SHOWNA = 8
#    SW_RESTORE = 9
#    SW_SHOWDEFAULT = 10
#    SW_FORCEMINIMIZE = 11
#    SW_MAX = 11
#
#
#    #hwnd = FindWindow(windowname)
#    from ctypes import POINTER, WINFUNCTYPE, windll, WinError
#    from ctypes.wintypes import BOOL, HWND, RECT, LPCSTR, UINT, c_int
#    #Not OK
#    prototype0 = WINFUNCTYPE(HWND, LPCSTR,LPCSTR)
#    paramflags = (1, "lpClassName"), (1, "lpWindowName")
#    FindWindow = prototype0(("FindWindow", windll.user32), paramflags)
#
#    # OK
#    prototype = WINFUNCTYPE(BOOL, HWND, POINTER(RECT))
#    paramflags = (1, "hwnd"), (2, "lprect")
#    GetWindowRect = prototype(("GetWindowRect", windll.user32), paramflags)
#    GetWindowRect.errcheck = _errcheck
#
#    #BringWindowToTop(hwnd)
#    prototype2 = WINFUNCTYPE(BOOL,HWND)
#    paramflags = (1, "hwnd"),
#    #Not Ok.
#    BringWindowToTop = prototype2(("BringWindowToTop", windll.user32),
#                                   paramflags)
#    # Ok
#    CloseWindow = prototype2(("CloseWindow", windll.user32), paramflags)
#    #Not ok
#    prototype3 = WINFUNCTYPE(BOOL, HWND, POINTER(UINT))
#    paramflags = (1, "hwnd"), (1, "ncmdshow")
#    ShowWindow = prototype3(("ShowWindow", windll.user32),paramflags)
#    import win32gui
#    h = win32gui.FindWindow(None,'PyLab')
#    win32gui.ShowWindow(h,)
