# /usr/bin/env python
'''
Module FIG
------------
Module for manipulating windows/figures created using
pylab or enthought.mayavi.mlab on the windows platform.

Figure manipulation involves
maximization, minimization, hiding, closing, stacking or tiling.

It is assumed that the figures are uniquely numbered in the following way:
Figure 1
Figure 2
....
or
TVTK scene 1
TVTK scene 2
TVTK scene 3
...

Examples
--------
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

# import win32api

import win32gui
import win32con
import numpy

from win32gui import (EnumWindows, MoveWindow, GetWindowRect, FindWindow,
                      ShowWindow, BringWindowToTop)
try:
    import wx
except ImportError as error:
    import warnings
    warnings.warn(str(error))
    wx = None

__all__ = ['close', 'cycle', 'hide', 'keep', 'maximize', 'minimize', 'pile',
           'restore', 'stack', 'tile', 'find_all_figure_numbers', 'set_size']

# Figure format strings to recognize in window title
FIGURE_TITLE_FORMATS = ('Figure', 'TVTK Scene', 'Chaco Plot Window: Figure')
_SCREENSIZE = None

if wx is not None:
    class CycleDialog(wx.Dialog):

        def _get_buttons(self):
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            buttons = ['Forward', 'Back', 'Cancel']
            callbacks = [self.on_forward, self.on_backward, self.on_cancel]
            for button, callback in zip(buttons, callbacks):
                button = wx.Button(self, -1, button, size=(70, 30))
                self.Bind(wx.EVT_BUTTON, callback, button)
                hbox.Add(button, 1, wx.ALIGN_CENTER)
            return hbox

        def _get_message(self):
            label = ('Press back or forward to display previous or next figure(s),'
                     ' respectively. Press cancel to quit.')
            message = wx.StaticText(self, label=label, size=(240, 25))
            return message

        def __init__(self, parent, interval=None, title='Cycle dialog'):
            super(CycleDialog, self).__init__(parent, title=title, size=(260, 130))
            if isinstance(interval, (float, int)):
                self.interval_milli_sec = interval * 1000
            else:
                self.interval_milli_sec = 30

            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self.on_forward, self.timer)

            vbox = wx.BoxSizer(wx.VERTICAL)
            vbox.Add(self._get_message(), 0, wx.ALIGN_CENTER | wx.TOP, 20)
            vbox.Add(self._get_buttons(), 1, wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, 10)
            self.SetSizer(vbox)

        def ShowModal(self, *args, **kwargs):
            self.timer.Start(self.interval_milli_sec, oneShot=True)
            return super(CycleDialog, self).ShowModal(*args, **kwargs)

        def on_forward(self, evt):
            self.EndModal(wx.ID_FORWARD)

        def on_backward(self, evt):
            self.EndModal(wx.ID_BACKWARD)

        def on_cancel(self, evt):
            self.EndModal(wx.ID_CANCEL)


    def _get_cycle_dialog(parent=None, interval=None):
        app = wx.GetApp()
        if not app:
            app = wx.App(redirect=False)
            frame = wx.Frame(None)
            app.SetTopWindow(frame)
        dlg = CycleDialog(parent, interval)
        return dlg


def get_window_position_and_size(window_handle):
    pos = GetWindowRect(window_handle)
    return pos[0], pos[1], pos[2] - pos[0], pos[3] - pos[1]


def get_screen_position_and_size(window_handles):
    """Return screen position; X, Y and size; width, height.

    Parameters
    ----------
    window_handles: list of handles to open window figures
      (Note: only needed the first time)

    Returns
    --------
    X : coordinate of the left side of the screen.
    Y : coordinate of the top of the screen.
    width : screen horizontal size
    height : screen vertical size

    """
    # pylint: disable=global-statement
    global _SCREENSIZE
    if _SCREENSIZE is None:
        window_handle = window_handles[0]
        pos = get_window_position_and_size(window_handle)
        _show_windows((window_handle,), win32con.SW_SHOWMAXIMIZED)
        _SCREENSIZE = get_window_position_and_size(window_handle)
        MoveWindow(window_handle, pos[0], pos[1], pos[2], pos[3], 1)
    return _SCREENSIZE


def _get_screen_size(wnds):
    screen_width, screen_height = get_screen_position_and_size(wnds)[2:4]
    return screen_width, screen_height


def _windowEnumerationHandler(handle, result_list):
    """Pass to win32gui.EnumWindows() to generate list of window handle, window
    text tuples."""
    # pylint: disable=no-member
    if win32gui.IsWindowVisible(handle):
        result_list.append((handle, win32gui.GetWindowText(handle)))


def _find_window_handles_and_titles(wantedTitle=None):
    """Return list of window handle and window title tuples.

    Parameter
    ---------
    wantedTitle:

    """
    handles_n_titles = []
    EnumWindows(_windowEnumerationHandler, handles_n_titles)
    if wantedTitle is None:
        return handles_n_titles
    else:
        return [(handle, title)
                for handle, title in handles_n_titles
                if title.startswith(wantedTitle)]


def find_figure_handles(*figure_numbers):
    """Find figure handles from figure numbers."""
    wnd_handles = []
    for figure_number in _parse_figure_numbers(*figure_numbers):
        for format_ in FIGURE_TITLE_FORMATS:
            winTitle = format_ + ' %d' % figure_number
            handle = FindWindow(None, winTitle)
            if not handle == 0:
                wnd_handles.append(handle)
    return wnd_handles


def find_all_figure_numbers():
    """Return list of all figure numbers.

    Examples
    --------
    >>> import fig
    >>> import pylab as p
    >>> for ix in range(5):
    ...     f = p.figure(ix)
    ...     p.draw()

    fig.find_all_figure_numbers()
    [0, 1, 2, 3, 4]

    >>> fig.close()

    """
    figure_numbers = []
    for wantedTitle in FIGURE_TITLE_FORMATS:
        handles_n_titles = _find_window_handles_and_titles(wantedTitle)
        for _handle, title in handles_n_titles:
            try:
                number = int(title.split()[-1])
                figure_numbers.append(number)
            except (TypeError, ValueError):
                pass
    # pylint: disable=no-member
    return numpy.unique(figure_numbers).tolist()


def _parse_figure_numbers(*args):
    figure_numbers = []
    for arg in args:
        if isinstance(arg, (list, tuple, set)):
            for val in arg:
                figure_numbers.append(int(val))
        elif isinstance(arg, int):
            figure_numbers.append(arg)
        elif arg == 'all':
            figure_numbers = find_all_figure_numbers()
            break
        else:
            raise TypeError('Only integers arguments accepted!')

    if len(figure_numbers) == 0:
        figure_numbers = find_all_figure_numbers()
    return figure_numbers


def _show_figure(figure_numbers, command):
    """Sets the specified figure's show state.

    Parameters
    ----------
    figure_numbers: list of figure numbers
    command: one of following commands:
    SW_FORCEMINIMIZE:
        Minimizes a window, even if the thread that owns the window is not
        responding. This flag should only be used when minimizing windows
        from a different thread.
    SW_HIDE:
        Hides the window and activates another window.
    SW_MAXIMIZE:
        Maximizes the specified window.
    SW_MINIMIZE:
        Minimizes the specified window and activates the next top-level window
        in the Z order.
    SW_RESTORE:
        Activates and displays the window. If the window is minimized or
        maximized, the system restores it to its original size and position.
        An application should specify this flag when restoring a minimized
        window.
    SW_SHOW:
        Activates the window and displays it in its current size and position.
    SW_SHOWDEFAULT:
        Sets the show state based on the SW_ value specified in the STARTUPINFO
        structure passed to the CreateProcess function by the program that
        started the application.
    SW_SHOWMAXIMIZED:
        Activates the window and displays it as a maximized window.
    SW_SHOWMINIMIZED:
        Activates the window and displays it as a minimized window.
    SW_SHOWMINNOACTIVE:
        Displays the window as a minimized window. This value is similar to
        SW_SHOWMINIMIZED, except the window is not activated.
    SW_SHOWNA:
        Displays the window in its current size and position. This value is
        similar to SW_SHOW, except the window is not activated.
    SW_SHOWNOACTIVATE:
        Displays a window in its most recent size and position. This value is
        similar to SW_SHOWNORMAL, except the window is not actived.
    SW_SHOWNORMAL:
        Activates and displays a window. If the window is minimized or
        maximized, the system restores it to its original size and position.
        An application should specify this flag when displaying the window for
        the first time.

    """
    for number in _parse_figure_numbers(*figure_numbers):
        for format_ in FIGURE_TITLE_FORMATS:
            title = format_ + ' %d' % number
            handle = FindWindow(None, title)
            if not handle == 0:
                BringWindowToTop(handle)
                ShowWindow(handle, command)


def _show_windows(handles, command, redraw_now=False):
    """Sets the specified window's show state.

    Parameters
    ----------
    handles: list of window handles
    command: one of following commands:
    SW_FORCEMINIMIZE:
        Minimizes a window, even if the thread that owns the window is not
        responding. This flag should only be used when minimizing windows
        from a different thread.
    SW_HIDE:
        Hides the window and activates another window.
    SW_MAXIMIZE:
        Maximizes the specified window.
    SW_MINIMIZE:
        Minimizes the specified window and activates the next top-level window
        in the Z order.
    SW_RESTORE:
        Activates and displays the window. If the window is minimized or
        maximized, the system restores it to its original size and position.
        An application should specify this flag when restoring a minimized
                    window.
    SW_SHOW:
        Activates the window and displays it in its current size and position.
    SW_SHOWDEFAULT:
        Sets the show state based on the SW_ value specified in the STARTUPINFO
        structure passed to the CreateProcess function by the program that
        started the application.
    SW_SHOWMAXIMIZED:
        Activates the window and displays it as a maximized window.
    SW_SHOWMINIMIZED:
        Activates the window and displays it as a minimized window.
    SW_SHOWMINNOACTIVE:
        Displays the window as a minimized window. This value is similar to
        SW_SHOWMINIMIZED, except the window is not activated.
    SW_SHOWNA:
        Displays the window in its current size and position. This value is
        similar to SW_SHOW, except the window is not activated.
    SW_SHOWNOACTIVATE:
        Displays a window in its most recent size and position. This value is
        similar to SW_SHOWNORMAL, except the window is not actived.
    SW_SHOWNORMAL:
        Activates and displays a window. If the window is minimized or
        maximized, the system restores it to its original size and position.
        An application should specify this flag when displaying the window for
        the first time.

    redraw_now :

    """
    # pylint: disable=no-member
    for handle in handles:
        if not handle == 0:
            BringWindowToTop(handle)
            ShowWindow(handle, command)
            if redraw_now:
                rect = GetWindowRect(handle)
                win32gui.RedrawWindow(handle, rect, None, win32con.RDW_UPDATENOW)


def keep(*figure_numbers):
    """Keeps figure windows of your choice and closes the rest.

    Parameters
    ----------
    figure_numbers : list of integers specifying which figures to keep.

    Examples
    --------
    # keep only figures 1,2,3,5 and 7
    >>> import pylab as p
    >>> import wafo.fig as fig
    >>> for ix in range(10):
    ...     f = p.figure(ix)
    >>> fig.keep( range(1,4),  5, 7)

    or
        fig.keep([range(1,4),  5, 7])
    >>> fig.close()

    See also
    --------
    fig.close

    """
    figs2keep = []
    for fig in figure_numbers:
        if isinstance(fig, (list, tuple, set)):
            for val in fig:
                figs2keep.append(int(val))
        elif isinstance(fig, int):
            figs2keep.append(fig)
        else:
            raise TypeError('Only integers arguments accepted!')

    if len(figs2keep) > 0:
        allfigs = set(find_all_figure_numbers())
        figs2delete = allfigs.difference(figs2keep)
        close(figs2delete)


def close(*figure_numbers):
    """  Close figure window(s)

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to close (default 'all').

    Examples
    --------
    >>> import pylab as p
    >>> import wafo.fig as fig
    >>> for ix in range(5):
    ...     f = p.figure(ix)
    >>> fig.close(3,4)   # close figure 3 and 4
    >>> fig.close('all') # close all remaining figures

    or even simpler
    fig.close() # close all remaining figures

    See also
    --------
    fig.keep

    """
    # pylint: disable=no-member
    for handle in find_figure_handles(*figure_numbers):
        if win32gui.SendMessage(handle, win32con.WM_CLOSE, 0, 0):
            win32gui.SendMessage(handle, win32con.WM_DESTROY, 0, 0)


def restore(*figure_numbers):
    """Restore figures window size and position to its default value.

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to restor (default 'all').

    Description
    -----------
    RESTORE Activates and displays the window. If the window is minimized
    or maximized, the system restores it to its original size and position.

    Examples
    ---------
      >>> import pylab as p
      >>> import wafo.fig as fig
      >>> for ix in range(5):
    ...     f = p.figure(ix)
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

    """
    SW_RESTORE = win32con.SW_RESTORE
    # SW_RESTORE = win32con.SW_SHOWDEFAULT
    # SW_RESTORE = win32con.SW_SHOWNORMAL
    _show_figure(figure_numbers, SW_RESTORE)


def hide(*figure_numbers):
    """hide figure(s) window.

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to hide (default 'all').

    Examples:
    --------
    >>> import wafo.fig as fig
     >>> import pylab as p
    >>> for ix in range(5):
    ...     f = p.figure(ix)
     >>> fig.hide('all')   #hides all unhidden figures
     >>> fig.hide()        #same as hide('all')
     >>> fig.hide(p.gcf().number)   #hides the current figure
     >>> fig.hide(3)       #hides figure 3
     >>> fig.hide([2, 4])   #hides figures 2 and 4

     or alternatively
         fig.hide(2, 4)
    >>> fig.restore(list(range(5)))
     >>> fig.close()

     See also
     --------
    fig.cycle,
    fig.keep,
    fig.restore

    """
    _show_figure(figure_numbers, win32con.SW_HIDE)


def minimize(*figure_numbers):
    """Minimize figure(s) window size.

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to minimize (default 'all').

    Examples:
    ---------
     >>> import wafo.fig as fig
     >>> import pylab as p
     >>> for ix in range(5):
     ...     f = p.figure(ix)
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
     fig.cycle,
     fig.keep,
     fig.restore

    """
    _show_figure(figure_numbers, win32con.SW_SHOWMINIMIZED)


def maximize(*figure_numbers):
    """Maximize figure(s) window size.

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to maximize (default 'all').

    Examples:
    ---------
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(5):
     ...     f = p.figure(ix)
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
     fig.cycle,
     fig.keep,
     fig.restore

    """
    _show_figure(figure_numbers, win32con.SW_SHOWMAXIMIZED)


def pile(*figure_numbers, **kwds):
    """Pile figure windows.

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to pile (default 'all').
    kwds : dict with the following keys
        position :
        width  :
        height :

    Description
    -------------
       PILE piles all open figure windows on top of eachother
       with complete overlap. PILE(FIGS) can be used to specify which
       figures that should be piled. Figures are not sorted when specified.

     Examples
     --------
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(7):
     ...     f = p.figure(ix)
     >>> fig.pile()                # pile all open figures
     >>> fig.pile(range(1,4), 5, 7)  # pile figure 1,2,3,5 and 7
     >>> fig.close()

     See also
     --------
     fig.cycle, fig.keep, fig.maximize, fig.restore,
             fig.stack, fig.tile

    """
    wnds = find_figure_handles(*figure_numbers)
    numfigs = len(wnds)
    if numfigs > 0:
        screen_width, screen_height = _get_screen_size(wnds)
        pos = kwds.get(
            'position', (int(screen_width / 5), int(screen_height / 4)))
        width = kwds.get('width', int(screen_width / 2.5))
        height = kwds.get('height', int(screen_height / 2))

        for wnd in wnds:
            MoveWindow(wnd, pos[0], pos[1], width, height, 1)
            BringWindowToTop(wnd)


def set_size(*figure_numbers, **kwds):
    """Set size for figure windows.

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to pile (default 'all').
    kwds : dict with the following keys
        width  :
        height :

    Description
    -------------
    Set size sets the size of all open figure windows. SET_SIZE(FIGS)
    can be used to specify which figures that should be resized.
    Figures are not sorted when specified.

     Examples
     --------
     >>> import pylab as p
     >>> import fig
     >>> for ix in range(7):
     ...     f = p.figure(ix)
     >>> fig.set_size(7, width=150, height=100)
     >>> fig.set_size(range(1,4), 5,width=250, height=170)
     >>> fig.close()

     See also
     --------
     fig.cycle, fig.keep, fig.maximize, fig.restore,
             fig.stack, fig.tile

    """
    handles = find_figure_handles(*figure_numbers)
    numfigs = len(handles)
    if numfigs > 0:
        screen_width, screen_height = _get_screen_size(handles)
        width = kwds.get('width', int(screen_width / 2.5))
        height = kwds.get('height', int(screen_height / 2))
        new_pos = kwds.get('position', None)
        pos = new_pos
        for handle in handles:
            if not new_pos:
                pos = get_window_position_and_size(handle)
            MoveWindow(handle, pos[0], pos[1], width, height, 1)
            BringWindowToTop(handle)


def stack(*figure_numbers, **kwds):
    """Stack figure windows.

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to stack (default 'all').
    kwds : dict with the following keys
        figs_per_stack :
            number of figures per stack (default depends on screenheight)

    Description
    -----------
       STACK stacks all open figure windows on top of eachother
       with maximum overlap. STACK(FIGS) can be used to specify which
       figures that should be stacked. Figures are not sorted when specified.

     Examples
     --------
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(7):
     ...     f = p.figure(ix)
     >>> fig.stack()                # stack all open figures
     >>> fig.stack(range(1,4), 5, 7)  # stack figure 1,2,3,5 and 7
     >>> fig.close()

     See also
     --------
      fig.cycle, fig.keep, fig.maximize, fig.restore,
             fig.pile, fig.tile

    """
    wnds = find_figure_handles(*figure_numbers)
    numfigs = len(wnds)
    if numfigs > 0:
        screenpos = get_screen_position_and_size(wnds)
        y_step = 25
        x_step = border = 5

        figs_per_stack = kwds.get(
            'figs_per_stack',
            int(numpy.fix(0.7 * (screenpos[3] - border) / y_step)))

        for iy in range(numfigs):
            pos = get_window_position_and_size(wnds[iy])
            # print('[x, y, w, h] = ', pos)
            ix = iy % figs_per_stack
            ypos = int(screenpos[1] + ix * y_step + border)
            xpos = int(screenpos[0] + ix * x_step + border)
            MoveWindow(wnds[iy], xpos, ypos, pos[2], pos[3], 1)
            BringWindowToTop(wnds[iy])


def tile(*figure_numbers, **kwds):
    """Tile figure windows.

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to tile (default 'all').
    kwds : dict with key pairs
        specifying how many pairs of figures that are tiled at a time

    Description
    -----------
       TILE places all open figure windows around on the screen with no
       overlap. TILE(FIGS) can be used to specify which figures that
       should be tiled. Figures are not sorted when specified.

     Examples
     --------
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(7):
     ...     f = p.figure(ix)
     >>> fig.tile()             # tile all open figures
     >>> fig.tile( range(1,4), 5, 7)    # tile figure 1,2,3,5 and 7
     >>> fig.tile(range(1,11), pairs=2) # tile figure 1 to 10 two at a time
     >>> fig.tile(range(1,11), pairs=3) # tile figure 1 to 10 three at a time
     >>> fig.close()

     See also
     --------
     fig.cycle, fig.keep, fig.maximize, fig.minimize
     fig.restore, fig.pile, fig.stack

    """
    wnds = find_figure_handles(*figure_numbers)

    nfigs = len(wnds)
    # Number of windows.

    if nfigs > 0:
        nfigspertile = kwds.get('pairs', nfigs)

        ceil = numpy.ceil
        sqrt = numpy.sqrt
        maximum = numpy.maximum

        nlayers = int(ceil(nfigs / nfigspertile))

        # Number of figures horisontally.
        nh = maximum(int(ceil(sqrt(nfigspertile))), 2)
        # Number of figures vertically.
        nv = maximum(int(ceil(nfigspertile / nh)), 2)

        screenpos = get_screen_position_and_size(wnds)
        screen_width, screen_heigth = screenpos[2:4]

        hspc = 10            # Horisontal space.
        topspc = 20            # Space above top figure.
        medspc = 10            # Space between figures.
        botspc = 20            # Space below bottom figure.

        figwid = (screen_width - (nh + 1) * hspc) / nh
        fighgt = (screen_heigth - (topspc + botspc) - (nv - 1) * medspc) / nv

        figwid = int(numpy.round(figwid))
        fighgt = int(numpy.round(fighgt))

        idx = 0
        for unused_ix in range(nlayers):
            for row in range(nv):
                figtop = int(screenpos[1] + topspc + row * (fighgt + medspc))
                for col in range(nh):
                    if (row) * nh + col < nfigspertile:
                        if idx < nfigs:
                            figlft = int(
                                screenpos[0] + (col + 1) * hspc + col * figwid)
                            fighnd = wnds[idx]
                            MoveWindow(fighnd, figlft, figtop, figwid, fighgt,
                                       1)
                            # Set position.
                            BringWindowToTop(fighnd)
                        idx += 1


class _CycleGenerator(object):

    """Cycle through figure windows.

    Parameters
    ----------
    figure_numbers : list of integers or string
        specifying which figures to cycle through (default 'all').
    kwds : dict with the following keys
        pairs : number of figures to cycle in pairs (default 1)
        maximize: If True maximize figure when viewing (default False)
        interval : pause interval in seconds

    Description
    -----------
       CYCLE brings up all open figure in ascending order and pauses after
       each figure. Press escape to quit cycling, backspace to display previous
       figure(s) and press any other key to display next figure(s)
       When done, the figures are sorted in ascending order.

    CYCLE(maximize=True) does the same thing, except figures are maximized.
       CYCLE(pairs=2)   cycle through all figures in pairs of 2.

     Examples:
     >>> import pylab as p
     >>> import wafo.fig as fig
     >>> for ix in range(4):
     ...     f = p.figure(ix)

     fig.cycle(range(3), interval=1)  # Cycle trough figure 0 to 2

     # Cycle trough figure 0 to 2 with figures maximized
     fig.cycle(range(3), maximize=True, interval=1)
     fig.cycle(interval=1)            # Cycle through all figures one at a time
     fig.tile(pairs=2, interval=1)
     fig.cycle(pairs=2, interval=2)   # Cycle through all figures two at a time

     fig.cycle(pairs=2)      # Manually cycle through all figures two at a time
     >>> fig.close()

    See also
    --------
     fig.keep, fig.maximize, fig.restore, fig.pile,
            fig.stack, fig.tile

    """
    escape_key = chr(27)
    backspace_key = chr(8)

    def __init__(self, **kwds):
        self.dialog = None
        maximize = kwds.get('maximize', False)
        pairs = kwds.get('pairs', 1)
        self.interval = kwds.get('interval', 'user_defined')
        self.nfigspercycle = 1
        if maximize:
            self.command = win32con.SW_SHOWMAXIMIZED
        else:
            self.command = win32con.SW_SHOWNORMAL
            if pairs is not None:
                self.nfigspercycle = pairs

    def _set_options(self, kwds):
        self.__init__(**kwds)

    def _iterate(self, handles):
        i = 0
        numfigs = len(handles)
        self.dialog = _get_cycle_dialog(parent=None, interval=self.interval)
        while 0 <= i and i < numfigs:
            iu = min(i + self.nfigspercycle, numfigs)
            yield handles[i:iu]
            i = self.next_index(i)
        self.dialog.Destroy()
        raise StopIteration

    def next_index(self, i):
        result = self.dialog.ShowModal()
        if result == wx.ID_FORWARD:
            i += self.nfigspercycle
        elif result == wx.ID_BACKWARD:
            i -= self.nfigspercycle
        else:
            i = -1
        return i

    def __call__(self, *figure_numbers, **kwds):
        handles = find_figure_handles(*figure_numbers)
        numfigs = len(handles)
        if numfigs > 0:
            self._set_options(kwds)
            for handle in self._iterate(handles):
                _show_windows(handle, self.command, redraw_now=True)

            _show_windows(handles, win32con.SW_SHOWNORMAL)

cycle = _CycleGenerator()


if __name__ == '__main__':
    from wafo.testing import test_docstrings
    import matplotlib
    matplotlib.interactive(True)
    test_docstrings(__file__)
