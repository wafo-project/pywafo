from __future__ import division
from numpy import round
from threading import Thread
from time import sleep
from win32gui import (InitCommonControls, CallWindowProc, CreateWindowEx,
                      CreateWindow, SetWindowLong, SendMessage, ShowWindow,
                      PumpWaitingMessages, PostQuitMessage, DestroyWindow,
                      MessageBox, EnumWindows, GetClassName)
from win32api import GetModuleHandle, GetSystemMetrics  # @UnresolvedImport
from win32api import SetWindowLong as api_SetWindowLong  # @UnresolvedImport
from commctrl import (TOOLTIPS_CLASS, TTM_GETDELAYTIME, TTM_SETDELAYTIME,
                      TTDT_INITIAL, TTDT_AUTOPOP)
import win32con

WM_USER = win32con.WM_USER
PBM_SETRANGE = (WM_USER + 1)
PBM_SETPOS = (WM_USER + 2)
PBM_DELTAPOS = (WM_USER + 3)
PBM_SETSTEP = (WM_USER + 4)
PBM_STEPIT = (WM_USER + 5)
PBM_SETRANGE32 = (WM_USER + 6)
PBM_GETRANGE = (WM_USER + 7)
PBM_GETPOS = (WM_USER + 8)
PBM_SETBARCOLOR = (WM_USER + 9)
PBM_SETMARQUEE = (WM_USER + 10)
PBM_GETSTEP = (WM_USER + 13)
PBM_GETBKCOLOR = (WM_USER + 14)
PBM_GETBARCOLOR = (WM_USER + 15)
PBM_SETSTATE = (WM_USER + 16)
PBM_GETSTATE = (WM_USER + 17)
PBS_SMOOTH = 0x01
PBS_VERTICAL = 0x04
PBS_MARQUEE = 0x08
PBS_SMOOTHREVERSE = 0x10
PBST_NORMAL = 1
PBST_ERROR = 2
PBST_PAUSED = 3
WC_DIALOG = 32770
WM_SETTEXT = win32con.WM_SETTEXT


def MAKELPARAM(a, b):
    return (a & 0xffff) | ((b & 0xffff) << 16)


def _get_tooltip_handles(hwnd, resultList):
    '''
    Adds a window handle to resultList if its class-name is 'tooltips_class32',
    i.e. the window is a tooltip.
    '''
    if GetClassName(hwnd) == TOOLTIPS_CLASS:
        resultList.append(hwnd)


def set_max_pop_delay_on_tooltip(tooltip):
    '''
    Sets maximum auto-pop delay (delay before hiding) on an instance of
    wx.ToolTip.
    NOTE: The tooltip's SetDelay method is used just to identify the correct
    tooltip.
    '''
    test_value = 12345
    # Set initial delay just to identify tooltip.
    tooltip.SetDelay(test_value)
    handles = []
    EnumWindows(_get_tooltip_handles, handles)
    for hwnd in handles:
        if SendMessage(hwnd, TTM_GETDELAYTIME, TTDT_INITIAL) == test_value:
            SendMessage(hwnd, TTM_SETDELAYTIME, TTDT_AUTOPOP, 32767)
    tooltip.SetDelay(500)   # Restore default value


class Waitbar(Thread):

    def __init__(self, title='Waitbar', can_abort=True, max_val=100):
        Thread.__init__(self)     # Initialize thread
        self.title = title
        self.can_abort = can_abort
        self.max_val = max_val
        InitCommonControls()
        self.hinst = GetModuleHandle(None)
        self.started = False
        self.position = 0
        self.do_update = False
        self.start()                        # Run the thread
        while not self.started:
            sleep(0.1)                      # Wait until the dialog is ready

    def DlgProc(self, hwnd, uMsg, wParam, lParam):
        if uMsg == win32con.WM_DESTROY:
            api_SetWindowLong(self.dialog,
                              win32con.GWL_WNDPROC,
                              self.oldWndProc)
        if uMsg == win32con.WM_CLOSE:
            self.started = False
        if uMsg == win32con.WM_COMMAND and self.can_abort:
            self.started = False
        return CallWindowProc(self.oldWndProc, hwnd, uMsg, wParam, lParam)

    def BuildWindow(self):
        width = 400
        height = 100
        self.dialog = CreateWindowEx(
            win32con.WS_EX_TOPMOST,
            WC_DIALOG,
            self.title + ' (0%)',
            win32con.WS_VISIBLE | win32con.WS_OVERLAPPEDWINDOW,
            int(round(
                GetSystemMetrics(win32con.SM_CXSCREEN) * .5 - width * .5)),
            int(round(
                GetSystemMetrics(win32con.SM_CYSCREEN) * .5 - height * .5)),
            width,
            height,
            0,
            0,
            self.hinst,
            None)
        self.progbar = CreateWindow(
            #                             win32con.WS_EX_DLGMODALFRAME,
            'msctls_progress32',
            '',
            win32con.WS_VISIBLE | win32con.WS_CHILD,
            10,
            10,
            width - 30,
            20,
            self.dialog,
            0,
            0,
            None)
        if self.can_abort:
            self.button = CreateWindow(
                #                             win32con.WS_EX_DLGMODALFRAME,
                'BUTTON',
                'Cancel',
                win32con.WS_VISIBLE | win32con.WS_CHILD | win32con.BS_PUSHBUTTON,  # @IgnorePep8
                int(width / 2.75),
                40,
                100,
                20,
                self.dialog,
                0,
                0,
                None)
        self.oldWndProc = SetWindowLong(
            self.dialog,
            win32con.GWL_WNDPROC,
            self.DlgProc)
        SendMessage(self.progbar, PBM_SETRANGE, 0, MAKELPARAM(0, self.max_val))
#        win32gui.SendMessage(self.progbar, PBM_SETSTEP, 0, 10)
#        win32gui.SendMessage(self.progbar, PBM_SETMARQUEE, 0, 0)
        ShowWindow(self.progbar, win32con.SW_SHOW)

    def run(self):
        self.BuildWindow()
        self.started = True
        while self.started:
            PumpWaitingMessages()
            if self.do_update:
                SendMessage(self.progbar, PBM_SETPOS,
                            int(self.position % self.max_val), 0)
                percentage = int(round(100.0 * self.position / self.max_val))
                SendMessage(self.dialog, WM_SETTEXT, 0,
                            self.title + ' (%d%%)' % percentage)
    #            SendMessage(self.progbar, PBM_STEPIT, 0, 0)
                self.do_update = False
            sleep(0.1)
        PostQuitMessage(0)
        DestroyWindow(self.dialog)

    def update(self, pos):
        if self.started:
            if not self.do_update:
                self.position = pos
                self.do_update = True
            return True
        return False

    def close(self):
        self.started = False


# class Waitbar2(Dialog, Thread):
#    def __init__(self, title='Waitbar'):
#        template = [[title, (0, 0, 215, 36),
#                     (win32con.DS_MODALFRAME | win32con.WS_POPUP |
#                      win32con.WS_VISIBLE | win32con.WS_CAPTION |
#                      win32con.WS_SYSMENU | win32con.DS_SETFONT |
#                      win32con.WS_GROUP | win32con.WS_EX_TOPMOST),
# | win32con.DS_SYSMODAL),
#                      None, (8, "MS Sans Serif")], ]
#        Dialog.__init__(self, id=template)
# Thread.__init__(self)     # Initialize thread
#        self.started = False
# self.start()                        # Run the thread
#        while not self.started:
# sleep(0.1)                      # Wait until the dialog is ready
#
#    def OnInitDialog(self):
#        rc = Dialog.OnInitDialog(self)
#        self.pbar = CreateProgressCtrl()
#        self.pbar.CreateWindow (win32con.WS_CHILD | win32con.WS_VISIBLE,
#                                (10, 10, 310, 24), self, 1001)
#        self.started = True
#        return rc
#
#    def run(self):
#        self.DoModal()
#
#    def update(self, pos):
#        self.pbar.SetPos(int(pos))
#
#    def close(self):
#        self.OnCancel()


class WarnDlg(Thread):

    def __init__(self, message='', title='Warning!'):
        Thread.__init__(self)     # Initialize thread
        self.title = title
        self.message = message
        self.start()              # Run the thread

    def run(self):
        # MessageBox(self.message, self.title, win32con.MB_ICONWARNING)
        MessageBox(0, self.message, self.title,
                   win32con.MB_ICONWARNING | win32con.MB_SYSTEMMODAL)


class ErrorDlg(Thread):

    def __init__(self, message='', title='Error!', blocking=False):
        Thread.__init__(self)     # Initialize thread
        self.title = title
        self.message = message
        if blocking:
            self.run()              # Run without threading
        else:
            self.start()            # Run in thread

    def run(self):
        # MessageBox(self.message, self.title, win32con.MB_ICONERROR)
        MessageBox(0, self.message, self.title,
                   win32con.MB_ICONERROR | win32con.MB_SYSTEMMODAL)


if __name__ == '__main__':
    WarnDlg('This is an example of a warning', 'Warning!')
    ErrorDlg('This is an example of an error message')
    wb = Waitbar('Waitbar example')
#    wb2 = Waitbar2('Waitbar example')
    for i in range(20):
        print(wb.update(i * 5))
#        wb2.update(i)
        sleep(0.1)
    wb.close()
#    wb2.close()
