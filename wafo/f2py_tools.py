# -*- coding: utf-8 -*-
"""
f2py c_library.pyf c_functions.c -c

See also http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW
"""
import os
import sys


def which(program):
    """
    Return filepath to program if it exists

    In order to test if a certain executable exists, it will search for the
    program name in the environment variables.
    If program is a full path to an executable, it will check it exists

    Copied from:
    http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python/
    It is supposed to mimic the UNIX command "which"
    """

    def is_exe(fpath):
        return os.path.exists(fpath) and os.access(fpath, os.X_OK)

    fpath, unused_fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def f2py_call_str():
    '''Return which f2py callable is in the path regardless of platform'''

    # define possible options:
    # on Arch Linux, python and f2py are the calls corresponding to python 3
    # and python2/f2py2 for python 2
    # other Linux versions might still use python/f2py for python 2

    if os.path.basename(sys.executable).endswith('2'):
        options = ('f2py2', 'f2py2.6', 'f2py2.7',)
    else:  # on Windows and other Linux using python/f2py
        options = ('f2py.exe', 'f2py.bat', 'f2py', 'f2py2.6', 'f2py2.7',
                   'f2py.py',)
    for k in options:
        if which(k):
            # Found the f2py path, no need to look further
            f2py_call = k
            f2py_path = which(k)
            break

    try:
        print('found f2py in:', f2py_path)
        return f2py_call
    except NameError:
        raise UserWarning('Couldn\'t locate f2py. '
                          'Should be part of NumPy installation.')
