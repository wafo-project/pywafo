"""
f2py c_library.pyf c_functions.c -c

See also http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW
"""
import os
import sys
from wafo.f2py_tools import f2py_call_str

def compile_all():
    f2py_call = f2py_call_str()
    print '=' * 75
    print 'compiling c_codes'
    print '=' * 75

    compile_format = f2py_call + ' %s %s -c'

    pyfs = ('c_library.pyf',)
    files = ('c_functions.c',)

    for pyf, file_ in zip(pyfs, files):
        os.system(compile_format % (pyf, file_))

if __name__ == '__main__':
    compile_all()
