"""
f2py c_library.pyf c_functions.c -c

gfortran -W -Wall -pedantic-errors -fbounds-check -Werror -c dsvdc.f mregmodule.f

"""
import os
import sys
from wafo.f2py_tools import f2py_call_str


def compile_all():
    f2py_call = f2py_call_str()
    print '=' * 75
    print 'compiling cov2mod'
    print '=' * 75

    files = ['dsvdc', 'mregmodule', 'intfcmod']
    compile1_format = 'gfortran -fPIC -c %s.f'
    format1 = '%s.o ' * len(files)
    for file_ in files:
        os.system(compile1_format % file_)
    file_objects = format1 % tuple(files)

    os.system(f2py_call + ' -m cov2mod  -c %s cov2mmpdfreg_intfc.f' %
              file_objects)


if __name__ == '__main__':
    compile_all()
