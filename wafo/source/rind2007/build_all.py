"""
Builds rindmod.pyd

See also
http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW

"""
import os
import sys
from wafo.f2py_tools import f2py_call_str


def compile_all():
    f2py_call = f2py_call_str()
    print '=' * 75
    print 'compiling rind2007'
    print '=' * 75

    files = ['intmodule',  'jacobmod', 'swapmod',
             'fimod', 'rindmod', 'rind71mod']
    compile1_format = 'gfortran -fPIC -c %s.f'
    format1 = '%s.o ' * len(files)
    for file_ in files:
        os.system(compile1_format % file_)
    file_objects = format1 % tuple(files)

    os.system(f2py_call + ' -m rindmod  -c %s rind_interface.f ' %
              file_objects)

if __name__ == '__main__':
    compile_all()
