"""builds mvnprdmod.pyd."""
import os
import sys
from wafo.f2py_tools import f2py_call_str


def compile_all():
    f2py_call = f2py_call_str()
    print '=' * 75
    print 'compiling mvnprd'
    print '=' * 75

    files = ['mvnprd', 'mvnprodcorrprb']
    compile1_format = 'gfortran -fPIC -c %s.f'
    for file_ in files:
        os.system(compile1_format % file_)
    file_objects = '%s.o %s.o' % tuple(files)

    # os.system('f2py.py -m mvnprdmod  -c %s mvnprd_interface.f
    # --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71' % file_objects)
    os.system(f2py_call + ' -m mvnprdmod  -c %s mvnprd_interface.f ' %
              file_objects)

if __name__ == '__main__':
    compile_all()
