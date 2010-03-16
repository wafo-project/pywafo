"""
f2py c_library.pyf c_functions.c -c
"""
import os

def compile_all():
    # Install gfortran and run the following to build the module:
    compile_format = 'f2py.py %s %s -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71'

    # Install microsoft visual c++ .NET 2003 and run the following to build the module:
    #compile_format = 'f2py.py %s %s -c'
    pyfs = ('c_library.pyf',)
    files =('c_functions.c',)

    for pyf,file in zip(pyfs,files):
        os.system(compile_format % (pyf,file))

if __name__=='__main__':
    compile_all()
