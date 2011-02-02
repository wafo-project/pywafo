"""
f2py c_library.pyf c_functions.c -c

See also http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW
"""
import os

def compile_all():
    
    # on Linux my linux version it is f2py2.6, don't know how that is on others
    if os.name == 'posix':
        # this might vary among specific cases: f2py, f2py2.7, f2py3.2, ...
        # TODO: more robust approach, find out what f2py is in the users path
        compile_format = 'f2py2.6 %s %s -c'
    
    # Install microsoft visual c++ .NET 2003 and run the following to build the module:
    elif os.name == 'nt':
        # compile_format = 'f2py.py %s %s -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71'
        compile_format = 'f2py.py %s %s -c'
    
    # give an Error for other OS-es
    else:
        raise UserWarning, \
        'Untested platform:', os.name
    
    pyfs = ('c_library.pyf',)
    files =('c_functions.c',)

    for pyf,file in zip(pyfs,files):
        os.system(compile_format % (pyf,file))
#        f2py.py c_library.pyf c_functions.c -c

if __name__=='__main__':
    compile_all()
