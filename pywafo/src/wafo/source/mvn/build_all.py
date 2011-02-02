"""
builds mvn.pyd
"""
import os

def compile_all():
    
    #os.system('f2py.py mvn.pyf mvndst.f -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71 ')
    
    # this might vary among specific cases: f2py, f2py2.7, f2py3.2, ...
    # TODO: more robust approach, find out what f2py is in the users path
    if os.name == 'posix':
        f2py_call = 'f2py2.6'
    
    # Install microsoft visual c++ .NET 2003 and run the following to build the module:
    elif os.name == 'nt':
        f2py_call = 'f2py.py'
    
    # give an Error for other OS-es
    else:
        raise UserWarning, \
        'Untested platform:', os.name
    
    os.system(f2py_call + ' mvn.pyf mvndst.f -c ')
    
    
if __name__=='__main__':
    compile_all()
