"""
f2py c_library.pyf c_functions.c -c

See also http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW
"""
import os

def which(program):
    """
    Test if program exists
    ======================
    
    In order to test if a certain executable exists, it will search for the 
    program name in the environment variables.
    If program is a full path to an executable, it will check it exists
    
    Copied from:
    http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python/
    It is supposed to mimic the UNIX command "which"
    """
    
    def is_exe(fpath):
        return os.path.exists(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def compile_all():
    
    # regardless of platform, try to figure out which f2py call is in the path
    # define possible options
    f2py_call_list = ('f2py','f2py2.6','f2py2.7','f2py.py',)
    
    no_f2py = True
    for k in f2py_call_list:
        # if the call command exists in the path, it will return the path as
        # a string, otherwise it will return None
        f2py_path = which(k)
        if not f2py_path:
            # didn't find the current call k, continue looking
            pass
        else:
            # current call k is in the path
            f2py_call = k
            no_f2py = False
            break
    
    # raise exception if f2py is not found
    if no_f2py:
        raise UserWarning, \
        'Couldn\'t locate f2py. Should be part of NumPy installation.'
    else:
        print '='*75
        print 'compiling c_codes'
        print '='*75
        print 'found f2py in:', f2py_path
    
    # on Windows: Install microsoft visual c++ .NET 2003 to run the following 
    # build command
    # on posix: install gcc and gfortran
    compile_format = f2py_call + ' %s %s -c'
    
#    # on Linux my linux version it is f2py2.6, don't know how that is on others
#    if os.name == 'posix':
#        # this might vary among specific cases: f2py, f2py2.7, f2py3.2, ...
#        # TODO: more robust approach, find out what f2py is in the users path
#        compile_format = 'f2py2.6 %s %s -c'
#    
#    # Install microsoft visual c++ .NET 2003 and run the following to build the module:
#    elif os.name == 'nt':
#        # compile_format = 'f2py.py %s %s -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71'
#        compile_format = 'f2py.py %s %s -c'
#    
#    # give an Error for other OS-es
#    else:
#        raise UserWarning, \
#        'Untested platform:', os.name
    
    pyfs = ('c_library.pyf',)
    files =('c_functions.c',)

    for pyf,file in zip(pyfs,files):
        os.system(compile_format % (pyf,file))
#        f2py.py c_library.pyf c_functions.c -c

if __name__=='__main__':
    compile_all()