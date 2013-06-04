"""
f2py c_library.pyf c_functions.c -c

See also http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW
"""
import os
import sys

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
    # regardless of platform, try to figure out which f2py call is in the path
    # define possible options
    
    # on Arch Linux, python and f2py are the calls corresponding to python 3
    # and python2/f2py2 for python 2
    # other Linux versions might still use python/f2py for python 2
    if os.path.basename(sys.executable).endswith('2'):
        for k in ('f2py2','f2py2.6','f2py2.7',):
            # if we found the f2py path, no need to look further
            if which(k):
                f2py_call = k
                f2py_path = which(k)
                break
    # on Windows and other Linux using python/f2py
    else:
        for k in ('f2py','f2py2.6','f2py2.7','f2py.py',):
            # if we found the f2py path, no need to look further
            if which(k):
                f2py_call = k
                f2py_path = which(k)
                break
    
    try:
        print 'found f2py in:', f2py_path
        return f2py_call
    # raise exception if f2py is not found, f2py_path variable will not exist
    except NameError:
        raise UserWarning, \
        'Couldn\'t locate f2py. Should be part of NumPy installation.'
#        # this might vary among specific cases: f2py, f2py2.7, f2py3.2, ...
#        # TODO: more robust approach, find out what f2py is in the users path
#    if os.name == 'posix':
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

def compile_all():
    f2py_call = f2py_call_str()    
    print '='*75
    print 'compiling c_codes'
    print '='*75
    
    compile_format = f2py_call + ' %s %s -c'
    
    pyfs = ('c_library.pyf',)
    files =('c_functions.c',)

    for pyf,file in zip(pyfs,files):
        os.system(compile_format % (pyf,file))
 
if __name__=='__main__':
    compile_all()
