"""
Builds rindmod.pyd
"""
import os

def compile_all():
    files = ['intmodule',  'jacobmod', 'swapmod', 'fimod','rindmod','rind71mod']
    compile1_format = 'gfortran -fPIC -c %s.f'
    format1 = '%s.o ' * len(files)
    for file in files:
        os.system(compile1_format % file)
    file_objects = format1  % tuple(files)
    
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
    
    os.system(f2py_call + ' -m rindmod  -c %s rind_interface.f ' % file_objects)
    
if __name__=='__main__':
    compile_all()
