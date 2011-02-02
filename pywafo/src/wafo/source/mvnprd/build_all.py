"""
builds mvnprdmod.pyd
"""
import os

def compile_all():
    files = ['mvnprd', 'mvnprodcorrprb']
    compile1_format = 'gfortran -fPIC -c %s.f'
    for file in files:
        os.system(compile1_format % file)
    file_objects = '%s.o %s.o' % tuple(files)
    
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
    
    #os.system('f2py.py -m mvnprdmod  -c %s mvnprd_interface.f --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71' % file_objects)
    os.system(f2py_call + ' -m mvnprdmod  -c %s mvnprd_interface.f ' % file_objects)

if __name__=='__main__':
    compile_all()
