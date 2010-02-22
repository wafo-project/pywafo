"""
f2py c_library.pyf c_functions.c -c
"""
import os

def compile_all():
    files = ['mvnprd', 'mvnprodcorrprb']
    compile1_format = 'gfortran -fPIC -c %s.f'
    for file in files:
        os.system(compile1_format % file)
    file_objects = '%s.o %s.o' % tuple(files)
    
    os.system('f2py -m mvnprdmod  -c %s mvnprd_interface.f --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71' % file_objects)
    #compile1_txt = 'gfortran -fPIC -c mvnprd.f'
    #compile2_txt = 'f2py -m mvnprdmod  -c mvnprd.o mvnprd_interface.f --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71'
    #os.system(compile1_txt)
    #os.system(compile2_txt)
    # Install gfortran and run the following to build the module:
    #compile_format = 'f2py %s %s -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71'

    # Install microsoft visual c++ .NET 2003 and run the following to build the module:
    #compile_format = 'f2py %s %s -c'
    #pyfs = ('c_library.pyf',)
    #files =('c_functions.c',)

    #for pyf,file in zip(pyfs,files):
    #    os.system(compile_format % (pyf,file))

if __name__=='__main__':
    compile_all()
