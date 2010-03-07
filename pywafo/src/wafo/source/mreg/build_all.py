"""
f2py c_library.pyf c_functions.c -c

gfortran -W -Wall -pedantic-errors -fbounds-check -Werror -c dsvdc.f mregmodule.f

"""
import os

def compile_all():
    files = ['mregmodule',  'dsvdc']
    compile1_format = 'gfortran -fPIC -c %s.f'
    format1 = '%s.o ' * len(files)
    for file in files:
        os.system(compile1_format % file)
    file_objects = format1  % tuple(files)
    #f2py --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71 -m mymod  -c mymod.f90

    os.system('f2py -m cov2mod  -c %s cov2mmpdfreg_intfc.f --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71' % file_objects)
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
