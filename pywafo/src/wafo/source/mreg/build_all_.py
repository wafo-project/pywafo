"""
f2py c_library.pyf c_functions.c -c

gfortran -W -Wall -pedantic-errors -fbounds-check -Werror -c dsvdc.f mregmodule.f

"""
import os

def compile_all():
    files = ['dsvdc','mregmodule']
    compile1_format = 'gfortran -fPIC -c %s.f'
    format1 = '%s.o ' * len(files)
    for file in files:
        os.system(compile1_format % file)
    file_objects = format1  % tuple(files)
    
    os.system('f2py.py -m cov2mod  -c %s cov2mmpdfreg_intfc.f' % file_objects)
     
    
if __name__=='__main__':
    compile_all()
