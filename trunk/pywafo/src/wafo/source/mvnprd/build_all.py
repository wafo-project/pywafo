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
    
    #os.system('f2py.py -m mvnprdmod  -c %s mvnprd_interface.f --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71' % file_objects)
    os.system('f2py.py -m mvnprdmod  -c %s mvnprd_interface.f ' % file_objects)

if __name__=='__main__':
    compile_all()
