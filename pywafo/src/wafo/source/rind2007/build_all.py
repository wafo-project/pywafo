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
    
    #os.system('f2py.py -m rindmod  -c %s rind_interface.f --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71' % file_objects)
    os.system('f2py.py -m rindmod  -c %s rind_interface.f ' % file_objects)
    
if __name__=='__main__':
    compile_all()
