"""
builds mvn.pyd
"""
import os

def compile_all():
    #os.system('f2py.py mvn.pyf mvndst.f -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71 ')
    os.system('f2py.py mvn.pyf mvndst.f -c ')
    
if __name__=='__main__':
    compile_all()
