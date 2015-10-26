"""
builds mvn.pyd
"""
import os
import sys


from wafo.f2py_tools import f2py_call_str

def compile_all():
    f2py_call = f2py_call_str()
    print '=' * 75
    print 'compiling mvn'
    print '=' * 75

    os.system(f2py_call + ' mvn.pyf mvndst.f -c ')


if __name__ == '__main__':
    compile_all()
