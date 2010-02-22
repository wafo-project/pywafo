import os

def compile_all():
    # Install gfortran and run the following to build the module:
    #compile_format = 'f2py %s %s -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71'

    # Install microsoft visual c++ .NET 2003 and run the following to build the module:
    compile_format = 'f2py %s %s -c'
    pyfs = ('rfc.pyf','diffsumfunq.pyf')
    files =('findrfc.c','disufq1.c')

    for pyf,file in zip(pyfs,files):
        os.system(compile_format % (pyf,file))

if __name__=='__main__':
    compile_all()
