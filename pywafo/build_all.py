"""
builds all extensions

In notepad (or your favourite text editor) create a new text file and enter the following in the file:
[build]
compiler = mingw32

Save the file as "C:\Python25\Lib\distutils\distutils.cfg".
This will tell python to use the MinGW compiler when compiling extensions


See also http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW
"""
import os
import shutil

def compile_all():
    wd = os.getcwd()
    pkg_name = 'wafo'
    root_dir = os.path.join(wd,'src',pkg_name)
    root_src = os.path.join(root_dir, 'source')
    buildscript = 'build_all.py'
    
    # if Linux
    if os.name == 'posix':
        build_call = 'python %s' % buildscript
        lib_ext = '.so'
    
    # On Windows
    elif os.name == 'nt':
        build_call = 'python.exe  %s' % buildscript
        lib_ext = '.pyd'
    
    # give an Error for other OS-es
    else:
        raise UserWarning, \
        'Untested platform:', os.name
    
    for root, dirs, files in os.walk(root_src):
        dir1 = [dir for dir in dirs if not os.path.exists(os.path.join(root,dir,buildscript))]
        for dir in dir1:
            dirs.remove(dir) ## don't visit directories without buildscript
        if buildscript in files:
            print('Building: ', root)
            #buildfile ='python.exe  %s' % os.path.join(root,buildscript) 
            os.chdir(root)
            t = os.system(build_call)
            print(t)
            
            for file in os.listdir('.'):
                if file.endswith(lib_ext):
                    dest_file = os.path.join(root_dir, file)
                    if os.path.exists(dest_file):
                        os.remove(dest_file)
                    shutil.copy(os.path.join(root,file), root_dir)
            
    os.chdir(wd)
    
if __name__=='__main__':
    compile_all()
