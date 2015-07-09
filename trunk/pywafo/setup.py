"""
Install wafo

Usage:

python setup.py develop
python setup.py install [, --prefix=$PREFIX]

python setup.py bdist_wininst

PyPi upload:

python setup.py sdist bdist_wininst upload --show-response

"""
#!/usr/bin/env python

import os
import shutil
import sys
import subprocess
import re
import warnings
from Cython.Build import cythonize
MAJOR               = 0
MINOR               = 1
MICRO               = 2
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
# sys.argv.append("build_src")
# sys.argv.append("build_ext")
# sys.argv.append("--inplace")
# sys.argv.append("develop")
# sys.argv.append("install")
DISTUTILS_DEBUG = True
PKG_NAME = 'wafo'
ROOT_DIR = os.path.join('src',PKG_NAME)

# make sure we import from this package, not an installed one:
sys.path.insert(0, ROOT_DIR)
import info

from setuptools import find_packages  # setup, Extension
from numpy.distutils.core import setup, Extension  # as FExtension

def svn_version():
    '''Return the svn version as a string, raise a ValueError otherwise'''
    from numpy.compat import asstr

    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    try:
        out = subprocess.Popen(['svn', 'info'], stdout=subprocess.PIPE,
                env=env).communicate()[0]
    except OSError:
        warnings.warn(" --- Could not run svn info --- ")
        return ""

    r = re.compile('Revision: ([0-9]+)')
    svnver = None
    for line in asstr(out).split('\n'):
        m = r.match(line)
        if m:
            svnver = m.group(1)

    if not svnver:
        raise ValueError("Error while parsing svn version ?")
    return svnver

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    # If in git or something, bypass the svn rev
    if os.path.exists('.svn'):
        FULLVERSION += svn_version()

def write_version_py(filename='version.py'):
    cnt = """\
# THIS FILE IS GENERATED FROM SETUP.PY
short_version='%(version)s'
version='%(version)s'
release=%(isrelease)s
"""
    fid = open(os.path.join(ROOT_DIR,filename), 'w')
    try:
        fid.write(cnt % {'version': VERSION, 'isrelease': str(ISRELEASED)})
    finally:
        fid.close()


def get_library_extension():
    '''Return extension of an executable library'''
    if os.name == 'posix':  # executable library on Linux has extension .so
        lib_ext = '.so'
    elif os.name == 'nt':  # extension on Windows is .pyd
        lib_ext = '.pyd'
    else:
        raise UserWarning('Platform not supported:', os.name)
    return lib_ext


def compile_all():
    wd = os.getcwd()
    root_dir = os.path.join(wd,'src',PKG_NAME)
    root_src = os.path.join(root_dir, 'source')
    buildscript = 'build_all.py'
    lib_ext = get_library_extension()

    if os.name == 'nt':  # On Windows
        build_call = 'python.exe  %s' % buildscript
    else:
        build_call = 'python %s' % buildscript
    
    for root, dirs, files in os.walk(root_src):
        dir1 = [dir for dir in dirs
                if not os.path.exists(os.path.join(root, dir, buildscript))]
        for dir in dir1:
            dirs.remove(dir)  # don't visit directories without buildscript
        if buildscript in files:
            print('Building: ', root) 
            os.chdir(root)
            t = os.system(build_call)
            print(t)

            for file in os.listdir('.'):
                if file.endswith(lib_ext):
                    dest_file = os.path.join(root_dir, file)
                    if os.path.exists(dest_file):
                        os.remove(dest_file)
                    shutil.copy(os.path.join(root, file), root_dir)
    os.chdir(wd)


def setup_package():
    write_version_py()
    join = os.path.join
    packages = find_packages('src')
    for p in packages:
        print(p)
    def convert_package2path(p):
        return p.replace(PKG_NAME + '.',
                         '').replace(PKG_NAME, '').replace('.', os.path.sep)
    package_paths = [convert_package2path(p) for p in packages]
    test_paths = [join(pkg_path, 'test') for pkg_path in package_paths
                  if os.path.exists(join(ROOT_DIR, pkg_path, 'test'))]
    testscripts = [join(subtst, f) for subtst in test_paths
                   for f in os.listdir(join(ROOT_DIR, subtst))
                   if not (f.startswith('.') or f.endswith('~') or
                           f.endswith('.old') or f.endswith('.bak'))]

    datadir = 'data'
    datafiles = [join(datadir, f)  for f in os.listdir(join(ROOT_DIR, datadir))
    				if  not (f.startswith('.') or f.endswith('~') or
                           f.endswith('.old') or f.endswith('.bak') or 
                           f.endswith('.py') or f.endswith('test') )]
    if 'build_ext' in sys.argv:
        compile_all()
    lib_ext = get_library_extension()
    libs = [f for f in os.listdir(join(ROOT_DIR)) if f.endswith(lib_ext)]
    
    packagedata = testscripts + datafiles + libs

#     ext_module_list =  cythonize(join(ROOT_DIR, "primes.pyx"))
# 
#     for ext_module in ext_module_list:
#         if not isinstance(ext_module, Extension):
#             ext_module.__class__ = Extension

#     for name, src_files in [('mvn',('mvn.pyf', 'mvndst.f')),
#                             ('c_library',('c_library.pyf', 'c_functions.c'))]: 
#         sources = [join(ROOT_DIR, 'source', name, f) for f in src_files]
#         ext_module_list.append(Extension(name='%s.%s' % (PKG_NAME, name),
#                                          sources=sources))

#     sources = [join(ROOT_DIR, 'source', 'mreg', 'cov2mmpdfreg_intfc.f'), ]
#     libs = [join(ROOT_DIR, 'source', 'mreg', f)
#             for f in ['dsvdc', 'mregmodule', 'intfcmod'] ]
#     ext_module_list.append(Extension(name='wafo.covmod', sources=sources,
#                                      libraries=libs))

#     mvn_sources = [join(root_mvn, 'source', 'mvn', 'mvn.pyf'),
#                    join(root_mvn, 'source', 'mvn','mvndst.f')]
#     ext_module_list.append(Extension(name='wafo.mvn', sources=mvn_sources))

    setup(
        version = VERSION,
        author='WAFO-group',
        author_email='wafo@maths.lth.se',
        description = 'Statistical analysis and simulation of random waves and random loads',
        long_description = info.__doc__,
    	install_requires = ['numpy>=1.4','numdifftools>=0.2'],
        license = "GPL",
        url='http://code.google.com/p/pywafo/',
    	name = PKG_NAME,
        package_dir = {'': 'src'},
        packages = packages,
        package_data = {'': packagedata}, 
        # ext_modules = ext_module_list,
        classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Education',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License (GPL)',
              'Operating System :: Microsoft :: Windows',
              'Programming Language :: Python :: 2.6',
              'Topic :: Scientific/Engineering :: Mathematics',
              ],
        )


if __name__=='__main__':
    setup_package()