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
import sys
import subprocess
import re
import warnings

MAJOR               = 0
MINOR               = 1
MICRO               = 2
ISRELEASED          = True
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


#sys.argv.append("develop")
#sys.argv.append("install")
DISTUTILS_DEBUG = True
pkg_name = 'wafo'
root_dir = os.path.join('src',pkg_name)

# make sure we import from this package, not an installed one:
sys.path.insert(0, root_dir)
import info
#import wafo

if  True: #__file__ == 'setupegg.py':
    # http://peak.telecommunity.com/DevCenter/setuptools
    from setuptools import setup, Extension, find_packages
else:
    from distutils.core import setup


# Return the svn version as a string, raise a ValueError otherwise
def svn_version():
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
    fid = open(os.path.join(root_dir,filename), 'w')
    try:
        fid.write(cnt % {'version': VERSION, 'isrelease': str(ISRELEASED)})
    finally:
        fid.close()

if __name__=='__main__':
    write_version_py()
    
    packages = find_packages('src')
    for p in packages:
        print(p)
    package_paths =[p.replace(pkg_name+'.','').replace(pkg_name,'').replace('.',os.path.sep) for p in packages]
    test_paths = [os.path.join(pkg_path,'test') for pkg_path in package_paths
                  if os.path.exists(os.path.join(root_dir,pkg_path,'test'))]
    testscripts = [os.path.join(subtst, f) for subtst in test_paths
        for f in os.listdir(os.path.join(root_dir, subtst))
                   if not (f.startswith('.') or f.endswith('~') or
                           f.endswith('.old') or f.endswith('.bak'))]
    
    #subpackages = ('spectrum','data','transform','covariance')
    #subpackagesfull = [os.path.join(pkg_name,f) for f in subpackages]
    
    #subtests = [os.path.join(subpkg,'test') for subpkg in subpackages]
    
    #testscripts = [os.path.join(subtst, f) for subtst in subtests
    #    for f in os.listdir(os.path.join(root_dir, subtst))
    #               if not (f.startswith('.') or f.endswith('~') or
    #                       f.endswith('.old') or f.endswith('.bak'))]
    datadir = 'data'
    datafiles = [os.path.join(datadir, f)   for f in os.listdir(os.path.join(root_dir, datadir))
    				if  not (f.startswith('.') or f.endswith('~') or
                           f.endswith('.old') or f.endswith('.bak') or 
                           f.endswith('.py') or f.endswith('test') )]
    libs = [f   for f in os.listdir(os.path.join(root_dir)) if  f.endswith('.pyd') ]
    packagedata = testscripts + datafiles + libs #['c_library.pyd'] #,'disufq1.c','diffsumfunq.pyd','diffsumfunq.pyf','findrfc.c','rfc.pyd','rfc.pyf']
    
    
    setup(
        version = VERSION,
        author='WAFO-group',
        author_email='wafo@maths.lth.se',
        decription = 'Statistical analysis and simulation of random waves and random loads',
        long_description = info.__doc__,
    	 install_requires = ['numpy>=1.4','numdifftools>=0.2'],
        license = "GPL",
        url='http://code.google.com/p/pywafo/',
    	name = pkg_name,
        package_dir = {'': 'src'},
        packages = packages,
        package_data = {'': packagedata}, 
        classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Education',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License (GPL)',
              'Operating System :: Microsoft :: Windows',
              'Programming Language :: Python :: 2.6',
              'Topic :: Scientific/Engineering :: Mathematics',
              ],
        #packages = [package_name,] + list(subpackagesfull),
        #package_data = {package_name: packagedata},
        #package_data = {'': ['wafo.cfg']},
        #scripts = [os.path.join('bin', f)
        #           for f in os.listdir('bin')
        #           if not (f.startswith('.') or f.endswith('~') or
        #                   f.endswith('.old') or f.endswith('.bak'))],
        )
