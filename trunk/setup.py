"""
Install wafo

Usage:

python setup.py install [, --prefix=$PREFIX]

python setup.py develop
python setup.py bdist_wininst
"""
#!/usr/bin/env python
import os, sys

# make sure we import from WAFO in this package, not an installed one:
sys.path.insert(0, os.path.join('src'))
import wafo

if  __file__ == 'setupegg.py':
    # http://peak.telecommunity.com/DevCenter/setuptools
    from setuptools import setup, Extension
else:
    from distutils.core import setup

package_name = "wafo"
subpackages = ('spectrum','data','transform','covariance')
subpackagesfull = [os.path.join(package_name,f) for f in subpackages]
subtests = [os.path.join(subpkg,'test') for subpkg in subpackages]

testscripts = [os.path.join(subtst, f) for subtst in subtests
    for f in os.listdir(os.path.join('src',package_name,subtst))
               if not (f.startswith('.') or f.endswith('~') or
                       f.endswith('.old') or f.endswith('.bak'))]
datadir = 'data'
datafiles = [os.path.join(datadir, f)   for f in os.listdir(os.path.join('src',package_name,datadir))
				if  not (f.endswith('.py') or f.endswith('test') )]
#docs = [os.path.join('doc', f) for f in os.listdir('doc')]
packagedata = testscripts + datafiles + ['c_library.pyd'] #,'disufq1.c','diffsumfunq.pyd','diffsumfunq.pyf','findrfc.c','rfc.pyd','rfc.pyf']


setup(
    version = '0.11',
    author='WAFO-group',
    author_email='wafo@maths.lth.se',
    description = wafo.__doc__,
    license = "GPL",
    url='http://www.maths.lth.se/matstat/wafo/',
	name = package_name.upper(),
    package_dir = {'': 'src'},
    packages = [package_name,] + list(subpackagesfull),
    package_data = {package_name: packagedata},
    #package_data = {'': ['wafo.cfg']},
    #scripts = [os.path.join('bin', f)
    #           for f in os.listdir('bin')
    #           if not (f.startswith('.') or f.endswith('~') or
    #                   f.endswith('.old') or f.endswith('.bak'))],
    )
