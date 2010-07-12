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
import os, sys

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

t = find_packages('src')
subpackages = ('spectrum','data','transform','covariance')
#subpackagesfull = [os.path.join(pkg_name,f) for f in subpackages]

subtests = [os.path.join(subpkg,'test') for subpkg in subpackages]

testscripts = [os.path.join(subtst, f) for subtst in subtests
    for f in os.listdir(os.path.join(root_dir, subtst))
               if not (f.startswith('.') or f.endswith('~') or
                       f.endswith('.old') or f.endswith('.bak'))]
datadir = 'data'
datafiles = [os.path.join(datadir, f)   for f in os.listdir(os.path.join(root_dir, datadir))
				if  not (f.startswith('.') or f.endswith('~') or
                       f.endswith('.old') or f.endswith('.bak') or 
                       f.endswith('.py') or f.endswith('test') )]
libs = [f   for f in os.listdir(os.path.join(root_dir)) if  f.endswith('.pyd') ]
packagedata = testscripts + datafiles + libs #['c_library.pyd'] #,'disufq1.c','diffsumfunq.pyd','diffsumfunq.pyf','findrfc.c','rfc.pyd','rfc.pyf']


setup(
    version = '0.11',
    author='WAFO-group',
    author_email='wafo@maths.lth.se',
    decription = 'Statistical analysis and simulation of random waves and random loads',
    long_description = info.__doc__,
	 install_requires = ['numpy>=1.4','numdifftools>=0.2'],
    license = "GPL",
    url='http://code.google.com/p/pywafo/',
	name = pkg_name,
    package_dir = {'': 'src'},
    packages = find_packages('src'),
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
