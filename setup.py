#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for wafo.


Usage:
Run all tests:
  python setup.py test

  python setup.py doctests

Build documentation

  python setup.py docs

Install
  python setup.py install [, --prefix=$PREFIX]

Build

  python setup.py bdist_wininst

  python setup.py bdist_wheel --universal

  python setup.py sdist

PyPi upload:
  git pull origin
git shortlog v0.9.20..HEAD -w80 --format="* %s" --reverse > log.txt  # update Changes.rst
# update CHANGELOG.rst with info from log.txt

  python build_package.py 0.10.0rc0
  git commit
  git tag v0.10.0rc0 master
  git push --tags
  twine check dist/*   # check
  twine upload dist/*  # wait until the travis report is OK before doing this step.

"""
import os
import re
import sys

# numpy.distutils will figure out if setuptools is available when imported
# this allows us to combine setuptools and f2py extensions
import setuptools
from setuptools import Command
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from distutils.command.sdist import sdist


HERE = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = 'wafo'


def read(*parts):
    with open(os.path.join(*parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(HERE, *file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)  # @UndefinedVariable
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


class Doctest(Command):
    description = 'Run doctests with Sphinx'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.application import Sphinx
        sph = Sphinx('./docs',  # source directory
                     './docs',  # directory containing conf.py
                     './docs/_build',  # output directory
                     './docs/_build/doctrees',  # doctree directory
                     'doctest')  # finally, specify the doctest builder
        sph.build()


def setup_package():
    extra_link_args = ["-static", "-static-libgfortran", "-static-libgcc"] if os.name == 'nt' else []
    config = Configuration(PACKAGE_NAME)

    # -------------------------------------------------------------------------
    # c_library
    config.add_extension('c_library',
                         sources=['source/c_library/c_library.pyf',
                                  'source/c_library/c_functions.c'])
    # -------------------------------------------------------------------------
    # mvn
    config.add_extension('mvn',
                         sources=['source/mvn/mvn.pyf',
                                  'source/mvn/mvndst.f'],
                         extra_link_args=extra_link_args)

    # -------------------------------------------------------------------------
    # mvnprdmod
    lib_mvnprdmod_src = ['source/mvnprd/mvnprd.f',
                         'source/mvnprd/mvnprodcorrprb.f']
    config.add_library('_mvnprdmod', sources=lib_mvnprdmod_src)
    config.add_extension('mvnprdmod',
                         sources=['source/mvnprd/mvnprd_interface.f'],
                         libraries=['_mvnprdmod'],
                         depends=(lib_mvnprdmod_src),
                         extra_link_args=extra_link_args)

    # -------------------------------------------------------------------------
    # cov2mod
    lib_cov2mod_src = ['source/mreg/dsvdc.f',
                       'source/mreg/mregmodule.f',
                       'source/mreg/intfcmod.f']
    config.add_library('_cov2mod', sources=lib_cov2mod_src)
    config.add_extension('cov2mod',
                         sources=['source/mreg/cov2mmpdfreg_intfc.f'],
                         libraries=['_cov2mod'],
                         include_dirs=['source/mreg/'],
                         depends=(lib_cov2mod_src),
                         extra_link_args=extra_link_args)

    # -------------------------------------------------------------------------
    # rindmod
    lib_rindmod_src = ['source/rind2007/intmodule.f',
                       'source/rind2007/jacobmod.f',
                       'source/rind2007/swapmod.f',
                       'source/rind2007/fimod.f',
                       'source/rind2007/rindmod.f',
                       'source/rind2007/rind71mod.f']
    config.add_library('_rindmod', sources=lib_rindmod_src)
    config.add_extension('rindmod',
                         sources=['source/rind2007/rind_interface.f'],
                         libraries=['_rindmod'],
                         include_dirs=['source/mreg/'],
                         depends=(lib_rindmod_src),
                         extra_link_args=extra_link_args)

    config.add_data_dir(('data', 'src/wafo/data'))

#     needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
#     sphinx = ['sphinx'] if needs_sphinx else []
#     setup(setup_requires=['six', 'pyscaffold>=2.4rc1,<2.5a0'] + sphinx,
#           tests_require=['pytest_cov', 'pytest'],
#           use_pyscaffold=True,
#           **config.todict())
    version = find_version('src', PACKAGE_NAME, "__init__.py")
    print("Version: {}".format(version))

    sphinx_requires = ['sphinx>=1.3.1']
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['numpydoc',
              'imgmath',
              'sphinx_rtd_theme>=0.1.7'] + sphinx_requires if needs_sphinx else []
    setup(setup_requires=["pytest-runner"] + sphinx,
          version=version,
          cmdclass={'doctest': Doctest,
                    'sdist': sdist},
          include_package_data=True,
          package_data={# If any package contains *.txt or *.rst files, include them:
                        "": ["*.dat"],},
          extras_require={'build_sphinx': sphinx_requires,},
          **config.todict()
          )


if __name__ == "__main__":
    setup_package()
