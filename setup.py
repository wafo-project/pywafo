#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for wafo.

    This file was generated with PyScaffold 2.4.2, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/

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
  twine upload dist/*

"""

from __future__ import division, absolute_import, print_function

# numpy.distutils will figure out if setuptools is available when imported
# this allows us to combine setuptools use_pyscaffold=True and f2py extensions
import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

import sys


def setup_package_pyscaffold():

    config = Configuration('wafo')

    # -------------------------------------------------------------------------
    # c_library
    config.add_extension('c_library',
                         sources=['wafo/source/c_library/c_library.pyf',
                                  'wafo/source/c_library/c_functions.c'])
    # -------------------------------------------------------------------------
    # mvn
    config.add_extension('mvn',
                         sources=['wafo/source/mvn/mvn.pyf',
                                 'wafo/source/mvn/mvndst.f'])

    # -------------------------------------------------------------------------
    # mvnprdmod
    lib_mvnprdmod_src = ['wafo/source/mvnprd/mvnprd.f',
                         'wafo/source/mvnprd/mvnprodcorrprb.f']
    config.add_library('_mvnprdmod', sources=lib_mvnprdmod_src)
    config.add_extension('mvnprdmod',
                         sources=['wafo/source/mvnprd/mvnprd_interface.f'],
                         libraries=['_mvnprdmod'],
                         depends=(lib_mvnprdmod_src))

    # -------------------------------------------------------------------------
    # cov2mod
    lib_cov2mod_src = ['wafo/source/mreg/dsvdc.f',
                       'wafo/source/mreg/mregmodule.f',
                       'wafo/source/mreg/intfcmod.f']
    config.add_library('_cov2mod', sources=lib_cov2mod_src)
    config.add_extension('cov2mod',
                         sources=['wafo/source/mreg/cov2mmpdfreg_intfc.f'],
                         libraries=['_cov2mod'],
                         include_dirs=['wafo/source/mreg/'],
                         depends=(lib_cov2mod_src))

    # -------------------------------------------------------------------------
    # rindmod
    lib_rindmod_src = ['wafo/source/rind2007/intmodule.f',
                       'wafo/source/rind2007/jacobmod.f',
                       'wafo/source/rind2007/swapmod.f',
                       'wafo/source/rind2007/fimod.f',
                       'wafo/source/rind2007/rindmod.f',
                       'wafo/source/rind2007/rind71mod.f']
    config.add_library('_rindmod', sources=lib_rindmod_src)
    config.add_extension('rindmod',
                         sources=['wafo/source/rind2007/rind_interface.f'],
                         libraries=['_rindmod'],
                         include_dirs=['wafo/source/mreg/'],
                         depends=(lib_rindmod_src))

    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['six', 'pyscaffold>=2.4rc1,<2.5a0'] + sphinx,
          tests_require=['pytest_cov', 'pytest'],
          use_pyscaffold=True,
          **config.todict())


if __name__ == "__main__":
    setup_package_pyscaffold()
