#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for wafo.

    This file was generated with PyScaffold 2.4.2, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

from __future__ import division, absolute_import, print_function

# numpy.distutils will figure out if setuptools is available when imported
# this allows us to combine setuptools use_pyscaffold=True and f2py extensions
import setuptools
from numpy.distutils.core import setup, Extension

import sys


def setup_package_pyscaffold():

    extensions = []

    c_lib_ext = Extension('wafo.c_library',
                          sources=['wafo/source/c_library/c_library.pyf',
                                   'wafo/source/c_library/c_functions.c'])
    extensions.append(c_lib_ext)

#    rind_ext = Extension('wafo.rindmod',
#                         extra_objects=['wafo/source/rind2007/intmodule.f',
#                                  'wafo/source/rind2007/jacobmod.f',
#                                  'wafo/source/rind2007/swapmod.f',
#                                  'wafo/source/rind2007/fimod.f',
#                                  'wafo/source/rind2007/rindmod.f',
#                                  'wafo/source/rind2007/rind71mod.f'],
#                          sources=['wafo/source/rind2007/rind_interface.f'])
#    extensions.append(rind_ext)
#
#    mreg_ext = Extension('wafo.cov2mod',
#                         sources=['wafo/source/mreg/dsvdc.f',
#                                  'wafo/source/mreg/mregmodule.f',
#                                  'wafo/source/mreg/intfcmod.f',
#                                  'wafo/source/mreg/cov2mmpdfreg_intfc.f'],
#                         include_dirs=['wafo/source/rind2007'])
#    extensions.append(mreg_ext)

#    mvn_ext = Extension('wafo.mvn',
#                        sources=['wafo/source/mvn/mvn.pyf',
#                                 'wafo/source/mvn/mvndst.f'])
#    extensions.append(mvn_ext)

#    mvnprd_ext = Extension('wafo.mvnprdmod',
#                           sources=['wafo/source/mvnprd/mvnprd.f',
#                                    'wafo/source/mvnprd/mvnprodcorrprb.f',
#                                    'wafo/source/mvnprd/mvnprd_interface.f'])
#    extensions.append(mvnprd_ext)


    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['six', 'pyscaffold>=2.4rc1,<2.5a0'] + sphinx,
          tests_require=['pytest_cov', 'pytest'],
          use_pyscaffold=True,
          ext_modules=extensions)

if __name__ == "__main__":
    setup_package_pyscaffold()
