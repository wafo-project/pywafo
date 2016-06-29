# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:55:34 2015

@author: dave
"""


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('wafo', parent_package, top_path)
    config.add_subpackage('source')
    config.make_config_py()
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
