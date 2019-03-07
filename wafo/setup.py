# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:55:34 2015

@author: dave
"""

if __name__ == "__main__":
    from setuptools import setup, find_packages
        setup(
            name = 'wafo',
            packages = find_packages(),
            no-vcs = 1,
            format = 'bdist_wheel'
        )
