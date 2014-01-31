# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:59:12 2012

@author: pab
"""
import numpy as np


def magic(n):
    ix = np.arange(n) + 1
    J, I = np.meshgrid(ix, ix)
    A = np.mod(I + J - (n + 3) / 2, n)
    B = np.mod(I + 2 * J - 2, n)
    M = n * A + B + 1
    return M
