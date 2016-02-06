'''
Created on 19. juli 2010

@author: pab
'''
import numpy as np
from wafo.wave_theory.dispersion_relation import w2k, k2w  # @UnusedImport


def test_k2w_infinite_water_depth():
    vals = k2w(np.arange(0.01, .5, 0.2))[0]
    true_vals = np.array([0.3132092,  1.43530485,  2.00551739])
    assert((np.abs(vals - true_vals) < 1e-7).all())


def test_k2w_finite_water_depth():
    vals = k2w(np.arange(0.01, .5, 0.2), h=20)[0]
    true_vals = np.array([0.13914927,  1.43498213,  2.00551724])
    assert((np.abs(vals - true_vals) < 1e-7).all())


def test_w2k_infinite_water_depth():
    vals = w2k(range(4))[0]
    true_vals = np.array([0.,  0.1019368,  0.4077472,  0.91743119])
    assert((np.abs(vals - true_vals) < 1e-7).all())


def test_w2k_finite_water_depth():
    vals = w2k(range(4), h=20)[0]
    true_vals = np.array([0.,  0.10503601,  0.40774726,  0.91743119])
    assert((np.abs(vals - true_vals) < 1e-7).all())

if __name__ == '__main__':
    import nose
    nose.run()
