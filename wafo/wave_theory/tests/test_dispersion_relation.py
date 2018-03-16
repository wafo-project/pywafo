'''
Created on 19. juli 2010

@author: pab
'''
import numpy as np
from numpy.testing import assert_allclose
from wafo.wave_theory.dispersion_relation import w2k, k2w  # @UnusedImport


def test_k2w_infinite_water_depth():
    vals = k2w(np.arange(0.01, .5, 0.2))[0]
    true_vals = np.array([0.3132092, 1.43530485, 2.00551739])
    assert_allclose(vals, true_vals)


def test_k2w_finite_water_depth():
    vals, theta = k2w(np.arange(0.01, .5, 0.2), h=20)
    true_vals = (0.13914927, 1.43498213, 2.00551724)
    assert_allclose(vals, true_vals)
    assert_allclose(theta, 0)


def test_k2w_finite_water_depth_with_negative_k():
    vals, theta = k2w(-np.arange(0.01, .5, 0.2), h=20)
    true_vals = [0.13914927, 1.43498213, 2.00551724]
    assert_allclose(vals, true_vals)
    assert_allclose(theta, np.pi)


def test_w2k_infinite_water_depth():
    vals, k2 = w2k(range(4))
    true_vals = np.array([0., 0.1019368, 0.4077472, 0.91743119])
    assert_allclose(vals, true_vals)
    assert_allclose(k2, 0)

def test_w2k_infinite_water_depth_with_negative_w():
    vals, k2 = w2k(-np.arange(4))
    true_vals = -1 * np.array([0., 0.1019368, 0.4077472, 0.91743119])
    assert_allclose(vals, true_vals)
    assert_allclose(k2, 0)

def test_w2k_finite_water_depth():
    vals, k2 = w2k(range(4), h=20)
    true_vals = np.array([0., 0.10503601, 0.40774726, 0.91743119])
    assert_allclose(vals, true_vals)
    assert_allclose(k2, 0)

def test_w2k_finite_water_depth_with_negative_w():
    vals, k2 = w2k(-np.arange(4), h=20)
    true_vals = -1 * np.array([0., 0.10503601, 0.40774726, 0.91743119])
    assert_allclose(vals, true_vals)
    assert_allclose(k2, 0)


if __name__ == '__main__':
    import nose
    nose.run()
