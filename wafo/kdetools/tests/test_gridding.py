'''
Created on 23. des. 2016

@author: pab
'''
from __future__ import division
import unittest
import numpy as np
from numpy.testing import assert_allclose
import wafo.kdetools.gridding as wkg
from wafo.kdetools.tests.data import DATA1D, DATA2D, DATA3D


class TestKdeTools(unittest.TestCase):
    @staticmethod
    def test_gridcount_1d():
        data = DATA1D
        x = np.linspace(0, max(data) + 1, 10)

        dx = x[1] - x[0]
        c = wkg.gridcount(data, x)
        assert_allclose(c.sum(), len(data))
        assert_allclose(c,
                        [0.1430937435034, 5.864465648665, 9.418694957317207,
                         2.9154367000439, 0.6583089504704, 0.0,
                         0.12255097773682266, 0.8774490222631774, 0.0, 0.0])
        t = np.trapz(c / dx / len(data), x)
        assert_allclose(t, 0.9964226564124143)

    @staticmethod
    def test_gridcount_2d():
        N = 20
        data = DATA2D
        x = np.linspace(0, max(np.ravel(data)) + 1, 5)
        dx = x[1] - x[0]
        X = np.vstack((x, x))
        c = wkg.gridcount(data, X)
        assert_allclose(c.sum(), N)
        assert_allclose(c,
                        [[0.38922806, 0.8987982,  0.34676493, 0.21042807,  0.],
                         [1.15012203, 5.16513541, 3.19250588, 0.55420752,  0.],
                         [0.74293418, 3.42517219, 1.97923195, 0.76076621,  0.],
                         [0.02063536, 0.31054405, 0.71865964, 0.13486633,  0.],
                         [0.,  0.,  0.,  0.,  0.]], 1e-5)

        t = np.trapz(np.trapz(c / (dx**2 * N), x), x)
        assert_allclose(t, 0.9011618785736376)

    @staticmethod
    def test_gridcount_3d():
        N = 20
        data = DATA3D
        x = np.linspace(0, max(np.ravel(data)) + 1, 3)
        dx = x[1] - x[0]
        X = np.vstack((x, x, x))
        c = wkg.gridcount(data, X)
        assert_allclose(c.sum(), N)
        assert_allclose(c,
                        [[[8.74229894e-01, 1.27910940e+00, 1.42033973e-01],
                          [1.94778915e+00, 2.59536282e+00, 3.28213680e-01],
                          [1.08429416e-01, 1.69571495e-01, 7.48896775e-03]],
                         [[1.44969128e+00, 2.58396370e+00, 2.45459949e-01],
                          [2.28951650e+00, 4.49653348e+00, 2.73167915e-01],
                          [1.10905565e-01, 3.18733817e-01, 1.12880816e-02]],
                         [[7.49265424e-02, 2.18142488e-01, 0.0],
                          [8.53886762e-02, 3.73415131e-01, 0.0],
                          [4.16196568e-04, 1.62218824e-02, 0.0]]])

        t = np.trapz(np.trapz(np.trapz(c / dx**3 / N, x), x), x)
        assert_allclose(t, 0.5164999727560187)

    @staticmethod
    def test_gridcount_4d():

        N = 10
        data = np.reshape(DATA2D, (4, N))
        x = np.linspace(0, max(np.ravel(data)) + 1, 3)
        dx = x[1] - x[0]
        X = np.vstack((x, x, x, x))
        c = wkg.gridcount(data, X)
        truth = [[[[1.77163904e-01, 1.87720108e-01, 0.0],
                   [5.72573585e-01, 6.09557834e-01, 0.0],
                   [3.48549923e-03, 4.05931870e-02, 0.0]],
                  [[1.83770124e-01, 2.56357594e-01, 0.0],
                   [4.35845892e-01, 6.14958970e-01, 0.0],
                   [3.07662204e-03, 3.58312786e-02, 0.0]],
                  [[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]]],
                 [[[3.41883175e-01, 5.97977973e-01, 0.0],
                   [5.72071865e-01, 8.58566538e-01, 0.0],
                     [3.46939323e-03, 4.04056116e-02, 0.0]],
                  [[3.58861043e-01, 6.28962785e-01, 0.0],
                     [8.80697705e-01, 1.47373158e+00, 0.0],
                     [2.22868504e-01, 1.18008528e-01, 0.0]],
                  [[2.91835067e-03, 2.60268355e-02, 0.0],
                     [3.63686503e-02, 1.07959459e-01, 0.0],
                     [1.88555613e-02, 7.06358976e-03, 0.0]]],
                 [[[3.13810608e-03, 2.11731327e-02, 0.0],
                   [6.71606255e-03, 4.53139824e-02, 0.0],
                     [0.0, 0.0, 0.0]],
                  [[7.05946179e-03, 5.44614852e-02, 0.0],
                   [1.09099593e-01, 1.95935584e-01, 0.0],
                     [6.61257395e-02, 2.47717418e-02, 0.0]],
                  [[6.38695629e-04, 5.69610302e-03, 0.0],
                     [1.00358265e-02, 2.44053065e-02, 0.0],
                     [5.67244468e-03, 2.12498697e-03, 0.0]]]]
        assert_allclose(c.sum(), N)
        assert_allclose(c, truth)

        t = np.trapz(np.trapz(np.trapz(np.trapz(c / dx**4 / N, x), x), x), x)
        assert_allclose(t, 0.4236703654904251)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
