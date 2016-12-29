'''
Created on 23. des. 2016

@author: pab
'''
from __future__ import division
import unittest
import numpy as np
from numpy.testing import assert_allclose
from numpy import inf
import wafo.kdetools.kernels as wkk
from wafo.kdetools.tests.data import DATA3D


class TestKernels(unittest.TestCase):
    def setUp(self):
        self.names = ['epanechnikov', 'biweight', 'triweight', 'logistic',
                      'p1epanechnikov', 'p1biweight', 'p1triweight',
                      'triangular', 'gaussian', 'rectangular', 'laplace']

    def test_stats(self):
        truth = {
            'biweight': (0.14285714285714285, 0.7142857142857143, 22.5),
            'logistic': (3.289868133696453, 1./6, 0.023809523809523808),
            'p1biweight': (0.14285714285714285, 0.7142857142857143, 22.5),
            'triangular': (0.16666666666666666, 0.6666666666666666, inf),
            'gaussian': (1, 0.28209479177387814, 0.21157109383040862),
            'epanechnikov': (0.2, 0.6, inf),
            'triweight': (0.1111111111111111, 0.8158508158508159, inf),
            'p1triweight': (0.1111111111111111, 0.8158508158508159, inf),
            'p1epanechnikov': (0.2, 0.6, inf),
            'rectangular': (0.3333333333333333, 0.5, inf),
            'laplace': (2, 0.25, inf)}
        for name in self.names:
            kernel = wkk.Kernel(name)
            assert_allclose(kernel.stats(), truth[name])
            # truth[name] = kernel.stats()
        # print(truth)

    def test_norm_factors_1d(self):
        truth = {
            'biweight': 1.0666666666666667, 'logistic': 1.0,
            'p1biweight': 1.0666666666666667, 'triangular': 1.0,
            'gaussian': 2.5066282746310002, 'epanechnikov': 1.3333333333333333,
            'triweight': 0.91428571428571426, 'laplace': 2,
            'p1triweight': 0.91428571428571426,
            'p1epanechnikov': 1.3333333333333333, 'rectangular': 2.0}
        for name in self.names:
            kernel = wkk.Kernel(name)
            assert_allclose(kernel.norm_factor(d=1, n=20), truth[name])
            # truth[name] = kernel.norm_factor(d=1, n=20)

    def test_effective_support(self):
        truth = {'biweight': (-1.0, 1.0), 'logistic': (-7.0, 7.0),
                 'p1biweight': (-1.0, 1.0), 'triangular': (-1.0, 1.0),
                 'gaussian': (-4.0, 4.0), 'epanechnikov': (-1.0, 1.0),
                 'triweight': (-1.0, 1.0), 'p1triweight': (-1.0, 1.0),
                 'p1epanechnikov': (-1.0, 1.0), 'rectangular': (-1.0, 1.0),
                 'laplace': (-7.0, 7.0)}
        for name in self.names:
            kernel = wkk.Kernel(name)
            assert_allclose(kernel.effective_support(), truth[name])
            # truth[name] = kernel.effective_support()
        # print(truth)
        # self.assertTrue(False)

    def test_that_kernel_is_a_pdf(self):

        for name in self.names:
            kernel = wkk.Kernel(name)
            xmin, xmax = kernel.effective_support()
            x = np.linspace(xmin, xmax, 4*1024+1)
            m0 = kernel.norm_factor(d=1, n=1)
            pdf = kernel(x)/m0
            #             print(name)
            #             print(pdf[0], pdf[-1])
            #             print(np.trapz(pdf, x) - 1)
            assert_allclose(np.trapz(pdf, x), 1, 1e-2)
        # self.assertTrue(False)


class TestSmoothing(unittest.TestCase):
    def setUp(self):
        self.data = DATA3D
        self.gauss = wkk.Kernel('gaussian')

    def test_hns(self):
        hs = self.gauss.hns(self.data)
        assert_allclose(hs, [0.18154437, 0.36207987, 0.37396219])

    def test_hos(self):
        hs = self.gauss.hos(self.data)
        assert_allclose(hs, [0.195209, 0.3893332, 0.40210988])

    def test_hms(self):
        hs = self.gauss.hmns(self.data)
        assert_allclose(hs, [[3.25196193e-01, -2.68892467e-02, 3.18932448e-04],
                             [-2.68892467e-02, 3.91283306e-01, 2.38654678e-02],
                             [3.18932448e-04, 2.38654678e-02, 4.05123874e-01]])
        hs = self.gauss.hmns(self.data[0])
        assert_allclose(hs, self.gauss.hns(self.data[0]))

        hs = wkk.Kernel('epan').hmns(self.data)
        assert_allclose(hs,
                        [[8.363847e-01, -6.915749e-02, 8.202747e-04],
                         [-6.915749e-02, 1.006357e+00, 6.138052e-02],
                         [8.202747e-04, 6.138052e-02, 1.041954e+00]],
                        rtol=1e-5)
        hs = wkk.Kernel('biwe').hmns(self.data[:2])
        assert_allclose(hs, [[0.868428, -0.071705],
                             [-0.071705, 1.04685]], rtol=1e-5)
        hs = wkk.Kernel('triwe').hmns(self.data[:2])
        assert_allclose(hs, [[0.975375, -0.080535],
                             [-0.080535, 1.17577]], rtol=1e-5)
        self.assertRaises(NotImplementedError,
                          wkk.Kernel('biwe').hmns, self.data)
        self.assertRaises(NotImplementedError,
                          wkk.Kernel('triwe').hmns, self.data)
        self.assertRaises(NotImplementedError,
                          wkk.Kernel('triangular').hmns, self.data)

    def test_hscv(self):
        hs = self.gauss.hscv(self.data)
        assert_allclose(hs, [0.1656318800590673, 0.3273938258112911,
                             0.31072126996412214])

    def test_hstt(self):
        hs = self.gauss.hstt(self.data)
        assert_allclose(hs, [0.18099075, 0.50409881, 0.11018912])

    def test_hste(self):
        hs = self.gauss.hste(self.data)
        assert_allclose(hs, [0.17035204677390572, 0.29851960273788863,
                             0.186685349741972])

    def test_hldpi(self):
        hs = self.gauss.hldpi(self.data)
        assert_allclose(hs, [0.1732289, 0.33159097, 0.3107633])

    def test_hisj(self):
        hs = self.gauss.hisj(self.data)
        assert_allclose(hs, [0.29542502, 0.74277133, 0.51899114])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
