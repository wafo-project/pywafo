# -*- coding:utf-8 -*-
"""
Created on 5. aug. 2010

@author: pab
"""

import unittest
from numpy.testing import TestCase, assert_array_almost_equal
import wafo.data  # @UnusedImport
import numpy as np  # @UnusedImport
import wafo.objects as wo
import wafo.spectrum.models as sm
import wafo.transform.models as tm


class TestObjects(TestCase):
    def test_timeseries(self):

        x = wafo.data.sea()
        ts = wo.mat2timeseries(x)
        assert_array_almost_equal(ts.sampling_period(), 0.25)

        S = ts.tospecdata()
        assert_array_almost_equal(S.data[:10],
                                  [0.00913087,  0.00881073,  0.00791944,
                                   0.00664244,  0.00522429, 0.00389816,
                                   0.00282753,  0.00207843,  0.00162678,
                                   0.0013916])

        rf = ts.tocovdata(lag=150)
        assert_array_almost_equal(rf.data[:10],
                                  [0.22368637,  0.20838473,  0.17110733,
                                   0.12237803,  0.07024054, 0.02064859,
                                   -0.02218831, -0.0555993, -0.07859847,
                                   -0.09166187])

    def test_timeseries_trdata(self):
        Hs = 7.0
        Sj = sm.Jonswap(Hm0=Hs)
        S = Sj.tospecdata()  # Make spectrum object from numerical values
        S.tr = tm.TrOchi(mean=0, skew=0.16, kurt=0, sigma=Hs/4, ysigma=Hs/4)
        xs = S.sim(ns=2**20, iseed=10)
        ts = wo.mat2timeseries(xs)
        g0, _gemp = ts.trdata(monitor=False)  # Not Monitor the development

        # Equal weight on all points
        g1, _gemp = ts.trdata(method='mnonlinear', gvar=0.5)

        # Less weight on the ends
        g2, _gemp = ts.trdata(method='nonlinear', gvar=[3.5, 0.5, 3.5])
        self.assert_(1.2 < S.tr.dist2gauss() < 1.6)
        self.assert_(1.65 < g0.dist2gauss() < 2.05)
        self.assert_(0.54 < g1.dist2gauss() < 0.95)
        self.assert_(1.5 < g2.dist2gauss() < 1.9)

    def test_cycles_and_levelcrossings(self):

        x = wafo.data.sea()
        ts = wo.mat2timeseries(x)

        tp = ts.turning_points()
        assert_array_almost_equal(tp.data[:10],
                                  [-1.200495,  0.839505, -0.090495, -0.020495,
                                   -0.090495, -0.040495, -0.160495,  0.259505,
                                   -0.430495, -0.080495]
                                  )

        mm = tp.cycle_pairs()
        assert_array_almost_equal(mm.data[:10],
                                  [0.839505, -0.020495, -0.040495,  0.259505,
                                   -0.080495, -0.080495, 0.349505,  0.859505,
                                   0.009505,  0.319505])

        lc = mm.level_crossings()

        assert_array_almost_equal(lc.data[:10],
                                  [0.,  1.,  2.,  2.,  3.,  4.,
                                   5.,  6.,  7.,  9.])

    def test_levelcrossings_extrapolate(self):
        x = wafo.data.sea()
        ts = wo.mat2timeseries(x)

        tp = ts.turning_points()
        mm = tp.cycle_pairs()
        lc = mm.level_crossings()

        s = x[:, 1].std()
        lc_gpd = lc.extrapolate(-2 * s, 2 * s, dist='rayleigh')
        assert_array_almost_equal(lc_gpd.data[:10],
                                  [1.789254e-37,   2.610988e-37,
                                   3.807130e-37,   5.546901e-37,
                                   8.075384e-37,   1.174724e-36,
                                   1.707531e-36,   2.480054e-36,
                                   3.599263e-36,   5.219466e-36])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
