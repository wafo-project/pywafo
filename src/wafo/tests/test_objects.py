# -*- coding:utf-8 -*-
"""
Created on 5. aug. 2010

@author: pab
"""

import unittest
from numpy.testing import TestCase, assert_allclose
import wafo.data
import wafo.objects as wo
import wafo.spectrum.models as sm
import wafo.transform.models as tm


class TestTimeSeries(TestCase):
    def setUp(self):
        x = wafo.data.sea()
        self.ts = wo.mat2timeseries(x)

    def test_sampling_period(self):
        ts = self.ts
        assert_allclose(ts.sampling_period(), 0.25)

    def test_tospecdata(self):
        S = self.ts.tospecdata(L=150)
        # print(S.data[:10].tolist())
        assert_allclose(S.data[:10],
                                  [0.0050789888306202345, 0.0049411187454784225,
                                   0.004553923924951667, 0.003990722577978725,
                                   0.00335482379127744, 0.002755110296973988,
                                   0.002281782794825119, 0.0019941282234629933,
                                   0.0019329154962902488, 0.002164040256079313])

    def test_tocovdata(self):
        rf = self.ts.tocovdata(lag=150)
        # print(rf.data[:10].tolist())
        rf_truth = [0.2236863694370409, 0.20838472806306807, 0.17110733468261707,
                    0.12237802905232559, 0.07024054173332359, 0.020648586944915163,
                    -0.022188309948145163, -0.05559930301041356, -0.07859846868575099,
                    -0.09166187410953541]

        assert_allclose(rf.data[:10], rf_truth)

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
        assert 1.2 < S.tr.dist2gauss() < 1.6
        assert 1.65 < g0.dist2gauss() < 2.05
        assert 0.54 < g1.dist2gauss() < 0.95
        assert 1.5 < g2.dist2gauss() < 1.9

    def test_timeseries_wave_periods(self):
        true_t = ([-0.69, -0.86, -1.05],
                  [0.42, 0.78, 1.37],
                  [0.09, 0.51, -0.85],
                  [-0.27, -0.08, 0.32],
                  [3.84377468, 6.35707656, 4.15490909],
                  [6.25273295, 6.10295202, 3.36978685],
                  [2.48364668, 4.74282402, 1.75553431],
                  [3.76908628, 1.360128, 1.61425254],
                  [-5.05027968, -9.16405436, -15.60113092],
                  [7.53392635, 13.90687837, 17.35666522],
                  [-0.2811934, -7.11392635, -13.12687837],
                  [4.05027968, 8.47405436, 14.74113092],
                  [2.03999996, 0.07, 0.05],
                  [-0.93, -0.07, -0.12],
                  [1.10999996, 0., -0.07],
                  [-0.86, -0.02, 0.3],
                  [0.93, -0.8, -0.2],
                  [1.10999996, 0., -0.07],
                  [-0.02, 0.3, -0.34],
                  [6.10295202, 3.36978685, 3.58501107],
                  [6.25273295, 6.10295202, 3.36978685],
                  )

        pdefs = ['t2c', 'c2t', 't2t', 'c2c',
                 'd2d', 'u2u', 'd2u', 'u2d',
                 'd2t', 't2u', 'u2c', 'c2d',
                 'm2M', 'M2m', 'm2m', 'M2M', 'all',
                 ]
        ts = wo.TimeSeries(self.ts.data[0:400, :2], self.ts.args[:400])
        for pdef, truth in zip(pdefs, true_t):
            T, _ix = ts.wave_periods(vh=0.0, pdef=pdef)
            # print(T[:3,])
            assert_allclose(T[:3], truth)

        true_t2 = ([1.10999996, 0., - 0.07],
                   [-0.02, 0.3, - 0.34],
                   [6.10295202, 3.369787,  3.585011],
                   [6.25273295,  6.102952,  3.369787],
                   [-0.27, -0.08, 0.32],
                   [-0.27, -0.08, 0.32])
        wdefs = ['mw', 'Mw', 'dw', 'uw', 'tw', 'cw', ]
        for wdef, truth in zip(wdefs, true_t2):
            pdef = '{0}2{0}'.format(wdef[0].lower())
            T, _ix = ts.wave_periods(vh=0.0, pdef=pdef, wdef=wdef)
            # print(T[:3])
            assert_allclose(T[:3], truth)


class TestObjects(TestCase):
    def setUp(self):
        x = wafo.data.sea()
        self.ts = wo.mat2timeseries(x)

    def test_cycles_and_levelcrossings(self):
        tp = self.ts.turning_points()

        tp_truth = [-1.2004945, 0.83950546, -0.09049454, -0.02049454, -0.09049454,
                    -0.04049454, -0.16049454, 0.25950546, -0.43049454, -0.08049454]
        assert_allclose(tp.data[:10], tp_truth)

        mm = tp.cycle_pairs()
        # print(mm.data[:10].tolist())

        mm_truth = [0.83950546, -0.02049454, -0.04049454, 0.25950546, -0.08049454,
                    -0.08049454, 0.34950546, 0.85950546, 0.0095054599, 0.31950546]


        assert_allclose(mm.data[:10], mm_truth)
        true_lcs = (([0., 1., 2., 2., 3., 4., 5., 6., 7., 9.],
                    [-1.7504945, -1.4404945, -1.4204945, -1.4004945,
                     -1.3704945, -1.3204945, -1.2704945, -1.2604945,
                     -1.2504945, -1.2004945]),
                    ([0., 1., 2., 3., 3., 4., 5., 6., 7., 9.],
                    [-1.7504945, -1.4404945, -1.4204945, -1.4004945,
                     -1.3704945, -1.3204945, -1.2704945, -1.2604945,
                     -1.2504945, -1.2004945]),
                    ([1.,  2.,  3.,  4.,  4.,  5.,  6.,  7.,  9., 11.],
                    [-1.7504945, -1.4404945, -1.4204945, -1.4004945,
                     -1.3704945, -1.3204945, -1.2704945, -1.2604945,
                     -1.2504945, -1.2004945]),
                    ([1.,  2.,  3.,  3.,  4.,  5.,  6.,  7.,  9., 11.],
                    [-1.7504945, -1.4404945, -1.4204945, -1.4004945,
                     -1.3704945, -1.3204945, -1.2704945, -1.2604945,
                     -1.2504945, -1.2004945]))
        for i, true_lc in enumerate(true_lcs):
            true_count, true_levels = true_lc
            lc = mm.level_crossings(kind=i+1)
            assert_allclose(lc.data[:10], true_count)
            assert_allclose(lc.args[:10], true_levels)

    def test_levelcrossings_extrapolate(self):
        tp = self.ts.turning_points()
        mm = tp.cycle_pairs()
        lc = mm.level_crossings()

        s = lc.sigma  # x[:, 1].std()
        ix = slice(0, 1000, 100)
        lc_ray = lc.extrapolate(-2 * s, 2 * s, dist='rayleigh')

        assert_allclose(lc_ray.data[ix],
                        [1.79048319e-37, 9.61443295e-23, 2.05330344e-11, 1.74404876e-03,
                         5.89169345e+01, 5.24000000e+02, 6.72611616e+01, 4.46072713e-01,
                         2.23424122e-04, 8.45157346e-09])
        lc_exp = lc.extrapolate(-2 * s, 2 * s, dist='expon')

        lc_gpd = lc.extrapolate(-2 * s, 2 * s, dist='genpareto')

        assert_allclose(lc_exp.data[ix],
                                  [6.51864195e-12, 1.13025876e-08,
                                   1.95974080e-05, 3.39796881e-02,
                                   5.89169345e+01, 5.24000000e+02,
                                   6.43476951e+01, 1.13478843e+00,
                                   2.00122906e-02, 3.52921977e-04])

        truth = [0.0, 0.0, 0.0, 0.0,
                 58.91693446677314, 524.0,
                 68.04847695471916, 0.14101938976341344,
                 0.0, 0.0]

        # print(lc_gpd.data[ix].tolist())
        assert_allclose(lc_gpd.data[ix], truth)



if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
