import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from wafo.spectrum.models import (Bretschneider, Jonswap, OchiHubble, Tmaspec,
                                  Torsethaugen, McCormick, Wallop, Spreading)


class TestCase(unittest.TestCase):
    def assertListAlmostEqual(self, list1, list2, decimal=5, msg=''):
        assert_array_almost_equal(list1, list2, decimal, msg)


class TestSpectra(TestCase):
    def test_bretschneider(self):
        S = Bretschneider(Hm0=6.5, Tp=10)
        vals = S((0, 1, 2, 3)).tolist()
        true_vals = [0.,  1.69350993,  0.06352698,  0.00844783]
        self.assertListAlmostEqual(vals, true_vals)

    def test_if_jonswap_with_gamma_one_equals_bretschneider(self):
        S = Jonswap(Hm0=7, Tp=11, gamma=1)
        vals = S((0, 1, 2, 3))
        true_vals = np.array([0.,  1.42694133,  0.05051648,  0.00669692])
        self.assertListAlmostEqual(vals, true_vals)
        w = np.linspace(0, 5)
        S2 = Bretschneider(Hm0=7, Tp=11)
        # JONSWAP with gamma=1 should be equal to Bretscneider:
        self.assertListAlmostEqual(S(w), S2(w))

    def test_tmaspec(self):
        S = Tmaspec(Hm0=7, Tp=11, gamma=1, h=10)
        vals = S((0, 1, 2, 3))
        true_vals = np.array([0.,  0.70106233,  0.05022433,  0.00669692])
        self.assertListAlmostEqual(vals, true_vals)

    def test_torsethaugen(self):
        S = Torsethaugen(Hm0=7, Tp=11, gamma=1, h=10)
        vals = S((0, 1, 2, 3))
        true_vals = np.array([0.,  1.19989709,  0.05819794,  0.0093541])
        self.assertListAlmostEqual(vals, true_vals)

        vals = S.wind(range(4))
        true_vals = np.array([0.,  1.13560528,  0.05529849,  0.00888989])
        self.assertListAlmostEqual(vals, true_vals)

        vals = S.swell(range(4))
        true_vals = np.array([0.,  0.0642918,  0.00289946,  0.00046421])
        self.assertListAlmostEqual(vals, true_vals)

    def test_ochihubble(self):

        S = OchiHubble(par=2)
        vals = S(range(4))
        true_vals = np.array([0.,  0.90155636,  0.04185445,  0.00583207])
        self.assertListAlmostEqual(vals, true_vals)

    def test_mccormick(self):

        S = McCormick(Hm0=6.5, Tp=10)
        vals = S(range(4))
        true_vals = np.array([0.,  1.87865908,  0.15050447,  0.02994663])
        self.assertListAlmostEqual(vals, true_vals)

    def test_wallop(self):
        S = Wallop(Hm0=6.5, Tp=10)
        vals = S(range(4))
        true_vals = np.array([0.00000000e+00, 9.36921871e-01, 2.76991078e-03,
                              7.72996150e-05])
        self.assertListAlmostEqual(vals, true_vals)


class TestSpreading(TestCase):
    def test_cos2s(self):
        theta = np.linspace(0, 2 * np.pi)
        d = Spreading(type='cos2s')
        dvals = [[1.10168934e+00],
                 [1.03576796e+00],
                 [8.60302298e-01],
                 [6.30309013e-01],
                 [4.06280137e-01],
                 [2.29514882e-01],
                 [1.13052757e-01],
                 [4.82339343e-02],
                 [1.76754409e-02],
                 [5.50490020e-03],
                 [1.43800617e-03],
                 [3.09907242e-04],
                 [5.39672445e-05],
                 [7.39553743e-06],
                 [7.70796579e-07],
                 [5.84247670e-08],
                 [3.03264905e-09],
                 [9.91950201e-11],
                 [1.81442131e-12],
                 [1.55028269e-14],
                 [4.63223469e-17],
                 [2.90526245e-20],
                 [1.35842977e-24],
                 [3.26077455e-31],
                 [1.65021852e-45],
                 [1.65021852e-45],
                 [3.26077455e-31],
                 [1.35842977e-24],
                 [2.90526245e-20],
                 [4.63223469e-17],
                 [1.55028269e-14],
                 [1.81442131e-12],
                 [9.91950201e-11],
                 [3.03264905e-09],
                 [5.84247670e-08],
                 [7.70796579e-07],
                 [7.39553743e-06],
                 [5.39672445e-05],
                 [3.09907242e-04],
                 [1.43800617e-03],
                 [5.50490020e-03],
                 [1.76754409e-02],
                 [4.82339343e-02],
                 [1.13052757e-01],
                 [2.29514882e-01],
                 [4.06280137e-01],
                 [6.30309013e-01],
                 [8.60302298e-01],
                 [1.03576796e+00],
                 [1.10168934e+00]]

        self.assertListAlmostEqual(d(theta)[0], dvals)


if __name__ == '__main__':
    unittest.main()
