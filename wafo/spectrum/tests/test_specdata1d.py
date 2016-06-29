import wafo.spectrum.models as sm
import wafo.transform.models as wtm
import wafo.objects as wo
from wafo.spectrum import SpecData1D
import numpy as np
from numpy import NAN
from numpy.testing import assert_array_almost_equal, assert_array_equal
import unittest


def slow(f):
    f.slow = True
    return f


class TestSpectrumHs7(unittest.TestCase):
    def setUp(self):
        self.Sj = sm.Jonswap(Hm0=7.0, Tp=11)
        self.S = self.Sj.tospecdata()

    def test_tocovmatrix(self):
        acfmat = self.S.tocov_matrix(nr=3, nt=256, dt=0.1)
        vals = acfmat[:2, :]
        true_vals = np.array([[3.06073383,  0.0000000, -1.67748256, 0.],
                              [3.05235423, -0.1674357, -1.66811444,
                               0.18693242]])
        assert_array_almost_equal(vals, true_vals)

    def test_tocovdata(self):

        Nt = len(self.S.data) - 1
        acf = self.S.tocovdata(nr=0, nt=Nt)
        vals = acf.data[:5]

        true_vals = np.array(
            [3.06090339,  2.22658399, 0.45307391, -1.17495501, -2.05649042])
        assert_array_almost_equal(vals, true_vals)
        assert((np.abs(vals - true_vals) < 1e-6).all())

    def test_to_t_pdf(self):
        f = self.S.to_t_pdf(pdef='Tc', paramt=(0, 10, 51), speed=7, seed=100)
        vals = ['%2.3f' % val for val in f.data[:10]]
        truevals = ['0.000', '0.014', '0.027', '0.040',
                    '0.050', '0.059', '0.067', '0.073', '0.077', '0.082']
        for t, v in zip(truevals, vals):
            assert(t == v)

        # estimated error bounds
        vals = ['%2.4f' % val for val in f.err[:10]]
        truevals = ['0.0000', '0.0003', '0.0003', '0.0004',
                    '0.0006', '0.0008', '0.0016', '0.0019', '0.0020', '0.0021']
        for t, v in zip(truevals, vals):
            assert(t == v)

    @slow
    def test_sim(self):
        S = self.S

        import scipy.stats as st
        x2 = S.sim(20000, 20)
        truth1 = [0, np.sqrt(S.moment(1)[0]), 0., 0.]
        funs = [np.mean, np.std, st.skew, st.kurtosis]
        for fun, trueval in zip(funs, truth1):
            res = fun(x2[:, 1::], axis=0)
            m = res.mean()
            sa = res.std()
            assert(np.abs(m - trueval) < sa)

    @slow
    def test_sim_nl(self):
        S = self.S

        import scipy.stats as st
        x2, _x1 = S.sim_nl(ns=20000, cases=40)
        truth1 = [0, np.sqrt(S.moment(1)[0][0])] + S.stats_nl(moments='sk')
        truth1[-1] = truth1[-1] - 3

        # truth1
        # [0, 1.7495200310090633, 0.18673120577479801, 0.061988521262417606]

        funs = [np.mean, np.std, st.skew, st.kurtosis]
        for fun, trueval in zip(funs, truth1):
            res = fun(x2.data, axis=0)
            m = res.mean()
            sa = res.std()
            # trueval, m, sa
            assert(np.abs(m - trueval) < 2 * sa)

    def test_stats_nl(self):
        S = self.S
        me, va, sk, ku = S.stats_nl(moments='mvsk')
        assert(me == 0.0)
        assert_array_almost_equal(va, 3.0608203389019537)
        assert_array_almost_equal(sk, 0.18673120577479801)
        assert_array_almost_equal(ku, 3.0619885212624176)

    def test_testgaussian(self):
        Hs = self.Sj.Hm0
        S0 = self.S
        # ns =100; dt = .2
        # x1 = S0.sim(ns, dt=dt)
        S = S0.copy()
        me, _va, sk, ku = S.stats_nl(moments='mvsk')
        S.tr = wtm.TrHermite(
            mean=me, sigma=Hs / 4, skew=sk, kurt=ku, ysigma=Hs / 4)
        ys = wo.mat2timeseries(S.sim(ns=2 ** 13))
        g0, _gemp = ys.trdata()
        t0 = g0.dist2gauss()
        t1 = S0.testgaussian(ns=2 ** 13, test0=None, cases=50)
        assert(sum(t1 > t0) < 5)


class TestSpectrumHs5(unittest.TestCase):
    def setUp(self):
        self.Sj = sm.Jonswap(Hm0=5.0)
        self.S = self.Sj.tospecdata()

    def test_moment(self):
        S = self.S
        vals, txt = S.moment()
        true_vals = [1.5614600345079888, 0.95567089481941048]
        true_txt = ['m0', 'm0tt']

        assert_array_almost_equal(vals, true_vals)
        for tv, v in zip(true_txt, txt):
            assert(tv == v)

    def test_nyquist_freq(self):
        S = self.S
        assert_array_almost_equal(S.nyquist_freq(), 3.0)

    def test_sampling_period(self):
        S = self.S
        assert_array_almost_equal(S.sampling_period(), 1.0471975511965976)

    def test_normalize(self):
        S = self.S
        mom, txt = S.moment(2)
        assert_array_almost_equal(mom,
                                  [1.5614600345079888, 0.95567089481941048])
        assert_array_equal(txt, ['m0', 'm0tt'])
        vals, _txt = S.moment(2)
        true_vals = [1.5614600345079888, 0.95567089481941048]
        assert_array_almost_equal(vals, true_vals)

        Sn = S.copy()
        Sn.normalize()

        # Now the moments should be one
        new_vals, _txt = Sn.moment(2)
        assert_array_almost_equal(new_vals, np.ones(2))

    def test_characteristic(self):
        S = self.S
        ch, R, txt = S.characteristic(1)
        assert_array_almost_equal(ch, 8.59007646)
        assert_array_almost_equal(R,  0.03040216)
        self.assert_(txt == ['Tm01'])

        ch, R, txt = S.characteristic([1, 2, 3])  # fact a vector of integers
        assert_array_almost_equal(ch, [8.59007646,  8.03139757,  5.62484314])
        assert_array_almost_equal(R,
                                  [[0.03040216,  0.02834263,         NAN],
                                   [0.02834263,  0.0274645,         NAN],
                                   [NAN,         NAN,  0.01500249]])
        assert_array_equal(txt, ['Tm01', 'Tm02', 'Tm24'])

        ch, R, txt = S.characteristic('Ss')  # fact a string
        assert_array_almost_equal(ch, [0.04963112])
        assert_array_almost_equal(R, [[2.63624782e-06]])
        assert_array_equal(txt, ['Ss'])

        # fact a list of strings
        ch, R, txt = S.characteristic(['Hm0', 'Tm02'])

        assert_array_almost_equal(ch,
                                  [4.99833578,  8.03139757])
        assert_array_almost_equal(R, [[0.05292989,  0.02511371],
                                      [0.02511371,  0.0274645]])
        assert_array_equal(txt, ['Hm0', 'Tm02'])


class TestSpectrumHs3(unittest.TestCase):
    def test_bandwidth(self):

        Sj = sm.Jonswap(Hm0=3, Tp=7)
        w = np.linspace(0, 4, 256)
        S = SpecData1D(Sj(w), w)  # Make spectrum object from numerical values
        vals = S.bandwidth([0, 1, 2, 3])
        true_vals = [0.73062845,  0.34476034,  0.68277527,  2.90817052]
        assert_array_almost_equal(vals, true_vals)


if __name__ == '__main__':
    import nose
    nose.run()
