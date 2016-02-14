import wafo.spectrum.models as sm
import wafo.transform.models as wtm
import wafo.objects as wo
from wafo.spectrum import SpecData1D
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest


def slow(f):
    f.slow = True
    return f


class TestSpectrum(unittest.TestCase):

    @slow
    def test_tocovmatrix(self):
        Sj = sm.Jonswap()
        S = Sj.tospecdata()
        acfmat = S.tocov_matrix(nr=3, nt=256, dt=0.1)
        vals = acfmat[:2, :]
        true_vals = np.array([[3.06073383,  0.0000000, -1.67748256, 0.],
                              [3.05235423, -0.1674357, -1.66811444,
                               0.18693242]])
        self.assertTrue((np.abs(vals - true_vals) < 1e-7).all())


def test_tocovdata():
    Sj = sm.Jonswap()
    S = Sj.tospecdata()
    Nt = len(S.data) - 1
    acf = S.tocovdata(nr=0, nt=Nt)
    vals = acf.data[:5]

    true_vals = np.array(
        [3.06090339,  2.22658399, 0.45307391, -1.17495501, -2.05649042])
    assert((np.abs(vals - true_vals) < 1e-6).all())


def test_to_t_pdf():
    Sj = sm.Jonswap()
    S = Sj.tospecdata()
    f = S.to_t_pdf(pdef='Tc', paramt=(0, 10, 51), speed=7, seed=100)
    vals = ['{0:2.3f}'.format(val) for val in f.data[:10]]
    truevals = ['0.000', '0.014', '0.027', '0.040',
                '0.050', '0.059', '0.067', '0.073', '0.077', '0.082']
    for t, v in zip(truevals, vals):
        assert(t == v)

    # estimated error bounds
    vals = ['{0:2.4f}'.format(val) for val in f.err[:10]]
    truevals = ['0.0000', '0.0003', '0.0003', '0.0004',
                '0.0006', '0.0008', '0.0016', '0.0019', '0.0020', '0.0021']
    for t, v in zip(truevals, vals):
        assert(t == v)


@slow
def test_sim():
    Sj = sm.Jonswap()
    S = Sj.tospecdata()
    #ns = 100
    #dt = .2
    #x1 = S.sim(ns, dt=dt)

    import scipy.stats as st
    x2 = S.sim(20000, 20)
    truth1 = [0, np.sqrt(S.moment(1)[0]), 0., 0.]
    funs = [np.mean, np.std, st.skew, st.kurtosis]
    for fun, trueval in zip(funs, truth1):
        res = fun(x2[:, 1::], axis=0)
        m = res.mean()
        sa = res.std()
        #trueval, m, sa
        assert(np.abs(m - trueval) < sa)


@slow
def test_sim_nl():

    Sj = sm.Jonswap()
    S = Sj.tospecdata()
#    ns = 100
#    dt = .2
#    x1 = S.sim_nl(ns, dt=dt)
    import scipy.stats as st
    x2, _x1 = S.sim_nl(ns=20000, cases=40)
    truth1 = [0, np.sqrt(S.moment(1)[0][0])] + S.stats_nl(moments='sk')
    truth1[-1] = truth1[-1] - 3

    # truth1
    #[0, 1.7495200310090633, 0.18673120577479801, 0.061988521262417606]

    funs = [np.mean, np.std, st.skew, st.kurtosis]
    for fun, trueval in zip(funs, truth1):
        res = fun(x2.data, axis=0)
        m = res.mean()
        sa = res.std()
        # trueval, m, sa
        assert(np.abs(m - trueval) < 2 * sa)


def test_stats_nl():

    Hs = 7.
    Sj = sm.Jonswap(Hm0=Hs, Tp=11)
    S = Sj.tospecdata()
    me, va, sk, ku = S.stats_nl(moments='mvsk')
    assert(me == 0.0)
    assert_array_almost_equal(va, 3.0608203389019537)
    assert_array_almost_equal(sk, 0.18673120577479801)
    assert_array_almost_equal(ku, 3.0619885212624176)


def test_testgaussian():

    Hs = 7
    Sj = sm.Jonswap(Hm0=Hs)
    S0 = Sj.tospecdata()
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


def test_moment():
    Sj = sm.Jonswap(Hm0=5)
    S = Sj.tospecdata()  # Make spectrum ob
    vals, txt = S.moment()
    true_vals = [1.5614600345079888, 0.95567089481941048]
    true_txt = ['m0', 'm0tt']
    for tv, v in zip(true_vals, vals):
        assert_array_almost_equal(tv, v)
    for tv, v in zip(true_txt, txt):
        assert(tv==v)


def test_nyquist_freq():
    Sj = sm.Jonswap(Hm0=5)
    S = Sj.tospecdata()  # Make spectrum ob
    assert(S.nyquist_freq() == 3.0)


def test_sampling_period():
    Sj = sm.Jonswap(Hm0=5)
    S = Sj.tospecdata()  # Make spectrum ob
    assert(S.sampling_period() == 1.0471975511965976)


def test_normalize():
    Sj = sm.Jonswap(Hm0=5)
    S = Sj.tospecdata()  # Make spectrum ob
    S.moment(2)
    ([1.5614600345079888, 0.95567089481941048], ['m0', 'm0tt'])
    vals, _txt = S.moment(2)
    true_vals = [1.5614600345079888, 0.95567089481941048]
    for tv, v in zip(true_vals, vals):
        assert_array_almost_equal(tv, v)

    Sn = S.copy()
    Sn.normalize()

    # Now the moments should be one
    new_vals, _txt = Sn.moment(2)
    for v in new_vals:
        assert(np.abs(v - 1.0) < 1e-7)


def test_characteristic():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=5)
    >>> S = Sj.tospecdata() #Make spectrum ob
    >>> S.characteristic(1)
    (array([ 8.59007646]), array([[ 0.03040216]]), ['Tm01'])

    >>> [ch, R, txt] = S.characteristic([1,2,3])  # fact a vector of integers
    >>> ch; R; txt
    array([ 8.59007646,  8.03139757,  5.62484314])
    array([[ 0.03040216,  0.02834263,         nan],
           [ 0.02834263,  0.0274645 ,         nan],
           [        nan,         nan,  0.01500249]])
    ['Tm01', 'Tm02', 'Tm24']

    >>> S.characteristic('Ss')               # fact a string
    (array([ 0.04963112]), array([[  2.63624782e-06]]), ['Ss'])

    >>> S.characteristic(['Hm0','Tm02'])   # fact a list of strings
    (array([ 4.99833578,  8.03139757]), array([[ 0.05292989,  0.02511371],
           [ 0.02511371,  0.0274645 ]]), ['Hm0', 'Tm02'])
    '''


def test_bandwidth():

    Sj = sm.Jonswap(Hm0=3, Tp=7)
    w = np.linspace(0, 4, 256)
    S = SpecData1D(Sj(w), w)  # Make spectrum object from numerical values
    vals = S.bandwidth([0, 1, 2, 3])
    true_vals = np.array([0.73062845,  0.34476034,  0.68277527,  2.90817052])
    assert((np.abs(vals - true_vals) < 1e-7).all())


def test_docstrings():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    import nose
    nose.run()
    # test_docstrings()
    # test_tocovdata()
    # test_tocovmatrix()
    # test_sim()
    # test_bandwidth()
