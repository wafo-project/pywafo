from numpy.testing import (run_module_suite, assert_equal,
                           assert_array_almost_equal)
# assert_almost_equal, assert_array_equal)
import wafo.spectrum.models as sm
# from wafo.covariance import CovData1D


def test_covariance():
    Sj = sm.Jonswap()
    S = Sj.tospecdata()  # Make spec
    R = S.tocovdata()
    dt = R.sampling_period()
    assert_equal(dt, 1.0471975511965976)
    S1 = R.tospecdata()
    assert_array_almost_equal(S.data[:10], S1.data[:10], 11)
    x = R.sim(ns=1000, dt=0.2, iseed=0)
    assert_array_almost_equal(x[:10, 0], [0.0, 1.04719755, 2.0943951,
                                          3.14159265, 4.1887902, 5.23598776,
                                          6.28318531, 7.33038286, 8.37758041,
                                          9.42477796], decimal=3)
    assert_array_almost_equal(x[:10, 1], [0.22155905, 1.21207066, 1.95670282,
                                          2.11634902, 1.57967273, 0.2665005,
                                          -0.79630253, -1.31908028,
                                          -2.20056021, -1.84451748], decimal=3)


if __name__ == '__main__':
    run_module_suite()
