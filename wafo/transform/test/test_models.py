from wafo.transform.models import TrHermite, TrOchi, TrLinear
import numpy as np
from numpy.testing import assert_array_almost_equal

def test_trhermite():

    std = 7. / 4
    g = TrHermite(sigma=std, ysigma=std)
    assert(np.abs(g.dist2gauss() - 0.88230868748851554) < 1e-7)

    assert(g.mean == 0.0)
    assert(g.sigma == 1.75)
    vals = g.dat2gauss([0, 1, 2, 3])
    true_vals = np.array([0.04654321, 1.03176393, 1.98871279, 2.91930895])
    assert((np.abs(vals - true_vals) < 1e-7).all())


def test_trochi():

    std = 7. / 4
    g = TrOchi(sigma=std, ysigma=std)
    assert_array_almost_equal(g.dist2gauss(), 1.4106988010566603)
    assert_array_almost_equal(g.mean, 0.0)
    assert_array_almost_equal(g.sigma, 1.75)
    vals = g.dat2gauss([0, 1, 2, 3])
    true_vals = np.array([6.21927960e-04, 9.90237621e-01, 1.96075606e+00,
                          2.91254576e+00])
    assert_array_almost_equal(vals, true_vals)
    # assert((np.abs(vals - true_vals) < 1e-7).all())


def test_trlinear():

    std = 7. / 4
    g = TrLinear(sigma=std, ysigma=std)
    assert(g.dist2gauss() == 0.0)
    assert(g.mean == 0.0)
    assert(g.sigma == 1.75)
    vals = g.dat2gauss([0, 1, 2, 3])
    true_vals = np.array([0., 1., 2., 3.])
    assert((np.abs(vals - true_vals) < 1e-7).all())

if __name__ == '__main__':
    import nose
    nose.run()
