from wafo.transform import TrData
import numpy as np


def test_trdata():
    '''
    Construct a linear transformation model
    '''
    sigma = 5
    mean = 1
    u = np.linspace(-5, 5)
    x = sigma * u + mean
    y = u
    g = TrData(y, x)
    assert(g.mean == 1.0)
    print(g.sigma)
    # assert(g.sigma==5.0)

    g = TrData(y, x, mean=1, sigma=5)
    assert(g.mean == 1)
    assert(g.sigma == 5.)
    # vals = g.dat2gauss(1, 2, 3)
    # true_vals = [np.array([0.]), np.array([0.4]), np.array([0.6])]

    vals = g.dat2gauss([0, 1, 2, 3])
    true_vals = np.array([-0.2, 0., 0.2, 0.4])
    assert((np.abs(vals - true_vals) < 1e-7).all())
    # Check that the departure from a Gaussian model is zero
    assert(g.dist2gauss() < 1e-16)


if __name__ == '__main__':
    import nose
    nose.run()
