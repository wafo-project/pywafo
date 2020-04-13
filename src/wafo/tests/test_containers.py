'''
Created on 29. jun. 2016

@author: pab
'''
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from wafo.containers import transformdata_1d, PlotData


class TestPlotData(unittest.TestCase):

    def setUp(self):
        x = np.linspace(0, np.pi, 5)
        self.d2 = PlotData(np.sin(x), x,
                           xlab='x', ylab='sin', title='sinus',
                           plot_args=['r.'])
        self.x = x

    def tearDown(self):
        pass

    def test_copy(self):
        d3 = self.d2.copy()  # shallow copy
        self.d2.args = None
        assert_array_almost_equal(d3.args, self.x)

    def test_labels_str(self):
        txt = str(self.d2.labels)
        self.assertEqual(txt,
                         'AxisLabels(title=sinus, xlab=x, ylab=sin, zlab=)')


class TestTransform(unittest.TestCase):
    def test_transformdata_1d(self):
        expectations = \
            ([0.25, 0.4330127, 0.5, 0.4330127, 0.25],
             [0.75, 0.5669873, 0.5, 0.5669873, 0.75],
             [0.17881231, 0.42307446, 0.66733662, 0.84614892],
             [0.82118769, 0.57692554, 0.33266338, 0.15385108],
             [-1.38629436, -0.83698822, -0.69314718, -0.83698822, -1.38629436],
             [-0.28768207, -0.56741838, -0.69314718, -0.56741838, -0.28768207],
             [-1.72141859, -0.86020708, -0.40446069, -0.1670599],
             [-0.19700358, -0.55004207, -1.10062416, -1.87177018],
             [-6.0205999, -3.6349936, -3.0103, -3.6349936, -6.0205999])

        x = np.linspace(0, np.pi, 7)[1:-1]
        f = np.sin(x)/2
        for i, truth in enumerate(expectations):
            plotflag = i * 10
            tf = transformdata_1d(x, f, plotflag)
            print(tf)
            assert_array_almost_equal(tf, truth)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
