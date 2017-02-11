'''
Created on 14. feb. 2016

@author: pab
'''
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
import wafo.dctpack as wd


class Test(unittest.TestCase):
    def test_shiftdim(self):
        a = np.arange(6).reshape((1, 1, 3, 1, 2))
        b = wd.shiftdim(a)
        c = wd.shiftdim(b, -2)
        assert_array_almost_equal(b.shape, (3, 1, 2))
        assert_array_almost_equal(c.shape, a.shape)
        assert_array_almost_equal(c, a)

    def _test_3d(self, dct, idct, dctn, idctn):
        a = np.array([[[0.51699637,  0.42946223,  0.89843545],
                       [0.27853391,  0.8931508,  0.34319118],
                       [0.51984431,  0.09217771,  0.78764716]],
                      [[0.25019845,  0.92622331,  0.06111409],
                       [0.81363641,  0.06093368,  0.13123373],
                       [0.47268657,  0.39635091,  0.77978269]],
                      [[0.86098829,  0.07901332,  0.82169182],
                       [0.12560088,  0.78210188,  0.69805434],
                       [0.33544628,  0.81540172,  0.9393219]]])
        d = dct(dct(dct(a).transpose(0, 2, 1)).transpose(2, 1, 0)
                ).transpose(2, 1, 0).transpose(0, 2, 1)
        d0 = dctn(a)
        e0 = idctn(d0)
        e = idct(idct(idct(d).transpose(0, 2, 1)).transpose(2, 1, 0)
                 ).transpose(2, 1, 0).transpose(0, 2, 1)
        assert_array_almost_equal(d, d0)
        assert_array_almost_equal(e, e0)
        assert_array_almost_equal(a, e)

    def test_3d_dct_idct_dctn_idctn(self):
        self._test_3d(wd.dct, wd.idct, wd.dctn, wd.idctn)

    def test_3d_dst_idst_dstn_idstn(self):
        self._test_3d(wd.dst, wd.idst, wd.dstn, wd.idstn)

    def test_dct_and_dctn(self):
        self._test_1d(wd.dct, wd.idct, wd.dctn, wd.idctn)

    def test_dst_and_dstn(self):
        self._test_1d(wd.dst, wd.idst, wd.dstn, wd.idstn)

    def _test_1d(self, dct, idct, dctn, idctn):
        a = np.arange(12)  # .reshape((3, -1))
        y = dct(a)
        yn = dctn(a)
        x = idct(y)
        xn = idctn(yn)

        assert_array_almost_equal(y, yn)
        assert_array_almost_equal(x, a)
        assert_array_almost_equal(xn, a)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
