'''
Created on 14. feb. 2016

@author: pab
'''
import unittest
import wafo.bitwise as wb
import numpy as np
from numpy.testing import assert_array_equal


class Test(unittest.TestCase):

    def test_getbit(self):

        assert_array_equal(wb.getbit(13, np.arange(3, -1, -1)),
                           [1, 1, 0, 1])
        assert_array_equal(wb.getbit(5, np.r_[0:4]), [1, 0, 1, 0])

    def test_setbit(self):
        """
        Set bit fifth bit in the five bit binary binary representation
        of 9 (01001)
        """
        assert_array_equal(wb.setbit(9, 4), 25)

    def test_setbits(self):
        assert_array_equal(wb.setbits([1, 1]), 3)
        assert_array_equal(wb.setbits([1, 0]), 1)

    def test_getbits(self):
        assert_array_equal(wb.getbits(3), [1, 1, 0, 0, 0, 0, 0, 0])
        assert_array_equal(wb.getbits(1), [1, 0, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
