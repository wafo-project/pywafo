from numpy.testing import (run_module_suite, assert_equal, assert_almost_equal,
                           assert_array_equal, assert_array_almost_equal)
import numpy as np
from numpy import array, cos, exp, linspace, pi, sin, diff, arange, ones

import wafo.spectrum.models as sm
from wafo.covariance import CovData1D

def test_covariance():
    Sj = sm.Jonswap()
    S = Sj.tospecdata()   #Make spec
    R = S.tocovdata()
    
    x = R.sim(ns=1000,dt=0.2)

if __name__ == '__main__':
    run_module_suite()
