
from info import __doc__
import misc
import data
import demos
import kdetools
import objects
import spectrum
import transform
import definitions
import polynomial
import stats
import interpolate
import fig
import dctpack

try:
    from wafo.version import version as __version__
except ImportError:
    __version__='nobuilt'
    
from numpy.testing import Tester
test = Tester().test 