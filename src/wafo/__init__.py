from .info import __doc__

from . import misc
from . import data
from . import demos
from . import kdetools
from . import objects
from . import spectrum
from . import transform
from . import definitions
from . import polynomial
from . import stats
from . import interpolate
from . import dctpack
from . import wave_theory
try:
    from . import fig
except ImportError:
    print('fig import only supported on Windows')

__version__ = "0.3.4"

from numpy.testing import Tester
test = Tester().test
