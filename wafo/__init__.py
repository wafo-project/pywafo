from __future__ import division, print_function, absolute_import

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
try:
    from . import fig
except ImportError:
    print('fig import only supported on Windows')

try:
    from wafo.version import version as __version__
except ImportError:
    __version__ = 'nobuilt'

from numpy.testing import Tester
test = Tester().test
