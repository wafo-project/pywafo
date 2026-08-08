import os
_dll_dir = os.path.join(os.path.dirname(__file__), '.libs')
if os.path.isdir(_dll_dir):
    try:
        os.add_dll_directory(_dll_dir)
    except AttributeError:
        pass
    os.environ['PATH'] += os.pathsep + _dll_dir

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

__version__ = "0.4.0"


def test(*options):
    """
    Run tests for module using pytest.

    Parameters
    ----------
    *options : optional
        options to pass to pytest. The most important ones include:
        '-v', '--verbose':
            increase verbosity.
        '-q', '--quiet':
            decrease verbosity.
        '--doctest-modules':
            run doctests in all .py modules
        '--cov':
            measure coverage for .py modules
        '-h', '--help':
            show full help message and display all possible options to use.

    Returns
    -------
    exit_code: scalar
        Exit code is 0 if all tests passed without failure.

    Examples
    --------
    import wafo
    wafo.test('-q', '--doctest-modules', '--cov', '--disable-warnings')
    """

    import pytest
    return pytest.main(['--pyargs', 'wafo'] + list(options))
