import logging
import scipy as sp
import numpy as np
from numpy import pi, reshape
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import rcParams
rcParams.update({"font.size": 10})


try:
    from win32api import LoadResource
except ImportError:
    pass

log = logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG)


def set_windows_title(title, log=None):
    if log is not None:
        log.info("Set windows title {}".format(title))
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.show()
