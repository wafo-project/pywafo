# Set this to eg. pylab to be able to plot
import numpy
try:
    from matplotlib import pyplot as plotbackend
    #from matplotlib import pyplot
    numpy.disp('Scipy.stats: plotbackend is set to matplotlib.pyplot')
except:
    numpy.disp('Scipy.stats: Unable to load matplotlib.pyplot as plotbackend')
    plotbackend = None