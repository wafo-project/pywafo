"""
    Modify this file if another plotbackend is wanted.
"""

if False:
    try:
        from scitools import easyviz as plotbackend
        print('wafo.wafodata: plotbackend is set to scitools.easyviz')
    except:
        print('wafo: Unable to load scitools.easyviz as plotbackend')
        plotbackend = None
else:
    try:
        from matplotlib import pyplot as plotbackend
        print('wafo.wafodata: plotbackend is set to matplotlib.pyplot')
    except:
        print('wafo: Unable to load matplotlib.pyplot as plotbackend')
        plotbackend = None