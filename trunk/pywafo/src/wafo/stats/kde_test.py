# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 16:02:47 2011

@author: pab
"""
import numpy as np
import wafo.kdetools as wk
n = 100
x = np.sort(5*np.random.rand(1,n)-2.5, axis=-1).ravel()
y = (np.cos(x)>2*np.random.rand(n, 1)-1).ravel()

kreg = wk.KRegression(x,y)
f = kreg(output='plotobj', title='Kernel regression', plotflag=1)
f.plot()