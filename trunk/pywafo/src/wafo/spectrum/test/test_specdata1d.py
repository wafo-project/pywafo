import wafo.spectrum.models
from wafo.spectrum import SpecData1D
def test_tocovmatrix():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap()
    >>> S = Sj.tospecdata()
    >>> acfmat = S.tocov_matrix(nr=3, nt=256, dt=0.1)
    >>> acfmat[:2,:]
    array([[ 3.06075987,  0.        , -1.67750289,  0.        ],
           [ 3.05246132, -0.16662376, -1.66819445,  0.18634189]])
    '''
def test_tocovdata():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap()
    >>> S = Sj.tospecdata()
    >>> Nt = len(S.data)-1
    >>> acf = S.tocovdata(nr=0, nt=Nt)
    >>> acf.data[:5]
    array([ 3.06093287,  2.23846752,  0.48630084, -1.1336035 , -2.03036854])
    '''

def test_to_t_pdf():
    '''
    The density of Tc is computed by:
    >>> from wafo.spectrum import models as sm
    >>> Sj = sm.Jonswap()
    >>> S = Sj.tospecdata()
    >>> f = S.to_t_pdf(pdef='Tc', paramt=(0, 10, 51), speed=7, seed=100) 
    >>> ['%2.3f' % val for val in f.data[:10]]
    ['0.000', '0.014', '0.027', '0.040', '0.050', '0.059', '0.067', '0.072', '0.077', '0.081']
                
    estimated error bounds
    >>> ['%2.4f' % val for val in f.err[:10]]  
    ['0.0000', '0.0003', '0.0003', '0.0004', '0.0006', '0.0009', '0.0016', '0.0019', '0.0020', '0.0021']
    '''
def test_sim():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap();S = Sj.tospecdata()
    >>> ns =100; dt = .2
    >>> x1 = S.sim(ns,dt=dt)

    >>> import numpy as np
    >>> import scipy.stats as st
    >>> x2 = S.sim(20000,20)
    >>> truth1 = [0,np.sqrt(S.moment(1)[0]),0., 0.]
    >>> funs = [np.mean,np.std,st.skew,st.kurtosis]
    >>> for fun,trueval in zip(funs,truth1):
    ...     res = fun(x2[:,1::],axis=0)
    ...     m = res.mean()
    ...     sa = res.std()
    ...     #trueval, m, sa
    ...     np.abs(m-trueval)<sa
    True
    array([ True], dtype=bool)
    True
    True
    '''
def test_sim_nl():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap();S = Sj.tospecdata()
    >>> ns =100; dt = .2
    >>> x1 = S.sim_nl(ns,dt=dt)

    >>> import numpy as np
    >>> import scipy.stats as st
    >>> x2, x1 = S.sim_nl(ns=20000,cases=40)
    >>> truth1 = [0,np.sqrt(S.moment(1)[0][0])] + S.stats_nl(moments='sk')
    >>> truth1[-1] = truth1[-1]-3
    >>> truth1
    [0, 1.7495200310090633, 0.18673120577479801, 0.061988521262417606]
     
    >>> funs = [np.mean,np.std,st.skew,st.kurtosis]
    >>> for fun,trueval in zip(funs,truth1):
    ...     res = fun(x2[:,1::], axis=0)
    ...     m = res.mean()
    ...     sa = res.std()
    ...     #trueval, m, sa
    ...     np.abs(m-trueval)<2*sa
    True
    True
    True
    True
    '''
def test_stats_nl():
    '''
    >>> import wafo.spectrum.models as sm    
    >>> Hs = 7.
    >>> Sj = sm.Jonswap(Hm0=Hs, Tp=11)
    >>> S = Sj.tospecdata()
    >>> me, va, sk, ku = S.stats_nl(moments='mvsk')
    >>> me; va; sk; ku
    0.0
    3.0608203389019537
    0.18673120577479801
    3.0619885212624176
    '''
def test_testgaussian():
    '''
    >>> import wafo.spectrum.models as sm
    >>> import wafo.transform.models as wtm
    >>> import wafo.objects as wo
    >>> Hs = 7
    >>> Sj = sm.Jonswap(Hm0=Hs)
    >>> S0 = Sj.tospecdata()
    >>> ns =100; dt = .2
    >>> x1 = S0.sim(ns, dt=dt)
    
    >>> S = S0.copy()
    >>> me, va, sk, ku = S.stats_nl(moments='mvsk')
    >>> S.tr = wtm.TrHermite(mean=me, sigma=Hs/4, skew=sk, kurt=ku, ysigma=Hs/4)
    >>> ys = wo.mat2timeseries(S.sim(ns=2**13))         
    >>> g0, gemp = ys.trdata()
    >>> t0 = g0.dist2gauss()
    >>> t1 = S0.testgaussian(ns=2**13, t0=t0, cases=50) 
    >>> sum(t1>t0)<5
    True
    '''
def test_moment():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=5)
    >>> S = Sj.tospecdata() #Make spectrum ob
    >>> S.moment()
    ([1.5614600345079888, 0.95567089481941048], ['m0', 'm0tt'])
    '''
    
def test_nyquist_freq():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=5)
    >>> S = Sj.tospecdata() #Make spectrum ob
    >>> S.nyquist_freq()
    3.0
    '''
def test_sampling_period():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=5)
    >>> S = Sj.tospecdata() #Make spectrum ob
    >>> S.sampling_period()
    1.0471975511965976
    '''
def test_normalize():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=5)
    >>> S = Sj.tospecdata() #Make spectrum ob
    >>> S.moment(2)
    ([1.5614600345079888, 0.95567089481941048], ['m0', 'm0tt'])
    
    >>> Sn = S.copy(); Sn.normalize()
    
    Now the moments should be one
    >>> Sn.moment(2)
    ([1.0000000000000004, 0.99999999999999967], ['m0', 'm0tt'])
    
    '''
def test_characteristic():
    '''
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=5)
    >>> S = Sj.tospecdata() #Make spectrum ob
    >>> S.characteristic(1)
    (array([ 8.59007646]), array([[ 0.03040216]]), ['Tm01'])

    >>> [ch, R, txt] = S.characteristic([1,2,3])  # fact a vector of integers
    >>> ch; R; txt
    array([ 8.59007646,  8.03139757,  5.62484314])
    array([[ 0.03040216,  0.02834263,         NaN],
           [ 0.02834263,  0.0274645 ,         NaN],
           [        NaN,         NaN,  0.01500249]])
    ['Tm01', 'Tm02', 'Tm24']
    
    >>> S.characteristic('Ss')               # fact a string
    (array([ 0.04963112]), array([[  2.63624782e-06]]), ['Ss'])

    >>> S.characteristic(['Hm0','Tm02'])   # fact a list of strings
    (array([ 4.99833578,  8.03139757]), array([[ 0.05292989,  0.02511371],
           [ 0.02511371,  0.0274645 ]]), ['Hm0', 'Tm02'])
    '''   
def test_bandwidth():
    '''
    >>> import numpy as np
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=3)
    >>> w = np.linspace(0,4,256)
    >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values
    >>> S.bandwidth([0,1,2,3])
    array([ 0.65354446,  0.3975428 ,  0.75688813,  2.00207912])
    '''
def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()