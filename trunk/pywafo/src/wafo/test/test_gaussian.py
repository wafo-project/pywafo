'''
Created on 17. juli 2010

@author: pab
'''
import numpy as np
from numpy import pi, inf
from wafo.gaussian import Rind, prbnormtndpc, prbnormndpc, prbnormnd, cdfnorm2d, prbnorm2d

def test_rind():
    '''
    >>> Et = 0.001946 # #  exact prob.
    >>> n = 5
    >>> Blo =-np.inf; Bup=-1.2; indI=[-1, n-1]  # Barriers
    >>> m = np.zeros(n); rho = 0.3;
    >>> Sc =(np.ones((n,n))-np.eye(n))*rho+np.eye(n)
    >>> rind = Rind()
    >>> E0, err0, terr0 = rind(Sc,m,Blo,Bup,indI); 
    
    >>> np.abs(E0-Et)< err0+terr0
    array([ True], dtype=bool)
    >>> 'E0 = %2.6f' % E0  
    'E0 = 0.001946'
    
    >>> A = np.repeat(Blo,n); B = np.repeat(Bup,n)  # Integration limits
    >>> E1, err1, terr1  = rind(np.triu(Sc),m,A,B); #same as E0
    >>> np.abs(E1-Et)< err0+terr0
    array([ True], dtype=bool)
    >>> 'E1 = %2.6f' % E1  
    'E1 = 0.001946'
    
    Compute expectation E( abs(X1*X2*...*X5) )
    >>> xc = np.zeros((0,1))
    >>> infinity = 37
    >>> dev = np.sqrt(np.diag(Sc))  # std
    >>> ind = np.nonzero(indI[1:])[0]
    >>> Bup, Blo = np.atleast_2d(Bup,Blo)
    >>> Bup[0,ind] = np.minimum(Bup[0,ind] , infinity*dev[indI[ind+1]])
    >>> Blo[0,ind] = np.maximum(Blo[0,ind] ,-infinity*dev[indI[ind+1]])
    >>> rind(Sc,m,Blo,Bup,indI, xc, nt=0)
    (array([ 0.05494076]), array([ 0.00083066]), array([  1.00000000e-10]))
    
    Compute expectation E( X1^{+}*X2^{+} ) with random
    correlation coefficient,Cov(X1,X2) = rho2.
    >>> m2  = [0, 0]; 
    >>> rho2 = 0.3 #np.random.rand(1)
    >>> Sc2 = [[1, rho2], [rho2 ,1]]
    >>> Blo2 = 0; Bup2 = np.inf; indI2 = [-1, 1]
    >>> rind2 = Rind(method=1)
    >>> g2 = lambda x : (x*(np.pi/2+np.arcsin(x))+np.sqrt(1-x**2))/(2*np.pi)
    >>> E2 = g2(rho2); E2   # exact value
    0.24137214191774381
    
    >>> E3, err3, terr3 = rind(Sc2,m2,Blo2,Bup2,indI2,nt=0); E3;err3;terr3
    array([ 0.24127499])
    array([ 0.00013838])
    array([  1.00000000e-10])
    
    >>> E4, err4, terr4 = rind2(Sc2,m2,Blo2,Bup2,indI2,nt=0); E4;err4;terr4
    array([ 0.24127499])
    array([ 0.00013838])
    array([  1.00000000e-10])
    
    >>> E5, err5, terr5 = rind2(Sc2,m2,Blo2,Bup2,indI2,nt=0,abseps=1e-4); E5;err5;terr5
    array([ 0.24127499])
    array([ 0.00013838])
    array([  1.00000000e-10])
    '''
def test_prbnormtndpc():
    '''
    >>> rho2 = np.random.rand(2); 
    >>> a2   = np.zeros(2);
    >>> b2   = np.repeat(np.inf,2);
    >>> [val2,err2, ift2] = prbnormtndpc(rho2,a2,b2)
    >>> g2 = lambda x : 0.25+np.arcsin(x[0]*x[1])/(2*pi)
    >>> E2 = g2(rho2)  #% exact value
    >>> np.abs(E2-val2)<err2
    True
    
    >>> rho3 = np.random.rand(3) 
    >>> a3   = np.zeros(3)
    >>> b3   = np.repeat(inf,3)
    >>> [val3,err3, ift3] = prbnormtndpc(rho3,a3,b3)  
    >>> g3 = lambda x : 0.5-sum(np.sort(np.arccos([x[0]*x[1],x[0]*x[2],x[1]*x[2]])))/(4*pi)
    >>> E3 = g3(rho3)   #  Exact value  
    >>> np.abs(E3-val3)<err3
    True
    '''
def test_prbnormndpc():
    '''
    >>> rho2 = np.random.rand(2); 
    >>> a2   = np.zeros(2);
    >>> b2   = np.repeat(np.inf,2);
    >>> [val2,err2, ift2] = prbnormndpc(rho2,a2,b2)
    >>> g2 = lambda x : 0.25+np.arcsin(x[0]*x[1])/(2*pi)
    >>> E2 = g2(rho2)  #% exact value
    >>> np.abs(E2-val2)<err2
    True
    
    >>> rho3 = np.random.rand(3) 
    >>> a3   = np.zeros(3)
    >>> b3   = np.repeat(inf,3)
    >>> [val3,err3, ift3] = prbnormndpc(rho3,a3,b3)  
    >>> g3 = lambda x : 0.5-sum(np.sort(np.arccos([x[0]*x[1],x[0]*x[2],x[1]*x[2]])))/(4*pi)
    >>> E3 = g3(rho3)   #  Exact value  
    >>> np.abs(E3-val3)<err2
    True
    '''
    
def test_prbnormnd():
    '''
    >>> Et = 0.001946 # #  exact prob.
    >>> n = 5; nt = n
    >>> Blo =-np.inf; Bup=0; indI=[-1, n-1]  # Barriers
    >>> m = 1.2*np.ones(n); rho = 0.3;
    >>> Sc =(np.ones((n,n))-np.eye(n))*rho+np.eye(n)
    >>> rind = Rind()
    >>> E0, err0, terr0 = rind(Sc,m,Blo,Bup,indI, nt=nt) 
    
    >>> A = np.repeat(Blo,n)
    >>> B = np.repeat(Bup,n)-m 
    >>> [val,err,inform] = prbnormnd(Sc,A,B);val;err;inform
    0.0019456719705212067
    1.0059406844578488e-05
    0
    >>> np.abs(val-Et)< err0+terr0
    array([ True], dtype=bool)
    >>> 'val = %2.6f' % val  
    'val = 0.001945'
    '''
def test_cdfnorm2d():
    '''
    >>> x = np.linspace(-3,3,3) 
    >>> [b1,b2] = np.meshgrid(x,x)
    >>> r  = 0.3
    >>> cdfnorm2d(b1,b2,r)
    array([[  2.38515157e-05,   1.14504149e-03,   1.34987703e-03],
           [  1.14504149e-03,   2.98493342e-01,   4.99795143e-01],
           [  1.34987703e-03,   4.99795143e-01,   9.97324055e-01]])
    '''
    
def test_prbnorm2d():
    '''
    >>> a = [-1, -2]
    >>> b = [1, 1] 
    >>> r = 0.3
    >>> prbnorm2d(a,b,r)
    array([ 0.56659121])
    '''
if __name__ == '__main__':
    import doctest
    doctest.testmod()