'''

pychip.py
chris.michalski@gmail.com
20090818

Piecewise cubic Hermite interpolation (monotonic...) in Python

References:

    Wikipedia:  Monotone cubic interpolation
                Cubic Hermite spline

A cubic Hermite spline is a third degree spline with each polynomial of the spline
in Hermite form.  The Hermite form consists of two control points and two control
tangents for each polynomial.  Each interpolation is performed on one sub-interval
at a time (piece-wise).  A monotone cubic interpolation is a variant of cubic
interpolation that preserves monotonicity of the data to be interpolated (in other
words, it controls overshoot).  Monotonicity is preserved by linear interpolation
but not by cubic interpolation.

Use:

There are two separate calls, the first call, pchip_slopes(),  computes the slopes that
the interpolator needs.  If there are a large number of points to compute,
it is more efficient to compute the slopes once, rather than for  every point
being evaluated.  The second call, pchip_eval(), takes the slopes computed by
pchip_slopes() along with X, Y, and a vector of desired "xnew"s and computes a vector
of "ynew"s.  If only a handful of points is needed, pchip() is a  third function
which combines a call to pchip_slopes() followed by pchip_eval().

'''
import warnings
import numpy as np
from matplotlib import pyplot as plt
from interpolate import slopes2, slopes, stineman_interp
from scipy.interpolate import PiecewisePolynomial
#=========================================================
def pchip(x, y, xnew):

    # Compute the slopes used by the piecewise cubic Hermite  interpolator
    m = pchip_slopes(x, y)
    
    # Use these slopes (along with the Hermite basis function) to  interpolate
    ynew = pchip_eval(x, y, m, xnew)
    
    return ynew

#=========================================================
def x_is_okay(x,xvec):
    # Make sure "x" and "xvec" satisfy the conditions for
    # running the pchip interpolator
    
    n = len(x)
    m = len(xvec)
    
    # Make sure "x" is in sorted order (brute force, but works...)
    xx = x.copy()
    xx.sort()
    total_matches = (xx == x).sum()
    if total_matches != n:
        warnings.warn( "x values weren't in sorted order --- aborting")
        return False
    
    # Make sure 'x' doesn't have any repeated values
    delta = x[1:] - x[:-1]
    if (delta == 0.0).any():
        warnings.warn( "x values weren't monotonic--- aborting")
        return False
    
    # Check for in-range xvec values (beyond upper edge)
    check = xvec > x[-1]
    if check.any():
        print "*" * 50
        print "x_is_okay()"
        print "Certain 'xvec' values are beyond the upper end of 'x'"
        print "x_max = ", x[-1]
        indices = np.compress(check, range(m))
        print "out-of-range xvec's = ", xvec[indices]
        print "out-of-range xvec indices = ", indices
        return False
    
    # Second - check for in-range xvec values (beyond lower edge)
    check = xvec< x[0]
    if check.any():
        print "*" * 50
        print "x_is_okay()"
        print "Certain 'xvec' values are beyond the lower end of 'x'"
        print "x_min = ", x[0]
        indices = np.compress(check, range(m))
        print "out-of-range xvec's = ", xvec[indices]
        print "out-of-range xvec indices = ", indices
        return False
    
    return True

#=========================================================
def pchip_eval(x, y, m, xvec):
    '''
     Evaluate the piecewise cubic Hermite interpolant with  monoticity preserved
    
        x = array containing the x-data
        y = array containing the y-data
        m = slopes at each (x,y) point [computed to preserve  monotonicity]
        xnew = new "x" value where the interpolation is desired
    
        x must be sorted low to high... (no repeats)
        y can have repeated values
    
     This works with either a scalar or vector of "xvec"
    '''
    
    ############################
    # Make sure there aren't problems with the input data
    ############################
    if not x_is_okay(x, xvec):
        print "pchip_eval2() - ill formed 'x' vector!!!!!!!!!!!!!"
    
        # Cause a hard crash...
        return #STOP_pchip_eval2
    
    # Find the indices "k" such that x[k] < xvec < x[k+1]
    k = np.searchsorted(x[1:-1], xvec)
    
    # Create the Hermite coefficients
    h = x[k+1] - x[k]
    t = (xvec - x[k]) / h[k]
    
    # Hermite basis functions
    h00 = (2 * t**3) - (3 * t**2) + 1
    h10 =      t**3  - (2 * t**2) + t
    h01 = (-2* t**3) + (3 * t**2)
    h11 =      t**3  -      t**2
    
    # Compute the interpolated value of "y"
    ynew = h00*y[k] + h10*h*m[k] + h01*y[k+1] + h11*h*m[k+1]
    
    return ynew

#=========================================================
def pchip_slopes(x,y, method='secant', tension=0, monotone=True):
    '''
    Return estimated slopes y'(x) 
    
    Parameters
    ----------
    x, y : array-like
        array containing the x- and y-data, respectively.
        x must be sorted low to high... (no repeats) while
        y can have repeated values.
    method : string
        defining method of estimation for yp. Valid options are:
        'secant' average secants 
            yp = 0.5*((y[k+1]-y[k])/(x[k+1]-x[k]) + (y[k]-y[k-1])/(x[k]-x[k-1]))
        'Catmull-Rom'  yp = (y[k+1]-y[k-1])/(x[k+1]-x[k-1])
        'Cardinal'     yp = (1-tension) * (y[k+1]-y[k-1])/(x[k+1]-x[k-1])
    tension : real scalar between 0 and 1.
        tension parameter used in Cardinal method
    monotone : bool
        If True modifies yp to preserve monoticity
    
    x input conditioning is assumed but not checked
    '''
    n = len(x)
    
    # Compute the slopes of the secant lines between successive points
    delta = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    
    # Initialize the tangents at every points as the average of the  secants
    m = np.zeros(n, dtype='d')
    
    # At the endpoints - use one-sided differences
    m[0] = delta[0]
    m[n-1] = delta[-1]
    method = method.lower()
    if method.startswith('secant'):
        # In the middle - use the average of the secants
        m[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    else: # Cardinal or Catmull-Rom method
        m[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
        if method.startswith('cardinal'):
            m = (1-tension) * m
       
    if monotone:
        # Special case: intervals where y[k] == y[k+1]
        
        # Setting these slopes to zero guarantees the spline connecting
        # these points will be flat which preserves monotonicity
        ii, = (delta == 0.0).nonzero()
        m[ii] = 0.0
        m[ii+1] = 0.0
        
        alpha = m[:-1]/delta
        beta  = m[1:]/delta
        dist  = alpha**2 + beta**2
        tau   = 3.0 / np.sqrt(dist)
        
        # To prevent overshoot or undershoot, restrict the position vector
        # (alpha, beta) to a circle of radius 3.  If (alpha**2 +  beta**2)>9,
        # then set m[k] = tau[k]alpha[k]delta[k] and m[k+1] =  tau[k]beta[b]delta[k]
        # where tau = 3/sqrt(alpha**2 + beta**2).
        
        # Find the indices that need adjustment
        indices_to_fix, = (dist > 9.0).nonzero() 
        for ii in indices_to_fix:
            m[ii]   = tau[ii] * alpha[ii] * delta[ii]
            m[ii+1] = tau[ii] * beta[ii]  * delta[ii]
    
    return m

def _edge_case(m0, d1):
    return np.where((d1==0) | (m0==0), 0.0, 1.0/(1.0/m0+1.0/d1))

def pchip_slopes2(x, y):
    # Determine the derivatives at the points y_k, d_k, by using
    #  PCHIP algorithm is:
    # We choose the derivatives at the point x_k by
    # Let m_k be the slope of the kth segment (between k and k+1)
    # If m_k=0 or m_{k-1}=0 or sgn(m_k) != sgn(m_{k-1}) then d_k == 0
    # else use weighted harmonic mean:
    #   w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
    #   1/d_k = 1/(w_1 + w_2)*(w_1 / m_k + w_2 / m_{k-1})
    #   where h_k is the spacing between x_k and x_{k+1}

    hk = x[1:] - x[:-1]
    mk = (y[1:] - y[:-1]) / hk
    smk = np.sign(mk)
    condition = ((smk[1:] != smk[:-1]) | (mk[1:]==0) | (mk[:-1]==0))

    w1 = 2*hk[1:] + hk[:-1]
    w2 = hk[1:] + 2*hk[:-1]
    whmean = 1.0/(w1+w2)*(w1/mk[1:] + w2/mk[:-1])

    dk = np.zeros_like(y)
    dk[1:-1][condition] = 0.0
    dk[1:-1][~condition] = 1.0/whmean[~condition]

    # For end-points choose d_0 so that 1/d_0 = 1/m_0 + 1/d_1 unless
    #  one of d_1 or m_0 is 0, then choose d_0 = 0

    dk[0] = _edge_case(mk[0],dk[1])
    dk[-1] = _edge_case(mk[-1],dk[-2])
    return dk

class StinemanInterp(PiecewisePolynomial):
    def __init__(self, x, y, yp=None, method='parabola'):
        if yp is None:
            yp = slopes2(x, y, method)
        super(StinemanInterp,self).__init__(x, zip(y,yp))


def CubicHermiteSpline2(x, y, xnew):
    '''
    Piecewise Cubic Hermite Interpolation using Catmull-Rom
    method for computing the slopes.
    '''
    # Non-montonic cubic Hermite spline interpolator using
    # Catmul-Rom method for computing slopes...
    m = pchip_slopes(x, y, method='catmull', monotone=False)
    
    # Use these slopes (along with the Hermite basis function) to  interpolate
    ynew = pchip_eval(x, y, m, xnew)
    
    return ynew

  

#==============================================================
def main():
    ############################################################
    # Sine wave test
    ############################################################
    
    # Create a example vector containing a sine wave.
    x = np.arange(30.0)/10.
    y = np.sin(x)
    
    # Interpolate the data above to the grid defined by "xvec"
    xvec = np.arange(250.)/100.
    
    # Initialize the interpolator slopes
    m = pchip_slopes(x,y)
    m1 = slopes(x, y)
    m2  = pchip_slopes(x,y,method='catmul',monotone=False)
    m3 = pchip_slopes2(x, y)
    # Call the monotonic piece-wise Hermite cubic interpolator
    yvec = pchip_eval(x, y, m, xvec)
    yvec1 = pchip_eval(x, y, m1, xvec)
    yvec2 = pchip_eval(x, y, m2, xvec)
    yvec3 = pchip_eval(x, y, m3, xvec)
    
    plt.figure(1)
    plt.plot(x,y, 'ro')
    plt.title("pchip() Sin test code")
    
    # Plot the interpolated points
    plt.plot(xvec, yvec, xvec, yvec1, xvec, yvec2,xvec, yvec3, )
    plt.legend(['true','m0','m1','m2','m3'])
     
    
    # Step function test...
    plt.figure(2)
    plt.title("pchip() step function test")
    
    # Create a step function (will demonstrate monotonicity)
    x = np.arange(7.0) - 3.0
    y = np.array([-1.0, -1,-1,0,1,1,1])
    
    # Interpolate using monotonic piecewise Hermite cubic spline
    xvec = np.arange(599.)/100. - 3.0
    
    # Create the pchip slopes
    m  = pchip_slopes(x,y)
    m1 = slopes(x, y)
    m2  = pchip_slopes(x,y,method='catmul',monotone=False)
    m3 = pchip_slopes2(x, y)
    # Interpolate...
    yvec = pchip_eval(x, y, m, xvec)
    
    # Call the Scipy cubic spline interpolator
    from scipy.interpolate import interpolate
    function = interpolate.interp1d(x, y, kind='cubic')
    yvec2 = function(xvec)
    
    # Non-montonic cubic Hermite spline interpolator using
    # Catmul-Rom method for computing slopes...
    yvec3 = CubicHermiteSpline(x,y)(xvec)
    yvec4 = StinemanInterp(x, y)(xvec)
    #yvec4 = stineman_interp(xvec, x, y, m)
    yvec5 = pchip_eval(x, y, m3, xvec)
    
    # Plot the results
    plt.plot(x,    y,     'ro')
    plt.plot(xvec, yvec,  'b')
    plt.plot(xvec, yvec2, 'k')
    plt.plot(xvec, yvec3, 'g')
    plt.plot(xvec, yvec4, 'm')
    #plt.plot(xvec, yvec5, 'y')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Comparing pypchip() vs. Scipy interp1d() vs. non-monotonic CHS")
    legends = ["Data", "pypchip()", "interp1d","CHS", 'SI']
    plt.legend(legends, loc="upper left")
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    ###################################################################
    main()