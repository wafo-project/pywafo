'''
Misc
'''
from __future__ import absolute_import, division
import collections
import sys
import fractions
import numpy as np
from numpy import (
    meshgrid,
    abs, amax, any, logical_and, arange, linspace, atleast_1d,
    asarray, ceil, floor, frexp, hypot,
    sqrt, arctan2, sin, cos, exp, log, log1p, mod, diff,
    finfo, inf, pi, interp, isscalar, zeros, ones, linalg,
    r_, sign, unique, hstack, vstack, nonzero, where, extract)
from scipy.special import gammaln
from scipy.integrate import trapz, simps
import warnings
from time import strftime, gmtime
from numdifftools.extrapolation import dea3
from .plotbackend import plotbackend
from collections import OrderedDict
try:
    from . import c_library as clib  # @UnresolvedImport
except ImportError:
    warnings.warn('c_library not found. Check its compilation.')
    clib = None
floatinfo = finfo(float)
_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps

__all__ = ['now', 'spaceline', 'narg_smallest', 'args_flat', 'is_numlike',
           'JITImport', 'DotDict', 'Bunch', 'printf', 'sub_dict_select',
           'parse_kwargs', 'detrendma', 'ecross', 'findcross', 'findextrema',
           'findpeaks', 'findrfc', 'rfcfilter', 'findtp', 'findtc',
           'findoutliers', 'common_shape', 'argsreduce', 'stirlerr',
           'getshipchar', 'dea3',
           'betaloge', 'gravity', 'nextpow2', 'discretize', 'polar2cart',
           'cart2polar', 'meshgrid', 'ndgrid', 'trangood', 'tranproc',
           'plot_histgrm', 'num2pistr', 'test_docstrings', 'lazywhere',
           'piecewise',
           'valarray']


def valarray(shape, value=np.NaN, typecode=None):
    """Return an array of all value.
    """
    if typecode is None:
        typecode = bool
    out = ones(shape, dtype=typecode) * value

    if not isinstance(out, np.ndarray):
        out = asarray(out)
    return out


def piecewise(xi, condlist, funclist, fill_value=0.0, args=(), **kw):
    """
    Evaluate a piecewise-defined function.

    Given a set of conditions and corresponding functions, evaluate each
    function on the input data wherever its condition is true.

    Parameters
    ----------
    xi : tuple
        input arguments to the functions in funclist, i.e., (x0, x1,...., xn)
    condlist : list of bool arrays
        Each boolean array corresponds to a function in `funclist`.  Wherever
        `condlist[i]` is True, `funclist[i](x0,x1,...,xn)` is used as the
        output value. Each boolean array in `condlist` selects a piece of `xi`,
        and should therefore be of the same shape as `xi`.

        The length of `condlist` must correspond to that of `funclist`.
        If one extra function is given, i.e. if
        ``len(funclist) - len(condlist) == 1``, then that extra function
        is the default value, used wherever all conditions are false.
    funclist : list of callables, f(*(xi + args), **kw), or scalars
        Each function is evaluated over `x` wherever its corresponding
        condition is True.  It should take an array as input and give an array
        or a scalar value as output.  If, instead of a callable,
        a scalar is provided then a constant function (``lambda x: scalar``) is
        assumed.
    fill_value : scalar
        fill value for out of range values. Default 0.
    args : tuple, optional
        Any further arguments given here passed to the functions
        upon execution, i.e., if called ``piecewise(..., ..., args=(1, 'a'))``,
        then each function is called as ``f(x0, x1,..., xn, 1, 'a')``.
    kw : dict, optional
        Keyword arguments used in calling `piecewise` are passed to the
        functions upon execution, i.e., if called
        ``piecewise(..., ..., lambda=1)``, then each function is called as
        ``f(x0, x1,..., xn, lambda=1)``.

    Returns
    -------
    out : ndarray
        The output is the same shape and type as x and is found by
        calling the functions in `funclist` on the appropriate portions of `x`,
        as defined by the boolean arrays in `condlist`.  Portions not covered
        by any condition have undefined values.


    See Also
    --------
    choose, select, where

    Notes
    -----
    This is similar to choose or select, except that functions are
    evaluated on elements of `xi` that satisfy the corresponding condition from
    `condlist`.

    The result is::

          |--
          |funclist[0](x0[condlist[0]],x1[condlist[0]],...,xn[condlist[0]])
    out = |funclist[1](x0[condlist[1]],x1[condlist[1]],...,xn[condlist[1]])
          |...
          |funclist[n2](x0[condlist[n2]],x1[condlist[n2]],...,xn[condlist[n2]])
          |--

    Examples
    --------
    Define the sigma function, which is -1 for ``x < 0`` and +1 for ``x >= 0``.

    >>> x = np.linspace(-2.5, 2.5, 6)
    >>> piecewise(x, [x < 0, x >= 0], [-1, 1])
    array([-1., -1., -1.,  1.,  1.,  1.])

    Define the absolute value, which is ``-x`` for ``x <0`` and ``x`` for
    ``x >= 0``.

    >>> piecewise((x,), [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    array([ 2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    Define the absolute value, which is ``-x*y`` for ``x*y <0`` and ``x*y`` for
    ``x*y >= 0``
    >>> X, Y = np.meshgrid(x, x)
    >>> piecewise((X, Y), [X * Y < 0, ],
    ...           [lambda x, y: -x * y, lambda x, y: x * y])
    array([[ 6.25,  3.75,  1.25,  1.25,  3.75,  6.25],
           [ 3.75,  2.25,  0.75,  0.75,  2.25,  3.75],
           [ 1.25,  0.75,  0.25,  0.25,  0.75,  1.25],
           [ 1.25,  0.75,  0.25,  0.25,  0.75,  1.25],
           [ 3.75,  2.25,  0.75,  0.75,  2.25,  3.75],
           [ 6.25,  3.75,  1.25,  1.25,  3.75,  6.25]])
    """
    def otherwise_condition(condlist):
        return ~np.logical_or.reduce(condlist, axis=0)

    def check_shapes(condlist, funclist):
        nc, nf = len(condlist), len(funclist)
        if nc not in [nf-1, nf]:
            raise ValueError("function list and condition list" +
                             " must be the same length")

    check_shapes(condlist, funclist)
    if not isinstance(xi, tuple):
        xi = (xi,)

    condlist = np.broadcast_arrays(*condlist)
    if len(condlist) == len(funclist)-1:
        condlist.append(otherwise_condition(condlist))

    arrays = np.broadcast_arrays(*xi)
    dtype = np.result_type(*arrays)

    out = valarray(arrays[0].shape, fill_value, dtype)
    for cond, func in zip(condlist, funclist):
        if isinstance(func, collections.Callable):
            temp = tuple(np.extract(cond, arr) for arr in arrays) + args
            np.place(out, cond, func(*temp, **kw))
        else:  # func is a scalar value
            np.place(out, cond, func)
    return out


def lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """
    np.where(cond, x, fillvalue) always evaluates x even where cond is False.
    This one only evaluates f(arr1[cond], arr2[cond], ...).
    For example,
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
    ...     return a*b
    >>> lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])

    Notice it assumes that all `arrays` are of the same shape, or can be
    broadcasted together.

    """
    if fillvalue is None:
        if f2 is None:
            raise ValueError("One of (fillvalue, f2) must be given.")
        else:
            fillvalue = np.nan
    else:
        if f2 is not None:
            raise ValueError("Only one of (fillvalue, f2) can be given.")

    arrays = np.broadcast_arrays(*arrays)
    temp = tuple(np.extract(cond, arr) for arr in arrays)
    out = valarray(np.shape(arrays[0]), value=fillvalue)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out


def rotation_matrix(heading, pitch, roll):
    '''

    Examples
    >>> import numpy as np
    >>> rotation_matrix(heading=0, pitch=0, roll=0)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    >>> np.all(np.abs(rotation_matrix(heading=180, pitch=0, roll=0)-
    ... np.array([[ -1.000000e+00,  -1.224647e-16,   0.000000e+00],
    ...       [  1.224647e-16,  -1.000000e+00,   0.000000e+00],
    ...       [ -0.000000e+00,   0.000000e+00,   1.000000e+00]]))<1e-7)
    True
    >>> np.all(np.abs(rotation_matrix(heading=0, pitch=180, roll=0)-
    ... np.array([[ -1.000000e+00,   0.000000e+00,   1.224647e-16],
    ...       [ -0.000000e+00,   1.000000e+00,   0.000000e+00],
    ...       [ -1.224647e-16,  -0.000000e+00,  -1.000000e+00]]))<1e-7)
    True
    >>> np.all(np.abs(rotation_matrix(heading=0, pitch=0, roll=180)-
    ... np.array([[  1.000000e+00,   0.000000e+00,   0.000000e+00],
    ...       [  0.000000e+00,  -1.000000e+00,  -1.224647e-16],
    ...       [ -0.000000e+00,   1.224647e-16,  -1.000000e+00]]))<1e-7)
    True
    '''
    data = np.diag(np.ones(3))  # No transform if H=P=R=0
    if heading != 0 or pitch != 0 or roll != 0:
        deg2rad = np.pi / 180
        H = heading * deg2rad
        P = pitch * deg2rad
        R = roll * deg2rad  # Convert to radians

        data.put(0, cos(H) * cos(P))
        data.put(1, cos(H) * sin(P) * sin(R) - sin(H) * cos(R))
        data.put(2, cos(H) * sin(P) * cos(R) + sin(H) * sin(R))
        data.put(3, sin(H) * cos(P))
        data.put(4, sin(H) * sin(P) * sin(R) + cos(H) * cos(R))
        data.put(5, sin(H) * sin(P) * cos(R) - cos(H) * sin(R))
        data.put(6, -sin(P))
        data.put(7, cos(P) * sin(R))
        data.put(8, cos(P) * cos(R))
    return data


def rotate(x, y, z, heading=0, pitch=0, roll=0):
    rot_param = rotation_matrix(heading, pitch, roll).ravel()
    X = x * rot_param[0] + y * rot_param[1] + z * rot_param[2]
    Y = x * rot_param[3] + y * rot_param[4] + z * rot_param[5]
    Z = x * rot_param[6] + y * rot_param[7] + z * rot_param[8]
    return X, Y, Z


def rotate_2d(x, y, angle_deg):
    '''
    Rotate points in the xy cartesian plane counter clockwise

    Examples
    --------
    >>> rotate_2d(x=1, y=0, angle_deg=0)
    (1.0, 0.0)
    >>> rotate_2d(x=1, y=0, angle_deg=90)
    (6.123233995736766e-17, 1.0)
    >>> rotate_2d(x=1, y=0, angle_deg=180)
    (-1.0, 1.2246467991473532e-16)
    >>> rotate_2d(x=1, y=0, angle_deg=360)
    (1.0, -2.4492935982947064e-16)
    '''
    angle_rad = angle_deg * pi / 180
    ch = cos(angle_rad)
    sh = sin(angle_rad)
    return ch * x - sh * y, sh * x + ch * y


def now(show_seconds=True):
    '''
    Return current date and time as a string
    '''
    if show_seconds:
        return strftime("%a, %d %b %Y %H:%M:%S", gmtime())
    else:
        return strftime("%a, %d %b %Y %H:%M", gmtime())


def _assert(cond, txt=''):
    if not cond:
        raise ValueError(txt)


def spaceline(start_point, stop_point, num=10):
    '''Return `num` evenly spaced points between the start and stop points.

    Parameters
    ----------
    start_point : vector, size=3
        The starting point of the sequence.
    stop_point : vector, size=3
        The end point of the sequence.
    num : int, optional
        Number of samples to generate. Default is 10.

    Returns
    -------
    space_points : ndarray of shape n x 3
        There are `num` equally spaced points in the closed interval
        ``[start, stop]``.

    See Also
    --------
    linspace : similar to spaceline, but in 1D.
    arange : Similiar to `linspace`, but uses a step size (instead of the
             number of samples).
    logspace : Samples uniformly distributed in log space.

    Example
    -------
    >>> import wafo.misc as pm
    >>> pm.spaceline((2,0,0), (3,0,0), num=5)
    array([[ 2.  ,  0.  ,  0.  ],
           [ 2.25,  0.  ,  0.  ],
           [ 2.5 ,  0.  ,  0.  ],
           [ 2.75,  0.  ,  0.  ],
           [ 3.  ,  0.  ,  0.  ]])
    '''
    num = int(num)
    e1, e2 = np.atleast_1d(start_point, stop_point)
    e2m1 = e2 - e1
    length = np.sqrt((e2m1 ** 2).sum())
    # length = sqrt((E2[0]-E1(1))^2 + (E2(2)-E1(2))^2 + (E2(3)-E1(3))^2)
    C = e2m1 / length
    delta = length / float(num - 1)
    return np.array([e1 + n * delta * C for n in range(num)])


def narg_smallest(n, arr):
    ''' Return the n smallest indicis to the arr
    '''
    return np.array(arr).argsort()[:n]


def args_flat(*args):
    '''
    Return x,y,z positions as a N x 3 ndarray

    Parameters
    ----------
    pos : array-like, shape N x 3
        [x,y,z] positions
    or

    x,y,z : array-like
        [x,y,z] positions

    Returns
    ------
    pos : ndarray, shape N x 3
        [x,y,z] positions
    common_shape : None or tuple
        common shape of x, y and z variables if given as triple input.

    Examples
    --------
    >>> x = [1,2,3]
    >>> pos, c_shape =args_flat(x,2,3)
    >>> pos
    array([[1, 2, 3],
           [2, 2, 3],
           [3, 2, 3]])
    >>> c_shape
    (3,)
    >>> pos1, c_shape1 = args_flat([1,2,3])
    >>> pos1
    array([[1, 2, 3]])
    >>> c_shape1 is None
    True
    >>> pos1, c_shape1 = args_flat(1,2,3)
    >>> pos1
    array([[1, 2, 3]])
    >>> c_shape1
    ()
    >>> pos1, c_shape1 = args_flat([1],2,3)
    >>> pos1
    array([[1, 2, 3]])
    >>> c_shape1
    (1,)

    '''
    nargin = len(args)

    if (nargin == 1):  # pos
        pos = np.atleast_2d(args[0])
        _assert((pos.shape[1] == 3) and (pos.ndim == 2),
                'POS array must be of shape N x 3!')
        return pos, None
    elif nargin == 3:
        x, y, z = np.broadcast_arrays(*args[:3])
        c_shape = x.shape
        return np.vstack((x.ravel(), y.ravel(), z.ravel())).T, c_shape
    else:
        raise ValueError('Number of arguments must be 1 or 3!')


def index2sub(shape, index, order='C'):
    '''
    Returns Multiple subscripts from linear index.

    Parameters
    ----------
    shape : array-like
        shape of array
    index :
        linear index into array
    order : {'C','F'}, optional
        The order of the linear index.
        'C' means C (row-major) order.
        'F' means Fortran (column-major) order.
        By default, 'C' order is used.

    This function is used to determine the equivalent subscript values
    corresponding to a given single index into an array.

    Example
    -------
    >>> shape = (3,3,4)
    >>> a = np.arange(np.prod(shape)).reshape(shape)
    >>> order = 'C'
    >>> a[1, 2, 3]
    23
    >>> i = sub2index(shape, 1, 2, 3, order=order)
    >>> a.ravel(order)[i]
    23
    >>> index2sub(shape, i, order=order)
    (1, 2, 3)

    See also
    --------
    sub2index
    '''
    return np.unravel_index(index, shape, order=order)


def sub2index(shape, *subscripts, **kwds):
    '''
    Returns linear index from multiple subscripts.

    Parameters
    ----------
    shape : array-like
        shape of array
    *subscripts :
        subscripts into array
    order : {'C','F'}, optional
        The order of the linear index.
        'C' means C (row-major) order.
        'F' means Fortran (column-major) order.
        By default, 'C' order is used.

    This function is used to determine the equivalent single index
    corresponding to a given set of subscript values into an array.

    Example
    -------
    >>> shape = (3,3,4)
    >>> a = np.arange(np.prod(shape)).reshape(shape)
    >>> order = 'C'
    >>> i = sub2index(shape, 1, 2, 3, order=order)
    >>> a[1, 2, 3]
    23
    >>> a.ravel(order)[i]
    23
    >>> index2sub(shape, i, order=order)
    (1, 2, 3)

    See also
    --------
    index2sub
    '''
    return np.ravel_multi_index(subscripts, shape, **kwds)


def is_numlike(obj):
    'return true if *obj* looks like a number'
    try:
        obj + 1
    except TypeError:
        return False
    else:
        return True


class JITImport(object):

    '''
    Just In Time Import of module

    Example
    -------
    >>> np = JITImport('numpy')
    >>> np.exp(0)==1.0
    True
    '''

    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None

    def __getattr__(self, attr):
        try:
            return getattr(self._module, attr)
        except:
            if self._module is None:
                self._module = __import__(self._module_name, None, None, ['*'])
                # assert(isinstance(self._module, types.ModuleType), 'module')
                return getattr(self._module, attr)
            else:
                raise


class DotDict(dict):

    ''' Implement dot access to dict values

      Example
      -------
       >>> d = DotDict(test1=1,test2=3)
       >>> d.test1
       1
    '''
    __getattr__ = dict.__getitem__


class Bunch(object):

    ''' Implement keyword argument initialization of class

    Example
    -------
    >>> d = Bunch(test1=1,test2=3)
    >>> d.test1
    1
    '''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def keys(self):
        return self.__dict__.keys()

    def update(self, ** kwargs):
        self.__dict__.update(kwargs)


def printf(format, *args):  # @ReservedAssignment
    sys.stdout.write(format % args)


def sub_dict_select(somedict, somekeys):
    '''
    Extracting a Subset from Dictionary

    Example
    --------
    # Update options dict from keyword arguments if
    # the keyword exists in options
    >>> opt = dict(arg1=2, arg2=3)
    >>> kwds = dict(arg2=100,arg3=1000)
    >>> sub_dict = sub_dict_select(kwds,opt.keys())
    >>> opt.update(sub_dict)
    >>> opt
    {'arg1': 2, 'arg2': 100}

    See also
    --------
    dict_intersection
    '''
    # slower: validKeys = set(somedict).intersection(somekeys)
    return dict((k, somedict[k]) for k in somekeys if k in somedict)


def parse_kwargs(options, **kwargs):
    '''
    Update options dict from keyword arguments if the keyword exists in options

    Example
    >>> opt = dict(arg1=2, arg2=3)
    >>> opt = parse_kwargs(opt,arg2=100)
    >>> print(opt)
    {'arg1': 2, 'arg2': 100}
    >>> opt2 = dict(arg2=101)
    >>> opt = parse_kwargs(opt,**opt2)

    See also sub_dict_select
    '''

    newopts = sub_dict_select(kwargs, options.keys())
    if len(newopts) > 0:
        options.update(newopts)
    return options


def detrendma(x, L):
    """
    Removes a trend from data using a moving average
           of size 2*L+1.  If 2*L+1 > len(x) then the mean is removed

    Parameters
    ----------
    x : vector or matrix of column vectors
        of data
    L : scalar, integer
        defines the size of the moving average window

    Returns
    -------
    y : ndarray
        detrended data

    Examples
    --------
    >>> import wafo.misc as wm
    >>> import pylab as plt
    >>> exp = plt.exp; cos = plt.cos; randn = plt.randn
    >>> x = plt.linspace(0,1,200)
    >>> y = exp(x)+cos(5*2*pi*x)+1e-1*randn(x.size)
    >>> y0 = wm.detrendma(y,20); tr = y-y0
    >>> h = plt.plot(x, y, x, y0, 'r', x, exp(x), 'k', x, tr, 'm')

    >>> plt.close('all')

    See also
    --------
    Reconstruct
    """

    if L <= 0:
        raise ValueError('L must be positive')
    if L != round(L):
        raise ValueError('L must be an integer')

    x1 = np.atleast_1d(x)
    if x1.shape[0] == 1:
        x1 = x1.ravel()

    n = x1.shape[0]
    if n < 2 * L + 1:  # only able to remove the mean
        return x1 - x1.mean(axis=0)

    mn = x1[0:2 * L + 1].mean(axis=0)
    y = np.empty_like(x1)
    y[0:L] = x1[0:L] - mn

    ix = np.r_[L:(n - L)]
    trend = ((x1[ix + L] - x1[ix - L]) / (2 * L + 1)).cumsum(axis=0) + mn
    y[ix] = x1[ix] - trend
    y[n - L::] = x1[n - L::] - trend[-1]
    return y


def ecross(t, f, ind, v=0):
    '''
    Extracts exact level v crossings

    ECROSS interpolates t and f linearly to find the exact level v
    crossings, i.e., the points where f(t0) = v

    Parameters
    ----------
    t,f : vectors
        of arguments and functions values, respectively.
    ind : ndarray of integers
        indices to level v crossings as found by findcross.
    v : scalar or vector (of size(ind))
        defining the level(s) to cross.

    Returns
    -------
    t0 : vector
        of  exact level v crossings.

    Example
    -------
    >>> from matplotlib import pylab as plt
    >>> import wafo.misc as wm
    >>> ones = np.ones
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)
    >>> ind = wm.findcross(x,0.75)
    >>> ind
    array([  9,  25,  80,  97, 151, 168, 223, 239])
    >>> t0 = wm.ecross(t,x,ind,0.75)
    >>> np.abs(t0 - np.array([0.84910514, 2.2933879 , 7.13205663, 8.57630119,
    ...        13.41484739, 14.85909194, 19.69776067, 21.14204343]))<1e-7
    array([ True,  True,  True,  True,  True,  True,  True,  True], dtype=bool)

    >>> a = plt.plot(t, x, '.', t[ind], x[ind], 'r.', t, ones(t.shape)*0.75,
    ...              t0, ones(t0.shape)*0.75, 'g.')
    >>> plt.close('all')

    See also
    --------
    findcross
    '''
    # Tested on: Python 2.5
    # revised pab Feb2004
    # By pab 18.06.2001
    return (t[ind] + (v - f[ind]) * (t[ind + 1] - t[ind]) /
            (f[ind + 1] - f[ind]))


def _findcross(xn):
    '''Return indices to zero up and downcrossings of a vector
    '''
    if clib is not None:
        ind, m = clib.findcross(xn, 0.0)
        return ind[:m]

    n = len(xn)
    iz, = (xn == 0).nonzero()
    if len(iz) > 0:
        # Trick to avoid turning points on the crossinglevel.
        if iz[0] == 0:
            if len(iz) == n:
                warnings.warn('All values are equal to crossing level!')
                return zeros(0, dtype=np.int)

            diz = diff(iz)
            if len(diz) > 0 and (diz > 1).any():
                ix = iz[(diz > 1).argmax()]
            else:
                ix = iz[-1]

            # x(ix) is a up crossing if  x(1:ix) = v and x(ix+1) > v.
            # x(ix) is a downcrossing if x(1:ix) = v and x(ix+1) < v.
            xn[0:ix + 1] = -xn[ix + 1]
            iz = iz[ix + 1::]

        for ix in iz.tolist():
            xn[ix] = xn[ix - 1]

    # indices to local level crossings ( without turningpoints)
    ind, = (xn[:n - 1] * xn[1:] < 0).nonzero()
    return ind


def xor(a, b):
    """
    Return True only when inputs differ.
    """
    return a ^ b


def findcross(x, v=0.0, kind=None):
    '''
    Return indices to level v up and/or downcrossings of a vector

    Parameters
    ----------
    x : array_like
        vector with sampled values.
    v : scalar, real
        level v.
    kind : string
        defines type of wave or crossing returned. Possible options are
        'dw' : downcrossing wave
        'uw' : upcrossing wave
        'cw' : crest wave
        'tw' : trough wave
        'd'  : downcrossings only
        'u'  : upcrossings only
        None : All crossings will be returned

    Returns
    -------
    ind : array-like
        indices to the crossings in the original sequence x.

    Example
    -------
    >>> from matplotlib import pylab as plt
    >>> import wafo.misc as wm
    >>> ones = np.ones
    >>> findcross([0, 1, -1, 1],0)
    array([0, 1, 2])
    >>> v = 0.75
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)
    >>> ind = wm.findcross(x,v) # all crossings
    >>> ind
    array([  9,  25,  80,  97, 151, 168, 223, 239])

    >>> t0 = plt.plot(t,x,'.',t[ind],x[ind],'r.', t, ones(t.shape)*v)
    >>> ind2 = wm.findcross(x,v,'u')
    >>> ind2
    array([  9,  80, 151, 223])

    >>> t0 = plt.plot(t[ind2],x[ind2],'o')
    >>> plt.close('all')

    See also
    --------
    crossdef
    wavedef
    '''
    xn = np.int8(sign(atleast_1d(x).ravel() - v))  # @UndefinedVariable
    ind = _findcross(xn)
    if ind.size == 0:
        warnings.warn('No level v = %0.5g crossings found in x' % v)
        return ind

    if kind not in ('du', 'all', None):
        if kind == 'd':  # downcrossings only
            t_0 = int(xn[ind[0] + 1] > 0)
            ind = ind[t_0::2]
        elif kind == 'u':  # upcrossings  only
            t_0 = int(xn[ind[0] + 1] < 0)
            ind = ind[t_0::2]
        elif kind in ('dw', 'uw', 'tw', 'cw'):
            # make sure the first is a level v down-crossing
            #   if wdef=='dw' or wdef=='tw'
            # or make sure the first is a level v up-crossing
            #    if wdef=='uw' or wdef=='cw'

            first_is_down_crossing = int(xn[ind[0]] > xn[ind[0] + 1])
            if xor(first_is_down_crossing, kind in ('dw', 'tw')):
                ind = ind[1::]

            # make sure the number of troughs and crests are according to the
            # wavedef, i.e., make sure length(ind) is odd if dw or uw
            # and even if tw or cw
            is_odd = mod(ind.size, 2)
            if xor(is_odd, kind in ('dw', 'uw')):
                ind = ind[:-1]
        else:
            raise ValueError('Unknown wave/crossing definition!')
    return ind


def findextrema(x):
    '''
    Return indices to minima and maxima of a vector

    Parameters
    ----------
    x : vector with sampled values.

    Returns
    -------
    ind : indices to minima and maxima in the original sequence x.

    Examples
    --------
    >>> import numpy as np
    >>> import pylab as plt
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)
    >>> ind = wm.findextrema(x)

    >>> a = plt.plot(t,x,'.',t[ind],x[ind],'r.')
    >>> plt.close('all')

    See also
    --------
    findcross
    crossdef
    '''
    xn = atleast_1d(x).ravel()
    return findcross(diff(xn), 0.0) + 1


def findpeaks(data, n=2, min_h=None, min_p=0.0):
    '''
    Find peaks of vector or matrix possibly rainflow filtered

    Parameters
    ----------
    data      = matrix or vector
    n     = The n highest peaks are found (if exist). (default 2)
    min_h  = The threshold in the rainflowfilter (default 0.05*range(S(:))).
                 A zero value will return all the peaks of S.
    min_p  = 0..1, Only the peaks that are higher than
                       min_p*max(max(S))  min_p*(the largest peak in S)
                       are returned (default  0).
    Returns
    ix =
        linear index to peaks of S

    Example:

    Find highest 8 peaks that are not
    less that 0.3*"global max" and have
    rainflow amplitude larger than 5.
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> x = np.arange(0,10,0.01)
    >>> data = x**2+10*np.sin(3*x)+0.5*np.sin(50*x)
    >>> wm.findpeaks(data, n=8, min_h=5, min_p=0.3)
    array([908, 694, 481])

    See also
    --------
    findtp
    '''
    S = np.atleast_1d(data)
    smax = S.max()
    if min_h is None:
        smin = S.min()
        min_h = 0.05 * (smax - smin)
    ndim = S.ndim
    S = np.atleast_2d(S)
    nrows, mcols = S.shape

    # Finding turningpoints of the spectrum
    # Returning only those with rainflowcycle heights greater than h_min
    indP = []  # indices to peaks
    ind = []
    for iy in range(nrows):  # find all peaks
        TuP = findtp(S[iy], min_h)
        if len(TuP):
            ind = TuP[1::2]  # extract indices to maxima only
        else:  # did not find any , try maximum
            ind = np.atleast_1d(S[iy].argmax())

        if ndim > 1:
            if iy == 0:
                ind2 = np.flatnonzero(S[iy, ind] > S[iy + 1, ind])
            elif iy == nrows - 1:
                ind2 = np.flatnonzero(S[iy, ind] > S[iy - 1, ind])
            else:
                ind2 = np.flatnonzero((S[iy, ind] > S[iy - 1, ind]) &
                                      (S[iy, ind] > S[iy + 1, ind]))

            if len(ind2):
                indP.append((ind[ind2] + iy * mcols))

    if ndim > 1:
        ind = np.hstack(indP) if len(indP) else []
    if len(ind) == 0:
        return []

    peaks = S.take(ind)
    ind2 = peaks.argsort()[::-1]

    # keeping only the Np most significant peak frequencies.
    nmax = min(n, len(ind))
    ind = ind[ind2[:nmax]]
    if (min_p > 0):
        # Keeping only peaks larger than min_p percent relative to the maximum
        # peak
        ind = ind[(S.take(ind) > min_p * smax)]

    return ind


def findrfc_astm(tp):
    """
    Return rainflow counted cycles

    Nieslony's Matlab implementation of the ASTM standard practice for rainflow
    counting ported to a Python C module.

    Parameters
    ----------
    tp : array-like
        vector of turningpoints (NB! Only values, not sampled times)

    Returns
    -------
    sig_rfc : array-like
        array of shape (n,3) with:
        sig_rfc[:,0] Cycles amplitude
        sig_rfc[:,1] Cycles mean value
        sig_rfc[:,2] Cycle type, half (=0.5) or full (=1.0)
    """

    y1 = atleast_1d(tp).ravel()
    sig_rfc, cnr = clib.findrfc3_astm(y1)
    # the sig_rfc was constructed too big in rainflow.rf3, so
    # reduce the sig_rfc array as done originally by a matlab mex c function
    n = len(sig_rfc)
    sig_rfc = sig_rfc.__getslice__(0, n - cnr[0])
    # sig_rfc holds the actual rainflow counted cycles, not the indices
    return sig_rfc


def findrfc(tp, h=0.0, method='clib'):
    '''
    Return indices to rainflow cycles of a sequence of TP.

    Parameters
    -----------
    tp : array-like
        vector of turningpoints (NB! Only values, not sampled times)
    h : real scalar
        rainflow threshold. If h>0, then all rainflow cycles with height
        smaller than h are removed.
    method : string, optional
        'clib' 'None'
        Specify 'clib' for calling the c_functions, otherwise fallback to
        the Python implementation.

    Returns
    -------
    ind : ndarray of int
        indices to the rainflow cycles of the original sequence TP.

    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)+0.1*np.sin(50*t)
    >>> ind = wm.findextrema(x)
    >>> ti, tp = t[ind], x[ind]

    >>> a = plt.plot(t,x,'.',ti,tp,'r.')
    >>> ind1 = wm.findrfc(tp,0.3); ind1
    array([  0,   9,  32,  53,  74,  95, 116, 137])
    >>> ind2 = wm.findrfc(tp,0.3, method=''); ind2
    array([  0,   9,  32,  53,  74,  95, 116, 137])
    >>> a = plt.plot(ti[ind1],tp[ind1])
    >>> plt.close('all')

    See also
    --------
    rfcfilter,
    findtp.
    '''
    y = atleast_1d(tp).ravel()

    t_start = int(y[0] > y[1])  # first is a max, ignore it
    y = y[t_start::]
    n = len(y)
    NC = np.floor(n / 2) - 1

    if (NC < 1):
        return zeros(0, dtype=np.int)  # No RFC cycles*/

    if (y[0] > y[1] and y[1] > y[2] or
            y[0] < y[1] and y[1] < y[2]):
        warnings.warn('This is not a sequence of turningpoints, exit')
        return zeros(0, dtype=np.int)

    if clib is None or method not in ('clib'):
        ind, ix = _findrfc(y, h)
    else:
        ind, ix = clib.findrfc(y, h)
    return np.sort(ind[:ix]) + t_start


def _findrfc(y, h):
    # TODO: merge rfcfilter and _findrfc
    t_start = 0

    n = len(y)
    NC = np.floor(n / 2) - 1
    ind = zeros(n, dtype=np.int)
    NC = np.int(NC)
    ix = 0
    for i in range(NC):
        Tmi = t_start + 2 * i
        Tpl = t_start + 2 * i + 2
        xminus = y[2 * i]
        xplus = y[2 * i + 2]

        if(i != 0):
            j = i - 1
            while ((j >= 0) and (y[2 * j + 1] <= y[2 * i + 1])):
                if (y[2 * j] < xminus):
                    xminus = y[2 * j]
                    Tmi = t_start + 2 * j
                j -= 1
        if (xminus >= xplus):
            if (y[2 * i + 1] - xminus >= h):
                ind[ix] = Tmi
                ix += 1
                ind[ix] = (t_start + 2 * i + 1)
                ix += 1
            # goto L180 continue
        else:
            j = i + 1
            while (j < NC):
                if (y[2 * j + 1] >= y[2 * i + 1]):
                    break  # goto L170
                if((y[2 * j + 2] <= xplus)):
                    xplus = y[2 * j + 2]
                    Tpl = (t_start + 2 * j + 2)
                j += 1
            else:
                if ((y[2 * i + 1] - xminus) >= h):
                    ind[ix] = Tmi
                    ix += 1
                    ind[ix] = (t_start + 2 * i + 1)
                    ix += 1
                # iy = i
                continue

            # goto L180
            # L170:
            if (xplus <= xminus):
                if ((y[2 * i + 1] - xminus) >= h):
                    ind[ix] = Tmi
                    ix += 1
                    ind[ix] = (t_start + 2 * i + 1)
                    ix += 1
            elif ((y[2 * i + 1] - xplus) >= h):
                ind[ix] = (t_start + 2 * i + 1)
                ix += 1
                ind[ix] = Tpl
                ix += 1

        # L180:
        # iy=i
    #  /* for i */
    return ind, ix


def mctp2rfc(fmM, fMm=None):
    '''
    Return Rainflow matrix given a Markov chain of turning points

    computes f_rfc = f_mM + F_mct(f_mM).

    Parameters
    ----------
    fmM =  the min2max Markov matrix,
    fMm = the max2min Markov matrix,

    Returns
    -------
    f_rfc = the rainflow matrix,

    Example:
    -------
    >>> fmM = np.array([[ 0.0183,    0.0160,    0.0002,    0.0000,         0],
    ...            [0.0178,    0.5405,    0.0952,         0,         0],
    ...            [0.0002,    0.0813,         0,         0,         0],
    ...            [0.0000,         0,         0,         0,         0],
    ...            [     0,         0,         0,         0,         0]])

    >>> np.abs(mctp2rfc(fmM)-np.array([[2.669981e-02, 7.799700e-03,
    ...        4.906077e-07, 0.000000e+00, 0.000000e+00],
    ...       [  9.599629e-03,   5.485009e-01,   9.539951e-02,   0.000000e+00,
    ...          0.000000e+00],
    ...       [  5.622974e-07,   8.149944e-02,   0.000000e+00,   0.000000e+00,
    ...          0.000000e+00],
    ...       [  0.000000e+00,   0.000000e+00,   0.000000e+00,   0.000000e+00,
    ...          0.000000e+00],
    ...       [  0.000000e+00,   0.000000e+00,   0.000000e+00,   0.000000e+00,
    ...          0.000000e+00]]))<1.e-7
    array([[ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]], dtype=bool)
    '''
    def _get_PMm(AA1, MA, nA):
        PMm = AA1.copy()
        for j in range(nA):
            norm = MA[j]
            if norm != 0:
                PMm[j, :] = PMm[j, :] / norm
                # end
            # end
        PMm = np.fliplr(PMm)
        return PMm

    if fMm is None:
        fmM = np.atleast_1d(fmM)
        fMm = fmM
    else:
        fmM, fMm = np.atleast_1d(fmM, fMm)
    f_mM, f_Mm = fmM.copy(), fMm.copy()
    N = max(f_mM.shape)
    f_max = np.sum(f_mM, axis=1)
    f_min = np.sum(f_mM, axis=0)
    f_rfc = zeros((N, N))
    f_rfc[N - 2, 0] = f_max[N - 2]
    f_rfc[0, N - 2] = f_min[N - 2]
    for k in range(2, N - 1):
        for i in range(1, k):
            AA = f_mM[N - 1 - k:N - 1 - k + i, k - i:k]
            AA1 = f_Mm[N - 1 - k:N - 1 - k + i, k - i:k]
            RAA = f_rfc[N - 1 - k:N - 1 - k + i, k - i:k]
            nA = max(AA.shape)
            MA = f_max[N - 1 - k:N - 1 - k + i]
            mA = f_min[k - i:k]
            SA = AA.sum()
            SRA = RAA.sum()

            DRFC = SA - SRA
            NT = min(mA[0] - sum(RAA[:, 0]), MA[0] - sum(RAA[0, :]))  # check!
            NT = max(NT, 0)  # ??check

            if NT > 1e-6 * max(MA[0], mA[0]):
                NN = MA - np.sum(AA, axis=1)  # T
                e = (mA - np.sum(AA, axis=0))  # T
                e = np.flipud(e)
                PmM = np.rot90(AA.copy())
                for j in range(nA):
                    norm = mA[nA - 1 - j]
                    if norm != 0:
                        PmM[j, :] = PmM[j, :] / norm
                        e[j] = e[j] / norm
                    # end
                # end
                fx = 0.0
                if (max(abs(e)) > 1e-6 and
                        max(abs(NN)) > 1e-6 * max(MA[0], mA[0])):
                    PMm = _get_PMm(AA1, MA, nA)

                    A = PMm
                    B = PmM

                    if nA == 1:
                        fx = NN * (A / (1 - B * A) * e)
                    else:
                        rh = np.eye(A.shape[0]) - np.dot(B, A)
                        # least squares
                        fx = np.dot(NN, np.dot(A, linalg.solve(rh, e)))
                    # end
                # end
                f_rfc[N - 1 - k, k - i] = fx + DRFC

                #  check2=[ DRFC  fx]
                # pause
            else:
                f_rfc[N - 1 - k, k - i] = 0.0
            # end
        # end
        m0 = max(0, f_min[0] - np.sum(f_rfc[N - k + 1:N, 0]))
        M0 = max(0, f_max[N - 1 - k] - np.sum(f_rfc[N - 1 - k, 1:k]))
        f_rfc[N - 1 - k, 0] = min(m0, M0)
        #  n_loops_left=N-k+1
    # end

    for k in range(1, N):
        M0 = max(0, f_max[0] - np.sum(f_rfc[0, N - k:N]))
        m0 = max(0, f_min[N - 1 - k] - np.sum(f_rfc[1:k + 1, N - 1 - k]))
        f_rfc[0, N - 1 - k] = min(m0, M0)
    # end

#    %clf
#    %subplot(1,2,2)
#    %pcolor(levels(paramm),levels(paramM),flipud(f_mM))
#    %  title('Markov matrix')
#    %  ylabel('max'), xlabel('min')
#    %axis([paramm(1) paramm(2) paramM(1) paramM(2)])
#    %axis('square')
#
#    %subplot(1,2,1)
#    %pcolor(levels(paramm),levels(paramM),flipud(f_rfc))
#    %  title('Rainflow matrix')
#    %  ylabel('max'), xlabel('rfc-min')
#    %axis([paramm(1) paramm(2) paramM(1) paramM(2)])
#    %axis('square')

    return f_rfc


def rfcfilter(x, h, method=0):
    """
    Rainflow filter a signal.

    Parameters
    -----------
    x : vector
        Signal.   [nx1]
    h : real, scalar
        Threshold for rainflow filter.
    method : scalar, integer
        0 : removes cycles with range < h. (default)
        1 : removes cycles with range <= h.

    Returns
    --------
    y   = Rainflow filtered signal.

    Examples:
    ---------
    # 1. Filtered signal y is the turning points of x.
    >>> import wafo.data as data
    >>> import wafo.misc as wm
    >>> x = data.sea()
    >>> y = wm.rfcfilter(x[:,1], h=0, method=1)
    >>> np.all(np.abs(y[0:5]-np.array([-1.2004945 , 0.83950546, -0.09049454,
    ...        -0.02049454, -0.09049454]))<1e-7)
    True
    >>> y.shape
    (2172,)

    # 2. This removes all rainflow cycles with range less than 0.5.
    >>> y1 = wm.rfcfilter(x[:,1], h=0.5)
    >>> y1.shape
    (863,)
    >>> np.all(np.abs(y1[0:5]-np.array([-1.2004945 , 0.83950546, -0.43049454,
    ...        0.34950546, -0.51049454]))<1e-7)
    True
    >>> ind = wm.findtp(x[:,1], h=0.5)
    >>> y2 = x[ind,1]
    >>> y2[0:5]
    array([-1.2004945 ,  0.83950546, -0.43049454,  0.34950546, -0.51049454])
    >>> y2[-5::]
    array([ 0.83950546, -0.64049454,  0.65950546, -1.0004945 ,  0.91950546])

    See also
    --------
    findrfc
    """
    # TODO merge rfcfilter and findrfc
    y = atleast_1d(x).ravel()
    n = len(y)
    t = zeros(n, dtype=np.int)
    j = 0
    t0 = 0
    y0 = y[t0]
    z0 = 0

    def aleb(a, b):
        return a <= b

    def altb(a, b):
        return a < b

    if method == 0:
        cmpfun1 = aleb
        cmpfun2 = altb
    else:
        cmpfun1 = altb
        cmpfun2 = aleb

    # The rainflow filter
    for tim1, yi in enumerate(y[1::]):
        fpi = y0 + h
        fmi = y0 - h
        ti = tim1 + 1
        # yi = y[ti]

        if z0 == 0:
            if cmpfun1(yi, fmi):
                z1 = -1
            elif cmpfun1(fpi, yi):
                z1 = +1
            else:
                z1 = 0
            t1, y1 = (t0, y0) if z1 == 0 else (ti, yi)
        else:
            if (((z0 == +1) & cmpfun1(yi, fmi)) |
                    ((z0 == -1) & cmpfun2(yi, fpi))):
                z1 = -1
            elif (((z0 == +1) & cmpfun2(fmi, yi)) |
                  ((z0 == -1) & cmpfun1(fpi, yi))):
                z1 = +1
            else:
                warnings.warn('Something wrong, i=%d' % tim1)

            # Update y1
            if z1 != z0:
                t1, y1 = ti, yi
            elif z1 == -1:
                # y1 = min([y0 xi])
                t1, y1 = (t0, y0) if y0 < yi else (ti, yi)
            elif z1 == +1:
                # y1 = max([y0 xi])
                t1, y1 = (t0, y0) if y0 > yi else (ti, yi)

        # Update y if y0 is a turning point
        if abs(z0 - z1) == 2:
            j += 1
            t[j] = t0

        # Update t0, y0, z0
        t0, y0, z0 = t1, y1, z1
    # end

    # Update y if last y0 is greater than (or equal) threshold
    if cmpfun1(h, abs(y0 - y[t[j]])):
        j += 1
        t[j] = t0
    return y[t[:j + 1]]


def findtp(x, h=0.0, kind=None):
    '''
    Return indices to turning points (tp) of data, optionally rainflowfiltered.

    Parameters
    ----------
    x : vector
        signal
    h : real, scalar
        rainflow threshold
         if  h<0, then ind = range(len(x))
         if  h=0, then  tp  is a sequence of turning points (default)
         if  h>0, then all rainflow cycles with height smaller than
                  h  are removed.
    kind : string
        defines the type of wave or indicate the ASTM rainflow counting method.
        Possible options are 'astm' 'mw' 'Mw' or 'none'.
        If None all rainflow filtered min and max
        will be returned, otherwise only the rainflow filtered
        min and max, which define a wave according to the
        wave definition, will be returned.

    Returns
    -------
    ind : arraylike
        indices to the turning points in the original sequence.

    Example:
    --------
    >>> import pylab as plt
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> x = np.hstack((t, np.cos(t) + 0.3 * np.sin(5*t)))
    >>> x1 = x[0:100,:]
    >>> itp = wm.findtp(x1[:,1],0,'Mw')
    >>> itph = wm.findtp(x1[:,1],0.3,'Mw')
    >>> tp = x1[itp,:]
    >>> tph = x1[itph,:]
    >>> a = plt.plot(x1[:,0],x1[:,1],
    ...             tp[:,0],tp[:,1],'ro',
    ...             tph[:,0],tph[:,1],'k.')
    >>> plt.close('all')
    >>> itp
    array([ 5, 18, 24, 38, 46, 57, 70, 76, 91, 98, 99])
    >>> itph
    array([91])

    See also
    ---------
    findtc
    findcross
    findextrema
    findrfc
    '''
    n = len(x)
    if h < 0.0:
        return arange(n)

    ind = findextrema(x)

    if ind.size < 2:
        return None

    # In order to get the exact up-crossing intensity from rfc by
    # mm2lc(tp2mm(rfc))  we have to add the indices to the last value
    # (and also the first if the sequence of turning points does not start
    # with a minimum).

    if kind == 'astm':
        # the Nieslony approach always put the first loading point as the first
        # turning point.
        # add the first turning point is the first of the signal
        if x[ind[0]] != x[0]:
            ind = np.r_[0, ind, n - 1]
        else:  # only add the last point of the signal
            ind = np.r_[ind, n - 1]
    else:
        if x[ind[0]] > x[ind[1]]:  # adds indices to  first and last value
            ind = r_[0, ind, n - 1]
        else:  # adds index to the last value
            ind = r_[ind, n - 1]

    if h > 0.0:
        ind1 = findrfc(x[ind], h)
        ind = ind[ind1]

    if kind in ('mw', 'Mw'):
        # make sure that the first is a Max if wdef == 'Mw'
        # or make sure that the first is a min if wdef == 'mw'
        first_is_max = (x[ind[0]] > x[ind[1]])

        remove_first = xor(first_is_max, kind.startswith('Mw'))
        if remove_first:
            ind = ind[1::]

        # make sure the number of minima and Maxima are according to the
        # wavedef. i.e., make sure Nm=length(ind) is odd
        if (mod(ind.size, 2)) != 1:
            ind = ind[:-1]
    return ind


def findtc(x_in, v=None, kind=None):
    """
    Return indices to troughs and crests of data.

    Parameters
    ----------
    x : vector
        surface elevation.
    v : real scalar
        reference level (default  v = mean of x).

    kind : string
        defines the type of wave. Possible options are
        'dw', 'uw', 'tw', 'cw' or None.
        If None indices to all troughs and crests will be returned,
        otherwise only the paired ones will be returned
        according to the wavedefinition.

    Returns
    --------
    tc_ind : vector of ints
        indices to the trough and crest turningpoints of sequence x.
    v_ind : vector of ints
        indices to the level v crossings of the original
        sequence x. (d,u)

    Example:
    --------
    >>> import pylab as plt
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> x = np.hstack((t, np.cos(t)))
    >>> x1 = x[0:200,:]
    >>> itc, iv = wm.findtc(x1[:,1],0,'dw')
    >>> tc = x1[itc,:]
    >>> a = plt.plot(x1[:,0],x1[:,1],tc[:,0],tc[:,1],'ro')
    >>> plt.close('all')

    See also
    --------
    findtp
    findcross,
    wavedef
    """

    x = atleast_1d(x_in)
    if v is None:
        v = x.mean()

    v_ind = findcross(x, v, kind)
    n_c = v_ind.size
    if n_c <= 2:
        warnings.warn('There are no waves!')
        return zeros(0, dtype=np.int), zeros(0, dtype=np.int)

    # determine the number of trough2crest (or crest2trough) cycles
    is_even = mod(n_c + 1, 2)
    n_tc = int((n_c - 1 - is_even) / 2)

    # allocate variables before the loop increases the speed
    ind = zeros(n_c - 1, dtype=np.int)

    first_is_down_crossing = (x[v_ind[0]] > x[v_ind[0] + 1])
    if first_is_down_crossing:
        for i in range(n_tc):
            # trough
            j = 2 * i
            ind[j] = x[v_ind[j] + 1:v_ind[j + 1] + 1].argmin()
            # crest
            ind[j + 1] = x[v_ind[j + 1] + 1:v_ind[j + 2] + 1].argmax()

        if (2 * n_tc + 1 < n_c) and (kind in (None, 'tw')):
            # trough
            ind[n_c - 2] = x[v_ind[n_c - 2] + 1:v_ind[n_c - 1]].argmin()

    else:  # the first is a up-crossing
        for i in range(n_tc):
            # crest
            j = 2 * i
            ind[j] = x[v_ind[j] + 1:v_ind[j + 1] + 1].argmax()
            # trough
            ind[j + 1] = x[v_ind[j + 1] + 1:v_ind[j + 2] + 1].argmin()

        if (2 * n_tc + 1 < n_c) and (kind in (None, 'cw')):
            # crest
            ind[n_c - 2] = x[v_ind[n_c - 2] + 1:v_ind[n_c - 1]].argmax()

    return v_ind[:n_c - 1] + ind + 1, v_ind


def findoutliers(x, zcrit=0.0, dcrit=None, ddcrit=None, verbose=False):
    """
    Return indices to spurious points of data

    Parameters
    ----------
    x : vector
        of data values.
    zcrit : real scalar
        critical distance between consecutive points.
    dcrit : real scalar
        critical distance of Dx used for determination of spurious
        points.  (Default 1.5 standard deviation of x)
    ddcrit : real scalar
        critical distance of DDx used for determination of spurious
        points.  (Default 1.5 standard deviation of x)

    Returns
    -------
    inds : ndarray of integers
        indices to spurious points.
    indg : ndarray of integers
        indices to the rest of the points.

    Notes
    -----
    Consecutive points less than zcrit apart  are considered as spurious.
    The point immediately after and before are also removed. Jumps greater than
    dcrit in Dxn and greater than ddcrit in D^2xn are also considered as
    spurious.
    (All distances to be interpreted in the vertical direction.)
    Another good choice for dcrit and ddcrit are:

        dcrit = 5*dT  and ddcrit = 9.81/2*dT**2

    where dT is the timestep between points.

    Examples
    --------
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> xx = np.hstack((t, np.cos(t)))
    >>> dt = np.diff(xx[:2,0])
    >>> dcrit = 5*dt
    >>> ddcrit = 9.81/2*dt*dt
    >>> zcrit = 0
    >>> [inds, indg] = wm.findoutliers(xx[:,1],zcrit,dcrit,ddcrit,verbose=True)
    Found 0 spurious positive jumps of Dx
    Found 0 spurious negative jumps of Dx
    Found 0 spurious positive jumps of D^2x
    Found 0 spurious negative jumps of D^2x
    Found 0 consecutive equal values
    Found the total of 0 spurious points


    #waveplot(xx,'-',xx(inds,:),1,1,1)

    See also
    --------
    waveplot, reconstruct
    """

    def _find_nans(xn):
        i_missing = np.flatnonzero(np.isnan(xn))
        if verbose:
            print('Found %d missing points' % i_missing.size)
        return i_missing

    def _find_spurious_jumps(dxn, dcrit, name='Dx'):
        i_p = np.flatnonzero(dxn > dcrit)
        if i_p.size > 0:
            i_p += 1  # the point after the jump
        if verbose:
            print('Found {0:d} spurious positive jumps of {1}'.format(i_p.size,
                                                                      name))

        i_n = np.flatnonzero(dxn < -dcrit)  # the point before the jump
        if verbose:
            print('Found {0:d} spurious negative jumps of {1}'.format(i_n.size,
                                                                      name))
        if i_n.size > 0:
            return hstack((i_p, i_n))
        return i_p

    def _find_consecutive_equal_values(dxn, zcrit):

        mask_small = (abs(dxn) <= zcrit)
        i_small = np.flatnonzero(mask_small)
        if verbose:
            if zcrit == 0.:
                print('Found %d consecutive equal values' % i_small.size)
            else:
                print('Found %d consecutive values less than %g apart.' %
                      (i_small.size, zcrit))
        if i_small.size > 0:
            i_small += 1
            # finding the beginning and end of consecutive equal values
            i_step = np.flatnonzero((diff(mask_small))) + 1
            # indices to consecutive equal points
            # removing the point before + all equal points + the point after

            return hstack((i_step - 1, i_small, i_step, i_step + 1))
        return i_small

    xn = asarray(x).flatten()

    if xn.size < 2:
        raise ValueError('The vector must have more than 2 elements!')

    i_missing = _find_nans(xn)
    if np.any(i_missing):
        xn[i_missing] = 0.  # set NaN's to zero
    if dcrit is None:
        dcrit = 1.5 * xn.std()
        if verbose:
            print('dcrit is set to %g' % dcrit)

    if ddcrit is None:
        ddcrit = 1.5 * xn.std()
        if verbose:
            print('ddcrit is set to %g' % ddcrit)

    dxn = diff(xn)
    ddxn = diff(dxn)

    ind = np.hstack((_find_spurious_jumps(dxn, dcrit, name='Dx'),
                     _find_spurious_jumps(ddxn, ddcrit, name='D^2x'),
                     _find_consecutive_equal_values(dxn, zcrit)))

    indg = ones(xn.size, dtype=bool)
    if ind.size > 1:
        ind = unique(ind)
        indg[ind] = 0
    indg, = nonzero(indg)

    if verbose:
        print('Found the total of %d spurious points' % ind.size)

    return ind, indg


def common_shape(*args, ** kwds):
    '''
    Return the common shape of a sequence of arrays

    Parameters
    -----------
    *args : arraylike
        sequence of arrays
    **kwds :
        shape

    Returns
    -------
    shape : tuple
        common shape of the elements of args.

    Raises
    ------
    An error is raised if some of the arrays do not conform
    to the common shape according to the broadcasting rules in numpy.

    Examples
    --------
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> A = np.ones((4,1))
    >>> B = 2
    >>> C = np.ones((1,5))*5
    >>> wm.common_shape(A,B,C)
    (4, 5)
    >>> wm.common_shape(A,B,C,shape=(3,4,1))
    (3, 4, 5)

    See also
    --------
    broadcast, broadcast_arrays
    '''
    shape = kwds.get('shape')
    x0 = 1 if shape is None else np.ones(shape)
    x1 = np.broadcast(x0, *args)
    return tuple(x1.shape)


def argsreduce(condition, * args):
    """ Return the elements of each input array that satisfy some condition.

    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements of each
        input array to extract. The shape of 'condition' must match the common
        shape of the input arrays according to the broadcasting rules in numpy.
    arg1, arg2, arg3, ... : array_like
        one or more input arrays.

    Returns
    -------
    narg1, narg2, narg3, ... : ndarray
        sequence of extracted copies of the input arrays converted to the same
        size as the nonzero values of condition.

    Example
    -------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> rand = np.random.random_sample
    >>> A = rand((4,5))
    >>> B = 2
    >>> C = rand((1,5))
    >>> cond = np.ones(A.shape)
    >>> [A1,B1,C1] = wm.argsreduce(cond,A,B,C)
    >>> B1.shape
    (20,)
    >>> cond[2,:] = 0
    >>> [A2,B2,C2] = wm.argsreduce(cond,A,B,C)
    >>> B2.shape
    (15,)

    See also
    --------
    numpy.extract
    """
    newargs = atleast_1d(*args)
    if not isinstance(newargs, list):
        newargs = [newargs, ]
    expand_arr = (condition == condition)
    return [extract(condition, arr1 * expand_arr) for arr1 in newargs]


def stirlerr(n):
    '''
    Return error of Stirling approximation,
        i.e., log(n!) - log( sqrt(2*pi*n)*(n/exp(1))**n )

    Example
    -------
    >>> import wafo.misc as wm
    >>> np.abs(wm.stirlerr(2)- 0.0413407)<1e-7
    array([ True], dtype=bool)

    See also
    ---------
    binom


    Reference
    -----------
    Catherine Loader (2000).
    Fast and Accurate Computation of Binomial Probabilities
    <http://lists.gnu.org/archive/html/octave-maintainers/2011-09/pdfK0uKOST642.pdf>
    '''

    S0 = 0.083333333333333333333  # /* 1/12 */
    S1 = 0.00277777777777777777778  # /* 1/360 */
    S2 = 0.00079365079365079365079365  # /* 1/1260 */
    S3 = 0.000595238095238095238095238  # /* 1/1680 */
    S4 = 0.0008417508417508417508417508  # /* 1/1188 */

    n1 = atleast_1d(n)

    y = gammaln(n1 + 1) - log(sqrt(2 * pi * n1) * (n1 / exp(1)) ** n1)

    nn = n1 * n1

    n500 = 500 < n1
    y[n500] = (S0 - S1 / nn[n500]) / n1[n500]
    n80 = logical_and(80 < n1, n1 <= 500)
    if any(n80):
        y[n80] = (S0 - (S1 - S2 / nn[n80]) / nn[n80]) / n1[n80]
    n35 = logical_and(35 < n1, n1 <= 80)
    if any(n35):
        nn35 = nn[n35]
        y[n35] = (S0 - (S1 - (S2 - S3 / nn35) / nn35) / nn35) / n1[n35]

    n15 = logical_and(15 < n1, n1 <= 35)
    if any(n15):
        nn15 = nn[n15]
        y[n15] = (
            S0 - (S1 - (S2 - (S3 - S4 / nn15) / nn15) / nn15) / nn15) / n1[n15]

    return y


def getshipchar(value=None, property="max_deadweight",  # @ReservedAssignment
                **kwds):  # @IgnorePep8
    '''
    Return ship characteristics from value of one ship-property

    Parameters
    ----------
    value : scalar
        value to use in the estimation.
    property : string
        defining the ship property used in the estimation. Options are:
           'max_deadweight','length','beam','draft','service_speed',
           'propeller_diameter'.
           The length was found from statistics of 40 vessels of size 85 to
           100000 tonn. An exponential curve through 0 was selected, and the
           factor and exponent that minimized the standard deviation of the
           relative error was selected. (The error returned is the same for
           any ship.) The servicespeed was found for ships above 1000 tonns
           only. The propeller diameter formula is from [1]_.

    Returns
    -------
    sc : dict
        containing estimated mean values and standard-deviations of ship
        characteristics:
            max_deadweight    [kkg], (weight of cargo, fuel etc.)
            length            [m]
            beam              [m]
            draught           [m]
            service_speed      [m/s]
            propeller_diameter [m]

    Example
    ---------
    >>> import wafo.misc as wm
    >>> sc = wm.getshipchar(10,'service_speed')
    >>> for key in sorted(sc): key, sc[key]
    ('beam', 29.0)
    ('beamSTD', 2.9000000000000004)
    ('draught', 9.6)
    ('draughtSTD', 2.112)
    ('length', 216.0)
    ('lengthSTD', 2.011309883194276)
    ('max_deadweight', 30969.0)
    ('max_deadweightSTD', 3096.9)
    ('propeller_diameter', 6.761165385916601)
    ('propeller_diameterSTD', 0.20267047566705432)
    ('service_speed', 10.0)
    ('service_speedSTD', 0)

    Other units: 1 ft = 0.3048 m and 1 knot = 0.5144 m/s


    Reference
    ---------
    .. [1] Gray and Greeley, (1978),
    "Source level model for propeller blade rate radiation for the world's
    merchant fleet", Bolt Beranek and Newman Technical Memorandum No. 458.
    '''
    if value is None:
        names = kwds.keys()
        if len(names) != 1:
            raise ValueError('Only on keyword')
        property = names[0]  # @ReservedAssignment
        value = kwds[property]
    value = np.atleast_1d(value)
    valid_props = dict(l='length', b='beam', d='draught', m='max_deadweigth',
                       s='service_speed', p='propeller_diameter')
    prop = valid_props[property[0]]

    prop2max_dw = dict(length=lambda x: (x / 3.45) ** (2.5),
                       beam=lambda x: ((x / 1.78) ** (1 / 0.27)),
                       draught=lambda x: ((x / 0.8) ** (1 / 0.24)),
                       service_speed=lambda x: ((x / 1.14) ** (1 / 0.21)),
                       propeller_diameter=lambda x: (((x / 0.12) ** (4 / 3) /
                                                      3.45) ** (2.5)))

    max_deadweight = prop2max_dw.get(prop, lambda x: x)(value)
    propertySTD = prop + 'STD'

    length = np.round(3.45 * max_deadweight ** 0.40)
    length_err = length ** 0.13

    beam = np.round(1.78 * max_deadweight ** 0.27 * 10) / 10
    beam_err = beam * 0.10

    draught = np.round(0.80 * max_deadweight ** 0.24 * 10) / 10
    draught_err = draught * 0.22

    # S    = round(2/3*(L)**0.525)
    speed = np.round(1.14 * max_deadweight ** 0.21 * 10) / 10
    speed_err = speed * 0.10

    p_diam = 0.12 * length ** (3.0 / 4.0)
    p_diam_err = 0.12 * length_err ** (3.0 / 4.0)

    max_deadweight = np.round(max_deadweight)
    max_deadweightSTD = 0.1 * max_deadweight

    shipchar = OrderedDict(beam=beam, beamSTD=beam_err,
                           draught=draught, draughtSTD=draught_err,
                           length=length, lengthSTD=length_err,
                           max_deadweight=max_deadweight,
                           max_deadweightSTD=max_deadweightSTD,
                           propeller_diameter=p_diam,
                           propeller_diameterSTD=p_diam_err,
                           service_speed=speed, service_speedSTD=speed_err)

    shipchar[propertySTD] = 0
    return shipchar


def binomln(z, w):
    '''
    Natural Logarithm of binomial coefficient.

    CALL binomln(z,w)

    BINOMLN computes the natural logarithm of the binomial
    function for corresponding elements of Z and W.   The arrays Z and
    W must be real and nonnegative. Both arrays must be the same size,
    or either can be scalar.  BETALOGE is defined as:

    y = LOG(binom(Z,W)) = gammaln(Z)-gammaln(W)-gammaln(Z-W)

    and is obtained without computing BINOM(Z,W). Since the binom
    function can range over very large or very small values, its
    logarithm is sometimes more useful.
    This implementation is more accurate than the log(BINOM(Z,W) implementation
    for large arguments

    Example
    -------

    >>> abs(binomln(3,2)- 1.09861229)<1e-7
    array([ True], dtype=bool)

    See also
    --------
    binom
    '''
    # log(n!) = stirlerr(n)  + log( sqrt(2*pi*n)*(n/exp(1))**n )
    # y = gammaln(z+1)-gammaln(w+1)-gammaln(z-w+1)
    zpw = z - w
    return (stirlerr(z + 1) - stirlerr(w + 1) - 0.5 * log(2 * pi) -
            (w + 0.5) * log1p(w) + (z + 0.5) * log1p(z) - stirlerr(zpw + 1) -
            (zpw + 0.5) * log1p(zpw) + 1)


def betaloge(z, w):
    '''
    Natural Logarithm of beta function.

    CALL betaloge(z,w)

    BETALOGE computes the natural logarithm of the beta
    function for corresponding elements of Z and W.   The arrays Z and
    W must be real and nonnegative. Both arrays must be the same size,
    or either can be scalar.  BETALOGE is defined as:

    y = LOG(BETA(Z,W)) = gammaln(Z)+gammaln(W)-gammaln(Z+W)

    and is obtained without computing BETA(Z,W). Since the beta
    function can range over very large or very small values, its
    logarithm is sometimes more useful.
    This implementation is more accurate than the BETALN implementation
    for large arguments

    Example
    -------
    >>> import wafo.misc as wm
    >>> abs(wm.betaloge(3,2)+2.48490665)<1e-7
    array([ True], dtype=bool)

    See also
    --------
    betaln, beta
    '''
    # y = gammaln(z)+gammaln(w)-gammaln(z+w)
    zpw = z + w
    return (stirlerr(z) + stirlerr(w) + 0.5 * log(2 * pi) +
            (w - 0.5) * log(w) + (z - 0.5) * log(z) - stirlerr(zpw) -
            (zpw - 0.5) * log(zpw))

    # stirlings approximation:
    #  (-(zpw-0.5).*log(zpw) +(w-0.5).*log(w)+(z-0.5).*log(z) +0.5*log(2*pi))
    # return y


def gravity(phi=45):
    ''' Returns the constant acceleration of gravity

    GRAVITY calculates the acceleration of gravity
    using the international gravitational formulae [1]_:

      g = 9.78049*(1+0.0052884*sin(phir)**2-0.0000059*sin(2*phir)**2)
    where
      phir = phi*pi/180

    Parameters
    ----------
    phi : {float, int}
         latitude in degrees

    Returns
    --------
    g : ndarray
        acceleration of gravity [m/s**2]

    Examples
    --------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> phi = np.linspace(0,45,5)
    >>> np.abs(wm.gravity(phi)-np.array([ 9.78049   ,  9.78245014,  9.78803583,
    ...            9.79640552,  9.80629387]))<1.e-7
    array([ True,  True,  True,  True,  True], dtype=bool)

    See also
    --------
    wdensity

    References
    ----------
    .. [1] Irgens, Fridtjov (1987)
            "Formelsamling i mekanikk:
            statikk, fasthetsl?re, dynamikk fluidmekanikk"
            tapir forlag, University of Trondheim,
            ISBN 82-519-0786-1, pp 19

    '''

    phir = phi * pi / 180.  # change from degrees to radians
    return 9.78049 * (1. + 0.0052884 * sin(phir) ** 2. -
                      0.0000059 * sin(2 * phir) ** 2.)


def nextpow2(x):
    '''
    Return next higher power of 2

    Example
    -------
    >>> import wafo.misc as wm
    >>> wm.nextpow2(10)
    4
    >>> wm.nextpow2(np.arange(5))
    3
    '''
    t = isscalar(x) or len(x)
    if (t > 1):
        f, n = frexp(t)
    else:
        f, n = frexp(abs(x))

    if (f == 0.5):
        n = n - 1
    return n


def discretize(fun, a, b, tol=0.005, n=5, method='linear'):
    '''
    Automatic discretization of function

    Parameters
    ----------
    fun : callable
        function to discretize
    a,b : real scalars
        evaluation limits
    tol : real, scalar
        absoute error tolerance
    n : scalar integer
        number of values
    method : string
        defining method of gridding, options are 'linear' and 'adaptive'

    Returns
    -------
    x : discretized values
    y : fun(x)

    Example
    -------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> import pylab as plt
    >>> x,y = wm.discretize(np.cos, 0, np.pi)
    >>> xa,ya = wm.discretize(np.cos, 0, np.pi, method='adaptive')
    >>> t = plt.plot(x, y, xa, ya, 'r.')

    plt.show()
    >>> plt.close('all')

    '''
    if method.startswith('a'):
        return _discretize_adaptive(fun, a, b, tol, n)
    else:
        return _discretize_linear(fun, a, b, tol, n)


def _discretize_linear(fun, a, b, tol=0.005, n=5):
    '''
    Automatic discretization of function, linear gridding
    '''
    x = linspace(a, b, n)
    y = fun(x)

    err0 = inf
    err = 10000
    nmax = 2 ** 20
    while (err != err0 and err > tol and n < nmax):
        err0 = err
        x0 = x
        y0 = y
        n = 2 * (n - 1) + 1
        x = linspace(a, b, n)
        y = fun(x)
        y00 = interp(x, x0, y0)
        err = 0.5 * amax(abs((y00 - y) / (abs(y00 + y) + _TINY)))
    return x, y


def _discretize_adaptive(fun, a, b, tol=0.005, n=5):
    '''
    Automatic discretization of function, adaptive gridding.
    '''
    n += (mod(n, 2) == 0)  # make sure n is odd
    x = linspace(a, b, n)
    fx = fun(x)

    n2 = (n - 1) / 2
    erri = hstack((zeros((n2, 1)), ones((n2, 1)))).ravel()
    err = erri.max()
    err0 = inf
    # while (err != err0 and err > tol and n < nmax):
    for j in range(50):
        if err != err0 and np.any(erri > tol):
            err0 = err
            # find top errors

            I, = where(erri > tol)
            # double the sample rate in intervals with the most error
            y = (vstack(((x[I] + x[I - 1]) / 2,
                         (x[I + 1] + x[I]) / 2)).T).ravel()
            fy = fun(y)

            fy0 = interp(y, x, fx)
            erri = 0.5 * (abs((fy0 - fy) / (abs(fy0 + fy) + _TINY)))

            err = erri.max()

            x = hstack((x, y))

            I = x.argsort()
            x = x[I]
            erri = hstack((zeros(len(fx)), erri))[I]
            fx = hstack((fx, fy))[I]

        else:
            break
    else:
        warnings.warn('Recursion level limit reached j=%d' % j)

    return x, fx


def polar2cart(theta, rho, z=None):
    '''
    Transform polar coordinates into 2D cartesian coordinates.

    Returns
    -------
    x, y : array-like
        Cartesian coordinates, x = rho*cos(theta), y = rho*sin(theta)

    See also
    --------
    cart2polar
    '''
    x, y = rho * cos(theta), rho * sin(theta)
    if z is None:
        return x, y
    else:
        return x, y, z
pol2cart = polar2cart


def cart2polar(x, y, z=None):
    ''' Transform 2D cartesian coordinates into polar coordinates.

    Returns
    -------
    theta : array-like
        radial angle, arctan2(y,x)
    rho : array-like
        radial distance, sqrt(x**2+y**2)

    See also
    --------
    polar2cart
    '''
    t, r = arctan2(y, x), hypot(x, y)
    if z is None:
        return t, r
    else:
        return t, r, z
cart2pol = cart2polar


def ndgrid(*args, **kwargs):
    """
    Same as calling meshgrid with indexing='ij' (see meshgrid for
    documentation).
    """
    kwargs['indexing'] = 'ij'
    return meshgrid(*args, ** kwargs)


def trangood(x, f, min_n=None, min_x=None, max_x=None, max_n=inf):
    """
    Make sure transformation is efficient.

    Parameters
    ------------
    x, f : array_like
        input transform function, (x,f(x)).
    min_n : scalar, int
        minimum number of points in the good transform.
               (Default  x.shape[0])
    min_x : scalar, real
        minimum x value to transform. (Default  min(x))
    max_x : scalar, real
        maximum x value to transform. (Default  max(x))
    max_n : scalar, int
        maximum number of points in the good transform
              (default inf)
    Returns
    -------
    x, f : array_like
        the good transform function.

    TRANGOOD interpolates f linearly  and optionally
    extrapolate it linearly outside the range of x
    with X uniformly spaced.

    See also
    ---------
    tranproc,
    numpy.interp
    """
    xo, fo = atleast_1d(x, f)

    if (xo.ndim != 1):
        raise ValueError('x must be a vector.')
    if (fo.ndim != 1):
        raise ValueError('f  must be a vector.')

    i = xo.argsort()
    xo, fo = xo[i], fo[i]
    del i
    dx = diff(xo)
    if (any(dx <= 0)):
        raise ValueError('Duplicate x-values not allowed.')

    nf = fo.shape[0]

    if max_x is None:
        max_x = xo[-1]
    if min_x is None:
        min_x = xo[0]
    if min_n is None:
        min_n = nf
    if (min_n < 2):
        min_n = 2
    if (max_n < 2):
        max_n = 2

    ddx = diff(dx)
    xn = xo[-1]
    x0 = xo[0]
    L = float(xn - x0)
    if ((nf < min_n) or (max_n < nf) or any(abs(ddx) > 10 * _EPS * (L))):
        # pab 07.01.2001: Always choose the stepsize df so that
        # it is an exactly representable number.
        # This is important when calculating numerical derivatives and is
        # accomplished by the following.
        dx = L / (min(min_n, max_n) - 1)
        dx = (dx + 2.) - 2.
        xi = arange(x0, xn + dx / 2., dx)
        # New call pab 11.11.2000: This is much quicker
        fo = interp(xi, xo, fo)
        xo = xi

    # x is now uniformly spaced
    dx = xo[1] - xo[0]

    # Extrapolate linearly outside the range of ff
    if (min_x < xo[0]):
        x1 = dx * arange(floor((min_x - xo[0]) / dx), -2)
        f2 = fo[0] + x1 * (fo[1] - fo[0]) / (xo[1] - xo[0])
        fo = hstack((f2, fo))
        xo = hstack((x1 + xo[0], xo))

    if (max_x > xo[-1]):
        x1 = dx * arange(1, ceil((max_x - xo[-1]) / dx) + 1)
        f2 = f[-1] + x1 * (f[-1] - f[-2]) / (xo[-1] - xo[-2])
        fo = hstack((fo, f2))
        xo = hstack((xo, x1 + xo[-1]))

    return xo, fo


def tranproc(x, f, x0, *xi):
    """
    Transforms process X and up to four derivatives
          using the transformation f.

    Parameters
    ----------
    x,f : array-like
        [x,f(x)], transform function, y = f(x).
    x0, x1,...,xn : vectors
        where xi is the i'th time derivative of x0. 0<=N<=4.

    Returns
    -------
    y0, y1,...,yn : vectors
        where yi is the i'th time derivative of y0 = f(x0).

    By the basic rules of derivation:
    Y1 = f'(X0)*X1
    Y2 = f''(X0)*X1^2 + f'(X0)*X2
    Y3 = f'''(X0)*X1^3 + f'(X0)*X3 + 3*f''(X0)*X1*X2
    Y4 = f''''(X0)*X1^4 + f'(X0)*X4 + 6*f'''(X0)*X1^2*X2
      + f''(X0)*(3*X2^2 + 4*X1*X3)

    The derivation of f is performed numerically with a central difference
    method with linear extrapolation towards the beginning and end of f,
    respectively.

    Example
    --------
    Derivative of g and the transformed Gaussian model.
    >>> import pylab as plt
    >>> import wafo.misc as wm
    >>> import wafo.transform.models as wtm
    >>> tr = wtm.TrHermite()
    >>> x = linspace(-5,5,501)
    >>> g = tr(x)
    >>> gder = wm.tranproc(x, g, x, ones(g.shape[0]))
    >>> h = plt.plot(x, g, x, gder[1])

    plt.plot(x,pdfnorm(g)*gder[1],x,pdfnorm(x))
    plt.legend('Transformed model','Gaussian model')

    >>> plt.close('all')

    See also
    --------
    trangood.
    """
    def _default_step(xo, N):
        hn = xo[1] - xo[0]
        if hn ** N < sqrt(_EPS):
            msg = ('Numerical problems may occur for the derivatives in ' +
                   'tranproc.\n' +
                   'The sampling of the transformation may be too small.')
            warnings.warn(msg)
        return hn

    def _diff(xo, fo, x0, N):
        hn = _default_step(xo, N)
        # Transform X with the derivatives of  f.
        fder = vstack((xo, fo))
        fxder = zeros((N, x0.size))
        for k in range(N):  # Derivation of f(x) using a difference method.
            n = fder.shape[-1]
            fder = vstack([(fder[0, 0:n - 1] + fder[0, 1:n]) / 2,
                           diff(fder[1, :]) / hn])
            fxder[k] = tranproc(fder[0], fder[1], x0)
        return fxder

    def _der_1(fxder, xi):
        """First time derivative of y: y1 = f'(x)*x1"""
        return fxder[0] * xi[0]

    def _der_2(fxder, xi):
        """Second time derivative of y: y2 = f''(x)*x1.^2+f'(x)*x2"""
        return fxder[1] * xi[0] ** 2. + fxder[0] * xi[1]

    def _der_3(fxder, xi):
        """Third time derivative of y:
        y3 = f'''(x)*x1.^3+f'(x)*x3 +3*f''(x)*x1*x2
        """
        return (fxder[2] * xi[0] ** 3 + fxder[0] * xi[2] +
                3 * fxder[1] * xi[0] * xi[1])

    def _der_4(fxder, xi):
        """Fourth time derivative of y:
            y4 = f''''(x)*x1.^4+f'(x)*x4 +
                 6*f'''(x)*x1^2*x2+f''(x)*(3*x2^2+4x1*x3)
        """
        return (fxder[3] * xi[0] ** 4. + fxder[0] * xi[3] +
                6. * fxder[2] * xi[0] ** 2. * xi[1] +
                fxder[1] * (3. * xi[1] ** 2. + 4. * xi[0] * xi[1]))

    xo, fo, x0 = atleast_1d(x, f, x0)
    xi = atleast_1d(*xi)
    if not isinstance(xi, list):
        xi = [xi, ]
    N = len(xi)  # N = number of derivatives
    nmax = ceil((xo.ptp()) * 10 ** (7. / max(N, 1)))
    xo, fo = trangood(xo, fo, min_x=min(x0), max_x=max(x0), max_n=nmax)

    n = f.shape[0]
    xu = (n - 1) * (x0 - xo[0]) / (xo[-1] - xo[0])

    fi = asarray(floor(xu), dtype=int)
    fi = where(fi == n - 1, fi - 1, fi)

    xu = xu - fi
    y0 = fo[fi] + (fo[fi + 1] - fo[fi]) * xu

    y = y0
    if N > 4:
        warnings.warn('Transformation of derivatives of order>4 is ' +
                      'not supported.')
        N = 4
    if N > 0:
        y = [y0]
        fxder = _diff(xo, fo, x0, N)
        # Calculate the transforms of the derivatives of X.
        dfuns = [_der_1, _der_2, _der_3, _der_4]
        for dfun in dfuns[:N]:
            y.append(dfun(fxder, xi))

    return y


def good_bins(data=None, range=None, num_bins=None,  # @ReservedAssignment
              num_data=None, odd=False, loose=True):
    ''' Return good bins for histogram

    Parameters
    ----------
    data : array-like
        the data
    range : (float, float)
        minimum and maximum range of bins (default data.min(), data.max())
    num_bins : scalar integer
        approximate number of bins wanted
        (default depending on num_data=len(data))
    odd : bool
        placement of bins (0 or 1) (default 0)
    loose : bool
        if True add extra space to min and max
        if False the bins are made tight to the min and max

    Example
    -------
    >>> import wafo.misc as wm
    >>> wm.good_bins(range=(0,5), num_bins=6)
    array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.])
    >>> wm.good_bins(range=(0,5), num_bins=6, loose=False)
    array([ 0.,  1.,  2.,  3.,  4.,  5.])
    >>> wm.good_bins(range=(0,5), num_bins=6, odd=True)
    array([-1.5, -0.5,  0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5])
    >>> wm.good_bins(range=(0,5), num_bins=6, odd=True, loose=False)
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5,  5.5])
    '''
    def _default_range(range_, x):
        return range_ if range_ else (x.min(), x.max())

    def _default_bins(num_bins, x):
        if num_bins is None:
            num_bins = np.ceil(4 * np.sqrt(np.sqrt(len(x))))
        return num_bins

    def _default_step(mn, mx, num_bins):
        d = float(mx - mn) / num_bins * 2
        e = np.floor(np.log(d) / np.log(10))
        m = np.clip(np.floor(d / 10 ** e), a_min=0, a_max=5)
        if 2 < m < 5:
            m = 2
        return m * 10 ** e

    if data is not None:
        x = np.atleast_1d(data)

    mn, mx = _default_range(range, x)
    num_bins = _default_bins(num_bins, x)
    d = _default_step(mn, mx, num_bins)
    mn = (np.floor(mn / d) - loose) * d - odd * d / 2
    mx = (np.ceil(mx / d) + loose) * d + odd * d / 2
    limits = np.arange(mn, mx + d / 2, d)
    return limits


def plot_histgrm(data, bins=None, range=None,  # @ReservedAssignment
                 normed=False, weights=None, lintype='b-'):
    '''
    Plot histogram

    Parameters
    -----------
    data : array-like
        the data
    bins : int or sequence of scalars, optional
        If an int, it defines the number of equal-width
        bins in the given range (4 * sqrt(sqrt(len(data)), by default).
        If a sequence, it defines the bin edges, including the
        rightmost edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(data.min(), data.max())``.  Values outside the range are
        ignored.
    normed : bool, optional
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability *density* function
        at the bin, normalized such that the *integral* over the range is 1.
    weights : array_like, optional
        An array of weights, of the same shape as `data`.  Each value in `data`
        only contributes its associated weight towards the bin count
        (instead of 1).  If `normed` is True, the weights are normalized,
        so that the integral of the density over the range remains 1
    lintype : specify color and lintype, see PLOT for possibilities.

    Returns
    -------
    h : list
        of plot-objects

    Example
    -------
    >>> import pylab as plt
    >>> import wafo.misc as wm
    >>> import wafo.stats as ws
    >>> R = ws.weibull_min.rvs(2,loc=0,scale=2, size=100)

    >>> h0 = wm.plot_histgrm(R, 20, normed=True)
    >>> x = np.linspace(-3,16,200)
    >>> h1 = plt.plot(x,ws.weibull_min.pdf(x,2,0,2),'r')
    >>> plt.close('all')

    See also
    --------
    wafo.misc.good_bins
    numpy.histogram
    '''

    x = np.atleast_1d(data)
    if bins is None:
        bins = np.ceil(4 * np.sqrt(np.sqrt(len(x))))

    bin_, limits = np.histogram(
        data, bins=bins, normed=normed, weights=weights)
    limits.shape = (-1, 1)
    xx = limits.repeat(3, axis=1)
    xx.shape = (-1,)
    xx = xx[1:-1]
    bin_.shape = (-1, 1)
    yy = bin_.repeat(3, axis=1)
    # yy[0,0] = 0.0 # pdf
    yy[:, 0] = 0.0  # histogram
    yy.shape = (-1,)
    yy = np.hstack((yy, 0.0))
    return plotbackend.plot(xx, yy, lintype, limits, limits * 0)


def num2pistr(x, n=3, numerator_max=10, denominator_max=10):
    '''
    Convert a scalar to a text string in fractions of pi
        if the numerator is less than 10 and not equal 0
               and if the denominator is less than 10.

    Parameters
    ----------
    x   = a scalar
    n   = maximum digits of precision. (default 3)

    Returns
    -------
    xtxt = a text string in fractions of pi

    Example
    -------
    >>> import wafo.misc as wm
    >>> wm.num2pistr(np.pi*3/4)=='3\\pi/4'
    True
    '''
    def _denominator_text(den):
        return '' if abs(den) == 1 else '/%d' % den

    def _numerator_text(num):
        if abs(num) == 1:
            return '-' if num == -1 else ''
        return '%d' % num
    frac = fractions.Fraction.from_float(x / pi).limit_denominator(int(1e+13))
    num, den = frac.numerator, frac.denominator
    if (den < denominator_max) and (num < numerator_max) and (num != 0):
        return _numerator_text(num) + r'\pi' + _denominator_text(den)
    fmt = '%0.' + '%dg' % n
    return fmt % x


def fourier(data, t=None, period=None, m=None, n=None, method='trapz'):
    '''
    Returns Fourier coefficients.

    Parameters
    ----------
    data : array-like
        vector or matrix of row vectors with data points shape p x n.
    t : array-like
        vector with n values indexed from 1 to N.
    period : real scalar, (default T = t[-1]-t[0])
        primitive period of signal, i.e., smallest period.
    m : scalar integer
        defines no of harmonics desired (default M = N)
    n : scalar integer
        no of data points (default len(t))
    method : string
        integration method used

    Returns
    -------
    a,b  = Fourier coefficients size m x p

    FOURIER finds the coefficients for a Fourier series representation
    of the signal x(t) (given in digital form).  It is assumed the signal
    is periodic over T.  N is the number of data points, and M-1 is the
    number of coefficients.

    The signal can be estimated by using M-1 harmonics by:
                        M-1
     x[i] = 0.5*a[0] + sum (a[n]*c[n,i] + b[n]*s[n,i])
                       n=1
    where
       c[n,i] = cos(2*pi*(n-1)*t[i]/T)
       s[n,i] = sin(2*pi*(n-1)*t[i]/T)

    Note that a[0] is the "dc value".
    Remaining values are a[1], a[2], ... , a[M-1].

    Example
    -------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> T = 2*np.pi
    >>> t = np.linspace(0,4*T)
    >>> x = np.sin(t)
    >>> a, b = wm.fourier(x, t, T=T, m=5)
    >>> np.abs(a.ravel())<1e-12
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(b.ravel()-np.array([ 0.,  4.,  0.,  0.,  0.]))<1e-12
    array([ True,  True,  True,  True,  True], dtype=bool)

    See also
    --------
    fft
    '''
    x = np.atleast_2d(data)
    p, n = x.shape
    t = np.arange(n) if t is None else np.atleast_1d(t)

    n = len(t) if n is None else n
    m = n if m is None else m
    period = t[-1] - t[0] if period is None else period
    intfun = trapz if method.startswith('trapz') else simps

    # Define the vectors for computing the Fourier coefficients
    t.shape = (1, -1)
    a = zeros((m, p))
    b = zeros((m, p))
    a[0] = intfun(x, t, axis=-1)

    # Compute M-1 more coefficients
    tmp = 2 * pi * t / period
    for i in range(1, m):
        a[i] = intfun(x * cos(i * tmp), t, axis=-1)
        b[i] = intfun(x * sin(i * tmp), t, axis=-1)

    a = a / pi
    b = b / pi

    # Alternative:  faster for large M, but gives different results than above.
#    nper = diff(t([1 end]))/T; %No of periods given
#    if nper == round(nper):
#        N1 = n/nper
#    else:
#        N1 = n
#
#
#
# Fourier coefficients by fft
#    Fcof1 = 2*ifft(x(1:N1,:),[],1);
#    Pcor = [1; exp(sqrt(-1)*(1:M-1).'*t(1))] # correction term to get
#                                             # the correct integration limits
#    Fcof = Fcof1(1:M,:).*Pcor(:,ones(1,P));
#    a = real(Fcof(1:M,:));
#    b = imag(Fcof(1:M,:));

    return a, b


def real_main0():
    x = np.arange(10000)
    y = np.arange(100).reshape(-1, 1)
    np.broadcast_arrays(x, y, x, x, x, x, x, x, x, x)


def real_main():
    x = np.arange(100000)
    y = np.arange(100).reshape(-1, 1)
    common_shape(x, y, x, x, x, x, x, x, x, x)


def profile_main1():
    # This is the main function for profiling
    # We've renamed our original main() above to real_main()
    import cProfile
    import pstats
    prof = cProfile.Profile()
    prof = prof.runctx("real_main()", globals(), locals())
    print("<pre>")
    stats = pstats.Stats(prof)
    stats.sort_stats("time")  # Or cumulative
    stats.print_stats(80)  # 80 = how many to print
    # The rest is optional.
    # stats.print_callees()
    # stats.print_callers()
    print("</pre>")


main = profile_main1


def test_docstrings():
    # np.set_printoptions(precision=6)
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == "__main__":
    test_docstrings()
