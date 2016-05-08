'''
Module extending the bitoperator capabilites of numpy
'''

from numpy import (bitwise_and, bitwise_or,
                   bitwise_not, binary_repr,  # @UnusedImport
                   bitwise_xor, where, arange)  # @UnusedImport
__all__ = ['bitwise_and', 'bitwise_or', 'bitwise_not', 'binary_repr',
           'bitwise_xor', 'getbit', 'setbit', 'getbits', 'setbits']


def getbit(i, bit):
    """
    Get bit at specified position

    Parameters
    ----------
    i : array-like of uints, longs
        value to
    bit : array-like of ints or longs
        bit position between 0 and the number of bits in the uint class.

    Examples
    --------
    >>> import numpy as np
    >>> binary_repr(13)
    '1101'
    >>> getbit(13,np.arange(3,-1,-1))
    array([1, 1, 0, 1])
    >>> getbit(5, np.r_[0:4])
    array([1, 0, 1, 0])
    """
    return bitwise_and(i, 1 << bit) >> bit


def getbits(i, numbits=8):
    """
    Returns bits of i in a list
    """
    return getbit(i, arange(0, numbits))


def setbit(i, bit, value=1):
    """
    Set bit at specified position

    Parameters
    ----------
    i : array-like of uints, longs
        value to
    bit : array-like of ints or longs
        bit position between 0 and the number of bits in the uint class.
    value : array-like of 0 or 1
        value to set the bit to.

    Examples
    --------
    Set bit fifth bit in the five bit binary binary representation of 9 (01001)
    yields 25 (11001)
    >>> setbit(9,4)
    array(25)
    """
    val1 = 1 << bit
    val0 = bitwise_not(val1)
    return where((value == 0) & (i == i) & (bit == bit), bitwise_and(i, val0),
                 bitwise_or(i, val1))


def setbits(bitlist):
    """
    Set bits of val to values in bitlist

    Example
    -------
    >>> setbits([1,1])
    3
    >>> setbits([1,0])
    1
    """
#    return bitlist[7]<<7 | bitlist[6]<<6 | bitlist[5]<<5 | bitlist[4]<<4 | \
#    bitlist[3]<<3 | bitlist[2]<<2 | bitlist[1]<<1 | bitlist[0]
    val = 0
    for i, j in enumerate(bitlist):
        val |= j << i
    return val


def test_docstrings():
    import doctest
    print('Testing docstrings in %s' % __file__)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

if __name__ == '__main__':
    test_docstrings()

#    t = set(np.arange(8),1,1)
#    t=get(0x84,np.arange(0,8))
#    t=getbyte(0x84)
#    t=get(0x84,[0, 1, 2, 3, 4, 5, 6, 7])
#    t=get(0x20, 6)
#    bit = [0 for i in range(8)]
#    bit[7]=1
#    t = setbits(bit)
#    print(hex(t))
