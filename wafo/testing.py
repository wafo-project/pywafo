'''
Created on 15. des. 2016

@author: pab
'''
import inspect


def test_docstrings(name=''):
    import doctest
    if not name:
        name = inspect.stack()[1][1]
    print('Testing docstrings in {}'.format(name))
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)

if __name__ == '__main__':
    pass
