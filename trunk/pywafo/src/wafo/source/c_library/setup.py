'''
python setup.py build_src build_ext --inplace

See also http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW
'''
# File setup.py


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('', parent_package, top_path)

    config.add_extension('c_library',
                         sources=['c_library.pyf', 'c_functions.c'])
    return config
if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
