'''
python setup.py build_src build_ext --inplace

See also http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW
'''

# File setup.py


def compile_all():
    import os
    files = ['mvnprd', 'mvnprodcorrprb']
    compile1_format = 'gfortran -fPIC -c %s.f'
    for file_ in files:
        os.system(compile1_format % file_)
    file_objects = ['%s.o' % file_ for file_ in files]
    return file_objects


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    libs = compile_all()
    config = Configuration('', parent_package, top_path)

    config.add_extension('mvnprdmod',
                         libraries=libs,
                         sources=['mvnprd_interface.f'])
    return config
if __name__ == "__main__":

    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
