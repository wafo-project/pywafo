|wafo_logo|
==========================================
Wave Analysis for Fatigue and Oceanography
==========================================

|pkg_img| |tests_img| |docs_img| |health_img| |coverage_img| |versions_img| |depsy_img|


Description
===========

WAFO is a toolbox Python routines for statistical analysis and simulation of
random waves and random loads. WAFO is freely redistributable software, see WAFO
icence, cf. the GNU General Public License (GPL) and contain tools for:
        
Fatigue Analysis
----------------

- Fatigue life prediction for random loads
- Theoretical density of rainflow cycles

Sea modelling
-------------

- Simulation of linear and non-linear Gaussian waves
- Estimation of seamodels (spectrums)
- Joint wave height, wave steepness, wave period distributions

Statistics
------------

- Extreme value analysis
- Kernel density estimation
- Hidden markov models

Classes
-------

* TimeSeries:
    Data analysis of time series. Example: extraction of turning points,
    estimation of spectrum and covariance function. Estimation transformation
    used in transformed Gaussian model.

* CovData:
    Computation of spectral functions, linear and non-linear time series
    simulation.

* SpecData:
    Computation of spectral moments and covariance functions, linear and
    non-linear time series simulation. Ex: common spectra implemented, 
    directional spectra, bandwidth measures, exact distributions for wave
    characteristics.

* CyclePairs:
    Cycle counting, discretization, and crossings, calculation of damage.
    Simulation of discrete Markov chains, switching Markov chains,
    harmonic oscillator. Ex:  Rainflow cycles and matrix, discretization of
    loads. Damage of a rainflow count or matrix, damage matrix, S-N plot.


Subpackages
-----------

* TRANSFORM
    Modelling with linear or transformed Gaussian waves.
* STATS
    Statistical tools and extreme-value distributions. Ex: generation of random
    numbers, estimation of parameters, evaluation of pdf and cdf
* KDETOOLS
    Kernel-density estimation.
* MISC
    Miscellaneous routines.
* DOCS
    Documentation of toolbox, definitions. An overview is given in the routine
    wafomenu.
* DATA
    Measurements from marine applications.

WAFO homepage: <http://www.maths.lth.se/matstat/wafo/>
On the WAFO home page you will find:
- The WAFO Tutorial
- List of publications related to WAFO.

Installation
============

WAFO contains some Fortran and C extensions that require a properly configured
compiler and NumPy/f2py.

Create a binary wheel package and place it in the dist folder as follows::

    python setup.py bdist_wheel -d dist

And install the wheel package with::

    pip install dist/wafo-X.Y.Z+abcd123-os_platform.whl


Unit tests
==========

To test if the toolbox is working paste the following in an interactive
python session::

   import wafo as wf
   wf.test(coverage=True, doctests=True)


Note
====

This project has been set up using PyScaffold 2.4.2. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.

.. |wafo_logo| image:: https://github.com/wafo-project/pywafo/blob/master/wafo/data/wafoLogoNewWithoutBorder.png
    :target: https://github.com/wafo-project/pywafo


.. |pkg_img| image:: https://badge.fury.io/py/wafo.png
    :target: https://pypi.python.org/pypi/wafo/

.. |tests_img| image:: https://travis-ci.org/wafo-project/pywafo.svg?branch=master
    :target: https://travis-ci.org/wafo-project/pywafo

.. |docs_img| image:: https://readthedocs.org/projects/pip/badge/?version=latest
    :target: http://pywafo.readthedocs.org/en/latest/

.. |health_img| image:: https://landscape.io/github/wafo-project/pywafo/master/landscape.svg?style=flat
   :target: https://landscape.io/github/wafo-project/pywafo/master
   :alt: Code Health

.. |coverage_img| image:: https://coveralls.io/repos/wafo-project/pywafo/badge.svg?branch=master
   :target: https://coveralls.io/github/wafo-project/pywafo?branch=master

.. |versions_img| image:: https://img.shields.io/pypi/pyversions/wafo.svg
   :target: https://github.com/wafo-project/pywafo


.. |depsy_img| image:: http://depsy.org/api/package/pypi/wafo/badge.svg
   :target: http://depsy.org/package/python/wafo