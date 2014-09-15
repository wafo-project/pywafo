"""
WAFO
====
   WAFO is a toolbox Python routines for statistical analysis and simulation of
   random waves and random loads.
   WAFO is freely redistributable software, see WAFO licence, cf. the
   GNU General Public License (GPL) and contain tools for:

Fatigue Analysis
----------------
-Fatigue life prediction for random loads
-Theoretical density of rainflow cycles

Sea modelling
-------------
-Simulation of linear and non-linear Gaussian waves
-Estimation of seamodels (spectrums)
-Joint wave height, wave steepness, wave period distributions

Statistics
------------
-Extreme value analysis
-Kernel density estimation
-Hidden markov models

 WAFO consists of several subpackages and classes with short descriptions given
 below.

 Classes:
 TimeSeries  - Data analysis of time series. Example: extraction of
              turning points, estimation of spectrum and covariance function.
              Estimation transformation used in transformed Gaussian model.
 CovData     - Computation of spectral functions, linear
              and non-linear time series simulation.
 SpecData    - Computation of spectral moments and covariance functions, linear
              and non-linear time series simulation.
              Ex: common spectra implemented, directional spectra,
              bandwidth measures, exact distributions for wave characteristics.

 CyclePairs  - Cycle counting, discretization, and crossings, calculation of
              damage. Simulation of discrete Markov chains, switching Markov
              chains, harmonic oscillator. Ex:  Rainflow cycles and matrix,
              discretization of loads. Damage of a rainflow count or
              matrix, damage matrix, S-N plot.


Subpackages:
 TRANSFORM  - Modelling with linear or transformed Gaussian waves. Ex:
 STATS      - Statistical tools and extreme-value distributions.
              Ex: generation of random numbers, estimation of parameters,
              evaluation of pdf and cdf
 KDETOOLS   - Kernel-density estimation.
 MISC       - Miscellaneous routines.
 DOCS       - Documentation of toolbox, definitions. An overview is given
              in the routine wafomenu.
 DATA       - Measurements from marine applications.

 WAFO homepage: <http://www.maths.lth.se/matstat/wafo/>
 On the WAFO home page you will find:
  - The WAFO Tutorial
  - New versions of WAFO to download.
  - Reported bugs.
  - List of publications related to WAFO.
"""
