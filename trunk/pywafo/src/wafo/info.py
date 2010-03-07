"""
WAFO
=====
   WAFO is a toolbox Python routines for statistical analysis and simulation of random waves and random loads. 
   WAFO is freely redistributable software, see WAFO licence, cf. the GNU General Public License (GPL) and
   contain tools for:

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

 WAFO consists of several modules with short descriptions below.
 The modules SPECTRUM, COVARIANCE, TRANSFORM, WAVEMODELS, and MULTIDIM are 
 mainly for oceanographic applications.
 The modules CYCLES, MARKOV, and DAMAGE are mainly for fatigue problems. 
 The contents file for each module is shown by typing 'help module-name' 
 Type 'help fatigue' for a presentation of all routines related to fatigue.

 The paths to the modules are initiated by the function 'initwafo'.

 ONEDIM     - Data analysis of time series. Example: extraction of 
              turning points, estimation of spectrum and covariance function. 
              Estimation transformation used in transformed Gaussian model.
 COVARIANCE - Computation of spectral functions, linear 
              and non-linear time series simulation. 
 SPECTRUM   - Computation of spectral moments and covariance functions, linear 
              and non-linear time series simulation. 
              Ex: common spectra implemented, directional spectra, 
              bandwidth measures, exact distributions for wave characteristics.
 TRANSFORM  - Modelling with linear or transformed Gaussian waves. Ex: 
              
 WAVEMODELS - Models for distributions of wave characteristics found in 
              the literature. Ex: parametric models for breaking 
              limited wave heights.
 MULTIDIM   - Multi-dimensional time series analysis.  (Under construction)
 CYCLES     - Cycle counting, discretization, and crossings, calculation of 
              damage. Simulation of discrete Markov chains, switching Markov
              chains, harmonic oscillator. Ex:  Rainflow cycles and matrix, 
              discretization of loads. Damage of a rainflow count or 
              matrix, damage matrix, S-N plot.
 MARKOV     - Routines for Markov loads, switching Markov loads, and 
              their connection to rainflow cycles.
 DAMAGE     - Calculation of damage. Ex: Damage of a rainflow count or 
              matrix, damage matrix, S-N plot.
 SIMTOOLS   - Simulation of random processes. Ex: spectral simulation, 
              simulation of discrete Markov chains, switching Markov
              chains, harmonic oscillator
 STATISTICS - Statistical tools and extreme-value distributions.
              Ex: generation of random numbers, estimation of parameters,
              evaluation of pdf and cdf
 KDETOOLS   - Kernel-density estimation.
 MISC       - Miscellaneous routines. Ex: numerical integration, smoothing 
              spline, binomial coefficient, water density.
 WDEMOS     - WAFO demos. 
 DOCS       - Documentation of toolbox, definitions. An overview is given 
              in the routine wafomenu. 
 DATA       - Measurements from marine applications.
 PAPERS     - Commands that generate figures in selected scientific 
              publications.
 SOURCE     - Fortran and C files. Information on compilation.
 EXEC       - Executable files (cf. SOURCE), pre-compiled for Solaris, 
              Alpha-Dec or Windows. 
 
 WAFO homepage: <http://www.maths.lth.se/matstat/wafo/>
 On the WAFO home page you will find:
  - The WAFO Tutorial
  - New versions of WAFO to download.
  - Reported bugs.
  - List of publications related to WAFO.
"""