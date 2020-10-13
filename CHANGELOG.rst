=========
Changelog
=========

Version 0.4.0 13 October 2020
=============================

Mattias Josefsson (5):
      * Update _discrete_distns.py
      * Fix import logsumexp for scipy>=1.0.0
      * Update polynomial.py
      * Fix import pade for scipy>=1.0.0
      * Include all files under wafo/data in installation

Per A Brodtkorb (144):
      * Updated CHANGELOG.rst and prepare for relase v0.4.0
      * Updated import of cumtrapz from scipy.integrate
      * Re-added included package_data=... in setup.py
      * Set scipy>=1.1 in requirements.txt
      * Removed THANKS.txt from MANIFEST.in Streamlined setup.cfg and setup.py
      * Updated coverage call in .travis.yml
      * Updated .travis.yml: added test for python 3.8 Added COPYRIGHTS
      * Added license.py Fixed a bug in build_package.py pep8
      * Updated build_package.py
      * Updated weibull_min._fitstart with a better start function.
      * Ongoing work to make Profile, ProfileProbability and ProfileQuantile
         more robust.
      * Made chapter5.py work again. Fixed a bug in _fitstart in
         _distn_infrastructure.py that did not allow loc, scale given as part of
         *args. Made FitDistribution and the ProfileXXX classes more robust.
      * Added returnperiod2sf and sf2returnperiod Simplied valarray with a call
         to numpy.full Updated the chapter5 tutorials.
      * Improved profiling methods in wafo.stats.estimation.py:  * Silenced all
         divide by zero and invalid value warnings.  * Replace
         scipy.optimize.fmin with scipy.optimize.minimize  * Fixed a bug due to
         changed behaviour of numpy.linspace:   - Previously the call
         np.linspace([0],[1], n)  returned a vector of length n, but now returns
         a n x 2  array.  * Added _approx_p_min_max and _p_min_max methods to
         ProfileProbability and ProfileQuantile to better optimize the profile
         loglikelihood.  * Updated tutorial scripts and made all the esitmation
         methods work again.
      * Updated the tutorial scripts as well as the path given to them in the
         info.py file. Replaced import of pylab with import of numpy and
         matplotlib in wafo.spectrum/models.py Replaced deprecated call
         "np.histogram(self.data, bins=limits, normed=True)" with
         "np.histogram(self.data, bins=limits, density=True)"
      * Fixed some issues with fortran source code:   * Removed unused labels
         and variables.   * Replaced tabs with spaces. Fix for the issue that
         Setuptools sdist doesn't include all the files in the extension sources
         list.
      * Added sonar-python and bandit plugins to .codeclimate.yml
      * Updated thresholds in .codeclimate.yml
      * Another attempt to get .codeclimate.yml right.
      * Updated .codeclimat.yml
      * Removed python 2.7 classifier in setup.cfg
      * Updated path to the wafo-logo in README.rst and info.py
      * Updated failing test_levelcrossings_extrapolate in test_objects.py
      * Patched fitstart to some scipy.stats distributions in
         wafo.stats.distributions.py Added test of maximum product of spacing
         estimation method to test_fit.py Also accept the fit as good if the
         pvalue of the fit is larger than 0.05 in test_fit.py
      * Added better _fitstart to the truncrayleigh distribution.
      * Readded wafo.stats.tests to the tests...
      * Simplified the wafo.stats.tests Updated path to the WAFO-logo in
         README.rst Replaced wafo.test function with a call to pytest.main
      * Removed dependence on pyscaffold and changed the layout of WAFO:   
        - Moved files from ./wafo to ./src/wafo in order to insulate the package
          from the setup.py folder.   
      * Updated setup.cfg and setup.py to support the /src layout.   
      * Moved sources from ./src/wafo to ./ and updated setup.py accordingly.   
      * Removed dependence on pyscaffold in setup.py
      * Added lfind to misc.py 
      * Replaced call to valarray with np.full 
      * Refactored code from detrendma into a new function moving_average. 
      * Made doctests more robust.
      * Added wavemodels.py
      * Added extra tests for dispersion_relation.py and qtf in spectrum/core.py.  
      * Deleted powerpoint.py
      * Fixed a bug in findextrema
      * Added possibility to change freqtype after creation of SpecData1D object
      * Made testgaussian test less strict.
      * Moved import of dea3 directly to integrate. Added import collections
      * Deleted MSO.py and MSPPT.py and updated doctest in misc.py
      * Simplified _genextreme_link
      * Moved mctp2rfc and mc2rfc to markov.py, added CycleMatrix to objects.py
      * Remove find_first_cross
      * Added numba_misc.py
      * Added markov.py
      * Drop support for python 3.4
      * Added test on python 3.6 and 3.7 + updated mvn/mvndst.f
      * Added .pylintrc
      * Fixed doctests
      * Splitted tests
      * Fixed interpolate.py
      * Fixed mvndst.f
      * Replaced dict().setdefault with use of defaultdict.
      * Fixed a TypeError: 'numpy.float64' object cannot be interpreted as an index
      * Fixed TypeError: "TypeError('only integer scalar arrays can be
         converted to a scalar index',)" exception error in findcross.
      * Replace definition of test_docstrings with an import from wafo.testing.
      * Moved sg_filter.py into a sg_filter package
      * Removed plt.show from doctest
      * Moved code from sg_filter.py to demo_sg.py
      * Added tests for dst, dstn idst and idstn
      * Refactored magic
      * Updated doctests
      * Added import of statsmodels to .travis.yml
      * Added doctest for kdetools/demo.py. 
      * Removed obsolete options from wafo.containers.py

Version 0.3.1 16 January 2017 
=============================
Bo Zhang @NAOC (1):
      * turn nd into int

Per A Brodtkorb (109):
      * Refactored _reduce_func
      * Moved demo from kdetools.py to demo.py. Increased test coverage.
      * Simplified wafo.transform/models.py
      * Added inc property to _KDE class
      * Refactored franke function + added test error_estimate from padua_fit
      * Moved common data into data.py
      * Altered demo
      * Updated testdata      
      * Replaced instance methods with static methods + lower case method names.
      * Moved test_kdetools.py to wafo.kdetools.tests
      * Removed obsolete code
      * Simplified TKDE._eval_grid_fast
      * Simplified _eval_grid_fast
      * Refactored duplicated code into a _estimate_psi function in kdetools.py
      * Simplified fixed_point
      * Simplified Kernel
      * Simplified _hmns_scale
      * fixed a bug in Kernel.name and simplified glevels and made it general.
      * Reduced the complexity of accum
      * Simplified gridcount
      * Fixed failing doctests
      * Refactored kdetools.py into a subpackage. 
      * Added dst, idst, dstn and idstn
      * Fixed a bug in detrendma and mctp2tc
      * Simplified out of bound check in mctp2tc
      * Reduced cyclomatic complexity in mctp2tc
      * Refactrored poly2str and poly2hstr
      * Avoid `None` as a redundant second argument to `dict.get()
      * added option for plotting confidence interval in plotesf
      * Added doctest to check_random_state and added ci_quantile and ci_sf to FitDistribution
      * Added numba_misc.py
      * Replace dict.keys() with list(dict) in order to make it python 3 compatible.
      * Removed plotting
      * Fixed a bug in TransformEstimator
      * Updated test_integrate_oscillating.py
      * Added check for TypeError
      * Simplified getshipchar
      * Added test_containers.py + fixed some codestyle issues
      * Replaced call to PiecewisePolynomial with BPoly
      * Added tutorial_init.py and moved rainflow_example to tutorial scripts
      * Simplified bandwith and dof calculation
      * Updated stats
      * Fixed a failing doctest. Reorganized _penalized_nnlf
      * Moved all links to estimation.py
      * Updated wafo.stats
      * Added test to lazywhere + pep8 on polynomial
      * Replaced string interpolation operator with 'format()'
      * Reduced complexity of _compute_cov
      * Updated call signature to Limit
      * Added pip install funcsigs
      * Ongoing work to simplify estimation
      * Simplified _nlogps
      * Simplified PlotData in containers.py
      * Simplified TKDE class
      * Refactored code in _get_g
      * Refactored code into _estimate_psi function
      * replace `not ... is` with `is not` + renamed misspelled test
      * Fixed a bug on comput_cov
      * Fixed a bug in ProfileQuantile and ProfileProbability
      * Try to silence optimizer.
      * Added rainflow_example.py
      * Added Getting started section to readme.
      * Fixed a bug in histogram
      * Added link functions to genextreme, exponweib
      * Fixed cmat2nt for kind=1
      * Added doctests to stirlerr
      * added nt2cmat and cmat2nt
      * updated spectrum.core
      * added import of pil in .travis, removed plots
      * Fixed a bug in kdetools
      * Added MSO.py and MSPPT.py to collect_ignore
      * Updated .travis.yml and setup.cfg
      * Added newest numdifftools
      * removed tox.ini
      * added integrate ocscilating + added doctest to .travis
      * Fixed failing test for piecewise
      * added --doctest-modules
      * Deleted namedtuple, fixed bug in piecewise
      * Added image to code-climate

Version 0.2.1 May 22 2016
=========================

Per A Brodtkorb (47):                                                                                                         
     * Removed tabs from c_functions.c refaactored quadgr
     * added test_sg_filter.py                                                                                               
     * pep8
     * Simplified delete_text_object
     * updated SmoothNd
     * Refactored smoothn into SmoothNd and _Filter classes
     * updated Kalman and HampelFilter
     * Simplified HampelFilter
     * Removed unused code and added test for shiftdim
     * Removed duplicated dea3 from integrate and misc.py import from
         numdifftools.extrapolate.dea3 instead pepified
     * Simplified common_shape
     * refactored findrfc
     * Deleted misc.hypgf function 
     * Refactored:  misc.findoutliers objects.TimeSeries.wave_periods
     * Added files: .codeclimate.yml test_bitwise.py test_dct_pack.py
     * Added wafo-logo to README.rst
     * Made test_integrate.py more robust
     * Replaced iteritems with a python 3 compatible iterable dict items.
     * Made sure arrays used as indices is of integer type
     * Made code python 3 compatible: Replaced round with numpy.round
     * made code python 3 compatible: Replaced xrange with range and map with list comprehension
     * Added from __future__ absolute_import
     * Deleted obsolete magic.py
     * Deleted wafodata.py
     * Made print statements python 3 compatible
     * Restored c_functions.c
     * removed test_numpy_utils.py + pepified test_trdata + disabled plot in test_specdata1d.py
     * Deleted obsolete test folder, numpy_utils.py + tests more robust
     * Updated tox.ini
     * Small refactoring FitDistribution
     * Added _util.py + fixed a bug in test_fit.py
     * Added numpy_utils.py
     * Added padua.py
     * Replaced sub2index and index2sub with calls to np.ravel_multi_index and np.unravel_index, respectively.
     * Added chebfit_dct chebvandernd chebfitnd chebvalnd chebgridnd
     * Replaced dct with call to scipy.fftpack.dct
     * build extensions for windows
     * Deleted c_library.pyd
     * Renamed c_codes -> c_library
     * Made doctests more robust
     * compiled fortran/c-code for windows pep8 
     * Moved smoothn from kdetools to sg_filter.py 
     * Simplified sg_filter.py, dctpack.py and added autumn.gif
     * Simplified interpolations and made dea3 more robust
     * Deleted statsmodels
     * Added fix for genpareto.logpdf
     * added test_estimation.py and test_continuous_extra.py
     * Fixed more bugs in distributions.py
     * Updated from wafo.stats from scipy.stats
     * Updated tutorial ipython notebook scripts
     * Fixed a bug in dispersion_idx
     * Compiled on win7 64 bit
     * Refactored Profile
     * fix ticket 1131
     * Improved beta distribution
     * moved test/test_dispersion_relation.py to the wave_theory/test
     * Added magic.py
     * Renamed test_all.py to nose_all.py
     * Updated to most recent scipy.stats.distributions
     * vectorizing depth as well in w2k
     * Generalized magic
     * Added magic square
     * Added pychip.py
     * Updated kreg_demo3
     * refactored parts of kreg_demo2 into kreg_demo3 and _get_data
     * Added SavitzkyGolay class to sg_filter.py Refined confidence intervals
         in kreg_demo2 in kdetools.py
     * Better confidence interval in kreg_demo2
     * Added savitzky_golay savitzky_golay_piecewise sgolay2d Added evar Added
         some work in progress
     * Fixed a bug in KRegression
     * Small updates
     * Added fig.py Fixed a bug in RegLogit Added epcolor and tallibing to graphutil.py
     * Fixed some bugs in RegLogit (still bugs left)
     * Added improved Sheater-Jones plugin estimate of the smoothing parameter
     * Replaced dct and idct with a call to the ones in scipy.fftpack. 
     * Added n-dimensional  version dctn and idctn to dctpack.py
     * Added dctn and idctn
     * Added kernel regression
     * Made interpolation more general and faster in TKDE._eval_grid_fast
     * Fixed some bugs
     * Fixed some bugs in kdetools.py + added more tests in test/test_kdetools.py
     * Added alternative version of  scikits statsmodels
     * Updated distributions.py according to the latest updates in scipy.stats.distributions.py
     * Small extension to plot2d
     * Added mctp2rfc to misc.py Fixed a bug in qlevels and cltext
     * Started work on SpecData1D.to_mmt_pdf
     * Fixed bugs in cov2mmpdfreg_intfc.f
     * Successfully made an interface to mregmodule. It still remains to check that it is correct.
     * Translated matlab tran function into a TransferFunction class
     * added import of k2w from dispersion_relation.py
     * Updated help header
     * Added qlevels2 + possibility to calculate weighted percentile
     * Added percentile
     * Added more work to LevelCrossings.extrapolate (not finished yet)
     * Copied stineman_interp from pylab to interpolate.py and fixed the
        annoying dividing by zero warnings.
     * misc fixes
     * added fourier (not finished) added TurningPoints.rainflow_filter
         +translated some parts of chapter4.py
     * updated __all__ attributes in modules
     * Made a baseclass _KDE for KDE  and TKDE + updated tests
     * Added bitwise operators
     * Fixed a bug in kde.eval_grid_fast + updated tests
     * Added test_distributions.py updated test_estimation.py
     * Fixed Scipy-ticket #1131:  ppf for Lognormal fails on array-like 'loc' or 'scale'

david.verelst (7):
     * ignore import error for fig.py: depends on windows only libraries
     * updated builds for Linux 64bit
     * build commands Linux: use python or python2
     * References and BSD license for Nieslony's rainflow algorithm
     * Nieslony's ASTM rainflow counting algorithm. Partially integrated, no
        support for the CyclePairs object yet.
     * More robust way to determine f2py call in /sources/c_codes/build_all.py,
        etc scripts
     * build_all.py scripts in source now call to f2py2.6 on posix systems
         (this might give issues on other installation, for instance when it is
         f2py, f2yp2.7, etc). On nt (windows) it remains f2py.py. The general
         setup.py and build_all.py scripts now copies the .so compiled libraries
         when on posix platform, on nt (windows) these are the .pyd files

davidovitch (13):
      * remove trailing white spaces in README
      * added installation section in README
      * add *.mod to .gitignore
      * merge re-organisation of sources, pip installable setup.py, see issue
         #14
      * change library names: lib will added as prefix automagically by
         distutils
      * setup now correctly compiles the extensions and the fortran
         objects/modules it depends upon
      * Merge pull request #10 from ocefpaf/scipy
      * fix formatting of old readme to rst format
      * [WIP] add first iteration and incomplete packaging files generated with
         pyscaffold
      * move directory structure up, change root to: pywafo/src/* > wafo/*
      * remove old packaging related scripts and configs
      * removed some of the obsolete(?) project files Eclips/epydoc
      * cleaning up, remove binaries and compiled modules

ocefpaf (2):
      * Fixed SciPy lib imports.
      * fix encoding

per.andreas.brodtkorb (120):
      * Updated setup.cfg
      * Dropped support for python 3.3
      * Replaced tabs with spaces....
      * Simplified wafo.stats:  -Deleted obsolete files.  -Requires scipy v0.16
         -._distn_infrastructure.py monkeypatch
         scipy.stats._distn_infrastructure.py
      * updated .travis.yml moved some funtions from numpy_utils -> misc. pep8
      * added sudo gfortran again
      * Updated bagdes in README.rst
      * added .checkignore for quantifycode
      * Commented out installation of gfortran on travis
      * Removed space in numbers
      * Updated mvnprd.f
      * Alternative build of mvnprd extension
      * Commented out compilation of mvnprd extension
      * Updated compilation of mvnprdmod extension
      * Try alternative build for fortran extensions
      * Try to compile fortran extensions again
      * Fixed misspelled modulename
      * Changed doctests into unittests
      * Changed doctest for normndprb into a unittest
      * Changed doctest to unittest
      * Try compile mvn extenstion
      * Added codecov to .travis.ymls
      * Renamed test folders to "tests"
      * Disabled wafo.stats tests + small cosmetiq fixes
      * set base to python2.7 in tox.ini
      * Python3 support: Replaced print statements with print(...)
      * Add try except when importing the compiled extensions
      * updated .travis.yml
      * Removed deprecated import Updated test/test_padua.py
      * Renamed TestFunctions to ExampleFunctions in order to not confuse
         pytest.
      * Added matplotlib to requirements.txt
      * Added test folders to setup.cfg
      * Attempt to fix the coverage
      * added .coveragerc file
      * commented out fortran extension
      * commented out building of fortran extensions
      * added .landscape.yml file
      * added installation of numdifftools on travis
      * Added missing mvn/mvn.pyf
      * updated path to pypi
      * added image badges..
      * Added build step to .travis.yml
      * Updated README.rst in order to test travis-CI
      * Added configuration file.travis.yml for setting up continous integration
         tests.
      * Added CubicHermiteSpline, StinemanInterp, Pchip pchip_slopes, slopes2
      * Added eval_points and integrate to the WafoData class
      * updated kreg_demo2
      * Completed smoothn. Tested on 3 examples and works OK.
      * Added CI to kreg_demo2
      * Added tallibing
      * Updated _nnlf()
      * Added RegLogit to core.py (not finished)
      * Fixed: nan-propagation errors (ticket #835) stats.lognorm.pdf(0,s)
         reports nan (ticket #1471)
      * Resolved issue 6: mctp2rfc is now working for the example given
      * Fixed a bug in the extrapolate method of LevelCrossings class.
      * Added truncated rayleigh
      * Resolved issues 2, 3 and 4: Test failures in test/test_gaussian.py,
         test_misc.py and test_objects.py
      * Updated chapter scripts + small fixes elsewhere
      * Added TimeSeries.wave_parameters
      * Added wave_height_steepness method to TimeSeries class
      * Fixed a bug in LevelCrossings.trdata Added plotflag to Plotter_1d and
         WafoData. Otherwise small cosmetic fixes
      * Cosmetic fixes
      * Fixed a bug in nlogps when ties occuring in the data.
      * Fixed a bug in SpecData1D.tocovdata
      * Added functionality to TimeSeries.trdata
      * Added tutorial_scripts
      * bugfix in SpecData1D.sim_nl + updated doctests
      * Added SpecData2D.moment
      * added histgrm + small bugfixes
      * Updated test examples in kdetools.py + cosmetic fixes to the rest
      * Added WafoData output of KDE. Added kde_demo1 and kde_demo2
      * Added plotobject output to KDE class
      * updated kdetools.py
      * Added KDE.eval_grid and KDE.eval_grid_fast
      * Updated example in TKDE
      * Fixed bugs in KDE Added TKDE + tests
      * Added test for estimation.py
      * Updated kdetools.py (but still not working correctly) Small cosmetic
         fixes to other files	  

Version 0.1.2 Oct 13 2010
=========================       

Per.Andreas.Brodtkorb (22):
      * Fixed a bug in setup.py Added functions to __all__ variable in stats.core.py
      * Added hessian_nlogps for more robust estimation of the covariance.
      * Fixed bugs in: link method of frechet_r_gen class _reduce_func method of
         rv_continuous and FitDistribution classes _myprbfun method in Profile class
      * Mostly updated the documentation and validated the examples.

      * Added version generation to setup.py Simplified __init__.py files to
         avoid duplicate inclusion of wafo.
      * Added more kernels
   
      * Added cdfnorm2d and prbnorm2d + tests
      * Moved mvn.pyf and mvndst.f to source/mvn directory + added the build script for it
      * Added default plot_args and plot_args_children to WafoData
      * Added more test to test/test_misc.py and test/test_gaussian.py
      * Removed reference to ppimport
      * Deleted ppimport.py
      * Added tests for misc.py
      * Added build_all and test_all scripts
      * Added functions to stats.core.py -reslife -extremal_idx and improved estimation.py
      * Deleted stats.plotbackend.py and test_ppimport.py
      * Fixed a bug in findcross in c_functions.c Recompiled binaries for Windows xp 32bit
      * Updated distributions.py so it is in accordance with scipy.stats.distributions.py
      * Revised the setup script

Version 0.11 Jun 11, 2010
=========================	   
  * First release