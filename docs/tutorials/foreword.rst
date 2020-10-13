Foreword
========

Foreword to 2017 edition
------------------------

This Wafo tutorial 2017 has been successfully tested with Matlab 2017a
on Windows 10.

The tutorial for Wafo 2.5 appeared 2011, with routines tested on Matlab
2010b. Since then, many users have commented on the toolbox, suggesting
clarifications and corrections to the routines and to the tutorial text.
We are grateful for all suggestions, which have helped to keep the Wafo
project alive.

Major updates and additions have also been made duringing the years,
many of them caused by new Matlab versions. The new graphics system
introduced with Matlab2014b motivated updates to all plotting routines.
Syntax changes and warnings for deprecated functions have required other
updates.

Several additions have also been made. In 2016, a new module, handling
non-linear Lagrange waves, was introduced. A special tutorial for the
Lagrange routines is included in the module ``lagrange``; (Wafo Lagrange
– a Wafo Module for Analysis of Random Lagrange Waves 2017). Two sets of
file- and string-utility routines were also added 2016.

During 2015 the Wafo-project moved from
``http://code.google.com/p/wafo/`` to to
``https://github.com/wafo-project/``, where it can now be found under
the generic name Wafo – no version number needed.

In order to facilitate the use of Wafo outside the Matlab environment,
most of the Wafo routines have been checked for use with Octave. On
``github`` one can also find a start of a Python-version, called pywafo.

Recurring changes in the Matlab language may continue to cause the
command window flood with warnings for deprecated functions. The
routines in this version of Wafo have been updated to work well with
Matlab2017a. We will continue to update the toolbox in the future,
keeping compatibility with older versions.

Foreword to 2011 edition
------------------------

This is a tutorial for how to use the Matlab toolbox Wafo for analysis
and simulation of random waves and random fatigue. The toolbox consists
of a number of Matlab m-files together with executable routines from
Fortran or C++ source, and it requires only a standard Matlab setup,
with no additional toolboxes.

A main and unique feature of Wafo is the module of routines for
computation of the exact statistical distributions of wave and cycle
characteristics in a Gaussian wave or load process. The routines are
described in a series of examples on wave data from sea surface
measurements and other load sequences. There are also sections for
fatigue analysis and for general extreme value analysis. Although the
main applications at hand are from marine and reliability engineering,
the routines are useful for many other applications of Gaussian and
related stochastic processes.

The routines are based on algorithms for extreme value and crossing
analysis, developed over many years by the authors as well as many
results available in the literature. References are given to the source
of the algorithms whenever it is possible. These references are given in
the Matlab-code for all the routines and they are also listed in the
Bibliography section of this tutorial. If the references are not used
explicitly in the tutorial; it means that it is referred to in one of
the Matlab m-files.

Besides the dedicated wave and fatigue analysis routines the toolbox
contains many statistical simulation and estimation routines for general
use, and it can therefore be used as a toolbox for statistical work.
These routines are listed, but not explicitly explained in this
tutorial.

The present toolbox represents a considerable development of two earlier
toolboxes, the Fat and Wat toolboxes, for fatigue and wave analysis,
respectively. These toolboxes were both Version 1; therefore Wafo has
been named Version 2. The routines in the tutorial are tested on
Wafo-version 2.5, which was made available in beta-version in January
2009 and in a stable version in February 2011.

The persons that take actively part in creating this tutorial are (in
alphabetical order): *Per Andreas Brodtkorb*\  [1]_, *Pär Johannesson*,
*Georg Lindgren*

, *Igor Rychlik*.

Many other people have contributed to our understanding of the problems
dealt with in this text, first of all Professor Ross Leadbetter at the
University of North Carolina at Chapel Hill and Professor Krzysztof
Podgórski, Mathematical Statistics, Lund University. We would also like
to particularly thank Michel Olagnon and Marc Provosto, at Institut
Français de Recherches pour l’Exploitation de la Mer (IFREMER), Brest,
who have contributed with many enlightening and fruitful discussions.

Other persons who have put a great deal of effort into Wafo and its
predecessors FAT and WAT are Mats Frendahl, Sylvie van Iseghem, Finn
Lindgren, Ulla Machado, Jesper Ryén, Eva Sjö, Martin Sköld, Sofia Åberg.

This tutorial was first made available for the beta version of Wafo
Version 2.5 in November 2009. In the present version some misprints have
been corrected and some more examples added. All examples in the
tutorial have been run with success on MATLAB up to 2010b.
