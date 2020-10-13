Technical information
=====================

-  Wafo was released in a stable version in February 2011. The most
   recent stable updated and expanded version of Wafo can be downloaded
   from

   ``https://github.com/wafo-project/``

   Older versions can also be downloaded from the Wafo homepage
   (WAFO-group 2000)

   ``http://www.maths.lth.se/matstat/wafo/``

-  To get access to the Wafo toolbox, unzip the downloaded file,
   identify the wafo package and save it in a folder of your choise.
   Take a look at the routines ``install.m``, ``startup.m``,
   ``initwafo.m`` in the ``WAFO`` and ``WAFO/docs`` folders to learn how
   Matlab can find Wafo.

-  To let Matlab start Wafo automatically, edit ``startup.m`` and save
   it in the starting folder for Matlab.

-  To start Wafo manually in Matlab, add the ``WAFO`` folder manually to
   the Matlab-path and run ``initwafo``.

-  In this tutorial, the word ``WAFO``, when used in path
   specifications, means the full name of the Wafo main catalogue, for
   instance ``C:/wafo/``

-  The Matlab code used for the examples in this tutorial can be found
   in the Wafo catalogue ``WAFO/papers/tutorcom/``

   The total time to run the examples in fast mode is less than fifteen
   minutes on a PC from 2017, running Windows 10 pro with Intel(R)
   Core(TM) i7-7700 CPU, 3.6 GHz, 32 GB RAM. All details on execution
   times given in this tutorial relates to that configuration.

-  Wafo is built of modules of platform independent Matlab m-files and a
   set of executable files from ``C++`` and ``Fortran`` source files.
   These executables are platform and Matlab-version dependent, and they
   have been tested with recent Matlab and Windows installations.

-  If you have many Matlab-toolboxes installed, name-conflicts may
   occur. Solution: arrange the Matlab-path with ``WAFO`` first.

-  For help on the toolbox, write ``help wafo``.

-  Comments and suggestions are solicited — send to
   ``wafo@maths.lth.se``
