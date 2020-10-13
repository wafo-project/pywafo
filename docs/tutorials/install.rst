.. _install:


=============
Install guide
=============

Before you can use WAFO, you'll need to get it installed. This guide will
guide you through a simple installation
that'll work while you walk through the introduction.


Install Python
==============

Being a Python library, WAFO requires Python. Preferably you ned version 3.4 or
newer, but you get the latest version of Python at
https://www.python.org/downloads/.

You can verify that Python is installed by typing ``python`` from the command shell;
you should see something like:

.. code-block:: console


    Python 3.6.3 (64-bit)| (default, Oct 15 2017, 03:27:45)
    [MSC v.1900 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>>


``pip`` is the Python installer. Make sure yours is up-to-date, as earlier versions can be less reliable:


.. code-block:: console

    $ pip install --upgrade pip


Dependencies
============
WAFO requires numpy 1.9 or newer, 
scipy 0.8 or newer, and Python 3.5 or newer. 
This tutorial assumes you are using Python 3. 



Install WAFO
====================

To install WAFO simply type in the 'command' shell:

.. code-block:: console

    $ pip install WAFO

to get the lastest stable version. Using pip also has the advantage 
that all requirements are automatically installed.


Verifying installation
======================
To verify that WAFO can be seen by Python, type ``python`` from your shell.
Then at the Python prompt, try to import WAFO:

.. parsed-literal::

    >>> import wafo
    >>> print(wafo.__version__)
    |release|


To test if the toolbox is working correctly paste the following in an interactive python prompt:

.. parsed-literal::

    wafo.test('--doctest-module')


If the result show no errors, you now have installed a fully functional toolbox.
Congratulations!


That's it!
==========

That's it -- you can now :doc:`move onto the getting started tutorial </tutorials/getting_started>`
