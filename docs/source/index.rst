.. HiPERCAM observation utilities documentation master file, created by
   sphinx-quickstart on Wed Mar 11 13:41:12 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: globals.rst

|hiperobs| manual
=================

.. image:: https://img.shields.io/pypi/v/hcam_obsutils.svg
        :target: https://pypi.python.org/pypi/hcam_obsutils

The |hiperobs| package provides a collection of utilities for working with |hiper| or |ultra|
observations, including scripts for checking the readout noise, calculating zeropoints, 
measuring gains, checking for missing bias frames, and more. 

Most of these scripts are intended to be used whilst observing with the instruments and 
are gathered together here in one package for ease of installation and to provide
version control. 

The |hiperobs| package also provides an API for working with |hiper| and |ultra| data, 
which may be useful after observations, including a `Calibrator <hcam_obsutils.calibrator.Calibrator>`_ 
class for calculating photometric zeropoints, and comparison star magnitudes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   scripts

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`