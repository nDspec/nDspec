.. nDspec documentation master file, created by
   sphinx-quickstart on Fri Oct 27 14:54:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the nDspec documentation!
====================================

This is the intial repository for the nDspec Python-based X-ray astronomy modeling software. nDspec is designed to allow users to model spectral, spectral timing and (in the future) spectral polarimetry and polarimetry timing X-ray data. The software allows users to fit power-spectra as a function of Fourier frequency, time-averaged spectra as a function of photon energy (channel), as well as cross spectra (and their related products, such as lag spectra) as a function of both Fourier frequency and energy. Bayesian sampling using the common package emcee is fully supported. The software comes with a small library of phenomenological models; alternatively, users can use their own Python-based models. Currently, nDspec does not support joint fitting multiple datasets together. 

Currently, the software is built on two core functionalities which users can use in their own code, outside of the fitting environment. The ResponseMatrix class allows user to fold one- and two-dimensional models through the response matrix of modern X-ray instruments - formally these would either be spectral-timing or spectral-polarimetry models, although the second dimension beyond photon energy does not matter. The observatories/instruments explicitely supported are RXTE/PCA, Swift/XRT, XMM-Newton, NuSTAR and NICER. The PowerSpectrum and CrossSpectrum classes can compute standard Fourier products like lag spectra from time- and/or energy- dependent, user-defined models. It is possible to input models defined in both the time and Fourier domains, as well as to combine multiple components.

Along with these two core functionalities, the current release of nDspec provides classes for modelling time-averaged spectra, power spectra, and cross-spectra as a function of both energy and frequency (or both). In the latter case, users can fit lags alone (in units of time), or jointly model the real and imaginary, or modulus and phase, as a unique dataset, without the need to instatiate multiple models and/or tie or define multiple parameters. nDspec provides classes for handling model and parameter management, chi-squared optimization and plotting, and to interface with the `emcee <https://emcee.readthedocs.io/en/stable/>`_ Python package for performing Bayesian inference. Finally, a small library of one and two dimensional phenomenological models is included. Alternatively, users can implement their own models, either by writing them in Python directly, or interfacing Python with their own compiled code (e.g. through the use of `pybind <https://pybind11.readthedocs.io/en/stable/basics.html>`_ or `f2py <https://numpy.org/doc/stable/f2py/>`_). An interface to call the Xspec-compatible models within Python is also included. 

The beta release of nDspec includes a new class, JointFit, which allows users to fit multiple datasets together, as well as link parameters between the datasets. This is useful for fitting multiple datasets from the same source, or to fit a model to both the time-averaged spectrum and the power spectrum of a source simultaneously. The JointFit class is designed to be flexible and can handle a wide range of fitting scenarios. The beta release has also added a new interface to Xspec models, as well as any models which adhere to the Xspec model format generally adopted by the X-ray astronomy community. This allows users to use their existing Xspec models directly in nDspec, without the need to rewrite them in Python.

Installation and testing
~~~~~~~~~~~~~~~~~~~~~~~~

The beta release of the software can only be installed from the repository. 

Unit tests utilize `py-test <https://pytest.org>`_. Running the unit tests simply requires opening the folder in which users downloaded the repository, and running the command `pytest` in the command line. 

Table of contents
~~~~~~~~~~~~~~~~~ 

Data fitting tutorials:

.. toctree::
    fit_psd
    fit_spec
    fit_cross_1d
    fit_cross_2d
    fit_joint

nDspec core functionality:

.. toctree::
    xspec_interface
    response_basics
    response_optimization   
    timing
    numerics

References and tables
~~~~~~~~~~~~~~~~~~~~~

* :ref:`search`
.. toctree::  
    api 
