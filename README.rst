======
nDspec
======

|Docs|

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-dimensional X-ray data modelling in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nDspec is a software package designed to model (primarly) X-ray astronomical data; in particular, it is meant to enable users to fit spectral, timing and (in the future) polarimetric data natively. 
Currently, nDspec provides the following features:

- a class to handle folding an energy-dependent model through a given X-ray instrument response matrix. We currently support existing and legacy missions (Swift/XRT, RXTE, NuSTAR, Nicer, XMM-Newton); next-generation missions like XRISM and Athena are not supported yet.
- Classes to handle the standard spectral-timing data products (power spectra, two-dimensional cross spectra, as well as their one-dimensional projections like lag-frequency spectra). Models need to be defined by the user either in the time domain (ie, an impulse response function) or in the Fourier domain (ie, a transfer functions). Additive models can be combined by the users before computing a given Fourier product.
- a small library of phenomenological spectral-timing models, currently including a one dimensional power-law, black body and gaussian, as well as example impulse response and/or transfer functions that parametrize the time dependence of these components.

The goal of the software is to provide users with an Xspec-like package, but to move beyond simple one-dimensional spectral fitting, which typically has relied on each group utilizing their own custom software. Being an open-source modular package, unlike legacy software nDspec allows users to interface with data science libraries for model handling or fitting. 

Documentation
-------------

The software documentation, including installation instructions, tutorials, and the full API, is `found on readthedocs <https://ndspec.readthedocs.io/en/latest/>`_. You can also find notebooks discussing the features of each class in the /notebooks/ folder.

Installation and testing
------------------------

The stable version of the software can be installed via pip; if you want to use (or contribute to!) features that are still in-developement, you should install the software directly from the repository. Unit tests make use of `py.test <https://pytest.org>`_. Running the unit tests simply requires opening the folder in which users downloaded the repository, and running the command `pytest` in the command line. 

Related Packages
----------------

- `STINGRAY <https://github.com/StingraySoftware/stingray>`_ is a library that allows users to produce typical spectral-timing products from observations. A typical use would consist of using Stingray to handle the data a user might be interested in (e.g. by converting event files into data products for modelling), and nDspec to handle modelling of that data.

- `Astropy <https://www.astropy.org/>`_ is the standard library for handling FITS files in a Python environment.

- `emcee <https://emcee.readthedocs.io/en/stable/>`_ implements one of the most commonly used Bayesian sampling algorithms in astronomy. 

- `lmfit <https://lmfit.github.io/lmfit-py/>`_ provides functions, classes and methods for curve optimization, such as least chi-squared fitting. 

Contributing
------------

nDspec is a fully open source software, so we welcome your contribution and feedback!
The best way to contact the developers is through the `issues`_ page - even a simple comment on what you find useful or intuitive (or vice versa) goes a long way in helping the project. 
If you would like to contribute a piece of code, you are welcome to do so either by opening an `issue`_ or submitting a `pull request`_. Before you do so, you should read the `code of conduct`_ and `coding conventions`_.

Citing
------

The initial release paper is not released yet; until then, please refer to this repository if you make use of the software in your work.

License
-------

All content © 2023 The Authors. The code is distributed under the MIT license; see `LICENSE <LICENSE>`_ for details.

.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://ndspec.readthedocs.io/en/latest/
.. _issues: https://github.com/nDspec/nDspec/issues
.. _issue: https://github.com/nDspec/nDspec/issues
.. _pull request: https://github.com/nDspec/nDspec/pulls
.. _code of conduct: https://github.com/nDspec/nDspec/blob/beta/code_of_conduct.md
.. _coding conventions: https://github.com/nDspec/nDspec/blob/beta/code_conventions.md
