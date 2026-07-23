Welcome to the nDspec documentation!
====================================

nDspec is a Python-based X-ray astronomy modelling package for spectral, spectral-timing and — in future releases — time-averaged and/or time-dependent polarimetry data. The software allows users to fit common one-dimensional products used in X-ray astronomy, like time-averaged spectra, power spectra, and cross spectra (including lags) as a function of photon energy/and or frequency. In addition, it is possible to fit arbitrary two-dimensional data, and to jointly fit different datasets and observations. Bayesian sampling using common MCMC and nested sampling packages is fully supported. The software comes with a small library of phenomenological models, and a Python interface to the Xspec model library; alternatively, users can use their own Python-based models. 

The software is built on core "Operator" classes which users can use in their own code, outside of the fitting environment. nDspec currently includes two "Operator" categories. The ResponseMatrix class allows user to fold one- and two-dimensional models through the response matrix of modern X-ray instruments - formally these would either be spectral-timing or spectral-polarimetry models, although the second dimension beyond photon energy does not matter. The observatories/instruments explicitly supported are RXTE/PCA, Swift/XRT, XMM-Newton, NuSTAR and NICER. The PowerSpectrum and CrossSpectrum classes can compute standard Fourier products like lag spectra from time- and/or energy- dependent, user-defined models. It is possible to input models defined in both the time and Fourier domains, as well as to combine multiple components. These classes allow users to convert the output of a given model to units that are comparable to data - for example, to arrays of counts per energy channel.

Along with the core functionality, the current release of nDspec provides classes for modelling a wide variety of data types. In the case of data that is multi-dimensional and/or (mathematically) complex, like for a cross spectrum, users can fit lags alone (in units of time), or simultaneously model the real and imaginary, or modulus and phase, as a unique dataset, without the need to instantiate multiple models and/or tie or define multiple parameters. nDspec provides classes for handling model and parameter management, likelihood optimization and plotting, and to interface with Bayesian samplers like the `emcee <https://emcee.readthedocs.io/en/stable/>`_ Python package for performing inference. Finally, a small library of one and two dimensional phenomenological models is included. 

Installation and testing
~~~~~~~~~~~~~~~~~~~~~~~~ 

The stable version of the software can be installed via pip:

.. code-block:: bash

   pip install ndspec 

Alternatively, If you want to use (or contribute to!) features that are still in developement, you should install the software directly from the repository:

.. code-block:: bash

   git clone https://github.com/nDspec/nDspec.git
   cd nDspec
   pip install .
   
For a development install, use ``pip install -e ".[dev]"``. 

Unit tests use `pytest <https://pytest.org>`_. From the root of the cloned repository:

.. code-block:: bash

   pytest

Tests that depend on Xspec/HEASoft are marked and will be skipped automatically if HEASoft is not available in your environment.

Table of contents
~~~~~~~~~~~~~~~~~ 

.. toctree::
   :maxdepth: 2

   tutorials
   core_functionality
   api
