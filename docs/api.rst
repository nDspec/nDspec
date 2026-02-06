.. _api:

nDspec API
==========

Docstrings for every class and function in the nDspec modelling software

Operator Class
~~~~~~~~~~~~~~

.. autoclass:: ndspec.Operator.nDspecOperator
   :members:

Response Matrix Class
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.Response.ResponseMatrix
   :members:

Timing Classes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.Timing.FourierProduct
   :members:

.. autoclass:: ndspec.Timing.PowerSpectrum
   :members:
   
.. autoclass:: ndspec.Timing.CrossSpectrum
   :members:
   
SimpleFit Classes 
~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.SimpleFit.SimpleFit
   :members:

.. autoclass:: ndspec.SimpleFit.EnergyDependentFit
   :members:
   
.. autoclass:: ndspec.SimpleFit.FrequencyDependentFit
   :members:
   
Data loading utilities
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ndspec.SimpleFit.load_lc

.. autofunction:: ndspec.SimpleFit.load_pha  
   
FitPowerSpectrum Class
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.FitPowerSpectrum.FitPowerSpectrum
   :members:
   
FitTimeAvgSpectrum Class
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.FitTimeAvgSpectrum.FitTimeAvgSpectrum
   :members:
   
FitCrossSpectrum Class
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.FitCrossSpectrum.FitCrossSpectrum
   :members:

JointFit Class
~~~~~~~~~~~~~~~~~

.. autoclass:: ndspec.JointFit.JointFit
   :members:

   
Sampling functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: ndspec.SamplingUtils.set_sampling_priors

.. autofunction:: ndspec.SamplingUtils.set_sampling_model

.. autofunction:: ndspec.SamplingUtils.set_sampling_data

.. autofunction:: ndspec.SamplingUtils.set_sampling_parameters

.. autofunction:: ndspec.SamplingUtils.initialise_mcmc

.. autofunction:: ndspec.SamplingUtils.reflect_parameter

.. autoclass:: ndspec.SamplingUtils.priorUniform
    :members:

.. autoclass:: ndspec.SamplingUtils.priorLogUniform
    :members:

.. autoclass:: ndspec.SamplingUtils.priorNormal
    :members:

.. autoclass:: ndspec.SamplingUtils.priorLogNormal
    :members:

.. autofunction:: ndspec.SamplingUtils.nested_sampling_priors

.. autofunction:: ndspec.SamplingUtils.log_priors

.. autofunction:: ndspec.SamplingUtils.sampling_cash_likelihood

.. autofunction:: ndspec.SamplingUtils.mcmc_cash_likelihood

.. autofunction:: ndspec.SamplingUtils.sampling_gaussian_likelihood

.. autofunction:: ndspec.SamplingUtils.mcmc_gaussian_likelihood

.. autofunction:: ndspec.SamplingUtils.process_emcee

Xspec library 
~~~~~~~~~~~~~

.. autoclass:: ndspec.XspecInterface.ModelInterface
    :members:
    
.. autoclass:: ndspec.XspecInterface.FortranInterface
    :members:
    
.. autoclass:: ndspec.XspecInterface.CInterface
    :members:

Model library
~~~~~~~~~~~~~

.. autofunction:: ndspec.Models.lorentz

.. autofunction:: ndspec.Models.cross_lorentz

.. autofunction:: ndspec.Models.powerlaw

.. autofunction:: ndspec.Models.brokenpower

.. autofunction:: ndspec.Models.gaussian

.. autofunction:: ndspec.Models.bbody

.. autofunction:: ndspec.Models.varbbody

.. autofunction:: ndspec.Models.gauss_fred

.. autofunction:: ndspec.Models.gauss_bkn

.. autofunction:: ndspec.Models.bbody_fred

.. autofunction:: ndspec.Models.bbody_bkn

.. autofunction:: ndspec.Models.pivoting_pl

.. autofunction:: ndspec.Models.plot_2d

Simulator utilities
~~~~~~~~~~~~~~~~~~~

.. autofunction:: ndspec.Simulator.simulate_lightcurve

.. autofunction:: ndspec.Simulator.simulate_time_lags

.. autofunction:: ndspec.Simulator.simulate_time_averaged
