import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))

from lmfit import Model as LM_Model
from lmfit import Parameters as LM_Parameters

from ndspec.SimpleFit import SimpleFit
from ndspec.FitPowerSpectrum import FitPowerSpectrum
from ndspec.Likelihoods import chisq, cstat, ratio
import ndspec.SamplingUtils as sampling_utils
from ndspec.SamplingUtils import (set_sampling_priors,
                                  set_sampling_model,
                                  set_sampling_data,
                                  set_sampling_parameters,
                                  initialise_mcmc,
                                  nested_sampling_priors,
                                  log_priors,
                                  sampling_gaussian_likelihood,
                                  sampling_cash_likelihood,
                                  reflect_parameter,
                                  priorUniform,
                                  priorLogUniform,
                                  priorNormal,
                                  priorLogNormal,
                                  priorTruncNormal)

import pytest

#all the tests use a constant model, for a number of data points set by _N_DATA
#with value set by the argument "value". We need a dummy fitter object throughout 
#the tests, so that means the independent variable of the function must be 
#calledd 'freq'
_N_DATA = 8

def flat_model(freq, value=1.0):
    return value*np.ones(_N_DATA)


def make_psd_fitter(data_value=1.0, data_err=1.0, likelihood="chisq"):
    """
    Build a minimal, valid FitPowerSpectrum whose data is a constant array.
    Used throughout the tests as dummy object to interface with the sampling 
    functions. 
    """
    data = data_value*np.ones(_N_DATA)
    err = data_err*np.ones(_N_DATA)
    freqs = np.linspace(1, _N_DATA, _N_DATA)

    fitter = FitPowerSpectrum(likelihood=likelihood)
    fitter.set_data(data, err, freqs)

    model = LM_Model(flat_model,independent_vars=['freq'])
    params = LM_Parameters()
    #"x" is the independent variable, "value" is the one free parameter
    params.add_many(('value', data_value, True, None, None))
    fitter.set_model(model)
    fitter.set_params(params)
    return fitter


def reset_globals():
    """
    Reset all module-level globals in SamplingUtils between tests. This is 
    necessary to avoid the global values contaminating tests they would not 
    be used in during regular use. 
    """
    sampling_utils.sampling_names = None
    sampling_utils.sampling_values = None
    sampling_utils.sampling_priors = None
    sampling_utils.sampling_data = None
    sampling_utils.sampling_data_err = None
    sampling_utils.sampling_model = None
    sampling_utils.sampling_noise = None
    sampling_utils.sampling_noise_err = None
    sampling_utils.sampling_exp = None
    sampling_utils.sampling_bins = None
    sampling_utils.sampling_params = None


#this tests checks that priors throw errors when their bounds are specified badly
class TestPriorErrors(object):
    
    #uniform priors need min < max bounds 
    def test_uniform_bounds(self):
        with pytest.raises(ValueError):
            priorUniform(1.0, 0.0)
        with pytest.raises(ValueError):
            priorUniform(1.0, 1.0)

    def test_loguniform_bounds(self):
        with pytest.raises(ValueError):
            priorLogUniform(1.0, 0.0)
        with pytest.raises(ValueError):
            priorLogUniform(1.0, 1.0)

    #gaussian priors need sigma >= 0
    def test_normal_sigma(self):
        with pytest.raises(ValueError):
            priorNormal(0.0, 0.0)
        with pytest.raises(ValueError):
            priorNormal(0.0, -1.0)

    def test_lognormal_sigma(self):
        with pytest.raises(ValueError):
            priorLogNormal(0.0, 0.0)
        with pytest.raises(ValueError):
            priorLogNormal(0.0, -1.0)

    #the truncated gaussian need min < max bounds and sigma >= 0
    def test_truncnormal_sigma(self):
        with pytest.raises(ValueError):
            priorTruncNormal(0.0, -1.0, 0.0, 1.0)

    def test_truncnormal_bounds(self):
        with pytest.raises(ValueError):
            priorTruncNormal(0.0, 1.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            priorTruncNormal(0.0, 1.0, 1.0, 1.0)

#this test checks that the sampling setup throws errors where appropriate
class TestSetupErrors(object):

    #before running any test, reset all the global variables 
    def setup_method(self, method):
        reset_globals()

    #test that set_sampling_parameters only accepts an lmfit Parameters object
    def test_set_parameters_type(self):
        with pytest.raises(AttributeError):
            set_sampling_parameters(np.ones(3))

    #test that set_sampling_model rejects anything that isn't a SimpleFit or a 
    #JointFit object 
    def test_set_model_type(self):
        with pytest.raises(TypeError):
            set_sampling_model(np.ones(3))

    #test that set_data also only takes SimpleFit or JointFit objects
    def test_set_data_type(self):
        with pytest.raises(TypeError):
            set_sampling_data(np.ones(3))

    #test that initialise_mcmc rejects invalid fit objects up front
    def test_initialise_mcmc_type(self):
        with pytest.raises(TypeError):
            initialise_mcmc(np.ones(3), {})

    #test that set_sampling_priors flags priors named after parameters not 
    #include in  the model
    def test_priors_unknown_parameter(self):
        fitter = make_psd_fitter()
        #"value" is valid, "not_a_parameter" is not 
        priors = {'value': priorUniform(0.0, 10.0),
                  'not_a_parameter': priorUniform(0.0, 1.0)}
        with pytest.raises(ValueError):
            set_sampling_priors(fitter, priors)

    #test that a free parameter that has no prior throws an error
    def test_priors_missing_prior(self):
        fitter = make_psd_fitter()
        priors = {}
        with pytest.raises(ValueError):
            set_sampling_priors(fitter, priors)

    #test that specifying a prior for a fixed parameter is also an error
    def test_priors_prior_for_fixed(self):
        fitter = make_psd_fitter()
        #freeze the only parameter, then try to give it a prior
        fitter.model_params['value'].vary = False
        priors = {'value': priorUniform(0.0, 10.0)}
        with pytest.raises(ValueError):
            set_sampling_priors(fitter, priors)

    #test that nested_sampling_priors needs the priors to have been set first
    #in order to be called 
    def test_nested_priors_no_priors(self):
        sampling_utils.sampling_priors = None
        with pytest.raises(AttributeError):
            nested_sampling_priors(np.array([0.5, 0.5]))

    #test that the priors actually check for the size of the parameter array 
    #and prior dictionary to be identical 
    def test_log_priors_size_mismatch(self):
        priors = {'a': priorUniform(0.0, 1.0), 'b': priorUniform(0.0, 1.0)}
        theta = np.array([0.5])
        with pytest.raises(AttributeError):
            log_priors(theta, priors)


#test the likelihood calculations in the sampling wrappers against known values 
class TestGaussianLikelihood(object):
    def setup_method(self, method):
        reset_globals()

    #test the case when the model and data are identical, and therefore the 
    #fit statistic is zero 
    def test_identical_model_and_data(self):
        fitter = make_psd_fitter(data_value=3.0, data_err=1.0)
        theta = initialise_mcmc(fitter, {'value': priorUniform(0.0, 10.0)})

        stat = sampling_gaussian_likelihood(theta)
        assert np.isclose(stat, 0.0)

    #test the case when the error valuesa are all unity. In this case, the 
    #ikelihood is just -0.5 * sum (data-model)**2.
    #Offsetting the model by a constant delta from the data gives
    #-0.5 * N * delta**2; here we pre-computed that value and compare against it.
    def test_constant_offset(self):
        data_value = 2.0
        delta = 0.5
        fitter = make_psd_fitter(data_value=data_value, data_err=1.0)
        #set the prior/globals up around the true value
        initialise_mcmc(fitter, {'value': priorUniform(0.0, 10.0)})
        #now evaluate the likelihood at a model offset by delta
        theta_offset = np.array([data_value + delta])
        stat = sampling_gaussian_likelihood(theta_offset)
        #now compare with the expected value 
        expected = -0.5*_N_DATA*delta**2
        assert np.isclose(stat, expected)

    #test that scaling the error bars by a factor f scales the likelihood by a
    #factor 1/f**2
    def test_error_scaling(self):
        data_value = 2.0
        delta = 1.0
        fitter_a = make_psd_fitter(data_value=data_value, data_err=1.0)
        initialise_mcmc(fitter_a, {'value': priorUniform(0.0, 10.0)})
        stat_a = sampling_gaussian_likelihood(np.array([data_value + delta]))

        reset_globals()
        fitter_b = make_psd_fitter(data_value=data_value, data_err=2.0)
        initialise_mcmc(fitter_b, {'value': priorUniform(0.0, 10.0)})
        stat_b = sampling_gaussian_likelihood(np.array([data_value + delta]))

        #err doubled -> residual halved -> squared residual quartered
        assert np.isclose(stat_b, stat_a/4.0)

    #test that if the data and model have identical values 
    #a) the likelihood is optimized, and 
    #b) whether we increase or decrease the model by 1, the likelihood changes 
    #by the same amount
    def test_identical_is_maximum(self):
        data_value = 4.0
        fitter = make_psd_fitter(data_value=data_value, data_err=1.0)
        initialise_mcmc(fitter, {'value': priorUniform(0.0, 10.0)})

        best = sampling_gaussian_likelihood(np.array([data_value]))
        worse_hi = sampling_gaussian_likelihood(np.array([data_value + 1.0]))
        worse_lo = sampling_gaussian_likelihood(np.array([data_value - 1.0]))
        #check the likelihood is optimized
        assert best >= worse_hi
        assert best >= worse_lo
        #check the likelihood is symmetric around the optimum 
        assert np.isclose(worse_hi, worse_lo)


#test the chi squared likelihood evaluations on their own, without the sampling 
#wrappers, against known values/cases
class TestChisqFunction(object):

    #test that for identical model and data, the residual is zero in all bins
    #and also when it is summed 
    def test_identical_is_zero(self):
        data = np.array([1.0, 2.0, 3.0])
        err = np.ones(3)
        res = chisq(data, err, data)
        assert np.allclose(res, 0.0)
        summed = chisq(data, err, data, summed=True) 
        assert np.isclose(summed, 0.0)

    #test that with unity errors and each bin off by 1, the summed chi-squared 
    #is the total number of data points N
    def test_unit_offset_summed(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        model = np.array([2.0, 1.0, 4.0, 3.0])  
        err = np.ones(4)
        assert np.isclose(chisq(data, err, model, summed=True), 4.0)

    #test that when a background is included, the residuals still return zero
    #in the case when the model explains the data (alone perfectly), 
    def test_background_subtraction(self):
        model = np.array([2.0, 2.0])
        noise = np.array([1.0, 1.0])
        data = model + noise
        err = np.ones(2)
        noise_err = np.ones(2)
        res = chisq(data, err, model, noise=noise, noise_err=noise_err)
        assert np.allclose(res, 0.0)

    #test that trying to use a background without its error (or vice versa) 
    #throws an error
    def test_background_without_error(self):
        data = np.ones(2)
        err = np.ones(2)
        model = np.ones(2)
        with pytest.raises(ValueError):
            chisq(data, err, model, noise=np.ones(2))
        with pytest.raises(ValueError):
            chisq(data, err, model, noise_err=np.ones(2))

#similar tests as above, but this time we are testing the Cash statistic. 
#cstat(data, model, exp, widths, ...) returns 2x the Cash statistic, per bin
#or summed. Using exp=1 and unit channel widths makes the counts equal to
#the input rates, so the arithmetic is easy to write down by hand.
class TestCstatFunction(object):
    
    #test that the statistic is zero when the model is identical to the data
    def test_identical_is_zero(self):
        data = np.array([3.0, 7.0, 2.0])
        stat = cstat(data, data, 1.0, np.ones(3), summed=True)
        assert np.isclose(stat, 0.0)

    #test a simple generic case - a single bin with data=2, model=1 gives 
    #2*(model - data + data*ln(data/model))
    def test_single_bin_known_value(self):
        data = np.array([2.0])
        model = np.array([1.0])
        stat = cstat(data, model, 1.0, np.ones(1), summed=True)
        expected = 2.0*(1.0 - 2.0 + 2.0*(np.log(2.0) - np.log(1.0)))
        assert np.isclose(stat, expected)

    #ensure that a data bin of exactly zero must not produce nan; in the limit 
    #the bin's contribution should be 2*model
    def test_zero_data_bin(self):
        data = np.array([0.0, 5.0])
        model = np.array([1.0, 5.0])
        per_bin = cstat(data, model, 1.0, np.ones(2))
        assert np.all(np.isfinite(per_bin))
        assert np.isclose(per_bin[0], 2.0*model[0])
        assert np.isclose(per_bin[1], 0.0)

    #test the per-bin contribution to the *summed* statistic. 
    def test_zero_data_bin_summed_value(self):
        #test a single empty bin: summed statistic is exactly 2*model
        model = np.array([3.7])
        stat = cstat(np.array([0.0]), model, 1.0, np.ones(1), summed=True)
        assert np.isclose(stat, 2.0*model[0])

        #test multiple empty bins: contributions add, giving 2*sum(model)
        model = np.array([1.0, 2.5, 0.4])
        stat = cstat(np.zeros(3), model, 1.0, np.ones(3), summed=True)
        assert np.isclose(stat, 2.0*np.sum(model))

        #test that the exposure/width conversion still applies to an empty bin: 
        #the contribution is 2*model*exp*width, so a non-trivial exp and width
        #scale it as expected
        model = np.array([2.0])
        exp = 5.0
        widths = np.array([0.3])
        stat = cstat(np.array([0.0]), model, exp, widths, summed=True)
        assert np.isclose(stat, 2.0*model[0]*exp*widths[0])

    #test that including a background also produces expected results 
    #because of how messy the Cstat becomes with background cases, this takes 
    #a whole lot of tests. Yay. 
    #First, test that passing a background with zero counts recovers the 
    #base case when we do not pass a background 
    def test_background_zero_reduces_to_nobkg(self):
        data = np.array([5.0, 2.0, 8.0])
        model = np.array([3.0, 4.0, 6.0])
        widths = np.ones(3)
        with_zero_bkg = cstat(data, model, 1.0, widths, noise=np.zeros(3))
        no_bkg = cstat(data, model, 1.0, widths)
        assert np.allclose(with_zero_bkg, no_bkg)

    #test the same, in the limit of a very small background 
    def test_background_small_limit(self):
        data = np.array([5.0])
        model = np.array([3.0])
        no_bkg = cstat(data, model, 1.0, np.ones(1))
        tiny_bkg = cstat(data, model, 1.0, np.ones(1), noise=np.array([1e-9]))
        assert np.allclose(tiny_bkg, no_bkg)

    #test that when the model is identical to data-background, the statistic 
    #is zero 
    def test_background_perfect_match_is_zero(self):
        model = np.array([3.0, 10.0])
        noise = np.array([1.0, 0.3])   # model > noise in both bins
        data = model + noise
        per_bin = cstat(data, model, 1.0, np.ones(2), noise=noise)
        assert np.allclose(per_bin, 0.0)

    #finally, test that two bins with identical background counts give exactly  
    #the summed statistic of a single bin, with that same number of counts
    def test_background_bin_addition(self):
        data, model, noise = 5.0, 3.0, 1.0
        one_bin = cstat(np.array([data]), np.array([model]), 
                        1.0, np.ones(1),
                        noise=np.array([noise]), summed=True)
        two_bin = cstat(np.array([data, data]), np.array([model, model]), 
                        1.0, np.ones(2),
                        noise=np.array([noise, noise]), summed=True)
        assert np.isclose(two_bin, 2.0*one_bin)

#this class tests that the reflect_parameter function behaves as intended 
class TestReflectParameter(object):

    #test that a value already inside the bounds is returned unchanged
    def test_in_bounds_unchanged(self):
        x = np.array([0.3, 0.7])
        y = reflect_parameter(x, 0.0, 1.0)
        assert np.allclose(y, x)

    #test the example from the docstring: x=1.2, a=0, b=1 -> 0.8
    def test_reflect_above(self):
        x = np.array([1.2])
        y = reflect_parameter(x, 0.0, 1.0)
        assert np.allclose(y, 0.8)

    #test the same as above, but the parameter is below the lower bound:
    #x=-0.2, a=0, b=1 -> 0.2
    def test_reflect_below(self):
        x = np.array([-0.2])
        y = reflect_parameter(x, 0.0, 1.0)
        assert np.allclose(y, 0.2)
