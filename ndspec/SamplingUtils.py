import numpy as np 
import corner
import copy
import inspect
import math
import lmfit

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
from matplotlib.colors import TwoSlopeNorm
import scipy.stats

from .JointFit import JointFit
from .SimpleFit import SimpleFit
from .FitTimeAvgSpectrum import FitTimeAvgSpectrum
from .Likelihoods import cstat 

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
fi = 22
plt.rcParams.update({'font.size': fi-5})

sampling_names = None 
sampling_values = None
sampling_priors = None
sampling_data = None 
sampling_data_err = None
sampling_model = None 
sampling_noise = None
sampling_noise_err = None
sampling_exp = None
sampling_bins = None
sampling_vectorize = False

def set_sampling_priors(fitobj,priors):
    """
    This function is used to set the priors to be used with emcee sampling.  
    These priors are saved in a global variable called sampling_priors; therefore,  
    users should never re-use the variable name sampling_priors in their code.
    
    Parameters:
    -----------
    fitobj: ndspec.Fit...Object or ndspec.JointFit 
        Object containing the data, specified model, and parameters
    priors: dict 
        A dictionary of priors to be used in emcee. The key of each dictionary 
        should be the name of the parameter. Each key should contain an object 
        with a method called "logprob", which returns the (negative) logarithm 
        of the prior evaluated at a given point.
    """
    
    global sampling_priors
    input_par_names = list(priors.keys())
    obj_par_names = list(fitobj.model_params.keys())

    if set(input_par_names) > set(obj_par_names):
        raise ValueError("Not all specified priors are parameters present in the model")
    
    for key in obj_par_names:
        if (fitobj.model_params[key].vary is True) and (key not in input_par_names):
            raise ValueError(f"{key} does not have a prior. Fix {key} or specify one.")
        elif (fitobj.model_params[key].vary is False) and (key in input_par_names):
            raise ValueError("Incorrectly specified a prior for a fixed parameter")
        else:
            continue
    sampling_priors = priors 
    return 

def set_sampling_model(fitobj,vectorize=False): 
    """
    This function is used to set the model to be used with emcee sampling.  
    This model is saved in a global variable called sampling_model; therefore,  
    users should never re-use the variable name sampling_model in their code.
    Whether the model is evaluated for one set of parameters at a time, or for 
    every set at once, is stored in a global variable called sampling_vectorize.
    
    Parameters:
    -----------
    fitobj: ndspec.Fit...Object or ndspec.JointFit 
        Object containing the data, specified model, and parameters
        
    vectorize: bool, default False 
        A boolean switch to evaluate the model for every set of parameters at 
        once, rather than one set at a time, when computing the likelihood. It 
        requires both a fitter object whose eval_model method supports 
        vectorized evaluations, and a model which returns an array with an 
        additional leading axis running over the parameter sets when it is 
        passed arrays, rather than floats, as parameter values.
    """
    
    global sampling_model
    global sampling_vectorize
    if type(fitobj) == JointFit:
        fitobj.flatten = True
    elif not issubclass(type(fitobj),SimpleFit):
        raise TypeError("Invalid fit object passed")

    if (vectorize is True and "vectorize" not in 
        inspect.signature(fitobj.eval_model).parameters):
        raise TypeError(("The fitter object passed does not support vectorized"
                         " model evaluations"))

    sampling_model = fitobj.eval_model
    sampling_vectorize = vectorize
    return 
    
def set_sampling_data(fitobj):
    """
    This function is used to set the data and its error to be used with emcee 
    sampling. These are saved in global variables called sampling_data and 
    sampling_data_err; therefore, users should never re-use the variable names 
    sampling_data and sampling_data_err in their code. If the fitter object includes 
    noise (e.g. a background spectrum) and exposure times, these are included as
    well. 
    
    Parameters:
    -----------
    fitobj: ndspec.Fit...Object or ndspec.JointFit 
        Object containing the data, specified model, and parameters
    """
    
    global sampling_data
    global sampling_data_err
    global sampling_noise 
    global sampling_noise_err
    global sampling_exp
    global sampling_bins 

    if type(fitobj) != JointFit and not issubclass(type(fitobj),SimpleFit):
        raise TypeError("Invalid fit object passed")
 
    if type(fitobj) == JointFit:
        sampling_data = []
        sampling_data_err = []
        sampling_noise = []
        sampling_noise_err = []
        sampling_exp = []
        sampling_bins = []
        for obs in fitobj.joint:
            if type(fitobj.joint[obs]) == list:
                for m in fitobj.joint[obs]:
                    sampling_data.append(m.data)
                    sampling_data_err.append(m.data_err)
                    if m.noise is not None:
                        sampling_noise.append(m.noise)
                        sampling_noise_err.append(m.noise_err)
                    if m.likelihood == "cstat":
                        sampling_exp.append(m.exposure)
                        sampling_bins.append(m.ewidths)
            else:
                sampling_data.append(fitobj.joint[obs].data)
                sampling_data_err.append(fitobj.joint[obs].data_err) 
                if fitobj.joint[obs].noise is not None:
                    sampling_noise.append(fitobj.joint[obs].noise)
                    sampling_noise_err.append(fitobj.joint[obs].noise_err)
                if fitobj.joint[obs].likelihood == "cstat": 
                    sampling_exp.append(fitobj.joint[obs].exposure)
                    sampling_bins.append(fitobj.joint[obs].ewidths)                
    else:
        sampling_data = fitobj.data 
        sampling_data_err = fitobj.data_err
        if fitobj.noise is not None:
            sampling_noise = fitobj.noise
            sampling_noise_err = fitobj.noise_err
        if fitobj.likelihood == "cstat":
            sampling_exp = fitobj.exposure
            sampling_bins = fitobj.ewidths
            
    return

def set_sampling_parameters(params):
    """
    This function is used to set the parameters of the model to be used with
    emcee sampling. The parameter object (containing all parameters), as well as
    the names and values of the variable parameters are saved in global 
    variables called sampling_names, sampling_values and sampling_params; therefore, 
    users should never re-use these variable names in their code.
    
    Parameters:
    -----------
    params: lmfit.Parameters
        The lmfit parameters object used in the model, including those kept
        constant. 
        
    Returns:
    --------
    theta: np.array 
        A numpy array containing the values of the free parameters in the model.
    """
    
    global sampling_names 
    global sampling_values 
    global sampling_params

    if not isinstance(params,lmfit.Parameters):
        raise AttributeError("The parameters input must be an LMFit Parameters object")
    
    sampling_params = copy.copy(params) 
    sampling_values = []
    sampling_names = []
    theta = []
    for key in params:
        if params[key].vary is True:
            sampling_names = np.append(sampling_names,params[key].name)
            sampling_values = np.append(sampling_values,params[key].value)
            theta = np.append(theta,params[key].value)  
    return theta

def initialise_mcmc(fitobj,priors,vectorize=False):
    """
    This function is used to initialise an MCMC run. The Fit...Object can be
    any of the particular data products, or a JointFit object containing
    multiple FitObjects.

    Parameters:
    -----------
    fitobj: ndspec.Fit...Object or ndspec.JointFit 
        Object containing the data, specified model, and parameters

    priors: dict
        A dictionary of priors to be used in emcee. The key of each dictionary 
        should be the name of the parameter. Each key should contain an object 
        with a method called "logprob", which returns the (negative) logarithm 
        of the prior evaluated at a given point.

    vectorize: bool, default False 
        A boolean switch to evaluate the model for every set of parameters at 
        once, rather than one set at a time, when computing the likelihood. See 
        the set_sampling_model function for the requirements this places on the 
        fitter object and the model. If it is set to True, the likelihoods can 
        be passed an array containing one set of parameters per row, and return 
        one log-likelihood per set - which is the format expected e.g. by emcee 
        when it is run with vectorize=True.

    Returns:
    --------
    theta: np.array 
        A numpy array containing the values of the free parameters in the model.
    """
    if type(fitobj) == JointFit:
        pass
    elif issubclass(type(fitobj),SimpleFit):
        pass
    else:
        raise TypeError("Invalid fit object passed")
    
    theta = set_sampling_parameters(fitobj.model_params)
    set_sampling_data(fitobj)
    set_sampling_model(fitobj,vectorize)
    set_sampling_priors(fitobj,priors)
    return theta

def reflect_parameter(x, a, b):
    """
    This function is used to bounce a parameter value off a hard-limit, defined 
    by e..g the limits of a uniform prior. It is useful to improve the behavior
    and acceptance rate of a mcmc chain
    
    Parameters:
    -----------
    x: np.array(float) or (float) 
        A value, or array of values, for a parameter which may or may not 
        exceed the limits defined by the priors 
        
    a: np.array(float) or (float) 
        A value, or array of values, containing the lower limit allowed for 
        each parameter
    
    b: np.array(float) or (float) 
        A value, or array of values, containing the upper limit allowed for 
        each parameter
        
    Returns:
    --------
    y: np.array(float) or (float) 
        The parameter value to be used in the model, "bounced" of the hard limit 
        if it exceeds it - e.g. if x=1.2, a=0, b=1, then y = 0.8.
    """
    y = x.copy()
    width = b - a
    y = (y - a) % (2*width)
    y = np.where(y <= width, a + y, b - (y - width))

    return y

class priorUniform():
    """
    This class is used to compute a uniform prior distribution during Bayesian
    sampling, for a given model parameter. 
    
    Parameters:
    -----------
    min: float 
        The lower bound of the distribution. 
        
    max: float 
        The upper bound of the distribution. 
    """
    
    def __init__(self,min,max):
        if min >= max:
            raise ValueError("Lower bound of the prior must be smaller than the upper bound")
        
        self.min = min
        self.max = max
        self.reflect = False
        self.distribution = scipy.stats.uniform(self.min,self.max-self.min)
        pass 
        
    def logprob(self,theta):
        """
        This method returns the log probability of the distribution - in this 
        case, an (arbitrary, for the purpose of likelihood optimization) 
        constant. 
        
        Parameters:
        -----------
        theta: float 
            The parameter value for which the likelihood is to be computed. 
            
        Returns:
        --------
            The value of the likelihood for the input parameter.
        """
        
        if self.min < theta < self.max:
            return 0.0
        return -np.inf
        
    def transform_prior(self,quantile):
        """
        This method returns the parameter value defined from the quantile of  
        the cumulative distribution defined in the prior. It effectively 
        converts an interval from 0-1 to one that samples the prior distribution
        and is used by nested sampling to define prior likelihoods. 
        
        Parameters:
        -----------
        quantile: float, 0-1 
            The quantile for which the prior is to be computed 
            
        Returns:
        --------
        prior: float 
            The parameter value for the chosen parameter prior distribution. 
        """
        prior = self.distribution.ppf(quantile)
        return prior 


class priorLogUniform():
    """
    This class is used to compute a log-uniform prior distribution (in base e)
    during Bayesian sampling, for a given model parameter.
    
    Parameters:
    -----------
    min: float 
        The lower bound of the distribution 
        
    max: float 
        The upper bound of the distribution 
    """    
    
    def __init__(self,min,max):
        if min >= max:
            raise ValueError("Lower bound of the prior must be smaller than the upper bound")

        self.min = min
        self.max = max
        self.reflect = False
        pass
    
    def logprob(self,theta):
        """
        This method returns the log probability of the distribution - in this 
        case, an (arbitrary, for the purpose of likelihood optimization) 
        constant. More explicitely, if x is our parameter and log10(x) is uniform,
        then p(log10(x)) = const, p(x) = p(log10(x))*dlog10(x)/dx = const/x.
        Therefore, the log-probability is (minus a constant)
        log10(p(x)) = log10(1/x) = -log10(x). 
        
        Parameters:
        -----------
        theta: float 
            The parameter value for which the likelihood is to be computed. 
            
        Returns:
        --------
            The value of the likelihood for the input parameter.
        """        
        
        if self.min < theta < self.max:
            return -np.log10(theta)
        return -np.inf

    def transform_prior(self,quantile):
        """
        This method returns the parameter value defined from the quantile of  
        the cumulative distribution defined in the prior. It effectively 
        converts an interval from 0-1 to one that samples the prior distribution
        and is used by nested sampling to define prior likelihoods. 
        
        Parameters:
        -----------
        quantile: float, 0-1 
            The quantile for which the prior is to be computed 
            
        Returns:
        --------
        prior: float 
            The parameter value for the chosen parameter prior distribution. 
        """

        prior = 10.**(self.min + (self.max-self.min) * quantile)
        return prior 

class priorNormal():
    """
    This class is used to compute a normal prior distribution during Bayesian
    sampling, for a given model parameter. 
    
    Parameters:
    -----------
    sigma: float 
        The standard deviation of the distribution. 
        
    mu: float 
        The expectation of the distribution. 
    """    
    
    def __init__(self,mu,sigma):
        if sigma <= 0:
            raise ValueError("Standard deviation of the prior must be positive")

        self.mu = mu
        self.sigma = sigma
        self.reflect = False
        self.distribution = scipy.stats.norm(self.mu, self.sigma)
        pass 

    def logprob(self,theta):
        """
        This method returns the log probability of the distribution for the 
        given parameter theta.
        
        Parameters:
        -----------
        theta: float 
            The parameter value for which the likelihood is to be computed. 
            
        Returns:
        --------
            The value of the likelihood for the input parameter.
        """
        
        logprior = -0.5*(theta-self.mu)**2/self.sigma**2-0.5*np.log(2.*np.pi*self.sigma**2)
        return logprior

    def transform_prior(self,quantile):
        """
        This method returns the parameter value defined from the quantile of  
        the cumulative distribution defined in the prior. It effectively 
        converts an interval from 0-1 to one that samples the prior distribution
        and is used by nested sampling to define prior likelihoods. 
        
        Parameters:
        -----------
        quantile: float, 0-1 
            The quantile for which the prior is to be computed 
            
        Returns:
        --------
        prior: float 
            The parameter value for the chosen parameter prior distribution. 
        """
        prior = self.distribution.ppf(quantile)
        return prior 

class priorLogNormal():
    """
    This class is used to compute a lognormal prior distribution during Bayesian
    sampling, for a given model parameter. 
    
    Parameters:
    -----------
    sigma: float 
        The standard deviation of the distribution. 
        
    mu: float 
        The expectation of the distribution. 
    """    
    
    def __init__(self,mu,sigma):
        if sigma <= 0:
            raise ValueError("Standard deviation of the prior must be positive")

        self.mu = mu
        self.sigma = sigma
        self.reflect = False
        self.distribution = scipy.stats.lognorm(self.mu, self.sigma)
        pass 

    def logprob(self,theta):
        """
        This method returns the log probability of the distribution for the 
        given parameter theta.
        
        Parameters:
        -----------
        theta: float 
            The parameter value for which the likelihood is to be computed. 
            
        Returns:
        --------
            The value of the likelihood for the input parameter.
        """
        logprior = -0.5*(np.log(theta)-self.mu)**2/self.sigma**2-0.5*np.log(2.*np.pi*self.sigma**2/theta**2)
        return logprior

    def transform_prior(self,quantile):
        """
        This method returns the parameter value defined from the quantile of  
        the cumulative distribution defined in the prior. It effectively 
        converts an interval from 0-1 to one that samples the prior distribution
        and is used by nested sampling to define prior likelihoods. 
        
        Parameters:
        -----------
        quantile: float, 0-1 
            The quantile for which the prior is to be computed 
            
        Returns:
        --------
        prior: float 
            The parameter value for the chosen parameter prior distribution. 
        """
        prior = self.distribution.ppf(quantile)
        return prior 

class priorTruncNormal():
    """
    This class is used to compute a truncated, normal prior distribution during 
    Bayesian sampling, for a given model parameter. 
    
    Parameters:
    -----------
    sigma: float 
        The standard deviation of the distribution. 
        
    mu: float 
        The expectation of the distribution. 

    min: float
        The minimum value below which the distribution is truncated.

    max: float
        The maximum value above which the distribution is truncated.
    """    
    
    def __init__(self,mu,sigma,min,max):
        if sigma <= 0:
            raise ValueError("Standard deviation of the prior must be positive")
        if min >= max:
            raise ValueError("Lower bound of the prior must be smaller than the upper bound")
                
        self.mu = mu
        self.sigma = sigma
        #due to the scipy conversion, min/max for them are in
        #units of sigma, NOT in actual model values like we're passing here
        self.min = (min-self.mu)/self.sigma
        self.max = (max-self.mu)/self.sigma
        self.reflect = False
        self.distribution = scipy.stats.truncnorm(a=self.min,b=self.max,
                                                  loc=self.mu, scale=self.sigma)
        pass 

    def logprob(self,theta):
        """
        This method returns the log probability of the distribution for the 
        given parameter theta.
        
        Parameters:
        -----------
        theta: float 
            The parameter value for which the likelihood is to be computed. 
            
        Returns:
        --------
            The value of the likelihood for the input parameter.
        """
        
        logprior = self.distribution.logpdf(theta)
        return logprior

    def transform_prior(self,quantile):
        """
        This method returns the parameter value defined from the quantile of  
        the cumulative distribution defined in the prior. It effectively 
        converts an interval from 0-1 to one that samples the prior distribution
        and is used by nested sampling to define prior likelihoods. 
        
        Parameters:
        -----------
        quantile: float, 0-1 
            The quantile for which the prior is to be computed 
            
        Returns:
        --------
        prior: float 
            The parameter value for the chosen parameter prior distribution. 
        """
        prior = self.distribution.ppf(quantile)
        return prior 

def nested_sampling_priors(quantile_cube):
    """
    This function samples the prior distribution for the parameters stored in 
    the sampling_params global variable. This function is to be used to sample 
    the prior distribution when  using nested sampling algorithms. Given
    N parameters, the function maps an N-dimensional unitary cube to the 
    corresponding quantile in the distribution for each prior. For example, 
    given a parameter with a Gaussian prior centered at 2, and another parameter
    with a uniform prior from 3 to 5, the quantile cube (0.5,0.5) would return 
    (2, 4) - the center of each distribution.
    
    Parameters:
    -----------
    quantile_cube: np.array(float), 0-1
        An array containing the quantile for each prior distribution from which 
        to sample a new value
        
    Returns:
    --------
    params: np.array(float) 
        An array containing the parameter values sampled from the priors, given 
        the input quantiles. 
    """
        
    
    global sampling_priors
    global sampling_params 
    
    if sampling_priors is None:
        raise AttributeError("Priors definition missing")

    params = quantile_cube.copy()
    
    for i, val in enumerate(params):
        name = list(sampling_priors.keys())[i] 
        params[i] = sampling_priors[name].transform_prior(val)
        
    return params
       
def log_priors(theta, prior_dict):
    """
    This function computes the total log-probability of a set of priors, given 
    a st of input parameter values. This function is called automatically within 
    the likelihood methods labelled "mcmc".
    
    Parameters:
    -----------
    theta: np.array(float)
        An array of parameter values for which to compute the priors 
            
    prior_dict: dictionary
        A dictionary of prior objects, each containing a method called .logprob 
        which returns the log-probability given the input parameter value 
        
    Returns:
    --------
    logprior: float 
        A float containing the log-probability of the set of parameters, given 
        their priors. 
    """

    if len(theta) != len(prior_dict):
        raise AttributeError("Size of parameter and prior arrays do not match")

    logprior = 0
    for (key, obj), val in zip(prior_dict.items(), theta):        
        logprior = logprior + obj.logprob(val) 
    return logprior

def _reflect_theta(theta):
    """
    This function bounces one set of parameter values off the hard limits set by
    their priors, for the parameters whose priors ask for it. It is called 
    internally by the likelihood functions used for MCMC sampling. It requires 
    the global variables sampling_priors and sampling_params to have been set 
    beforehand.
    
    Parameters:
    -----------
    theta: np.array(float)
        An array containing one set of values for the free parameters of the 
        model.
        
    Returns:
    --------
    theta_r: np.array(float)
        The array of parameter values, with the values of the parameters whose 
        priors require it bounced off the limits of those priors.
        
    rejected: bool 
        A boolean which is True if one of the parameters lies so far outside of 
        the limits of its prior that the whole set is to be rejected.
    """

    global sampling_priors 
    global sampling_params

    theta_r = theta.copy()
    rejected = False
    index = 0
    for name in sampling_params:
        if sampling_params[name].vary is True:
            if sampling_priors[name].reflect is True:
                min_value = sampling_priors[name].min
                max_value = sampling_priors[name].max   
                #if we're too far from the original boundary just set a hard bound 
                #on the likelihood
                if (theta[index] < 0.5*min_value or theta[index] > 2.*max_value):
                    rejected = True
                    return theta_r, rejected
                #otherwise, just bounce the value off the limits
                ref_value = reflect_parameter(theta[index],min_value,max_value)
                theta_r[index] = ref_value
                index = index + 1          
    return theta_r, rejected

def _vectorized_mcmc_likelihood(theta,likelihood_function):
    """
    This function computes the log-likelihood, including priors, for several 
    sets of parameter values at once. It is called internally by the likelihood 
    functions used for MCMC sampling when they are passed more than one set of 
    parameters, e.g. by emcee running with vectorize=True. The priors are 
    evaluated one set at a time, because the model evaluation is what dominates 
    the run time; sets that are rejected by the priors are not passed on to the 
    model at all.
    
    Parameters:
    -----------
    theta: np.array(float,float)
        An array of size (n_sets x n_free), containing one set of values of the 
        free parameters of the model per row.
        
    likelihood_function: function 
        The function used to compute the log-likelihood excluding the priors, 
        e.g. sampling_gaussian_likelihood or sampling_cash_likelihood.
        
    Returns:
    --------
    likelihood: np.array(float)
        An array of size (n_sets), containing the log-likelihood of each set of 
        input parameter values.
    """

    global sampling_priors

    n_sets = np.shape(theta)[0]
    theta_r = np.zeros(np.shape(theta))
    logpriors = np.zeros(n_sets)
    for index in range(n_sets):
        theta_r[index], rejected = _reflect_theta(theta[index])
        if rejected is True:
            logpriors[index] = -np.inf
        else:
            logpriors[index] = log_priors(theta_r[index],sampling_priors)

    likelihood = np.full(n_sets,-np.inf)
    sampled = np.isfinite(logpriors)
    if np.any(sampled):
        likelihood[sampled] = (likelihood_function(theta_r[sampled]) +
                               logpriors[sampled])
    return likelihood

def sampling_cash_likelihood(theta):
    """
    This function computes the log-likelihood of Poisson-distributed data 
    excluding priors, for a given set of parameter values theta. This is the 
    likelihood that should be passed to nested sampling algorithms, which 
    evaluate the priors separately from the likelihood. It requires the global 
    variables sampling_names, sampling_params, sampling_data, sampling_noise,
    sampling_exp, and sampling_bins beforehand.  
    
    Parameters:
    -----------
    theta: np.array(float) or np.array(float,float)
        An array of parameter values for which to compute the log likelihood. If 
        the model was set up for vectorized evaluations, this can also be an 
        array of size (n_sets x n_free) containing one set of parameter values 
        per row. 
        
    Returns:
    --------
    likelihood: float or np.array(float)
        The value of the summed Cash log-likelihood for the given parameter 
        values, or one value per set of parameter values if several sets were 
        passed.
    """

    global sampling_names
    global sampling_params
    global sampling_data
    global sampling_model 
    global sampling_noise
    global sampling_exp
    global sampling_bins
    global sampling_vectorize

    #when computing the likelihood of several sets of parameters at once, the 
    #values are passed to the fitter object directly rather than through the 
    #lmfit Parameters object
    theta = np.asarray(theta)
    if np.ndim(theta) == 2:
        if sampling_vectorize is not True:
            raise AttributeError(("Vectorized model evaluations are not enabled;"
                                  " enable them with the set_sampling_model or"
                                  " initialise_mcmc functions"))
        model = sampling_model(params=theta,vectorize=True)
    else:
        for name, val in zip(sampling_names, theta):
            sampling_params[name].value = val    
        model = sampling_model(params=sampling_params) 
 
    if isinstance(sampling_data, np.ndarray):
        residual = cstat(sampling_data,model,sampling_exp,sampling_bins,sampling_noise,summed=True)
    else:
        residual = 0
        for index in range(len(sampling_data)):
            if index == 0:
                bins_old = 0
            else:
                bins_old = bins_new
            bins_new = bins_old + len(sampling_data[index])
            residual = residual+ cstat(sampling_data[index],
                                       model[bins_old:bins_new],
                                       sampling_exp[index],sampling_bins[index],
                                       sampling_noise[index],
                                       summed=True)                         
    likelihood = -residual
    
    return likelihood

def mcmc_cash_likelihood(theta):
    """
    This function computes the log-likelihood of Poisson-distributed data, and 
    including priors, for a given set of parameter values theta. This is the 
    likelihood that should be passed to MCMC sampling algorithms, which evaluate
    the priors together with the likelihood. It requires the global variables 
    sampling_priors, sampling_names, sampling_params, sampling_data, 
    sampling_noise, sampling_exp, and sampling_bins beforehand. 
    
    Parameters:
    -----------
    theta: np.array(float) or np.array(float,float)
        An array of parameter values for which to compute the log likelihood. If 
        the model was set up for vectorized evaluations, this can also be an 
        array of size (n_sets x n_free) containing one set of parameter values 
        per row - which is the format emcee uses when it is run with 
        vectorize=True. 
        
    Returns:
    --------
    likelihood: float or np.array(float)
        The value of the summed Cash log-likelihood for the given parameter 
        values, or one value per set of parameter values if several sets were 
        passed.
    """
    
    global sampling_priors 
    global sampling_params

    #when several sets of parameters are passed at once, both the priors and the 
    #likelihood are computed for every set
    theta = np.asarray(theta)
    if np.ndim(theta) == 2:
        return _vectorized_mcmc_likelihood(theta,sampling_cash_likelihood)

    #reflect parameters before computing the priors 
    theta_r, rejected = _reflect_theta(theta)
    if rejected is True:
        return -np.inf
    
    logpriors = log_priors(theta_r, sampling_priors)
    
    if not np.isfinite(logpriors):
        return -np.inf        

    likelihood = sampling_cash_likelihood(theta_r) + logpriors                      
    return likelihood

def sampling_gaussian_likelihood(theta):
    """
    This function computes the log-likelihood, using a Gaussian distribution
    and including priors, for a given set of parameter values theta. This is the 
    likelihood that should be passed to nested sampling algorithms, which 
    evaluate the priors separately from the likelihood. It requires the user to 
    have set the global variables sampling_names, sampling_params, 
    sampling_data, sampling_data_err and sampling_model beforehand. 
    
    Parameters: 
    -----------
    theta: np.array(float) or np.array(float,float)
        An array of parameter values for which to compute the log likelihood. If 
        the model was set up for vectorized evaluations, this can also be an 
        array of size (n_sets x n_free) containing one set of parameter values 
        per row. 
        
    Returns:
    --------
    likelihood: float or np.array(float)
        The value of the chi-square log-likelihood for the given parameter 
        values, or one value per set of parameter values if several sets were 
        passed.
    """   

    global sampling_names 
    global sampling_params
    global sampling_data
    global sampling_data_err
    global sampling_noise
    global sampling_noise_err
    global sampling_model     
    global sampling_vectorize

    #when computing the likelihood of several sets of parameters at once, the 
    #values are passed to the fitter object directly rather than through the 
    #lmfit Parameters object
    theta = np.asarray(theta)
    if np.ndim(theta) == 2:
        if sampling_vectorize is not True:
            raise AttributeError(("Vectorized model evaluations are not enabled;"
                                  " enable them with the set_sampling_model or"
                                  " initialise_mcmc functions"))
        model = sampling_model(params=theta,vectorize=True)
    else:
        for name, val in zip(sampling_names, theta):
            sampling_params[name].value = val 
        model = sampling_model(params=sampling_params)

    #flatten arrays if necessary
    if isinstance(sampling_data, list):
        data = []
        for array in sampling_data:
            data.extend(array)
        data = np.asarray(data)
        
        err = []
        for array in sampling_data_err:
            err.extend(array)
        err = np.asarray(err)
        
        noise_err = [] 
        for array in sampling_noise_err:
            noise_err.extend(array)
        noise_err = np.asarray(noise_err)
    else:
        data = sampling_data
        err = sampling_data_err 
        noise_err = sampling_noise_err
    
    if noise_err is not None:
        err = np.sqrt(err**2+noise_err**2)

    residual = (data-model)/err
    statistic = -0.5*np.sum(residual**2,axis=-1)
    
    return statistic 

    
def mcmc_gaussian_likelihood(theta):
    """
    This function computes the log-likelihood, using a Gaussian distribution
    and including priors, for a given set of parameter values theta. It requires
    the user to have set the global variables sampling_priors, sampling_names, 
    sampling_params, sampling_data, sampling_data_err and sampling_model beforehand. 
    
    Parameters: 
    -----------
    theta: np.array(float) or np.array(float,float)
        An array of parameter values for which to compute the log likelihood. If 
        the model was set up for vectorized evaluations, this can also be an 
        array of size (n_sets x n_free) containing one set of parameter values 
        per row - which is the format emcee uses when it is run with 
        vectorize=True. 
        
    Returns:
    --------
    likelihood: float or np.array(float)
        The value of the chi-square log-likelihood for the given parameter 
        values, or one value per set of parameter values if several sets were 
        passed.
    """    

    global sampling_priors
    global sampling_names 
    global sampling_params
    global sampling_data
    global sampling_data_err
    global sampling_noise
    global sampling_noise_err
    global sampling_model 

    #when several sets of parameters are passed at once, both the priors and the 
    #likelihood are computed for every set
    theta = np.asarray(theta)
    if np.ndim(theta) == 2:
        return _vectorized_mcmc_likelihood(theta,sampling_gaussian_likelihood)
    
    #reflect parameters before computing the priors 
    theta_r, rejected = _reflect_theta(theta)
    if rejected is True:
        return -np.inf
    
    logpriors = log_priors(theta_r, sampling_priors)

    if not np.isfinite(logpriors):
        return -np.inf        

    likelihood = sampling_gaussian_likelihood(theta_r) + logpriors
    
    return likelihood
   
def process_emcee(sampler,labels=None,discard=2000,thin=100,values=None,get_autocorr=True):
    """
    Given a sampler emcee EnsamleSampler object, this function calculates and 
    prints the autocorrelation length, and plots the trace plots of the walkers, 
    the acceptance fraction, and the corner plot for the posteriors. 
    
    This function is meant for a quick look at the output of a chain, rather 
    than for publication quality plots. All the plots produced by this function 
    have more customization options than the default ones used here.  
    
    Parameters:
    -----------
    sampler: emcee.EnsamleSampler
        The sampler from which to plot the data 
    
    labels: list(str) 
        A list of strings to use for naming the parameters in both the trace and
        corner plots 
        
    discard: int, default 2000
        The number of steps used to define the burn-in period 
        
    thin: int, default 100
        Use one every "thin" steps in the chain. Used to make plots clearer. 
        
    values: np.array(float), default None
        An array of parameter values used to show the best fit or "true" value 
        of each parameter in the corner plot. 
    """

    #print auto correlation lengths 
    if get_autocorr is True:
        tau = sampler.get_autocorr_time()
        with np.printoptions(threshold=np.inf):
            print("Autocorrelation lengths: ",tau)
    
    #print trace plots
    if labels is not None and len(labels) != ndim:
        raise ValueError("Size of labels does not match the number of parameters")
    ndim = sampler.ndim
    size = math.ceil(14/9*ndim)
    fig, axes = plt.subplots(ndim, figsize=(9, size), sharex=True)
    samples = sampler.get_chain(discard=discard, thin=thin)    
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        if labels is not None:
            ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)    
    axes[-1].set_xlabel("Step number");
    
    #print acceptance fraction
    frac = sampler.acceptance_fraction
    nwalkers = len(frac)    
    fig, ax = plt.subplots(1, figsize=(6, 4), sharex=True)
    ax.scatter(np.linspace(0,nwalkers,nwalkers), frac, marker='o', alpha=0.8,color='black')
    ax.set_xlim(-1, nwalkers+1)
    ax.set_ylabel("Acceptance fraction")
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylim(0.9*np.min(frac),1.1*np.max(frac))    
    ax.set_xlabel("Walker");
    
    #corner plot,values from user input if desired         
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    fig = corner.corner(flat_samples, labels=labels, truths=values);
    return
