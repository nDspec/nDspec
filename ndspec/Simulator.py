import numpy as np
from stingray.simulator import simulator
from pyfftw.interfaces.numpy_fft import (
    fft,
    fftfreq,
)

def simulate_lightcurve(psd_obj,obs_time,dt,countrate,rms=None,
                        params=None):
    """
    This function is used to simulate a lightcurve of the set model
    for a given set of parameters for a given timespan at a given
    time resolution. By default, this will simulate using the specified
    model and model parameters and will self-consistently calculate the
    root mean square of the lightcurve from the power spectrum model.
    The count rate must always be provided, as the mean flux of the
    lightcurve cannot be determined from the model alone. This method
    does not take into account the instrument response, and therefore
    all lightcurves simulated with this method will be mono-energetic
    and will not include any energy-dependent effects.

    Practically, this method is a wrapper for the stingray simulator
    module (https://docs.stingray.science/en/stable/api.html#simulator).    
    
    Parameters:
    -----------       
    psd_obj: ndspec.PowerSpectrum
        An instance of the PowerSpectrum class, which contains the model
        to use for simulating the lightcurve. This object is used to evaluate 
        the power spectrum at the frequencies measurable by the observation 
        time and time resolution.

    obs_time: float
        The total observation time, in seconds, over which the lightcurve
        is simulated. This is used to determine the number of bins in the
        simulated lightcurve.

    dt: float
        The time resolution of the simulated lightcurve, in seconds. This
        determines the time binning of the simulated lightcurve.

    countrate: float
        The mean count rate of the simulated lightcurve, in counts per
        second. This is used to set the mean flux of the simulated lightcurve.

    rms: float, default None
        The root mean square of the simulated lightcurve, which is used to
        set the variability of the simulated lightcurve. By default, the rms
        is calculated from the power spectrum model. If a specific rms value 
        is provided, it will be used instead.
                    
    params: lmfit.Parameters, default None
        The parameter values to use in evaluating the model. If none are 
        provided, the model_params attribute of the FitTimeAvgSpectrum 
        is used.
        
    Returns:
    --------
    lightcurve: stingray.lightcurve.Lightcurve
        The resulting lightcurve object, containing the simulated
        lightcurve of the model evaluated over the given Fourier frequency
        array, for the given input parameters.
    """
    if psd_obj.model is None:
        raise AttributeError("No model defined. Please define a model before simulating a lightcurve.")
    if psd_obj.freqs is None:
        raise AttributeError("No frequency grid defined. Please set a frequency grid before simulating a lightcurve.")
    if params is None:
        params = psd_obj.model_params
    else:
        psd_obj.set_model(psd_obj.model,params)

    # Transform the observation time and time resolution into a number of bins
    # and a frequency grid for the simulation
    N = int(obs_time/dt)
    w = np.fft.fftfreq(N, d=dt)
    w = w[w>0]

    #simulate
    psd_obj.compute_psd(params=params,freq=w)
    power_spectrum = psd_obj.power_spec

    if rms is None:
        # Calculate the rms from the power spectrum, integrating from 0
        bins = np.append(0,w)
        rms = np.sqrt(np.sum(power_spectrum**2*np.diff(bins)))

    mean_flux = countrate * dt  # mean count rate per bin
    sim = simulator.Simulator(N=N, mean=mean_flux, dt=dt, rms=rms,poisson=True)
    # Simulate
    lc = sim.simulate(power_spectrum)
    return lc

def simulate_lag_energy(response,time_avg_model,cross_model,
                       bkg_rate,freq_bounds,ref_bounds,coh,pow,exposure):
    """
    This method will simulate a lag-energy spectrum based a set of user defined 
    models. The methodology used here is identical to that described in section 
    3 of Ingram et al. 2022, https://ui.adsabs.harvard.edu/abs/2022MNRAS.509..619I/abstract.
    
    Note that users must be extremely careful in their assumption in order for 
    the simulation to make sense. In particular, in order to calculate the noise 
    and therefore error bars correctly, they must take care that the models for 
    the time-averaged spectrum and cross spectrum passed produce consistent 
    absolute rms. 

    Parameters
    ----------- 
    response: nDspec.ResponseMatrix
        The response matrix to be used in the simulation. This should always be 
        re-binned to a coarse channel grid, as is standard for all lag-energy 
        spectra. 
        
    time_avg_model: LMFit.CompositeModel
        A Model or CompositeModel LMFit object that stores the assumed
        time-averaged spectrum of the source. Necessary to calculate the noise 
        and lag errors correctly.
        
    cross_model: nDspec.CrossSpectrum
        An nDspec energy-dependent cross spectrum model from which to derive 
        the lag spectra, as well as the real and imaginary parts of the cross 
        spectrum (as required for calculating the errors). This object MUST 
        contain the computed cross spectrum beforehand.
        
    bkg_rate: np.array(float) 
        A Numpy array containing the background count rate in each channel of 
        the assumed instrument response. 
        
    freq_bounds: (float,foat)
        The minimum and maximum Fourier frequencies over which to calculate the 
        lag-energy spectrum. 
            
    ref_bounds: (float,float)
        The energy bounds of the reference band used to calculate the cross 
        spectrum. These must be identical to those used to calculate the model 
        in the cross_model object. 
        
    coh: float 
        The assumed coherence in the Fourier frequency interval over which the 
        lag spectrum is to be simulated. 
        
    pow: float 
        The assumed power in fractional rms in the Fourier frequency interval 
        over which the lag spectrum is to be simulated.
        
    exposure: float 
        The total exposure time in seconds.
    
    Returns
    --------
    lagsim: np.array(float)
        An array containing the lag values in each energy channel. 
        
    dlag: np.array(float)
        An array containing the error on the lags in each energy channel.
        
    response.emin, response.emax: np.array(float) 
        The lower and upper energy bounds of each energy channel.    
    """
    #convolve our time-averaged and cross spectra with the response
    spectrum_model = response.convolve_response(time_avg_model)
    convolved_model = response.convolve_response(cross_model)    
    #get the lags, real and imaginary parts of our convolved cross spectrum
    lag_model = convolved_model.lag_energy(freq_bounds)    
    real_model = convolved_model.real_energy(freq_bounds)
    imag_model = convolved_model.imag_energy(freq_bounds)
    
    #now we calculate the power and noise in the reference band, given the models we have
    ref_ilo = np.argmin(np.abs(response.emin-ref_bounds[0]))
    ref_ihi = np.argmin(np.abs(response.emax-ref_bounds[1]))
    # Calculate background in reference band - the +1 is necessary because the bounds are open rather than closed, ie [,) rather than [,]
    br = np.sum(bkg_rate[ref_ilo:ref_ihi+1])
    # Calculate reference band power (absolute rms^2)
    ref_pow = pow*(freq_bounds[1]+freq_bounds[0])*0.5
    Pr = ref_pow * np.sum(real_model[ref_ilo:ref_ihi+1])
    # Calculate reference band Poisson noise (absolute rms^2)
    mur = np.sum(spectrum_model[ref_ilo:ref_ihi+1])
    # Calculate total noise
    Prnoise = 2.0 * (br + mur)

    #next we set up the channels of interest
    sub_Elo = response.emin
    sub_Ehi = response.emax    
    #we already have rebinned the matrix in the channels appropriately, so we only need one set of
    #indexes rather than two like for the reference which can cover multiple channels
    sub_i = np.array([np.argmin(np.abs(response.emin - e)) for e in sub_Elo],dtype=int)
    
    #now finally run the simulation, given the input coherence and exposure   
    lagsim = []
    dlag = []

    for index in sub_i:
        #calculate the total noise in each subject band
        mus = np.sum(spectrum_model[index:index+1])
        bs = np.sum(bkg_rate[index:index+1])
        Psnoise = 2*(mus+bs)
        #calculate the cross spectrum contributions for the error
        realcross = np.sum(real_model[index:index+1])
        imagcross = np.sum(imag_model[index:index+1])
        crosssquared = pow**2*(realcross**2+imagcross**2)
        #calculate the error
        lag_err = 1+Prnoise/Pr
        lag_err *= (crosssquared*(1-coh) - Psnoise*Pr)
        lag_err /= (crosssquared*coh)
        lag_err /= (2*exposure * (freq_bounds[1]-freq_bounds[0]))
        lag_err = np.abs(lag_err)**0.5
        lag_err /= (2.0*np.pi*np.diff(freq_bounds))
        #generate lags with Gaussian noise rescaled by the error
        lag_val = lag_model[index] + np.random.normal(loc=0,scale=1,size=1) * lag_err
        dlag = np.append(dlag,lag_err)
        lagsim = np.append(lagsim,lag_val)
    
    return lagsim,dlag, response.emin, response.emax


def simulate_time_averaged(res_obj,model,params,exposure_time=None):
    """
    This method simulates a time-averaged spectrum given a set of parameters, 
    by evaluating the model and folding it through the response. It is used 
    to generate synthetic spectra for testing purposes. 
    
    Parameters:
    -----------
    res_obj: ndspec.ResponseMatrix
        An instance of the ResponseMatrix class, which contains the response 
        matrix of the instrument for which to simulate the data 
        
    model: lmfit.model or lmfit.compositemodel 
        An instance of an LMFit model object which contains the model to use 
        to simulate the spectrum

    params: lmfit.Parameters, default None
        The parameter values to use in evaluating the model. If none are 
        provided, the model_params attribute is used.
        
    exposure_time: float, default None
        The exposure time to use for the simulation. If None, the exposure
        time stored in the response matrix is used. This is used to convert
        the model count rate to expected counts in each channel.

    ear: np.array(float), default None
        The array of photon energy channel edges over which to evaluate the model. 
        If none are provided, the same grid contained in the instrument response
        is used. 
    
    Returns:
    --------
    simulated_spectrum: np.array(float)
        The simulated spectrum evaluated over the noticed energy channels
        and Poisson sampled. The spectrum is in units of counts/channel.
    """

    #set the energy grids as appropriate
    ear = np.append(res_obj.energ_lo,res_obj.energ_hi[-1])   
    energ = 0.5*(res_obj.energ_hi+res_obj.energ_lo)
    energ_bounds = res_obj.energ_hi-res_obj.energ_lo

    # evaluate the model with the given parameters and fold it through the response
    simulated_spectrum = model.eval(params,energ=energ,ear=ear)*energ_bounds
    simulated_spectrum = res_obj.convolve_response(simulated_spectrum,units_in="xspec",units_out="channel") 
    # multiply by exposure time to get expected counts
    if exposure_time is None:
        exposure_time = res_obj.exposure_time
    simulated_spectrum = simulated_spectrum*exposure_time 
    # Poisson sample the spectrum
    simulated_spectrum = np.random.poisson(simulated_spectrum)
 
    return simulated_spectrum
