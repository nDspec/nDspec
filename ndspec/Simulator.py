import numpy as np
from stingray.simulator import simulator

def simulate_lightcurve(Fitobj,obs_time,dt,countrate,rms=None,
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
    Fitobj: ndspec.FitPowerSpectrum
        An instance of the FitPowerSpectrum class, which contains the model
        to use for simulating the lightcurve. The FitPowerSpectrum object 
        must have a model defined and a frequency grid set before calling 
        this function. The model is used to evaluate the power spectrum 
        at the frequencies measurable by the observation time and time
        resolution.

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
    if Fitobj.model is None:
        raise AttributeError("No model defined. Please define a model before simulating a lightcurve.")
    if Fitobj.freqs is None:
        raise AttributeError("No frequency grid defined. Please set a frequency grid before simulating a lightcurve.")
    if params is None:
        params = Fitobj.model_params

    # Transform the observation time and time resolution into a number of bins
    # and a frequency grid for the simulation
    N = int(obs_time/dt)
    w = np.fft.rfftfreq(N, d=dt)[1:]
    #simulate
    power_spectrum = Fitobj.eval_model(params=params,freq=w)

    if rms is None:
        # Calculate the rms from the power spectrum
        rms = np.sqrt(np.sum(power_spectrum**2*np.diff(w)))

    mean_flux = countrate * dt  # mean count rate per bin
    sim = simulator.Simulator(N=N, mean=mean_flux, dt=dt, rms=rms,poisson=True)
    # Simulate
    lc = sim.simulate(power_spectrum)
    return lc

def simulate_time_lags(fitobj,ref_Elo,ref_Ehi,sub_Elo,sub_Ehi,Texp,
                              coh2,pow,time_avg_model,
                              bkg_file_path=None,params=None):
    r"""
    This method will simulate a lag spectrum based on the model. The model
    must be able to evaluate the cross spectrum and a background file must be provided.
    You must also provide the energy bounds of the reference and subject bands,
    the exposure time, the coherence squared value, and the power in rms/mean$^2$/Hz units
    ($\alpha_{\nu}$), as well as a time-averaged model that shares the same physical
    assumptions (and thus share relevant parameters) as the set cross-spectrum model. 
    The method will return a one-dimensional array containing the simulated lags for 
    each energy channel, based on the model and the specified parameters. 

    For further explanation for how the lag spectrum is simulated, see section 3 of
    Ingram et al. 2022, https://ui.adsabs.harvard.edu/abs/2022MNRAS.509..619I/abstract.

    Parameters
    ----------- 
    fitobj: ndspec.FitCrossSpectrum
        An instance of the FitCrossSpectrum class, which contains the model
        to use for simulating the lag spectrum. The FitCrossSpectrum object
        must have a model defined, a frequency grid set, and an instrument
        response set before calling this function. 

    ref_Elo: float
        The lower energy bound of the reference band in keV.

    ref_Ehi: float
        The upper energy bound of the reference band in keV.

    sub_Elo: float
        The lower energy bound of the subject band in keV.

    sub_Ehi: float 
        The upper energy bound of the subject band in keV.

    Texp: float
        The exposure time in seconds for which the lag spectrum is simulated.

    coh2: float
        The coherence squared value, which is a measure of the correlation 
        between the reference and subject bands.
    
    pow: float
        The power in rms/mean$^2$/Hz units ($\alpha_{\nu}$), which is used to scale the
        cross spectrum.

    time_avg_model: lmfit.Model or lmfit.CompositeModel
        A model that evaluates the time-averaged power spectrum, which is 
        used to calculate the background noise in the reference and subject 
        bands. This model should share the same physical assumptions as the
        model used to evaluate the cross spectrum and share all relevant
        physical parameters

    bkg_file_path: str, optional
        The path to the background file containing the background counts 
        for each energy channel. If not provided, the method will default to
        the background file already set in the class. A background file must be
        provided to simulate the lag spectrum.
    
    params: lmfit.Parameters, optional
        The parameters to use for evaluating the model. If not provided, the
        default parameters stored in the model_params attribute will be used.
    
    Returns
    --------
    lagsim: np.array(float)
        A one-dimensional array containing the simulated lags for each energy 
        channel, based on the model and the specified parameters. The size of 
        this array is equal to the number of energy channels defined in the 
        instrument response matrix.
    
    """
    if fitobj.model is None:
        raise AttributeError("Model not set. Please set a model before simulating.")
    if bkg_file_path is None and fitobj.needbkg is True:
        raise AttributeError("Background file not set. Please provide a background file to simulate the lag spectrum.")
    if params is None and fitobj.model_params is None:
        raise AttributeError("Model parameters not set. Please provide parameters to simulate the lag spectrum.")
    if fitobj.response is None:
        raise AttributeError("Instrument response not set. Please set the instrument response before simulating.")
    if fitobj.freqs is None:
        raise AttributeError("Frequency grid not set. Please set the frequency grid before simulating.")
    
    ear = fitobj.energs
    ne = len(ear)
    flo = fitobj.freqs[:-1]
    fhi = fitobj.freqs[1:]
    fc = (flo+fhi)/2.
    # Read in background array if needed
    if fitobj.needbkg or bkg_file_path is not None:
        fitobj.set_background(bkg_file_path)
        fitobj.needbkg = False

    #saves units and dependence to reset after simulation
    reset_units = fitobj.units
    reset_dependence = fitobj.dependence

    #evaluate the folded lags model
    fitobj.set_product_dependence("energy")
    fitobj.set_coordinates("lags")
    lags = fitobj.eval_model(params=params,fold=True)

    #evaluate the cross spectrum
    fitobj.set_coordinates("cartesian")
    cross_spectrum = fitobj.eval_model(params=params)

    #evaluate the time-averaged spectrum
    time_avg_spectrum = time_avg_model.eval(params=params)
    #finds the closest eneergy channels to the reference band edges
    #ilo is reference band channel number low, ihi is channel number high
    ilo = np.argmin(np.abs(ear-ref_Elo))
    ihi = np.argmin(np.abs(ear-ref_Ehi))
    #find the closest energy channels to the subject band edges
    #Elo is subject band channel number low, Ehi is channel number high
    Elo = np.argmin(np.abs(ear-sub_Elo))
    Ehi = np.argmin(np.abs(ear-sub_Ehi))

    # Calculate background in reference band
    br = np.sum(fitobj.bkg_rate[ilo:ihi+1])

    # Calculate background in subject
    bs = np.sum(fitobj.bkg_rate[Elo:Ehi+1])

    # Calculate reference band power (absolute rms^2)
    Pr = pow * np.sum(cross_spectrum[0,Elo:Ehi+1])
    # Calculate reference band Poisson noise (absolute rms^2)
    mur = np.sum(time_avg_spectrum[Elo:Ehi+1])
    # Calculate total noise
    Prnoise = 2.0 * (br + mur)

    # Loop through energy bins
    lagsim = np.zeros(ne)
    dlag = np.zeros(ne)
    for i in range(1,ne):
        mus = np.sum(time_avg_spectrum[ear[i-1]:ear[i]])
        Psnoise = 2.0 * (mus + bs[i])
        ReG = np.sum(cross_spectrum[0,ear[i-1]:ear[i]])
        ImG = np.sum(cross_spectrum[1,ear[i-1]:ear[i]])
        G2 = pow**2 * (ReG**2 + ImG**2)
        # Calculate error
        dlag[i] = 1.0 + Prnoise/Pr
        dlag[i] *= (G2*(1.0-coh2) + Psnoise*Pr)
        dlag[i] /= (coh2*G2)
        dlag[i] /= (2.0 * Texp * (fhi-flo))
        dlag[i] = np.sqrt(dlag[i])
        dlag[i] /= (2.0 * np.pi * fc)
        # Generate simulated data
        lagsim[i] = lags[i] + np.random.normal(loc=0,scale=1,size=1) * dlag[i]

    # Reset units and dependence
    fitobj.set_product_dependence(reset_dependence)
    fitobj.set_coordinates(reset_units)

    return lagsim


def simulate_time_averaged(fitobj,params=None,mask=False,exposure_time=None,ear=None):
    """
    This method simulates a time-averaged spectrum given a set of parameters, 
    by evaluating the model and folding it through the response. It is used 
    to generate synthetic spectra for testing purposes. 
    
    Parameters:
    -----------
    fitobj: ndspec.FitTimeAvgSpectrum
        An instance of the FitTimeAvgSpectrum class, which contains the model
        to use for simulating the spectrum. The FitTimeAvgSpectrum object must
        have a model defined and a response matrix set before calling this function.

    params: lmfit.Parameters, default None
        The parameter values to use in evaluating the model. If none are 
        provided, the model_params attribute is used.
        
    mask: bool, default False
        A boolean switch to choose whether to mask the model output to only 
        include the noticed energy channels, or to also return the ones 
        that have been ignored by the users. Default is False, so that
        the simulated spectrum is returned in the same energy grid as the
        full response matrix.

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
    if fitobj.response is None:
        raise AttributeError("No response matrix set. Please set a response matrix " \
        "before simulating a spectrum using either set_data() or set_response().")
    if fitobj.model is None:
        raise AttributeError("No model set. Please set a model before simulating a spectrum.")
    if params is None:
        params = fitobj.model_params
    if ear is None:
        ear = fitobj.ear
        energ = fitobj.energs 
        energ_bounds = fitobj.energ_bounds
    else:
        energ = 0.5*(ear[1:]+ear[:-1])
        energ_bounds = ear[1:]-ear[:-1]

    # evaluate the model with the given parameters and fold it through the response
    simulated_spectrum = fitobj.model.eval(params,energ=energ,ear=ear)*energ_bounds
    simulated_spectrum = fitobj.response.convolve_response(simulated_spectrum,units_in="xspec",units_out="channel") 
    # multiply by exposure time to get expected counts
    if exposure_time is None:
        exposure_time = fitobj.response.exposure_time
    simulated_spectrum = simulated_spectrum*exposure_time 
    # Poisson sample the spectrum
    simulated_spectrum = np.random.poisson(simulated_spectrum)
    #mask the simulated spectrum if desired 
    if mask is True:
        model = np.extract(fitobj.ebounds_mask,simulated_spectrum)   
    return simulated_spectrum
