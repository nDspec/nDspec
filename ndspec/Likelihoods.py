import numpy as np

def chisq(data,err,model,noise=None,noise_err=None,summed=False):
    """
    This function is used to calculate the traditional chi-squared statistic 
    drawn from Gaussian-distributed data. It also accounts for a background/noise 
    term, by subtracting the (optional) user-provided noise from the data, and 
    summing the uncertainties of the noise and data in quadrature. 
    
    Parameters:
    -----------
    data: np.array(float)
        The value of datapoints from which to calculate the statistic 
    
    err: np.array(float)
        The uncertainties on each data point 
    
    model: np.array(float)
        The model values to be compared to the data in the statistic 
    
    noise: np.array(float), default None
        The optional background values to be subtracted from the data when 
        calculating the statistic. If it is not specified, no subtraction is 
        performed
        
    noise_err: np.array(float), default None
        The optional background uncertainty values to be added in quadrature to 
        the error on the data. If it is not specified, no uncertainty is added.
        
    summed: bool, default False 
        A boolean to either return the summed statistic (True) or an array with 
        the contribution of each bin to the total statistic (False)
        
    Returns:
    --------
    chisq: np.array(float)
        Either a single value containing the summed statistic (if residuals is 
        True), or an array containing the contribution of each bin to the total 
        statistic (if residuals is False).
    """

    if (noise is None and noise_err is not None):
        raise ValueError("The background is not defined but its error is")
    if (noise is not None and noise_err is None):
        raise ValueError("The background is defined but its error is not")
    
    if noise_err is not None:
        err = np.sqrt(err**2+noise_err**2)
        chisq = (data-noise-model)/err
    else:
        chisq = (data-model)/err
    
    if summed is True:
        return np.sum(chisq)
    else:
        return chisq
    
def ratio(data,err,model,noise=None,noise_err=None,summed=True,bars=True):
    """
    This function is used to calculate the ratio of the data to a model. If a 
    background and its uncertainty are provided, these are subtracted from the 
    data and summed in quadrature to the error, respectively.
    
    Parameters:
    -----------
    data: np.array(float)
        The value of datapoints from which to calculate the ratio 
    
    err: np.array(float)
        The uncertainties on each data point 
    
    model: np.array(float)
        The model values to be compared to the data in the ratio 
    
    noise: np.array(float), default None
        The optional background values to be subtracted from the data when 
        calculating the ratio. If it is not specified, no subtraction is 
        performed
        
    noise_err: np.array(float), default None
        The optional background uncertainty values to be added in quadrature to 
        the error on the data. If it is not specified, no uncertainty is added.
        
    summed: bool, default False 
        A boolean to either return the summed ratios (False) or an array with 
        the ratio in each bin to the total statistic (True)
        
    bars: bool, default True 
        A boolean to control whether to return both an array with the ratios and 
        one with the corresponding error bars (True), or just the ratios (False).
        
    Returns:
    --------
    ratio: np.array(float)
        Either a single value containing the summed ratios (if summed is 
        True), or an array containing the ratio in each bin (if summed is False).
        
    bars: np.array(float) 
        An array containing the error bars of the data normalized by the model. 
        Used for ratio plots.
    """

    if noise is not None:
        data = data-noise 
    if noise_err is not None:
        err = np.sqrt(err**2+noise_err**2)

    ratio = data/model 
    
    if summed is True:
        return np.sum(ratio)

    if bars is True:
        bars = err/model 
        return ratio, bars
    else: 
        return ratio

def cstat(data,model,exp,widths,noise=None,summed=False):
    """
    This function calculates the Cash statistic (Cash 1979, DOI: 10.1086/156922)
    for Poisson-distributed count data (typically, a time-averaged X-ray 
    spectrum). Optionally, users can also provide a background count rate, which 
    is also assumed to be Poisson-distributed. The implementation is identical 
    to that of Xspec, with the additional assumption that the exposure times for 
    the background and data are identical. For more details of the math used, 
    see the Xspec documentation:
    https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node340.html
    
    Parameters:
    -----------
    data: np.array(float)
        The data for which to calculate the statistic. This MUST be provided in 
        units of counts/s/keV (e.g., identical to the array stored in a ndspec 
        FitTimeAvgSpectrum object).
        
    model: np.array(float)
        The model values for which to calculate the statistic. This MUST be 
        provided in units of counts/s/keV (e.g., identical to the array 
        calculated by the eval_model() method of a ndspec FitTimeAvgSpectrum
        object).
        
    exp: float 
        The exposure time (in seconds) of the data and background. 
        
    widths: np.array(float)
        An array of channel widths, in units of keV; this can come, for example, 
        from the difference between the "emax" and "emin" columns of a OGIP
        compatible response matrix file.
        
    noise: np.array(float), default None
        The background spectrum to be included in calculating the statistic; 
        thiis MUST be provided in units of counts/s/keV (e.g., identical to the 
        array stored in a ndspec FitTimeAvgSpectrum object).
        
    summed: bool, default False 
        A boolean to either return the summed statistic (True) or an array with 
        the contribution of each bin to the total statistic (False)
    
    Returns:
    --------
    cstat: np.array(float)
        Either a single value containing the summed statistic (if residuals is 
        True), or an array containing the contribution of each bin to the total 
        statistic (if residuals is False).        
    """       
    
    #fix the factor needed to convert to integrated counts, then calculate 
    #the model and data in integrated counts  
    conv_factor = exp*widths
    data = data*conv_factor
    model = model*conv_factor

    if noise is None:   
        cstat = model - data + data*(np.log(data)-np.log(model))        
    else:
        #convert the background array to counts 
        noise = noise*conv_factor       
        #see https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node340.html
        #if the bkg and source exposures are identical, our model array is
        #== ts*mi from the xspec docs which simplifies things
        #first we calculate the factor fi in the xspec docs
        cstat = np.zeros(len(data))
        bkg_num = np.zeros(len(data))
        bkg_den = np.zeros(len(data))
        
        test_sign = 2.*model - data - noise 
        #define the bkg model - fi in the xspec docs
        d_factor = np.sqrt(test_sign**2.+8.*model*noise)   
        
        mask = (test_sign>=0)        
        bkg_num[mask] = 2.*noise[mask]*model[mask]/(exp)
        bkg_den[mask] = test_sign[mask] + d_factor[mask]    
        bkg_num[~mask] = d_factor[~mask] - test_sign[~mask] 
        bkg_den[~mask] = 2.*exp             
        bkg_model = bkg_num/bkg_den*exp 
        
        #handle the exception of data being 0
        mask = (data==0)
        mask_model = model[mask]
        mask_noise = noise[mask]
        cstat[mask] = mask_model - mask_noise*np.log(0.5)

        #handle the exceptionof noise being 0 and low counts
        mask = (noise==0) & (test_sign<0)
        mask_model = model[mask]
        mask_data = data[mask] 
        cstat[mask] = -mask_model - mask_data*np.log(0.5)
        
        #handle the exception of noise being 0 and high counts 
        mask = (noise==0) & (test_sign>=0) 
        mask_model = model[mask]
        mask_data = data[mask] 
        cstat[mask] = (mask_model - mask_data + 
                      mask_data*(np.log(mask_data) - np.log(mask_model)))
       
        #handle the bins without exceptions
        mask = (data != 0) & (noise !=0)
        mask_model = model[mask]
        mask_data = data[mask]
        mask_noise = noise[mask]
        mask_bkg_model = bkg_model[mask]                
        cstat[mask] = (mask_model + 2.*mask_bkg_model - 
                       mask_data*np.log(mask_model+mask_bkg_model) -
                       mask_noise*np.log(mask_bkg_model) - 
                       mask_data*(1.-np.log(mask_data)) - 
                       mask_noise*(1.-np.log(mask_noise)))
    
    if summed is True:
        return 2.*np.sum(cstat)
    else:
        return 2.*cstat
