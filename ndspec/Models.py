import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

#required by the polarimetry models 
from .Polarimetry import PolarimetryProduct

colorscale = pl.cm.PuRd(np.linspace(0.,1.,5))

def lorentz(array,params):
    """
    This model is a Lorentzian function, defined identically to Uttley and 
    Malzac 2023. 

    Parameters:
    -----------
    array: np.array(float)
        The array (typically Fourier frequency) over which the Lorentzian is
        to be computed.

    params: array_like(float)
        The model parameters, in the following order:

        - f_pk: the peak frequency of the Lorentzian
        - q: the q-factor of the Lorentzian
        - rms: the normalization of the Lorentzian

    Output:
    -------
    model: np.array(float)
        A one-dimensional array of the same size as the input array.
    """

    if params.ndim == 1:
        f_pk = params[0]
        q = params[1]
        rms = params[2]
    elif params.ndim == 2:
        f_pk = params[:,0][:,np.newaxis]
        q = params[:,1][:,np.newaxis]
        rms = params[:,2][:,np.newaxis]
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    f_res = f_pk/(1.0+(1.0/(4.0*q**2)))**0.5
    r = rms/(0.5-np.arctan(-2.0*q)/np.pi)**0.5
    lorentz_num = (1/np.pi)*2*r**2*q*f_res
    lorentz_den = 4*q**2*(array-f_res)**2
    model = lorentz_num/(f_res**2+lorentz_den)
    return np.nan_to_num(model)

def cross_lorentz(array1,array2,params):
    """
    This model is a complex Lorentzian function, defined identically to Uttley 
    and Malzac 2023, and shifted by a fixed phase, defined identically to Mendez
    et al. 2023. This model is meant exclusively for fitting complex data like 
    cross spectra.

    Parameters:
    -----------
    array1: np.array(float)
        The array (typically energy) over which the model is tiled; the output 
        does not depend on this axis.

    array2: np.array(float)
        The Fourier frequency array over which the Lorentzian is to be
        computed.

    params: array_like(float)
        The model parameters, in the following order:

        - f_pk: the peak frequency of the Lorentzian
        - q: the q-factor of the Lorentzian
        - rms: the normalization of the Lorentzian
        - phase: the phase lag associated with the Lorentzian

    Output:
    -------
    twod_lorentz: np.array(complex), shape (len(array2),len(array1))
        A two-dimensional array, constant over array1 (which in the case of a  
        cross spectrum corresponds to the energy) and tiled across it.
    """
    n_energs = len(array1)
    n_freqs = len(array2)
    if params.ndim == 1:
        f_pk = params[0]
        q = params[1]
        rms = params[2]
        phase = params[3]
    elif params.ndim == 2:
        f_pk = params[:,0][:,np.newaxis]
        q = params[:,1][:,np.newaxis]
        rms = params[:,2][:,np.newaxis]
        phase = params[:,3][:,np.newaxis]
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    lorentz_arr = lorentz(array2,params)*np.exp(1j*phase)
    twod_lorentz = np.tile(lorentz_arr,n_energs).reshape((n_energs,n_freqs))
    twod_lorentz = np.transpose(twod_lorentz)
    return twod_lorentz

def powerlaw(array,params):
    """
    This model is a standard power-law.

    Parameters:
    -----------
    array: np.array(float)
        The array grid over which to compute the power-law.

    params: array_like(float)
        The model parameters, in the following order:

        - norm: the normalization of the power-law
        - slope: the slope of the power-law. Unlike in Xspec, this parameter
          does not implicitely assume a minus sign; it must be specified by
          the user.

    Output:
    -------
    model: np.array(float)
        A one-dimensional array of the same size as array, containing the
        power-law evaluated over it.
    """
    if params.ndim == 1:
        norm = params[0]
        slope = params[1]
        model = norm*array**slope
    elif params.ndim == 2:
        norm = params[:,0]
        slope = params[:,1]
        model = norm[:,np.newaxis]*np.power(array,slope[:,np.newaxis])
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    return model

def brokenpower(array,params):
    """
    This model is a smoothly broken powerlaw, defined identically to eq. 10 in 
    Ghisellini and Tavecchio 2009.

    Parameters:
    -----------
    array: np.array(float)
        The array grid over which to compute the broken power-law.

    params: array_like(float)
        The model parameters, in the following order:

        - norm: the normalization of the broken power-law
        - slope1: the slope of the broken power-law before the break
        - slope2: the slope of the broken power-law after the break
        - brk: the location of the break in the power-law

    Output:
    -------
    model: np.array(float)
        A one-dimensional array of the same size as array, containing the
        broken power-law evaluated over it.
    """
    if params.ndim == 1:
        norm = params[0]
        slope1 = params[1]
        slope2 = params[2]
        brk = params[3]
        scaled_array = array/brk
        num = norm*scaled_array**slope1
        den = 1.+scaled_array**(slope1-slope2)
        model = num/den
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        slope1 = params[:,1][:,np.newaxis]
        slope2 = params[:,2][:,np.newaxis]
        brk = params[:,3][:,np.newaxis]
        scaled_array = np.divide(array,brk)
        num = norm*np.power(scaled_array,slope1)
        den = 1.+np.power(scaled_array,slope1-slope2)
        model = np.divide(num,den)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    return model 

def gaussian(array,params):
    """
    This model is a Gaussian function.

    Parameters:
    -----------
    array: np.array(float)
        The array (typically energy) over which the Gaussian is defined.

    params: array_like(float)
        The model parameters, in the following order:

        - center: the centroid of the Gaussian
        - width: the width of the Gaussian
        - gauss_norm: the normalization of the Gaussian

    Output:
    -------
    line: np.array(float)
        A one-dimensional array of the same size as array, containing the
        Gaussian line evaluated over it.
    """
    if params.ndim == 1:
        center = params[0]
        width = params[1]
        gauss_norm = params[2]
        norm = (2.0*np.pi)**0.5*width
        shape = np.exp(-((array - center)/width)**2/2)
        line = gauss_norm*shape/norm 
    elif params.ndim == 2:
        center = params[:,0][:,np.newaxis]
        width = params[:,1][:,np.newaxis]
        gauss_norm = params[:,1][:,np.newaxis]
        norm = np.multiply(np.sqrt(2.0*np.pi),width)
        shape = np.exp(-np.power((array - center)/width,2.0)/2)
        line = gauss_norm*shape/norm 
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    return line

def bbody(array,params):
    """
    This model is a constant black body, identical to that included in Xspec.

    Parameters:
    -----------
    array: np.array(float)
        The array of energy bin centers over which the spectrum is defined.

    params: array_like(float)
        The model parameters, in the following order:

        - norm: the normalization of the black body, defined identically to
          that of the Xspec model
        - temp: the temperature, in keV

    Output:
    -------
    model: np.array(float)
        A one-dimensional array of the same size as array, containing the
        black body spectrum evaluated over it.
    """
    if params.ndim == 1:
        #boltzkamnn constant in kev
        norm = params[0]
        temp = params[1]
        renorm = 8.0525*norm/(temp**4)
        #safeguard against diverging exponentials, e.g. for low temperature BB
        #calculated at highx energy:
        with np.errstate(over='ignore', invalid='ignore'):
            planck = np.exp(array/temp)-1.
            planck[planck>1e20] = 1e20
            model = renorm*array**2/planck
        #if nans appear in the temperature/whatever, just set the bin to 0
        model = np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        temp = params[:,1][:,np.newaxis]
        renorm = 8.0525*norm/np.power(temp,4.)
        #safeguard against diverging exponentials, e.g. for low temperature BB
        #calculated at highx energy:
        with np.errstate(over='ignore', invalid='ignore'):
            planck = np.exp(array/temp)-1.
            planck[planck>1e20] = 1e20
            model = renorm*np.power(array,2.)/planck
        #if nans appear in the temperature/whatever, just set the bin to 0
        model = np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    return model

    
def varbbody(array,params):
    """
    This model is a variable black body, defined identically to Uttley and 
    Malzac 2023. It is meant for use in spectral-timing models.

    Parameters:
    -----------
    array: np.array(float)
        The array energy bin centers over which the spectrum is defined.

    params: array_like(float)
        The model parameters, in the following order:

        - norm: the normalization of the black body, defined identically to
          that of the Xspec model
        - temp: the temperature, in keV

    Output:
    -------
    model: np.array(float)
        A one-dimensional array of the same size as array, containing the
        black body spectrum evaluated over it.
    """
    if params.ndim == 1:
        #boltzkamnn constant in kev
        norm = params[0]
        temp = params[1]
        #safeguard against diverging exponentials, e.g. for low temperature BB
        #calculated at highx energy:
        with np.errstate(over='ignore', invalid='ignore'):
            planck = np.exp(array/temp)
            planck[planck>1e30] = 1e30
            denom = planck-1.
            renorm = 2.013*norm/(temp**5)*planck
            renorm[renorm>1e30] = 1e30
            model = renorm*array**3/denom**2
        #if nans appear in the temperature/whatever, just set the bin to 0
        model = np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        temp = params[:,1][:,np.newaxis]
        #safeguard against diverging exponentials, e.g. for low temperature BB
        #calculated at highx energy:
        with np.errstate(over='ignore', invalid='ignore'):
            planck = np.exp(array/temp)
            planck[planck>1e30] = 1e30
            denom = planck-1.
            renorm = 2.013*norm/np.power(temp,5.)*planck
            renorm[renorm>1e30] = 1e30
            model = renorm*np.power(array,3.)/np.power(denom,2)
        #if nans appear in the temperature/whatever, just set the bin to 0
        model = np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    return model     
    
def gauss_fred(array1,array2,params,return_full=False):
    """
    This model is a two-dimensional model for an impulse response function. The  
    time dependence is a fast rise, exponential decay pulse. The dependence over  
    the second axis (typically energy) is a Gaussian line narrowing over time 
    following a powerlaw. The total model is the product of the two dependences. 
    This model is meant exclusively for cross spectra.
    
    Parameters:
    -----------
    array1: np.array(float)
        The time array over which the pulse is defined.

    array2: np.array(float)
        The second array (typically energy) over which the model is defined.

    params: array_like(float)
        The model parameters, in the following order:

        - norm: the total model normalization
        - width: the initial width of the Gaussian
        - center: the centroid of the Gaussian
        - rise_t: the rise pulse timescale
        - decay_t: the decay pulse timescale
        - decay_w: the slope of the energy width power-law decay

    return_full: bool, default=False
        A boolean to choose whether to return just the two-dimensional model
        (done by default), or also the additional projections over the two
        model axes.

    Output:
    -------
    fred_pulse: np.array(float), shape (len(array2),len(array1))
        A two-dimensional array containing the impulse response function over
        energy and time; if params is two-dimensional, fred_pulse has shape
        (n_sets,len(array2),len(array1)) instead. 

    line_profile: np.array(float), optional
        The projection of fred_pulse over time, i.e. the time-averaged
        spectrum; only returned if return_full is True.

    pulse_profile: np.array(float), optional
        The projection of fred_pulse over energy, i.e. the energy-integrated
        pulse profile; only returned if return_full is True.
    """
    times = array1
    energy = array2
    if params.ndim == 1:
        norm = params[0]
        width = params[1]
        center = params[2]
        rise_t = params[3]
        decay_t = params[4]
        decay_w = params[5]
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma = np.nan_to_num(width*powerlaw(times/times[0],np.array([1.,decay_w])))
            sigma[0] = width
            fred_profile = np.exp(np.nan_to_num(-rise_t/times)-\
                                  np.nan_to_num(times/decay_t))
        fred_pulse = np.zeros((len(energy),len(times)))
        line_profile = np.zeros(len(energy))
        pulse_profile = np.zeros(len(times))
        for i in range(len(times)):
            fred_pulse[:,i] = gaussian(energy,np.array([center,sigma[i]],norm))*fred_profile[i]    
        line_profile = np.sum(fred_pulse,axis=1)
        pulse_profile = np.sum(fred_pulse,axis=0)
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        width = params[:,1][:,np.newaxis]
        center = params[:,2][:,np.newaxis]
        rise_t = params[:,3][:,np.newaxis]
        decay_t = params[:,4][:,np.newaxis]
        decay_w = params[:,5][:,np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            powerlaw_shape = powerlaw(times/times[0],
                                      np.concatenate([np.ones(decay_w.shape),
                                                      decay_w],axis=1))
            sigma = np.nan_to_num(width*powerlaw_shape)
            sigma[:,0] = width.T
            fred_profile = np.exp(np.nan_to_num(-rise_t/times)-\
                                  np.nan_to_num(times/decay_t))
        fred_pulse = np.zeros((params.shape[0],len(energy),len(times)))
        line_profile = np.zeros((params.shape[0],len(energy)))
        pulse_profile = np.zeros((params.shape[0],len(times)))
        for j in range(params.shape[0]):
            for i in range(len(times)):
                par = np.array([center[j,0],sigma[j,i],norm[j,0]])
                fred_pulse[j,:,i] = gaussian(energy,par)*fred_profile[j,i]    
            line_profile[j] = np.sum(fred_pulse[j],axis=1)
            pulse_profile[j] = np.sum(fred_pulse[j],axis=0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    if return_full is True:
        return fred_pulse, line_profile, pulse_profile
    else:
        return fred_pulse
    
def gauss_bkn(array1,array2,params,return_full=False):
    """
    This is a two-dimensional model for an impulse response function. The time 
    dependence is a smoothly broken powerlaw pulse. The dependence over the 
    second axis (typically energy) is a Gaussian line narrowing over time 
    following a powerlaw. The total model is the product of the two dependences. 
    This model is meant exclusively for cross spectra.

    Parameters:
    -----------
    array1: np.array(float)
        The time array over which the pulse is defined.

    array2: np.array(float)
        The second array (typically energy) over which the model is defined.

    params: array_like(float)
        The model parameters, in the following order:

        - norm: the total model normalization
        - width: the initial width of the Gaussian
        - center: the centroid of the Gaussian
        - rise_slope: the rise pulse slope
        - decay_slope: the decay pulse slope
        - break_time: the time at which the broken power-law changes from
          rise to decay slope
        - decay_w: the slope of the energy width power-law decay

    return_full: bool, default=False
        A boolean to choose whether to return just the two-dimensional model
        (done by default), or also the additional projections over the two
        model axes.

    Output:
    -------
    brk_pulse: np.array(float), shape (len(array2),len(array1))
        A two-dimensional array containing the impulse response function over
        energy and time; if params is two-dimensional, brk_pulse has shape
        (n_sets,len(array2),len(array1)) instead. 

    line_profile: np.array(float), optional
        The projection of brk_pulse over time, i.e. the time-averaged
        spectrum; only returned if return_full is True.

    pulse_profile: np.array(float), optional
        The projection of brk_pulse over energy, i.e. the energy-integrated
        pulse profile; only returned if return_full is True.
    """
    times = array1
    energy = array2
    if params.ndim == 1:
        norm = params[0]
        width = params[1]
        center = params[2]
        rise_slope = params[3]
        decay_slope = params[4]
        break_time = params[5]
        decay_w = params[6]
        sigma = width*powerlaw(times/times[0],np.array([1.,decay_w]))
        bkn_profile = brokenpower(times/times[0],np.array([1.,rise_slope,decay_slope,break_time]))
        brk_pulse = np.zeros((len(energy),len(times)))
        line_profile = np.zeros(len(energy))
        pulse_profile = np.zeros(len(times))
        for i in range(len(times)):
            brk_pulse[:,i] = gaussian(energy,np.array([center,sigma[i],norm]))*bkn_profile[i]    
        line_profile = np.sum(brk_pulse,axis=1)
        pulse_profile = np.sum(brk_pulse,axis=0)
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        width = params[:,1][:,np.newaxis]
        center = params[:,2][:,np.newaxis]
        rise_slope = params[:,3][:,np.newaxis]
        decay_slope = params[:,4][:,np.newaxis]
        break_time = params[:,5][:,np.newaxis]
        decay_w = params[:,6][:,np.newaxis]
        powerlaw_shape = powerlaw(times/times[0],
                                  np.concatenate([np.ones(decay_w.shape),
                                                  decay_w],axis=1))
        sigma = width*powerlaw_shape
        pars = np.concatenate([np.ones(decay_slope.shape),rise_slope,decay_slope,
                               break_time],axis=1)
        bkn_profile = brokenpower(times/times[0],pars)
        brk_pulse = np.zeros((params.shape[0],len(energy),len(times)))
        line_profile = np.zeros((params.shape[0],len(energy)))
        pulse_profile = np.zeros((params.shape[0],len(times)))
        for j in range(params.shape[0]):
            for i in range(len(times)):
                par = np.array([center[j,0],sigma[j,i],norm[j,0]])
                brk_pulse[j,:,i] = gaussian(energy,par)*bkn_profile[j,i]    
            line_profile[j] = np.sum(brk_pulse[j],axis=1)
            pulse_profile[j] = np.sum(brk_pulse[j],axis=0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    if return_full is True:
        return brk_pulse, line_profile, pulse_profile
    else:
        return brk_pulse
       
def bbody_fred(array1,array2,params,return_full=False):
    """
    This is a two-dimensional model for an impulse response function. The time 
    dependence is a fast rise, exponential decay pulse. The dependence over the 
    second energy is a variable black body, cooling over time following a
    powerlaw. The total model is the product of the two dependences. This model 
    is meant exclusively for cross spectra.

    Parameters:
    -----------
    array1: np.array(float)
        The time array over which the pulse is defined.

    array2: np.array(float)
        The second array (typically energy) over which the model is defined.

    params: array_like(float)
        The model parameters, in the following order:

        - norm: the total model normalization
        - temp: the initial temperature
        - rise_t: the rise pulse timescale
        - decay_t: the decay pulse timescale
        - decay_temp: the slope of the temperature power-law decay

    return_full: bool, default=False
        A boolean to choose whether to return just the two-dimensional model
        (done by default), or also the additional projections over the two
        model axes.

    Output:
    -------
    fred_pulse: np.array(float), shape (len(array2),len(array1))
        A two-dimensional array containing the impulse response function over
        energy and time; if params is two-dimensional, fred_pulse has shape
        (n_sets,len(array2),len(array1)) instead. 

    model_profile: np.array(float), optional
        The projection of fred_pulse over time, i.e. the time-averaged
        spectrum; only returned if return_full is True.

    pulse_profile: np.array(float), optional
        The projection of fred_pulse over energy, i.e. the energy-integrated
        pulse profile; only returned if return_full is True.
    """
    times = array1
    energy = array2 
    if params.ndim == 1:
        norm = params[0]
        temp = params[1]
        rise_t = params[2]
        decay_t = params[3]
        decay_temp = params[4]
        with np.errstate(divide='ignore', invalid='ignore'):
            temp_profile = np.nan_to_num(temp*powerlaw(times/times[0],np.array([1.,decay_temp])))
            temp_profile[temp_profile<=1e-6] = 1e-6
            fred_profile = np.exp(np.nan_to_num(-rise_t/times)-\
                                  np.nan_to_num(times/decay_t))   
        fred_pulse = np.zeros((len(energy),len(times)))
        model_profile = np.zeros(len(energy))
        pulse_profile = np.zeros(len(times))
        for i in range(len(times)): 
            fred_pulse[:,i] = varbbody(energy,np.array([norm,temp_profile[i]]))*fred_profile[i]
        model_profile = np.sum(fred_pulse,axis=1)
        pulse_profile = np.sum(fred_pulse,axis=0)
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        temp = params[:,1][:,np.newaxis]
        rise_t = params[:,2][:,np.newaxis]
        decay_t = params[:,3][:,np.newaxis]
        decay_temp = params[:,4][:,np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            temp_profile = np.nan_to_num(temp*powerlaw(times/times[0],
                                                       np.concatenate([np.ones(decay_temp.shape),
                                                                       decay_temp],axis=1)))
            temp_profile[temp_profile<=1e-6] = 1e-6
            fred_profile = np.exp(np.nan_to_num(-rise_t/times)-\
                                  np.nan_to_num(times/decay_t))   
        fred_pulse = np.zeros((params.shape[0],len(energy),len(times)))
        model_profile = np.zeros((params.shape[0],len(energy)))
        pulse_profile = np.zeros((params.shape[0],len(times)))
        for j in range(params.shape[0]):
            for i in range(len(times)):
                par = np.array([norm[j,0],temp_profile[j,i]])
                fred_pulse[j,:,i] = norm[j,0]*varbbody(energy,par)*fred_profile[j,i]    
            model_profile[j] = np.sum(fred_pulse[j],axis=1)
            pulse_profile[j] = np.sum(fred_pulse[j],axis=0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    if return_full is True:
        return fred_pulse, model_profile, pulse_profile
    else:
        return fred_pulse
    
def bbody_bkn(array1,array2,params,return_full=False):
    """
    This is a two-dimensional model for an impulse response function. The time 
    dependence is a smoothly broken powerlaw pulse. The dependence over the 
    second energy is a variable black body, cooling over time following a
    powerlaw. The total model is the product of the two dependences. This model 
    is meant exclusively for cross spectra.

    Parameters:
    -----------
    array1: np.array(float)
        The time array over which the pulse is defined.

    array2: np.array(float)
        The second array (typically energy) over which the model is defined.

    params: array_like(float)
        The model parameters, in the following order:

        - norm: the total model normalization
        - temp: the initial temperature
        - rise_slope: the rise pulse slope
        - decay_slope: the decay pulse slope
        - break_time: the time at which the broken power-law changes from
          rise to decay slope
        - decay_temp: the slope of the temperature power-law decay

    return_full: bool, default=False
        A boolean to choose whether to return just the two-dimensional model
        (done by default), or also the additional projections over the two
        model axes.

    Output:
    -------
    brk_pulse: np.array(float), shape (len(array2),len(array1))
        A two-dimensional array containing the impulse response function over
        energy and time; if params is two-dimensional, brk_pulse has shape
        (n_sets,len(array2),len(array1)) instead. 
        
    model_profile: np.array(float), optional
        The projection of brk_pulse over time, i.e. the time-averaged
        spectrum; only returned if return_full is True.

    pulse_profile: np.array(float), optional
        The projection of brk_pulse over energy, i.e. the energy-integrated
        pulse profile; only returned if return_full is True.
    """
    times = array1
    energy = array2 
    if params.ndim == 1:
        norm = params[0]
        temp = params[1]
        rise_slope = params[2]
        decay_slope = params[3]
        break_time = params[4]
        decay_temp = params[5]
        temp_profile = temp*powerlaw(times/times[0],np.array([1.,decay_temp]))
        temp_profile[temp_profile<=1e-6] = 1e-6
        bkn_profile = brokenpower(times,np.array([1.,rise_slope,decay_slope,break_time]))
        brk_pulse = np.zeros((len(energy),len(times)))
        model_profile = np.zeros(len(energy))
        pulse_profile = np.zeros(len(times))
        for i in range(len(times)):
            brk_pulse[:,i] = varbbody(energy,np.array([norm,temp_profile[i]]))*bkn_profile[i]
        model_profile = np.sum(brk_pulse,axis=1)
        pulse_profile = np.sum(brk_pulse,axis=0)
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        temp = params[:,1][:,np.newaxis]
        rise_slope = params[:,2][:,np.newaxis]
        decay_slope = params[:,3][:,np.newaxis]
        break_time = params[:,4][:,np.newaxis]
        decay_temp = params[:,5][:,np.newaxis]
        temp_profile = temp*powerlaw(times/times[0],np.concatenate([np.ones(decay_temp.shape),
                                                                    decay_temp],axis=1)) 
        temp_profile[temp_profile<=1e-6] = 1e-6
        pars = np.concatenate([np.ones(decay_slope.shape),rise_slope,decay_slope,
                               break_time],axis=1)
        bkn_profile = brokenpower(times,pars)
        brk_pulse = np.zeros((params.shape[0],len(energy),len(times)))
        model_profile = np.zeros((params.shape[0],len(energy)))
        pulse_profile = np.zeros((params.shape[0],len(times)))
        for j in range(params.shape[0]):
            for i in range(len(times)):
                par = np.array([norm[j,0],temp_profile[j,i]])
                brk_pulse[j,:,i] = norm[j,0]*varbbody(energy,par)*bkn_profile[j,i]    
            model_profile[j] = np.sum(brk_pulse[j],axis=1)
            pulse_profile[j] = np.sum(brk_pulse[j],axis=0)
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")
    if return_full is True:
        return brk_pulse, model_profile, pulse_profile
    else:
        return brk_pulse  

def pivoting_pl(array1,array2,params):
    """
    This is a pivoting power-law model for a transfer fuction, similar to that 
    implemented in reltrans (Mastroserio et al. 2021). The main difference is 
    that this implementation  expresses the dependence of the paramters gamma 
    and phi_ab (in the paper above) explicitely. This model is meant exclusively  
    for cross spectra.

    Parameters:
    -----------
    array1: np.array(float)
        The Fourier frequencies over which to compute the model.

    array2: np.array(float)
        The second array (typically energy) over which to compute the model.

    params: array_like(float)
        The model parameters, in the following order:

        - norm: the model normalization
        - pl_index: the slope of the power-law
        - gamma_0: the gamma parameter in Mastroserio et al. 2021, defined at
          a frequency nu_0
        - gamma_slope: the dependence of the gamma parameter with Fourier
          frequency, which is assumed to be log-linear
        - phi_0: the phi_AB parameter in Mastroserio et al. 2021, defined at
          a frequency nu_0
        - phi_slope: the dependence of the phi_AB parameter with Fourier
          frequency, which is assumed to be log-linear
        - nu_0: the initial frequency from which the pivoting parameters are
          defined

        params can either be a one-dimensional array containing a single set
        of parameters, or a two-dimensional array containing multiple sets of
        parameters to be computed simultaneously.

    Output:
    -------
    pivoting: np.array(complex), shape (len(array2),len(array1))
        A two-dimensional array containing the complex pivoting power-law
        evaluated over energy and Fourier frequency. This model is intended
        for use as a cross spectral timing product, tracking energy-dependent
        spectral pivoting as a function of Fourier frequency.
    """
    freqs = array1
    energy = array2
    if params.ndim == 1:
        norm = params[0]
        pl_index = params[1]
        gamma_0 = params[2]
        gamma_slope = params[3]
        phi_0 = params[4]
        phi_slope = params[5]
        nu_0 = params[6]
        pivoting = np.zeros((len(energy),len(freqs)),dtype=complex)
        powerlaw_shape = norm*powerlaw(energy,np.array([norm,pl_index]))
        phase = phi_0 + np.log10(freqs/nu_0)*phi_slope
        if phi_0 < 0:
            phase[phase<-0.99*np.pi] = -0.99*np.pi
        elif phi_0 > 0:
            phase[phase>0.99*np.pi] = 0.99*np.pi
        gamma = gamma_0 + np.log10(freqs/nu_0)*gamma_slope
        gamma[gamma<0] = 0
        #attempting new formalism for phi_0
        #*powerlaw(freqs/nu_0,np.array([1.,phi_slope]))
        #temp hack to avoid phase wrapping

        #the reshaping is to avoid for loops and to use matrix multiplication instead
        piv_factor = 1 - gamma*np.exp(1j*phase).reshape((1,len(freqs)))*np.log(energy).reshape((len(energy),1))
        pivoting = piv_factor*powerlaw_shape.reshape(len(energy),1)
    elif params.ndim == 2:
        norm = params[:,0][:,np.newaxis]
        pl_index = params[:,1][:,np.newaxis]
        gamma_0 = params[:,2][:,np.newaxis]
        gamma_slope = params[:,4][:,np.newaxis]
        phi_0 = params[:,4][:,np.newaxis]
        phi_slope = params[:,5][:,np.newaxis]
        nu_0 = params[:,6][:,np.newaxis]
        pivoting = np.zeros((len(energy),len(freqs)),dtype=complex)
        powerlaw_shape = norm*powerlaw(energy,
                                       np.concatenate([norm,pl_index],axis=1))
        #really not sure that this is correct 
        phase = phi_0 + np.log10(freqs/nu_0)*np.concatenate(phi_slope,axis=1)
        gamma = gamma_0 + np.log10(freqs/nu_0)*np.concatenate(gamma_slope,axis=1)
        gamma[gamma<0] = 0
        #phase = phi_0*powerlaw(freqs/nu_0,
        #                       np.concatenate([np.ones(phi_slope.shape),
        #                                       phi_slope],axis=1)) 
        #the reshaping is to avoid for loops and to use matrix multiplication instead
        log_energ = np.repeat(np.log(energy)[np.newaxis,:,np.newaxis],
                              params.shape[0],axis=0)
        piv_factor = 1 - (gamma*np.exp(1j*phase))[:,np.newaxis,:]*log_energ
        pivoting = piv_factor*powerlaw_shape[:,:,np.newaxis]
    else:
        raise TypeError("Params has too many dimensions, limit to 1 or 2 dimensions")      
    return pivoting    

def pol_constant(energs,params,grid_edges=False):
    """
    This model returns a polarization degree and angle which are both constant 
    with energy. It should be used as a multiplicative model against an array 
    containing Stokes I, Q and U vectors.

    Parameters:
    -----------
    energs: np.array(float)
        The array of photon energies over which to compute the model.

    params: array_like(float)
        The model parameters, in the following order:

        - pol_degree: the polarization degree, defined between 0 and 1
        - pol_angle: the polarization angle, in degrees

    grid_edges: bool, default=False
        Specifies whether energs contains all the edges of a binned grid
        (identically to Xspec), or the grid midpoints.

    Output:
    -------
    model: np.array(float), shape (3,len(energs))
        A two-dimensional, normalized Stokes vector (stokes_I, stokes_Q,
        stokes_U). 
    """
    if grid_edges is True:
        energs = 0.5*(energs[1:]+energs[:-1])
    if params.ndim != 1:
        raise TypeError("Params has too many dimensions, limit to 1 dimension")
    pol_degree = params[0]*np.ones(len(energs))
    pol_angle = np.radians(params[1])*np.ones(len(energs))
    #a normalized Stokes I of unity turns polarization_to_stokes into exactly 
    #the normalized Stokes vector this model is required to return
    product = PolarimetryProduct(energs,input_type='polarization')
    product.set_polarization(np.ones(len(energs)),pol_degree,pol_angle)
    stokes_I, stokes_Q, stokes_U = product.polarization_to_stokes()
    model = np.array([stokes_I,stokes_Q,stokes_U])
    return model
    
def pol_degree_linear(energs,params,grid_edges=False):
    """
    This model returns a polarization degree which varies linearly with energy, 
    and a polarization angle which is constant with it. It should be used as a 
    multiplicative model against an array  containing Stokes I, Q and U vectors. 

    Parameters:
    -----------
    energs: np.array(float)
        The array of photon energies over which to compute the model.

    params: array_like(float)
        The model parameters, in the following order:

        - pol_degree: the polarization degree at the pivot energy, defined
          between 0 and 1
        - degree_slope: the slope of the polarization degree, in units of
          inverse keV
        - pol_angle: the polarization angle, in degrees
        - energ_pivot: the pivot energy, in keV, at which the polarization
          degree is equal to pol_degree

    grid_edges: bool, default=False
        Specifies whether energs contains all the edges of a binned grid
        (identically to Xspec), or the grid midpoints.

    Output:
    -------
    model: np.array(float), shape (3,len(energs))
        A two-dimensional, normalized Stokes vector (stokes_I, stokes_Q,
        stokes_U). 
    """
    if grid_edges is True:
        energs = 0.5*(energs[1:]+energs[:-1])
    if params.ndim != 1:
        raise TypeError("Params has too many dimensions, limit to 1 dimension")
    pol_degree = params[0]+params[1]*(energs-params[3])
    pol_angle = np.radians(params[2])*np.ones(len(energs))
    product = PolarimetryProduct(energs,input_type='polarization')
    product.set_polarization(np.ones(len(energs)),pol_degree,pol_angle)
    stokes_I, stokes_Q, stokes_U = product.polarization_to_stokes()
    model = np.array([stokes_I,stokes_Q,stokes_U])
    return model    

def pol_angle_linear(energs,params,grid_edges=False):
    """
    This model returns a polarization angle which varies linearly with energy, 
    and a polarization degree which is constant with it. It should be used as a 
    multiplicative model against an array  containing Stokes I, Q and U vectors. 

    Parameters:
    -----------
    energs: np.array(float)
        The array of photon energies over which to compute the model.

    params: array_like(float)
        The model parameters, in the following order:

        - pol_degree: the polarization degree, defined between 0 and 1
        - pol_angle: the polarization angle at the pivot energy, in degrees
        - angle_slope: the slope of the polarization angle, in units of
          degrees per keV
        - energ_pivot: the pivot energy, in keV, at which the polarization
          angle is equal to pol_angle

    grid_edges: bool, default=False
        Specifies whether energs contains all the edges of a binned grid
        (identically to Xspec), or the grid midpoints.

    Output:
    -------
    model: np.array(float), shape (3,len(energs))
        A two-dimensional, normalized Stokes vector (stokes_I, stokes_Q,
        stokes_U).
    """
    if grid_edges is True:
        energs = 0.5*(energs[1:]+energs[:-1])
    if params.ndim != 1:
        raise TypeError("Params has too many dimensions, limit to 1 dimension")
    pol_degree = params[0]*np.ones(len(energs))
    pol_angle = np.radians(params[1]+params[2]*(energs-params[3]))
    product = PolarimetryProduct(energs,input_type='polarization')
    product.set_polarization(np.ones(len(energs)),pol_degree,pol_angle)
    stokes_I, stokes_Q, stokes_U = product.polarization_to_stokes()
    model = np.array([stokes_I,stokes_Q,stokes_U])
    return model

def pol_linear(energs,params,grid_edges=False):
    """
    This model returns a polarization degree and angle which both vary linearly 
    with energy. It should be used as a multiplicative model against an array 
    containing Stokes I, Q and U vectors.

    Parameters:
    -----------
    energs: np.array(float)
        The array of photon energies over which to compute the model.

    params: array_like(float)
        The model parameters, in the following order:

        - pol_degree: the polarization degree at the pivot energy, defined
          between 0 and 1
        - degree_slope: the slope of the polarization degree, in units of
          inverse keV
        - pol_angle: the polarization angle at the pivot energy, in degrees
        - angle_slope: the slope of the polarization angle, in units of
          degrees per keV
        - energ_pivot: the pivot energy, in keV, at which the polarization
          degree and angle are equal to pol_degree and pol_angle

    grid_edges: bool, default=False
        Specifies whether energs contains all the edges of a binned grid
        (identically to Xspec), or the grid midpoints.

    Output:
    -------
    model: np.array(float), shape (3,len(energs))
        A two-dimensional, normalized Stokes vector (stokes_I, stokes_Q,
        stokes_U).
    """
    if grid_edges is True:
        energs = 0.5*(energs[1:]+energs[:-1])
    if params.ndim != 1:
        raise TypeError("Params has too many dimensions, limit to 1 dimension")
    pol_degree = params[0]+params[1]*(energs-params[4])
    pol_angle = np.radians(params[2]+params[3]*(energs-params[4]))
    product = PolarimetryProduct(energs,input_type='polarization')
    product.set_polarization(np.ones(len(energs)),pol_degree,pol_angle)
    stokes_I, stokes_Q, stokes_U = product.polarization_to_stokes()
    model = np.array([stokes_I,stokes_Q,stokes_U])
    return model

def pol_rotation(seed,params):
    """
    This model rotates the polarization angle of another polarization model by 
    the same amount at every energy, leaving the polarization degree unchanged. 
        
    Unlike the other polarization models, this is a convolution rather than 
    multiplicative model. It takes as input the (3, len(energs)) Stokes seed 
    vector already returned by another polarization model, and transforms into 
    a new Stokes vector of the same shape; for instance;
    
     .. code-block:: python
     
        base = pol_angle_linear(energs,base_params)
        rotated = pol_rotation(base,rotation_params)

    Parameters:
    -----------
    seed: np.array(float), shape (3,len(energs))
        The Stokes vector returned by another polarization model, to be
        rotated. Its first row is Stokes I, and the second and third rows are
        the corresponding Stokes Q and U.

    params: array_like(float)
        The model parameters, in the following order:

        - angle_rotation: the angle by which the whole polarization angle
          model is rotated, in degrees

    Output:
    -------
    model: np.array(float), shape (3,len(energs))
        A two-dimensional, normalized Stokes vector (stokes_I, stokes_Q,
        stokes_U).
    """
    if params.ndim != 1:
        raise TypeError("Params has too many dimensions, limit to 1 dimension")
    angle_rotation = params[0]
    bins = np.arange(seed.shape[1])
    seed_product = PolarimetryProduct(bins,input_type='stokes')
    seed_product.set_stokes(seed[0],seed[1],seed[2])
    model = seed_product.rotate_polarization(angle_rotation)
    return model
    
def plot_2d(xaxis,yaxis,impulse_2d,impulse_x,impulse_y,
            xlim=[0.,400.],ylim=[0.1,10.5],xlog=False,ylog=False,
            return_plot=False,normalize_en=True):
    """
    A simple automated plotter for the impulse response function models above. 

    Parameters:
    -----------
    xaxis, yaxis: np.array(float)
        The two grids (typically time and energy) over which the model is
        defined.

    impulse_2d: np.array(float), shape (len(yaxis),len(xaxis))
        The two-dimensional model to plot.

    impulse_x, impulse_y: np.array(float)
        The projections of the model over the x/y axis.

    xlim, ylim: array_like(float), default=[0.,400.]/[0.1,10.5]
        The limits of the x/y axis to show in the plot.

    xlog, ylog: bool, default=False
        Booleans to switch between linear and log scales in each axis.

    return_plot: bool, default=False
        A boolean to decide whether to return the figure object containing
        the plot or not.

    normalize_en: bool, default=True
        A boolean to choose whether to multiply the energy dependence (on the
        y axis) by the y axis values squared. Useful to highlight the model
        energy dependence.

    Output:
    -------
    fig: matplotlib.figure, optional
        The plot object produced by the method, showing the impulse response
        function together with its time and energy projections. Only
        returned if return_plot is True.
    """
    fig = plt.figure(figsize=(9.,7.5))

    gs = gridspec.GridSpec(200,200)
    gs.update(wspace=0,hspace=0)
    ax = plt.subplot(gs[:-50,:-50])
    side = plt.subplot(gs[:-50,-50:200])
    below = plt.subplot(gs[-50:200,:-50])

    if normalize_en is True:
        impulse_2d = yaxis.reshape(len(yaxis),1)**2*impulse_2d
        impulse_y = impulse_y*yaxis**2

    c = ax.pcolormesh(xaxis,yaxis,impulse_2d,cmap="PuRd",
                  shading='auto',linewidth=0,rasterized=True)
    ax.set_xticklabels([])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_visible(False)
    ax.set_ylabel("Energy (keV)",fontsize=18)

    below.semilogy(xaxis,impulse_x,linewidth=2.5,color=colorscale[3])
    below.set_xlabel("Time ($\\rm{R_g}/c$)",fontsize=18)
    below.set_ylabel("Response",fontsize=18)
    below.set_xlim(xlim)
    below.set_ylim([1e-4*(max(impulse_x)),2.5*max(impulse_x)])

    side.step(impulse_y,yaxis,linewidth=2.5,color=colorscale[3],where='mid')
    side.invert_xaxis()
    side.yaxis.tick_right()
    side.yaxis.set_label_position('right')
    side.yaxis.set_ticks_position('both')
    side.set_xlabel("Spectrum \n (arb. units)",fontsize=18)
    side.yaxis.set_visible(False)
    side.set_ylim(ylim)
    fig.colorbar(c, ax=side)
    
    if ylog is True:
        ax.set_yscale("log",base=10)
        side.set_yscale("log",base=10)

    if xlog is True:
        ax.set_xscale("log",base=10)
        side.set_xscale("log",base=10)

    plt.show()
    if return_plot is True:
        return fig 
    else:
        return
