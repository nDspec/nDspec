import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

#required by the polarimetry models 
from .Polarimetry import PolarimetryProduct

from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
fi = 22
plt.rcParams.update({'font.size': fi-5})

colorscale = pl.cm.PuRd(np.linspace(0.,1.,5))

def lorentz(array,params):
    """
    This model is a Lorentzian function, defined identically to Uttley and Malzac
    2023. The input parameters are:
    
    array: the array over which the Lorentzian is to be computed \n
    f_pk: the peak frequency of the Lorentzian \n   
    q: the q-factor of the Lorentzian \n    
    rms: the normalization of the Lorentzian
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
    et al. 2023. The input parameters are:
    
    array: the array over which the Lorentzian is to be computed \n    
    f_pk: the peak frequency of the Lorentzian \n    
    q: the q-factor of the Lorentzian  \n    
    rms: the normalization of the Lorentzian \n    
    phase: the phase lag associated with the Lorentzian
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
    lorentz_arr = lorentz(array2,params,grid_edges=grid_edges)*np.exp(1j*phase)
    twod_lorentz = np.tile(lorentz_arr,n_energs).reshape((n_energs,n_freqs))
    twod_lorentz = np.transpose(twod_lorentz)
    return twod_lorentz

def powerlaw(array,params):
    """
    This model is a standard power-law. The input parameters are: 
    
    array: the array grid over which to compute the power-law \n    
    norm: the normalization of the powerlaw \n    
    slope: the slope over the powerlaw. Unlike in Xspec, this parameter does not 
    implicitely assume a minus sign; it must be specified by the user.
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
    Ghisellini and Tavecchio 2009. The input parameters are:
    
    array: the array over which to compute the broken powerlaw \n    
    norm: the normalization of the broken powerlaw \n    
    slope1: the slope of the broken powerlaw before the break \n    
    slope2: the slope of the broken powerlaw after the break \n    
    brk: the location of the break in the powerlaw 
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
    This model is a Gaussian function. The input parameters are: 
    
    array: the array over which the Gaussian is defined \n    
    center: the centroid of the Gaussian \n    
    width: the width of the Gaussian \n
    gauss_norm: the normalization of the Gaussian 
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
    This model is a constant black body. The input parameters are:
    
    array: the array over which the spectrum is defined \n    
    norm: the normalization of the black body, defined identically to that of 
    the Xspec model \n    
    temp: the temperature in keV
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
    Malzac 2023. The input parameters are:
    
    array: the array over which the spectrum is defined \n    
    norm: the normalization of the black body, defined identically to that of 
    the Xspec model \n    
    temp: the temperature in keV
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
    This is a two-dimensional model for an impulse response function. The time 
    dependence is a fast rise, exponential decay pulse. The dependence over the 
    second axis (typically energy) is a Gaussian line narrowing over time 
    following a powerlaw. The total model is the product of the two dependences. 
    The input paramters are:
    
    array1: the time over which the pulse is defined \n    
    array2: the second direction over which the model is defined \n    
    norm: the total model normalization \n
    width: the initial width of the Gaussian \n    
    center: the centroid of the Gaussian \n    
    rise_t: the rise pulse timescale \n    
    decay_t: the decay pulse timescale \n    
    decay_w: the slope of the energy width powerlaw decay \n    
    return_full: a boolean to choose whether to return just the 2d model (done 
    by default), or the additional projections over the two model axis
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
    The input paramters are:
    
    array1: the time over which the pulse is defined \n    
    array2: the second direction over which the model is defined \n    
    norm: the total model normalization \n    
    center: the centroid of the Gaussian \n    
    width: the initial width of the Gaussian \n    
    rise_slope: the rise pulse slope \n    
    decay_slope: the decay pulse slope \n    
    break_time: the time at which the broken powerlaw changes from rise to decay 
    slope \n    
    decay_w: the slope of the energy width powerlaw decay \n    
    return_full: a boolean to choose whether to return just the 2d model (done 
    by default), or the additional projections over the two model axis
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
    powerlaw. The total model is the product of the two dependences. The input  
    paramters are:
    
    array1: the time over which the pulse is defined \n    
    array2: the second direction over which the model is defined \n    
    norm: the total model normalization \n    
    temp: the initial temperature \n    
    rise_slope: the rise pulse slope \n    
    decay_slope: the decay pulse slope \n    
    break_time: the time at which the broken powerlaw changes from rise to decay 
    slope \n    
    decay_temp: the slope of the temperature powerlaw decay \n    
    return_full: a boolean to choose whether to return just the 2d model (done 
    by default), or the additional projections over the two model axis
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
    powerlaw. The total model is the product of the two dependences. The input  
    paramters are:
    
    array1: the time over which the pulse is defined \n    
    array2: the second grid over which the model is defined \n    
    norm: the total model normalization \n    
    temp: the initial temperature \n    
    rise_slope: the rise pulse slope \n      
    decay_slope: the decay pulse slope \n    
    break_time: the time at which the broken powerlaw changes from rise to decay 
    slope\n    
    decay_temp: the slope of the temperature powerlaw decay\n
    return_full: a boolean to choose whether to return just the 2d model (done 
    by default), or the additional projections over the two model axis
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
    This is a pivoting power-law model similar to that implemented in reltrans
    (Mastroserio et al. 2021). The main difference is that this implementation 
    expresses the dependence of the paramters gamma and phi_ab (in the paper 
    above) explicitely. The input paramters are:
    
    array1: the Fourier frequencies over which to compute the model \n    
    array2: the second direction (typically energy) over which to compute the 
    model \n    
    norm: the model normalzation \n    
    pl_index: the slope over the powerlaw \n    
    gamma_0: the gamma parameter in Mastroserio et al. 2021, defined at a 
    frequency nu_0 \n    
    gamma_slope: the dependence of the gamma parameter with Fourier frequency, 
    which is assumed to be log-linear \n    
    phi_0: the phi_AB parameter in Mastroserio et al. 2021, defined at a 
    frequency nu_0 \n    
    nu_0: the initial frequency from which the pivoting parameters are defined
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
    containing Stokes I, Q and U vectors. The input parameters are:
    
    energs: the array of photon energies over which to compute the model \n    
    pol_degree: the polarization degree, defined between 0 and 1 \n    
    pol_angle: the polarization angle, in degrees \n    
    grid_edges: specifies whether the input array contains all the edges of a 
    binned grid (identically to xspec), or the grid midpoints
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
    The input parameters are:
    
    energs: the array of photon energies over which to compute the model \n    
    pol_degree: the polarization degree at the pivot energy, defined between 0 
    and 1 \n    
    degree_slope: the slope of the polarization degree, in units of inverse keV 
    \n    
    pol_angle: the polarization angle, in degrees \n    
    energ_pivot: the pivot energy, in keV, at which the polarization degree is 
    equal to pol_degree \n    
    grid_edges: specifies whether the input array contains all the edges of a 
    binned grid (identically to xspec), or the grid midpoints
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
    The input parameters are:
    
    energs: the array of photon energies over which to compute the model \n    
    pol_degree: the polarization degree, defined between 0 and 1 \n    
    pol_angle: the polarization angle at the pivot energy, in degrees \n    
    angle_slope: the slope of the polarization angle, in units of degrees per 
    keV \n    
    energ_pivot: the pivot energy, in keV, at which the polarization angle is 
    equal to pol_angle \n    
    grid_edges: specifies whether the input array contains all the edges of a 
    binned grid (identically to xspec), or the grid midpoints
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
    containing Stokes I, Q and U vectors. The input parameters are:
        
    energs: the array of photon energies over which to compute the model \n    
    pol_degree: the polarization degree at the pivot energy, defined between 0 
    and 1 \n    
    degree_slope: the slope of the polarization degree, in units of inverse keV 
    \n    
    pol_angle: the polarization angle at the pivot energy, in degrees \n    
    angle_slope: the slope of the polarization angle, in units of degrees per 
    keV \n    
    energ_pivot: the pivot energy, in keV, at which the polarization degree and 
    angle are equal to pol_degree and pol_angle \n    
    grid_edges: specifies whether the input array contains all the edges of a 
    binned grid (identically to xspec), or the grid midpoints
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
        
    
    The input parameters are:
    
    seed: np.array(float), shape (3, len(energs))
        The Stokes vector returned by another polarization model, to be 
        rotated. Its first row is Stokes I, and the second and third are the 
        corresponding Stokes Q and U \n    
    params: array_like(float)
        angle_rotation: the angle by which the whole polarization angle model 
        is rotated, in degrees
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
    The input parameters are:
    
    xaxis, yaxis: the two grids over which the model is defined \n    
    impulsed_2d: the two-dimensional model to plot \n
    impulse_x,impulse_y: the projections of the model over the x/y axis \n    
    xlim,ylim: the limits of the x/y axis to show in the plot \n    
    xlog,ylog: booleans to switch between linera and log scales in each axis \n    
    return_plot: boolean to return the figure object for storage/saving \n    
    normalize_en: boolean to multiply the energy dependence (on the y axis) by 
    the y axis values squared. Useful to highlight the model energy dependence.
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
