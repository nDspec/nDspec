import numpy as np
import os
import warnings
import scipy
import pyfftw
from pyfftw.interfaces.numpy_fft import (
    fft,
    fftfreq,
)

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams.update({'font.size': 17})

from .Operator import nDspecOperator

pyfftw.interfaces.cache.enable()

class FourierProduct(nDspecOperator):
    """
    Parent class for all Fourier operators in the software. It is mainly used 
    to handle performing Fourier transforms from the time to the frequency 
    domain, which is implemented using two methods. It is not to be instantiated
    by itself.
    
    The first simply uses the numpy fftw to calculate a standard fast Fourier 
    transform; the second uses the sinc function decomposition described in 
    Uttley and Malzac 2023 (arXiv:2312.08302), eq. 27. 
    
    Depending on the input provided to the constructor, the class can switch 
    automatically between the two implementations to ensure the Fourier 
    transform calculation is correct.
    
    Parameters: 
    -----------  
    times: np.array(float)
        The array of times over which the quantity to be transformed is defined.
        If using the fft method, this array also defines the frequencies over
        which the transform is computed. 
    
    freqs: np.array(float) 
        Defines the range of Fourier frequencies over which the input quantity 
        is to be transformed, only if using the sinc function method. Has no 
        effect with fft.
    
    method: string{ "fft" | "sinc" }, default "fft"
        The computational method to calculate the Fourier tranform of input 
        quantities. Options are "fft", for a standard Fast Fourier Transform 
        as implemented by numpy.fft, and "sinc", for the sinc function 
        decomposition described in Uttley and Malzac 2023 (arXiv:2312.08302).
    
    Attributes: 
    -----------  
    times: np.array(float) 
        The array of times over which quantities are to be Fourier transformed.
        Necessary to utilize the sinc transform method. This array needs to be 
        linearly spaced to use the fft method; if this is not the case, the 
        class defaults to the sinc method. 
    
    time_bins: np.array(float) 
        The array of widths of each bin in the "time" array. The FFT method can 
        only be called if all the elements of the array are equal to the first, 
        or in other words if the "time" array is linearly spaced.    
    
    n_times: int 
        The size of the "times" and "time_bins" arrays.
    
    method: string{"fft" | "sinc" }
        The computational method to calculate the Fourier tranform of input 
        quantities. Options are "fft", for a standard Fast Fourier Transform 
        as implemented by numpy.fft, and "sinc", for the sinc function 
        decomposition described in Uttley and Malzac 2023 (arXiv:2312.08302).
    
    freqs: np.array(float)
        The array of Fourier frequencies over which the Fourier transform is 
        computed. When using the "fft" method, this is derived from the "times"
        array, considering only the positive frequency bins. When using the 
        "sinc" method, this is set by the user when the operator object is first
        initialized. 
    
    n_freqs: int
        The size of the "freqs" array.   
        
    irf_sinc_arr: np.array(complex,complex)
        A matrix of size (n_freqs) times n_times, which stores how each time bin
        in the "times" array maps to a given sinc function with frequency given 
        in the "freqs" array. This is required for use of the sinc method, and 
        therefore is updated automatically every time the user changes frequency
        grid.       
    """

    def __init__(self,times,freqs=0,method='fft',verbose=False):

        if (np.diff(times)<0).any():
            raise ValueError("Input time array is not monotonically increasing")

        self.times = times
        self.time_bins = np.diff(self.times)
        self.n_times = self.times.size
        
        if (np.allclose(self.time_bins, self.time_bins[0]) is False):
            if verbose is True:
                warnings.warn("Bin sizes not constant over time array, defaulting method to sinc",
                               UserWarning)
            self.method = 'sinc'
        elif np.isin(method,(['sinc'],['fft'],['sinc_cumul'])):
            self.method = method
        else:
            if verbose is True:
                warnings.warn("Unknown transform method, defaulting to fftw",
                               UserWarning)
            self.method ='fft'
        #if we're going to do fft the freqyency array is pre-determined, 
        #otherwise it  should be in the arguments
        
        #add safeguard for sinc method with freq array undefined
        self.freqs = self._set_frequencies(freqs)     
        if (self.method == 'sinc') or (self.method == 'sinc_cumul'):
            self.irf_sinc_arr = self._sinc_decomp()        
        pass
       
    def _set_frequencies(self,freqs=0,rebin=False):
        """   
        This method is used to the internal frequency array and its size. If 
        using the sinc transform method, the decomposition matrix irc_sinc_arr 
        is also updated automatically.
        
        Parameters:
        -----------
        freqs: np.array(float) 
            If using the sinc method, this is the array that will be used as the
            objet Fourier frequency grid. If using the fft method, the array is
            computed from the "n_times" and "time_bins" attributes, and is 
            linearly spaced in Fourier frequency, unless the "rebin" flag is 
            true (see below).
            
        rebin: bool, default=False 
            Used only for the fft method. If false, the method assumes we are 
            initializing the object for the first time, and therefore sets the 
            frequency grid to the output of the fftfreq function. If true, it 
            means we want to switch from the linear frequency grid provided by
            fftfreq to some other user-specified grid through the appropriate 
            methods of each object.

        Returns:
        --------
        new_freqs: np.array(float) 
            The updated frequency array, which can then be assigned to the
            "freqs" method. 
            
        """        
        #if the Fourier transforms were computed with fft but we are rebinning 
        #a spectrum, we need to update the frequency array and its size;
        #otherwise, if no rebinning is done and we are just setting the 
        #frequency grid, we get the frequency array from the time array and 
        #the fftfreq() function in pyfftw.
        if (self.method == 'fft'):
            if rebin is True:
                new_freqs = freqs
            else:
                fgt0 = self._positive_fft_bins()
                new_freqs = fftfreq(self.n_times,self.time_bins[0])[fgt0]
            self.n_freqs = len(new_freqs)
        elif (self.method == 'sinc') or (self.method == 'sinc_cumul'):
            if np.shape(freqs) == () or len(np.shape(freqs)) > 1:
                raise TypeError("Input frequency grid in incorrect format")
            if (np.diff(freqs)<0).any():
                raise ValueError("Input frequency array is not monotonically increasing")                                          
            self.n_freqs = len(freqs)
            new_freqs = freqs            
        else:
            raise AttributeError("Fourier transform method not recognized")
        return new_freqs
   
    def _compute_fft(self,input_array):
        """   
        This method computes the fast Fourier transform of the input array. The 
        normalization convention used is that both the forwards and backwards
        transforms are normalized by a factor sqrt(N), in order to return the 
        same normalization as the sinc function method.
        
        Parameters:
        -----------
        input_array: np.array(float) 
            The array to be Fourier transformed. 

        Returns:
        --------
        transform: np.array(complex) 
            The Fourier transform of the input array, including only the values 
            defined at positive, non-zero frequency bins. 
        """        
        
        fgt0 = self._positive_fft_bins()
        #An additional normalization factor (dt*t_tot), where t_tot is the
        #length of the time series sampled, also needs to be included because  
        #the normalization with fftw otherwise becomes resolution dependent
        norm = np.sqrt(np.size(input_array)/(self.time_bins[0]*self.times[-1])) 
        if (np.isnan(norm)):
            warnings.warn("FFT normalisation nan, setting it to unity",
                           UserWarning)
            norm = 1  
        transform = fft(input_array)[fgt0]/norm             
        return transform

    def _positive_fft_bins(self,include_zero=False):
        """
        This method return the range of positive frequencies of a complex FFT 
        and discards the negative ones.
        This function is copied from the Stingray library 
        (https://docs.stingray.science/en/v1.1.2.4/)_ 
    
        This assumes we are using Numpy's FFT, or something compatible
        with it, like ``pyfftw.interfaces.numpy_fft``, where the positive
        frequencies come before the negative ones, the Nyquist frequency is
        included in the negative frequencies but only in even number of bins,
        and so on.
        This is mostly to avoid using the ``freq > 0`` mask, which is
        memory-hungry and inefficient with large arrays. We use instead a
        slice object, giving the range of bins of the positive frequencies.
    
        See https://numpy.org/doc/stable/reference/
                    routines.fft.html#implementation-details
    
        Parameters:
        -----------
        include_zero : bool, default=False
            Include the zero frequency in the output slice.
    
        Returns:
        --------
        positive_bins : `slice`
            Slice object encoding the positive frequency bins.
        """        
        # The zeroth bin is 0 Hz. We usually don't include it, but
        # if the user wants it, we do.
        minbin = 1
        if include_zero:
            minbin = 0            
        
        if self.n_times % 2 == 0:
            return slice(minbin, self.n_times // 2)                        
        return slice(minbin, (self.n_times + 1) // 2)

    def _sinc_decomp(self):
        """   
        This method calculates the mapping between time delay bins and Fourier  
        frequency when using the sinc decomposition method, following eq. 27 in  
        Uttley and Malzac (https://arxiv.org/abs/2312.08302).
        
        Returns:
        --------
        decomp: np.array(complex,complex) 
            A two-d array of size (n-times x n_freqs), containing the 
            mapping of time delay bins to Fourier frequency. This is the
            same the term delta_tau * sinc(delta_tau*nu) * exp(-i 2 pi nu tau) 
            in Malzac and Uttley.
        
        """
        
        deltau = np.zeros(len(self.times))
        #the reason for setting the first element manually is to allow for start 
        #points of the time grid that are greater than 0
        deltau[0] = self.times[0]
        deltau[1:len(self.times)] = self.time_bins
        #all the reshaping here is to ensure that decomp is the correct 
        #format - ie, a matrix of size (n_times x n_freqs)
        decomp = deltau.reshape((len(deltau),1))* \
                 np.sinc(deltau.reshape((len(deltau),1))* \
                         self.freqs.reshape((1,self.n_freqs)))* \
                         np.exp(-2*np.pi*self.freqs.reshape((1,self.n_freqs))* \
                         self.times.reshape((len(deltau),1))*1j)        
        return decomp 
    
    def _compute_sinc(self,input_array):
        """   
        This method computes the Fourier transform of the array using the sinc 
        decomposition method, which can be more computationally efficient than 
        fft in some cases. In this method, we assume that we are inputting a 
        single impulse response function, essentially neglecting the summatory
        in eq. 27 in Uttley and Malzac 2023.
        
        Parameters:
        -----------
        input_array: np.array(float) 
            The array to be Fourier transformed. 

        Returns:
        --------
        transform: np.array(complex) 
            The Fourier transform of the input array for the set of time delay 
            and Fourier frequency bins set by _sinc_decomp(). 
        """
        
        if hasattr(self,"irf_sinc_arr") is False:
            raise AttributeError("Sinc base not defined")
        transform = np.matmul(input_array,self.irf_sinc_arr)                
        return transform

    def _compute_sinc_cumul(self,input_array):
        """   
        This method computes the Fourier transform of the array using the sinc 
        decomposition method, which can be more computationally efficient than 
        fft in some cases. In this method, we assume that the impulse response 
        at the i-th time bin is the response at that time bin, plus the response 
        up to the i-1th bin, identically to eq. 27 in Uttley and Malzac 2023.
        
        Parameters:
        -----------
        input_array: np.array(float) 
            The array to be Fourier transformed. 

        Returns:
        --------
        transform: np.array(complex) 
            The Fourier transform of the input array for the set of time delay 
            and Fourier frequency bins set by _sinc_decomp(). 
        """   
        
        if hasattr(self,"irf_sinc_arr") is False:
            raise AttributeError("Sinc base not defined")
        transform = np.cumsum(np.matmul(input_array.reshape(1,self.n_times),
                                        self.irf_sinc_arr),axis=1)        
        return transform 
        
    def transform(self,input_array):
        """   
        This method acts like a wrapper to Fourier transform array, and calls 
        the appropriate method depending on user settings. 
        
        Parameters:
        -----------
        input_array: np.array(float) 
            The array to be Fourier transformed. 

        Returns:
        --------
        transform: np.array(complex) 
            The Fourier transform of the input array.
        """        
        
        if (self.method == 'fft'):
            transform = self._compute_fft(input_array)
        elif (self.method == 'sinc'):
            transform = self._compute_sinc(input_array)
        elif (self.method =='sinc_cumul'):
            transform = self._compute_sinc_cumul(input_array)
        else:
            raise AttributeError("Fourier transform method not recognized")        
        return transform 

    
class PowerSpectrum(FourierProduct):
    """    
    This class computes model power spectra from a given signal input. The 
    public methods in the class call those in nDspecOperator and FourierProduct  
    nas ecessary to Fourier transform input quantities and handle changes to the
    internal frequency grid. 
    
    Parameters inherited from FourierProduct:
    ----------------------------------------- 
    times: np.array(float)
        The array of times over which the input signal is defined. 
    
    freqs: np.array(float) 
        The Fourier frequency array over which the power spectrum is defined. 
    
    method: string{"fft" | "sinc" }, default='fft'
        The computational method to calculate the power spectrum. Options are 
        "fft", for a standard Fast Fourier Transform  as implemented by 
        numpy.fft, and "sinc", for the sinc function decomposition described in 
        Uttley and Malzac 2023 (arXiv:2312.08302).
        
    verbose: bool
        A setting to display initialization warnings.
     
    Attributes inherited from FourierProduct:  
    -----------------------------------------  
    times: np.array 
        The array of times over which the input signal is defined. 
    
    time_bins: np.array 
        The array of widths of each bin in the "time" array. The FFT method can 
        only be used if all the elements of the array are equal to the first,
        or in other words if the "time" array is linearly spaced.
    
    n_times: int 
        The size of the "times" and "time_bins" arrays.
    
    method:  {"fft" | "sinc" }
        The computational method to calculate the Fourier tranform of input 
        quantities. Options are "fft", for a standard Fast Fourier Transform 
        as implemented by numpy.fft, and "sinc", for the sinc function 
        decomposition described in Uttley and Malzac 2023 (arXiv:2312.08302).
    
    freqs: np.array(float)
        The array of Fourier frequencies over which the power spectrum is 
        defined and/or computed.
    
    n_freqs: int
        The size of the "freqs" array.   
        
    irf_sinc_arr: np.array(complex,complex)
        A matrix of size (n_freqs x n_times), which stores how each time bin in 
        the "times" array maps to a given sinc function with frequency given in
        the "freqs" array. This is required for use of the sinc method, and 
        therefore is updated automatically every time the user changes frequency
        grid.  
        
    Other Attributes: 
    -----------------   
    power_spec: np.array(float)
        An array of size(n_freqs), storing the un-normalized power spectrum of 
        an input signal.     
    """
    
    def __init__(self,times,freqs=0,method='fft',verbose=False):        
        FourierProduct.__init__(self,times,freqs,method,verbose)        
        pass 

    def compute_psd(self,signal=None,freq=None,params=None):
        """   
        This method calculates the power spectrum, defined over the class 
        "freqs" Fourier frequency array, of an input array, and assigns it to   
        the power_spec attribute contained in the class instance.
        
        Parameters:
        -----------
        signal: np.array(float), default None
            The quantity from which to calculate the power spectrum. If it is 
            not provided, the class will use the model object stored in 
            the ``model'' attribute instead.
            
        freq: np.array(float), default None 
            The array of Fourier frequencies over which to compute the model. 
            If none is passed and evaluating a Fourier-domain model, uses the
            frequency stored in the object at initialization. 
            
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are 
            provided, the ''model_params'' attribute stored in the object is 
            used.    
        """  
        
        if signal is None:
            if freq is None:
                freq = self.freqs   
            if params is None:
                model = self.model.eval(self.model_params,freq=freq)
            else:
                model = self.model.eval(params,freq=freq)
            self.power_spec = model          
        else:
            if len(signal) != self.n_times:
                raise TypeError("Input signal size different from time array")
            transform = self.transform(signal)
            self.power_spec = np.multiply(transform,np.conj(transform))                
        return 

    def set_psd_model(self,model,params=None):
        """
        This method sets a LMFit CompositeModel object to be used to evaluate the 
        power spectrum; useful if you want your model to be defined in the Fourier 
        rather than time domain.
    
        Parameters:
        -----------            
        model: lmfit.model or lmfit.compositemodel 
            The lmfit wrapper of the model one wants to fit to the data. 
            
        params: lmfit.Parameters, default: None 
            The parameter values from which to start evalauting the model during
            the fit. If it is not provided, all model parameters will default 
            to 0, set to be free, and have no minimum or maximum bound. 
        """

        if ((getattr(model, '__module__', None) != "lmfit.compositemodel")&
            (getattr(model, '__module__', None) != "lmfit.model")):  
            raise AttributeError("The model input must be an LMFit Model or CompositeModel object")

        self.model = model 
        if params is None:
            self.model_params = self.model.make_params(verbose=True)
        else:
            self.model_params = params
        return 

    def rebin_frequency(self,new_grid,use_log=True):
        """   
        This methods rebins the class freqs array, and also updates all relevant 
        attributes (n_freqs, power_spec). The re-binned power spectrum is 
        calculated by interpolating the previous power over the new grid. 
        
        Parameters:
        -----------
        new_grid: np.array(float) 
            The new grid of Fourier frequencies over which to rebin the power 
            spectrum.   
            
        Other parameters:
        -----------------
        use_log: bool, default True
            Switches between interpolating the power spectrum, or the base 10 
            logarithm of the power spectrum, which is more accurate if the power 
            varies by many orders of magnitude with frequency
        """
        
        new_power = self._interpolate(self.power_spec,self.freqs,new_grid,use_log)
        self.freqs = self._set_frequencies(new_grid,rebin=True)
        self.power_spec = new_power                
        return 
    
    def plot_psd(self,units='Power*freq',return_plot=False):
        """ 
        This method plots the either the power or power per unit frequency 
        as a function of frequency, stored in the class instance.  
        
        Parameters:
        -----------
        units: string , default="Power*freq"
            Sets the units to plot on the y axis - the default "power*freq" 
            displays the power at that frequency, "power" instead uses the power
            per unit frequency.  
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        
        fig, ((ax1)) = plt.subplots(1,1,figsize=(9.,6.))   
        
        if units == 'Power':
            ax1.plot(self.freqs,self.power_spec)
            ax1.set_ylabel("Power")
        elif units == "Power*freq":
            ax1.plot(self.freqs,self.power_spec*self.freqs)
            ax1.set_ylabel("Power$\\times$frequency")
        else:
            raise ValueError("Y axis units not recognized")
        
        ax1.set_xscale("log",base=10)
        ax1.set_yscale("log",base=10)
        ax1.set_xlabel("Frequency")  
        
        plt.tight_layout()
        plt.show()        
        
        if return_plot is True:
            return fig 
        else:
            return   
        
        
class CrossSpectrum(FourierProduct):
    """    
    This class computes model cross spectra from a given input, assumed to be 
    either a user-provided impulse response function or a user-provided model 
    (such as a transfer function) already defined in Fourier space. It can 
    handle both one-dimensional, frequency dependent cross spectra between just 
    two energy bands, or two-dimensional cross spectra defined over multiple 
    energy and frequency grids. 
    
    Parameters inherited from FourierProduct:
    -----------------------------------------  
    times: np.array(float)
        The array of times over which the input signal is defined. 
    
    freqs: np.array(float) 
        The Fourier frequency array over which the power spectrum is defined. 
    
    method: string{"fft" | "sinc" }, default="fft"
        The computational method to calculate the cross spectrum. Options are 
        "fft", for a standard Fast Fourier Transform  as implemented by 
        numpy.fft, and "sinc", for the sinc function decomposition described in 
        Uttley and Malzac 2023 (arXiv:2312.08302). 
        
    verbose: bool
        A setting to trigger initialization warnings.   

    Other parameters: 
    ----------------- 
    energ: np.array(float) 
        The array of energy (or energy channels) bin mid-points over which a
        two-dimensional cross spectrum is defined. If we are only considering 
        the cross-spectrum between two energy bands, this defaults to "None". 

    Attributes inherited from FourierProduct:  
    -----------------------------------------  
    times: np.array(float) 
        The array of times over which the input signal is defined. 
    
    time_bins: np.array(float) 
        The array of widths of each bin in the "time" array. The FFT method can 
        only be used if all the elements of the array are equal to the first,
        or in other words if the "time" array is linearly spaced.
    
    n_times: int 
        The size of the "times" and "time_bins" arrays.
    
    method: string{"fft" | "sinc" }
        The computational method to calculate the Fourier tranform of input 
        quantities. Options are "fft", for a standard Fast Fourier Transform 
        as implemented by numpy.fft, and "sinc", for the sinc function 
        decomposition described in Uttley and Malzac 2023 (arXiv:2312.08302).
    
    freqs: np.array(float)
        The array of Fourier frequencies over which the power spectrum is 
        defined and/or computed.
    
    n_freqs: int
        The size of the "freqs" array.   
        
    irf_sinc_arr: np.array(complex,complex)
        A matrix of size (n_freqs x n_times), which stores how each time bin in 
        the "times" array maps to a given sinc function with frequency given in
        the "freqs" array. This is required for use of the sinc method, and 
        therefore is updated automatically every time the user changes frequency
        grid.  
    
    Other attributes: 
    ----------------- 
    n_chans: int 
        The size of the "energy" array. Defaults to 1 for a one-dimensional 
        cross spectrum between two energy bands. 
    
    chans: np.array(int) 
        An array containing the indexes of each energy bin (or channel) in the 
        cross spectrum. Defaults to 0 for a one-dimensional cross spectrum 
        between two energy bands.  
    
    power_spec: np.array(float) 
        A one-dimensional array of size (n_freqs) containing the assumed shape 
        of the power spectrum. This is necessary to a) calculate the cross 
        product from an impulse response and b) calculate Fourier frequency 
        averaged products like lag-energy spectra in a given frequency range. 
    
    imp_resp: np.array(float,float) 
        An array of size (n_chans x n_times), storing the model impulse 
        response function from which to compute the cross spectrum. 
    
    ref: np.array(float) 
        An array of size (n_times), storing the reference band signal. The 
        reference band can either be computed directly from the imp_resp 
        attribute, using the set_reference_idx method, or it can be provided 
        by the user, using the set_reference_lc method. 
        
    correct_ref: bool 
        A bool that sets whether the reference band used in the cross spectrum 
        calculations is corrected by subtracting the channel of interest 
        (correct_ref=True) or if it is kept the same (correct_ref=False).  
    
    trans_func: np.array(complex,complex) 
        An array of size (n_chans x n_freqs), containing the Fourier transform  
        of each channel of the impulse response function provided by the user - 
        formally, this is the transfer function. It is used together with the 
        weighing power spectrum power_spec to calculate the cross spectrum, and 
        is necessary to compute one-dimensional frequency-dependent products 
        (such as lag-frequency spectra) from a two-dimensional cross spectrum. 
        If users are not providing an impulse response function, this attribute 
        can be set by hand (e.g. from a model defined in Fourier space).

    cross: np.array(complex,complex)  
        An array of size (n_chans x n_freqs) containing the full (one or two 
        dimensional) cross spectrum, defined as the cross product between the
        Fourier transforms of one or more channels of interest and of the 
        reference band, and weighed by the power spectrum. 
        If users are not providing an impulse response or transfer function, 
        this attribute can be set by hand (e.g. from a model defined in Fourier 
        space).     
    """
    
    def __init__(self,times,freqs=0,energ=None,method='fft',verbose=False):
        FourierProduct.__init__(self,times,freqs,method,verbose)
        self.energ = energ
        #energ=none is used to treat 1-d cross spectra between only two energy
        #channels
        if energ is None:
            self.n_chans = 1
            self.chans = 0
        else:
            if (np.diff(energ)<0).any():
                raise ValueError("Input energy grid is not monotonically increasing")        
            self.n_chans = len(self.energ)
            self.chans = np.linspace(0,self.n_chans-1,self.n_chans,
                                     dtype=np.int16)  
        pass      

    def set_psd_weights(self,input_power):
        """  
        This method sets the weighing power spectrum power_spec from a given 
        input.  
        
        Parameters:
        -----------
        input_power: np.array(float) or PowerSpectrum
            Either an array of size (n_freqs) that is to be used as the weighing  
            power spectrum when computing the cross spectrum, or an nDspec 
            PowerSpectrum object. Both have to be defined over the same Fourier 
            frequency array. 
        """

        if isinstance(input_power,PowerSpectrum):
            if (np.allclose(input_power.freqs,self.freqs) is False):
                raise ValueError("Frequency array of PSD does not match the existing one")
            self.power_spec = input_power.power_spec
        else:                    
            if (len(input_power)) != self.n_freqs:
                raise TypeError("Input PSD array size different from frequency array")        
            self.power_spec = input_power       
        return

    def set_impulse(self,signal):
        """   
        This method sets the impulse response function imp_resp from which the 
        cross spectrum can be calculated.
        
        Parameters:
        -----------
        signal: np.array(float,float) 
            An array of size (n_chans x n_times) containing the model impulse 
            response function. 
        """
        
        if (signal.shape[0] != self.n_chans):
            raise TypeError("Input matrix size incorrect; x axis is not n_chans")
        if (signal.shape[1] != self.n_times):
            raise TypeError("Input matrix size incorrect; y axis is not n_times")
        
        self.imp_resp = signal
        #reshaping ensures the correct irf input format for a 1d cross spectrum        
        if self.n_chans == 1:
            self.imp_resp = np.reshape(self.imp_resp,(1,self.n_times))        
        return

    def set_transfer(self,signal):
        """   
        This method sets the transfer function trans_func from which the cross
        spectrum can be calculated.
        
        Parameters:
        -----------
        signal: np.array(float,float) 
            An array of size (n_chans x n_freqs) containing the model transfer 
            function. 
        """

        if (signal.shape[0] != self.n_chans):
            raise TypeError("Input matrix size incorrect; x axis is not n_chans")
        if (signal.shape[1] != self.n_freqs):
            raise TypeError("Input matrix size incorrect; y axis is not n_freqs")
        
        self.trans_func = signal
        #reshaping ensures  the correct transfer function input format for a 1d 
        #cross spectrum        
        if self.n_chans == 1:
            self.trans_func = np.reshape(self.trans_func,(1,self.n_freqs))        
        return

    def set_reference_energ(self,ref_bounds,correct_ref=True):
        """   
        This method sets the reference band from a range of energies provided by
        the user. Users can also specify whether or not they want  
        each channel of interest to be subtracted from the reference band when 
        computing the cross spectrum.
          
        Parameters:
        -----------          
        ref_bounds: np.array(float)
            A list with lower and upper energy channel bounds to be used in the 
            reference band. By default, we assume that the reference band is
            identical to that used in calculating the cross spectrum. As 
            implemented here, the limits specified in ref_bounds are included in
            the reference - e.g. [0.3,10.0] keV rather than (0.3,10.) keV.
            
        correct_ref: bool, default=True
            Flag to correct the reference band by removing each channel of 
            interest or not.    
        """   
        
        idx_ref = np.where(np.logical_and(self.energ>=ref_bounds[0],
                                          self.energ<=ref_bounds[1]))
        #idx_ref returns a tuple, not an array. The array of channel indexes is
        #stored in the first element of the tuple, ie idx_ref[0]
        if len(idx_ref[0]) == 0:
            raise ValueError("No full bins found within the reference band bounds")
                                

        if hasattr(self,"trans_func"):
            self.ref = np.reshape(np.sum(self.trans_func[idx_ref,:],axis=1),
                                 (self.n_freqs))  
        elif hasattr(self,"imp_resp"):
            self.ref = np.reshape(np.sum(self.imp_resp[idx_ref,:],axis=1),
                                 (self.n_times))      
        else:
            raise AttributeError("Neither impulse response nor transfer function defined")  
            
        self.correct_ref = correct_ref
        return 

    def set_reference_lc(self,input_lc,correct_ref=False):
        """   
        This method sets the reference band from an array (e.g. count rate as a
        function of energy) provided by the user. Users can also specify 
        whether or not they want each channel of interest to be subtracted from
        the reference band when computing the cross spectrum.
        
        Parameters:
        -----------
        input_lc: np.array(float) 
            An arrray of size (n_chans) containing either the reference band 
            lightcurve (if the model is defined in the time domain) or its 
            Fourier transform (for models defined in the Fourier domain).
            
        correct_ref: bool, default=False
            Flag to correct the reference band by removing each channel of 
            interest or not.    
        """
        
        if ((len(input_lc)) != self.n_times) and hasattr(self,"imp_resp"):
            raise TypeError("Reference array is not the same size as time array")
        if ((len(input_lc)) != self.n_freqs) and hasattr(self,"trans_func"):
            raise TypeError("Reference array is not the same size as frequency array")
        
        self.ref = input_lc
        self.correct_ref = correct_ref        
        return
    
    def cross_from_transfer(self,signal=None,ref_bounds=None,power=None):
        """   
        This method computes thecross spectrum from an input defined in Fourier
        space (such as a transfer function), reference band, and power spectra
        provided by the  user. These can either be already stored by the setter  
        methods set_psd_weights, set_transfer, and set_reference_idx or 
        set_reference_lc (which is the default behavior), or they can be passed
        as arguments of  this method. In the latter case, the reference can 
        only be provided in array, rather than channel index, form.
        
        Parameters:
        -----------
        signal: np.array(float,float), default=self.trans_func 
            An array of size (n_chans x n_freqs) containing a  model transfer 
            function defined in Fourier space. 
            
        ref_bounds: np.array(float)
            A list with lower and upper energy channel bounds to be used in the 
            reference band. By default, we assume that the reference band is
            identical to that used in calculating the cross spectrum. As 
            implemented here, the limits specified in ref_bounds are included in
            the reference - e.g. [0.3,10.0] keV rather than (0.3,10.) keV.
            
        power: np.array(float) or PowerSpectrum, default=self.power_spec 
            Either an array of size (n_freqs), or a PowerSpectrum nDspec object, 
            that is to be used as the weighing power spectrum when computing the
            cross spectrum.
        """
        
        if signal is not None:
            self.set_transfer(signal)
        transfer = self.trans_func
        
        if ref_bounds is not None:
            self.set_reference_energ(ref_bounds)
        ref_ft = self.ref
        
        if power is None:
            power_spec = self.power_spec 
        else:
            self.set_psd_weights(power) 
            if isinstance(power,PowerSpectrum): 
                power_spec = power.power_spec
            else:
                power_spec = power 
        
        self.cross = []
                
        for index in range(self.n_chans):
            if self.correct_ref is False:
                cross_prod = power_spec*np.multiply(transfer[index,:],
                                                    np.conj(ref_ft))
            else:
                #correct the reference by removing each channel of interest
                #note that we can do it after the FT to save computational time 
                #and not re-do the FT of the reference at each step of the loop
                #because the Fourier transform is a linear operator
                corr_ft = ref_ft - transfer[index,:]
                cross_prod = power_spec*np.multiply(transfer[index,:],
                                                    np.conj(corr_ft))              
            self.cross.append(cross_prod)  

        #reshaping ensures a consistent correct cross spectrum format between
        #1d and 2d cases
        self.cross = np.reshape(np.array(self.cross),
                               (self.n_chans,self.n_freqs))
        return   

    def transfer_from_irf(self,signal=None):
        """   
        This method computes and store the transfer function from a given 
        impulse response provided by the user. This can either be already stored
        by the setter method set_impulse (which is the default behavior), or it 
        can be passed as argument of this method. 
        
        Parameters:
        -----------
        signal: np.array(float,float), default=self.imp_resp 
            An array of size (n_chans x n_times) containing the model impulse 
            response function. 
        """

        if signal is None:
            signal = self.imp_resp
        else:
            self.set_impulse(signal)        
        
        self.trans_func = []
        
        for index in range(self.n_chans):
            ci_ft = self.transform(signal[index,:])
            self.trans_func.append(ci_ft)               

        self.trans_func = np.reshape(np.array(self.trans_func),
                                    (self.n_chans,self.n_freqs))        
        return

    #maybe throw in a wrapper like with transform 
    def cross_from_irf(self,signal=None,ref_bounds=None,power=None):
        """   
        This method computes the transfer function and cross spectrum from the 
        impulse response, reference band, and power spectra provided by the 
        user. These can either be already stored by the setter methods 
        set_psd_weights, set_impulse, and set_reference_idx or set_reference_lc
        (which is the default behavior), or they can be passed as arguments of 
        this method. In the latter case, the reference can only be provided in 
        array, rather than channel index, form.
        
        Parameters:
        -----------
        signal: np.array(float,float), default=self.imp_resp 
            An array of size (n_chans x n_times) containing the model impulse 
            response function. 
            
        ref_bounds: np.array(float)
            A list with lower and upper energy channel bounds to be used in the 
            reference band. By default, we assume that the reference band is
            identical to that used in calculating the cross spectrum. As 
            implemented here, the limits specified in ref_bounds are included in
            the reference - e.g. [0.3,10.0] keV rather than (0.3,10.) keV.
            
        power: np.array(float) or PowerSpectrum, default=self.power_spec 
            Either an array of size (n_freqs), or a PowerSpectrum nDspec object, 
            that is to be used as the weighing power spectrum when computing the
            cross spectrum.
        """
        
        if signal is not None:
            self.set_impulse(signal)
        signal = self.imp_resp
        
        if ref_bounds is not None:
            self.set_reference_energ(ref_bounds)
        reference = self.ref
        
        if power is None:
            power_spec = self.power_spec 
        else:
            self.set_psd_weights(power)  
            if isinstance(power,PowerSpectrum): 
                power_spec = power.power_spec
            else:
                power_spec = power 
        
        self.cross = []
        self.trans_func = []
        
        ref_ft = self.transform(reference)  
        for index in range(self.n_chans):
            ci_ft = self.transform(signal[index,:])
            self.trans_func.append(ci_ft)
            if self.correct_ref is False:
                cross_prod = power_spec*np.multiply(ci_ft,np.conj(ref_ft))
            else:
                #correct the reference by removing each channel of interest
                #note that we can do it after the FT to save computational time 
                #and not re-do the FT of the reference at each step of the loop
                #because the Fourier transform is a linear operator
                corr_ft = ref_ft - ci_ft
                cross_prod = power_spec*np.multiply(ci_ft,np.conj(corr_ft))                
            self.cross.append(cross_prod)                 

        #reshaping ensures a consistent correct cross spectrum and transfer
        #function format between1d and 2d cases  
        self.cross = np.reshape(np.array(self.cross),
                               (self.n_chans,self.n_freqs))
        self.trans_func = np.reshape(np.array(self.trans_func),
                                    (self.n_chans,self.n_freqs))        
        return

    def rebin_frequency(self,new_grid):
        """   
        This method rebins the cross spectrum and transfer function stored in 
        the object to a new frequency grid, and updates the relevant class 
        attributes as necessary.
        
        Parameters:
        -----------
        new_grid: np.array(float)
            An array of arbitrary size, containing the Fourier frequency bin 
            midpoints of the new grid.
        """        
        
        new_cross = []
        new_trans = []
        new_power = self._interpolate(self.power_spec,self.freqs,new_grid)
        
        for index in range(self.n_chans):
            cross_rebin = self._interpolate(self.cross[index,:],
                                            self.freqs,new_grid)
            trans_rebin = self._interpolate(self.trans_func[index,:],
                                            self.freqs,new_grid)
            new_cross.append(cross_rebin)
            new_trans.append(trans_rebin)
        
        self.freqs = self._set_frequencies(new_grid,rebin=True)
        self.power_spec = new_power
        self.cross = np.reshape(np.array(new_cross),(self.n_chans,self.n_freqs))
        self.trans_func = np.reshape(np.array(new_trans),(self.n_chans,
                                                          self.n_freqs))         
        return
    
    def real(self):
        """   
        This method returns the real part of the cross spectrum, both for one 
        and two dimensional cases.

        Returns:
        --------
        real: np.array(float) 
            An array of real parts of the cross spectrum, of size 
            (n_chans x n_freqs)
        """

        if not hasattr(self,"cross"):
            raise AttributeError("Cross spectrum not computed")
        
        real = np.real(self.cross)        
        return real

    def imag(self):
        """   
        This method returns the imaginary part of the cross spectrum, both for  
        one and two dimensional cases.

        Returns:
        --------
        imag: np.array(float) 
            An array of imaginary parts of the cross spectrum, of size 
            (n_chans x n_freqs)
        """

        if not hasattr(self,"cross"):
            raise AttributeError("Cross spectrum not computed")

        imag = np.imag(self.cross)       
        return imag

    def mod(self):
        """   
        This method returns the modulus of the cross spectrum, both for one and
        and two dimensional cases.

        Returns:
        --------
        mod: np.array(float)
            An array of moduli of the cross spectrum, of size 
            (n_chans x n_freqs)
        """

        if not hasattr(self,"cross"):
            raise AttributeError("Cross spectrum not computed")
        
        mod = np.absolute(self.cross)        
        return mod 

    def phase(self):
        """   
        This method returns the phase of the cross spectrum, both for one and
        and two dimensional cases.

        Returns:
        --------
        phases: np.array(float) 
            An array of phases of the cross spectrum, of size 
            (n_chans x n_freqs)
        """

        if not hasattr(self,"cross"):
            raise AttributeError("Cross spectrum not computed")
        
        phase = np.angle(self.cross)        
        return phase

    def lag(self):
        """   
        This method converts the phase of the cross spectrum, both for one and
        and two dimensional cases, into time lags.

        Returns:
        --------
        lags: np.array(float) 
            An array of phases of the cross spectrum, of size 
            (n_chans x n_freqs)
        """
        
        lag_conv = 2.*np.pi*self.freqs
        lags = self.phase()/lag_conv        
        return lags 

    def _oned_cross(self,int_bounds,ref_bounds=None):
        """   
        This method returns a frequency-dependent, one-dimensional cross 
        specctrum either from a two-dimensional transfer function stored in the 
        "trans" attribute, or from the two-dimensional "cross" attribute, 
        depending on whether a new reference band is used or not. The one-d 
        cross spectrum is produced  by summing over energy channels specified 
        by the user and, if a different reference band is chosen from the one 
        saved in the "ref" attribute,  re-calculating the cross product between 
        the reference band and channels of interest. As implemented here, the l
        imits specified in ref_bounds and int_bounds are ncluded in the r
        eference - e.g. [0.3,10.0] keV rather than (0.3,10.) keV.
        
        Parameters:
        -----------
        int_bounds: np.array(float) 
            A list of energy channel bounds to be used in the channels of 
            interest - using the convention used in X-ray spectral timing, this 
            should be the energy band where reverberation lags appear with
            negative values.
            
        ref_bounds: np.array(float), default=None  
            A list of energy channel bounds to be used in the reference band.
            By default, this assumes that the reference band is identical to  
            that used in calculating the cross spectrum. 

        Returns:
        --------
        cross: np.array(complex) 
            A one-dimensional array of size (n_freqs) containing the one 
            dimensional cross spectrum between the channels of interest and the 
            reference band provided by the user. 
        """
        
        idx_int = np.where(np.logical_and(self.energ>=int_bounds[0],
                                          self.energ<=int_bounds[1]))
        #idx_int returns a tuple, not an array. The array of channel indexes is
        #stored in the first element of the tuple, ie idx_int[0]
        if (len(idx_int[0]) == 0):
            raise ValueError("No bins found within the channel of interest bounds")  
        if ref_bounds is None:
            #if we do not define a new reference band, take the frequency 
            #dependent part from the existing cross spectrum simply by summing 
            #over the input energy channels 
            cross = np.sum(self.cross[idx_int,:],axis=1)
        else:
            #otherwise, we need to re-calculate a new cross spectrum with a new 
            #reference band, from the existing transfer function 
            idx_ref = np.where(np.logical_and(self.energ>=ref_bounds[0],
                                              self.energ<=ref_bounds[1]))
            #idx_ref returns a tuple, not an array. The array of channel indexes
            #is stored in the first element of the tuple, ie idx_ref[0].
            if (len(idx_ref[0]) == 0):
                raise ValueError("No bins found within the reference band bounds")                                              
            ref = np.sum(self.trans_func[idx_ref,:],axis=1)                                         
            ch_int = np.sum(self.trans_func[idx_int,:],axis=1)
            cross = self.power_spec*np.multiply(ch_int,np.conj(ref))        
        return cross

    def real_frequency(self,int_bounds,ref_bounds=None):
        """   
        This method computes the real part of a one-dimensional, Fourier 
        frequency dependent cross spectrum, by summing over energy channels 
        specified by the user and (if a new reference band is used) 
        re-calculating the cross product between the reference band and channels 
        of interest.
        
        Parameters:
        -----------
        int_bounds: np.array(float) 
            A list with lower and upper energy channel bounds to be used in the 
            channels of  interest - using the convention used in X-ray spectral
            timing, this  should be the energy band where reverberation lags 
            appear with negative values.
            
        ref_bounds: np.array(float), default=None  
            A list with lower and upper energy channel bounds to be used in the 
            reference band. By default, we assume that the reference band is
            identical to that used in calculating the cross spectrum. 

        Returns:
        --------
        real_spectrum: np.array(float) 
            A one-dimensional array of size (n_freqs) containing the real part 
            of he one dimensional cross spectrum between the channels of  
            interest and the reference band provided by the user, as a function  
            of Fourier frequency.
        """
        
        real_spectrum = np.real(self._oned_cross(int_bounds,ref_bounds))
        real_spectrum = np.reshape(real_spectrum,self.n_freqs)       
        return real_spectrum
    
    def imag_frequency(self,int_bounds,ref_bounds=None):
        """   
        This method computes the imaginary part of a one-dimensional, Fourier 
        frequency dependent cross spectrum, by summing over energy channels 
        specified by the user and (if a new reference band is used) 
        re-calculating the cross product between the reference band and channels 
        of interest.
        
        Parameters:
        -----------
        int_bounds: np.array(float) 
            A list with lower and upper energy channel bounds to be used in the 
            channels of  interest - using the convention used in X-ray spectral
            timing, this  should be the energy band where reverberation lags 
            appear with negative values.
            
        ref_bounds: np.array(float), default=None  
            A list with lower and upper energy channel bounds to be used in the 
            reference band. By default, we assume that the reference band is
            identical to that used in calculating the cross spectrum. 

        Returns:
        --------
        imag_spectrum: np.array(float) 
            A one-dimensional array of size (n_freqs) containing the imaginary 
            part of the one dimensional cross spectrum between the channels of  
            interest and the reference band provided by the user, as a function 
            of Fourier frequency. 
        """
        
        imag_spectrum = np.imag(self._oned_cross(int_bounds,ref_bounds))
        imag_spectrum = np.reshape(imag_spectrum,self.n_freqs)        
        return imag_spectrum
    
    def mod_frequency(self,int_bounds,ref_bounds=None):
        """   
        This method computes the modulus of a one-dimensional, Fourier frequency
        dependent cross spectrum, by summing over energy channels specified by
        the user and (if a new reference band is used) re-calculating the cross 
        product between the reference band and channels of interest.
        
        Parameters:
        -----------
        int_bounds: np.array(float) 
            A list with lower and upper energy channel bounds to be used in the 
            channels of  interest - using the convention used in X-ray spectral
            timing, this  should be the energy band where reverberation lags 
            appear with negative values.
            
        ref_bounds: np.array(float), default=None  
            A list with lower and upper energy channel bounds to be used in the 
            reference band. By default, we assume that the reference band is
            identical to that used in calculating the cross spectrum. 

        Returns;
        --------
        mod_spectrum: np.array(float)
            A one-dimensional array of size (n_freqs) containing the modulus 
            of the one dimensional cross spectrum between the channels of  
            interest and the  reference band provided by the user. 
        """
        
        mod_spectrum = np.absolute(self._oned_cross(int_bounds,ref_bounds))
        mod_spectrum = np.reshape(mod_spectrum,self.n_freqs)       
        return mod_spectrum
    
    def phase_frequency(self,int_bounds,ref_bounds=None):
        """   
        This method computes the phase of a one-dimensional, Fourier frequency
        dependent cross spectrum, by summing over energy channels specified by
        the user and (if a new reference band is used) re-calculating the cross 
        product between the reference band and channels of interest.
        
        Parameters:
        -----------
        int_bounds: np.array(float) 
            A list with lower and upper energy channel bounds to be used in the 
            channels of  interest - using the convention used in X-ray spectral
            timing, this  should be the energy band where reverberation lags 
            appear with negative values.
            
        ref_bounds: np.array(float), default=None  
            A list with lower and upper energy channel bounds to be used in the 
            reference band. By default, we assume that the reference band is
            identical to that used in calculating the cross spectrum. 

        Returns:
        --------
        phase_spectrum: np.array(float) 
            A one-dimensional array of size (n_freqs) containing the phase 
            one dimensional cross spectrum between the channels of  
            interest and the  reference band provided by the user, as a function 
            of Fourier frequency. 
        """
        
        phase_spectrum = np.angle(self._oned_cross(int_bounds,ref_bounds))
        phase_spectrum = np.reshape(phase_spectrum,self.n_freqs)       
        return phase_spectrum
    
    def lag_frequency(self,int_bounds,ref_bounds=None):        
        """   
        This method computes the time lag of a one-dimensional, Fourier 
        frequency dependent cross spectrum, by summing over energy channels 
        specified by the user and (if a new reference band is used) 
        re-calculating the cross product between the reference band and channels 
        of interest.
        
        Parameters:
        -----------
        int_bounds: np.array(float) 
            A list with lower and upper energy channel bounds to be used in the 
            channels of  interest - using the convention used in X-ray spectral
            timing, this  should be the energy band where reverberation lags 
            appear with negative values.
            
        ref_bounds: np.array(float), default=None  
            A list with lower and upper energy channel bounds to be used in the 
            reference band. By default, we assume that the reference band is
            identical to that used in calculating the cross spectrum. 

        Returns:
        --------
        mod_spectrum: np.array(float) 
            A one-dimensional array of size (n_freqs) containing the time lags 
            of the one dimensional cross spectrum between the channels of  
            interest and the reference band provided by the user, as a function 
            of Fourier frequency.  
        """
        
        lag_spectrum = self.phase_frequency(int_bounds,ref_bounds)/ \
                       (2.*np.pi*self.freqs)               
        return lag_spectrum
        
    def real_energy(self,freq_bounds):
        """ 
        This method computes the real part of a one-dimensional cross spectrum 
        as a function of energy, by averaging the attribute "cross" over a given
        frequency range specified by the user. 
        
        Parameters:
        -----------
        freq_bounds: np.array(float) 
            A list with lower and upper frequency bounds over which to average 
            the two dimensinoal cross spectrum

        Returns:
        --------
        real_sectrum: np.array(float) 
            An array of size (n_chans) containing the real part of the Fourier 
            frequency averaged cross spectrum, as a function of energy. 
        """
        
        nu_min = freq_bounds[0]
        nu_max = freq_bounds[1]        
        integrated_resp = self._integrate_range(self.cross,self.freqs,
                                                nu_min,nu_max,axis=1)
        real_spectrum = np.real(integrated_resp/(nu_max-nu_min))
        real_spectrum = np.reshape(real_spectrum,self.n_chans)        
        return real_spectrum

    def imag_energy(self,freq_bounds):
        """ 
        This method computes the imaginary part of a one-dimensional cross  
        spectrum as a function of energy, by averaging the attribute "cross" 
        over a given frequency range specified by the user. 
        
        Parameters:
        -----------
        freq_bounds: np.array(float) 
            A list with lower and upper frequency bounds over which to average 
            the two dimensinoal cross spectrum
            
        Returns:
        --------
        imag_spectrum: np.array(float)  
            An array of size (n_chans) containing the imaginary part of the  
            Fourier frequency averaged cross spectrum, as a function of energy. 
        """
        
        nu_min = freq_bounds[0]
        nu_max = freq_bounds[1]                
        integrated_resp = self._integrate_range(self.cross,self.freqs,
                                                nu_min,nu_max,axis=1)
        imag_spectrum = np.imag(integrated_resp/(nu_max-nu_min))
        imag_spectrum = np.reshape(imag_spectrum,self.n_chans)        
        return imag_spectrum
    
    def mod_energy(self,freq_bounds):
        """ 
        This method computes the modulus of a one-dimensional cross spectrum as 
        a function of energy, by averaging the attribute "cross" over a given 
        frequency range specified by the user. 
        
        Parameters:
        -----------
        freq_bounds: np.array(float) 
            A list with lower and upper frequency bounds over which to average 
            the two dimensinoal cross spectrum
            
        Returns:
        --------
        mod_spectrum: np.array(float)  
            An array of size (n_chans) containing the modulus of the Fourier
            frequency averaged cross spectrum, as a function of energy. 
        """
        
        nu_min = freq_bounds[0]
        nu_max = freq_bounds[1]        
        integrated_resp = self._integrate_range(self.cross,self.freqs,
                                                nu_min,nu_max,axis=1)
        mod_spectrum = np.absolute(integrated_resp/(nu_max-nu_min))
        mod_spectrum = np.reshape(mod_spectrum,self.n_chans)        
        return mod_spectrum
    
    def phase_energy(self,freq_bounds):
        """ 
        This method computes the phase of a one-dimensional cross spectrum
        as a function of energy, by averaging the attribute "cross" over a given
        frequency range specified by the user. 
        
        Parameters:
        -----------
        freq_bounds: np.array(float) 
            A list with lower and upper frequency bounds over which to average 
            the two dimensinoal cross spectrum
            
        Returns:
        --------
        phase_spectrum: np.array(float)  
            An array of size (n_chans) containing the phase of the Fourier
            frequency averaged cross spectrum, as a function of energy. 
        """

        nu_min = freq_bounds[0]
        nu_max = freq_bounds[1]        
        integrated_resp = self._integrate_range(self.cross,self.freqs,
                                                nu_min,nu_max,axis=1)
        phase_spectrum = np.angle(integrated_resp/(nu_max-nu_min))
        phase_spectrum = np.reshape(phase_spectrum,self.n_chans)        
        return phase_spectrum
    
    def lag_energy(self,freq_bounds):
        """ 
        This method computes the time lags of a one-dimensional cross spectrum
        as a function of energy, by averaging the attribute "cross" over a given
        frequency range specified by the user. 
        
        Parameters:
        -----------
        freq_bounds: np.array(float) 
            A list with lower and upper frequency bounds over which to average 
            the two dimensinoal cross spectrum
            
        Returns:
        --------
        lag_spectrum: np.array(float)  
            An array of size (n_chans) containing the time lags of the Fourier
            frequency averaged cross spectrum, as a function of energy. 
        """

        nu_min = freq_bounds[0]
        nu_max = freq_bounds[1]       
        nu_c = 0.5*(nu_min+nu_max)
        lag_spectrum = self.phase_energy(freq_bounds)/(2.*np.pi*nu_c)        
        return lag_spectrum        

    def plot_cross_1d(self,form="polar",return_plot=False):
        """   
        This method plots the a one-dimensional cross spectrum as a function of  
        Fourier frequency.
        
        Parameters:
        -----------
        form: string, default="polar" 
            A qualifier to choose in which units to plot the cross spectrum. 
            By default, form="polar" will plot the modulus, phase, and lag 
            frequency spectrum. Alternatively, form="cartesian" plots the real 
            and imaginary parts of the cross spectrum.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        
        #same as power spectrum, double check plotting style
        if form == "cartesian":
            fig, ((ax1,ax2)) = plt.subplots(1,2,figsize=(10.,5.))   
            
            ax1.plot(self.freqs,np.transpose(self.real()))
            ax1.set_xscale("log",base=10)
            ax1.set_ylabel("Real")
            ax1.set_xlabel("Frequency")  
            
            ax2.plot(self.freqs,np.transpose(self.imag()))
            ax2.set_xscale("log",base=10)
            ax2.set_ylabel("Imaginary")
            ax2.set_xlabel("Frequency")
            
            plt.tight_layout()
            plt.show()
        elif form == "polar":
            zero_line = np.zeros(self.n_freqs)
            #tbd: sort out the limits for the lag plot
            phase_wrap = 1/(2.*self.freqs)   
            
            fig, ((ax1,ax2,ax3)) = plt.subplots(1,3,figsize=(15.,5.))
            
            ax1.plot(self.freqs,np.transpose(self.mod()))
            ax1.set_xscale("log",base=10)
            ax1.set_yscale("log",base=10)
            ax1.set_xlabel("Frequency")
            ax1.set_ylabel("Modulus")    
            
            ax2.plot(self.freqs,np.transpose(self.phase()))
            ax2.plot(self.freqs,zero_line,linestyle='dotted',color='black')
            ax2.set_xscale("log",base=10)
            ax2.set_ylabel("Phase")
            ax2.set_xlabel("Frequency") 

            lag_min = np.min(np.transpose(self.lag()))
            lag_max = np.max(np.transpose(self.lag()))
            
            ax3.plot(self.freqs,zero_line,linestyle='dotted',color='black')
            ax3.plot(self.freqs,np.transpose(self.lag()))
            ax3.plot(self.freqs,phase_wrap,
                     linestyle='dotted',color='tab:orange')
            ax3.plot(self.freqs,-phase_wrap,
                     linestyle='dotted',color='tab:orange')
            ax3.set_xscale("log",base=10)
            ax3.set_ylabel("Time")
            ax3.set_xlabel("Frequency")
            ax3.set_ylim([min(0,lag_min)-0.05*lag_max,
                          max(0,lag_max)+0.05*lag_min])
            
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError("plot format not supported")

        if return_plot is True:
            return fig 
        else:
            return   

    def _plot_limits(self,plot_input):
        """   
        This method computes the normalization and ticks for two-dimensional 
        plots of the cross spectrum. 
        
        Parameters:
        -----------
        plot_input: np.array(float,float) 
            The two-dimensional array to be plotted, from which to define the 
            axis limits for the plots.

        Returns:
        --------
        norm: np.float 
            The normalization to be used for the colorbar in the plot. 
            
        ticks: np.array 
            The list of ticks to show on the colorbar. 
        """
       
        lim_min = np.min(plot_input)
        lim_max = np.max(plot_input)

        if (lim_max > 0 and lim_min < 0):
            norm = TwoSlopeNorm(vmin=lim_min,vcenter=0,vmax=lim_max)
        elif (lim_max < 0):
            norm = TwoSlopeNorm(vmin=lim_min,vcenter=0,vmax=-lim_max)   
        elif (lim_min > 0 ):
            norm = TwoSlopeNorm(vmin=-lim_min,vcenter=0,vmax=lim_max) 
        else:
            print(lim_min,lim_max)
            raise ValueError("Both lower and upper plot limits are 0")  
                         
        ticks_negative = np.linspace(0.99*lim_min,0,5)
        ticks_positive = np.linspace(0,1.01*lim_max,5)
        ticks = np.append(ticks_negative[:-1],ticks_positive)
        
        return norm,ticks
   
    def plot_cross_2d(self,form="polar",energy_limits=[0.3,10.5],
                      return_plot=False,normalize_en=True):
        """   
        Plots the a two-dimensional cross spectrum as a function of Fourier 
        frequency and energy.
        
        Parameters:
        -----------
        form: string,default="polar" 
            A qualifier to choose in which units to plot the cross spectrum. 
            By default, form="polar" will plot the modulus, phase, and lag 
            spectrum. Alternatively, form="cartesian" plots the real 
            and imaginary parts of the cross spectrum.
            
        energy_limits: list(float)
            The lower and upper bound to be used in the energy axis of the 
            cross spectrum.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
               
        energy_indexes = np.where(np.logical_and(self.energ>energy_limits[0],
                                                 self.energ<energy_limits[1]))
        
        if form == "cartesian":
            if normalize_en is True:
                plot_real = self.energ.reshape(self.n_chans,1)**2*self.real()
                plot_imag = self.energ.reshape(self.n_chans,1)**2*self.imag()
            else:
                plot_real = self.real()
                plot_real = self.imag()
                
            norm_real, ticks_real = self._plot_limits(
                                    plot_real[energy_indexes,:])
            norm_imag, ticks_imag = self._plot_limits(
                                    plot_imag[energy_indexes,:])
                       
            fig, ((ax1,ax2)) = plt.subplots(1,2,figsize=(12.5,5.))
            
            real = ax1.pcolormesh(self.freqs,self.energ,self.real(),cmap="PuOr",
                                  shading='auto',linewidth=0,
                                  rasterized=True,norm=norm_real)
            cb = fig.colorbar(real,ax=ax1,ticks=ticks_real,format="%.1f")
            ax1.set_xscale("log",base=10)
            ax1.set_title("Real")
            ax1.set_xlabel("Frequency")
            ax1.set_ylabel("Energy")
            ax1.set_ylim([energy_limits[0],energy_limits[1]])
            
            imag = ax2.pcolormesh(self.freqs,self.energ,self.imag(),cmap="PuOr",
                                  shading='auto',linewidth=0,
                                  rasterized=True,norm=norm_imag)
            cb = fig.colorbar(imag,ax=ax2,ticks=ticks_imag,format="%.1f")              
            ax2.set_xscale("log",base=10)
            ax2.set_title("Imaginary")
            ax2.set_xlabel("Frequency")
            ax2.set_ylabel("Energy")
            ax2.set_ylim([energy_limits[0],energy_limits[1]])
            
            plt.tight_layout()
            plt.show()
        elif form == "polar":
            if normalize_en is True:
                plot_mod = self.energ.reshape(self.n_chans,1)**2*self.mod()
            else:
                plot_mod = self.mod()
        
            norm_phase, ticks_phase = self._plot_limits(
                                      self.phase()[energy_indexes,:])            
            norm_lag, ticks_lag = self._plot_limits(
                                  self.lag()[energy_indexes,:])
            
            fig, ((ax1,ax2,ax3)) = plt.subplots(1,3,figsize=(15.,5.))
            modulus = ax1.pcolormesh(self.freqs,self.energ,np.log10(plot_mod),
                                     cmap="magma",shading='auto',
                                     linewidth=0,rasterized=True)
            cb = fig.colorbar(modulus,ax=ax1,format="%.1f")            
            ax1.set_xscale("log",base=10)
            ax1.set_title("Log10(Modulus)")
            ax1.set_xlabel("Frequency")
            ax1.set_ylabel("Energy")
            ax1.set_ylim([energy_limits[0],energy_limits[1]])
           
            phase = ax2.pcolormesh(self.freqs,self.energ,self.phase(),cmap="twilight",
                                   shading='auto',linewidth=0,
                                   rasterized=True,norm=norm_phase)
            cb = fig.colorbar(phase,ax=ax2,ticks=ticks_phase,format="%.2f")           
            ax2.set_xscale("log",base=10)
            ax2.set_title("Phase")
            ax2.set_xlabel("Frequency")
            ax2.set_ylabel("Energy")
            ax2.set_ylim([energy_limits[0],energy_limits[1]])
            
            lags = ax3.pcolormesh(self.freqs,self.energ,self.lag(),cmap="twilight",
                                  shading='auto',linewidth=0,
                                  rasterized=True,norm=norm_lag)
            cb = fig.colorbar(lags,ax=ax3,ticks=ticks_lag,format="%.2f")
            ax3.set_xscale("log",base=10)
            ax3.set_title("Lag")
            ax3.set_xlabel("Frequency")
            ax3.set_ylabel("Energy")
            ax3.set_ylim([energy_limits[0],energy_limits[1]])            
            
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError("plot mode not supported")
            
        if return_plot is True:
            return fig 
        else:
            return     
