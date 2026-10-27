import numpy as np
import warnings

import pyfftw
from pyfftw.interfaces.numpy_fft import (
    fft,
    rfftfreq,
)

from lmfit import Model as LM_Model
from lmfit import Parameters as LM_Parameters

from astropy.io import fits

from stingray import AveragedCrossspectrum, AveragedPowerspectrum
from stingray.fourier import poisson_level, get_average_ctrate

from .Response import ResponseMatrix
from . import Plotting
from .Timing import PowerSpectrum, CrossSpectrum
from .SimpleFit import SimpleFit, EnergyDependentFit, FrequencyDependentFit, load_pha
from .Likelihoods import chisq, ratio

pyfftw.interfaces.cache.enable()

class FitCrossSpectrum(SimpleFit,EnergyDependentFit,FrequencyDependentFit):
    """
    Least-chi squares fitter class for the cross spectrum. The class supports 
    both one-dimensional data between a reference and subject band as a function 
    of Fourier frequency, and two-dimensional data between many subjects bands 
    and a common reference band as a function of both energy and Fourier
    frequency. 
    
    Given an instrument response, an input spectrum, its error and a model, this
    class handles fitting internally using the lmfit library. The model can 
    either be in the form of the actual values of the (complex) cross spectrum, 
    a transfer function (defined in the Fourier domain), or an impulse response 
    function (defined in the time domain). In all cases, the conversion from 
    model units to cross spectral products (e.g. lag vs frequency, or real part 
    vs energy in a fixed frequency interval) is handled automatically by the 
    class. 
           
    Due to the variety of spectral-timing products related to or derived from 
    the cross spectrum, users have several ways of defining the input data. In 
    general, however, they are encouraged to do their own analysis separately 
    using e.g. stingray, before moving on to fitting it.
    
    Attributes inherited from SimpleFit:
    ------------------------------------
    model: lmfit.CompositeModel 
        A lmfit CompositeModel object, which contains a wrapper to the model 
        component(s) one wants to fit to the data. 
   
    model_params: lmfit.Parameters 
        A lmfit Parameters object, which contains the parameters for the model 
        components.
   
    likelihood: str
        A string that allows to switch between different fit statistics; which 
        one is available depends on the type of fitter object. Uses chi-squared 
        likelihood by default. Users can set different likelihoods either at 
        initialization or with the appropriate setter method.
        
    custom_likelihood: function, optional  
        A function users can set to bypass the supported likelihoods and instead 
        provide their own. 
        
    custom_args: tuple
        A tuple including any custom arguments (in addition to the data and 
        model values to be compared) necessary to calculate the custom 
        likelihood
   
    fit_result: lmfit.MinimizeResult
        A lmfit MinimizeResult, which stores the result (including best-fitting 
        parameter values, fit statistics etc) of a fit after it has been run.         
   
    data: np.array(float)
        An array storing the data to be fitted. If the data is complex and/or 
        multi-dimensional, it is flattened to a single dimension in order to be 
        compatible with the LMFit fitter methods.
   
    data_err: np.array(float)
        An array containing the uncertainty on the data to be fitted. It is also 
        stored as a one-dimensional array regardless of the type or dimensionality 
        of the initial data.  
        
    _data_unmasked, _data_err_unmasked: np.array(float)
        The arrays of every data bin and its error, regardless of which ones are
        ignored or noticed during the fit. Used exclusively to enable book 
        keeping internal to the fitter class.          
    
    Attributes inherited from EnergyDependentFit:
    ---------------------------------------------    
    energs: np.array(float)
        The array of physical photon energies over which the model is computed. 
        Defined as the middle of each bin in the energy range stored in the 
        instrument response provided.    
        
    energ_bounds: np.array(float)
        The array of energy bin widths, for each bin over which the model is 
        computed. Defined as the difference between the uppoer and lower bounds 
        of the energy bins stored in the insrument response provided. 
               
    ear: np.array(float) 
        The array of energy bin bounds, for each bin over which the model is 
        computed. Only necessary when calling Xspec models due to their unique 
        input structure.

    ebounds: np.array(float) 
        The array of energy channel bin centers for the instrument energy
        channels,  as stored in the instrument response provided. Only contains 
        the channels that are noticed during the fit.

    ewidths: np.array(float) 
        The array of energy channel bin widths for the instrument energy
        channels, as stored in the instrument response provided. Only contains 
        the channels that are noticed during the fit.

    emin, emax: np.array(float)
        The array of lower and upper bounds of the noticed energy channels.
        
    ebounds_mask: np.array(bool)
        The array of instrument energy channels that are either ignored or 
        noticed during the fit. A given channel i is noticed if ebounds_mask[i]
        is True, and ignored if it is false. 
        
    n_chans: int 
        The number of channels that are to be noticed during the fit.
        
    _all_chans: int 
        The total number of channels in the loaded response matrix.
        
    n_bins: int 
        The number of noticed channels times the number of noticed bins in 
        Fourier frequency.
        
    _all_bins: int 
        The total number of  channels, times the total number of bins in Fourier 
        frequency.
                
    _emin_unmasked, _emax_unmasked, _ebounds_unmasked, _ewidths_unmasked: np.array(float)
        The array of every lower bound, upper bound, channel center and channel 
        widths stored in the response, regardless of which ones are ignored or 
        noticed during the fit. Used exclusively to facilitate book-keeping 
        internal to the fitter class. 

    gain_params: lmfit.Parameters, default None 
        A lmfit Parameters object, which contains the parameters for the gain  
        correction model components if it is enabled. Defaults to None. For 
        cross spectral fitting, no gain correction is currently supported.

    Attributes inherited from FrequencyDependentFit:
    ------------------------------------------------
    _freqs_unmasked: np.array(float)
        If the data and model explicitely depend on Fourier frequency (e.g. a
        power spectrum), this is the array of Fourier frequency over which all 
        data and model are defined, including bins that are ignored in the fit. 
        
        If instead the data depends on energy, it contains both noticed and 
        ignored frequency intervals over which to  produce spectral-timing 
        products. For example, a user might input a set of 7 ranges of 
        frequencies to calculate lag energy spectra, but only want to consider 
        the first and last 3, and ignore the middle one.
    
    freqs_mask: np.array(bool)
        The array of Fourier frequencies that are either ignored or noticed 
        during the fit. A given channel i is noticed if freqs_mask[i] is True,
        and ignored if it is false.      
    
    n_freqs: int 
        The number of Fourier frequency bins that are noticed in the fit.  
    
    Other attributes:
    -----------------
    response: nDspec.ResponseMatrix
        The instrument response matrix corresponding to the spectrum to be 
        fitted. It is required to define the energy grids over which model and
        data are defined.  

    units: str 
        A string that checks the units which the user is providing  - "lags" for 
        fitting lag spectra alone, "polar" or fitting modulus and phase together, 
        and "cartesian" for fitting real and imaginary parts together.        
            
    ref_band: [np.float,np.float]
        The minimum/maximum energy bounds over which to take the reference band. 
        Necessary to calculate spectral timing products (like lag spectra) from 
        the input model.
    
    freqs: np.array(float)
        If the data explicitely depends on Fourier frequency, it is the range of
        Fourier frequencies over which both the data and model are defined. 
        Otherwise, it is the internal Fourier frequency grid over which the 
        model is computed before being converted into spectral-timing products 
        (e.g. lag-energy spectra).
        
    fmin, fmax: np.array(float)
        The arrays of noticed Fourier frequency bounds over which 
        energy-dependent data and model are averaged over, in order to handle
        energy-dependent spectral-timing products. 
    
    _times: np.array(float)
        The array of times corresponding to the Fourier frequency array. Used 
        internally for model calculations/book-keeping.        
        
    crossspec: nDspec.CrossSpectrum
        A nDspec CrossSpectrum object used to store model evaluations, and to 
        convert model evaluations into spectral-timing data products. 
     
    renorm_phase: bool 
        Allows users to apply a small phase renormalization when fitting energy 
        dependent products. This is necessary to account for imperfections in 
        the instrument response/calibration. For more discussion, see Appendix 
        E in Mastroserio et al. 2018: 
        https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.4027M/abstract
        This setting will NOT affect the modulus of a cross spectrum, only the 
        phase (and therefore it will affect the real and imaginary parts).
    
    renorm_modulus: bool 
        Allows users to apply a small modulus renormalization when fitting energy 
        dependent products.  This can be useful when defining models from transfer 
        or impulse response functions, in order to re-normalize the model in each
        Fourier frequency bin to match the data. Physically, this allows one to 
        take into account differences between the power spectrum shape assumed 
        in the model to calculate the spectral timing products (e.g. modulus vs 
        energy), and the "true" underyling power spectrum in the source. 
        This setting will NOT affect the phase of a cross spectrum, only the 
        modulus (and therefore it will affect the real and imaginary parts).   

    dependence: str 
        A string recording whether the data and model to be fitted are in units 
        of frequency or energy (e.g. lag frequency vs lag energy spectra).

    _supported_coordinates: str
        A string that checks the units models/data can be defined as. "lags" is 
        for fitting lag spectra alone, "polar" is for fitting modulus and phase 
        together, and "cartesian" is for fitting real and imaginary parts 
        together.
    
    _supported_models: str
        A string that checks the type of model defined. "cross" indicates models 
        that already return the full cross spectrum, and thus need limited 
        operations applied before being compared to the data. "transfer" 
        indicates models of transfer functions, which need to be crossed with 
        a reference band to calculate the cross spectrum. "irf" indicates a 
        models of impulse response functions, which need to be Fourier transformed 
        and then crossed with a reference band.
    
    _supported_products: str 
        A string that checks whether the data provided (e.g. lags) is a function 
        of Fourier frequency or energy.    
    """
    
    def __init__(self,likelihood="chisq"):
        SimpleFit.__init__(self,likelihood)
        self.ref_band = None
        self.freqs = None 
        self._times = None
        self.crossspec = None
        self._supported_coordinates = ["cartesian","polar","lags"]
        self._supported_models = ["irf","transfer","cross"]
        self._supported_products = ["frequency","energy"]
        self.renorm_phase = False
        self.renorm_modulus = False
        pass

    def set_product_dependence(self,depend):
        """
        This method allows users to specify whether the data they intend to fit is 
        a function of Fourier frequency (e.g. lag frequency spectra), or energy 
        (e.g. lag vs energy spectra). Polarimetric data is not supported at this 
        time. 
        
        Parameters:
        -----------
        depend: str 
            A string containing the dependence of the data. Must be one of the 
            supported dependence stored in the class - ie "frequency" or "energy".
        """
        if depend not in self._supported_products:
            raise TypeError("Unsopprted products for the cross spectrum")
        else:
            self.dependence = depend
        return 

    def set_coordinates(self,units):  
        """
        This method allows users to specify the coordinate system of the data 
        they intend to fit. The possible choices are "lags", for time lags alone,
        "polar", for modulus and phase together, or "cartesian", for real and 
        imaginary parts together.
        
        Parameters:
        -----------
        units: str 
            A string containing the dependence of the data. Must be one of the 
            supported units stored in the class - ie "lags", "polar" or 
            "cartesian".
        """      
        if units not in self._supported_coordinates:
            raise TypeError("Unsopprted units for the cross spectrum")
        else:
            self.units = units
        return 

    #explicitely show in the documentation that there are many ways to build
    #data and users have a lot of freedom.
    def set_data(self,response,ref_bounds,sub_bounds,data,
                 data_err=None,freq_grid=None,time_grid=None,
                 freq_bins=None,time_res=None,seg_size=None,norm=None,
                 frebin_fac=None):
        """
        This method is used to set the cross-spectrum data to be fitted. The 
        exact data is determined by the set_product_dependence and set_coordinates
        setter methods.  
        
        The method requires an input instrument response, bounds defined for the 
        reference band as well as the subject band(s), and the actual data. This 
        can be in the form of an array, in which case users also need to specify 
        the errors, or (only for frequency-dependent data) a stingray.events 
        object. Additionally, users need to specify the time resolution and 
        segment size of the lightcurves used to build the data.          
        
        When loading energy-dependent products (e.g. many lag-energy spectra), 
        it is necessary to specify the Fourier frequency intervals over which
        each lag spectrum is computed, using the freq_bins argument. 
        
        When loading freqency-dependent products from a stingray event file, it 
        is possible to specify the normalization to be used. By default, this 
        will be asbsolute rms normalization.
        
        Finally, for both products, users can specify by hand the Fourier 
        frequency and time grids to be used internally for model computations. 
        In some cases, this can be useful in speeding up model evaluations. 
        
        Parameters:
        -----------
        response: nDspec.ResponseMatrix
            The instrument response matrix corresponding to the spectrum to be 
            fitted. It is required to define the energy grids over which model 
            and data are defined. It is rebinned automatically such that the 
            subject/reference bands are in separate channels.
        
        ref_bounds: [np.float,np.float]
            The minimum/maximum energy bounds over which to take the reference 
            band.
        
        sub_bounds: np.array([float,float])
            An array of minimum/maximum energy bounds over which each channel of 
            interest is taken. 
        
        data: np.array(float) or stingray.events 
            The data to be fitted, either in the form of a one-dimensional array 
            or a stingray event file. If passing an array, the data has to be 
            stored such that all the real values or moduli are contained in the 
            first half of the array, with the imaginary values or phases in the 
            second half. 

        data_err: np.array(float), optional
            Only required when not passing a stingray event file. Needs to be 
            in the same format as the data array - all moduli/real parts first, 
            all phases/imaginary parts second. 

        freq_grid: np.array(float), optional 
            The grid of Fourier frequency over which to compute the model. If it 
            is not passed explicitely, it is computed from the time resolution 
            and segment size arguments. 
        
        time_grid: np.array(float), optional 
            The grid of times used to generate the Fourier frequency grid. Used 
            for internal model calcluations, if using an impulse response 
            function and the users wishes to use the sinc decomposition method.

        freq_bins: np.array(float), optional 
            The Fourier frequency grids over which energy-dependent data has 
            been averaged. Required for fitting energy-dependent data (e.g. lag 
            energy spectra), and not used for frequency-dependent data.
        
        time_res: np.float, optional 
            The time resolution of the lightcurves used to build the data. It is 
            necessary to build the grid of Fourier frequency over which to 
            compute the model.
        
        seg_size: np.float, optional 
            The size of the segments in which the lightcurves were divided 
            to build the data. It is necessary to build the grid of Fourier 
            frequency over which to compute the model.     
            
        norm; str, optional default = "abs" 
            The normalization of the data products, if they are calculated from 
            a stingray event file. If not specified, absolute rms normalization 
            is used.
            
        frebin_fac: float, default: None
            A user-specified factor by which to rebin the data logarithmically 
            over Fourier frequency when it is loaded through Stingray, should 
            users choose to do so.  
        """
                
        if self.units is None:
            raise AttributeError("Cross spectrum units not defined") 
        if self.dependence is None:
            raise AttributeError("Cross spectrum dependence not defined") 
        if norm is None and getattr(data, '__module__', None) != "stingray.events":
            norm = "abs"
        #combine the edges of the reference and subject bands with those of the matrix
        #then sort+keep only the ones that are not repeated, and rebin the matrix
        #to this grid of channels
        n_bins = len(sub_bounds)+len(ref_bounds)
        rebin_bounds = np.append(sub_bounds,ref_bounds).reshape(n_bins)
        rebin_bounds = np.append(rebin_bounds,response.emin[0])
        rebin_bounds = np.append(rebin_bounds,response.emax[-1])
        rebin_bounds = np.unique(np.sort(rebin_bounds))
        bounds_lo = rebin_bounds[:-1]
        bounds_hi = rebin_bounds[1:] 
        self.response = response.rebin_channels(bounds_lo,bounds_hi) 

        if ref_bounds[0] < self.response.energ_lo[0]:
            ref_bounds[0] = self.response.energ_lo[0]
            warnings.warn("Lower bound of the reference band defined below the "\
                          "start of the instrument response; re-setting to the lowest"\
                          "energy bin instead",UserWarning)   
        if ref_bounds[1] > self.response.energ_hi[-1]:
            ref_bounds[1] = self.response.energ_hi[-1]
            warnings.warn("Upper bound of the reference band defined above the "\
                          "start of the instrument response; re-setting to the highest"\
                          "energy bin instead",UserWarning)                                 
                       
        self.ref_band = ref_bounds
        EnergyDependentFit.__init__(self)  
        self.n_chans = self.ebounds_mask[self.ebounds_mask==True].size
        
        if self.dependence == "frequency":
            self._freq_dependent_cross(data,data_err,
                                       freq_grid,time_grid,
                                       time_res,seg_size,norm,frebin_fac)
        elif self.dependence == "energy":
            self._energ_dependent_cross(freq_bins,data,data_err,
                                        freq_grid,time_grid,
                                        time_res,seg_size)
        else:
            raise AttributeError("Data dependency not found! Set frequency or energy")    
        
        if len(self.freqs) < 50:
            warnings.warn(f"Model frequency grid contains only {len(self.freqs)} bins,"\
                           "model computations might be inaccurate! Suggest you use "\
                           "at least 50!",UserWarning)   
        
        self._set_unmasked_data()
        return

    def _freq_dependent_cross(self,data,data_err=None,
                              freq_grid=None,time_grid=None,
                              time_res=None,seg_size=None,norm=None,
                              frebin_fac=None):
        """
        This method handles loading a cross spectrum to be fitted, when users 
        specify that they want the data to depend from Fourier frequency rather 
        than energy (e.g., in the case of lag-frequency spectra).
        
        Parameters:
        ----------- 
        data: np.array(float) or stingray.events 
            The data to be fitted, either in the form of a one-dimensional array 
            or a stingray event file. If passing an array, the data has to be 
            stored such that all the real values or moduli are contained in the 
            first half of the array, with the imaginary values or phases in the 
            second half. 

        data_err: np.array(float), optional
            Only required when not passing a stingray event file. Needs to be 
            in the same format as the data array - all moduli/real parts first, 
            all phases/imaginary parts second. 

        freq_grid: np.array(float), optional 
            The grid of Fourier frequency over which to compute the model. If it 
            is not passed explicitely, it is computed from the time resolution 
            and segment size arguments. 
        
        time_grid: np.array(float), optional 
            The grid of times used to generate the Fourier frequency grid. Used 
            for internal model calcluations, if using an impulse response 
            function and the users wishes to use the sinc decomposition method.

        time_res: np.float, optional 
            The time resolution of the lightcurves used to build the data. It is 
            necessary to build the grid of Fourier frequency over which to 
            compute the model.
        
        seg_size: np.float, optional 
            The size of the segments in which the lightcurves were divided 
            to build the data. It is necessary to build the grid of Fourier 
            frequency over which to compute the model.  
            
        norm; str, optionalm default = "abs" 
            The normalization of the data products, if they are calculated from 
            a stingray event file. If not specified, absolute rms normalization 
            is used. 
            
        frebin_fac: float, default: None
            A user-specified factor by which to rebin the data logarithmically 
            over Fourier frequency when it is loaded through Stingray, should 
            users choose to do so.         
        """
        
        if getattr(data, '__module__', None) == "stingray.events":
            if time_res is None: 
                raise ValueError("time_res needs to be defined to load from an event file")
            if seg_size is None:
                raise ValueError("seg_size needs to be defined to load from an event file")
            if norm is None:
                raise ValueError("norm needs to be defined to load from an event file")

            events_ref = data.filter_energy_range(self.ref_band)
            ps_ref = AveragedPowerspectrum.from_events(events_ref,
                                                       segment_size=seg_size,
                                                       dt=time_res,norm=norm,
                                                       silent=True)
            if frebin_fac is not None:
                ps_ref = ps_ref.rebin_log(frebin_fac)
            ctrate_ref = get_average_ctrate(events_ref.time,events_ref.gti,seg_size)
            noise_ref = poisson_level(norm=norm, meanrate=ctrate_ref)     
                
            #If we use stingray, we always keep linearly-spaced frequency and 
            #time grids unless the data also gets rebinned, in which case we 
            #use a geometrically spaced time grid
            lc_length = ps_ref.n*time_res
            time_samples = int(lc_length/time_res)
            if frebin_fac is None:
                self._times = np.linspace(time_res,lc_length,time_samples)
            else:
                self._times = np.geomspace(time_res,lc_length,time_samples)
            self.freqs = np.array(ps_ref.freq)

            self.data = []
            self.data_err = []
            
            for i in range(self.n_chans):
                events_sub = data.filter_energy_range([self.response.emin[i],
                                                       self.response.emax[i]])
                cs = AveragedCrossspectrum.from_events(events_sub,events_ref,
                                                       segment_size=seg_size,
                                                       dt=time_res,norm=norm,
                                                       silent=True)
                if frebin_fac is not None:
                    cs = cs.rebin_log(frebin_fac)
                
                if self.units == "lags":
                    lag, lag_err = cs.time_lag() 
                    self.data = np.append(self.data,lag)
                    self.data_err = np.append(self.data_err,lag_err)
                else:
                    ps_sub = AveragedPowerspectrum.from_events(events_sub,
                                                               segment_size=seg_size,
                                                               dt=time_res,norm=norm,
                                                               silent=True)    
                    if frebin_fac is not None:
                        ps_sub = ps_sub.rebin_log(frebin_fac)
                    ctrate_sub = get_average_ctrate(events_sub.time,events_sub.gti,seg_size)                    
                    noise_sub = poisson_level(norm=norm, meanrate=ctrate_sub)                      
                    data_size = len(cs.freq)
                    
                    if self.units == "cartesian":    
                        data_first_dim = np.real(cs.power)
                        data_second_dim = np.imag(cs.power)             
                        error_first_dim = np.sqrt((ps_sub.power*ps_ref.power+ \
                                                   np.real(cs.power)**2- \
                                                   np.imag(cs.power)**2)/(2.*cs.m))
                        error_second_dim = np.sqrt((ps_sub.power*ps_ref.power- \
                                                    np.real(cs.power)**2+ \
                                                    np.imag(cs.power)**2)/(2.*cs.m))
                    elif self.units == "polar":
                        data_first_dim = np.absolute(cs.power)
                        error_first_dim = np.sqrt(ps_sub.power*ps_ref.power/(2.*cs.m))
                        data_second_dim, error_second_dim = cs.phase_lag()
                    
                    if i == 0:
                        self.data = np.append(self.data,data_first_dim)
                        self.data_err = np.append(self.data_err,error_first_dim)
                    else:
                        self.data = np.insert(self.data,i*data_size,data_first_dim)
                        self.data_err = np.insert(self.data_err,i*data_size,error_first_dim) 
                    self.data = np.append(self.data,data_second_dim)
                    self.data_err = np.append(self.data_err,error_second_dim)
        else:
            #when we do not use stingray, we can explicitely pass frequency and 
            #time grids
            if (time_grid is not None and freq_grid is not None):
                self._times = time_grid
                self.freqs = freq_grid
            #or we can explicitely pass a frequency grid alone, and the time 
            #grid is reconstructed automatically 
            elif freq_grid is not None:
                time_res = 0.5/(self.freqs[-1]+self.freqs[0])
                time_samples = (self.freqs.size+1)*2
                lc_length = time_samples*time_res
                #since we are passing a grid by hand, we just default to the 
                #sinc method for safety 
                self._times = np.geomspace(time_res,lc_length,time_samples)        
            else:
                raise ValueError("Frequency and/or time grids undefined")
            self.data = data
            self.data_err = data_err  

        FrequencyDependentFit.__init__(self,self.freqs)  
        self.n_freqs = self.freqs.size
        
        if len(self.data) != len(self.data_err):
            raise AttributeError("Size of data and error are not the same")
        if self.units == "lags":
            if len(self.data)/self.n_chans != self.n_freqs:
                raise AttributeError("Size of frequency grid does not match the data")
        else:
            if len(self.data)/(2.*self.n_chans) != self.n_freqs:
                raise AttributeError("Size of frequency grid does not match the data")
        return

    def _energ_dependent_cross(self,freq_bins,data,data_err,
                               freq_grid=None,time_grid=None,
                               time_res=None,seg_size=None):
        """
        This method handles loading a cross spectrum to be fitted, when users 
        specify that they want the data to depend from photon energy rather 
        than Fourier frequency (e.g., in the case of lag-energy spectra).
        
        Parameters:
        ----------- 
        freq_bins: np.array(float), optional 
            The bounds of the bins in Fourier frequencies over which energy-
            dependent data has been averaged. 
        
        data: np.array(float) or stingray.events 
            The data to be fitted, in the form of a one-dimensional array. The 
            data has to be stored such that all the real values or moduli are 
            contained in the first half of the array, with the imaginary values 
            or phases in the second half. 

        data_err: np.array(float)
            The errors on the data to be fitted. Needs to be in the same format 
            as the data array - all moduli/real parts first, all phases/
            imaginary parts second. 
        
        freq_grid: np.array(float), optional 
            The grid of Fourier frequency over which to compute the model. If it 
            is not passed explicitely, it is computed from the time resolution 
            and segment size arguments. 
        
        time_grid: np.array(float), optional 
            The grid of times used to generate the Fourier frequency grid. Used 
            for internal model calcluations, if using an impulse response 
            function and the users wishes to use the sinc decomposition method.

        time_res: np.float, optional 
            The time resolution of the lightcurves used to build the data. It is 
            necessary to build the grid of Fourier frequency over which to 
            compute the model.
        
        seg_size: np.float, optional 
            The size of the segments in which the lightcurves were divided 
            to build the data. It is necessary to build the grid of Fourier 
            frequency over which to compute the model.     
        """
        
        self.data = data
        self.data_err = data_err
        
        if freq_bins is None:
            raise AttributeError("Fourier frequency bins for data products not defined")             
        FrequencyDependentFit.__init__(self,freq_bins) 

        #we can explicitely pass frequency and time grids
        if (time_grid is not None )&(freq_grid is not None):
            self._times = time_grid
            self.freqs = freq_grid
        #or do so implicetly with a segment size and time resolution
        elif (time_res is not None)&(seg_size is not None):
            n_samples = int(seg_size/time_res)
            freqs = rfftfreq(n_samples,time_res)
            self.freqs = freqs[1:(n_samples+1)//2]        
            self._times = np.linspace(time_res,n_samples*time_res,n_samples)
        #or we can explicitely pass a frequency grid alone, and the time grid is 
        #reconstructed automatically 
        elif freq_grid is not None:
            self.freqs = freq_grid
            time_res = 0.5/(self.freqs[-1]+self.freqs[0])
            time_samples = (self.freqs.size+1)*2
            lc_length = time_samples*time_res
            #since we are passing a grid by hand, we just default to the 
            #sinc method for safety 
            self._times = np.geomspace(time_res,lc_length,time_samples)               
        else:
            raise ValueError("Frequency and/or time grids undefined")         
        
        if len(self.data) != len(self.data_err):
            raise AttributeError("Size of data and error are not the same")
        if self.units == "lags":
            if len(self.data)/self.n_chans != self.n_freqs:
                raise AttributeError("Size of frequency grid does not match the data")
        else:
            if len(self.data)/(2.*self.n_chans) != self.n_freqs:
                raise AttributeError("Size of frequency grid does not match the data")
        return

    def set_model(self,model,model_type="irf",params=None):
        """
        This method is used to pass the model users want to fit to the data. 
        Users also need to specify what quantity the model actually computes - 
        a time-dependent impulse respnse function, a Fourier-frequency dependent
        function, or an explicit value for the cross spectrum. Based on the model 
        type, the class then converts model output to the appropriate spectral 
        timing products automatically. Optionally it is also possible to pass 
        the initial parameter values of the model. 
        
        Parameters:
        -----------            
        model: lmfit.CompositeModel 
            The lmfit wrapper of the model one wants to fit to the data. 
            
        model_type: str, default="irf" 
            A string describing the type of models users want to define. Options 
            are "irf" for an impuplse response function, "transfer" for a 
            transfer functino, and "cross" for the actual cross spectrum.
            
        params: lmfit.Parameters, default: None 
            The parameter values from which to start evalauting the model during
            the fit. If it is not provided, all model parameters will default 
            to 0, set to be free, and have no minimum or maximum bound. 
        """
    
        if model_type not in self._supported_models:
            raise AttributeError("Unsupported model type")  
        self.model_type = model_type
        self.crossspec = CrossSpectrum(self._times,freqs=self.freqs,energ=self.energs)
        self.model = model 
        if params is None:
            self.model_params = self.model.make_params(verbose=True)
        else:
            self.model_params = params
        return 
        
    def set_psd_weights(self,psd_weights): 
        """
        This method is necessary when users define models from an impulse 
        response or transfer functins, and sets the power spectrum used as 
        weights when calculating spectral timing products.
        
        Parameters:
        -----------
        input_power: np.array(float) or PowerSpectrum
            Either an array of size (len(freqs)) that is to be used as the 
            weighing power spectrum when computing the cross spectrum, or an 
            nDspec PowerSpectrum object. Both have to be defined over the class 
            internal Fourier frequency grid. 
        """      
        if self.model_type != "cross":
            self.crossspec.set_psd_weights(psd_weights)
        else:
            raise UserWarning("Power spectrum weights not needed, skipping")
        return 
    
    def eval_model(self,params=None,fold=True,mask=True):
        """
        This method is used to evaluate and return the model values for a given 
        set of parameters, over the internal energy and frequency grids. By 
        default the model is evaluated using the parameters values stored 
        internally in the model_params attribute. The model is always folded 
        through the instrument response, returning either all or only the 
        noticed channels. The reference band is always that set in set_data.
        
        Parameters:
        -----------                         
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are 
            provided, the model_params attribute is used.
            
        fold: bool, default True
            A boolean switch to choose whether to fold the model through the 
            instrument response or not.

        mask: bool, default True
            A boolean switch to choose whether to mask the model output to only 
            include the noticed energy channels, or to also return the ones 
            that have been ignored by the users. 
            
        Returns:
        --------
        model: np.array(float)
            The model evaluated over the given energy grid, for the given input 
            parameters.  
        """  
               
        #evaluate the model for the chosen parameters
        if params is None:
            params= self.model_params
        model_eval = self.model.eval(params,ear=self.ear,energs=self.energs,
                                     freqs=self.freqs,times=self._times)
        #store the model in the cross spectrum, depending on the type
        #transposing is required to ensure the units are correct 
        crossspec = self._to_cross_spec(model_eval)
            
        #fold the instrument response:
        if fold is True:
            model_eval = self.response.convolve_response(crossspec,
                                                          units_in="rate",
                                                          units_out="channel")
        else:
            model_eval = crossspec 
            
        #return the appropriately structured products
        model = self._return_dependent_model(model_eval,params)

        if mask is True:
            model = self._filter_2d_by_mask(model)

        return model
    
    def _to_cross_spec(self,model_eval):
        """
        This method converts the model evaluation results into the cross spectrum format
        and sets the cross spectrum attribute appropriately. Returns the cross-spectrum
        result.

        Parameters:
        -----------
        model_eval: np.array(float)
            The model evaluation results to convert into the cross spectrum format.

        Returns:
        --------
        crossspec: CrossSpectrum
            The cross spectrum object containing the converted model evaluation results.
        """
        if self.model_type == "irf":
            self.crossspec.cross_from_irf(signal=np.transpose(model_eval),
                                          ref_bounds=self.ref_band)
        elif self.model_type == "transfer":
            self.crossspec.cross_from_transfer(signal=np.transpose(model_eval),
                                               ref_bounds=self.ref_band)
        elif self.model_type == "cross":           
            self.crossspec.cross = np.transpose(model_eval)
        else:
            raise AttributeError("Model type not supported")
        
        return self.crossspec
    
    def _return_dependent_model(self,model_eval,params=None):
        """
        This method returns the appropriately structured
        products based on the energy or frequency dependence
        of the model.
        """
        if self.dependence == "frequency":
            model = self._freq_dependent_model(model_eval)
        elif self.dependence == "energy":
            model = self._energ_dependent_model(model_eval,params)
        else:
            raise AttributeError("Product dependency not supported")
        
        return model

    def _freq_dependent_model(self,cross_eval):
        """
        This method takes a model cross spectrum evaluated by the class and 
        folded through the instrument response, and converts it to the 
        Fourier-frequency dependent spectral timing products chosen by the user 
        through the ``settings'' attribute. 
        
        Parameters:
        -----------
        cross_eval: np.array(float, float)
            A matrix of size (n_freq,_all_chans) containing the model cross 
            spectrum folded thruogh the instrument response matrix 
            
        Returns:
        --------
        model: np.array(float) 
            A one-dimensional array of size (n_freq*_all_chans) if fitting lags 
            and (2*n_freq*_all_chans) otherwise, containing the spectral timing 
            products resulting from the model evaluation. This is the array that 
            is compared to the data when fitting, and the two therefore must have 
            the same format. 
        """
    
        model = []
        sub_bounds = np.array([self._ebounds_unmasked-0.5*self._ewidths_unmasked,
                               self._ebounds_unmasked+0.5*self._ewidths_unmasked])
        sub_bounds = np.transpose(sub_bounds)
        
        if self.units == "lags":
            for i in range(self._all_chans):
                model_eval = cross_eval.lag_frequency(sub_bounds[i])
                model = np.append(model,model_eval)
        elif self.units == "cartesian":
            real = []
            imag = []             
            for i in range(self._all_chans):
                real_eval = cross_eval.real_frequency(sub_bounds[i])
                imag_eval = cross_eval.imag_frequency(sub_bounds[i])            
                real = np.append(real,real_eval)
                imag = np.append(imag,imag_eval)
            model = np.concatenate((real,imag))
        elif self.units == "polar":
            mod = []
            phase = []            
            for i in range(self._all_chans):
                mod_eval = cross_eval.mod_frequency(sub_bounds[i])
                phase_eval = cross_eval.phase_frequency(sub_bounds[i])         
                mod = np.append(mod,mod_eval)
                phase = np.append(phase,phase_eval)            
            model = np.concatenate((mod,phase))
        else:
            raise AttributeError("Incorrect model units, set lags, cartesian or polar")               
        return model

    def _energ_dependent_model(self,cross_eval,params):
        """
        This method takes a model cross spectrum evaluated by the class and 
        folded through the instrument response, and converts it to the energy
        dependent spectral timing products chosen by the user through the
        ``settings'' attribute. Depending on fitter settings, it also enables 
        the automatic renormalization of phases and modulii to correct for 
        instrument calibration and/or knowledge of the underlying physical 
        powerspectrum responsible for the variability.  For more discussion,
        see Appendix E in Mastroserio et al. 2018: 
        https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.4027M/abstract
        
        Parameters:
        -----------
        cross_eval: np.array(float, float)
            A matrix of size (n_freq,_all_chans) containing the model cross 
            spectrum folded thruogh the instrument response matrix 
            
        params: lmfit.parameters 
            An lmfit parameters object used to store the model parameter values 
            used for the evaluation. It is necessary when users choose to 
            renormalize phases or modulii. 
            
        Returns:
        --------
        model: np.array(float) 
            A one-dimensional array of size (n_freq*_all_chans) if fitting lags 
            and (2*n_freq*_all_chans) otherwise, containing the spectral timing 
            products resulting from the model evaluation. This is the array that 
            is compared to the data when fitting, and the two therefore must have 
            the same format. 
        """
        
        model = []
        
        if self.units == "lags":  
            for i in range(self._all_freqs):
                f_mean = 0.5*(self._freqs_unmasked[1:]+self._freqs_unmasked[:-1])
                if self.renorm_phase is True:
                    par_key = 'phase_renorm_'+str(i+1)
                    phase_pars = LM_Parameters()
                    phase_pars.add('renorm',value=params[par_key].value,
                                   min=params[par_key].min,max=params[par_key].max,
                                   vary=params[par_key].vary)
                    
                    phase_model = cross_eval.phase_energy([self._freqs_unmasked[i],self._freqs_unmasked[i+1]])
                    model_eval = self.phase_renorm_model.eval(phase_pars,array=phase_model)/(2*np.pi*f_mean[i])  
                else:
                    model_eval = cross_eval.lag_energy([self._freqs_unmasked[i],self._freqs_unmasked[i+1]])
                model = np.append(model,model_eval)
        elif self.units == "cartesian":
            real = []
            imag = [] 
            for i in range(self._all_freqs):
                real_eval = cross_eval.real_energy([self._freqs_unmasked[i],self._freqs_unmasked[i+1]])
                imag_eval = cross_eval.imag_energy([self._freqs_unmasked[i],self._freqs_unmasked[i+1]])
                if self.renorm_modulus is True:
                    par_key = 'mods_renorm_'+str(i+1)
                    mods_pars = LM_Parameters()
                    mods_pars.add('renorm',value=params[par_key].value,
                                   min=params[par_key].min,max=params[par_key].max,
                                   vary=params[par_key].vary)                                       
                    real_eval = self.mod_renorm_model.eval(mods_pars,array=real_eval)                                     
                    imag_eval = self.mod_renorm_model.eval(mods_pars,array=imag_eval)
                if self.renorm_phase is True:
                    par_key = 'phase_renorm_'+str(i+1)
                    angle = self.model_params[par_key].value
                    real_eval = np.cos(angle)*real_eval - np.sin(angle)*imag_eval
                    imag_eval = np.cos(angle)*imag_eval + np.sin(angle)*real_eval
                real = np.append(real,real_eval)
                imag = np.append(imag,imag_eval)
            model = np.concatenate((real,imag))
        elif self.units == "polar":
            mod = []
            phase = []
            for i in range(self._all_freqs):
                mod_model = cross_eval.mod_energy([self._freqs_unmasked[i],self._freqs_unmasked[i+1]])
                phase_model = cross_eval.phase_energy([self._freqs_unmasked[i],self._freqs_unmasked[i+1]])
                if self.renorm_modulus is True:
                    par_key = 'mods_renorm_'+str(i+1)
                    mods_pars = LM_Parameters()
                    mods_pars.add('renorm',value=params[par_key].value,
                                   min=params[par_key].min,max=params[par_key].max,
                                   vary=params[par_key].vary)             
                    mod_model = self.mod_renorm_model.eval(mods_pars,array=mod_model)                     
                if self.renorm_phase is True:                    
                    par_key = 'phase_renorm_'+str(i+1)
                    phase_pars = LM_Parameters()
                    phase_pars.add('renorm',value=params[par_key].value,
                                   min=params[par_key].min,max=params[par_key].max,
                                   vary=params[par_key].vary)
                    phase_model = self.phase_renorm_model.eval(phase_pars,array=phase_model)                   
                mod = np.append(mod,mod_model)
                phase = np.append(phase,phase_model)
            model = np.concatenate((mod,phase))
        else:
            raise AttributeError("Incorrect model units, set lags, cartesian or polar")                  
        return model

    def renorm_phases(self,switch):
        """
        Setter method to enable the phase renormalization when fitting energy 
        depenent products. This renormalization is intended to correct for 
        small uncertainties in the instrument response function, which can 
        affect the phase of the cross spectrum. For more discussion, see  
        Appendix E in Mastroserio et al. 2018: 
        https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.4027M/abstract
        This setting will NOT affect the modulus of a cross spectrum, only the 
        phase (and therefore it will affect the real and imaginary parts).
        
        Parameters:
        -----------
        switch: bool 
            A boolean to track whether phase renormalization is enabled or not.
            If it is, the method modifies the defined model and its parameters 
            automatically. 
        """
        #add complaint if people activate this for freq dependency        
        self.renorm_phase = switch
        if self.renorm_phase is True:
            #if we choose to renormalize the phase, we need to modify the model 
            #definition and its parameters to include the phase renormalization 
            #factors 
            self.phase_renorm_model = LM_Model(self._renorm_phase)
            phase_pars = LM_Parameters()
            for index in range(self.n_freqs):   
                phase_pars.add('phase_renorm_'+str(index+1), 
                               value=0,min=-0.05,max=0.05,vary=True)            
            self.model_params = self.model_params + phase_pars       
        return
        
    def _renorm_phase(self,array,renorm):
        """
        This method contains a model function to add a phase to an exisisting 
        array, and is used exclusively after being wrapped by lmfit to renormalize 
        the phases of energy-dependent cross spectral products.
       
        Parameters:
        -----------
        array: np.array(float)
            The array of energy depdent cross spectrum phases in a given Fourier
            frequnecy interval to be renormalized 
            
        renorm: float 
            The small phase to be added to re-normalize the cross spectrum. 
            
        Returns:
        --------
        array+renorm 
            The new, renormalized phase. 
        """
    
        return array + renorm

    def renorm_mods(self,switch):
        """
        Setter method to enable the modulus renormalization when fitting energy 
        depenent products. This renormalization is intended to correct for 
        not knowing the correct shape of the power spectrum responsible the 
        observed variability. See Mastroserio et al. 2018 for further discussion:
        https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.4027M/abstract
        This setting will NOT affect the phase of a cross spectrum, only the 
        modulus (and therefore it will affect the real and imaginary parts).   
        
        Parameters:
        -----------
        switch: bool 
            A boolean to track whether modulus renormalization is enabled or not.
            If it is, the method modifies the defined model and its parameters 
            automatically. 
        """
        #add complaint if people activate this for freq dependency
        self.renorm_modulus = switch
        if self.renorm_modulus is True:
            #if we choose to renormalize the modulus, we need to modify the model 
            #definition and its parameters to include the modulus renormalization 
            #factors 
            self.mod_renorm_model = LM_Model(self._renorm_modulus)
            mods_pars = LM_Parameters()
            for index in range(self.n_freqs):   
                mods_pars.add('mods_renorm_'+str(index+1), 
                               value=1,min=0.5,max=2.0,vary=True)            
            self.model_params = self.model_params + mods_pars               
        return

    def _renorm_modulus(self,array,renorm):
        """
        This method contains a model function to add renormalize the modulus of
        an exisisting array, and is used exclusively after being wrapped by 
        lmfit to renormalize the modulus of energy-dependent cross spectral
        products.
       
        Parameters:
        -----------
        array: np.array(float)
            The array of energy depdent cross spectrum modulus in a given Fourier
            frequnecy interval to be renormalized 
            
        renorm: float 
            The renormalization factor by which to multiply the modulus of the 
            cross spectrum. 
            
        Returns:
        --------
        array*renorm 
            The new, renormalized modulus. 
        """
    
        return renorm*array

    def _minimizer(self,params):
        """
        This method is used exclusively when running a minimization algorithm.
        It evaluates the model for an input set of parameters, and then returns 
        the residuals in units of contribution to the total chi squared 
        statistic.
        
        Parameters:
        -----------                         
        params: lmfit.Parameters
            The parameter values to use in evaluating the model. These will vary 
            as the fit runs.
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each bin.            
        """
        model = self.eval_model(params)  
          
        if self.likelihood == "chisq":
            residuals, _ = self.get_residuals("chisq",model=model,mask=True)
        elif self.likelihood == "custom":
            residuals, _ = self.get_residuals("custom",model=model,mask=True)
        else:
            raise AttributeError("Likelihood type not supported")
        
        return residuals

    def _plot_format(self,array,twod=False):
        """
        This method reshapes a flat array defined over every energy and frequency 
        bin (data, errors, model...) for plotting, with ignored bins set to NaN 
        so that they are ignored correctly by all matplotlib functions. 
        
        Parameters:
        -----------
        array: np.array(float)
            The array to reshape, of size (_all_bins) for lags and 
            (2*_all_bins) otherwise.
            
        twod: bool, default=False
            If False, the array is returned as one row per noticed spectrum, 
            for one-dimensional plots. If True, it is returned over the full 
            energy and Fourier frequency grid, for two-dimensional plots.
            
        Returns:
        --------
        array: np.array(float)
            The reshaped array. If twod is False, it has size 
            (n_quantities,n_spectra,n_x), where n_x is _all_freqs or _all_chans 
            as appropriate for dependence of the data. If twod is True, it has 
            size (n_quantities,_all_chans,_all_freqs). n_quantities is 1 for 
            lags and 2 otherwise.
        """
        
        if self.dependence == "frequency":
            array = array.reshape((-1,self._all_chans,self._all_freqs))
        else:
            array = array.reshape((-1,self._all_freqs,self._all_chans))
            array = np.transpose(array,(0,2,1))
        twod_mask = self.ebounds_mask.reshape((self._all_chans,1))*self.freqs_mask
        array = np.where(twod_mask,array,np.nan)
        
        if twod is True:
            return array
        elif self.dependence == "frequency":
            return array[:,self.ebounds_mask,:]
        else:
            return np.transpose(array,(0,2,1))[:,self.freqs_mask,:]
    
    def plot_data(self,panels=None,return_plot=False,data_kwargs=None):
        """
        This method plots the cross spectrum loaded by the user as a function of 
        the unit dependence specified (ie, Fourier frequency or energy). If the 
        data is loaded is two-dimensional - as is the case for frequency 
        dependent products in multiple subject bands, or energy dependent products 
        in multiple Fourier frequency bins, the method plots every loaded 
        spectrum in a single plot. 
        
        Regardless of the data format, only the noticed energy channels are 
        plotted. Note that if the user is ignoring energy bins that are not at 
        the limit of the channel grid (e.g., between 5 and 7 keV for a grid of
        channels between 0.5 and 10 keV), then the automated legend will not 
        label the spectra correctly.         
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------          
        panels : matplotlib.axes or numpy.ndarray(matplotlib.axes), default=None
            An existing panel to draw into, e.g as returned by calling plot_data 
            from a second fitter. If None, a new figure is created.
                  
        return_plot: bool, default=False
            A boolean to decide whether to return the figure object containing 
            the plot or not.
            
        data_kwargs: dict, default=None 
            Keyword arguments for the data points in the plot.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.

        panels: matplotlib.axes, optional 
            The panel containing the plot produced by the method.
        """
    
        if self.dependence == "frequency":
            x_axis = self._freqs_unmasked
            x_bars = 0.5*np.append(np.diff(x_axis),x_axis[-1]-x_axis[-2])
            x_label = "Frequency (Hz)"
            spec_lo = self.emin
            spec_hi = self.emax
            units = "keV"
        else:
            x_axis = self._ebounds_unmasked
            x_bars = 0.5*self._ewidths_unmasked
            x_label = "Energy (keV)"
            spec_lo = self.fmin
            spec_hi = self.fmax
            units = "Hz"
        
        if self.units == "lags":
            y_labels = ["Lag (s)"]
        elif self.units == "polar":
            y_labels = ["Modulus","Phase"]
        else:
            y_labels = ["Real part","Imaginary part"]
        n_quant = len(y_labels)
        
        data = self._plot_format(self._data_unmasked)
        data_err = self._plot_format(self._data_err_unmasked)
        
        layout_kwargs = dict(ncols=n_quant,figsize=(6.5*n_quant,4.5))
        if panels is None:
            plot_layout = Plotting.make_layout(**layout_kwargs)
            fig, panels = Plotting.make_panels(plot_layout,squeeze=False)
        else:
            #This method only ever draws one row and multiple columns 
            #plotting the model and data in 2d might require more 
            panels = np.reshape(np.asarray(panels,dtype=object),(1,n_quant))
            fig = panels[0,0].get_figure()

        for col in range(n_quant):
            #the first quantity is plotted in log space unless it is a lag 
            log_yaxis = (col == 0) and (self.units != "lags")
            if log_yaxis is False:
                panels[0,col].axhline(0,color='black',ls=':',zorder=3)        
            for i in range(len(spec_lo)):
                spec_kwargs = dict(label=f"{spec_lo[i]:.1f}-{spec_hi[i]:.1f} {units}")
                if data_kwargs is not None:
                    spec_kwargs.update(data_kwargs)
                panel_data = Plotting.make_panel_data(x_points=x_axis,
                                                      x_bars=x_bars,
                                                      y_points=data[col,i],
                                                      y_bars=data_err[col,i],
                                                      x_label=x_label,
                                                      y_label=y_labels[col])
                Plotting.draw_main_panel(panels[0,col],panel_data,color="C"+str(i),
                                         draw_model=False,log_yaxis=log_yaxis,
                                         data_kwargs=spec_kwargs)
                                     
        panels[0,-1].legend(loc="best",ncol=2,fontsize=12)

        panels = panels[0]
        if self.units == "lags":
            panels = panels[0]
        
        if return_plot is True:
            return fig, panels
        else:
            return
            
    def plot_data_2d(self,use_phase=False,cmaps=None,return_plot=False,
                     mesh_kwargs=None): 
        """
        This method plots the cross spectrum loaded by the user in two dimensions 
        as a function of both Fourier frequency or energy. Regardless of the data
        format, only the noticed energy channels are plotted. Note that due to 
        how matplotlib handles two-dimensional plots, the bounds of the bins 
        shown in the plot will differ slightly from those where the data is 
        defined.  
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        use_phase: bool, default=False
            A boolean used exclusively when plotting time lags. If it is true, 
            it converts the lags to phases for ease of visualization over a large 
            range in lag timescales.

        cmaps: list(str), default=None
            A list of colormaps of each panel. If set to none, lags use 
            `twilight`, modulus uses `magma`,real+imaginary use`PuOr`.            
                  
        return_plot: bool, default=False
            A boolean to decide whether to return the figure object containing 
            the plot or not.          
            
        mesh_kwargs: dict, default=None 
            Keyword arguments for the 2d colormesh, applied to every panel.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
            
        panels: numpy.ndarray(matplotlib.axes), optional 
            The panels containing the plot.
        """

        if self.dependence == "frequency":
            x_axis = self._freqs_unmasked
        else:
            x_axis = 0.5*(self._freqs_unmasked[1:]+self._freqs_unmasked[:-1])
        ylim = [self.emin[0],self.emax[-1]]
        
        if self.units == "lags":
            z_labels = ["Lag (s)"]
            default_cmaps = ["twilight"]
        elif self.units == "polar":
            z_labels = ["log10(Modulus)","Phase (rad)"]
            default_cmaps = ["magma","twilight"]
        else:
            z_labels = ["Real part","Imaginary part"]
            default_cmaps = ["PuOr","PuOr"]
        
        if cmaps is None:
            cmaps = default_cmaps
        
        n_quant = len(z_labels)
        
        values = self._plot_format(self._data_unmasked,twod=True)
        
        if self.units == "polar":
            values[0] = np.log10(values[0])
        elif self.units == "lags" and use_phase is True:
            values = values*2.*np.pi*x_axis
            z_labels = ["Phase (rad)"]
        
        plot_layout = Plotting.make_layout(ncols=n_quant,colorbars=True,
                                           figsize=(6.5*n_quant,4.5))
        fig, panels = Plotting.make_panels(plot_layout,squeeze=False)
        panels = np.atleast_1d(panels)
        
        for col in range(n_quant):
            mesh_data = Plotting.make_mesh_data(x_points=x_axis,
                                                y_points=self._ebounds_unmasked,
                                                z_values=values[col],
                                                x_label="Frequency (Hz)",
                                                y_label="Energy (keV)",
                                                z_label=z_labels[col])
            Plotting.draw_colormesh_panel(panels[0,col],mesh_data,cmap=cmaps[col],
                                          diverging=(self.units != "polar" or col == 1),
                                          log_xaxis=True,log_yaxis=True,ylim=ylim,
                                          mesh_kwargs=mesh_kwargs)
        panels = panels[0]
        if self.units == "lags":
            panels = panels[0]
        
        if return_plot is True:
            return fig, panels
        else:
            return 

    def plot_model(self,plot_data=True,params=None,use_phase=False,
                      residuals="chisq",panels=None,return_plot=False,
                      data_kwargs=None,model_kwargs=None,resid_kwargs=None):
        """
        This method plots the model defined by the user as a function of  the  
        unit dependence specified (ie, Fourier frequency or energy). 
        
        By default the method also plots the data loaded as well as the 
        residuals. Additionally, by default the model is evaluated using the 
        parameter values currently stored in the class, but it is also posssible 
        to pass a different set of parameters for model evaluation.
        
        If the  data is loaded is two-dimensional - as is the case for  
        frequency dependent products in multiple subject bands, or energy
        dependent products in multiple Fourier frequency bins, the method plots 
        every loaded spectrum in a single plot. 
        
        Regardless of the data format, only the noticed energy channels are 
        plotted. Note that if the user is ignoring energy bins that are not at 
        the limit of the channel grid (e.g., between 5 and 7 keV for a grid of
        channels between 0.5 and 10 keV), then the automated legend will not 
        label the spectra correctly.         
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------            
        plot_data: bool, default=True
            If true, both model and data are plotted; if false, just the model. 
                        
        params: lmfit.parameters, default=None 
            The parameters to be used to evaluate the model. If False, the set 
            of parameters stored in the class is used   
            
        residuals: str, default="chisq"
            The units to use for the residuals. If residuals="chisq", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model. If using a custom  
            likelihood, the residuals are computed from it.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure object containing 
            the plot or not.
            
        data_kwargs, model_kwargs, resid_kwargs: dict, default=None 
            Keyword arguments for each element of the plot.

        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.

        panels: matplotlib.axes or numpy.ndarray(matplotlib.axes), optional 
            The panel(s) containing the plot produced by the method.
        """

        if self.dependence == "frequency":
            x_axis = self._freqs_unmasked
            x_bars = 0.5*np.append(np.diff(x_axis),x_axis[-1]-x_axis[-2])
            x_label = "Frequency (Hz)"
            spec_lo = self.emin
            spec_hi = self.emax
            units = "keV"
        else:
            x_axis = self._ebounds_unmasked
            x_bars = 0.5*self._ewidths_unmasked
            x_label = "Energy (keV)"
            spec_lo = self.fmin
            spec_hi = self.fmax
            units = "Hz"
        
        if plot_data is True:
            if residuals == "chisq":
                res_label = "$\\Delta\\chi$"
            elif residuals == "ratio":
                res_label = "Data/model"
            else:
                raise ValueError("Residual format not supported")
        
        if self.units == "lags":
            y_labels = ["Lag (s)"]
        elif self.units == "polar":
            y_labels = ["Modulus","Phase"]
        else:
            y_labels = ["Real part","Imaginary part"]
        n_quant = len(y_labels)

        model = self.eval_model(params=params,mask=False)
        model_vals = self._plot_format(model)

        if self.dependence == "frequency":
            model_points = x_axis 
            model_edges = None
        elif self.dependence == "energy":
            model_points = None 
            model_edges = np.append(self._emin_unmasked,self._emax_unmasked[-1]) 

        if residuals is None:
            residuals = self.likelihood   
        
        resid, reserr = None, None
        data, data_err = None, None 
        data = self._plot_format(self._data_unmasked)
        data_err = self._plot_format(self._data_err_unmasked)
        #this horrible jank here is necessary to avoid raising annoying 
        #warnings about bins that are not being noticed, while still
        #supporting having gaps in the noticed bins 
        res, err = self.get_residuals(residuals,
                                      model=self._filter_2d_by_mask(model))
        noticed = self._filter_2d_by_mask(np.arange(self._data_unmasked.size))
        resid = np.full(self._data_unmasked.size,np.nan)
        reserr = np.full(self._data_unmasked.size,np.nan)
        resid[noticed] = res
        reserr[noticed] = err
        resid = self._plot_format(resid)
        reserr = self._plot_format(reserr)

        layout_kwargs = dict(figsize=(6.5*n_quant,4.5),ncols=n_quant)
        if plot_data is True:
            layout_kwargs = dict(nrows=2,ncols=n_quant,height_ratios=[2,1],
                                 sharex=True,figsize=(6.5*n_quant,6.))        
        if panels is None:
            plot_layout = Plotting.make_layout(**layout_kwargs)
            fig, panels = Plotting.make_panels(plot_layout,squeeze=False)
        else:
            #To lay out the panels correctly we need to reshape to a nrowsxncols
            #grid layout
            panels = np.reshape(np.asarray(panels,dtype=object),
                                (layout_kwargs.get("nrows",1),n_quant))
            fig = panels[0,0].get_figure()
        
        for col in range(n_quant):
            log_yaxis = (col == 0) and (self.units != "lags")
            if log_yaxis is False:
                panels[0,col].axhline(0,color='black',ls=':',zorder=3)        
            for i in range(len(spec_lo)):
                spec_kwargs = dict(label=f"{spec_lo[i]:.1f}-{spec_hi[i]:.1f} {units}")
                if data_kwargs is not None:
                    spec_kwargs.update(data_kwargs)
                panel_data = Plotting.make_panel_data(x_points=x_axis,
                                                      x_bars=x_bars,
                                                      y_points=data[col,i],
                                                      y_bars=data_err[col,i],
                                                      model_edges=model_edges,
                                                      model_points=model_points,
                                                      model_vals=model_vals[col,i], 
                                                      resid=resid[col,i],
                                                      reserr=reserr[col,i], 
                                                      x_label=x_label,
                                                      y_label=y_labels[col],
                                                      res_label=res_label)
                Plotting.draw_main_panel(panels[0,col],panel_data,color="C"+str(i),
                                         log_yaxis=log_yaxis,
                                         data_kwargs=spec_kwargs,
                                         model_kwargs=model_kwargs)
                if plot_data is True:
                    panels[0,col].set_xlabel("")
                    Plotting.draw_residual_panel(panels[1,col],panel_data,residuals,
                                                 color="C"+str(i),
                                                 resid_kwargs=resid_kwargs)
                          
        panels[0,-1].legend(loc="best",ncol=2,fontsize=12)

        if self.units == "lags":
            panels = panels[:,0]
        if plot_data is False:
            panels = panels[0]        
        if return_plot is True:
            return fig, panels
        else:
            return        
        
    def plot_model_2d(self,plot_data=True,params=None,use_phase=False,cmaps=None,
                      residuals=None,return_plot=False,mesh_kwargs=None):
        """
        This method plots the model and data loaded by the user in two dimensions 
        as a function of both Fourier frequency or energy. Regardless of the data
        format, only the noticed energy channels are plotted. Note that due to 
        how matplotlib handles two-dimensional plots, the bounds of the bins 
        shown in the plot will differ slightly from those where the data is 
        defined.           
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------           
        plot_data: bool, default=True
            If true, both model and data are plotted; if false, just the model. 

        params: lmfit.parameters, default=None 
            The parameters to be used to evaluate the model. If False, the set 
            of parameters stored in the class is used 
            
        use_phase: bool, default=False
            A boolean used exclusively when plotting time lags. If it is true, 
            it converts the lags to phases for ease of visualization over a large 
            range in lag timescales.

        cmaps: list(str)
            A list of colormaps of each panel. If set to none, lags use 
            `twilight`, modulus uses `magma`,real+imaginary use`PuOr`, and 
            residuals use `BrBG`.  

        residuals: str, default=None
            The units to use for the residuals. If residuals="chisq", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model. If using a custom  
            likelihood, the residuals are computed from it. If nothing is 
            specified, the method defaults to whatever likelihood was used in 
            the fit.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure object containing 
            the plot or not.
            
        mesh_kwargs: dict, default=None 
            Keyword arguments for the 2d colormesh, applied to every panel.

        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.

        panels: matplotlib.axes or numpy.ndarray(matplotlib.axes), optional 
            The panel(s) containing the plot produced by the method.
        """

        if self.dependence == "frequency":
            x_axis = self._freqs_unmasked
        else:
            x_axis = 0.5*(self._freqs_unmasked[1:]+self._freqs_unmasked[:-1])
        ylim = [self.emin[0],self.emax[-1]]
        
        if self.units == "lags" and use_phase is True:
            z_labels = ["Phase (rad)"]
            default_cmaps = ["twilight","BrBG"]
        elif self.units == "lags":
            z_labels = ["Lag (s)"]
            default_cmaps = ["twilight","BrBG"]
        elif self.units == "polar":
            z_labels = ["log10(Modulus)","Phase (rad)"]
            default_cmaps = ["magma","twilight","BrBG"]
        else:
            z_labels = ["Real part","Imaginary part"]
            default_cmaps = ["PuOr","PuOr","BrBG"]
        if cmaps is None:
            cmaps = default_cmaps
        n_quant = len(z_labels)         

        if params is None:
            params = self.model_params
        model = self.eval_model(params=params,mask=False)
        
        rows = ["model"]
        meshes = dict(model=self._plot_format(model,twod=True))
        if plot_data is True:
            rows = ["data","model","residuals"]
            meshes["data"] = self._plot_format(self._data_unmasked,twod=True)

            if residuals == "chisq":
                res_label = "$\\Delta\\chi$"
            elif residuals == "ratio":
                res_label = "Data/model"
            else:
                res_label = "Residuals"
            if residuals is None:
                residuals = self.likelihood
            res, _ = self.get_residuals(residuals,
                                        model=self._filter_2d_by_mask(model))
            noticed = self._filter_2d_by_mask(np.arange(self._data_unmasked.size))
            #this jank is necessary to support ignored bins in the 2d plot
            resid = np.full(self._data_unmasked.size,np.nan)
            resid[noticed] = res
            meshes["residuals"] = self._plot_format(resid,twod=True)

        for name in meshes.keys():
            if name == "residuals":
                continue
            if self.units == "polar":
                meshes[name][0] = np.log10(meshes[name][0])
            elif self.units == "lags" and use_phase is True:
                meshes[name] = meshes[name]*2.*np.pi*x_axis

        if self.units == "lags":
            plot_layout = Plotting.make_layout(ncols=len(rows),colorbars=True,
                                               figsize=(4.9*len(rows),4.0))
        else: 
            plot_layout = Plotting.make_layout(nrows=len(rows),ncols=n_quant,
                                               colorbars=True,
                                               figsize=(4.9*n_quant,4.0*len(rows)))
        fig, panels = Plotting.make_panels(plot_layout,squeeze=False)            
        if self.units == "lags":
            panels = panels.T
             
        for col in range(n_quant):
            #the modulus is the only quantity without a diverging colour scale
            diverging = (self.units != "polar") or (col == 1)
            #data and model share the colour scale of the data
            shared = {}
            ticks = None
            if plot_data is True and diverging is True:
                shared["norm"], _ = Plotting.get_diverging_norm(meshes["data"][col])
            elif plot_data is True:
                shared["vmin"] = np.nanmin(meshes["data"][col])
                shared["vmax"] = np.nanmax(meshes["data"][col])            
            for row, name in enumerate(rows):
                if name == "residuals":
                    cmap = cmaps[-1]
                    z_label = res_label
                    style = {}
                    panel_diverging = True
                else:
                    cmap = cmaps[col]
                    z_label = z_labels[col]
                    style = dict(shared)
                    panel_diverging = diverging and len(shared) == 0
                if mesh_kwargs is not None:
                    style.update(mesh_kwargs)
                mesh_data = Plotting.make_mesh_data(x_points=x_axis,
                                                    y_points=self._ebounds_unmasked,
                                                    z_values=meshes[name][col],
                                                    x_label="Frequency (Hz)",
                                                    y_label="Energy (keV)",
                                                    z_label=z_label)
                mesh = Plotting.draw_colormesh_panel(panels[row,col],mesh_data,cmap=cmap,
                                                     diverging=panel_diverging,
                                                     colorbar=(name == "residuals" or plot_data is False),
                                                     log_xaxis=True,log_yaxis=True,
                                                     ylim=ylim,mesh_kwargs=style)
                panels[row,col].set_title(name.capitalize())
                if name == "model":
                    mesh_model = mesh
                #set labels so only the leftmost/bottom panels retain them 
                if (self.units == "lags" and row > 0) or col > 0:
                    panels[row,col].set_ylabel("")
                    panels[row,col].tick_params(labelleft=False)
                if self.units != "lags" and row < len(rows)-1:
                    panels[row,col].set_xlabel("")
                    panels[row,col].tick_params(labelbottom=False)
            
            if plot_data is True:
                #set aspect to keep the bar width consistent for all plots 
                aspect = 20
                if self.units != "lags":
                    aspect = 40
                bar = fig.colorbar(mesh_model,ax=[panels[0,col],panels[1,col]],
                                   ticks=ticks,aspect=aspect)
                bar.set_label(z_labels[col])
            
        if self.units == "lags":
            panels = panels[:,0]
            if plot_data is False:
                panels = panels[0]
        
        if return_plot is True:
            return fig, panels
        else:
            return
