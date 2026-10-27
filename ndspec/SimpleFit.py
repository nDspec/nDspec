import numpy as np
import warnings

from lmfit import fit_report, minimize 
from lmfit import Parameters as LM_Parameters
from lmfit.printfuncs import gformat

from .Likelihoods import cstat, chisq, ratio

class SimpleFit():
    """
    Generic least-chi squared fitter class, used internally to store methods 
    that are shared between all the fitter types. 
           
    Attributes:
    -----------
    model: lmfit.CompositeModel 
        A lmfit CompositeModel object, which contains a wrapper to the model 
        component(s) one wants to fit to the data. 
   
    model_params: lmfit.Parameters 
        A lmfit Parameters object, which contains the parameters for the model 
        components.
   
    likelihood: str
        A string that allows to switch between different fit statistics; which 
        one is available depends on the type of fitter object. Uses chi-squared 
        likelihood by default. Users can set different likelihoods with the 
        appropriate setter method.
        
    custom_likelihood: function 
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
        
    noise: np.array(float) or None
        If loaded, an array containing the background spectrum, including only 
        the channels noticed in the fit.
        
    noise_err: np.array(float or None) 
        If loaded, an array containing the sqrt of the background counts, only 
        in the channels noticed during the fit. Used to compute the fit statistic.

    _data_unmasked, _data_err_unmasked, _noise_unmasked: np.array(float)
        The arrays of every data bin, its error and (if loaded) the backgruond, 
        regardless of which ones are ignored or noticed during the fit.
        Used exclusively to enable book keeping internal to the fitter class.            
    """ 

    def __init__(self,likelihood="chisq"):
        self.model = None
        self.model_params = None
        self.likelihood = likelihood
        self.custom_likelihood = None
        self.custom_args = None
        self.fit_result = None
        self.data = None
        self.data_err = None
        self.noise = None 
        self.noise_err = None 
    pass

    def _set_unmasked_data(self):
        """
        This initializer method is used to set up the unmasked arrays for later
        book-keeping. Depending on the dependence of the fit, it initializes
        different internal unmasked arrays.
        """

        self._data_unmasked = self.data
        self._data_err_unmasked = self.data_err
        #if our fit object has a background (e.g. a spectral background, or
        #Poisson noise) we also must store it to subtract it correctly from the
        #data
        if self.noise is not None:
            self._noise_unmasked = self.noise
            self._noise_err_unmasked = self.noise_err
        else:
            self._noise_unmasked = None
            self._noise_err_unmasked = None

        self._set_axis_arrays()
        return

    def _set_axis_arrays(self):
        """
        This method sets up the unmasked bookkeeping arrays (per-axis grids and
        bin counts) for whichever combination of axis-dependent mixins this
        fitter combines - energy, Fourier frequency, and/or polarization. 
        """

        has_energy = isinstance(self,EnergyDependentFit)
        has_freq = isinstance(self,FrequencyDependentFit)
        has_pol = isinstance(self,StokesDependentFit)

        if has_energy is True:
            self._emin_unmasked = self.response.emin
            self._emax_unmasked = self.response.emax
            self._ebounds_unmasked = self.ebounds
            self._ewidths_unmasked = self.ewidths
            self._all_chans = self._ebounds_unmasked.size
            self.n_chans = self._all_chans

        if has_pol is True:
            self._pol_emin_unmasked = self.response_pol.emin
            self._pol_emax_unmasked = self.response_pol.emax
            self._pol_ebounds_unmasked = self.pol_ebounds
            self._pol_ewidths_unmasked = self.pol_ewidths
            self._all_pol_chans = self._pol_ebounds_unmasked.size
            self.n_pol_chans = self._all_pol_chans

        #the total number of flattened data bins depends on which combination of
        #axes is present; every supported combination is listed explicitly so
        #that an unsupported one fails loudly instead of silently
        if (has_energy is True and has_freq is True):
            self._all_bins = self._all_freqs*self._all_chans
        elif (has_energy is True and has_pol is True):
            self._all_bins = self._all_chans+2*self._all_pol_chans
        elif (has_freq is True and has_pol is True):
            #e.g. a modulation-angle x Fourier-frequency fit with no energy axis
            raise NotImplementedError(
                "Frequency+polarization fits are not yet supported")
        elif has_energy is True:
            self._all_bins = self._all_chans
        elif has_freq is True:
            self._all_bins = self._all_freqs
        else:
            self._all_bins = None
        self.n_bins = self._all_bins
        return

    def _reflatten_data(self):
        """
        This method re-derives self.data, self.data_err, self.noise and
        self.noise_err from the unmasked arrays, after an ignore_*/notice_*
        method updates one of this fit's masks. 
        
        New combinations of dimensions/data (e.g. ModulationDependentFit for 
        stochastic polarimetry timing) should be added as an extra branch here.
        """

        has_energy = isinstance(self,EnergyDependentFit)
        has_freq = isinstance(self,FrequencyDependentFit)
        has_pol = isinstance(self,StokesDependentFit)

        #filter data for spectral timing
        if (has_energy is True and has_freq is True):
            self.data = self._filter_2d_by_mask(self._data_unmasked)
            self.data_err = self._filter_2d_by_mask(self._data_err_unmasked)
        #filter data for spectro-polarimetry
        elif has_pol is True:
            self.data = self._filter_stokes_by_mask(self._data_unmasked)
            self.data_err = self._filter_stokes_by_mask(self._data_err_unmasked)
            if self.noise is not None:
                self.noise = self._filter_stokes_by_mask(self._noise_unmasked)
                self.noise_err = self._filter_stokes_by_mask(self._noise_err_unmasked)
        #filter data for time averaged spectra
        elif has_energy is True:
            self.data = np.extract(self.ebounds_mask,self._data_unmasked)
            self.data_err = np.extract(self.ebounds_mask,self._data_err_unmasked)
            if self.noise is not None:
                self.noise = np.extract(self.ebounds_mask,self._noise_unmasked)
                self.noise_err = np.extract(self.ebounds_mask,self._noise_err_unmasked)
        #filter data for PSDs 
        elif has_freq is True:
            self.data = np.extract(self.freqs_mask,self._data_unmasked)
            self.data_err = np.extract(self.freqs_mask,self._data_err_unmasked)
            if self.noise is not None:
                self.noise = np.extract(self.freqs_mask,self._noise_unmasked)
        else:
            raise NotImplementedError("No known way to flatten this fit's data")
        return

    def _filter_2d_by_mask(self,array):
        """
        This method is used to filter two-dimensional data (for example, a cross
        spectrum) after users define a range of energy channels, or Fourier 
        frequency, or other data bins, bins to ignore. 
        
        Parametrers:
        ------------
        array: np.float 
            The one-dimensional array containing the (flattened) two-dimensional 
            data or model to be filtered 
            
        Output:
        -------
        filter_arr: np.float 
            The one-dimensional array filtered by the two-d mask defind by the 
            noticed frequency bins and channels.
        """

        if self.dependence == "generic":
            self.n_bins = self.rows*self.columns
        else:    
            self.n_bins = self.n_chans*self.n_freqs
        
        if self.dependence == "energy":
            twod_mask = self.freqs_mask.reshape((self._all_freqs,1))* \
                        self.ebounds_mask.reshape((1,self._all_chans))
        elif self.dependence == "frequency":
            twod_mask = self.ebounds_mask.reshape((self._all_chans,1))* \
                        self.freqs_mask.reshape((1,self._all_freqs))  
        elif self.dependence == "generic":
            twod_mask = self.row_mask.reshape((self._all_rows,1))* \
                        self.column_mask.reshape((1,self._all_columns))
        else:
            raise AttributeError("Data dependence not specified")
        twod_mask = np.array(twod_mask).flatten()

        #handle the case of complex data being loaded first, in which case we 
        #need to mask both real and imaginary parts
        if self.units != "lags" and self.units != "real":
            filter_first_dim = np.extract(twod_mask,array[:self._all_bins])
            filter_second_dim = np.extract(twod_mask,array[self._all_bins:],)
            filter_arr = np.append(filter_first_dim,filter_second_dim)          
        #otherwise if we only have real data (or lags in a cross spectrum) the 
        #mask can be applied just once 
        else:
            filter_arr = np.extract(twod_mask,array)
        return filter_arr
    
    def set_model(self,model,params=None):
        """
        This method is used to pass the model users want to fit to the data. 
        Optionally it is also possible to pass the initial parameter values of 
        the model. If a user had previously stored a model in the fitter which 
        included a calibration component, that component is removed or reset 
        (depending on the exact fitter) when passing the new model from calling 
        this method.
        
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
        self._reset_calibration()
        self.model = model 
        if params is None:
            self.model_params = self.model.make_params(verbose=True)
        else:
            self.model_params = params
        return 

    def set_params(self,params):
        """
        This method is used to set the model parameter names and values. It can
        be used both to initialize a fit, and to test different parameter values 
        before actually running the minimization algorithm.
        
        Parameters:
        -----------                       
        params: lmfit.parameter
            The parameter values from which to start evalauting the model during
            the fit.  
        """
        
        #maybe find a way to go through the parameters of the model, and make sure 
        #the object passed contains the same parameters?
        if getattr(params, '__module__', None) != "lmfit.parameter":  
            raise AttributeError("The parameters input must be an LMFit Parameters object")
        
        self.model_params = params
        return 

    def set_custom_likelihood(self,likelihood_function,*args):
        """
        This method allows users to define their own custom likelihood function,
        which can then optimized during a fit. In addition, this sets the
        value of the class "likelihood" string to custom, to signal to the other
        methods that a custom likelihood is in use and should be used for plots,
        residuals etc.
        
        Parameters:
        -----------
        likelihood_function: function
            The name of the function which calculates the model residuals; e.g.,
            if we want to minimize the difference between data and model, we 
            would define:
            
            def diff(data,model):
                return data-model 
                
            and call set_custom_likelihood(diff).  
            
        \*args:  
            Additional arguments to be passed to the likelihood calculation, 
            excluding the data and model (which are always included 
            automatically by the class). Following the example above:
            
            def diff(data,model,factor):
                return factor*(data-model)
            
            and call set_custom_likelihood(diff,5) - if we want to set "factor"
            to 5.
        """

        self.custom_likelihood = likelihood_function
        self.custom_args = args
        self.likelihood = "custom"
        return
        
    def get_residuals(self,res_type,model=None,mask=True):    
        """
        This methods return the residuals (either as data/model, or as 
        contribution to the total chi squared) of the input model, given the 
        parameters set in model_parameters, with respect to the data. 
        
        Parameters:
        -----------
        res_type: string 
            If set to "ratio", the method returns the residuals defined as 
            data/model. If set to "chisq", it returns the contribution of 
            each energy channel to the total chi squared. If set to "custom", 
            the residuals are based on whatever custom likelihood the user 
            defined.

        mask: bool, default True 
            A flag to decide whether to compare the model against the masked or 
            unmasked data.
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each channel.
            
        bars: np.array(float)
            An array of the same size as the residuals, containing the one sigma 
            range for each contribution to the residuals.           
        """

        if self.noise is None:
            noise = np.zeros(len(self.data))
            noise_err = np.zeros(len(self.data))
        else:
            noise = self.noise
            noise_err = self.noise_err
        
        if model is None:
            model = self.eval_model(mask=mask)
        
        #separate the case of Cash vs non cash statistic because in the former 
        #case subtracting/accounting for the background is not straightforward
        #and is taken care of within the cstat function call        
        if (mask is True and res_type != "cstat"):
            data = self.data - noise 
            error = np.sqrt(self.data_err**2+noise_err**2)
        elif (mask is False and res_type != "cstat"):
            #ugly hack to handle 2d cross spectrum plots - ugh
            if self._noise_unmasked is None:
                noise = np.zeros(len(self._data_unmasked))
                noise_err = np.zeros(len(self._data_unmasked))
            else:
                noise = self._noise_unmasked
                noise_err = self._noise_err_unmasked
            data = self._data_unmasked - noise
            error = np.sqrt(self._data_err_unmasked**2+noise_err**2)
        elif mask is True:
            data = self.data 
        elif mask is False:
            data = self._data_unmasked

        if res_type == "ratio":
            residuals, bars = ratio(data,error,model,summed=False)
        elif res_type == "chisq":
            residuals = chisq(data,error,model,summed=False)
            bars = np.ones(len(self.data))
        elif res_type == "cstat":
            exp = self.exposure 
            bins = self.ewidths
            residuals = cstat(data,model,exp,bins,noise=noise,summed=False)
            bars = np.ones(len(self.data))
        elif res_type == "custom":
            custom_args = [model,data]
            if self.custom_args is not None:
                for arg in self.custom_args:
                    custom_args.append(arg)
            residuals = self.custom_likelihood(*custom_args)
            bars = np.ones(len(self.data))
        else:
            raise ValueError("The supported residual types are ratio, chisq, and cstat")    
        return residuals, bars

    def print_fit_stat(self):
        """
        This method compares the model defined by the user, using the last set
        of parameters to have been set in the class, to the data stored. It then
        prints the chi-squared goodness-of-fit to terminal, along with the 
        number of data bins, free parameters and degrees of freedom. 
        """
        
        if self.likelihood == "chisq":
            res, _ = self.get_residuals("chisq")
        elif self.likelihood == "cstat":
            res, _ = self.get_residuals("cstat")
        elif self.likelihood == "custom":
            res, _ = self.get_residuals("custom")
       
        freepars = 0
        for key, value in self.model_params.items():
            param = self.model_params[key]
            if param.vary is True:
                freepars += 1
        dof = len(self.data) - freepars

        if self.likelihood == "chisq":
            fit_statistic = np.sum(res**2)
        elif self.likelihood == "cstat":
            fit_statistic = np.sum(res)  
        elif self.likelihood == "custom":
            fit_statistic = np.sum(res) 
        reduced_stat = fit_statistic/dof

        print("Goodness of fit metrics:")
        print(f"Fit statistic: {self.likelihood}")
        print("Fit statistic" + "{0: <11}".format(" ") + str(fit_statistic))
        print("Reduced fit stat" + "{0: <8}".format(" ") + str(reduced_stat))
        print("Data bins:" + "{0: <14}".format(" ") + str(len(self.data)))
        print("Free parameters:" + "{0: <8}".format(" ") + str(freepars))
        print("Degrees of freedom:" + "{0: <5}".format(" ") + str(dof))         

        return 

    def fit_data(self,algorithm='leastsq'):
        """
        This method attempts to minimize the residuals of the model with respect 
        to the data defined by the user. The fit always starts from the set of 
        parameters defined with .set_params(). Once the algorithm has completed 
        its run, it prints to terminal the best-fitting parameters, fit 
        statistics, and simple selection criteria (reduced chi-squared, Akaike
        information criterion, and Bayesian information criterion). 
        
        Parameters:
        -----------
        algorithm: str, default="leastsq"
            The fitting algorithm to be used in the minimization. The possible 
            choices are detailed on the LMFit documentation page:
            https://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table.
        """
        if np.all(self.data) == None:
            raise ValueError("No data to fit. Please set the data using the .set_data() method.")
        elif np.all(self.data_err) == None:
            raise ValueError("No data error to fit. Please set the data error using the .set_data() method.")

        if self.model == None:
            raise ValueError("No model to fit. Please set the model using the .set_model() method.")
        elif self.model_params == None:
            raise ValueError("No model parameters to fit. Please set the model parameters using the .set_params() method.")

        if algorithm == 'emcee':
            raise ValueError("EMCEE IS NOT AN OPTIMIZER AND SHOULD NOT BE USED SUCH! PICK A DIFFERENT METHOD")
        
        self.fit_result = minimize(self._minimizer,self.model_params,
                                   method=algorithm)
        fit_params = self.fit_result.params
        self.set_params(fit_params)
        
        self.print_fit_report()
        return
    
    def print_model(self):
        """
        This method prints out model components, model parameters, and their
        settings.
        """
        print("-----------------------")
        print(self.model.name)
        print("-----------------------")
        print("Parameters")
        print("-----------------------")
        self.model_params.pretty_print()
        return
        
    def print_fit_report(self):
        """
        This method prints the current fit result.
        """
        
        result = self.fit_result
        print("-----------------------")
        print("[[Fit Statistics]]")
        print(f"    # fitting method   = {result.method}")
        print(f"    # function evals   = {result.nfev}")
        print(f"    # data points      = {result.ndata}")
        print(f"    # variables        = {result.nvarys}")
        if self.likelihood == "chisq":        
            fit_statistic = result.chisqr 
            reduced_stat = result.redchi
        #for the Cash statistic, the chisqr/redchi stored in the result object 
        #is nonesense (since it assumes gaussian statistics etc)
        else:
            res, _ = self.get_residuals(self.likelihood) 
            fit_statistic = np.sum(res)  
            reduced_stat = fit_statistic/(result.ndata-result.nvarys)    
        print(f"    fit statistic      = {fit_statistic}")
        print(f"    reduced statistic  = {reduced_stat}")
        #no idea how to get these two for Cash statistic
        if self.likelihood == "chisq":
            print(f"    Akaike info crit   = {result.aic}")
            print(f"    Bayesian info crit = {result.bic}")
        
        namelen = max(len(n) for n in list(result.params.keys()))
        parnames_varying = [par for par in result.params if result.params[par].vary]
        #report parameteres that didn't vary/are stuck
        for name in parnames_varying:
            par = result.params[name]
            space = ' '*(namelen-len(name))
            if par.init_value and np.allclose(par.value, par.init_value):
                print(f'    {name}:{space}  at initial value')
            if (np.allclose(par.value, par.min) or np.allclose(par.value, par.max)):
                print(f'    {name}:{space}  at boundary')
        
        #report parameter values
        print("[[Parameters]]")
        modelpars = result.params
        for name in result.params.keys():
            par = result.params[name]
            space = ' '*(namelen-len(name))
            nout = f"{name}:{space}"
            inval = '(init = ?)'
            if par.init_value is not None:
                inval = f'(init = {par.init_value:.7g})'
            if modelpars is not None and name in modelpars:
                inval = f'{inval}, model_value = {modelpars[name].value:.7g}'
            try:
                sval = gformat(par.value)
            except (TypeError, ValueError):
                sval = ' Non numeric value found in parameter'
            if par.stderr is not None:
                serr = gformat(par.stderr)
                try:
                    spercent = f'({abs(par.stderr/par.value):.2%})'
                except ZeroDivisionError:
                    spercent = ''
                sval = f'{sval} +/-{serr} {spercent}'
            if par.vary:
                print(f"    {nout} {sval} {inval}")
            elif par.expr is not None:
                print(f"    {nout} {sval} == '{par.expr}'")
            else:
                print(f"    {nout} {par.value: .7g} (fixed)")
        return

    def _reset_calibration(self):
        """
        This method clears any gain instrument calibration correction which was 
        set on top of the model, and is called by set_model. The only correction 
        currently supported this way is gain calibration, so this method only 
        does anything if the fit is of an energy-dependent quantity.
        """
        
        if isinstance(self,EnergyDependentFit) and self.gain_params is not None:
            warnings.warn(("WARNING: changing the model has reset the gain"
                           " correction, call set_gain again to re-apply it"),
                          UserWarning)
            self.gain_params = None
            self._gain_keys = None
        
        return 

class EnergyDependentFit():
    """
    Internal book-keeping class used to manage noticing or ignoring energy 
    channels, for cases when the data requires an instrument response. 
    
    Stores the full (unmasked) energy center/bounds, and data arrays, a mask
    used to track which channels/data points are noticed or ignored, as well as 
    the masked arrays containing only the noticed bins. 

    Attributes:
    -----------    
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
        channels,  as stored in the instrument response provided. Only contains 
        the channels that are noticed during the fit.
        
    ebounds_mask: np.array(bool)
        The array of instrument energy channels that are either ignored or 
        noticed during the fit. A given channel i is noticed if ebounds_mask[i]
        is True, and ignored if it is false. 
        
    n_chans: int 
        The number of channels that are to be noticed during the fit.
        
    _all_chans: int 
        The total number of channels in the loaded response matrix.
        
    n_bins: int 
        Only used for two-dimensional data fitting. Defined as the number of 
        noticed channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).
        
    _all_bins: int 
        Only used for two-dimensional data fitting. Defined as the total number 
        of  channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).
                
    _emin_unmasked, _emax_unmasked, _ebounds_unmasked, _ewidths_unmasked: np.array(float)
        The array of every lower bound, upper bound, channel center and channel 
        widths stored in the response, regardless of which ones are ignored or 
        noticed during the fit. Used exclusively to facilitate book-keeping 
        internal to the fitter class.  
        
    gain_params: lmfit.Parameters, default None 
        A lmfit Parameters object, which contains the parameters for the gain  
        correction model components if it is enabled. Defaults to None.
        
    _gain_keys: list, default None 
        A list of keys we use to keep track of the names of the gain parameters. 
        Necessary during joint fits with multiple instruments/detectors, if more 
        than one detector requires gain fitting (e.g., fitting 3 IXPE DUs, each 
        with its set of gain parameters).      
    """
    def __init__(self):   
        self.energs = 0.5*(self.response.energ_hi+self.response.energ_lo)
        self.energ_bounds = self.response.energ_hi-self.response.energ_lo
        self.ear = np.append(self.response.energ_lo,self.response.energ_hi[-1])        
        self.ebounds = 0.5*(self.response.emax+self.response.emin)
        self.ewidths = self.response.emax - self.response.emin
        self.ebounds_mask = np.full((self.response.n_chans), True)
        self.gain_params = None
        self._gain_keys = None
        pass
       
    def ignore_energies(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        ignore selected channels based on their energy bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored energy interval.
        bound_hi : float
            Higher bound of ignored energy interval.    
        """
        
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Energy bounds must be floats or integers")
        
        self.ebounds_mask = ((self._emin_unmasked<bound_lo)|
                             (self._emax_unmasked>bound_hi))&self.ebounds_mask
       
        #take the unmasked arrays and keep only the bounds we want
        self.emin = np.extract(self.ebounds_mask,self._emin_unmasked)
        self.emax = np.extract(self.ebounds_mask,self._emax_unmasked)
        self.ebounds = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        self.ewidths = np.extract(self.ebounds_mask,self._ewidths_unmasked)   
        self.n_chans = self.ebounds_mask[self.ebounds_mask==True].size

        #filter and re-flatten the data using the same mask 
        self._reflatten_data()      
        return
   
    def notice_energies(self,bound_lo,bound_hi):
        """
        This method adjusts the data arrays stored such that they (and the fit) 
        notice selected (previously ignore) channels  based on their energy 
        bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored energy interval.
        bound_hi : float,
            Higher bound of ignored energy interval.     
        """
        
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Energy bounds must be floats or integers")        
              
        #if bounds of channel lie in noticed energies, notice channel
        self.ebounds_mask = self.ebounds_mask|np.logical_not(
                            (self._emin_unmasked<bound_lo)|
                            (self._emax_unmasked>bound_hi))

        #take the unmasked arrays and keep only the bounds we want
        self.emin = np.extract(self.ebounds_mask,self._emin_unmasked)
        self.emax = np.extract(self.ebounds_mask,self._emax_unmasked)
        self.ebounds = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        self.ewidths = np.extract(self.ebounds_mask,self._ewidths_unmasked)   
        self.n_chans = self.ebounds_mask[self.ebounds_mask==True].size        

        #filter and re-flatten the data using the same mask 
        self._reflatten_data() 
        return

    def set_gain(self,slope=1.0,offset=0.0,vary=True,
                 slope_bounds=(0.9,1.1),offset_bounds=(-0.3,0.3),
                 label=None):
        """
        This method sets a gain correction to the instrument channel to energy
        conversion, which is applied to the model after it has been folded 
        through the response.
        nDspec follows the same convention as Xspec, in which the true photon
        energy E' collected by a channel with nominal bound E is defined as

        E' = E/slope - offset,

        with the offset in units of keV.

        The gain parameters are appended to the model_params attribute, so this
        method has to be called after the model has been set. Since they are 
        ordinary lmfit parameters, users should set a prior on them through the 
        usual interfaces in SamplingUtils when using Bayesian sampling.

        Parameters:
        -----------
        slope: float, default=1.0
            The starting value of the multiplicative term of the gain shift.

        offset: float, default=0.0
            The starting value of the additive term of the gain shift, in units
            of keV.

        vary: bool, default=True
            A boolean switch to choose whether the gain parameters are free
            during the fit, or frozen at their starting values.

        slope_bounds: tuple, default=(0.9,1.1)
            The minimum and maximum values the slope is allowed to take.

        offset_bounds: tuple, default=(-0.3,0.3)
            The minimum and maximum values the offset is allowed to take, in
            units of keV.
            
        label: string, default=None
            A string appended to the names of the gain parameters, in order to
            keep the energy scales of different detectors independent from one
            another in a joint fit.    
        """

        if self.model_params is None:
            raise AttributeError(("The model has to be set before setting a"
                                  " gain correction"))

        self._check_gain_bounds(slope_bounds,offset_bounds)
        
        if label is None:
            self._gain_keys = ("gain_slope","gain_offset")
        else:
            self._gain_keys = ("gain_slope_"+str(label),
                               "gain_offset_"+str(label))

        self.gain_params = LM_Parameters()
        self.gain_params.add(self._gain_keys[0],value=slope,vary=vary,
                             min=slope_bounds[0],max=slope_bounds[1])
        self.gain_params.add(self._gain_keys[1],value=offset,vary=vary,
                             min=offset_bounds[0],max=offset_bounds[1])
        self.model_params.update(self.gain_params)
        return

    def _check_gain_bounds(self,slope_bounds,offset_bounds):
        """
        This method checks the bounds on the gain parameters against every
        channel grid tracked by the fitter. Classes which track more than one
        response are expected to override it, calling _check_gain_grid once for
        each grid.
 
        Parameters:
        -----------
        slope_bounds: tuple
            The minimum and maximum values the slope is allowed to take.
 
        offset_bounds: tuple
            The minimum and maximum values the offset is allowed to take, in
            units of keV.
        """
 
        self._check_gain_grid(self.ebounds_mask,self._emin_unmasked,
                              self._emax_unmasked,slope_bounds,offset_bounds)
        return 
        
    def _check_gain_grid(self,mask,emin_unmasked,emax_unmasked,slope_bounds,
                         offset_bounds,label="the response"):
        """
        This method checks that every channel noticed during the fit remains
        fully covered by a given channel grid, for every gain shift allowed by
        the bounds on the gain parameters. 
 
        If the check fails, users should either tighten the bounds on the gain
        parameters, or ignore additional channels at the edges of the grid.
 
        Parameters:
        -----------
        mask: np.array(bool)
            The mask of the channels noticed during the fit, over the grid to be
            checked.
 
        emin_unmasked, emax_unmasked: np.array(float)
            The lower and upper bounds of every channel in the grid to be
            checked, regardless of which ones are noticed or ignored.
 
        slope_bounds: tuple
            The minimum and maximum values the slope is allowed to take.
 
        offset_bounds: tuple
            The minimum and maximum values the offset is allowed to take, in
            units of keV.
 
        label: string, default="the response"
            A string identifying the grid being checked, used to tell users
            which response caused the check to fail in fits which track more
            than one.
        """

        noticed_min = np.min(np.extract(mask,emin_unmasked))
        noticed_max = np.max(np.extract(mask,emax_unmasked))
        shift_min = noticed_min/slope_bounds[1] - offset_bounds[1]
        shift_max = noticed_max/slope_bounds[0] - offset_bounds[0]
 
        if shift_min < emin_unmasked[0]:
            raise ValueError(("The lower bound of the noticed channels can be"
                              " shifted below the channel grid stored in "
                              +label+"; either tighten the bounds on the gain"
                              " parameters, or ignore more channels at low"
                              " energy"))
        if shift_max > emax_unmasked[-1]:
            raise ValueError(("The upper bound of the noticed channels can be"
                              " shifted above the channel grid stored in "
                              +label+"; either tighten the bounds on the gain"
                              " parameters, or ignore more channels at high"
                              " energy"))
        return
 
    def _check_gain_bounds(self,slope_bounds,offset_bounds):
        """
        This method checks the bounds on the gain parameters against every
        channel grid tracked by the fitter. Classes which track more than one
        response are expected to override it, calling _check_gain_grid once for
        each grid.
 
        Parameters:
        -----------
        slope_bounds: tuple
            The minimum and maximum values the slope is allowed to take.
 
        offset_bounds: tuple
            The minimum and maximum values the offset is allowed to take, in
            units of keV.
        """
 
        self._check_gain_grid(self.ebounds_mask,self._emin_unmasked,
                              self._emax_unmasked,slope_bounds,offset_bounds)
        return
    
    def _apply_gain(self,model,params,response=None):
        """
        This method applies the gain correction set by the user to a model which
        has already been folded through the instrument response, and returns it
        unchanged if no gain correction was set. The model must be defined over 
        every channel in the response, rather than over just the noticed ones, 
        because the channels which are ignored during the fit might supply some 
        counts after shifting.

        Parameters:
        -----------
        model: np.array(float)
            The folded model, of size (_all_chans).

        params: lmfit.Parameters
            The parameter values to use in applying the gain. If none are
            provided, the gain_params attribute is used.

        response: nDspec.ResponseMatrix, default=None
            The response through which the model was folded, and whose channel
            grid the gain is therefore applied over. If none is provided, the
            response attribute is used. 

        Returns:
        --------
        model: np.array(float)
            The folded model, shifted by the gain correction.
        """

        if self.gain_params is None:
            return model
        if response is None:
            response = self.response
        model = response.apply_gain(model,params[self._gain_keys[0]].value,
                                    params[self._gain_keys[1]].value)

        return model

class FrequencyDependentFit():
    """
    Internal book-keeping class used to manage noticing or ignoring Fourier  
    frequency bins. 
    
    Stores the full (unmasked) Fourier bins, and data arrays, a mask
    used to track which bins/data points are noticed or ignored.

    Attributes:
    -----------    
    _freqs_unmasked: np.array(float)
        If the data and model explicitely depend on Fourier frequency (e.g. a
        power spectrum), this is the array of Fourier frequency over which all 
        data and model are defined, including bins that are ignored in the fit. 
        
        If instead the data depends from some other energy (e.g. energy), it 
        contains both noticed and ignored frequency intervals over which to 
        produce spectral-timing products. For example, a user might input a set 
        of 7 ranges of frequencies to calculate lag energy spectra, but only 
        want to consider the first and last 3, and ignore the middle one.
    
    freqs_mask np.array(bool)
        The array of Fourier frequencies that are either ignored or noticed 
        during the fit. A given channel i is noticed if freqs_mask[i] is True,
        and ignored if it is false.      
    
    n_freqs: int 
        The number of Fourier frequency bins that are noticed in the fit. 
    
    n_bins: int 
        Only used for two-dimensional data fitting. Defined as the number of 
        noticed channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).
        
    _all_bins: int 
        Only used for two-dimensional data fitting. Defined as the total number 
        of  channels, times the number of bins in the second dimension 
        (e.g. Fourier frequency).   
    """

    def __init__(self,freqs):
        self._freqs_unmasked = freqs
        self.freqs = self._freqs_unmasked
        if self.dependence == "frequency":
            self._all_freqs = self._freqs_unmasked.size
        else:
            self._all_freqs = self._freqs_unmasked.size-1
        self.n_freqs = self._all_freqs
        self.freqs_mask = np.full((self._all_freqs), True)
        pass

    def ignore_frequencies(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        ignore selected frequencies based on user-supplied bounds bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored frequency interval.
        bound_hi : float
            Higher bound of ignored frequency interval.    
        """
        
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Frequency bounds must be floats or integers")
       
        if self.dependence == "frequency":
            #this is called for a regular frequency-dependent product
            self.freqs_mask = ((self._freqs_unmasked<bound_lo)|
                               (self._freqs_unmasked>bound_hi))&self.freqs_mask
            self.freqs = np.extract(self.freqs_mask,self._freqs_unmasked)
            self.n_freqs = self.freqs_mask[self.freqs_mask==True].size
        else:
            #this is for products that do not depend on energy explicitely, but 
            #only implicitely - for example, lag-energy data.
            fmin = self._freqs_unmasked[:-1]
            fmax = self._freqs_unmasked[1:]
            self.freqs_mask = ((fmin<bound_lo)|
                               (fmax>bound_hi))&self.freqs_mask
            self.freq_bounds = np.extract(self.freqs_mask,self._freqs_unmasked)
            self.n_freqs = self.freqs_mask[self.freqs_mask==True].size#-1 

        #filter and re-flatten the data using the same mask 
        self._reflatten_data()   
        return

    def notice_frequencies(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        ignore selected frequencies based on user-supplied bounds bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored frequency interval.
        bound_hi : float
            Higher bound of ignored frequency interval.    
        """
        
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Frequency bounds must be floats or integers")

        if self.dependence == "frequency":
            self.freqs_mask = self.freqs_mask|np.logical_not(
                              (self._freqs_unmasked<bound_lo)|
                              (self._freqs_unmasked>bound_hi))
            self.freqs = np.extract(self.freqs_mask,self._freqs_unmasked)
            self.n_freqs = self.freqs_mask[self.freqs_mask==True].size        
        else:
            fmin = self._freqs_unmasked[:-1]
            fmax = self._freqs_unmasked[1:]
            self.freqs_mask = self.freqs_mask|np.logical_not(
                              (fmin<bound_lo)|
                              (fmax>bound_hi))
            self.freq_bounds = np.extract(self.freqs_mask,self._freqs_unmasked)
            self.n_freqs = self.freqs_mask[self.freqs_mask==True].size-1 

        #filter and re-flatten the data using the same mask 
        self._reflatten_data()  
        return

class StokesDependentFit():
    """
    Internal book-keeping class used during spectro-polarimetry modelling to 
    manage noticing or ignoring the energy channels of the Stokes Q and U 
    spectra, for cases when the data consists of all three Stokes parameters. 
    It is currently only used together with the EnergyDependentFit class, which 
    handles the Stokes I channel grid.
    
    The two grids are tracked separately because Stokes Q and U are almost 
    always binned more coarsely than Stokes I: they are the difference between 
    two large numbers, and therefore require far more counts per channel to be 
    measured with the same significance. Stokes Q and U are instead always 
    tracked together, because they are measured from the same events and are 
    therefore always defined over the same channel grid.
    
    The data of all three Stokes parameters is stored in a single, flattened 
    array in the order I, Q, U, identically to how two-dimensional 
    spectral-timing products are handled elsewhere in the library. 
        
    Attributes:
    -----------
    pol_emin, pol_emax: np.array(float)
        The arrays of lower and upper energy channel bounds for the Stokes Q and 
        U instrument energy channels, as stored in the instrument response 
        provided. Only contain the channels that are noticed during the fit.
    
    pol_ebounds: np.array(float) 
        The array of energy channel bin centers for the Stokes Q and U 
        instrument energy channels, as stored in the instrument response 
        provided. Only contains the channels that are noticed during the fit.

    pol_ewidths: np.array(float) 
        The array of energy channel bin widths for the Stokes Q and U instrument 
        energy channels, as stored in the instrument response provided. Only 
        contains the channels that are noticed during the fit.
        
    pol_ebounds_mask: np.array(bool)
        The array of Stokes Q and U instrument energy channels that are either 
        ignored or noticed during the fit. A given channel i is noticed if 
        pol_ebounds_mask[i] is True, and ignored if it is false.
        
    n_pol_chans: int 
        The number of Stokes Q (and U) channels that are to be noticed during 
        the fit.
        
    n_bins: int 
        The total number of data bins noticed during the fit, defined as the 
        number of noticed Stokes I channels, plus twice the number of noticed 
        Stokes Q/U channels.
        
    _all_pol_chans: int 
        The total number of channels in the loaded Stokes Q/U response matrix.
        
    _all_bins: int 
        The total number of data bins loaded, defined as the total number of 
        Stokes I channels, plus twice the total number of Stokes Q/U channels.
        
    _pol_emin_unmasked, _pol_emax_unmasked, _pol_ebounds_unmasked, _pol_ewidths_unmasked: np.array(float)
        The array of every lower bound, upper bound, channel center and channel 
        widths stored in the Stokes Q/U response, regardless of which ones are 
        ignored or noticed during the fit. Used exclusively to facilitate 
        book-keeping internal to the fitter class.    
    """

    def __init__(self):
        self.pol_emin = self.response_pol.emin
        self.pol_emax = self.response_pol.emax
        self.pol_ebounds = 0.5*(self.response_pol.emax+self.response_pol.emin)
        self.pol_ewidths = self.response_pol.emax-self.response_pol.emin
        self.pol_ebounds_mask = np.full((self.response_pol.n_chans), True)
        pass

    def _stokes_slice(self,index,mask=True):
        """
        This method returns the slice of the flattened data (or model) array 
        that corresponds to a given Stokes parameter.
        
        Parameters:
        -----------
        index: int 
            The index of the Stokes parameter to be returned; 0 for Stokes I, 
            1 for Stokes Q, and 2 for Stokes U.
            
        mask: bool, default True 
            A flag to decide whether the slice refers to an array containing 
            only the noticed channels, or to one containing every channel.
            
        Output:
        -------
        stokes_slice: slice 
            The slice of the flattened array corresponding to the chosen Stokes 
            parameter.
        """
        
        if mask is True:
            n_chans = self.n_chans
            n_pol_chans = self.n_pol_chans
        else:
            n_chans = self._all_chans
            n_pol_chans = self._all_pol_chans
        
        if index == 0:
            stokes_slice = slice(0,n_chans)
        elif index == 1:
            stokes_slice = slice(n_chans,n_chans+n_pol_chans)
        elif index == 2:
            stokes_slice = slice(n_chans+n_pol_chans,n_chans+2*n_pol_chans)
        else:
            raise ValueError("Stokes index must be 0 (I), 1 (Q) or 2 (U)")
        return stokes_slice

    def split_stokes(self,array,mask=True):
        """
        This method splits a flattened array containing all three Stokes 
        parameters - for instance the data, or the output of eval_model - into 
        three separate arrays, each containing one of the Stokes parameters.
        
        Parameters:
        -----------
        array: np.array(float)
            The flattened array containing the Stokes I, Q and U values, in this 
            order.
            
        mask: bool, default True 
            A flag to decide whether the input array contains only the noticed 
            channels, or every channel.
            
        Output:
        -------
        stokes_I, stokes_Q, stokes_U: np.array(float)
            The three arrays containing the values of each Stokes parameter.
        """
        
        stokes_I = array[self._stokes_slice(0,mask=mask)]
        stokes_Q = array[self._stokes_slice(1,mask=mask)]
        stokes_U = array[self._stokes_slice(2,mask=mask)]
        return stokes_I, stokes_Q, stokes_U

    def _filter_stokes_by_mask(self,array):
        """
        This method is used to filter the flattened spectro-polarimetric data 
        after users define a range of energy channels to ignore. It is necessary 
        because the Stokes I channel grid is typically different from that of 
        Stokes Q and U, and therefore the two require separate masks.
        
        Parameters:
        -----------
        array: np.array(float)
            The flattened array containing the Stokes I, Q and U values in every 
            channel, to be filtered.
            
        Output:
        -------
        filter_arr: np.array(float)
            The flattened array filtered by the masks defined by the noticed 
            Stokes I and Stokes Q/U channels.
        """
        
        self.n_bins = self.n_chans+2*self.n_pol_chans
        
        filter_stokes_I = np.extract(self.ebounds_mask,
                                     array[self._stokes_slice(0,mask=False)])
        filter_stokes_Q = np.extract(self.pol_ebounds_mask,
                                     array[self._stokes_slice(1,mask=False)])
        filter_stokes_U = np.extract(self.pol_ebounds_mask,
                                     array[self._stokes_slice(2,mask=False)])
        filter_arr = np.concatenate((filter_stokes_I,filter_stokes_Q,
                                     filter_stokes_U))
        return filter_arr

    def _update_polarization_grids(self):
        """
        This method takes the unmasked Stokes Q/U channel arrays and keeps only 
        the channels that are noticed in the fit, after the Stokes Q/U mask has 
        been updated.
        """
        
        self.pol_emin = np.extract(self.pol_ebounds_mask,
                                   self._pol_emin_unmasked)
        self.pol_emax = np.extract(self.pol_ebounds_mask,
                                   self._pol_emax_unmasked)
        self.pol_ebounds = np.extract(self.pol_ebounds_mask,
                                      self._pol_ebounds_unmasked)
        self.pol_ewidths = np.extract(self.pol_ebounds_mask,
                                      self._pol_ewidths_unmasked)
        self.n_pol_chans = self.pol_ebounds_mask[self.pol_ebounds_mask==True].size
        return

    def ignore_polarization_energies(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        ignore selected Stokes Q and U channels based on their energy bounds. 
        The Stokes I channels are left untouched; users should call the 
        ignore_energies method of the FitSpectroPolarimetry class, which handles
        both channel grids at once.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored energy interval.
        bound_hi : float
            Higher bound of ignored energy interval.    
        """
        
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Energy bounds must be floats or integers")
        
        self.pol_ebounds_mask = ((self._pol_emin_unmasked<bound_lo)|
                                 (self._pol_emax_unmasked>bound_hi))& \
                                 self.pol_ebounds_mask
        
        self._update_polarization_grids()
        #filter and re-flatten the data using the same mask 
        self._reflatten_data() 
        return

    def notice_polarization_energies(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        notice selected (previously ignored) Stokes Q and U channels based on 
        their energy bounds. The Stokes I channels are left untouched; users 
        should call the notice_energies method of the FitSpectroPolarimetry 
        class, which handles both channel grids at once.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of noticed energy interval.
        bound_hi : float
            Higher bound of noticed energy interval.    
        """
        
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Energy bounds must be floats or integers")
        
        self.pol_ebounds_mask = self.pol_ebounds_mask|np.logical_not(
                                (self._pol_emin_unmasked<bound_lo)|
                                (self._pol_emax_unmasked>bound_hi))
        
        self._update_polarization_grids()
        #filter and re-flatten the data using the same mask 
        self._reflatten_data() 
        return

def load_pha(path,response):
    '''
    This function loads an X-ray spectrum , given an input path to an OGIP-compatible
    file and a nDspec ResponseMatrix object to be applied to the spectrum. 
  
    
    Parameters:
    -----------
    path: str 
        A string pointing to the spectrum file to be loaded 
        
    response: nDspec.ResponseMatrix 
        The instrument response matrix, loaded in nDspec, corresponding to the 
        spectrum to be loaded 
        
    Returns:
    --------
    bin_bounds_lo: np.array(float)
        An array of lower energy channel bounds, in keV, as contained in the 
        input file. If the spectrum was grouped, this contains the lower bounds 
        of the spectrum after rebinning.
        
    bin_bounds_hi: np.array(float)
        An array of upper energy channel bounds, in keV, as contained in the 
        input file. If the spectrum was grouped, this contains the lower bounds 
        of the spectrum after rebinning.
        
    counts_per_group: np.array(float)
        The total number of photon counts in each energy channel. If the spectrum 
        was grouped, this contains the counts in each channel after rebinning. 
        
    spectrum_error: np.array(float)
        The error on the counts in each group, including both Poisson and (if
        present) systematic errors
        
    exposure: float
        The exposure time contained in the spectrum file.   
        
    backscal: float 
        The background scaling factor. Typically used by imaging instruments to 
        account for different extraction region size for the source and background.    
    '''
    from astropy.io import fits
    from astropy.io.fits.card import Undefined, UNDEFINED
    
    with fits.open(path,filemap=False) as spectrum:
        extnames = np.array([h.name for h in spectrum])
        hdr = spectrum["SPECTRUM"].header
        spectrum_data = spectrum['SPECTRUM'].data
        channels = spectrum_data['CHANNEL']
        #check if exposure is present in either the primary or spectrum headers
        try:
            exposure = spectrum['PRIMARY'].header['EXPOSURE']
        except KeyError:
             try:
                exposure = spectrum['SPECTRUM'].header['EXPOSURE']
             except KeyError:
                exposure = 1.        
        try:         
            counts = spectrum_data['COUNTS']
        except KeyError:
            try:         
                counts = spectrum_data['RATE']*exposure
            except KeyError:
                raise FileNotFoundError("Fits file format incompatible, ensure it is OGIP compliant")        
        try:
            backscal = spectrum['PRIMARY'].header['BACKSCAL']
        except KeyError:
            try:
                backscal = spectrum['PRIMARY'].header['BACKSCAL']
            except KeyError:
                backscal = 1.
                warnings.warn("WARNING: backscal keyword not found, check file format",
                              UserWarning)     
        if backscal == 0.:
            backscal = 1.
            warnings.warn("WARNING: found backscal=0, assuming it is 1",
                          UserWarning)               
        
        #check that the spectrum and response have the same mission and channel 
        #number         
        mission_spectrum = hdr["TELESCOP"]
        instrument_spectrum = hdr["INSTRUME"]
        if mission_spectrum != response.mission:
            raise NameError("Observatory in the spectrum different from the response")
        if instrument_spectrum != response.instrument:
            raise NameError("Instrument in the spectrum different from the response")        
        #check if systematic errors are applied
        try: 
            sys_err = spectrum_data['SYS_ERR']   
            has_sys_err = True
        except KeyError:
            sys_err = np.zeros(len(counts))
            has_sys_err = False
        #check if the spectrum has been grouped
        try: 
            grouping_data = spectrum_data['GROUPING']  
            has_grouping = True
        except KeyError:
            has_grouping = False
        #calculate errors including systematics if present
        #note: we are summing the systematic and Poisson errors in quadrature
        #so the factor sqrt in the Poisson error factors out
        if has_sys_err:
            counts_err = np.sqrt(counts+(counts*sys_err)**2)
        else:
            counts_err = np.sqrt(counts)
        #calculate the spectrum whether it has been grouped or not, along with 
        #the energy bounds, width, and errors for each bin in either case
        if has_grouping:
            group_start = np.where(grouping_data==1)[0]
            total_groups = len(group_start)
            counts_per_group = np.zeros(total_groups,dtype=int)
            bin_bounds_lo = np.zeros(total_groups)
            bin_bounds_hi = np.zeros(total_groups)
            avg_sys = np.zeros(total_groups)
            for i in range(total_groups-1):
                counts_per_group[i] = np.sum(counts[group_start[i]:group_start[i+1]])
                avg_sys[i] = np.mean(sys_err[group_start[i]:group_start[i+1]])
                bin_bounds_lo[i] = response.emin[group_start[i]]
                #the upper bounds of this bin are the starting point of the next bin up in the grouping
                bin_bounds_hi[i] = response.emin[group_start[i+1]]    
            #the last bin needs to be accounted for explicitely because the photons may not end up
            #being regrouped
            counts_per_group[-1] = np.sum(counts[group_start[total_groups-1]:])
            avg_sys[-1] = np.mean(sys_err[group_start[total_groups-1]:])
            bin_bounds_lo[-1] = bin_bounds_hi[-2]
            bin_bounds_hi[-1] = response.emax[-1]
            sys_err_per_group = counts_per_group*avg_sys
            spectrum_error = np.sqrt(sys_err_per_group**2+counts_per_group)
        else:
            bin_bounds_lo = response.emin
            bin_bounds_hi = response.emax
            counts_per_group = counts
            spectrum_error = counts_err
        return bin_bounds_lo, bin_bounds_hi, counts_per_group, spectrum_error, exposure, backscal


def load_stokes_pha(path,response,stokes=None):
    '''
    This function loads a Stokes parameter spectrum, given an input path to an 
    OGIP-compatible file and a nDspec ResponseMatrix object to be applied to the 
    spectrum. 
    
    It is analogous to the load_pha function, but unlike a time-averaged 
    spectrum, the Stokes Q and U spectra are the difference between two 
    Poisson-distributed quantities: their counts can be negative, and their 
    errors can not be computed from the counts alone. For this reason this 
    function requires a STAT_ERR column to be present in the file, and sums the 
    errors in quadrature when the spectrum has been grouped. 
    
    Parameters:
    -----------
    path: str 
        A string pointing to the Stokes spectrum file to be loaded 
        
    response: nDspec.ResponseMatrix 
        The instrument response matrix, loaded in nDspec, corresponding to the 
        spectrum to be loaded. For Stokes Q and U this is typically the 
        modulation response function, rather than the standard response.
        
    stokes: str, default None 
        The Stokes parameter ("I", "Q" or "U") the file is expected to contain. 
        If provided, it is checked against the STOKES keyword in the file 
        header, if the latter is present.
        
    Returns:
    --------
    bin_bounds_lo: np.array(float)
        An array of lower energy channel bounds, in keV, as contained in the 
        input file. If the spectrum was grouped, this contains the lower bounds 
        of the spectrum after rebinning.
        
    bin_bounds_hi: np.array(float)
        An array of upper energy channel bounds, in keV, as contained in the 
        input file. If the spectrum was grouped, this contains the upper bounds 
        of the spectrum after rebinning.
        
    counts_per_group: np.array(float)
        The total number of photon counts in each energy channel. If the 
        spectrum was grouped, this contains the counts in each channel after 
        rebinning. For Stokes Q and U these can be negative.
        
    spectrum_error: np.array(float)
        The error on the counts in each group, including both statistical and 
        (if present) systematic errors
        
    exposure: float
        The exposure time contained in the spectrum file.   
        
    backscal: float 
        The background scaling factor. Typically used to account for different 
        extraction region size for the source and background.    
    '''
    from astropy.io import fits
    
    with fits.open(path,filemap=False) as spectrum:
        hdr = spectrum["SPECTRUM"].header
        spectrum_data = spectrum['SPECTRUM'].data
        #check if exposure is present in either the primary or spectrum headers
        try:
            exposure = spectrum['PRIMARY'].header['EXPOSURE']
        except KeyError:
             try:
                exposure = spectrum['SPECTRUM'].header['EXPOSURE']
             except KeyError:
                exposure = 1.        
        try:         
            counts = spectrum_data['COUNTS']
            is_rate = False
        except KeyError:
            try:         
                counts = spectrum_data['RATE']*exposure
                is_rate = True
            except KeyError:
                raise FileNotFoundError("Fits file format incompatible, ensure it is OGIP compliant")        
        try:
            backscal = spectrum['SPECTRUM'].header['BACKSCAL']
        except KeyError:
            try:
                backscal = spectrum['PRIMARY'].header['BACKSCAL']
            except KeyError:
                backscal = 1.
                warnings.warn("WARNING: backscal keyword not found, check file format",
                              UserWarning)     
        if backscal == 0.:
            backscal = 1.
            warnings.warn("WARNING: found backscal=0, assuming it is 1",
                          UserWarning)               
        
        #check that the spectrum and response have the same mission and channel 
        #number         
        mission_spectrum = hdr["TELESCOP"]
        instrument_spectrum = hdr["INSTRUME"]
        if mission_spectrum != response.mission:
            raise NameError("Observatory in the spectrum different from the response")
        if instrument_spectrum != response.instrument:
            raise NameError("Instrument in the spectrum different from the response")        
        #check the file contains the Stokes parameter the user expects, if the 
        #keyword tracking it is present
        if stokes is not None:
            stokes_index = {"I":0, "Q":1, "U":2}
            try:
                stokes_keyword = hdr["STOKES"]
                if stokes_keyword != stokes_index[stokes]:
                    raise NameError("Stokes parameter in the spectrum different from the one requested")
            except KeyError:
                warnings.warn("WARNING: stokes keyword not found, check file format",
                              UserWarning)
        #unlike for a time-averaged spectrum, the errors can not be computed 
        #from the counts, so they have to be stored in the file
        try: 
            counts_err = spectrum_data['STAT_ERR']   
            if is_rate:
                counts_err = counts_err*exposure
        except KeyError:
            if np.min(counts) < 0:
                raise FileNotFoundError("Stokes spectrum contains negative counts but no STAT_ERR column")
            counts_err = np.sqrt(counts)
            warnings.warn("WARNING: no STAT_ERR column found, assuming Poisson errors",
                          UserWarning)
        #check if systematic errors are applied
        try: 
            sys_err = spectrum_data['SYS_ERR']   
        except KeyError:
            sys_err = np.zeros(len(counts))
        counts_err = np.sqrt(counts_err**2+(counts*sys_err)**2)
        #check if the spectrum has been grouped
        try: 
            grouping_data = spectrum_data['GROUPING']  
            has_grouping = True
        except KeyError:
            has_grouping = False
        #calculate the spectrum whether it has been grouped or not, along with 
        #the energy bounds and errors for each bin in either case
        if has_grouping:
            group_start = np.where(grouping_data==1)[0]
            total_groups = len(group_start)
            counts_per_group = np.zeros(total_groups)
            spectrum_error = np.zeros(total_groups)
            bin_bounds_lo = np.zeros(total_groups)
            bin_bounds_hi = np.zeros(total_groups)
            for i in range(total_groups-1):
                counts_per_group[i] = np.sum(counts[group_start[i]:group_start[i+1]])
                spectrum_error[i] = np.sqrt(np.sum(counts_err[group_start[i]:group_start[i+1]]**2))
                bin_bounds_lo[i] = response.emin[group_start[i]]
                #the upper bounds of this bin are the starting point of the next bin up in the grouping
                bin_bounds_hi[i] = response.emin[group_start[i+1]]    
            #the last bin needs to be accounted for explicitely because the photons may not end up
            #being regrouped
            counts_per_group[-1] = np.sum(counts[group_start[total_groups-1]:])
            spectrum_error[-1] = np.sqrt(np.sum(counts_err[group_start[total_groups-1]:]**2))
            bin_bounds_lo[-1] = bin_bounds_hi[-2]
            bin_bounds_hi[-1] = response.emax[-1]
        else:
            bin_bounds_lo = response.emin
            bin_bounds_hi = response.emax
            counts_per_group = counts
            spectrum_error = counts_err
        return bin_bounds_lo, bin_bounds_hi, counts_per_group, spectrum_error, exposure, backscal
 


def load_lc(path):
    '''
    This function loads an X-ray lightcurve, given an input path to an 
    OGIP-compatible file.
    
    Parameters:
    -----------
    path: str 
        A string pointing to the lightcurve file to be loaded 
   
    Returns:
    --------
    time_bins: np.array(float)
        An array of time stamps covered by the lightcurve 
        
    counts: np.array(float) 
        An array of counts rates (defined in counts per second) contained in the 
        lightcurve 
        
    gti: list([float,float])
        A list of good time intervals over which the lightcurve is defined. 
    '''
    from astropy.io import fits

    with fits.open(path,filemap=False) as lightcurve:
        extnames = np.array([h.name for h in lightcurve])
        lightcurve_data = lightcurve['RATE'].data
        time_bins = lightcurve_data['TIME']
        counts = lightcurve_data['RATE']
        gti_data = lightcurve['GTI'].data
        #convert from astropy to numpy - this is annoying
        #for 2d arrays hence the horror below
        gti = np.zeros((len(gti_data),2))
        for i in range(len(gti_data)):
            gti[i][0] = gti_data[i][0]-gti_data[0][0]
            gti[i][1] = gti_data[i][1]-gti_data[0][0]

    return time_bins, counts, gti
