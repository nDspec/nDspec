import numpy as np

from .Response import ResponseMatrix
from . import Plotting
from .SimpleFit import SimpleFit, EnergyDependentFit, load_pha
from .Likelihoods import cstat, chisq, ratio
from .Utils import model_expand

class FitTimeAvgSpectrum(SimpleFit,EnergyDependentFit):
    """
    Least-chi squared fitter class for a time averaged spectrum, defined as the  
    count rate as a function of photon channel energy bound. 
    
    Given an instrument response, a count rate spectrum, its error and a 
    model (defined in energy space), this class handles fitting internally 
    using the lmfit library.    
        
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
       
    _emin_unmasked, _emax_unmasked, _ebounds_unmasked, _ewidths_unmasked: np.array(float)
        The array of every lower bound, upper bound, channel center and channel 
        widths stored in the response, regardless of which ones are ignored or 
        noticed during the fit. Used exclusively to facilitate book-keeping 
        internal to the fitter class.         

    gain_params: lmfit.Parameters, default None 
        A lmfit Parameters object, which contains the parameters for the gain  
        correction model components if it is enabled. Defaults to None. 

    Other attributes:
    -----------------
    response: nDspec.ResponseMatrix
        The instrument response matrix corresponding to the spectrum to be 
        fitted. It is required to define the energy grids over which model and
        data are defined. 
        
    exposure: np.float 
        The exposure time of the observation. Only used for calculating 
        Poisson-type likelihoods.
    """ 
    
    def __init__(self,likelihood="chisq"):
        SimpleFit.__init__(self,likelihood)
        self.response = None    
        pass

    def set_data(self,response,data,background=None):
        """
        This method sets the data to be fitted, its error, and the  energy and 
        channel grids given an input spectrum and its associated response matrix. 
        
        If the file provided was grouped with heatools, the method loads the 
        grouped data and adjusts the channel grid automatically. The data is 
        assumed to be background-subtracted (or to have negligible background).
        
        Parameters:
        -----------
        response: nDspec.ResponseMatrix
            An instrument response (including both rmf and arf) loaded into a 
            nDspec ResponseMatrix object. 
        
        data: str 
            A string pointing to the path of an X-ray spectrum file, stored in 
            a type 1 OGIP-formatted file (such as a pha file produced by a
            typical instrument reduction pipeline).
            
        background: str, default None 
            a string pointing to the path of an X-ray spectrum background file, 
            stored in a type 1 OGIP-formatted file. If not provided, the software 
            assumes the data is either already background-subtracted, or that 
            the user wants to ignore or model the background themselves. 
        """

        bounds_lo, bounds_hi, counts, error, exposure, src_backsc = load_pha(data,response)
        self.response = response.rebin_channels(bounds_lo,bounds_hi) 
        EnergyDependentFit.__init__(self)  

        #this loads the spectrum in units of counts/s/keV
        self.data = counts/exposure/self.ewidths
        self.data_err = error/exposure/self.ewidths
        self.exposure = exposure
        
        if background is not None:
            bounds_bkg_lo, bounds_bkg_hi, bkg_counts, bkg_error, _, bkg_backsc = load_pha(background,response)       
            backfac = src_backsc/bkg_backsc
            self.noise = self.response._rebin_sum(bkg_counts,
                                                  [bounds_bkg_lo, bounds_bkg_hi],
                                                  [bounds_lo, bounds_hi])
            #for imaging instruments, this factor acconuts for cases when the 
            #area of extracted spectra+backgrounds is different. 
            self.noise_err = np.sqrt(self.noise)*backfac/exposure/self.ewidths
            self.noise = self.noise*backfac/exposure/self.ewidths

        self._set_unmasked_data()
        return 
    
    def set_response(self,response):
        """
        This method sets the response matrix for the observation. It defines
        the energy grids over which model and data are defined. Generally,
        this method should only be called if the user is intending to simulate
        data from a model, as the response is not rebinned to reflect the
        data loaded by the user. Use the set_data method instead to set the
        data and response together.
        
        Parameters:
        -----------
        response: nDspec.ResponseMatrix
            An instrument response (including both rmf and arf) loaded into a 
            nDspec ResponseMatrix object. 
        """
        if not isinstance(response,ResponseMatrix):
            raise TypeError("Response must be an instance of nDspec.ResponseMatrix")
        self.response = response
        EnergyDependentFit.__init__(self)  
        return

    def set_fit_statistic(self,stat):
        """
        This method is used to set the statistic to be optimized during the fit.
        By default, the optimizer will optimize the chi-squared statistic. 
        
        Parameters:
        -----------
        stat: str 
            A string with the name of the fit statistic to be used. Supported 
            statistics currently are "chisq" (the standard chi squared statistic, 
            appropriate for data in the Gaussian regime) and "cstat" (the Cash 
            statistic, see https://ui.adsabs.harvard.edu/abs/1979ApJ...228..939C/abstract,
            appropriate for Poisson-distributed data). 
        """
        if (stat != "chisq" and stat != "cstat" and stat != "custom"):
            raise ValueError("Fit statistic not recognized")
        self.likelihood = stat 

    def eval_model(self,params=None,ear=None,fold=True,mask=True):    
        """
        This method is used to evaluate and return the model values for a given 
        set of parameters, over a given model energy grid. By default it  
        will evaluate the model over the energy grid defined in the response,
        using the parameters values stored internally in the model_params 
        attribute, without folding the model through the response.        
        
        Parameters:
        -----------                         
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are 
            provided, the model_params attribute is used.

        ear: np.array(float), default None
            The array of photon energy bin edges over which to evaluate the 
            model.  If none are provided, the same grid contained in the 
            instrument response is used. 
            
        fold: bool, default True
            A boolean switch to choose whether to fold the evaluated model 
            through the instrument response or not. Not that in order for the 
            model to be folded, the energy grid over which it is defined MUST 
            be identical to that stored in the response matrix/class.
            
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
            
        if ear is None:
            ear = self.ear
            energ = self.energs 
            energ_bounds = self.energ_bounds
        else:
            energ = 0.5*(ear[1:]+ear[:-1])
            energ_bounds = ear[1:]-ear[:-1]
            
        if params is None:
            params = self.model_params

        model = self.model.eval(params,energ=energ,ear=ear)*energ_bounds

        if fold is True:
            model = self.response.convolve_response(model) 
            model = self._apply_calibration(model,params)
        elif mask is True:
            raise ValueError(("mask=True requires fold=True: the ignore/"
                              "notice mask is defined in channel space, and "
                              "only applies to the model once it has been "
                              "folded through the response.")) 

        if mask is True:
            model = np.extract(self.ebounds_mask,model)            

        return model

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
        elif self.likelihood == "cstat":
            residuals, _ = self.get_residuals("cstat",model=model,mask=True)
            #lmfit minimizes the sum of the squares of whatever this method
            #returns, so the residuals passed to it have to be the square root
            #of the contribution of each bin to the total statistic; the clip
            #guards against bins floating point error for bins with 0 counts 
            residuals = np.sqrt(np.clip(residuals,0.,None))
        elif self.likelihood == "custom":
            residuals, _ = self.get_residuals("custom",model=model,mask=True)
        else:
            raise AttributeError("Chosen likelihood not supported")
        return residuals

    def _to_plot_units(self,array,units):
        """
        This method converts an unmasked channel-space array (data, model, 
        background...) to the requested plotting units, returning only the 
        array bins in the noticed channels 
        
        Parameters:
        -----------
        array: np.array(float)
            The array whose units are going to be converted.  
            
        units: str
            The units to be included in the plot. Options are "data", for a 
            count spectrum in counts/s/keV, and "unfold", for unfolding the 
            response. Unfolded arrays can be multiplied by factors of energy 
            identically to the Xspec convention, e.g. "units=unfold" returns 
            units of counts/s/kev/cm^2, "units=eeunfold" returns keV/s/cm^2.
            
        Returns: 
        --------
        converted_array: np.array(float)
            The input array, converted to the units requested. 
        """
        if len(array) != self._all_chans:
            raise ValueError(f"The input array should have length {self._all_chans}")

        #the ignored channels become NaN to show gaps when plotting with 
        #ax.stairs         
        array = np.where(self.ebounds_mask,array,np.nan)
        if units == "data":
            converted_array = array
        elif units.count("unfold"):
            power = units.count("e")
            converted_array = self.response.unfold_response(array)* \
                              self._ebounds_unmasked**power
        else: 
            raise ValueError("Plot units not recognized")
        
        return converted_array        

    def plot_data(self,units="data",plot_bkg=False,panels=None,return_plot=False,
                  data_kwargs=None,bkg_kwargs=None):
        """
        This method plots the spectrum loaded by the user as a function of 
        energy. It is possible to plot both in detector and "unfolded" space, 
        with the caveat that unfolding data is fairly dangerous and should
        be interpreted with care (or not at all). 
        
        The definition of unfolded data is subjective; nDspec adopts the same 
        convention as ISIS, and defines an unfolded count spectrum Uf(h) as a 
        function of energy channel h as :
        Uf(h) = C(h)/sum(R(E)),
        where C(h) is the detector space spectrum, R(E) is the instrument response 
        and sum denotes the sum over energy bins. This definition has the 
        advantage of being model-independent and is analogous to the Xspec 
        (model-dependent) definition of unfolding data when the model is a 
        constant. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        units: str, default="data"
            The units to use for the y axis. units="data", the default, plots 
            the data in detector space in units of counts/s/keV. units="unfold" 
            instead plots unfolded data and follows the Xspec convention for the 
            y axis - the y axis is in units of counts/s/keV/cm^2, times one 
            additional factor "keV" for each "e" that appears in the string. 
            For instance, units="eeunfold" plots units of kev^2 counts/s/keV/cm^2,
            i.e. units of nuFnu. 
            
        plot_bkg; bool, default="False:
            A boolean to choose whether you want to plot the background
        
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

        panel: matplotlib.axes, optional 
            The panel containing the plot produced by the method.
        """
        
        if units == "data":
            y_label =  "Counts/s/keV"
        elif units.count("unfold"):
            power = units.count("e")
            labels = {0: "Counts/s/keV/cm$^{2}$",
                      1: "Flux density (Counts/s/cm$^{2}$)",
                      2: "Flux (keV/s/cm$^{2}$)"}
            y_label =  labels.get(power,"keV$^{"+str(power)+"}$/s/keV/cm$^{2}$")    
        else: 
            raise ValueError("Plot units not recognized")   
        
        bkg = None
        if plot_bkg is True:
            if self.noise is None:
                raise AttributeError("No background loaded")
            bkg = self._to_plot_units(self._noise_unmasked,units)
            
        data = self._to_plot_units(self._data_unmasked,units)
        data_err = self._to_plot_units(self._data_err_unmasked,units)
        
        panel_data = Plotting.make_panel_data(x_points=self._ebounds_unmasked,
                                              y_points=data,
                                              x_bars=0.5*self._ewidths_unmasked,
                                              y_bars=data_err,
                                              bkg_vals=bkg,
                                              x_label="Energy (keV)",
                                              y_label=y_label)
        
        if panels is None:
            plot_layout = Plotting.make_layout(figsize=(6.5,4.5))
            fig, panel = Plotting.make_panels(plot_layout)
        else:
            #this fitter only ever works on a single panel, other fitters will
            #have more complex layouts 
            panel = panels
            fig = panel.get_figure()   
            
        Plotting.draw_main_panel(panel,panel_data,draw_model=False,draw_bkg=plot_bkg,
                                 data_kwargs=data_kwargs,bkg_kwargs=bkg_kwargs)
        
        if return_plot is True:
            return fig, panel
        else:
            return

    def plot_model(self,plot_data=True,plot_components=False,plot_bkg=False,
                   params=None,units="data",residuals=None,panels=None,
                   return_plot=False,data_kwargs=None,model_kwargs=None,
                   component_kwargs=None,resid_kwargs=None):
        """
        This method plots the model defined by the user as a function of 
        energy, as well as (optionally) its components, and the data plus model
        residuals. It is possible to plot both in detector and "unfolded" space, 
        with the caveat that unfolding data is EXTREMELY dangerous and should
        be interpreted with care (or not at all). 
        
        The definition of unfolded data is subjective; nDspec adopts the same 
        convention as ISIS, and defines an unfolded count spectrum Uf(h) as a 
        function of energy channel h as :
        
        Uf(h) = C(h)/sum(R(E)),
        
        where C(h) is the detector space spectrum, R(E) is the instrument response 
        and sum denotes the sum over energy bins. This definition has the 
        advantage of being model-independent and is analogous to the Xspec 
        (model-dependent) definition of unfolding data when the model is a 
        constant. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        plot_data: bool, default=True
            If true, both model and data are plotted; if false, just the model. 
            
        plot_components: bool or str, default=False 
            If True, the additive model components are overplotted, with 
            multiplicative and convolution factors applied to each. If 
            "additive", the additive components are returned without being 
            affected by  any multiplicative or convolution factors.
            
        plot_bkg; str, default=False:
            A boolean to choose whether you want to plot the background

        params: lmfit.parameters, default=None 
            The parameters to be used to evaluate the model. If False, the set 
            of parameters stored in the class is used 
        
        units: str, default="data"
            The units to use for the y axis. units="data", the detector, plots 
            the data in detector space in units of counts/s/keV. units="unfold" 
            instead plots unfolded data and follows the Xspec convention for the 
            y axis - the y axis is in units of counts/s/keV/cm^2, times one 
            additional factor "keV" for each "e" that appears in the string. 
            For instance, units="eeunfold" plots units of kev^2 counts/s/keV/cm^2,
            i.e. units of nuFnu. 
            
        residuals: str, default=None
            The units to use for the residuals. If residuals="chisq", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model; if residuals "cstat", the 
            plot shows the contribution of each bin to the Cash statistic. If
            residual units are not specified, they are computed from the  
            likelihood set by the user. 

        panels : matplotlib.axes or numpy.ndarray(matplotlib.axes), default=None
            An existing panel to draw into, e.g as returned by calling plot_data 
            from a second fitter. If None, a new figure is created.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure object containing 
            the plot or not.
            
        data_kwargs, model_kwargs, component_kwargs, resid_kwargs: dict, default=None 
            Keyword arguments for each element of the plot.

        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.

        panels: matplotlib.axes or numpy.ndarray(matplotlib.axes), optional 
            The panel(s) containing the plot produced by the method.
        """        

        if units == "data":
            y_label =  "Counts/s/keV"
        elif units.count("unfold"):
            power = units.count("e")
            labels = {0: "Counts/s/keV/cm$^{2}$",
                      1: "Flux density (Counts/s/cm$^{2}$)",
                      2: "Flux (keV/s/cm$^{2}$)"}
            y_label =  labels.get(power,"keV$^{"+str(power)+"}$/s/keV/cm$^{2}$")    
        else: 
            raise ValueError("Plot units not recognized")   
        
        if residuals is None:
            residuals = self.likelihood              
        
        if params is None:
            params = self.model_params  
        model_fold = self.eval_model(params=params,mask=False)
        model_vals = self._to_plot_units(model_fold,units)
        model_edges=np.append(self._emin_unmasked,
                              self._emax_unmasked[-1])

        resid, reserr = None, None
        data, data_err = None, None 
        res_label = None 
        if plot_data is True:            
            if residuals == "chisq":
                res_label = "$\\Delta\\chi$"
            elif residuals == "ratio":
                res_label = "Data/model"
            elif residuals == "cstat":
                res_label = "$\\Delta C$"
            elif residuals == "custom":
                res_label = "Residuals"
            else:
                raise ValueError("Residual format not supported")
            data = self._to_plot_units(self._data_unmasked,units)
            data_err = self._to_plot_units(self._data_err_unmasked,units)            
            #this horrible jank here is necessary to avoid raising annoying 
            #warnings about channels that are not being noticed, while still
            #supporting having gaps in the noticed channels 
            res, err = self.get_residuals(residuals,
                                          model=np.extract(self.ebounds_mask,model_fold))
            resid = np.full(len(self.ebounds_mask),np.nan)
            reserr = np.full(len(self.ebounds_mask),np.nan)
            resid[self.ebounds_mask] = res
            reserr[self.ebounds_mask] = err
        
        bkg = None
        if plot_bkg is True:
            if self.noise is None:
                raise AttributeError("No background loaded")
            bkg = self._to_plot_units(self._noise_unmasked,units)
        
        components = None
        if plot_components is not False:
            components = model_expand(self.model,params,
                                      distribute=(plot_components != "additive"),
                                      energ=self.energs,ear=self.ear)
            for key in components.keys():
                components[key] = self.response.convolve_response(components[key]*self.energ_bounds)
                components[key] = self._apply_calibration(components[key],params)
                components[key] = self._to_plot_units(components[key],units)

        panel_data = Plotting.make_panel_data(x_points=self._ebounds_unmasked,
                                              y_points=data,
                                              x_bars=0.5*self._ewidths_unmasked,
                                              y_bars=data_err,
                                              bkg_vals = bkg,
                                              model_edges=model_edges,
                                              model_vals=model_vals,                                        
                                              component_vals=components,
                                              resid=resid,
                                              reserr=reserr,
                                              x_label="Energy (keV)",
                                              y_label=y_label,
                                              res_label=res_label)

        layout_kwargs = dict(figsize=(6.5,4.5))
        if plot_data is True:
            #include residual panel if plotting data
            layout_kwargs = dict(nrows=2,height_ratios=[2,1],sharex=True,
                                 figsize=(6.5,6.))
        
        if panels is None:
            plot_layout = Plotting.make_layout(**layout_kwargs)
            fig, panels = Plotting.make_panels(plot_layout,squeeze=False)
        else:
            #This fitter only ever draws one column, but more complex polarimety 
            #or spectral timing stuff draws more and the functions need generic 
            #formatting to work in JointFit
            panels = np.reshape(np.asarray(panels,dtype=object),
                                (layout_kwargs.get("nrows",1),1))
            fig = panels[0,0].get_figure()

        Plotting.draw_main_panel(panels[0,0],panel_data,draw_data=plot_data,
                                 draw_components = (plot_components is not False),
                                 data_kwargs=data_kwargs,
                                 model_kwargs=model_kwargs,
                                 component_kwargs=component_kwargs) 
        
        if plot_data is True:
            panels[0,0].set_xlabel("")
            Plotting.draw_residual_panel(panels[1,0],panel_data,residuals,
                                         resid_kwargs=resid_kwargs)
        
        panels = panels[:,0]
        if plot_data is False:
            panels = panels[0]
        
        if return_plot is True:
            return fig, panels
        else:
            return
