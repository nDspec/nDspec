import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams.update({'font.size': 17})

from lmfit.model import ModelResult as LM_result

from .Response import ResponseMatrix
from .SimpleFit import SimpleFit, EnergyDependentFit, load_pha

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
   
    likelihood: None
        Work in progress; currently the software defaults to chi squared 
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

    Other attributes:
    -----------------
    response: nDspec.ResponseMatrix
        The instrument response matrix corresponding to the spectrum to be 
        fitted. It is required to define the energy grids over which model and
        data are defined.   
    """ 
    
    def __init__(self):
        SimpleFit.__init__(self)
        pass

    def set_data(self,response,data):
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
        """

        bounds_lo, bounds_hi, counts, error, exposure = load_pha(data,response)
        self.response = response.rebin_channels(bounds_lo,bounds_hi)   
        EnergyDependentFit.__init__(self)  
        #this loads the spectrum in units of counts/s/keV
        self.data = counts/exposure/self.ewidths
        self.data_err = error/exposure/self.ewidths
        self._set_unmasked_data()
        return 

    def eval_model(self,params=None,energ=None,fold=True,mask=True):    
        """
        This method is used to evaluate and return the model values for a given 
        set of parameters,  over a given model energy grid. By default it  
        will evaluate the model over the energy grid defined in the response,
        using the parameters values stored internally in the model_params 
        attribute, without folding the model through the response.        
        
        Parameters:
        -----------                         
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are 
            provided, the model_params attribute is used.
            
        energ: np.array(float), default None
            The the photon energies over which to evaluted the model. If 
            none are provided, the same grid contained in the instrument response  
            is used. 
            
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
    
        if energ is None:
            energ = self.energs

        if params is None:
            model = self.model.eval(self.model_params,energ=energ)*self.energ_bounds
        else:
            model = self.model.eval(params,energ=energ)*self.energ_bounds

        if fold is True:
            model = self.response.convolve_response(model) 

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
    
        if self.likelihood is None:
            model = self.eval_model(params,energ=self.energs)
            #convolve = np.extract(self.ebounds_mask,model)
            residuals = (self.data-model)/self.data_err
        else:
            raise AttributeError("custom likelihood not implemented yet")
        return residuals

    def plot_data(self,units="data",return_plot=False):
        """
        This method plots the spectrum loaded by the user as a function of 
        energy. It is possible to plot both in detector and ``unfolded'' space, 
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
        units: str, default="data"
            The units to use for the y axis. units="data", the detector, plots 
            the data in detector space in units of counts/s/keV. units="unfold" 
            instead plots unfolded data and follows the Xspec convention for the 
            y axis - the y axis is in units of counts/s/keV/cm^2, times one 
            additional factor "keV" for each "e" that appears in the string. 
            For instance, units="eeunfold" plots units of kev^2 counts/s/keV/cm^2,
            i.e. units of nuFnu. 
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
    
        energies = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        xerror = 0.5*np.extract(self.ebounds_mask,self._ewidths_unmasked)   
        
        if units == "data":
            data = self.data
            yerror = self.data_err
            ylabel = "Folded counts/s/keV"
        elif units.count("unfold"):
            power = units.count("e")            
            data = self.response.unfold_response(self._data_unmasked)* \
                   self._ebounds_unmasked**power
            error = self.response.unfold_response(self._data_err_unmasked)* \
                    self._ebounds_unmasked**power  
            data = np.extract(self.ebounds_mask,data)
            yerror = np.extract(self.ebounds_mask,error)
            if power == 0:
                ylabel = "Counts/s/keV/cm$^{2}$"
            elif power == 1:
                ylabel = "Flux density (Counts/s/cm$^{2}$)"
            elif power == 2:
                ylabel = "Flux (keV/s/cm$^{2}$)"
            #with weird units, use a generic label
            else:
                ylabel == "keV^{}/s/keV/cm$^{2}$".format(str(power))
        else:
            raise ValueError("Y axis units not supported")
        
        fig, ((ax1)) = plt.subplots(1,1,figsize=(6.,4.5))   
        
        ax1.errorbar(energies,data,yerr=yerror,xerr=xerror,
                     linestyle='',marker='o')
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel("Energy (keV)")          
        ax1.set_xscale("log",base=10)
        ax1.set_yscale("log",base=10)
        
        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return 

    def plot_model(self,plot_data=True,plot_components=False,params=None,
                   units="data",residuals="delchi",return_plot=False):
        """
        This method plots the model defined by the user as a function of 
        energy, as well as (optionally) its components, and the data plus model
        residuals. It is possible to plot both in detector and ``unfolded'' space, 
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
            
        plot_components: bool, default=False 
            If true, the model components are overplotted; if false, they are 
            not. Only additive model components will display their values 
            correctly. 
            
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
            
        residuals: str, default="delchi"
            The units to use for the residuals. If residuals="delchi", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """           
                                     
        energies = np.extract(self.ebounds_mask,self._ebounds_unmasked)
        xerror = 0.5*np.extract(self.ebounds_mask,self._ewidths_unmasked)       
        
        #first; get the model in the correct units
        model_fold = self.eval_model(params=params,energ=self.energs,mask=False)
        if units == "data":   
            model = np.extract(self.ebounds_mask,model_fold)   
            ylabel = "Folded counts/s/keV"
        elif units.count("unfold"):
            power = units.count("e") 
            model = self.response.unfold_response(model_fold)
            if power == 0:
                ylabel = "Counts/s/keV/cm$^{2}$"
            elif power == 1:
                ylabel = "Flux density (Counts/s/cm$^{2}$)"
            elif power == 2:
                ylabel = "Flux (keV/s/cm$^{2}$)"
            #with weird units, use a generic label
            else:
                ylabel == "keV^{}/s/keV/cm$^{2}$".format(str(power))  
            model = np.extract(self.ebounds_mask,model)
            model = model*self.ebounds**power
        else:
            raise ValueError("Y axis units not supported")
            
        #if we're also plotting data, get the data in the same units
        #as well as the residuals
        if plot_data is True:
            model_res,res_errors = self.get_residuals(residuals,model=model)
            if residuals == "delchi":
                reslabel = "$\\Delta\\chi$"
            elif residuals == "ratio":
                reslabel = "Data/model"
            else:
                raise ValueError("Residual format not supported")   
                            
            if units == "data":
                data = self.data
                yerror = self.data_err
                ylabel = "Folded counts/s/keV"
            elif units.count("unfold"):        
                data = self.response.unfold_response(self._data_unmasked)* \
                       self._ebounds_unmasked**power
                error = self.response.unfold_response(self._data_err_unmasked)* \
                        self._ebounds_unmasked**power  
                data = np.extract(self.ebounds_mask,data)
                yerror = np.extract(self.ebounds_mask,error)
            
        if plot_data is False:
            fig, (ax1) = plt.subplots(1,1,figsize=(6.,4.5))   
        else:
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6.,6.),
                                          sharex=True,
                                          gridspec_kw={'height_ratios': [2, 1]})

        if plot_data is True:
            ax1.errorbar(energies,data,yerr=yerror,xerr=xerror,
                         ls="",marker='o')       

        ax1.plot(energies,model,lw=3,zorder=3)

        if plot_components is True:
            #we need to allocate a ModelResult object in order to retrieve the components
            comps = LM_result(model=self.model,params=self.model_params).eval_components(energ=self.energs)
            for key in comps.keys():
                comp_folded = self.response.convolve_response(comps[key]*self.energ_bounds)
                #do it better here
                if units == "data":   
                    comp = np.extract(self.ebounds_mask,comp_folded)
                    ax1.plot(energies,comp,label=key,lw=2)
                elif units.count("unfold"):
                    comp_unfold = self.response.unfold_response(comp_folded)
                    comp = np.extract(self.ebounds_mask,comp_unfold)
                    ax1.plot(energies,comp*energies**power,label=key,lw=2)
            ax1.legend(loc='best')
        
        ax1.set_xscale("log",base=10)
        ax1.set_yscale("log",base=10)
        if plot_data is False:
            ax1.set_xlabel("Energy (keV)")
        ax1.set_ylabel(ylabel)

        if plot_data is True:
            ax1.set_ylim([0.85*np.min(data),1.15*np.max(data)])
            ax2.errorbar(energies,model_res,yerr=res_errors,
                         linestyle='',marker='o')
            if residuals == "delchi":
                ax2.plot(energies,np.zeros(len(energies)),
                         ls=":",lw=2,color='black')
            elif residuals == "ratio":
                ax2.plot(energies,np.ones(len(energies)),
                         ls=":",lw=2,color='black')                
            ax2.set_xlabel("Energy (keV)")
            ax2.set_ylabel(reslabel)

        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return  
