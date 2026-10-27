import numpy as np

from .SimpleFit import SimpleFit, FrequencyDependentFit
from .Likelihoods import chisq, ratio
from .Utils import model_expand
from . import Plotting

class FitPowerSpectrum(SimpleFit,FrequencyDependentFit):
    """
    Least-chi squared fitter class for a powerspectrum, defined as the product 
    between a Fourier-transformed lightcurve and its complex conjugate. 
    
    Given an array of Fourier frequencies, a power spectrum, its error and a 
    model (defined in Fourier space), this class handles fitting internally 
    using the lmfit library.    
    
    Poisson noise in the data is not accounted for explicitely. Users should 
    either pass noise-subtracted data, or include a constant component in the 
    model definition (see below).   
        
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
        An array storing the data to be fitted. Only contains noticed bins. 
   
    data_err: np.array(float)
        An array containing the uncertainty on the data to be fitted. Only 
        contains noticed bins. 
        
    _data_unmasked, _data_err_unmasked: np.array(float)
        The arrays of every data bin and its error, regardless of which ones are
        ignored or noticed during the fit. Used exclusively to enable book 
        keeping internal to the fitter class.   
        
    Attributes inherited from FrequencyDependentFit: 
    ------------------------------------------------    
    _freqs_unmasked: np.array(float) 
        The array of Fourier frequency over which all data and model are 
        defined, including bins that are ignored in the fit. 
    
    freqs_mask: np.array(bool)
        The array of Fourier frequencies that are either ignored or noticed 
        during the fit. A given channel i is noticed if freqs_mask[i] is True,
        and ignored if it is false.      
    
    n_freqs: int 
        The number of Fourier frequency bins that are noticed in the fit. 
        
    Other attributes:
    -----------------    
    freqs: np.array(float)
        The Fourier frequency over which both the data and model are defined, 
        in units of Hz. Only contains noticed bins.          
    """ 

    def __init__(self,likelihood="chisq"):
        self.dependence = "frequency"
        SimpleFit.__init__(self,likelihood)
        self.freqs = None 
        pass

    def set_data(self,data,data_err=None,data_grid=None,frebin_fac=None):
        """
        This method is used to pass the data users want to fit. The input can
        be either three arrays including the power, its erorr, and the Fourier 
        frequency grid, or a Stingray Powerspectrum object, in which case the 
        error and frequency array are handled automatically.
        
        Parameters:
        -----------           
        data: np.array(float) or stingray.powerspectrum
            The power spectrum to be fitted, either as a numpy array or a 
            stingray object 
            
        data_err: np.array(float), default: None 
            The error on the power spectrum. If passing a stingray object, this 
            is not necessary and is therefore ignored
            
        data_grid: np.array(float), default: None 
            The Fourier frequency grid over which the data (and model) are 
            defined. If passing a stingray object, this is not necessary and is 
            therefore ignored      
            
        frebin_fac: float, default: None
            A user-specified factor by which to rebin the data logarithmically 
            when it is loaded through Stingray, should users choose to do so.
        """

        if getattr(data, '__module__', None) == "stingray.powerspectrum":
            if frebin_fac is not None:
                data = data.rebin_log(frebin_fac)         
            self.data = data.power
            self.data_err = data.power_err
            FrequencyDependentFit.__init__(self,data.freq)            
        else:
            if len(data) != len(data_err):
                raise AttributeError("Input data and error arrays are different")
            if len(data) != len(data_grid):
                raise AttributeError("Input data and frequency arrays are different")        
            self.data = data
            self.data_err = data_err
            FrequencyDependentFit.__init__(self,data_grid)    
        self._set_unmasked_data()
        return
       
    def eval_model(self,params=None,freq=None,mask=True):
        """
        This method is used to evaluate and return the model values for a given 
        set of parameters,  over a given Fourier frequency grid. By default it  
        will evaluate the model over the data Fourier frequency grid and using  
        the values stored internally in the model_params attribute, but passing 
        a different grid/set of parameters is also possible.         
        
        Parameters:
        -----------                         
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are 
            provided, the model_params attribute is used.
            
        freq: np.array(float), default self._freqs_unmasked
            The the Fourier frequencies over which to evaluted the model. If 
            none are provided, the same frequencies over which the data is 
            defined are used. 
            
        mask: bool, default True
            A boolean switch to choose whether to mask the model output to only 
            include the noticed frequency bins, or to also return the ones that
            have been ignored by the users. 
            
        Returns:
        --------
        model: np.array(float)
            The model evaluated over the given Fourier frequency array, for the 
            given input parameters.   
        """
        
        if freq is None:
            freq = self._freqs_unmasked
        elif mask is True:
            raise ValueError(("mask=True requires freq=None to evaluate the model "
                              "on the stored frequency grid."))
        
        if params is None:
            params = self.model_params
        
        model = self.model.eval(params,freq=freq)          
        
        if mask is True:
            model = np.extract(self.freqs_mask,model)
            
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
        model = self.model.eval(params,freq=self.freqs)     
           
        if self.likelihood == "chisq":
            residuals, _ = self.get_residuals("chisq",model=model,mask=True)
        elif self.likelihood == "custom":
            residuals, _ = self.get_residuals("custom",model=model,mask=True)
        else:
            raise AttributeError("Likelihood type not supported")
            
        return residuals
    
    def plot_data(self,units="fpower",panels=None,return_plot=False,
                  data_kwargs=None):
        """
        This method plots the powerspectrum loaded by the user as a function of 
        Fourier frequency. It is possible to plot both units of power, and power 
        times frequency, depending on user input. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        units: str, default="fpower"
            The units to use for the y axis. units="fpower", the default, plots 
            the data in units of power*frequency. units="power" instead plots 
            the data in units of power. 
            
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
        
        if units == "power":
            scale = np.ones(len(self._data_unmasked))
            y_label = "Power"
        elif units == "fpower":
            scale = self._freqs_unmasked
            y_label = "Power$\\times$frequency"
        else:
            raise ValueError("Y axis units not recognized")

        freqs = self._freqs_unmasked
        x_bars = 0.5*np.append(np.diff(freqs),freqs[-1]-freqs[-2])
        #nan jank to correctly lay out gaps throughout the library 
        y_points = np.where(self.freqs_mask,self._data_unmasked,np.nan)*scale
        y_bars = np.where(self.freqs_mask,self._data_err_unmasked,np.nan)*scale
        
        panel_data = Plotting.make_panel_data(x_points=freqs,
                                              x_bars=x_bars,
                                              y_points=y_points,
                                              y_bars=y_bars,
                                              x_label="Frequency (Hz)",
                                              y_label=y_label)

        if panels is None:
            plot_layout = Plotting.make_layout(figsize=(6.5,4.5))
            fig, panel = Plotting.make_panels(plot_layout)
        else:
            #this fitter only ever works on a single panel, other fitters will
            #have more complex layouts 
            panel = panels
            fig = panel.get_figure()   
        
        Plotting.draw_main_panel(panel,panel_data,draw_data=True,draw_model=False,
                                 data_kwargs=data_kwargs)
        
        if return_plot is True:
            return fig, panel
        else:
            return         
        
    def plot_model(self,plot_data=True,plot_components=False,params=None,
                   units="fpower",residuals=None,panels=None,return_plot=False,
                   data_kwargs=None,model_kwargs=None,component_kwargs=None,
                   resid_kwargs=None):
        """
        This method plots the model defined by the user as a function of 
        Fourier frequency, as well as (optionally) its components, and the data
        plus model residuals. It is possible to plot both units of power, and 
        power times frequency, depending on user input. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        plot_data: bool, default=True
            If true, both model and data are plotted; if false, just the model. 
            
        plot_components: bool or str, default=False 
            If True, the additive model components are overplotted, with 
            multiplicative and convolution factors applied to each. If 
            "additive", the additive components are plotted without being 
            affected by  any multiplicative or convolution factors. 
            
        params: lmfit.parameters, default=None 
            The parameters to be used to evaluate the model. If None, the set 
            of parameters stored in the class is used 
        
        units: str, default="fpower"
            The units to use for the y axis. units="fpower", the default, plots 
            the data in units of power*frequency. units="power" instead plots 
            the data in units of power. 
            
        residuals: str, default=None
            The units to use for the residuals. If residuals="chisq", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model. If a custom likelihood 
            is defined, then the residuals are computed from that likelihood. If
            residual units are not specified, they are computed from the  
            likelihood set by the user. 

        panels : matplotlib.axes or numpy.ndarray(matplotlib.axes), default=None
            Existing panels to draw into, e.g as returned by calling plot_model 
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
        
        if params is None:
            params = self.model_params
        if residuals is None:
            residuals = self.likelihood
        if residuals == "chisq":
            res_label = "$\\Delta\\chi$"
        elif residuals == "ratio":
            res_label = "Data/model"
        elif residuals == "custom":
            res_label = "Residuals"
        else:
            raise ValueError("Residual format not supported")
        
        if units == "power":
            scale = np.ones(len(self._data_unmasked))
            y_label = "Power"
        elif units == "fpower":
            scale = self._freqs_unmasked
            y_label = "Power$\\times$frequency"
        else:
            raise ValueError("Y axis units not supported")

        freqs = self._freqs_unmasked
        x_bars = 0.5*np.append(np.diff(freqs),freqs[-1]-freqs[-2])        
        
        #horrible nan jank to handle data with gaps across the library  
        model = self.eval_model(params=params,mask=False)
        model_vals = np.where(self.freqs_mask,model*scale,np.nan)
        
        data = np.where(self.freqs_mask,self._data_unmasked*scale,np.nan)
        data_err = np.where(self.freqs_mask,self._data_err_unmasked*scale,np.nan)

        components = None
        if plot_components is not False:
            components = model_expand(self.model,params,freq=freqs,
                                      distribute=(plot_components != "additive"))
            for key in components.keys():
                components[key] = np.where(self.freqs_mask,
                                           components[key]*scale,np.nan)

        res, err = self.get_residuals(res_type=residuals,
                                      model=np.extract(self.freqs_mask,model))
        #same nan jank as above
        resid = np.full(len(freqs),np.nan)
        reserr = np.full(len(freqs),np.nan)
        resid[self.freqs_mask] = res
        reserr[self.freqs_mask] = err

        panel_data = Plotting.make_panel_data(x_points=freqs,
                                              x_bars=x_bars,
                                              y_points=data,
                                              y_bars=data_err,
                                              model_points=freqs,
                                              model_vals=model_vals,
                                              component_vals=components,
                                              resid=resid,
                                              reserr=reserr,
                                              x_label="Frequency (Hz)",
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
                                 draw_components=(plot_components is not False),
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
