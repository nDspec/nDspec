import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams.update({'font.size': 17})

from lmfit.model import ModelResult as LM_result

from .SimpleFit import SimpleFit, FrequencyDependentFit
from .Likelihoods import chisq, ratio

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
        likelihood by default.
        
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
     _freqs_unmasked 
    
    freqs_mask     
    
    n_freqs
        
    Other attributes:
    -----------------    
    freqs: np.array(float)
        The Fourier frequency over which both the data and model are defined, 
        in units of Hz. Only contains noticed bins.          
    """ 

    def __init__(self):
        SimpleFit.__init__(self)
        self.freqs = None 
        self.dependence = "frequency"
        pass

    def set_data(self,data,data_err=None,data_grid=None):
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
        """

        if getattr(data, '__module__', None) == "stingray.powerspectrum":         
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
            
        freq: np.array(float), default None
            The the Fourier frequencies over which to evaluted the model. If 
            none are provided, the same frequencies over which the data is 
            defined are used. 
            
        mask: bool, default True
            A boolean switch to choose whether to mask the model output to only 
            include the noticed energy channels, or to also return the ones 
            that have been ignored by the users. 
            
        Returns:
        --------
        model: np.array(float)
            The model evaluated over the given Fourier frequency array, for the 
            given input parameters.   
        """
        
        if freq is None:
            freq = self.freqs
        
        if params is None:
            model = self.model.eval(self.model_params,freq=freq)
        else:
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
    
    def plot_data(self,units="fpower",return_plot=False):
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
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        
        fig, ((ax1)) = plt.subplots(1,1,figsize=(6.,4.5))   
        
        if units == 'power':
            ax1.errorbar(self.freqs,self.data,yerr=self.data_err,
                         linestyle='',marker='o')
            ax1.set_ylabel("Power")
        elif units == "fpower":
            ax1.errorbar(self.freqs,self.data*self.freqs,yerr=self.data_err*self.freqs,
                         linestyle='',marker='o')
            ax1.set_ylabel("Power$\\times$frequency")
        else:
            raise ValueError("Y axis units not recognized")
        
        ax1.set_xscale("log",base=10)
        ax1.set_yscale("log",base=10)
        ax1.set_xlabel("Frequency (Hz)")  
        
        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return           
        
    def plot_model(self,plot_data=True,plot_components=False,params=None,
                   units="fpower",residuals="chisq",return_plot=False):
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
            
        plot_components: bool, default=False 
            If true, the model components are overplotted; if false, they are 
            not. Only additive model components will display their values 
            correctly. 
            
        params: lmfit.parameters, default=None 
            The parameters to be used to evaluate the model. If False, the set 
            of parameters stored in the class is used 
        
        units: str, default="fpower"
            The units to use for the y axis. units="fpower", the default, plots 
            the data in units of power*frequency. units="power" instead plots 
            the data in units of power. 
            
        residuals: str, default="chisq"
            The units to use for the residuals. If residuals="chisq", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model. If a custom likelihood 
            is defined, then the residuals are computed from that likelihood.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        
        if params is None:
            model = self.eval_model(params=self.model_params,mask=False)
        else:
            model = self.eval_model(params=params,mask=False)

        if plot_data is True:
            model_res, res_errors = self.get_residuals(res_type=residuals,model=model)

        if residuals == "chisq":
            reslabel = "$\\Delta\\chi$"
        elif residuals == "ratio":
            reslabel = "Data/model"
        elif self.likelihood == "custom":
            reslabel = "Residuals"
        else:
            raise ValueError("Residual format not supported")
            
        if units == "power":
            data = self.data
            error = self.data_err
            ylabel = "Power"
        elif units == "fpower":
            data = self.data*self.freqs
            error = self.data_err*self.freqs
            model = model*self.freqs
            ylabel= "Power$\\times$frequency"
        else:
            raise ValueError("Y axis units not supported")
            
        if plot_data is False:
            fig, (ax1) = plt.subplots(1,1,figsize=(6.,4.5))   
        else:
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6.,6.),sharex=True,
                                          gridspec_kw={'height_ratios': [2, 1]})

        if plot_data is True:
            ax1.errorbar(self.freqs,data,yerr=error,
                         linestyle='',marker='o')       
       
        ax1.plot(self.freqs,model,lw=3,zorder=3)

        if plot_components is True:
            #we need to allocate a ModelResult object in order to retrieve the components
            comps = LM_result(model=self.model,params=self.model_params)
            comps = comps.eval_components(freq=self.freqs)
            for key in comps.keys():
                if units == "power":                
                    ax1.plot(self.freqs,comps[key],label=key,lw=2)
                elif units == "fpower":
                    ax1.plot(self.freqs,comps[key]*self.freqs,label=key,lw=2)
            ax1.legend(loc='best')
        
        ax1.set_xscale("log",base=10)
        ax1.set_yscale("log",base=10)
        if plot_data is False:
            ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel(ylabel)
        ax1.set_ylim([0.3*np.min(model),3.*np.max(model)])

        if plot_data is True:
            ax2.errorbar(self.freqs,model_res,yerr=res_errors,
                         linestyle='',marker='o')
            if residuals == "chisq":
                ax2.plot(self.freqs,np.zeros(len(self.freqs)),
                         ls=":",lw=2,color='black')
            elif residuals == "ratio":
                ax2.plot(self.freqs,np.ones(len(self.freqs)),
                         ls=":",lw=2,color='black')                
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel(reslabel)

        plt.tight_layout()      
        
        if return_plot is True:
            return fig 
        else:
            return   
