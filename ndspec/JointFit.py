import numpy as np
import lmfit
from lmfit import fit_report, minimize
from lmfit import Parameters
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams.update({'font.size': 17})

from scipy.interpolate import interp1d, RegularGridInterpolator

from .SimpleFit import SimpleFit, EnergyDependentFit, FrequencyDependentFit
from .FitCrossSpectrum import FitCrossSpectrum
from .FitTimeAvgSpectrum import FitTimeAvgSpectrum
from .FitPowerSpectrum import FitPowerSpectrum

class JointFit():
    """
    Generic joint inference class. Use this class if you have multiple 
    datasets that you want to fit jointly in some way. 

    Users can add other Fit...Objs from ndspec to an instance of this class, and
    that instance this will handle evaluating the model, sharing parameters 
    and/or whole models between Fit....Objs objects, and perform inference
    and/or optimization. There is no restriction on the type or number of 
    Fit...Objs that can be added. 
    
    Note that JointFit does not perform extra performance enhancements to make
    evaluations run faster, so optimization and joint inference on many 
    parameters is still subject to the usual computational problems that come
    with such scenario. 
    
    Attributes:
    ------------
    joint : dict{Fit... objects and/or list(Fit... objects)}
        Dictionary containing named Fit... objects to be joint fitted. By
        default, datasets that share parameters completely (simultaneous 
        observations, or different data products of observations) are packaged 
        together in lists.
        
    joint_params: dict{lists(str)}
        Dictionary containing the names of model parameters for each distinct
        fit object.
        
    fit_result: lmfit.MinimizeResult
        A lmfit MinimizeResult, which stores the result (including best-fitting 
        parameter values, fit statistics etc) of a fit after it has been run. 

    model_params: lmfit.Parameters
        The parameter values from which to start evalauting the model during
        the fit.  
    """
    
    def __init__(self):
        self.joint = {}
        self.joint_params = {}
        self.fit_result = None
        self.model_params = None
        self.energy_grid = None

    def add_fitobj(self,fitobj,name):
        """
        Adds one or more fitters to the joint fit. 

        Parameters
        ----------
        fitobj : Fit... object or list of Fit... objects
            the Fit... object(s) being included in the joint fit.
        
        name: str
            name(s) to be assigned to the fitter objects loaded. These will be 
            used as keys of a dictionary to retrieve the individual fitter 
            objects and their methods. 
        """
        if type(fitobj) == list:
            for obj in fitobj:
                if issubclass(type(obj),SimpleFit):
                    pass
                else:
                    raise TypeError("Invalid object passed")
        else:
            if issubclass(type(fitobj),SimpleFit):
                pass
            else:
                raise TypeError("Invalid object passed")
            
        if type(fitobj) == list: 
            #multiple observations loaded at the same time 
            for counter, fitter_name  in enumerate(name):
                self._add_single_fitobj(fitobj[counter],fitter_name)  
        else: 
            #single observation loaded each time 
            self._add_single_fitobj(fitobj,name)
        return 
    
    def _add_single_fitobj(self,fitobj,name):
        """
        Adds a single fit object to the JointFit instance, and looks through the 
        model parameters stored in the fit for all the parameters that will be 
        used in the joint fit. 

        Note that if two fitter objects have models with parameters that happen 
        to have identical names (e.g. temperature), then the joint fitter will 
        assumme these two parameters are meant to be identical, and therefore 
        the joint fit will forcefully tie them. Use different names for your 
        individual fitter parameters (e.g. temperature_high and temperature_low)
        to avoid this behavior.        

        Parameters
        ----------
        fitobj : Fit... object 
            the individual Fit... object being included.
        
        name: str
            name to be assigned to the fitter objects fitobj.     
        """
        
        #we are passing a single fit object that may or may not share 
        #models/parameters with the other objects 
        self.joint[name] = fitobj
        #if first added object, add model params
        if self.model_params == None:
                self.model_params = Parameters()
                for par in fitobj.model_params:
                    self.model_params.add_many(fitobj.model_params[par])
        #pulls parameters names and saves to dictionary for model
        #evaluation later
        params = []
        for key in fitobj.model_params.valuesdict().keys():
            for joint_obs in self.joint_params:
                if key in self.joint_params[joint_obs]:
                    print(f"""
                          Caution: {key} is already a model parameter.
                          Do you intend for these parameters to be linked?
                          If not, give it a different name to differentiate
                          between multiple instances of the same type for
                          different models.
                          """)
                else:
                    self.model_params.add_many(fitobj.model_params[key])
            params.append(key)
        self.joint_params[name] = params
        return 
     
    def eval_model(self,params=None,names=None,flatten=True):
        """
        This method is used to evaluate and return the model values of models 
        in the hierarchy.
        
        Parameters:
        ------------
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model/models. If 
            none are provided, the model_params attribute is used.

        names: list(str), default None
            Names of the fitters that should evalualate their models. Defaults 
            to evaluating the models of all fitters.
            
        flatten: bool, default True 
            A boolean to switch between returning model evaluations as a 
            dictionary or numpy array (see below). 
        Returns:
        --------
        model_hierarchy: either dict(np.array(float)) or np.array(float)
            Models are evaluated and returned either as a dictionary, with keys 
            defined by the fitter names, or by flattened numpy arrays. The 
            former allows easy access to the evaluated model for each fitter, 
            the latter is necessary for lmfit optimzers and/or likelihood \
            calculations. 
        
        """
        
        if names == None: #retrieves all models
            names = self.joint.keys()
        if params == None:
            params = self.model_params
        #creates structure to return model results
        model_hierarchy = {}

        #when we use a custom grid, need to check that we're grabbing the right 
        #parameters - ugh. We can't grab the fit parameters from each fitter either.        
        for name in names:
            if name not in self.joint.keys():
                raise AttributeError(f"{name} is not among the stored fitters")
            #retrieves model or models based on dictionary name
            fitobjs = self.joint[name] 
            #tbd: grid stuff here 
            if type(fitobjs) == list: 
                model_results = []
                for fit_obj in fitobjs:
                    model_results.append(fit_obj.eval_model(params))
            else:
                model_results = fitobjs.eval_model(params)
            model_hierarchy[name] = model_results
        
        if flatten == False:
            return model_hierarchy
        else:
            model = np.array([])
            for key in model_hierarchy:
                model = np.concatenate([model,model_hierarchy[key]])
            return model
 
    def _minimizer(self,params,names = None):
        """
        This method is used exclusively when running a minimization algorithm.
        It evaluates the models for an input set of parameters, and then returns 
        the residuals in units of contribution to the total chi squared 
        statistic.
        
        Parameters:
        -----------                         
        params: lmfit.Parameters or list[lmfit.Parameters]
            The parameter values to use in evaluating the model. These will vary 
            as the fit runs.
        
        names: list(str), default None
            names of the models that should be evalualated. Defaults to
            evaluating all models.
            
        Returns:
        --------
        residuals: np.array(float)
            An array of the same size as the data, containing the model 
            residuals in each bin.            
        """
        if self.joint == {}:
            raise AttributeError("No loaded observations or models")
            
        if names == None: #retrieves all models
            names = list(self.joint.keys())
        elif type(names) == str:
            names = [names]
            
        if type(names) != list:
            raise TypeError("Inputted names are not valid type")
        else:
            model_dict = self.eval_model(params,names,flatten=False)
            residuals = np.array([])
            for name in names:
                model = model_dict[name]
                if self.joint[name].noise is None:
                    resids = (self.joint[name].data-model)/self.joint[name].data_err
                else:
                    resids = (self.joint[name].data-self.joint[name].noise-model)/self.joint[name].data_err    
                residuals = np.concatenate([residuals,np.asarray(resids).flatten()])
            residuals = np.asarray(residuals).flatten()
        return residuals
    
    def fit_data(self,algorithm='leastsq',names=None):
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
        
        names: list(str), default None
            names of the fit objects that should be part of the fit. Defaults to
            using all the fitters stored in the JointFit object instance.
        """
        if names == None:
            names = list(self.joint.keys())
        
        self.fit_result = minimize(self._minimizer,self.model_params,
                                   method=algorithm,args=[names])
        print(fit_report(self.fit_result,show_correl=False))
        fit_params = self.fit_result.params
        self.set_params(fit_params)
        return

    def share_energy_grid(self,grid_bounds):
        """
        This method defines a custom energy grid over which to evaluate all 
        loaded time-averaged spectra. The new grid MUST cover a wider energy 
        range than all the ones in the time-averaged spectra loaded in the joint 
        fitter instance. 
        
        Parameters:
        -----------
        grid_bounds: np.array(float)
            An array of energy bounds, starting from the bottom edge of the 
            first bin and finishing up to the top edge of the last bin in the 
            new grid.
        """
        names = list(self.joint.keys())
        for name in names:
            if getattr(self.joint[name], '__module__', None) == "ndspec.FitTimeAvgSpectrum":
                if grid_bounds[0] > self.joint[name].energs[0]:
                    raise ValueError(f"Custom grid bound above the minimum energy of {name}")
                if grid_bounds[-1] < self.joint[name].energs[-1]:
                    raise ValueError(f"Custom grid bound below the maximum energy of {name}")    
        
        self.energy_grid = dict(ear=grid_bounds,
                                energ=0.5*(grid_bounds[1:]+grid_bounds[:-1]),
                                energ_bounds=grid_bounds.ear[1:]-grid_bounds.ear[:-1])
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
        if type(params) != lmfit.Parameters:  
            raise AttributeError("The parameters input must be an LMFit Parameters object")
        #updates the individually linked parameters rather than overwrites them.
        for par in self.model_params:
            self.model_params[par] = params[par]
            for key in self.joint:
                if type(self.joint[key]) == list:
                    for m in self.joint[key]:
                        if par in list(m.model_params.keys()):
                            m.model_params[par] = params[par]
                else:
                    if par in list(self.joint[key].model_params.keys()):
                        self.joint[key].model_params[par] = params[par]
        return 

    def print_models(self,names=None):
        """
        Prints the models contained within the joint fit. Defaults to printing
        all models, but users can filter models using the names parameter.

        Parameters
        ----------
        names : str or list(str), optional
            names of the models to be printed. The default is to print all 
            models.
        """
        if names == None:
            names = list(self.joint.keys())
        
        if type(names) == list:
            for name in names:
                print(f"{name}: \n")
                print("-----------------------")
                self.joint[name].print_model()
                print("-----------------------")
        else:
            print(f"{names}: \n")
            print("-----------------------")
            self.joint[names].print_model()
            print("-----------------------")
        
    def print_fit_results(self):
        """
        This method prints the current fit results.
        """
        if self.fit_result != None:
            print(fit_report(self.fit_result,show_correl=False))
        else:
            print("No current fit result.")
    
    def __getitem__(self, key):
        """
        This method returns a particular fit object stored within
        the class.
        """
        return self.joint[key]

    #these methods are banished to the shadow realm down here while I figure out what to do 
    #with the multiple loading/shared grid etc 
    def _model_decompose(self,model):
        """
        Decomposes lmfit composite models into their base Models.
        Mainly useful for retrieving parameter names from complex
        composite models, and is only for internal model use.

        Parameters
        ----------
        model: lmfit.compositemodel
            composite model to be decomposed

        Returns
        -------
        models: list(lmfit.model)
            list of component lmfit.model objects.
        """
        #catches and returns inputted lmfit.models as list
        if type(model) == lmfit.Model:
            return [model] 
        
        if type(model) != lmfit.CompositeModel:
            raise TypeError("Not a lmfit composite model")
        models = []
        
        if type(model.left) == lmfit.Model:
            models.append(model.left)
        else:
            models.extend(self._model_decompose(model.left))
        
        if type(model.right) ==  lmfit.Model:
            models.append(model.right)
        else:
            models.extend(self._model_decompose(model.right))
        
        return models

    def _share_params(self,first_fitobj,second_fitobj,param_names=None):
        """
        Shares parameters between models and links the parameters of individual 
        models that compose the joint fit to the parameters inferred in the 
        optimization process.

        Parameters
        ----------
        first_fitobj : Fit... object 
            primary fit object that the secondary fit object is linked to.
        second_fitobj : Fit... object 
            secondary fit object that is linked to the primary.
        param_names : str or list(str), optional
            Names of parameters (with the same name) to share between models. The default 
            is to share all parameters together

        """
        #checks that both models are correctly specified
        if (((type(first_fitobj.model) != lmfit.CompositeModel)&(type(first_fitobj.model) != lmfit.Model))|
           ((type(second_fitobj.model) != lmfit.CompositeModel)&((type(second_fitobj.model) != lmfit.Model)))):  
            raise AttributeError("The model input must be an LMFit Model or CompositeModel object")
        
        #adds all base models into list (decomposes CompositeModels into Models)
        models = []
        #adds all models from first fit object as a list of models
        models.append(self._model_decompose(first_fitobj.model))
        #adds all models from second fit object as a list of models
        models.append(self._model_decompose(second_fitobj.model))

        if param_names == None: #defaults to all parameters (models are identical)
            second_fitobj.model_params = first_fitobj.model_params
        elif type(param_names) == list: #correct format
            pass
        elif type(param_names) == str: #translates to correct format
            param_names = [param_names]
        else:
            raise TypeError("Input parameter name or list of parameter names")

        #first check that all specified parameters are present in both fit objects
        for fit_obj in models:
            check = set(param_names)
            for m in fit_obj: #iterates through all basic models
                check = check - set(m.param_names)
            if check == set(): #if check is an empty set, all parameter names are present in object
                continue
            else:
                #if parameters are not shared, soft error
                print("Not all parameters inputted are in models")
                return
        
        for name in param_names:
            #find parameter name in first fit objects models
            second_fitobj.model_params[name] = first_fitobj.model_params[name]

    def joint_plot(self,units,plot_bkg=False,xrange=None,yrange=None,return_plot=False):
        """
        This method loops over all stored fitter objects and plots the data, 
        model (given the parameters stored), and residuals for all the fits 
        together. Note that this is useful only if the data you are trying to 
        plot is of the same time (e.g. all time-averaged spectra).
        
        Parameters:
        -----------
        units: str
            The units to use for the y axis. For more info, see the documentation 
            of the individual fitter classes. 
            
        plot_bkg; str, default=False:
            A boolean to choose whether you want to plot the background
            
        xrange, yrange: (float, float) 
            The limits of the plot on the x and y axis 

        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.            
        """

        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6.,6.),sharex=True,gridspec_kw={'height_ratios': [2., 1]})

        if xrange is not None:
            ax1.set_xlim(xrange)
            ax2.set_xlim(xrange)
        
        if yrange is not None:
            ax1.set_ylim(yrange)
        
        i=0
        for key in self.joint:
            if getattr(self, '__module__', None) == "ndspec.FitCrossSpectrum":
                raise TypeError("You can not display fits to 1d and 2d data on the same plot!")
            else:
                plot = self.joint[key].plot_model(units=units,return_plot=True,plot_bkg=plot_bkg)
            plot.axes[0].set_title(str(key))
            plot.tight_layout()
            
            ax1_data, ax2_data = plot.axes
            
            # Extract data points and errors from Collection 0 (horizontal errorbars and y data)
            segments_x = ax1_data.collections[0].get_segments()
            x_midpoints = np.mean([[seg[0, 0], seg[1, 0]] for seg in segments_x], axis=1)
            y_data = np.array([seg[0, 1] for seg in segments_x])  
            x_errors = np.abs(np.array([[seg[0, 0], seg[1, 0]] for seg in segments_x]).T - x_midpoints)
        
            # Extract data points and errors from Collection 1 (vertical errorbars and x data)
            segments_y = ax1_data.collections[1].get_segments()
            y_midpoints = np.mean([[seg[0, 1], seg[1, 1]] for seg in segments_y], axis=1)
            x_data = np.array([seg[0, 0] for seg in segments_y])  
            y_errors = np.abs(np.array([[seg[0, 1], seg[1, 1]] for seg in segments_y]).T - y_midpoints)
        
            col="C"+str(i)
            i = i+1
            ax1.errorbar(x_data, y_data, xerr=x_errors, yerr=y_errors, fmt='o',alpha=0.1, color=col)
            lines = ax1_data.get_lines()
            line = lines[1]
            ax1.plot(line.get_xdata(), line.get_ydata(),
                     linestyle=line.get_linestyle(),
                     linewidth=line.get_linewidth(),
                     color=col,
                     zorder=10)
            
            ax1.set_xscale("log",base=10)
            ax1.set_yscale("log",base=10)    
        
            #now extract the residuals as above
            segments_y = ax2_data.collections[0].get_segments()
            residuals = np.mean([[seg[0, 1], seg[1, 1]] for seg in segments_y], axis=1)
            x_data = np.array([seg[0, 0] for seg in segments_y])  
            y_errors = np.abs(np.array([[seg[0, 1], seg[1, 1]] for seg in segments_y]).T - residuals)
            
            ax2.errorbar(x_data, residuals, xerr=x_errors, yerr=y_errors, fmt='o',alpha=0.35, color=col)
        
        ax2.plot(x_data,np.zeros(len(x_data)),ls=":",lw=2,color='black',zorder=10)
        ax2.set_xscale("log",base=10)
        
        ax1.set_xlabel(ax1_data.get_xlabel())
        ax1.set_ylabel(ax1_data.get_ylabel())
        ax2.set_xlabel(ax2_data.get_xlabel())
        ax2.set_ylabel(ax2_data.get_ylabel())

        fig.tight_layout()

        if return_plot is True:
            return fig 
        else:
            return   
        
    def all_plots(self,units,plot_bkg=None,return_plot=False):
        """
        This method loops over all stored fitter objects and plots the data, 
        model (given the parameters stored), and residuals for all the fits 
        separately. For two-dimensional fits (like a cross spectrum), the method 
        still plots one-dimensional plots. This method is meant for quick-look 
        analysis of a joint fit.
        
        Parameters:
        -----------
        units: str
            The units to use for the y axis. For more info, see the documentation 
            of the individual fitter classes. Has no impact if the fitter being 
            looked over is a cross spectrum. 
            
        plot_bkg; str, default=False:
            A boolean to choose whether you want to plot the background

        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.           
        """   
        
        if return_plot is not False:
            figs = []
        
        for key in self.joint:
            if getattr(self, '__module__', None) == "ndspec.FitCrossSpectrum":
                plot = self.joint[key].plot_model_1d(return_plot=True)
            else:
                plot = self.joint[key].plot_model(units=units,return_plot=True,plot_bkg=plot_bkg)
            plot.axes[0].set_title(str(key))
            plot.tight_layout()
            if return_plot is True:
                figs = np.append(figs,plot)
        
        if return_plot is True:
            return figs 
        else:
            return         
