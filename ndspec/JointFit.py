import warnings

import numpy as np
import lmfit
from lmfit import fit_report, minimize
from lmfit import Parameters
from lmfit.printfuncs import gformat
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams.update({'font.size': 17})
import matplotlib.patheffects as path_effects

from scipy.interpolate import interp1d, RegularGridInterpolator

from lmfit import Model as LM_Model
from lmfit import Parameters as LM_Parameters

from .SimpleFit import SimpleFit, EnergyDependentFit, FrequencyDependentFit
from .FitCrossSpectrum import FitCrossSpectrum
from .FitTimeAvgSpectrum import FitTimeAvgSpectrum
from .Utils import get_plot_info, darken_colour
from .Likelihoods import cstat, chisq, ratio

class JointFit():
    """
    Generic joint fitting class. Use this class if you have multiple datasets
    that you want to fit jointly in some way. 

    Users can add other Fit...Objs from ndspec to an instance of this class, and
    that instance this will handle evaluating the model, sharing parameters 
    and/or whole models between Fit....Objs objects, and perform inference
    and/or optimization. There is no restriction on the type or number of 
    Fit...Objs that can be added. 
    
    Attributes:
    ------------
    joint : dict{Fit... objects and/or list(Fit... objects)}
        Dictionary containing named Fit... objects to be joint-fitted. 
        
    joint_params: dict{lists(str)}
        Dictionary containing the names of model parameters for each distinct
        fit object.
        
    fit_result: lmfit.MinimizeResult
        A lmfit MinimizeResult, which stores the result (including best-fitting 
        parameter values, fit statistics etc) of a fit after it has been run. 

    model_params: lmfit.Parameters
        The parameters of all the fitter objects combined, which are used during 
        the joint fit.
        
    energy_grid: dict{np.arrays}
        An optional dictionary containing  the details of a user-defined grid of 
        energy bins and bounds over which to evaluate a model. Usable only with 
        time-averaged spectra using a single shared model. Using a shared grid 
        can improve performance in some cases. 
        
    shared_energy_grid: bool 
        A boolean that tracks whether the joint fitter object has a shared 
        energy grid for evaluating the time-averaged spectrum model or not. 
        
    shared_model: lmfit.CompositeModel
        The model shared among all FitTimeAvgSpectrum objects to be evaluated on 
        the same energy grid. 
        
    spec_renorm_model: lmfit.Model 
        An optional lmfit model object used to include a constant component in 
        a set of chosen fitters for time-averaged spectra. Used to account for 
        instrument flux cross-calibration.
        
    shared_keys: list 
        A list containing the keys (names) of the parameters that are identical 
        and therefore shared between the loaded fitter objects.
        
    renorm_names: list[str]
        A list with the names of the time-averaged fitter objects to which the 
        cross-calibration constants should be applied. 
    """
    
    def __init__(self):
        self.joint = {}
        self.joint_params = {}
        self.fit_result = None
        self.model_params = None
        self.energy_grid = None
        self.shared_energy_grid = False
        self.shared_model = None
        self.shared_keys = []
        self.spec_renorm_model = None
        self.renorm_spectra = False 
        self.renorm_names = None

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
        
        if self.shared_keys != []:
            print(f"""The following parameters have been shared: {self.shared_keys}""")
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
                if key in self.shared_keys:
                    pass
                elif key in self.joint_params[joint_obs]:
                    self.shared_keys.append(key)
                else:
                    self.model_params.add_many(fitobj.model_params[key])
                params.append(key)
        self.joint_params[name] = params
        return 
     
    def eval_model(self,params=None,names=None,flatten=True):
        """
        This method is used to evaluate and return the model values of models 
        stored in the fitter objects that make up the joint fit.
        
        Parameters:
        ------------
        params: lmfit.Parameters, default None
            The parameters to use in evaluating the model/models. If none are 
            provided, the model_params attribute is used.

        names: list(str), default None
            Names of the fitters that should evalualate their models. Defaults 
            to evaluating the models of all fitters stored in the joint fit.
            
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
        elif type(names) == str:
            names = [names]
            
        if params == None:
            params = self.model_params
        #creates structure to return model results
        model_hierarchy = {}
        
        #check if we shared the energy grid. If yes, evalute the model on the 
        #shared grid here. We then evaluate the model, with no folding or 
        #masking of ignored bins
        if self.shared_energy_grid is True:
            joint_eval = self.shared_model.eval(params,
                                                energ=self.energy_grid["energ"],
                                                ear=self.energy_grid["ear"],
                                                fold=False,
                                                mask=False)
            model_interp = interp1d(self.energy_grid["energ"],joint_eval,
                                    fill_value='extrapolate',kind='linear')
    
        for name in names:
            if name not in self.joint.keys():
                raise AttributeError(f"{name} is not among the stored fitters")
            #retrieves model or models based on dictionary name
            fitobj = self.joint[name]     
            #if we are evaluating a fit on a shared grid, interpolate, fold and mask
            #otherwise evaluate normally      
            if (self.shared_energy_grid is True and type(fitobj)==FitTimeAvgSpectrum):
                model_results = model_interp(fitobj.energs)*fitobj.energ_bounds
                model_results = fitobj.response.convolve_response(model_results) 
                model_results = np.extract(fitobj.ebounds_mask,model_results) 
            else:
                model_results = fitobj.eval_model(params)
            #if the fitter is in the list of spectra to re-normalize for 
            #cross calibration, do so now
            if self.renorm_spectra is True:
                if name in self.renorm_names:
                    par_key = 'renorm_'+str(name)
                    renorm_pars = LM_Parameters()
                    renorm_pars.add('renorm',value=params[par_key].value,
                                    min=params[par_key].min,max=params[par_key].max,
                                    vary=params[par_key].vary)
                    model_results = self.spec_renorm_model.eval(renorm_pars,array=model_results)      
            model_hierarchy[name] = model_results
        
        if flatten == False:
            return model_hierarchy
        else:
            model = np.array([])
            for key in model_hierarchy:
                model = np.concatenate([model,model_hierarchy[key]])
            return model

    def get_residuals(self,params=None,names=None,flatten=True):
        """
        This methods return the residuals based on the statistic, model and 
        data defined in each of the loaded fitter objects. 
        
        Parameters:
        -----------
        params: lmfit.Parameters or list[lmfit.Parameters]
            The parameter values to use in evaluating the models. Defaults to 
            those stored in the JointFit instance. 
        
        names: list(str), default None
            names of the models that should be evalualated. Defaults to
            evaluating all models.
            
        flatten: bool, default True 
            A boolean to switch between returning model evaluations as a 
            dictionary or numpy array (see below).             
            
        Returns:
        --------
        residuals: np.array(float) or dict{np.array(float)}
            Either an array of the same size as the total number of data points,
            or a dictionary with keys named after each loaded SimpleFit instance,
            containing the model residuals in each channel.         
        """

        if self.joint == {}:
            raise AttributeError("No loaded observations or models")
            
        if names == None: #retrieves all models
            names = list(self.joint.keys())
        elif type(names) == str:
            names = [names]

        residual_hierarchy = {}            
        if type(names) != list:
            raise TypeError("Inputted names are not valid type")
        else:
            model_dict = self.eval_model(params,names,flatten=False)
            residuals = np.array([])
            for name in names:
                model = model_dict[name]          
                likelihood = self.joint[name].likelihood    
                resids, _ = self.joint[name].get_residuals(likelihood,model=model)
                residual_hierarchy[name] = resids
                
        if flatten == False:
            return residual_hierarchy
        else:
            residuals = np.array([])
            for key in residual_hierarchy:
                residuals = np.concatenate([residuals,residual_hierarchy[key]])
            return residuals
 
    def _minimizer(self,params,names = None):
        """
        This method is used (almost) exclusively when running a minimization 
        algorithm. It evaluates the models for an input set of parameters, and 
        then returns the residuals in units of contribution to the total chi 
        squared statistic.
        
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
        
        '''if self.joint == {}:
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
                likelihood = self.joint[name].likelihood    
                resids = self.joint[name].get_residuals(likelihood,model=model)
                residuals = np.concatenate([residuals,np.asarray(resids).flatten()])
            residuals = np.asarray(residuals).flatten()
        '''
        
        residuals = self.get_residuals(params,names,flatten=True)
            
        return residuals
    
    def fit_data(self,algorithm='leastsq',names=None,report_result=True):
        """
        This method attempts to minimize the residuals of the model with respect 
        to the data defined by the user. The fit either starts from the set of 
        parameters defined with .set_params(), or from the parameters set in the 
        individual fitter obejcts loaded. Once the algorithm has completed its
        run, it optionally prints to terminal the best-fitting parameters, fit 
        statistics, and parameter values. 
        
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
        fit_params = self.fit_result.params
        self.set_params(fit_params)

        self.print_fit_report()
        return

    def set_energy_grid(self,grid_bounds):
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
        raise AttributeError("THIS METHOD CURRENTLY DOES NOT WORK, DO NOT USE")
        
        names = list(self.joint.keys())
        model_list = []
        for name in names:
            if type(self.joint[name]) == FitTimeAvgSpectrum:
                model_list = np.append(model_list,self.joint[name].model)
                if grid_bounds[0] > self.joint[name].energs[0]:
                    raise ValueError(f"Custom grid bound above the minimum energy of {name}")
                if grid_bounds[-1] < self.joint[name].energs[-1]:
                    raise ValueError(f"Custom grid bound below the maximum energy of {name}")    
        
        #check that all the models in the time averaged spectrum fitters are the 
        #same, and assign the model ot be used if that is true
        
        first_model = model_list[0]
        if all(model == first_model for model in model_list):
            self.shared_model = first_model
        else:
            raise AttributeError("Not all models in the fitters are identical!")
        
        self.energy_grid = dict(ear=grid_bounds,
                                energ=0.5*(grid_bounds[1:]+grid_bounds[:-1]),
                                energ_bounds=grid_bounds[1:]-grid_bounds[:-1])
        self.shared_energy_grid = True
        return

    def renorm_timeavg(self,switch,names=None):
        """
        Setter method to enable the fitter objects passed with the "names" 
        attribute to include an additional constant component, in other to 
        account e.g. for the cross-calibration uncertainty between instruments.
        Only applicable to time-averaged spectra.      
        
        Parameters:
        -----------
        switch: bool 
            A boolean to track whether the spectra renormalization is enabled or 
            not. If it is, the method modifies the defined model and its parameters 
            automatically. 
            
        name: list(str) or str, default None
            A list of strings with the name of the fitter objects to which users 
            wish to apply a cross calibration constant. By default this is None 
            and all the spectra receive the constant.
        """
        self.renorm_spectra = switch
        if self.renorm_spectra is True:
            #retrieve all time averaged spectra loaded
            if names is None:
                searched_names = self.joint.keys()
            else:
                searched_names = names 
            self.renorm_names = []
            for name in searched_names:
                if type(self.joint[name]) == FitTimeAvgSpectrum:
                    self.renorm_names = np.append(self.renorm_names,name)        
            #add constant models to the selected fitters 
            self.spec_renorm_model = LM_Model(self._renorm_spectrum)
            renorm_pars = LM_Parameters()
            for name in self.renorm_names:       
                renorm_pars.add('renorm_'+str(name),
                                value=1,min=0.5,max=1.5,vary=True)
            self.model_params = self.model_params + renorm_pars          
        return

    def _renorm_spectrum(self,array,renorm):
        """
        This method contains a model function to renormalize an input array, 
        assumed to be a model time averaged spectrum, by some constant number. 
        This is to be used as a cross calibration constant for different 
        instruments. 
       
        Parameters:
        -----------
        array: np.array(float)
            The array of energy depdent time averaged spectrum to be 
            renormalized 
            
        renorm: float 
            The renormalization factor by which to multiply the model time 
            averaged spectrum
            
        Returns:
        --------
        array*renorm 
            The new, renormalized model array. 
        """
    
        return renorm*array

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
        #updates the individually linked parameters rather than overwriting them.
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
        
    def print_fit_report(self):
        """
        This method prints the current fit result.
        """
        
        result = self.fit_result
        print("-----------------------")
        print("[[Fit Statistics]]")
        print(f"    # fitting method   = {result.method}")
        print(f"    # function evals   = {result.nfev}")
        var = 0
        for key in self.model_params:
            if self.model_params[key].vary is True:
                var += 1
        print(f"    # variables        = {var}")
        
        total_fit_stat = 0
        total_n_points = 0
        
        print("-----------------------")        
        for name in self.joint.keys():
            print(f"    Dataset: {name}")    
            model =  self.joint[name].eval_model(params=self.model_params)  
            if (self.renorm_spectra is True and type(self.joint[name]) == FitTimeAvgSpectrum):
                model = model*self.model_params['renorm_'+str(name)].value  
            res, _ = self.joint[name].get_residuals(self.joint[name].likelihood,model=model) 
            if self.joint[name].likelihood == "chisq":
                fit_statistic = np.sum(res**2)
            else:
                fit_statistic = np.sum(res)
            dof = len(self.joint[name].data) - var
            reduced_stat = fit_statistic/dof    
            total_fit_stat = total_fit_stat + fit_statistic
            total_n_points = total_n_points + len(self.joint[name].data)
            print(f"    fit statistic      = {fit_statistic}")
            print(f"    reduced statistic  = {reduced_stat}")
            print(f"    # data points      = {len(self.joint[name].data)}")
        reduced_stat = total_fit_stat/(total_n_points-var) 
        print("-----------------------")
        print(f"    total fit stat         = {total_fit_stat}")
        print(f"    total reduced stat     = {reduced_stat}")        
        print(f"    total data points      = {total_n_points}")
        print("-----------------------")
        
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
                inval = f'(init = {par.init_value:.7g})'
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
    
    def __getitem__(self, key):
        """
        This method returns a particular fit object stored within
        the class.
        """
        return self.joint[key]

    def joint_plot(self,units,residuals="chisq",plot_bkg=False,xrange=None,yrange=None,return_plot=False,names=None):
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
            
        residuals: str, default="chisq"
            The units to use for the residuals. If residuals="chisq", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model. For cross spectra this 
            key word is ignored and only delta chi residuals can be shown.
            
        plot_bkg; str, default=False:
            A boolean to choose whether you want to plot the background
            
        xrange, yrange: (float, float) 
            The limits of the plot on the x and y axis 

        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.     
            
        names: list(str)
            A list of the fitters to be included in the joint plot. By default, 
            all the loaded fitters are included.        
        """
       
        if names is None:
            names = list(self.joint.keys())
        elif type(names) is str: 
            names = [names]

        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6.,6.),sharex=True,
                                      gridspec_kw={'height_ratios': [2, 1]})

        if xrange is not None:
            ax1.set_xlim(xrange)
            ax2.set_xlim(xrange)
        
        if yrange is not None:
            ax1.set_ylim(yrange)
        
        i=0
        max_xrange = 0.
        min_xrange = 1e30
        
        for key in names:
            if type(self.joint[key]) == FitCrossSpectrum:
                raise TypeError("You can not display fits to 1d and 2d data on the same plot!")
            else:
                plot = self.joint[key].plot_model(residuals=residuals,
                                                  units=units,
                                                  plot_bkg=plot_bkg,
                                                  return_plot=True,
                                                  params=self.model_params)                       
            
            plot_data = get_plot_info(plot,residuals=residuals)
            
            #sort the plot bounds 
            if plot_data["x_points"][0] < min_xrange:
                min_xrange = plot_data["x_points"][0]   
            if plot_data["x_points"][-1] > max_xrange: 
                max_xrange = plot_data["x_points"][-1]  
                    
            col="C"+str(i)
            i = i+1
            ax1.errorbar(plot_data["x_points"], plot_data["y_points"], 
                         xerr=plot_data["x_bars"], yerr=plot_data["y_bars"], 
                         fmt='o',alpha=0.35, color=col)
            
            model = plot_data["model_vals"] 
                
            #renormalize if necessary 
            if (self.renorm_spectra is True):
                model = model*self.model_params['renorm_'+str(key)].value            
            
            ax1.plot(plot_data["x_points"], model,
                     linestyle=plot_data["linestyle"][0],
                     linewidth= plot_data["linewidth"][0],
                     color=darken_colour(col),zorder=10)
            
            ax1.set_xscale("log",base=10)
            ax1.set_yscale("log",base=10)    
            
            y_res = plot_data["resid"]
            y_reserr = plot_data["reserr"]
            
            #if the spectra were renormalized, we have to over-write the residuals
            if (self.renorm_spectra is True and residuals=="chisq"):
                model_folded = self.eval_model(names=key,flatten=True)
                y_res = chisq(self.joint[key].data,self.joint[key].data_err,
                              model_folded,
                              noise=self.joint[key].noise,
                              noise_err=self.joint[key].noise_err,
                              )
                y_reserr = plot_data["reserr"]
            elif (self.renorm_spectra is True and residuals=="ratio"):
                model_folded = self.eval_model(names=key,flatten=True)
                res, bars = ratio(self.joint[key].data,
                                  self.joint[key].data_err,
                                  model_folded,
                                  noise=self.joint[key].noise,
                                  noise_err=self.joint[key].noise_err,
                                  summed=False,
                                  bars=True)
                y_res = res 
                y_reserr = bars
                                    
            ax2.errorbar(plot_data["x_points"], y_res, 
                         xerr=plot_data["x_bars"], yerr=y_reserr, 
                         fmt='o',alpha=0.5, color=col)
        
        x_points = np.linspace(min_xrange,max_xrange,100)

        if residuals == "chisq":
            ax2.plot(x_points,np.zeros(len(x_points)),
                     ls=":",lw=2,color='black',zorder=10)
        elif residuals == "ratio":
            ax2.plot(x_points,np.ones(len(x_points)),
                     ls=":",lw=2,color='black',zorder=10)
        ax2.set_xscale("log",base=10)
        
        ax1.set_xlabel(plot_data["ax1_data"].get_xlabel())
        ax1.set_ylabel(plot_data["ax1_data"].get_ylabel())
        ax2.set_xlabel(plot_data["ax2_data"].get_xlabel())
        ax2.set_ylabel(plot_data["ax2_data"].get_ylabel())
        
        ax1.set_xlim([0.95*min_xrange,1.05*max_xrange])
        ax2.set_xlim([0.95*min_xrange,1.05*max_xrange])

        fig.tight_layout()

        if return_plot is True:
            return fig 
        else:
            return   
        
    def all_plots(self,units,residuals="chisq",plot_bkg=None,return_plot=False):
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
            
        residuals: str, default="chisq"
            The units to use for the residuals. If residuals="chisq", the plot 
            shows the residuals in units of data-model/error; if residuals="ratio",
            the plot instead uses units of data/model. For cross spectra this 
            key word is ignored and only delta chi residuals can be shown.
            
        plot_bkg; str, default=False:
            A boolean to choose whether you want to plot the background

        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.           
        """
        
        if self.renorm_spectra is True:
            warnings.warn("Fit cross-calibration constants enabled! Models might be off!",
                           UserWarning)   
        
        if return_plot is not False:
            figs = []
        
        for key in self.joint:
            if type(self.joint[key]) == FitCrossSpectrum:
                plot = self.joint[key].plot_model_1d(return_plot=True)
            else:
                plot = self.joint[key].plot_model(residuals=residuals,
                                                  units=units,
                                                  plot_bkg=plot_bkg,
                                                  return_plot=True)
            plot.axes[0].set_title(str(key))
            plot.tight_layout()
            if return_plot is True:
                figs = np.append(figs,plot)
        
        if return_plot is True:
            return figs 
        else:
            return            
