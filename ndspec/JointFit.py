import warnings

import numpy as np
import lmfit
from lmfit import fit_report, minimize
from lmfit import Parameters
from lmfit.printfuncs import gformat
from scipy.interpolate import interp1d

from scipy.interpolate import interp1d, RegularGridInterpolator

from .SimpleFit import SimpleFit, EnergyDependentFit, FrequencyDependentFit
from .FitCrossSpectrum import FitCrossSpectrum
from .FitPowerSpectrum import FitPowerSpectrum
from .FitTimeAvgSpectrum import FitTimeAvgSpectrum
from . import Plotting
from .Likelihoods import cstat, chisq, ratio

ndspec_oned_fit_types = (FitTimeAvgSpectrum, FitPowerSpectrum)

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
                    raise TypeError("Attempted to load invalid fitter object(s)")
        else:
            if issubclass(type(fitobj),SimpleFit):
                pass
            else:
                raise TypeError("Attempted to load invalid fitter object(s)")
            
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

        params = list(fitobj.model_params.valuesdict().keys())
        for key in params:
            #if the parameter has already been shared among all fitters in the 
            #joint fit, we do not need to do anything
            if key in self.shared_keys:
                continue
            #if the parameter has an identical name to one from a different 
            #fitter object previously included in the joint fitter, we mark it 
            #as one that will be shared across the joint fit 
            if any(key in self.joint_params[joint_obs] for joint_obs in self.joint_params):
                self.shared_keys.append(key)
            #if we have never seen the parameter from the individual fitter
            #before, we store a copy within the joint fitter which is necessary 
            #for correct joint evaluations
            elif key not in self.model_params:
                self.model_params.add_many(fitobj.model_params[key])
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
                                    fill_value='extrapolate',kind='cubic')
    
        for name in names:
            if name not in self.joint.keys():
                raise AttributeError(f"{name} is not among the stored fitters")
            #retrieves model or models based on dictionary name
            fitobj = self.joint[name]     
            #if we are evaluating a fit on a shared grid, interpolate, fold, 
            #correct for gain/cross calibration, and mask
            #otherwise evaluate normally      
            if (self.shared_energy_grid is True and type(fitobj)==FitTimeAvgSpectrum):
                model_results = model_interp(fitobj.energs)*fitobj.energ_bounds
                model_results = fitobj.response.convolve_response(model_results) 
                model_results = fitobj._apply_calibration(model_results,params)
                model_results = np.extract(fitobj.ebounds_mask,model_results) 
            else:
                model_results = fitobj.eval_model(params)
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
            raise AttributeError("No loaded fitters found")
            
        if names == None: #retrieves all models
            names = list(self.joint.keys())
        elif type(names) == str:
            names = [names]

        residual_hierarchy = {}            
        if type(names) != list:
            raise TypeError("Input names do not match loaded fitter objects")
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
        warnings.warn(f"WARNING: using coarse energy grids can introduce numerical "\
                       "errors in model evaluations! Check your model against the "\
                       "default energy grids!"
                       ,UserWarning)
        
        names = list(self.joint.keys())
        model_list = []
        for name in names:
            if type(self.joint[name]) == FitTimeAvgSpectrum:
                model_list = np.append(model_list,self.joint[name].model)
                if grid_bounds[0] > self.joint[name].energs[0]:
                    raise ValueError(f"Custom grid lower bound above the minimum energy of {name} fitter object")
                if grid_bounds[-1] < self.joint[name].energs[-1]:
                    raise ValueError(f"Custom grid upper bound below the maximum energy of {name} fitter object")
                    
        if len(model_list) == 0:
            raise AttributeError("Only time-averaged spectrum fitters can use a shared energy grid")    
        
        #check that all the models in the time averaged spectrum fitters are the 
        #same, and assign the model ot be used if that is true        
        first_model = model_list[0]
        if all(model == first_model for model in model_list):
            self.shared_model = first_model
        else:
            raise AttributeError("The same model has to be defined across all fitters to share energy grids")
        
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
            not. If set to True, it lets the cross-calibration constants free 
            and defines them if they were not already present. If set to False, 
            it freezes them to the last value stored. 
            
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
            for name in searched_names:
                if name not in self.joint.keys():
                    raise AttributeError(f"{name} is not among the stored fitters")
            
            self.renorm_names = []
            for name in searched_names:
                if type(self.joint[name]) == FitTimeAvgSpectrum:
                    self.renorm_names = np.append(self.renorm_names,name)        
            #add a renormalisation constant to each selected fitter, and add 
            #the constants to the joint parameters. If the renormalisation was 
            #already active, simply set it to be free to vary
            for name in self.renorm_names:
                key = "renorm_"+str(name)
                if key in self.model_params:
                    self.model_params[key].set(vary=True)
                else:
                    self.joint[name].set_renorm(label=name)
                    self.model_params = self.model_params + self.joint[name].renorm_params
        elif self.renorm_names is not None:
            #disabling the renormalization fixes the constants to unity
            for name in self.renorm_names:
                self.model_params["renorm_"+str(name)].set(vary=False)
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
        #updates the individually linked parameters rather than overwriting them.
        for par in self.model_params:
            self.model_params[par] = params[par]
            for key in self.joint:
                if type(self.joint[key]) == list:
                    for m in self.joint[key]:
                        if par in list(m.model_params.keys()):
                            m.model_params[par].min = params[par].min
                            m.model_params[par].max = params[par].max
                            m.model_params[par].value = params[par].value                     
                else:
                    if par in list(self.joint[key].model_params.keys()):
                        self.joint[key].model_params[par].min = params[par].min
                        self.joint[key].model_params[par].max = params[par].max
                        self.joint[key].model_params[par].value = params[par].value
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
                if name not in self.joint.keys():
                    raise AttributeError(f"{name} is not among the stored fitters")
            for name in names:
                print(f"{name}:" )
                self.joint[name].print_model()
                print("----------------------- \n")
        else:
            if names not in self.joint.keys():
                raise AttributeError(f"{names} is not among the stored fitters")
            print(f"{names}:")
            self.joint[names].print_model()
            print("----------------------- \n")
        return
        
    def print_fit_report(self):
        """
        This method prints the current fit result. It is essentially identical 
        to the same function in LMFit.
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

    def _plot_groups(self,method_name,groups,plot_kwargs,kwargs):
        """
        This helper method draws a plot through specified fitter objects,    
        calling a plotting method specified by method_name ("plot_data" or 
        "plot_model"). The fitters to be called can be divided into gropus, in 
        which case each group will share a single plot (e.g. to make multiple
        time-averaged spectra share a single plot). If one-d methods are not 
        available, the  method defaults to calling plot_method_name_2d.
         
        Parameters:
        -----------
        method_name: str 
            The name of the method to be called, either "plot_data" or 
            "plot_model".
            
        groups: list(list(str))
            One or more lists with the name of the groups that need to share a 
            single figure; for example:
            
            .. code-block:: python
            
                groups=[["fpma","fpmb"],["ixpe_du1","ixpe_du2","ixpe_du3"]]
            
            will plot the first two fitters in one figure, and the last three 
            in another. 
            
        plot_kwargs: dict 
            A dictionary containing the extra keywords to pass to the plotter
            methods. It is meant to pass information for each individual fitter:
            
            .. code-block:: python
            
                plot_kwargs={"fpma": dict(units="eeunfold",
                              data_kwargs=dict(marker="s"), 
                              resid_kwargs=dict(alpha=0.5)),
                             "fpmb": dict(units="eeunfold"),
                             "nicer": dict(units="eeunfold",plot_components=True,
                              component_kwargs={"ndspec_diskbb": {"label":"Disk"}})}
                                        
        kwargs: dict 
            A dictionary containing keywords shared between all fitters that 
            the plotter method will call. For example, if calling plot_model 
            below, one might have:
            
            .. code-block:: python
            
                joint.plot_model(units="eeunfold",residuals="chisq")
            
            in which case "units" and "residuals" become kwargs passed to every 
            plot. 
            
        Returns: 
        --------
        fig, panels: list(fig), list(np.array(axes))
            Lists of figures and arrays of panels, each containing one entry 
            per group as if it were called with return_plot=True.
        """
        
        if groups is None:
            groups = [[name] for name in self.joint.keys()]
        if plot_kwargs is None:
            plot_kwargs = {}
        model_args = {}
        if method_name == "plot_model":
            model_args = dict(params=self.model_params)
        
        figures = []
        for group in groups:
            for name in group:
                if name not in self.joint.keys():
                    raise AttributeError(f"{name} is not among the stored fitters")
            
            #if a fitter has plot_data/plot_model call those, otherwise default 
            #to plot_2d
            if len(group) == 1:
                fitobj = self.joint[group[0]]
                plotter = getattr(fitobj,method_name,None)
                if plotter is None:
                    plotter = getattr(fitobj,method_name+"_2d")
                #take all the kwargs, and add the appropriate ones for this 
                #fitter from fitter_kwargs
                fitter_kwargs = dict(kwargs)
                fitter_kwargs.update(plot_kwargs.get(group[0],{}))
                fig, panels = plotter(return_plot=True,**model_args,**fitter_kwargs)
                panels[0].set_title(name)
            #fitters in the same group are drawn on top of each other
            else:
                panels = None
                for i, name in enumerate(group):
                    plotter = getattr(self.joint[name],method_name,None)
                    if plotter is None:
                        raise TypeError(f"{name} only has two-dimensional plots")
                    #take all the kwargs, and add the appropriate ones for this 
                    #fitter from fitter_kwargs
                    fitter_kwargs = dict(kwargs)
                    fitter_kwargs.update(plot_kwargs.get(name,{}))                    
                    color = "C"+str(i)
                    styles = dict(data_kwargs=dict(color=color,label=name))
                    if method_name == "plot_model":
                        styles["model_kwargs"] = dict(color=Plotting.darken_color(color))
                        styles["resid_kwargs"] = dict(color=color)
                    for key in styles.keys():
                        styles[key].update(fitter_kwargs.pop(key,None) or {})
                    fig, panels = plotter(panels=panels,return_plot=True,
                                          **model_args,**styles,**fitter_kwargs)
                np.ravel(panels)[0].legend(loc="best")
            figures.append((fig,panels))
        
        #return a list of figures/plots 
        return [fig for fig, _ in figures], [panels for _, panels in figures]

    def plot_data(self,groups=None,plot_kwargs=None,return_plot=False,**kwargs):
        """
        This method calls plot_data for all the specified fitter objects. The 
        fitters to be called can be divided into gropus, in which case each 
        group will share a single plot (e.g. to make multiple time-averaged
        spectra share a single plot). If one-d methods are not available, the 
        method defaults to calling plot_data_2d.
         
        Parameters:
        -----------
        groups: list(list(str))
            One or more lists with the name of the groups that need to share a 
            single figure; for example:
            
            .. code-block:: python
            
                groups=[["fpma","fpmb"],["ixpe_du1","ixpe_du2","ixpe_du3"]]
            
            will plot the first two fitters in one figure, and the last three 
            in another. 
            
        plot_kwargs: dict 
            A dictionary containing the extra keywords to pass to the plotter
            methods. It is meant to pass information for each individual fitter:
            
            .. code-block:: python
            
                plot_kwargs={"fpma": dict(units="eeunfold",
                              data_kwargs=dict(marker="s"), 
                              resid_kwargs=dict(alpha=0.5)),
                             "fpmb": dict(units="eeunfold"),
                             "nicer": dict(units="eeunfold",plot_components=True,
                              component_kwargs={"ndspec_diskbb": {"label":"Disk"}})}
                                        
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objects containing 
            the plot or not.

        kwargs: dict 
            A dictionary containing keywords shared between all fitters that 
            the plotter method will call. For example, if calling plot_model 
            below, one might have:
            
            .. code-block:: python
            
                joint.plot_model(units="eeunfold",residuals="chisq")
            
            in which case "units" and "residuals" become kwargs passed to every 
            plot. 
            
        Returns: 
        --------
        fig, panels: list(fig), list(np.array(axes))
            Lists of figures and arrays of panels, each containing one entry 
            per group as if it were called with return_plot=True.   
        """ 
        figs, panels = self._plot_groups("plot_data",groups,plot_kwargs,kwargs)
        if return_plot is True:
          return figs, panels
        else:
          return
        
    def plot_model(self,groups=None,plot_kwargs=None,return_plot=False,**kwargs):
        """
        This method calls plot_model for all the specified fitter objects. The 
        fitters to be called can be divided into gropus, in which case each 
        group will share a single plot (e.g. to make multiple time-averaged
        spectra share a single plot). If one-d methods are not available, the 
        method defaults to calling plot_model_2d.
         
        Parameters:
        -----------
        groups: list(list(str))
            One or more lists with the name of the groups that need to share a 
            single figure; for example:
            
            .. code-block:: python
            
                groups=[["fpma","fpmb"],["ixpe_du1","ixpe_du2","ixpe_du3"]]
            
            will plot the first two fitters in one figure, and the last three 
            in another. 
            
        plot_kwargs: dict 
            A dictionary containing the extra keywords to pass to the plotter
            methods. It is meant to pass information for each individual fitter:
            
            .. code-block:: python
            
                plot_kwargs={"fpma": dict(units="eeunfold",
                              data_kwargs=dict(marker="s"), 
                              resid_kwargs=dict(alpha=0.5)),
                             "fpmb": dict(units="eeunfold"),
                             "nicer": dict(units="eeunfold",plot_components=True,
                              component_kwargs={"ndspec_diskbb": {"label":"Disk"}})}
                                        
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objects containing 
            the plot or not.

        kwargs: dict 
            A dictionary containing keywords shared between all fitters that 
            the plotter method will call. For example, if calling plot_model 
            below, one might have:
            
            .. code-block:: python
            
                joint.plot_model(units="eeunfold",residuals="chisq")
            
            in which case "units" and "residuals" become kwargs passed to every 
            plot. 
            
        Returns: 
        --------
        fig, panels: list(fig), list(np.array(axes))
            Lists of figures and arrays of panels, each containing one entry 
            per group as if it were called with return_plot=True.   
        """ 
        figs, panels = self._plot_groups("plot_model",groups,plot_kwargs,kwargs)
        if return_plot is True:
          return figs, panels
        else:
          return
