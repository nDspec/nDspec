import numpy as np
import lmfit
from lmfit import fit_report, minimize
from lmfit import Parameters
from .SimpleFit import SimpleFit 

class JointFit():
    """
    Generic joint fitting class. Use this class if you have multiple 
    non-simultaneous observations that you wish to share parameters 
    between or mutliple simultaneous observations that share a single model.

    Users can simply add other Fit...Objs from ndspec to this structural class
    and this will handle evaluating the model, sharing parameters between 
    models, sharing whole models between observations, and perform optimization 
    of an appropriate fit statistic. There is no restriction on the type of
    Fit...Objs that can be added, nor the number of Fit...Objs, which 
    technically allows for extremely numerous joint inference of observations.
    
    Note that JointFit does not perform extra performance enhancements to make
    evaluations run faster, so optimization and joint inference on many 
    parameters is still subject to the usual computational problems that come
    in this case. 
    
    Attributes:
    ------------
    joint : dict{Fit... objects and/or list(Fit... objects)}
        Dictionary containing named Fit... objects to be joint fitted. By
        default, observations that share parameters completely (simultaneous or
        different data products of observations) are packaged together in lists.
        
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
    
    def __init__(self,flatten=False):
        self.joint = {}
        self.joint_params = {}
        self.fit_result = None
        self.model_params = None
        #defines whether model results should be returned in a dictionary or
        #simply as a flat 1-d numpy array
        self.flatten = flatten 

        
    def add_fitobj(self,fitobj,name):
        """
        Adds a model to the joint fitting hierarchy. If a list of 
        fitting objects is added, it is assumed that the objects are intended
        as simultaneous observations (i.e. a NuSTAR and XMM-Newton observation)
        which share a model.

        Parameters
        ----------
        fitobj : Fit... object or list of Fit... objects
            the Fit... object(s) containing the dataset(s) and model(s) to be 
            fit. In the case of multiple fit objects, they must all share the 
            same underlying model as all of their parameters will be tied.
        
        name: str
            Key assigned to call each fitter stored in the joint fitter object.
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
            #In this case we are passing multiple observations assumed to have   
            #the same underlying model 
            self.joint[name] = fitobj
            if len(fitobj) > 1: #check if actually multiple models
                first_fitobj = fitobj[0]
                #if first added object, add model params
                if self.model_params == None:
                    self.model_params = Parameters()
                    for par in fitobj[0].model_params:
                        self.model_params.add_many(fitobj[0].model_params[par])
                #links all model parameters to first model in list
                for i in range(1,len(fitobj)): 
                    self.share_params(first_fitobj, fitobj[i])
                #pulls parameters names and saves to dictionary for model
                #evaluation later
                params = []
                for key in first_fitobj.model_params.valuesdict().keys():
                    #iterates through current fit objects
                    for joint_obs in self.joint_params:
                        #checks if parameter matches any previous model 
                        #parameters
                        if key in self.joint_params[joint_obs]:
                            print(f"""
                                  Caution: {key} is already a model parameter.
                                  Do you intend for these parameters to be linked?
                                  If not, give it a different name to differentiate
                                  between multiple instances of the same type for
                                  different models.
                                  """)
                        else:
                            self.model_params.add_many(first_fitobj.model_params[key])
                    params.append(key)
                self.joint_params[name] = params
            else:
                raise TypeError("""
                                The list of fit objects contains only a
                                single entry, either add other fit objects 
                                containing simultaneous observations to the 
                                list or only add the fit object as a single
                                entry.
                                """)
        else: 
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
    
    def _model_decompose(self,model):
        """
        Decomposes lmfit composite models into their base Models.
        Mainly useful for retrieving parameter names from complex
        composite models, and is only for internal model use.

        Parameters
        ----------
        model: lmfit.CompositeModel
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
            raise TypeError("Not an lmfit composite model")
        
        models = []
        if type(model.left) == lmfit.Model:
            models.append(model.left)
        else:
            models.extend(self._model_decompose(model.left))
        
        if type(model.right) == lmfit.Model:
            models.append(model.right)
        else:
            models.extend(self._model_decompose(model.right))
        
        return models

    def share_params(self,first_fitobj,second_fitobj,param_names=None):
        """
        Shares parameters between models, and links the parameters of individual 
        models that compose the joint model to a parameter object. This object 
        is what is passed to the optimizer during the fitting process.

        Parameters
        ----------
        first_fitobj : Fit... object 
            primary fit object that the secondary fit object is linked to.
        second_fitobj : Fit... object 
            secondary fit object that is linked to the primary.
        param_names : str or list(str), optional
            Names of parameters (with the same name) to share between models. 
            The default is to share all parameters together.
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
            #iterates through all fit objects
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
                print("""Not all parameters inputted are in the models of both 
                         fitter objects""")
                return
        
        for name in param_names:
            #find parameter name in first fit objects models
            second_fitobj.model_params[name] = first_fitobj.model_params[name]
    
    def eval_model(self,params=None,names=None):
        """
        This method is used to evaluate and return the model values of models 
        stored in the joint fitter object.
        
        Parameters:
        ------------
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model/models. If 
            none are provided, the model_params attribute is used.

        names: list(str), default None
            names of the models that should be evalualated. Defaults to
            evaluating all models.
        
        Returns:
        --------
        model_hierarchy: dict(np.array(float))
            All models are evaluated and returned as a dictionary, corresponding
            to the top-level hierarchy.
        """
        if names == None: #retrieves all models
            names = self.joint.keys()
        if params == None:
            params = self.model_params
        #creates structure to return model results
        model_hierarchy = {}
        
        for name in names:
            if name not in self.joint.keys():
                raise AttributeError(f"{name} is not in model hierarchy")
            #retrieves model or models based on dictionary name
            fitobjs = self.joint[name] 
            if type(fitobjs) == list: #if simultaneous, evaluate each one.
                model_results = []
                for fit_obj in fitobjs:
                    model_results.append(fit_obj.eval_model(params))
            else:
                model_results = fitobjs.eval_model(params)
            model_hierarchy[name] = model_results
        
        if self.flatten == False:
            return model_hierarchy
        else:
            model = np.array([])
            for key in model_hierarchy:
                model = np.concatenate([model,model_hierarchy[key]])
            return model
        
    def _minimizer(self,params,names=None):
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
            model_dict = self.eval_model(params,names)
            residuals = np.array([])
            for name in names:
                model = model_dict[name]
                resids = (self.joint[name].data-model)/self.joint[name].data_err
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
            names of the models that should be evalualated. Defaults to
            evaluating all models.
        """
        if names == None:
            names = list(self.joint.keys())
        
        self.fit_result = minimize(self._minimizer,self.model_params,
                                   method=algorithm,args=[names])
        print(fit_report(self.fit_result,show_correl=False))
        fit_params = self.fit_result.params
        self.set_params(fit_params)
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
