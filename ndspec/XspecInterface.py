import ctypes as ct
import numpy as np
from functools import wraps
import os
import warnings
from sys import platform

class ModelInterface():
    """
    This class allows users to load a library file containing Xspec-compatible 
    models (including the entire library that comes with a typical HEASOFT 
    installation), initialize them into the class objects as class methods, and 
    evaluate them in their own Python code. ModelInterface serves as the parent 
    for two more classes which can interface with models in Fortran (including
    the default Xspec library) or C respectively.
    
    Attributes:
    -----------
    models_info: dict 
        A dictionary to store model information. The keywords store the
        initialized model names (models_info['nthcomp']), the model type and 
        parameters (e.g. models_info[['nthcomp']['type']), and the names, 
        minimum and maximum parameter values, and units of each (e.g. 
        models_info[['nthcomp']['parameters']['kTe']['unit'] = "keV").
        
    lib:  DLL ctype 
        The compiled .so (if using a Linux system) or .dylib (if using MacOS) 
        library file loaded. By default, the code looks for the Xspec models 
        at path_to_heasoft/Xspec/platform_name/lib/libXSFunctions.so (or dylib)
        file produced by an Xspec installation, and it contains (among other 
        functions) the entire library of Xspec spectral models.
        
    _all_info: dict 
        A dictionary containing the information of every model in the loaded 
        library,  regardless of whether the user has initialized it for use or 
        not. The structure is identical to models_info.         
    """    
    def __init__(self, lib_path, pars_path):
        self._all_info = self.parse_models(pars_path)
        self.models_info = {}
    
        # Load the library
        self.lib = ct.cdll.LoadLibrary(lib_path)

    def parse_models(self,input_file):
        """
        This method parses the Xspec file with the model name, type, parameters
        values and units, etc, and stores the necessary information in the 
        _all_info dictionary. 
        
        Parameters:
        -----------
        input_file: str 
            A path to the model file to be parsed. By default, the method looks 
            for the Xspec model list in path_to_heasoft/Xspec/src/manager/. 
            
        Output:
        -------
        models_info: dict 
            A dictionary containing model names and types, parameter values,
            minimum nad maximum bounds, and parameter units for all models in 
            the library. 
        """

        with open(input_file, 'r') as file:
            file_content = file.read()
       
        # Split the content by empty lines to separate different models
        model_sections = file_content.strip().split('\n\n')        
        models_info = {}
        
        for section in model_sections:           
            lines = section.strip().split('\n')        
            if not lines:
                continue
        
            first_line_parts = lines[0].split()
            model_name = first_line_parts[0].lower()
            model_type = first_line_parts[5] 
            func_call = first_line_parts[4]
            #print("test name:", model_name, " function call:", )
        
            parameters = {}   
            #Parse each parameter line, ignoring the first one (since it contains the model definition)
            for line in lines[1:]:
                parts = line.split()
                #All the exception catches are to deal with non-standard formatting like empty lines, 
                #switch parameters without bounds, weird formatting in the units, etc
                try: 
                    param_name = parts[0]
                    unit = parts[1]
                except IndexError:
                    continue            
                if param_name[0] == "$" or param_name[0] == "*":
                    param_name = param_name.strip('$*')
                    unit = "n/a"
                    try:
                        value = float(parts[1])
                        min_val = None
                        max_val = None
                    except ValueError:
                        try:
                            value = float(parts[2])
                            min_val = None
                            max_val = None
                        except ValueError:
                            value = float(parts[3])
                            min_val = None
                            max_val = None
                elif unit == '"':    
                    unit = "n/a"
                    value = float(parts[3])
                    min_val = float(parts[4])
                    max_val = float(parts[7])
                else:
                    try:
                        value = float(parts[2])
                        min_val = float(parts[3])
                        max_val = float(parts[6])
                    except ValueError:
                        unit = parts[1]+parts[2]
                        value = float(parts[3])
                        min_val = float(parts[4])
                        max_val = float(parts[7])                
                unit = unit.strip('"')
                parameters[param_name] = {
                    'value': value,
                    'min': min_val,
                    'max': max_val,
                    'unit': unit
                }       
            if model_type == 'add':
                parameters['norm'] = {
                    'value': 1,
                    'min': 0,
                    'max': 1e20,
                    'unit': "n/a"
                }                     
            models_info[model_name] = {
                'func_call': func_call,
                'type': model_type,
                'parameters': parameters
            }
            
        return models_info    

    def print_model_info(self):
        """
        This method prints to terminal a list of all the models that are 
        currently initialized and ready for use, as well as their model type, 
        parameter names, as well as default/min/max values and units.
        """
    
        print()
        print("Initialized Xspec models:")
        for component, details in self.models_info.items():
            print(f"{component}:")
            print(f"  type: {details['type']}")
            print(f"  function called: {details['func_call']}")
            print("  parameters:")
            for param, values in details['parameters'].items():
                value_str = ', '.join(f"{key}: {value}" for key, value in values.items())
                print(f"    {param}: {value_str}")
            print()

    def check_param_values(self,model_name,params):
        """
        This method ensures that the input parameters for a given model 
        evaluation do not exceed its allowed bounds, by comparing the input 
        values with the minimum and maximum stored in the models_info dictionary
        
        Parameters:
        -----------
        model_name: str 
            A string containing the name of the model being computed 
            
        params: np.array, dtype=float32 
            An array of parameter values stored as float32 being used in the 
            model computation 
            
        Output:
        -------
        test_pars: bool 
            A boolean which returns false if one of the number of parameters 
            being passed is incorrect, or if one of the values is out of bounds. 
            In the latter case, the model evaluation returns NaN. 
            If none of the above happen, the method evalutes to True.
        """    
        #check that we're within bounds. This takes a microsecond to loop over ~15 parameters so we just call it every time
        test_pars = True 
        
        par_data = self.models_info[model_name]['parameters']
        if len(par_data) != len(params):
            warnings.warn(f"Wrong parameter number {len(par_data)} required but {len(params)} passed")
            test_pars = False 
            return test_pars 
        for i, key in enumerate(par_data):
            if (params[i] < par_data[key]['min']):
                params[i] = par_data[key]['min'] 
                warnings.warn(f"Model parameter {par_data[key]} value {params[i]} out of bounds") 
                test_pars = False 
                return test_pars 
            elif (params[i] > par_data[key]['max']):
                params[i] = par_data[key]['max']
                warnings.warn(f"Model parameter {par_data[key]} value {params[i]}  out of bounds") 
                test_pars = False 
                return test_pars 
            else:
                return test_pars

    def load_models(self, models):
        """
        This method allows users to initialized multiple models simultaneously 
        by passing a dictionary with model names and calling functions. 
        
        Parameters:
        -----------
        models: dict 
            A dictionary whose keys are identical to the function names users 
            want to intialize in the class. 
        """
    
        for model_name, model_func in models.items():
            # Apply the decorator to each model function
            self.add_model(model_func)

class FortranInterface(ModelInterface):
    """
    This class implements the methods inherited from ModelInterface to import 
    and execute Xspec-compatible Fortran models. Users can use it either for 
    their own code, or to load the entire Xspec library that is included in a 
    typical HEASOFT installation. 
    """    
    def __init__(self, lib_path=None, pars_path=None):
        default_xspec = False
        if lib_path is None:
            headas_path = os.environ.get("HEADAS")
            input_file =  headas_path + f"/../Xspec/src/manager/model.dat"       
            if headas_path:
                if platform == "linux" or platform == "linux2":
                    lib_path = headas_path + f"/../Xspec/{os.path.basename(headas_path)}/lib/libXSFunctions.so"
                elif platform == "darwin":
                    lib_path = headas_path + f"/../Xspec/{os.path.basename(headas_path)}/lib/libXSFunctions.dylib"
                else:
                    raise OSError("Your platform is not supported.")
                default_xspec = True
            else:
                raise EnvironmentError("HEADAS environment variable not set.")
        if pars_path is None:
            headas_path = os.environ.get("HEADAS")
            pars_path =  headas_path + f"/../Xspec/src/manager/model.dat"     
        
        ModelInterface.__init__(self,lib_path,pars_path)
        
        if default_xspec is True:
            self.initialize_heasoft()
        pass

    def add_model(self, func, symbol=None):
        """
        This method initializes a given model by adding it to the library object
        as one of its methods - for example:
        
        def powerlaw(ear, params):
            pass
        
        lib.add_model(powerlaw)
        model = lib.powerlaw(arguments)
        
        Parameters:
        -----------
        func: function 
            An empty function with the same name and input parameters as the 
            model to be added to the library object. 

        symbol: str, optional
            A string containing the name of the function in the library to be
            called. If not provided, it defaults to the function name. Generally, 
            a user does not need to provide this argument, as the function name 
            is usually sufficient to identify the model in the library. However, 
            if a user wants to use a different name for the function they are 
            calling via ndspec, they can provide the original function name here.
        """
        func_name = func.__name__.rstrip('_')

        if symbol is None:
            symbol = func_name + "_"

        #sort out model parameters
        self.models_info[func_name] = self._all_info[func_name] 
    
        #prepare the model call for the given model name
        lib_func = getattr(self.lib, f"{symbol}")
        lib_func.argtypes = [
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_float)
        ]
        lib_func.restype = None

        #specify the exact model call depending on whether the model is additive
        #multiplicative or convolutional.
        if self.models_info[func_name]['type'] == "add":        
            @wraps(func)
            def wrapper(ear, params):
                ear = np.asarray(ear, dtype=np.float32)
                params = np.asarray(params, dtype=np.float32)
                ne = len(ear) - 1
                photar = np.zeros(ne, dtype=np.float32)
                photer = np.zeros(ne, dtype=np.float32)

                par_test = self.check_param_values(func_name,params)
                if par_test is False:
                    photar = np.nan
                else:
                    lib_func(
                        ear.ctypes.data_as(ct.POINTER(ct.c_float)),
                        ct.byref(ct.c_int(ne)),
                        params.ctypes.data_as(ct.POINTER(ct.c_float)),
                        ct.byref(ct.c_int(1)),
                        photar.ctypes.data_as(ct.POINTER(ct.c_float)),
                        photer.ctypes.data_as(ct.POINTER(ct.c_float))
                    )
                    photar = photar/np.diff(ear)
                return photar*params[-1]
        elif self.models_info[func_name]['type'] == "mul":        
            @wraps(func)
            def wrapper(ear, params):
                ear = np.asarray(ear, dtype=np.float32)
                params = np.asarray(params, dtype=np.float32)
                ne = len(ear) - 1
                photar = np.zeros(ne, dtype=np.float32)
                photer = np.zeros(ne, dtype=np.float32)
                
                par_test = self.check_param_values(func_name,params)
                if par_test is False:
                    photar = np.nan
                else:        
                    lib_func(
                        ear.ctypes.data_as(ct.POINTER(ct.c_float)),
                        ct.byref(ct.c_int(ne)),
                        params.ctypes.data_as(ct.POINTER(ct.c_float)),
                        ct.byref(ct.c_int(1)),
                        photar.ctypes.data_as(ct.POINTER(ct.c_float)),
                        photer.ctypes.data_as(ct.POINTER(ct.c_float))
                    )
                return photar
        elif self.models_info[func_name]['type'] == "con":  
            @wraps(func)
            def wrapper(ear, params, seed):
                ear = np.asarray(ear, dtype=np.float32)
                params = np.asarray(params, dtype=np.float32)
                ne = len(ear) - 1
                seed = np.array(seed,dtype = np.float32)
                photer = np.zeros(ne, dtype = np.float32)
                
                par_test = self.check_param_values(func_name,params)
                if par_test is False:
                    photar = np.nan
                else:
                    lib_func(
                        ear.ctypes.data_as(ct.POINTER(ct.c_float)),
                         ct.byref(ct.c_int(ne)),
                         params.ctypes.data_as(ct.POINTER(ct.c_float)),
                         ct.byref(ct.c_int(1)),
                         seed.ctypes.data_as(ct.POINTER(ct.c_float)),
                         photer.ctypes.data_as(ct.POINTER(ct.c_float))
                    )
                return seed
                
        # Attach the wrapper to the class 
        setattr(self, func_name, wrapper)        
        return func

    def initialize_heasoft(self):
        """
        This method calls the fninit HEASOFT function, which initializes cross 
        sections and abundances and is required to correctly evaluate Xspec 
        models outside of the Xspec command interface. 
        """    
        init_call = self.lib.fninit_
        init_call()

class CInterface(ModelInterface):
    """
    This class implements the methods inherited from ModelInterface to import 
    and execute Xspec-compatible C models. It is meant to be used with custom 
    models not immediately available with Xspec installations, such as 
    Relxill and its various flavours.
    """   
    def __init__(self, lib_path,pars_path):
        ModelInterface.__init__(self,lib_path,pars_path)    
        pass

    #try to let users pass any name they want
    def add_model(self, func):
        """
        This method initializes a given model by adding it to the library object
        as one of its methods - for example:
        
        def powerlaw(ear, params):
            pass
        
        lib.add_model(powerlaw)
        model = lib.powerlaw(arguments)
        
        Parameters:
        -----------
        func: function 
            An empty function with the same name and input parameters as the 
            model to be added to the library object. 
        """
        func_name = func.__name__.rstrip('_')
        
        #sort out model parameters
        self.models_info[func_name] = self._all_info[func_name]
        func_call = self.models_info[func_name]['func_call'].strip('c_')
        #prepare the model call for the given model name
        lib_func = getattr(self.lib, f"{func_call}")
        lib_func.argtypes = [
            ct.POINTER(ct.c_double),  
            ct.c_int,                     
            ct.POINTER(ct.c_double),  
            ct.c_int,                     
            ct.POINTER(ct.c_double),  
            ct.POINTER(ct.c_double), 
            ct.c_char_p    
        ]
        lib_func.restype = None

        #specify the exact model call depending on whether the model is additive
        #multiplicative or convolutional.      
        if self.models_info[func_name]['type'] == "add":        
            @wraps(func)
            def wrapper(ear, params):
                ear = np.asarray(ear, dtype=np.float64)
                params = np.asarray(params, dtype=np.float64)
                ne = len(ear) - 1
                photar = np.zeros(ne, dtype=np.float64)
                photer = np.zeros(ne, dtype=np.float64)
                params = np.asarray(params, dtype=np.float64)
                init_string = "1"
                spectrum = 1

                par_test = self.check_param_values(func_name,params)
                if par_test is False:
                    photar = np.nan
                else:
                    lib_func(
                        ear.ctypes.data_as(ct.POINTER(ct.c_double)),
                        ct.c_int(ne),
                        params.ctypes.data_as(ct.POINTER(ct.c_double)),
                        ct.c_int(spectrum),
                        photar.ctypes.data_as(ct.POINTER(ct.c_double)),
                        photer.ctypes.data_as(ct.POINTER(ct.c_double)),
                        init_string.encode('utf-8')
                    )
                photar = photar/np.diff(ear)
                return photar*params[-1]
        elif self.models_info[func_name]['type'] == "mul":        
            @wraps(func)
            def wrapper(ear, params):
                ear = np.asarray(ear, dtype=np.float64)
                params = np.asarray(params, dtype=np.float64)
                ne = len(ear) - 1
                photar = np.zeros(ne, dtype=np.float64)
                photer = np.zeros(ne, dtype=np.float64)
                params = np.asarray(params, dtype=np.float64)
                init_string = "1"
                spectrum = 1

                par_test = self.check_param_values(func_name,params)
                if par_test is False:
                    photar = np.nan
                else:
                    lib_func(
                        ear.ctypes.data_as(ct.POINTER(ct.c_double)),
                        ct.c_int(ne),
                        params.ctypes.data_as(ct.POINTER(ct.c_double)),
                        ct.c_int(spectrum),
                        photar.ctypes.data_as(ct.POINTER(ct.c_double)),
                        photer.ctypes.data_as(ct.POINTER(ct.c_double)),
                        init_string.encode('utf-8')
                    )
                return photar
        elif self.models_info[func_name]['type'] == "con":  
            @wraps(func)
            def wrapper(ear, params, seed):
                ear = np.asarray(ear, dtype=np.float64)
                params = np.asarray(params, dtype=np.float64)
                ne = len(ear) - 1
                seed = np.zeros(ne, dtype=np.float64)
                photer = np.zeros(ne, dtype=np.float64)
                params = np.asarray(params, dtype=np.float64)
                init_string = "1"                
                spectrum = 1                
                
                par_test = self.check_param_values(func_name,params)
                if par_test is False:
                    photar = np.nan
                else:
                    lib_func(
                        ear.ctypes.data_as(ct.POINTER(ct.c_double)),
                        ct.c_int(ne),
                        params.ctypes.data_as(ct.POINTER(ct.c_double)),
                        ct.c_int(spectrum),
                        seed.ctypes.data_as(ct.POINTER(ct.c_double)),
                        photer.ctypes.data_as(ct.POINTER(ct.c_double)),
                        init_string.encode('utf-8')
                    )
                return seed

        # Attach the wrapper to the class 
        setattr(self, func_name, wrapper)        
        return func
