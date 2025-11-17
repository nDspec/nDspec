import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams.update({'font.size': 17})

def parse_plot_axes(plot):
    """
    This function is used to parse the contents of a matplotlib plot object. 
    Given a plot object return by e.g. one of the nDspec fitter object, it 
    loops over all the axis objects in the plots, and prints information about 
    the collections (which contain data points, colors, linewidth etc) in the 
    plot object.
    
    Parameters: 
    -----------
    plot: matplotlib.figure.Figure
        A plot object of which you want to list the contents of the axes
    """

    for i, ax in enumerate(plot.axes):
        print(f"\n--- Plot axis #{i+1} ---")
        for j, collection in enumerate(ax.collections):
            print(f"--- Collection {j} ---")
            print(f"Type: {type(collection)}")
        
            # Print the number of segments (for errorbars, lines, etc.)
            if hasattr(collection, 'get_segments'):
                segments = collection.get_segments()
                print(f"Number of segments: {len(segments)}")
                print(f"First segment:\n{segments[0]}")
        
            # Print other useful attributes
            if hasattr(collection, 'get_array'):
                array = collection.get_array()
                print(f"Array data (e.g., colors, sizes): {array}")
        
            if hasattr(collection, 'get_linewidths'):
                linewidths = collection.get_linewidths()
                print(f"Line widths: {linewidths}")
        
            if hasattr(collection, 'get_colors'):
                colors = collection.get_colors()
                print(f"Colors: {colors}") 
                
def parse_plot_lines(plot):
    """
    This function is used to parse the contents of a matplotlib plot object. 
    Given a plot object return by e.g. one of the nDspec fitter object, it 
    loops over all the axis objects in the plots, and prints information about 
    the lines (which in nDspec contain e.g. the model info) stored in the plot. 
    
    Parameters: 
    -----------
    plot: matplotlib.figure.Figure
        A plot object of which you want to list the contents of the axes
    """
    for j, ax in enumerate(plot.axes):
        print(f"\n--- Plot axis #{j+1} ---")
        for i, line in enumerate(ax.get_lines()):
            print(f"--- Line {i} ---")
            print(f"Label: {line.get_label()}")
            print(f"Color: {line.get_color()}")
            print(f"Linestyle: {line.get_linestyle()}")
            print(f"Linewidth: {line.get_linewidth()}")
            print(f"X data length: {len(line.get_xdata())}")
            print(f"Y data length: {len(line.get_ydata())}")
            print(f"First 3 (x, y) points: {list(zip(line.get_xdata()[:3], line.get_ydata()[:3]))}")
            
def get_plot_info(plot):
    """
    This function is used to return the data points, as well as x and y error 
    bars, and model values, of an input matplotlib plot object created with the
    plot_model methods in the nDspec fitter objects. 
    
    Parameters: 
    -----------
    plot: matplotlib.figure.Figure
        A plot object of which you want to list the contents of the axes. 
        
    Returns:
    --------
    plot_info: dict 
        A dictionary containing the information parsed from the plot. The arrays
        stored in each keyword are, respectively:    
        
        x_points: The array used to define the x axis (e.g. energy in keV)'         
        
        x_points: The array containing the actual data points. 
        
        x_bars: The array containing the width of the error bar in the x 
                  direction. 
        
        y_bars: The array containing the errors on the data. 
        
        resid: The array containing the value of the residuals
        
        reserr: The array containing the error bars on the residuals. 
        
        model_points: The array containing the values of the x axis over which 
                      the model is defined.
        
        model_vals: The array containing the model values used in the plot.
        
        model_ls: The line style used to plot the model 
        
        model_lw: The width of the line used to plot the model   
    """
    
    plot_info = {}
    
    #save plot info and close the plot object 
    ax1_data, ax2_data = plot.axes
    lines = ax1_data.get_lines()
    plt.close(plot)
    
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
    
    # Extract data points and errors for the resiuals. The x axis is the same as above. 
    segments_res = ax2_data.collections[0].get_segments()
    y_res = np.mean([[seg[0, 1], seg[1, 1]] for seg in segments_res], axis=1)
    y_reserr = np.abs(np.array([[seg[0, 1], seg[1, 1]] for seg in segments_res]).T - y_res)
  
    #Extract model lines 
    model_line = lines[1]    
    model_xvals = model_line.get_xdata()
    model_yvals = model_line.get_ydata()
    model_ls = model_line.get_linestyle(),
    model_lw = model_line.get_linewidth(),
    
    plot_info = dict(x_points=x_data,
                     y_points=y_data,
                     x_bars=x_errors,
                     y_bars=y_errors,
                     resid=y_res,
                     reserr=y_reserr,
                     model_points=model_xvals,
                     model_vals=model_yvals,
                     linestyle=model_ls,
                     linewidth=model_lw,
                     ax1_data=ax1_data,
                     ax2_data=ax2_data)
  
    return plot_info

def model_decompose(model):
    """
    Decomposes lmfit composite models into their base Models.
    Mainly useful for retrieving parameter names from complex
    composite models, and is only for internal model use.

    Parameters:
    -----------
    model: lmfit.compositemodel
        composite model to be decomposed

    Returns:
    --------
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
        models.extend(model_decompose(model.left))
    
    if type(model.right) ==  lmfit.Model:
        models.append(model.right)
    else:
        models.extend(model_decompose(model.right))
    
    return models

def share_params(first_fitobj,second_fitobj,param_names=None):
    """
    Shares parameters between models and links the parameters of individual 
    models that compose the joint fit to the parameters inferred in the 
    optimization process.

    Parameters:
    -----------
    first_fitobj : Fit... object 
        primary fit object that the secondary fit object is linked to.
    second_fitobj : Fit... object 
        secondary fit object that is linked to the primary.
    param_names : str or list(str), optional
        Names of parameters (with the same name) to share between models. The default 
        is to share all parameters together
        
    Returns:
    --------
    """
        
    #checks that both models are correctly specified
    if (((type(first_fitobj.model) != lmfit.CompositeModel)&(type(first_fitobj.model) != lmfit.Model))|
       ((type(second_fitobj.model) != lmfit.CompositeModel)&((type(second_fitobj.model) != lmfit.Model)))):  
        raise AttributeError("The model input must be an LMFit Model or CompositeModel object")
    
    #adds all base models into list (decomposes CompositeModels into Models)
    models = []
    #adds all models from first fit object as a list of models
    models.append(model_decompose(first_fitobj.model))
    #adds all models from second fit object as a list of models
    models.append(model_decompose(second_fitobj.model))

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
        
    return
