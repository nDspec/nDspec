import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.colors as mcolors
import operator as mathop

import lmfit

def model_expand(model,params,distribute=True,**kwargs):
    """
    This function expands a composite lmfit model into its additive terms. For 
    a model of the form a*(b+c), the arrays returned are either those of a*b 
    and a*c, or those of b and c on their own, depending on the distribute 
    argument. 
    
    Parameters:
    -----------
    model: lmfit.Model or lmfit.CompositeModel
        The model to be expanded.
        
    params: lmfit.Parameters 
        The parameters used to evaluate the model.
        
    distribute: bool, default=True 
        A boolean to choose whether the multiplicative and convolution 
        components of the model are applied to the additive terms - e.g.
        if it is True, the function return a*b and a*c from a*(b+c)
        
    kwargs: 
        Kwargs containing additional info like the independent variables 
        of the model
    
    Returns:
    --------
    components: dict 
        The evaluated additive terms, keyed by the prefix of the additive 
        component of each term. 
    """

    #count the number of additive components in the model
    leaves = _additive_leaves(model)
    
    components = {}
    #if there is only one component, return nothing
    if len(leaves) < 2:
        return components
    
    for leaf in leaves:
        if distribute is True:
            values = _eval_term(model,leaf,params,**kwargs)
        else:
            values = leaf.eval(params,**kwargs)
        label = _leaf_label(leaf)
        if label in components:
            raise ValueError(f"Two additive components share the label {label}")
        components[label] = values
    
    return components
 
def _additive_leaves(model):
    """
    This function returns the leaves of a model tree that include only its 
    additive terms. 
    
    Parameters:
    -----------
    model: lmfit.Model or lmfit.CompositeModel
        The model whose tree is to be inspected.
    
    Returns:
    --------
    leaves: list(lmfit.Model)
        The additive components of the model.
    """
    
    if not isinstance(model,lmfit.CompositeModel):
        return [model]
    
    if model.op is mathop.add:
        return _additive_leaves(model.left)+_additive_leaves(model.right)
    
    left_leaves = _additive_leaves(model.left)
    right_leaves = _additive_leaves(model.right)
    
    #a model like (a+b)*(c+d) can not obviously be separated in additive 
    #and multiplicative components like e.g. a*(b+d*c), so we must throw 
    #an error to avoid returning nonsense
    if len(left_leaves) > 1 and len(right_leaves) > 1:
        raise ValueError("The model structure does not allow a non-ambigous "
                         "separation of all the components")
    
    if len(right_leaves) >= len(left_leaves):
        return right_leaves
    
    return left_leaves
 
def _eval_term(model,leaf,params,**kwargs):
    """
    This function takes the additive model leafs identified by
    _additive_leaves, and evaluates them while also isolating any
    multiplicative components that may need to be applied to them.
    
    Parameters:
    -----------
    model: lmfit.Model or lmfit.CompositeModel
        The model to be evaluated.
        
    leaf: lmfit.Model 
        The additive component identifying the term to be evaluated.
        
    params: lmfit.Parameters 
        The parameters used to evaluate the model.
        
    kwargs: 
        The independent variables of the model.
    
    Returns:
    --------
    values: numpy.ndarray 
        The term, evaluated over the independent variables provided.
    """

    #this is the case of a model component that does not need any extra 
    #multiplicative or whatever components distributed to it
    if not isinstance(model,lmfit.CompositeModel):
        return model.eval(params,**kwargs)

    #this catches the case when two components are being added together,
    #and figures out which one corresponds to the particular leaf we are 
    #interested in for this function call
    if model.op is mathop.add:
        if _contains(model.left,leaf):
            return _eval_term(model.left,leaf,params,**kwargs)
        return _eval_term(model.right,leaf,params,**kwargs)

    #this catches every other operation, like multiplication
    return model.op(_eval_term(model.left,leaf,params,**kwargs),
                    _eval_term(model.right,leaf,params,**kwargs))
 
 
def _contains(model,leaf):
    """
    This function checks whether a given component appears anywhere in a  
    model tree.
    
    Parameters:
    -----------
    model: lmfit.Model or lmfit.CompositeModel
        The model whose tree is to be searched.
        
    leaf: lmfit.Model 
        The component to search for.
    
    Returns:
    --------
    found: bool 
        Whether the component appears in the tree.
    """
    
    if model is leaf:
        return True
    
    if not isinstance(model,lmfit.CompositeModel):
        return False
    
    return _contains(model.left,leaf) or _contains(model.right,leaf)
 
def _leaf_label(leaf):
    """
    This function returns the label used to identify a model component in a 
    plot legend, taken from the prefix the user assigned to it and falling back 
    on the name of the function the model wraps.
    
    Parameters:
    -----------
    leaf: lmfit.Model 
        The component to be labelled.
    
    Returns:
    --------
    label: str 
        The label of the component.
    """
    
    if leaf.prefix:
        return leaf.prefix.rstrip('_')
    
    return leaf.func.__name__

def check_shape(arr,size,name,dtype=float):
    """
    This function checks that the length in the first axis of an input array 
    is identical to that provided. 
 
    Parameters:
    -----------
    arr: array_like 
        The array of floats to be checked.
 
    size: int 
        The size of the dimension expected.
 
    name: str 
        The name of the array used in throwing the error.
 
    dtype: dtype, default=float 
        The type of data to cast the array to after checking the size.
 
    Returns:
    --------
    arr: array_like 
        The array after checking its size.
    """
    
    arr = np.atleast_1d(np.asarray(arr,dtype=dtype))
    if arr.shape[0] != size:
        raise ValueError(name+" has shape "+str(arr.shape)+", expected first "
                         "dimension n_bins="+str(size))
    
    return arr
 
def check_matching_length(name_a,array_a,name_b,array_b):
    """
    This function checks that two arrays which are expected to be defined 
    over the same grid have the same length, by calling check_shape. If 
    either array is None, no check is performed, so that optional arrays can 
    be validated against each other without requiring both to be present.
 
    Parameters:
    -----------
    name_a, name_b: str 
        The names of the two arrays, used to build the error message.
 
    array_a, array_b: array_like or None 
        The two arrays whose lengths are compared.
    """
    
    if array_a is not None and array_b is not None:
        check_shape(array_b,len(array_a),name_b)
    
    return

def check_two_arrays(name_a,array_a,name_b,array_b):
    """
    This function checks that two arrays which are only meaningful together 
    are either both provided or both left out. If only one of the two is 
    None, the function throws an error.
 
    Parameters:
    -----------
    name_a, name_b: str 
        The names of the two arrays, used to build the error message.
 
    array_a, array_b: array_like or None 
        The two arrays whose presence is compared.
    """
    
    if (array_a is None) != (array_b is None):
        present = name_a if array_a is not None else name_b
        missing = name_b if array_a is not None else name_a
        raise ValueError(present+" was provided without "+missing+", but "
                         "the two must be given together or not at all")
    
    return
