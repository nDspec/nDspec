import numpy as np
import warnings

from . import Plotting
from .Response import ResponseMatrix
from .SimpleFit import SimpleFit
from .Likelihoods import chisq, ratio

class FitTwoD(SimpleFit):
    """
    This class is designed for fitting generic types of two-dimensional data, 
    regardless of what units it may be in. Models used in this fitter are 
    expected to be provided already in the same unit as the data. Common 
    examples of using this class might be time-dependent spectroscopy, or 
    fitting a dynamical power spectrum. 
    
    As an exception, users can optionally pass an instrument response matrix 
    object, in which case the y axis is assumed to be in units of photon 
    channels and the model should produce units of integrated photon flux 
    over that axis.
    
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
        
    Other attributes:
    -----------------
    rows, columns: int 
        Two integers keeping track of the number of rows and columns in the 
        data loaded. Only tracks the number of rows/columns that are noticed 
        during the fit. 
        
    column_grid: np.array(float)
        The grid of values along which the x-axis of the model is computed.        
    
    column_mask: np.array(bool)
        A masking array used to ignore or notice data bins along the x axis. 
    
    row_grid: np.array(float)
        The grid of values along which the y-axis of the model is computed. If 
        an instrument response is loaded, it contains the photon channels.       
    
    column_mask: np.array(bool)
        A masking array used to ignore or notice data bins along the y axis. 
        
    _column_grid_unmasked,_row_grid_unmasked: np.array(float)
        The unmasked arrays over which the entire data is define, whether they 
        are noticed duing the fit or not.

    _all_rows,_all_columns,_all_bins: int
        Integers keeping track of all the rows, columns, or total number of 
        data points, whether they are noticed in the fit or not.  
   
    dependence, units: str
        Two strings used to specify the units the data is in, used to handle 
        filtering in/out data bins in the fitter.    
    
    response: nDspec.ResponseMatrix, default None 
        The instrument response matrix corresponding to the data to be fitted.
        It is required to define the energy grids over which model and data are
        defined. The following arrays are only defined when a response matrix 
        is loaded:      
        
    energs: np.array(float)
        The array of physical photon energies over which the model is computed. 
        Defined as the middle of each bin in the energy range stored in the 
        instrument response provided.    
        
    energ_bounds: np.array(float)
        The array of energy bin widths, for each bin over which the model is 
        computed. Defined as the difference between the uppoer and lower bounds 
        of the energy bins stored in the insrument response provided. 
        
    ear: np.array(float) 
        The array of energy bin bounds, for each bin over which the model is 
        computed. Only necessary when calling Xspec models due to their unique 
        input structure.
               
    ebounds: np.array(float) 
        The array of energy channel bin centers for the instrument energy
        channels,  as stored in the instrument response provided. Only contains 
        the channels that are noticed during the fit.

    ewidths: np.array(float) 
        The array of energy channel bin widths for the instrument energy
        channels,  as stored in the instrument response provided. Only contains 
        the channels that are noticed during the fit.
    """

    def __init__(self,likelihood="chisq"):
        SimpleFit.__init__(self,likelihood)
        self.response=None
        self.dependence="generic"
        self.units="real"
        pass

    def set_data(self,data,data_err,column_grid,row_grid,response=None,noise=None,noise_err=None):
        """
        This method is used to pass the data users want to fit. The input has 
        to be in the form of numpy arrays for the data, its errors, and the
        grids over which it is defined. Users can optionally pass an instrument 
        response, as well as background arrays. 
        
        Parameters:
        -----------        
        data, data_err: np.array([float,float])
            Two-dimensional arrays containing the data to be fitted and its 
            uncertainties. 
            
        column_grid, row_grid: np.array(float)
            The arrays over which the data is defined. If users pass an 
            instrument response (see below), then row_grid is the array of 
            energy bin bounds, for each channel over which the data is defined.
            This is identical to the 'ear' array in Xspec models. 
            
        response: nDspec.ResponseMatrix, default None 
            An instrument response (including both rmf and arf) loaded into a 
            nDspec ResponseMatrix object. 
            
        noise, noise_err: np.array(float), default None
            Optional arrays including the background/noise floor and its error. 
        """
        
        if data.shape != data_err.shape:
            raise AttributeError("Size of data and error are not the same")
        if data.shape[1] != len(column_grid):
            raise AttributeError("Size of column grid does not match the data")
        if response is None and data.shape[0] != len(row_grid):
            raise AttributeError("Size of row grid does not match the data")
        elif response is not None and data.shape[0] != len(row_grid)-1:
            raise AttributeError("Size of row grid does not match the data")
        if noise is not None and noise.shape != data.shape:
            raise AttributeError("Size of noise and data are not the same")


        self.rows = data.shape[0]
        self.columns = data.shape[1]
        self.data = data.T.flatten()
        self.data_err = data_err.T.flatten()
        self.column_grid = column_grid
        self.column_mask = np.full((self.columns), True)  
        
        if noise is not None:
            self.noise = noise.flatten()
            self.noise_err = noise_err.flatten()
        if response is not None:
            #in this case, the row grid is the equilvaent of the ear array in 1d models
            self._set_energy_arrays(response,row_grid)
        else:
            self.row_grid = row_grid
            self.row_mask = np.full((self.rows), True)    
        #do unmasked data things here 
        self._set_unmasked_data()
        return

    def _set_energy_arrays(self,response,grid):
        """
        This initializer method is used to set the appropriate energy-dependent 
        arrays in case an instrument response is used in the fit.        
        """
        self.row_grid = grid
        #rebin the response to row_grid here
        bounds_lo = self.row_grid[:-1]
        bounds_hi = self.row_grid[1:]
        self.response = response.rebin_channels(bounds_lo,bounds_hi) 

        #assign the arrays from the response - note that ear is called row_grid to generalize in this class
        self.energs = 0.5*(self.response.energ_hi+self.response.energ_lo)
        self.energ_bounds = self.response.energ_hi-self.response.energ_lo 
        self.ear = np.append(self.response.energ_lo,self.response.energ_hi[-1])    
        self.ebounds = 0.5*(self.response.emax+self.response.emin)
        self.ewidths = self.response.emax - self.response.emin
        self.row_mask = np.full((self.response.n_chans), True)
        return

    def _set_unmasked_data(self):
        """
        This initializer method is used to set up the unmasked arrays for later 
        book-keeping.      
        """
        self._data_unmasked = self.data
        self._data_err_unmasked = self.data_err  
        self._column_grid_unmasked = self.column_grid
        self._row_grid_unmasked = self.row_grid
        #save the number of unmasked rows and columns for book keeping
        self._all_rows = self.rows
        self._all_columns = self.columns
        self._all_bins = self._all_columns*self._all_rows
        self.n_bins = self._all_bins
        #if our fit object has a background (e.g. a spectral background, or 
        #Poisson noise) we also must store it to subtract it correctly from the 
        #data
        if self.noise is not None:
            self._noise_unmasked = self.noise
            self._noise_err_unmasked = self.noise_err
        else:
            self._noise_unmasked = None
            self._noise_err_unmasked = None           
        if self.response is not None:
            self._emin_unmasked = self.response.emin
            self._emax_unmasked = self.response.emax
            self._ebounds_unmasked = self.ebounds
            self._ewidths_unmasked = self.ewidths
        return 

    def set_fit_statistic(self,stat):
        """
        This method is used to set the statistic to be optimized during the fit.
        By default, the optimizer will optimize the chi-squared statistic. 
        
        Parameters:
        -----------
        stat: str 
            A string with the name of the fit statistic to be used. Supported 
            statistics currently are "chisq" (the standard chi squared statistic, 
            appropriate for data in the Gaussian regime) and "cstat" (the Cash 
            statistic, see https://ui.adsabs.harvard.edu/abs/1979ApJ...228..939C/abstract,
            appropriate for Poisson-distributed data). 
        """
        
        if (stat != "chisq" and stat != "cstat" and stat != "custom"):
            raise ValueError("Fit statistic not recognized")
        self.likelihood = stat 

    def ignore_columns(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        ignore selected columns based on their bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored column interval.
        bound_hi : float
            Higher bound of ignored column interval.    
        """
    
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Grid bounds must be floats or integers")        
        
        self.column_mask = ((self._column_grid_unmasked<bound_lo)|
                            (self._column_grid_unmasked>bound_hi))&self.column_mask        
        
        #now update the column grid; in this case it will never be energies
        #unlike the rows
        self.columns = self.column_mask[self.column_mask==True].size
        self.column_grid = np.extract(self.column_mask,self._column_grid_unmasked)

        #and now call the mask to filter the data
        self.data = self._filter_2d_by_mask(self._data_unmasked)
        self.data_err = self._filter_2d_by_mask(self._data_err_unmasked)
        if self.noise is not None:
            self.noise = self._filter_2d_by_mask(self._noise_unmasked)
            self.noise_err = self._filter_2d_by_mask(self._noise_err_unmasked)            
        return

    def notice_columns(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        notice selected columns based on their bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of noticed column interval.
        bound_hi : float
            Higher bound of noticed column interval.    
        """
    
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Grid bounds must be floats or integers")   

        self.column_mask = self.column_mask|np.logical_not(
                            (self._column_grid_unmasked<bound_lo)|
                            (self._column_grid_unmasked>bound_hi))

        #now update the column grid; in this case it will never be energies
        #unlike the rows
        self.columns = self.column_mask[self.column_mask==True].size
        self.column_grid = np.extract(self.column_mask,self._column_grid_unmasked)

        #and now call the mask to filter the data
        self.data = self._filter_2d_by_mask(self._data_unmasked)
        self.data_err = self._filter_2d_by_mask(self._data_err_unmasked)
        if self.noise is not None:
            self.noise = self._filter_2d_by_mask(self._noise_unmasked)
            self.noise_err = self._filter_2d_by_mask(self._noise_err_unmasked)            
        return

    def ignore_rows(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        ignore selected columns rows on their bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of ignored rows interval.
        bound_hi : float
            Higher bound of ignored rows interval.    
        """
    
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Grid bounds must be floats or integers")
        
        if self.response is not None:
            self.row_mask = ((self._emin_unmasked<bound_lo)|
                             (self._emax_unmasked>bound_hi))&self.row_mask
        else:
            self.row_mask = ((self._row_grid_unmasked<bound_lo)|
                             (self._row_grid_unmasked>bound_hi))&self.row_mask            
        
        #extract the grid from the 1d mask
        self.rows = self.row_mask[self.row_mask==True].size
        self.row_grid = np.extract(self.row_mask,self._row_grid_unmasked)
        #filter the additional arrays that come with an instrument response
        if self.response is not None:
            self.ebounds = np.extract(self.row_mask,self._ebounds_unmasked)
            self.ewidths = np.extract(self.row_mask,self._ewidths_unmasked)

        #and now call the mask to filter the data
        self.data = self._filter_2d_by_mask(self._data_unmasked)
        self.data_err = self._filter_2d_by_mask(self._data_err_unmasked)
        if self.noise is not None:
            self.noise = self._filter_2d_by_mask(self._noise_unmasked)
            self.noise_err = self._filter_2d_by_mask(self._noise_err_unmasked)          
        return

    def notice_rows(self,bound_lo,bound_hi):
        """
        This method adjusts the arrays stored such that they (and the fit) 
        notice selected columns rows on their bounds.

        Parameters:
        -----------
        bound_lo : float
            Lower bound of noticed rows interval.
        bound_hi : float
            Higher bound of noticed rows interval.    
        """
    
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Grid bounds must be floats or integers")

        if self.response is not None:
            self.row_mask = self.row_mask|np.logical_not(
                            (self._emin_unmasked<bound_lo)|
                            (self._emax_unmasked>bound_hi)) 
        else:
            self.row_mask = self.row_mask|np.logical_not(
                            (self._row_grid_unmasked<bound_lo)|
                            (self._row_grid_unmasked>bound_hi)) 
        
        #extract the grid from the 1d mask
        self.rows = self.row_mask[self.row_mask==True].size
        self.row_grid = np.extract(self.row_mask,self._row_grid_unmasked)
        #filter the additional arrays that come with an instrument response
        if self.response is not None:
            self.ebounds = np.extract(self.row_mask,self._ebounds_unmasked)
            self.ewidths = np.extract(self.row_mask,self._ewidths_unmasked)

        #and now call the mask to filter the data
        self.data = self._filter_2d_by_mask(self._data_unmasked)
        self.data_err = self._filter_2d_by_mask(self._data_err_unmasked)
        if self.noise is not None:
            self.noise = self._filter_2d_by_mask(self._noise_unmasked)
            self.noise_err = self._filter_2d_by_mask(self._noise_err_unmasked)    
        return

    def eval_model(self,params=None,column_grid=None,row_grid=None,fold=True,mask=True):
        """
        This method is used to evaluate and return the model values for a given 
        set of parameters, over given row and column grids. If a response is 
        loaded, by default it  will evaluate the model over the energy grid 
        defined in the response, using the parameters values stored internally 
        in the model_params attribute, without folding the model through the 
        response.  
        
        Parameters:
        -----------                         
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are 
            provided, the model_params attribute is used.

        column_grid: np.array(float), default None
            The array of values in the x-axis over which to evaluate the model.
            If none are provided, the same grid contained in the fitter object 
            is used.  
            
        row_grid: np.array(float), default None
            The array of values in the y-axis over which to evaluate the model.
            If none are provided, the same grid contained in the fitter object 
            is used. If the fitter contains an instrument response, this array 
            is the energy grid contained in that response. 
            
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
        
        if column_grid is None:
            column_grid = self._column_grid_unmasked
        #handle the simple case of not needing a response
        if row_grid is None and self.response is None:
            row_grid = self._row_grid_unmasked
        #if we do need a response we also must prepare to multiply by bin width 
        #to keep units consistent across fits 
        elif self.response is not None:
            row_grid = self.ear                      
        
        if params is None:
            model = self.model.eval(self.model_params,x_axis=column_grid,y_axis=row_grid)
        else:
            model = self.model.eval(params,x_axis=column_grid,y_axis=row_grid)            

        #add folding of the response if necessary here 
        #the transpositions are necessary because of the weirdness introduce 
        #by .flatten(). TBD check that this makes sense with xpsec models
        if self.response is not None and fold is True:
            model = self.response.convolve_response(model.T).T 
        
        if mask is True:
            model = self._filter_2d_by_mask(model)

        model = model.flatten()
        
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

        model = self.eval_model(params)
    
        if self.likelihood == "chisq":
            residuals, _ = self.get_residuals("chisq",model=model,mask=True)
        elif self.likelihood == "cstat":
            residuals, _ = self.get_residuals("cstat",model=model,mask=True)
        elif self.likelihood == "custom":
            residuals, _ = self.get_residuals("custom",model=model,mask=True)
        else:
            raise AttributeError("Chosen likelihood not supported")
        return residuals

    def _plot_format(self,array):
        """
        This method reshapes a flat array defined over every the input data grid
        (data, errors, model...) for plotting, with ignored bins set to NaN so
        that they are ignored correctly by all matplotlib functions. 
        
        Parameters:
        -----------
        array: np.array(float)
            The array to reshape, of size (_all_bins).
            
        Returns:
        --------
        array: np.array(float)
            The reshaped array, of size (_all_rows,_all_columns).
        """
        
        twod_mask = self.column_mask.reshape((self._all_columns,1))*self.row_mask
        array = array.reshape((self._all_columns,self._all_rows))
        return np.transpose(np.where(twod_mask,array,np.nan))

    def plot_data_2d(self,x_name=None,y_name=None,z_name=None,cmap='plasma',
                     log_xaxis=False,log_yaxis=False,log_zaxis=False,
                     return_plot=False,mesh_kwargs=None):
        """
        This method creates a 2d plot of the data loaded by the user as a 
        function of whatever units are in used for the rows and columns. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        x_name: str, default="None"
            The units to use to label the X axis.
            
        y_name: str, default="None"
            The units to use to label the Y axis.
            
        z_name: str, default="None"
            The units to use to label the Y axis.
            
        cmap: str, default='plasma'
            The colormap to use for the panel. 

        log_xaxis, log_yaxis: bool, default=False 
            Booleans to choose whether each axis of the mesh uses a logarithmic 
            scale.

        log_zaxis: bool, default=False 
            A boolean to choose whether the the z axis of the mesh uses a 
            logarithmic scale.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure object containing 
            the plot or not.
            
        mesh_kwargs: dict, default=None 
            Keyword arguments for the 2d colormesh.

        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.

        panels: matplotlib.axes, optional 
            The panel containing the plot produced by the method.     
        """
    
        if x_name is None:
            x_name = "X axis"
        if y_name is None:
            y_name = "Y axis"

        x_axis = self._column_grid_unmasked
        if self.response is None:
            y_axis = self._row_grid_unmasked
        else:
            y_axis = self._ebounds_unmasked
 
        ymin, ymax = None, None
        if self.response is not None:
            ymin = np.max([self.ebounds[0]-0.5*self.ewidths[0],1e-1])
            ymax = self.ebounds[-1]+0.5*self.ewidths[-1]
        
        data_vals = self._plot_format(self._data_unmasked)
        if log_zaxis is True:
            data_vals = np.log10(np.where(data_vals > 0,data_vals,np.nan))
            z_name = "log10(" + z_name + ")"    

        plot_layout = Plotting.make_layout(ncols=1,colorbars=True,figsize=(6.5,4.5))
        fig, panel = Plotting.make_panels(plot_layout)              
        
        data = Plotting.make_mesh_data(x_points=x_axis,
                               y_points=y_axis,
                               z_values=data_vals,
                               x_label=x_name,
                               y_label=y_name,
                               z_label=z_name)
                               

        Plotting.draw_colormesh_panel(panel,data,cmap=cmap,
                                      log_xaxis=log_xaxis,
                                      log_yaxis=log_yaxis,
                                      ylim=[ymin,ymax],
                                      mesh_kwargs=mesh_kwargs)      
        
        if return_plot is True:
            return fig, panel
        else:
            return 
            
    def plot_model_2d(self,plot_data=True,params=None,residuals=None,x_name=None,
                      y_name=None,z_name=None,cmaps=None,log_xaxis=False,
                      log_yaxis=False,log_zaxis=False,return_plot=False,
                      mesh_kwargs=None):
        """
        This method creates a 2d plot of the data loaded by the user, the 
        model for the given parameters (either passed as an object, or already 
        loaded in the fitter), and residuals, as a unction of whatever units 
        are in used for the rows and columns. 
        
        It is also possible to return the figure object, for instance in order 
        to save it to file.
        
        Parameters:
        -----------
        residuals: str, default="chisq"
            The units to be used in the residuals. 
            
        params: lmfit.parameters, default=None 
            The parameters to be used to evaluate the model. If False, the set 
            of parameters stored in the class is used. 
        
        x_name: str, default="None"
            The units to use to label the X axis.
            
        y_name: str, default="None"
            The units to use to label the Y axis.
        
        return_plot: bool, default=False
            A boolean to decide whether to return the figure object containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.        
        """
        
        if params is None:
            params = self.model_params
        
        if y_name is None:
            y_name = "Y axis"

        if cmaps is None:
            cmaps = ["plasma","RdYlBu"]
        elif isinstance(cmaps,str):
            cmaps = [cmaps,cmaps]
        else:
            raise ValueError("cmaps needs to be a string")

        x_axis = self._column_grid_unmasked
        if self.response is None:
            y_axis = self._row_grid_unmasked
        else:
            y_axis = self._ebounds_unmasked
 
        ymin, ymax = None, None
        if self.response is not None:
            ymin = np.max([self.ebounds[0]-0.5*self.ewidths[0],1e-1])
            ymax = self.ebounds[-1]+0.5*self.ewidths[-1]

        model = self.eval_model(params=params,mask=False)
        model_vals = self._plot_format(model)
        if log_zaxis is True:
            model_vals = np.log10(np.where(model_vals > 0,model_vals,np.nan))
            
        columns = ["model"]
        meshes = dict(model=model_vals)        
        if plot_data is True:
            columns = ["data","model","residuals"]
            data_vals = self._plot_format(self._data_unmasked)
            if log_zaxis is True:
                data_vals = np.log10(np.where(data_vals > 0,data_vals,np.nan))
            meshes["data"] = data_vals
            
            if residuals == "chisq":
                res_label = "$\\Delta\\chi$"
            elif residuals == "ratio":
                res_label = "Data/model"
            else:
                res_label = "Residuals"
            if residuals is None:
                residuals = self.likelihood
            res,_ = self.get_residuals(residuals,model=self._filter_2d_by_mask(model))
            noticed = self._filter_2d_by_mask(np.arange(self._data_unmasked.size))
            #this jank is necessary to support ignored bins in the 2d plot
            resid = np.full(self._data_unmasked.size,np.nan)
            resid[noticed] = res
            meshes["residuals"] = self._plot_format(resid)

        plot_layout = Plotting.make_layout(ncols=len(columns),colorbars=True,
                                           figsize=(4.9*len(columns),4.0))
        fig, panels = Plotting.make_panels(plot_layout,squeeze=False) 

        shared = {} 
        if plot_data is True:
            shared["vmin"] = np.nanmin(meshes["data"])
            shared["vmax"] = np.nanmax(meshes["data"])   
        else:            
            shared["vmin"] = np.nanmin(meshes["model"])
            shared["vmax"] = np.nanmax(meshes["model"])   
        
        for col, name in enumerate(columns):
            if col == 2:
                cmap = cmaps[-1]
                z_label = res_label
                style = {}
            else:            
                cmap = cmaps[0]    
                z_label = columns[col]
                style = dict(shared)
            if mesh_kwargs is not None:
                style.update(mesh_kwargs)
            
            mesh_data = Plotting.make_mesh_data(x_points=x_axis,
                                                y_points=y_axis,
                                                z_values=meshes[name],
                                                x_label=x_name,
                                                y_label=y_name,
                                                z_label=z_label)
            mesh = Plotting.draw_colormesh_panel(panels[0,col],mesh_data,cmap=cmap,
                                                 diverging=(name == "residuals"),
                                                 colorbar=(plot_data is False),
                                                 log_xaxis=log_xaxis,
                                                 log_yaxis=log_yaxis,
                                                 ylim=[ymin,ymax],
                                                 mesh_kwargs=style)
            panels[0,col].set_title(name.capitalize())
            if name == "model":
                mesh_model = mesh
            if col > 0:
                panels[0,col].set_ylabel("")
                panels[0,col].tick_params(labelleft=False)
            if name == "residuals":
                _, ticks = Plotting.get_diverging_norm(meshes["residuals"])
                bar = fig.colorbar(mesh,ax=panels[0,col],pad=0.01,ticks=ticks)
                bar.set_label(res_label)
        
        if plot_data is True:
            bar = fig.colorbar(mesh_model,ax=[panels[0,0],panels[0,1]],pad=0.01)
            bar.set_label(z_name)
        else:
            panels = panels[0]              
        
        if return_plot is True:
            return fig, panels
        else:
            return
