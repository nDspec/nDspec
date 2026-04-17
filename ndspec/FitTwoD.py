import numpy as np
import warnings

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
from matplotlib.colors import TwoSlopeNorm
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams.update({'font.size': 17})

from lmfit import Model as LM_Model
from lmfit import Parameters as LM_Parameters

from .Response import ResponseMatrix
from .SimpleFit import SimpleFit
from .Likelihoods import chisq, ratio

class TwoDFit(SimpleFit):
    def __init__(self,likelihood="chisq"):
        SimpleFit.__init__(self,likelihood)
        self.response=None
        self.dependence="generic"
        self.units="real"
        pass

    def set_data(self,data,data_err,column_grid,row_grid,response=None,noise=None,noise_err=None):
        self.rows = data.shape[0]
        self.columns = data.shape[1]
        #here: compare the size of the grid, with rows/columns.
        self.data = data.flatten()
        self.data_err = data_err.flatten()
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
        self.row_grid = grid
        #rebin the response to row_grid here
        bounds_lo = self.row_grid[:-1]
        bounds_hi = self.row_grid[1:]
        self.response = response.rebin_channels(bounds_lo,bounds_hi) 

        #assign the arrays from the response - note that ear is called row_grid to generalize in this class
        self.energs = 0.5*(self.response.energ_hi+self.response.energ_lo)
        self.energ_bounds = self.response.energ_hi-self.response.energ_lo    
        self.ebounds = 0.5*(self.response.emax+self.response.emin)
        self.ewidths = self.response.emax - self.response.emin
        self.row_mask = np.full((self.response.n_chans), True)
        return

    def _set_unmasked_data(self):
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
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Grid bounds bounds must be floats or integers")        
        
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
            self.noise = self._filter_2d_by_mask(self._data_unmasked)
            self.noise_err = self._filter_2d_by_mask(self._data_err_unmasked)            
        return

    def notice_columns(self,bound_lo,bound_hi):
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Grid bounds bounds must be floats or integers")   

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
            self.noise = self._filter_2d_by_mask(self._data_unmasked)
            self.noise_err = self._filter_2d_by_mask(self._data_err_unmasked)            
        return

    def ignore_rows(self,bound_lo,bound_hi):
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Energy bounds must be floats or integers")
        
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
            self.noise = self._filter_2d_by_mask(self._data_unmasked)
            self.noise_err = self._filter_2d_by_mask(self._data_err_unmasked)          
        return

    def notice_rows(self,bound_lo,bound_hi):
        if ((isinstance(bound_lo, (np.floating, float, int)) != True)|
            (isinstance(bound_hi, (np.floating, float, int)) != True)):
            raise TypeError("Energy bounds must be floats or integers")

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
            self.noise = self._filter_2d_by_mask(self._data_unmasked)
            self.noise_err = self._filter_2d_by_mask(self._data_err_unmasked)    
        return

    def eval_model(self,params=None,column_grid=None,row_grid=None,fold=True,mask=True):
        
        if column_grid is None:
            column_grid = self._column_grid_unmasked
        if row_grid is None:
            row_grid = self._row_grid_unmasked
            if self.response is not None:
                energ_bounds = self.energ_bounds
                #this is only used for the ndspec energy dependent models 
                energ = self.energs
        #set up the rest of the energy grid if we're providing the interval ourselves
        elif self.response is not None:
            energ = 0.5*(row_grid[1:]+row_grid[:-1])
            energ_bounds = row_grid[1:]-row_grid[:-1]            
        
        if params is None:
            model = self.model.eval(self.model_params,x_axis=column_grid,y_axis=row_grid)
        else:
            model = self.model.eval(params,x_axis=column_grid,y_axis=row_grid)            

        #add folding of the response if necessary here 
        if self.response is not None and fold is True:
            model = model*energ_bounds
            model = self.response.convolve_response(model) 
        
        if mask is True:
            model = self._filter_2d_by_mask(model)
        
        return model 

    def plot_data(self,x_name=None,y_name=None,return_plot=False):
        fig, ((ax1)) = plt.subplots(1,1,figsize=(6.,4.5))

        if x_name is None:
            x_name = "X axis"
        if y_name is None:
            y_name = "Y axis"

        x_axis = self._column_grid_unmasked
        y_axis = self._row_grid_unmasked
        
        twod_mask = self.row_mask.reshape((1,self._all_rows))* \
                    self.column_mask.reshape((self._all_columns,1))
        twod_mask = twod_mask.reshape((self._all_columns,self._all_rows))
        twod_mask = np.logical_not(twod_mask) 
        
        plot_data = self._data_unmasked.reshape((self._all_columns,self._all_rows))
        plot_data = np.transpose(np.ma.masked_where(twod_mask, plot_data))        
        
        data_plot = ax1.pcolormesh(x_axis,y_axis,plot_data,cmap="plasma",shading='auto',linewidth=0)
        fig.colorbar(data_plot, ax=ax1)
        ax1.set_xlabel(x_name)
        ax1.set_ylabel(y_name)
        if self.response is not None:
            ymin = np.max([self.ebounds[0]-0.5*self.ewidths[0],1e-1])
            ymax = self.ebounds[-1]+0.5*self.ewidths[-1]
            ax1.set_ylim([ymin,ymax])
        
        fig.tight_layout()
        if return_plot is True:
            return fig
        else:
            return  
            
    def plot_model(self,x_name=None,y_name=None,residuals="chisq",return_plot=False):
        model = self.eval_model(mask=False)
        model_res,_ = self.get_residuals(residuals,model=model,mask=False)
        
        fig, ((ax1),(ax2),(ax3)) = plt.subplots(1, 3, figsize=(15.,5.), sharex=True) 

        if x_name is None:
            x_name = "X axis"
        if y_name is None:
            y_name = "Y axis"

        x_axis = self._column_grid_unmasked
        y_axis = self._row_grid_unmasked
        
        twod_mask = self.row_mask.reshape((1,self._all_rows))* \
                    self.column_mask.reshape((self._all_columns,1))
        twod_mask = twod_mask.reshape((self._all_columns,self._all_rows))
        twod_mask = np.logical_not(twod_mask) 
        
        plot_data = self._data_unmasked.reshape((self._all_columns,self._all_rows))
        plot_data = np.transpose(np.ma.masked_where(twod_mask, plot_data))    

        plot_model = model.reshape((self._all_columns,self._all_rows))
        plot_model = np.transpose(np.ma.masked_where(twod_mask, plot_model)) 

        plot_res = model_res.reshape((self._all_columns,self._all_rows))
        plot_res = np.transpose(np.ma.masked_where(twod_mask, plot_res)) 
        
        data_plot = ax1.pcolormesh(x_axis,y_axis,plot_data,cmap="plasma",shading='auto',linewidth=0)
        fig.colorbar(data_plot, ax=ax1)
        ax1.set_xlabel(x_name)
        ax1.set_ylabel(y_name)
        ax1.set_title("Data") 
        if self.response is not None:
            ymin = np.max([self.ebounds[0]-0.5*self.ewidths[0],1e-1])
            ymax = self.ebounds[-1]+0.5*self.ewidths[-1]
            ax1.set_ylim([ymin,ymax])

        model_plot = ax2.pcolormesh(x_axis,y_axis,plot_model,cmap="plasma",shading='auto',linewidth=0)
        fig.colorbar(model_plot, ax=ax2)
        ax2.set_xlabel(x_name)
        ax2.set_ylabel(y_name)
        ax2.set_title("Model") 
        if self.response is not None:
            ymin = np.max([self.ebounds[0]-0.5*self.ewidths[0],1e-1])
            ymax = self.ebounds[-1]+0.5*self.ewidths[-1]
            ax2.set_ylim([ymin,ymax])

        res_min = np.min([np.min(plot_res),-1])
        res_max = np.max([np.max(plot_res),1])
        TwoSlopeNorm(vmin=res_min,vcenter=0,vmax=res_max) 
        res_plot = ax3.pcolormesh(x_axis,y_axis,plot_res,cmap="RdYlBu",shading='auto',linewidth=0)
        fig.colorbar(res_plot, ax=ax3)
        ax3.set_xlabel(x_name)
        ax3.set_ylabel(y_name)
        ax3.set_title("Residuals") 
        if self.response is not None:
            ymin = np.max([self.ebounds[0]-0.5*self.ewidths[0],1e-1])
            ymax = self.ebounds[-1]+0.5*self.ewidths[-1]
            ax3.set_ylim([ymin,ymax])
        
        fig.tight_layout()
        if return_plot is True:
            return fig
        else:
            return  
