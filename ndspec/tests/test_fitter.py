import sys
import os
import warnings
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))

from lmfit import Model as LM_Model
from lmfit import Parameters as LM_Parameters

from stingray import EventList

from ndspec.Response import ResponseMatrix
from ndspec.SimpleFit import SimpleFit, load_pha
from ndspec.FitPowerSpectrum import FitPowerSpectrum
from ndspec.FitTimeAvgSpectrum import FitTimeAvgSpectrum
from ndspec.FitCrossSpectrum import FitCrossSpectrum
from ndspec.FitTwoD import FitTwoD
from ndspec.JointFit import JointFit
import ndspec.Models as models

import pytest

def ones_model(len):
    return np.ones(len)
    
def zeros_model(len):
    return np.zeros(len)

def cross_const(energs,freqs):
    n_energs = len(energs)
    n_freqs = len(freqs)
    model = np.ones((n_freqs,n_energs))
    return model

def twod_ones_model(x_axis,y_axis):
    return np.ones((len(x_axis),len(y_axis)))
    
def sin_wave(phase,norm,center):
    return norm*np.sin(phase*2*np.pi)+center

def pulsing_bb(ear,x_axis,norm_bb,kT,norm_mod):
    var_norm = sin_wave(x_axis,norm_mod,norm_bb)
    energ = 0.5*(ear[1:]+ear[:-1])
    model = np.zeros((len(x_axis),len(energ)))
    for i, phasenorm in enumerate(var_norm):
        bb_array = np.array([phasenorm,kT])
        model[i,:] = models.bbody(energ,bb_array)*np.diff(ear)
    model = model.T
    return model

def pulsing_bb_eval(y_axis,x_axis,norm_bb,kT,norm_mod):
    model = pulsing_bb(y_axis,x_axis,norm_bb,kT,norm_mod)
    model = model.T
    return model
     

class TestFitPowerSpectrum(object):
 
    @classmethod
    def setup_class(cls):
        dummy_psd = np.ones(10)
        freqs = np.linspace(1,10,10)
        cls.test_psd = FitPowerSpectrum()
        cls.test_psd.set_data(dummy_psd,0.1*dummy_psd,freqs) 
        return 
 
    def test_psd_eval(self):
        psd_model = LM_Model(ones_model)
        model_parameters = LM_Parameters()
        model_parameters.add_many(('len', 10, False, 1, 20))
 
        self.test_psd.set_model(psd_model)
        self.test_psd.set_params(model_parameters)
        test_residuals = self.test_psd.get_residuals("chisq")
        assert(np.allclose(test_residuals[0],np.zeros(10)))
 
    def test_set_psd_data(self):
        wrong_data = np.ones(8)
        wrong_err = np.ones(9)
        wrong_freq = np.ones(9)
        #test that the class doesn't allow data and grid to have different sizes
        with pytest.raises(AttributeError):
            self.test_psd.set_data(wrong_data,wrong_err)
        with pytest.raises(AttributeError):
            self.test_psd.set_data(wrong_data,wrong_data,data_grid=wrong_freq)
            
    def test_psd_likelihood(self):      
        #test that the class doesn't calculate the likelihood if it is not defined 
        #correctly
        with pytest.raises(AttributeError):
            self.test_psd.likelihood = "error"  
            err = self.test_psd._minimizer(params=self.model_params)
    
    def test_psd_plot_errors(self):
        #test that plots do not allow weird things to be rendered
        with pytest.raises(ValueError):
            self.test_psd.plot_data(units="wrong")
        with pytest.raises(ValueError):
            self.test_psd.plot_model(residuals="wrong")
        with pytest.raises(ValueError):
            self.test_psd.plot_model(units="wrong")

class TestFitTimeAvgSpectrum(object):
 
    @classmethod
    def setup_class(cls):
        #set up the response to be used 
        rmffile = os.getcwd()+"/ndspec/tests/data/xrt.rmf"
        arffile = os.getcwd()+"/ndspec/tests/data/xrt.arf"
        cls.response = ResponseMatrix(rmffile)
        cls.response.load_arf(arffile)
 
        #set up data+fitter to test the time averaged spectrum 
        cls.test_spec = FitTimeAvgSpectrum()
        cls.test_spec.set_data(cls.response,os.getcwd()+"/ndspec/tests/data/xrt.fak") 
        return 
 
    def test_spec_eval(self):
        spec_model = LM_Model(ones_model)
        model_parameters = LM_Parameters()
        model_parameters.add_many(('len', 2400, False, 1, 3000))
 
        self.test_spec.set_model(spec_model,params=model_parameters)
        test_residuals = self.test_spec.get_residuals("ratio")
        #ignore the bins that are nan/inf because of the swift response 
        n_dof = self.test_spec.n_chans - 1 - 27
        test_stat = np.sum(test_residuals[0][27:-1])/n_dof
        #tolerance for Poisson noise in the simulated spectrum
        tol = 5e-3
        assert(np.allclose(test_stat,1,rtol=tol))
 
    def test_spec_likelihood(self):     
        #test that the class doesn't calculate the likelihood if it is not defined 
        #correctly               
        with pytest.raises(AttributeError):
            self.test_spec.likelihood = "error"
            err = self.test_spec._minimizer(params=None)
            
    def test_spec_plot_errors(self):
        #test that plots do not allow weird things to be rendered
        with pytest.raises(ValueError):
            self.test_spec.plot_data(units="wrong")
        with pytest.raises(ValueError):
            self.test_spec.plot_model(residuals="wrong")
        with pytest.raises(ValueError):
            self.test_spec.plot_model(units="wrong")        
 
    #test that users can't require weird formats for residuals    
    #note that this tests only one class because get_residuals is shared 
    #between all fitters anyway  
    def test_residual_errors(self):      
        with pytest.raises(ValueError):
            spec_model = LM_Model(ones_model)
            model_parameters = LM_Parameters()
            model_parameters.add_many(('len', 2400, False, 1, 3000))
            self.test_spec.set_model(spec_model,params=model_parameters)
            test_residuals = self.test_spec.get_residuals("wrong")               
 

class TestFitCrossSpectrum(object):
 
    @classmethod
    def setup_class(cls):
        #set up the response to be used 
        rmffile = os.getcwd()+"/ndspec/tests/data/xrt.rmf"
        arffile = os.getcwd()+"/ndspec/tests/data/xrt.arf"
        cls.response = ResponseMatrix(rmffile)
        cls.response.load_arf(arffile)
        
        cls.new_channels = np.linspace(cls.response.emin[0],cls.response.emax[-1],6)
        cls.rebin_matrix = cls.response.rebin_channels(cls.new_channels[:-1],cls.new_channels[1:])
        new_grid = 0.5*(cls.rebin_matrix.emax+cls.rebin_matrix.emin)
        new_width = cls.rebin_matrix.emax-cls.rebin_matrix.emin
        cls.new_edges = np.append(new_grid-0.5*new_width,new_grid[-1]+0.5*new_width[-1])
 
        #set up data+fitter to test the cross spectrum 
        cls.cross_freqs = np.linspace(0.2,1.0,5)
        cls.test_cross = FitCrossSpectrum()
        cls.test_cross.set_coordinates("polar")
        cls.test_cross.set_product_dependence("energy")
 
        dummy_mods = np.ones((4,5))
        dummy_phase = np.ones((4,5))
 
        dummy_cross = np.append(dummy_mods.flatten(),dummy_phase.flatten())
        dummy_cross_err = 0*dummy_cross
 
        cls.test_cross.set_data(cls.rebin_matrix,
                                [cls.new_channels[0],cls.new_channels[-1]],
                                cls.new_edges,
                                dummy_cross,dummy_cross_err,
                                freq_bins=cls.cross_freqs,
                                freq_grid=np.linspace(0.2,1.0,1000))
        
        #set the objects to test noticing/ignoring ranges         
        cls.test_select = FitCrossSpectrum()
        cls.test_select.set_coordinates("lags")
        cls.test_select.set_product_dependence("energy")
        
        cls.dummy_data = np.array([[1,2,3,4,5],
                                   [6,7,8,9,10],
                                   [11,12,13,14,15],
                                   [16,17,18,19,20]]).flatten()
        dummy_err = 0*cls.dummy_data.flatten()
 
        cls.test_select.set_data(cls.rebin_matrix,
                                 [cls.new_channels[0],cls.new_channels[-1]],
                                 cls.new_edges,
                                 cls.dummy_data,dummy_err,
                                 freq_bins=cls.cross_freqs,
                                 freq_grid=np.linspace(0.2,1.0,1000)) 
        return 
 
    def test_cross_eval(self):
        cross_model = LM_Model(cross_const,independent_vars=['energs','freqs'])
        cross_pars = LM_Parameters()
 
        self.test_cross.set_model(cross_model,model_type="cross")
        self.test_cross.set_params(cross_pars)
        test_model = self.test_cross.eval_model(fold=False)
        assert(np.allclose(test_model,self.test_cross.data,rtol=5e-3))
        
    def test_select_bins(self):
        #test ignore frequencies:
        self.test_select.ignore_frequencies(0,0.4)
        self.test_select.ignore_frequencies(0.8,1.0)
        known_data = np.array([6,7,8,9,10,11,12,13,14,15])
        assert(np.allclose(known_data,self.test_select.data))
 
        #test notice frequencies
        self.test_select.notice_frequencies(0,0.4)
        self.test_select.notice_frequencies(0.8,1.2)
        assert(np.allclose(self.dummy_data,self.test_select.data))
 
        #test ignore energies
        self.test_select.ignore_energies(0,self.new_edges[1])
        self.test_select.ignore_energies(self.new_edges[-2],self.new_edges[-1])
        known_data = np.array([2,3,4,7,8,9,12,13,14,17,18,19])
        assert(np.allclose(known_data,self.test_select.data))
 
        #test notice energies
        self.test_select.notice_energies(0,self.new_edges[1])
        self.test_select.notice_energies(self.new_edges[-2],self.new_edges[-1])
        assert(np.allclose(self.dummy_data,self.test_select.data))
 
        #test both together
        self.test_select.ignore_frequencies(0,0.4)
        self.test_select.ignore_frequencies(0.8,1.0)
        self.test_select.ignore_energies(0,self.new_edges[1])
        self.test_select.ignore_energies(self.new_edges[-2],self.new_edges[-1])
        known_data = np.array([7,8,9,12,13,14])
        assert(np.allclose(known_data,self.test_select.data))
 
    def test_cross_setup(self):
        #test that the class does not allow non-supported coordinates or 
        #unit dependences 
        with pytest.raises(TypeError):
            self.test_cross.set_product_dependence("wrong")
        with pytest.raises(TypeError):
            self.test_cross.set_coordinates("wrong")
        #test that the class does not allow data to be loaded without first 
        #stating the units and dependence of the data 
        with pytest.raises(AttributeError):
            self.test_cross.units = None
            self.test_cross.set_data(self.rebin_matrix,
                                     [self.new_channels[0],self.new_channels[-1]],
                                     self.new_edges,
                                     self.dummy_data,self.dummy_data,
                                     freq_bins=self.cross_freqs,
                                     time_res=0.1,seg_size=10)          
        with pytest.raises(AttributeError):
            self.test_cross.dependence = None
            self.test_cross.set_data(self.rebin_matrix,
                                     [self.new_channels[0],self.new_channels[-1]],
                                     self.new_edges,
                                     self.dummy_data,self.dummy_data,
                                     freq_bins=self.cross_freqs,
                                     time_res=0.1,seg_size=10)  
        
    #test that weird things can't happen when loading frequency dependent data
    def test_cross_load_freq(self):
        self.test_cross.set_coordinates("polar")
        self.test_cross.set_product_dependence("frequency") 
        times = [0.5, 1.1, 2.2, 3.7]
        mjdref=58000.
        events = EventList(times, mjdref=mjdref)
 
        #test that when loading stingray events the class looks for the time 
        #resolution/segment size/normalization 
        with pytest.raises(ValueError):
            self.test_cross.set_data(self.rebin_matrix,
                                     [self.new_channels[0],self.new_channels[-1]],
                                     self.new_edges,
                                     events,
                                     time_res=None,seg_size=None,norm=None)          
        with pytest.raises(ValueError):
            self.test_cross.set_data(self.rebin_matrix,
                                     [self.new_channels[0],self.new_channels[-1]],
                                     self.new_edges,
                                     events,time_res=0.5,seg_size=None,norm=None)         
        with pytest.raises(ValueError):
            self.test_cross.set_data(self.rebin_matrix,
                                     [self.new_channels[0],self.new_channels[-1]],
                                     self.new_edges,
                                     events,
                                     time_res=0.5,seg_size=10.,norm=None)   
        #test that when loading arrays the class looks for the time and frequency 
        #grids 
        with pytest.raises(ValueError):
            self.test_cross.set_data(self.rebin_matrix,
                                     [self.new_channels[0],self.new_channels[-1]],
                                     self.new_edges,
                                     self.dummy_data,self.dummy_data)
        #check that the code does not allow incorrectly sized data to be loaded 
        with pytest.raises(AttributeError):
            self.test_select.set_data(self.rebin_matrix,
                                      [self.new_channels[0],self.new_channels[-1]],
                                      self.new_edges,
                                      self.dummy_data,self.dummy_data[1:-1],
                                      freq_bins=self.cross_freqs,
                                      time_res=0.1,seg_size=10)            
        with pytest.raises(AttributeError):
            self.test_select.set_data(self.rebin_matrix,
                                      [self.new_channels[0],self.new_channels[-1]],
                                      self.new_edges,
                                      self.dummy_data,self.dummy_data,
                                      freq_bins=self.cross_freqs[:-1],
                                      time_res=0.1,seg_size=10)  
        with pytest.raises(AttributeError):
            reduced_data = self.dummy_data[:int(len(self.dummy_data)/2)]
            self.test_select.set_coordinates("lags")          
            self.test_select.set_data(self.rebin_matrix,
                                      [self.new_channels[0],self.new_channels[-1]],
                                      self.new_edges,
                                      reduced_data,reduced_data,
                                      freq_bins=self.cross_freqs[:-1],
                                      time_res=0.1,seg_size=10)    
        #self.test_select.set_coordinates("polar")     
 
    #same as above but with energy dependent data             
    def test_cross_load_energ(self):
        self.test_cross.set_coordinates("polar")         
        self.test_cross.set_product_dependence("energy")   
        #test that when loading arrays the class looks for the time and frequency 
        #grids 
        with pytest.raises(AttributeError):
            self.test_cross.set_data(self.rebin_matrix,
                                     [self.new_channels[0],self.new_channels[-1]],
                                     self.new_edges,
                                     self.dummy_data,self.dummy_data)     
        with pytest.raises(ValueError):
            self.test_cross.set_data(self.rebin_matrix,
                                     [self.new_channels[0],self.new_channels[-1]],
                                     self.new_edges,
                                     self.dummy_data,self.dummy_data,
                                     freq_bins=self.cross_freqs)             
        #check that the code does not allow incorrectly sized data to be loaded 
        with pytest.raises(AttributeError):
            self.test_cross.set_data(self.rebin_matrix,
                                      [self.new_channels[0],self.new_channels[-1]],
                                      self.new_edges,
                                      self.dummy_data,self.dummy_data[1:-1],
                                      freq_bins=self.cross_freqs,
                                      time_res=0.1,seg_size=10)            
        with pytest.raises(AttributeError):
            self.test_cross.set_data(self.rebin_matrix,
                                      [self.new_channels[0],self.new_channels[-1]],
                                      self.new_edges,
                                      self.dummy_data,self.dummy_data,
                                      freq_bins=self.cross_freqs[:-1],
                                      time_res=0.1,seg_size=10)  
        with pytest.raises(AttributeError):
            reduced_data = self.dummy_data[:int(len(self.dummy_data)/2)]
            self.test_cross.set_coordinates("lags")          
            self.test_cross.set_data(self.rebin_matrix,
                                      [self.new_channels[0],self.new_channels[-1]],
                                      self.new_edges,
                                      reduced_data,reduced_data,
                                      freq_bins=self.cross_freqs[:-1],
                                      time_res=0.1,seg_size=10)             
            
    #check that the class raises an error if trying to define a weird model type 
    #and that users are prevented from hard-coding unsupported model types and 
    #model coordinates 
    def test_cross_model_type(self): 
        self.test_cross.set_coordinates("polar")
        self.test_cross.set_product_dependence("energy")
        cross_model = LM_Model(cross_const,independent_vars=['energs','freqs'])          
        with pytest.raises(AttributeError):
            self.test_cross.set_model(cross_model,model_type="spectral")            
        with pytest.raises(AttributeError):    
            self.test_cross.set_model(cross_model,model_type="cross")            
            self.test_cross.model_type = None 
            test = self.test_cross.eval_model()
        with pytest.raises(AttributeError):    
            self.test_cross.set_model(cross_model,model_type="cross")              
            self.test_cross.dependence = "wrong" 
            test = self.test_cross.eval_model()
        with pytest.raises(AttributeError):    
            self.test_cross.set_model(cross_model,model_type="cross")              
            self.test_cross.set_product_dependence("frequency")
            self.test_cross.units = None
            test = self.test_cross.eval_model()            
        with pytest.raises(AttributeError):    
            self.test_cross.set_model(cross_model,model_type="cross")              
            self.test_cross.set_product_dependence("energy")
            self.test_cross.units = None
            test = self.test_cross.eval_model()                
            
    #test that when turning on phase+modulus normalization, the class adds 
    #the normalization parameters 
    def test_cross_renorm_params(self):                
        self.test_cross.set_coordinates("polar")
        self.test_cross.set_product_dependence("energy")          
        dummy_mods = np.ones((4,5))
        dummy_phase = np.ones((4,5))
        dummy_cross = np.append(dummy_mods.flatten(),dummy_phase.flatten())
        dummy_cross_err = 0*dummy_cross
        self.test_cross.set_data(self.rebin_matrix,
                                [self.new_channels[0],self.new_channels[-1]],
                                self.new_edges,
                                dummy_cross,dummy_cross_err,
                                freq_bins=self.cross_freqs,
                                freq_grid=np.linspace(0.2,1.0,1000))
        cross_model = LM_Model(cross_const,independent_vars=['energs','freqs'])
        cross_pars = LM_Parameters()
        self.test_cross.set_model(cross_model,model_type="cross")
        self.test_cross.set_params(cross_pars)            
        assert len(self.test_cross.model_params) == 0        
        self.test_cross.renorm_phases(True)    
        assert len(self.test_cross.model_params) == 4           
        self.test_cross.renorm_mods(True)    
        assert len(self.test_cross.model_params) == 8
        
    #test that the class doesn't calculate the likelihood if it is not defined 
    #correctly     
    def test_cross_likelihood(self):               
        with pytest.raises(AttributeError):
            self.test_cross.likelihood = "error"
            err = self.test_cross._minimizer(params=None)         

class TestFitTwoD(object):
 
    @classmethod
    def setup_class(cls):
        #set up a response, purely to test the response-dependent branch of 
        #set_data
        rmffile = os.getcwd()+"/ndspec/tests/data/xrt.rmf"
        arffile = os.getcwd()+"/ndspec/tests/data/xrt.arf"
        cls.response = ResponseMatrix(rmffile)
        cls.response.load_arf(arffile)
 
        #a minimal, valid 2d dataset and model, needed so that methods further 
        #down the pipeline (e.g. _minimizer) can be tested without also 
        #having to test the underlying model calculation
        cls.dummy_data = np.ones((3,4))
        cls.dummy_err = 0.1*np.ones((3,4))
        cls.column_grid = np.linspace(0,3,4)
        cls.row_grid = np.linspace(0,2,3)
 
        cls.test_twod_base = FitTwoD()
        cls.test_twod_base.set_data(cls.dummy_data,cls.dummy_err,
                                    cls.column_grid,cls.row_grid)
 
        twod_model = LM_Model(twod_ones_model,independent_vars=['x_axis','y_axis'])
        twod_pars = LM_Parameters()
        cls.test_twod_base.set_model(twod_model)
        cls.test_twod_base.set_params(twod_pars)
        
        #setup to test the energy dependent case
        #first build a sensible response 
        rebin_bounds = np.geomspace(0.5,10,20)
        rebin_bounds = np.append(np.min(cls.response.emin),rebin_bounds)
        rebin_bounds = np.append(rebin_bounds,np.max(cls.response.emax))
        rebin_bounds_lo = rebin_bounds[:-1]
        rebin_bounds_hi = rebin_bounds[1:]
        cls.rebin_response = cls.response.rebin_channels(rebin_bounds_lo,rebin_bounds_hi)
        
        #then build a pulsing BB model and get data from it
        ear_bins = np.append(cls.rebin_response.energ_lo,cls.rebin_response.energ_hi[-1])
        energ_bins = 0.5*(cls.rebin_response.energ_lo+cls.rebin_response.energ_hi)
        phase_grid = np.linspace(0,1,30)
        model_varbb = pulsing_bb(ear_bins,np.linspace(0,1,30),1,1,0.3)
        folded_varbb = cls.rebin_response.convolve_response(model_varbb)
        simulate_pulse = folded_varbb
        simulate_pulse_err = 0.05*folded_varbb
        pulse_model = LM_Model(pulsing_bb_eval,independent_vars=['y_axis','x_axis'])
        start_params = pulse_model.make_params(norm_bb=dict(value=1,min=0,max=10),
                                               kT=dict(value=1,min=0,max=10),
                                               norm_mod=dict(value=0.3,min=0,max=1),
                                               )       
        
        #set up the second fitter 
        cls.test_twod_energ = FitTwoD()
        cls.test_twod_energ.set_data(simulate_pulse,simulate_pulse_err,
                                     column_grid=phase_grid,row_grid=rebin_bounds,
                                     response=cls.rebin_response)
        cls.test_twod_energ.set_model(pulse_model)
        cls.test_twod_energ.set_params(start_params)        
        return
 
    #test that the class does not allow data, error, and grids of mismatched 
    #sizes to be loaded
    def test_set_data_errors(self):
        #data and its error must have the same shape
        with pytest.raises(AttributeError):
            wrong_err = np.ones((2,4))
            self.test_twod_base.set_data(self.dummy_data,wrong_err,
                                         self.column_grid,self.row_grid)
        #the column grid must match the number of columns in the data
        with pytest.raises(AttributeError):
            wrong_columns = np.linspace(0,3,3)
            self.test_twod_base.set_data(self.dummy_data,self.dummy_err,
                                         wrong_columns,self.row_grid)
        #the row grid must match the number of rows in the data, if no 
        #response is loaded
        with pytest.raises(AttributeError):
            wrong_rows = np.linspace(0,2,2)
            self.test_twod_base.set_data(self.dummy_data,self.dummy_err,
                                         self.column_grid,wrong_rows)
        #the row grid (i.e. the ear array) must have one more bin edge than 
        #the number of rows in the data, if a response is loaded
        with pytest.raises(AttributeError):
            wrong_ear = np.linspace(0,2,3)
            self.test_twod_base.set_data(self.dummy_data,self.dummy_err,
                                         self.column_grid,wrong_ear,
                                         response=self.response)
        #noise and data must have the same shape
        with pytest.raises(AttributeError):
            wrong_noise = np.ones((2,4))
            self.test_twod_base.set_data(self.dummy_data,self.dummy_err,
                                         self.column_grid,self.row_grid,
                                         noise=wrong_noise,noise_err=wrong_noise)
 
    #test that the class does not allow an unsupported fit statistic to be set
    def test_set_fit_statistic_errors(self):
        with pytest.raises(ValueError):
            self.test_twod_base.set_fit_statistic("wrong")
 
    #test that the ignore/notice methods require float or integer bounds
    def test_ignore_notice_errors(self):
        with pytest.raises(TypeError):
            self.test_twod_base.ignore_columns("wrong",1)
        with pytest.raises(TypeError):
            self.test_twod_base.notice_columns("wrong",1)
        with pytest.raises(TypeError):
            self.test_twod_base.ignore_rows("wrong",1)
        with pytest.raises(TypeError):
            self.test_twod_base.notice_rows("wrong",1)
 
    #test that the class doesn't calculate the likelihood if it is not 
    #defined correctly
    def test_twod_base_likelihood(self):
        with pytest.raises(AttributeError):
            self.test_twod_base.likelihood = "error"
            err = self.test_twod_base._minimizer(params=None)

    #test that the model and residuals are calculated correctly in the energy 
    #independent case
    def test_twod_base_residuals(self):
        test = self.test_twod_base.eval_model()
        assert np.allclose(test,np.ones(12)) == True
        test, _ = self.test_twod_base.get_residuals(res_type="chisq")
        assert np.allclose(test,np.zeros(12)) == True
        test, _ = self.test_twod_base.get_residuals(res_type="ratio")
        assert np.allclose(test,np.ones(12)) == True
        
    #now do the same in the energy independent case
    def test_twod_energ_residuals(self):
        test, _ = self.test_twod_energ.get_residuals(res_type="chisq")
        assert np.allclose(test,np.zeros(630)) == True
        test, _ = self.test_twod_energ.get_residuals(res_type="ratio")
        assert np.allclose(test,np.ones(630)) == True

class TestJointFit(object):
    @classmethod
    def setup_class(cls):
        #a minimal power spectrum fitter
        dummy_psd = np.ones(10)
        freqs = np.linspace(1,10,10)
        cls.psd_fit = FitPowerSpectrum()
        cls.psd_fit.set_data(dummy_psd,0.1*dummy_psd,freqs)
        psd_model = LM_Model(ones_model)
        psd_pars = LM_Parameters()
        psd_pars.add_many(('len', 10, False, 1, 20))
        cls.psd_fit.set_model(psd_model)
        cls.psd_fit.set_params(psd_pars)
        
        #response used for the energy dependant tests 
        rmffile = os.getcwd()+"/ndspec/tests/data/xrt.rmf"
        arffile = os.getcwd()+"/ndspec/tests/data/xrt.arf"
        test_response = ResponseMatrix(rmffile)
        test_response.load_arf(arffile)

        #a minimal time-averaged spectrum fitter
        cls.spec_fit = FitTimeAvgSpectrum()
        cls.spec_fit.set_data(test_response,os.getcwd()+"/ndspec/tests/data/xrt.fak")
        spec_model = LM_Model(ones_model)
        spec_pars = LM_Parameters()
        spec_pars.add_many(('len', 2400, False, 1, 3000))
        cls.spec_fit.set_model(spec_model,params=spec_pars)

        #a second time-averaged spectrum fitter, used only to test that the class  
        #rejects mismatched shared grid if models are also not shared 
        cls.spec_fit2 = FitTimeAvgSpectrum()
        cls.spec_fit2.set_data(test_response,os.getcwd()+"/ndspec/tests/data/xrt.fak")
        spec_model2 = LM_Model(zeros_model)
        spec_pars2 = LM_Parameters()
        spec_pars2.add_many(('len', 2400, False, 1, 3000))
        cls.spec_fit2.set_model(spec_model2,params=spec_pars2)
        
        #a minimal two-dimensional fitter to check plotting errors only 
        dummy_data = np.ones((3,4))
        dummy_err = 0.1*np.ones((3,4))
        column_grid = np.linspace(0,3,4)
        row_grid = np.linspace(0,2,3)

        cls.twod_fit = FitTwoD()
        cls.twod_fit.set_data(dummy_data,dummy_err,column_grid,row_grid)

        twod_model = LM_Model(twod_ones_model,independent_vars=['x_axis','y_axis'])
        twod_pars = LM_Parameters()
        cls.twod_fit.set_model(twod_model)
        cls.twod_fit.set_params(twod_pars)
        return 

    #test that only Fit... objects (or lists thereof) can be added to a joint fit
    def test_add_fitobj_errors(self):
        joint = JointFit()
        with pytest.raises(TypeError):
            joint.add_fitobj(np.ones(3),"wrong")
        with pytest.raises(TypeError):
            joint.add_fitobj([self.psd_fit,np.ones(3)],["psd","wrong"])

    #test that residuals throw errors if the fitters/arguments are poorly defined
    def test_get_residuals_errors(self):
        joint = JointFit()
        with pytest.raises(AttributeError):
            JointFit().get_residuals()
        joint.add_fitobj(self.psd_fit,"psd")
        with pytest.raises(TypeError):
            joint.get_residuals(names=123)

    #test that shared energy grids throw errors when the wrong fitters are passed,
    #or when bad grid bounds are passed 
    def test_set_energy_grid_errors(self):
        joint = JointFit()
        joint.add_fitobj(self.psd_fit,"psd")
        with pytest.raises(AttributeError):
            joint.set_energy_grid(np.linspace(0.1,10,50))
        joint = JointFit()
        joint.add_fitobj(self.spec_fit,"spec")
        low = self.spec_fit.energs[0]
        high = self.spec_fit.energs[-1]
        with pytest.raises(ValueError):
            newgrid = np.linspace(1.5*low,high,50)
            joint.set_energy_grid(newgrid)
        with pytest.raises(ValueError):
            newgrid = np.linspace(low,0.5*high,50)
            joint.set_energy_grid(newgrid)
        joint = JointFit()
        joint.add_fitobj(self.spec_fit,"spec1")
        joint.add_fitobj(self.spec_fit2,"spec2")
        newgrid = np.linspace(0.1*low,1.5*high,50)
        with pytest.raises(AttributeError):
            joint.set_energy_grid(newgrid)

    #test that parameters can only be set through lmfit
    def test_set_params_errors(self):
        joint = JointFit()
        joint.add_fitobj(self.psd_fit,"psd")
        with pytest.raises(AttributeError):
            joint.set_params(np.ones(3))

    #test that renorm_timeavg doesn't look for fitters that do not exist
    def test_renorm_timeavg_errors(self):
        joint = JointFit()
        joint.add_fitobj(self.spec_fit,"spec")
        with pytest.raises(AttributeError):
            joint.renorm_timeavg(True,names=["not_loaded"])

    #check that print model doesn't look for fitters that do not exist
    def test_print_models_errors(self):
        joint = JointFit()
        joint.add_fitobj(self.psd_fit,"psd")
        with pytest.raises(AttributeError):
            joint.print_models(names=["not_loaded"])
        with pytest.raises(AttributeError):
            joint.print_models(names="not_loaded")
 
    #test that the class tells users it can't jointly plot a fitter that 
    #hasn't been loaded and can't plot 1d and 2d fitters together
    def test_joint_plot_errors(self):
        joint = JointFit()
        joint.add_fitobj(self.psd_fit,"psd")
        with pytest.raises(ValueError):
            joint.joint_plot(units="wrong")
        with pytest.raises(AttributeError):
            joint.joint_plot(units="fpower",names=["not_loaded"])
        joint = JointFit()
        joint.add_fitobj(self.twod_fit,"twod")
        with pytest.raises(TypeError):
            joint.joint_plot(units="eeunfold")

class TestSimpleFit(object):
 
    @classmethod
    def setup_class(cls):
        #generic SimpleFit object to test the shared methods 
        cls.test_shared = SimpleFit()
 
        return 
 
    #test that users can't assign silly things to models/parameters                 
    def test_generic_setters(self):
        with pytest.raises(AttributeError):
            wrong_input = np.ones(1)
            self.test_shared.set_model(wrong_input)                
        with pytest.raises(AttributeError):
            wrong_input = np.ones(1)
            self.test_shared.set_params(wrong_input) 

