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

import pytest


def ones_model(len):
    return np.ones(len)

def cross_const(energs,freqs):
    n_energs = len(energs)
    n_freqs = len(freqs)
    model = np.ones((n_freqs,n_energs))
    return model

#I really would like to split this into testing each fitter separately 
class TestFitter(object):

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
        
        #set up data+fitter to test the PSD
        dummy_psd = np.ones(10)
        freqs = np.linspace(1,10,10)
        cls.test_psd = FitPowerSpectrum()
        cls.test_psd.set_data(dummy_psd,0.1*dummy_psd,freqs)
        
        #set up data+fitter to test the time averaged spectrum 
        cls.test_spec = FitTimeAvgSpectrum()
        cls.test_spec.set_data(cls.response,os.getcwd()+"/ndspec/tests/data/xrt.fak")
        
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
                                time_res=0.1,seg_size=10)
        
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
                                 time_res=0.1,seg_size=10)

        #generic SimpleFit object to test the shared methods 
        cls.test_shared = SimpleFit()
        
        return 
       
    def test_psd_eval(self):
        psd_model = LM_Model(ones_model)
        model_parameters = LM_Parameters()
        model_parameters.add_many(('len', 10, False, 1, 20))

        self.test_psd.set_model(psd_model)
        self.test_psd.set_params(model_parameters)
        test_residuals = self.test_psd.get_residuals("delchi")
        assert(np.allclose(test_residuals[0],np.zeros(10)))
     
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
        
    def test_cross_eval(self):
        cross_model = LM_Model(cross_const,independent_vars=['energs','freqs'])
        cross_pars = LM_Parameters()

        self.test_cross.set_model(cross_model,model_type="cross")
        self.test_cross.set_params(cross_pars)
        test_model = self.test_cross.eval_model(fold=False)
        assert(np.allclose(test_model,self.test_cross.data))
        
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
            err = self.test_psd._minimizer(params=None)
    
    def test_psd_plot_errors(self):
        #test that plots do not allow weird things to be rendered
        with pytest.raises(ValueError):
            self.test_psd.plot_data(units="wrong")
        with pytest.raises(ValueError):
            self.test_psd.plot_model(residuals="wrong")
        with pytest.raises(ValueError):
            self.test_psd.plot_model(units="wrong")
            
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
                                time_res=0.1,seg_size=10)
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

    #test that users can't assign silly things to models/parameters                 
    def test_generic_setters(self):
        with pytest.raises(AttributeError):
            wrong_input = np.ones(1)
            self.test_shared.set_model(wrong_input)                
        with pytest.raises(AttributeError):
            wrong_input = np.ones(1)
            self.test_shared.set_params(wrong_input) 
    
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
