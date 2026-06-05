import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))

from pyfftw.interfaces.numpy_fft import (
    fft,
    fftfreq,
)
import pytest
import warnings
from astropy.io import fits
from scipy.interpolate import interp1d

import ndspec.Timing as timing 
import ndspec.Models as models

class TestTiming(object):

    #set up energy, time, frequency grids and models to be used in the tests
    @classmethod
    def setup_class(cls):
        cls.time_res_fft = 2500
        cls.times_fft = np.linspace(1.,1.e3,cls.time_res_fft)
        time_bin = np.diff(cls.times_fft)[0]
        freqs_all = fftfreq(cls.time_res_fft,time_bin)
        cls.freqs_fft = freqs_all[freqs_all>0]
 
        sinc_tstart = np.log10(cls.times_fft[0])
        sinc_tend = np.log10(cls.times_fft[-1])
        sinc_fstart = np.log10(cls.freqs_fft[0])
        sinc_fend = np.log10(cls.freqs_fft[-1])
        cls.times_sinc = np.logspace(sinc_tstart,sinc_tend,500)
        cls.freqs_sinc = np.logspace(sinc_fstart,sinc_fend,200)
        
        rmffile = "docs/data/nicer-rmf6s-teamonly-array50.rmf"
        with fits.open(rmffile, memmap=False) as response:
            extnames = np.array([h.name for h in response])
            h = response["MATRIX"]
            data = h.data
            energ_lo = np.array(data.field("ENERG_LO"))
            energ_hi = np.array(data.field("ENERG_HI"))
            cls.energies = 0.5*(energ_lo+energ_hi)

        cls.sin_wave = np.sin(cls.times_sinc)

        cls.lorentz_fft = models.lorentz(cls.freqs_fft,
                                         np.array([0.0075,0.1,0.1])) + \
                          models.lorentz(cls.freqs_fft,
                                         np.array([0.05,0.1,0.1]))               
        cls.lorentz_sinc = models.lorentz(cls.freqs_sinc,
                                         np.array([0.0075,0.1,0.1])) + \
                          models.lorentz(cls.freqs_sinc,
                                         np.array([0.05,0.1,0.1]))                     

        cls.bbflash_fft = models.bbody_bkn(cls.times_fft,
                                           cls.energies,
                                           np.array([1.,0.5,1.,-1.5,3.,-0.05]))
        cls.bbflash_sinc = models.bbody_bkn(cls.times_sinc,
                                            cls.energies,
                                            np.array([1.,0.5,1.,-1.5,3.,-0.05]))                                           
        
        cls.gauss_fft = models.gauss_bkn(cls.times_fft,cls.energies,
                                         np.array([3.e-3,2.,6.5,2.,-3.,4.,-0.25]))
        cls.gauss_sinc = models.gauss_bkn(cls.times_sinc,cls.energies,
                                          np.array([3.e-3,2.,6.5,2.,-3.,4.,-0.25]))           
        
        cls.pivoting_fft = models.pivoting_pl(cls.freqs_fft,cls.energies,
                           np.array([1.,-1.9,0.2,-0.05,-0.5,-0.05,1e-3]))
        cls.pivoting_sinc = models.pivoting_pl(cls.freqs_sinc,cls.energies,
                            np.array([1.,-1.9,0.2,-0.05,-0.5,-0.05,1e-3]))                                   
        return 

    #check that the code returns the appropriate errors if users provide incorrect
    #grid formats for time and frequency        
    def test_powerspectrum_init(self):
        with pytest.raises(ValueError):
            wrong_times = np.array([0,1,2,3,4,6,5,7,8,9,10])
            powerspectrum = timing.PowerSpectrum(times=wrong_times)
        with pytest.raises(TypeError):
            wrong_frequencies = np.array(([3,2],[1,3]))
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = wrong_frequencies,
                                              method='sinc')
        with pytest.raises(ValueError):
            wrong_frequencies = np.array([0,1,2,3,4,6,5,7,8,9,10])
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = wrong_frequencies,
                                              method='sinc')
        with pytest.warns(UserWarning):
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                                 freqs = self.freqs_sinc,
                                                 method='wrong',
                                                 verbose=True)            
        with pytest.raises(AttributeError):
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                                 freqs = self.freqs_sinc,
                                                 method='sinc')
            powerspectrum.method='wrong'
            powerspectrum.power_spec=self.lorentz_sinc
            powerspectrum.rebin_frequency(self.freqs_sinc)

    #check that the code doesn't compute the sinc decomposition if the sinc 
    #matrix somehow is not present
    def test_sinc_exists(self):
        with pytest.raises(AttributeError):
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              method='sinc')
            del powerspectrum.irf_sinc_arr 
            powerspectrum.compute_psd(self.sin_wave)   

    #check that the code doesn't randomly attempt a fft if no method is defined            
    def test_method_exists(self):                                    
        with pytest.raises(AttributeError):
            powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              method='sinc')
            del powerspectrum.method 
            powerspectrum.compute_psd(self.sin_wave)  

    #check that the code only computes the psd if the input is of the appropriate
    #size             
    def test_psd_array_size(self):
        with pytest.raises(TypeError):
            powerspectrum = timing.PowerSpectrum(self.times_fft)
            powerspectrum.compute_psd(self.sin_wave)
  
    #check that the code does not extrapolate when rebinning frequencies           
    def test_interpolate_bounds(self):
        powerspectrum = timing.PowerSpectrum(self.times_sinc,
                                      freqs = self.freqs_sinc,
                                      method='sinc')
        powerspectrum.compute_psd(self.sin_wave)  
        new_grid_lo = 0.5*self.freqs_sinc
        new_grid_hi = 1.5*self.freqs_sinc
        with pytest.raises(ValueError):
            powerspectrum.rebin_frequency(new_grid_lo)
        with pytest.raises(ValueError):
            powerspectrum.rebin_frequency(new_grid_hi)

    #check that the crossspectrum fails to bild correctly when reading incorrect 
    #input formats
    def test_crossspectrum_init(self):
        with pytest.raises(ValueError):
            wrong_energs = np.array([0,1,2,3,4,6,5,7,8,9,10])
            crossspectrum = timing.CrossSpectrum(self.times_fft,
                                                 energ=wrong_energs)   
        with pytest.raises(ValueError):
            wrong_times = np.array([0,1,2,3,4,6,5,7,8,9,10])
            crossspectrum = timing.CrossSpectrum(times=wrong_times,
                                                 energ=self.energies)
        with pytest.raises(TypeError):
            wrong_frequencies = np.array(([3,2],[1,3]))
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = wrong_frequencies,
                                              energ=self.energies,
                                              method='sinc')
        with pytest.raises(ValueError):
            wrong_frequencies = np.array([0,1,2,3,4,6,5,7,8,9,10])
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = wrong_frequencies,
                                              energ=self.energies,
                                              method='sinc')
        with pytest.warns(UserWarning):
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              energ=self.energies,
                                              method='wrong',
                                              verbose=True)            
        with pytest.raises(AttributeError):
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              energ=self.energies,
                                              method='sinc')
            crossspectrum.set_impulse(self.bbflash_sinc)
            crossspectrum.set_reference_lc(self.bbflash_sinc[0,:])
            crossspectrum.set_psd_weights(self.lorentz_sinc)
            crossspectrum.cross_from_irf()
            crossspectrum.method='wrong'
            crossspectrum.rebin_frequency(self.freqs_sinc)
 
    #check that the input psd is the same format as the crossspectrum object       
    def test_set_psd_weights(self):
        with pytest.raises(TypeError):
            crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                              freqs = self.freqs_sinc,
                                              energ=self.energies,
                                              method='sinc')
            crossspectrum.set_psd_weights(self.lorentz_fft)

    #check that the input irf is not set it uses the wrong size       
    def test_set_impulse(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                      freqs = self.freqs_sinc,
                                      energ=self.energies,
                                      method='sinc')
        with pytest.raises(TypeError):
            crossspectrum.set_impulse(self.bbflash_sinc[1:,:]) 
        with pytest.raises(TypeError):    
            crossspectrum.set_impulse(self.bbflash_fft)

    #check that the input transfer function is not set it uses the wrong size                                          
    def test_set_transfer(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                      freqs = self.freqs_sinc,
                                      energ=self.energies,
                                      method='sinc')
        with pytest.raises(TypeError):
            crossspectrum.set_transfer(self.pivoting_sinc[1:,:]) 
        with pytest.raises(TypeError):    
            crossspectrum.set_transfer(self.pivoting_fft)

    #check that the input reference band is not set if it input incorrectly                                                    
    def test_set_reference(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                      freqs = self.freqs_sinc,
                                      energ=self.energies,
                                      method='sinc')
        with pytest.raises(AttributeError):
            crossspectrum.set_reference_energ([1.0,2.])         
        crossspectrum.set_impulse(self.bbflash_sinc)
        with pytest.raises(ValueError): 
            crossspectrum.set_reference_energ([300.,400.])  
        with pytest.raises(ValueError): 
            crossspectrum.set_reference_energ([1.0,1.0001])  

    #check that the code raises erros if users try to retrieve the cross spectrum 
    #before computing it correctly
    def test_cross_found(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                  freqs = self.freqs_sinc,
                                  energ=self.energies,
                                  method='sinc')
        with pytest.raises(AttributeError):
            test = crossspectrum.real()
        with pytest.raises(AttributeError):
            test = crossspectrum.imag()
        with pytest.raises(AttributeError):
            test = crossspectrum.mod()    
        with pytest.raises(AttributeError):
            test = crossspectrum.phase()          
        with pytest.raises(AttributeError):
            test = crossspectrum.lag()
 
    #check that oned spectra can not be retrieved if users input the energy or 
    #frequency ranges incorrectly             
    def test_oned_range_errors(self):
        crossspectrum = timing.CrossSpectrum(self.times_sinc,
                                             freqs = self.freqs_sinc,
                                             energ=self.energies,
                                             method='sinc')
        crossspectrum.set_impulse(self.bbflash_sinc)
        crossspectrum.set_reference_energ([self.energies[0],self.energies[-1]])
        crossspectrum.set_psd_weights(self.lorentz_sinc)
        with pytest.raises(ValueError):
            error = crossspectrum.lag_frequency(int_bounds=[1.0,1.0001])  
        with pytest.raises(ValueError):
            error = crossspectrum.lag_frequency(int_bounds=[300.,400.])  
        crossspectrum.cross_from_irf()
        with pytest.raises(ValueError):
            error = crossspectrum.phase_energy([self.freqs_sinc[4],
                                                self.freqs_sinc[2]])
  
    #set up a model for the tests below        
    def cross_model_setup(self):  
        self.rev_fft = timing.CrossSpectrum(self.times_fft,energ=self.energies)
        self.rev_fft.set_impulse(self.gauss_fft)
        self.rev_fft.set_psd_weights(self.lorentz_fft)
        self.rev_fft.set_reference_energ([7.,8.])
        self.rev_fft.cross_from_irf()
        
        self.rev_sinc = timing.CrossSpectrum(self.times_sinc,
                                            energ=self.energies,
                                            freqs=self.freqs_sinc,
                                            method='sinc')
        self.rev_sinc.set_impulse(self.gauss_sinc)
        self.rev_sinc.set_psd_weights(self.lorentz_sinc)
        self.rev_sinc.set_reference_energ([7.,8.])
        self.rev_sinc.cross_from_irf()  
        self.ener_lim = [6.,7.]
        self.lowf_lim  = [self.rev_fft.freqs[0],5.*self.rev_fft.freqs[0]]
        self.highf_lim  = [0.02*self.rev_fft.freqs[-1],0.07*self.rev_fft.freqs[-1]]      
    pass 

    #all the testes below check that the result of the fft and sinc methods are 
    #consistent with each other, when calculating on the same grid, using both 
    #oned frequency and energy dependent products.     
    #Numerics are complicated so here we only require a 5percent accuracy  
    def test_onedcross_frequency_lin_grid(self):
        self.cross_model_setup()     
        interp_real_to_lin = interp1d(self.rev_sinc.freqs,
                                      self.rev_sinc.real_frequency(self.ener_lim),
                                      fill_value='extrapolate')
        array_lin = interp_real_to_lin(self.rev_fft.freqs)
        indexes = np.where(np.logical_and(self.rev_fft.freqs>=self.lowf_lim[1],
                                          self.rev_fft.freqs<=self.highf_lim[1]))
        r_tol = 5e-2
        assert(np.allclose(self.rev_fft.real_frequency(self.ener_lim)[indexes],
                           array_lin[indexes],
                           rtol=r_tol))

        interp_imag_to_lin = interp1d(self.rev_sinc.freqs,
                                      self.rev_sinc.imag_frequency(self.ener_lim),
                                      fill_value='extrapolate')
        array_lin = interp_imag_to_lin(self.rev_fft.freqs)
        indexes = np.where(np.logical_and(self.rev_fft.freqs>=self.lowf_lim[1],
                                          self.rev_fft.freqs<=self.lowf_lim[1]))
        r_tol = 5e-2
        assert(np.allclose(self.rev_fft.imag_frequency(self.ener_lim)[indexes],
                           array_lin[indexes],
                           rtol=r_tol))

    def test_onedcross_frequency_log_grid(self):
        self.cross_model_setup()     
        interp_real_to_log = interp1d(self.rev_fft.freqs,
                                      self.rev_fft.real_frequency(self.ener_lim),
                                      fill_value='extrapolate')
        array_log = interp_real_to_log(self.rev_sinc.freqs)
        indexes = np.where(np.logical_and(self.rev_sinc.freqs>=self.lowf_lim[1],
                                          self.rev_sinc.freqs<=self.highf_lim[1]))
        r_tol = 5e-2
        assert(np.allclose(self.rev_sinc.real_frequency(self.ener_lim)[indexes],
                           array_log[indexes],
                           rtol=r_tol))

        interp_imag_to_log = interp1d(self.rev_fft.freqs,
                                      self.rev_fft.imag_frequency(self.ener_lim),
                                      fill_value='extrapolate')
        array_log = interp_imag_to_log(self.rev_sinc.freqs)
        indexes = np.where(np.logical_and(self.rev_sinc.freqs>=self.lowf_lim[1],
                                          self.rev_sinc.freqs<=self.lowf_lim[1]))
        r_tol = 5e-2
        assert(np.allclose(self.rev_sinc.imag_frequency(self.ener_lim)[indexes],
                           array_log[indexes],
                           rtol=r_tol))
    
    def test_onedcross_energy_lin_grid_lowf(self):
        self.cross_model_setup()  
        interp_real_to_lin = interp1d(self.rev_sinc.energ,
                                      self.rev_sinc.real_energy(self.lowf_lim),
                                      fill_value='extrapolate')
        array_lin = interp_real_to_lin(self.rev_fft.energ)
        r_tol = 1e-2
        a_tol = 1.5e-4
        assert(np.allclose(array_lin,self.rev_fft.real_energy(self.lowf_lim),
                          rtol=r_tol,atol=a_tol))
        interp_imag_to_lin = interp1d(self.rev_sinc.energ,
                                      self.rev_sinc.imag_energy(self.lowf_lim),
                                      fill_value='extrapolate')
        array_lin = interp_imag_to_lin(self.rev_fft.energ)
        r_tol = 1e-2
        a_tol = 4e-6
        assert(np.allclose(array_lin,self.rev_fft.imag_energy(self.lowf_lim),
                          rtol=r_tol,atol=a_tol))        
                                             
    def test_onedcross_energy_lin_grid_highf(self):
        self.cross_model_setup()  
        interp_real_to_lin = interp1d(self.rev_sinc.energ,
                                      self.rev_sinc.real_energy(self.highf_lim),
                                      fill_value='extrapolate')
        array_lin = interp_real_to_lin(self.rev_fft.energ)
        r_tol = 1e-2
        a_tol = 1.5e-6
        assert(np.allclose(array_lin,self.rev_fft.real_energy(self.highf_lim),
                          rtol=r_tol,atol=a_tol))
        interp_imag_to_lin = interp1d(self.rev_sinc.energ,
                                      self.rev_sinc.imag_energy(self.highf_lim),
                                      fill_value='extrapolate')
        array_lin = interp_imag_to_lin(self.rev_fft.energ)
        r_tol = 1e-2
        a_tol = 1.5e-7
        assert(np.allclose(array_lin,self.rev_fft.imag_energy(self.highf_lim),
                          rtol=r_tol,atol=a_tol)) 
                          
    def test_onedcross_energy_log_grid_lowf(self):
        self.cross_model_setup()  
        interp_real_to_log = interp1d(self.rev_fft.energ,
                                      self.rev_fft.real_energy(self.lowf_lim),
                                      fill_value='extrapolate')
        array_log = interp_real_to_log(self.rev_sinc.energ)
        r_tol = 1e-2
        a_tol = 1.5e-4
        assert(np.allclose(array_log,self.rev_sinc.real_energy(self.lowf_lim),
                           rtol=r_tol,atol=a_tol))
        #then the imaginary
        interp_imag_to_log = interp1d(self.rev_fft.energ,
                                      self.rev_fft.imag_energy(self.lowf_lim),
                                      fill_value='extrapolate')
        array_log = interp_imag_to_log(self.rev_sinc.energ)
        r_tol = 1e-2
        a_tol = 4e-6
        print(np.allclose(array_log,self.rev_sinc.imag_energy(self.lowf_lim),
                          rtol=r_tol,atol=a_tol))
                          
    def test_onedcross_energy_log_grid_highf(self):
        self.cross_model_setup()  
        interp_real_to_log = interp1d(self.rev_fft.energ,
                                      self.rev_fft.real_energy(self.highf_lim),
                                      fill_value='extrapolate')
        array_log = interp_real_to_log(self.rev_sinc.energ)
        r_tol = 1e-2
        a_tol = 1.5e-6
        assert(np.allclose(array_log,self.rev_sinc.real_energy(self.highf_lim),
                           rtol=r_tol,atol=a_tol))
        #then the imaginary
        interp_imag_to_log = interp1d(self.rev_fft.energ,
                                      self.rev_fft.imag_energy(self.highf_lim),
                                      fill_value='extrapolate')
        array_log = interp_imag_to_log(self.rev_sinc.energ)
        r_tol = 1e-2
        a_tol = 1.5e-7
        print(np.allclose(array_log,self.rev_sinc.imag_energy(self.highf_lim),
                          rtol=r_tol,atol=a_tol))
