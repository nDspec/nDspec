import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))

import pytest
import warnings

import ndspec.Polarimetry as polarimetry
import ndspec.Models as models

class TestPolarimetry(object):

    #set up the energy grid, modulation angle grid, and the Stokes parameters
    #and polarization degree/angle to be used in the tests.
    @classmethod
    def setup_class(cls):
        cls.energs = np.linspace(2.,8.,25)
        cls.n_bins = len(cls.energs)
        cls.mod_angles = np.linspace(0.,np.pi,4000)
        cls.mod_factor = 0.35*np.ones(cls.n_bins)

        cls.stokes_I = models.powerlaw(cls.energs,np.array([1.,-2.]))
        cls.pol_degree = 0.15*np.ones(cls.n_bins)
        cls.pol_angle = np.radians(30.)*np.ones(cls.n_bins)
        cls.stokes_Q = cls.stokes_I*cls.pol_degree*np.cos(2.*cls.pol_angle)
        cls.stokes_U = cls.stokes_I*cls.pol_degree*np.sin(2.*cls.pol_angle)
        return

    def stokes_setup(self):
        product = polarimetry.PolarimetryProduct(self.energs,
                                                 input_type='stokes')
        product.set_stokes(self.stokes_I,self.stokes_Q,self.stokes_U)
        return product

    def polarization_setup(self):
        product = polarimetry.PolarimetryProduct(self.energs,
                                                 input_type='polarization')
        product.set_polarization(self.stokes_I,self.pol_degree,self.pol_angle)
        return product

    #test that the code returns the appropriate error if users provide an
    #unsupported input type, and that the bins are stored correctly otherwise
    def test_polarimetry_init(self):
        with pytest.raises(ValueError):
            product = polarimetry.PolarimetryProduct(self.energs,
                                                     input_type='wrong')
        product = polarimetry.PolarimetryProduct(self.energs,
                                                 input_type='stokes')
        assert(product.n_bins == self.n_bins)
        assert(np.allclose(product.bins,self.energs))

    #test that each setter method can only be called on a product that was
    #initialized with the matching input type
    def test_setter_input_type(self):
        with pytest.raises(ValueError):
            product = polarimetry.PolarimetryProduct(self.energs,
                                                     input_type='polarization')
            product.set_stokes(self.stokes_I,self.stokes_Q,self.stokes_U)
        with pytest.raises(ValueError):
            product = polarimetry.PolarimetryProduct(self.energs,
                                                     input_type='stokes')
            product.set_polarization(self.stokes_I,self.pol_degree,
                                     self.pol_angle)

    #test that the code only stores input arrays defined over the same number
    #of bins as the object
    def test_input_array_size(self):
        product = polarimetry.PolarimetryProduct(self.energs,
                                                 input_type='stokes')
        with pytest.raises(ValueError):
            product.set_stokes(self.stokes_I[1:],self.stokes_Q,self.stokes_U)
        with pytest.raises(ValueError):
            product.set_stokes(self.stokes_I,self.stokes_Q[1:],self.stokes_U)
        with pytest.raises(ValueError):
            product.set_stokes(self.stokes_I,self.stokes_Q,self.stokes_U[1:])
        product = polarimetry.PolarimetryProduct(self.energs,
                                                 input_type='polarization')
        with pytest.raises(ValueError):
            product.set_polarization(self.stokes_I,self.pol_degree[1:],
                                     self.pol_angle)
        with pytest.raises(ValueError):
            product.set_polarization(self.stokes_I,self.pol_degree,
                                     self.pol_angle[1:])

    #test that the modulation factor is only stored if it is either a scalar,
    #or defined over every bin in the object
    def test_modulation_factor_size(self):
        product = self.stokes_setup()
        product.set_modulation_factor(0.35)
        assert(product.mod_factor.size == 1)
        #mod_factor is defined at initialization, so it has length 35
        product.set_modulation_factor(self.mod_factor)
        assert(product.mod_factor.size == self.n_bins)
        with pytest.raises(ValueError):
            product.set_modulation_factor(self.mod_factor[1:])

    #test that the code does not compute any product before the quantities it
    #depends on have been set by the user
    def test_products_require_input(self):
        product = polarimetry.PolarimetryProduct(self.energs,
                                                 input_type='stokes')
        with pytest.raises(ValueError):
            test = product.stokes_to_polarization()
        with pytest.raises(ValueError):
            test = product.stokes_to_modulation()
        #the modulation curve additionally requires the modulation angles and
        #factors`
        product.set_stokes(self.stokes_I,self.stokes_Q,self.stokes_U)
        with pytest.raises(ValueError):
            test = product.stokes_to_modulation()
        product.set_modulation_angles(self.mod_angles)
        with pytest.raises(ValueError):
            test = product.stokes_to_modulation()

        product = polarimetry.PolarimetryProduct(self.energs,
                                                 input_type='polarization')
        with pytest.raises(ValueError):
            test = product.polarization_to_stokes()
        with pytest.raises(ValueError):
            test = product.polarization_to_modulation()
        product.set_polarization(self.stokes_I,self.pol_degree,self.pol_angle)
        with pytest.raises(ValueError):
            test = product.polarization_to_modulation()

    #test that the Stokes parameters are converted correctly into polarization 
    #degree and angle
    def test_stokes_to_polarization(self):
        product = self.stokes_setup()
        pol_degree, pol_angle = product.stokes_to_polarization()
        assert(np.allclose(pol_degree,self.pol_degree))
        assert(np.allclose(pol_angle,self.pol_angle))

    #test that the polarization degree and angle are converted into the Stokes
    #parameters correctly 
    def test_polarization_to_stokes(self):
        product = self.polarization_setup()
        stokes_I, stokes_Q, stokes_U = product.polarization_to_stokes()
        assert(np.allclose(stokes_I,self.stokes_I))
        assert(np.allclose(stokes_Q,self.stokes_Q))
        assert(np.allclose(stokes_U,self.stokes_U))

    #test that converting from Stokes parameters to polarization degree/angle
    #and back returns the initial input. 
    def test_conversion_roundtrip(self):
        product = self.stokes_setup()
        pol_degree, pol_angle = product.stokes_to_polarization()
        roundtrip = polarimetry.PolarimetryProduct(self.energs,
                                                   input_type='polarization')
        roundtrip.set_polarization(product.stokes_I,pol_degree,pol_angle)
        stokes_I, stokes_Q, stokes_U = roundtrip.polarization_to_stokes()
        assert(np.allclose(stokes_I,self.stokes_I))
        assert(np.allclose(stokes_Q,self.stokes_Q))
        assert(np.allclose(stokes_U,self.stokes_U))

    #test that the modulation curve is identical whether it is computed from
    #the Stokes parameters or from the polarization degree and angle
    def test_modulation_consistency(self):
        stokes_product = self.stokes_setup()
        stokes_product.set_modulation_angles(self.mod_angles)
        stokes_product.set_modulation_factor(self.mod_factor)
        stokes_curve = stokes_product.stokes_to_modulation()

        polar_product = self.polarization_setup()
        polar_product.set_modulation_angles(self.mod_angles)
        polar_product.set_modulation_factor(self.mod_factor)
        polar_curve = polar_product.polarization_to_modulation()

        assert(np.shape(stokes_curve) == (self.n_bins,len(self.mod_angles)))
        assert(np.allclose(stokes_curve,polar_curve))

    #test that the modulation curve has the amplitude and normalization
    #expected from the polarization state used to compute it. 
    def test_modulation_amplitude(self):
        product = self.polarization_setup()
        product.set_modulation_angles(self.mod_angles)
        product.set_modulation_factor(self.mod_factor)
        curve = product.polarization_to_modulation()

        curve_max = np.max(curve,axis=1)
        curve_min = np.min(curve,axis=1)
        amplitude = (curve_max-curve_min)/(curve_max+curve_min)
        assert(np.allclose(amplitude,self.mod_factor*self.pol_degree))
        #the cosine averages out over a full period, leaving only Stokes I
        assert(np.allclose(np.mean(curve,axis=1),self.stokes_I/(2.*np.pi),))

    #test that rotating the Stokes parameters shifts the polarization angle by
    #the rotation angle, and leaves the polarization degree unchanged
    def test_rotate_polarization(self):
        product = self.stokes_setup()
        rotation = 25.
        model = product.rotate_polarization(rotation)
        assert(np.shape(model) == (3,self.n_bins))
        pol_degree, pol_angle = product.stokes_to_polarization()
        assert(np.allclose(pol_degree,self.pol_degree))
        assert(np.allclose(pol_angle,self.pol_angle+np.radians(rotation)))
        #Stokes I sets the absolute scale and is unaffected by the rotation
        assert(np.allclose(model[0],self.stokes_I))

    #test that rotating twice by the same angle is identical to rotating once
    #by twice that angle, and that a rotation by 180 degrees returns the
    #initial Stokes parameters
    def test_rotation_is_additive(self):
        product = self.stokes_setup()
        product.rotate_polarization(20.)
        product.rotate_polarization(20.)
        reference = self.stokes_setup()
        reference.rotate_polarization(40.)
        assert(np.allclose(product.stokes_Q,reference.stokes_Q))
        assert(np.allclose(product.stokes_U,reference.stokes_U))

        product = self.stokes_setup()
        product.rotate_polarization(180.)
        assert(np.allclose(product.stokes_Q,self.stokes_Q))
        assert(np.allclose(product.stokes_U,self.stokes_U))

    #test that the code only rotates by a single angle, and only if the Stokes
    #parameters have been set first
    def test_rotation_errors(self):
        product = polarimetry.PolarimetryProduct(self.energs,
                                                 input_type='stokes')
        with pytest.raises(ValueError):
            product.rotate_polarization(25.)
        product.set_stokes(self.stokes_I,self.stokes_Q,self.stokes_U)
        with pytest.raises(ValueError):
            product.rotate_polarization(np.array([25.,45.]))

    #test that the same rotation applied through the pol_rotation model
    #returns the same Stokes parameters as the method itself
    def test_rotation_model(self):
        seed = np.array([self.stokes_I,self.stokes_Q,self.stokes_U])
        model = models.pol_rotation(seed,np.array([25.]))
        product = self.stokes_setup()
        reference = product.rotate_polarization(25.)
        assert(np.allclose(model,reference))

    #test that the code does not attempt to plot any quantity before it has
    #been computed
    def test_plot_errors(self):
        product = polarimetry.PolarimetryProduct(self.energs,
                                                 input_type='stokes')
        with pytest.raises(ValueError):
            product.plot_stokes()
        with pytest.raises(ValueError):
            product.plot_polarization_1d()
        with pytest.raises(ValueError):
            product.plot_polarization_slice()
        with pytest.raises(ValueError):
            product.plot_modulation()
        #the modulation curve can only be plotted once it has been computed,
        #even if the Stokes parameters themselves are already set
        product.set_stokes(self.stokes_I,self.stokes_Q,self.stokes_U)
        with pytest.raises(ValueError):
            product.plot_modulation()
