import numpy as np
import copy

import scipy.sparse
 
from .Response import ResponseMatrix
from .Polarimetry import PolarimetryProduct
from .SimpleFit import SimpleFit, EnergyDependentFit, \
                       StokesDependentFit, load_stokes_pha
from .Likelihoods import cstat, chisq, ratio
from .Utils import model_expand
from . import Plotting

class FitSpectroPolarimetry(SimpleFit,EnergyDependentFit,
                            StokesDependentFit):
    """
    Least-chi squared fitter class for a spectro-polarimetric observation,
    defined as the Stokes I, Q and U count rate spectra as a function of photon
    channel energy bound.

    Given a pair of instrument responses (one for Stokes I, one for Stokes Q and
    U), the three Stokes spectra, their errors, and a model (defined in energy
    space) returning all three Stokes parameters, this class handles fitting
    internally using the lmfit library.

    The three Stokes parameters are always fitted simultaneously, but they are
    stored (and can be noticed or ignored) separately. This is necessary
    because Stokes Q and U are typically binned more coarsely than Stokes I.

    Attributes inherited from SimpleFit:
    ------------------------------------
    model: lmfit.CompositeModel
        A lmfit CompositeModel object, which contains a wrapper to the model
        component(s) one wants to fit to the data. For this fitter, it is 
        required to return an array of shape (3,len(energs)) containing the 
        Stokes I, Q and U photon spectra.

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
        An array storing the data to be fitted, in units of counts/s/keV. It
        contains the Stokes I, Q and U spectra, in this order, flattened to a
        single dimension and including only the channels noticed in the fit.

    data_err: np.array(float)
        An array containing the uncertainty on the data to be fitted, stored
        identically to the data attribute.

    noise: np.array(float) or None
        If loaded, an array containing the Stokes I, Q and U background spectra,
        stored identically to the data attribute.

    noise_err: np.array(float or None)
        If loaded, an array containing the error on the background counts,
        stored identically to the data attribute. Used to compute the fit
        statistic.

    _data_unmasked, _data_err_unmasked, _noise_unmasked: np.array(float)
        The arrays of every data bin, its error and (if loaded) the background,
        regardless of which ones are ignored or noticed during the fit.
        Used exclusively to enable book keeping internal to the fitter class.

    Attributes inherited from EnergyDependentFit:
    ---------------------------------------------
    energs: np.array(float)
        The array of physical photon energies over which the model is computed.
        Defined as the middle of each bin in the energy range stored in the
        instrument response provided.

    energ_bounds: np.array(float)
        The array of energy bin widths, for each bin over which the model is
        computed. Defined as the difference between the upper and lower bounds
        of the energy bins stored in the instrument response provided.

    ear: np.array(float)
        The array of energy bin bounds, for each bin over which the model is
        computed. Only necessary when calling Xspec models due to their unique
        input structure.

    ebounds: np.array(float)
        The array of energy channel bin centers for the Stokes I instrument
        energy channels, as stored in the instrument response provided. Only
        contains the channels that are noticed during the fit.

    ewidths: np.array(float)
        The array of energy channel bin widths for the Stokes I instrument
        energy channels, as stored in the instrument response provided. Only
        contains the channels that are noticed during the fit.

    ebounds_mask: np.array(bool)
        The array of Stokes I instrument energy channels that are either ignored
        or noticed during the fit. A given channel i is noticed if
        ebounds_mask[i] is True, and ignored if it is false.

    n_chans: int
        The number of Stokes I channels that are to be noticed during the fit.

    _all_chans: int
        The total number of channels in the loaded Stokes I response matrix.

    _emin_unmasked, _emax_unmasked, _ebounds_unmasked, _ewidths_unmasked: np.array(float)
        The array of every lower bound, upper bound, channel center and channel
        widths stored in the Stokes I response, regardless of which ones are
        ignored or noticed during the fit. Used exclusively to facilitate
        book-keeping internal to the fitter class.

    gain_params: lmfit.Parameters, default None 
        A lmfit Parameters object, which contains the parameters for the gain  
        correction model components if it is enabled. Defaults to None. 

    Attributes inherited from StokesDependentFit:
    ---------------------------------------------------
    pol_emin, pol_emax: np.array(float)
        The arrays of lower and upper energy channel bounds for the Stokes Q and
        U instrument energy channels. Only contain the channels that are noticed
        during the fit.

    pol_ebounds: np.array(float)
        The array of energy channel bin centers for the Stokes Q and U
        instrument energy channels. Only contains the channels that are noticed
        during the fit.

    pol_ewidths: np.array(float)
        The array of energy channel bin widths for the Stokes Q and U instrument
        energy channels. Only contains the channels that are noticed during the
        fit.

    pol_ebounds_mask: np.array(bool)
        The array of Stokes Q and U instrument energy channels that are either
        ignored or noticed during the fit. A given channel i is noticed if
        pol_ebounds_mask[i] is True, and ignored if it is false.

    n_pol_chans: int
        The number of Stokes Q (and U) channels that are to be noticed during
        the fit.

    n_bins: int
        The total number of data bins noticed during the fit, defined as
        n_chans+2*n_pol_chans.

    _all_pol_chans: int
        The total number of channels in the loaded Stokes Q/U response matrix.

    _all_bins: int
        The total number of data bins loaded, defined as
        _all_chans+2*_all_pol_chans.

    _pol_emin_unmasked, _pol_emax_unmasked, _pol_ebounds_unmasked, _pol_ewidths_unmasked: np.array(float)
        The array of every lower bound, upper bound, channel center and channel
        widths stored in the Stokes Q/U response, regardless of which ones are
        ignored or noticed during the fit. Used exclusively to facilitate
        book-keeping internal to the fitter class.

    Other attributes:
    -----------------
    response: nDspec.ResponseMatrix
        The instrument response matrix corresponding to the Stokes I spectrum to
        be fitted. It is required to define the energy grids over which model
        and data are defined.

    response_pol: nDspec.ResponseMatrix
        The instrument response matrix corresponding to the Stokes Q and U
        spectra to be fitted, rebinned over the (typically coarser) Stokes Q/U
        channel grid. For most polarimeters this is the modulation response
        function (MRF), ie the response weighted by the energy-dependent 
        modulation factor of the instrument.

    response_polgrid: nDspec.ResponseMatrix
        The Stokes I instrument response matrix, rebinned over the Stokes Q/U
        channel grid. It is only used to compute the polarization degree in
        detector space, which requires all three Stokes parameters to be defined
        over the same channel grid. The reason this is different from the above 
        is that response_pol contains the MRF, response_polgrid the ARF. 

    mod_factor: np.array(float) or None
        If the Stokes Q/U response was built by rescaling the Stokes I response,
        the array of modulation factors used to do so, defined over the photon
        energy grid of the response. If instead a modulation response function
        was loaded directly, it is set to None.

    exposure: float
        The exposure time of the observation. Only used for calculating
        Poisson-type likelihoods.

    _data_stokes_I_unmasked, _data_stokes_I_err_unmasked: np.array(float)
        The Stokes I data (and its error), background-subtracted if applicable,
        rebinned onto the Stokes Q/U channel grid and normalized to units of
        counts/s/keV. Computed once when the data is loaded, and re-used every
        time the polarization degree and angle of the data are needed.

    """

    def __init__(self,likelihood="chisq"):
        SimpleFit.__init__(self,likelihood)
        self.response = None
        self.response_pol = None
        self.response_polgrid = None
        self.mod_factor = None
        self.exposure = None
        self._data_stokes_I_unmasked = None
        self._data_stokes_I_err_unmasked = None
        self._supported_stokes = ["all","I","QU","Q","U"]
        pass
 
    def set_data(self,response,stokes_I,stokes_Q,stokes_U,response_pol=None,
                 mod_factor=None,background_I=None,background_Q=None,
                 background_U=None):
        """
        This method sets the Stokes I, Q and U data to be fitted, their errors,
        and the energy and channel grids, given three input spectra and their
        associated response matrices.
 
        The response to be applied to Stokes Q and U can be provided in one of
        two ways. Users can either pass a modulation response function directly,
        through the response_pol argument; or they can pass an array of 
        modulation factors through the mod_factor argument, in which case the 
        Stokes Q/U response is built internally by rescaling the effective area 
        of the Stokes I response by the modulation factor in each energy bin. 
        Exactly one of the two has to be provided.
 
        If the files provided were grouped with heatools, the method loads the
        grouped data and adjusts the channel grids automatically. Stokes Q and U
        are required to share the same channel grid, but this can (and typically
        will) be coarser than that of Stokes I. The data is assumed to be
        background-subtracted, unless the appropriate background files are
        provided.
 
        Parameters:
        -----------
        response: nDspec.ResponseMatrix
            The instrument response (including both rmf and arf) to be applied
            to Stokes I, loaded into a nDspec ResponseMatrix object.
 
        stokes_I, stokes_Q, stokes_U: str
            Strings pointing to the paths of the Stokes I, Q and U spectrum
            files, each stored in a type 1 OGIP-formatted file (such as the pha
            files produced by a typical polarimetry reduction pipeline).
 
        response_pol: nDspec.ResponseMatrix, default None
            The instrument response to be applied to Stokes Q and U, loaded into
            a nDspec ResponseMatrix object. For most polarimeters this is the
            modulation response function, ie the response weighted by the
            energy-dependent modulation factor of the instrument. It is required
            to be defined over the same photon energy grid as the Stokes I
            response. Mutually exclusive with mod_factor.
 
        mod_factor: float or np.array(float), default None
            The modulation factor of the instrument, either as a single value or
            as an array defined over the photon energy grid of the Stokes I
            response. If it is provided, the Stokes Q/U response is built by
            rescaling the Stokes I response by it. Mutually exclusive with
            response_pol.
 
        background_I, background_Q, background_U: str, default None
            Strings pointing to the paths of the Stokes I, Q and U background
            files, each stored in a type 1 OGIP-formatted file. If they are not
            provided, the software assumes the data is either already
            background-subtracted, or that the user wants to ignore or model the
            background themselves.
        """
 
        if not isinstance(response,ResponseMatrix):
            raise TypeError("Response must be an instance of nDspec.ResponseMatrix")
        if ((response_pol is None)&(mod_factor is None)):
            raise ValueError(("Please provide either a modulation response "
                              "function, or an array of modulation factors"))
        if ((response_pol is not None)&(mod_factor is not None)):
            raise ValueError(("Please provide either a modulation response "
                              "function, or an array of modulation factors, "
                              "but not both"))
 
        if mod_factor is not None:
            response_pol = self._build_modulation_response(response,mod_factor)
        else:
            if not isinstance(response_pol,ResponseMatrix):
                raise TypeError("Polarized response must be an instance of nDspec.ResponseMatrix")
            if len(response.energ_lo) != len(response_pol.energ_lo):
                raise ValueError("Stokes I and Q/U responses have different energy grids")
            if not np.allclose(response.energ_lo,response_pol.energ_lo):
                raise ValueError("Stokes I and Q/U responses have different energy grids")
 
        bounds_lo, bounds_hi, counts, error, exposure, src_backsc = \
            load_stokes_pha(stokes_I,response,stokes="I")
        bounds_q_lo, bounds_q_hi, counts_q, error_q, exposure_q, _ = \
            load_stokes_pha(stokes_Q,response_pol,stokes="Q")
        bounds_u_lo, bounds_u_hi, counts_u, error_u, exposure_u, _ = \
            load_stokes_pha(stokes_U,response_pol,stokes="U")
 
        if len(bounds_q_lo) != len(bounds_u_lo):
            raise ValueError("Stokes Q and U spectra have different channel grids")
        if ((np.allclose(bounds_q_lo,bounds_u_lo) is False)|
            (np.allclose(bounds_q_hi,bounds_u_hi) is False)):
            raise ValueError("Stokes Q and U spectra have different channel grids")
        if ((np.isclose(exposure,exposure_q) is False)|
            (np.isclose(exposure,exposure_u) is False)):
            raise ValueError("Stokes I, Q and U spectra have different exposures")
 
        #the Stokes I response is rebinned twice: once over the Stokes I channel
        #grid, which is used for the fit, and once over the Stokes Q/U channel
        #grid, which is only used to compute the polarization degree
        self.response = response.rebin_channels(bounds_lo,bounds_hi)
        self.response_pol = response_pol.rebin_channels(bounds_q_lo,bounds_q_hi)
        self.response_polgrid = response.rebin_channels(bounds_q_lo,bounds_q_hi)
        EnergyDependentFit.__init__(self)
        StokesDependentFit.__init__(self)
 
        #this loads the three Stokes spectra in units of counts/s/keV
        stokes_I_data = counts/exposure/self.ewidths
        stokes_Q_data = counts_q/exposure/self.pol_ewidths
        stokes_U_data = counts_u/exposure/self.pol_ewidths
        stokes_I_err = error/exposure/self.ewidths
        stokes_Q_err = error_q/exposure/self.pol_ewidths
        stokes_U_err = error_u/exposure/self.pol_ewidths
        self.data = np.concatenate((stokes_I_data,stokes_Q_data,stokes_U_data))
        self.data_err = np.concatenate((stokes_I_err,stokes_Q_err,stokes_U_err))
        self.exposure = exposure
 
        self._bounds_I = [bounds_lo, bounds_hi]
        self._bounds_pol = [bounds_q_lo, bounds_q_hi]
 
        self._set_unmasked_data()
 
        #the background has to be loaded after the unmasked arrays are set, as
        #it is stored over the channel grids of all three Stokes parameters.
        #if a Stokes I background was provided, its rebinned raw counts are
        #returned directly, rather than stored on self, since they are only
        #needed once more, to subtract them below
        bkg_counts_I, bkg_counts_var_I = None, None
        if ((background_I is not None)|(background_Q is not None)|
            (background_U is not None)):
            bkg_counts_I, bkg_counts_var_I = self._set_background(
                                             response,response_pol,src_backsc,
                                             background_I,background_Q,
                                             background_U)
 
        #save the Stokes I data on the (coarser) Q/U channel grid, which we 
        #need to convert the data stokes Q/U into polarization degree/angle
        self._compute_data_stokes_I(counts,error,bkg_counts_I,bkg_counts_var_I)
        return
 
    def _compute_data_stokes_I(self,counts_I,counts_I_err,bkg_counts_I=None,
                               bkg_counts_var_I=None):
        """
        This method rebins the Stokes I counts (and, if provided, the Stokes I
        background counts) from the Stokes I channel grid onto the coarser
        Stokes Q/U channel grid, and stores the result in units of counts/s/keV.
        This is required because the polarization degree and angle of the data
        can only be computed once all three Stokes parameters are defined over
        the same channels.
 
        Parameters:
        -----------
        counts_I, counts_I_err: np.array(float)
            The raw Stokes I source counts and their error, on the Stokes I
            channel grid.
 
        bkg_counts_I, bkg_counts_var_I: np.array(float) or None, default None
            The raw Stokes I background counts and their variance, already
            rebinned onto the Stokes I channel grid and scaled by the
            background scaling factor, as returned by _set_background. Left as
            None if no Stokes I background was loaded.
        """
 
        counts_I_var = counts_I_err**2
        if bkg_counts_I is not None:
            counts_I = counts_I - bkg_counts_I
            counts_I_var = counts_I_var+bkg_counts_var_I
        stokes_I = self.response._rebin_sum(counts_I,self._bounds_I,
                                            self._bounds_pol)
        stokes_I_var = self.response._rebin_sum(counts_I_var,self._bounds_I,
                                                self._bounds_pol)
        self._data_stokes_I_unmasked = stokes_I/self.exposure/ \
                                       self._pol_ewidths_unmasked
        self._data_stokes_I_err_unmasked = np.sqrt(stokes_I_var)/self.exposure/ \
                                           self._pol_ewidths_unmasked
        return
 
    def _build_modulation_response(self,response,mod_factor):
        """
        This method builds the response to be applied to the Stokes Q and U
        spectra, by rescaling the effective area of the Stokes I response by the
        modulation factor of the instrument in each photon energy bin. It is
        only used when users do not supply a modulation response function
        directly.
 
        Parameters:
        -----------
        response: nDspec.ResponseMatrix
            The instrument response to be applied to Stokes I, before any
            rebinning over the channel grid of the data.
 
        mod_factor: float or np.array(float)
            The modulation factor of the instrument, either as a single value or
            as an array defined over the photon energy grid of the response.
 
        Returns:
        --------
        response_pol: nDspec.ResponseMatrix
            The instrument response to be applied to Stokes Q and U.
        """
 
        mod_factor = np.atleast_1d(np.asarray(mod_factor,dtype=float))
        if mod_factor.size == 1:
            mod_factor = mod_factor[0]*np.ones(response.n_energs)
        elif mod_factor.size != response.n_energs:
            raise ValueError(("The modulation factor must either be a scalar, "
                              "or have the same size as the photon energy grid "
                              "of the response"))
        if ((np.min(mod_factor) < 0.)|(np.max(mod_factor) > 1.)):
            raise ValueError("The modulation factor must be between 0 and 1")
 
        #the modulation factor multiplies the effective area, so it is applied
        #to every channel of a given photon energy bin identically
        response_pol = copy.copy(response)
        response_pol.resp_matrix = (response.resp_matrix.T * scipy.sparse.diags(mod_factor)).T
        if response.has_arf is True:
            response_pol.specresp = response.specresp*mod_factor
        self.mod_factor = mod_factor
        return response_pol
 
    def _set_background(self,response,response_pol,src_backsc,
                        background_I,background_Q,background_U):
        """
        This method loads the Stokes I, Q and U backgrounds, rebins them over
        the channel grid of the corresponding source spectrum, and stores them
        in the noise and noise_err attributes in units of counts/s/keV. Any
        Stokes parameter for which a background is not provided is assumed to
        have a negligible background, and is set to zero.
 
        Parameters:
        -----------
        response: nDspec.ResponseMatrix
            The instrument response to be applied to Stokes I, before any
            rebinning over the channel grid of the data.
 
        response_pol: nDspec.ResponseMatrix
            The instrument response to be applied to Stokes Q and U, before any
            rebinning over the channel grid of the data.
 
        src_backsc: float
            The background scaling factor of the source spectrum, used together
            with that of the background spectrum to account for cases where the
            source and background extraction regions have a different size.
 
        background_I, background_Q, background_U: str or None
            Strings pointing to the paths of the Stokes I, Q and U background
            files.
 
        Returns:
        --------
        bkg_counts_I, bkg_counts_var_I: np.array(float) or None
            If a Stokes I background was loaded, its raw counts and variance,
            rebinned onto the Stokes I channel grid of the source spectrum and
            scaled by the background scaling factor. They are returned, rather
            than stored, because they are only needed once more, in
            _compute_data_stokes_I. If no Stokes I background was provided,
            both are None.
        """
 
        noise = np.zeros(self._all_bins)
        noise_err = np.zeros(self._all_bins)
        paths = [background_I, background_Q, background_U]
        stokes_names = ["I", "Q", "U"]
        responses = [response, response_pol, response_pol]
        widths = [self.ewidths, self.pol_ewidths, self.pol_ewidths]
        grids = [self._bounds_I, self._bounds_pol, self._bounds_pol]
 
        bkg_counts_I, bkg_counts_var_I = None, None
        for k, path in enumerate(paths):
            if path is None:
                continue
            bkg_lo, bkg_hi, bkg_counts, bkg_error, _, bkg_backsc = \
                load_stokes_pha(path,responses[k],stokes=stokes_names[k])
            backfac = src_backsc/bkg_backsc
            #rebin onto the grid of the source spectrum, and scale by the
            #background scaling factor, while still in raw counts
            bkg_rebin = responses[k]._rebin_sum(bkg_counts,[bkg_lo, bkg_hi],
                                                grids[k])*backfac
            bkg_rebin_var = responses[k]._rebin_sum(bkg_error**2,
                                                    [bkg_lo, bkg_hi],
                                                    grids[k])*backfac**2
            noise[self._stokes_slice(k,mask=False)] = bkg_rebin/self.exposure/ \
                                                       widths[k]
            noise_err[self._stokes_slice(k,mask=False)] = np.sqrt(
                                                           bkg_rebin_var)/ \
                                                           self.exposure/widths[k]
            #the Stokes I background is also returned in raw counts, on the
            #Stokes I channel grid, so that it can be subtracted before Stokes I
            #is rebinned onto the (coarser) Stokes Q/U channel grid
            if k == 0:
                bkg_counts_I = bkg_rebin
                bkg_counts_var_I = bkg_rebin_var
 
        self.noise = noise
        self.noise_err = noise_err
        self._noise_unmasked = noise
        self._noise_err_unmasked = noise_err
        return bkg_counts_I, bkg_counts_var_I
 
    def ignore_energies(self,bound_lo,bound_hi,stokes="all"):
        """
        This method adjusts the arrays stored such that they (and the fit)
        ignore selected channels based on their energy bounds. Because Stokes I
        is typically binned more finely than Stokes Q and U, it is possible to
        ignore channels either in all three Stokes parameters at once, or in
        Stokes I and Stokes Q/U separately.
 
        Parameters:
        -----------
        bound_lo: float
            Lower bound of ignored energy interval.
 
        bound_hi: float
            Higher bound of ignored energy interval.
 
        stokes: str, default "all"
            A string that sets which Stokes parameters the energy bounds should
            be applied to. "all" ignores the interval in all three Stokes
            parameters, "I" only in Stokes I, and "QU" only in Stokes Q and U.
        """
 
        if stokes not in self._supported_stokes:
            raise ValueError("Stokes parameter not recognized")
 
        if stokes in ("all", "I"):
            #the reason this is called this way is we are overloarding the 
            #function name, so if we call self.ignore_energies we end up in an
            #infinite recursive loop
            EnergyDependentFit.ignore_energies(self,bound_lo,bound_hi)
        if stokes in ("all", "QU", "Q", "U"):
            self.ignore_polarization_energies(bound_lo,bound_hi)
        return
 
    def notice_energies(self,bound_lo,bound_hi,stokes="all"):
        """
        This method adjusts the data arrays stored such that they (and the fit)
        notice selected (previously ignored) channels based on their energy
        bounds. Because Stokes I is typically binned more finely than Stokes Q
        and U, it is possible to notice channels either in all three Stokes
        parameters at once, or in Stokes I and Stokes Q/U separately.
 
        Parameters:
        -----------
        bound_lo: float
            Lower bound of noticed energy interval.
 
        bound_hi: float
            Higher bound of noticed energy interval.
 
        stokes: str, default "all"
            A string that sets which Stokes parameters the energy bounds should
            be applied to. "all" notices the interval in all three Stokes
            parameters, "I" only in Stokes I, and "QU" only in Stokes Q and U.
        """
 
        if stokes not in self._supported_stokes:
            raise ValueError("Stokes parameter not recognized")
 
        if stokes in ("all", "I"):
            #the reason this is called this way is we are overloarding the 
            #function name, so if we call self.ignore_energies we end up in an
            #infinite recursive loop
            EnergyDependentFit.notice_energies(self,bound_lo,bound_hi)
        if stokes in ("all", "QU", "Q", "U"):
            self.notice_polarization_energies(bound_lo,bound_hi)
        return

    def _check_gain_bounds(self,slope_bounds,offset_bounds):
        """
        This method checks the bounds on the gain parameters against both the
        Stokes I and the Stokes Q/U channel grids, which are typically grouped
        differently from one another. 
 
        Parameters:
        -----------
        slope_bounds: tuple
            The minimum and maximum values the slope is allowed to take.
 
        offset_bounds: tuple
            The minimum and maximum values the offset is allowed to take, in
            units of keV.
        """
 
        self._check_gain_grid(self.ebounds_mask,self._emin_unmasked,
                              self._emax_unmasked,slope_bounds,offset_bounds,
                              label="the Stokes I response")
        self._check_gain_grid(self.pol_ebounds_mask,self._pol_emin_unmasked,
                              self._pol_emax_unmasked,slope_bounds,
                              offset_bounds,
                              label="the Stokes Q and U response")
        return
 
    def set_fit_statistic(self,stat):
        """
        This method is used to set the statistic to be optimized during the fit.
        By default, the optimizer will optimize the chi-squared statistic.
 
        Note that, unlike for a time-averaged spectrum, Poisson-type statistics
        are not supported: Stokes Q and U are defined as the difference between
        two Poisson-distributed quantities, and are therefore not themselves
        Poisson distributed. In practice this is not a limitation, because the
        channel grids over which Stokes parameters are defined are always coarse
        enough for the Gaussian regime to apply.
 
        Parameters:
        -----------
        stat: str
            A string with the name of the fit statistic to be used. Supported
            statistics currently are "chisq" (the standard chi squared statistic,
            appropriate for data in the Gaussian regime) and "custom" (a
            likelihood defined by the user through set_custom_likelihood).
        """
 
        if (stat != "chisq" and stat != "custom"):
            raise ValueError("Fit statistic not recognized")
        self.likelihood = stat
        return
 
    def eval_model(self,params=None,ear=None,fold=True,mask=True):
        """
        This method is used to evaluate and return the model Stokes parameters
        for a given set of parameters, over a given model energy grid. By
        default it will evaluate the model over the energy grid defined in the
        responses, using the parameter values stored internally in the
        model_params attribute, and fold each Stokes parameter through the
        appropriate instrument response.
 
        The model set by the user is required to return an array of shape
        (3,len(energs)) containing the Stokes I, Q and U photon spectra. In
        practice this is built by multiplying a spectral model, which returns the
        Stokes I spectrum, by one of the multiplicative polarization models
        included in nDspec.Models.
 
        Parameters:
        -----------
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are
            provided, the model_params attribute is used.
 
        ear: np.array(float), default None
            The array of photon energy bin edges over which to evaluate the
            model. If none are provided, the same grid contained in the
            instrument responses is used.
 
        fold: bool, default True
            A boolean switch to choose whether to fold the evaluated model
            through the instrument responses or not. Note that in order for the
            model to be folded, the energy grid over which it is defined MUST
            be identical to that stored in the response matrices.
 
        mask: bool, default True
            A boolean switch to choose whether to mask the model output to only
            include the noticed energy channels, or to also return the ones
            that have been ignored by the users.
 
        Returns:
        --------
        model: np.array(float)
            The Stokes I, Q and U models evaluated over the given energy grid,
            for the given input parameters, and flattened into a single array in
            this order.
        """
 
        if self.model is None:
            raise AttributeError("No model set, use set_model first")
 
        if ear is None:
            ear = self.ear
            energ = self.energs
            energ_bounds = self.energ_bounds
        else:
            energ = 0.5*(ear[1:]+ear[:-1])
            energ_bounds = ear[1:]-ear[:-1]
 
        if params is None:
            params = self.model_params
            
        stokes = self.model.eval(params,energ=energ,ear=ear)*energ_bounds
        if np.shape(stokes)[0] != 3:
            raise TypeError(("The model must return an array of Stokes I, Q "
                             "and U; multiply your spectral model by one of "
                             "the polarization models in nDspec.Models or "
                             "update your model format."))
        stokes_I = stokes[0]
        stokes_Q = stokes[1]
        stokes_U = stokes[2]
        if fold is True:
            stokes_I = self.response.convolve_response(stokes_I)
            stokes_Q = self.response_pol.convolve_response(stokes_Q)
            stokes_U = self.response_pol.convolve_response(stokes_U)
            stokes_I = self._apply_calibration(stokes_I,params,self.response)
            stokes_Q = self._apply_calibration(stokes_Q,params,self.response_pol)
            stokes_U = self._apply_calibration(stokes_U,params,self.response_pol)
        elif mask is True:
            raise ValueError(("mask=True requires fold=True: the ignore/"
                              "notice mask is defined in channel space, and "
                              "only applies to the model once it has been "
                              "folded through the response.")) 

        model = np.concatenate((stokes_I,stokes_Q,stokes_U))

        if mask is True:
            model = self._filter_stokes_by_mask(model)
        return model
 
    def derive_folded_polarization(self,params=None,mask=True):
        """
        This method evaluates the model polarization angle and degree in
        detector space. Because the computation is in detector space, the 
        polarization degree is multiplied by the modulation factor $\\mu(E)$.
        
        Both products are computed over the energy channel grid of the Stokes Q 
        and U spectra. 
 
        Parameters:
        -----------
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are
            provided, the model_params attribute is used.
 
        mask: bool, default True
            A boolean switch to choose whether to mask the model output to only
            include the noticed energy channels, or to also return the ones
            that have been ignored by the users.
 
        Returns:
        --------
        pol_degree: np.array(float)
            The model polarization degree in each Stokes Q/U energy channel.
 
        pol_angle: np.array(float)
            The model polarization angle, in radians, in each Stokes Q/U energy
            channel.
        """

        if params is None:
            params = self.model_params

        model = self.eval_model(params=params,fold=False,mask=False)
        #the reason we call np.split (which divides the array in 3) rather than 
        #_split_stokes (which accounts for different binning in Stokes I/Q/U) is 
        #that since we did not yet fold with the response to get "model", the 
        #three arrays have identical size 
        stokes_I, stokes_Q, stokes_U = np.split(model,3)
                
        #Stokes I has to be folded over the Q/U channel grid, otherwise the
        #three Stokes parameters are not defined over the same channels
        stokes_I = self.response_polgrid.convolve_response(stokes_I)
        stokes_Q = self.response_pol.convolve_response(stokes_Q)
        stokes_U = self.response_pol.convolve_response(stokes_U)
        stokes_I = self._apply_calibration(stokes_I,params,self.response_polgrid)
        stokes_Q = self._apply_calibration(stokes_Q,params,self.response_pol)
        stokes_U = self._apply_calibration(stokes_U,params,self.response_pol)
 
        model_product = PolarimetryProduct(self._pol_ebounds_unmasked,
                                           input_type='stokes')
        model_product.set_stokes(stokes_I,stokes_Q,stokes_U)
        #channels with no effective area have Stokes I identically zero, and
        #therefore an undefined polarization degree; they are always ignored in
        #a fit, but they still need to be handled when returning every channel
        with np.errstate(divide='ignore',invalid='ignore'):
            pol_degree, pol_angle = model_product.stokes_to_polarization()
        pol_degree = np.nan_to_num(pol_degree)
        pol_angle = np.nan_to_num(pol_angle)
 
        if mask is True:
            pol_degree = np.extract(self.pol_ebounds_mask,pol_degree)
            pol_angle = np.extract(self.pol_ebounds_mask,pol_angle)
        return pol_degree, pol_angle
 
    def get_data_polarization(self,mask=True):
        """
        This method computes the polarization degree and angle of the data. Due 
        to how data is recorded in X-ray polarimeters, the polarization degree 
        returned by this method is modulated by the modulation factor $\\mu$(E).
        
        The polarization angle and (modulated) polarization degree are computed 
        over the energy channel grid of the Stokes Q and U spectra, along with 
        their errors. The Stokes I counts are rebinned over the Stokes Q/U 
        channel grid first, since the three Stokes parameters need to be defined 
        over the same channels.
 
        The errors are computed by propagating the errors on the three Stokes
        parameters, and neglecting their covariance; users interested in
        rigorous confidence intervals on the polarization degree and angle
        should instead sample the posterior of the fit. Note also that the
        polarization degree is a positive definite quantity, and is therefore
        biased upwards in channels with a low signal to noise ratio.
 
        Parameters:
        -----------
        mask: bool, default True
            A boolean switch to choose whether to return only the noticed energy
            channels, or also the ones that have been ignored by the users.
 
        Returns:
        --------
        pol_degree: np.array(float)
            The polarization degree of the data in each Stokes Q/U channel.
 
        pol_angle: np.array(float)
            The polarization angle of the data, in radians, in each Stokes Q/U
            channel.
 
        pol_degree_err: np.array(float)
            The error on the polarization degree in each Stokes Q/U channel.
 
        pol_angle_err: np.array(float)
            The error on the polarization angle, in radians, in each Stokes Q/U
            channel.
        """
 
        #the Stokes I data on the Q/U channel grid was already rebinned and
        #normalized when the data was loaded, since nothing it depends on
        #changes afterward
        stokes_I = self._data_stokes_I_unmasked
        stokes_I_err = self._data_stokes_I_err_unmasked
        _, stokes_Q, stokes_U = self.split_stokes(self._data_unmasked,mask=False)
        _, stokes_Q_err, stokes_U_err = self.split_stokes(
                                        self._data_err_unmasked,mask=False)

        if self.noise is not None:
            _, noise_Q, noise_U = self.split_stokes(self._noise_unmasked,
                                                    mask=False)
            _, noise_Q_err, noise_U_err = self.split_stokes(
                                          self._noise_err_unmasked,mask=False)
            stokes_Q = stokes_Q - noise_Q
            stokes_U = stokes_U - noise_U
            stokes_Q_err = np.sqrt(stokes_Q_err**2+noise_Q_err**2)
            stokes_U_err = np.sqrt(stokes_U_err**2+noise_U_err**2)
 
        data_product = PolarimetryProduct(self._pol_ebounds_unmasked,
                                          input_type='stokes')
        data_product.set_stokes(stokes_I,stokes_Q,stokes_U)
        #channels with no counts have an undefined polarization degree; they are
        #always ignored in a fit, but they still need to be handled when
        #returning every channel
        with np.errstate(divide='ignore',invalid='ignore'):
            pol_degree, pol_angle = data_product.stokes_to_polarization()
            pol_degree_err, pol_angle_err = self._polarization_errors(
                                            stokes_I,stokes_Q,stokes_U,
                                            stokes_I_err,stokes_Q_err,
                                            stokes_U_err)
        pol_degree = np.nan_to_num(pol_degree)
        pol_angle = np.nan_to_num(pol_angle)
 
        if mask is True:
            pol_degree = np.extract(self.pol_ebounds_mask,pol_degree)
            pol_angle = np.extract(self.pol_ebounds_mask,pol_angle)
            pol_degree_err = np.extract(self.pol_ebounds_mask,pol_degree_err)
            pol_angle_err = np.extract(self.pol_ebounds_mask,pol_angle_err)
        return pol_degree, pol_angle, pol_degree_err, pol_angle_err
 
    def _polarization_errors(self,stokes_I,stokes_Q,stokes_U,
                             stokes_I_err,stokes_Q_err,stokes_U_err):
        """
        This method propagates the errors on the three Stokes parameters into
        errors on the polarization degree and angle, neglecting the covariance
        between the Stokes parameters.
 
        Parameters:
        -----------
        stokes_I, stokes_Q, stokes_U: np.array(float)
            The arrays containing the three Stokes parameters, defined over the
            same channel grid.
 
        stokes_I_err, stokes_Q_err, stokes_U_err: np.array(float)
            The arrays containing the errors on the three Stokes parameters.
 
        Returns:
        --------
        pol_degree_err: np.array(float)
            The error on the polarization degree in each channel.
 
        pol_angle_err: np.array(float)
            The error on the polarization angle, in radians, in each channel.
        """
 
        pol_flux = np.sqrt(stokes_Q**2+stokes_U**2)
        pol_degree = pol_flux/stokes_I
        #the error on the polarized flux, propagated from Q and U
        pol_flux_err = np.sqrt((stokes_Q*stokes_Q_err)**2+
                               (stokes_U*stokes_U_err)**2)/pol_flux
        pol_degree_err = pol_degree*np.sqrt((pol_flux_err/pol_flux)**2+
                                            (stokes_I_err/stokes_I)**2)
        pol_angle_err = 0.5*np.sqrt((stokes_Q*stokes_U_err)**2+
                                    (stokes_U*stokes_Q_err)**2)/pol_flux**2
        return np.nan_to_num(pol_degree_err), np.nan_to_num(pol_angle_err)
 
    def get_polarization_residuals(self,res_type,params=None):
        """
        This method returns the residuals of the model polarization degree and
        angle with respect to those computed from the data. It is only used for
        visualization purposes: the fit itself is always performed on the Stokes
        parameters, which unlike the polarization degree and angle have well
        behaved, approximately Gaussian errors.
 
        Parameters:
        -----------
        res_type: str
            If set to "ratio", the method returns the residuals defined as
            data/model. If set to "chisq", it returns the contribution of each
            energy channel to the total chi squared.
 
        params: lmfit.Parameters, default None
            The parameter values to use in evaluating the model. If none are
            provided, the model_params attribute is used.
 
        Returns:
        --------
        residuals: list(np.array(float))
            A list containing the residuals of the polarization degree and
            angle, in this order.
 
        bars: list(np.array(float))
            A list containing the one sigma range for each contribution to the
            residuals of the polarization degree and angle, in this order.
        """
 
        model_degree, model_angle = self.derive_folded_polarization(params=params)
        data_degree, data_angle, degree_err, angle_err = \
            self.get_data_polarization()
        #get rid of the phase wraps when the difference between data and model 
        #is more than pi; e.g. if data_angle is -89 and model_angle is 89, then 
        #taking the difference would return 178, this returns 4.
        data_angle = model_angle+0.5*np.angle(np.exp(2j*(data_angle-model_angle)))
 
        if res_type == "ratio":
            degree_res, degree_bars = ratio(data_degree,degree_err,
                                            model_degree,summed=False)
            angle_res, angle_bars = ratio(data_angle,angle_err,
                                          model_angle,summed=False)
        elif res_type == "chisq":
            degree_res = chisq(data_degree,degree_err,model_degree,summed=False)
            angle_res = chisq(data_angle,angle_err,model_angle,summed=False)
            degree_bars = np.ones(len(degree_res))
            angle_bars = np.ones(len(angle_res))
        else:
            raise ValueError("The supported residual types are ratio and chisq")
        return [degree_res, angle_res], [degree_bars, angle_bars]
 
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
        elif self.likelihood == "custom":
            residuals, _ = self.get_residuals("custom",model=model,mask=True)
        else:
            raise AttributeError("Chosen likelihood not supported")
        return residuals
 
    def _plot_format(self,array):
        """
        This method splits a flattened array defined over every channel into its 
        three Stokes parameters, with ignored channels set to NaN so that they 
        are left blank in plots. 
        
        Parameters:
        -----------
        array: np.array(float)
            The flattened array containing the Stokes I, Q and U values in every 
            channel, in this order.
            
        Returns:
        --------
        stokes: list(np.array(float))
            A list containing the Stokes I, Q and U arrays to be plotted. 
        """
        
        stokes_I, stokes_Q, stokes_U = self.split_stokes(array,mask=False)
        return [np.where(self.ebounds_mask,stokes_I,np.nan),
                np.where(self.pol_ebounds_mask,stokes_Q,np.nan),
                np.where(self.pol_ebounds_mask,stokes_U,np.nan)]


    def plot_data(self,units="stokes",plot_bkg=False,panels=None,
                  return_plot=False,data_kwargs=None,bkg_kwargs=None):
        """
        This method plots the spectro-polarimetric data loaded by the user as a
        function of energy, either as the three Stokes parameters or as the 
        polarization degree and angle computed from them. Ignored channels are 
        left blank.
 
        Parameters:
        -----------
        units: str, default="stokes"
            The quantities to plot. units="stokes" plots the three Stokes
            parameters in detector space, in units of counts/s/keV;
            units="polarization" instead plots the polarization degree and
            angle, computed from the data over the Stokes Q/U channel grid.
 
        plot_bkg: bool, default=False
            A boolean to choose whether to overplot the background. Only 
            supported when plotting Stokes parameters.
 
        panels : matplotlib.axes or numpy.ndarray(matplotlib.axes), default=None
            Existing panels to draw into, e.g as returned by calling plot_model 
            from a second fitter. If None, a new figure is created.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure object containing 
            the plot or not.
            
        data_kwargs, bkg_kwargs: dict, default=None 
            Keyword arguments for the data and background points.
 
        Returns:
        --------
        fig: matplotlib.figure, optional
            The plot object produced by the method.
            
        panels: numpy.ndarray(matplotlib.axes), optional 
            The panels containing the plot, one per quantity.
        """

        #bulid the bin centers, their widths, and their edges into arrays
        #for both stokes I and Q/U (or polarization/degree)
        grid_I = (self._ebounds_unmasked,0.5*self._ewidths_unmasked,
                  np.append(self._emin_unmasked,self._emax_unmasked[-1]))
        grid_pol = (self._pol_ebounds_unmasked,0.5*self._pol_ewidths_unmasked,
                    np.append(self._pol_emin_unmasked,self._pol_emax_unmasked[-1]))
 
        bkg = None
        if units == "stokes":
            data = self._plot_format(self._data_unmasked)
            data_err = self._plot_format(self._data_err_unmasked)
            if plot_bkg is True:
                if self.noise is None:
                    raise AttributeError("No background loaded")
                bkg = self._plot_format(self._noise_unmasked)
            grids = [grid_I,grid_pol,grid_pol]
            y_labels = ["Stokes I (counts/s/keV)","Stokes Q (counts/s/keV)",
                        "Stokes U (counts/s/keV)"]
        elif units == "polarization":
            if plot_bkg is True:
                raise ValueError("The background can only be plotted for "
                                 "Stokes parameters")
            degree, angle, degree_err, angle_err = \
                self.get_data_polarization(mask=False)
            data = [degree,np.degrees(angle)]
            data_err = [degree_err,np.degrees(angle_err)]
            data = [np.where(self.pol_ebounds_mask,v,np.nan) for v in data]
            data_err = [np.where(self.pol_ebounds_mask,e,np.nan) for e in data_err]
            grids = [grid_pol,grid_pol]
            y_labels = ["Polarization degree $\\times\\ \\mu(E)$",
                        "Polarization angle (deg)"]
        else:
            raise ValueError("Plot units not supported")
        n_quant = len(data)

        layout_kwargs = dict(ncols=n_quant,figsize=(6.5*n_quant,4.5))
        if panels is None:
            plot_layout = Plotting.make_layout(**layout_kwargs)
            fig, panels = Plotting.make_panels(plot_layout,squeeze=False)
        else:
            #The panels need to be sorted to lay out the plot correctly 
            panels = np.reshape(np.asarray(panels,dtype=object),(1,n_quant))
            fig = panels[0,0].get_figure()        
     
        for k in range(n_quant):
            x_axis, x_bars, _ = grids[k]
            #only Stokes I is positive definite
            log_yaxis = (units == "stokes" and k == 0)
            if units == "stokes" and k > 0:
                panels[0,k].axhline(0,color='black',ls=':',zorder=3)
            panel_data = Plotting.make_panel_data(x_points=x_axis,
                                                  y_points=data[k],
                                                  x_bars=x_bars,
                                                  y_bars=data_err[k],
                                                  bkg_vals=None if bkg is None else bkg[k],
                                                  x_label="Energy (keV)",
                                                  y_label=y_labels[k])
            Plotting.draw_main_panel(panels[0,k],panel_data,draw_model=False,
                                     draw_bkg=plot_bkg,log_yaxis=log_yaxis,
                                     data_kwargs=data_kwargs,
                                     bkg_kwargs=bkg_kwargs)
 
        panels = panels[0]
        if return_plot is True:
            return fig, panels
        else:
            return

    def plot_model(self,plot_data=True,plot_components=False,plot_bkg=False,
                   params=None,units="stokes",residuals=None,panels=None,
                   return_plot=False,data_kwargs=None,model_kwargs=None,
                   bkg_kwargs=None,component_kwargs=None,resid_kwargs=None):
        """
        This method plots the model as a function of energy, optionally with 
        its additive components, the data, and the residuals, either as the 
        three Stokes parameters or as the polarization degree and angle. 
        Ignored channels are left blank.
 
        When plotting the polarization degree and angle, the residuals are 
        computed from those quantities directly; they are only indicative, 
        since unlike the Stokes parameters they do not have Gaussian errors, 
        and they do not add up to the fit statistic. 
 
        Parameters:
        -----------
        plot_data: bool, default=True
            If True, the data and a residual panel are drawn below the model.
 
        plot_components: bool, default=False
            If True, the additive model components are overplotted on each 
            Stokes parameter, folded through the responses and gain. Only 
            supported when plotting Stokes parameters, since the polarization 
            degree and angle are not additive.
 
        plot_bkg: bool, default=False
            A boolean to choose whether to overplot the background. Only 
            supported when plotting Stokes parameters.
 
        params: lmfit.Parameters, default=None
            The parameters used to evaluate the model. If None, model_params 
            is used.
 
        units: str, default="stokes"
            The quantities to plot, as in plot_data.
 
        residuals: str, default=None
            The residual units: "chisq" for (data-model)/error, "ratio" for 
            data/model, or "custom". If None, the fit likelihood is used.
 
        panels : matplotlib.axes or numpy.ndarray(matplotlib.axes), default=None
            Existing panels to draw into, e.g as returned by calling plot_model 
            from a second fitter. If None, a new figure is created.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure object containing 
            the plot or not.
            
        data_kwargs, model_kwargs, bkg_kwargs, component_kwargs, resid_kwargs: dict, default=None 
            Keyword arguments for each element of the plot.
 
        Returns:
        --------
        fig: matplotlib.figure, optional
            The plot object produced by the method.
            
        panels: numpy.ndarray(matplotlib.axes), optional 
            The panels containing the plot: one per quantity if plot_data is 
            False, otherwise an array of size (2,n_quantities) with the 
            residuals in the second row.
        """
 
        if params is None:
            params = self.model_params
        if residuals is None:
            residuals = self.likelihood
        if residuals == "chisq":
            res_label = "$\\Delta\\chi$"
        elif residuals == "ratio":
            res_label = "Data/model"
        elif residuals == "custom" and units == "stokes":
            res_label = "Residuals"
        else:
            raise ValueError("Residual format not supported")
        if plot_bkg is True and self.noise is None:
            raise AttributeError("No background loaded")
        if units != "stokes" and (plot_components is True or plot_bkg is True):
            raise ValueError("Components and background can only be plotted "
                             "for Stokes parameters")

        #bulid the bin centers, their widths, and their edges into arrays
        #for both stokes I and Q/U (or polarization/degree)
        grid_I = (self._ebounds_unmasked,0.5*self._ewidths_unmasked,
                  np.append(self._emin_unmasked,self._emax_unmasked[-1]))
        grid_pol = (self._pol_ebounds_unmasked,0.5*self._pol_ewidths_unmasked,
                    np.append(self._pol_emin_unmasked,self._pol_emax_unmasked[-1]))
 
        bkg, components = None, None
        if units == "stokes":
            y_labels = ["Stokes I (counts/s/keV)","Stokes Q (counts/s/keV)",
                        "Stokes U (counts/s/keV)"]
            grids = [grid_I,grid_pol,grid_pol]
            model = self.eval_model(params=params,mask=False)
            model_vals = self._plot_format(model)            
            
            data = self._plot_format(self._data_unmasked)
            data_err = self._plot_format(self._data_err_unmasked)
            #this horrible jank here is necessary to avoid raising annoying 
            #warnings about bins that are not being noticed, while still
            #supporting having gaps in the noticed bins 
            res, err = self.get_residuals(residuals,
                              model=self._filter_stokes_by_mask(model))
            noticed = self._filter_stokes_by_mask(np.arange(model.size))
            resid = np.full(model.size,np.nan)
            reserr = np.full(model.size,np.nan)
            resid[noticed] = res
            reserr[noticed] = err
            resid = self._plot_format(resid)
            reserr = self._plot_format(reserr)
            
            if plot_bkg is True:
                bkg = self._plot_format(self._noise_unmasked)
            
            if plot_components is True:
                #the Stokes parameters are additive, so each component is folded 
                #and plotted separately in every Stokes parameter
                comps = model_expand(self.model,params,energ=self.energs,
                                     ear=self.ear)
                folded = {}
                for key in comps.keys():
                    comp = comps[key]*self.energ_bounds
                    comp_I = self._apply_calibration(self.response.convolve_response(comp[0]),
                                              params,self.response)
                    comp_Q = self._apply_calibration(self.response_pol.convolve_response(comp[1]),
                                              params,self.response_pol)
                    comp_U = self._apply_calibration(self.response_pol.convolve_response(comp[2]),
                                              params,self.response_pol)
                    folded[key] = self._plot_format(np.concatenate((comp_I,comp_Q,comp_U)))
                components = [{key: folded[key][k] for key in folded.keys()} for k in range(3)]
        
        elif units == "polarization":
            grids = [grid_pol,grid_pol]
            y_labels = ["Polarization degree $\\times\\ \\mu(E)$",
                        "Polarization angle (deg)"]
            model_degree, model_angle = self.derive_folded_polarization(params=params,
                                                                        mask=False)
            model_vals = [model_degree,np.degrees(model_angle)]
            model_vals = [np.where(self.pol_ebounds_mask,v,np.nan) for v in model_vals]
            
            degree, angle, degree_err, angle_err = \
                self.get_data_polarization(mask=False)            
            data = [degree,np.degrees(angle)]
            data_err = [degree_err,np.degrees(angle_err)]            
            data = [np.where(self.pol_ebounds_mask,v,np.nan) for v in data]
            data_err = [np.where(self.pol_ebounds_mask,v,np.nan) for v in data_err]
            
            res, err = self.get_polarization_residuals(residuals,params=params)
            resid, reserr = [], []
            #this horrible jank here is necessary to avoid raising annoying 
            #warnings about bins that are not being noticed, while still
            #supporting having gaps in the noticed bins 
            for k in range(2):
                resid.append(np.full(self._all_pol_chans,np.nan))
                reserr.append(np.full(self._all_pol_chans,np.nan))
                resid[k][self.pol_ebounds_mask] = res[k]
                reserr[k][self.pol_ebounds_mask] = err[k]
        else:
            raise ValueError("Plot units not supported")
        n_quant = len(model_vals)
        
        layout_kwargs = dict(figsize=(6.5*n_quant,4.5),ncols=n_quant)
        if plot_data is True:
            layout_kwargs = dict(nrows=2,ncols=n_quant,height_ratios=[2,1],
                                 sharex=True,figsize=(6.5*n_quant,6.))  
        if panels is None:
            plot_layout = Plotting.make_layout(**layout_kwargs)
            fig, panels = Plotting.make_panels(plot_layout,squeeze=False)
        else:
            #To lay out the panels correctly we need to reshape to a nrowsxncols
            #grid layout
            panels = np.reshape(np.asarray(panels,dtype=object),
                                (layout_kwargs.get("nrows",1),n_quant))
            fig = panels[0,0].get_figure()
 
        for k in range(n_quant):
            x_axis, x_error, x_edges = grids[k]
            #only Stokes I is positive definite
            log_yaxis = (units == "stokes" and k == 0)
            if units == "stokes" and k > 0:
                panels[0,k].axhline(0,color='black',ls=':',zorder=3)
            panel_data = Plotting.make_panel_data(x_points=x_axis,
                                                  y_points=data[k],
                                                  x_bars=x_error,
                                                  y_bars=data_err[k],
                                                  model_vals=model_vals[k],
                                                  model_edges=x_edges,
                                                  bkg_vals=None if bkg is None else bkg[k],
                                                  component_vals=None if components is None else components[k],
                                                  resid=resid[k],
                                                  reserr=reserr[k],
                                                  x_label="Energy (keV)",
                                                  y_label=y_labels[k],
                                                  res_label=res_label)
            Plotting.draw_main_panel(panels[0,k],panel_data,draw_data=plot_data,
                                     draw_bkg=plot_bkg,
                                     draw_components=plot_components,
                                     log_yaxis=log_yaxis,
                                     data_kwargs=data_kwargs,
                                     model_kwargs=model_kwargs,
                                     bkg_kwargs=bkg_kwargs,
                                     component_kwargs=component_kwargs)
            if plot_data is True:
                panels[0,k].set_xlabel("")
                Plotting.draw_residual_panel(panels[1,k],panel_data,residuals,
                                             resid_kwargs=resid_kwargs)
        if plot_data is False:
            panels = panels[0]
        
        if return_plot is True:
            return fig, panels
        else:
            return


    def plot_polarization_slice(self,params=None,plot_data=True,
                                plot_model=True,cmap="viridis",
                                angle_range=None,degree_range=None,
                                return_plot=False,data_kwargs=None,
                                model_kwargs=None,track_kwargs=None):
        """
        This method plots the polarization degree and angle in polar
        coordinates, with each energy channel shown as an ellipse colored by
        the energy of the channel. The data is shown as its one sigma
        confidence region; the model is shown as a filled ellipse of variable 
        size. 
 
        Due to the ambiguity in X-ray polarimeters, the polarization angle is 
        only defined modulo 180 degrees. Users can choose any angle range (e.g 
        -30 to 30 degrees), in which case the data and model are wrapped in the 
        correct quadrant.
 
        Parameters:
        -----------
        params: lmfit.Parameters, default=None
            The parameters used to evaluate the model. If None, model_params 
            is used.
 
        plot_data, plot_model: bool, default=True
            Booleans to choose whether the data and the model are shown.
 
        cmap: str, default="viridis"
            The colormap used to color each energy channel.
 
        angle_range: list(float), default=None
            The lower and upper bounds of the polarization angles shown, in 
            degrees, less than 180 degrees apart. Ranges straddling the origin 
            can be given as (-30,30). If None, the full 0 to 180 degree range 
            is shown.
 
        degree_range: list(float), default=None
            The lower and upper bounds of the polarization degrees shown. If 
            None, the axis runs from zero to slightly above the largest value 
            plotted.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure and panel.
            
        data_kwargs, model_kwargs, track_kwargs: dict, default=None 
            Keyword arguments for the data ellipses, model markers and track.
 
        Returns:
        --------
        fig: matplotlib.figure, optional
            The plot object produced by the method.
            
        panel: matplotlib.axes, optional 
            The panel containing the plot produced by the method.
        """
 
        if plot_data is False and plot_model is False:
            raise ValueError("At least one of data or model must be plotted")
 
        degree, angle, degree_err, angle_err = None, None, None, None
        model_degree, model_angle = None, None
        if plot_data is True:
            degree, angle, degree_err, angle_err = self.get_data_polarization()
        if plot_model is True:
            model_degree, model_angle = self.derive_folded_polarization(params=params)
 
        plot_layout = Plotting.make_layout(figsize=(6.5,4.5),projections=["polar"])
        fig, panel = Plotting.make_panels(plot_layout)
        data = Plotting.make_polar_data(angle=angle,degree=degree,
                                        angle_err=angle_err,
                                        degree_err=degree_err,
                                        model_angle=model_angle,
                                        model_degree=model_degree,
                                        color_values=self.pol_ebounds,
                                        color_label="Energy (keV)",
                                        title="Polarization angle/degree$\\times\\mu(E)$")
        Plotting.draw_polar_panel(panel,data,cmap=cmap,draw_data=plot_data,
                                  draw_model=plot_model,angle_range=angle_range,
                                  degree_range=degree_range,
                                  data_kwargs=data_kwargs,
                                  model_kwargs=model_kwargs,
                                  track_kwargs=track_kwargs)
 
        if return_plot is True:
            return fig, panel
        else:
            return
