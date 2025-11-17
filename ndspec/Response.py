import numpy as np
import copy
import warnings
from astropy.io import fits
from scipy.interpolate import interp1d
#import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams.update({'font.size': 17})

colorscale = pl.cm.PuRd(np.linspace(0.,1.,5))

from .Operator import nDspecOperator
from .Timing import CrossSpectrum

class ResponseMatrix(nDspecOperator):
    """
    This class handles folding an energy-dependent, multi-dimensional product,
    through a given X-ray instrument response matrix, including the effects of
    both the redistribution matrix and the effective area. 

    Currently the X-ray observatories explicitely supported are NICER, NuSTAR 
    and RXTE. Swift/XRT/XMM/Chandra/Insight-HXMT/Astrosat should be compatible 
    but have not yet been tested. Support for these missions is pending. 
    
    XRISM and Athena are NOT compatible with this code due to a) the large size 
    of their responses and b) the unusual format of the contents of the 
    responses when read with Astropy. Fixes for both of these issues are 
    pending.     
    
    Parameters: 
    -----------  
    resp_path: string 
        The path to the response file (either .rmf or .rsp) to load.
    
    Other parameters:
    -----------------  
    arf_path: string 
        Optional path to an effective area file (.arf) to load along with the 
        redistribution matrix.  Note that some mission tools produce .rmf files 
        which also contain the telescope effective area, in which case loading 
        the arf is not necessary. 
        
    Attributes:
    -----------
    chans: numpy.array(int)
        An array of integers of size (n_chans) representing each channel in the 
        response.
        
    n_chans: int
        The length of the chans array.
    
    emin, emax: numpy.array(float)
        The arrays with the minimum/maximum energies in each channel in the  
        chans array. After rebinning on a new grid, it contains the minimum and  
        maximum energies of each bin in the new grid.
        
    energ_lo, energ_hi: numpy.array(float)
        The arrays with the minimum/maximum energies bins that sample the 
        instrument rmf/arf.
        
    n_energs: int
        The length of the energ_lo/energ_hi arrays.
        
    resp_matrix: numpy.array(float x float)
        The instrument response in matrix form, with size (n_energs x n_chans). 
        Only contains the redistribution matrix if no arf is loaded yet.

    specresp: numpy.array(float)
        The instrument effective area as a function of energy as contained in 
        the arf file.
        
    energ_rebin: bool
        A flag to check whether the response was re-binned over energies, in 
        which case folding requires a renormalization constant to conserve the
        total photon counts  
        
    has_arf: bool
        A flag to check whether only a rmf file has been loaded, or a full 
        rmf+arf response. This varies between observatories.  
        
    mission: str 
        A string which tracks the observatory for which the response is defined. 
        
    instrument: str 
        A string which tracks the instrument for which the response is defined.         
    """ 
    
    def __init__(self, resp_path, arf_path=None):
        self.load_rmf(resp_path)
        if (arf_path is not None):
            self.load_arf(arf_path)
        #add check to also load the arf if provided
        pass
    
    def load_rmf(self,filepath):
        """
        This method loads either a rmf or rsp file (containing the 
        redistribution matrix or full response, respectively), and sets the  
        class attributes from it using astropy.
        
        Parameters:
        -----------             
        filepath: string 
            The path to the .rmf or .rsp file to be loaded.
        """
        
        self.rmfpath = filepath 
        with fits.open(filepath, memmap=False) as response:
            # get all the extension names
            extnames = np.array([h.name for h in response])
            self.bounds = response["EBOUNDS"]
            #figure out the right table to use, and decide 
            #whether we are also loading the arf or not 
            #redo this check some matrices are weird          
           
            if "MATRIX" in extnames:
                h = response["MATRIX"]
                self.has_arf = False
                print("Arf missing, please load it")
            elif "SPECRESP MATRIX" in extnames:
                h = response["SPECRESP MATRIX"]
                self.has_arf = True
                print("Response file includes arf")
            channel_info = self.bounds.data
            data = h.data
            hdr = h.header
            if (hdr["TELESCOP"] == "ATHENA") or (hdr["TELESCOP"] == "XRiSM"):
                raise AttributeError(hdr["TELESCOP"],"data not supported!")
            if hdr["HDUCLASS"] != "OGIP":
                raise TypeError("File is not OGIP compliant")   
            self.mission = hdr["TELESCOP"]
            self.instrument = hdr["INSTRUME"]
        
        self.emin = np.array(channel_info.field("E_MIN"))
        self.emax = np.array(channel_info.field("E_MAX"))
        self.chans = np.array(channel_info.field("CHANNEL"))
        self.n_chans = len(self.chans)  

        self.energ_lo = np.array(data.field("ENERG_LO"))
        self.energ_hi = np.array(data.field("ENERG_HI"))
        self.n_energs = len(self.energ_lo)
        
        self.offset = self._get_tlmin(h)
        #by default we assume we're loading a file that was not rebinned in 
        #energy. This is needed to track whether a normalization constant due 
        #to energy rebinning is required when folding
        self.energ_rebin = False
        
        #We only need this information to convert the response to matrix format
        #so there is no need to store these as attributes 
        n_grp = np.array(data.field("N_GRP"))
        f_chan = np.array(data.field("F_CHAN"))
        n_chan = np.array(data.field("N_CHAN"))
        matrix = np.array(data.field("MATRIX"))
        
        self.resp_matrix = self._read_matrix(n_grp,f_chan,n_chan,matrix)        
        return
        
    def _read_matrix(self,n_grp,f_chan,n_chan,matrix):
        """
        This method converts the information in the n_grp, f_chan, n_chan and 
        matrix columns of a response file into a (n_energs x n_chans) matrix.
        
        Parameters:
        ----------- 
        n_grp: np.array(int)
            The number of sets of non-zero elements stored in the matrix. 
        
        f_chan: np.array(int)
            The first channel in each set of non-zero elements stored in the 
            matrix. 
        
        n_chan: np.array(int)
            The number of channels after f_chan that are stored in each set 
            labelled from n_grp.
        
        matrix: np.array(float) 
            The non-zero values of the instrument response stored in each set 
            marked by n_grp, starting at channel f_chan and ending at channel 
            f_chan+n_chan.          
        
        Returns:
        --------
        resp_matrix: np.array(float,float)
            The instrument response matrix, loaded in an array of dimensions
            (n_energs x n_chans). The elements that are not present in the 
            response file are hard-coded to 0.
        """        
        #start with an empty matrix - we need to figure out 
        #which elements from the FITS file are not zero. These are the only 
        #values reported in the FITS file, where the matrix has format 
        #(number of successive energy bins with no empty values n_grp) x
        #(number of bins that are not empty n_chan) x energy
        #we are trying to convert this to a matrix with format channel x energy
        resp_matrix = np.zeros((self.n_energs,self.n_chans),dtype=np.float32)
        #loop over the detector energies over which the rmf is binned 
        for j in range(self.n_energs):
            i = 0
            #loop over the number of channels in each channel set of consecutive
            #bins with no empty values
            for k in range(n_grp[j]):
                #Sometimes there are more than one groups of entries per row
                #As a result, we loop over the groups and assign the matrix  
                #values in the appropariate channel range as below:
                if any(m>1 for m in n_grp):
                    for l in range(f_chan[j][k],n_chan[j][k]+f_chan[j][k]):
                        resp_matrix[j][l] = resp_matrix[j][l] + matrix[j][i]
                        i = i + 1
                #In this case, the length of j-th row of the "Matrix" array is 
                #n_chan[j]+f_chan[j] corresponding to channel indexes 
                #f_chan[j]+1 to n_chan[j]+f_chan[j]. We set those values in 
                #coordinates j,l in the matrix array resp_matrix
                #IMPORTANT: loading the matrix this way doesn't read the very 
                #last channel correctly. This is fine because nobody sane is 
                #ever going to use the very last detector channel for science 
                #anyway.
                else:
                    for l in range(f_chan[j],n_chan[j]+f_chan[j]):
                        resp_matrix[j][l] = resp_matrix[j][l] + matrix[j][i]  
                        i = i + 1              
        return resp_matrix        
    
    def load_arf(self,filepath):       
        """
        This method reads an effective area .arf file, and applies it to a
        redistribution matrix previously loaded with the load_rmf method. The
        arf-corrected matrix over-writes the class attribute resp_matrix.
        
        Additionally, the array of effective area vs energy is stored in the 
        specresp class attribute.
        
        Parameters:
        -----------     
        filepath: string 
            The path to the .arf file to be loaded.
        """
    
        self.arfpath = filepath
        self.has_arf = True
        with fits.open(filepath) as response:
            extnames = np.array([h.name for h in response])
            h = response["SPECRESP"]
            data = h.data
            hdr = h.header
            
            if hdr["HDUCLASS"] != "OGIP":
                raise TypeError("File is not OGIP compliant")   
        
        arf_emin = np.array(data.field("ENERG_LO"))
        arf_emax = np.array(data.field("ENERG_HI"))        
        
        if (np.allclose(arf_emin,self.energ_lo) is False or 
            np.allclose(arf_emax,self.energ_hi) is False):
            raise ValueError("Energy grids in rmf and arf do not match")
        
        self.specresp = np.array(data.field("SPECRESP"))
        
        if "EXPOSURE" in list(hdr.keys()):
            self.exposure = hdr["EXPOSURE"]
            print(("Exposure header found in ARF, setting exposure time to"
                  f" {self.exposure} second"))
        else:
            self.exposure = 1.0
            
        
        for k in range(self.n_chans):
            for j in range(self.n_energs):
                self.resp_matrix[j][k] = self.resp_matrix[j][k]* \
                                         self.specresp[j]*self.exposure      
        print("Arf loaded")
        return 
        
    def _get_tlmin(self, h):
        """
        Get the tlmin keyword for `F_CHAN`.

        Parameters:
        -----------
        h : an astropy.io.fits.hdu.table.BinTableHDU object
            The extension containing the `F_CHAN` column

        Returns:
        --------
        tlmin : int
            The tlmin keyword
        """
        # get the header
        hdr = h.header
        # get the keys of all
        keys = np.array(list(hdr.keys()))

        # find the place where the tlmin keyword is defined
        t = np.array(["TLMIN" in k for k in keys])

        # get the index of the TLMIN keyword
        tlmin_idx = np.hstack(np.where(t))[0]

        # get the corresponding value
        tlmin = int(list(hdr.items())[tlmin_idx][1])
        return tlmin

    def rebin_channels(self,new_bounds_lo,new_bounds_hi):
        """
        This method rebins the response matrix resp_matrix to an input, 
        continuous grid of channels, and updates the emin, emax and n_chans  
        attributes appropriately. In order to avoid potential issues with gain 
        shifting due bin edges, the method also re-aligns the input grid such 
        that its bounds match the (finer) existing grid. With this method, both 
        the probability to assign an energy channel to a recorded photon, and 
        the total number of photons in the model, are conserved after rebinning.
        
        Parameters:
        -----------    
        new_bounds_lo: np.array(float)
            An array of energies with the lower bound of each energy channel. 
            
        new_bounds_hi: np.array(float)  
            An array of energies with the upper bound of each energy channel.     
        
        Returns:
        --------
        bin_resp: ResponseMatrix
            A ResponseMatrix object containing the same response loaded in the 
            self object, but rebinned over the channel axis to the input grid.
        """
    
        if new_bounds_lo[0] < self.emin[0]:
            raise ValueError("New channel grid below lower limit of existing one")
        if  new_bounds_hi[-1] > self.emax[-1]:
            raise ValueError("New channel grid above upper limit of existing one")
        if len(new_bounds_lo) != len(new_bounds_hi):
            raise TypeError("Lower and upper bounds of new channel grid have different size")
        if len(new_bounds_lo) > self.n_chans:
            raise TypeError("You can not rebin to a finer channel grid")
        #raise error if channels have 0 spacing
        
        #shift the new bounds to coincide match with the existing channel bounds
        new_bounds_lo = self._align_grid(self.emin,new_bounds_lo)
        new_bounds_hi = self._align_grid(self.emax,new_bounds_hi)
        rebinned_response = np.zeros((self.n_energs,len(new_bounds_lo)))

        #rebin by summing over all energy bins
        for j in range(self.n_energs):
            rebinned_response[j,:] = self._rebin_sum(self.resp_matrix[j,:],
                                                     (self.emin,self.emax),
                                             (new_bounds_lo,new_bounds_hi))
        
        bin_resp = copy.copy(self)
        bin_resp.emin = new_bounds_lo
        bin_resp.emax = new_bounds_hi
        bin_resp.n_chans = len(new_bounds_lo)
        bin_resp.chans = np.linspace(0,bin_resp.n_chans-1,bin_resp.n_chans)
        bin_resp.resp_matrix = rebinned_response
        return bin_resp

    def rebin_energies(self,factor):
        """
        This method rebins the response matrix resp_matrix in energy by grouping
        the a numer of existing binning (defined by the energ_lo and energ_hi 
        attributes) in coarser bins. The input parameter "factor" controls how 
        many bins of the old grid will be grouped in the new grid. With this 
        method, both the probability to assign an energy channel to a recorded 
        photon, and the total number of photons in the model, are conserved 
        after rebinning.        
        Note that rebinning in energy is VERY dangerous and should only be done
        very carefully.
        
        Parameters:
        -----------   
        factor: integer
            The number of bins of the old energy grid that will be grouped in 
            the new energy grid  
        
        Returns:
        -------- 
        bin_resp: ResponseMatrix
            A ResponseMatrix object containing the same response loaded in the 
            self object, but rebinned over the energy axis to the input grid.
        """    
        
        if factor < 1:
            raise ValueError("You can not rebin to a finer energy grid")

        if (isinstance(factor, int) != True):
            raise TypeError("Rebinning factor must be an integer!") 

        warnings.warn("WARNING: rebinning a response in energy is dangerous, use at your own risk!",
                      UserWarning) 

        new_bounds_lo = self._integer_slice(self.energ_lo,factor)
        new_bounds_hi = np.append(new_bounds_lo[1:],self.energ_hi[-1])
        
        rebinned_response = np.zeros((len(new_bounds_lo),self.n_chans))
        
        for j in range(self.n_chans):
            rebinned_response[:,j] = self._rebin_int(self.resp_matrix[:,j],
                                             (self.energ_lo,self.energ_hi),
                                             (new_bounds_lo,new_bounds_hi),
                                                               renorm=True)
        
        bin_resp = copy.copy(self)
        bin_resp.energ_lo = new_bounds_lo
        bin_resp.energ_hi = new_bounds_hi
        bin_resp.n_energs = len(new_bounds_lo)
        bin_resp.resp_matrix = rebinned_response
        return bin_resp

    def convolve_response(self,model_input,units_in="xspec",units_out="kev"):
        """
        This method applies the response matrix loaded in the class to a user
        defined mode. 
        Two input model normalizations are supported: "rate" normalization, 
        which assumes the input is in units of count rate, or "xspec" 
        normalization, which assumes the model is in units of count rate times
        energy bin width. 
        Two output normalizations are supported: "kev", in which case the model 
        is returned in units of counts/s/kev, or "channel", in which case the  
        model is returned in units of counts/s/channel.
        
        Parameters:
        -----------      
        model_input: np.array(float,float) or CrossSpectrum
            Either a) a 2-d array of size (n_energs x arbirtrary length), 
            containing the input model as a function of energy and optionally an 
            additional quantity (Fourier frequency, time, pulse phase, etc.), 
            or b) a CrossSpectrum object from nDspec, containing the model cross
            spectrum to be folded with the instrument response.
            
        units_in: string, default="xspec"
            A string detailing the normalization of the iinput model. The base 
            "xspec" normalization assumes the input is in units of integrated 
            counts per unit time in each energy bin; "rate" normalization assumes
            the input is in units of integrated count rate per unit time and 
            energy in each bin. 
            
        units_out: string, default="kev"
            A string setting the normalization of the output model. The default 
            "kev" normalizations returns a model in units of counts/s/kev, 
            "channel" instead returns a model in units of counts/s/channel.
        
        Returns:
        -------- 
        conv_model, np.array(float,float) or CrossSpectrum
            Either a) a 2-d array of size (n_chans x arbitrary length), 
            containing the input model as a function of energy channel and a 
            secondary quantity identical to the input model_input (Fourier 
            frequency, time, pulse phase, etc.), or a CrossSpectrum object from
            nDspec, containing the folded model cross spectrum as a function of
            energy channel and Fourier frequency.
        """

        #if passing a nDspec CrossSpectrum object, we are returning a new class 
        #instance, otherwise just a matrix with the folded input model 
        if isinstance(model_input,CrossSpectrum):
            unfolded_model = model_input.cross 
            resp_energs = self._grid_bounds_to_midpoint(self.emin,self.emax)
            output_model = CrossSpectrum(model_input.times,
                                         energ = resp_energs,
                                         freqs = model_input.freqs,
                                         method = model_input.method)
            if hasattr(model_input,"power_spec"):
                output_model.set_psd_weights(model_input.power_spec)
        else: 
           unfolded_model = model_input 
    
        if np.shape(self.resp_matrix)[0] != np.shape(unfolded_model)[0]:
            raise TypeError(("Model energy grid has a different size from"
                             " response"))    

        #all the transpose calls are to get the right format for the matrix 
        #multiplication                 
        if units_in == "rate":
            bin_widths = self.energ_hi-self.energ_lo
            renorm_model = np.multiply(np.transpose(unfolded_model),bin_widths)
            conv_model = np.matmul(renorm_model,self.resp_matrix)
        elif units_in == "xspec":
            trans_model = np.transpose(unfolded_model)
            conv_model = np.matmul(trans_model,self.resp_matrix)
        else:
            raise ValueError(("Please specify units of either count rate or"
                              " count rate normalized to bin width"))
            
        #convert to per kev units if desired:
        if units_out == "kev":
            bin_widths = self.emax-self.emin
            conv_model = conv_model/bin_widths 
        elif units_out != "channel":
            raise ValueError(("Output units incorrect, specify either kev or"
                              " channel"))
        #finally transpose to obtain the correct format 
        conv_model = np.transpose(conv_model)
        
        if isinstance(model_input,CrossSpectrum):
            output_model.cross = conv_model
        else:
            output_model = conv_model                
        return output_model

    def plot_response(self,plot_type="channel",return_plot=False):
        """
        Plots the instrument response as a function of incoming energy and 
        instrument channel. For ease of visualization, the z-axis plots 
        the base-10 logarithm of the response matrix. 
        
        Parameters:
        -----------             
        plot_type: string, default="channel"
            Sets the units of the X-axis to be either the channel number (by 
            default) or the bounds of each channel (plot_type="energy").
        
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
    
        fig = plt.figure(figsize=(9.,7.5))
        
        if plot_type == "channel":
            x_axis = self.chans
            plt.xlabel("Channel")
        elif plot_type == "energy":
            x_axis = (self.emax+self.emin)/2.
            plt.xlabel("Bounds (keV)")
        else:
            raise TypeError("Specify either channel or energy for the x-axis")
                
        energy_array = (self.energ_hi+self.energ_lo)/2.
        p = plt.pcolormesh(x_axis,energy_array,np.log10(self.resp_matrix),
                           cmap="PuRd",shading='auto',linewidth=0,
                           rasterized=True)
        fig.colorbar(p)
        plt.ylabel("Energy (keV)")
        plt.title("log10(Response)")
        plt.show()
        
        if return_plot is True:
            return fig 
        else:
            return   

        
    def plot_arf(self,plot_scale="log",return_plot=False):
        """
        Plots the instrument effective area, if one has been loaded, as a 
        function of energy. 
        
        Parameters:
        -----------             
        plot_scale: string, default="log"
            Switches between log10(arf) (plot_scale="log", the default behavior)
            and just the arf (plot_scale="lin").
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
    
        #tbd: only allow this to happen if specresp is defined
        energy_array = (self.energ_hi+self.energ_lo)/2.
        fig = plt.figure(figsize=(9.,7.5))
        plt.plot(energy_array,self.specresp,linewidth=2.5,color=colorscale[3])
        plt.xlabel("Energy (keV)")
        plt.ylabel("Effective area (cm$^{2}$)")
        plt.yscale("log",base=10)
        
        if plot_scale == "log":
            plt.xscale("log",base=10)
        elif plot_scale != "lin":
            raise TypeError(("Please specify either linear (lin) or"
                             " logarithmic (log) x scale")) 
        
        plt.show()
        
        if return_plot is True:
            return fig 
        else:
            return   
        
    def diagonal_matrix(self,num):
        """
        Returns a diagonal identity matrix, which by definition contains only
        ones on the diagonal and zeroes  otherwise.
        
        Parameters:
        ----------             
        num: int
            The dimension of the desired matrix.
            
        Returns: 
        --------
        diag_resp: np.array(float,float)
            An identity matrix of size (num x num).   
        """
    
        diag_resp = np.diag(np.ones(num))
        return diag_resp            
              
    def unfold_response(self,array,units_in="kev"):    
        """
        Unfolds an array through the instrument response. Note that plotting 
        data in this fashion can be EXTREMELY misleading and should be done 
        with care. In nDspec we define an unfolded model as:
        unfolded(H) = counts(H)/exposure*sum(rmf*arf),
        where H is a given bounds in energy channels, exposure is the exposure 
        time of the observation, rmf*arf is the instrument response, and the sum 
        is carried out every the energy bins of the response. Users also need to 
        specify whether the input array is in units of counts/s/keV or 
        counts/s/channel; if they do so correctly, the output of this method is 
        in photon density - counts/s/keV/cm^2. Assuming 0 background, this 
        definition of unfolding is identical to Isis, regardless of model 
        choice, and Xspec, as long as the model is a constant in each energy 
        bin. Converting to flux units - ie, energy/s/area, then requires 
        multiplying the output of this method by H^2, identically to the 
        "eeunfold" method in Xspec. 
        Unlike Isis and Xspec, this method also supports unfolding two-d arrays,
        e.g. cross spectra. 
        
        Parameters:
        -----------             
        array: np.array(int,int)
            The input array to be unfolded, of size (n_energs,arbitrary). 
            In the x-axis it needs to be defined over the instrument energy 
            grid, in the y-axis it can be any size (e.g., over a grid of Fourier
            frequencies).
            
        Returns: 
        -------- 
        unfold_model: np.array(float,float)
            The array unfolded through the instrument response, of size 
            (n_chans,arbitrary), defined over the energy bounds of each channel 
            in the response. The y-axis is identical to the input.
        """
        #reshaping the input array needed to treat a 1d array like a 2d one and
        #use the same array operations later
        if (array.size == self.n_chans):
            array = array.reshape(self.n_chans,1)         
        #reshaping the energy and channel arrays is necessary to get the right 
        #dimensions when unfolding 2d arrays 
        energy_widths = self.energ_hi - self.energ_lo 
        unfold_matrix = energy_widths.reshape(self.n_energs,1)*self.resp_matrix
        unfold_array = np.sum(unfold_matrix,axis=0).reshape(self.n_chans,1) 

        if units_in == "channel":
            unfold_model = array/unfold_array
        elif units_in == "kev":
            channel_widths = (self.emax - self.emin).reshape(self.n_chans,1)
            unfold_model = array/unfold_array*channel_widths 
        else:
            raise ValueError("Specify whether the input array is normalized per channel or per keV")
        
        #if we had a 1d array as input, we convert back to a 1d array; otherwise 
        #this confuses matplotlib and produces weird plots
        if (unfold_model.size == self.n_chans):
            unfold_model = unfold_model.reshape(self.n_chans)      
        
        return unfold_model
       
    def set_exposure_time(self,time):
        """
        Adjusts the response matrix to a new time. Some mission's FITS files
        do not contain this information and thus need to be set manually.

        Parameters
        ----------
        time : float
            Exposure time of observation.

        Returns
        -------
        None.

        """
        if (isinstance(time, (np.floating, float, int)) != True):
            raise TypeError("Time must be a float or integer")
        
        factor = time/self.exposure
        
        self.resp_matrix = self.resp_matrix*factor
        self.exposure = time        
        return
        
