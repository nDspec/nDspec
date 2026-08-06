import numpy as np
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

from matplotlib import rc, rcParams
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams.update({'font.size': 17})

from .Operator import nDspecOperator

class PolarimetryProduct(Operator.nDspecOperator):
    """
    This class is used to operate on polarimetric model products, in 
    particular (but not limited to) spectro-polarimetry. It handles 
    conversions between model Stokes parameters (I, Q, U), polarization
    degree/angle (Pi, psi), and modulation curves.

    The object is initialized in one of two modes at construction, 
    depending on the initial inputs:

    - 'stokes':       the user supplies Stokes parameters, which can then 
                      be converted to polarization degree/angle, or to a
                      modulation curve per data bin
    - 'polarization': the user supplies stokes I (an array of count rates  
                      per bin), polarization degree Pi and polarization 
                      angle psi.

    Parameters:
    -----------
    bins : array_like(float)
        Energy (or time, or other dimensions) bin centers.
    input_type : {'stokes', 'polar'}
        Specifies which units the user wishes to define as input.

    Attributes:
    -----------
    n_bins: int
        The length of the arrays containing stokes parameters or 
        polarization degree/angle.

    stokes_I, stokes_Q, stokes_U: array_like(float)
        The arrays containing the Stokes parameters. Note that the
        intensity in stokes_I is always required, since it sets the 
        absolute scale of the modulation curve and is necessary 
        to recover stokes Q/U from polarization degree/angle.

    pol_degree, pol_angle: array_like(float)
        The arrays containing the polarization degree/angle over all
        bins. Note that the polarization angle is defined in radians,
        NOT degrees.

    mod_angles: array_like(float)
        An optional array containing the grid of modulation angles 
        over which to compute the modulation curve

    mod_factor: array_like(float)
        An optional array containing the grid of modulation factors 
        for each bin in `bins`, and used to compute the modulation 
        curve

    modulation_curve: array_like(float)
        An array containing the modulation curve(s) for each bin, 
        computed from the input Stokes parameters or polarization
        degree/angle
    """

    _valid_types = ('stokes', 'polarization')

    def __init__(self, bins, input_type):
        #this has the problem of being bin centers and not edges but eh
        #keep for now, fix plots later
        self.bins = np.asarray(bins, dtype=float)
        self.n_bins = self.bins.size

        if input_type not in self._valid_types:
            raise ValueError(
                f"input_type must be one of {self._valid_types}, got {input_type!r}"
            )
        self.input_type = input_type

        self.stokes_I = None
        self.stokes_Q = None
        self.stokes_U = None
        self.pol_degree = None
        self.pol_angle = None

        self.mod_angles = None
        self.mod_factor = None
        self.modulation_curve = None
        pass

    def set_stokes(self, I, Q, U):
        """
        This setter method is used to define all three Stokes
        parameters over each bin covered by the object. 

        Parameters:
        I, Q, U: array_like(float)
            The arrays containing the Stokes parameters to be stored
        -----------
        """
        if self.input_type != 'stokes':
            raise ValueError(
                f"This object was initialized with input_type={self.input_type!r}; "
                "set_stokes() is only valid for input_type='stokes'."
            )
        self.stokes_I = self._check_shape(I, self.n_bins, "I")
        self.stokes_Q = self._check_shape(Q, self.n_bins, "Q")
        self.stokes_U = self._check_shape(U, self.n_bins, "U")
        return

    def set_polarization(self, I, degree, angle):
        """
        This setter method is used to define the Stokes I 
        intensity parameter, as well as the polarization degree
        and angle, in every bincovered by the object

        Parameters:
        -----------
        I: array_like(float)
            An array containing the Stokes I values for each bin 

        degree: array_like(float)
            An array containing the polarization degree for each bin

        angle: array_like(float)
            An array containing the polarization angle for each bin        
        """
        if self.input_type != 'polarization':
            raise ValueError(
                f"This object was initialized with input_type={self.input_type!r}; "
                "set_polarization() is only valid for input_type='polarization'."
            )
        self.stokes_I = self._check_shape(I, self.n_bins, "I")
        self.pol_degree = self._check_shape(degree, self.n_bins, "polarization degree")
        self.pol_angle = self._check_shape(angle, self.n_bins, "polarization angle")
        return

    def set_modulation_angles(self, angles):
        """
        This setter method is used to  define the grid of modulation angles
        to use when calculating the modulation curve. Typically, this should 
        be between 0 and pi, since the modulation angles are defined in radians.

        Parameters:
        -----------
        angles: array_like(float)
            An array of modulation angles to be used.
        """
        self.mod_angles = np.asarray(angles, dtype=float)
        return

    def rotate_polarization(self,rotation):
        """
        This method rotates the Stokes Q and U stored in the object by a given 
        angle, following the standard rotation of Stokes parameters:
 
        q' = q*cos(2*delta) - u*sin(2*delta) \n
        u' = q*sin(2*delta) + u*cos(2*delta)
 
        where delta is the rotation angle. Stokes I, and therefore the 
        polarization degree, are left unchanged by the rotation. The rotated 
        Stokes Q and U are stored back into the stokes_Q and stokes_U arrays.
        
        Parameters:
        -----------
        angle_rotation: float 
            The angle, in degrees, by which to rotate the stored polarization 
            state. 
 
        Returns:
        --------
        model: np.array(float), shape (3, len(energs))
            The rotated Stokes vector (stokes_I, stokes_Q, stokes_U), identical 
            to what is now stored in this object.
        """
        if len(rotation) != 1:
            raise ValueError(f"This method only supports rotating by a single angle, "
                               "but input size is {len(rotation)}")

        self._require('stokes_I', 'stokes_Q', 'stokes_U')
        delta = np.radians(angle_rotation)
        cos_rotation = np.cos(2.*delta)
        sin_rotation = np.sin(2.*delta)
        stokes_Q = self.stokes_Q*cos_rotation-self.stokes_U*sin_rotation
        stokes_U = self.stokes_Q*sin_rotation+self.stokes_U*cos_rotation
        self.stokes_Q = stokes_Q
        self.stokes_U = stokes_U
        model = np.array([self.stokes_I,self.stokes_Q,self.stokes_U])
        return model

    def set_modulation_factor(self, mu):
        """
        This setter method is used to define the values of the modulation
        factors in each data bin, when calculating the modulation curve.

        Parameters:
        -----------
        mu: array_like(float)
            An array of modulation factors to be used.
        """
        mu = np.asarray(mu, dtype=float)
        if mu.size not in (1, self.n_bins):
            raise ValueError(
                f"mod_factor must be scalar or have length n_bins={self.n_bins}, "
                f"got shape {mu.shape}"
            )
        self.mod_factor = mu
        return
    
    def stokes_to_polarization(self):
        """
        This method converts the stored Stokes parameters into arrays 
        of polarization degree/angle, and stores them internally.

        Returns:
        --------
        self.pol_degree: np.array(float)
            An array containing the polarization degree in ech bin.

        self.pol_angle: np.array(float)
            An array containing the polarization angle in ech bin.
        """
        self._require('stokes_I', 'stokes_Q', 'stokes_U')
        self.pol_degree = (
            np.sqrt(self.stokes_Q**2 + self.stokes_U**2) / self.stokes_I
        )
        # arctan2, not arctan: keeps the correct quadrant and avoids a
        # divide-by-zero warning when Q == 0.
        self.pol_angle = 0.5 * np.arctan2(self.stokes_U, self.stokes_Q)
        return self.pol_degree, self.pol_angle

    def polarization_to_stokes(self):
        """
        This method converts the stored Stokes I, polarization degree, 
        and polarization angle arrays into Stokes Q and U, and stores 
        them internally.

        Returns:
        --------
        self.stokes_I: np.array(float)
            An array containing the value of Stokes I in ech bin.

        self.stokes_Q: np.array(float)
            An array containing the value of Stokes Q in ech bin.

        self.stokes_U: np.array(float)
            An array containing the value of Stokes U in ech bin.
        """
        
        self._require('stokes_I', 'pol_degree', 'pol_angle')
        self.stokes_Q = self.stokes_I * self.pol_degree * np.cos(2 * self.pol_angle)
        self.stokes_U = self.stokes_I * self.pol_degree * np.sin(2 * self.pol_angle)
        return self.stokes_I, self.stokes_Q, self.stokes_U

    def stokes_to_modulation(self):
        """
        This method computes the modulation curve over a grid of 
        modulation angles, and in each bin, starting from the stored 
        Stokes parameters. Mathematically:
        
        mod(bin, phi) = [I + mu*(Q*cos(2*phi) + U*sin(2*phi))] / (2*pi)

        Returns:
        --------
        self.modulation_curve: array_like(n_bins, n_angles)
            A two-dimensional array containing the modulation curve in
            each data and modulation angle bin set in the object.
        """
        self._require('stokes_I', 'stokes_Q', 'stokes_U', 'mod_angles', 'mod_factor')
        I = self._as_column(self.stokes_I)
        Q = self._as_column(self.stokes_Q)
        U = self._as_column(self.stokes_U)
        mu = self._as_column(self.mod_factor)
        phi = self._as_row(self.mod_angles)

        self.modulation_curve = (
            I + mu * (Q * np.cos(2 * phi) + U * np.sin(2 * phi))
        ) / (2 * np.pi)
        return self.modulation_curve

    def polarization_to_modulation(self):
        """
        This method computes the modulation curve over a grid of 
        modulation angles, and in each bin, starting from the stored 
        polarization degree and angle. Mathematically:
        
        mod(bin, phi) = I/(2*pi) * [1 + mu*Pi*cos(2*(phi - psi))]

        Returns:
        --------
        self.modulation_curve: array_like(n_bins, n_angles)
            A two-dimensional array containing the modulation curve in
            each data and modulation angle bin set in the object.
        """
        self._require('stokes_I', 'pol_degree', 'pol_angle', 'mod_angles', 'mod_factor')
        I = self._as_column(self.stokes_I)
        Pi = self._as_column(self.pol_degree)
        psi = self._as_column(self.pol_angle)
        mu = self._as_column(self.mod_factor)
        phi = self._as_row(self.mod_angles)

        self.modulation_curve = (
            I / (2 * np.pi) * (1 + mu * Pi * np.cos(2 * (phi - psi)))
        )
        return self.modulation_curve

    def plot_stokes(self, x_label="bin", return_plot=False):
        """
        This method plots Stokes I, Q, U vs. all the bins defined in the 
        object.

        Parameters:
        -----------
        x_label: str
            An optional string to label the x-axis of the plot.

        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        
        self._require('stokes_I', 'stokes_Q', 'stokes_U')
        labels = ['Stokes I', 'Stokes Q', 'Stokes U']
        arrays = [self.stokes_I, self.stokes_Q, self.stokes_U]

        fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15, 5))
        for ax, label, arr in zip(axes, labels, arrays):
            ax.plot(self.bins, arr, marker='o', ms=3)
            ax.set_title(label)
            ax.set_xlabel(x_label)
        
        plt.tight_layout()
        plt.show()        
        
        if return_plot is True:
            return fig 
        else:
            return  

    def plot_polarization_1d(self, x_label="bin", return_plot=False):
        """
        This method plots polarizatoin degree and angle vs. all the bins defined 
        in the object, using regular one-dimensional plots.

        Parameters:
        -----------
        x_label: str
            An optional string to label the x-axis of the plot.

        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        
        self._require('pol_degree', 'pol_angle')
        fig, ((ax1,ax2)) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))

        ax1.plot(self.bins, self.pol_degree, marker='o', ms=3)
        ax1.set_ylabel('Polarization degree')
        ax1.set_xlabel(x_label)
        
        ax2.plot(self.bins, np.degrees(self.pol_angle),marker='o', ms=3)
        ax2.set_ylabel('Polarization angle (deg)')
        ax2.set_xlabel(x_label)

        plt.tight_layout()
        plt.show()        
        
        if return_plot is True:
            return fig 
        else:
            return  

    def plot_polarization_slice(self, cmap='viridis', marker='o', return_plot=False):
        """
        This method plots polarizatoin degree and angle vs. all the bins defined 
        in the object, showing polar coordinates. Note that due to the ambiguity 
        in X-ray detectors, the angles shown are limited from 0 to 180 degrees 
        (or 0 to pi radians) only. The markers shown are colored by polarization
        degree.

        Parameters:
        -----------
        cmap: str, default='viridis'
            Name of the colormap to color the markers.

        marker: str, default='o'
            The maker to use to plot the computed values.

        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        self._require('pol_degree', 'pol_angle')
 
        # wrap angle into [0, pi) since EVPA is only defined mod pi
        psi_wrapped = np.mod(self.pol_angle, np.pi)
 
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_thetamin(0)
        ax.set_thetamax(180)
 
        sc = ax.scatter(
            psi_wrapped, self.pol_degree,
            c=self.bins, cmap=cmap, marker=marker,
            s=350, edgecolors='k', linewidths=0.5
        )

        ax.set_title('Polarization degree / angle by bin')
        fig.colorbar(sc, ax=ax, label='Bin', pad=0.1)

        plt.tight_layout()
        plt.show()        
        
        if return_plot is True:
            return fig 
        else:
            return  

    def plot_modulation(self, bin_index=None, y_label="bins", renormalize=True,
                        cmap='viridis', return_plot=False):
        """
        This method plots the modulation curve in the bins chosen by th euser.

        If bin_index is given (or there is only one bin), the method plots 
        a single 1D curve vs. modulation angle. Otherwise plots the full (n_bins,
        n_angles) modulation curve as a 2D image.

        For clarity, the plot optionally be re-normalized by dividing the 
        modulation curve by stokes I. This can help in cases where stokes I
        varies very strongly from one data bin to the next (for instance,
        if it is a power-law).

        Parameters:
        -----------
        bin_index: array_like(int)
            One or more integers to pick which bins for which to plot the
            modulation curve 

        y_label: str, default=`bins`
            The name to give to the y axis of the plot, when plotting in two
            dimensions against all bins 

        renormalize: bool, default=True
            A boolean to choose whether to re-normalize the modulation curve 
            by stokes I for visualization purposes.

        cmap: str, default=`viridis`
            The name of the color map to use when plotting all the modulation 
            curves in two dimensions
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        if renormalize is True:
            self._require('stokes_I','modulation_curve', 'mod_angles')
        else:
            self._require('modulation_curve', 'mod_angles')
        
        curve = self.modulation_curve
        mod_name = "Modulation"
        
        if renormalize is True:
            I = self._as_column(self.stokes_I)
            curve = curve/I
            mod_name = "Normalized modulation"
        
        fig, ax = plt.subplots()
        if bin_index is not None:
            for index in bin_index:
                ax.plot(self.mod_angles, curve[index, :])
            ax.set_xlabel('Modulation angle (rad)')
            ax.set_ylabel(mod_name)
        elif curve.shape[0] == 1:
            ax.plot(self.mod_angles, curve[0, :])
            ax.set_xlabel('Modulation angle (rad)')
            ax.set_ylabel(mod_name)
        else:
            im = ax.pcolormesh(self.mod_angles, self.bins, curve, 
                               cmap=cmap, shading='auto',
                               rasterized=True,linewidth=0)
            ax.set_xlabel('Modulation angle (rad)')
            ax.set_ylabel(y_label)
            fig.colorbar(im, ax=ax, label=mod_name)
        
        plt.tight_layout()
        plt.show()        
        
        if return_plot is True:
            return fig 
        else:
            return  
