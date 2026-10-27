import numpy as np
import os
import warnings

from .Operator import nDspecOperator
from . import Plotting

class PolarimetryProduct(nDspecOperator):
    """
    This class is used to operate on polarimetric model products, in particular
    (but not limited to) spectro-polarimetry. It handles conversions between
    model Stokes parameters (I, Q, U), polarization degree/angle, and modulation
    curves.

    The object is initialized in one of two modes:

    - 'stokes':       the user supplies Stokes parameters, which can then be
                      converted to polarization degree/angle, or to a modulation
                      curve, per data bin
    - 'polarization': the user supplies Stokes I (an array of count rates per 
                      bin), polarization degree Pi and polarization angle psi,
                      and can then derive Stokes Q/U and/or modulation curves.                     

    Parameters:
    -----------
    bins : array_like(float)
        Energy (or time, or other dimensions) bin centers.
    input_type : {'stokes', 'polar'}
        Specifies which units the user wishes to define as input.

    Attributes:
    -----------
    n_bins: int
        The length of the arrays containing stokes parameters or polarization
        degree/angle.

    stokes_I, stokes_Q, stokes_U: array_like(float)
        The arrays containing the input Stokes parameters over all bins. 

    pol_degree, pol_angle: array_like(float)
        The arrays containing the input polarization degree/angle over all bins
        bins. The polarization angle is defined in radians.

    mod_angles: array_like(float)
        An optional array containing the grid of modulation angles over which to
        compute the modulation curve.

    mod_factor: array_like(float)
        An optional array containing the grid of modulation factors for each bin
        in `bins`; it is only used to compute the modulation curve.

    modulation_curve: array_like(float)
        An array containing the modulation curve(s) for each bin, computed from 
        the input Stokes parameters or polarization degree/angle.
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
        This setter method is used to define all three Stokes parameters over 
        each bin covered by the object. 

        Parameters:
        -----------
        I, Q, U: array_like(float)
            The arrays containing the Stokes parameters to be stored.      
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
        This setter method is used to define the polarization degree
        and angle, as well as the Stokes I parameter, in every bin covered by 
        the object.

        Parameters:
        -----------
        I: array_like(float)
            An array containing the Stokes I values for each bin. 

        degree: array_like(float)
            An array containing the fractional polarization degree for each bin.

        angle: array_like(float)
            An array containing the polarization angle in radians for each bin.        
        """
        if self.input_type != 'polarization':
            raise ValueError(
                f"This object was initialized with input_type={self.input_type!r}; "
                "set_polarization() is only valid for input_type='polarization'."
            )
        if np.any(degree < 0.) or np.any(degree > 1.):
            raise ValueError("The polarization degree must be between 0 and 1")
            
        self.stokes_I = self._check_shape(I, self.n_bins, "I")
        self.pol_degree = self._check_shape(degree, self.n_bins, "polarization degree")
        self.pol_angle = self._check_shape(angle, self.n_bins, "polarization angle")
        return

    def set_modulation_angles(self, angles):
        """
        This setter method is used to define the grid of modulation angles (in
        radians) to use when calculating the modulation curve. 

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
 
        where delta is the rotation angle. T
        
        Parameters:
        -----------
        rotation: float 
            The angle, in degrees, by which to rotate the stored polarization 
            state. 
 
        Returns:
        --------
        model: np.array(float), shape (3, len(energs))
            The rotated Stokes vector (stokes_I, stokes_Q, stokes_U) now stored 
            in the object instance
        """
        rotation = np.asarray(rotation, dtype=float)
        if rotation.size != 1:

            raise ValueError(f"This method only supports rotating by a single angle, "
                               "but input size is {len(rotation)}")

        self._require('stokes_I', 'stokes_Q', 'stokes_U')
        delta = np.radians(rotation)
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
        This method converts the stored Stokes parameters into arrays of
        polarization degree/angle, and stores them internally.

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
        This method converts the stored Stokes I, polarization degree, and 
        polarization angle arrays into Stokes Q and U, and stores them
        internally.

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
        This method computes the modulation curve over the stored grid of
        modulation angles, starting from the stored Stokes parameters. By 
        defintion:
        
        mod(bin, phi) = [I + mu*(Q*cos(2*phi) + U*sin(2*phi))] / (2*pi)

        Returns:
        --------
        self.modulation_curve: array_like(n_bins, n_angles)
            A two-dimensional array containing the modulation curve in each data
            and modulation angle bin set in the object.
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
        This method computes the modulation curve over the stored grid of
        modulation angles, and in each bin, starting from the stored 
        polarization degree and angle. By definition:
        
        mod(bin, phi) = I/(2*pi) * [1 + mu*Pi*cos(2*(phi - psi))]

        Returns:
        --------
        self.modulation_curve: array_like(n_bins, n_angles)
            A two-dimensional array containing the modulation curve in each data
            and modulation angle bin set in the object.
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

    def plot_stokes(self, x_label="bin", return_plot=False, stokes_kwargs=None):
        """
        This method plots Stokes I, Q, U vs. all the bins defined in the object.

        Parameters:
        -----------
        x_label: str
            An optional string to label the x-axis of the plot.

        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.

        stokes_kwargs: dict, default=None 
            Keyword arguments for the stokes paramters plots
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
            
        panels: np.array(matplotlib.axes), optional 
            The panels containing the plot produced by the method.
        """
        
        self._require('stokes_I', 'stokes_Q', 'stokes_U')
        labels = ['Stokes I', 'Stokes Q', 'Stokes U']
        arrays = [self.stokes_I, self.stokes_Q, self.stokes_U]
        x_axis = self.bins

        plot_layout = Plotting.make_layout(ncols=len(arrays),
                                           panel_size=(6.5,4.5))
        fig, panels = Plotting.make_panels(plot_layout)

        model_style = dict(drawstyle="steps-mid")
        if stokes_kwargs is not None:
            model_style.update(stokes_kwargs)
        
        for panel, array, label in zip(panels,arrays,labels):
            data = Plotting.make_panel_data(model_points=x_axis,
                                            model_vals=array,
                                            x_label=x_label,
                                            y_label=label)
            #log scale only for Stokes I
            Plotting.draw_main_panel(panel,data,draw_data=False,draw_model=True,
                                     log_yaxis=(label == "Stokes I"),
                                     log_xaxis=(label == "Stokes I"),
                                     model_kwargs=model_style)
            if label != "Stokes I":
                panel.axhline(0.,linestyle=':',linewidth=2.,color='black')         
        
        if return_plot is True:
            return fig, panels 
        else:
            return  

    def plot_polarization_1d(self, x_label="bin", return_plot=False, 
                             pol_kwargs=None):
        """
        This method plots polarizatoin degree and angle vs. all the bins defined 
        in the object, using one-dimensional plots.

        Parameters:
        -----------
        x_label: str
            An optional string to label the x-axis of the plot.

        return_plot: bool, default=False
            A boolean to decide whether to return the figure objected containing 
            the plot or not.

        pol_kwargs: dict, default=None 
            Keyword arguments for the polarization degree/angle plots
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
        """
        
        self._require('pol_degree', 'pol_angle')
        labels = ['Polarization degree', 'Polarization angle (deg)']
        arrays = [self.pol_degree, np.degrees(self.pol_angle)]
        x_axis = self.bins
        
        plot_layout = Plotting.make_layout(ncols=len(arrays),
                                           panel_size=(6.5,4.5))
        fig, panels = Plotting.make_panels(plot_layout)

        model_style = dict(drawstyle="steps-mid")
        if pol_kwargs is not None:
            model_style.update(pol_kwargs)
        
        for panel, array, label in zip(panels,arrays,labels):
            data = Plotting.make_panel_data(model_points=x_axis,
                                            model_vals=array,
                                            x_label=x_label,
                                            y_label=label)
            Plotting.draw_main_panel(panel,data,draw_data=False,draw_model=True,
                                     log_yaxis=False,log_xaxis=False,
                                     model_kwargs=model_style)   
        
        if return_plot is True:
            return fig, panels 
        else:
            return  

    def plot_polarization_slice(self,cmap="viridis",angle_range=None,
                                    degree_range=None,return_plot=False, 
                                    pol_kwargs=None):
            """
            This method plots polarization degree and angle for all the bins 
            defined in the object as markers in polar coordinates, with the angle 
            as the azimuth and the degree as the radius. Due to the ambiguity in 
            X-ray detectors, the angle is only defined modulo 180 degrees. The  
            markers are colored by bin.

            Parameters:
            -----------
            cmap: str, default="viridis"
                The colormap used to color the markers by bin.
                
            angle_range: list(float), default=None 
                The lower and upper bounds of the polarization angles shown, in 
                degrees. If None, the full 0 to 180 degree range is shown.
                
            degree_range: list(float), default=None 
                The lower and upper bounds of the polarization degrees shown. If 
                None, the axis runs from zero to slightly above the largest 
                value.

            return_plot: bool, default=False
                A boolean to decide whether to return the figure and panel 
                containing the plot or not.

            pol_kwargs: dict, default=None 
                Keyword arguments for the markers.
                
            Returns: 
            --------
            fig: matplotlib.figure, optional 
                The plot object produced by the method.
                
            panel: matplotlib.axes, optional 
                The panel containing the plot produced by the method.
            """
            self._require('pol_degree', 'pol_angle')
            
            plot_layout = Plotting.make_layout(panel_size=(6.5,4.5),
                                   projections=["polar"])
            fig, panel = Plotting.make_panels(plot_layout)
            
            data = Plotting.make_polar_data(model_angle=self.pol_angle,
                                            model_degree=self.pol_degree,
                                            color_values=self.bins,
                                            color_label="Bin",
                                            title="Polarization angle/degree")
            
            Plotting.draw_polar_panel(panel,data,cmap=cmap,draw_data=False,
                                      angle_range=angle_range,
                                      degree_range=degree_range,
                                      model_kwargs=pol_kwargs)
            
            if return_plot is True:
                return fig, panel
            else:
                return

    def plot_modulation(self, bin_index=None, y_label="bin", renormalize=True,
                        cmap="viridis", colors=None, return_plot=False, 
                        mod_kwargs=None):
        """
        This method plots the modulation curve in the bins chosen by the user.

        If bin_index is given (or there is only one bin), the method plots  
        a single 1D curve vs. modulation angle. Otherwise plots the full (n_bins,
        n_angles) modulation curve as a 2D plot.

        For clarity, the plot can optionally be re-normalized by dividing the 
        modulation curve by stokes I. This can help in cases where stokes I
        varies very strongly from one data bin to the next (for instance,
        if it is a power-law).

        Parameters:
        -----------
        bin_index: int or array_like(int), default=None
            One or more indexes of the bins for which to plot the modulation 
            curve.

        y_label: str, default="bin"
            The label of the y axis when plotting all bins in two dimensions.

        renormalize: bool, default=True
            A boolean to choose whether to re-normalize the modulation curve 
            by stokes I for visualization purposes.

        cmap: str, default="viridis"
            The colormap used when plotting all bins in two dimensions.

        colors: list(str), default=None
            The colors of each modulation curve when plotting in one dimension, 
            one per entry of bin_index. By default, the curves follow the 
            matplotlib color cycle.
            
        return_plot: bool, default=False
            A boolean to decide whether to return the figure and panel 
            containing the plot or not.

        mod_kwargs: dict, default=None 
            Keyword arguments for the modulation curves, or for the colormesh 
            when plotting in two dimensions.
            
        Returns: 
        --------
        fig: matplotlib.figure, optional 
            The plot object produced by the method.
            
        panel: matplotlib.axes, optional 
            The panel containing the plot produced by the method.
        """
        if renormalize is True:
            self._require('stokes_I','modulation_curve','mod_angles')
        else:
            self._require('modulation_curve','mod_angles')
        
        curve = self.modulation_curve
        mod_name = "Modulation"
        
        if renormalize is True:
            curve = curve/self._as_column(self.stokes_I)
            mod_name = "Normalised modulation"

        if bin_index is None and curve.shape[0] == 1:
            bin_index = [0]
        
        plot_layout = Plotting.make_layout(panel_size=(6.5,4.5))
        fig, panel = Plotting.make_panels(plot_layout)
        
        if bin_index is None:
            data = Plotting.make_mesh_data(x_points=self.mod_angles,
                                           y_points=self.bins,
                                           z_values=curve,
                                           x_label="Modulation angle (rad)",
                                           y_label=y_label,
                                           z_label=mod_name)
            mesh_style = dict(rasterized=True,linewidth=0)
            if mod_kwargs is not None:
                mesh_style.update(mod_kwargs)
            Plotting.draw_colormesh_panel(panel,data,cmap=cmap,
                                          log_yaxis=False,
                                          mesh_kwargs=mesh_style)
        else:
            bin_index = np.atleast_1d(bin_index)
            if colors is None:
                colors = ["C"+str(count) for count in range(len(bin_index))]
            elif len(colors) != len(bin_index):
                raise ValueError("Specify one color for each bin to be plotted")
            for index, color in zip(bin_index,colors):
                data = Plotting.make_panel_data(model_points=self.mod_angles,
                                                model_vals=curve[index,:],
                                                x_label="Modulation angle (rad)",
                                                y_label=mod_name)
                model_style = dict(label="bin = {:g}".format(self.bins[index]))
                if mod_kwargs is not None:
                    model_style.update(mod_kwargs)
                Plotting.draw_main_panel(panel,data,color=color,
                                         draw_data=False,draw_model=True,
                                         log_xaxis=False,log_yaxis=False,
                                         model_kwargs=model_style)
            panel.legend(loc="best")                               
        
        if return_plot is True:
            return fig, panel
        else:
            return
