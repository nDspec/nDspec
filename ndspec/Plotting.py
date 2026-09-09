import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib import rc
import colorsys

def set_plot_style(use_tex=True,font_size=17,font_family='serif',font=None):
    """
    This function sets the matplotlib style used throughout the library. It is a 
    separate function to avoid a missing LateX installation from messing with
    importing nDspec.
    
    Parameters:
    -----------
    use_tex: bool, default=True
        A boolean to choose whether text is rendered through LaTeX. If LaTeX is 
        not available on the system, this should be set to False. 
        
    font_size: float, default=17
        The base font size scale for all plots.
        
    font_family: string, default='serif'
        The matplotlib font family to use. Standard choices are 'serif', 
        'sans-serif', 'cursive', 'fantasy' and 'monospace'.
        
    font: string, default=None
        The specific font within font_family to use. If use_tex is True, this 
        should be a LaTeX-supported font (e.g., 'Computer Modern' for serif, or 
        'Helvetica' for sans-serif). If None, the matplotlib/LaTeX default for 
        the chosen font_family is used instead.
    """
    
    if font is None:
        if font_family == 'serif':
            font = 'Computer Modern'
        elif font_family == 'sans-serif':
            font = 'Helvetica'
    
    if use_tex is True:
        rc('text',usetex=True)
        rc('font',**{'family':font_family,font_family:[font]})
    else:
        plt.rcParams['font.family'] = font_family
        if font is not None:
            plt.rcParams[font_family] = [font]
    plt.rcParams.update({'font.size':font_size})
    
    return

def make_panel_data(x_points=None,y_points=None,x_bars=None,y_bars=None,
                    model_points=None,model_vals=None,model_edges=None,
                    bkg_vals=None,component_vals=None,
                    resid=None,reserr=None,x_label="",y_label="",
                    res_label=""):
    """
    This function assembles the dictionary of arrays that can be included in 
    a one-dimensional panel from either operator or fitter objects. 
    
    Parameters:
    -----------
    x_points: numpy.ndarray, default=None 
        The position of bin centers.
        
    y_points: numpy.ndarray, default=None 
        The value of each data/model point.
        
    x_bars, y_bars: numpy.ndarray, default=None 
        The half-width of the errorbar along each axis.
        
    model_points: numpy.ndarray, default=None 
        The grid over which the model is evaluated. 
        
    model_vals: numpy.ndarray, default=None 
        The total model values, evaluated over model_points.
        
    model_edges: numpy.ndarray, default=None 
        The bin edges of the noticed channels, of length one greater than 
        model_vals. If this is provided, the model and its components are drawn 
        as a step histogram over these edges; if it is None, they are drawn as 
        a line through model_points instead.
        
    bkg_vals: numpy.ndarray, default=None 
        The background, evaluated over x_points. 
        
    component_vals: dict, default=None 
        The individual additive components of the model.
        
    resid, reserr: numpy.ndarray, default=None 
        The residuals and the half-width of their errorbars.
        
    x_label, y_label, res_label: str, default="" 
        The labels of the x axis, of the y axis, and of the y axis of the 
        residual panel.
    
    Returns:
    --------
    panel_data: dict 
        A dictionary containing every key listed above.
    """
    
    panel_data = dict(x_points=x_points,
                      y_points=y_points,
                      x_bars=x_bars,
                      y_bars=y_bars,
                      model_points=model_points,
                      model_vals=model_vals,
                      model_edges=model_edges,
                      bkg_vals=bkg_vals,
                      component_vals=component_vals,
                      resid=resid,
                      reserr=reserr,
                      x_label=x_label,
                      y_label=y_label,
                      res_label=res_label)
    
    return panel_data

def make_mesh_data(x_points=None,y_points=None,z_values=None,x_label="",
                   y_label="",z_label=""):
    """
    This function assembles the dictionary of arrays that can be included in 
    a two-dimensional panel from either operator or fitter objects. 
    
    Parameters:
    -----------
    x_points, y_points: numpy.ndarray, default=None 
        The grids over which the mesh is defined.
        
    z_values: numpy.ndarray, default=None 
        The two-dimensional array to be displayed, masked and oriented so that 
        its first axis corresponds to y_points.
        
    x_label, y_label, z_label: str, default="" 
        The labels of the two axes and of the colorbar.
    
    Returns:
    --------
    mesh_data: dict 
        A dictionary containing every key listed above.
    """
    
    mesh_data = dict(x_points=x_points,
                     y_points=y_points,
                     z_values=z_values,
                     x_label=x_label,
                     y_label=y_label,
                     z_label=z_label)
    
    return mesh_data


def make_layout(nrows=1,ncols=1,height_ratios=None,width_ratios=None,
                sharex=False,projections=None,colorbars=False,
                panel_size=(6.,4.5),panels=None):
    """
    This function assembles the dictionary describing the geometry of a panel 
    grid, without creating any plot object. It is used by the get_plot_layout 
    methods of the fitters and operators to declare how much canvas they 
    require.
    
    Parameters:
    -----------
    nrows, ncols: int, default=1
        The number of rows and columns in the panel grid.
        
    height_ratios: list(float), default=None 
        The relative height of each row in the grid. If None, every row is 
        given the same height.
        
    width_ratios: list(float), default=None 
        The relative width of each column in the grid. If None, every column 
        is given the same width.
        
    sharex: bool, default=False 
        A boolean to choose whether the panels in each column share their x 
        axis. 
        
    projections: list(str), default=None 
        The matplotlib projection of each panel, listed in row-major order. 
        Used to treat plots both in a rectilinear projection, and in the polar 
        projection (e.g. for polarization).
        
    colorbars: bool, default=False 
        A boolean to choose whether the panels in the grid require space to be 
        reserved for a colorbar. 
        
    panel_size: tuple(float), default=(6.,4.5)
        The size in inches of a single panel of the grid. 
        
    panels: dict, default=None 
        The role of each panel of the grid, keyed by name and containing the 
        row and column index of the corresponding panel - for example 
        panels={"data":(0,0),"residuals":(1,0)}.
    
    Returns:
    --------
    layout: dict 
        A dictionary containing every key listed above.
    """
    
    layout = dict(nrows=nrows,
                  ncols=ncols,
                  height_ratios=height_ratios,
                  width_ratios=width_ratios,
                  sharex=sharex,
                  projections=projections,
                  colorbars=colorbars,
                  panel_size=panel_size,
                  panels=panels)
    
    return layout

def make_panels(layout,fig=None,subplot_spec=None,squeeze=True,hspace=None,wspace=None):
    """
    This function builds the grid of axis objects described by an input layout 
    dictionary. If an existing figure and a subplot specification are provided, 
    the grid  is nested inside the figure/subplot.
    
    Parameters:
    -----------
    layout: dict 
        The panel geometry to be built, as returned by make_layout.
        
    fig: matplotlib.figure.Figure, default=None 
        An existing figure in which to nest the grid. If None, a new figure is 
        created.
        
    subplot_spec: matplotlib.gridspec.SubplotSpec, default=None 
        The region of an existing figure in which to nest the grid. If None, 
        the grid occupies the entire figure.
        
    squeeze: bool, default=True 
        A boolean to choose whether axes of length one are removed from the 
        returned array. 

    hspace, wspace: float, default=None 
        The vertical and horizontal space between the panels of the grid, as a 
        fraction of the size of a panel.
    
    Returns:
    --------
    fig: matplotlib.figure.Figure
        The figure containing the grid.
        
    axes: matplotlib.axes.Axes or numpy.ndarray(matplotlib.axes.Axes)
        The panels of the grid, in row-major order.
    """

    #set the spacing and if it's not passed, default to constrained layout
    spacing = dict(hspace=hspace,wspace=wspace)
    constrained = (hspace is None and wspace is None)
    
    if fig is None:
        figsize = (layout["ncols"]*layout["panel_size"][0],
                   layout["nrows"]*layout["panel_size"][1])
        fig = plt.figure(figsize=figsize,constrained_layout=constrained)
    
    if subplot_spec is None:
        grid = gridspec.GridSpec(layout["nrows"],layout["ncols"],figure=fig,
                                 height_ratios=layout["height_ratios"],
                                 width_ratios=layout["width_ratios"],
                                 **spacing)
    else:
        grid = gridspec.GridSpecFromSubplotSpec(
                                 layout["nrows"],layout["ncols"],
                                 subplot_spec=subplot_spec,
                                 height_ratios=layout["height_ratios"],
                                 width_ratios=layout["width_ratios"],
                                 **spacing)
    
    axes = np.empty((layout["nrows"],layout["ncols"]),dtype=object)
    for row in range(layout["nrows"]):
        for col in range(layout["ncols"]):
            index = row*layout["ncols"] + col
            projection = None
            if layout["projections"] is not None:
                projection = layout["projections"][index]
            shared_axis = None
            if layout["sharex"] is True and row > 0:
                shared_axis = axes[0,col]
            axes[row,col] = fig.add_subplot(grid[row,col],
                                            projection=projection,
                                            sharex=shared_axis)
    
    if squeeze is True:
        if axes.size == 1:
            return fig, axes[0,0]
        axes = np.squeeze(axes)
    
    return fig, axes

def _merge_style(defaults,overrides):
    """
    This function allows the appearance of every element of a plot to be 
    controlled by sorting through a set of keyword arguments provided by 
    the user.
    
    Parameters:
    -----------
    defaults: dict 
        The default keyword arguments of whatever is to be plotted.
        
    overrides: dict or None 
        The keyword arguments supplied by the user. If None, the defaults are 
        returned unchanged.
    
    Returns:
    --------
    style: dict 
        The merged keyword arguments, to be passed on to matplotlib.
    """
    
    style = dict(defaults)
    if overrides is not None:
        style.update(overrides)
    
    return style

def draw_main_panel(axes,panel_data,colour="C0",draw_data=True,
                    draw_model=True,draw_bkg=False,draw_components=False,
                    log_xaxis=True,log_yaxis=True,xlim=None,ylim=None,
                    data_kwargs=None,model_kwargs=None,
                    bkg_kwargs=None,component_kwargs=None):
    """
    This function draws the contents of an input dictionary from make_panel_data
    on an input axis object from make_panels. It is used for one-dimensional plots 
    from fitter objects (excluding residuals), and from operator classes.
         
    The appearance of every element can be overridden through the corresponding 
    keyword dictionary. For example, passing data_kwargs={"alpha":1.,"marker":"s"} 
    draws the data points opaque and square.
    
    Parameters:
    -----------
    axes: matplotlib.axes.Axes
        The panel onto which the contents are drawn.
        
    panel_data: dict 
        The dictionary of arrays to be displayed, as returned by make_panel_data. 
        If an element (e.g. the model) is not contained in the dictionary, it
        just get skipped.
        
    colour: str, default="C0"
        The base colour of the plot added to the panel. By default, the model is
        changed to a darkened shade of it to better separate it from the data.
        
    draw_data: bool, default=True 
        A boolean to choose whether the data points will be drawn.
        
    draw_model: bool, default=True 
        A boolean to choose whether the total model will be drawn.
        
    draw_bkg: bool, default=False 
        A boolean to choose whether the background will be drawn.
        
    draw_components: bool, default=False 
        A boolean to choose whether the individual additive components of the 
        model will be drawn.

    xlim, ylim: (float,float), default=None
        If specified, these set the x and y ranges for the plot.
        
    log_xaxis, log_yaxis: bool, default=True 
        Booleans to choose whether each axis uses a logarithmic scale.
        
    data_kwargs, model_kwargs, bkg_kwargs, component_kwargs: dict, default=None 
        Keyword arguments passed to matplotlib for each element of the panel, 
        overriding the defaults described above.
    """
    
    if draw_data is True:
        style = _merge_style(dict(linestyle='',marker='o',alpha=0.35,
                                  color=colour),data_kwargs)
        axes.errorbar(panel_data["x_points"],panel_data["y_points"],
                      xerr=panel_data["x_bars"],yerr=panel_data["y_bars"],
                      **style)
    
    if draw_bkg is True and panel_data.get("bkg_vals") is not None:
        style = _merge_style(dict(linestyle='',marker='s',alpha=0.35,
                                  color=colour),bkg_kwargs)
        axes.errorbar(panel_data["x_points"],panel_data["bkg_vals"],
                      xerr=panel_data["x_bars"],**style)
    
    if draw_model is True:
        style = _merge_style(dict(linewidth=2.5,zorder=10,
                                  color=darken_colour(colour)),model_kwargs)
        #if panel_data contains bin edges, draw the model as a histogram
        #otherwise just plot a line for each point
        if panel_data.get("model_edges") is not None:
            axes.stairs(panel_data["model_vals"],panel_data["model_edges"],
                        baseline=None,**style)
        else:
            axes.plot(panel_data["model_points"],panel_data["model_vals"],
                      **style)
    
    if draw_components is True:
        draw_model_components(axes,panel_data,colour=colour,
                              component_kwargs=component_kwargs)
    
    if log_xaxis is True:
        axes.set_xscale("log",base=10)
    if log_yaxis is True:
        axes.set_yscale("log",base=10)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    
    axes.set_xlabel(panel_data["x_label"])
    axes.set_ylabel(panel_data["y_label"])
    
    return

def draw_model_components(axes,panel_data,colour="C0",
                          component_kwargs=None):
    """
    This function draws the individual additive components of a model onto an 
    existing axis object. The components are those returned by the model_expand 
    function in Utils.py, which unpacks a composite model into just its additive
    components. 
    
    Parameters:
    -----------
    axes: matplotlib.axes.Axes
        The panel onto which the components are drawn.
        
    panel_data: dict 
        The arrays to be displayed, as returned by make_panel_data. The 
        component_vals entry is required.
        
    colour: str, default="C0"
        The base colour of the dataset the components belong to.
        
    component_kwargs: dict, default=None 
        Keyword arguments passed to matplotlib, overriding the defaults.
    """
    
    if panel_data.get("component_vals") is None:
        return
    
    style = _merge_style(dict(linestyle='-',linewidth=2.,alpha=0.8),
                         component_kwargs)
    
    for key in panel_data["component_vals"].keys():
        if panel_data.get("model_edges") is not None:
            axes.stairs(panel_data["component_vals"][key],
                        panel_data["model_edges"],baseline=None,label=key,
                        **style)
        else:
            axes.plot(panel_data["model_points"],
                      panel_data["component_vals"][key],label=key,**style)
    
    axes.legend(loc="best")
    
    return


def draw_residual_panel(axes,panel_data,residuals,colour="C0",
                        log_xaxis=True,log_yaxis=False,
                        resid_kwargs=None,line_kwargs=None):
    """
    This function draws the residuals stored in an input dictionary onto an 
    existing axis object, together with a horizontal reference line 
    appropriate to the residuals in use. 
    
    Parameters:
    -----------
    axes: matplotlib.axes.Axes
        The panel onto which the residuals are drawn.
        
    panel_data: dict 
        The arrays to be displayed, created by make_panel_data.
        
    residuals: str 
        The units of the residuals being displayed. This sets the value of the 
        horizontal reference line.
        
    colour: str, default="C0"
        The colour of the residual points, ignored if resid_kwargs sets a 
        colour of its own.

    log_xaxis, log_yaxis: bool, default=True, False
        Booleans to choose whether each axis uses a logarithmic scale.
        
    resid_kwargs: dict, default=None 
        Keyword arguments passed to matplotlib for the residual points, 
        overriding the defaults.
        
    line_kwargs: dict, default=None 
        Keyword arguments passed to matplotlib for the horizontal reference 
        line, overriding the defaults.
    """
    
    reference = 0.
    if residuals == "ratio":
        reference = 1.
    
    style = _merge_style(dict(linestyle='',marker='o',color=colour),
                         resid_kwargs)
    axes.errorbar(panel_data["x_points"],panel_data["resid"],
                  xerr=panel_data["x_bars"],yerr=panel_data["reserr"],**style)
    
    style = _merge_style(dict(linestyle=':',linewidth=2.,color='black'),
                         line_kwargs)
    axes.axhline(reference,**style)
    
    axes.set_xlabel(panel_data["x_label"])
    axes.set_ylabel(panel_data["res_label"])

    if log_xaxis is True:
        axes.set_xscale("log",base=10)
    if log_yaxis is True:
        axes.set_yscale("log",base=10)
    
    return

def draw_colormesh_panel(axes,panel_data,cmap="viridis",diverging=False,
                         log_xaxis=False,log_yaxis=True,
                         xlim=None,ylim=None,colorbar=True,                         
                         mesh_kwargs=None):
    """
    This function draws a two-dimensional dataset or model stored in an input 
    input dictionary onto an existing axis object, as a colormesh with an 
    optional colorbar.
    
    Parameters:
    -----------
    axes: matplotlib.axes.Axes
        The panel onto which the colormesh is drawn.
        
    panel_data: dict 
        The arrays to be displayed, typically created by make_mesh_data. The x_points, 
        y_points and z_values entries are required; z_values is expected to be masked 
        and oriented so that its first axis corresponds to y_points.
        
    cmap: str, default="viridis"
        The colormap used to display z_values.
        
    diverging: bool, default=False 
        A boolean to choose whether the colormap is normalized symmetrically 
        around zero, through get_diverging_norm. This is appropriate for 
        quantities that change sign, such as time lags.
        
    log_xaxis, log_yaxis: bool, default=False, True 
        Booleans to choose whether each axis uses a logarithmic scale.

    xlim, ylim: (float,float), default=None
        If specified, these set the x and y ranges for the plot.
        
    colorbar: bool, default=True 
        A boolean to choose whether a colorbar is attached to the panel. This 
        is set to False when several panels share a single colorbar.
        
    mesh_kwargs: dict, default=None 
        Keyword arguments passed to matplotlib for the colormesh, overriding 
        the defaults.
    
    Returns:
    --------
    mesh: matplotlib.collections.QuadMesh
        The colormesh that was drawn.
    """
    
    norm = None
    ticks = None
    if diverging is True:
        norm, ticks = get_diverging_norm(panel_data["z_values"])
    
    style = _merge_style(dict(cmap=plt.get_cmap(cmap),norm=norm,
                              shading="auto"),mesh_kwargs)
    mesh = axes.pcolormesh(panel_data["x_points"],panel_data["y_points"],
                           panel_data["z_values"],**style)
    
    if colorbar is True:
        bar = axes.get_figure().colorbar(mesh,ax=axes,ticks=ticks)
        bar.set_label(panel_data["z_label"])
    
    if log_xaxis is True:
        axes.set_xscale("log",base=10)
    if log_yaxis is True:
        axes.set_yscale("log",base=10)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    
    axes.set_xlabel(panel_data["x_label"])
    axes.set_ylabel(panel_data["y_label"])
    #the reason we return mesh is so that a colorbar can be attached to it
    #somewhere else on the grid/plot
    return mesh


def draw_polarization_ellipse(axes,mod_angle,pol_degree,pol_error,
                              angle_error,colour="C0",fill=True):
    """
    This function draws a single confidence ellipse in the polarization plane 
    onto an existing polar axis object. 
    
    Parameters:
    -----------
    axes: matplotlib.axes.Axes
        The panel onto which the ellipse is drawn. This must have been created 
        with the polar projection.
        
    mod_angle: float 
        The polarization angle at the centre of the ellipse, in radians.
        
    pol_degree: float 
        The polarization degree at the centre of the ellipse.
        
    pol_error: float 
        The one sigma uncertainty on the polarization degree, which sets the 
        radial extent of the ellipse.
        
    angle_error: float 
        The one sigma uncertainty on the polarization angle in radians, which 
        sets the azimuthal extent of the ellipse.
        
    colour: str, default="C0"
        The colour of the ellipse.
        
    fill: bool, default=True 
        A boolean to choose whether the ellipse is filled or drawn as an 
        outline only.
    """

    #sort out which parts from the polarimetry classes are needed here
    print("Work in progress!")
    
    return

def get_diverging_norm(values,n_ticks=5):
    """
    This function returns the colormap normalization and tick positions used to 
    display a quantity that changes sign, so that the midpoint of the colormap 
    is always zero.
    
    Parameters:
    -----------
    values: numpy.ndarray 
        The array to be displayed, which may be masked.
        
    n_ticks: int, default=5 
        The number of ticks to place on the colorbar.
    
    Returns:
    --------
    norm: matplotlib.colors.TwoSlopeNorm
        The normalization to be passed to pcolormesh.
        
    ticks: numpy.ndarray 
        The positions of the colorbar ticks. 
    """
    
    minimum = np.min(values)
    maximum = np.max(values)
    
    #if the array is only positive or only negative, set an 
    #artificial cap/floor to create the ticks and normalization
    if minimum >= 0.:
        minimum = -0.05*maximum
    if maximum <= 0.:
        maximum = -0.05*minimum
    
    norm = mcolors.TwoSlopeNorm(vmin=minimum,vcenter=0.,vmax=maximum)
    ticks = np.linspace(minimum,maximum,n_ticks)
    
    return norm, ticks

def plot_marginal_colormesh(xaxis,yaxis,values,marginal_x=None,
                            marginal_y=None,x_label="",y_label="",z_label="",
                            marginal_labels=("",""),cmap="PuRd",
                            diverging=False,log_xaxis=False,log_yaxis=False,
                            log_zaxis=False,colour=None,panel_size=(9.0,9.0),
                            mesh_kwargs=None,bottom_kwargs=None,side_kwargs=None):
    """
    This function displays a two-dimensional array as a colormesh, together 
    with a one-dimensional projection along each axis drawn in a panel 
    adjacent to the corresponding side of the mesh. 
    
    Parameters:
    -----------
    xaxis, yaxis: numpy.ndarray 
        The grids over which the array is defined.
        
    values: numpy.ndarray 
        The two-dimensional array to be displayed.
        
    marginal_x, marginal_y: numpy.ndarray 
        The one-dimensional projections of the array along each axis. If either is 
        None, it is computed as the sum of the array over the other axis; if a 
        projection is not wanted at all, the corresponding panel is left empty by
        passing an array of zeros.

    x_label, y_label, z_label: str, default="" 
        The labels of the two axes of the mesh and of its colorbar.
    
    marginal_labels: tuple(str), default=("","") 
        The labels of the two projections, in the order (x,y). These are drawn 
        on the axis of each projection that is not shared with the mesh.
        
    cmap: str, default="PuRd"
        The colormap used to display the array.

    diverging: bool, default=False 
        A boolean to choose whether the colormap is normalized symmetrically 
        around zero, as described in draw_colormesh_panel.

    log_xaxis, log_yaxis: bool, default=False 
        Booleans to choose whether each axis of the mesh, and the matching axis 
        of each projection, uses a logarithmic scale.
        
    log_zaxis: bool, default=False 
        A boolean to choose whether the two projections use a logarithmic scale 
        along the axis holding their values. This does not affect the mesh, 
        whose scaling is set through diverging instead.
        
    colour: str or tuple, default=None 
        The colour of the two projections. If it is None, a colour is sampled 
        from the colormap of the mesh.
    
    panel_size: tuple(float), default=(3.5,3.5)
        The size in inches of a single panel of the figure.
        
    mesh_kwargs, bottom_kwargs, side_kwargs: dict, default=None 
        Keyword arguments passed to matplotlib for the mesh and for each 
        projection, overriding the defaults.        
    
    Returns:
    --------
    fig: matplotlib.figure.Figure
        The figure containing the plot, if return_plot is True.
        
    axes: numpy.ndarray(matplotlib.axes.Axes)
        The panels of the plot, if return_plot is True. The mesh occupies the 
        lower left panel, the projection along the x axis the upper left panel, 
        and the projection along the y axis the lower right panel; the upper 
        right panel of the grid is unused and is removed.

    """
    #set the color of the projection
    if colour is None:
        colour = plt.get_cmap(cmap)(0.75)

    #set the kwargs and arrays on the side panels
    bottom_kwargs = _merge_style(dict(color=colour),bottom_kwargs)
    side_kwargs = _merge_style(dict(color=colour),side_kwargs)    
    if marginal_x is None:
        marginal_x = np.sum(values,axis=0)
    if marginal_y is None:
        marginal_y = np.sum(values,axis=1)

    #prepare the layout+panels and remove the unused bottom right panel
    panels = {"mesh":(0,0),"bottom":(1,0),"right":(0,1),"unused":(1,1)}
    layout = make_layout(nrows=2,ncols=2,height_ratios=[3,1],
                         width_ratios=[3,1],sharex=False,
                         panel_size=(0.5*panel_size[0],0.5*panel_size[1]),
                         panels=panels)
    fig, axes = make_panels(layout,squeeze=False,hspace=0,wspace=0)
    axes[panels["unused"]].remove()

    #align the panels properly
    axes[panels["bottom"]].sharex(axes[panels["mesh"]])
    axes[panels["right"]].sharey(axes[panels["mesh"]])

    #draw the main twod plot
    mesh_data = make_mesh_data(x_points=xaxis,y_points=yaxis,z_values=values,
                               x_label=x_label,y_label=y_label,
                               z_label=z_label)
    mesh = draw_colormesh_panel(axes[panels["mesh"]],mesh_data,cmap=cmap,
                                diverging=diverging,log_xaxis=log_xaxis,
                                log_yaxis=log_yaxis,colorbar=False,
                                mesh_kwargs=mesh_kwargs)
    #attach the colorbar to the mesh
    bar = fig.colorbar(mesh,ax=axes[panels["right"]])
    bar.set_label(z_label)
    
    #draw the bottom projection of the 2d plot
    bottom_data = make_panel_data(model_points=xaxis,model_vals=marginal_x,
                                  x_label=x_label,
                                  y_label=marginal_labels[0])
    draw_main_panel(axes[panels["bottom"]],bottom_data,draw_data=False,
                    log_xaxis=log_xaxis,log_yaxis=log_zaxis,
                    model_kwargs=bottom_kwargs)
    
    #draw the side projection of the 2d plot; it is done explicitely 
    #because it is rotated 90 degrees
    style = _merge_style(dict(linewidth=2.5),side_kwargs)
    axes[panels["right"]].plot(marginal_y,yaxis,**style)
    axes[panels["right"]].set_xlabel(marginal_labels[1])
    axes[panels["right"]].invert_xaxis()
    if log_zaxis is True:
        axes[panels["right"]].set_xscale("log",base=10)
    if log_yaxis is True:
        axes[panels["right"]].set_yscale("log",base=10)

    #set the labels
    axes[panels["mesh"]].set_xlabel("")
    axes[panels["mesh"]].tick_params(labelbottom=False)
    axes[panels["right"]].set_ylabel("")
    axes[panels["right"]].tick_params(labelleft=False)
    
    return fig, axes

def darken_colour(colour,factor=0.6):
    """
    This function returns a darker shade of an input colour, and is used to 
    draw a model in a shade of the colour of the data it is being compared to.
    
    Parameters:
    -----------
    colour: str or tuple 
        The colour to be darkened, in any format understood by matplotlib.
        
    factor: float, default=0.6 
        The factor by which the lightness of the colour is multiplied.
    
    Returns:
    --------
    darker_colour: tuple 
        The darkened colour, as an rgb tuple.
    """
    
    rgb = mcolors.to_rgb(colour)
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    darker_colour = colorsys.hls_to_rgb(hue,factor*lightness,saturation)
    
    return darker_colour
