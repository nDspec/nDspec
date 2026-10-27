import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__/ndspec/'))))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import pytest

import ndspec.Plotting as Plotting

#make_panel_data
def test_make_panel_data_returns_all_keys():
    panel_data = Plotting.make_panel_data(x_points=np.arange(5),
                                          y_points=np.arange(5))
    expected_keys = {"x_points","y_points","x_bars","y_bars","model_points",
                     "model_vals","model_edges","bkg_vals","component_vals",
                     "resid","reserr","x_label","y_label","res_label"}
    assert set(panel_data.keys()) == expected_keys

def test_make_panel_data_mismatched_data_raises():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(x_points=np.arange(5),y_points=np.arange(4))

def test_make_panel_data_mismatched_bars_raise():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(x_points=np.arange(5),y_points=np.arange(5),
                                 x_bars=np.arange(3))
    with pytest.raises(ValueError):
        Plotting.make_panel_data(x_points=np.arange(5),y_points=np.arange(5),
                                 y_bars=np.arange(3))

def test_make_panel_data_mismatched_bkg_raises():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(x_points=np.arange(5),y_points=np.arange(5),
                                 bkg_vals=np.arange(4))

def test_make_panel_data_resid_must_match_data():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(x_points=np.arange(5),y_points=np.arange(5),
                                 resid=np.arange(4),reserr=np.ones(4))

def test_make_panel_data_reserr_must_match_resid():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(resid=np.arange(5),reserr=np.ones(4))

def test_make_panel_data_resid_without_reserr_raises():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(resid=np.zeros(5))

def test_make_panel_data_reserr_without_resid_raises():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(reserr=np.ones(5))

def test_make_panel_data_matching_resid_ok():
    panel_data = Plotting.make_panel_data(x_points=np.arange(5),
                                          y_points=np.arange(5),
                                          resid=np.zeros(5),
                                          reserr=np.ones(5))
    assert len(panel_data["resid"]) == len(panel_data["x_points"])

def test_make_panel_data_mismatched_model_raises():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(model_points=np.arange(5),
                                 model_vals=np.arange(4))

def test_make_panel_data_bad_model_edges_raises():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(model_vals=np.arange(5),
                                 model_edges=np.arange(5))

def test_make_panel_data_valid_model_edges_ok():
    panel_data = Plotting.make_panel_data(model_vals=np.arange(5),
                                          model_edges=np.arange(6))
    assert len(panel_data["model_edges"]) == len(panel_data["model_vals"])+1

def test_make_panel_data_mismatched_component_raises():
    with pytest.raises(ValueError):
        Plotting.make_panel_data(model_points=np.arange(5),
                                 model_vals=np.arange(5),
                                 component_vals={"comp1":np.arange(4)})

#make_mesh_data
def test_make_mesh_data_returns_all_keys():
    mesh_data = Plotting.make_mesh_data(x_points=np.arange(3),
                                        y_points=np.arange(4),
                                        z_values=np.zeros((4,3)))
    expected_keys = {"x_points","y_points","z_values","x_label","y_label",
                     "z_label"}
    assert set(mesh_data.keys()) == expected_keys

def test_make_mesh_data_wrong_ndim_raises():
    with pytest.raises(ValueError):
        Plotting.make_mesh_data(x_points=np.arange(3),y_points=np.arange(4),
                                z_values=np.zeros(12))

def test_make_mesh_data_transposed_shape_raises():
    with pytest.raises(ValueError):
        Plotting.make_mesh_data(x_points=np.arange(3),y_points=np.arange(4),
                                z_values=np.zeros((3,4)))

#make_layout
def test_make_layout_returns_all_keys():
    layout = Plotting.make_layout(nrows=2,ncols=2)
    expected_keys = {"nrows","ncols","height_ratios","width_ratios","sharex",
                     "projections","colorbars","panel_size","panels"}
    assert set(layout.keys()) == expected_keys

def test_make_layout_mismatched_height_ratios_raises():
    with pytest.raises(ValueError):
        Plotting.make_layout(nrows=2,ncols=1,height_ratios=[1,2,3])

def test_make_layout_mismatched_width_ratios_raises():
    with pytest.raises(ValueError):
        Plotting.make_layout(nrows=1,ncols=2,width_ratios=[1])

def test_make_layout_mismatched_projections_raises():
    with pytest.raises(ValueError):
        Plotting.make_layout(nrows=1,ncols=2,projections=["polar"])

def test_make_layout_out_of_bounds_panel_raises():
    with pytest.raises(ValueError):
        Plotting.make_layout(nrows=1,ncols=1,panels={"data":(1,0)})

def test_make_layout_valid_panels_ok():
    layout = Plotting.make_layout(nrows=2,ncols=1,
                                  panels={"data":(0,0),"residuals":(1,0)})
    assert layout["panels"]["residuals"] == (1,0)

#make_panels
def test_make_panels_single_panel_returns_one_axis():
    layout = Plotting.make_layout(nrows=1,ncols=1)
    fig, axes = Plotting.make_panels(layout)
    assert axes.__class__.__name__ == "Axes"
    plt.close(fig)

def test_make_panels_grid_returns_array_of_axes():
    layout = Plotting.make_layout(nrows=2,ncols=2)
    fig, axes = Plotting.make_panels(layout)
    assert axes.shape == (2,2)
    plt.close(fig)

def test_make_panels_squeeze_false_keeps_shape():
    layout = Plotting.make_layout(nrows=1,ncols=1)
    fig, axes = Plotting.make_panels(layout,squeeze=False)
    assert axes.shape == (1,1)
    plt.close(fig)

#_merge_style
def test_merge_style_none_overrides_returns_defaults():
    defaults = dict(color="C0",alpha=0.5)
    assert Plotting._merge_style(defaults,None) == defaults

def test_merge_style_overrides_take_precedence():
    style = Plotting._merge_style(dict(color="C0",alpha=0.5),dict(color="C1"))
    assert style["color"] == "C1"
    assert style["alpha"] == 0.5

def test_merge_style_does_not_mutate_defaults():
    defaults = dict(color="C0")
    Plotting._merge_style(defaults,dict(color="C1"))
    assert defaults["color"] == "C0"

#draw_main_panel
def test_draw_main_panel_draws_data_and_model():
    fig, axes = plt.subplots()
    panel_data = Plotting.make_panel_data(x_points=np.arange(1,6),
                                          y_points=np.arange(1,6),
                                          model_points=np.arange(1,6),
                                          model_vals=np.arange(1,6))
    Plotting.draw_main_panel(axes,panel_data)
    assert len(axes.containers) == 1
    assert len(axes.lines) > 0
    plt.close(fig)

def test_draw_main_panel_missing_data_raises():
    fig, axes = plt.subplots()
    panel_data = Plotting.make_panel_data(model_points=np.arange(1,6),
                                          model_vals=np.arange(1,6))
    with pytest.raises(ValueError):
        Plotting.draw_main_panel(axes,panel_data,draw_data=True,
                                 draw_model=False)
    plt.close(fig)

def test_draw_main_panel_missing_model_raises():
    fig, axes = plt.subplots()
    panel_data = Plotting.make_panel_data(x_points=np.arange(1,6),
                                          y_points=np.arange(1,6))
    with pytest.raises(ValueError):
        Plotting.draw_main_panel(axes,panel_data,draw_data=False,
                                 draw_model=True)
    plt.close(fig)

def test_draw_main_panel_missing_bkg_raises():
    fig, axes = plt.subplots()
    panel_data = Plotting.make_panel_data(x_points=np.arange(1,6),
                                          y_points=np.arange(1,6))
    with pytest.raises(ValueError):
        Plotting.draw_main_panel(axes,panel_data,draw_model=False,
                                 draw_bkg=True)
    plt.close(fig)

def test_draw_main_panel_bkg_drawn_when_present():
    fig, axes = plt.subplots()
    panel_data = Plotting.make_panel_data(x_points=np.arange(1,6),
                                          y_points=np.arange(1,6),
                                          bkg_vals=np.ones(5))
    Plotting.draw_main_panel(axes,panel_data,draw_model=False,draw_bkg=True)
    assert len(axes.containers) == 2
    plt.close(fig)

#draw_model_components
def test_draw_model_components_draws_each_component():
    fig, axes = plt.subplots()
    panel_data = Plotting.make_panel_data(
        model_points=np.arange(1,6),model_vals=np.arange(1,6),
        component_vals={"comp1":np.arange(1,6),"comp2":np.arange(1,6)*2})
    Plotting.draw_model_components(axes,panel_data)
    assert len(axes.lines) == 2
    plt.close(fig)

def test_draw_model_components_labels_match_keys():
    fig, axes = plt.subplots()
    panel_data = Plotting.make_panel_data(
        model_points=np.arange(1,6),model_vals=np.arange(1,6),
        component_vals={"comp1":np.arange(1,6),"comp2":np.arange(1,6)*2})
    Plotting.draw_model_components(axes,panel_data)
    labels = [line.get_label() for line in axes.lines]
    assert labels == ["comp1","comp2"]
    plt.close(fig)

#draw_residual_panel
def test_draw_residual_panel_missing_resid_raises():
    fig, axes = plt.subplots()
    panel_data = Plotting.make_panel_data(x_points=np.arange(1,6))
    with pytest.raises(ValueError):
        Plotting.draw_residual_panel(axes,panel_data,"delta")
    plt.close(fig)

#draw_colormesh_panel
def test_draw_colormesh_panel_returns_mesh():
    fig, axes = plt.subplots()
    panel_data = Plotting.make_mesh_data(x_points=np.arange(1,4),
                                         y_points=np.arange(1,5),
                                         z_values=np.random.rand(4,3))
    mesh = Plotting.draw_colormesh_panel(axes,panel_data,colorbar=False)
    assert mesh.__class__.__name__ == "QuadMesh"
    plt.close(fig)

def test_draw_colormesh_panel_diverging_norm_centred_on_zero():
    fig, axes = plt.subplots()
    values = np.linspace(-1.,1.,12).reshape((4,3))
    panel_data = Plotting.make_mesh_data(x_points=np.arange(1,4),
                                         y_points=np.arange(1,5),
                                         z_values=values)
    mesh = Plotting.draw_colormesh_panel(axes,panel_data,diverging=True,
                                         colorbar=False)
    assert mesh.norm.vcenter == 0.
    plt.close(fig)

#plot_marginal_colormesh
def test_plot_marginal_colormesh_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Plotting.plot_marginal_colormesh(np.arange(1,4),np.arange(1,5),
                                         np.zeros((3,3)))

def test_plot_marginal_colormesh_marginal_x_mismatch_raises():
    with pytest.raises(ValueError):
        Plotting.plot_marginal_colormesh(np.arange(1,4),np.arange(1,5),
                                         np.zeros((4,3)),
                                         marginal_x=np.zeros(2))

def test_plot_marginal_colormesh_marginal_y_mismatch_raises():
    with pytest.raises(ValueError):
        Plotting.plot_marginal_colormesh(np.arange(1,4),np.arange(1,5),
                                         np.zeros((4,3)),
                                         marginal_y=np.zeros(2))
