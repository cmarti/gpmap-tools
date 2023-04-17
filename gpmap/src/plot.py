#!/usr/bin/env python
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.graph_objects as go
import datashader as ds
import holoviews as hv

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from holoviews.operation.datashader import datashade

from gpmap.src.settings import PLOTS_FORMAT
from gpmap.src.seq import guess_space_configuration
from gpmap.src.genotypes import (get_edges_coords, get_nodes_df_highlight,
                                 minimize_nodes_distance)
import warnings
from gpmap.src.utils import check_error


# Functions
def init_fig(nrow=1, ncol=1, figsize=None, style='ticks',
             colsize=3, rowsize=3):
    sns.set_style(style)
    if figsize is None:
        figsize = (colsize * ncol, rowsize * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    return(fig, axes)


def init_single_fig(figsize=None, style='ticks',
             colsize=3, rowsize=3, is_3d=False):
    sns.set_style(style)
    if figsize is None:
        figsize = (colsize, rowsize)
    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(1, 1, 1, projection='3d' if is_3d else None)
    return(fig, axes)


def savefig(fig, fpath=None, tight=True, fmt=PLOTS_FORMAT, dpi=360):
    if tight:
        fig.tight_layout()
    if fpath is not None:
        fpath = '{}.{}'.format(fpath, fmt)
        fig.savefig(fpath, format=fmt, dpi=dpi)
        plt.close()
    else:
        plt.show()
    

def save_plotly(fig, fpath=None):
    if fpath is not None:
        fpath = '{}.html'.format(fpath)
        fig.write_html(fpath)


def empty_axes(axes):
    sns.despine(ax=axes, left=True, bottom=True)
    axes.set_xticks([])
    axes.set_yticks([])


def create_patches_legend(axes, colors_dict, loc=1, **kwargs):
    axes.legend(handles=[mpatches.Patch(color=color, label=label)
                         for label, color in colors_dict.items()],
                loc=loc, **kwargs)


def plot_relaxation_times(decay_df, axes=None, fpath=None, log_scale=False,
                          neutral_time=None, kwargs={}):
    '''
    Plots the relaxation times associated to each of the calculated components
    from using ```WMWSWalk.calc_visualization`
    
    Parameters
    ----------
    decay_df : pd.DataFrame of shape (n_components, 3)
        ``pd.DataFrame`` containing the decay rates and the associated mean
        relaxation times for each of the calculated components
        
    axes : matplotlib Axes object (None)
        ``Axes`` where to plot. If not provided, a new figure will be created 
        automatically for this plot and save in the path provided by ``fpath``
        
    fpath : str (None)
        File path to store the plot. If ``fpath=None``, ``axes`` argument must
        be provided for plotting.
        
    log_scale : bool (False)
        Plot the relaxation times in log scale
        
    neutral_time : float (None)
        If provided, an additional horizontal line will be plotted representing
        the relaxation time associated to the neutral process. This is useful
        when selecting the number of relevant dimensions to plot
    
    kwargs :  dict 
        Additional key-word arguments dictionary provided for ``axes.plot`` and
        ``axes.scatter`` e.g. color.
    
    '''
    
    if axes is None and fpath is None:
        msg = 'Either axes or fpath argument must be provided'
        raise ValueError(msg)
    
    fig = None
    if axes is None:
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3)
    
    axes.plot(decay_df['k'], decay_df['relaxation_time'],
              linewidth=1, **kwargs, label='Selection')
    axes.scatter(decay_df['k'], decay_df['relaxation_time'],
                 s=15, **kwargs)
    xlims = axes.get_xlim()
    if neutral_time is not None:
        axes.plot(xlims, (neutral_time, neutral_time), lw=0.5, c='orange',
                  linestyle='--', label='Neutral')
    
    if log_scale:
        axes.set(yscale='log')
    
    axes.legend(loc=1)
    axes.set(xlabel=r'Eigenvalue order $k$', 
             ylabel=r'Relaxation time $\frac{-1}{\lambda_{k}}$',
             xticks=decay_df['k'], xlim=xlims)
    
    if fig is not None:
        savefig(fig, fpath)


def plot_edges(axes, nodes_df, edges_df, x='1', y='2', z=None,
               color='grey', width=0.5, cmap='binary',
               alpha=0.1, zorder=1, avoid_dups=False,
               max_width=1, min_width=0.1):
    '''
    Plots the edges representing the connections between states that are conneted
    in the discrete space under a particular embedding
    
    Parameters
    ----------
    axes : matplotlib ``Axes`` in which to plot the edges.
    
    nodes_df : pd.DataFrame of shape (n_genotypes, n_components + 2)
        ``pd.DataFrame`` containing the coordinates in every of the ``n_components``
        in addition to the "function" and "stationary_freq" columns. Additional
        columns are also allowed
        
    edges_df : pd.DataFrame of shape (n_edges, 2)
        ``pd.DataFrame`` the connectivity information between states of the
        discrete space to plot. It has columns "i" and "j" for the indexes
        of the pairs of states that are connected.
    
    x : str ('1')
        Column in ``nodes_df`` to use for plotting the genotypes on the x-axis
    
    y : str ('2')
        Column in ``nodes_df`` to use for plotting the genotypes on the y-axis 
    
    z : str (None)
        Column in ``nodes_df`` to use for plotting the genotypes on the z-axis.
        If provided, then a 3D plot will be produced as long as the provided
        ``axes`` object allows it.
    
    color : str  ('grey')
        Column name for the values according to which edges will be colored or
        the specific color to use for plotting the edges
    
    width : float or str
        Width of the lines representing the edges. If a ``float`` is provided,
        that will be the width used to plot every edges. If ``str``, then
        widths will be scaled according to the corresponding column
        in ``edges_df``.  
    
    cmap :  colormap or str
        Colormap to use for coloring the edges according to column ``color``
    
    alpha : float (0.1)
        Transparency of lines representing the edges
    
    zorder : int (1)
        Order in which the edges will be rendered relative to other elements.
        Generally, we would want this to be smaller than the ``zorder``
        used for plotting the nodes
    
    avoid_dups : bool (False)
        Plot edges after removing redundant lines representing connections
        between the same genotypes in two different directions
    
    max_width : float (1)
        Maximum linewidth for the edges when scaled by
        
    min_width : float (0.1)
        Maximum linewidth for the edges when scaled by
    
    Returns
    -------
    line_collection : LineCollection or Line3DCollection
    
    '''
    # TODO: get colors and width as either fixed values or from edges_df
    edges_coords = get_edges_coords(nodes_df, edges_df, x=x, y=y, z=z,
                                    avoid_dups=avoid_dups)

    if color in edges_df.columns:
        cmap = cm.get_cmap(cmap)
        color = cmap(edges_df[color])
        
    if width in edges_df.columns:
        w = edges_df[width]
        width = min_width + w * (max_width - min_width) / (w.max() - w.min())
        
    if z is None:
        ln_coll = LineCollection(edges_coords, colors=color, linewidths=width,
                                 alpha=alpha, zorder=zorder)
    else:
        ln_coll = Line3DCollection(edges_coords, color=color, linewidths=width,
                                   alpha=alpha, zorder=zorder)
    axes.add_collection(ln_coll)
    return(ln_coll)


def get_axis_lims(nodes_df, x, y, z=None):
    axis_max = max(nodes_df[x].max(), nodes_df[y].max())
    axis_min = min(nodes_df[x].min(), nodes_df[y].min())
    if z is not None:
        axis_max = max(axis_max, nodes_df[z].max())
        axis_min = min(axis_min, nodes_df[z].min())
    
    axis_range = axis_max - axis_min
    axis_lims = (axis_min - 0.05 * axis_range, axis_max + 0.05 * axis_range)
    return(axis_lims)


def plot_nodes(axes, nodes_df, x='1', y='2', z=None,
               color='function', size=2.5, cmap='viridis',
               cbar=True, cbar_axes=None, palette='Set1',
               clabel='Function',
               alpha=1, zorder=2, max_size=40, min_size=1,
               edgecolor='black', lw=0, label=None,
               sort=True, sort_by=None, ascending=False, 
               vcenter=None, vmax=None, vmin=None, fontsize=12, legendloc=0,
               autoscale_axis=False):
    '''
    Plots the nodes representing the states of the discrete space on the
    provided coordinates
    
    Parameters
    ----------
    axes : matplotlib ``Axes`` in which to plot the nodes or states.
    
    nodes_df : pd.DataFrame of shape (n_genotypes, n_components + 2)
        ``pd.DataFrame`` containing the coordinates in every of the ``n_components``
        in addition to the "function" and "stationary_freq" columns. Additional
        columns are also allowed
        
    x : str ('1')
        Column in ``nodes_df`` to use for plotting the genotypes on the x-axis
    
    y : str ('2')
        Column in ``nodes_df`` to use for plotting the genotypes on the y-axis 
    
    z : str (None)
        Column in ``nodes_df`` to use for plotting the genotypes on the z-axis.
        If provided, then a 3D plot will be produced as long as the provided
        ``axes`` object allows it.
    
    color : str  ('grey')
        Column name for the values according to which states will be colored or
        the specific color to use for plotting the states
        
    vmax : float
        Maximum value to show in the colormap
    
    vmin : float
        Minimum value to show in the colormap
        
    cmap : colormap or str
        Colormap to use for coloring the nodes according to column ``color``
    
    clabel : str
        Label for the colorbar associated to the nodes color scale
    
    cbar : bool
        Boolean variable representing whether to show the colorbar
    
    cbar_axes : matplotlib ``Axes`` 
        Axes to plot the colorbar. If not provided, it will be automatically
        adjusted to the current Axes
    
    palette : dict
        Dictionary containing the colors associated to the categories specified
        by the column ``color`` in ``nodes_df``, if they express categories
        rather than numerical values
    
    size : float (2.5)
        Size of the markers provided for plotting to ``axes.scatter``. If a
        ``float`` is provided, that will be the size used to plot every nodes.
        If ``str``, then node sizes will be scaled according to the
        corresponding column in ``nodes_df``.  
    
    alpha : float (1)
        Transparency of markers representing the nodes
    
    zorder : int (2)
        Order in which the nodes will be rendered relative to other elements.
        Generally, we would want this to be bigger than the ``zorder``
        used for plotting the edges
    
    max_size : float (1)
        Maximum linewidth for the edges when scaled by
        
    min_size : float (0.1)
        Maximum linewidth for the edges when scaled by
    
    edgecolor : str ('black')
        Color of the line edges delimiting the markers representing the nodes
        
    lw : float (0)
        Width of the line edges delimiting the markers representing the nodes
        
    sort : bool (True)
        Whether to plot nodes in according to the order of values in column
        ``sort_by``
    
    sort_by : str (None)
        Column name in ``nodes_df`` to specify the order in which nodes are
        plotted
    
    ascending : bool (False)
        Use ascending order as nodes plotting order
     
    fontsize : float (12)
        Fontsize to use for axis and legend labels
    
    legendloc : int or tuple
        Location of the legend in case of coloring according to a categoric 
        variable
        
    autoscale_axis : bool (False)
        Scale axis limits so that the span the same range of values on all axis
        for a more realistic representation of distances
    
    Returns
    -------
    line_collection : LineCollection or Line3DCollection
    
    '''
    add_cbar, add_legend = False, False
    if color in nodes_df.columns:
        
        # Categorical color map
        if nodes_df[color].dtype == object:
            if isinstance(palette, str):
                labels = np.unique(nodes_df[color])
                n_colors = labels.shape[0]
                c = sns.color_palette(palette, n_colors)
                palette = dict(zip(labels, c))
            elif not isinstance(palette, dict):
                raise ValueError('palette must be a str or dict')
            color = np.array([palette[label] for label in nodes_df[color]])
            add_legend = True
            
        # Continuous color map
        elif nodes_df[color].dtype in (float, int):
            if sort:
                if sort_by is None:
                    sort_by = color
                nodes_df = nodes_df.sort_values(sort_by, ascending=ascending)
            cmap = cm.get_cmap(cmap)
            color = nodes_df[color]
            add_cbar = True
        else:
            msg = 'color dtype is not compatible: {}'.format(nodes_df[color].dtype)
            raise ValueError(msg)
    
    if size in nodes_df.columns and not isinstance(size, int):
        s = np.power(nodes_df[size], 2)
        size = min_size + s * (max_size - min_size) / (s.max() - s.min())

    axis_lims = get_axis_lims(nodes_df, x, y, z=z)
    
    norm = None if vcenter is None else colors.CenteredNorm()
    if z is not None:
        sc = axes.scatter(nodes_df[x], nodes_df[y], zs=nodes_df[z], c=color,
                          linewidth=lw, s=size, zorder=zorder, alpha=alpha,
                          edgecolor=edgecolor, cmap=cmap, label=label,
                          vmax=vmax, vmin=vmin, norm=norm)
        axes.set_zlabel('Diffusion axis {}'.format(z), fontsize=fontsize)
        axes.set_zlim(axis_lims)
    
    else:
        sc = axes.scatter(nodes_df[x], nodes_df[y], c=color,
                          linewidth=lw, s=size, zorder=zorder, alpha=alpha,
                          edgecolor=edgecolor, cmap=cmap, label=label,
                          vmax=vmax, vmin=vmin, norm=norm)
    
    if add_cbar and cbar:
        if cbar_axes is None:
            plt.colorbar(sc, ax=axes, fraction=0.1, pad=0.02).set_label(label=clabel, fontsize=fontsize)
        else:
            plt.colorbar(sc, cax=cbar_axes, fraction=0.1, pad=0.02).set_label(label=clabel, fontsize=fontsize)
    if add_legend:
        create_patches_legend(axes, palette, loc=legendloc, fontsize=fontsize)
        
    axes.set_xlabel('Diffusion axis {}'.format(x), fontsize=fontsize)
    axes.set_ylabel('Diffusion axis {}'.format(y), fontsize=fontsize)
    if autoscale_axis:
        axes.set(xlim=axis_lims, ylim=axis_lims)


def highlight_genotype_groups(axes, nodes_df, genotype_groups,
                              x='1', y='2', z=None, size=50, edgecolor='black',
                              lw=1, palette='colorblind', legendloc=1,
                              fontsize=12, is_prot=False,
                              alphabet_type='dna', codon_table='Standard'):
    nodes_df = get_nodes_df_highlight(nodes_df, genotype_groups,
                                      alphabet_type=alphabet_type,
                                      codon_table=codon_table,
                                      is_prot=is_prot)
    
    plot_nodes(axes, nodes_df, x=x, y=y, z=z, color='group', size=size,
               palette=palette, edgecolor=edgecolor, lw=lw,
               fontsize=fontsize, legendloc=legendloc, autoscale_axis=False)


def plot_visualization(axes, nodes_df, edges_df=None, x='1', y='2', z=None,
                       nodes_color='function', nodes_size=2.5, nodes_cmap='viridis',
                       cbar=True, cbar_axes=None, palette=None, nodes_alpha=1,
                       nodes_min_size=1, nodes_max_size=40,
                       nodes_edgecolor='black', nodes_lw=0, 
                       nodes_cmap_label='Function', nodes_vmin=None, nodes_vmax=None,
                       edges_color='grey', edges_width=0.5, edges_cmap='binary',
                       edges_alpha=0.1, edges_max_width=1, edges_min_width=0.1, 
                       sort_nodes=True, ascending=False, sort_by=None,
                       fontsize=12, prev_nodes_df=None, autoscale_axis=False):
    
    if prev_nodes_df is not None:
        axis = [x, y] if z is None else [x, y, z]
        nodes_df = minimize_nodes_distance(nodes_df, prev_nodes_df, axis)
    
    plot_nodes(axes, nodes_df=nodes_df, x=x, y=y, z=z,
               color=nodes_color, size=nodes_size, cmap=nodes_cmap,
               cbar=cbar, cbar_axes=cbar_axes,
               palette=palette, alpha=nodes_alpha, zorder=2,
               max_size=nodes_max_size, min_size=nodes_min_size,
               edgecolor=nodes_edgecolor, lw=nodes_lw,
               label=None, clabel=nodes_cmap_label,
               sort=sort_nodes, sort_by=sort_by, ascending=ascending, 
               vmax=nodes_vmax, vmin=nodes_vmin, fontsize=fontsize,
               autoscale_axis=autoscale_axis)
    
    if edges_df is not None:
        plot_edges(axes, nodes_df, edges_df, x=x, y=y, z=z,
                   color=edges_color, width=edges_width, cmap=edges_cmap,
                   alpha=edges_alpha, zorder=1, avoid_dups=True,
                   max_width=edges_max_width, min_width=edges_min_width)


def get_lines_from_edges_df(nodes_df, edges_df, x=1, y=2, z=None,
                            avoid_dups=True):
    edges = get_edges_coords(nodes_df, edges_df, x=x, y=y, z=z,
                             avoid_dups=avoid_dups)
    nans = np.full((edges.shape[0], 1), fill_value=np.nan)
    line_coords = np.vstack([np.hstack([edges[:, :, i], nans]).flatten()
                             for i in range(edges.shape[2])]).T
    return(line_coords)


def plot_interactive(nodes_df, edges_df=None, fpath=None, x='1', y='2', z=None,
                     nodes_color='function', nodes_size=4,
                     nodes_cmap='viridis', nodes_cmap_label='Function',
                     edges_width=0.5, edges_color='#888', edges_alpha=0.2,
                     text=None):
    '''
    Makes an interactive plot of fitness landscape with genotypes as nodes
    and single point mutations as edges using plotly 
    
    Parameters
    ----------
    nodes_df : pd.DataFrame of shape (n_genotypes, n_components + 2)
        ``pd.DataFrame`` containing the coordinates in every of the ``n_components``
        in addition to the "function" and "stationary_freq" columns. Additional
        columns are also allowed
    
    edges_df : pd.DataFrame of shape (n_edges, 2)
        ``pd.DataFrame`` the connectivity information between states of the
        discrete space to plot. It has columns "i" and "j" for the indexes
        of the pairs of states that are connected.
        
    fpath : str
        File path in which to store the interactive plot as an html file
        
    x : str ('1')
        Column in ``nodes_df`` to use for plotting the genotypes on the x-axis
    
    y : str ('2')
        Column in ``nodes_df`` to use for plotting the genotypes on the y-axis 
    
    z : str (None)
        Column in ``nodes_df`` to use for plotting the genotypes on the z-axis.
        If provided, then a 3D plot will be produced as long as the provided
        ``axes`` object allows it.
    
    nodes_color : str  ('function')
        Column name for the values according to which states will be colored or
        the specific color to use for plotting the states
    
    nodes_size : float (2.5)
        Size of the markers provided for plotting to ``axes.scatter``. If a
        ``float`` is provided, that will be the size used to plot every nodes.
        If ``str``, then node sizes will be scaled according to the
        corresponding column in ``nodes_df``.
        
    nodes_cmap : colormap or str
        Colormap to use for coloring the nodes according to column ``color``
    
    nodes_cmap_label : str
        Label for colorbar
    
    edges_width : float or str
        Width of the lines representing the edges. If a ``float`` is provided,
        that will be the width used to plot every edges. If ``str``, then
        widths will be scaled according to the corresponding column
        in ``edges_df``.  
    
    edges_color : str
        Column name for the values according to which edges will be colored or
        the specific color to use for plotting the edges
     
    edges_alpha : float (0.2)
        Transparency of lines representing the edges
     
    text : array-like of shape (nodes_df.shape[0]) (None)
        Labels to show for each state when hovering over the markers representing
        them. If not provided, rownames of the nodes_df DataFrame will be used
    '''
    
    # Create figure
    fig = go.Figure()

    # Create nodes plot
    colorbar = dict(thickness=25, title=nodes_cmap_label, xanchor='left',
                    titleside='right', len=0.8)
    marker = dict(showscale=True, colorscale=nodes_cmap, reversescale=False,
                  color=nodes_df[nodes_color], size=nodes_size, colorbar=colorbar,
                  line_width=2)
    if text is None:
        text = nodes_df.index
    
    if z is None:
        node_trace = go.Scatter(x=nodes_df[x], y=nodes_df[y],
                                mode='markers', hoverinfo='text',
                                marker=marker, text=text, name='Genotypes')
    else:
        node_trace = go.Scatter3d(x=nodes_df[x], y=nodes_df[y], z=nodes_df[z],
                                  mode='markers', hoverinfo='text',
                                  marker=marker, text=text, name='Genotypes')
    fig.add_trace(node_trace)

    # Create edges        
    if edges_df is not None:
        edges = get_lines_from_edges_df(nodes_df, edges_df, x=x, y=y, z=z)
        if z is None:
            edge_trace = go.Scatter(x=edges[:, 0], y=edges[:, 1],
                                    line=dict(width=edges_width, color=edges_color),
                                    hoverinfo='none', mode='lines',
                                    opacity=edges_alpha, name='Mutations')
        else:
            edge_trace = go.Scatter3d(x=edges[:, 0], y=edges[:, 1], z=edges[:, 2],
                                      line=dict(width=edges_width, color=edges_color),
                                      hoverinfo='none', mode='lines',
                                      opacity=edges_alpha, name='Mutations')
        fig.add_trace(edge_trace)
    
    # Update layout        
    scene = dict(xaxis_title='Diffusion axis {}'.format(x),
                 yaxis_title='Diffusion axis {}'.format(y))
    if z is not None:
        scene['zaxis_title'] = 'Diffusion axis {}'.format(z)
        
    fig.update_layout(title="Landscape visualization", 
                      hovermode='closest', template='simple_white',
                      xaxis_title='Diffusion axis {}'.format(x),
                      yaxis_title='Diffusion axis {}'.format(y),
                      scene=scene)
    
    save_plotly(fig, fpath=fpath)
    return(fig)


def figure_visualization(nodes_df, edges_df=None, fpath=None, x='1', y='2', z=None,
                         nodes_color='function', nodes_size=None, nodes_cmap='viridis',
                         nodes_alpha=1, nodes_min_size=1, nodes_max_size=40,
                         nodes_edgecolor='black', nodes_lw=0, 
                         nodes_cmap_label='Function',
                         nodes_vmin=None, nodes_vmax=None,
                         edges_color='grey', edges_width=0.5, edges_cmap='binary',
                         edges_alpha=0.1, edges_max_width=1, edges_min_width=0.1, 
                         sort_nodes=True, sort_by=None, ascending=True,
                         fontsize=12, prev_nodes_df=None,
                         highlight_genotypes=None, is_prot=False,
                         highlight_size=200, palette='colorblind',
                         figsize=None, unit_size=2, interactive=False, 
                         alphabet_type=None, fmt='png'):
    
    if nodes_size is None:
        nodes_size = 15 if z is None else 4 if interactive else 50
    
    if interactive:
        text = nodes_df['protein'] if is_prot else nodes_df.index
        plot_interactive(nodes_df, edges_df=edges_df, fpath=fpath,
                         x=x, y=y, z=z,
                         nodes_color=nodes_color, nodes_size=nodes_size,
                         cmap=nodes_cmap, nodes_cmap_label=nodes_cmap_label,
                         edges_width=edges_width, edges_color=edges_color,
                         text=text)
    else:
        if figsize is None:
            axis_lims = get_axis_lims(nodes_df, x, y, z=z)
            axis_size = axis_lims[1] - axis_lims[0]
            figsize = (unit_size * axis_size / 0.85, unit_size * axis_size)
            
        fig, axes = init_single_fig(figsize=figsize, is_3d=z is not None)
        plot_visualization(axes, nodes_df=nodes_df, edges_df=edges_df,
                           x=x, y=y, z=z,
                           nodes_color=nodes_color, nodes_size=nodes_size,
                           nodes_cmap=nodes_cmap, nodes_alpha=nodes_alpha,
                           nodes_min_size=nodes_min_size, nodes_max_size=nodes_max_size,
                           nodes_edgecolor=nodes_edgecolor, nodes_lw=nodes_lw, 
                           nodes_cmap_label=nodes_cmap_label, nodes_vmin=nodes_vmin,
                           nodes_vmax=nodes_vmax,
                           edges_color=edges_color, edges_width=edges_width,
                           edges_cmap=edges_cmap,
                           edges_alpha=edges_alpha, edges_max_width=edges_max_width,
                           edges_min_width=edges_min_width, 
                           sort_nodes=sort_nodes, sort_by=sort_by, ascending=ascending,
                           fontsize=fontsize,
                           prev_nodes_df=prev_nodes_df)
        
        if highlight_genotypes is not None:
            highlight_genotype_groups(axes, nodes_df, highlight_genotypes,
                                      is_prot=is_prot, x=x, y=y, z=z,
                                      alphabet_type=alphabet_type,
                                      size=highlight_size, palette=palette)
        savefig(fig, fpath, fmt=fmt)


def figure_allele_grid(nodes_df, edges_df=None, fpath=None, x='1', y='2',
                       allele_color='orange', background_color='lightgrey',
                       nodes_size=None, edges_color='grey', edges_width=0.5,
                       positions=None, position_labels=None, autoscale_axis=False,
                       colsize=3, rowsize=2.7, xpos_label=0.05, ypos_label=0.92,
                       fmt='png'):
    
    config = guess_space_configuration(nodes_df.index.values)
    length, n_alleles = config['length'], np.max(config['n_alleles'])

    if position_labels is None:
        position_labels = np.arange(length) + 1

    if positions is None:
        positions = np.arange(length)
        
    fig, subplots = init_fig(n_alleles, positions.shape[0], colsize=colsize, rowsize=rowsize)
    for col, j in enumerate(positions):
        for i in range(n_alleles):
            axes = subplots[i][col]
            
            try:
                allele = config['alphabet'][col][i]
                plot_visualization(axes, nodes_df, edges_df=edges_df, x=x, y=y,
                                   nodes_color=background_color, nodes_size=nodes_size,
                                   edges_color=edges_color, edges_width=edges_width,
                                   autoscale_axis=autoscale_axis)
                sel_idxs = np.array([seq[j] == allele for seq in nodes_df.index])
                plot_nodes(axes, nodes_df.loc[sel_idxs, :], x=x, y=y, color=allele_color,
                           size=nodes_size, autoscale_axis=False)
                
                if i < n_alleles - 1:
                    axes.set_xlabel('')
                    axes.set_xticks([])
                if col > 0:
                    axes.set_ylabel('')
                    axes.set_yticks([])
                    
                xlims, ylims = axes.get_xlim(), axes.get_ylim()
                xpos = xlims[0] + xpos_label * (xlims[1] - xlims[0])
                ypos = ylims[0] + ypos_label * (ylims[1] - ylims[0])
                axes.text(xpos, ypos, '{}{}'.format(allele, position_labels[j]),
                          ha='left')
            except IndexError:
                empty_axes(axes)
    
    savefig(fig, fpath, fmt=fmt)



def figure_axis_grid(nodes_df, max_axis=None, edges_df=None, fpath=None,
                     nodes_size=None, nodes_color='function',
                     nodes_cmap='viridis', nodes_cmap_label='Function', 
                     edges_color='grey', edges_width=0.5, edges_cmap='binary',
                     edges_alpha=0.1, colsize=3, rowsize=2.7,
                     fmt='png'):
    
    if max_axis is None:
        max_axis = 1
        while str(max_axis) in nodes_df.columns:
            max_axis += 1
        max_axis -= 1
    else:
        msg = 'max_axis should be in nodes_df.columns'
        check_error(str(max_axis) in nodes_df.columns, msg=msg)
    max_axis -= 1
    
    fig, subplots = init_fig(max_axis, max_axis,
                             colsize=colsize, rowsize=rowsize)
    fig.subplots_adjust(right=0.90)
    cbar_axes = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plot_cbar = True
    
    for i in range(max_axis):
        y = str(i+2)
        for j in range(max_axis):
            x = str(j+1)
            axes = subplots[i][j]
            
            if j > i:
                empty_axes(axes)
            else:
                plot_visualization(axes, nodes_df, edges_df=edges_df,
                                   x=x, y=y, nodes_color=nodes_color,
                                   nodes_size=nodes_size, nodes_cmap=nodes_cmap,
                                   cbar=plot_cbar, cbar_axes=cbar_axes if plot_cbar else None,
                                   nodes_cmap_label=nodes_cmap_label, 
                                   edges_color=edges_color, edges_width=edges_width,
                                   edges_cmap=edges_cmap, edges_alpha=edges_alpha,  
                                   sort_nodes=True, ascending=True, sort_by=None)
                plot_cbar = False
                
            if i < max_axis - 1:
                axes.set_xlabel('')
                axes.set_xticks([])
            if j > 0:
                axes.set_ylabel('')
                axes.set_yticks([])
                    
    savefig(fig, fpath, fmt=fmt, tight=False)


def figure_Ns_grid(rw, fpath=None, fmin=None, fmax=None,
                   ncol=4, nrow=3, show_edges=True,
                   nodes_color='function', nodes_size=2.5, nodes_cmap='viridis', nodes_alpha=1,
                   nodes_min_size=1, nodes_max_size=40,
                   nodes_edgecolor='black', nodes_lw=0, 
                   nodes_cmap_label='Function', nodes_vmin=None, nodes_vmax=None,
                   edges_color='grey', edges_width=0.5, edges_cmap='binary',
                   edges_alpha=0.1, edges_max_width=1, edges_min_width=0.1, 
                   sort_nodes=True, ascending=False, sort_by=None,
                   fontsize=12, x='1', y='2'):
    f = rw.space.y
    if fmin is None:
        fmin = f.mean() + 0.05 * (f.max() - f.mean())
    if fmax is None:
        fmax = f.mean() + 0.8 * (f.max() - f.mean())

    mean_fs = np.linspace(fmin, fmax, ncol*nrow)
    
    fig, subplots = init_fig(nrow, ncol, colsize=3, rowsize=2.7)
    fig.subplots_adjust(right=0.90)
    cbar_axes = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    subplots = subplots.flatten()
    
    prev_nodes_df = None
    xmin, xmax, ymin, ymax = None, None, None, None
    for i, (mean_function, axes) in enumerate(zip(mean_fs, subplots)):
        rw.calc_visualization(mean_function=mean_function, n_components=3, eig_tol=0.01)
        is_last_plot = i == mean_fs.shape[0] - 1
        
        edges_df = None if not show_edges else rw.space.get_edges_df()
        plot_visualization(axes, rw.nodes_df, edges_df=edges_df, x=x, y=y,
                           nodes_color=nodes_color, nodes_size=nodes_size,
                           nodes_cmap=nodes_cmap, nodes_alpha=nodes_alpha,
                           cbar=is_last_plot, cbar_axes=cbar_axes if is_last_plot else None, 
                           nodes_min_size=nodes_min_size, nodes_max_size=nodes_max_size,
                           nodes_edgecolor=nodes_edgecolor, nodes_lw=nodes_lw, 
                           nodes_cmap_label=nodes_cmap_label if (i+1) % ncol == 0 else None,
                           nodes_vmin=nodes_vmin,
                           nodes_vmax=nodes_vmax, edges_color=edges_color,
                           edges_width=edges_width, edges_cmap=edges_cmap,
                           edges_alpha=edges_alpha, 
                           edges_max_width=edges_max_width, edges_min_width=edges_min_width, 
                           sort_nodes=sort_nodes, ascending=ascending, sort_by=sort_by,
                           fontsize=fontsize, prev_nodes_df=prev_nodes_df)
        prev_nodes_df = rw.nodes_df
        
        axes.set_title('Stationary F = {:.2f}'.format(mean_function))
        
        if i // ncol != nrow - 1:
            axes.set_xlabel('')
            axes.set_xticks([])
        
        if i % ncol != 0:
            axes.set_ylabel('')
            axes.set_yticks([])
            
        # Check and store global lims values
        xlims, ylims = axes.get_xlim(), axes.get_ylim()
        
        if xmin is None or xlims[0] < xmin:
            xmin = xlims[0]
        if xmax is None or xlims[1] > xmax:
            xmax = xlims[1] 
            
        if ymin is None or ylims[0] < ymin:
            ymin = ylims[0]
        if ymax is None or ylims[1] > ymax:
            ymax = ylims[1]
    
    for axes in subplots:
        axes.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
    savefig(fig, fpath, tight=False)


def figure_shifts_grid(nodes_df, seq, edges_df=None, fpath=None, x='1', y='2',
                       allele_color='orange', background_color='lightgrey',
                       nodes_size=None, edges_color='grey', edges_width=0.5,
                       positions=None, position_labels=None, autoscale_axis=True,
                       colsize=3, rowsize=2.7, xpos_label=0.05, ypos_label=0.92,
                       is_prot=False, alphabet_type='rna', codon_table='Standard',
                       ncol=None, nrow=1, labels_full_seq=False):
    
    length = len(nodes_df.index[0]) - len(seq) + 1

    if position_labels is None:
        position_labels = np.arange(length) + 1

    if positions is None:
        positions = np.arange(length)
        
    if ncol is None:
        ncol = positions.shape[0]
    
    fig, subplots = init_fig(nrow, ncol, colsize=colsize, rowsize=rowsize)
    subplots = subplots.flatten()
    
    for col, j in enumerate(positions):
        if alphabet_type in ['rna', 'dna']:
            genotype_groups = ['N' * col + seq + 'N' * (length - col - 1)]
        else:
            genotype_groups = ['X' * col + seq + 'X' * (length - col - 1)]
        axes = subplots[col]
        plot_visualization(axes, nodes_df, edges_df=edges_df, x=x, y=y,
                           nodes_color=background_color, nodes_size=nodes_size,
                           edges_color=edges_color, edges_width=edges_width,
                           autoscale_axis=autoscale_axis)
        sel_nodes_df = get_nodes_df_highlight(nodes_df,
                                              genotype_groups=genotype_groups,
                                              is_prot=is_prot,
                                              alphabet_type=alphabet_type,
                                              codon_table=codon_table)
        plot_nodes(axes, sel_nodes_df, x=x, y=y, color=allele_color,
                   size=nodes_size, autoscale_axis=False)
        
        if col // ncol != nrow - 1:
            axes.set_xlabel('')
            axes.set_xticks([])
    
        if col % ncol != 0:
            axes.set_ylabel('')
            axes.set_yticks([])
            
        xlims, ylims = axes.get_xlim(), axes.get_ylim()
        xpos = xlims[0] + xpos_label * (xlims[1] - xlims[0])
        ypos = ylims[0] + ypos_label * (ylims[1] - ylims[0])
        
        panel_label = genotype_groups[0] if labels_full_seq else '{}{}'.format(seq, position_labels[j])
        axes.text(xpos, ypos, panel_label, ha='left')
    
    savefig(fig, fpath)
    

def plot_edges_datashader(nodes_df, edges_df, x='1', y='2', cmap='grey',
                          width=0.5, alpha=0.2, color='grey',
                          shade=True, resolution=800):
    line_coords = get_lines_from_edges_df(nodes_df, edges_df, x=x, y=y, z=None)
    dsg = hv.Curve(line_coords)
    if shade:
        dsg = datashade(dsg, cmap=cmap, width=resolution, height=resolution)
    else:
        dsg = dsg.opts(color=color, linewidth=width, alpha=alpha)
    return(dsg)


def plot_nodes_datashader(nodes_df, x='1', y='2', color='function', cmap='viridis',
                          size=5, linewidth=0, edgecolor='black',
                          vmin=None, vmax=None,
                          sort_by=None, ascending=True,
                          shade=True, resolution=800):
    if sort_by is not None:
        nodes_df = nodes_df.sort_values(sort_by, ascending=ascending)
        
    if vmin is None:
        vmin = nodes_df[color].min()
    if vmax is None:
        vmax = nodes_df[color].max()
    
    if shade:
        nodes = hv.Points(nodes_df, kdims=[x, y], label='Nodes')
        if sort_by is not None:
            dsg = datashade(nodes, cmap=cmap,
                            width=resolution, height=resolution,
                            aggregator=ds.first(color))
        else:
            dsg = datashade(nodes, cmap=cmap, 
                            width=resolution, height=resolution,
                            aggregator=ds.max(color))
    else:
        hv.extension('matplotlib')
        colnames = [x, y]
        if color not in colnames: # avoid adding color in case is already a selected field
            colnames.append(color)
        scatter = hv.Scatter(nodes_df[colnames])
        dsg = scatter.opts(color=color, cmap=cmap, clim=(vmin, vmax),
                           s=size, linewidth=linewidth, 
                           edgecolor=edgecolor)
    
    return(dsg)


def plot_holoview(nodes_df, x='1', y='2', edges_df=None,
                  nodes_color='function', nodes_cmap='viridis',
                  nodes_size=5, nodes_vmin=None, nodes_vmax=None,
                  linewidth=0, edgecolor='black',
                  sort_by=None, ascending=False,
                  edges_width=0.5, edges_alpha=0.2, edges_color='grey',
                  edges_cmap='grey', background_color='white',
                  nodes_resolution=800, edges_resolution=1200,
                  shade_nodes=True, shade_edges=True):
    dsg = plot_nodes_datashader(nodes_df, x, y, nodes_color, nodes_cmap,
                                linewidth=linewidth, edgecolor=edgecolor,
                                size=nodes_size,
                                vmin=nodes_vmin, vmax=nodes_vmax,
                                sort_by=sort_by, ascending=ascending,
                                resolution=nodes_resolution,
                                shade=shade_nodes)
    
    if edges_df is not None:
        edges_dsg = plot_edges_datashader(nodes_df, edges_df, x, y,
                                          cmap=edges_cmap,
                                          width=edges_width, 
                                          alpha=edges_alpha,
                                          color=edges_color,
                                          resolution=edges_resolution,
                                          shade=shade_edges)
        dsg = edges_dsg * dsg
    
    dsg.opts(xlabel='Diffusion axis {}'.format(x),
             ylabel='Diffusion axis {}'.format(y),
             bgcolor=background_color, padding=0.1)
    
    return(dsg)


def save_holoviews(dsg, fpath, fmt='png'):
    fig = hv.render(dsg)
    savefig(fig, fpath, tight=False, fmt=fmt)


def figure_allele_grid_datashader(nodes_df, fpath, x='1', y='2', edges_df=None,
                                  positions=None, position_labels=None,
                                  edges_cmap='grey', background_color='white',
                                  nodes_resolution=800, edges_resolution=1200,
                                  fmt='png'):
    if edges_df is not None:
        edges = plot_edges_datashader(nodes_df, edges_df, x, y,
                                      cmap=edges_cmap,
                                      resolution=edges_resolution)
    else:
        edges = None
        
    config = guess_space_configuration(nodes_df.index.values)
    length, n_alleles = config['length'], np.max(config['n_alleles'])

    if position_labels is None:
        position_labels = np.arange(length) + 1

    if positions is None:
        positions = np.arange(length)

    plots = None
    
    nc = {i: np.array([seq[i] for seq in nodes_df.index])
          for i in range(length)}
    
    for i in range(n_alleles):
        for col, j in enumerate(positions):
            try:
                allele  = config['alphabet'][j][i]
                nodes_df['allele'] = (nc[col] == allele).astype(int)
            except IndexError:
                allele = ''
                nodes_df['allele'] = np.nan
                
            nodes = plot_nodes_datashader(nodes_df.copy(),
                                          x, y,
                                          color='allele', cmap='viridis',
                                          resolution=nodes_resolution,
                                          shade=True)
            nodes = nodes.relabel('{}{}'.format(j+1, allele))
            dsg = nodes if edges is None else edges * nodes
            
            dsg.opts(xlabel='Diffusion axis {}'.format(x),
                     ylabel='Diffusion axis {}'.format(y),
                     bgcolor=background_color,
                     title='{}{}'.format(position_labels[j], allele))
            
            if i < n_alleles - 1:
                dsg.opts(xlabel='')
            if col > 0:
                dsg.opts(ylabel='')
                
            if plots is None:
                plots = dsg
            else:
                plots += dsg
    dsg = plots.cols(length)
    fig = hv.render(dsg)
    savefig(fig, fpath, tight=False, fmt=fmt)


def plot_hyperparam_cv(df, axes, x='log_a', y='logL', err_bars='stderr',
                       xlabel=r'$\log_{10}(a)$',
                       ylabel='log(L)',  show_folds=True, highlight='max'):
    
    sdf = df.groupby(x, as_index=False).agg({y: ('mean', 'std', 'count')})
    sdf.columns = ['x', 'mean', 'sd', 'count']
    sdf['stderr'] = sdf['sd'] / np.sqrt(sdf['count'])
    
    idx = sdf['mean'].argmax() if highlight == 'max' else sdf['mean'].argmin()
    x_star = sdf['x'][idx]
    y_star = sdf['mean'][idx]
    
    if show_folds:
        sns.lineplot(x=x, y=y, hue='fold', ax=axes, legend=False,
                     data=df.sort_values(x), alpha=0.4, linewidth=0.5,
                     zorder=1)
    axes.plot(sdf['x'], sdf['mean'], color='black', lw=1, zorder=1)
    axes.scatter(sdf['x'], sdf['mean'], color='black', s=15)
    axes.scatter(x_star, y_star, color='red', s=25, zorder=10, alpha=1)
    
    for a, m, s in zip(sdf['x'], sdf['mean'], sdf[err_bars]):
        color = 'red' if a == x_star else 'black'
        
        if a == x_star:
            label = '{}* = {:.1f}'.format(xlabel, x_star)
        else:
            label = None
        
        axes.plot((a, a), (m-s, m+s), lw=1, color=color, label=label)
    
    xlims, ylims = axes.get_xlim(), axes.get_ylim()
    ylims = (ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.1)
    
    for r in sdf.loc[np.isinf(sdf['x']), :].to_dict(orient='index').values():
        x, y = r['x'], r['mean']
        label = r'{} = $\infty$'.format(xlabel)
        if y < 0:
            label = r'{} = -$\infty$'.format(xlabel)
        if not np.isnan(y):
            axes.plot(xlims, (y, y), lw=0.5, c='darkred', linestyle='--',
                      label=label)
    axes.legend(loc=1)
    axes.set(xlabel=xlabel, ylabel='Out of sample {}'.format(ylabel),
             ylim=ylims, xlim=xlims)


def plot_density_vs_frequency(seq_density, axes):
    with warnings.catch_warnings():
        logf = np.log10(seq_density['frequency'])
        logq = np.log10(seq_density['Q_star'])
        warnings.simplefilter("ignore")
        data = pd.DataFrame({'logR': logf, 'logQ': logq}).dropna()
                             
    axes.scatter(data['logR'], data['logQ'],
                 color='black', s=5, alpha=0.4, zorder=2)
    zero_counts_logq = logq[np.isinf(logf)]
    fake_logf = (data['logQ'].min()-1) * np.ones(zero_counts_logq.shape)
    axes.scatter(fake_logf, zero_counts_logq, marker='<',
                 color='red', s=5, alpha=0.2, zorder=2)
    
    xlims, ylims = axes.get_xlim(), axes.get_ylim()
    lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1])
    axes.plot(lims, lims, color='grey', linewidth=0.5, alpha=0.5, zorder=1)
    axes.set(xlabel=r'$log_{10}$(Frequency)', ylabel=r'$log_{10}$(Q*)', 
             xlim=lims, ylim=lims)


def plot_SeqDEFT_summary(log_Ls, seq_density=None, err_bars='stderr',
                         show_folds=True):
    '''
    Generates a 2 panel figure showing how the cross-validated likelihood
    changes with ``a`` hyperparameter and the best selected value for model
    fitting.  
    
    Parameters
    ----------
        log_Ls : pd.DataFrame of shape (num_a, 3)
            DataFrame containing the column names ``a``, ``logL`` and
            ``fold```
        
        seq_density : pd.DataFrame of shape (n_genotypes, >= 2)
            DataFrame with column names ``frequency``, ``Q_star`` with the 
            observed frequencies and estimated densities for each possible sequence
            respectively. If not provided only a 1 panel figure with the
            cross-validated likelihood curve will be provided
            
        err_bars : str
            What to show in the error bars: sd standard deviation across the 
            different folds or stderr for standard error of the mean
            
        show_folds: bool
            Whether to show the out of sample log likelihoods for the different
            folds in the cross-validation procedure separately
            
    Returns
    -------
        fig : matplotlib.figure object
            ``Figure`` object containing the resulting plots
    
    '''
    if seq_density is None:
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3.5)
    else:
        fig, subplots = init_fig(1, 2, colsize=4, rowsize=3.5)
        plot_density_vs_frequency(seq_density, subplots[1])
        axes = subplots[0]
        
    plot_hyperparam_cv(log_Ls, axes, err_bars=err_bars,
                       show_folds=show_folds)
    
    if seq_density is not None:
        fig.tight_layout()
    return(fig)
