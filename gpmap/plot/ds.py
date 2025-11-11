#!/usr/bin/env python
import numpy as np
import datashader as ds
import holoviews as hv

from holoviews.operation.datashader import datashade

from gpmap.settings import PLOTS_FORMAT
from gpmap.seq import guess_space_configuration
from gpmap.plot.utils import get_lines_from_edges_df, sort_nodes, get_vmin_max


def calc_ds_size(nodes_df, x, y, resolution, square=True):
    if square:
        xlim = nodes_df[x].min(), nodes_df[x].max()
        ylim = nodes_df[y].min(), nodes_df[y].max()
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
        w, h = resolution, resolution * dx / dy
    else:
        w, h = resolution, resolution
    return (int(w), int(h))


def plot_nodes(
    nodes_df,
    x="1",
    y="2",
    color="function",
    cmap="viridis",
    vmin=None,
    vmax=None,
    size=5,
    linewidth=0,
    edgecolor="black",
    sort_by=None,
    sort_ascending=True,
    shade=True,
    resolution=800,
    square=False,
):
    """
    Plot nodes with various customization options.

    Parameters
    ----------
    nodes_df : pandas.DataFrame
        DataFrame containing node data.
    x : str, optional
        Column name for the x-axis, by default "1".
    y : str, optional
        Column name for the y-axis, by default "2".
    color : str, optional
        Column name for the color values, by default "function".
    cmap : str, optional
        Colormap to use for coloring nodes, by default "viridis".
    vmin : float, optional
        Minimum value for color scaling, by default None.
    vmax : float, optional
        Maximum value for color scaling, by default None.
    size : int, optional
        Size of the nodes, by default 5.
    linewidth : int, optional
        Line width of the node edges, by default 0.
    edgecolor : str, optional
        Color of the node edges, by default "black".
    sort_by : str, optional
        Column name to sort nodes by, by default None.
    sort_ascending : bool, optional
        Whether to sort nodes in ascending order, by default True.
    shade : bool, optional
        Whether to use datashader for rendering, by default True.
    resolution : int, optional
        Resolution of the plot, by default 800.
    square : bool, optional
        Whether to enforce a square aspect ratio, by default False.

    Returns
    -------
    holoviews.Element
        A Holoviews element representing the plotted nodes.
    """
    ndf = sort_nodes(nodes_df, sort_by, sort_ascending, color)
    vmin, vmax = get_vmin_max(ndf, color, vmin=vmin, vmax=vmax)

    if shade:
        nodes = hv.Points(ndf, kdims=[x, y], label="Nodes")
        w, h = calc_ds_size(ndf, x, y, resolution, square=square)
        if sort_by is not None:
            dsg = datashade(
                nodes, cmap=cmap, width=w, height=h, aggregator=ds.first(color)
            )
        else:
            dsg = datashade(
                nodes, cmap=cmap, width=w, height=h, aggregator=ds.max(color)
            )
    else:
        hv.extension("matplotlib")
        colnames = [x, y]
        if (
            color not in colnames
        ):  # avoid adding color in case is already a selected field
            colnames.append(color)
        scatter = hv.Scatter(ndf[colnames])
        dsg = scatter.opts(
            color=color,
            cmap=cmap,
            clim=(vmin, vmax),
            s=size,
            linewidth=linewidth,
            edgecolor=edgecolor,
        )
    aspect = "square" if square else "equal"
    dsg.opts(aspect=aspect)
    return dsg


def plot_edges(
    nodes_df,
    edges_df,
    x="1",
    y="2",
    cmap="grey",
    width=0.5,
    alpha=0.2,
    color="grey",
    shade=True,
    resolution=800,
    square=True,
):
    """
    Plot edges.

    Parameters
    ----------
    nodes_df : pandas.DataFrame
        DataFrame containing node data.
    edges_df : pandas.DataFrame
        DataFrame containing edge data.
    x : str, optional
        Column name for the x-axis, by default "1".
    y : str, optional
        Column name for the y-axis, by default "2".
    cmap : str, optional
        Colormap to use for coloring edges, by default "grey".
    width : float, optional
        Line width of the edges, by default 0.5.
    alpha : float, optional
        Transparency level of the edges, by default 0.2.
    color : str, optional
        Color of the edges, by default "grey".
    shade : bool, optional
        Whether to use datashader for rendering, by default True.
    resolution : int, optional
        Resolution of the plot, by default 800.
    square : bool, optional
        Whether to enforce a square aspect ratio, by default True.

    Returns
    -------
    holoviews.Element
        A Holoviews element representing the plotted edges.
    """
    line_coords = get_lines_from_edges_df(nodes_df, edges_df, x=x, y=y, z=None)
    dsg = hv.Curve(line_coords)
    if shade:
        w, h = calc_ds_size(nodes_df, x, y, resolution, square=square)
        dsg = datashade(dsg, cmap=cmap, width=w, height=h)
    else:
        dsg = dsg.opts(color=color, linewidth=width, alpha=alpha)
    aspect = "square" if square else "equal"
    dsg.opts(aspect=aspect)
    return dsg


def plot_visualization(
    nodes_df,
    x="1",
    y="2",
    edges_df=None,
    nodes_color="function",
    nodes_cmap="viridis",
    nodes_size=5,
    nodes_vmin=None,
    nodes_vmax=None,
    linewidth=0,
    edgecolor="black",
    sort_by=None,
    sort_ascending=False,
    edges_width=0.5,
    edges_alpha=1,
    edges_color="grey",
    edges_cmap="grey",
    background_color="white",
    nodes_resolution=800,
    edges_resolution=1200,
    shade_nodes=True,
    shade_edges=True,
    square=True,
):
    """
    Plots the nodes representing the states of the discrete space on the
    provided coordinates and the edges representing the connections between
    states if provided.

    Parameters
    ----------
    nodes_df : pandas.DataFrame
        DataFrame containing node data.
    x : str, optional
        Column name for the x-axis, by default "1".
    y : str, optional
        Column name for the y-axis, by default "2".
    edges_df : pandas.DataFrame, optional
        DataFrame containing edge data, by default None.
    nodes_color : str, optional
        Column name for the color values, by default "function".
    nodes_cmap : str, optional
        Colormap to use for coloring nodes, by default "viridis".
    nodes_size : int, optional
        Size of the nodes, by default 5.
    nodes_vmin : float, optional
        Minimum value for color scaling, by default None.
    nodes_vmax : float, optional
        Maximum value for color scaling, by default None.
    linewidth : int, optional
        Line width of the node edges, by default 0.
    edgecolor : str, optional
        Color of the node edges, by default "black".
    sort_by : str, optional
        Column name to sort nodes by, by default None.
    sort_ascending : bool, optional
        Whether to sort nodes in ascending order, by default False.
    edges_width : float, optional
        Line width of the edges, by default 0.5.
    edges_alpha : float, optional
        Transparency level of the edges, by default 1.
    edges_color : str, optional
        Color of the edges, by default "grey".
    edges_cmap : str, optional
        Colormap to use for coloring edges, by default "grey".
    background_color : str, optional
        Background color of the plot, by default "white".
    nodes_resolution : int, optional
        Resolution of the nodes plot, by default 800.
    edges_resolution : int, optional
        Resolution of the edges plot, by default 1200.
    shade_nodes : bool, optional
        Whether to use datashader for rendering nodes, by default True.
    shade_edges : bool, optional
        Whether to use datashader for rendering edges, by default True.
    square : bool, optional
        Whether to enforce a square aspect ratio, by default True.

    Returns
    -------
    holoviews.Element
        A Holoviews element representing the plotted visualization.
    """
    dsg = plot_nodes(
        nodes_df,
        x,
        y,
        nodes_color,
        nodes_cmap,
        linewidth=linewidth,
        edgecolor=edgecolor,
        size=nodes_size,
        vmin=nodes_vmin,
        vmax=nodes_vmax,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        resolution=nodes_resolution,
        shade=shade_nodes,
        square=square,
    )

    if edges_df is not None:
        edges_dsg = plot_edges(
            nodes_df,
            edges_df,
            x,
            y,
            cmap=edges_cmap,
            width=edges_width,
            alpha=edges_alpha,
            color=edges_color,
            resolution=edges_resolution,
            shade=shade_edges,
            square=square,
        )
        dsg = edges_dsg * dsg

    dsg.opts(
        xlabel="Diffusion axis {}".format(x),
        ylabel="Diffusion axis {}".format(y),
        bgcolor=background_color,
        padding=0.1,
    )

    return dsg


def dsg_to_fig(dsg):
    """
    Convert a Holoviews element to a Matplotlib figure.

    Parameters
    ----------
    dsg : holoviews.Element
        A Holoviews element to be converted.

    Returns
    -------
    matplotlib.figure.Figure
        A Matplotlib figure object representing the Holoviews element.
    """
    return hv.render(dsg)


def savefig(
    dsg, fpath=None, tight=True, fmt=PLOTS_FORMAT, dpi=360, figsize=None
):
    fig = dsg_to_fig(dsg)
    if tight:
        fig.tight_layout()

    if figsize is not None:
        fig.set_size_inches(figsize)

    if fpath is not None:
        fpath = "{}.{}".format(fpath, fmt)
        fig.savefig(fpath, format=fmt, dpi=dpi)


def _get_allele_panel(
    nodes_df,
    x,
    y,
    edges_dsg,
    seq_pos,
    allele,
    pos_label,
    nodes_resolution,
    square,
    background_color,
    sort_by="function",
    sort_ascending=True,
):
    if seq_pos is None:
        nodes_df["allele"] = np.nan
    else:
        nodes_df["allele"] = (seq_pos == allele).astype(int)

    nodes = plot_nodes(
        nodes_df.copy(),
        x,
        y,
        color="allele",
        cmap="viridis",
        resolution=nodes_resolution,
        shade=True,
        square=square,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
    )

    dsg = nodes if edges_dsg is None else edges_dsg * nodes
    dsg.opts(
        xlabel="Diffusion axis {}".format(x),
        ylabel="Diffusion axis {}".format(y),
        bgcolor=background_color,
        padding=0.1,
        title="{}{}".format(pos_label, allele),
    )
    return dsg


def _get_edges_dsg(nodes_df, edges_df, x, y, cmap, resolution, square):
    if edges_df is not None:
        edges_dsg = plot_edges(
            nodes_df,
            edges_df,
            x,
            y,
            cmap=cmap,
            resolution=resolution,
            square=square,
        )
    else:
        edges_dsg = None
    return edges_dsg


def figure_allele_grid(
    nodes_df,
    fpath,
    x="1",
    y="2",
    edges_df=None,
    positions=None,
    position_labels=None,
    edges_cmap="grey",
    background_color="white",
    nodes_resolution=800,
    edges_resolution=1200,
    sort_by=None,
    sort_ascending=False,
    fmt="png",
    figsize=None,
    square=True,
    **kwargs,
):
    """
    Generate a grid of allele visualizations and save the resulting figure.

    Parameters
    ----------
    nodes_df : pandas.DataFrame
        DataFrame containing node data.
    fpath : str
        File path to save the resulting figure.
    x : str, optional
        Column name for the x-axis, by default "1".
    y : str, optional
        Column name for the y-axis, by default "2".
    edges_df : pandas.DataFrame, optional
        DataFrame containing edge data, by default None.
    positions : list or numpy.ndarray, optional
        List or array of positions to visualize, by default None.
    position_labels : list or numpy.ndarray, optional
        Labels for the positions, by default None.
    edges_cmap : str, optional
        Colormap to use for coloring edges, by default "grey".
    background_color : str, optional
        Background color of the plot, by default "white".
    nodes_resolution : int, optional
        Resolution of the nodes plot, by default 800.
    edges_resolution : int, optional
        Resolution of the edges plot, by default 1200.
    sort_by : str, optional
        Column name to sort nodes by, by default None.
    sort_ascending : bool, optional
        Whether to sort nodes in ascending order, by default False.
    fmt : str, optional
        Format to save the figure, by default "png".
    figsize : tuple, optional
        Size of the figure in inches, by default None.
    square : bool, optional
        Whether to enforce a square aspect ratio, by default True.
    """
    edges_dsg = _get_edges_dsg(
        nodes_df, edges_df, x, y, edges_cmap, edges_resolution, square
    )
    config = guess_space_configuration(nodes_df.index.values)
    alphabet = config["alphabet"]
    length, n_alleles = config["length"], np.max(config["n_alleles"])

    if position_labels is None:
        position_labels = np.arange(length) + 1

    if positions is None:
        positions = np.arange(length)
    else:
        length = len(positions)

    nc = {
        label: np.array([seq[i] for seq in nodes_df.index])
        for i, label in zip(positions, position_labels)
    }
    plots = None
    for j, (pos, pos_label) in enumerate(zip(positions, position_labels)):
        alleles = alphabet[pos]
        for i, allele in enumerate(alleles):
            seq_pos = nc[pos_label]
            dsg = _get_allele_panel(
                nodes_df,
                x,
                y,
                edges_dsg,
                seq_pos,
                allele,
                pos_label,
                nodes_resolution,
                square,
                background_color,
                sort_by=sort_by,
                sort_ascending=sort_ascending,
            )

            if i < n_alleles - 1:
                dsg.opts(xlabel="")
            if j > 0:
                dsg.opts(ylabel="")
            plots = dsg if plots is None else plots + dsg

    dsg = hv.Layout(plots).cols(length)
    savefig(dsg, fpath, tight=False, fmt=fmt, figsize=figsize)
