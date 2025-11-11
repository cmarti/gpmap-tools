#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from itertools import product, chain, cycle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    from skimage.draw import line, line_aa
except ImportError:
    pass

from gpmap.settings import PLOTS_FORMAT
from gpmap.utils import check_error
from gpmap.seq import guess_space_configuration
from gpmap.genotypes import (
    get_edges_coords,
    get_nodes_df_highlight,
    minimize_nodes_distance,
)
from gpmap.plot.utils import sort_nodes


def init_fig(
    nrow=1,
    ncol=1,
    figsize=None,
    colsize=3,
    rowsize=3,
    sharex=False,
    sharey=False,
    hspace=None,
    wspace=None,
):
    if figsize is None:
        figsize = (colsize * ncol, rowsize * nrow)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw={"wspace": wspace, "hspace": hspace},
    )
    return (fig, axes)


def get_hist_inset_axes(axes, pos=(0.2, 0.7), width=0.4, height=0.25):
    ax = axes.inset_axes((pos[0], pos[1], width, height))
    ax.patch.set_alpha(0)
    return ax


def get_cbar_inset_axes(
    axes, horizontal=False, pos=(0.2, 0.6), width=0.1, height=0.4
):
    if horizontal:
        width, height = height, width
    ax = axes.inset_axes((pos[0], pos[1], width, height))
    return ax


def set_centered_spines(
    axes,
    xlabel="",
    ylabel="",
    xlabel_pos=(1.1, 0.1),
    ylabel_pos=(0.1, 0.94),
    add_grid=True,
    zorder=3,
    alpha=0.5,
    fontsize=None,
):
    
    axes.spines["left"].set(position=("data", 0), zorder=zorder, alpha=alpha)
    axes.spines["bottom"].set(position=("data", 0), zorder=zorder, alpha=alpha)
    axes.tick_params(axis="both", color=(0, 0, 0, alpha))
    axes.set(xlabel="", ylabel="")

    for t in chain(axes.get_xticklabels(), axes.get_yticklabels()):
        t.set_alpha(alpha)

    if add_grid:
        axes.grid(alpha=0.2)

    axes.plot(
        (1),
        (0),
        ls="",
        marker=">",
        ms=5,
        color="k",
        alpha=alpha,
        transform=axes.get_yaxis_transform(),
        clip_on=False,
    )
    axes.plot(
        (0),
        (1),
        ls="",
        marker="^",
        ms=5,
        color="k",
        alpha=alpha,
        transform=axes.get_xaxis_transform(),
        clip_on=False,
    )

    axes.annotate(
        xlabel,
        xy=xlabel_pos,
        xycoords=("axes fraction", "data"),
        fontsize=fontsize,
        ha="right",
        va="center",
    )
    axes.annotate(
        ylabel,
        xy=ylabel_pos,
        xycoords=("data", "axes fraction"),
        fontsize=fontsize,
        ha="left",
        va="bottom",
    )
    axes.spines[["right", "top"]].set_visible(False)


def savefig(
    fig, fpath=None, tight=True, fmt=PLOTS_FORMAT, dpi=360, figsize=None
):
    if tight:
        fig.tight_layout()

    if figsize is not None:
        fig.set_size_inches(*figsize)

    if fpath is not None:
        fpath = "{}.{}".format(fpath, fmt)
        fig.savefig(fpath, format=fmt, dpi=dpi)
        plt.close()
    else:
        plt.show()


def empty_axes(axes):
    axes.spines[["left", "bottom", "right", "top"]].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([])


def create_patches_legend(axes, colors_dict, loc=1, **kwargs):
    axes.legend(
        handles=[
            mpatches.Patch(color=color, label=label)
            for label, color in colors_dict.items()
        ],
        loc=loc,
        **kwargs,
    )


def plot_relaxation_times(
    decay_df,
    axes=None,
    fpath=None,
    log_scale=False,
    neutral_time=None,
    kwargs={},
):
    """
    Plot relaxation times for calculated components.

    Parameters
    ----------
    decay_df : pd.DataFrame
        DataFrame with shape (n_components, 3) containing decay rates and
        associated mean relaxation times for each calculated component.

    axes : matplotlib.axes.Axes, optional
        Axes object to plot on. If not provided, a new figure will be created
        and saved to the path specified by `fpath`.

    fpath : str, optional
        File path to save the plot. If `None`, the `axes` argument must be
        provided for plotting.

    log_scale : bool, default=False
        Whether to plot relaxation times on a logarithmic scale.

    neutral_time : float, optional
        If provided, a horizontal line representing the neutral process
        relaxation time will be added to the plot. Useful for selecting
        relevant dimensions.

    kwargs : dict, optional
        Additional keyword arguments for `axes.plot` and `axes.scatter`,
        such as color or marker style.

    """

    if axes is None and fpath is None:
        msg = "Either axes or fpath argument must be provided"
        raise ValueError(msg)

    fig = None
    if axes is None:
        fig, axes = init_fig(1, 1, colsize=4, rowsize=3)

    axes.plot(
        decay_df["k"],
        decay_df["relaxation_time"],
        linewidth=1,
        **kwargs,
        label="Selection",
    )
    axes.scatter(decay_df["k"], decay_df["relaxation_time"], s=15, **kwargs)
    xlims = axes.get_xlim()
    if neutral_time is not None:
        axes.plot(
            xlims,
            (neutral_time, neutral_time),
            lw=0.5,
            c="orange",
            linestyle="--",
            label="Neutral",
        )
        axes.legend(loc=1)

    if log_scale:
        axes.set(yscale="log")

    axes.set(
        xlabel=r"Eigenvalue order $k$",
        ylabel=r"Relaxation time $\frac{-1}{\lambda_{k}}$",
        xticks=decay_df["k"],
        xlim=xlims,
    )

    if fig is not None:
        savefig(fig, fpath)


def plot_edges(
    axes,
    nodes_df,
    edges_df,
    x="1",
    y="2",
    z=None,
    alpha=0.1,
    zorder=1,
    color="grey",
    cbar=True,
    cmap="binary",
    cbar_axes=None,
    cbar_orientation="vertical",
    cbar_label="",
    palette=None,
    legend=True,
    legend_loc=0,
    width=0.5,
    max_width=1,
    min_width=0.1,
    fontsize=None,
    rasterized=False,
):
    """
    Plots the edges representing connections between states in the discrete
    space under a particular embedding.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The matplotlib Axes object in which to plot the edges.

    nodes_df : pandas.DataFrame
        DataFrame of shape (n_genotypes, n_components + 2) containing the
        coordinates in each of the `n_components`, along with additional
        columns such as "function" and "stationary_freq". Additional columns
        are also allowed.

    edges_df : pandas.DataFrame
        DataFrame of shape (n_edges, 2) containing the connectivity
        information between states of the discrete space to plot. It has
        columns "i" and "j" for the indices of the pairs of states that are
        connected.

    x : str, default='1'
        Column in `nodes_df` to use for plotting the genotypes on the x-axis.

    y : str, default='2'
        Column in `nodes_df` to use for plotting the genotypes on the y-axis.

    z : str, optional
        Column in `nodes_df` to use for plotting the genotypes on the z-axis.
        If provided, a 3D plot will be produced, provided the `axes` object
        supports it.

    alpha : float, default=0.1
        Transparency of the lines representing the edges.

    zorder : int, default=1
        Order in which the edges will be rendered relative to other elements.
        Typically, this should be smaller than the `zorder` used for plotting
        nodes.

    color : str, default='grey'
        Column name in `edges_df` for the values used to color the edges, or a
        specific color to use for all edges.

    cmap : str or matplotlib.colors.Colormap, default='binary'
        Colormap to use for coloring the edges based on the `color` column.

    width : float or str
        Width of the lines representing the edges. If a float is provided, it
        will be used for all edges. If a string is provided, edge widths will
        be scaled based on the corresponding column in `edges_df`.

    max_width : float, default=1
        Maximum width for the edges when scaled.

    min_width : float, default=0.1
        Minimum width for the edges when scaled.

    rasterized : bool, optional, default=False
        Whether to rasterize the plot for better performance with large datasets.

    Returns
    -------
    line_collection : matplotlib.collections.LineCollection or
    mpl_toolkits.mplot3d.art3d.Line3DCollection
        The collection of lines representing the edges.
    """
    # TODO: get colors and width as either fixed values or from edges_df
    edges_coords = get_edges_coords(
        nodes_df, edges_df, x=x, y=y, z=z, avoid_dups=True
    )
    c, cbar, palette, legend, vmin, vmax, _ = get_element_color(
        edges_df, color, palette, cbar, legend
    )
    widths = get_element_sizes(edges_df, width, min_width, max_width)
    get_lines = LineCollection if z is None else Line3DCollection
    lines = get_lines(
        edges_coords,
        colors=c,
        linewidths=widths,
        alpha=alpha,
        zorder=zorder,
        cmap=mpl.colormaps[cmap],
        rasterized=rasterized,
    )
    axes.add_collection(lines)
    add_color_info(
        lines,
        axes,
        cbar,
        cbar_axes,
        cbar_label,
        legend,
        palette,
        legend_loc,
        fontsize,
        cbar_orientation,
        vmin,
        vmax,
    )
    return lines


def plot_nodes(
    axes,
    nodes_df,
    x="1",
    y="2",
    z=None,
    alpha=1,
    zorder=2,
    sort_by=None,
    sort_ascending=False,
    color="function",
    cmap="viridis",
    cbar=True,
    cbar_axes=None,
    cbar_label="Function",
    cbar_orientation="vertical",
    vcenter=None,
    vmax=None,
    vmin=None,
    palette="Set1",
    size=2.5,
    max_size=40,
    min_size=1,
    lw=0,
    edgecolor="black",
    legend=True,
    legend_loc=0,
    rasterized=False,
):
    """
    Plots the nodes representing the states of the discrete space on the
    provided coordinates.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The matplotlib Axes object in which to plot the nodes or states.

    nodes_df : pandas.DataFrame
        DataFrame of shape (n_genotypes, n_components + 2) containing the coordinates
        in each of the `n_components`, along with additional columns such as
        "function" and "stationary_freq". Additional columns are also allowed.

    x : str, default='1'
        Column in `nodes_df` to use for plotting the genotypes on the x-axis.

    y : str, default='2'
        Column in `nodes_df` to use for plotting the genotypes on the y-axis.

    z : str, optional
        Column in `nodes_df` to use for plotting the genotypes on the z-axis.
        If provided, a 3D plot will be produced, provided the `axes` object supports it.

    alpha : float, default=1
        Transparency of markers representing the nodes.

    zorder : int, default=2
        Order in which the nodes will be rendered relative to other elements.
        Typically, this should be greater than the `zorder` used for plotting edges.

    color : str, default='grey'
        Column name in `nodes_df` for the values used to color the nodes, or a specific
        color to use for all nodes.

    vcenter : bool, default=False
        Whether to center the color scale around the 0 value.

    vmax : float, optional
        Maximum value for the colormap.

    vmin : float, optional
        Minimum value for the colormap.

    cmap : str or matplotlib.colors.Colormap, default='viridis'
        Colormap to use for coloring the nodes based on the `color` column.

    cbar : bool, default=True
        Whether to display the colorbar.

    cbar_label : str, optional
        Label for the colorbar associated with the nodes' color scale.

    cbar_axes : matplotlib.axes.Axes, optional
        Axes to plot the colorbar. If not provided, it will be automatically
        adjusted to the current Axes.

    palette : dict, optional
        Dictionary mapping categories in the `color` column to specific colors,
        if the column represents categorical data.

    size : float or str, default=2.5
        Size of the markers for the nodes. If a float is provided, it will be used
        for all nodes. If a string is provided, node sizes will be scaled based
        on the corresponding column in `nodes_df`.

    max_size : float, default=40
        Maximum size for the nodes when scaled.

    min_size : float, default=1
        Minimum size for the nodes when scaled.

    lw : float, default=0
        Line width of the edges around the markers representing the nodes.

    edgecolor : str, default='black'
        Color of the edges around the markers representing the nodes.

    legend : bool, default=True
        Whether to display a legend on the plot.

    legend_loc : int or tuple, default=0
        Location of the legend if coloring is based on a categorical variable.

    rasterized : bool, default=False
        Whether to rasterize the scatterplot when rendering the plot in vector format.
    """

    ndf = sort_nodes(nodes_df, sort_by, sort_ascending, color)
    s = get_element_sizes(ndf, size, min_size, max_size)
    c, cbar, palette, legend, vmin, vmax, continuous = get_element_color(
        ndf, color, palette, cbar, legend, vmin, vmax
    )
    kwargs = {
        "linewidth": lw,
        "s": s,
        "zorder": zorder,
        "alpha": alpha,
        "edgecolor": edgecolor,
        "rasterized": rasterized,
    }
    if continuous:
        kwargs["cmap"] = mpl.colormaps[cmap]

    if vcenter:
        kwargs["norm"] = colors.CenteredNorm()
    else:
        kwargs["vmax"] = vmax
        kwargs["vmin"] = vmin

    if z is not None:
        kwargs["zs"] = ndf[z]
    sc = axes.scatter(ndf[x], ndf[y], c=c, **kwargs)
    add_color_info(
        sc,
        axes,
        cbar,
        cbar_axes,
        cbar_label,
        legend,
        palette,
        legend_loc,
        cbar_orientation,
        vmin,
        vmax,
    )


def plot_color_hist(axes, values, cmap="viridis", bins=50, fontsize=None):
    """
    Plot a histogram with bars colored according to a colormap.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The matplotlib axes object where the histogram will be plotted.
    values : array-like
        The data values to be plotted in the histogram.
    cmap : str, optional
        The name of the colormap to use for coloring the bars. Defaults to "viridis".
    bins : int, optional
        The number of bins to use for the histogram. Defaults to 50.
    fontsize : int or None, optional
        Font size for the y-axis label. If None, the default font size is used.

    """
    _, bins, bars = axes.hist(values, bins=bins)
    values = 0.5 * (bins[1:] + bins[:-1])
    values = (values - bins[0]) / (bins[-1] - bins[0])

    cmap = mpl.colormaps[cmap]
    for value, bar in zip(values, bars):
        color = cmap(value)
        if hasattr(bar, "patches"):
            for patch in bar.patches:
                patch.set_facecolor(color)
        else:
            bar.set_facecolor(color)

    axes.spines[["right", "top"]].set_visible(False)
    axes.set(yticks=[], xticks=[], xlim=(bins[0], bins[-1]))
    axes.set_ylabel(ylabel="Frequency", fontsize=fontsize)


def add_axis_labels(axes, x, y, z=None, fontsize=None, center_spines=False):
    xlabel, ylabel = (
        "Diffusion axis {}".format(x),
        "Diffusion axis {}".format(y),
    )

    if center_spines:
        msg = "Centered spines option is incompatible with z axis for 3D plot"
        check_error(z is None, msg=msg)
        set_centered_spines(axes, xlabel, ylabel, add_grid=True)
        axes.set(aspect="equal")

    else:
        axes.set_xlabel(xlabel, fontsize=fontsize)
        axes.set_ylabel(ylabel, fontsize=fontsize)

        if hasattr(axes, "set_zlabel"):
            axes.set_zlabel("Diffusion axis {}".format(z), fontsize=fontsize)


def add_color_info(
    sc,
    axes,
    cbar,
    cbar_axes,
    cbar_label,
    legend,
    palette,
    legend_loc,
    orientation,
    vmin,
    vmax,
    fontsize=None,
):
    if cbar:
        add_cbar(
            sc,
            axes,
            cbar_axes=cbar_axes,
            label=cbar_label,
            fontsize=fontsize,
            fraction=0.1,
            pad=0.02,
            orientation=orientation,
            vmin=vmin,
            vmax=vmax,
        )
    if legend:
        create_patches_legend(axes, palette, loc=legend_loc, fontsize=fontsize)


def get_axis_lims(nodes_df, x, y, z=None):
    axis_max = max(nodes_df[x].max(), nodes_df[y].max())
    axis_min = min(nodes_df[x].min(), nodes_df[y].min())
    if z is not None:
        axis_max = max(axis_max, nodes_df[z].max())
        axis_min = min(axis_min, nodes_df[z].min())

    axis_range = axis_max - axis_min
    axis_lims = (axis_min - 0.05 * axis_range, axis_max + 0.05 * axis_range)
    return axis_lims


def get_element_sizes(df, size, min_size, max_size):
    if size in df.columns and not isinstance(size, int):
        s = np.power(df[size], 2)
        size = min_size + s * (max_size - min_size) / (s.max() - s.min())
    return size


def color_palette(name, n):
    cmap = mpl.colormaps[name]
    mpl_qual_pals = {
        "tab10": 10,
        "tab20": 20,
        "tab20b": 20,
        "tab20c": 20,
        "Set1": 9,
        "Set2": 8,
        "Set3": 12,
        "Accent": 8,
        "Paired": 12,
        "Pastel1": 9,
        "Pastel2": 8,
        "Dark2": 8,
    }
    if name in mpl_qual_pals:
        bins = np.linspace(0, 1, mpl_qual_pals[name])[:n]
    else:
        bins = np.linspace(0, 1, int(n) + 2)[1:-1]
    palette = list(map(tuple, cmap(bins)[:, :3]))
    return palette


def get_element_color(df, color, palette, cbar, legend, vmin=None, vmax=None):
    continuous = False
    if color in df.columns:
        # Categorical color map
        if df[color].dtype in (object, bool, str):
            if isinstance(palette, str):
                labels = np.unique(df[color])
                n_colors = labels.shape[0]
                c = color_palette(palette, n_colors)
                palette = dict(zip(labels, c))
            elif not isinstance(palette, dict):
                raise ValueError("palette must be a str or dict")
            color = np.array([palette[label] for label in df[color]])
            legend, cbar = legend, False

        # Continuous color map
        elif df[color].dtype in (float, int):
            color = df[color]
            legend, cbar = False, cbar
            if vmin is None:
                vmin = color.min()
            if vmax is None:
                vmax = color.max()
            continuous = True

        else:
            msg = "color dtype is not compatible: {}".format(df[color].dtype)
            raise ValueError(msg)
    else:
        cbar, legend = False, False

    return (color, cbar, palette, legend, vmin, vmax, continuous)


def add_cbar(
    sc,
    axes,
    cbar_axes=None,
    label="Function",
    fontsize=None,
    fraction=0.1,
    shrink=0.7,
    pad=0.02,
    orientation="vertical",
    nticks=5,
    vmin=None,
    vmax=None,
):
    ax, cax = (axes, None) if cbar_axes is None else (None, cbar_axes)
    cbar = plt.colorbar(
        sc,
        ax=ax,
        cax=cax,
        fraction=fraction,
        pad=pad,
        orientation=orientation,
        shrink=shrink,
    )
    cbar.set_label(label=label, fontsize=fontsize)

    if vmin is not None and vmax is not None:
        ticks = np.linspace(vmin, vmax, nticks)
        cbar.set_ticks(ticks)
        ticklabels = ["{:.1f}".format(x) for x in ticks]
        cbar.set_ticklabels(ticklabels, fontsize=fontsize)


def draw_cbar(
    axes,
    cmap,
    label,
    vmin=0,
    vmax=1,
    fontsize=None,
    orientation="vertical",
    nticks=5,
    width=12,
):
    values = np.linspace(0, 1, 256)
    values = np.vstack([values] * width)

    ticks = np.linspace(0, 256, nticks)
    ticklabels = ["{:.1f}".format(x) for x in np.linspace(vmin, vmax, nticks)]

    if orientation == "vertical":
        values = values.T
        xlabel, ylabel = "", label
        xticks, yticks = [], ticks
        set_ticklabels = axes.set_yticklabels
        xlims, ylims = (None, None), (0, 256)
    elif orientation == "horizontal":
        xlabel, ylabel = label, ""
        xticks, yticks = ticks, []
        set_ticklabels = axes.set_xticklabels
        xlims, ylims = (0, 256), (None, None)
    else:
        raise ValueError("orientation not allowed: {}".format(orientation))

    axes.imshow(values, cmap=cmap)

    axes.set_ylabel(ylabel, fontsize=fontsize)
    axes.set_xlabel(xlabel, fontsize=fontsize)
    axes.set(xticks=xticks, yticks=yticks, xlim=xlims, ylim=ylims)
    set_ticklabels(ticklabels, fontsize=fontsize)


def add_cbar_hist_inset(
    axes,
    values,
    cmap="viridis",
    label="Function",
    fontsize=None,
    pos=(0.6, 0.5),
    width=0.4,
    height=0.2,
    bins=50,
):
    vmin, vmax = values.min(), values.max()
    ax1 = get_cbar_inset_axes(
        axes, pos=(pos[0], pos[1]), horizontal=True, height=width
    )
    draw_cbar(
        ax1,
        cmap=cmap,
        label=label,
        vmin=vmin,
        vmax=vmax,
        width=16,
        orientation="horizontal",
        fontsize=fontsize,
    )

    # TODO: fix problems when setting different heights
    ax2 = get_hist_inset_axes(
        axes, pos=(pos[0], pos[1] + height - 0.12), width=width, height=height
    )
    plot_color_hist(ax2, values, cmap=cmap, bins=bins, fontsize=fontsize)


def plot_genotypes_box(
    axes,
    xlims,
    ylims,
    lw=1,
    c="black",
    facecolor="none",
    title=None,
    title_pos="top",
    fontsize=None,
):
    dx, dy = xlims[1] - xlims[0], ylims[1] - ylims[0]
    rect = mpatches.Rectangle(
        (xlims[0], ylims[0]),
        dx,
        dy,
        linewidth=lw,
        edgecolor=c,
        facecolor=facecolor,
    )
    axes.add_patch(rect)

    if title is not None:
        if title_pos == "top":
            axes.text(
                xlims[0] + dx / 2,
                ylims[0] + 1.1 * dy,
                title,
                va="bottom",
                ha="center",
                fontsize=fontsize,
            )
        elif title_pos == "right":
            axes.text(
                xlims[0] + 1.1 * dx,
                ylims[0] + dy / 2,
                title,
                va="center",
                ha="left",
                fontsize=fontsize,
            )
        elif title_pos == "left":
            axes.text(
                xlims[0] - 0.1 * dx,
                ylims[0] + dy / 2,
                title,
                va="center",
                ha="right",
                fontsize=fontsize,
            )
        elif title_pos == "bottom":
            axes.text(
                xlims[0] + 0.5 * dx,
                ylims[0] - 0.1 * dy,
                title,
                va="top",
                ha="center",
                fontsize=fontsize,
            )
        else:
            msg = "Incorrect position for title: "
            raise ValueError(msg + 'try {"top", "bottom", "left", "right"}')


def plot_visualization(
    axes,
    nodes_df,
    edges_df=None,
    x="1",
    y="2",
    z=None,
    nodes_alpha=1,
    nodes_zorder=2,
    nodes_color="function",
    nodes_cmap="viridis",
    nodes_palette=None,
    nodes_vmin=None,
    nodes_vmax=None,
    nodes_vcenter=False,
    nodes_cbar=True,
    nodes_cbar_axes=None,
    nodes_cmap_label="Function",
    nodes_size=2.5,
    nodes_min_size=1,
    nodes_max_size=40,
    nodes_lw=0,
    nodes_edgecolor="black",
    edges_alpha=0.1,
    edges_zorder=1,
    edges_color="grey",
    edges_cmap="binary",
    edges_palete=None,
    edges_cbar=False,
    edges_cbar_axes=None,
    edges_width=0.5,
    edges_max_width=1,
    edges_min_width=0.1,
    sort_by=None,
    sort_ascending=True,
    center_spines=False,
    add_hist=False,
    inset_cbar=False,
    inset_pos=(0.7, 0.7),
    prev_nodes_df=None,
    rasterized=False,
):
    """
    Plots the nodes representing the states of the discrete space on the
    provided coordinates and the edges representing the connections between
    states if provided.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Matplotlib `Axes` object in which to plot the edges and nodes.

    nodes_df : pd.DataFrame of shape (n_genotypes, n_variables)
        DataFrame containing the coordinates in each of the `n_components`,
        in addition to the "function" and "stationary_freq" columns. Additional
        columns are also allowed.

    edges_df : pd.DataFrame of shape (n_edges, 2), optional
        DataFrame containing the connectivity information between states of the
        discrete space to plot. It has columns "i" and "j" for the indexes
        of the pairs of states that are connected. If not provided, only nodes
        will be plotted.

    x : str, optional, default='1'
        Column in `nodes_df` to use for plotting the genotypes on the x-axis.

    y : str, optional, default='2'
        Column in `nodes_df` to use for plotting the genotypes on the y-axis.

    z : str, optional, default=None
        Column in `nodes_df` to use for plotting the genotypes on the z-axis.
        If provided, a 3D plot will be produced.

    nodes_alpha : float, optional, default=1
        Transparency of the markers representing the nodes.

    nodes_zorder : int, optional, default=2
        Order in which the nodes will be rendered relative to other elements.

    nodes_color : str, optional, default='function'
        Column name in `nodes_df` for the values used to color the nodes, or a
        specific color to use for all nodes.

    nodes_cmap : str, optional, default='viridis'
        Colormap to use for coloring the nodes based on the `nodes_color` column.

    nodes_palette : dict, optional, default=None
        Dictionary mapping categories in the `nodes_color` column to specific colors,
        if the column represents categorical data.

    nodes_vmin : float, optional, default=None
        Minimum value for the colormap.

    nodes_vmax : float, optional, default=None
        Maximum value for the colormap.

    nodes_vcenter : bool, optional, default=False
        Whether to center the color scale around the 0 value.

    nodes_cbar : bool, optional, default=True
        Whether to display the colorbar for the nodes.

    nodes_cbar_axes : matplotlib.axes.Axes, optional, default=None
        Axes to plot the colorbar. If not provided, it will be automatically
        adjusted to the current Axes.

    nodes_cmap_label : str, optional, default='Function'
        Label for the colorbar associated with the nodes' color scale.

    nodes_size : float, optional, default=2.5
        Size of the markers for the nodes.

    nodes_min_size : float, optional, default=1
        Minimum size for the nodes when scaled.

    nodes_max_size : float, optional, default=40
        Maximum size for the nodes when scaled.

    nodes_lw : float, optional, default=0
        Line width of the edges around the markers representing the nodes.

    nodes_edgecolor : str, optional, default='black'
        Color of the edges around the markers representing the nodes.

    edges_alpha : float, optional, default=0.1
        Transparency of the lines representing the edges.

    edges_zorder : int, optional, default=1
        Order in which the edges will be rendered relative to other elements.

    edges_color : str, optional, default='grey'
        Column name in `edges_df` for the values used to color the edges, or a
        specific color to use for all edges.

    edges_cmap : str, optional, default='binary'
        Colormap to use for coloring the edges based on the `edges_color` column.

    edges_palette : dict, optional, default=None
        Dictionary mapping categories in the `edges_color` column to specific colors,
        if the column represents categorical data.

    edges_cbar : bool, optional, default=False
        Whether to display the colorbar for the edges.

    edges_cbar_axes : matplotlib.axes.Axes, optional, default=None
        Axes to plot the colorbar for the edges. If not provided, it will be
        automatically adjusted to the current Axes.

    edges_width : float, optional, default=0.5
        Width of the lines representing the edges.

    edges_max_width : float, optional, default=1
        Maximum width for the edges when scaled.

    edges_min_width : float, optional, default=0.1
        Minimum width for the edges when scaled.

    sort_by : str, optional, default=None
        Column in `nodes_df` to use for sorting the nodes before plotting.

    sort_ascending : bool, optional, default=False
        Whether to sort the nodes in ascending order based on the `sort_by` column.

    center_spines : bool, optional, default=False
        Whether to center the spines of the plot at (0, 0).

    add_hist : bool, optional, default=False
        Whether to add a histogram inset showing the distribution of node colors.

    inset_cbar : bool, optional, default=False
        Whether to add an inset colorbar for the nodes.

    inset_pos : tuple, optional, default=(0.7, 0.7)
        Position of the inset colorbar or histogram as a fraction of the Axes.

    prev_nodes_df : pd.DataFrame, optional, default=None
        DataFrame containing the previous positions of the nodes. If provided,
        the current nodes will be aligned to minimize their distance from the
        previous positions.

    rasterized : bool, optional, default=False
        Whether to rasterize the plot for better performance with large datasets.
    """

    if prev_nodes_df is not None:
        axis = [x, y] if z is None else [x, y, z]
        nodes_df = minimize_nodes_distance(nodes_df, prev_nodes_df, axis)

    if edges_df is not None:
        plot_edges(
            axes,
            nodes_df,
            edges_df,
            x=x,
            y=y,
            z=z,
            alpha=edges_alpha,
            zorder=edges_zorder,
            color=edges_color,
            cmap=edges_cmap,
            cbar_axes=edges_cbar_axes,
            cbar=edges_cbar,
            palette=edges_palete,
            width=edges_width,
            max_width=edges_max_width,
            min_width=edges_min_width,
            rasterized=rasterized,
        )

    orientation = "vertical"
    if add_hist:
        check_error(
            inset_cbar, msg="inset_cbar must be true for adding histogram"
        )
        orientation = "horizontal"
        nodes_hist_axes = get_hist_inset_axes(axes, pos=inset_pos)
        if nodes_color in nodes_df.columns:
            plot_color_hist(
                nodes_hist_axes, nodes_df[nodes_color], nodes_cmap, bins=20
            )
    if inset_cbar:
        nodes_cbar_axes = get_cbar_inset_axes(
            axes, horizontal=orientation == "horizontal", pos=inset_pos
        )

    plot_nodes(
        axes,
        nodes_df=nodes_df,
        x=x,
        y=y,
        z=z,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        color=nodes_color,
        size=nodes_size,
        cmap=nodes_cmap,
        cbar=nodes_cbar,
        cbar_axes=nodes_cbar_axes,
        cbar_label=nodes_cmap_label,
        cbar_orientation=orientation,
        palette=nodes_palette,
        alpha=nodes_alpha,
        zorder=nodes_zorder,
        max_size=nodes_max_size,
        min_size=nodes_min_size,
        edgecolor=nodes_edgecolor,
        lw=nodes_lw,
        vmax=nodes_vmax,
        vmin=nodes_vmin,
        vcenter=nodes_vcenter,
        rasterized=rasterized,
    )

    add_axis_labels(axes, x, y, z=z, center_spines=center_spines)


def calc_stationary_ymeans(y, n, ymin=None, ymax=None, pmin=0.05, pmax=0.8):
    ymean = y.mean()
    dy = y.max() - ymean

    if ymin is None:
        ymin = ymean + pmin * dy
    if ymax is None:
        ymax = ymean + pmax * dy

    ymeans = np.linspace(ymin, ymax, n)
    return ymeans


def figure_Ns_grid(
    rw,
    x="1",
    y="2",
    pmin=0,
    pmax=0.8,
    ncol=4,
    nrow=3,
    show_edges=True,
    fpath=None,
    **kwargs,
):
    """
    Generate a grid of visualizations for different stationary mean functions.

    Parameters
    ----------
    rw : object
        An object containing the space and nodes information, as well as 
        methods for calculating visualizations.

    x : str, optional
        Column in the nodes DataFrame to use for plotting the x-axis. Default 
        is "1".

    y : str, optional
        Column in the nodes DataFrame to use for plotting the y-axis. Default 
        is "2".

    pmin : float, optional
        Minimum proportion of the range to use for calculating mean functions. 
        Default is 0.

    pmax : float, optional
        Maximum proportion of the range to use for calculating mean functions. 
        Default is 0.8.

    ncol : int, optional
        Number of columns in the grid. Default is 4.

    nrow : int, optional
        Number of rows in the grid. Default is 3.

    show_edges : bool, optional
        Whether to include edges in the visualization. Default is True.

    fpath : str, optional
        File path to save the figure. If None, the figure will not be saved. 
        Default is None.
    """
    fig, subplots = init_fig(
        nrow,
        ncol,
        colsize=3,
        rowsize=2.7,
        sharex=True,
        sharey=True,
        hspace=0.2,
        wspace=0.2,
    )
    vspace = max(0.25 - 0.05 * nrow, 0.1)
    fig.subplots_adjust(right=0.9, left=0.085, bottom=vspace, top=1 - vspace)
    cbar_axes = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    subplots = subplots.flatten()
    ymeans = calc_stationary_ymeans(
        rw.space.y, n=ncol * nrow, pmin=pmin, pmax=pmax
    )

    prev_nodes_df = None
    for i, (mean_function, axes) in enumerate(zip(ymeans, subplots)):
        rw.calc_visualization(mean_function=mean_function, n_components=3)
        is_last_plot = i == ymeans.shape[0] - 1

        ndf = rw.nodes_df
        edf = None if not show_edges else rw.space.get_edges_df()

        plot_visualization(
            axes,
            ndf,
            edf,
            x=x,
            y=y,
            nodes_cbar=is_last_plot,
            nodes_cbar_axes=cbar_axes if is_last_plot else None,
            prev_nodes_df=prev_nodes_df,
            **kwargs,
        )
        prev_nodes_df = rw.nodes_df
        title = r"$\bar{y}$" + " = {:.2f}; Ns={:.2f}".format(
            mean_function, rw.Ns
        )
        axes.set(xlabel="", ylabel="", title=title)

    fig.supxlabel(
        "Diffusion axis {}".format(x), y=0.05, ha="center", va="center"
    )
    fig.supylabel(
        "Diffusion axis {}".format(y), x=0.05, ha="center", va="center"
    )
    savefig(fig, fpath, tight=False)


def get_max_axis(max_axis, nodes_df):
    if max_axis is None:
        max_axis = 1
        while str(max_axis) in nodes_df.columns:
            max_axis += 1
        max_axis -= 1
    else:
        msg = "max_axis should be in nodes_df.columns"
        check_error(str(max_axis) in nodes_df.columns, msg=msg)
    max_axis -= 1
    return max_axis


def figure_axis_grid(
    nodes_df,
    max_axis=None,
    edges_df=None,
    fpath=None,
    colsize=3,
    rowsize=2.7,
    fmt="png",
    **kwargs,
):
    max_axis = get_max_axis(max_axis, nodes_df)
    fig, subplots = init_fig(
        max_axis, max_axis, colsize=colsize, rowsize=rowsize
    )
    fig.subplots_adjust(right=0.87, left=0.13)
    cbar_axes = fig.add_axes([0.9, 0.2, 0.015, 0.6])

    for i, j in product(list(range(max_axis)), repeat=2):
        is_last_plot = (i == max_axis - 1) and (j == max_axis - 1)
        axes = subplots[i][j]
        x, y = str(j + 1), str(i + 2)

        if j > i:
            empty_axes(axes)
        else:
            plot_visualization(
                axes,
                nodes_df,
                edges_df=edges_df,
                x=x,
                y=y,
                nodes_cbar=is_last_plot,
                nodes_cbar_axes=cbar_axes if is_last_plot else None,
                **kwargs,
            )

        if i < max_axis - 1:
            axes.set(xlabel="", xticks=[])
        if j > 0:
            axes.set(ylabel="", yticks=[])

    savefig(fig, fpath, fmt=fmt, tight=False)


def figure_allele_grid(
    nodes_df,
    edges_df=None,
    allele_color="orange",
    background_color="lightgrey",
    positions=None,
    position_labels=None,
    colsize=3,
    rowsize=2.7,
    xpos_label=0.05,
    ypos_label=0.92,
    fmt="png",
    fpath=None,
    **kwargs,
):
    """
    Generate a grid of visualizations for alleles at specific positions.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame containing the nodes' information, including their coordinates
        and attributes.

    edges_df : pd.DataFrame, optional
        DataFrame containing the edges' information, including connectivity
        between nodes. If None, edges will not be plotted. Default is None.

    allele_color : str, optional
        Color used to highlight nodes corresponding to specific alleles. Default is "orange".

    background_color : str, optional
        Color used for the background nodes. Default is "lightgrey".

    positions : array-like, optional
        List of positions to visualize. If None, all positions will be used. Default is None.

    position_labels : array-like, optional
        Labels for the positions. If None, positions will be labeled sequentially. Default is None.

    colsize : int, optional
        Width of each column in the grid. Default is 3.

    rowsize : float, optional
        Height of each row in the grid. Default is 2.7.

    xpos_label : float, optional
        Horizontal position of the allele label within each subplot, as a fraction
        of the axes width. Default is 0.05.

    ypos_label : float, optional
        Vertical position of the allele label within each subplot, as a fraction
        of the axes height. Default is 0.92.

    fmt : str, optional
        Format to save the figure, e.g., "png" or "pdf". Default is "png".

    fpath : str, optional
        File path to save the figure. If None, the figure will not be saved. Default is None.
    """
    config = guess_space_configuration(nodes_df.index.values)
    length, n_alleles = config["length"], np.max(config["n_alleles"])

    if position_labels is None:
        position_labels = np.arange(length) + 1

    if positions is None:
        positions = np.arange(length)

    fig, subplots = init_fig(
        n_alleles,
        positions.shape[0],
        colsize=colsize,
        rowsize=rowsize,
        sharex=True,
        sharey=True,
    )

    for i, axes_row in enumerate(subplots):
        for j, (pos, axes) in enumerate(zip(positions, axes_row)):
            try:
                allele = config["alphabet"][j][i]
            except IndexError:
                empty_axes(axes)
                continue

            plot_visualization(
                axes,
                nodes_df,
                edges_df=edges_df,
                nodes_color=background_color,
                **kwargs,
            )
            idxs = np.array([seq[pos] == allele for seq in nodes_df.index])
            plot_visualization(
                axes, nodes_df.loc[idxs, :], nodes_color=allele_color, **kwargs
            )

            axes.text(
                xpos_label,
                ypos_label,
                "{}{}".format(allele, position_labels[j]),
                ha="left",
                transform=axes.transAxes,
            )
            axes.set(xlabel="", ylabel="")

    x, y = "1", "2"
    if "x" in kwargs:
        x = kwargs["x"]
    if "y" in kwargs:
        y = kwargs["y"]

    fig.supxlabel(
        "Diffusion axis {}".format(x), y=0.05, ha="center", va="center"
    )
    fig.supylabel(
        "Diffusion axis {}".format(y), x=0.05, ha="center", va="center"
    )
    savefig(fig, fpath, fmt=fmt, tight=False)


def figure_shifts_grid(
    nodes_df,
    seq,
    edges_df=None,
    fpath=None,
    x="1",
    y="2",
    allele_color="orange",
    background_color="lightgrey",
    nodes_size=None,
    edges_color="grey",
    edges_width=0.5,
    positions=None,
    position_labels=None,
    colsize=3,
    rowsize=2.7,
    xpos_label=0.05,
    ypos_label=0.92,
    is_prot=False,
    alphabet_type="rna",
    codon_table="Standard",
    ncol=None,
    nrow=1,
    labels_full_seq=False,
):
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
        if alphabet_type in ["rna", "dna"]:
            genotype_groups = ["N" * col + seq + "N" * (length - col - 1)]
        else:
            genotype_groups = ["X" * col + seq + "X" * (length - col - 1)]
        axes = subplots[col]
        plot_visualization(
            axes,
            nodes_df,
            edges_df=edges_df,
            x=x,
            y=y,
            nodes_color=background_color,
            nodes_size=nodes_size,
            edges_color=edges_color,
            edges_width=edges_width,
        )
        sel_nodes_df = get_nodes_df_highlight(
            nodes_df,
            genotype_groups=genotype_groups,
            is_prot=is_prot,
            alphabet_type=alphabet_type,
            codon_table=codon_table,
        )
        plot_nodes(
            axes, sel_nodes_df, x=x, y=y, color=allele_color, size=nodes_size
        )

        if col // ncol != nrow - 1:
            axes.set_xlabel("")
            axes.set_xticks([])

        if col % ncol != 0:
            axes.set_ylabel("")
            axes.set_yticks([])

        xlims, ylims = axes.get_xlim(), axes.get_ylim()
        xpos = xlims[0] + xpos_label * (xlims[1] - xlims[0])
        ypos = ylims[0] + ypos_label * (ylims[1] - ylims[0])

        panel_label = (
            genotype_groups[0]
            if labels_full_seq
            else "{}{}".format(seq, position_labels[j])
        )
        axes.text(xpos, ypos, panel_label, ha="left")

    savefig(fig, fpath)


"""
Rasterization functionality
"""


def raster_edges(
    nodes_df,
    edges_df,
    x="1",
    y="2",
    color=None,
    aa=True,
    only_first=False,
    resolution=800,
):
    cs = cycle([1]) if color is None else edges_df[color]
    xy = digitize_coords(nodes_df, x, y, resolution)

    vs = np.zeros(xy.max(0) + 1)
    edges = np.hstack([xy[edges_df["i"], :], xy[edges_df["j"], :]])

    for (r0, c0, r1, c1), c in zip(edges, cs):
        if aa:
            rr, cc, v = line_aa(r0, c0, r1, c1)
        else:
            rr, cc = line(r0, c0, r1, c1)
            v = 1.0

        v = v * c
        if only_first:
            v = v * (vs[rr, cc] == 0.0).astype(float)
        vs[rr, cc] += v

    return vs.T[::-1, :]


def digitize_coords(nodes_df, x, y, resolution):
    xmin, xmax = nodes_df[x].min(), nodes_df[x].max()
    ymin, ymax = nodes_df[y].min(), nodes_df[y].max()
    ratio = (xmax - xmin) / (ymax - ymin)
    xres, yres = int(resolution * ratio), int(resolution / ratio)
    xbins = np.linspace(xmin, xmax, xres + 1)
    xbins[-1] += 1e-12
    ybins = np.linspace(ymin, ymax, yres + 1)
    ybins[-1] += 1e-12
    x, y = (
        np.digitize(nodes_df[x], xbins) - 1,
        np.digitize(nodes_df[y], ybins) - 1,
    )
    xy = np.vstack([x, y]).T
    return xy


def prep_nodes_raster_df(nodes_df, x, y, resolution, color):
    xy = digitize_coords(nodes_df, x, y, resolution)
    df = pd.DataFrame(xy, columns=["x", "y"])
    if color is None:
        df["c"] = 1.0
    else:
        df["c"] = nodes_df[color].values
    return df


def raster_nodes(
    nodes_df,
    x="1",
    y="2",
    color="function",
    palette=None,
    sort_by=None,
    sort_ascending=False,
    only_first=True,
    resolution=800,
):
    df = prep_nodes_raster_df(nodes_df, x, y, resolution, color)

    if sort_by is not None:
        df.sort_values(sort_by, ascending=sort_ascending, inplace=True)

    if only_first:
        v = df.groupby(["x", "y"])["c"].first().reset_index()
    else:
        v = df.groupby(["x", "y"])["c"].sum().reset_index()

    z = np.zeros(v[["x", "y"]].max() + 1)
    x, y = v["x"].values, v["y"].values
    z[x, y] = v["c"].values
    return z.T[::-1, :]


def calc_raster(
    nodes_df,
    x="1",
    y="2",
    edges_df=None,
    nodes_resolution=800,
    edges_resolution=1200,
    antialias=True,
):
    rasters = raster_nodes(nodes_df, resolution=nodes_resolution)

    if edges_df is not None:
        edges_raster = raster_edges(
            nodes_df, edges_df, resolution=edges_resolution, aa=antialias
        )
        rasters = (rasters, edges_raster)
    extent = (
        nodes_df[x].min(),
        nodes_df[x].max(),
        nodes_df[y].min(),
        nodes_df[y].max(),
    )
    return rasters + (extent,)


def plot_raster(
    axes,
    z,
    extent=(0, 1, 0, 1),
    cmap="viridis",
    vmin=None,
    vmax=None,
    alpha=1,
    cbar=False,
    cbar_axes=None,
    cbar_label="",
    fontsize=None,
    orientation="veritcal",
    set_zero_transparent=True,
):
    if set_zero_transparent:
        z[z == 0.0] = np.nan
    im = axes.imshow(
        z, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, extent=extent
    )
    if cbar:
        add_cbar(
            im,
            axes,
            cbar_axes=cbar_axes,
            label=cbar_label,
            fontsize=fontsize,
            orientation=orientation,
            vmin=vmin,
            vmax=vmax,
        )


def plot_visualization_raster(
    axes,
    nodes_raster,
    extent,
    edges_raster=None,
    x="1",
    y="2",
    nodes_cmap="viridis",
    nodes_vmin=None,
    nodes_vmax=None,
    nodes_cbar=True,
    nodes_cbar_axes=None,
    nodes_cmap_label="Function",
    edges_cmap="Greys",
    edges_alpha=0.5,
    center_spines=False,
    inset_cbar=False,
    inset_pos=(0.7, 0.7),
    axis_fontsize=None,
    fontsize=None,
):
    if edges_raster is not None:
        z = np.log(edges_raster + 1)
        z = z / z.max()
        plot_raster(axes, z, extent, cmap=edges_cmap, alpha=edges_alpha)

    orientation = "vertical"
    if inset_cbar:
        nodes_cbar_axes = get_cbar_inset_axes(
            axes, horizontal=orientation == "horizontal", pos=inset_pos
        )

    plot_raster(
        axes,
        nodes_raster,
        extent,
        cmap=nodes_cmap,
        vmin=nodes_vmin,
        vmax=nodes_vmax,
        cbar=nodes_cbar,
        cbar_axes=nodes_cbar_axes,
        cbar_label=nodes_cmap_label,
        fontsize=fontsize,
        orientation=orientation,
    )

    add_axis_labels(
        axes, x, y, fontsize=axis_fontsize, center_spines=center_spines
    )

    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    padding = 0.1
    xlim = (xlim[0] - padding * dx, xlim[1] + padding * dx)
    ylim = (ylim[0] - padding * dy, ylim[1] + padding * dy)
    axes.set(xlim=xlim, ylim=ylim)


"""
Auxiliary plots
"""


def plot_hyperparam_cv(
    df,
    axes,
    x="log_a",
    y="logL",
    err_bars="stderr",
    xlabel=r"$\log_{10}(a)$",
    ylabel="log(L)",
    show_folds=True,
    highlight="max",
    legend_loc=1,
    refx=None,
    size=15,
):
    if refx is not None and np.isin(refx, df[x].values):
        refdf = df.loc[df[x] == refx, :].set_index("fold")[[y]]
        df = df.join(refdf, on="fold", rsuffix="_ref")
        df[y] = df[y] - df["{}_ref".format(y)]

    sdf = df.groupby(x, as_index=False).agg({y: ("mean", "std", "count")})
    sdf.columns = ["x", "mean", "sd", "count"]
    sdf["stderr"] = sdf["sd"] / np.sqrt(sdf["count"])

    idx = sdf["mean"].argmax() if highlight == "max" else sdf["mean"].argmin()
    x_star = sdf["x"][idx]
    y_star = sdf["mean"][idx]

    if show_folds:
        for _, fold_df in df.groupby("fold"):
            axes.plot(
                fold_df[x], fold_df[y], c="black", lw=0.5, alpha=0.4, zorder=1
            )

    axes.plot(sdf["x"], sdf["mean"], color="black", lw=1, zorder=1)
    axes.scatter(sdf["x"], sdf["mean"], color="black", s=size)
    axes.scatter(x_star, y_star, color="red", s=size + 1, zorder=10, alpha=1)

    for a, m, s in zip(sdf["x"], sdf["mean"], sdf[err_bars]):
        color = "red" if a == x_star else "black"

        if a == x_star:
            label = "{}* = {:.1f}".format(xlabel, x_star)
        else:
            label = None

        axes.plot((a, a), (m - s, m + s), lw=1, color=color, label=label)

    xlims, ylims = axes.get_xlim(), axes.get_ylim()
    ylims = (ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.1)

    for r in sdf.loc[np.isinf(sdf["x"]), :].to_dict(orient="index").values():
        x, y = r["x"], r["mean"]
        sign = "" if x > 0 else "-"
        label = "{}".format(xlabel) + " = {}".format(sign) + r"$\infty$"

        if not np.isnan(y):
            axes.plot(
                xlims, (y, y), lw=0.5, c="darkred", linestyle="--", label=label
            )

    axes.legend(loc=legend_loc, fontsize=9)
    axes.set(
        xlabel=xlabel,
        ylabel="Out of sample {}".format(ylabel),
        ylim=ylims,
        xlim=xlims,
    )


def plot_density_vs_frequency(seq_density, axes):
    with np.errstate(divide="ignore"):
        logf = np.log10(seq_density["frequency"])
        logq = np.log10(seq_density["Q_star"])
        data = pd.DataFrame({"logR": logf, "logQ": logq}).dropna()

    axes.scatter(
        data["logR"],
        data["logQ"],
        color="black",
        s=5,
        alpha=0.4,
        zorder=2,
        label="Observed sequences",
    )

    mask = np.isinf(logf)
    zero_counts_logq = logq[mask]
    fake_logf = np.full(zero_counts_logq.shape, logf[~mask].min() - 0.5)
    axes.scatter(
        fake_logf,
        zero_counts_logq,
        marker="<",
        color="red",
        s=5,
        alpha=0.2,
        zorder=2,
        label="Unobserved sequences",
    )

    xlims, ylims = axes.get_xlim(), axes.get_ylim()
    lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1])
    axes.plot(lims, lims, color="grey", linewidth=0.5, alpha=0.5, zorder=1)
    axes.set(
        xlabel=r"$log_{10}$(Frequency)",
        ylabel=r"$log_{10}$(Q*)",
        xlim=lims,
        ylim=lims,
    )
    axes.legend(loc=2, fontsize=9)


def figure_SeqDEFT_summary(
    log_Ls,
    seq_density=None,
    err_bars="stderr",
    show_folds=False,
    legend_loc=1,
    normalize_logL=True,
):
    """
    Generates a 2-panel figure summarizing the SeqDEFT model results.

    The first panel shows how the cross-validated likelihood changes with the
    ``a`` hyperparameter and highlights the best-selected value for model
    fitting. If sequence density data is provided, the second panel visualizes
    the relationship between observed sequence frequencies and estimated
    densities.

    Parameters
    ----------
    log_Ls : pd.DataFrame
        A DataFrame of shape (num_a, 3) containing the columns:

        - ``a``: The hyperparameter values.
        - ``logL``: The log-likelihood values.
        - ``fold``: The cross-validation fold identifiers.

    seq_density : pd.DataFrame, optional
        A DataFrame of shape (n_genotypes, >= 2) with the following columns:

        - ``frequency``: Observed frequencies for each sequence.
        - ``Q``: Estimated densities for each sequence.
        If not provided, only a single-panel figure with the cross-validated
        likelihood curve will be generated.

    err_bars : str, default='stderr'
        Specifies the type of error bars to display:

        - ``'sd'``: Standard deviation across the different folds.
        - ``'stderr'``: Standard error of the mean.

    show_folds : bool, default=False
        Whether to display the out-of-sample log-likelihoods for the individual
        folds in the cross-validation procedure.

    legend_loc : int, default=1
        The location of the legend in the plot. Follows matplotlib's legend
        location codes.

    normalize_logL : bool, default=True
        If True, normalizes the log-likelihood values relative to the value at
        ``a = ∞``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure object containing the generated plots.
    """
    if seq_density is None:
        fig, axes = init_fig(1, 1, colsize=5, rowsize=4.3)
    else:
        fig, subplots = init_fig(1, 2, colsize=4, rowsize=3.5)
        plot_density_vs_frequency(seq_density, subplots[1])
        axes = subplots[0]

    ylabel = r"log(L) relative to a=$\infty$" if normalize_logL else "log(L)"
    plot_hyperparam_cv(
        log_Ls,
        axes,
        err_bars=err_bars,
        show_folds=show_folds,
        legend_loc=legend_loc,
        ylabel=ylabel,
        refx=np.inf if normalize_logL else None,
    )

    if seq_density is not None:
        fig.tight_layout()
    return fig
