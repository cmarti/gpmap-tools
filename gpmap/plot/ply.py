#!/usr/bin/env python
import plotly.graph_objects as go

from gpmap.plot.utils import get_lines_from_edges_df


def savefig(fig, fpath=None):
    if fpath is not None:
        fpath = "{}.html".format(fpath)
        fig.write_html(fpath)


def plot_visualization(
    nodes_df,
    edges_df=None,
    x="1",
    y="2",
    z=None,
    nodes_color="function",
    nodes_size=4,
    nodes_cmap="viridis",
    nodes_cmap_label="Function",
    edges_width=0.5,
    edges_color="#888",
    edges_alpha=0.2,
    text=None,
    fpath=None,
):
    """
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

    fpath : str
        File path in which to store the interactive plot as an html file
    """

    # Create figure
    fig = go.Figure()

    # Create nodes plot
    colorbar = dict(
        thickness=25,
        title=nodes_cmap_label,
        xanchor="left",
        titleside="right",
        len=0.8,
    )
    marker = dict(
        showscale=True,
        colorscale=nodes_cmap,
        reversescale=False,
        color=nodes_df[nodes_color],
        size=nodes_size,
        colorbar=colorbar,
        line_width=2,
    )
    if text is None:
        text = nodes_df.index

    if z is None:
        node_trace = go.Scatter(
            x=nodes_df[x],
            y=nodes_df[y],
            mode="markers",
            hoverinfo="text",
            marker=marker,
            text=text,
            name="Genotypes",
        )
    else:
        node_trace = go.Scatter3d(
            x=nodes_df[x],
            y=nodes_df[y],
            z=nodes_df[z],
            mode="markers",
            hoverinfo="text",
            marker=marker,
            text=text,
            name="Genotypes",
        )
    fig.add_trace(node_trace)

    # Create edges
    if edges_df is not None:
        edges = get_lines_from_edges_df(nodes_df, edges_df, x=x, y=y, z=z)
        if z is None:
            edge_trace = go.Scatter(
                x=edges[:, 0],
                y=edges[:, 1],
                line=dict(width=edges_width, color=edges_color),
                hoverinfo="none",
                mode="lines",
                opacity=edges_alpha,
                name="Mutations",
            )
        else:
            edge_trace = go.Scatter3d(
                x=edges[:, 0],
                y=edges[:, 1],
                z=edges[:, 2],
                line=dict(width=edges_width, color=edges_color),
                hoverinfo="none",
                mode="lines",
                opacity=edges_alpha,
                name="Mutations",
            )
        fig.add_trace(edge_trace)

    # Update layout
    scene = dict(
        xaxis_title="Diffusion axis {}".format(x),
        yaxis_title="Diffusion axis {}".format(y),
    )
    if z is not None:
        scene["zaxis_title"] = "Diffusion axis {}".format(z)

    fig.update_layout(
        title="Landscape visualization",
        hovermode="closest",
        template="simple_white",
        xaxis_title="Diffusion axis {}".format(x),
        yaxis_title="Diffusion axis {}".format(y),
        scene=scene,
    )

    savefig(fig, fpath=fpath)
    return fig
