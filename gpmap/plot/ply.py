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
    Creates an interactive plot of a fitness landscape with genotypes as nodes
    and single point mutations as edges using Plotly.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame containing genotype information. Must include columns for
        coordinates (e.g., "1", "2", "3"), "function", and optionally other
        metadata.

    edges_df : pd.DataFrame, optional
        DataFrame containing edge connectivity information. Must include
        columns "i" and "j" for connected node indices.

    x : str, default '1'
        Column name in `nodes_df` for the x-axis coordinates.

    y : str, default '2'
        Column name in `nodes_df` for the y-axis coordinates.

    z : str, optional
        Column name in `nodes_df` for the z-axis coordinates. If provided,
        a 3D plot will be generated.

    nodes_color : str, default 'function'
        Column name in `nodes_df` for node coloring or a specific color value.

    nodes_size : float, default 4
        Size of the nodes. Can be a constant or a column name in `nodes_df`.

    nodes_cmap : str, default 'viridis'
        Colormap for node coloring.

    nodes_cmap_label : str, default 'Function'
        Label for the colorbar associated with node coloring.

    edges_width : float, default 0.5
        Width of the edges. Can be a constant or a column name in `edges_df`.

    edges_color : str, default '#888'
        Color of the edges.

    edges_alpha : float, default 0.2
        Transparency of the edges.

    text : array-like, optional
        Labels for nodes to display on hover. Defaults to `nodes_df.index`.

    fpath : str, optional
        File path to save the interactive plot as an HTML file.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated Plotly figure.
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
