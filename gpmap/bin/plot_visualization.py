#!/usr/bin/env python
import argparse

import matplotlib
import matplotlib.pyplot as plt
import gpmap.plot.mpl as mplot
import gpmap.plot.ds as dplot
import gpmap.plot.ply as pplot

from gpmap.utils import LogTrack, read_dataframe, read_edges


def main():
    description = "Plot the low dimensional representation of a pre-processed"
    description += "landscape with calc_visualization"

    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group("Input")
    help_msg = "CSV file containing nodes coordinates and attributes"
    input_group.add_argument("nodes", help=help_msg)

    nodes_group = parser.add_argument_group("Nodes options")
    nodes_group.add_argument(
        "-nc",
        "--nodes_color",
        default="function",
        help="Color genotypes according to this property (function)",
    )
    nodes_group.add_argument(
        "-l", "--label", default=None, help="Function label to show on colorbar"
    )
    nodes_group.add_argument(
        "--cmap", default="viridis", help="Colormap to use (viridis)"
    )
    nodes_group.add_argument(
        "-ns", "--nodes_size", default=50, type=float, help="Nodes size (50)"
    )
    help_msg = "Sort nodes according to this property before plotting (f)"
    nodes_group.add_argument("-s", "--sort_by", default="f", help=help_msg)
    nodes_group.add_argument(
        "--ascending",
        default=False,
        action="store_true",
        help="Sort nodes for plotting in ascending order",
    )

    edges_group = parser.add_argument_group("Edges options")
    edges_group.add_argument(
        "-e",
        "--edges",
        default=None,
        help="npz or csv file containing edges data for plotting",
    )
    edges_group.add_argument(
        "-ec", "--edges_color", default="grey", help="Edges color (grey)"
    )
    edges_group.add_argument(
        "-ea",
        "--edges_alpha",
        default=0.1,
        type=float,
        help="Edges transparency (0.1)",
    )
    edges_group.add_argument(
        "-ew",
        "--edges_width",
        default=0.5,
        type=float,
        help="Edges width (0.5)",
    )

    fig_group = parser.add_argument_group("Figure options")
    help_msg = "Comma separated list of diffusion axis to display (1,2)"
    fig_group.add_argument("-a", "--axis", default="1,2", help=help_msg)
    fig_group.add_argument(
        "--interactive",
        default=False,
        action="store_true",
        help="Make interactive html",
    )
    fig_group.add_argument(
        "--datashader",
        default=False,
        action="store_true",
        help="Use datashader for plotting. Recommended for big landscapes",
    )
    fig_group.add_argument(
        "-H",
        "--height",
        default=5,
        type=float,
        help="Figure height in inches when using datashader (5)",
    )
    fig_group.add_argument(
        "-W",
        "--width",
        default=5,
        type=float,
        help="Figure width in inches when using datashader (5)",
    )
    fig_group.add_argument(
        "-nr",
        "--nodes_resolution",
        default=600,
        type=int,
        help="Resolution for datashader plotting of nodes (600)",
    )
    fig_group.add_argument(
        "-er",
        "--edges_resolution",
        default=1200,
        type=int,
        help="Resolution for datashader plotting of edges (1200)",
    )
    fig_group.add_argument(
        "-f", "--format", default="png", help="Figure format (png)"
    )

    layouts_group = parser.add_argument_group("Special layouts")
    help_msg = "Layout with panels highlighting alleles at each position"
    layouts_group.add_argument(
        "--alleles", default=False, action="store_true", help=help_msg
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", required=True, help="Output file"
    )

    # Parse arguments
    parsed_args = parser.parse_args()
    nodes_fpath = parsed_args.nodes

    nodes_color = parsed_args.nodes_color
    label = parsed_args.label
    nodes_cmap = parsed_args.cmap
    nodes_size = parsed_args.nodes_size
    sort_by = parsed_args.sort_by
    ascending = parsed_args.ascending

    edges_fpath = parsed_args.edges
    edges_color = parsed_args.edges_color
    edges_alpha = parsed_args.edges_alpha
    edges_width = parsed_args.edges_width

    axis = parsed_args.axis.split(",")
    if len(axis) == 2:
        axis.append(None)
    x, y, z = axis
    interactive = parsed_args.interactive
    use_datashader = parsed_args.datashader
    height = parsed_args.height
    width = parsed_args.width
    nodes_resolution = parsed_args.nodes_resolution
    edges_resolution = parsed_args.edges_resolution

    alleles_grid = parsed_args.alleles

    fmt = parsed_args.format
    out_fpath = parsed_args.output

    # Load data
    log = LogTrack()
    log.write("Start analysis")

    log.write("Reading genotype data from {}".format(nodes_fpath))
    nodes_df = read_dataframe(nodes_fpath)
    edges_df = read_edges(edges_fpath)

    log.write("Plot visualization")
    figsize = (width, height)

    if use_datashader:
        matplotlib.use("Agg")  # Needed to resize figure out of datashader
        if alleles_grid:
            dplot.figure_allele_grid(
                nodes_df,
                out_fpath,
                x=x,
                y=y,
                edges_df=edges_df,
                edges_cmap="grey",
                background_color="white",
                nodes_resolution=nodes_resolution,
                edges_resolution=edges_resolution,
                fmt=fmt,
                figsize=figsize,
            )

        else:
            dsg = dplot.plot_visualization(
                nodes_df,
                x=x,
                y=y,
                edges_df=edges_df,
                nodes_color=nodes_color,
                nodes_cmap=nodes_cmap,
                edges_cmap="grey",
                background_color="white",
                nodes_resolution=nodes_resolution,
                edges_resolution=edges_resolution,
            )
            log.write("Setting figure size to {}".format(figsize))
            dplot.savefig(dsg, out_fpath, fmt=fmt, figsize=figsize)

    else:
        if alleles_grid:
            mplot.figure_allele_grid(
                nodes_df,
                edges_df=edges_df,
                fpath=out_fpath,
                x=x,
                y=y,
                allele_color="orange",
                background_color="lightgrey",
                nodes_size=nodes_size,
                edges_color=edges_color,
                edges_width=edges_width,
                colsize=3,
                rowsize=2.7,
                xpos_label=0.05,
                ypos_label=0.92,
                fmt=fmt,
            )
        else:
            if interactive:
                pplot.plot_visualization(
                    nodes_df,
                    edges_df,
                    x,
                    y,
                    z,
                    nodes_color,
                    nodes_size,
                    nodes_cmap=nodes_cmap,
                    nodes_cmap_label=label,
                    edges_width=edges_width,
                    edges_color=edges_color,
                    edges_alpha=edges_alpha,
                    fpath=out_fpath,
                )
            else:
                fig, axes = plt.subplots(1, 1, figsize=figsize)
                mplot.plot_visualization(
                    axes,
                    nodes_df,
                    edges_df=edges_df,
                    x=x,
                    y=y,
                    z=z,
                    nodes_color=nodes_color,
                    nodes_cmap=nodes_cmap,
                    nodes_size=nodes_size,
                    nodes_cmap_label=label,
                    edges_color=edges_color,
                    edges_width=edges_width,
                    edges_alpha=edges_alpha,
                    sort_by=sort_by,
                    sort_ascending=ascending,
                )
                mplot.savefig(fig, out_fpath, fmt=fmt, figsize=figsize)

    log.finish()


if __name__ == "__main__":
    main()
