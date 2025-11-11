#!/usr/bin/env python
import argparse

from gpmap.genotypes import select_genotypes
from gpmap.utils import (
    LogTrack,
    read_dataframe,
    read_edges,
    write_dataframe,
    write_edges,
)


def main():
    description = "Filters nodes_df and edges_df to obtain a reduced version "
    description += "of the landscape"

    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group("nodes")
    help_msg = "File containing nodes coordinates and attributes"
    input_group.add_argument("nodes", help=help_msg)
    input_group.add_argument(
        "-e",
        "--edges",
        default=None,
        help="File containing edges data for plotting",
    )

    nodes_group = parser.add_argument_group("Filtering options")
    nodes_group.add_argument(
        "-m",
        "--min_value",
        default=None,
        type=float,
        help="Minumum value to filter",
    )
    nodes_group.add_argument(
        "-M",
        "--max_value",
        default=None,
        type=float,
        help="Maximum value to filter",
    )
    nodes_group.add_argument(
        "-l",
        "--label",
        default="function",
        help="Attribute to filter genotypes (function)",
    )

    nodes_group.add_argument(
        "-n",
        "--n_genotypes",
        default=None,
        type=int,
        help='Number of genotypes to select from sorting "label"',
    )
    nodes_group.add_argument(
        "--bottom",
        default=False,
        action="store_true",
        help="Take bottom n_genotypes instead of top",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", required=True, help="Output prefix"
    )
    output_group.add_argument(
        "-nf",
        "--nodes_format",
        default="pq",
        help="Nodes format [pq, csv] (pq)",
    )
    output_group.add_argument(
        "-ef",
        "--edges_format",
        default="npz",
        help="Edges format [npz, csv] (npz)",
    )

    # Parse arguments
    parsed_args = parser.parse_args()
    nodes_fpath = parsed_args.nodes
    edges_fpath = parsed_args.edges

    min_value = parsed_args.min_value
    max_value = parsed_args.max_value
    label = parsed_args.label
    n_genotypes = parsed_args.n_genotypes
    bottom = parsed_args.bottom

    out_prefix = parsed_args.output
    nodes_format = parsed_args.nodes_format
    edges_format = parsed_args.edges_format

    # Load data
    log = LogTrack()
    log.write("Start analysis")

    log.write("Reading genotype data from {}".format(nodes_fpath))
    nodes_df = read_dataframe(nodes_fpath)
    edges = read_edges(edges_fpath, return_df=False)

    if min_value is not None:
        log.write("Selecting genotypes with {} > {}".format(label, min_value))
        genotypes = nodes_df[label] > min_value
    elif max_value is not None:
        log.write("Selecting genotypes with {} < {}".format(label, max_value))
        genotypes = nodes_df[label] < min_value
    elif n_genotypes is not None:
        sorted_values = nodes_df[label].sort_values(ascending=True)
        if bottom:
            log.write(
                "Selecting bottom {} genotypes for {}".format(
                    n_genotypes, label
                )
            )
            genotypes = nodes_df[label] < sorted_values[n_genotypes]
        else:
            log.write(
                "Selecting top {} genotypes for {}".format(n_genotypes, label)
            )
            genotypes = nodes_df[label] > sorted_values[-n_genotypes]

    log.write("Filtering genotypes")
    ndf = select_genotypes(nodes_df, genotypes, edges=edges)
    if edges is not None:
        ndf, edges = ndf

    # Write filtered landscape
    fpath = "{}.nodes.{}".format(out_prefix, nodes_format)
    log.write("Saving filtered nodes data at {}".format(fpath))
    write_dataframe(ndf, fpath)

    if edges is not None:
        fpath = "{}.edges.{}".format(out_prefix, edges_format)
        log.write("Saving filtered edges data at {}".format(fpath))
        write_edges(edges, fpath)

    log.finish()


if __name__ == "__main__":
    main()
