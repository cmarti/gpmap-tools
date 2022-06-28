#!/usr/bin/env python
import argparse

import pandas as pd
from scipy.sparse._matrix_io import load_npz

from gpmap.utils import LogTrack
from gpmap.visualization import filter_genotypes
from gpmap.src.genotypes import read_edges

        
def main():
    description = 'Filters nodes_df and edges_df to obtain a reduced version '
    description += 'of the landscape'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('nodes')
    help_msg = 'CSV file containing nodes coordinates and attributes'
    input_group.add_argument('nodes', help=help_msg)
    input_group.add_argument('-e', '--edges', default=None,
                             help='CSV files containing edges data for plotting')
    
    nodes_group = parser.add_argument_group('Filtering options')
    nodes_group.add_argument('-m', '--min_value', default=None, type=float,
                             help='Minumum value to filter')
    nodes_group.add_argument('-M', '--max_value', default=None, type=float,
                             help='Maximum value to filter')
    nodes_group.add_argument('-l', '--label', default='f',
                             help='Attribute to filter genotypes (f)')
    nodes_group.add_argument('-n', '--n_genotypes', default=None, type=int,
                             help='Number of genotypes to select from sorting "label"')
    nodes_group.add_argument('--bottom', default=False, action='store_true',
                             help='Take bottom n_genotypes instead of top')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True,
                              help='Output prefix')

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

    # Load data
    log = LogTrack()
    log.write('Start analysis')
    
    log.write('Reading genotype data from {}'.format(nodes_fpath))
    nodes_df = pd.read_csv(nodes_fpath, index_col=0)
    edges_df = read_edges(edges_fpath)
    
    if min_value is not None:
        log.write('Selecting genotypes with {} > {}'.format(label, min_value))
        genotypes = nodes_df[label] > min_value
    elif max_value is not None:
        log.write('Selecting genotypes with {} < {}'.format(label, max_value))
        genotypes = nodes_df[label] < min_value
    elif n_genotypes is not None:
        sorted_values = nodes_df[label].sort_values(ascending=True)
        if bottom:
            log.write('Selecting bottom {} genotypes for {}'.format(n_genotypes, label))
            genotypes = nodes_df[label] < sorted_values[n_genotypes]
        else:
            log.write('Selecting top {} genotypes for {}'.format(n_genotypes, label))
            genotypes = nodes_df[label] > sorted_values[-n_genotypes]
    
    log.write('Filtering genotypes')
    if edges_df is None:
        nodes_df = filter_genotypes(nodes_df, genotypes)
    else:
        nodes_df, edges_df = filter_genotypes(nodes_df, genotypes,
                                              edges_df=edges_df)

    log.write('Saving filtered nodes data at {}.nodes.csv'.format(out_prefix))
    nodes_df.to_csv('{}.nodes.csv'.format(out_prefix))
    if edges_df is not None:
        log.write('Saving filtered edges data at {}.edges.csv'.format(out_prefix))
        edges_df.to_csv('{}.edges.csv'.format(out_prefix))
    
    log.finish()


if __name__ == '__main__':
    main()
