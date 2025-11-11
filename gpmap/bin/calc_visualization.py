#!/usr/bin/env python
import argparse

from os.path import exists

import pandas as pd

from gpmap.utils import LogTrack, read_dataframe
from gpmap.space import SequenceSpace
from gpmap.randwalk import WMWalk
from gpmap.seq import get_custom_codon_table

        
def main():
    description = 'Calculates the eigen-decomposition of the time reversible '
    description += 'rate matrix under a weak selection weak mutation regime '
    description += '(WMWS) on a landscape to obtain a low dimensional '
    description += 'representation'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    help_msg = 'CSV or Parquet file with sequence-function map sorted by sequence'
    input_group.add_argument('input', help=help_msg)

    coding_group = parser.add_argument_group('Coding options')
    coding_group.add_argument('-C', '--use_codon_model', default=False,
                              action='store_true',
                              help='Use codon model for visualization of protein landscape')
    help_msg = 'NCBI Codon table to use for equivalence (None) or '
    help_msg += 'CSV file with codon-aa correspondance information. If provided'
    help_msg += 'without -C option, it will restrict transitions in protein '
    help_msg += 'sequences to those allowed by this codon table'
    coding_group.add_argument('-c', '--codon_table', default=None, help=help_msg)
    coding_group.add_argument('-sf', '--stop_f', default=None, type=float,
                              help='Function for stop codons (minimum in dataset by default)')
    
    viz_group = parser.add_argument_group('Visualization options')
    viz_group.add_argument('-k', '--n_components', default=5, type=int,
                           help='Number of eigenvectors to compute (5)')
    help_msg = 'Scaled effective population size (Ns) for building the '
    help_msg += 'rate matrix'
    viz_group.add_argument('-Ns', '--Ns', default=None, type=float, help=help_msg)
    viz_group.add_argument('-m', '--mean_function', default=None, type=float,
                           help='Set Ns to reach this mean function at stationarity')
    help_msg = 'Set Ns to reach a mean function equal to this percentile in '
    help_msg += 'the distribution'
    viz_group.add_argument('-p', '--percentile_function', default=None,
                           type=float, help=help_msg)
    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True,
                              help='Output file')
    output_group.add_argument('-e', '--edges', default=False, action='store_true',
                              help='Write edges')
    output_group.add_argument('-nf', '--nodes_format', default='pq',
                              help='Nodes format [pq, csv] (pq)')
    output_group.add_argument('-ef', '--edges_format', default='npz',
                              help='Edges format [npz, csv] (npz)')
    

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.input
    
    n_components = parsed_args.n_components
    Ns = parsed_args.Ns
    mean_function = parsed_args.mean_function
    mean_function_perc = parsed_args.percentile_function
    
    use_codon_model = parsed_args.use_codon_model
    codon_table = parsed_args.codon_table
    stop_y = parsed_args.stop_f
    
    out_fpath = parsed_args.output
    write_edges = parsed_args.edges
    nodes_format = parsed_args.nodes_format
    edges_format = parsed_args.edges_format
    
    # Load data
    log = LogTrack()
    log.write('Start analysis')
    data = read_dataframe(data_fpath)
    
    # Build space
    X, y = data.index.values, data.iloc[:, 0].values
    
    if stop_y is None and use_codon_model:
        stop_y = y.min()
        
    space = SequenceSpace(X=X, y=y, stop_y=stop_y)
        
    # Transform to nucleotide space if required
    if codon_table is not None:
        if exists(codon_table):
            aa_mapping = pd.read_csv(codon_table)
            codon_table = get_custom_codon_table(aa_mapping)
        
        if use_codon_model:
            space = space.to_nucleotide_space(codon_table)
        else:
            space.remove_codon_incompatible_transitions(codon_table)
    
    # Create random walk and calculate coordinates
    rw = WMWalk(space)
    rw.calc_visualization(Ns=Ns, mean_function=mean_function,
                          mean_function_perc=mean_function_perc,
                          n_components=n_components)
    rw.write_tables(out_fpath, write_edges=write_edges,
                    nodes_format=nodes_format, edges_format=edges_format)
    
    log.finish()


if __name__ == '__main__':
    main()
