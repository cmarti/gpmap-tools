#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.utils import LogTrack
from gpmap.visualization import Visualization
from gpmap.src.space import SequenceSpace
from gpmap.src.randwalk import WMWSWalk
from gpmap.src.seq import guess_space_configuration

        
def main():
    description = 'Calculates the eigen-decomposition of the time reversible '
    description += 'rate matrix under a weak selection weak mutation regime '
    description += '(WMWS) on a landscape to obtain a low dimensional '
    description += 'representation'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    help_msg = 'CSV file with sequence-function map sorted by sequence'
    input_group.add_argument('input', help=help_msg)

    options_group = parser.add_argument_group('Landscape options')
    options_group.add_argument('-A', '--alphabet_type', default='guess',
                               help='Alphabet type [guess, dna, rna, protein, custom] (guess)')
    options_group.add_argument('-a', '--n_alleles', default=None, type=int,
                               help='Number of alleles to use for custom alphabet')
    
    coding_group = parser.add_argument_group('Coding options')
    coding_group.add_argument('-C', '--use_codon_model', default=False,
                              action='store_true',
                              help='Use codon model for visualization of protein landscape')
    coding_group.add_argument('-c', '--codon_table', default=None,
                              help='NCBI Codon table to use for equivalence (None)')
    coding_group.add_argument('-sf', '--stop_f', default=-10, type=float,
                              help='Function for stop codons')
    
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
    output_group.add_argument('-f', '--edges_format', default='npz',
                              help='Edges format [npz, csv] (npz)')
    

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.input
    alphabet_type = parsed_args.alphabet_type
    n_alleles = parsed_args.n_alleles
    
    n_components = parsed_args.n_components
    Ns = parsed_args.Ns
    mean_function = parsed_args.mean_function
    mean_function_perc = parsed_args.percentile_function
    
    use_codon_model = parsed_args.use_codon_model
    codon_table = parsed_args.codon_table
    stop_function = parsed_args.stop_f
    
    out_fpath = parsed_args.output
    write_edges = parsed_args.edges
    edges_format = parsed_args.edges_format
    
    # Load data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(data_fpath, index_col=0)
    genotypes = data.index.values
    seq_length = len(genotypes[0])

    alphabet = None
    if alphabet_type == 'guess':
        config = guess_space_configuration(genotypes)
        seq_length = config['length']
        alphabet = config['alphabet']
        alphabet_type = 'custom'
        n_alleles = None
    
    # Load annotation data
    space = SequenceSpace(seq_length=seq_length, n_alleles=n_alleles,
                          alphabet=alphabet, alphabet_type=alphabet_type,
                          function=data.iloc[:, 0].values,
                          use_codon_model=use_codon_model, 
                          codon_table=codon_table, stop_function=stop_function)
    
    mc = WMWSWalk(space)
    mc.calc_visualization(Ns=Ns, mean_function=mean_function,
                          mean_function_perc=mean_function_perc,
                          n_components=n_components)
    mc.write_tables(out_fpath, write_edges=write_edges, edges_format=edges_format)
    
    log.finish()


if __name__ == '__main__':
    main()
