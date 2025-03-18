#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.utils import LogTrack
from gpmap.visualization import Visualization

        
def main():
    description = 'Calculates relevant properties of nodes and edges under '
    description += 'Transition Path Theory between two sets of genotypes '
    description += 'in a landscape using a Weak Mutation Weak Selection (WMWS) '
    description += 'evolutionary model'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    help_msg = 'CSV file with sequence-function map sorted by sequence'
    input_group.add_argument('input', help=help_msg)
    help_msg = 'Comma separated list of starting genotypes (IUPAC codes allowed)'
    input_group.add_argument('-gt1', '--genotypes_1', help=help_msg)
    help_msg = 'Comma separated list of ending genotypes (IUPAC codes allowed)'
    input_group.add_argument('-gt2', '--genotypes_2', help=help_msg)

    options_group = parser.add_argument_group('Landscape options')
    options_group.add_argument('-A', '--alphabet_type', default='dna',
                               help='Alphabet type [dna, rna, protein, custom] (dna)')
    options_group.add_argument('-a', '--n_alleles', default=None, type=int,
                               help='Number of alleles to use for custom alphabet')
    
    coding_group = parser.add_argument_group('Coding options')
    coding_group.add_argument('-C', '--use_coding_sequence', default=False,
                              action='store_true',
                              help='Use codon model for visualization of protein landscape')
    coding_group.add_argument('-c', '--codon_table', default='Standard',
                              help='NCBI Codon table to use for equivalence (Standard)')
    
    viz_group = parser.add_argument_group('Evolutionary model options')
    help_msg = 'Scaled effective population size (Ns) for building the '
    help_msg += 'rate matrix'
    viz_group.add_argument('-Ns', '--Ns', default=None, type=float, help=help_msg)
    viz_group.add_argument('-m', '--mean_function', default=None, type=float,
                           help='Set Ns to reach this mean function at stationarity')
    help_msg = 'Set Ns to reach a mean function equal to this percentile in '
    help_msg += 'the distribution'
    viz_group.add_argument('-p', '--percentile_function', default=None,
                           type=float, help=help_msg)
    
    options_group = parser.add_argument_group('Transition paths options')
    options_group.add_argument('-K', '--max_paths', default=None, type=int,
                               help='Maximum number of dominant paths to calculate')
    help_msg = 'Maximum proportion of flow to missing in the set of dominant paths (0.01)'
    options_group.add_argument('-P', '--max_missing_flow_p', default=0.01,
                               type=float, help=help_msg)
    help_msg = 'Skip calculation of return probabilities to avoid inversion of large matrix'
    options_group.add_argument('--skip_p_return', default=False,
                               action='store_true', help=help_msg)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True,
                              help='Output prefix')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.input
    genotypes1 = parsed_args.genotypes_1.split(',')
    genotypes2 = parsed_args.genotypes_2.split(',')
    
    alphabet_type = parsed_args.alphabet_type
    n_alleles = parsed_args.n_alleles
    
    Ns = parsed_args.Ns
    mean_function = parsed_args.mean_function
    percentile_function = parsed_args.percentile_function
    
    max_paths = parsed_args.max_paths
    max_missing_flow_p = parsed_args.max_missing_flow_p
    skip_p_return = parsed_args.skip_p_return
    
    out_prefix = parsed_args.output
    
    use_coding_sequence = parsed_args.use_coding_sequence
    codon_table = parsed_args.codon_table if use_coding_sequence else None
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(data_fpath, index_col=0)
    length = len(data.index[0])
    if use_coding_sequence:
        length = length * 3

    # Load annotation data
    v = Visualization(length, alphabet_type=alphabet_type,
                      n_alleles=n_alleles, log=log)
    v.set_function(data.iloc[:, 0], codon_table=codon_table)

    if Ns is None:
        Ns = v.calc_Ns(stationary_function=mean_function, perc=percentile_function)
        
    v.calc_stationary_frequencies(Ns=Ns)
    v.calc_rate_matrix(Ns)
    
    objects = v.calc_transition_path_objects(genotypes1, genotypes2,
                                             max_missing_flow_p=max_missing_flow_p,
                                             max_paths=max_paths,
                                             skip_p_return=skip_p_return)
    v.write_tpt_objects(objects, out_prefix)
    
    log.finish()


if __name__ == '__main__':
    main()
