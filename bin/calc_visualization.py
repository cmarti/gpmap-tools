#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.utils import LogTrack
from gpmap.visualization import Visualization

        
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
    options_group.add_argument('-A', '--alphabet_type', default='dna',
                               help='Alphabet type [dna, rna, protein, custom] (dna)')
    options_group.add_argument('-a', '--n_alleles', default=None,
                               help='Number of alleles to use for custom alphabet')
    
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

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.input
    alphabet_type = parsed_args.alphabet_type
    n_alleles = parsed_args.n_alleles
    
    n_components = parsed_args.n_components
    Ns = parsed_args.Ns
    mean_function = parsed_args.mean_function
    percentile_function = parsed_args.percentile_function
    out_fpath = parsed_args.output
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(data_fpath, index_col=0)
    length = len(data.index[0])

    # Load annotation data
    gpmap = Visualization(length, alphabet_type=alphabet_type,
                          n_alleles=n_alleles, log=log)
    gpmap.set_function(data.iloc[:, 0])
    gpmap.calc_visualization(Ns=Ns, meanf=mean_function,
                             perc_function=percentile_function,
                             n_components=n_components)
    gpmap.save(out_fpath)
    
    log.finish()


if __name__ == '__main__':
    main()
