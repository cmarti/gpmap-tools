#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.utils import LogTrack
from gpmap.inference import SeqDEFT

        
def main():
    description = 'Runs sequence density estimation using Field Theory (SeqDEFT)'
    description += ' from a table with the number of time each sequence was'
    description += ' observed e.g. the number of times each splice donor or'
    description += ' acceptor is found in the human genome'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('counts', help='CSV table with sequence counts')

    options_group = parser.add_argument_group('Options')
    help_msg = 'Value of the hyperparameter a for SeqDEFT. If not provided, '
    help_msg += 'grid search with cross-validated likelihood will be used'
    options_group.add_argument('-a', '--a_value', default=None,
                               help=help_msg)
    help_msg = 'P for the delta_P operator to build prior (2)'
    options_group.add_argument('-P', '--delta_P', default=2, help=help_msg)

    options_group.add_argument('-A', '--alphabet_type', default='dna',
                               help='Alphabet type [dna, rna, protein, custom] (dna)')
    options_group.add_argument('-n', '--n_alleles', default=None,
                               help='Number of alleles to use for custom alphabet')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True,
                              help='Output file')

    # Parse arguments
    parsed_args = parser.parse_args()
    counts_fpath = parsed_args.counts
    alphabet_type = parsed_args.alphabet_type
    n_alleles = parsed_args.n_alleles
    P = parsed_args.delta_P
    a = parsed_args.a_value
    out_fpath = parsed_args.output
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(counts_fpath, index_col=0)
    length = len(data.index[0])
    print(length)

    # Load annotation data
    seqdeft = SeqDEFT(length, P, alphabet_type=alphabet_type,
                      n_alleles=n_alleles, log=log)
    seqdeft.fit(data['counts'], resolution=0.1, max_a_max=1e6, num_a=50)
    seqdeft.plot_summary(fname=out_fpath)
    seqdeft.write_output(out_fpath)
    
    log.finish()


if __name__ == '__main__':
    main()
