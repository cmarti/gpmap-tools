#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.src.utils import LogTrack
from gpmap.src.inference import SeqDEFT
from gpmap.src.plot import plot_SeqDEFT_summary, savefig

        
def main():
    description = 'Runs Sequence Density Estimation using Field Theory (SeqDEFT)'
    description += ' from a table with the number of time each sequence was'
    description += ' observed e.g. the number of times each splice donor or'
    description += ' acceptor is found in the human genome'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('counts', help='CSV table with sequence counts')

    options_group = parser.add_argument_group('SeqDEFT options')
    help_msg = 'Value of the hyperparameter a for SeqDEFT. If not provided, '
    help_msg += 'grid search with cross-validated likelihood will be used'
    options_group.add_argument('-a', '--a_value', default=None, type=float,
                               help=help_msg)
    options_group.add_argument('-k', '--num_a', default=20, type=int,
                               help='Number of a values to test (20)')
    help_msg = 'P for the Delta(P) operator to build prior (2)'
    options_group.add_argument('-P', '--P_delta', default=2, help=help_msg,
                               type=int)
    options_group.add_argument('--get_a_values', default=False, action='store_true',
                               help='Only calculate the series of a values for CV')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True,
                              help='Output file')

    # Parse arguments
    parsed_args = parser.parse_args()
    counts_fpath = parsed_args.counts

    P = parsed_args.P_delta
    a_value = parsed_args.a_value
    num_a = parsed_args.num_a
    get_a_values = parsed_args.get_a_values
    
    out_fpath = parsed_args.output
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(counts_fpath, index_col=0)

    # Load annotation data
    seqdeft = SeqDEFT(P, a=a_value, num_reg=num_a)
    if get_a_values:
        seqdeft.init(genotypes=data.index.values)
        seqdeft.set_data(X=data.index.values, y=data['counts'].values)
        log.write('Calculating only a values')
        with open(out_fpath, 'w') as fhand:
            for a_value in seqdeft.get_a_values():
                fhand.write('{}\n'.format(a_value))
    else:
        result = seqdeft.fit(X=data.index.values, y=data['counts'].values)
        result.to_csv(out_fpath)
        
        fig = plot_SeqDEFT_summary(seqdeft.logL_df, result)
        savefig(fig, out_fpath)
    
    log.finish()


if __name__ == '__main__':
    main()
