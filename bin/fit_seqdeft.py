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
    options_group.add_argument('-P', '--delta_P', default=2, help=help_msg)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True,
                              help='Output file')

    # Parse arguments
    parsed_args = parser.parse_args()
    counts_fpath = parsed_args.counts

    P = parsed_args.delta_P
    a_value = parsed_args.a_value
    num_a = parsed_args.num_a
    
    out_fpath = parsed_args.output
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(counts_fpath, index_col=0)

    # Load annotation data
    seqdeft = SeqDEFT(P)
    result = seqdeft.fit(X=data.index.values, counts=data['counts'].values,
                         a_value=a_value, num_a=num_a)
    result.to_csv(out_fpath)
    
    fig = plot_SeqDEFT_summary(seqdeft.log_Ls, result)
    savefig(fig, out_fpath)
    
    log.finish()


if __name__ == '__main__':
    main()
