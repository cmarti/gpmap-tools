#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.src.utils import read_split_data, calc_r2_values


def main():
    description = 'Calculates R2 values of a set of predicted genotype-phenotype'
    description += ' data subsets'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('data', help='CSV table with genotype-phenotype data')
    input_group.add_argument('-p', '--prefix', required=True,
                             help='Files prefix')
    input_group.add_argument('-s', '--suffix', default=None,
                             help='Files suffix')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, help='Output file')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    prefix = parsed_args.prefix
    suffix = parsed_args.suffix
    out_fpath = parsed_args.output
    
    # Load data
    data = pd.read_csv(data_fpath, index_col=0)
    splits = read_split_data(prefix, suffix=suffix)
    r2 = calc_r2_values(splits, data)
    r2.to_csv(out_fpath, index=False)


if __name__ == '__main__':
    main()
