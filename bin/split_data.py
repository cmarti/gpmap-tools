#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd

from gpmap.src.utils import (LogTrack, get_CV_splits,
                             generate_p_training_config,
                             get_training_p_splits, write_split_data)


def main():
    description = 'Splits data into smaller datasets for model evaluation. '
    description += 'Two modes are available: generating regular Cross-validation'
    description += 'split datasets or training and test data for a range '
    description += ' of training proportions. It supports both quantitative '
    description += 'measurements or count based data.'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('data', help='CSV table with genotype-phenotype data')
    input_group.add_argument('--counts', default=False, action='store_true',
                             help='Data consists on counts')
    input_group.add_argument('--seed', default=None, type=int,
                             help='Random seed')

    cv_group = parser.add_argument_group('Cross-validation options')
    cv_group.add_argument('--cv', default=False, action='store_true',
                          help='Generate cross-validation datasets')
    cv_group.add_argument('-k', '--k_folds', default=10, type=int,
                          help='Number of folds for cross-validation (10)')
    
    trainp_group = parser.add_argument_group('Training proportions options')
    trainp_group.add_argument('-r', '--nreps', default=3, type=int,
                              help='Number of replicates for each proportion (3)')
    trainp_group.add_argument('-n', '--n_ps', default=10, type=int,
                              help='Number of different training proportions (10)')
    trainp_group.add_argument('-m', '--max_pred', default=None, type=int,
                              help='Max number of test sequences to generate')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, help='Output file')
    output_group.add_argument('-p', '--prefix', required=True,
                              help='Prefix for sub-datasets output files')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    count_data = parsed_args.counts
    seed = parsed_args.seed
    
    run_cv = parsed_args.cv
    nfolds = parsed_args.k_folds
    
    nreps = parsed_args.nreps
    n_ps = parsed_args.n_ps
    max_pred = parsed_args.max_pred

    out_fpath = parsed_args.output
    out_prefix = parsed_args.prefix
    
    # Load counts data
    log = LogTrack()
    log.write('Loading data')
    data = pd.read_csv(data_fpath, dtype=str)
    data = data.set_index(data.columns[0])
    
    if seed is not None:
        np.random.seed(seed)

    # Get processed data        
    X = data.index.values
    y = data.values[:, 0]
    y_var = data.values[:, 1] if data.shape[1] > 1 else None 
    
    if run_cv:
        log.write('Generating {}-fold Cross-valiation sets'.format(nfolds))
        splits = get_CV_splits(X, y, y_var=y_var, nfolds=nfolds,
                               count_data=count_data, max_pred=max_pred)
    else:
        log.write('Generating training and test sets with variable proportions')
        config = generate_p_training_config(n_ps=n_ps, nreps=nreps)
        config.to_csv(out_fpath)
        
        splits = get_training_p_splits(config, X, y, y_var=y_var, 
                                       count_data=count_data, max_pred=max_pred)
        
    write_split_data(out_prefix, splits)
    
    log.finish()


if __name__ == '__main__':
    main()
