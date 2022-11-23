#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.src.utils import (LogTrack, get_CV_splits,
                             generate_p_training_config,
                             get_training_p_data)
from tqdm import tqdm


def write_seqs(seqs, fpath):
    with open(fpath, 'w') as fhand:
        for seq in seqs:
            fhand.write('{}\n'.format(seq))

        
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

    cv_group = parser.add_argument_group('Cross-validation options')
    cv_group.add_argument('--cv', default=False, action='store_true',
                          help='Generate cross-validation datasets')
    cv_group.add_argument('-k', '--k_folds', default=10, type=int,
                          help='Number of folds for cross-validation')
    
    trainp_group = parser.add_argument_group('Training proportions options')
    trainp_group.add_argument('-r', '--nreps', default=3,
                              help='Number of replicates for each proportion')
    trainp_group.add_argument('-n', '--n_ps', default=10,
                              help='Number of different training proportions')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, help='Output file')
    output_group.add_argument('-p', '--prefix', required=True,
                              help='Prefix for sub-datasets output files')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    count_data = parsed_args.counts
    
    run_cv = parsed_args.cv
    nfolds = parsed_args.k_folds
    
    nreps = parsed_args.nreps
    n_ps = parsed_args.n_ps

    out_fpath = parsed_args.output
    out_prefix = parsed_args.prefix
    
    # Load counts data
    log = LogTrack()
    log.write('Loading data')
    data = pd.read_csv(data_fpath, dtype=str)
    data = data.set_index(data.columns[0])
    
    # Get processed data
    X = data.index.values
    y = data.values[:, 0]
    y_var = data.values[:, 1] if data.shape[1] > 1 else None 
    
    if run_cv:
        log.write('Generating {}-fold Cross-valiation sets'.format(nfolds))
        splits = get_CV_splits(X, y, y_var=y_var, 
                               nfolds=nfolds, count_data=count_data)
        for i, (train, test) in tqdm(enumerate(splits), total=nfolds):
            train_x, train_y = train[:2]
            test_x = test[0]

            train_data = {'y': train_y}
            if len(train) > 2 and train[2] is not None:
                train_data['y_var'] = train[2]            
            train_data = pd.DataFrame(train_data, index=train_x)
            fpath = '{}.{}.train.csv'.format(out_prefix, i)
            train_data.to_csv(fpath)
            
            fpath = '{}.{}.test.txt'.format(out_prefix, i)
            write_seqs(test_x, fpath)
    
    else:
        log.write('Generating training and test sets with variable proportions')
        config = generate_p_training_config(n_ps=n_ps, nreps=nreps)
        config.to_csv(out_fpath)
        
        for i, p in tqdm(zip(config['id'], config['p']), total=config.shape[0]):
            train, test = get_training_p_data(X, y, p, y_var=y_var,
                                              count_data=count_data)
            train_x, train_y = train[:2]
            test_x = test[0]

            train_data = {'y': train_y}
            if len(train) > 2 and train[2] is not None:
                train_data['y_var'] = train[2]
            train_data = pd.DataFrame(train_data, index=train_x)
            fpath = '{}.{}.train.csv'.format(out_prefix, i)
            train_data.to_csv(fpath)
            
            fpath = '{}.{}.test.txt'.format(out_prefix, i)
            write_seqs(test_x, fpath)
    
    log.finish()


if __name__ == '__main__':
    main()
