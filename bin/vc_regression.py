#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd

from gpmap.src.utils import LogTrack
from gpmap.src.inference import VCregression

        
def main():
    description = 'Runs Variance Component regression using data from'
    description += ' quantitative phenotypes associated to their corresponding'
    description += ' sequences. If provided, the variance of the estimated '
    description += ' quantitative measure can be incorporated into the model'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('data', help='CSV table with genotype-phenotype data')

    options_group = parser.add_argument_group('VC options')
    options_group.add_argument('--lambdas', default=None,
                               help='File containing known lambdas to use for prediction directly')
    options_group.add_argument('-r', '--regularize',
                               action='store_true', default=False,
                               help='Regularize variance components to exponential decay through CV')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True, help='Output file')
    output_group.add_argument('-p', '--pred',
                              help='File containing sequencse for predicting genotype')
    output_group.add_argument('--var', action='store_true',
                              help='Report also posterior variance for the predicted values')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data
    
    regularize = parsed_args.regularize
    lambdas_fpath = parsed_args.lambdas

    pred_fpath = parsed_args.pred
    out_fpath = parsed_args.output
    estimate_variance = parsed_args.var
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    data = pd.read_csv(data_fpath, dtype=str)
    data = data.set_index(data.columns[0]).astype(float)
    
    # Get processed data
    X = data.index.values
    y = data.values[:, 0]
    y_var = data.values[:, 1] if data.shape[1] > 1 else None 
    Xpred = [line.strip().strip('"')
             for line in open(pred_fpath)] if pred_fpath is not None else None
    
    # Create VC model, fit and predict
    vc = VCregression()
    
    if lambdas_fpath is None:
        log.write('Estimate variance components through kernel alignment')
        vc.fit(X=X, y=y, variance=y_var, cross_validation=regularize)
        lambdas = vc.lambdas
    else:
        log.write('Load variance components from {}'.format(lambdas_fpath))
        lambdas = np.array([float(x.strip()) for x in open(lambdas_fpath)])
        
    log.write('Obtain phenotypic predictions')
    result = vc.predict(X=X, y=y, variance=y_var,
                        Xpred=Xpred, estimate_variance=estimate_variance,
                        lambdas=lambdas)
    result.to_csv(out_fpath)
    
    log.finish()


if __name__ == '__main__':
    main()
