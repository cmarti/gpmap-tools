#!/usr/bin/env python
import sys
import unittest
import numpy as np

from itertools import product
from scipy.stats import pearsonr

from gpmap.inference import GaussianProcessRegressor
from gpmap.linop import ConnectednessKernel, VarianceComponentKernel


class GPTests(unittest.TestCase):
    def test_prior_sample(self):
        seq_length, a = 5, 4
        
        # Sample from fixed rho kernel
        kernel = ConnectednessKernel(a, seq_length, rho=0.5)
        model = GaussianProcessRegressor(kernel)
        y = model.sample_prior()
        assert y.shape[0] == a**seq_length
        
        # Sample from variable rho kernel
        kernel = ConnectednessKernel(
            a, seq_length, rho=np.array([0.2, 0.1, 0.8, 0.4, 0.5])
        )
        model = GaussianProcessRegressor(kernel)
        y = model.sample_prior()
        assert y.shape[0] == a**seq_length
        
        # Sample from VC kernel
        kernel = VarianceComponentKernel(
            a, seq_length, lambdas=2.0 ** -np.arange(seq_length + 1)
        )
        model = GaussianProcessRegressor(kernel)
        y = model.sample_prior()
        assert y.shape[0] == a**seq_length
        
    def xtest_gp_fit(self):
        alleles = ['A', 'C', 'G', 'T']
        seq_length, a = 5, len(alleles)
        X = np.array(["".join(c) for c in product(alleles, repeat=seq_length)])
        
        # Create model and simulate
        model = GaussianProcessRegressor(VarianceComponentKernel)
        model.define_kernel(a, seq_length)
        log_lambdas = 3 - np.arange(seq_length + 1)
        model.K.set_params(log_lambdas)
        y_true = model.sample()
        y = np.random.normal(y_true, scale=0.5)
        y_var = np.full(y.shape[0], fill_value=0.25)
        
        # Create model and infer
        model = GaussianProcessRegressor(VarianceComponentKernel)
        params = model.fit(X, y, y_var=y_var)
        r = pearsonr(log_lambdas, params)[0]
        assert(r > 0.9)

        
if __name__ == '__main__':
    sys.argv = ['', 'GPTests']
    unittest.main()
