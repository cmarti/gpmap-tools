#!/usr/bin/env python
import sys
import unittest
import numpy as np

from os.path import join
from subprocess import check_call

from scipy.stats.mstats_basic import pearsonr
from scipy.special._basic import comb

from gpmap.src.inference import VCregression, GaussianProcessRegressor
from gpmap.src.settings import BIN_DIR
from gpmap.src.linop import LaplacianOperator, ProjectionOperator,\
    ConnectednessKernelOperator, VarianceComponentKernelOperator
from gpmap.src.space import SequenceSpace
from tempfile import NamedTemporaryFile
from itertools import product


class GPTests(unittest.TestCase):
    def test_prior_sample(self):
        l, a = 5, 4
        
        # Sample from fixed rho kernel
        kernel = ConnectednessKernelOperator(a, l)
        kernel.set_rho(0.5)
        model = GaussianProcessRegressor(kernel=kernel)
        y = model.sample()
        assert(y.shape[0] == a**l)
        
        # Sample from variable rho kernel
        kernel = ConnectednessKernelOperator(a, l)
        kernel.set_rho([0.2, 0.1, 0.8, 0.4, 0.5])
        model = GaussianProcessRegressor(kernel=kernel)
        y = model.sample()
        assert(y.shape[0] == a**l)
        
        # Sample from VC kernel
        kernel = VarianceComponentKernelOperator(a, l)
        kernel.set_lambdas(2.**-np.arange(l+1))
        model = GaussianProcessRegressor(kernel=kernel)
        y = model.sample()
        assert(y.shape[0] == a**l)
        
    def test_gp_fit(self):
        alleles = ['A', 'C', 'G', 'T']
        l, a = 5, len(alleles)
        X = np.array([''.join(c) for c in product(alleles, repeat=l)])
        
        # Create model and simulate
        model = GaussianProcessRegressor(VarianceComponentKernelOperator)
        model.define_kernel(a, l)
        log_lambdas = 3-np.arange(l+1)
        model.K.set_params(log_lambdas)
        y_true = model.sample()
        print(y_true)
        y = np.random.normal(y_true, scale=0.5)
        y_var = np.full(y.shape[0], fill_value=0.25)
        
        # Create model and infer
        model = GaussianProcessRegressor(VarianceComponentKernelOperator)
        params = model.fit(X, y, y_var=y_var)
        r = pearsonr(log_lambdas, params)[0]
        assert(r > 0.9)

        
if __name__ == '__main__':
    sys.argv = ['', 'GPTests.test_gp_fit']
    unittest.main()
