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

        
if __name__ == '__main__':
    sys.argv = ['', 'GPTests']
    unittest.main()
