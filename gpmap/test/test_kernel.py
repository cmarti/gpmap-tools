#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd
from scipy.stats.mstats_basic import pearsonr

from gpmap.src.inference import VCregression
from scipy.special._basic import comb
from timeit import timeit
from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from os.path import join
from subprocess import check_call
from gpmap.src.linop import LaplacianOperator, KernelAligner
from gpmap.src.kernel import VarianceComponentKernel, SequenceKernel


class KernelTest(unittest.TestCase):
    def test_sequence_kernel(self):
        kernel = SequenceKernel(2, 2)
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        kernel.set_data(X, alleles=['A', 'B'])
        
        # Test encoding
        onehot = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 1, 0, 1]])
        assert(np.allclose(kernel.x1, onehot))
        assert(np.allclose(kernel.x2, onehot))
        
        # Test distance
        d = np.array([[0, 1, 1, 2],
                      [1, 0, 2, 1],
                      [1, 2, 0, 1],
                      [2, 1, 1, 0]])
        hamming = kernel.calc_hamming_distance(kernel.x1, kernel.x2)
        assert(np.allclose(hamming, d))
    
    def test_vc_kernel(self):
        kernel = VarianceComponentKernel(2, 2)
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        kernel.set_data(X, alleles=['A', 'B'])
        
        # Test constant covariance
        lambdas = np.array([1, 0, 0])
        cov = kernel(lambdas=lambdas)
        assert(np.allclose(cov, 1))
        
        # Test additive covariance
        lambdas = np.array([0, 1, 0])
        cov = kernel(lambdas=lambdas)
        add = np.array([[ 2.,  0.,  0., -2.],
                        [ 0.,  2., -2.,  0.],
                        [ 0., -2.,  2.,  0.],
                        [-2.,  0.,  0.,  2.]])
        assert(np.allclose(cov, add))
        
        # Test 2nd order covariance
        lambdas = np.array([0, 0, 1])
        cov = kernel(lambdas=lambdas)
        add = np.array([[ 1., -1., -1.,  1.],
                        [-1.,  1.,  1., -1.],
                        [-1.,  1.,  1., -1.],
                        [ 1., -1., -1.,  1.]])
        assert(np.allclose(cov, add))
    
    def test_vc_kernel_ps(self):
        kernel = VarianceComponentKernel(2, 2, use_p=True)
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        kernel.set_data(X, alleles=['A', 'B'])
        
        # Test constant covariance
        lambdas = np.array([1, 0, 0])
        cov = kernel(lambdas=lambdas)
        assert(np.allclose(cov, 1))
        
        # Test additive covariance
        lambdas = np.array([0, 1, 0])
        cov = kernel(lambdas=lambdas)
        add = np.array([[ 2.,  0.,  0., -2.],
                        [ 0.,  2., -2.,  0.],
                        [ 0., -2.,  2.,  0.],
                        [-2.,  0.,  0.,  2.]])
        assert(np.allclose(cov, add))
        
        # Test 2nd order covariance
        lambdas = np.array([0, 0, 1])
        cov = kernel(lambdas=lambdas)
        add = np.array([[ 1., -1., -1.,  1.],
                        [-1.,  1.,  1., -1.],
                        [-1.,  1.,  1., -1.],
                        [ 1., -1., -1.,  1.]])
        assert(np.allclose(cov, add))
        
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelTest']
    unittest.main()
