#!/usr/bin/env python
import unittest

import numpy as np

from gpmap.src.inference import VCregression
from gpmap.src.kernel import (VarianceComponentKernel, SequenceKernel,
                              KernelAligner)


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
        log_p = np.log(0.5) * np.ones((2, 2))
        cov = kernel(lambdas=lambdas, log_p=log_p)
        assert(np.allclose(cov, 1))
        
        # Test additive covariance
        lambdas = np.array([0, 1, 0])
        cov = kernel(lambdas=lambdas, log_p=log_p)
        add = np.array([[ 2.,  0.,  0., -2.],
                        [ 0.,  2., -2.,  0.],
                        [ 0., -2.,  2.,  0.],
                        [-2.,  0.,  0.,  2.]])
        assert(np.allclose(cov, add))
        
        # Test 2nd order covariance
        lambdas = np.array([0, 0, 1])
        cov = kernel(lambdas=lambdas, log_p=log_p)
        add = np.array([[ 1., -1., -1.,  1.],
                        [-1.,  1.,  1., -1.],
                        [-1.,  1.,  1., -1.],
                        [ 1., -1., -1.,  1.]])
        assert(np.allclose(cov, add))
    
    def test_kernel_alignment(self):
        np.random.seed(1)
        seq_length, n_alleles = 5, 4
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        
        vc = VCregression()
        vc.init(seq_length, n_alleles)
        aligner = KernelAligner(seq_length, n_alleles)
        
        data = vc.simulate(lambdas)
        vc.set_data(X=data.index.values, y=data.y.values)
        cov, n = vc.calc_emp_dist_cov()
        aligner.set_data(cov, n)
        
        # Align with beta = 0
        lambdas_star = aligner.fit()
        pred = aligner.predict(lambdas_star)
        print(cov - pred)
        assert(np.allclose(cov, pred, rtol=0.01))
        assert(np.allclose(lambdas, lambdas_star, rtol=0.5))
        
        # Align with beta > 0
        aligner.set_beta(0.01)
        lambdas_star = aligner.fit()
        pred = aligner.predict(lambdas_star)
        assert(np.allclose(cov, pred))
        assert(np.allclose(lambdas, lambdas_star, rtol=0.5))
        
        # Problematic case
        aligner.set_beta(0)
        cov = [ 5.38809419, 3.10647274, 1.47573831,
               0.39676481, -0.30354594, -0.70018283]
        n = [819, 9834, 58980, 176754, 265160, 159214]
        aligner.set_data(cov, n)
        lambdas = aligner.fit()
        print(lambdas)
        assert(np.all(np.isnan(lambdas) == False)) 
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelTest']
    unittest.main()
