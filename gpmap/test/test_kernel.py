#!/usr/bin/env python
import unittest

import numpy as np

from gpmap.src.inference import VCregression
from gpmap.src.kernel import (VarianceComponentKernel, SequenceKernel,
                              KernelAligner, FullKernelAligner)
from scipy.special._basic import comb


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
        
    def test_mse(self):
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        kernel = VarianceComponentKernel(2, 2, use_p=False)
        aligner = FullKernelAligner(kernel=kernel)
        
        # Constant function
        y = np.array([1, 1, 1, 1])
        lambdas = np.array([1, 0, 0])
        aligner.set_data(X, y, alleles=['A', 'B'])        
        mse = aligner.mse(lambdas=lambdas)
        mse_grad = aligner.mse_grad(lambdas=lambdas)
        assert(mse == 0)
        assert(np.allclose(mse_grad, 0))
        
        # Additive function
        y = np.array([-2, 0, 0, 2])
        lambdas = np.array([0, 1, 0])
        aligner.set_data(X, y, alleles=['A', 'B'])        
        mse = aligner.mse(lambdas=lambdas)
        mse_grad = aligner.mse_grad(lambdas=lambdas)
        assert(mse == 2)
        assert(np.allclose(mse_grad, 0))
        
        # Pairwise function
        y = np.array([1, -1, -1, 1])
        lambdas = np.array([0, 0, 1])
        aligner.set_data(X, y, alleles=['A', 'B'])        
        mse = aligner.mse(lambdas=lambdas)
        mse_grad = aligner.mse_grad(lambdas=lambdas)
        assert(mse == 0)
        assert(np.allclose(mse_grad, 0))
        
        # Additive function with non zero gradient
        y = np.array([-1, 0, 0, 1])
        lambdas = np.array([0, 0.75, 0])
        aligner.set_data(X, y, alleles=['A', 'B'])        
        mse = aligner.mse(lambdas=lambdas)
        mse_grad = aligner.mse_grad(lambdas=lambdas)
        assert(mse == 0.625)
        assert(np.allclose(mse_grad, [0, 12, 0]))
    
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
        cov = [5.38809419, 3.10647274, 1.47573831,
               0.39676481, -0.30354594, -0.70018283]
        n = [819, 9834, 58980, 176754, 265160, 159214]
        aligner.set_data(cov, n)
        lambdas = aligner.fit()
        assert(np.all(np.isnan(lambdas) == False))
    
    def test_full_kernel_alignment_small(self):
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        alleles = ['A', 'B']
        
        kernel = VarianceComponentKernel(2, 2)
        aligner = FullKernelAligner(kernel=kernel)
        
        # Constant model
        y = np.array([1, 1, 1, 1])
        aligner.set_data(X=X, y=y, alleles=alleles)
        lambdas_star = aligner.fit()
        assert(np.allclose(lambdas_star, [1, 0, 0], atol=0.01))
        
        # Additive model
        y = np.array([-2, 0, 0, 2])
        aligner.set_data(X=X, y=y, alleles=alleles)
        lambdas_star = aligner.fit()
        assert(np.allclose(lambdas_star, [0, 1, 0], atol=0.01))
        
        # Pairwise model
        y = np.array([1, -1, -1, 1])
        aligner.set_data(X=X, y=y, alleles=alleles)
        lambdas_star = aligner.fit()
        assert(np.allclose(lambdas_star, [0, 0, 1], atol=0.01))
        
    def test_full_kernel_alignment_medium(self):
        np.random.seed(0)
        l, a = 4, 4
        lambdas = np.zeros(l+1)
        lambdas[1] = 1
        
        vc = VCregression()
        vc.init(l, a)
        data = vc.simulate(lambdas)
        X, y = data.index.values, data.y.values
        alleles = np.unique(vc.alphabet)
        
        kernel = VarianceComponentKernel(l, a)
        aligner = FullKernelAligner(kernel=kernel)
        
        aligner.set_data(X=X, y=y, alleles=alleles)
        lambdas_star = aligner.fit()
        assert(np.allclose(lambdas_star / lambdas_star[1], lambdas))
        
        # With known noise
        y_var = 0.05 * np.ones(y.shape)
        y = np.random.normal(y, y_var)
        aligner.set_data(X=X, y=y, y_var=y_var, alleles=alleles)
        lambdas_star = aligner.fit()
        assert(np.allclose(lambdas_star / lambdas_star[1], lambdas))
    
    def test_NK_lambdas(self):
        l, a = 7, 4
        P = 3
        kernel = VarianceComponentKernel(l, a)
        lambdas = np.array([a**(-P) * comb(l-k, l-P) for k in range(l+1)])
        # lambdas = np.array([a**(-P) * comb(P, k) for k in range(l+1)])
        print(lambdas)
        print(kernel.W_kd.dot(lambdas))
        cov = np.array([comb(l-d, P) for d in range(l+1)])
        print(cov)
        
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelTest.test_NK_lambdas']
    unittest.main()
