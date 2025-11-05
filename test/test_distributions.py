#!/usr/bin/env python
import unittest
import numpy as np

from scipy.sparse.linalg import aslinearoperator
from gpmap.distributions import Gaussian
from gpmap.linop import RhoProjectionOperator, DiagonalOperator, ProjectionOperator


class DistributionsTests(unittest.TestCase):
    def setUp(self):
        n_alleles = 2
        seq_length = 3
        n = n_alleles ** seq_length
        lambdas = np.array([0, 1e1, 1e0, 1e-1])
        
        self.mean = np.zeros(n)
        self.K = ProjectionOperator(n_alleles, seq_length, lambdas=lambdas)
        self.D = DiagonalOperator(diag=0.2 * np.ones(n))
        self.Sigma = self.K.todense()
        self.corr = self.Sigma / self.Sigma[0, 0]
        self.Sigma2 = (self.K + self.D) @ np.eye(n)
        self.corr2 = self.Sigma2 / self.Sigma2[0, 0]
        # self.C = self.K.inv()
        return super().setUp()
    
    def test_gaussian_sample(self):
        mean = np.zeros(3)
        K = np.array([[1, 0.5, 0.25],
                      [0.5, 1, 0.5],
                      [0.25, 0.5, 1]])
        
        # Direct sampling
        gaussian = Gaussian(mean=mean, K=K)
        x = gaussian.sample(n_samples=10000)
        corr = np.corrcoef(x)
        print(corr)
        
        # CG sampling
        gaussian = Gaussian(mean=mean, K=aslinearoperator(K))
        x = gaussian.sample(n_samples=10000)
        corr = np.corrcoef(x)
        print(corr)
        
    
    def test_gaussian_sample_dense(self):
        gaussian = Gaussian(mean=self.mean, K=self.Sigma)
        x = gaussian.sample(n_samples=10000)
        
        corr = np.corrcoef(x)
        rmse = np.sqrt(np.mean((corr - self.corr) ** 2))
        assert(rmse < 1e-1)
    
    def test_gaussian_sample_linop_chol(self):
        gaussian = Gaussian(mean=self.mean, K=self.K)
        x = gaussian.sample(n_samples=10000)
        
        corr = np.corrcoef(x)
        rmse = np.sqrt(np.mean((corr - self.corr) ** 2))
        assert(rmse < 1e-1)
    
    def test_gaussian_sample_linop_cg_cov(self):
        gaussian = Gaussian(mean=self.mean, K=self.K + self.D)
        x = gaussian.sample(n_samples=10000)
        
        corr = np.corrcoef(x)
        rmse = np.sqrt(np.mean((corr - self.corr2) ** 2))
        print(corr[0, :])
        print(self.corr2[0, :])
        assert(rmse < 1e-1)
        
    def test_gaussian_sample_linop_cg_precision(self):
        C = self.K.inv() + self.D.inv()
        gaussian = Gaussian(mean=self.mean, C=C)
        x = gaussian.sample(n_samples=10000)
        
        corr = np.corrcoef(x)
        print(corr[0, :])
        print(self.corr2[0, :])
        rmse = np.sqrt(np.mean((corr - self.corr2) ** 2))
        print(rmse)
        assert(rmse < 1e-1)
        
    def test_gaussian_sample_cov_cholesky(self):
        gaussian = Gaussian(mean=self.mean, K=self.K)
        x = gaussian.sample(n_samples=10000)
        
        c1 = np.corrcoef(x)
        c2 = self.K.todense() / self.K.d
        rmse = np.sqrt(np.mean((c1 - c2) ** 2))
        assert(rmse < 1e-1)
    

if __name__ == "__main__":
    import sys

    sys.argv = ["", "LinOpsTests"]
    unittest.main()
