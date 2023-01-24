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


class VCTests(unittest.TestCase):
    def test_laplacian(self):
        L = LaplacianOperator(2, 2)
        L.calc_L()
        v = np.array([1, 2, 3, 0.])
        
        print(L.dot0(v))
        print(L.dot1(v))
        
        L = LaplacianOperator(4, 7)
        L.calc_L()
        v = np.random.normal(size=L.shape[0])
        
        print(timeit(lambda: L.dot0(v), number=10))
        print(timeit(lambda : L.dot1(v), number=10))
        
        assert(np.allclose(L.dot0(v), L.dot1(v)))
        
    def test_get_gt_to_data_matrix(self):
        vc = VCregression()
        vc.init(3, 2)

        m = vc.get_gt_to_data_matrix().todense()
        assert(np.all(m == np.eye(8)))
        
        m = vc.get_gt_to_data_matrix(idx=np.array([0, 1, 2, 3])).todense()
        assert(np.all(m[:4, :] == np.eye(4)))
        assert(np.all(m[4:, :] == 0))
        assert(m.shape == (8, 4))
    
    def test_calc_polynomial_coeffs(self):
        vc = VCregression()
        vc.init(2, 2)
        lambdas = np.ones(vc.seq_length + 1)
        
        # Ensure that if all eigenvalues are 1 then we would end up with I
        vc.calc_polynomial_coeffs()
        assert(np.allclose(vc.B.sum(1), [1, 0, 0]))

        # The same using the method        
        bs = vc.get_polynomial_coeffs(lambdas=lambdas)
        assert(np.allclose(bs, [1, 0, 0]))
        
    def test_calc_polynomial_coeffs_analytical(self):
        for l in range(3, 9):
            vc = VCregression()
            vc.init(l, 4)
            y = np.random.normal(size=vc.n_genotypes)
            
            b1 = vc.calc_polynomial_coeffs()
            p1 = vc.project(y, k=l-1)
            b2 = vc.calc_polynomial_coeffs(numeric=True)
            p2 = vc.project(y, k=l-1)
            
            # Test coefficients of the polynomials
            assert(np.allclose(b1, b2))
            
            # Test that projections are also equal
            assert(np.allclose(p1, p2))
    
    def test_projection(self):
        vc = VCregression()
        vc.init(2, 2)
        
        # Purely additive function
        y = np.array([-1.5, -0.5, 0.5, 1.5])
        assert(np.allclose(vc.project(y, k=2), 0))
        assert(np.allclose(vc.project(y, k=1), y))
        assert(np.allclose(vc.project(y, k=0), 0))
        
        # Non-zero orthogonal projections
        y = np.array([-1.5, -0.5, 0.5, 4])
        y0 = vc.project(y, k=0)
        y1 = vc.project(y, k=1)
        y2 = vc.project(y, k=2) 
        
        assert(not np.allclose(y0, 0))
        assert(not np.allclose(y1, y))
        assert(not np.allclose(y2, 0))
        
        assert(np.allclose(y0.T.dot(y1), 0))
        assert(np.allclose(y0.T.dot(y2), 0))
        assert(np.allclose(y1.T.dot(y2), 0))
    
    def test_simulate_vc(self):
        np.random.seed(1)
        sigma = 0.1
        lambdas = np.array([0, 200, 20, 2, 0.2])
        
        vc = VCregression()
        vc.init(4, 4)
        
        data = vc.simulate(lambdas, sigma)
        assert(data.shape[0] == 256)
        
        data = vc.simulate(lambdas, sigma, p_missing=0.1)
        assert(data.dropna().shape[0] < 256)
    
    def test_compute_empirical_rho(self):
        np.random.seed(1)
        sigma, lambdas = 0.1, np.array([0, 200, 20, 2, 0.2])
        
        seq_length, n_alleles = 4, 4
        
        vc = VCregression()
        vc.init(seq_length, n_alleles)
        data = vc.simulate(lambdas, sigma)
        
        obs_idx = vc.get_obs_idx(data.index)
        rho, n = vc.compute_empirical_rho(obs_idx, data['y'])
        
        # Ensure we get the expected number of pairs per distance category
        for d in range(seq_length + 1):
            total_genotypes = n_alleles ** seq_length
            d_combs = comb(seq_length, d)
            d_sites_genotypes = (n_alleles - 1) ** d
            n[d] = total_genotypes * d_combs * d_sites_genotypes
            
        # Ensure anticorrelated distances
        assert(rho[3] < 0)
        assert(rho[4] < 0)
    
    def test_kernel_alignment(self):
        np.random.seed(1)
        seq_length, n_alleles = 5, 4
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        
        vc = VCregression()
        vc.init(seq_length, n_alleles)
        data = vc.simulate(lambdas)
        vc.set_data(X=data.index.values, y=data.y.values)
        rho, n = vc.compute_empirical_rho(vc.obs_idx, vc.y)
        
        aligner = KernelAligner(seq_length, n_alleles)
        aligner.set_data(rho, n)
        lambdas_star = aligner.fit()
        pred = aligner.predict(lambdas_star)
        assert(np.allclose(rho, pred))
        assert(np.allclose(lambdas, lambdas_star, rtol=0.5))
    
    def test_vc_fit(self):
        lambdas = np.array([0, 200, 20, 2, 0.2, 0.02])
        fpath = join(TEST_DATA_DIR, 'vc.data.csv')
        data = pd.read_csv(fpath, dtype={'seq': str}).set_index('seq')
        
        # Ensure MSE is within a small range
        vc = VCregression()
        vc.fit(data.index.values, data['y'], variance=data['var'],
               cross_validation=False)
        sd1 = np.log2((vc.lambdas[1:]+1e-6) / (lambdas[1:]+1e-6)).std()
        assert(sd1 < 2)
        
        # Try with regularization and CV and 
        vc = VCregression()
        vc.fit(data.index, data['y'], variance=data['var'],
               cross_validation=True)
        sd2 = np.log2((vc.lambdas[1:]+1e-6) / (lambdas[1:]+1e-6)).std()
        assert(sd2 < 1)
        
        # Ensure regularization improves results
        assert(sd1 > sd2)
    
    def test_vc_fit_bin(self):
        data_fpath = join(TEST_DATA_DIR, 'vc.data.csv')
        lambdas_fpath = join(TEST_DATA_DIR, 'vc.lambdas.csv')
        xpred_fpath = join(TEST_DATA_DIR, 'vc.xpred.txt')
        out_fpath = join(TEST_DATA_DIR, 'seqdeft.output.csv')
        bin_fpath = join(BIN_DIR, 'vc_regression.py')
        
        # Direct kernel alignment
        cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath]
        check_call(cmd)
        
        # Perform regularization
        cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-r']
        check_call(cmd)
        
        # With known lambdas
        cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-r',
               '--lambdas', lambdas_fpath]
        check_call(cmd)
        
        # Predict few sequences and their variances
        cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-r',
               '--var', '-p', xpred_fpath]
        check_call(cmd)
    
    def test_vc_predict(self):
        lambdas = np.array([0, 200, 20, 2, 0.2, 0.02])
        fpath = join(TEST_DATA_DIR, 'vc.data.csv')
        data = pd.read_csv(fpath, dtype={'seq': str}).set_index('seq')
        
        # Using the a priori known variance components
        vc = VCregression()
        vc.set_data(X=data.index, y=data['y'], variance=data['var'])
        vc.set_lambdas(lambdas)
        pred = vc.predict()
        
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.05)
        
        # Estimate posterior variances
        pred = vc.predict(estimate_variance=True)
        assert('var' in pred.columns)
        assert(np.all(pred['var'] > 0))
        
        # Capture error with incomplete input
        vc = VCregression()
        vc.set_data(X=data.index, y=data['y'], variance=data['var'])
        try:
            pred = vc.predict()
            self.fail()
        except ValueError:
            pass
    
    def test_vc_predict_from_incomplete_data(self):
        np.random.seed(0)
        fpath = join(TEST_DATA_DIR, 'vc.data.csv')
        data = pd.read_csv(fpath, dtype={'seq': str}).set_index('seq')
        
        filtered = data.loc[np.random.choice(data.index, size=950), :]
        vc = VCregression(save_memory=False)
        vc.fit(filtered.index, filtered['y'], variance=filtered['var'], 
               cross_validation=True)
        pred = vc.predict().sort_index()
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.3)
    
    def test_vc_predict_save_memory(self):
        np.random.seed(0)
        fpath = join(TEST_DATA_DIR, 'vc.data.csv')
        data = pd.read_csv(fpath, dtype={'seq': str}).set_index('seq')
        
        filtered = data.loc[np.random.choice(data.index, size=950), :]
        vc = VCregression(save_memory=True)
        vc.fit(filtered.index, filtered['y'], variance=filtered['var'], 
               cross_validation=True)
        pred = vc.predict().sort_index()
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.3)
    
    def test_skewed_VC(self):
        ps = np.array([[0.4, 0.6],
                       [0.3, 0.7]])
        sk_vc = VCregression()
        sk_vc.init(2, 2, ps=ps)
        
        vc = VCregression()
        vc.init(2, 2)
        
        # Ensure that we maintain the right eigenvalues
        lambdas = np.linalg.eig(sk_vc.M.todense())[0]
        assert(np.allclose(lambdas, [2, 1, 0, 1]))
        
        # Ensure that stationary frequencies add up to 1
        assert(sk_vc.D_pi.data.sum() == 1)
        
        # Test projection into the constant subspace
        f = np.random.normal(0, 1, size=sk_vc.n_genotypes)
        sk_f_0 = sk_vc.project(f, k=0)
        assert(np.allclose(sk_f_0, sk_f_0[0]))

        # Test simulation and projection consistency
        f = sk_vc.simulate(lambdas=[10, 5, 1])['y_true'].values
        f_0 = sk_vc.project(f, k=0)
        f_1 = sk_vc.project(f, k=1)
        f_2 = sk_vc.project(f, k=2)
        assert(np.allclose(f, f_0 + f_1 + f_2))
        
        # Ensure D_pi orthogonality of projections
        assert(np.allclose(sk_vc.project(f_0, k=1), 0))
        assert(np.allclose(sk_vc.project(f_0, k=2), 0))
        assert(np.allclose(sk_vc.project(f_1, k=0), 0))
        assert(np.allclose(sk_vc.project(f_1, k=2), 0))
        assert(np.allclose(sk_vc.project(f_2, k=0), 0))
        assert(np.allclose(sk_vc.project(f_2, k=1), 0))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'VCTests.test_kernel_alignment']
    unittest.main()
