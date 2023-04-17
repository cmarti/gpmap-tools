#!/usr/bin/env python
import unittest
import numpy as np
import pandas as pd

from os.path import join
from itertools import combinations
from timeit import timeit
from subprocess import check_call

from scipy.stats.mstats_basic import pearsonr
from scipy.special._basic import comb

from gpmap.src.inference import VCregression
from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from gpmap.src.linop import (LaplacianOperator, ProjectionOperator,
                             VjProjectionOperator)
from gpmap.src.utils import get_sparse_diag_matrix
from gpmap.src.space import SequenceSpace


class VCTests(unittest.TestCase):
    def test_get_gt_to_data_matrix(self):
        vc = VCregression()
        vc.init(3, 2)

        m = vc.get_gt_to_data_matrix().todense()
        assert(np.all(m == np.eye(8)))
        
        m = vc.get_gt_to_data_matrix(idx=np.array([0, 1, 2, 3])).todense()
        assert(np.all(m[:4, :] == np.eye(4)))
        assert(np.all(m[4:, :] == 0))
        assert(m.shape == (8, 4))
    
    def test_simulate_vc(self):
        np.random.seed(1)
        sigma = 0.1
        l, a = 4, 4
        vc = VCregression()
        vc.init(l, a)
        
        # Test functioning and size
        lambdas = np.array([0, 200, 20, 2, 0.2])
        data = vc.simulate(lambdas, sigma)
        f = data['y_true'].values
        assert(data.shape[0] == 256)
        
        # Test missing genotypes
        data = vc.simulate(lambdas, sigma, p_missing=0.1)
        assert(data.dropna().shape[0] < 256)
        
        # Test pure components
        W = ProjectionOperator(a, l)
        for k1 in range(l + 1):
            lambdas = np.zeros(l+1)
            lambdas[k1] = 1
            
            data = vc.simulate(lambdas)
            f = data['y_true'].values
        
            for k2 in range(l+1):
                W.set_lambdas(k=k2)
                f_k_rq = W.rayleigh_quotient(f)
                assert(np.allclose(f_k_rq, k1 == k2))
    
    def test_simulate_skewed_vc(self):  
        np.random.seed(1)
        l, a = 2, 2 
        vc = VCregression()
        
        # With p=1
        ps = 1 * np.ones((l, a))
        L = LaplacianOperator(a, l, ps=ps)
        vc.init(l, a, ps=ps)
        W = ProjectionOperator(L=L)
        for k1 in range(l + 1):
            lambdas = np.zeros(l+1)
            lambdas[k1] = 1
            
            data = vc.simulate(lambdas)
            f = data['y_true'].values
        
            for k2 in range(l+1):
                W.set_lambdas(k=k2)
                f_k_rq = W.rayleigh_quotient(f)
                assert(np.allclose(f_k_rq, k1 == k2))
        
        # with variable ps
        ps = np.random.dirichlet(np.ones(a), size=l) * a
        L = LaplacianOperator(a, l, ps=ps)
        vc.init(l, a, ps=ps)
        W = ProjectionOperator(L=L)
        
        for k1 in range(l + 1):
            lambdas = np.zeros(l+1)
            lambdas[k1] = 1
            
            data = vc.simulate(lambdas)
            f = data['y_true'].values
        
            for k2 in range(l+1):
                W.set_lambdas(k=k2)
                f_k_rq = W.rayleigh_quotient(f)
                assert(np.allclose(f_k_rq, k1 == k2))
    
    def test_calc_emp_dist_cov(self):
        np.random.seed(1)
        sigma, lambdas = 0.1, np.array([0, 200, 20, 2, 0.2])
        
        seq_length, n_alleles = 4, 4
        
        vc = VCregression()
        vc.init(seq_length, n_alleles)
        data = vc.simulate(lambdas, sigma)
        vc.set_data(X=data.index.values, y=data.y.values)
        rho, n = vc.calc_emp_dist_cov()
        
        # Ensure we get the expected number of pairs per distance category
        for d in range(seq_length + 1):
            total_genotypes = n_alleles ** seq_length
            d_combs = comb(seq_length, d)
            d_sites_genotypes = (n_alleles - 1) ** d
            n[d] = total_genotypes * d_combs * d_sites_genotypes
            
        # Ensure anticorrelated distances
        assert(rho[3] < 0)
        assert(rho[4] < 0)
    
    def test_vc_fit(self):
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        fpath = join(TEST_DATA_DIR, 'vc.data.csv')
        data = pd.read_csv(fpath, dtype={'seq': str}).set_index('seq')
        
        # Ensure MSE is within a small range
        vc = VCregression()
        vc.fit(data.index.values, data['y'], y_var=data['var'])
        sd1 = np.log2((vc.lambdas[1:]+1e-6) / (lambdas[1:]+1e-6)).std()
        assert(sd1 < 2)
        
        # Try with regularization and CV and 
        vc = VCregression(cross_validation=True)
        vc.fit(data.index, data['y'], y_var=data['var'])
        sd2 = np.log2((vc.lambdas[1:]+1e-6) / (lambdas[1:]+1e-6)).std()
        assert(sd2 < 1)
        assert(vc.beta > 0)
        
        vc.cv_loss_df.to_csv(join(TEST_DATA_DIR, 'vc.cv_loss.csv'))
        
        # Ensure regularization improves results
        assert(sd1 > sd2)
    
    def test_calculate_variance_components(self):
        vc = VCregression()
        vc.init(n_alleles=4, seq_length=8)
        lambdas0 = np.array([0, 10, 5, 2, 1, 0.5, 0.2, 0, 0])
        data = vc.simulate(lambdas=lambdas0)
        
        # Ensure kernel alignment and calculation of variance components is the same
        space = SequenceSpace(X=data.index.values, y=data.y.values)
        lambdas1 = space.calc_variance_components()
        vc.fit(X=data.index.values, y=data.y.values)
        lambdas2 = vc.lambdas
        assert(np.allclose(lambdas2, lambdas1))
    
    def test_vc_smn1(self):
        data_fpath = join(TEST_DATA_DIR, 'smn1.0.train.csv')
        out_fpath = join(TEST_DATA_DIR, 'smn1.0.out')
        bin_fpath = join(BIN_DIR, 'vc_regression.py')
        
        # Perform regularization
        cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-r']
        check_call(cmd)
        
    def test_vc_predict(self):
        lambdas = np.array([0, 200, 20, 2, 0.2, 0.02])
        fpath = join(TEST_DATA_DIR, 'vc.data.csv')
        data = pd.read_csv(fpath, dtype={'seq': str}).set_index('seq')
        
        # Using the a priori known variance components
        vc = VCregression()
        vc.set_data(X=data.index, y=data['y'], y_var=data['var'])
        vc.set_lambdas(lambdas)
        pred = vc.predict()
        
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.05)
        
        # Estimate posterior variances
        pred = vc.predict(calc_variance=True)
        assert('var' in pred.columns)
        assert(np.all(pred['var'] > 0))
        
        # Capture error with incomplete input
        vc = VCregression()
        vc.set_data(X=data.index, y=data['y'], y_var=data['var'])
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
        vc = VCregression()
        vc.fit(filtered.index, filtered['y'], y_var=filtered['var'], 
               cross_validation=True)
        pred = vc.predict().sort_index()
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.3)
    
    def test_vc_predict_max_L_size(self):
        np.random.seed(0)
        fpath = join(TEST_DATA_DIR, 'vc.data.csv')
        data = pd.read_csv(fpath, dtype={'seq': str}).set_index('seq')
        
        filtered = data.loc[np.random.choice(data.index, size=950), :]
        vc = VCregression(max_L_size=100, cross_validation=True)
        vc.fit(filtered.index, filtered['y'], y_var=filtered['var'])
        pred = vc.predict().sort_index()
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.3)
    
    def test_skewed_VC(self):
        ps = np.array([[0.4, 0.6],
                       [0.3, 0.7]])
        vc = VCregression()
        vc.init(2, 2, ps=ps)
        
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
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'VCTests.test_calculate_variance_components']
    unittest.main()
