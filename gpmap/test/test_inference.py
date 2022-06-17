#!/usr/bin/env python
import unittest

import numpy as np
from scipy.stats.mstats_basic import pearsonr

from gpmap.src.inference import VCregression
from scipy.special._basic import comb


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
    
    def test_calc_L_powers_unique_entries_matrix(self):
        vc = VCregression()
        vc.init(5, 4)
        exp_mat = [[1, 15, 240, 4020, 69720, 1242600],
                   [0, -1, -28, -628, -13120, -266296],
                   [0,  0, 2,  78,  2168, 52680],
                   [0, 0, 0, -6, -288, -9240],
                   [0, 0, 0, 0,  24, 1320],
                   [0 ,0 ,0, 0, 0, -120]]
        assert(np.all(exp_mat == vc.L_powers_unique_entries))
    
    def test_calc_L_powers(self):
        vc = VCregression()
        vc.init(3, 2)
        
        np.random.seed(0)
        v = np.random.normal(size=8)
        L_powers = vc.calc_L_powers(v)
        assert(L_powers.shape == (8, 4))
    
    def test_simulate_vc(self):
        np.random.seed(1)
        sigma = 0.1
        lambdas = np.array([0, 200, 20, 2, 0.2])
        
        vc = VCregression()
        vc.init(4, 4)
        
        data = vc.simulate(lambdas, sigma)
        assert(data.shape[0] == 256)
        
        data = vc.simulate(lambdas, sigma, p_missing=0.1)
        assert(data.shape[0] < 256)
    
    def test_compute_empirical_rho(self):
        np.random.seed(1)
        sigma, lambdas = 0.1, np.array([0, 200, 20, 2, 0.2])
        
        seq_length, n_alleles = 4, 4
        
        vc = VCregression()
        vc.init(seq_length, n_alleles)
        data = vc.simulate(lambdas, sigma)
        
        obs_idx = vc.get_obs_idx(data.index)
        rho, n = vc.compute_empirical_rho(obs_idx, data['function'])
        
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
        np.random.seed(1)
        sigma, lambdas = 0.2, np.array([0, 200, 20, 2, 0.2, 0.02])
        loglambdas = np.log10(lambdas[1:])
        
        vc = VCregression()
        vc.init(5, 4)
        data = vc.simulate(lambdas, sigma, p_missing=0.1)
        
        # Ensure MSE is within a small range
        vc = VCregression()
        vc.fit(data.index, data['function'], variance=data['variance'],
               cross_validation=False)
        assert(np.mean((loglambdas - np.log10(vc.lambdas[1:] + 1e-2))**2) < 0.5)
        
        # Try with regularization and CV
        vc = VCregression()
        vc.fit(data.index, data['function'], variance=data['variance'],
               cross_validation=True)
        assert(np.mean((loglambdas - np.log10(vc.lambdas[1:]))**2) < 0.25)
    
    def test_vc_predict(self):
        np.random.seed(0)
        sigma, lambdas = 0.2, np.array([0, 200, 20, 2, 0.2, 0.02])
        
        vc = VCregression()
        vc.init(5, 4)
        data = vc.simulate(lambdas, sigma)
        real = vc.real_function
        
        # Estimating also the variance components
        vc = VCregression()
        vc.fit(data.index, data['function'], variance=data['variance'])
        pred = vc.predict()
        mse = np.mean((pred['function'] - real['function']) ** 2)
        rho = pearsonr(pred['function'], real['function'])[0]
        assert(rho > 0.95)
        assert(mse < 0.05)
        
        # Using the a priori known variance components
        vc = VCregression()
        pred = vc.predict(X=data.index, y=data['function'],
                          variance=data['variance'], lambdas=lambdas)
        mse = np.mean((pred['function'] - real['function']) ** 2)
        rho = pearsonr(pred['function'], real['function'])[0]
        assert(rho > 0.95)
        assert(mse < 0.05)
        
        # Estimate posterior variances
        vc = VCregression()
        pred = vc.predict(X=data.index, y=data['function'],
                          variance=data['variance'], lambdas=lambdas, 
                          estimate_variance=True)
        assert('variance' in pred.columns)
        assert(np.all(pred['variance'] > 0))
        
        # Capture error with incomplete input
        try:
            pred = vc.predict(X=data.index, y=data['function'],
                              variance=data['variance'])
            self.fail()
        except ValueError:
            pass
        
        # With incomplete genotype sampling data: larger MSE
        data = data.loc[np.random.choice(data.index, size=950), :]
        vc = VCregression()
        vc.fit(data.index, data['function'], variance=data['variance'])
        pred = vc.predict()
        mse = np.mean((pred['function'] - real['function']) ** 2)
        rho = pearsonr(pred['function'], real['function'])[0]
        assert(rho > 0.95)
        assert(mse < 0.3)
        
    def xtest_estimate_f_variance(self):
        np.random.seed(0)
        lambdas = [2.80160850e02, 1.11396023, 2.56638126e-01,
                   3.51032105e-02, 1.10928480e-02, 6.69724738e-03]

        gpmap = get_test_gpmap()        
        v = gpmap.estimate_f_variance(lambdas)
        print(v)
        
    def xtest_W_dot_precision(self):
        gpmap = VCregression(5, 4)
        f = np.random.normal(size=gpmap.n_genotypes)
        
        ### Test first with pure 2nd order ###
        k = 2

        # Calculating projection directly        
        D = gpmap.calc_distance_matrix()
        W_k = gpmap.W_kd[k][D]
        f_k1 = W_k.dot(f)

        # Use Laplacian powers
        lambdas = np.zeros(gpmap.length + 1)
        lambdas[k] = 1
        L_powers_coeffs = gpmap.calc_L_powers_coeffs(lambdas)
        f_k2 = gpmap.W_dot(f, L_powers_coeffs)
        assert(np.allclose(f_k1, f_k2))
        
        ### Test first all orders simultaneously ###
        
        # Calculating projection directly 
        lambdas = [0, 10, 5, 2, 1, 0.1]
        D = gpmap.calc_distance_matrix()
        W = 0
        for k in range(gpmap.length + 1):
            W += gpmap.W_kd[k][D] * lambdas[k]
        f_1 = W.dot(f)

        # Use Laplacian powers
        L_powers_coeffs = gpmap.calc_L_powers_coeffs(lambdas)
        f_2 = gpmap.W_dot(f, L_powers_coeffs)
        
        # Use cov matrix directly
        W = gpmap.calc_cov_dense(lambdas)
        f_3 = W.dot(f)
        
        assert(np.allclose(f_1, f_2))
        assert(np.allclose(f_1, f_3))
    
    
    def xtest_calc_generalized_laplacian(self):
        gpmap = VCregression(2, 2, alphabet_type='custom')
        p = np.array([[0.3, 0.5],
                      [0.7, 0.5]])
        gpmap.calc_generalized_laplacian(p)
        L = np.array([[ 1.2, -0.5, -0.45825757, 0.],
                      [-0.5,  1.2,  0.,  -0.45825757],
                      [-0.45825757,  0.,   0.8, -0.5],
                      [ 0., -0.45825757, -0.5, 0.8]])
        assert(np.allclose(gpmap.L.todense(), L))
        assert(np.allclose(gpmap.L.dot(np.sqrt(gpmap.probability)), 0))
        l = np.linalg.eigvalsh(gpmap.L.todense())
        assert(np.allclose(l,  [0, 1, 1, 2]))
    
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'VCTests']
    unittest.main()
