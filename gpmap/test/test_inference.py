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
    
    def test_calc_L_polynomial_coeffs(self):
        vc = VCregression()
        vc.init(2, 2)
        lambdas = np.ones(vc.seq_length + 1)
        
        # Ensure that if all eigenvalues are 1 then we would end up with I
        vc.calc_L_polynomial_coeffs()
        assert(np.allclose(vc.B.sum(1), [1, 0, 0]))

        # The same using the method        
        bs = vc.calc_scaled_L_polynomial_coeffs(lambdas)
        assert(np.allclose(bs, [1, 0, 0]))
        
    def test_calc_L_polynomial_coeffs_analytical(self):
        for l in range(3, 9):
            vc = VCregression()
            vc.init(l, 4)
            y = np.random.normal(size=vc.n_genotypes)
            
            b1 = vc.calc_L_polynomial_coeffs()
            p1 = vc.project(y, k=l-1)
            b2 = vc.calc_L_polynomial_coeffs_num()
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
    
    def test_vc_fit(self):
        np.random.seed(0)
        sigma, lambdas = 0., np.array([0, 200, 20, 2, 0.2, 0.02])
        
        vc = VCregression()
        vc.init(5, 4)
        data = vc.simulate(lambdas, sigma=sigma, p_missing=0.05).dropna()
        
        # Ensure MSE is within a small range
        vc = VCregression()
        vc.fit(data.index, data['y'], variance=data['var'],
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
    
    def test_vc_predict(self):
        np.random.seed(0)
        sigma, lambdas = 0.2, np.array([0, 200, 20, 2, 0.2, 0.02])
        
        vc = VCregression()
        vc.init(5, 4)
        data = vc.simulate(lambdas, sigma)
        
        # Estimating also the variance components
        vc = VCregression()
        vc.fit(data.index, data['y'], variance=data['var'])
        pred = vc.predict()
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.05)
        
        # Using the a priori known variance components
        vc = VCregression()
        pred = vc.predict(X=data.index, y=data['y'],
                          variance=data['var'], lambdas=lambdas)
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.05)
        
        # Estimate posterior variances
        vc = VCregression()
        pred = vc.predict(X=data.index, y=data['y'],
                          variance=data['var'], lambdas=lambdas, 
                          estimate_variance=True)
        assert('var' in pred.columns)
        assert(np.all(pred['var'] > 0))
        
        # Capture error with incomplete input
        try:
            pred = vc.predict(X=data.index, y=data['y'],
                              variance=data['var'])
            self.fail()
        except ValueError:
            pass
        
        # With incomplete genotype sampling data: larger MSE
        filtered = data.loc[np.random.choice(data.index, size=950), :]
        vc = VCregression()
        vc.fit(filtered.index, filtered['y'], variance=filtered['var'])
        pred = vc.predict()
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
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
