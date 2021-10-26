#!/usr/bin/env python
import unittest
from os.path import join

import numpy as np
import pandas as pd
from scipy.stats.mstats_basic import pearsonr

from gpmap.settings import TEST_DATA_DIR
from gpmap.utils import LogTrack
from gpmap.inference import VCregression


def get_test_gpmap(add_missing=False):
    log = LogTrack()
    fpath = join(TEST_DATA_DIR, 'pentamer.csv')
    data = pd.read_csv(fpath).sort_values('seq').set_index('seq')
    if add_missing:
        idx = np.random.choice(np.arange(data.shape[0]),
                               size=int(data.shape[0] * 0.9), replace=False)
        data = data.iloc[idx, :]
    
    gpmap = VCregression(5, 4, log=log)
    gpmap.load_data(f_obs=data['mean'], variance=data['variance'],
                    seqs=data.index)
    return(gpmap)


class VCTests(unittest.TestCase):
    def test_load_data(self):
        gpmap = get_test_gpmap()
        assert(gpmap.n_obs == (1024))
        assert(np.all(gpmap.genotype_labels == gpmap.obs.index))
    
    def test_calc_MAT(self):
        gpmap = get_test_gpmap()
        gpmap.calc_MAT()
        exp_mat = [[1, 15, 240, 4020, 69720, 1242600],
                   [0, -1, -28, -628, -13120, -266296],
                   [0,  0, 2,  78,  2168, 52680],
                   [0, 0, 0, -6, -288, -9240],
                   [0, 0, 0, 0,  24, 1320],
                   [0 ,0 ,0, 0, 0, -120]]
        assert(np.all(exp_mat == gpmap.MAT))
    
    def test_compute_empirical_rho(self):
        gpmap = get_test_gpmap()
        gpmap.calc_laplacian()
        gpmap.calc_MAT()
        rho_d = gpmap.compute_empirical_rho()[0]
        
        expected_rho = [0.3289592165429687, 0.29711601889225264, 0.2829828261439887,
                        0.27518021926897074, 0.271383375496395, 0.27036637220315385]
        assert(np.allclose(expected_rho, rho_d))
        
    def test_vc_inference(self):
        gpmap = get_test_gpmap()
        gpmap.estimate_variance_components()
        expected_lambdas = [2.80160850e02, 1.11396023, 2.56638126e-01,
                             3.51032105e-02, 1.10928480e-02, 6.69724738e-03]
        
        lambdas = gpmap.estimate_variance_components(regularize=False)
        rho = pearsonr(np.log(lambdas), np.log(expected_lambdas))[0]
        logd = np.log(lambdas) - np.log(expected_lambdas)
        assert(rho > 0.99)
        assert(logd.mean() < 0.1)
        
        vc = gpmap.lambdas_to_variance(lambdas)
        expected_vc = [0.29507236, 0.40941648, 0.1731748, 0.0882358, 0.03410057]
        assert(np.allclose(vc, expected_vc, atol=0.01))
    
        lambdas = gpmap.estimate_variance_components(regularize=True)
        rho = pearsonr(np.log(lambdas), np.log(expected_lambdas))[0]
        logd = np.log(lambdas) - np.log(expected_lambdas)
        assert(rho > 0.99)
        assert(logd.mean() < 0.1)
        
        vc = gpmap.lambdas_to_variance(lambdas)
        assert(np.allclose(vc, expected_vc, atol=0.05))
        
    def test_vc_inference2(self):
        np.random.seed(1)
        lambdas = [0, 100, 10, 1, 0.1, 0.01]
        vcreg = VCregression(5, 4)
        vc = vcreg.lambdas_to_variance(lambdas)
        vcreg.simulate(lambdas=lambdas, sigma=0.1, p_missing=0.1)
        
        lambdas_inferred = vcreg.estimate_lambdas(regularize=False)
        vc_inferred = vcreg.lambdas_to_variance(lambdas_inferred)
        assert(np.allclose(vc, vc_inferred, atol=0.05))
        
        lambdas_inferred = vcreg.estimate_lambdas(regularize=True)
        vc_inferred = vcreg.lambdas_to_variance(lambdas_inferred)
        assert(np.allclose(vc, vc_inferred, atol=0.05))
    
    def test_estimate_f(self):
        np.random.seed(0)
        lambdas = [2.80160850e02, 1.11396023, 2.56638126e-01,
                   3.51032105e-02, 1.10928480e-02, 6.69724738e-03]
        exp_f1 = [0.46865088, 0.71775025, 0.32067001, 0.62541312, 0.63366557]
        exp_f2 = [0.76384838, 0.87255378, 0.94822025, 0.87959252, 0.89394908]

        gpmap = get_test_gpmap()        
        f = gpmap.estimate_f(lambdas)
        assert(np.allclose(f[:5], exp_f1))
        assert(np.allclose(f[-5:], exp_f2))
        
        gpmap = get_test_gpmap(add_missing=True)
        f = gpmap.estimate_f(lambdas)
        assert(f.shape[0] == 1024)
        assert(np.allclose(f[:5], exp_f1, atol=0.01))
        assert(np.allclose(f[-5:], exp_f2, atol=0.01))
    
    def test_estimate_f_variance(self):
        np.random.seed(0)
        lambdas = [2.80160850e02, 1.11396023, 2.56638126e-01,
                   3.51032105e-02, 1.10928480e-02, 6.69724738e-03]

        gpmap = get_test_gpmap()        
        v = gpmap.estimate_f_variance(lambdas)
        print(v)
        
    def test_simulate_f(self):
        np.random.seed(1)
        sigma = 0.01
        gpmap = VCregression(4, 4)
        gpmap.calc_L_eigendecomposition()
        
        estimated = []
        for _ in range(100):
            lambdas = np.array([0, 200, 20, 0, 0])
            gpmap.simulate(lambdas, sigma)
            vc = gpmap.lambdas_to_variance(lambdas)
            lambdas_star = gpmap.estimate_variance_components()
            vc_star = gpmap.lambdas_to_variance(lambdas_star)
            record = {'vc': np.abs(vc_star[:2] - vc[:2]).mean(),
                      'lambda': np.abs(np.log(lambdas_star[1:3] / lambdas[1:3])).mean()}
            estimated.append(record)
            
        estimated = pd.DataFrame(estimated).mean()
        assert(np.all(estimated < [0.1, 0.3]))
    
    def test_calc_laplacian_powers(self):
        gpmap = VCregression(3, 2)
        gpmap.simulate([0, 1, 0.5, 0.1])
        L_powers = gpmap.calc_laplacian_powers(gpmap.obs)
        assert(L_powers.shape == (8, 4))
    
    def test_calc_distance_matrix(self):
        gpmap = VCregression(2, 2)
        D = gpmap.calc_distance_matrix()
        expected_D = np.array([[0, 1, 1, 2],
                               [1, 0, 2, 1],
                               [1, 2, 0, 1],
                               [2, 1, 1, 0]])
        assert(np.all(D == expected_D))
        assert(D.dtype == 'int')
    
    def test_W_dot_precision(self):
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
    
    def test_stan_model(self):
        np.random.seed(1)
        sigma = 0.05
        gpmap = VCregression(4, 4)
        lambdas = np.array([0, 200, 20, 2, 0.2])
        gpmap.simulate(lambdas, sigma)
        vc = gpmap.lambdas_to_variance(lambdas)
        print(vc)
        fit = gpmap.stan_fit(sigma)
        print(gpmap.lambdas_to_variance(fit['lambdas'].mean(0)))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'VCTests']
    unittest.main()
