#!/usr/bin/env python
import sys
import unittest
import numpy as np

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

from scipy.stats import pearsonr
from scipy.special import comb

from gpmap.src.inference import VCregression
from gpmap.src.settings import BIN_DIR
from gpmap.src.linop import (LaplacianOperator, ProjectionOperator,
                             calc_variance_components, calc_covariance_distance)
from gpmap.src.space import SequenceSpace


class VCTests(unittest.TestCase):
    def test_lambdas_to_variance_p(self):
        lambdas = np.array([0, 1, 0.5, 0.1])
        vc = VCregression(n_alleles=2, seq_length=3, lambdas=lambdas)
        v = vc.lambdas_to_variance(lambdas)
        assert(np.allclose(v, [3/4.6, 1.5/4.6, 0.1/4.6]))
    
    def test_simulate_vc(self):
        np.random.seed(1)
        sigma = 0.1
        l, a = 4, 4

        lambdas = np.array([0, 200, 20, 2, 0.2])
        vc = VCregression(n_alleles=a, seq_length=l, lambdas=lambdas)
        
        # Test functioning and size
        data = vc.simulate(sigma)
        f = data['y_true'].values
        assert(data.shape[0] == 256)
        
        # Test missing genotypes
        data = vc.simulate(sigma, p_missing=0.1)
        assert(data.dropna().shape[0] < 256)
        
        # Test pure components
        
        for k1 in range(l + 1):
            vc.set_lambdas(k=k1)    
            data = vc.simulate()
            f = data['y_true'].values
        
            for k2 in range(l+1):
                W = ProjectionOperator(a, l, k=k2)
                f_k_rq = W.rayleigh_quotient(f)
                assert(np.allclose(f_k_rq, k1 == k2))
    
    def test_calc_distance_covariance(self):
        np.random.seed(1)
        sigma, lambdas = 0.1, np.array([0, 200, 20, 2, 0.2])
        n_alleles, seq_length = 4, 4
        
        vc = VCregression(n_alleles=n_alleles, seq_length=seq_length,
                          lambdas=lambdas)
        data = vc.simulate(sigma)
        rho, n = calc_covariance_distance(data.y.values, n_alleles, seq_length)
        
        # Ensure we get the expected number of pairs per distance category
        for d in range(seq_length + 1):
            total_genotypes = n_alleles ** seq_length
            d_combs = comb(seq_length, d)
            d_sites_genotypes = (n_alleles - 1) ** d
            n[d] = total_genotypes * d_combs * d_sites_genotypes
            
        # Ensure anticorrelated distances
        assert(rho[3] < 0)
        assert(rho[4] < 0)
        
        # With missing data
        data = vc.simulate(sigma, p_missing=0.1).dropna()
        idx = vc.get_obs_idx(data.index.values)
        rho, n = calc_covariance_distance(data.y.values, n_alleles, seq_length, idx=idx)
            
        # Ensure anticorrelated distances
        assert(rho[3] < 0)
        assert(rho[4] < 0)
    
    def test_vc_fit(self):
        # Simulate data
        np.random.seed(0)
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        a, l = 4, lambdas.shape[0]-1
        vc = VCregression(n_alleles=a, seq_length=l, lambdas=lambdas)
        data = vc.simulate(sigma=0.1)
        
        # Ensure MSE is within a small range
        vc = VCregression()
        vc.fit(X=data.index.values, y=data['y'], y_var=data['y_var'])
        sd1 = np.log2((vc.lambdas[1:]+1e-6) / (lambdas[1:]+1e-6)).std()
        assert(sd1 < 2)
        
        # Try with regularization and CV
        vc = VCregression(cross_validation=True)
        vc.fit(data.index, data['y'], y_var=data['y_var'])
        sd2 = np.log2((vc.lambdas[1:]+1e-6) / (lambdas[1:]+1e-6)).std()
        assert(sd2 < 1)
        assert(vc.beta > 0)
        
        # Ensure regularization improves results
        assert(sd1 > sd2)
        
    def test_vc_predict(self):
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        a, l = 4, lambdas.shape[0]-1
        vc = VCregression(seq_length=l, n_alleles=a, lambdas=lambdas)
        data = vc.simulate(sigma=0.1)
        
        # Using the a priori known variance components
        vc.set_data(X=data.index, y=data['y'], y_var=data['y_var'])
        pred = vc.predict()
        mse = np.mean((pred['ypred'] - data['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.05)
        
        # Estimate posterior variances
        pred = vc.predict(calc_variance=True)
        assert('var' in pred.columns)
        assert(np.all(pred['var'] > 0))
        
        # Capture error with missing lambdas
        vc = VCregression()
        vc.set_data(X=data.index, y=data['y'], y_var=data['y_var'])
        try:
            pred = vc.predict()
            self.fail()
        except ValueError:
            pass
    
        # Subset the data
        idx = np.random.uniform(size=data.shape[0]) < 0.9
        train, test = data.loc[idx, :], data.loc[~idx, :]

        # Using known lambdas        
        vc = VCregression()
        vc.set_data(X=train.index, y=train['y'], y_var=train['y_var'])
        vc.set_lambdas(lambdas)
        pred = vc.predict(X_pred=test.index.values)
        mse = np.mean((pred['ypred'] - test['y_true']) ** 2)
        rho = pearsonr(pred['ypred'], test['y_true'])[0]
        assert(rho > 0.6)
        assert(mse < 0.3)
        
        # Calculate variances and check calibration
        pred = vc.predict(X_pred=test.index.values, calc_variance=True)
        sigma = np.sqrt(pred['var'])
        pred['lower'] = pred['ypred'] - 2 * sigma
        pred['upper'] = pred['ypred'] + 2 * sigma
        p = np.logical_and(test['y_true'] > pred['lower'],
                           test['y_true'] < pred['upper']).mean()
        assert(p > 0.9)
    
    def test_vc_regression_bin(self):
        bin_fpath = join(BIN_DIR, 'vc_regression.py')

        with NamedTemporaryFile() as fhand:
            data_fpath = '{}.data.csv'.format(fhand.name)
            lambdas_fpath = '{}.lambdas.csv'.format(fhand.name)
            out_fpath = '{}.out.csv'.format(fhand.name)
            xpred_fpath = '{}.xpred.txt'.format(fhand.name)
            
            # Simulate data
            a, l = 4, 5
            lambdas = np.exp(-np.arange(0, l+1))
            
            vc = VCregression(alphabet_type='dna', seq_length=l,
                              lambdas=lambdas)
            data = vc.simulate(sigma=0.05)
            idx = np.random.uniform(size=data.shape[0]) < 0.99
            train, test = data.loc[idx, :], data.loc[~idx, :]
            
            # Save simulated data in temporary files
            train.to_csv(data_fpath)
            with open(xpred_fpath, 'w') as fhand:
                for seq in test.index:
                    fhand.write(seq + '\n')
            
            with open(lambdas_fpath, 'w') as fhand:
                for l in lambdas:
                    fhand.write('{}\n'.format(l))
        
            # Direct kernel alignment
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath]
            check_call(cmd)
            
            # With known lambdas
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-r',
                   '--lambdas', lambdas_fpath]
            check_call(cmd)
            
            # Run with regularization
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-r']
            check_call(cmd)
            
            # Predict few sequences and their variances under known lambdas
            cmd = [sys.executable, bin_fpath, data_fpath, '-o', out_fpath, '-r',
                   '--var', '-p', xpred_fpath, '--lambdas', lambdas_fpath]
            check_call(cmd)
            

class SkewedVCTests(unittest.TestCase):
    def xtest_simulate_skewed_vc(self):  
        np.random.seed(1)
        l, a = 2, 2 
        vc = VCregression()
        
        # With p=1
        ps = 1 * np.ones((l, a))
        L = LaplacianOperator(a, l)
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
        
        
if __name__ == '__main__':
    sys.argv = ['', 'VCTests']
    unittest.main()
