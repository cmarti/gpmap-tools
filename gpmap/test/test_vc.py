#!/usr/bin/env python
import sys
import unittest
import numpy as np
import pandas as pd

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

from scipy.stats import pearsonr
from scipy.special import comb

from gpmap.src.inference import VCregression
from gpmap.src.settings import BIN_DIR
from gpmap.src.matrix import rayleigh_quotient
from gpmap.src.linop import (LaplacianOperator, ProjectionOperator,
                             calc_covariance_distance)


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
                f_k_rq = rayleigh_quotient(W, f)
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
        
    def test_vc_process_data(self):
        # Simulate data
        np.random.seed(0)
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        a, l = 4, lambdas.shape[0]-1
        vc = VCregression(n_alleles=a, seq_length=l, lambdas=lambdas)
        data = vc.simulate(sigma=0.1)
        data = (data.index.values, data['y'], data['y_var'])
        
        vc = VCregression(n_alleles=a, seq_length=l)
        X, _, _, _, ns = vc.process_data(data)
        assert(np.isclose(X.shape[0] ** 2, ns.sum()))
            
    def test_vc_docs(self):
        # Simulate data
        np.random.seed(0)
        lambdas_true = np.array([1e3, 1e3, 2e2, 1e0, 1e-1, 3e-3, 1e-5])
        model = VCregression(seq_length=6, alphabet_type='dna', lambdas=lambdas_true)
        data = model.simulate(sigma=0.2, p_missing=0.1)
        obs = data.dropna()
        
        # Without regularization
        model.fit(X=obs.index.values, y=obs.y.values, y_var=obs.y_var.values)
        
        # Try with regularization and CV
        cvmodel = VCregression(cross_validation=True)
        cvmodel.fit(X=obs.index.values, y=obs.y.values, y_var=obs.y_var.values)
        
        # Try with different CV metric
        cvmodel = VCregression(cross_validation=True, cv_loss_function='logL')
        cvmodel.fit(X=obs.index.values, y=obs.y.values, y_var=obs.y_var.values)

    def test_vc_calc_posterior(self):
        np.random.seed(0)
        lambdas = np.array([1, 200, 20, 2, 0.2])
        a, l = 4, lambdas.shape[0]-1
        vc = VCregression(seq_length=l, n_alleles=a, lambdas=lambdas, progress=False)
        data = vc.simulate(sigma=0.1, p_missing=0.05)
        test = data.loc[np.isnan(data['y']), :].iloc[:3, :]
        X_pred = test.index.values
        n = test.shape[0]
        train = data.dropna()
        
        # Ensure that the posterior has the right form
        vc.set_data(X=train.index.values, y=train['y'], y_var=train['y_var'])
        m, S = vc.calc_posterior(X_pred=X_pred)
        assert(m.shape == (n, ))
        assert(S.shape == (n, n))

        # Test that the diagonal correspond to the posterior variances calculated individually
        S = S @ np.eye(S.shape[1])
        y_var1 = np.diag(S)
        y_var2 = vc.calc_posterior_variance(X_pred=X_pred)
        assert(np.allclose(y_var1, y_var2))

        # Test posterior of linear combination
        B = np.array([[0, 1, -1],
                      [1, -1, 0]])
        m, S = vc.calc_posterior(X_pred=X_pred, B=B)
        S = S @ np.eye(S.shape[1])
    
    def test_vc_contrasts(self):
        np.random.seed(0)
        lambdas = np.array([1, 200, 20, 2, 0.2])
        l = lambdas.shape[0]-1
        vc = VCregression(seq_length=l, alphabet_type='dna',
                          lambdas=lambdas, progress=False)
        data = vc.simulate(sigma=0.1, p_missing=0.05)
        test = data.loc[np.isnan(data['y']), :].iloc[:3, :]
        train = data.dropna()
        vc.set_data(X=train.index.values, y=train['y'], y_var=train['y_var'])

        # Test a single contrast
        contrast_matrix = pd.DataFrame({'c1': [0, 1, -1]},
                                        index=test.index.values)
        results = vc.make_contrasts(contrast_matrix)
        assert(results.shape == (1, 5))
        
        # Make contrasts between random test points
        contrast_matrix = pd.DataFrame({'c1': [0, 1, -1], 
                                        'c2': [0.5, 0.5, 0]},
                                        index=test.index.values)
        results = vc.make_contrasts(contrast_matrix)
        assert(results.shape == (2, 5))

        # Make contrasts for mutational effect and epistatic coefficient
        seqs = ['AGCT', 'AGCC', 'TGCT', 'TGCC']
        contrast_matrix = pd.DataFrame({'T4C':     [0, 0, 1, -1], 
                                        'T4C:A1T': [1, -1, -1, 1]},
                                        index=seqs)
        results = vc.make_contrasts(contrast_matrix)
        assert(results.shape == (2, 5))

    def test_vc_predict(self):
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        a, l = 4, lambdas.shape[0]-1
        vc = VCregression(seq_length=l, n_alleles=a, lambdas=lambdas)
        data = vc.simulate(sigma=0.1)
        
        # Using the a priori known variance components
        vc.set_data(X=data.index, y=data['y'], y_var=data['y_var'])
        pred = vc.predict()
        mse = np.mean((pred['y'] - data['y_true']) ** 2)
        rho = pearsonr(pred['y'], data['y_true'])[0]
        assert(rho > 0.95)
        assert(mse < 0.05)
        
        # Estimate posterior variances
        pred = vc.predict(calc_variance=True)
        assert('y_var' in pred.columns)
        assert(np.all(pred['y_var'] > 0))
        
        # Capture error with missing lambdas
        vc = VCregression()
        vc.set_data(X=data.index, y=data['y'], y_var=data['y_var'])
        try:
            pred = vc.predict()
            self.fail()
        except:
            pass
    
        # Subset the data
        idx = np.random.uniform(size=data.shape[0]) < 0.9
        train, test = data.loc[idx, :], data.loc[~idx, :]

        # Using known lambdas        
        vc = VCregression()
        vc.set_data(X=train.index, y=train['y'], y_var=train['y_var'])
        vc.set_lambdas(lambdas)
        pred = vc.predict(X_pred=test.index.values)
        mse = np.mean((pred['y'] - test['y_true']) ** 2)
        rho = pearsonr(pred['y'], test['y_true'])[0]
        assert(rho > 0.6)
        assert(mse < 0.3)
        
        # Calculate variances and check calibration
        pred = vc.predict(X_pred=test.index.values, calc_variance=True)
        sigma = np.sqrt(pred['y_var'])
        pred['lower'] = pred['y'] - 2 * sigma
        pred['upper'] = pred['y'] + 2 * sigma
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
    
    def test_sd(self):
        fpath = '/home/martigo/elzar/projects/shine_dalgarno/data/dmsc_processed.csv'
        df = pd.read_csv(fpath, index_col=0).dropna(subset=['y', 'y_var_post'])
        # print(df)
        # exit()
        X, y, y_var = df.index.values, df.y.values, df.y_var_post.values
        
        model = VCregression(beta=1e5)
        model.fit(X, y, y_var=y_var)
        print(model.get_variance_component_df(model.lambdas))
        # model.set_lambdas(model.lambdas + y_var[0])
        print(model.predict())
            

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
                f_k_rq = rayleigh_quotient(W, f)
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
                f_k_rq = rayleigh_quotient(W, f)
                assert(np.allclose(f_k_rq, k1 == k2))
        
        
if __name__ == '__main__':
    sys.argv = ['', 'VCTests']
    unittest.main()
