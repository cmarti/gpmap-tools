#!/usr/bin/env python
import sys
import unittest

import numpy as np
from scipy.stats import pearsonr

from gpmap.inference import ConnectednessModelRegression


class ConnectednessModelTests(unittest.TestCase):
    def test_simulate(self):
        mu = np.array([0.5, 0.1, 0.2, 0.1, 0.05])
        model = ConnectednessModelRegression(seq_length=5,
                                             n_alleles=4,
                                             mu=mu, sigma2=1.)
        f = model.sample_prior()
        assert(f.shape[0] == 4 ** 5)
    
    def test_predict(self):
        # Partial dataset that can recapitulate MEI
        X = np.array(["AA", "AB", "BA"])
        y = np.array([0, 1, 1.0])
        y_var = np.array([0.1] * 3)
        
        # Test with equal decay factors
        mu = np.array([0.2, 0.2])
        sigma2 = 1.
        model = ConnectednessModelRegression(seq_length=2,
                                             n_alleles=2,
                                             genotypes=X,
                                             mu=mu, sigma2=sigma2)
        model.set_data(X, y, y_var)
        fhat, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert np.allclose(fhat[1], fhat[2])
        assert Sigma[0, 0] < Sigma[3, 3]
        
        # Test with different decay factors
        mu = np.array([0.2, 0.1])
        sigma2 = 1.
        model = ConnectednessModelRegression(seq_length=2,
                                             n_alleles=2,
                                             genotypes=X,
                                             mu=mu, sigma2=sigma2)
        model.set_data(X, y, y_var)
        fhat, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert not np.allclose(fhat[1], fhat[2])
        assert Sigma[0, 0] < Sigma[3, 3]
    
        # Test with larger sequence space from simulated data
        mu = np.array([0.5, 0.1, 0.2, 0.1, 0.05])
        model = ConnectednessModelRegression(seq_length=5,
                                             n_alleles=4,
                                             mu=mu, sigma2=1.)
        f = model.sample_prior()
        y_var = np.full_like(f, 0.01)
        y = np.random.normal(f, 0.01)
        X = model.genotypes
        
        model.set_data(X, y, y_var)
        pred = model.predict()
        r = pearsonr(pred['f'], f)[0]
        assert(r > 0.8)
        
        # Compute posterior variance: uniform because we have all sequences
        X_pred = np.random.choice(X, size=10)
        pred = model.predict(X_pred, calc_variance=True)
        assert(np.allclose(pred['f_var'], 0.007665, atol=1e-4))
        
    def test_fit(self):
        np.random.seed(0)
        mu = np.array([0.5, 0.1, 0.2, 0.1, 0.05])
        model = ConnectednessModelRegression(seq_length=5,
                                             n_alleles=4,
                                             mu=mu, sigma2=1.)
        f = model.sample_prior()
        y_var = np.full_like(f, 0.01)
        y = np.random.normal(f, 0.01)
        X = model.genotypes
        
        model.fit(X, y, y_var)
        df = model.get_decay_factors()
        r = pearsonr(df['mu'], mu)[0]
        assert(r > 0.8)
        assert(np.all(df.columns == ['mu', 'decay_factor']))


if __name__ == "__main__":
    sys.argv = ["", "ConnectednessModelTests"]
    unittest.main()
