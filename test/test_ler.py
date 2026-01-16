#!/usr/bin/env python
import sys
import unittest
import numpy as np

from itertools import product
from gpmap.inference import LocalEpistasisRegression


class LERTests(unittest.TestCase):
    def test_initializations(self):
        configs = [{'seq_length': 2, 'n_alleles': 2},
                   {'seq_length': 2, 'alphabet_type': 'rna'},
                   {'genotypes': ['AA', 'AB', 'BA', 'BB']}]
        for config in configs:
            model = LocalEpistasisRegression(**config)
        
    def test_predict(self):
        # Partial dataset that can recapitulate MEI
        X = np.array(["AA", "AB", "BA"])
        y = np.array([0, 1, 1.0])
        y_var = np.array([0.1] * 3)
        a_values = np.array([1.])

        model = LocalEpistasisRegression(seq_length=2,
                                         n_alleles=2,
                                         genotypes=X, a_values=a_values)
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert np.allclose(mu, [0, 1, 1, 2])
        assert Sigma[0, 0] < Sigma[3, 3]

        # Test in a bigger landscape
        X = np.array(["AAA", "AAB", "ABA", "BAA", "BAB", "BBA"])
        y = np.array([0, 1, 1.0, 0, 1, 1.0])
        y_var = np.array([0.1] * 6)
        a_values = np.array([1.0, 0.5, 0.25])

        model = LocalEpistasisRegression(
            seq_length=3, n_alleles=2, genotypes=X, a_values=a_values
        )
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(8)
        assert np.allclose(mu, [0, 1, 1, 2, 0, 1, 1, 2])
        assert Sigma[0, 0] < Sigma[3, 3]

        # Test with different a_values
        X = np.array([''.join(x) for x in product(['A', 'B'], repeat=3)])
        y = np.random.normal(size=X.shape[0])
        y_var = np.array([0.1] * X.shape[0])

        a_values = np.array([1.0, 0.5, 0.25])
        model = LocalEpistasisRegression(
            seq_length=3, n_alleles=2, genotypes=X, a_values=a_values
        )
        model.set_data(X, y, y_var)
        mu1 = model.calc_posterior_mean()

        a_values = np.array([0.5, 0.25, 1])
        model.set_a_values(a_values)
        mu2 = model.calc_posterior_mean()
        assert( not np.allclose(mu1, mu2, atol=1e-4))
    
    def test_fit(self):
        X = np.array(["AAA", "AAB", "ABA", "BAA", "BAB", "BBA"])
        y_var = np.array([0.1] * 6)
        a_values = np.array([1.0, 0.5, 0.25])

        model = LocalEpistasisRegression(
            seq_length=3, n_alleles=2, genotypes=X, a_values=a_values
        )

        # Fit a purely additive function
        y = np.array([0, 1, 1.0, 0, 1, 1.0])
        model.fit(X, y, y_var)
        assert np.all(model.a_values > 1e16)

        # Fit a more complicated function with epistasis across a pair of sites
        X = np.array(["".join(x) for x in product(["A", "B"], repeat=3)])
        y = np.array([0, 1, 1, 3,
                      1, 2, 2, 4])
        y_var = np.array([0.1] * X.shape[0])
        model.fit(X, y, y_var)
        assert np.all(model.a_values[:2] > 1e16)
        assert np.allclose(model.a_values[-1], 0.5)


if __name__ == "__main__":
    sys.argv = ["", "MEITests"]
    unittest.main()
