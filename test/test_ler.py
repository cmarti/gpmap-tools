#!/usr/bin/env python
import sys
import unittest
import numpy as np

from itertools import product
from gpmap.inference import LocalEpistasisRegression, LocalEpistasisMinimizer


class LEMTests(unittest.TestCase):
    def test_initializations(self):
        configs = [
            {"seq_length": 2, "n_alleles": 2},
            {"seq_length": 2, "alphabet_type": "rna"},
            {"genotypes": ["AA", "AB", "BA", "BB"]},
        ]
        for config in configs:
            LocalEpistasisMinimizer(**config)

    def test_predict(self):
        # Partial dataset that can recapitulate MEI
        X = np.array(["AA", "AB", "BA"])
        y = np.array([0, 1, 1.0])
        y_var = np.array([0.1] * 3)
        a_values = np.array([1.0])

        model = LocalEpistasisMinimizer(
            seq_length=2, n_alleles=2, genotypes=X, a_values=a_values
        )
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

        model = LocalEpistasisMinimizer(
            seq_length=3, n_alleles=2, genotypes=X, a_values=a_values
        )
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(8)
        assert np.allclose(mu, [0, 1, 1, 2, 0, 1, 1, 2])
        assert Sigma[0, 0] < Sigma[3, 3]

        # Test with different a_values
        X = np.array(["".join(x) for x in product(["A", "B"], repeat=3)])
        y = np.random.normal(size=X.shape[0])
        y_var = np.array([0.1] * X.shape[0])

        a_values = np.array([1.0, 0.5, 0.25])
        model = LocalEpistasisMinimizer(
            seq_length=3, n_alleles=2, genotypes=X, a_values=a_values
        )
        model.set_data(X, y, y_var)
        mu1 = model.calc_posterior_mean()

        a_values = np.array([0.5, 0.25, 1])
        model.set_a_values(a_values)
        mu2 = model.calc_posterior_mean()
        assert not np.allclose(mu1, mu2, atol=1e-4)

    def test_fit(self):
        X = np.array(["AAA", "AAB", "ABA", "BAA", "BAB", "BBA"])
        model = LocalEpistasisMinimizer(seq_length=3, n_alleles=2, genotypes=X)

        # Fit a constant function
        y = np.ones(X.shape)
        model.fit(X, y)
        print(model.a_values)
        assert np.all(model.a_values > 1e16)

        # Fit a purely additive function
        y = np.array([0, 1, 1.0, 0, 1, 1.0])
        model.fit(X, y)
        print(model.a_values)
        assert np.all(model.a_values > 1e16)

        # Fit a more complicated function with epistasis across a pair of sites
        X = np.array(["".join(x) for x in product(["A", "B"], repeat=3)])
        y = np.array([0, 1, 1, 3, 1, 2, 2, 4])
        y_var = np.array([0.1] * X.shape[0])
        model.fit(X, y, y_var)
        assert np.all(model.a_values[:2] > 1e16)
        assert np.allclose(model.a_values[-1], 0.5)


class LERTests(unittest.TestCase):
    def test_initializations(self):
        configs = [
            {"seq_length": 2, "n_alleles": 2},
            {"seq_length": 2, "alphabet_type": "rna"},
            {"genotypes": ["AA", "AB", "BA", "BB"]},
        ]
        for config in configs:
            LocalEpistasisRegression(**config)

    def test_predict(self):
        # Partial dataset that can recapitulate MEI
        X = np.array(["AA", "AB", "BA"])
        y = np.array([0, 1, 1.0])
        y_var = np.array([0.1] * 3)
        a_values = np.array([1.0])
        lambda_U = np.array([1e3, 10, 10])

        model = LocalEpistasisRegression(
            seq_length=2,
            n_alleles=2,
            genotypes=X,
            a_values=a_values,
            lambda_U_lower_than_P=lambda_U,
        )
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert np.allclose(mu, [0, 1, 1, 2], atol=0.5)
        assert Sigma[0, 0] < Sigma[3, 3]

    def test_fit_complete(self):
        X = np.array([''.join(x) for x in product(list('AB'), repeat=3)])
        model = LocalEpistasisRegression(seq_length=3, n_alleles=2, genotypes=X)

        # Fit a constant function
        f = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        model.fit(X, f)
        assert np.all(model.a_values > 1e12)
        assert np.all(model.lambda_U_lower_than_P > 0)
        assert np.all(model.lambda_U_lower_than_P[1:] < 1e-12)
        assert np.allclose(model.lambda_U_lower_than_P[0], 8)
        
        # Fit a purely additive function
        f = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        model.fit(X, f)
        assert np.all(model.a_values > 1e12)
        assert np.all(model.lambda_U_lower_than_P > 0)
        assert np.all(model.lambda_U_lower_than_P[:-1] < 1e-12)
        assert np.allclose(model.lambda_U_lower_than_P[-1], 8)
        
        # Fit a pure local interaction
        f = np.array([1, -1, -1, 1, 1, -1, -1, 1])
        model.fit(X, f)
        assert np.all(model.a_values[-1] < 10)
        assert np.all(model.a_values[:-1] > 1e12)
        assert np.all(model.lambda_U_lower_than_P < 1e-12)
        
        # More complex interactions
        f = np.array([0, 1, 1, 3, 1, 2, 2, 4])
        model.fit(X, f)
        assert np.all(model.a_values[:2] > 1e12)
        assert np.allclose(model.a_values[-1], 0.5, rtol=1)
        assert np.allclose(model.lambda_U_lower_than_P, [24.5, 4.5, 4.5, 2])
    
    def test_fit_complete_noise(self):
        np.random.seed(1234)
        X = np.array([''.join(x) for x in product(list('AB'), repeat=3)])
        y_var = np.full(X.shape, 0.005)
        y_sd = np.sqrt(y_var)
        noise = np.random.normal(0, y_sd)
        model = LocalEpistasisRegression(seq_length=3, n_alleles=2, genotypes=X)

        # Fit a constant function
        f = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y = f + noise
        model.fit(X, y, y_var=y_var)
        assert np.all(model.a_values > 20)
        assert np.all(model.lambda_U_lower_than_P > 0)
        assert np.all(model.lambda_U_lower_than_P[1:] < 1e-2)
        assert np.allclose(model.lambda_U_lower_than_P[0], 8, atol=0.5)
        
        # Fit a purely additive function
        f = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y = f + noise
        model.fit(X, y, y_var=y_var)
        assert np.all(model.a_values > 20)
        assert np.all(model.lambda_U_lower_than_P > 0)
        assert np.all(model.lambda_U_lower_than_P[:-1] < 1e-2)
        assert np.allclose(model.lambda_U_lower_than_P[-1], 8, atol=0.5)
        
        # Fit a pure local interaction
        f = np.array([1, -1, -1, 1, 1, -1, -1, 1])
        y = f + noise
        model.fit(X, y, y_var=y_var)
        assert np.all(model.a_values[-1] < 10)
        assert np.all(model.a_values[:-1] > 20)
        assert np.all(model.lambda_U_lower_than_P < 1e-2)
        
        # More complex interactions
        f = np.array([0, 1, 1, 3, 1, 2, 2, 4])
        y = f + noise
        model.fit(X, y, y_var)
        assert np.all(model.a_values[:2] > 20)
        assert np.allclose(model.a_values[-1], 0.5, rtol=1)
        assert np.allclose(model.lambda_U_lower_than_P, [24.5, 4.5, 4.5, 2], atol=1)
    
    def test_fit_noise_incomplete(self):
        np.random.seed(1234)
        X = np.array(["AAA", "AAB", "ABA", "BAA", "BAB", "BBA"])
        y_var = np.full(X.shape, 0.005)
        y_sd = np.sqrt(y_var)
        noise = np.random.normal(0, y_sd)
        model = LocalEpistasisRegression(
            seq_length=3, n_alleles=2, genotypes=X
        )

        # Fit a purely additive function
        f = np.array([0, 1, 1.0, 0, 1, 1.0])
        y = f + noise
        model.fit(X, y, y_var)
        assert np.all(model.a_values > 0)
        assert np.all(model.a_values[:2] > 1e16)
        assert np.all(model.lambda_U_lower_than_P > 0)
        assert np.all(model.lambda_U_lower_than_P[0] > 1)
        assert np.all(model.lambda_U_lower_than_P[-1] < 1e-12)



if __name__ == "__main__":
    sys.argv = ["", "MEITests"]
    unittest.main()
