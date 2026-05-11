import sys
import unittest
from itertools import product

import numpy as np

from gpmap.inference import LocalEpistasisRegression


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
    
    def test_predict_variable_noise(self):
        # Partial dataset that can recapitulate MEI
        X = np.array(["AA", "AB", "BA"])
        y = np.array([0, 1, 1.0])
        a_values = np.array([1.0])
        lambda_U = np.array([1e3, 10, 10])

        model = LocalEpistasisRegression(
            seq_length=2,
            n_alleles=2,
            genotypes=X,
            a_values=a_values,
            lambda_U_lower_than_P=lambda_U,
        )
        
        # No noise
        model.set_data(X, y)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert np.allclose(mu, [0, 1, 1, 2], atol=0.5)
        assert Sigma[0, 0] < Sigma[3, 3]
        
        # Heteroscedastic noise
        y_var = np.array([0.1, 0.001, 0.05])
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert np.allclose(mu, [0, 1, 1, 2], atol=0.5)
        assert Sigma[0, 0] < Sigma[3, 3]
        
        # Homoskedastic large noise
        y_var = np.array([1, 1, 1])
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert np.allclose(mu, [0, 1, 1, 2], atol=0.5)
        assert Sigma[0, 0] < Sigma[3, 3]
    
    def test_predict_conditioning(self):
        X = np.array(["AA", "AB", "BA"])
        y = np.array([0, 1, 1.0])
        a_values = np.array([1.0])
        
        # Large variance for the constant component induces bad conditioning on K
        lambda_U = np.array([1e5, 10, 10])
        model = LocalEpistasisRegression(
            seq_length=2,
            n_alleles=2,
            genotypes=X,
            a_values=a_values,
            lambda_U_lower_than_P=lambda_U,
        )
        
        y_var = np.array([1e-4]  * 3)
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert np.allclose(mu, [0, 1, 1, 2], atol=0.5)
        assert Sigma[0, 0] < Sigma[3, 3]

    def test_fit_complete(self):
        X = np.array(["".join(x) for x in product(list("AB"), repeat=3)])
        model = LocalEpistasisRegression(
            seq_length=3, n_alleles=2, genotypes=X
        )

        # Fit a constant function
        f = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        model.fit(X, f)
        assert np.all(model.a_values > 50)
        assert np.all(model.lambda_U_lower_than_P >= 0)
        assert np.all(model.lambda_U_lower_than_P[1:] < 1e-2)
        assert np.allclose(model.lambda_U_lower_than_P[0], 8)

        # Fit a purely additive function
        f = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        model.fit(X, f)
        assert np.all(model.a_values > 50)
        assert np.all(model.lambda_U_lower_than_P >= 0)
        assert np.all(model.lambda_U_lower_than_P[:-1] < 1e-2)
        assert np.allclose(model.lambda_U_lower_than_P[-1], 8)

        # Fit a pure local interaction
        f = np.array([1, -1, -1, 1, 1, -1, -1, 1])
        model.fit(X, f)
        assert np.all(model.a_values[-1] < 10)
        assert np.all(model.a_values[:-1] > 50)
        assert np.all(model.lambda_U_lower_than_P < 1e-2)

        # More complex interactions
        f = np.array([0, 1, 1, 3, 1, 2, 2, 4])
        model.fit(X, f)
        assert np.all(model.a_values[:2] > 50)
        assert np.allclose(model.a_values[-1], 0.5, rtol=1)
        assert np.allclose(model.lambda_U_lower_than_P, [24.5, 4.5, 4.5, 2])

    def test_fit_complete_noise(self):
        np.random.seed(1234)
        X = np.array(["".join(x) for x in product(list("AB"), repeat=3)])
        y_var = np.full(X.shape, 0.005)
        y_sd = np.sqrt(y_var)
        noise = np.random.normal(0, y_sd)
        model = LocalEpistasisRegression(
            seq_length=3, n_alleles=2, genotypes=X
        )

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
        assert np.allclose(
            model.lambda_U_lower_than_P, [24.5, 4.5, 4.5, 2], atol=1
        )

    def test_get_parameters(self):
        X = np.array(["".join(x) for x in product(list("AB"), repeat=3)])
        model = LocalEpistasisRegression(
            seq_length=3, n_alleles=2, genotypes=X
        )
        f = np.array([0, 1, 1, 3, 1, 2, 2, 4])
        model.fit(X, f)

        cor_df = model.get_empirical_pred_correlations_df()
        assert cor_df.shape == (8, 5)
        assert np.all(
            cor_df.columns == ["d", "n", "emp_cor", "pred_cor", "d_jittered"]
        )
        assert np.allclose(cor_df['emp_cor'], cor_df['pred_cor'])

        a_df = model.get_a_values()
        assert a_df.shape == (3, 4)
        assert np.all(
            a_df.columns == ["site1", "site2", "a_U", "interaction_strength"]
        )

        lambda_U_df = model.get_lambda_U_values()
        assert lambda_U_df.shape == (4, 3)
        assert np.all(lambda_U_df.columns == ["U", "k", "lambda_U"])

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
        assert np.all(model.a_values[:2] > 50)
        assert np.all(model.lambda_U_lower_than_P > 0)
        assert np.all(model.lambda_U_lower_than_P[0] > 1)
        assert np.all(model.lambda_U_lower_than_P[-1] < 1e-2)


if __name__ == "__main__":
    sys.argv = ["", "LERTests"]
    unittest.main()
