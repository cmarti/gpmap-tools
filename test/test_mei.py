#!/usr/bin/env python
import sys
import unittest

import numpy as np
from scipy.sparse.linalg import aslinearoperator
from scipy.stats import pearsonr

from gpmap.inference import (
    GaussianProcessRegressor,
    MinimizerRegressor,
    MinimumEpistasisInterpolator,
)
from gpmap.linop import VarianceComponentKernel


class MEITests(unittest.TestCase):
    def test_interpolation(self):
        # Compute posterior mean with no epistasis
        model = MinimumEpistasisInterpolator(seq_length=2, n_alleles=2, P=2)
        X = np.array(["AA", "AB", "BA"])
        y = np.array([0, 1, 1.0])
        model.set_data(X, y)
        f_pred = model.calc_posterior_mean()
        assert f_pred[-1] == 2.0
        assert model.calc_loss_prior(f_pred) == 0.0

        # Ensure smoothing does not change predictions
        f_pred = model.smooth(f_pred)
        assert np.allclose(f_pred, [0, 1, 1, 2])
        assert model.calc_loss_prior(f_pred) == 0.0

        # Compute posterior variance under a=1
        try:
            model.calc_posterior_covariance()
        except ValueError:
            pass
        model.set_a(1.0)
        Sigma = model.calc_posterior_covariance()
        Sigma = Sigma @ np.eye(Sigma.shape[1])
        m = Sigma.copy()
        m[-1, -1] -= 1
        assert np.allclose(m, 0)

        # Compute posterior mean with epistasis
        model = MinimumEpistasisInterpolator(seq_length=3, n_alleles=2, P=2)
        X = np.array(["AAA", "ABB", "BAA", "BBB"])
        y = np.array([1, 0, 0, 1])
        model.set_data(X, y)

        # Check uniqueness of solution error
        try:
            f_pred = model.calc_posterior()[0]
        except ValueError:
            pass

        # Ensure epistasis is larger than 0
        model.set_data(X, y)
        X = np.array(["AAA", "AAB", "ABA", "BAA", "BBB"])
        y = np.array([1, 0, 0, 0, 1])
        model.set_data(X, y)
        f_pred = model.calc_posterior()[0]
        cost1 = model.calc_loss_prior(f_pred)
        assert np.allclose(y, model.likelihood.Xop @ f_pred)
        assert cost1 > 1e-16

        # Ensure smoothing decreases epistasis
        f_pred_smoothed = model.smooth(f_pred)
        cost2 = model.calc_loss_prior(f_pred_smoothed)
        Z = model.likelihood.Zop
        X = model.likelihood.Xop
        assert np.allclose(Z @ f_pred_smoothed, Z @ f_pred)
        assert not np.allclose(y, X @ f_pred_smoothed)
        assert cost1 > 0
        assert cost2 < cost1

        # Compute posterior variance with estimated a from MAP
        X = np.array(["AAA", "BBA", "ABB", "BAA", "AAB", "BBB"])
        y = np.array([1, 0, 0, 0, 0, 1])
        try:
            model.calc_posterior_covariance()
        except ValueError:
            pass
        model.fit(X, y)
        Sigma = model.calc_posterior_covariance()
        Sigma = Sigma @ np.eye(Sigma.shape[1])
        m = Sigma.copy()
        m[2, 2] -= 0.444444444
        m[-3, -3] -= 0.44444444
        assert np.allclose(m, 0)

    def test_minimizer(self):
        X = np.array(["AA", "AB", "BA", "BB"])
        y = np.array([0, 0.9, 1.0, 2.1])
        y_var = np.array([0.1] * 4)

        # With standard GP formulation
        kernel = VarianceComponentKernel(2, 2, lambdas=np.array([10, 1, 0.1]))
        model1 = GaussianProcessRegressor(kernel)
        model1.set_data(X, y, y_var)
        mu1, Sigma1 = model1.calc_posterior()

        # With regularizer formulation
        K_inv = np.linalg.inv(kernel @ np.eye(4))
        C = aslinearoperator(K_inv)
        model2 = MinimizerRegressor(seq_length=2, n_alleles=2)
        model2.set_data(X, y, y_var)
        model2.C = C
        mu2, Sigma2 = model2.calc_posterior()

        # With incomplete data
        X = np.array(["AA", "AB", "BA"])
        y = np.array([0, 0.9, 1.0])
        y_var = np.array([0.1] * 3)

        model1.set_data(X, y, y_var)
        mu1, Sigma1 = model1.calc_posterior()

        model2.set_data(X, y, y_var)
        mu2, Sigma2 = model2.calc_posterior()

        assert np.allclose(mu1, mu2)
        assert np.allclose(Sigma1 @ np.eye(4), Sigma2 @ np.eye(4))

        # Ensure predict methods return same values
        pred1 = model1.predict(calc_variance=True)
        pred2 = model2.predict(calc_variance=True)
        assert np.allclose(pred1, pred2)

        # Run on larger simulated dataset
        np.random.seed(0)
        n_alleles, seq_length = 4, 5
        lambdas = np.array([1e3, 1e3, 1e2, 1e1, 1e0, 1e-1])
        kernel = VarianceComponentKernel(n_alleles, seq_length, lambdas=lambdas)
        model1 = GaussianProcessRegressor(kernel, cg_rtol=1e-5)
        model1.define_space(seq_length=seq_length, n_alleles=n_alleles)
        f, X, y, y_var = model1.simulate(y_var=0.1, p_missing=0.1)
        model1.set_data(X, y, y_var)
        mu1, Sigma1 = model1.calc_posterior()
        r1 = pearsonr(mu1, f)[0]
        assert r1 > 0.4

        # With operator inverse method
        model2 = MinimizerRegressor(seq_length, n_alleles, cg_rtol=1e-5)
        model2.set_data(X, y, y_var)
        model2.C = kernel.inv()
        mu2, Sigma2 = model2.calc_posterior()
        assert np.allclose(mu1, mu2, atol=1e-2)

    def test_regression(self):
        # Partial dataset that can recapitulate MEI
        X = np.array(["AA", "AB", "BA"])
        y = np.array([0, 1, 1.0])
        y_var = np.array([0.1] * 3)

        model = MinimumEpistasisInterpolator(genotypes=X, a=10, P=2)
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert np.allclose(mu, [0, 1, 1, 2])
        assert Sigma[0, 0] < Sigma[3, 3]

        # Complete dataset with epistasis
        X = np.array(["AA", "AB", "BA", "BB"])
        y = np.array([0, 0.9, 1.0, 2.1])
        y_var = np.array([0.1] * 4)

        model = MinimumEpistasisInterpolator(genotypes=X, a=10, P=2)
        model.set_data(X, y, y_var)
        mu, _ = model.calc_posterior()

        # Check that the inferred landscape is less epistatic
        # than the data: action of the prior
        cost1 = model.calc_loss_prior(y) / model.a
        cost2 = model.calc_loss_prior(mu) / model.a
        assert cost1 > cost2

        # Check that epistasis decreases as a increases
        model = MinimumEpistasisInterpolator(genotypes=X, a=100, P=2)
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        cost3 = model.calc_loss_prior(mu) / model.a
        assert cost3 < cost2

        # Check predict function works as expected
        post_var = np.diag(Sigma @ np.eye(4))
        pred = model.predict(calc_variance=True)
        assert np.allclose(mu, pred["f"])
        assert np.allclose(post_var, pred["f_var"])

    def test_regression_fit(self):
        # Simulate data
        np.random.seed(0)
        n_alleles, seq_length = 4, 6
        a = 100
        model = MinimumEpistasisInterpolator(
            P=2,
            n_alleles=n_alleles,
            seq_length=seq_length,
            a=a,
        )
        f, X, y, y_var = model.simulate(y_var=1.0)
        idx = np.random.uniform(size=X.shape[0]) < 0.98
        X_test, f_test_true = X[~idx], f[~idx]
        X, y, y_var = X[idx], y[idx], y_var[idx]

        # Make interpolation predictions
        model.set_data(X, y)
        pred = model.predict()
        assert np.allclose(pred.loc[idx, "f"], y)

        r = pearsonr(pred.loc[X_test, "f"], f_test_true)[0]
        assert r > 0.5

        # Make predictions with noisy data
        model.set_data(X, y, y_var)
        pred = model.predict()
        r = pearsonr(pred.loc[X_test, "f"], f_test_true)[0]
        assert r > 0.5

        # Evaluate coverage of posterior intervals
        pred = model.predict(X_test, calc_variance=True)
        coverage = np.mean(
            (pred["ci_95_lower"] < f_test_true)
            & (f_test_true < pred["ci_95_upper"])
        )
        assert coverage > 0.9

    def test_mei_fit(self):
        a_true = 100
        np.random.seed(0)
        model = MinimumEpistasisInterpolator(
            seq_length=5, alphabet_type="dna", a=a_true
        )
        f, X = model.simulate(p_missing=0.1)[:2]
        idx = model.get_obs_idx(X)
        model.fit(X, f[idx])
        assert np.allclose(model.a, a_true, rtol=0.2)

    def test_mei_predict(self):
        np.random.seed(0)
        model = MinimumEpistasisInterpolator(
            seq_length=5, alphabet_type="dna", a=100
        )
        f, X, y, y_var = model.simulate(y_var=0.1, p_missing=0.1)
        idx = model.genotype_idxs.loc[X]
        X_test = np.delete(model.genotypes, idx)
        f_test = np.delete(f, idx)

        # Interpolation solution
        model.set_data(X, y)
        pred = model.predict(X_test)
        r = pearsonr(pred["f"], f_test)[0]
        assert r > 0.5

        # Interpolation solution with variances
        model = MinimumEpistasisInterpolator(
            seq_length=5, alphabet_type="dna", a=100
        )
        model.set_data(X, y)
        pred = model.predict(X_test, calc_variance=True)
        r = pearsonr(pred["f"], f_test)[0]
        p = np.mean(
            (pred["ci_95_lower"] < f_test) & (f_test < pred["ci_95_upper"])
        )
        assert r > 0.5
        assert p > 0.9

        # GP with precision matrix and experimental variances
        model = MinimumEpistasisInterpolator(
            seq_length=5, alphabet_type="dna", a=100
        )
        model.set_data(X, y, y_var)
        pred = model.predict(X_test, calc_variance=True)
        r = pearsonr(pred["f"], f_test)[0]
        p = np.mean(
            (pred["ci_95_lower"] < f_test) & (f_test < pred["ci_95_upper"])
        )
        assert r > 0.5
        assert p > 0.9


if __name__ == "__main__":
    sys.argv = ["", "MEITests"]
    unittest.main()
