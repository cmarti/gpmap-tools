#!/usr/bin/env python
import sys
import unittest
import numpy as np


from scipy.stats import pearsonr
from scipy.sparse.linalg import aslinearoperator

from gpmap.src.linop import ConnectednessKernel
from gpmap.src.inference import (MinimumEpistasisInterpolator, MinimumEpistasisRegression,
                                 MinimizerRegressor, GaussianProcessRegressor,
                                 VCregression)


class MEITests(unittest.TestCase):
    def test_interpolation(self):
        model = MinimumEpistasisInterpolator(P=2)
        X = np.array(['AA', 'AB', 'BA'])
        y = np.array([0, 1, 1.])
        model.set_data(X, y)
        y_pred = model.calc_posterior_mean()
        assert(y_pred[-1] == 2.)
        assert(model.calc_cost(y_pred) == 0.)
        
        y_pred = model.smooth(y_pred)
        assert(np.allclose(y_pred, [0, 1, 1, 2]))
        assert(model.calc_cost(y_pred) == 0.)
        
        model = MinimumEpistasisInterpolator(P=2)
        X = np.array(['AAA', 'ABB', 'BAA', 'BBB'])
        y = np.array([1, 0, 0, 1])
        model.set_data(X, y)
        y_pred = model.calc_posterior_mean()
        cost1 = model.calc_cost(y_pred)
        assert(np.allclose(y, y_pred[model.obs_idx]))
        assert(cost1 > 0.)
        
        y_pred_smoothed = model.smooth(y_pred)
        cost2 = model.calc_cost(y_pred_smoothed)
        assert(np.allclose(y_pred_smoothed[model.pred_idx],
                           y_pred[model.pred_idx]))
        assert(not np.allclose(y, y_pred_smoothed[model.obs_idx]))
        assert(cost1 > 0)
        assert(cost2 < cost1)
    
    def test_minimizer(self):
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = np.array([0, 0.9, 1., 2.1])
        y_var = np.array([0.1] * 4)
        
        # With standard GP formulation
        kernel = ConnectednessKernel(2, 2, rho=np.array([0.2, 0.2]))
        model1 = GaussianProcessRegressor(kernel)
        model1.set_data(X, y, y_var)
        mu1, Sigma1 = model1.calc_posterior()
        
        # With regularizer formulation
        C = aslinearoperator(np.linalg.inv(kernel @ np.eye(4)))
        model2 = MinimizerRegressor()
        model2.set_data(X, y, y_var)
        model2.C = C
        mu2, Sigma2 = model2.calc_posterior()
        
        # With incomplete data
        X = np.array(['AA', 'AB', 'BA'])
        y = np.array([0, 0.9, 1.])
        y_var = np.array([0.1] * 3)
        
        model1.set_data(X, y, y_var)
        mu1, Sigma1 = model1.calc_posterior()
        
        model2.set_data(X, y, y_var)
        mu2, Sigma2 = model2.calc_posterior()
        assert(np.allclose(mu1, mu2))

        assert(np.allclose(mu1, mu2))
        assert(np.allclose(Sigma1 @ np.eye(4), Sigma2 @ np.eye(4)))

        # Ensure predict methods return same values
        pred1 = model1.predict(calc_variance=True)
        pred2 = model2.predict(calc_variance=True)
        assert(np.allclose(pred1, pred2))
        
    def test_regression(self):
        # Partial dataset that can recapitulate MEI
        X = np.array(['AA', 'AB', 'BA'])
        y = np.array([0, 1, 1.])
        y_var = np.array([0.1] * 3)
        
        model = MinimumEpistasisRegression(a=10, P=2)
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        Sigma = Sigma @ np.eye(4)
        assert(np.allclose(mu, [0, 1, 1, 2]))
        assert(Sigma[0, 0] < Sigma[3, 3])
        
        # Complete dataset with epistasis
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        y = np.array([0, 0.9, 1., 2.1])
        y_var = np.array([0.1] * 4)
        
        model = MinimumEpistasisRegression(a=10, P=2)
        model.set_data(X, y, y_var)
        mu, _ = model.calc_posterior()
        
        # Check that the inferred landscape is less epistatic
        # than the data: action of the prior
        cost1 = model.calc_loss_prior(y) / model.a
        cost2 = model.calc_loss_prior(mu) / model.a
        assert(cost1 > cost2)
        
        # Check that epistasis decreases as a increases
        model = MinimumEpistasisRegression(a=100, P=2)
        model.set_data(X, y, y_var)
        mu, Sigma = model.calc_posterior()
        cost3 = model.calc_loss_prior(mu) / model.a
        assert(cost3 < cost2)

        # Check predict function works as expected
        post_var = np.diag(Sigma @ np.eye(4))
        pred = model.predict(calc_variance=True)
        assert(np.allclose(mu, pred['y']))
        assert(np.allclose(post_var, pred['y_var']))

    def test_regression_fit(self):
        # Simulate data
        np.random.seed(0)
        lambdas = np.array([1, 200, 20, 2, 0.2, 0.02])
        a, length = 4, lambdas.shape[0]-1
        vc = VCregression(n_alleles=a, seq_length=length, lambdas=lambdas)
        data = vc.simulate(sigma=1)

        idx = np.random.uniform(size=data.shape[0]) < 0.8
        train, test = data.loc[idx, :], data.loc[~idx, :]
        
        X = train.index.values
        y = train['y']
        y_var = train['y_var']
        X_test = test.index.values

        # Fit model with empirical epistatic coeffs
        model = MinimumEpistasisRegression(P=2, n_alleles=a, seq_length=length)
        model.fit(X, y, y_var, cross_validation=False)
        
        pred = model.predict(X_test, calc_variance=True)
        r = pearsonr(pred['y'], test['y_true'])[0]
        calibration = np.mean((pred['ci_95_lower'] < test['y_true']) &
                              (test['y_true'] < pred['ci_95_upper']))
        assert(r > 0.9)
        assert(calibration > 0.9)

        # Fit model with cross validation
        model = MinimumEpistasisRegression(P=2, n_alleles=a, seq_length=length)
        model.fit(X, y, y_var, cross_validation=True)
        
        pred = model.predict(X_test, calc_variance=True)
        r = pearsonr(pred['y'], test['y_true'])[0]
        calibration = np.mean((pred['ci_95_lower'] < test['y_true']) &
                              (test['y_true'] < pred['ci_95_upper']))
        assert(r > 0.9)
        assert(calibration > 0.9)
        
        
if __name__ == '__main__':
    sys.argv = ['', 'MEITests']
    unittest.main()
