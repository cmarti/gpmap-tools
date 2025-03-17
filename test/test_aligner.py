#!/usr/bin/env python
import unittest

import numpy as np

from gpmap.aligner import VCKernelAligner, RhoKernelAligner
from gpmap.linop import (RhoProjectionOperator, ProjectionOperator,
                             calc_covariance_vjs,
                             calc_covariance_distance, calc_variance_components)


class KernelAlignerTest(unittest.TestCase):
    def test_frobenius_loss(self):
        # Additive covariances
        a, l = 2, 2
        cov, ns = [0.5, 0, -0.5], [4, 8, 4]
        log_lambdas = np.array([-16, 0, -16])
        aligner = VCKernelAligner(a, l)
        aligner.set_data(cov, ns)
        loss = aligner.calc_loss(log_lambdas)
        assert(loss < 1e-12)
        
        # With simulated data from a pure pairwise model
        np.random.seed(1)
        a, l, k = 4, 5, 2
        P = ProjectionOperator(a, l, k=k)
        log_lambdas = np.full(l + 1, -16.)
        log_lambdas[k] = 0
        y = P @ np.random.normal(size=P.shape[1])
        cov, ns = calc_covariance_distance(y, a, l)
        exp_lambdas = calc_variance_components(y, a, l)
        exp_log_lambdas = np.log(np.abs(exp_lambdas))
        
        aligner = VCKernelAligner(a, l)
        aligner.set_data(cov, ns)
        loss = aligner.calc_loss(exp_log_lambdas)
        assert(loss < 1e-12)
        
        # With the true covariances
        a, l, k = 4, 5, 2
        log_lambdas_true = np.full(l + 1, -16)
        log_lambdas_true[k] = 1
        cov_true = aligner.predict(np.exp(log_lambdas_true))
        aligner.set_data(cov_true, ns)
        loss, grad = aligner.calc_loss(log_lambdas_true, return_grad=True)
        assert(loss < 1e-12)
        assert(np.allclose(grad, 0, rtol=1e-10))

    def test_vc_kernel_alignment(self):
        # Simulate data
        np.random.seed(1)
        beta = 1e5
        sigma2 = 0.1
        a, l, rho = 4, 5, 0.5
        P = 5 * RhoProjectionOperator(a, l, rho=rho).matrix_sqrt()
        y_true = P @ np.random.normal(size=P.shape[1])
        cov_true, ns = calc_covariance_distance(y_true, a, l)
        lambdas_true = calc_variance_components(y_true, a, l)
        y = np.random.normal(y_true, np.sqrt(sigma2))
        cov_obs, ns = calc_covariance_distance(y, a, l)

        # Define kernel aligner and fit unregularized model
        aligner = VCKernelAligner(a, l)
        lambdas_star_1 = aligner.fit(cov_true, ns)
        cov_pred = aligner.predict(lambdas_star_1)
        loss, grad = aligner.calc_loss(np.log(lambdas_star_1), return_grad=True)
        assert(loss < 1e-12)
        assert(np.allclose(grad, 0, rtol=1e-10))
        assert(np.allclose(cov_true, cov_pred, rtol=0.01))
        assert(np.allclose(lambdas_true, lambdas_star_1, rtol=0.5))
        
        # Align with beta > 0
        aligner = VCKernelAligner(a, l, beta=beta)
        lambdas_star_2 = aligner.fit(cov_true, ns)
        cov_pred = aligner.predict(lambdas_star_2)
        assert(np.allclose(cov_true, cov_pred, rtol=0.01))
        assert(np.allclose(lambdas_true, lambdas_star_2, rtol=0.5))

        # Ensure loss is lower than unregularized fit
        loss1 = aligner.calc_loss(np.log(lambdas_star_1))
        loss2 = aligner.calc_loss(np.log(lambdas_star_2))
        assert(loss2 < loss1)

        # Add known measurement error sigma^2
        aligner = VCKernelAligner(a, l)
        lambdas_star_1 = aligner.fit(cov_obs, ns)
        lambdas_star_2 = aligner.fit(cov_obs, ns, sigma2=sigma2)
        cov_obs_pred = aligner.predict(lambdas_star_2 + sigma2)
        assert(not np.allclose(lambdas_star_1, lambdas_star_2, rtol=0.05))
        assert(np.allclose(cov_obs, cov_obs_pred, rtol=0.05))
        assert(np.allclose(lambdas_true, lambdas_star_2, rtol=0.5))

        # Align with beta > 0
        aligner = VCKernelAligner(a, l, beta=beta)
        lambdas_star_3 = aligner.fit(cov_obs, ns, sigma2=sigma2)
        cov_pred = aligner.predict(lambdas_star_3)
        assert(np.allclose(lambdas_true, lambdas_star_3, rtol=0.5))

        # Ensure loss is lower than unregularized fit
        loss2 = aligner.calc_loss(np.log(lambdas_star_2))
        loss3 = aligner.calc_loss(np.log(lambdas_star_3))
        assert(loss3 < loss2)
        
    def test_rho_kernel_alignment(self):
        # Simulate data
        np.random.seed(1)
        log_mu0 = 0
        a, l = 4, 5
        logit_rho0 = np.array([-4, 0, 1., 1, -5])
        rho0 = np.exp(logit_rho0) / (1 + np.exp(logit_rho0))
        P = RhoProjectionOperator(a, l, rho=np.sqrt(rho0))
        y = P @ np.random.normal(size=P.shape[1])
        y = y / y.std()
        covs, ns, sites_matrix = calc_covariance_vjs(y, a, l)
        
        # Define kernel aligner and fit to empirical covariances
        aligner = RhoKernelAligner(a, l)
        log_mu, logit_rho = aligner.fit(covs, ns, sites_matrix)

        # Ensure big differences remain
        assert(np.all(logit_rho[0] < logit_rho[1:4] - 1))
        assert(np.all(logit_rho[-1] < logit_rho[1:4] - 1))

        # Ensure loss is lower than with the generating rhos
        loss0 = aligner.frobenius_norm(aligner.params_to_x(log_mu0, logit_rho0))
        loss = aligner.frobenius_norm(aligner.params_to_x(log_mu, logit_rho))
        assert(loss < loss0)
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'KernelAlignerTest']
    unittest.main()
