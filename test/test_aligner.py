#!/usr/bin/env python
import unittest

import numpy as np

from gpmap.aligner import (
    DeltaPKernelAligner,
    VCKernelAligner,
    DeltaUKernelAligner,
    VUKernelAligner,
    ConnectednessKernelAligner,
)
from gpmap.linop import (
    ConnectednessProjectionOpererator,
    ProjectionOperator,
    calc_covariance_distance,
    calc_variance_components,
)


class KernelAlignerTest(unittest.TestCase):
    def test_frobenius_loss(self):
        # Additive covariances
        a, sl = 2, 2
        cov, ns = [0.5, 0, -0.5], [4, 8, 4]
        log_lambdas = np.array([-16, 0, -16])
        aligner = VCKernelAligner(a, sl)
        aligner.set_data(cov, ns)
        loss = aligner.calc_loss(log_lambdas)
        assert loss < 1e-12

        # With simulated data from a pure pairwise model
        np.random.seed(1)
        a, sl, k = 4, 5, 2
        P = ProjectionOperator(a, sl, k=k)
        log_lambdas = np.full(sl + 1, -16.0)
        log_lambdas[k] = 0
        y = P @ np.random.normal(size=P.shape[1])
        cov, ns = calc_covariance_distance(y, a, sl)
        exp_lambdas = calc_variance_components(y, a, sl)
        exp_log_lambdas = np.log(np.abs(exp_lambdas))

        aligner = VCKernelAligner(a, sl)
        aligner.set_data(cov, ns)
        loss = aligner.calc_loss(exp_log_lambdas)
        assert loss < 1e-12

        # With the true covariances
        a, sl, k = 4, 5, 2
        log_lambdas_true = np.full(sl + 1, -16)
        log_lambdas_true[k] = 1
        cov_true = aligner.predict(np.exp(log_lambdas_true))
        ns = np.ones_like(cov_true)
        aligner.set_data(cov_true, ns)
        loss, grad = aligner.calc_loss(log_lambdas_true, return_grad=True)
        assert loss < 1e-12
        assert np.allclose(grad, 0, rtol=1e-10)

    def test_VC_kernel_alignment(self):
        # Simulate data
        np.random.seed(1)
        beta = 1e5
        sigma2 = 0.1
        a, sl, rho = 4, 5, 0.5
        P = 5 * ConnectednessProjectionOpererator(a, sl, rho=rho).matrix_sqrt()
        y_true = P @ np.random.normal(size=P.shape[1])
        cov_true, ns = calc_covariance_distance(y_true, a, sl)
        lambdas_true = calc_variance_components(y_true, a, sl)
        y = np.random.normal(y_true, np.sqrt(sigma2))
        cov_obs, ns = calc_covariance_distance(y, a, sl)

        # Define kernel aligner and fit unregularized model
        aligner = VCKernelAligner(a, sl)
        lambdas_star_1 = aligner.fit(cov_true, ns)
        cov_pred = aligner.predict(lambdas_star_1)
        loss, grad = aligner.calc_loss(np.log(lambdas_star_1), return_grad=True)
        assert loss < 1e-12
        assert np.allclose(grad, 0, rtol=1e-10)
        assert np.allclose(cov_true, cov_pred, rtol=0.01)
        assert np.allclose(lambdas_true, lambdas_star_1, rtol=0.5)

        # Align with beta > 0
        aligner = VCKernelAligner(a, sl, beta=beta)
        lambdas_star_2 = aligner.fit(cov_true, ns)
        cov_pred = aligner.predict(lambdas_star_2)
        assert np.allclose(cov_true, cov_pred, rtol=0.01)
        assert np.allclose(lambdas_true, lambdas_star_2, rtol=0.5)

        # Ensure loss is lower than unregularized fit
        loss1 = aligner.calc_loss(np.log(lambdas_star_1))
        loss2 = aligner.calc_loss(np.log(lambdas_star_2))
        assert loss2 < loss1

        # Add known measurement error sigma^2
        aligner = VCKernelAligner(a, sl)
        lambdas_star_1 = aligner.fit(cov_obs, ns)
        lambdas_star_2 = aligner.fit(cov_obs, ns, sigma2=sigma2)
        cov_obs_pred = aligner.predict(lambdas_star_2 + sigma2)
        assert not np.allclose(lambdas_star_1, lambdas_star_2, rtol=0.05)
        assert np.allclose(cov_obs, cov_obs_pred, rtol=0.05)
        assert np.allclose(lambdas_true, lambdas_star_2, rtol=0.5)

        # Align with beta > 0
        aligner = VCKernelAligner(a, sl, beta=beta)
        lambdas_star_3 = aligner.fit(cov_obs, ns, sigma2=sigma2)
        cov_pred = aligner.predict(lambdas_star_3)
        assert np.allclose(lambdas_true, lambdas_star_3, rtol=0.5)

        # Ensure loss is lower than unregularized fit
        loss2 = aligner.calc_loss(np.log(lambdas_star_2))
        loss3 = aligner.calc_loss(np.log(lambdas_star_3))
        assert loss3 < loss2
    
    def test_DeltaP_kernel_alignment(self):
        a, sl = 4, 5
        aligner = DeltaPKernelAligner(a, sl, P=2)
        a_true = 0.01
        cov_true = aligner.predict(a_true)
        ns = np.ones_like(cov_true)
        
        a_star = aligner.fit(cov_true, ns)
        cov_pred = aligner.predict(a_star)
        loss = aligner.frobenius_norm(np.log(a_star))
        assert loss < 1e-12
        assert np.allclose(cov_true, cov_pred, rtol=0.01)
        assert np.allclose(a_star, a_true)

    def test_VU_kernel_alignment(self):
        # Ensure inner matrix works well
        aligner = VUKernelAligner(n_alleles=3, seq_length=2)

        c0 = aligner.W_sU @ np.array([1, 0, 0, 0])
        assert np.allclose(c0, 1 / 9)

        c1 = aligner.W_sU @ np.array([0, 1, 1, 0])
        assert np.allclose(c1, [4 / 9, 1 / 9, 1 / 9, -2 / 9])

        c2 = aligner.W_sU @ np.array([0, 0, 0, 1])
        assert np.allclose(c2, [4 / 9, -2 / 9, -2 / 9, 1 / 9])

        log_lambda_U = np.array([1, 0.5, 0, -1])
        cov = aligner.calc_cov(log_lambda_U)
        ns = np.ones_like(cov)
        log_lambda_U_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_lambda_U_hat, log_lambda_U)

        # Try now with a larger case
        aligner = VUKernelAligner(n_alleles=4, seq_length=3)
        log_lambda_U = np.array([1, 0.5, 0, 0.25, -1, -2, -1.5, -3])
        cov = aligner.calc_cov(log_lambda_U)
        ns = np.ones_like(cov)
        log_lambda_U_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_lambda_U_hat, log_lambda_U)

    def test_DeltaU_kernel_alignment(self):
        # Ensure inner matrix works well
        aligner = DeltaUKernelAligner(n_alleles=3, seq_length=2, P=2)

        c0 = aligner.W_sU @ np.array([1, 0, 0, 0])
        assert np.allclose(c0, 1 / 9)

        c1 = aligner.W_sU @ np.array([0, 1, 1, 0])
        assert np.allclose(c1, [4 / 9, 1 / 9, 1 / 9, -2 / 9])

        c2 = aligner.W_sU @ np.array([0, 0, 0, 1])
        assert np.allclose(c2, [4 / 9, -2 / 9, -2 / 9, 1 / 9])

        # Test a_to_lambda_U
        lambda_U = aligner.a_to_lambda_U([1])
        assert np.allclose(lambda_U, [0, 0, 0, 1 / 9])

        log_a = np.array([-5.]) 
        cov = aligner.calc_cov(log_a)
        ns = np.ones_like(cov)
        log_a_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_a_hat, log_a)

        # Try now with a larger case
        aligner = DeltaUKernelAligner(n_alleles=4, seq_length=3, P=2)
        log_a = np.array([-5.0, -2.0, -1.0])
        cov = aligner.calc_cov(log_a)
        ns = np.ones_like(cov)
        log_a_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_a_hat, log_a)

    def test_connectedness_kernel_alignment(self):
        aligner = ConnectednessKernelAligner(n_alleles=3, seq_length=2)
        logit_mu, log_mu0 = np.array([0., 0]), np.array([0.])
        cov = aligner.predict(logit_mu, log_mu0)
        ns = np.ones_like(cov)
        assert(np.allclose(cov[1], cov[2]))

        log_mu0_hat, logit_mu_hat = aligner.fit(cov, ns)
        assert np.allclose(log_mu0_hat, log_mu0)
        assert np.allclose(logit_mu_hat, logit_mu)

        # Try now with a larger case
        aligner = ConnectednessKernelAligner(n_alleles=4, seq_length=4)
        logit_mu, log_mu0 = np.array([0.5, -0.5, 0, -1]), np.array([0.0])
        cov = aligner.predict(logit_mu, log_mu0)
        ns = np.ones_like(cov)
        log_mu0_hat, logit_mu_hat = aligner.fit(cov, ns)
        assert np.allclose(log_mu0_hat, log_mu0)
        assert np.allclose(logit_mu_hat, logit_mu)


if __name__ == "__main__":
    import sys

    sys.argv = ["", "KernelAlignerTest"]
    unittest.main()
