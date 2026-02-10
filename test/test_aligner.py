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


class KernelAlignerTest(unittest.TestCase):
    def test_frobenius_loss(self):
        a, sl = 2, 2
        cov, ns = [0.5, 0, -0.5], [4, 8, 4]
        log_lambdas = np.array([-16, 0, -16])
        aligner = VCKernelAligner(a, sl)
        aligner.set_data(cov, ns)
        loss, grad = aligner.calc_loss(log_lambdas, return_grad=True)
        assert loss < 1e-12
        assert np.allclose(grad, 0, rtol=1e-10)
        
        log_lambdas = np.array([0, 0, 0])
        loss, grad = aligner.calc_loss(log_lambdas, return_grad=True)
        assert loss > 1
        assert np.allclose(grad, [2, 0, 2], rtol=1e-10)

    def test_VC_kernel_alignment_predict(self):
        aligner = VCKernelAligner(n_alleles=3, seq_length=2)

        c0 = aligner.predict(np.array([1, 0, 0]))
        assert np.allclose(c0, 1 / 9)

        c1 = aligner.predict(np.array([0, 1, 0]))
        assert np.allclose(c1, [4 / 9, 1 / 9, -2 / 9])

        c2 = aligner.predict(np.array([0, 0, 1]))
        assert np.allclose(c2, [4 / 9, -2 / 9, 1 / 9])

    def test_VC_kernel_alignment_fit(self):
        aligner = VCKernelAligner(n_alleles=3, seq_length=2)
        log_lambda_k = np.array([1, 0, -1])
        cov = aligner.calc_cov(log_lambda_k)
        ns = np.ones_like(cov)
        log_lambda_k_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_lambda_k_hat, log_lambda_k)

        aligner = VCKernelAligner(n_alleles=4, seq_length=3)
        log_lambda_k = np.array([1, 0, -1, -2])
        cov = aligner.calc_cov(log_lambda_k)
        ns = np.ones_like(cov)
        log_lambda_k_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_lambda_k_hat, log_lambda_k)
    
    def test_DeltaP_kernel_alignment_fit(self):
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

    def test_VU_kernel_alignment_predict(self):
        aligner = VUKernelAligner(n_alleles=3, seq_length=2)

        c0 = aligner.predict(np.array([1, 0, 0, 0]))
        assert np.allclose(c0, 1 / 9)

        c1 = aligner.predict(np.array([0, 1, 1, 0]))
        assert np.allclose(c1, [4 / 9, 1 / 9, 1 / 9, -2 / 9])

        c2 = aligner.predict(np.array([0, 0, 0, 1]))
        assert np.allclose(c2, [4 / 9, -2 / 9, -2 / 9, 1 / 9])

    def test_VU_kernel_alignment_fit(self):
        aligner = VUKernelAligner(n_alleles=3, seq_length=2)
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

        # Test get_lambda_U
        lambda_U = aligner.get_lambda_U([1])
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
    
    def test_DeltaU_kernel_alignment_include_lower_order(self):
        aligner = DeltaUKernelAligner(n_alleles=3, seq_length=2, P=2,
                                      include_lower_P=True)
        assert aligner.n_a_values == 1
        assert aligner.n_U_lower_than_P == 3
        assert aligner.n_params == 4
        
        x = np.array([-16, -16, 0, 16])
        cov = aligner.calc_cov(x)
        ns = np.ones_like(cov)
        x_hat = np.log(aligner.fit(cov, ns))
        cov2 = aligner.calc_cov(x_hat)
        assert np.allclose(cov, cov2)
        assert np.allclose(np.exp(x[:-1]), np.exp(x_hat[:-1]), atol=1e-4)
        assert np.exp(x[-1]) > 20
        
        x = np.array([-16, -16, -16, -5.0])
        cov = aligner.calc_cov(x)
        ns = np.ones_like(cov)
        x_hat = np.log(aligner.fit(cov, ns))
        cov2 = aligner.calc_cov(x_hat)
        assert np.allclose(np.exp(x), np.exp(x_hat), atol=1e-4)
        assert np.allclose(cov, cov2)

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
