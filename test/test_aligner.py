#!/usr/bin/env python
import unittest

import numpy as np

from functools import partial
from gpmap.aligner import (
    DeltaUtoVUTransform,
    DeltaPtoVkTransform,
    DeltaPKernelAligner,
    VCKernelAligner,
    DeltaUKernelAligner,
    VCUKernelAligner,
    ConnectednessKernelAligner,
)


class KernelAlignerTest(unittest.TestCase):
    def _finite_diff_grad(self, f, x, eps=1e-6):
        grad = np.zeros_like(x)
        for i in range(x.shape[0]):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad

    def test_VC_kernel_frobenius_loss(self):
        a, sl = 2, 2
        cov, ns = np.array([0.5, 0, -0.5]), np.array([4, 8, 4])
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

    def test_VC_kernel_predict(self):
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
        cov = aligner.predict(np.exp(log_lambda_k))
        ns = np.ones_like(cov)
        log_lambda_k_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_lambda_k_hat, log_lambda_k)

        aligner = VCKernelAligner(n_alleles=4, seq_length=3)
        log_lambda_k = np.array([1, 0, -1, -2])
        cov = aligner.predict(np.exp(log_lambda_k))
        ns = np.ones_like(cov)
        log_lambda_k_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_lambda_k_hat, log_lambda_k)

    def test_VC_kernel_alignment_fit_regularized(self):
        aligner = VCKernelAligner(n_alleles=4, seq_length=3)
        log_lambda_k = np.array([1, 0, -1.2, -1.8])
        cov = aligner.predict(np.exp(log_lambda_k))
        ns = np.ones_like(cov)
        log_lambda_k_hat1 = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_lambda_k_hat1, log_lambda_k)

        aligner = VCKernelAligner(n_alleles=4, seq_length=3, beta=10)
        log_lambda_k = np.array([1, 0, -1.2, -1.8])
        cov = aligner.predict(np.exp(log_lambda_k))
        ns = np.ones_like(cov)
        log_lambda_k_hat2 = np.log(aligner.fit(cov, ns))
        assert not np.allclose(log_lambda_k_hat2, log_lambda_k)

        reg1 = aligner.regularizer(log_lambda_k_hat1)
        reg2 = aligner.regularizer(log_lambda_k_hat2)
        assert reg2 < reg1
    
    def test_DeltaP_to_Vk_transform(self):
        transform = DeltaPtoVkTransform(4, 2, P=2)

        x = np.array([-16, -16, 0])
        log_lambda_k = transform(x, return_grad=False)
        lambda_k = np.exp(log_lambda_k)
        assert np.allclose(lambda_k, [0, 0, 1 / 16], atol=1e-6)

        x = np.array([0, 1, 1])
        log_lambda_k, grad = transform(x, return_grad=True)
        assert np.allclose(log_lambda_k, [0, 1, -1 - 2 * np.log(4)])
        assert np.allclose(
            grad, [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        )

        # Test with 3 sites
        transform = DeltaPtoVkTransform(4, 3, P=2)
        x = np.array([-16, -16, 0])
        log_lambda_k, grad = transform(x, return_grad=True)
        lambda_k = np.exp(log_lambda_k)
        exp = np.array([0, 0, 1 / 16.0, 1 / 48.0])
        assert np.allclose(lambda_k, exp, atol=1e-4)
        assert np.allclose(grad, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -1]])

    def test_DeltaP_kernel_alignment_fit(self):
        a, sl = 4, 5
        aligner = DeltaPKernelAligner(a, sl, P=2)
        x_true = np.array([-16, 3, -1])
        cov_true = aligner.predict(x_true)
        ns = np.ones_like(cov_true)
        aligner.set_data(cov_true, ns)

        # Ensure loss and gradients are zero at the right solution
        loss, grad = aligner.calc_loss(x_true, return_grad=True)
        assert(np.allclose(loss, 0))
        assert np.allclose(grad, 0)

        # Train initializing at the true values
        x_hat = np.log(aligner.fit(cov_true, ns, x0=x_true))
        cov_pred = aligner.predict(x_hat)
        loss, grad = aligner.calc_loss(x_hat, return_grad=True)
        assert loss < 1e-12
        assert np.allclose(grad, 0, atol=1e-6)
        assert np.allclose(cov_true, cov_pred, rtol=0.01)
        assert np.allclose(x_true, x_hat)

    def test_VCU_kernel_predict(self):
        aligner = VCUKernelAligner(n_alleles=3, seq_length=2)

        c0 = aligner.predict(np.array([1, 0, 0, 0]))
        assert np.allclose(c0, 1 / 9)

        c1 = aligner.predict(np.array([0, 1, 1, 0]))
        assert np.allclose(c1, [4 / 9, 1 / 9, 1 / 9, -2 / 9])

        c2 = aligner.predict(np.array([0, 0, 0, 1]))
        assert np.allclose(c2, [4 / 9, -2 / 9, -2 / 9, 1 / 9])

    def test_VCU_kernel_alignment_fit(self):
        aligner = VCUKernelAligner(n_alleles=3, seq_length=2)
        log_lambda_U = np.array([1, 0.5, 0, -1])
        cov = aligner.predict(np.exp(log_lambda_U))
        ns = np.ones_like(cov)
        log_lambda_U_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_lambda_U_hat, log_lambda_U)

        aligner = VCUKernelAligner(n_alleles=4, seq_length=3)
        log_lambda_U = np.array([1, 0.5, 0, 0.25, -1, -2, -1.5, -3])
        cov = aligner.predict(np.exp(log_lambda_U))
        ns = np.ones_like(cov)
        log_lambda_U_hat = np.log(aligner.fit(cov, ns))
        assert np.allclose(log_lambda_U_hat, log_lambda_U)

    def test_DeltaU_to_VU_transform(self):
        transform = DeltaUtoVUTransform(4, 2, P=2)

        x = np.array([-16, -16, -16, 0])
        log_lambda_U = transform(x, return_grad=False)
        lambda_U = np.exp(log_lambda_U)
        assert np.allclose(lambda_U, [0, 0, 0, 1 / 16], atol=1e-6)

        x = np.array([0, 1, 1, 1])
        log_lambda_U, grad = transform(x, return_grad=True)
        assert np.allclose(log_lambda_U, [0, 1, 1, -1 - 2 * np.log(4)])
        assert np.allclose(
            grad, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        )

        # Test with 3 sites
        transform = DeltaUtoVUTransform(4, 3, P=2)

        x = np.array([-16, -16, -16, -16, 0, 16, 16])
        log_lambda_U = transform(x, return_grad=False)
        lambda_U = np.exp(log_lambda_U)
        assert np.allclose(lambda_U, [0, 0, 0, 0, 0, 0, 1 / 16.0, 0], atol=1e-6)

        x = np.array([-16, -16, -16, -16, 0, 0, 0])
        log_lambda_U = transform(x, return_grad=False)
        lambda_U = np.exp(log_lambda_U)
        exp = np.array([0, 0, 0, 1 / 16.0, 0, 1 / 16, 1 / 16, 1 / 48.0])
        assert np.allclose(lambda_U, exp, atol=1e-4)

    def test_DeltaU_kernel_predict(self):
        aligner = DeltaUKernelAligner(n_alleles=2, seq_length=3, P=2)
        x = np.array([-16, -16, -16, -16, 0, 0, 0])
        cov = aligner.predict(x)
        exp = [
            0.10416684,
            -0.04166664,
            -0.04166664,
            -0.02083333,
            -0.04166664,
            -0.02083333,
            -0.02083333,
            0.0833333,
        ]
        assert np.allclose(cov, exp)

    def test_DeltaU_kernel_alignment_grad(self):
        aligner = DeltaUKernelAligner(n_alleles=4, seq_length=3, P=2)

        x_true = np.random.normal(size=aligner.n_params)
        cov_true = aligner.predict(x_true)
        ns = np.ones_like(cov_true)
        aligner.set_data(cov_true, ns)
        frob, grad = aligner.calc_loss(x_true, return_grad=True)
        assert np.allclose(frob, 0)
        assert np.allclose(grad, 0)

        cov_obs = cov_true + 1e-2 * np.random.normal(size=cov_true.shape)
        aligner.set_data(cov_obs, ns)
        x = np.random.normal(size=aligner.n_params)
        grad = aligner.calc_loss(x, return_grad=True)[1]
        f = partial(aligner.calc_loss, return_grad=False)
        grad_num = self._finite_diff_grad(f, x)
        assert np.allclose(grad, grad_num, rtol=1e-5, atol=1e-8)

    def test_DeltaU_kernel_alignment_fit(self):
        aligner = DeltaUKernelAligner(n_alleles=4, seq_length=3, P=2)
        xs = np.array(
            [
                [-16, -16, -16, -16, -5.0, -5, -5],
                [-16, -16, -16, -16, -5.0, -4, -2],
                [-16, -16, -16, -16, -5.0, 3, 3],
            ]
        )

        for x in xs:
            cov_true = aligner.predict(x)
            ns = np.ones_like(cov_true)
            x_hat = np.log(aligner.fit(cov_true, ns))
            cov_pred = aligner.predict(x_hat)
            assert np.allclose(cov_pred, cov_true, atol=1e-2)
            assert np.allclose(np.exp(x), np.exp(x_hat), atol=1e-2, rtol=0.1)

    def test_connectedness_kernel_alignment(self):
        aligner = ConnectednessKernelAligner(n_alleles=3, seq_length=2)

        x = np.zeros(3)
        cov_true = aligner.predict(x)
        ns = np.ones_like(cov_true)
        assert np.allclose(cov_true[1], cov_true[2])

        x_hat = np.log(aligner.fit(cov_true, ns))
        cov_pred = aligner.predict(x_hat)
        assert np.allclose(cov_pred, cov_true)
        assert np.allclose(np.exp(x_hat), np.exp(x))

        # Test with a larger space
        aligner = ConnectednessKernelAligner(n_alleles=4, seq_length=4)
        x = np.array([0, -0.5, -0.5, -1, -0.69])
        cov_true = aligner.predict(x)
        ns = np.ones_like(cov_true)
        x_hat = np.log(aligner.fit(cov_true, ns))
        cov_pred = aligner.predict(x_hat)
        assert np.allclose(cov_pred, cov_true)
        assert np.allclose(np.exp(x_hat), np.exp(x))


if __name__ == "__main__":
    import sys

    sys.argv = ["", "KernelAlignerTest"]
    unittest.main()
