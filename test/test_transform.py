#!/usr/bin/env python
import unittest

import numpy as np

from gpmap.aligner import (
    ConnectednessToVUTransform,
    DeltaPtoVkTransform,
    DeltaUtoVUTransform,
)


class KernelAlignerTest(unittest.TestCase):
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

    def test_Connectedness_to_VU_transform(self):
        transform = ConnectednessToVUTransform(4, 2)

        # Approximately constant kernel
        x = np.array([0, -16, -16])
        log_lambda_U, grad = transform(x, return_grad=True)
        lambda_U = np.exp(log_lambda_U)
        assert np.allclose(lambda_U, [1, 0, 0, 0], atol=1e-6)
        assert np.allclose(grad, [[2, 1, 1, 0],
                                  [0, 0, 1, 1],
                                  [0, 1, 0, 1]], atol=1e-6)
        
        # Exponential kernel
        x = np.array([0, -1, -1])
        log_lambda_U, grad = transform(x, return_grad=True)
        lambda_U = np.exp(log_lambda_U)
        assert np.allclose(lambda_U, [1, np.exp(-1), np.exp(-1), np.exp(-2)], atol=1e-6)
        assert np.allclose(grad, [[2, 1, 1, 0],
                                  [0, 0, 1, 1],
                                  [0, 1, 0, 1]], atol=1e-6)
        
        # Site-specific kernel
        x = np.array([0, -1, -2])
        log_lambda_U, grad = transform(x, return_grad=True)
        lambda_U = np.exp(log_lambda_U)
        assert np.allclose(lambda_U, [1, np.exp(-2), np.exp(-1), np.exp(-3)], atol=1e-6)
        assert np.allclose(grad, [[2, 1, 1, 0],
                                  [0, 0, 1, 1],
                                  [0, 1, 0, 1]], atol=1e-6)

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


if __name__ == "__main__":
    import sys

    sys.argv = ["", "KernelAlignerTest"]
    unittest.main()
