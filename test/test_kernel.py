#!/usr/bin/env python
import unittest

import numpy as np

from scipy.special import comb

from gpmap.aligner import FullKernelAligner
from gpmap.inference import VCregression, GaussianProcessRegressor
from gpmap.kernel import (
    VarianceComponentKernel,
    SequenceKernel,
    SkewedVarianceComponentKernel,
    ConnectednessKernel,
)
from gpmap.seq import generate_possible_sequences


class KernelTest(unittest.TestCase):
    def xtest_sequence_kernel(self):
        kernel = SequenceKernel(2, 2)
        X = np.array(["AA", "AB", "BA", "BB"])
        kernel.set_data(X, alleles=["A", "B"])

        # Test encoding
        onehot = np.array(
            [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]]
        )
        assert np.allclose(kernel.x1, onehot)
        assert np.allclose(kernel.x2, onehot)

        # Test distance
        d = np.array([[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]])
        hamming = kernel.calc_hamming_distance(kernel.x1, kernel.x2)
        assert np.allclose(hamming, d)

    def xtest_vc_kernel(self):
        kernel = VarianceComponentKernel(2, 2)
        X = np.array(["AA", "AB", "BA", "BB"])
        kernel.set_data(X, alleles=["A", "B"])

        # Test constant covariance
        lambdas = np.array([1, 0, 0])
        cov = kernel(lambdas=lambdas)
        assert np.allclose(cov, 1 / 4)

        # Test additive covariance
        lambdas = np.array([0, 1, 0])
        cov = kernel(lambdas=lambdas)
        add = np.array(
            [
                [2.0, 0.0, 0.0, -2.0],
                [0.0, 2.0, -2.0, 0.0],
                [0.0, -2.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0, 2.0],
            ]
        )
        assert np.allclose(cov, 1 / 4 * add)

        # Test 2nd order covariance
        lambdas = np.array([0, 0, 1])
        cov = kernel(lambdas=lambdas)
        pair = np.array(
            [
                [1.0, -1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ]
        )
        assert np.allclose(cov, 1 / 4 * pair)

    def xtest_rho_kernel(self):
        kernel = ConnectednessKernel(2, 2)
        X = np.array(["AA", "AB", "BA", "BB"])
        kernel.set_data(X, alleles=["A", "B"])

        cov = kernel(rho=[0.5])
        assert np.allclose(cov[0], np.array([9 / 16.0, 3 / 16, 3 / 16, 1 / 16]))

        cov = kernel(rho=[0.5, 0.2])
        assert np.allclose(cov[0], 1 / 4 * np.array([1.8, 1.2, 0.6, 0.4]))

    def xtest_rho_kernel_grad(self):
        X = np.array(["AA", "AB", "BA", "BB"])

        # Equal sites
        kernel = ConnectednessKernel(2, 2, sites_equal=True)
        kernel.set_data(X, alleles=["A", "B"])
        kernel(rho=[0.5])
        grad = list(kernel.backward())[0]
        assert np.allclose(grad[0], np.array([0.75, -0.25, -0.25, -0.25]) / 4)

        # Variable rho across sites
        kernel = ConnectednessKernel(2, 2, sites_equal=False)
        kernel.set_data(X, alleles=["A", "B"])
        kernel(rho=[0.5, 0.5])
        grads = list(kernel.backward())

        # Partial derivatives depending on the site that is different
        assert np.allclose(
            grads[0][0], np.array([0.375, 0.125, -0.375, -0.125]) / 4
        )
        assert np.allclose(
            grads[1][0], np.array([0.375, -0.375, 0.125, -0.125]) / 4
        )

        # Variable rho method should give the same solution when rho are fixed
        grad = grads[0] + grads[1]
        assert np.allclose(grad[0], np.array([0.75, -0.25, -0.25, -0.25]) / 4)

    def xtest_frob2(self):
        X = np.array(["AA", "AB", "BA", "BB"])
        kernel = VarianceComponentKernel(2, 2)
        aligner = FullKernelAligner(kernel=kernel)

        # Constant function
        y = np.array([1, 1, 1, 1]) / 2
        lambdas = np.array([1, 0, 0])
        aligner.set_data(X, y, alleles=["A", "B"])
        frob2 = aligner.frob2(lambdas=lambdas)
        frob2_grad = aligner.frob2_grad(lambdas=lambdas)
        assert frob2 == 0
        assert np.allclose(frob2_grad, 0)

        # Additive function
        y = np.array([-2, 0, 0, 2]) / 2
        lambdas = np.array([0, 1, 0])
        aligner.set_data(X, y, alleles=["A", "B"])
        frob2 = aligner.frob2(lambdas=lambdas)
        frob2_grad = aligner.frob2_grad(lambdas=lambdas)
        assert frob2 == 2
        assert np.allclose(frob2_grad, 0)

        # Pairwise function
        y = np.array([1, -1, -1, 1]) / 2
        lambdas = np.array([0, 0, 1])
        aligner.set_data(X, y, alleles=["A", "B"])
        frob2 = aligner.frob2(lambdas=lambdas)
        frob2_grad = aligner.frob2_grad(lambdas=lambdas)
        assert frob2 == 0
        assert np.allclose(frob2_grad, 0)

        # Additive function with non zero gradient
        y = np.array([-1, 0, 0, 1]) / 2
        lambdas = np.array([0, 0.75, 0])
        aligner.set_data(X, y, alleles=["A", "B"])
        frob2 = aligner.frob2(lambdas=lambdas)
        frob2_grad = aligner.frob2_grad(lambdas=lambdas)
        assert frob2 == 0.625
        assert np.allclose(frob2_grad, [0, 1.5, 0])

    def xtest_full_kernel_alignment(self):
        X = np.array(["AA", "AB", "BA", "BB"])
        alleles = ["A", "B"]

        kernel = VarianceComponentKernel(2, 2)
        aligner = FullKernelAligner(kernel=kernel)

        # Constant model
        y = np.array([1, 1, 1, 1])
        aligner.set_data(X=X, y=y, alleles=alleles)
        lambdas_star = aligner.fit()
        assert np.allclose(lambdas_star, [4, 0, 0], atol=0.01)

        # Additive model
        y = np.array([-2, 0, 0, 2])
        aligner.set_data(X=X, y=y, alleles=alleles)
        lambdas_star = aligner.fit()
        assert np.allclose(lambdas_star, [0, 4, 0], atol=0.01)

        # Pairwise model
        y = np.array([1, -1, -1, 1])
        aligner.set_data(X=X, y=y, alleles=alleles)
        lambdas_star = aligner.fit()
        assert np.allclose(lambdas_star, [0, 0, 4], atol=0.01)

    def xtest_full_kernel_alignment_comparison(self):
        np.random.seed(0)
        l, a = 4, 4
        alleles = ["A", "C", "G", "T"]
        X = np.array(list(generate_possible_sequences(l, alleles)))

        # Simulate
        lambdas0 = 2.0 ** -np.arange(l + 1)
        kernel = VarianceComponentKernel(a, l, lambdas=lambdas0)
        model = GaussianProcessRegressor(kernel=kernel)
        y = model.sample()

        # Run collapsed kernel alignment
        model = VCregression()
        model.init(l, a)
        lambdas_star1 = model.fit(X, y)
        rel_err = np.abs(np.log2(lambdas_star1 / lambdas0))
        assert np.all(rel_err[1:] < 0.5)

        # Run full kernel alignment
        kernel = VarianceComponentKernel(l, a)
        aligner = FullKernelAligner(kernel=kernel)
        aligner.set_data(X=X, y=y, alleles=alleles)
        lambdas_star2 = aligner.fit()
        assert np.allclose(lambdas_star1, lambdas_star2, atol=1e-4)

    def xtest_full_kernel_alignment_rho(self):
        X = np.array(["AA", "AB", "BA", "BB"])
        alleles = ["A", "B"]

        kernel = ConnectednessKernel(2, 2)
        aligner = FullKernelAligner(kernel=kernel)
        y = np.array([1, 1, 1, 1])
        aligner.set_data(X=X, y=y, alleles=alleles)

        # Set target covariance to one that can be
        # perfectly represented by a combination of rho
        aligner.target = (
            np.array(
                [
                    [1.8, 1.2, 0.6, 0.4],
                    [1.2, 1.8, 0.4, 0.6],
                    [0.6, 0.4, 1.8, 1.2],
                    [0.4, 0.6, 1.2, 1.8],
                ]
            )
            / 4
        )

        rho_star = aligner.fit()
        assert np.allclose(rho_star, [0.5, 0.2])

    def xtest_full_kernel_alignment_rho_comparison(self):
        np.random.seed(0)
        l, a = 5, 4
        alleles = ["A", "C", "G", "T"]
        X = np.array(list(generate_possible_sequences(l, alleles)))

        # Simulate
        rho0 = 0.5
        kernel = ConnectednessKernel(a, l, rho=rho0)
        model = GaussianProcessRegressor(kernel=kernel)
        y = model.sample()

        # Run kernel alignment
        kernel = ConnectednessKernel(l, a, sites_equal=True)
        aligner = FullKernelAligner(kernel=kernel)
        aligner.set_data(X=X, y=y, alleles=alleles)
        rho_star = aligner.fit()[0]
        assert np.allclose(rho0, rho_star, atol=0.01)

    def xtest_full_kernel_alignment_variable_rho(self):
        np.random.seed(1)
        rho0 = [0.5, 0.1, 0.7, 0.9]
        l, a = len(rho0), 4
        alleles = ["A", "C", "G", "T"]
        X = np.array(list(generate_possible_sequences(l, alleles)))

        # Simulate
        kernel = ConnectednessKernel(a, l, rho=rho0)
        model = GaussianProcessRegressor(kernel=kernel)
        y = model.sample()

        # Run kernel alignment
        kernel = ConnectednessKernel(l, a, sites_equal=False)
        aligner = FullKernelAligner(kernel=kernel)
        aligner.set_data(X=X, y=y, alleles=alleles)
        rho_star = aligner.fit()
        rel_err = np.abs(np.log2(rho_star / rho0))
        assert np.all(rel_err[1:] < 0.5)

    def xtest_vc_kernel_ps(self):
        kernel = SkewedVarianceComponentKernel(2, 2, use_p=True)
        X = np.array(['AA', 'AB', 'BA', 'BB'])
        kernel.set_data(X, alleles=['A', 'B'])

        # Test constant covariance
        lambdas = np.array([1, 0, 0])
        log_p = np.log(0.5) * np.ones((2, 2))
        cov = kernel(lambdas=lambdas, log_p=log_p)
        assert(np.allclose(cov, 1))

        # Test additive covariance
        lambdas = np.array([0, 1, 0])
        cov = kernel(lambdas=lambdas, log_p=log_p)
        add = np.array([[ 2.,  0.,  0., -2.],
                        [ 0.,  2., -2.,  0.],
                        [ 0., -2.,  2.,  0.],
                        [-2.,  0.,  0.,  2.]])
        assert(np.allclose(cov, add))

        # Test 2nd order covariance
        lambdas = np.array([0, 0, 1])
        cov = kernel(lambdas=lambdas, log_p=log_p)
        add = np.array([[ 1., -1., -1.,  1.],
                        [-1.,  1.,  1., -1.],
                        [-1.,  1.,  1., -1.],
                        [ 1., -1., -1.,  1.]])
        assert(np.allclose(cov, add))

    def xtest_NK_lambdas(self):
        l, a = 7, 4
        P = 3
        kernel = VarianceComponentKernel(l, a)
        lambdas = np.array([a**(-P) * comb(l-k, l-P) for k in range(l+1)])
        # lambdas = np.array([a**(-P) * comb(P, k) for k in range(l+1)])
        print(lambdas)
        print(kernel.W_kd.dot(lambdas))
        cov = np.array([comb(l-d, P) for d in range(l+1)])
        print(cov)


if __name__ == "__main__":
    import sys

    sys.argv = ["", "KernelTest"]
    unittest.main()
