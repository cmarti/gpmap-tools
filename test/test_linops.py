#!/usr/bin/env python
import unittest
import numpy as np

from itertools import combinations
from scipy.special import comb
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import solve_triangular
from scipy.stats import multivariate_normal

from gpmap.seq import generate_possible_sequences
from gpmap.matrix import inv_dot, quad
from gpmap.linop import (
    LaplacianOperator,
    ProjectionOperator,
    VjProjectionOperator,
    DiagonalOperator,
    IdentityOperator,
    DeltaPOperator,
    RhoProjectionOperator,
    ExtendedDeltaPOperator,
    MatMulOperator,
    VjBasisOperator,
    KernelOperator,
    EigenBasisOperator,
    DeltaKernelBasisOperator,
    DeltaKernelRegularizerOperator,
    KronOperator,
    PolynomialOperator,
    CovarianceDistanceOperator,
    CovarianceVjOperator,
    SelIdxOperator,
    ExpandIdxOperator,
    StackedOperator,
    InverseOperator,
    TriangularInverseOperator,
    KronTriangularInverseOperator,
    MultivariateGaussian,
)


class LinOpsTests(unittest.TestCase):
    def test_diag_operator(self):
        D = DiagonalOperator(diag=np.array([2, 1, 2]))

        # Test matvec
        v = np.ones(3)
        assert np.allclose(D.dot(v), [2, 1, 2])

        v = 2 * np.ones(3)
        assert np.allclose(D.dot(v), [4, 2, 4])

        # Test matmat
        B = np.ones((3, 2))
        assert np.allclose(D.dot(B), [[2, 2], [1, 1], [2, 2]])

    def test_identity_operator(self):
        n = 3
        Identity = IdentityOperator(n)
        v = np.random.normal(size=n)
        assert np.allclose(v, Identity @ v)
        assert np.allclose(v, Identity.transpose() @ v)

    def test_tri_inv_operator(self):
        A = np.tril(np.random.normal(size=(5, 5)))
        b = np.random.normal(size=5)

        A_inv = TriangularInverseOperator(A)
        x1 = A_inv @ b
        x2 = solve_triangular(A, b, lower=True)
        x3 = np.linalg.inv(A) @ b
        assert np.allclose(x1, x2)
        assert np.allclose(x1, x3)

    def test_matmul_operator(self):
        m = np.array([[1, 2], [-1, 1], [2, 0]])

        # Chain 2 operators
        v = np.random.normal(size=m.shape[0])
        X = m @ m.T
        u1 = X.dot(v)

        M = MatMulOperator([m, m.T])
        u2 = M.dot(v)
        assert np.allclose(u1, u2)

        # Chain more than 2 operators
        v = np.random.normal(size=m.shape[0])
        X = m.T @ m @ m.T
        u1 = X.dot(v)

        M = MatMulOperator([m.T, m, m.T])
        u2 = M.dot(v)
        assert np.allclose(u1, u2)

        # Check dimensions error
        try:
            M = MatMulOperator([m, m])
        except ValueError:
            pass

    def test_stacked_operator(self):
        m = np.array([[1, 2], [-1, 1], [2, 0.0]])

        A = aslinearoperator(m)

        B = StackedOperator([A, A], axis=1)
        assert B.shape == (3, 4)
        assert np.allclose(B.todense(), np.hstack([m, m]))

        C = StackedOperator([A, A], axis=0)
        assert C.shape == (6, 2)
        assert np.allclose(C.todense(), np.vstack([m, m]))

        D = B.transpose()
        assert D.shape == (4, 3)
        assert np.allclose(D.todense(), np.hstack([m, m]).T)

        E = C.transpose()
        assert E.shape == (2, 6)
        assert np.allclose(E.todense(), np.vstack([m, m]).T)

    def test_sel_idxs_operator(self):
        m = np.array([[1, 2, 0], [-1, 1, 1], [2, 0, -1]])
        i, j = np.array([0, 1]), np.array([0, 2])
        op1 = SelIdxOperator(
            n=3,
            idx=i,
        )
        op2 = ExpandIdxOperator(n=3, idx=j)

        B = MatMulOperator([op1, m, op2])
        A = m[i, :][:, j]
        C = MatMulOperator([m]).submatrix(i, j)
        assert B.shape == A.shape
        assert B.shape == C.shape

        v = np.random.normal(size=B.shape[1])
        u1 = A @ v
        u2 = B @ v
        u3 = C @ v
        assert np.allclose(u1, u2)
        assert np.allclose(u1, u3)

    def test_laplacian_operator(self):
        sl = LaplacianOperator(2, 2)

        v = np.ones(4)
        assert np.allclose(sl.dot(v), 0)
        assert np.allclose(sl.dot(2 * v), 0)

        v = np.array([1, 2, 1, 0])
        u = np.array([-1, 3, 1, -3])
        assert np.allclose(sl.dot(v), u)

        sl = LaplacianOperator(2, 2)
        assert np.allclose(sl.lambdas, [0, 2, 4])

        sl = LaplacianOperator(2, 3)
        assert np.allclose(sl.lambdas, [0, 2, 4, 6])

        sl = LaplacianOperator(3, 2)
        assert np.allclose(sl.lambdas, [0, 3, 6])

    def test_polynomial_operator(self):
        A = np.array([[2, 1.0], [1, 0.0]])
        coeffs = np.array([1, -1, 2.0])
        P1 = coeffs[0] * np.eye(2) + coeffs[1] * A + coeffs[2] * (A @ A)
        P2 = PolynomialOperator(A, coeffs)

        v = np.array([1, 0.0])
        assert np.allclose(P2.dot(v), P1[:, 0])
        v = np.array([0, 1.0])
        assert np.allclose(P2.dot(v), P1[:, 1])
        v = np.random.normal(size=A.shape[1])
        assert np.allclose(P1.dot(v), P2.dot(v))

    def test_projection_operator_coefficients(self):
        a, sl = 4, 5
        lambdas = 10 ** np.linspace(2, -2, 6)
        W = ProjectionOperator(a, sl, lambdas=lambdas)
        V = W.calc_eig_vandermonde_matrix()
        V_inv1 = W.calc_eig_vandermonde_matrix_inverse(numeric=True)
        V_inv2 = W.calc_eig_vandermonde_matrix_inverse(numeric=False)
        assert np.allclose(V_inv1 @ V, np.eye(V.shape[0]))
        assert np.allclose(V_inv2 @ V, np.eye(V.shape[0]))

    def test_projection_operator(self):
        # Purely additive function
        y = np.array([-1.5, -0.5, 0.5, 1.5])

        W = ProjectionOperator(2, 2, k=2)
        assert np.allclose(W.dot(y), 0)

        W = ProjectionOperator(2, 2, k=1)
        assert np.allclose(W.dot(y), y)

        W = ProjectionOperator(2, 2, k=0)
        assert np.allclose(W.dot(y), 0)

        # Non-zero orthogonal projections
        y = np.array([-1.5, -0.5, 0.5, 4])

        W = ProjectionOperator(2, 2, k=0)
        y0 = W.dot(y)

        W = ProjectionOperator(2, 2, k=1)
        y1 = W.dot(y)

        W = ProjectionOperator(2, 2, k=2)
        y2 = W.dot(y)

        assert not np.allclose(y0, 0)
        assert not np.allclose(y1, y)
        assert not np.allclose(y2, 0)

        # Ensure they are orthogonal to each other
        assert np.allclose(y0.T.dot(y1), 0)
        assert np.allclose(y0.T.dot(y2), 0)
        assert np.allclose(y1.T.dot(y2), 0)

        # Test inverse
        W = ProjectionOperator(2, 2, lambdas=np.array([1, 10, 1.0]))
        W_inv = W.inv()
        assert np.allclose(W_inv.dot(W.dot(y)), y)

    def test_deltap_operator(self):
        DP2 = DeltaPOperator(P=2, n_alleles=2, seq_length=3)
        DP3 = DeltaPOperator(P=3, n_alleles=2, seq_length=3)

        # Additive landscape
        v = np.array([0, 1, 1, 2, 0, 1, 1, 2])
        assert quad(DP2, v) == 0
        assert quad(DP3, v) == 0

        # Pairwise landscape
        v = np.array([0, 1, 1, 3, 0, 1, 1, 3])
        assert quad(DP2, v) > 0
        assert quad(DP3, v) == 0

        # Test eigenvalues
        sl, a, P = 5, 4, 2
        DP = DeltaPOperator(P=P, n_alleles=a, seq_length=sl)
        DP.calc_lambdas()

        for k in range(P, sl + 1):
            lambda_k = a**P * comb(k, P)
            assert DP.lambdas[k] == lambda_k

    def test_extended_deltap_operator(self):
        op = ExtendedDeltaPOperator(
            n_alleles=2, seq_length=3, P=2, lambdas0=np.array([1, 1.0])
        )
        assert np.allclose(op.lambdas, [1, 1, 4, 12])

    def test_kron_operator(self):
        np.random.seed(0)
        m1 = np.random.normal(size=(2, 2))
        m2 = np.random.normal(size=(2, 2))
        m3 = np.random.normal(size=(2, 1))
        m4 = np.random.normal(size=(2, 3))

        # With 2 matrices
        K = KronOperator([m1, m2])
        m = np.kron(m1, m2)
        v = np.random.normal(size=K.shape[1])
        u1 = m @ v
        u2 = K @ v
        assert np.allclose(m, K.todense())
        assert np.allclose(u1, u2)

        # With 2 matrices of different sizes
        K = KronOperator([m1, m3])
        m = np.kron(m1, m3)
        v = np.random.normal(size=K.shape[1])
        assert np.allclose(m.dot(v), K.dot(v))
        assert np.allclose(m, K.todense())

        # With 3 matrices
        K = KronOperator([m2, m1, m1])
        m = np.kron(m2, np.kron(m1, m1))
        v = np.random.normal(size=K.shape[1])
        assert np.allclose(m.dot(v), K.dot(v))
        assert np.allclose(m, K.todense())

        # Try different sizes
        K = KronOperator([m3, m1, m2])
        m = np.kron(m3, np.kron(m1, m2))
        v = np.random.normal(size=K.shape[1])
        assert np.allclose(m.dot(v), K.dot(v))
        assert np.allclose(m, K.todense())

        # Try with random matrices of different sizes
        K = KronOperator([m1, m2, m3, m4])
        m = np.kron(m1, np.kron(m2, np.kron(m3, m4)))
        v = np.random.normal(size=K.shape[1])
        assert np.allclose(m.dot(v), K.dot(v))
        assert np.allclose(m, K.todense())

        # Test transpose
        K_transpose = K.transpose()
        v = np.random.normal(size=K_transpose.shape[1])
        assert np.allclose(m.T.dot(v), K_transpose.dot(v))
        assert np.allclose(m.T, K_transpose.todense())

    def test_kron_cholesky_operator(self):
        L_i = np.array([[1, 0], [0.4, 0.8]])
        K_0 = L_i @ L_i.T

        # Classic approach
        K1 = np.kron(K_0, K_0)
        L1 = np.linalg.cholesky(K1)

        # Using LinearOperator
        K2 = KronOperator([K_0, K_0])
        L2 = K2.cholesky()

        # Test matvec
        b = np.random.normal(size=K1.shape[1])
        x1 = L1 @ b
        x2 = L2 @ b
        assert np.allclose(x1, x2)

        # Test inverse operator
        L_inv = KronTriangularInverseOperator(L2)
        x1 = solve_triangular(L1, b, lower=True)
        x2 = L_inv @ b
        assert np.allclose(x1, x2)

    def test_vj_projection_operator(self):
        a, sl = 2, 2

        # Purely additive function
        y = np.array([-1.5, -0.5, 0.5, 1.5])
        y01 = np.array([-1, -1, 1, 1])
        y10 = np.array([-0.5, 0.5, -0.5, 0.5])

        Pj = VjProjectionOperator(a, sl, j=[0])
        f01 = Pj.dot(y)
        assert np.allclose(f01, y01)

        Pj = VjProjectionOperator(a, sl, j=[1])
        f10 = Pj.dot(y)
        assert np.allclose(f10, y10)

        Pj = VjProjectionOperator(a, sl, j=[])
        f00 = Pj.dot(y)
        assert np.allclose(f00, 0)

        Pj = VjProjectionOperator(a, sl, j=[0, 1])
        f11 = Pj.dot(y)
        assert np.allclose(f11, 0)

        # Tests that projections add up to the whole subspace in larger case
        a, sl = 4, 5
        v = np.random.normal(size=a**sl)

        for k in range(1, 6):
            W = ProjectionOperator(a, sl, k=k)
            u1 = W.dot(v)

            u2 = np.zeros(v.shape[0])
            for j in combinations(np.arange(W.seq_length), k):
                Pj = VjProjectionOperator(a, sl, j=list(j))
                u2 += Pj.dot(v)

            assert np.allclose(u1, u2)

    def test_vj_projection_operator_sq_norm(self):
        a, sl = 2, 2
        y = np.array([-1.5, -0.5, 0.5, 1.5])

        for j in [[], [0], [1], [0, 1]]:
            Pj = VjProjectionOperator(a, sl, j=j)
            fsqn = Pj.dot_square_norm(y)
            exp = np.sum(Pj.dot(y) ** 2)
            assert np.allclose(fsqn, exp)

        # Test with bigger operator
        Pj = VjProjectionOperator(4, 8, j=[0, 3, 5])
        y = np.random.normal(size=Pj.shape[1])
        exp = np.sum(Pj.dot(y) ** 2)
        fsqn = Pj.dot_square_norm(y)
        assert np.allclose(fsqn, exp)

    def test_vj_basis_operator(self):
        # Test in small case
        B = VjBasisOperator(2, 2, j=(0, 1))
        b = B.todense()
        assert b.shape == B.shape
        assert b.shape == (4, 1)

        dense_b = np.array([0.5, -0.5, -0.5, 0.5])
        assert np.allclose(b.flatten(), dense_b)

        B = VjBasisOperator(2, 2, j=(0,))
        dense_b = np.array([-0.5, -0.5, 0.5, 0.5])
        assert np.allclose(B.todense().flatten(), dense_b)

        B = VjBasisOperator(2, 2, j=(1,))
        dense_b = np.array([-0.5, 0.5, -0.5, 0.5])
        assert np.allclose(B.todense().flatten(), dense_b)

        # Test dimensions in larger operators
        B = VjBasisOperator(4, 5, j=(0, 2))
        assert B.shape == (4**5, 9)

        v = np.random.normal(size=(9,))
        u = B @ v
        assert u.shape[0] == 4**5

    def test_k_eigen_basis_operator(self):
        a, sl = 4, 5
        v = np.random.normal(size=a**sl)

        for k in range(0, sl + 1):
            W = ProjectionOperator(a, sl, k=k)
            B = EigenBasisOperator(a, sl, k=k)
            u1 = W @ v
            u2 = B @ B.transpose() @ v
            assert np.allclose(u1, u2)

    def test_delta_kernel_basis_operator(self):
        # Test small example explicitly
        a, sl, P = 2, 2, 2
        B = DeltaKernelBasisOperator(a, sl, P=P)
        B_dense = np.array(
            [
                [1 / 2, 1 / 2, 1 / 2, 1 / 2],
                [-1 / 2, -1 / 2, 1 / 2, 1 / 2],
                [-1 / 2, 1 / 2, -1 / 2, 1 / 2],
            ]
        )
        b = np.vstack([B.dot(v) for v in np.eye(3)])
        assert np.allclose(b, B_dense)

        # Test in a larger case
        a, sl, P = 4, 5, 2
        B = DeltaKernelBasisOperator(a, sl, P=P)

        # Ensure it is in the null space of DeltaP operator
        DP = DeltaPOperator(a, sl, P=P)
        v = np.random.normal(size=B.shape[1])
        f = B.dot(v)
        assert np.allclose(DP.dot(f), 0.0)

        # Ensure it provides a valid projection matrix
        v = np.random.normal(size=B.shape[0])
        u1 = B @ B.transpose() @ v
        u2 = B @ B.transpose() @ u1
        assert np.allclose(u1, u2)

        # Ensure it provides the right projection matrix
        v = np.random.normal(size=B.shape[0])
        u1 = B @ B.transpose() @ v
        u2 = 0.0
        for k in range(P):
            W = ProjectionOperator(a, sl, k=k)
            u2 += W.dot(v)
        assert np.allclose(u1, u2)

    def test_delta_kernel_regularizer_operator(self):
        # Test small example explicitly
        a, sl, P = 2, 2, 2

        lda = np.array([1e-16, 1])
        B = DeltaKernelBasisOperator(a, sl, P=P)
        b = np.random.normal(scale=[0, 1, 1])
        phi = B @ b

        # Ensure it is in the null space
        DP = DeltaPOperator(a, sl, P)
        c3 = np.dot(phi, DP @ phi)
        assert np.allclose(c3, 0)

        D = DeltaKernelRegularizerOperator(B, lambdas_inv=lda)
        c1 = np.sum(b[1:] ** 2)
        c2 = np.dot(phi, D @ phi)
        assert np.allclose(c1, c2)

    def test_rho_projection_operator(self):
        np.random.seed(0)

        # Small landscape
        a, sl = 2, 2
        P = RhoProjectionOperator(a, sl, rho=0.5)
        v = np.random.normal(size=P.n)
        u1 = P.dot(v)

        lambdas = np.array([1, 0.5, 0.25])
        W = ProjectionOperator(a, sl, lambdas=lambdas)
        u2 = W.dot(v)
        assert np.allclose(u1, u2)

        # Different rho per site
        P = RhoProjectionOperator(a, sl, rho=[0.5, 0.8])
        v = np.random.normal(size=P.n)
        u1 = P.dot(v)

        # Test inverse operator
        P_inv = P.inv()
        assert(np.allclose(P_inv @ P @ v, v))

        # Do calculations using Vj Projections
        ws = [1 * 1, 1 * 0.8, 0.5 * 1, 0.8 * 0.5]
        js = [[], [1], [0], [0, 1]]
        u2 = np.zeros(v.shape)
        for w, j in zip(ws, js):
            Pj = VjProjectionOperator(a, sl, j=j)
            u2 += w * Pj.dot(v)
        assert np.allclose(u1, u2)

        # Larger landscape with single rho model
        a, sl = 4, 6
        P = RhoProjectionOperator(a, sl, rho=0.5)
        v = np.random.normal(size=P.shape[1])
        u1 = P.dot(v)

        lambdas = np.array([2**-(i) for i in range(sl + 1)])
        W = ProjectionOperator(a, sl, lambdas=lambdas)
        u2 = W.dot(v)
        assert np.allclose(u1, u2)

        # Test inverse operator

    def test_kernel_operator(self):
        A = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])

        # Test plain kernel operator
        K = KernelOperator(A)
        v = np.random.normal(size=K.shape[1])
        assert np.allclose(A.dot(v), K.dot(v))

        # Test transpose
        assert np.allclose(K._rmatmat(v), K.dot(v))
        assert np.allclose(K.transpose().dot(v), K.dot(v))

        # Solve using CG
        v = np.array([1, 2.0, 1])
        u = inv_dot(K, v, method="cg", atol=1e-14)
        assert np.allclose(K.dot(u), v)

        # Test different indexings
        x1 = np.array([0, 1], dtype=int)
        K = KernelOperator(A, x1=x1)
        assert K.shape == (2, 3)

        v = np.random.normal(size=K.shape[1])
        assert np.allclose(A[:2, :].dot(v), K.dot(v))

        x2 = np.array([1, 2], dtype=int)
        K = KernelOperator(A, x2=x2)
        assert K.shape == (3, 2)

        v = np.random.normal(size=K.shape[1])
        assert np.allclose(A[:, 1:].dot(v), K.dot(v))

        K = KernelOperator(A, x1=x1, x2=x2)
        assert K.shape == (2, 2)

        v = np.random.normal(size=K.shape[1])
        assert np.allclose(A[:2, 1:].dot(v), K.dot(v))

        # Test different indexings transpose
        K = KernelOperator(A, x1=x1)
        v = np.random.normal(size=(K.shape[0], 1))
        assert np.allclose(A[x1, :].T @ v, K._rmatmat(v))

        # Test adding diagonal
        D = DiagonalOperator(np.ones(2))
        B = A[:2, 1:] + np.eye(2)
        K = KernelOperator(A, x1=x1, x2=x2) + D
        assert K.shape == (2, 2)

        v = np.random.normal(size=K.shape[1])
        assert np.allclose(B.dot(v), K.dot(v))

    def test_inverse_operator(self):
        np.random.seed(0)

        # With small matrix we can verify the inverse
        A = np.array([[1, 0.5], [0.5, 1]])
        A_inv = np.linalg.inv(A)
        A_inv_op = InverseOperator(aslinearoperator(A), method="cg").todense()
        assert np.allclose(A_inv_op, A_inv)

        # With a large operator
        A = RhoProjectionOperator(4, 8, rho=0.5)  # + D
        A_inv = InverseOperator(A, method="cg")
        v = np.random.normal(size=A.shape[1])
        u = A @ (A_inv @ v)
        assert np.allclose(u, v, atol=1e-4)

    def test_mv_gaussian(self):
        A = np.array([[1, 0.5], [0.5, 1]])
        Sigma1 = np.kron(A, A)
        Sigma2 = KronOperator([A, A])
        mu = np.zeros(Sigma1.shape[0])
        x = np.random.normal(size=mu.shape[0])

        gaussian1 = multivariate_normal(mu, Sigma1)
        gaussian2 = MultivariateGaussian(mu, Sigma2)
        logp1 = gaussian1.logpdf(x)
        logp2 = gaussian2.logp(x)
        assert np.allclose(logp1, logp2)

        # Test sampling
        x = gaussian2.sample(n_samples=10000)
        Sigma_hat = np.cov(x)
        assert np.allclose(Sigma1, Sigma_hat, atol=0.05)

        # Test with large matrices
        Sigma1 = np.kron(Sigma1, Sigma1)
        Sigma2 = KronOperator([A] * 4)
        mu = np.zeros(Sigma1.shape[0])
        x = np.random.normal(size=mu.shape[0])

        gaussian1 = multivariate_normal(mu, Sigma1)
        gaussian2 = MultivariateGaussian(mu, Sigma2)
        logp1 = gaussian1.logpdf(x)
        logp2 = gaussian2.logp(x)
        assert np.allclose(logp1, logp2)

    def test_covariance_distance_operator(self):
        a, sl = 2, 2
        v = np.random.normal(size=(a**sl, 1))
        S = v @ v.T

        # Ensure the sum over all possible distances matches
        ss1 = S.sum()
        ss2 = 0
        for d in range(sl + 1):
            C = CovarianceDistanceOperator(a, sl, d)
            ss2 += quad(C, v)
        assert np.allclose(ss1, ss2)

        # Check distance=0
        s0 = np.sum(v**2)
        C0 = CovarianceDistanceOperator(a, sl, distance=0)
        assert np.allclose(quad(C0, v), s0)

        # Check distance=1
        C1 = CovarianceDistanceOperator(a, sl, distance=1)
        m1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
        assert np.allclose(np.sum(m1 * S), quad(C1, v))

        # Check distance=2
        C2 = CovarianceDistanceOperator(a, sl, distance=2)
        m2 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        assert np.allclose(np.sum(m2 * S), quad(C2, v))

    def test_covariance_vj_operator(self):
        a, sl = 2, 2
        sites = np.arange(sl)
        v = np.random.normal(size=(a**sl, 1))
        S = v @ v.T

        # Check 00
        s00 = np.sum(v**2)
        C00 = CovarianceVjOperator(a, sl, j=[])
        assert np.allclose(np.eye(a**sl), C00.todense())
        assert np.allclose(quad(C00, v), s00)

        # Check 01
        C01 = CovarianceVjOperator(a, sl, j=[0])
        m01 = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                        [1, 0, 0, 0], [0, 1, 0, 0]])
        assert np.allclose(m01, C01.todense())
        assert np.allclose(np.sum(m01 * S), quad(C01, v))

        # Check 10
        C10 = CovarianceVjOperator(a, sl, j=[1])
        m10 = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                        [0, 0, 0, 1], [0, 0, 1, 0]])
        assert np.allclose(m10, C10.todense())
        assert np.allclose(np.sum(m10 * S), quad(C10, v))

        # Check distance=2
        C11 = CovarianceVjOperator(a, sl, j=(0, 1))
        m11 = np.array([[0, 0, 0, 1], [0, 0, 1, 0],
                        [0, 1, 0, 0], [1, 0, 0, 0]])
        assert np.allclose(m11, C11.todense())
        assert np.allclose(np.sum(m11 * S), quad(C11, v))

        # Ensure the sum over all possible distances matches
        ss1 = S.sum()
        ss2 = 0
        for k in range(sl + 1):
            for j in combinations(sites, k):
                C = CovarianceVjOperator(a, sl, j=j)
                ss2 += quad(C, v)
        assert np.allclose(ss1, ss2)

# class SkewedLinOpsTests(unittest.TestCase):
#     def xtest_skewed_kernel_operator(self):
#         ps = np.array([[0.3, 0.7], [0.5, 0.5]])
#         log_p = np.log(ps)
#         sl, a = ps.shape

#         # Define Laplacian based kernel
#         sl = LaplacianOperator(a, sl, ps=ps)
#         W = ProjectionOperator(sl=sl)
#         K1 = VarianceComponentKernelOperator(W)

#         # Define full kernel function
#         kernel = VarianceComponentKernel(sl, a, use_p=True)
#         x = np.array(["AA", "AB", "BA", "BB"])
#         kernel.set_data(x1=x, alleles=["A", "B"])

#         # Constant component
#         lambdas = [1, 0, 0]
#         K1.set_lambdas(lambdas)
#         k1 = K1.todense()
#         assert np.allclose(k1, 1)

#         K2 = kernel(lambdas=lambdas, log_p=log_p)
#         assert np.allclose(K2, 1)

#         # Additive component
#         lambdas = [0, 1, 0]
#         K1.set_lambdas(lambdas)
#         k1 = K1.todense()
#         k2 = kernel(lambdas=lambdas, log_p=log_p)
#         assert np.allclose(k1, k2)

#         # Pairwise component
#         lambdas = [0, 0, 1]
#         K1.set_lambdas(lambdas)
#         k1 = K1.todense()
#         k2 = kernel(lambdas=lambdas, log_p=log_p)
#         assert np.allclose(k1, k2)

#     def xtest_skewed_kernel_operator_big(self):
#         sl, a = 4, 4
#         alleles = ALPHABET[:a]
#         ps = np.random.dirichlet(alpha=np.ones(a), size=sl)
#         log_p = np.log(ps)

#         # Define Laplacian based kernel
#         sl = LaplacianOperator(a, sl, ps=ps)
#         W = ProjectionOperator(sl=sl)
#         K1 = VarianceComponentKernelOperator(W)
#         I = np.eye(K1.shape[0])

#         # Define full kernel function
#         kernel = VarianceComponentKernel(sl, a, use_p=True)
#         x = np.array([x for x in generate_possible_sequences(sl, alleles)])
#         kernel.set_data(x1=x, alleles=alleles)

#         # Test components
#         for k in range(sl + 1):
#             lambdas = np.zeros(sl + 1)
#             lambdas[k] = 1
#             K1.set_lambdas(lambdas)
#             k1 = K1.dot(I)
#             k2 = kernel(lambdas=lambdas, log_p=log_p)
#             assert np.allclose(k1, k2)


if __name__ == "__main__":
    import sys

    sys.argv = ["", "LinOpsTests"]
    unittest.main()
