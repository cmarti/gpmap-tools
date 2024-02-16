#!/usr/bin/env python
import unittest
import numpy as np

from timeit import timeit
from itertools import combinations
from scipy.special import comb

from gpmap.src.datasets import DataSet
from gpmap.src.settings import ALPHABET
from gpmap.src.seq import generate_possible_sequences
from gpmap.src.matrix import get_sparse_diag_matrix
from gpmap.src.kernel import VarianceComponentKernel
from gpmap.src.linop import (LaplacianOperator, ProjectionOperator,
                             VjProjectionOperator,
                             VarianceComponentKernelOperator,
                             DeltaPOperator, ProjectionOperator2,
                             RhoProjectionOperator, ConnectednessKernelOperator,
                             ExtendedLinearOperator, VjBasisOperator,
                             EigenBasisOperator, DeltaKernelBasisOperator,
                             KronOperator, PolynomialOperator,
                             KernelOperator,
                             calc_variance_components,
                             calc_vjs_variance_components)


class LinOpsTests(unittest.TestCase):
    def test_laplacian_operator(self):
        L = LaplacianOperator(2, 2)
        
        v = np.ones(4)
        assert(np.allclose(L.dot(v), 0))
        assert(np.allclose(L.dot(2*v), 0))
        
        v = np.array([1, 2, 1, 0])
        u = np.array([-1, 3, 1, -3])
        assert(np.allclose(L.dot(v), u))
    
        L = LaplacianOperator(2, 2)
        assert(np.allclose(L.lambdas, [0, 2, 4]))
        
        L = LaplacianOperator(2, 3)
        assert(np.allclose(L.lambdas, [0, 2, 4, 6]))
        
        L = LaplacianOperator(3, 2)
        assert(np.allclose(L.lambdas, [0, 3, 6]))
    
    def test_polynomial_operator(self):
        A = np.array([[2, 1.],
                      [1, 0.]])
        coeffs = np.array([1, -1, 2.])
        P1 = coeffs[0] * np.eye(2) + coeffs[1] * A + coeffs[2] * (A @ A)
        P2 = PolynomialOperator(A, coeffs)
        
        v = np.array([1, 0.])
        assert(np.allclose(P2.dot(v), P1[:, 0]))
        v = np.array([0, 1.])
        assert(np.allclose(P2.dot(v), P1[:, 1]))
        v = np.random.normal(size=A.shape[1])
        assert(np.allclose(P1.dot(v), P2.dot(v)))
        
    def test_projection_operator_coefficients(self):
        a, l = 4, 5
        lambdas = 10 ** np.linspace(2, -2, 6)
        W = ProjectionOperator(a, l, lambdas=lambdas)
        V = W.calc_eig_vandermonde_matrix()
        V_inv1 = W.calc_eig_vandermonde_matrix_inverse(numeric=True)
        V_inv2 = W.calc_eig_vandermonde_matrix_inverse(numeric=False)
        assert(np.allclose(V_inv1 @ V, np.eye(V.shape[0])))
        assert(np.allclose(V_inv2 @ V, np.eye(V.shape[0])))
            
    def test_projection_operator(self):
        # Purely additive function
        y = np.array([-1.5, -0.5, 0.5, 1.5])
        
        W = ProjectionOperator(2, 2, k=2)
        assert(np.allclose(W.dot(y), 0))
        
        W = ProjectionOperator(2, 2, k=1)
        assert(np.allclose(W.dot(y), y))
        
        W = ProjectionOperator(2, 2, k=0)
        assert(np.allclose(W.dot(y), 0))
        
        # Non-zero orthogonal projections
        y = np.array([-1.5, -0.5, 0.5, 4])
        
        W = ProjectionOperator(2, 2, k=0)
        y0 = W.dot(y)

        W = ProjectionOperator(2, 2, k=1)
        y1 = W.dot(y)
        
        W = ProjectionOperator(2, 2, k=2)
        y2 = W.dot(y) 
        
        assert(not np.allclose(y0, 0))
        assert(not np.allclose(y1, y))
        assert(not np.allclose(y2, 0))
        
        # Ensure they are orthogonal to each other
        assert(np.allclose(y0.T.dot(y1), 0))
        assert(np.allclose(y0.T.dot(y2), 0))
        assert(np.allclose(y1.T.dot(y2), 0))
        
        # Test inverse
        W = ProjectionOperator(2, 2, lambdas=np.array([1, 10, 1.]))
        W_inv = W.inv()
        assert(np.allclose(W_inv.dot(W.dot(y)), y))
    
    def test_deltap_operator(self):
        DP2 = DeltaPOperator(P=2, n_alleles=2, seq_length=3)
        DP3 = DeltaPOperator(P=3, n_alleles=2, seq_length=3)

        # Additive landscape        
        v = np.array([0, 1, 1, 2,
                      0, 1, 1, 2])
        assert(DP2.quad(v) == 0)
        assert(DP3.quad(v) == 0)
        
        # Pairwise landscape
        v = np.array([0, 1, 1, 3,
                      0, 1, 1, 3])
        assert(DP2.quad(v) > 0)
        assert(DP3.quad(v) == 0)
    
        # Test eigenvalues
        l, a, P = 5, 4, 2
        DP = DeltaPOperator(P=P, n_alleles=a, seq_length=l)
        DP.calc_lambdas()
        
        for k in range(P, l+1):
            lambda_k = a ** P * comb(k, P)
            assert(DP.lambdas[k] == lambda_k)
    
    def test_kron_operator(self):
        m1 = np.random.normal(size=(2, 2))
        m2 = np.random.normal(size=(2, 2))
        m3 = np.random.normal(size=(2, 1))
        m4 = np.random.normal(size=(2, 3))
        
        # With 2 matrices
        K = KronOperator([m1, m2])
        m = np.kron(m1, m2)
        v = np.random.normal(size=K.shape[1])
        assert(np.allclose(m.dot(v), K.dot(v)))
        assert(np.allclose(m, K.todense()))

        # With 3 matrices
        K = KronOperator([m2, m1, m1])
        m = np.kron(m2, np.kron(m1, m1))
        v = np.random.normal(size=K.shape[1])
        assert(np.allclose(m.dot(v), K.dot(v)))
        assert(np.allclose(m, K.todense()))

        # Try different sizes
        K = KronOperator([m3, m1, m2])
        m = np.kron(m3, np.kron(m1, m2))
        v = np.random.normal(size=K.shape[1])
        assert(np.allclose(m.dot(v), K.dot(v)))
        assert(np.allclose(m, K.todense()))

        # Try with random matrices of different sizes
        K = KronOperator([m1, m2, m3, m4])
        m = np.kron(m1, np.kron(m2, np.kron(m3, m4)))
        v = np.random.normal(size=K.shape[1])
        assert(np.allclose(m.dot(v), K.dot(v)))
        assert(np.allclose(m, K.todense()))
        
        # Test transpose
        K_transpose = K.transpose()
        v = np.random.normal(size=K_transpose.shape[1])
        assert(np.allclose(m.T.dot(v), K_transpose.dot(v)))
        assert(np.allclose(m.T, K_transpose.todense()))
    
    def test_vj_projection_operator(self):
        a, l = 2, 2
        
        # Purely additive function
        y = np.array([-1.5, -0.5, 0.5, 1.5])
        y01 = np.array([-1, -1, 1, 1])
        y10 = np.array([-0.5, 0.5, -0.5, 0.5])
        
        Pj = VjProjectionOperator(a, l, j=[0])
        f01 = Pj.dot(y)
        assert(np.allclose(f01, y01))
        
        Pj = VjProjectionOperator(a, l, j=[1])
        f10 = Pj.dot(y)
        assert(np.allclose(f10, y10))
        
        Pj = VjProjectionOperator(a, l, j=[])
        f00 = Pj.dot(y)
        assert(np.allclose(f00, 0))
        
        Pj = VjProjectionOperator(a, l, j=[0, 1])
        f11 = Pj.dot(y)
        assert(np.allclose(f11, 0))
        
        # Tests that projections add up to the whole subspace in larger case
        a, l = 4, 5
        v = np.random.normal(size=a ** l)

        for k in range(1, 6):
            W = ProjectionOperator(a, l, k=k)
            u1 = W.dot(v)
            
            u2 = np.zeros(v.shape[0])
            for j in combinations(np.arange(W.l), k):
                Pj = VjProjectionOperator(a, l, j=list(j))
                u2 += Pj.dot(v)
            
            assert(np.allclose(u1, u2))
    
    def test_vj_projection_operator_sq_norm(self):
        a, l = 2, 2
        y = np.array([-1.5, -0.5, 0.5, 1.5])
        
        for j in [[], [0], [1], [0, 1]]:
            Pj = VjProjectionOperator(a, l, j=j)
            fsqn = Pj.dot_square_norm(y)
            exp = np.sum(Pj.dot(y) ** 2)
            assert(np.allclose(fsqn, exp))
        
        # Test with bigger operator
        Pj = VjProjectionOperator(4, 8, j=[0, 3, 5])
        y = np.random.normal(size=Pj.shape[1])
        exp = np.sum(Pj.dot(y) ** 2)
        fsqn = Pj.dot_square_norm(y)
        assert(np.allclose(fsqn, exp))
        
    def test_vj_basis_operator(self):
        # Test in small case
        B = VjBasisOperator(2, 2, j=(0, 1))
        b = B.todense()
        assert(b.shape == B.shape)
        assert(b.shape == (4, 1))
        
        dense_b = np.array([0.5, -0.5, -0.5, 0.5])
        assert(np.allclose(b.flatten(), dense_b))
        
        B = VjBasisOperator(2, 2, j=(0,))
        dense_b = np.array([-0.5, -0.5, 0.5, 0.5])
        print(B.todense().flatten())
        assert(np.allclose(B.todense().flatten(), dense_b))
        
        B = VjBasisOperator(2, 2, j=(1,))
        dense_b = np.array([-0.5, 0.5, -0.5, 0.5])
        assert(np.allclose(B.todense().flatten(), dense_b))

        # Test dimensions in larger operators        
        B = VjBasisOperator(4, 5, j=(0, 2))
        assert(B.shape == (4 ** 5, 9))

        v = np.random.normal(size=(9,))
        u = B @ v
        assert(u.shape[0] == 4 ** 5)

    def test_k_eigen_basis_operator(self):
        a, l = 4, 5
        v = np.random.normal(size=a ** l)
        
        for k in range(0, l+1):
            W = ProjectionOperator(a, l, k=k)
            B = EigenBasisOperator(a, l, k=k)
            u1 = W @ v
            u2 = B @ B.transpose_dot(v)
            assert(np.allclose(u1, u2))
    
    def test_delta_kernel_basis_operator(self):
        # Test small example explicitly
        a, l, P = 2, 2, 2
        B = DeltaKernelBasisOperator(a, l, P=P)
        B_dense = np.array([[1/2,  1/2,  1/2, 1/2],
                            [-1/2, -1/2, 1/2, 1/2],
                            [-1/2, 1/2, -1/2, 1/2]])
        b = np.vstack([B.dot(v) for v in np.eye(3)])
        assert(np.allclose(b, B_dense))
        
        # Test in a larger case
        a, l, P = 4, 5, 2
        B = DeltaKernelBasisOperator(a, l, P=P)
        
        # Ensure it is in the null space of DeltaP operator
        DP = DeltaPOperator(a, l, P=P)
        v = np.random.normal(size=B.shape[1])
        f = B.dot(v)
        assert(np.allclose(DP.dot(f), 0.))
        
        # Ensure it provides a valid projection matrix
        v = np.random.normal(size=B.shape[0])
        u1 = B @ B.transpose_dot(v)
        u2 = B @ B.transpose_dot(u1)
        assert(np.allclose(u1, u2))
        
        # Ensure it provides the right projection matrix
        v = np.random.normal(size=B.shape[0])
        u1 = B @ B.transpose_dot(v)
        u2 = 0.
        for k in range(P):
            W = ProjectionOperator(a, l, k=k)
            u2 += W.dot(v)
        assert(np.allclose(u1, u2))
    
    def test_rho_projection_operator(self):
        np.random.seed(0)
        
        # Small landscape
        a, l = 2, 2
        P = RhoProjectionOperator(a, l, rho=0.5)
        v = np.random.normal(size=P.n)
        u1 = P.dot(v)
        
        lambdas = np.array([1, 0.5, 0.25])
        W = ProjectionOperator(a, l, lambdas=lambdas)
        u2 = W.dot(v)
        assert(np.allclose(u1, u2))
        
        # Different rho per site
        P = RhoProjectionOperator(a, l, rho=[0.5, 0.8])
        v = np.random.normal(size=P.n)
        u1 = P.dot(v)
        
        # Do calculations using Vj Projections
        ws = [1 * 1, 1 * 0.8, 0.5 * 1, 0.8 * 0.5]
        js = [[], [1], [0], [0, 1]]
        u2 = np.zeros(v.shape)
        for w, j in zip(ws, js): 
            Pj = VjProjectionOperator(a, l, j=j)
            u2 += w * Pj.dot(v)
        assert(np.allclose(u1, u2))
        
        # Larger landscape with single rho model
        a, l = 4, 6
        P = RhoProjectionOperator(a, l, rho=0.5)
        v = np.random.normal(size=P.shape[1])
        u1 = P.dot(v)
        
        lambdas = np.array([2**-(i) for i in range(l+1)])
        W = ProjectionOperator(a, l, lambdas=lambdas)
        u2 = W.dot(v)
        assert(np.allclose(u1, u2))
        
    def test_kernel_operator(self):
        A = np.array([[1.0, 0.5, 0.5],
                      [0.5, 1.0, 0.5],
                      [0.5, 0.5, 1.0]])
        
        # Test plain kernel operator
        K = KernelOperator(A)
        v = np.random.normal(size=K.shape[1])
        assert(np.allclose(A.dot(v), K.dot(v)))
        
        # Solve using CG
        v = np.array([1, 2., 1])
        u = K.inv_dot(v)
        assert(np.allclose(K.dot(u), v))
        
        # Test different indexings
        x1 = np.array([0, 1], dtype=int)
        K = KernelOperator(A, x1=x1)
        assert(K.shape == [2, 3])
        
        v = np.random.normal(size=K.shape[1])
        assert(np.allclose(A[:2, :].dot(v), K.dot(v)))
        
        x2 = np.array([1, 2], dtype=int)
        K = KernelOperator(A, x2=x2)
        assert(K.shape == [3, 2])
        
        v = np.random.normal(size=K.shape[1])
        assert(np.allclose(A[:, 1:].dot(v), K.dot(v)))
        
        K = KernelOperator(A, x1=x1, x2=x2)
        assert(K.shape == [2, 2])
        
        v = np.random.normal(size=K.shape[1])
        assert(np.allclose(A[:2, 1:].dot(v), K.dot(v)))
        
        # Test additing diagonal
        y_var_diag = np.ones(2)
        B = A[:2, 1:] + np.eye(2)
        K = KernelOperator(A, x1=x1, x2=x2,
                           y_var_diag=y_var_diag)
        assert(K.shape == [2, 2])
        
        v = np.random.normal(size=K.shape[1])
        assert(np.allclose(B.dot(v), K.dot(v)))
        
    def test_calculate_variance_components(self):
        space = DataSet('gb1').to_sequence_space()
        lambdas = calc_variance_components(space)
        assert(np.all(lambdas > 0))
        
    def test_calc_vjs_variance_components(self):
        space = DataSet('gb1').to_sequence_space()
        
        vj1 = calc_vjs_variance_components(space, k=1)  
        assert(vj1[(2,)] > vj1[(0,)])
        assert(vj1[(3,)] > vj1[(1,)])
        
        vj2 = calc_vjs_variance_components(space, k=2)
        for v in vj2.values():
            assert(vj2[(2,3)] >= v)
        

class SkewedLinOpsTests(unittest.TestCase):
    def xtest_skewed_kernel_operator(self):
        ps = np.array([[0.3, 0.7], [0.5, 0.5]])
        log_p = np.log(ps)
        l, a = ps.shape
        
        # Define Laplacian based kernel
        L = LaplacianOperator(a, l, ps=ps)
        W = ProjectionOperator(L=L)
        K1 = VarianceComponentKernelOperator(W)
        
        # Define full kernel function
        kernel = VarianceComponentKernel(l, a, use_p=True)
        x = np.array(['AA', 'AB', 'BA', 'BB'])
        kernel.set_data(x1=x, alleles=['A', 'B'])
        
        # Constant component
        lambdas = [1, 0, 0]
        K1.set_lambdas(lambdas)
        k1 = K1.todense()
        assert(np.allclose(k1, 1))
        
        K2 = kernel(lambdas=lambdas, log_p=log_p)
        assert(np.allclose(K2, 1))
        
        # Additive component
        lambdas = [0, 1, 0]
        K1.set_lambdas(lambdas)
        k1 = K1.todense()
        k2 = kernel(lambdas=lambdas, log_p=log_p)
        assert(np.allclose(k1, k2))
        
        # Pairwise component
        lambdas = [0, 0, 1]
        K1.set_lambdas(lambdas)
        k1 = K1.todense()
        k2 = kernel(lambdas=lambdas, log_p=log_p)
        assert(np.allclose(k1, k2))
    
    def xtest_skewed_kernel_operator_big(self):
        l, a = 4, 4
        alleles = ALPHABET[:a]
        ps = np.random.dirichlet(alpha=np.ones(a), size=l)
        log_p = np.log(ps)
        
        # Define Laplacian based kernel
        L = LaplacianOperator(a, l, ps=ps)
        W = ProjectionOperator(L=L)
        K1 = VarianceComponentKernelOperator(W)
        I = np.eye(K1.shape[0])
        
        # Define full kernel function
        kernel = VarianceComponentKernel(l, a, use_p=True)
        x = np.array([x for x in generate_possible_sequences(l, alleles)])
        kernel.set_data(x1=x, alleles=alleles)
        
        # Test components
        for k in range(l+1):
            lambdas = np.zeros(l+1)
            lambdas[k] = 1
            K1.set_lambdas(lambdas)
            k1 = K1.dot(I)
            k2 = kernel(lambdas=lambdas, log_p=log_p)
            assert(np.allclose(k1, k2))

        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'LinOpsTests']
    unittest.main()
