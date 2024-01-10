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
                             ExtendedLinearOperator,
                             calc_variance_components,
                             calc_vjs_variance_components)


class LinOpsTests(unittest.TestCase):
    def test_laplacian(self):
        L = LaplacianOperator(2, 2)
        
        v = np.ones(4)
        assert(np.allclose(L.dot(v), 0))
        assert(np.allclose(L.dot(2*v), 0))
        
        v = np.array([1, 2, 1, 0])
        u = np.array([-1, 3, 1, -3])
        assert(np.allclose(L.dot(v), u))
    
    def test_laplacian_eigenvalues(self):
        L = LaplacianOperator(2, 2)
        assert(np.allclose(L.lambdas, [0, 2, 4]))
        
        L = LaplacianOperator(2, 3)
        assert(np.allclose(L.lambdas, [0, 2, 4, 6]))
        
        L = LaplacianOperator(3, 2)
        assert(np.allclose(L.lambdas, [0, 3, 6]))
        
    def test_calc_polynomial_coeffs_analytical(self):
        for l in range(3, 9):
            W = ProjectionOperator(4, l)
            y = np.random.normal(size=W.n)
            
            B1 = W.calc_polynomial_coeffs(numeric=True)
            W.set_lambdas(k=l-1)
            p1 = W.dot(y)
            
            B2 = W.calc_polynomial_coeffs(numeric=False)
            W.set_lambdas(k=l-1)
            p2 = W.dot(y)
            
            # Test that coefficients are equal
            assert(np.allclose(B1, B2))
            
            # Test that projections are also equal
            assert(np.allclose(p1, p2))
    
    def test_calc_coefficients(self):
        for l in range(2, 14):
            W = ProjectionOperator(2, l)
            lambdas0 = 10 ** (-np.cumsum(np.exp(np.random.normal(size=W.lp1))))
            
            coeffs1 = W.lambdas_to_coeffs(lambdas0, use_lu=False)
            lambdas1 = W.V.dot(coeffs1)
            
            coeffs2 = W.lambdas_to_coeffs(lambdas0, use_lu=True)
            lambdas2 = W.V.dot(coeffs2) 
            
            assert(np.allclose(lambdas0, lambdas1, atol=1e-3))
            assert(np.allclose(lambdas0, lambdas2, atol=1e-3))
            assert(np.all(lambdas1 > -1e-6))
            assert(np.all(lambdas2 > -1e-6))
            
    def test_projection_operator(self):
        W = ProjectionOperator(2, 2)
        
        # Purely additive function
        y = np.array([-1.5, -0.5, 0.5, 1.5])
        
        W.set_lambdas(k=2)
        assert(np.allclose(W.dot(y), 0))
        
        W.set_lambdas(k=1)
        assert(np.allclose(W.dot(y), y))
        
        W.set_lambdas(k=0)
        assert(np.allclose(W.dot(y), 0))
        
        # Non-zero orthogonal projections
        y = np.array([-1.5, -0.5, 0.5, 4])
        
        W.set_lambdas(k=0)
        y0 = W.dot(y)
        W.set_lambdas(k=1)
        y1 = W.dot(y)
        W.set_lambdas(k=2)
        y2 = W.dot(y) 
        
        assert(not np.allclose(y0, 0))
        assert(not np.allclose(y1, y))
        assert(not np.allclose(y2, 0))
        
        # Ensure they are orthogonal to each other
        assert(np.allclose(y0.T.dot(y1), 0))
        assert(np.allclose(y0.T.dot(y2), 0))
        assert(np.allclose(y1.T.dot(y2), 0))
        
        # Test inverse
        W.set_lambdas(np.array([1, 10, 1.]))
        assert(np.allclose(W.inv_dot(W.dot(y)), y))
    
    def test_projection_operator_large(self):
        for l in range(10, 14):
            W = ProjectionOperator(seq_length=l, n_alleles=2)
            
            for _ in range(10):
                v = np.random.normal(size=W.shape[0])
                
                for k in range(l+1):
                    W.set_lambdas(k=k)
                    u = W.dot(v)
                    q = np.sum(u**2)
                    assert(q >= 0)
    
    def xtest_projection_operator2(self):
        # TODO: recursion problem
        W = ProjectionOperator2(2, 2)
         
        # Purely additive function
        y = np.array([-1.5, -0.5, 0.5, 1.5])
         
        W.set_lambdas(k=2)
        assert(np.allclose(W.dot(y), 0))
         
        W.set_lambdas(k=1)
        assert(np.allclose(W.dot(y), y))
         
        W.set_lambdas(k=0)
        assert(np.allclose(W.dot(y), 0))
         
        # Non-zero orthogonal projections
        y = np.array([-1.5, -0.5, 0.5, 4])
         
        W.set_lambdas(k=0)
        y0 = W.dot(y)
        W.set_lambdas(k=1)
        y1 = W.dot(y)
        W.set_lambdas(k=2)
        y2 = W.dot(y) 
         
        assert(not np.allclose(y0, 0))
        assert(not np.allclose(y1, y))
        assert(not np.allclose(y2, 0))
         
        # Ensure they are orthogonal to each other
        assert(np.allclose(y0.T.dot(y1), 0))
        assert(np.allclose(y0.T.dot(y2), 0))
        assert(np.allclose(y1.T.dot(y2), 0))
         
        # Test inverse
        W.set_lambdas(np.array([1, 10, 1.]))
        assert(np.allclose(W.inv_dot(W.dot(y)), y))
    
    def test_projection_operator_times(self):
        a, l = 4, 7
        W1 = ProjectionOperator(seq_length=l, n_alleles=a)
        W2 = ProjectionOperator2(seq_length=l, n_alleles=a)
        
        v = np.random.normal(size=W1.n)
        
        lambdas = 10.**(-np.arange(l+1))
        W1.set_lambdas(lambdas=lambdas)
        W2.set_lambdas(lambdas=lambdas)
        
        u1 = W1.dot(v)
        u2 = W2.dot(v)
        assert(np.allclose(u1, u2))
        
        W1.set_lambdas(k=2)
        W2.set_lambdas(k=2)
        
        print(timeit(lambda : W1.dot(v), number=10))
        print(timeit(lambda : W2.dot(v), number=10))
    
    def test_vj_projection_operator(self):
        vjp = VjProjectionOperator(2, 2)
        
        # Purely additive function
        y = np.array([-1.5, -0.5, 0.5, 1.5])
        y01 = np.array([-1, -1, 1, 1])
        y10 = np.array([-0.5, 0.5, -0.5, 0.5])
        
        vjp.set_j([0])
        f01 = vjp.dot(y)
        assert(np.allclose(f01, y01))
        
        vjp.set_j([1])
        f10 = vjp.dot(y)
        assert(np.allclose(f10, y10))
        
        vjp.set_j([])
        assert(np.allclose(vjp.dot(y), 0))
        
        vjp.set_j([0, 1])
        assert(np.allclose(vjp.dot(y), 0))
        
        # Tests that projections add up to the whole subspace in larger case
        W = ProjectionOperator(4, 5)
        vjp = VjProjectionOperator(4, 5)
        v = np.random.normal(size=W.n)

        for k in range(1, 6):
            W.set_lambdas(k=k)
            u1 = W.dot(v)
            
            u2 = np.zeros(v.shape[0])
            for j in combinations(np.arange(W.l), k):
                vjp.set_j(list(j))
                u2 += vjp.dot(v)
            
            assert(np.allclose(u1, u2))
    
    def test_vj_projection_operator_sq_norm(self):
        vjp = VjProjectionOperator(2, 2)
        y = np.array([-1.5, -0.5, 0.5, 1.5])
        
        vjp.set_j([0])
        f = vjp.dot(y)
        fsqn = vjp.dot_square_norm(y)
        assert(np.allclose(fsqn, np.sum(f**2)))
        
        vjp.set_j([1])
        f = vjp.dot(y)
        fsqn = vjp.dot_square_norm(y)
        assert(np.allclose(fsqn, np.sum(f**2)))
        
        vjp.set_j([])
        f = vjp.dot(y)
        fsqn = vjp.dot_square_norm(y)
        assert(np.allclose(fsqn, np.sum(f**2)))
        
        vjp.set_j([0, 1])
        f = vjp.dot(y)
        fsqn = vjp.dot_square_norm(y)
        assert(np.allclose(fsqn, np.sum(f**2)))
        
    def test_vj_projection_operator_sq_norm_big(self):
        vjp = VjProjectionOperator(4, 8)
        vjp.set_j([0, 3, 5])
        y = np.random.normal(size=vjp.n)
        f = vjp.dot(y)
        fsqn = vjp.dot_square_norm(y)
        assert(np.allclose(fsqn, np.sum(f**2)))
    
    def test_rho_projection_operator(self):
        # Small landscape
        P = RhoProjectionOperator(2, 2)
        P.set_rho(0.5)
        
        np.random.seed(0)
        v = np.random.normal(size=P.n)
        u1 = P.dot(v)
        
        W = ProjectionOperator(2, 2)
        W.set_lambdas(np.array([1, 0.5, 0.25]))
        u2 = W.dot(v)
        assert(np.allclose(u1, u2))
        
        # Larger landscape
        P = RhoProjectionOperator(4, 6)
        P.set_rho(0.5)
        
        np.random.seed(0)
        v = np.random.normal(size=P.n)
        u1 = P.dot(v)
        
        W = ProjectionOperator(4, 6)
        W.set_lambdas(np.array([2**-(i) for i in range(W.lp1)]))
        u2 = W.dot(v)
        assert(np.allclose(u1, u2))
        
        # Different rho per site
        P = RhoProjectionOperator(2, 2)
        P.set_rho([0.5, 0.8])
        
        np.random.seed(0)
        v = np.random.normal(size=P.n)
        u1 = P.dot(v)
        
        W = VjProjectionOperator(2, 2)
        u2 = np.zeros(v.shape)
        
        W.set_j([])
        u2 += 1 * 1 * W.dot(v)
        
        W.set_j([1])
        u2 += 1 * 0.8 * W.dot(v)
        
        W.set_j([0])
        u2 += 0.5 * 1 * W.dot(v)
        
        W.set_j([0, 1])
        u2 += 0.5 * 0.8 * W.dot(v)
        assert(np.allclose(u1, u2))
        
    def test_kernel_opt_get_gt_to_data_matrix(self):
        K = VarianceComponentKernelOperator(n_alleles=2, seq_length=3)
        obs_idx = np.arange(K.n)
        K.calc_gt_to_data_matrix(obs_idx)
        m = K.gt2data.todense()
        assert(np.all(m == np.eye(8)))
        
        K.calc_gt_to_data_matrix(obs_idx=np.array([0, 1, 2, 3]))
        m = K.gt2data.todense()
        assert(np.all(m[:4, :] == np.eye(4)))
        assert(np.all(m[4:, :] == 0))
        assert(m.shape == (8, 4))
    
    def test_vc_kernel_operator_large(self):
        l = 9
        K = VarianceComponentKernelOperator(n_alleles=4, seq_length=l)
        K.set_lambdas(lambdas=np.append([0], 2-np.arange(l)))
        K.set_y_var(y_var=0.1 * np.ones(K.n), obs_idx=np.arange(K.n))
        v = np.random.normal(size=K.n_obs)
        u = K.inv_dot(K.dot(v))
        assert(np.allclose(u, v))
    
    def test_rho_kernel_operator(self):
        l = 8
        K = ConnectednessKernelOperator(n_alleles=4, seq_length=l)
        K.set_rho(0.8)
        K.set_y_var(y_var=0.1 * np.ones(K.n), obs_idx=np.arange(K.n))
        v = np.random.normal(size=K.n_obs)
        u = K.inv_dot(K.dot(v))
        assert(np.allclose(u, v))
    
    def test_DeltaP_operator(self):
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
    
    def test_DeltaP_operator_eigenvalues(self):
        l, a, P = 5, 4, 2
        DP = DeltaPOperator(P=P, n_alleles=a, seq_length=l)
        DP.calc_lambdas()
        
        for k in range(P, l+1):
            lambda_k = a ** P * comb(k, P)
            assert(DP.lambdas[k] == lambda_k)
    
    def test_DP_calc_kernel_basis(self):
        DP = DeltaPOperator(P=2, n_alleles=4, seq_length=5)
        DP.calc_kernel_basis()
        basis = DP.kernel_basis
        
        # Ensure basis is orthonormal
        prod = basis.T.dot(basis)
        identity = get_sparse_diag_matrix(np.ones(prod.shape[0]))
        assert(np.allclose((prod - identity).todense(), 0))
        
        # Ensure basis is sparse
        max_values = basis.shape[0] * basis.shape[1]
        assert(basis.data.shape[0] < max_values) 
        
        # Ensure they generate good projection matrices
        u = np.dot(basis, basis.T).dot(basis)
        error = (basis - u).mean()
        assert(np.allclose(error, 0))
    
    def test_calc_trace(self):
        P = RhoProjectionOperator(4, 5)
        P.set_rho(0.5)
        trace = P.calc_trace(exact=True)
        assert(np.allclose(trace, P.get_diag().sum()))
        
        L = LaplacianOperator(4, 5)
        trace = L.calc_trace(exact=True)
        assert(np.allclose(trace, L.get_diag().sum()))
        
        W = ProjectionOperator(4, 5)
        W.set_lambdas(lambdas=[1, 0.5, 0.4, 0.1, 0.02, 0.001])
        trace = W.calc_trace(exact=True)
        assert(np.allclose(trace, W.get_diag().sum()))
        
        K = VarianceComponentKernelOperator(4, 5)
        K.set_lambdas(lambdas=[1, 0.5, 0.4, 0.1, 0.02, 0.001])
        trace = K.calc_trace(exact=True)
        assert(np.allclose(trace, W.get_diag().sum()))
        
        obs_idx = np.arange(200)
        y_var = np.exp(np.random.normal(scale=0.1, size=200))
        K.set_y_var(y_var, obs_idx)
        trace = K.calc_trace(exact=True)
        assert(trace - y_var.sum() < W.calc_trace())
        assert(np.allclose(trace, K.get_diag().sum()))
        
        K = ConnectednessKernelOperator(4, 5)
        K.set_rho(0.5)
        trace = K.calc_trace(exact=True)
        assert(np.allclose(trace, P.get_diag().sum()))
        
        obs_idx = np.arange(200)
        y_var = np.exp(np.random.normal(scale=0.1, size=200))
        K.set_y_var(y_var, obs_idx)
        trace = K.calc_trace(exact=True)
        assert(trace - y_var.sum() < P.calc_trace())
        assert(np.allclose(trace, K.get_diag().sum()))
    
    def test_calc_trace_approx(self):
        P = RhoProjectionOperator(4, 5)
        P.set_rho(0.5)
        trace = P.calc_trace(exact=False, n_vectors=100)
        assert(np.allclose(trace, P.get_diag().sum(), rtol=0.01))
        
        L = LaplacianOperator(4, 5)
        trace = L.calc_trace(exact=False, n_vectors=100)
        assert(np.allclose(trace, L.get_diag().sum(), rtol=0.01))
        
        W = ProjectionOperator(4, 5)
        W.set_lambdas(lambdas=[1, 0.5, 0.4, 0.1, 0.02, 0.001])
        trace = W.calc_trace(exact=False, n_vectors=100)
        assert(np.allclose(trace, W.get_diag().sum(), rtol=0.01))
    
    def test_calc_projection_log_det(self):
        W = ProjectionOperator(2, 2)
        
        # zero determinant
        W.set_lambdas(lambdas=np.array([0, 1, 0]))
        log_det = W.calc_log_det()
        assert(np.allclose(log_det, -np.inf))
        
        # Non-zero determinant
        W.set_lambdas(lambdas=np.array([1, 0.5, 0.1]))
        log_det = W.calc_log_det()
        assert(np.allclose(log_det, 0 + 2 * np.log(0.5) + np.log(0.1)))
    
    def test_calc_log_det_approx(self):
        A = np.array([[0.5, 0.15, 0.05, 0.3],
                      [0.15, 0.5, 0.3, 0.05],
                      [0.05, 0.3, 0.5, 0.15],
                      [0.3, 0.05, 0.15, 0.5]])
        linop = ExtendedLinearOperator(shape=(4, 4), matvec=A.dot)
        true_logdet = linop.calc_log_det(degree=10, n_vectors=100, method='naive')
        
#         log_det = linop.calc_log_det(degree=20, method='taylor')
#         assert(np.allclose(log_det, true_logdet, rtol=0.1))
#         
#         log_det = linop.calc_log_det(degree=20, n_vectors=100, method='barry_pace99')
#         assert(np.allclose(log_det, true_logdet, rtol=0.1))

        log_det = linop.calc_log_det(degree=20, n_vectors=100, method='SLQ')        
        assert(np.allclose(log_det, true_logdet, rtol=0.1))
    
    def test_calc_log_det_large_operator(self):
        lambdas = np.array([100, 50, 25, 12, 5, 2.5, 1, 0.5, 0.25])
        K = VarianceComponentKernelOperator(4, 8, lambdas=lambdas)
        K.set_y_var(y_var=np.ones(K.n), obs_idx=np.arange(K.n))
        true_logdet = np.sum(K.m_k * np.log(1 + lambdas))
        log_det = K.calc_log_det(degree=10, n_vectors=1, method='SLQ')
        assert(np.allclose(log_det, true_logdet, rtol=0.1))
    
    def test_arnoldi(self):
        A = np.array([[0.5, 0.15, 0.05, 0.2],
                      [0.15, 0.5, 0.2, 0.05],
                      [0.05, 0.2, 0.5, 0.15],
                      [0.2, 0.05, 0.15, 0.5]])
        linop = ExtendedLinearOperator(shape=(4, 4), matvec=A.dot)
        v0 = np.array([1, 0, 0, 0])
        Q, H = linop.arnoldi(v0, n_vectors=4)
        assert(np.allclose(Q.T @ Q, np.eye(Q.shape[1])))
        assert(np.allclose(A.dot(Q), Q.dot(H)))
        
        assert(np.allclose(np.diag(H, 1), np.diag(H, -1)))
        assert(np.allclose(np.diag(H, 2), 0))
        assert(np.allclose(np.diag(H, 3), 0))
    
    def test_lanczos(self):
        A = np.array([[0.5, 0.15, 0.05, 0.2],
                      [0.15, 0.5, 0.2, 0.05],
                      [0.05, 0.2, 0.5, 0.15],
                      [0.2, 0.05, 0.15, 0.5]])
        linop = ExtendedLinearOperator(shape=(4, 4), matvec=A.dot)
        v0 = np.array([1, 0, 0, 0])
        
        Q, T = linop.lanczos(v0, n_vectors=4, full_orth=False)
        assert(np.allclose(Q.T @ Q, np.eye(Q.shape[1])))
        assert(np.allclose(A.dot(Q), Q.dot(T)))
        
        Q, T = linop.lanczos(v0, n_vectors=4, full_orth=True)
        assert(np.allclose(Q.T @ Q, np.eye(Q.shape[1])))
        assert(np.allclose(A.dot(Q), Q.dot(T)))
        
        T2 = linop.lanczos(v0, n_vectors=4, return_Q=False)[1]
        assert(np.allclose(T, T2))
    
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
    def test_skewed_kernel_operator(self):
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
    
    def test_skewed_kernel_operator_big(self):
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
