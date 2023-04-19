#!/usr/bin/env python
import unittest
import numpy as np

from itertools import combinations
from timeit import timeit

from gpmap.src.settings import ALPHABET
from gpmap.src.seq import generate_possible_sequences
from gpmap.src.utils import get_sparse_diag_matrix
from gpmap.src.kernel import VarianceComponentKernel
from gpmap.src.linop import (LaplacianOperator, ProjectionOperator,
                             VjProjectionOperator, KernelOperator,
                             DeltaPOperator)


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
        
        ps = np.array([[0.4, 0.6], [0.5, 0.5]])
        L = LaplacianOperator(2, 2, ps=ps)
        assert(np.allclose(L.lambdas, [0, 1, 2]))

    def test_laplacian_split(self):
        L1 = LaplacianOperator(4, 7)
        L2 = LaplacianOperator(4, 7, max_size=500)
        
        for d in L2.Kns_shape:
            assert(d[0] < 500)

        v = np.ones(L1.n)
        assert(np.allclose(L1.dot(v), 0))
        assert(np.allclose(L2.dot(v), 0))
        
        v = np.random.normal(size=L1.shape[0])
        L2.dot(v)
        
        print(timeit(lambda: L1.dot0(v), number=10))
        print(timeit(lambda : L1.dot1(v), number=10))
        print(timeit(lambda : L2.dot1(v), number=10))
        
        assert(np.allclose(L1.dot0(v), L1.dot1(v)))
        assert(np.allclose(L1.dot0(v), L2.dot1(v)))
        
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
    
    def test_vj_projection_operator_mixed(self):
        vjp = VjProjectionOperator(4, 5)
        v = np.random.normal(size=vjp.n)
        vjp.set_j([2, 3])
        u1 = vjp.dot(v)
        
        vjp2 = VjProjectionOperator(4, 5, max_size=100)
        vjp2.set_j([2, 3])
        u2 = vjp2.dot(v)
        assert(np.allclose(u1, u2))
    
    def test_kernel_operator(self):
        lambdas = np.array([0, 1, 0])
        y_var = 0.1 * np.ones(4)
        obs_idx = np.arange(4)
        W = ProjectionOperator(2, 2)
        W.set_lambdas(lambdas)
        K = KernelOperator(W)
        K.set_y_var(y_var=y_var, obs_idx=obs_idx)
        v = np.random.normal(size=K.n_obs)
        u = K.inv_dot(K.dot(v))
        assert(np.allclose(u, v))
        
        # Large matrix
        l = 9
        K = KernelOperator(ProjectionOperator(4, l))
        K.set_y_var(y_var=0.1 * np.ones(K.n), obs_idx=np.arange(K.n))
        K.set_lambdas(lambdas=np.append([0], 2-np.arange(l)))
        
        v = np.random.normal(size=K.n_obs)
        u = K.inv_dot(K.dot(v))
        assert(np.allclose(u, v))
    
    def test_kernel_opt_get_gt_to_data_matrix(self):
        vc = KernelOperator(ProjectionOperator(2, 3))
        obs_idx = np.arange(vc.n)
        vc.calc_gt_to_data_matrix(obs_idx)
        m = vc.gt2data.todense()
        assert(np.all(m == np.eye(8)))
        
        vc.calc_gt_to_data_matrix(obs_idx=np.array([0, 1, 2, 3]))
        m = vc.gt2data.todense()
        assert(np.all(m[:4, :] == np.eye(4)))
        assert(np.all(m[4:, :] == 0))
        assert(m.shape == (8, 4))
    
    def test_skewed_kernel_operator(self):
        ps = np.array([[0.3, 0.7], [0.5, 0.5]])
        log_p = np.log(ps)
        l, a = ps.shape
        
        # Define Laplacian based kernel
        L = LaplacianOperator(a, l, ps=ps)
        W = ProjectionOperator(L=L)
        K1 = KernelOperator(W)
        
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
        K1 = KernelOperator(W)
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
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'LinOpsTests']
    unittest.main()
