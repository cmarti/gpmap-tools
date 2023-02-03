#!/usr/bin/env python
import unittest
import numpy as np

from itertools import combinations
from timeit import timeit

from gpmap.src.linop import (LaplacianOperator, ProjectionOperator,
                             VjProjectionOperator, KernelOperator)


class LinOpsTests(unittest.TestCase):
    def test_laplacian(self):
        L = LaplacianOperator(2, 2)
        
        v = np.ones(4)
        assert(np.allclose(L.dot(v), 0))
        assert(np.allclose(L.dot(2*v), 0))
        
        v = np.array([1, 2, 1, 0])
        u = np.array([-1, 3, 1, -3])
        assert(np.allclose(L.dot(v), u))
    
    def test_laplacian_D_pi(self):
        L = LaplacianOperator(2, 2)
        assert(np.allclose(L.D_pi.data, 1))
        
        L = LaplacianOperator(2, 2, ps=np.array([[0.4, 0.6], [0.5, 0.5]]))
        pi = np.array([0.2, 0.2, 0.3, 0.3])
        assert(np.allclose(L.D_pi.data, pi))
    
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
    
    def test_projection_operator(self):
        W = ProjectionOperator(2, 2)
        
        # Purely additive function
        y = np.array([-1.5, -0.5, 0.5, 1.5])
        
        W.set_lambdas(k=2)
        print(W.dot(y))
        # assert(np.allclose(W.dot(y), 0))
        
        W.set_lambdas(k=1)
        print(W.dot(y))
        # assert(np.allclose(W.dot(y), y))
        
        W.set_lambdas(k=0)
        print(W.dot(y))
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
                vjp.set_j(j)
                u2 += vjp.dot(v)
            
            assert(np.allclose(u1, u2))
    
    def test_kernel_operator(self):
        lambdas = np.array([0, 1, 0])
        y_var = 0.1 * np.ones(4)
        obs_idx = np.arange(4)
        W = ProjectionOperator(2, 2)
        W.set_lambdas(lambdas)
        K = KernelOperator(W, y_var=y_var, obs_idx=obs_idx)
        v = np.random.normal(size=K.n_obs)        
        u = K.inv_dot(K.dot(v))
        assert(np.allclose(u, v))
        
        # Large matrix
        l = 9
        W = ProjectionOperator(4, l)
        lambdas = np.append([0], 2-np.arange(l))
        W.set_lambdas(lambdas)

        y_var = 0.1 * np.ones(W.n)
        obs_idx = np.arange(W.n)
        K = KernelOperator(W, y_var=y_var, obs_idx=obs_idx)
        v = np.random.normal(size=K.n_obs)
        u = K.inv_dot(K.dot(v))
        assert(np.allclose(u, v))
    
    def test_vj_projection_operator_ss(self):
        vjp = VjProjectionOperator(4, 5)
        v = np.random.normal(size=vjp.n)
        
        vjp.set_j([0, 2, 3])
        print(timeit(lambda: vjp.quad(v), number=10))
        print(timeit(lambda : vjp.dot_square_norm(v), number=10))
        
        # for k in range(1, 6):
        #     for j in combinations(np.arange(vjp.l), k):
        #         vjp.set_j(j)
        #         u = vjp.dot(v)
        #         ss1 = np.sum(u * u)
        #         ss2 = vjp.dot_square_norm(v)
        #         ss3 = vjp.quad(v) # only applies in equally weighted case
        #         assert(np.allclose(ss1, ss2))
        #         assert(np.allclose(ss1, ss3))
    
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'LinOpsTests']
    unittest.main()
