#!/usr/bin/env python
import unittest
import numpy as np

from scipy.sparse import csr_matrix

from gpmap.src.matrix import (calc_cartesian_product, calc_cartesian_product_dot, 
                              diag_pre_multiply, diag_post_multiply,
                              lanczos_conjugate_gradient, rate_to_jump_matrix)


class MatrixTests(unittest.TestCase):
    def test_diag_pre_multiply(self):
        d = np.array([2, 1])
        m = np.array([[1, 2],
                      [2, 3]])
        v = np.array([1, 2])
        
        r = diag_pre_multiply(d, v)
        assert(np.allclose(r, np.array([2, 2])))
        
        r = diag_pre_multiply(d, m)
        assert(np.allclose(r, np.array([[2, 4], [2, 3]])))
        
    def test_diag_post_multiply(self):
        d = np.array([2, 1])
        m = np.array([[1, 2],
                      [2, 3]])
        v = np.array([1, 2])
        
        r = diag_post_multiply(d, v)
        assert(np.allclose(r, np.array([2, 2])))
        
        r = diag_post_multiply(d, m)
        assert(np.allclose(r, np.array([[2, 2], [4, 3]])))
        
    def test_cartesian_product(self):
        # With adjacency matrices
        matrix = np.array([[0, 1],
                           [1, 0]])
        expected = np.array([[0, 1, 1, 0],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 1, 0]])
        
        result = calc_cartesian_product([matrix, matrix])
        assert(np.all(result == expected))
        
        # With more than 2 matrices
        matrix = np.array([[0, 1],
                           [1, 0]])
        expected = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                             [1, 0, 0, 1, 0, 1, 0, 0],
                             [1, 0, 0, 1, 0, 0, 1, 0],
                             [0, 1, 1, 0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0, 1, 1, 0],
                             [0, 1, 0, 0, 1, 0, 0, 1],
                             [0, 0, 1, 0, 1, 0, 0, 1],
                             [0, 0, 0, 1, 0, 1, 1, 0]])
        
        result = calc_cartesian_product([matrix, matrix, matrix])
        assert(np.all(result == expected))
        
        # With pseudo-rate matrices (without diagonals)
        matrix = np.array([[0, 0.6],
                           [0.4, 0]])
        expected = np.array([[0, 0.6, 0.6, 0],
                             [0.4, 0, 0, 0.6],
                             [0.4, 0, 0, 0.6],
                             [0, 0.4, 0.4, 0]])
        
        result = calc_cartesian_product([matrix, matrix])
        assert(np.all(result == expected))
        
        # With different pseudo-rate matrices (without diagonals)
        matrix1 = np.array([[0, 0.6],
                            [0.4, 0]])
        matrix2 = np.array([[0, 0.7],
                            [0.3, 0]])
        expected = np.array([[0, 0.7, 0.6, 0],
                             [0.3, 0, 0, 0.6],
                             [0.4, 0, 0, 0.7],
                             [0, 0.4, 0.3, 0]])
        result = calc_cartesian_product([matrix1, matrix2])
        assert(np.all(result == expected))
        
        # With boolean matrices
        matrix = np.array([[False, True],
                           [True, False]])
        expected = np.array([[False, True, True, False],
                             [True, False, False, True],
                             [True, False, False, True],
                             [False, True, True, False]])
        
        result = calc_cartesian_product([matrix, matrix])
        assert(result.dtype == bool)
        assert(np.all(result == expected))
    
    def test_cartesian_product_dot(self):
        m1 = np.array([[0, 1],
                      [1, 0]])
        ms = [m1, m1]
        m2 = calc_cartesian_product(ms)
        v = np.random.normal(size=m2.shape[0])
        u1 = m2.dot(v)
        u2 = calc_cartesian_product_dot([m1, m1], v)
        assert(np.allclose(u1, u2))
        
        ms = [m1, m1, m1, m1, m1, m1, m1]
        m2 = calc_cartesian_product(ms)
        v = np.random.normal(size=m2.shape[0])
        u1 = m2.dot(v)
        u2 = calc_cartesian_product_dot(ms, v)
        assert(np.allclose(u1, u2))
        
    def test_lanczos_conjugate_gradient(self):
        A = np.ones((4, 4)) + np.diag(np.ones(4))
        x_true = np.random.normal(size=4)
        b = A.dot(x_true)
        print(A)
        print(x_true)
        print(b)
        
        tol = 1e-6
        x = lanczos_conjugate_gradient(A, b, tol=tol)
        print(x)
        assert(np.allclose(x, x_true, atol=tol))
    
    def test_rate_to_jump_matrix(self):
        # Test easy matrix
        Q = np.array([[-2, 1, 1],
                      [1, -2, 1],
                      [1, 1, -2]])
        P = rate_to_jump_matrix(Q)
        assert(np.allclose(np.diag(P), 0))
        assert(np.allclose(P.sum(1), 1))
        assert(np.allclose(P[0, 1], 0.5))
        assert(np.allclose(P[1, 0], 0.5))
        assert(np.allclose(P[2, 0], 0.5))
        
        # Test matrix with absorbing state
        Q = np.array([[-2, 1, 1],
                      [1, -2, 1],
                      [0, 0, 0]])
        P1 = rate_to_jump_matrix(Q)
        assert(np.allclose(np.diag(P1[:-1, :-1]), 0))
        assert(np.allclose(P1.sum(1), 1))
        assert(np.allclose(P1[-1, -1], 1))
        assert(np.allclose(P1[0, 1], 0.5))
        assert(np.allclose(P1[1, 0], 0.5))
        
        # Ensure it works with csr_matrix as well
        P2 = rate_to_jump_matrix(csr_matrix(Q)).todense()
        assert(np.allclose(P1, P2))
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'MatrixTests']
    unittest.main()
