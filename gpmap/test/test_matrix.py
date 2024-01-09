#!/usr/bin/env python
import unittest
import numpy as np

from scipy.sparse import csr_matrix

from gpmap.src.matrix import (calc_cartesian_product, calc_tensor_product,
                              calc_cartesian_product_dot, calc_tensor_product_dot,
                              calc_tensor_product_quad, quad,
                              calc_tensor_product_dot2, kron_dot,
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
        
    def test_tensor_product(self):
        matrix1 = np.array([[0.6],
                            [0.4]])
        matrix2 = np.array([[0.7],
                            [0.3]])
        expected = np.array([[0.42],
                             [0.18],
                             [0.28],
                             [0.12]])
        result = calc_tensor_product([matrix1, matrix2])
        assert(np.allclose(result, expected))
    
    def test_tensor_product_dot(self):
        m1 = 0.5 * np.array([[1, 1],
                             [1, 1]])
        m2 = 0.5 * np.array([[1, -1],
                             [-1, 1]])
        ms = [m1, m2]
        m = calc_tensor_product(ms)
        
        v = np.ones(4)
        u1 = m.dot(v)
        u2 = calc_tensor_product_dot(ms, v)
        assert(np.allclose(u1, u2))
        
        # Test in larger scenario
        ms = [m1, m2, m2, m1, m2, m1, m1]
        m = calc_tensor_product(ms)
        v = np.random.normal(size=m.shape[0])
        u1 = m.dot(v)
        u2 = calc_tensor_product_dot(ms, v)
        assert(np.allclose(u1, u2))
        
    def test_tensor_product_dot2(self):
        v = np.random.normal(size=4)
        m1 = 0.5 * np.array([[1, 1],
                             [1, 1]])
        m2 = 0.5 * np.array([[1, -1],
                             [-1, 1]])
        
        m = np.kron(m1, m2)
        u1 = m.dot(v)
        u2 = calc_tensor_product_dot2(m1, m2, v)
        assert(np.allclose(u1, u2))
        
        m = np.kron(m1, m1)
        u1 = m.dot(v)
        u2 = calc_tensor_product_dot2(m1, m1, v)
        assert(np.allclose(u1, u2))
        
        m = np.kron(m2, m2)
        u1 = m.dot(v)
        u2 = calc_tensor_product_dot2(m2, m2, v)
        assert(np.allclose(u1, u2))

        v = np.random.normal(size=8)
        m3 = np.kron(m1, m2)
        m = np.kron(m2, m3)
        u1 = m.dot(v)
        u2 = calc_tensor_product_dot2(m2, m3, v)
        assert(np.allclose(u1, u2))

        v = np.array([1, 1, 1, 1, 0 , 0, 0, 0])
        m = np.kron(m2, np.kron(m1, m1))
        u1 = m.dot(v)
        u2 = kron_dot([m2, m1, m1], v)
        assert(np.allclose(u1, u2))
        
    def test_tensor_product_quad(self):
        m1 = 0.5 * np.array([[1, 1],
                             [1, 1]])
        m2 = 0.5 * np.array([[1, -1],
                             [-1, 1]])
        
        # Test small case
        ms = [m1, m2]
        m = calc_tensor_product(ms)
        v = np.random.normal(size=m.shape[0])
        u1 = quad(m, v)
        u2 = calc_tensor_product_quad(ms, v, v)
        assert(np.allclose(u1, u2))

        # Test in larger scenario
        ms = [m1, m2, m2]
        m = calc_tensor_product(ms)
        v = np.random.normal(size=m.shape[0])
        u1 = quad(m, v)
        u2 = calc_tensor_product_quad(ms, v)
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
