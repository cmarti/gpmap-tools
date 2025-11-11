#!/usr/bin/env python
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from gpmap.matrix import (
    calc_cartesian_product,
    calc_cartesian_product_dot,
    diag_post_multiply,
    diag_pre_multiply,
    dot_log,
    kron,
    pivoted_cholesky_with_diag,
    rate_to_jump_matrix,
    tensordot,
)


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
        
    def test_tensordot(self):
        m = np.random.normal(size=(1, 2))
        
        v = np.random.normal(size=(2, 2, 4))
        assert(np.allclose(v, v.reshape(2, 8).reshape(2, 2, 4)))
        
        x1 = np.tensordot(m, v, axes=([1], [0]))
        x2 = tensordot(m, v, axis=0)
        assert(np.allclose(x1, x2))
        
        v = np.random.normal(size=(2, 2, 4))
        x1 = np.tensordot(m, v, axes=([1], [1]))
        x2 = tensordot(m, v, axis=1)
        assert(np.allclose(x1, x2))
        
        v = np.random.normal(size=(4, 2, 2))
        x1 = np.tensordot(m, v, axes=([1], [2]))
        x2 = tensordot(m, v, axis=2)
        assert(np.allclose(x1, x2))
        
    def test_diag_post_multiply(self):
        d = np.array([2, 1])
        m = np.array([[1, 2],
                      [2, 3]])
        v = np.array([1, 2])
        
        r = diag_post_multiply(d, v)
        assert(np.allclose(r, np.array([2, 2])))
        
        r = diag_post_multiply(d, m)
        assert(np.allclose(r, np.array([[2, 2], [4, 3]])))
    
    # def test_pivoted_cholesky(self):
    #     np.random.seed(1)
    #     M = 0.25 * np.exp(np.random.normal(size=(4, 4)))
    #     M = np.eye(4) + M @ M.T
        
    #     rank = 4
    #     U, piv = pivoted_cholesky_with_diag(M, np.diag(M), rank=rank)
    #     i = np.arange(rank)
    #     U[:, i] = U[:, piv]
    #     M_reconstructed = U.T @ U
    #     L = U.T

    #     L_inv = np.linalg.inv(L[:rank, :rank])
    #     M_inv_approx = L_inv.T @ L_inv
    #     P = np.eye(4)[piv]
    #     M_inv_reconstructed = P.T @ M_inv_approx @ P

    #     print('here')
    #     print(M)
    #     print(M_reconstructed)

    #     print(M_inv_reconstructed @ M)

    #     assert(np.allclose(M_reconstructed, M))
    #     assert(np.allclose(L @ L.T, M))
    
    def test_dot_log(self):
        # Matrix vector test
        m = np.array([[1, 2],
                      [2, 3]])
        v = np.array([4, 2])
        expected = np.log(m @ v)
        logm, signm = np.log(np.abs(m)), np.sign(m)
        logv, signv = np.log(np.abs(v)), np.sign(v)
        res, sign = dot_log(logm, signm, logv, signv)
        assert(np.allclose(expected, res))
        assert(np.allclose(sign, 1))

        # Test with negative entries in output
        m = np.array([[1, 2],
                      [2, 3]])
        v = np.array([1, -2])
        res = m @ v.T
        expected = np.log(np.abs(res))
        expected_sign = np.sign(res)
        logm, signm = np.log(np.abs(m)), np.sign(m)
        logv, signv = np.log(np.abs(v)), np.sign(v)
        res, sign = dot_log(logm, signm, logv, signv)
        assert(np.allclose(expected, res))
        assert(np.allclose(expected_sign, sign))

        # Test matrix-matrix product
        m = np.array([[1, 2],
                      [2, 3]])
        v = np.array([[4, 2],
                      [1, 1]])
        expected = np.log(m @ v)
        logm, signm = np.log(np.abs(m)), np.sign(m)
        logv, signv = np.log(np.abs(v)), np.sign(v)
        res, sign = dot_log(logm, signm, logv, signv)
        assert(np.allclose(expected, res))
        assert(np.allclose(sign, 1))

    def test_kron_product(self):
        m = np.array([[1, 2],
                      [2, 3]])
        n = np.array([[4, 2],
                      [2, 2]])
        
        # Test kron of two matrices
        p1 = np.kron(m, m)
        p2 = kron([m, m])
        assert(np.allclose(p1, p2))
        
        p1 = np.kron(m, n)
        p2 = kron([m, n])
        assert(np.allclose(p1, p2))
        
        # Test more than 2 matrices
        p1 = np.kron(m, np.kron(n, m))
        p2 = kron([m, n, m])
        assert(np.allclose(p1, p2))
        
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
