#!/usr/bin/env python
import unittest
import numpy as np

from gpmap.linop import (LaplacianOperator, ProjectionOperator,
                             VarianceComponentKernel,
                             RhoProjectionOperator, ConnectednessKernel,
                             ExtendedLinearOperator)


class LinalgTests(unittest.TestCase):
    def xtest_calc_trace(self):
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
        
        K = VarianceComponentKernel(4, 5)
        K.set_lambdas(lambdas=[1, 0.5, 0.4, 0.1, 0.02, 0.001])
        trace = K.calc_trace(exact=True)
        assert(np.allclose(trace, W.get_diag().sum()))
        
        obs_idx = np.arange(200)
        y_var = np.exp(np.random.normal(scale=0.1, size=200))
        K.set_y_var(y_var, obs_idx)
        trace = K.calc_trace(exact=True)
        assert(trace - y_var.sum() < W.calc_trace())
        assert(np.allclose(trace, K.get_diag().sum()))
        
        K = ConnectednessKernel(4, 5)
        K.set_rho(0.5)
        trace = K.calc_trace(exact=True)
        assert(np.allclose(trace, P.get_diag().sum()))
        
        obs_idx = np.arange(200)
        y_var = np.exp(np.random.normal(scale=0.1, size=200))
        K.set_y_var(y_var, obs_idx)
        trace = K.calc_trace(exact=True)
        assert(trace - y_var.sum() < P.calc_trace())
        assert(np.allclose(trace, K.get_diag().sum()))
    
    def xtest_calc_trace_approx(self):
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
    
    def xtest_calc_projection_log_det(self):
        W = ProjectionOperator(2, 2)
        
        # zero determinant
        W.set_lambdas(lambdas=np.array([0, 1, 0]))
        log_det = W.calc_log_det()
        assert(np.allclose(log_det, -np.inf))
        
        # Non-zero determinant
        W.set_lambdas(lambdas=np.array([1, 0.5, 0.1]))
        log_det = W.calc_log_det()
        assert(np.allclose(log_det, 0 + 2 * np.log(0.5) + np.log(0.1)))
    
    def xtest_calc_log_det_approx(self):
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
    
    def xtest_calc_log_det_large_operator(self):
        lambdas = np.array([100, 50, 25, 12, 5, 2.5, 1, 0.5, 0.25])
        K = VarianceComponentKernelOperator(4, 8, lambdas=lambdas)
        K.set_y_var(y_var=np.ones(K.n), obs_idx=np.arange(K.n))
        true_logdet = np.sum(K.m_k * np.log(1 + lambdas))
        log_det = K.calc_log_det(degree=10, n_vectors=1, method='SLQ')
        assert(np.allclose(log_det, true_logdet, rtol=0.1))
    
    def xtest_arnoldi(self):
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
    
    def xtest_lanczos(self):
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
    
    # def xtest_lanczos_conjugate_gradient(self):
    #     A = np.ones((4, 4)) + np.diag(np.ones(4))
    #     x_true = np.random.normal(size=4)
    #     b = A.dot(x_true)
    #     print(A)
    #     print(x_true)
    #     print(b)
        
    #     tol = 1e-6
    #     x = lanczos_conjugate_gradient(A, b, tol=tol)
    #     print(x)
    #     assert(np.allclose(x, x_true, atol=tol))
    

if __name__ == '__main__':
    import sys;sys.argv = ['', 'LinalgTests']
    unittest.main()
