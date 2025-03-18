#!/usr/bin/env python
import numpy as np

try:
    from scipy.sparse.linalg.interface import _CustomLinearOperator
except ImportError:
    from scipy.sparse.linalg._interface import _CustomLinearOperator

from scipy.linalg import eigh_tridiagonal

from gpmap.utils import check_error
from gpmap.linop import SubMatrixOperator
from gpmap.matrix import inner_product


class ExtendedLinearOperator(_CustomLinearOperator):
    def _init_dtype(self):
        v = np.random.normal(size=3)
        self.dtype = v.dtype

    def rowsum(self):
        v = np.ones(self.shape[0])
        return self.dot(v)

    def submatrix(self, row_idx=None, col_idx=None):
        return SubMatrixOperator(self, row_idx, col_idx)

    def todense(self):
        return self.dot(np.eye(self.shape[1]))

    def quad(self, v):
        return np.sum(v * self.dot(v))

    def rayleigh_quotient(self, v, metric=None):
        return self.quad(v) / inner_product(v, v, metric=metric)

    def get_column(self, i):
        vec = np.zeros(self.shape[1])
        vec[i] = 1
        return self.dot(vec)

    def get_diag(self):
        return np.array([self.get_column(i)[i] for i in range(self.shape[0])])

    def calc_trace_hutchinson(self, n_vectors):
        """
        Stochastic trace estimator from

        Hutchinson, M. F. (1990). A stochastic estimator of the trace of
        the influence matrix for laplacian smoothing splines. Communications
        in Statistics - Simulation and Computation, 19(2), 433–450.
        """
        trace = np.array(
            [
                self.quad(np.random.normal(size=self.shape[1]))
                for _ in range(n_vectors)
            ]
        )
        return trace

    def calc_trace(self, exact=True, n_vectors=10):
        if exact or n_vectors > self.shape[1]:
            if hasattr(self, "_calc_trace"):
                trace = self._calc_trace()
            else:
                trace = self.get_diag().sum()
        else:
            trace = self.calc_trace_hutchinson(n_vectors).mean()

        return trace

    def calc_eigenvalue_upper_bound(self):
        return self.rowsum().max()

    def arnoldi(self, r, n_vectors):
        """
        Arnoldi algorithm based on
        https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf
        """
        n_vectors = min(n_vectors, self.shape[1])
        Q = np.expand_dims(r / np.norm(r), 1)
        H = np.zeros((n_vectors + 1, n_vectors))

        for j in range(n_vectors):
            q_j = Q[:, -1]
            r = self.dot(q_j)
            for i in range(j + 1):
                q_i = Q[:, i]
                p = np.dot(q_i, r)
                r -= p * q_i
                H[i, j] = p
            r_norm = np.linalg.norm(r)
            q = r / r_norm
            H[j + 1, j] = r_norm
            Q = np.append(Q, np.expand_dims(q, 1), 1)

            if np.allclose(r_norm, 0, atol=np.finfo(q.dtype).eps):
                return (Q, H[:j, :][:, :j])

        return (Q[:, :-1], H[:-1, :])

    def lanczos(self, r, n_vectors, full_orth=False, return_Q=True):
        """
        Lanczos tridiagonalization algorithm based on
        https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf
        """
        n_vectors = min(n_vectors, self.shape[1])
        q_j = r / np.linalg.norm(r)
        T = np.zeros((n_vectors + 1, n_vectors + 1))
        Q = None
        beta = None
        q_j_1 = None

        for j in range(n_vectors):
            if return_Q or full_orth:
                if Q is None:
                    Q = np.expand_dims(q_j, 1)
                else:
                    Q = np.append(Q, np.expand_dims(q_j, 1), 1)

            r_j = self.dot(q_j)

            # Substract projection into previous vector
            alpha = np.dot(q_j, r_j)
            r_j -= alpha * q_j
            T[j, j] = alpha

            # Substract projection into previous vector
            if q_j_1 is not None:
                r_j -= beta * q_j_1
                T[j, j - 1], T[j - 1, j] = beta, beta

            # Substract projection all other q's
            if full_orth:
                r_j -= Q[:, :-1].dot(Q[:, :-1].T.dot(r_j))

            r_norm = np.linalg.norm(r_j)
            if np.allclose(r_norm, 0, atol=np.finfo(q_j.dtype).eps):
                return (Q, T[: j + 1, :][:, : j + 1])

            q_j_1 = q_j
            q_j = r_j / r_norm
            beta = r_norm

        return (Q, T[:, :-1][:-1, :])

    def get_slq_config(self, lambda_min, lambda_max, epsilon=0.01, eta=0.05):
        kappa = lambda_max / lambda_min
        k1 = lambda_max * np.sqrt(kappa) * np.log(lambda_min + lambda_max)
        degree = np.int(np.sqrt(kappa) / 4 * np.log(k1 / epsilon))
        n_vectors = int(
            24 / epsilon**2 * np.log(1 + kappa) ** 2 * np.log(2 / eta)
        )
        return (n_vectors, degree)

    def calc_log_det(
        self,
        method="SLQ",
        n_vectors=10,
        degree=None,
        epsilon=0.01,
        eta=0.05,
        lambda_min=None,
        lambda_max=None,
    ):
        if method == "naive":
            sign, log_det = np.linalg.slogdet(self.todense())
            msg = "Negative determinant found. Ensure LinearOperator"
            msg += " has positive eigenvalues"
            check_error(sign > 0, msg=msg)
            return log_det

        elif method == "SLQ":
            log_det = 0
            if lambda_min is not None and lambda_max is not None:
                n_vectors, degree = self.get_slq_config(
                    lambda_min, lambda_max, epsilon=epsilon, eta=eta
                )

            for _ in range(n_vectors):
                u = 0.5 - (np.random.uniform(size=self.shape[0]) > 0.5).astype(
                    float
                )
                T = self.lanczos(u, degree, return_Q=False)[1]
                Theta, Y = eigh_tridiagonal(np.diag(T), np.diag(T, 1))
                tau = Y[0, :]
                log_det += np.sum(
                    [
                        np.log(theta_k) * tau_k**2
                        for tau_k, theta_k in zip(tau, Theta)
                    ]
                )
            log_det = log_det * self.shape[0] / n_vectors
            return log_det

        else:
            msg = "Method not properly working or implemented yet"
            raise ValueError(msg)
            upper = self.calc_eigenvalue_upper_bound()
            alpha = 1 / upper
            if degree is None:
                degree = np.log(self.shape[0])

            mat_log = TruncatedMatrixLog(self, degree, alpha)
            if method == "barry_pace99":
                v_i = mat_log.calc_trace_hutchinson(n_vectors)
                log_det = self.shape[0] * v_i.mean()
                #                 err = self.shape[0] * alpha ** (degree - 1) / (degree + 1) /(1 - alpha)
                #                 err += 1.96 * np.std(v_i) / np.sqrt(n_vectors)
                #                 bounds = (log_det - err, log_det + err)
                return log_det

            elif method == "taylor":
                return mat_log.calc_trace(exact=True)

            else:
                msg = "Unknown method for log_det estimation: {}".format(method)
                raise ValueError(msg)


# def lanczos_conjugate_gradient(A, b, tol=1e-6, max_iter=100):
#     x = np.zeros(b.shape)
#     p = b
#     r = b
#     w = 0
#     gamma = 1
#     prev_u = 0
#     u = np.zeros(b.shape)
#     r = A.dot(u) - b
#     r_norm = np.linalg.norm(r)
#     d = r

#     for _ in range(max_iter):
#         v = A.dot(r)
#         alpha = r_norm / np.dot(d, v)
#         u = u + alpha * r
#         r = r - alpha * v
#         r_norm_prev = r_norm
#         r_norm = np.linalg.norm(r)

#         if r_norm < tol:
#             break

#         beta = r_norm / r_norm_prev
#         d = r - beta * d

#     return u
