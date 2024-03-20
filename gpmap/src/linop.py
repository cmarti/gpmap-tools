#!/usr/bin/env python
import numpy as np

from itertools import combinations
from numpy.linalg.linalg import norm, matrix_power
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import eigh_tridiagonal, orth
from scipy.sparse import csr_matrix
from scipy.special import comb, factorial
from scipy.sparse.linalg import minres, cg
try:
    from scipy.sparse.linalg.interface import _CustomLinearOperator
except ImportError:
    from scipy.sparse.linalg._interface import _CustomLinearOperator

from gpmap.src.utils import check_error
from gpmap.src.matrix import (calc_cartesian_product,
                              calc_cartesian_product_dot,
                              inner_product, kron_dot, diag_pre_multiply,
                              reciprocal)


class ExtendedLinearOperator(_CustomLinearOperator):
    def _init_dtype(self):
        v = np.random.normal(size=3)
        self.dtype = v.dtype
    
    def rowsum(self):
        v = np.ones(self.shape[0])
        return(self.dot(v))
    
    def todense(self):
        return(self.dot(np.eye(self.shape[1])))
    
    def quad(self, v):
        return(np.sum(v * self.dot(v)))
    
    def rayleigh_quotient(self, v, metric=None):
        return(self.quad(v) / inner_product(v, v, metric=metric))
    
    def inv_dot(self, v, **kwargs):
        res = minres(self, v, **kwargs)
        self.res = res[1]
        return(res[0])

    def inv_quad(self, v, **kwargs):
        u = self.inv_dot(v, **kwargs)
        return(np.sum(v * u))
    
    def get_column(self, i):
        vec = np.zeros(self.shape[1])
        vec[i] = 1
        return(self.dot(vec))
    
    def get_diag(self):
        return(np.array([self.get_column(i)[i] for i in range(self.shape[0])]))
    
    def calc_trace_hutchinson(self, n_vectors):
        '''
        Stochastic trace estimator from 
        
        Hutchinson, M. F. (1990). A stochastic estimator of the trace of
        the influence matrix for laplacian smoothing splines. Communications
        in Statistics - Simulation and Computation, 19(2), 433–450.
        '''
        trace = np.array([self.quad(np.random.normal(size=self.shape[1]))
                          for _ in range(n_vectors)])
        return(trace)
    
    def calc_trace(self, exact=True, n_vectors=10):
        if exact or n_vectors > self.shape[1]:
            if hasattr(self, '_calc_trace'):
                trace = self._calc_trace()
            else:
                trace = self.get_diag().sum()
        else:
            trace = self.calc_trace_hutchinson(n_vectors).mean()
            
        return(trace)
    
    def calc_eigenvalue_upper_bound(self):
        return(self.rowsum().max())
    
    def arnoldi(self, r, n_vectors):
        '''
        Arnoldi algorithm based on 
        https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf
        '''
        n_vectors = min(n_vectors, self.shape[1])
        Q = np.expand_dims(r / norm(r), 1)
        H = np.zeros((n_vectors+1, n_vectors)) 
        
        for j in range(n_vectors):
            q_j = Q[:, -1]
            r = self.dot(q_j)
            for i in range(j+1):
                q_i = Q[:, i]
                p = np.dot(q_i, r)
                r -= p * q_i
                H[i, j] = p
            r_norm = norm(r)
            q = r / r_norm 
            H[j+1, j] = r_norm
            Q = np.append(Q, np.expand_dims(q, 1), 1)
            
            if np.allclose(r_norm, 0, atol=np.finfo(q.dtype).eps):
                return(Q, H[:j, :][:, :j])
            
        return(Q[:, :-1], H[:-1, :])
    
    def lanczos(self, r, n_vectors, full_orth=False, return_Q=True):
        '''
        Lanczos tridiagonalization algorithm based on 
        https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf
        '''
        n_vectors = min(n_vectors, self.shape[1])
        q_j = r / norm(r)
        T = np.zeros((n_vectors+1, n_vectors+1))
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
                T[j, j-1], T[j-1, j] = beta, beta
            
            # Substract projection all other q's
            if full_orth:
                r_j -= Q[:, :-1].dot(Q[:, :-1].T.dot(r_j))
            
            r_norm = norm(r_j)
            if np.allclose(r_norm, 0, atol=np.finfo(q_j.dtype).eps):
                return(Q, T[:j+1, :][:, :j+1])
            
            q_j_1 = q_j
            q_j = r_j / r_norm
            beta = r_norm
            
        return(Q, T[:, :-1][:-1, :])
    
    def get_slq_config(self, lambda_min, lambda_max, epsilon=0.01, eta=0.05):
        kappa = lambda_max / lambda_min
        k1 = lambda_max * np.sqrt(kappa) * np.log(lambda_min + lambda_max)
        degree = np.int(np.sqrt(kappa) / 4 * np.log(k1 / epsilon))
        n_vectors = int(24 / epsilon ** 2 * np.log(1 + kappa) ** 2 * np.log(2 / eta))
        return(n_vectors, degree)
    
    def calc_log_det(self, method='SLQ', n_vectors=10, degree=None,
                     epsilon=0.01, eta=0.05, lambda_min=None, lambda_max=None):
        if method == 'naive':
            sign, log_det = np.linalg.slogdet(self.todense())
            msg = 'Negative determinant found. Ensure LinearOperator has positive eigenvalues'
            check_error(sign > 0, msg=msg)
            return(log_det)
        
        elif method == 'SLQ':
            log_det = 0
            if lambda_min is not None and lambda_max is not None:
                n_vectors, degree = self.get_slq_config(lambda_min, lambda_max,
                                                        epsilon=epsilon, eta=eta)
            
            for _ in range(n_vectors):
                u = 0.5 - (np.random.uniform(size=self.shape[0]) > 0.5).astype(float)
                T = self.lanczos(u, degree, return_Q=False)[1]
                Theta, Y = eigh_tridiagonal(np.diag(T), np.diag(T, 1))
                tau = Y[0, :]
                log_det += np.sum([np.log(theta_k) * tau_k ** 2
                                   for tau_k, theta_k in zip(tau, Theta)])
            log_det = log_det * self.shape[0] / n_vectors
            return(log_det)
        
        else:
            msg = 'Method not properly working or implemented yet'
            raise ValueError(msg)
            upper = self.calc_eigenvalue_upper_bound()
            alpha = 1 / upper
            if degree is None:
                degree = np.log(self.shape[0])
            
            mat_log = TruncatedMatrixLog(self, degree, alpha)
            if method == 'barry_pace99':
                v_i = mat_log.calc_trace_hutchinson(n_vectors)
                log_det = self.shape[0] * v_i.mean()
#                 err = self.shape[0] * alpha ** (degree - 1) / (degree + 1) /(1 - alpha)
#                 err += 1.96 * np.std(v_i) / np.sqrt(n_vectors)
#                 bounds = (log_det - err, log_det + err)
                return(log_det)
            
            elif method == 'taylor':
                return(mat_log.calc_trace(exact=True))
            
            else:
                msg = 'Unknown method for log_det estimation: {}'.format(method)
                raise ValueError(msg)

class KronOperator(ExtendedLinearOperator):
    def __init__(self, matrices):
        self.matrices = matrices
        self.v_shape = [m_i.shape[1] for m_i in self.matrices]
        self.shape = (np.prod([m_i.shape[0] for m_i in self.matrices]),
                      np.prod(self.v_shape))

    def _matvec(self, v):
        check_error(v.shape[0] == self.shape[1],
                    msg='Incorrect dimensions of matrices and `v`')
        u_tensor = v.reshape(self.v_shape)
        for i, m in enumerate(self.matrices):
            u_tensor = np.tensordot(m, u_tensor, axes=([1], [i]))
        u = u_tensor.transpose().flatten()
        return(u)
    
    def _todense(self, matrices):
        if len(matrices) == 1:
            return(matrices[0])
        matrices = matrices[:-2] + [np.kron(matrices[-2], matrices[-1])]
        return(self._todense(matrices))

    def todense(self):
        return(self._todense(self.matrices))
    
    def transpose(self):
        return(KronOperator([m.T for m in self.matrices]))
    

class PolynomialOperator(ExtendedLinearOperator):
    def __init__(self, linop, coeffs):
        self.linop = linop
        self.shape = linop.shape
        self.set_coeffs(coeffs)
        self._init_dtype()

    def set_coeffs(self, coeffs):
        self.coeffs = np.array(coeffs)
        self.degree = self.coeffs.shape[0]
        
    def _matvec(self, v):
        power = v
        u = self.coeffs[0] * v
        for c in self.coeffs[1:]:
            power = self.linop.dot(power)
            u += c * power
        return(u)

    def calc_trace_hutchinson(self, n_vectors):
        trace = []
        for _ in range(n_vectors):
            v = 0.5 - (np.random.uniform(size=self.shape[1]) > 0.5).astype(float)
            power = v
            trace_i = 0
            for c in self.coeffs[1:]:
                if c == 0:
                    continue
                power = self.linop.dot(power)
                trace_i += c * np.sum(v * power)
            trace.append(trace_i)
        return(np.array(trace))
    

class TruncatedMatrixExp(PolynomialOperator):
    def __init__(self, linop, m):
        coeffs = np.array([1/factorial(i) for i in range(m+1)])
        super().__init__(linop, coeffs)


class TruncatedMatrixLog(PolynomialOperator):
    def __init__(self, linop, m, alpha=1):
        self.shape = linop.shape
        j = np.arange(1, m)
        coeffs = -alpha ** j / j
        coeffs = np.append([0], coeffs)
        A = ExtendedLinearOperator(shape=self.shape, dtype=float, 
                                   matvec=lambda v: v - alpha * linop.dot(v))
        
        super().__init__(A, coeffs)
        self.alpha = alpha


class SeqOperator(ExtendedLinearOperator):
    def __init__(self, n_alleles, seq_length):
        self.alpha = n_alleles
        self.l = seq_length
        self.lp1 = seq_length + 1
        self.n = self.alpha ** self.l
        self.shape = (self.n, self.n)
        self.shape_contracted = tuple([self.alpha]*self.l)
        self.positions = np.arange(self.l)
    
    def contract_v(self, v):
        return(v.reshape(self.shape_contracted))

    def expand_v(self, v):
        return(v.reshape(self.n))


class ConstantDiagSeqOperator(SeqOperator):
    def get_diag(self):
        return(np.full(self.n, self.d))
        
    def _calc_trace(self):
        return(self.n * self.d)
    

class LaplacianOperator(ConstantDiagSeqOperator):
    def __init__(self, n_alleles, seq_length):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.d = (self.alpha - 1) * self.l
        self.lambdas = np.arange(self.l + 1) * self.alpha
        self.lambdas_multiplicity = [comb(self.l, k) * (self.alpha-1) ** k
                                     for k in range(self.lp1)]
    
    def _matvec(self, v):
        v = self.contract_v(v)
        
        u = np.zeros(v.shape)
        for i in range(self.l):
            u += np.expand_dims(v.sum(i), axis=i)
        
        u = self.l * self.alpha * v - u
        return(self.expand_v(u))
    

class DeltaPOperator(ConstantDiagSeqOperator):
    def __init__(self, n_alleles, seq_length, P):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.L = LaplacianOperator(n_alleles=n_alleles, seq_length=seq_length)
        self.m_k = self.L.lambdas_multiplicity
        self.set_P(P)
        self.calc_kernel_dimension()
        self.calc_n_p_faces()

    def calc_kernel_dimension(self):
        self.kernel_dimension = np.sum(self.m_k[:self.P])
    
    def calc_n_p_faces(self):
        n_p_sites = comb(self.l, self.P)
        n_p_faces_per_sites = comb(self.alpha, 2) ** self.P
        allelic_comb_remaining_sites = self.alpha ** (self.l - self.P)
        self.n_p_faces = n_p_sites * n_p_faces_per_sites * allelic_comb_remaining_sites
        
    def set_P(self, P):
        self.P = P
        if self.P == (self.lp1):
            msg = '"P" = l+1, the optimal density is equal to the empirical frequency.'
            raise ValueError(msg)
        elif not 1 <= self.P <= self.l:
            msg = '"P" not in the right range.'
            raise ValueError(msg)
        self.Pfactorial = factorial(self.P)
        self.d = comb(self.l, self.P) * (self.alpha - 1) ** self.P
    
    def _L_minus_p_a_dot(self, v, p=0):
        return(self.L.dot(v) - p * self.alpha * v)
    
    def _matvec(self, v):
        dotv = v.copy()
        for p in range(self.P):
            dotv = self._L_minus_p_a_dot(dotv, p)
        return(dotv / self.Pfactorial)

    def calc_lambdas(self):
        lambdas = []
        for L_lambda_k in self.L.lambdas:
            lambda_k = 1
            for p in range(self.P):
                lambda_k *= L_lambda_k - p * self.alpha
            lambdas.append(lambda_k / self.Pfactorial)
        self.lambdas = np.array(lambdas)
    
    def calc_log_det(self):
        return(self.m_k[self.P:] * np.log(self.lambdas[self.P:]))
    

class KrawtchoukOperator(SeqOperator, PolynomialOperator):
    def __init__(self, n_alleles, seq_length, **params):
        SeqOperator.__init__(self, n_alleles=n_alleles, seq_length=seq_length)
        L = LaplacianOperator(n_alleles=n_alleles, seq_length=seq_length)
        self.L_lambdas = L.lambdas
        self.m_k = L.lambdas_multiplicity
        self.calc_eig_vandermonde_matrix_inverse(numeric=False)
        self.calc_polynomial_coefficients(**params)
        PolynomialOperator.__init__(self, L, self.coeffs)

    def calc_eig_vandermonde_matrix(self):
        self.V = np.vstack([self.L_lambdas ** i for i in range(self.lp1)]).T
        self.V_LU = lu_factor(self.V)
        return(self.V)

    def calc_eig_vandermonde_matrix_inverse(self, numeric=False):
        '''
        Calculates the coefficients of the polynomial in L that represent 
        projection matrices into each of the kth eigenspaces.
        
        Returns
        -------
        B : array-like of shape (seq_length + 1, seq_length + 1)
            Matrix containing the b_i,k coefficients for power i on rows
            and order k on columns. One can obtain the coefficients for any
            combination of $\lambda_k$ values by scaling the coefficients 
            for each eigenspace by its eigenvalue and adding them up across
            different powers
        '''
        if numeric:
            self.calc_eig_vandermonde_matrix()
            self.V_inv = np.linalg.inv(self.V)

        else:        
            V_inv = np.zeros((self.lp1, self.lp1))
            idx = np.arange(self.lp1)
            
            for k in idx:
                k_idx = idx != k
                k_lambdas = self.L_lambdas[k_idx]
                norm_factor = 1 / np.prod(k_lambdas - self.L_lambdas[k])
    
                for power in idx:
                    p = np.sum([np.product(v) for v in combinations(k_lambdas, self.l - power)])
                    V_inv[power, k] = norm_factor * (-1) ** (power) * p
            
            self.V_inv = V_inv
            
        return(self.V_inv)

    def calc_w(self, k, d):
        """return value of the Krawtchouk polynomial for k, d"""
        l, a = self.l, self.alpha
        s = 0
        for q in range(k + 1):
            s += (-1)**q * (a - 1)**(k - q) * comb(d, q) * comb(l - d, k - q)
        return(1 / a**l * s)
    
    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.lp1, self.lp1])
        for k in range(self.lp1):
            for d in range(self.lp1):
                self.W_kd[k, d] = self.calc_w(k, d)
    

class ProjectionOperator(ConstantDiagSeqOperator, KrawtchoukOperator):
    def __init__(self, n_alleles, seq_length, k=None, lambdas=None):
        KrawtchoukOperator.__init__(self, n_alleles=n_alleles, seq_length=seq_length,
                                    k=k, lambdas=lambdas)
        self._init_dtype()
    
    def calc_polynomial_coefficients(self, k=None, lambdas=None):
        self.lambdas = self.get_lambdas(lambdas=lambdas, k=k)
        self.coeffs = self.lambdas_to_coeffs(self.lambdas)

    def get_lambdas(self, lambdas=None, k=None):
        msg = 'Only one "k" or "lambdas" can and must be provided'
        check_error((lambdas is None) ^ (k is None), msg=msg)
        
        if lambdas is None:
            lambdas = np.zeros(self.l + 1)
            lambdas[k] = 1
            
        return(lambdas)

    def lambdas_to_coeffs(self, lambdas, use_lu=False):
        if use_lu:
            coeffs = lu_solve(self.V_LU, lambdas)
        else:
            coeffs = self.V_inv.dot(lambdas)
        return(coeffs)
    
    @property
    def d(self):
        if not hasattr(self, '_d'):
            self._d = self.calc_covariance_distance()[0]
        return(self._d)

    def calc_covariance_distance(self):
        self.calc_W_kd_matrix()
        return(self.W_kd.T.dot(self.lambdas))
    
    def inv(self):
        return(ProjectionOperator(self.alpha, self.l, lambdas=1./self.lambdas))
    
    def calc_log_det(self):
        if np.any(self.lambdas == 0.):
            return(-np.inf)
        return(np.sum(np.log(self.lambdas) * self.m_k))
    
    def matrix_sqrt(self):
        return(ProjectionOperator(self.alpha, self.l, lambdas=np.sqrt(self.lambdas)))


class ExtendedDeltaPOperator(ProjectionOperator):
    def __init__(self, n_alleles, seq_length, P, lambdas0, **params):
        msg = 'Ensure that lambdas0 has size P'
        check_error(lambdas0.shape[0] == P, msg=msg)
        DP = DeltaPOperator(n_alleles, seq_length, P)
        DP.calc_lambdas()
        lambdas = DP.lambdas
        lambdas[:P] = 1 / lambdas0
        super().__init__(n_alleles=n_alleles, seq_length=seq_length,
                         lambdas=lambdas, **params)

class CovarianceDistanceOperator(SeqOperator, PolynomialOperator):
    def __init__(self, n_alleles, seq_length, distance):
        SeqOperator.__init__(self, n_alleles=n_alleles, seq_length=seq_length)
        L = LaplacianOperator(n_alleles=n_alleles, seq_length=seq_length)
        self.L_lambdas = L.lambdas
        self.m_k = L.lambdas_multiplicity
        self.calc_polynomial_coefficients(distance=distance)
        PolynomialOperator.__init__(self, L, self.coeffs)
    
    def calc_L_powers_distance_matrix_inverse(self):
        """Construct entries of powers of L. 
        Column: powers of L. 
        Row: Hamming distance"""
        
        l, a, s = self.l, self.alpha, self.lp1
    
        # Auxiliary matrices
        C = np.zeros([s, s])
        for i in range(s):
            for j in range(s):
                if i == j:
                    C[i, j] = i * (a - 2)
                if i == j + 1:
                    C[i, j] = i
                if i == j - 1:
                    C[i, j] = (l - j + 1) * (a - 1)
        D = np.array(np.diag(l * (a - 1) * np.ones(s), 0))
        B = D - C
        u = np.zeros(s)
        u[0], u[1] = l * (a - 1), -1
    
        # Construct L_powers_d column by column
        L_powers_d = np.zeros([s, s])
        L_powers_d[0, 0] = 1
        for j in range(1, s):
            L_powers_d[:, j] = matrix_power(B, j-1).dot(u)
    
        self.L_powers_d_inv = np.linalg.inv(L_powers_d)
    
    def calc_polynomial_coefficients(self, distance):
        self.calc_L_powers_distance_matrix_inverse()
        self.coeffs = self.L_powers_d_inv[:, distance]


class CovarianceVjOperator(ConstantDiagSeqOperator, KronOperator):
    def __init__(self, n_alleles, seq_length, j):
        self.j = j

        ConstantDiagSeqOperator.__init__(self, n_alleles=n_alleles, seq_length=seq_length)
        KronOperator.__init__(self, self.get_matrices())
    
    def get_matrices(self):
        C0 = np.eye(self.alpha)
        C1 = np.ones((self.alpha, self.alpha)) - C0
        return([C1 if i in self.j else C0 for i in range(self.l)])


class VjOperator(ConstantDiagSeqOperator, KronOperator):
    def __init__(self, n_alleles, seq_length, j):
        self.j = j
        self.k = len(j) 

        ConstantDiagSeqOperator.__init__(self, n_alleles=n_alleles, seq_length=seq_length)
        self.repeats = self.alpha ** (self.l - self.k)
        
        KronOperator.__init__(self, self.get_matrices(j))
        

class VjBasisOperator(VjOperator):
    def get_matrices(self, j):
        site_L = self.alpha * np.eye(self.alpha) - np.ones((self.alpha, self.alpha))
        b = [np.full((self.alpha, 1), 1 / np.sqrt(self.alpha)), orth(site_L)]
        return([b[int(i in j)] for i in range(self.l)])
    
class VjProjectionOperator(VjOperator):
    def get_matrices(self, j):
        self.W0 = np.full((self.alpha, self.alpha), fill_value=1./self.alpha)
        self.W1 = np.eye(self.alpha) - self.W0
        W = [self.W0, self.W1]
        return([W[int(i in j)] for i in range(self.l)])
    
    def dot_square_norm(self, v):
        axis = tuple([p for p in range(self.l)
                      if p not in self.j])
        u = self.contract_v(v)
        if axis:
            u = u.mean(axis=axis)

        if self.k == 0:
            sqnorm = self.repeats * u ** 2
        else:
            matrices = [self.W1] * self.k
            sqnorm = self.repeats * np.sum(kron_dot(matrices, u.flatten()) ** 2)
        return(sqnorm)
    
class RhoProjectionOperator(ConstantDiagSeqOperator, KronOperator):
    def __init__(self, n_alleles, seq_length, rho):
        ConstantDiagSeqOperator.__init__(self, n_alleles=n_alleles, seq_length=seq_length)
        
        self.set_rho(rho)
        KronOperator.__init__(self, self.get_matrices())

    def get_matrices(self):
        W0 = np.full((self.alpha, self.alpha), fill_value=1./self.alpha)
        W1 = np.eye(self.alpha) - W0
        return([W0 + r * W1 for r in self.rho])

    def check_rho(self, rho, ignore_bound=False):
        msg = 'rho vector size must be equal to sequence length'
        check_error(rho.shape[0] == self.l, msg=msg)

        checked = rho > 0
        msg = 'rho larger than 0'
        if not ignore_bound:
            checked = checked & (rho < 1)
            msg = 'rho must be between 0 and 1' 
        check_error(np.all(checked), msg=msg)
    
    def set_rho(self, rho, ignore_bound=False):
        self.rho  = np.full(self.l, rho) if isinstance(rho, float) else np.array(rho)
        self.check_rho(self.rho, ignore_bound=ignore_bound)
        self.d = np.prod([1 + (self.alpha - 1) * r for r in self.rho]) / self.n
    
    def inv(self):
        return(RhoProjectionOperator(self.alpha, self.l, 1./self.rho))

    def calc_log_det(self):
        log_rho = np.log(self.rho)
        k = np.sum([comb(self.l, i-1) for i in range(self.l)])
        return(k * np.sum(log_rho))


class EigenBasisOperator(SeqOperator):
    def __init__(self, n_alleles, seq_length, k):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.k = k
        self.m = (self.alpha - 1) ** self.k
        self.shape = (self.n, int(comb(self.l, self.k) * self.m))

    def _matvec(self, v):
        positions = np.arange(self.l)
        u = 0
        for i, j in enumerate(combinations(positions, self.k)):
            B = VjBasisOperator(self.alpha, self.l, j)
            v_i = v[self.m*i:self.m*(i+1)]
            u += B @ v_i
        return(u)
    
    def transpose_dot(self, v):
        positions = np.arange(self.l)
        u = np.zeros(self.shape[1])
        for i, j in enumerate(combinations(positions, self.k)):
            B_transpose = VjBasisOperator(self.alpha, self.l, j).transpose()
            u[self.m*i:self.m*(i+1)] = B_transpose.dot(v)
        return(u)


class DeltaKernelBasisOperator(SeqOperator):
    def __init__(self, n_alleles, seq_length, P):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.P = P
        m = int(np.sum([comb(self.l, k) * (self.alpha - 1) ** k
                        for k in range(P)]))
        self.shape = (self.n, m)

    def _matvec(self, v):
        u = 0.
        start = 0
        for k in range(self.P):
            B = EigenBasisOperator(self.alpha, self.l, k=k)
            end = start + B.shape[1]
            u += B @ v[start:end]
            start = end
        return(u)
    
    def transpose_dot(self, v):
        u = np.zeros(self.shape[1])
        start = 0
        for k in range(self.P):
            B = EigenBasisOperator(self.alpha, self.l, k=k)
            end = start + B.shape[1]
            u[start:end] = B.transpose_dot(v)
            start = end
        return(u)
    
class ProjectionOperator2(SeqOperator):
    def __init__(self, n_alleles, seq_length):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.calc_b()
        
    def set_lambdas(self, lambdas=None, k=None):
        msg = 'Only one "k" or "lambdas" can and must be provided'
        check_error((lambdas is None) ^ (k is None), msg=msg)
        
        if lambdas is None:
            lambdas = np.zeros(self.l + 1)
            lambdas[k] = 1
            
        self.lambdas = lambdas
        self.coeffs = self.lambdas_to_coeffs(lambdas)
        
    def calc_b(self):
        b = np.zeros((self.lp1, self.lp1))
        for i in range(self.lp1):
            for k, j in enumerate(range(i, self.lp1)):
                b[i, j] = (-1)**k * comb(self.l-i, k)
        self.b = b
    
    def lambdas_to_coeffs(self, lambdas):
        return(self.b.dot(lambdas))
    
    def _matvec(self, v):
        u = self.contract_v(v)
        r = np.zeros(self.shape_contracted)

        for k in range(self.lp1):
            c_k = self.coeffs[k]
            
            if c_k == 0:
                continue
            
            for j in combinations(self.positions, k):
                axis = tuple([p for p in self.positions if p not in j])
                r += c_k * np.expand_dims(u.mean(axis=axis), axis=axis)
                
        return(self.expand_v(r))


class KernelOperator(ExtendedLinearOperator):
    def __init__(self, linop, x1=None, x2=None, y_var_diag=None):
        self.linop = linop
        self.x1 = x1
        self.x2 = x2
        self.shape = [linop.shape[0], linop.shape[1]]
        self.m1, self.m2 = None, None

        if x1 is not None:
            self.m1 = self.calc_x_matrix(self.linop.shape[0], x1).T
            self.shape[0] = self.m1.shape[0]

        if x2 is not None:
            self.m2 = self.calc_x_matrix(self.linop.shape[0], x2)
            self.shape[1] = self.m2.shape[1]
        
        if y_var_diag is not None:
            check_error(self.shape[1] == y_var_diag.shape[0], 
                        'Ensure that `y_var_diag` has the same dimension as `x2`')
        self.y_var_diag = y_var_diag
        self._init_dtype()
    
    def calc_x_matrix(self, m, x):
        n = x.shape[0]
        return(csr_matrix((np.ones(n), (x, np.arange(n))),
                          shape=(self.linop.shape[1], n)))
    
    def _matvec(self, v):
        u = self.m2 @ v if self.m2 is not None else v
        u = self.linop @ u
        if self.m1 is not None:
            u = self.m1 @ u
        if self.y_var_diag is not None:
            u += diag_pre_multiply(self.y_var_diag, v)
        return(u)
    
    def inv_dot(self, v, atol=1e-16):
        res = cg(self, v, atol=atol)
        self.res = res[1]
        return(res[0])

class BaseKernelOperator(SeqOperator):
    def set_y_var(self, y_var=None, obs_idx=None):
        
        if y_var is not None and obs_idx is not None:
            msg = 'y_var and obs_idx should have same dimension: {} vs {}'
            msg = msg.format(y_var.shape[0], obs_idx.shape[0])
            check_error(y_var.shape[0] == obs_idx.shape[0], msg=msg)
            self.known_var = True
            self.homoscedastic = np.unique(y_var).shape[0] == 1
            self.mean_var = y_var.mean()
            self.y_var = y_var
            self.n_obs = obs_idx.shape[0]
            self.calc_gt_to_data_matrix(obs_idx)
            self.shape = (self.n_obs, self.n_obs)

        else:
            msg = 'y_var and obs_idx must be provided with each other'
            check_error(y_var is None and obs_idx is None, msg=msg)
    
    def set_mode(self, all_rows=False, add_y_var_diag=True, full_v=False):
        self.all_rows = all_rows
        self.add_y_var_diag = add_y_var_diag
        self.full_v = full_v
        
        nrows, ncols = (self.n, self.n)
        if not self.all_rows and self.known_var:
            nrows = self.n_obs
        if not full_v and self.known_var:
            ncols = self.n_obs
            
        self.shape = (nrows, ncols)
    
    def _matvec(self, v):
        if self.full_v or not self.known_var:
            u = self._dot(v)
        else:
            u = self._dot(self.gt2data.dot(v))
            if self.add_y_var_diag:
                u += self.gt2data.dot(diag_pre_multiply(self.y_var, v))

        if not self.all_rows and self.known_var:
            u = self.gt2data.T.dot(u)
        return(u)

    def calc_gt_to_data_matrix(self, obs_idx):
        n_obs = obs_idx.shape[0]
        self.gt2data = csr_matrix((np.ones(n_obs), (obs_idx, np.arange(n_obs))),
                                  shape=(self.n, n_obs))   
    
    def inv_quad(self, v, show=False):
        u = self.inv_dot(v, show=show)
        return(np.sum(u * v))
    
    
class VarianceComponentKernelOperator(BaseKernelOperator):
    def __init__(self, n_alleles, seq_length, lambdas=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.W = ProjectionOperator(n_alleles=n_alleles, seq_length=seq_length)
        self.m_k = self.W.L.lambdas_multiplicity
        self.n = self.W.n
        self.shape = (self.n, self.n)
        self.known_var = False
        self.set_mode()
        
        if lambdas is not None:
            self.set_lambdas(lambdas)
    
    def set_params(self, params):
        self.set_lambdas(lambdas=np.exp(params))
    
    def get_params(self):
        return(np.log(self.get_lambdas()))
    
    def set_lambdas(self, lambdas):
        self.W.set_lambdas(lambdas)
    
    def get_lambdas(self):
        return(self.W.lambdas)
    
    def _dot(self, v):
        return(self.W.dot(v))
    
    def one_half_power_dot(self, v):
        lambdas = self.get_lambdas()
        self.set_lambdas(np.sqrt(lambdas))
        u = self.W.dot(v)
        self.set_lambdas(lambdas)
        return(u)
    
    def _get_diag(self):
        msg = 'lambdas need to be set to get diagonal'
        check_error(hasattr(self, 'lambdas'), msg=msg)
        if hasattr(self, 'n_obs'):
            return(self.W.d + self.y_var)
        else:
            return(np.full(self.W.d, self.n))
    
    def _calc_trace(self):
        msg = 'lambdas need to be set to calculate trace'
        check_error(hasattr(self.W, 'lambdas'), msg=msg)
        if hasattr(self, 'n_obs'):
            return(self.n_obs * self.W.d + np.sum(self.y_var))
        else:
            return(self.n * self.W.d)
    
    def get_params0(self):
        return(np.random.normal(size=self.lp1))
        return(-np.arange(self.lp1))
    
    def grad(self):
        for k, lambda_k in enumerate(self.get_lambdas()):
            K_grad = ProjectionOperator(n_alleles=self.alpha,
                                        seq_length=self.l)
            K_grad.set_lambdas(k=k)
            yield(K_grad * lambda_k)


class ConnectednessKernelOperator(BaseKernelOperator):
    def __init__(self, n_alleles, seq_length, rho=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.P = RhoProjectionOperator(n_alleles=n_alleles, seq_length=seq_length)
        self.n = self.P.n
        self.shape = (self.n, self.n)
        self.known_var = False
        self.set_mode()
        self.set_params = self.set_rho
        self.get_params = self.get_rho
        
        if rho is not None:
            self.set_rho(rho)
    
    def set_rho(self, rho):
        self.P.set_rho(rho)
    
    def get_rho(self):
        return(self.P.rho)
    
    def _dot(self, v):
        return(self.P.dot(v))
    
    def one_half_power_dot(self, v):
        rho = self.get_rho()
        self.set_rho(np.sqrt(rho))
        u = self.P.dot(v)
        self.set_rho(rho)
        return(u)
    
    def _get_diag(self):
        msg = 'rho need to be set to get diagonal'
        check_error(hasattr(self, 'rho'), msg=msg)
        if hasattr(self, 'n_obs'):
            return(self.P.d + self.y_var)
        else:
            return(np.full(self.P.d, self.n))
    
    def _calc_trace(self):
        msg = 'rho need to be set to calculate trace'
        check_error(hasattr(self.P, 'rho'), msg=msg)
        if hasattr(self, 'n_obs'):
            return(self.n_obs * self.P.d + np.sum(self.y_var))
        else:
            return(self.n * self.P.d)
    

#################### Skewed operators ##################################

# class SkewedLaplacianOperator(SeqOperator):
#     def __init__(self, n_alleles, seq_length, ps=None, max_size=None):
#         super().__init__(n_alleles=n_alleles, seq_length=seq_length)
#         self.set_ps(ps)
#         self.calc_Kns()
#         self.calc_lambdas(ps=ps)
#         self.calc_lambdas_multiplicity()
        
#         self.max_size = max_size
#         if max_size is None:
#             self.calc_L()
#             self._matvec = self.dot1
#         else:
#             self._matvec = self.dot2
    
#     def guess_n_products(self):
#         if self.max_size is None:
#             return(None)
        
#         size = 1
#         for k in range(self.l):
#             size *= self.alpha
#             if size >= self.max_size:
#                 break
#         return(k)
    
#     def set_ps(self, ps):
#         self.variable_ps = ps is not None
        
#         if ps is None:
#             ps = [np.ones(self.alpha)] * self.l
#         check_error(len(ps) == self.l, msg='Number of ps should be equal to length')

#         # Normalize ps to have the eigenvalues in the right scale
#         self.ps = np.vstack([p / p.sum() * self.alpha for p in ps])
#         self.pi = calc_tensor_product([p.reshape((p.shape[0], 1)) for p in ps]).flatten()
    
#     def calc_lambdas(self, ps=None):
#         self.lambdas = np.arange(self.l + 1) * self.alpha
    
#     def calc_lambdas_multiplicity(self):
#         self.lambdas_multiplicity = [comb(self.l, k) * (self.alpha-1) ** k
#                                      for k in range(self.lp1)]
    
#     def calc_Kn(self, p):
#         if self.variable_ps:
#             Kn = np.vstack([p] * p.shape[0])
#             np.fill_diagonal(Kn, np.zeros(Kn.shape[0]))
#         else:
#             Kn = np.full((self.alpha, self.alpha), True)
#             np.fill_diagonal(Kn, np.full(self.alpha, False))
#         return(Kn)
    
#     def calc_Kns(self):
#         Kns = [self.calc_Kn(p) for p in self.ps]
#         if self.max_size is not None:
#             size = self.guess_n_products()
#             i = self.l - size
#             Kns = Kns[:i] + [calc_cartesian_product([csr_matrix(m) for m in Kns[i:]]).tocsr()]
#         self.Kns = Kns
#         self.Kns_shape = [x.shape for x in Kns]
    
#     def calc_L(self):
#         L = -calc_cartesian_product(self.Kns).astype(np.float64)
#         L.setdiag(-L.sum(1).A1)
#         self.L = L.tocsr()
    
#     def dot1(self, v):
#         return(self.L.dot(v))
    
#     def dot2(self, v):
#         return(self.d * v - calc_cartesian_product_dot(self.Kns, v))


# class LapDepOperator(SeqOperator):
#     def __init__(self, n_alleles=None, seq_length=None, L=None):
#         if L is None:
#             msg = 'either L or both seq_length and n_alleles must be given'
#             check_error(n_alleles is not None and seq_length is not None, msg=msg)
#             L = LaplacianOperator(n_alleles, seq_length)
#         else:
#             msg = 'either L or both seq_length and n_alleles must be given'
#             check_error(L is not None, msg=msg)
#             n_alleles, seq_length = L.alpha, L.l
            
#         super().__init__(n_alleles=n_alleles, seq_length=seq_length)
#         self.L = L


# class SkewedVjProjectionOperator(LapDepOperator):
#     def __init__(self, n_alleles=None, seq_length=None, L=None, max_size=None):
#         super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L)
#         self.calc_elementary_W()
#         self.max_size = max_size
#         self.size = self.guess_n_products()
#         self.cache = {}
    
#     def calc_elementary_W(self):
#         if self.L.ps is None:
#             self.b = [np.ones(self.alpha)] * self.l
#         else:
#             self.b = self.L.ps
#         self.D_pi = [np.diag(b) for b in self.b]
#         self.pi = calc_tensor_product([b.reshape((b.shape[0], 1)) for b in self.b]).flatten()
#         self.W0 = [np.outer(b, b).dot(D) / np.sum(b * D.dot(b))
#                    for b, D in zip(self.b, self.D_pi)]
#         self.W1 = [np.eye(self.alpha) - w0 for w0 in self.W0]
#         self.Ws = [[w0, w1] for w0, w1 in zip(self.W0, self.W1)]
        
#     def get_Vj_matrix(self, js):
#         js_tuple = tuple(js)
#         if js_tuple not in self.cache:
#             m = self.l - self.size
#             ms = [self.Ws[m+p][j] for p, j in enumerate(js)]
#             self.cache[js_tuple] = calc_tensor_product(ms)
#         return(self.cache[js_tuple])
    
#     def set_j(self, positions):
#         js = np.zeros(self.l, dtype=int)
#         js[positions] = 1

#         if self.max_size is None:
#             self.matrices = [self.Ws[p][j] for p, j in enumerate(js)]
#         else:
#             m = self.l - self.size
#             self.matrices = [self.Ws[p+m][j] for p, j in enumerate(js[:m])]
#             self.matrices.append(self.get_Vj_matrix(js[m:]))
    
#     def _matvec(self, v):
#         return(calc_tensor_product_dot(self.matrices, v))
    
#     def dot_square_norm(self, v):
#         '''
#         Note: we are calculating the squared D_pi-norm of the projection to be 
#         able to do it directly through recursive product
#         '''
#         return(calc_tensor_product_quad(self.matrices, v1=self.pi * v, v2=v))


# class SkewedKernelOperator(SeqOperator):
#     def __init__(self, W):
#         self.W = W
#         self.sqrt_pi_inv = 1 / np.sqrt(W.L.pi)
#         self.pi_inv = 1 / W.L.pi
#         self.n = W.n
#         self.shape = (self.n, self.n)
#         self.known_var = False
    
#     def set_y_var(self, y_var=None, obs_idx=None):
        
#         if y_var is not None and obs_idx is not None:
#             msg = 'y_var and obs_idx should have same dimension: {} vs {}'
#             msg = msg.format(y_var.shape[0], obs_idx.shape[0])
#             check_error(y_var.shape[0] == obs_idx.shape[0], msg=msg)
#             self.known_var = True
#             self.homoscedastic = np.unique(y_var).shape[0] == 1
#             self.mean_var = y_var.mean()
#             self.y_var = y_var
#             self.n_obs = obs_idx.shape[0]
#             self.calc_gt_to_data_matrix(obs_idx)
#             self.shape = (self.n_obs, self.n_obs)

#         else:
#             msg = 'y_var and obs_idx must be provided with each other'
#             check_error(y_var is None and obs_idx is None, msg=msg)

#     @property
#     def lambdas_multiplicity(self):
#         return(self.W.L.lambdas_multiplicity)
        
#     def set_lambdas(self, lambdas):
#         self.W.set_lambdas(lambdas)
    
#     def get_lambdas(self):
#         return(self.W.lambdas)
    
#     def _dot(self, v):
#         if hasattr(self.W.L, 'variable_ps') and self.W.L.variable_ps:
#             return(self.W.dot(self.D_pi_inv.dot(v)))
#         else:
#             return(self.W.dot(v))

#     def calc_gt_to_data_matrix(self, obs_idx):
#         n_obs = obs_idx.shape[0]
#         self.gt2data = csr_matrix((np.ones(n_obs), (obs_idx, np.arange(n_obs))),
#                                   shape=(self.n, n_obs))   
    
#     def _matvec(self, v, all_rows=False, add_y_var_diag=True, full_v=False):
#         if full_v or not self.known_var:
#             u = self._dot(v)
#         else:
#             u = self._dot(self.gt2data.dot(v))
#             if add_y_var_diag:
#                 u += self.gt2data.dot(diag_pre_multiply(self.y_var, v))

#         if not all_rows and self.known_var:
#             u = self.gt2data.T.dot(u)
#         return(u)
    
#     @property
#     def Kop(self):
#         if not hasattr(self, '_Kop'):
#             self._Kop = LinearOperator((self.n_obs, self.n_obs), matvec=self.dot)
#         return(self._Kop)
    
#     def inv_dot(self, v, show=False):
#         res = minres(self.Kop, v, tol=1e-6, show=show)
#         self.res = res[1]
#         return(res[0])
    
#     def inv_quad(self, v, show=False):
#         u = self.inv_dot(v, show=show)
#         return(np.sum(u * v))


def _get_seq_values_and_obs_seqs(y, n_alleles, seq_length, idx=None):
    n = n_alleles ** seq_length
    if idx is not None:
        seq_values, observed_seqs = np.zeros(n), np.zeros(n)
        seq_values[idx], observed_seqs[idx] = y, 1.
    else:
        seq_values, observed_seqs = y, np.ones(n, dtype=float)
    return(seq_values, observed_seqs)


def calc_covariance_distance(y, n_alleles, seq_length, idx=None):
    seq_values, obs_seqs = _get_seq_values_and_obs_seqs(y, n_alleles,
                                                        seq_length, idx=None)

    cov, ns = np.zeros(seq_length + 1), np.zeros(seq_length + 1)
    for d in range(seq_length + 1):
        P = CovarianceDistanceOperator(n_alleles, seq_length, distance=d)
        quad = P.quad(seq_values)
        ns[d] = P.quad(obs_seqs)
        cov[d] = reciprocal(quad, ns[d])
    return(cov, ns)


def calc_covariance_vjs(y, n_alleles, seq_length, idx=None):
    lp1 = seq_length + 1
    seq_values, obs_seqs = _get_seq_values_and_obs_seqs(y, n_alleles,
                                                        seq_length, idx=idx)

    cov, ns = [], []
    sites = np.arange(seq_length)
    sites_matrix = []
    for k in range(lp1):
        for j in combinations(sites, k):
            P = CovarianceVjOperator(n_alleles, seq_length, j=j)
            quad = P.quad(seq_values)
            nj = P.quad(obs_seqs)
            z = np.array([i not in j for i in range(seq_length)], dtype=float)
            
            cov.append(reciprocal(quad, nj))
            ns.append(nj)
            sites_matrix.append(z)

    sites_matrix = np.array(sites_matrix)
    cov, ns = np.array(cov), np.array(ns)
    return(cov, ns, sites_matrix)


def calc_variance_components(space):
    '''
    Calculates the variance components associated to the function
    along the SequenceSpace. It returns the squared module of the
    projection into each of the l+1 eigenspaces of the graph Laplacian
    representing the variance associated to epistatic interations of order k
    
    See Zhou et al. 2021
    https://www.pnas.org/doi/suppl/10.1073/pnas.2204233119
    
    Returns
    -------
    lambdas: array-like of shape (seq_length + 1, )
        Vector containing the squared module of the projections into the
        k'th eigenspaces in increasing order of k.
     
    '''

    n_alleles = np.unique(space.n_alleles)
    msg = 'Variance components can only be calculated for spaces'
    msg += ' with constant number of alleles across sites'
    check_error(n_alleles.shape[0] == 1, msg)
    n_alleles = n_alleles[0]
    
    lambdas = []
    for k in np.arange(space.seq_length + 1):
        W = ProjectionOperator(n_alleles=n_alleles, seq_length=space.seq_length, k=k)
        lambdas.append(np.sum(W.quad(space.y)) / W.m_k[k])
    return(np.array(lambdas))


def calc_vjs_variance_components(space, k):
    '''
    Calculates the squared module of the projection into the `Vj` subspaces
    of order `k` defined by each individual combination of `k` sites as
    defined by `j` 
    
    Parameters
    ----------
    space : SequenceSpace
        SequenceSpace object for which to calculate the Vj's variance components
    
    k : int from 0 to seq_length + 1
        Order of interaction to calculate
        
    
    Returns
    -------
    lambdas: dict
        Dictionary with combinations of `k` sites as keys and the associated
        squared modules of the projection into the individual subspaces 
    '''
    
    n_alleles = np.unique(space.n_alleles)
    msg = 'Variance components can only be calculated for spaces'
    msg += ' with constant number of alleles across sites'
    check_error(n_alleles.shape[0] == 1, msg)
    n_alleles = n_alleles[0]
        
    positions = np.arange(space.seq_length)
    dimension = (n_alleles - 1) ** float(k)
    variances = {}
    for j in combinations(positions, k):
        Pj = VjProjectionOperator(n_alleles, space.seq_length, j=j)
        variances[j] = np.sum(Pj.dot(space.y) ** 2) / dimension 
    return(variances)
