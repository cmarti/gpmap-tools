#!/usr/bin/env python
import numpy as np

from math import factorial
from itertools import combinations

from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.csr import csr_matrix
from scipy.linalg.decomp_svd import orth
from scipy.special._basic import comb
from scipy.sparse.linalg.isolve import minres
from scipy.sparse.linalg.interface import LinearOperator

from gpmap.src.utils import (calc_cartesian_product, check_error,
                             calc_matrix_polynomial_dot, calc_tensor_product,
                             calc_cartesian_product_dot,
                             calc_tensor_product_dot, calc_tensor_product_quad, 
                             get_sparse_diag_matrix, inner_product, kron_dot)


class SeqLinOperator(object):
    def __init__(self, n_alleles, seq_length, max_size=None):
        self.alpha = n_alleles
        self.l = seq_length
        self.lp1 = seq_length + 1
        self.n = self.alpha ** self.l
        self.d = (self.alpha - 1) * self.l
        self.shape = (self.n, self.n)
        self.max_size = max_size
    
    def contract_v(self, v):
        return(v.reshape(tuple([self.alpha]*self.l)))

    def expand_v(self, v):
        return(v.reshape(self.n))
    
    def guess_n_products(self):
        if self.max_size is None:
            return(None)
        
        size = 1
        for k in range(self.l):
            size *= self.alpha
            if size >= self.max_size:
                break
        return(k)
    
    def todense(self):
        return(self.dot(np.eye(self.shape[0])))
    
    def quad(self, v):
        return(np.sum(v * self.dot(v)))
    
    def rayleigh_quotient(self, v, metric=None):
        return(self.quad(v) / inner_product(v, v, metric=metric))


class LaplacianOperator(SeqLinOperator):
    def __init__(self, n_alleles, seq_length):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
            
        self.calc_lambdas()
        self.calc_lambdas_multiplicity()
        self.calc_W_kd_matrix()
    
    def dot(self, v):
        v = self.contract_v(v)
        u = np.zeros(v.shape)
        for i in range(self.l):
            u += np.stack([v.sum(i)] * self.alpha, axis=i)
        u = self.l * self.alpha * v - u
        return(self.expand_v(u))
    
    def calc_lambdas(self):
        self.lambdas = np.arange(self.l + 1) * self.alpha
    
    def calc_lambdas_multiplicity(self):
        self.lambdas_multiplicity = [comb(self.l, k) * (self.alpha-1) ** k
                                     for k in range(self.lp1)]
    
    def get_Vj_basis(self, j):
        if not hasattr(self, 'vj'):
            site_L = self.alpha * np.eye(self.alpha) - np.ones((self.alpha, self.alpha))
            v0 = np.full((self.alpha, 1), 1 / np.sqrt(self.alpha)) 
            v1 = orth(site_L)
            
            msg = 'Basis for subspaces V0 and V1 are not orthonormal'
            check_error(np.allclose(v1.T.dot(v0), 0), msg)
            
            self.vj = {0: v0, 1: v1, (0,): v0, (1,): v1}

        if j not in self.vj:
            self.vj[j] = calc_tensor_product([self.vj[j[0]], self.get_Vj_basis(j[1:])])
            
        return(self.vj[j])
    
    def calc_eigenspace_basis(self, k):
        basis = []
        for idxs in combinations(range(self.l), k):
            idxs = np.array(idxs, dtype=int)
            j = np.zeros(self.l, dtype=int)
            j[idxs] = 1
            basis.append(self.get_Vj_basis(tuple(j)))
        basis = np.hstack(basis)
        return(basis)
    
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


class SkewedLaplacianOperator(SeqLinOperator):
    def __init__(self, n_alleles, seq_length, ps=None, max_size=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length,
                         max_size=max_size)
        self.set_ps(ps)
        self.calc_Kns()
        self.calc_lambdas(ps=ps)
        self.calc_lambdas_multiplicity()
        
        if max_size is None:
            self.calc_L()
            self.dot = self.dot1
        else:
            self.dot = self.dot2
    
    def set_ps(self, ps):
        self.variable_ps = ps is not None
        
        if ps is None:
            ps = [np.ones(self.alpha)] * self.l
        check_error(len(ps) == self.l, msg='Number of ps should be equal to length')

        # Normalize ps to have the eigenvalues in the right scale
        self.ps = np.vstack([p / p.sum() * self.alpha for p in ps])
        self.pi = calc_tensor_product([p.reshape((p.shape[0], 1)) for p in ps]).flatten()
    
    def calc_lambdas(self, ps=None):
        self.lambdas = np.arange(self.l + 1) * self.alpha
    
    def calc_lambdas_multiplicity(self):
        self.lambdas_multiplicity = [comb(self.l, k) * (self.alpha-1) ** k
                                     for k in range(self.lp1)]
    
    def calc_Kn(self, p):
        if self.variable_ps:
            Kn = np.vstack([p] * p.shape[0])
            np.fill_diagonal(Kn, np.zeros(Kn.shape[0]))
        else:
            Kn = np.full((self.alpha, self.alpha), True)
            np.fill_diagonal(Kn, np.full(self.alpha, False))
        return(Kn)
    
    def calc_Kns(self):
        Kns = [self.calc_Kn(p) for p in self.ps]
        if self.max_size is not None:
            size = self.guess_n_products()
            i = self.l - size
            Kns = Kns[:i] + [calc_cartesian_product([csr_matrix(m) for m in Kns[i:]]).tocsr()]
        self.Kns = Kns
        self.Kns_shape = [x.shape for x in Kns]
    
    def calc_A(self):
        self.A = calc_cartesian_product(self.Kns)
    
    def calc_L(self):
        L = -calc_cartesian_product(self.Kns).astype(np.float64)
        L.setdiag(-L.sum(1).A1)
        self.L = L.tocsr()
    
    def dot1(self, v):
        return(self.L.dot(v))
    
    def dot2(self, v):
        return(self.d * v - calc_cartesian_product_dot(self.Kns, v))
                

class LapDepOperator(SeqLinOperator):
    def __init__(self, n_alleles=None, seq_length=None, L=None):
        if L is None:
            msg = 'either L or both seq_length and n_alleles must be given'
            check_error(n_alleles is not None and seq_length is not None, msg=msg)
            L = LaplacianOperator(n_alleles, seq_length)
        else:
            msg = 'either L or both seq_length and n_alleles must be given'
            check_error(L is not None, msg=msg)
            n_alleles, seq_length = L.alpha, L.l
            
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.L = L
    

class DeltaPOperator(LapDepOperator):
    def __init__(self, P, n_alleles=None, seq_length=None, L=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L)
        self.set_P(P)
        self.calc_kernel_dimension()
        self.calc_n_p_faces()
        self.calc_lambdas()

    def calc_kernel_dimension(self):
        self.kernel_dimension = np.sum(self.L.lambdas_multiplicity[:self.P])
    
    def calc_n_p_faces(self):
        n_p_sites = comb(self.l, self.P)
        n_p_faces_per_sites = comb(self.alpha, 2) ** self.P
        allelic_comb_remaining_sites = self.alpha ** (self.l - self.P)
        self.n_p_faces = n_p_sites * n_p_faces_per_sites * allelic_comb_remaining_sites
        
    def set_P(self, P):
        self.P = P
        if self.P == (self.l + 1):
            msg = '"P" = l+1, the optimal density is equal to the empirical frequency.'
            raise ValueError(msg)
        elif not 1 <= self.P <= self.l:
            msg = '"P" not in the right range.'
            raise ValueError(msg)
        self.Pfactorial = factorial(self.P)
    
    def dot(self, v):
        dotv = v.copy()
        for p in range(self.P):
            dotv = self.L.dot(v) - p * self.alpha * v
        return(dotv / self.Pfactorial)

    def calc_kernel_basis(self, max_value_to_zero=1e-12):
        basis = np.hstack([self.L.calc_eigenspace_basis(k=p) for p in range(self.P)])
        basis[np.abs(basis) < max_value_to_zero] = 0
        self.kernel_basis = csr_matrix(basis)
    
    def calc_lambdas(self):
        lambdas = []
        for L_lambda_k in self.L.lambdas:
            lambda_k = 1
            for p in range(self.P):
                lambda_k *= L_lambda_k - p * self.alpha
            lambdas.append(lambda_k / self.Pfactorial)
        self.lambdas = np.array(lambdas)


class VjProjectionOperator(SeqLinOperator):
    def __init__(self, n_alleles, seq_length):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.calc_elementary_W()
    
    def calc_elementary_W(self):
        b = np.ones(self.alpha)
        self.b = b
        self.W0 = np.outer(b, b) / np.sum(b ** 2)
        self.W1 = np.eye(self.alpha) - self.W0
        
    def set_j(self, positions):
        self.positions = positions
        self.k = len(positions) 
        self.repeats = self.alpha ** (self.l - self.k)
        self.matrices = [self.W1 if p in positions else self.W0
                         for p in range(self.l)]
    
    def dot(self, v):
        return(kron_dot(self.matrices, v))
    
    def dot_square_norm(self, v):
        axis = [p not in self.positions for p in range(self.l)]
        u = self.contract_v(v).mean(axis=axis)
        
        matrices = [self.W1] * self.k
        sqnorm = self.repeats * self.np.sum(kron_dot(matrices, u) ** 2)
        return(sqnorm)


class SkewedVjProjectionOperator(LapDepOperator):
    def __init__(self, n_alleles=None, seq_length=None, L=None, max_size=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L)
        self.calc_elementary_W()
        self.max_size = max_size
        self.size = self.guess_n_products()
        self.cache = {}
    
    def calc_elementary_W(self):
        if self.L.ps is None:
            self.b = [np.ones(self.alpha)] * self.l
        else:
            self.b = self.L.ps
        self.D_pi = [np.diag(b) for b in self.b]
        self.pi = calc_tensor_product([b.reshape((b.shape[0], 1)) for b in self.b]).flatten()
        self.W0 = [np.outer(b, b).dot(D) / np.sum(b * D.dot(b))
                   for b, D in zip(self.b, self.D_pi)]
        self.W1 = [np.eye(self.alpha) - w0 for w0 in self.W0]
        self.Ws = [[w0, w1] for w0, w1 in zip(self.W0, self.W1)]
        
    def get_Vj_matrix(self, js):
        js_tuple = tuple(js)
        if js_tuple not in self.cache:
            m = self.l - self.size
            ms = [self.Ws[m+p][j] for p, j in enumerate(js)]
            self.cache[js_tuple] = calc_tensor_product(ms)
        return(self.cache[js_tuple])
    
    def set_j(self, positions):
        js = np.zeros(self.l, dtype=int)
        js[positions] = 1

        if self.max_size is None:
            self.matrices = [self.Ws[p][j] for p, j in enumerate(js)]
        else:
            m = self.l - self.size
            self.matrices = [self.Ws[p+m][j] for p, j in enumerate(js[:m])]
            self.matrices.append(self.get_Vj_matrix(js[m:]))
    
    def dot(self, v):
        return(calc_tensor_product_dot(self.matrices, v))
    
    def dot_square_norm(self, v):
        '''
        Note: we are calculating the squared D_pi-norm of the projection to be 
        able to do it directly through recursive product
        '''
        return(calc_tensor_product_quad(self.matrices, v1=self.pi * v, v2=v))

    
class ProjectionOperator(LapDepOperator):
    def __init__(self, n_alleles=None, seq_length=None, L=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L)
        self.calc_eig_vandermonde_matrix()
        self.calc_polynomial_coeffs(numeric=False)
    
    def calc_eig_vandermonde_matrix(self):
        self.V = np.vstack([self.L.lambdas ** i for i in range(self.lp1)]).T
        self.V_LU = lu_factor(self.V)
        return(self.V)
    
    def calc_polynomial_coeffs(self, numeric=False):
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
            self.B = np.linalg.inv(self.V)

        else:        
            B = np.zeros((self.lp1, self.lp1))
            idx = np.arange(self.lp1)
            
            for k in idx:
                k_idx = idx != k
                k_lambdas = self.L.lambdas[k_idx]
                norm_factor = 1 / np.prod(k_lambdas - self.L.lambdas[k])
    
                for power in idx:
                    p = np.sum([np.product(v) for v in combinations(k_lambdas, self.l - power)])
                    B[power, k] = norm_factor * (-1) ** (power) * p
            
            self.B = B
            
        return(self.B)
    
    def lambdas_to_coeffs(self, lambdas, use_lu=False):
        if use_lu:
            coeffs = lu_solve(self.V_LU, lambdas)
        else:
            coeffs = self.B.dot(lambdas)
        return(coeffs)
    
    def set_lambdas(self, lambdas=None, k=None):
        msg = 'Only one "k" or "lambdas" can and must be provided'
        check_error((lambdas is None) ^ (k is None), msg=msg)
        
        if lambdas is None:
            lambdas = np.zeros(self.l + 1)
            lambdas[k] = 1
            
        self.lambdas = lambdas
        self.coeffs = self.lambdas_to_coeffs(lambdas)
    
    def calc_covariance_distance(self):
        return(self.L.W_kd.T.dot(self.lambdas))
    
    def dot(self, v):
        check_error(hasattr(self, 'coeffs'),
                    msg='"lambdas" must be defined for projection')
        projection = calc_matrix_polynomial_dot(self.coeffs, self.L, v)
        return(projection)
    
    def inv_dot(self, v):
        lambdas = self.lambdas.copy()
        check_error(np.all(lambdas > 0), msg='All lambdas must be > 0')
        self.set_lambdas(1 / lambdas)
        u = self.dot(v)
        self.set_lambdas(lambdas)
        return(u)
    

class KernelOperator(SeqLinOperator):
    def __init__(self, W):
        self.W = W
        self.D_sqrt_pi_inv = get_sparse_diag_matrix(1 / np.sqrt(W.L.pi))
        self.D_pi_inv = get_sparse_diag_matrix(1 / W.L.pi)
        self.n = W.n
        self.shape = (self.n, self.n)
        self.known_var = False
    
    def set_y_var(self, y_var=None, obs_idx=None):
        
        if y_var is not None and obs_idx is not None:
            msg = 'y_var and obs_idx should have same dimension: {} vs {}'
            msg = msg.format(y_var.shape[0], obs_idx.shape[0])
            check_error(y_var.shape[0] == obs_idx.shape[0], msg=msg)
            self.known_var = True
            self.homoscedastic = np.unique(y_var).shape[0] == 1
            self.mean_var = y_var.mean()
            self.y_var_diag = get_sparse_diag_matrix(y_var)
            self.n_obs = obs_idx.shape[0]
            self.calc_gt_to_data_matrix(obs_idx)
            self.shape = (self.n_obs, self.n_obs)

        else:
            msg = 'y_var and obs_idx must be provided with each other'
            check_error(y_var is None and obs_idx is None, msg=msg)

    @property
    def lambdas_multiplicity(self):
        return(self.W.L.lambdas_multiplicity)
        
    def set_lambdas(self, lambdas):
        self.W.set_lambdas(lambdas)
    
    def get_lambdas(self):
        return(self.W.lambdas)
    
    def _dot(self, v):
        if hasattr(self.W.L, 'variable_ps') and self.W.L.variable_ps:
            return(self.W.dot(self.D_pi_inv.dot(v)))
        else:
            return(self.W.dot(v))

    def calc_gt_to_data_matrix(self, obs_idx):
        n_obs = obs_idx.shape[0]
        self.gt2data = csr_matrix((np.ones(n_obs), (obs_idx, np.arange(n_obs))),
                                  shape=(self.n, n_obs))   
    
    def dot(self, v, all_rows=False, add_y_var_diag=True, full_v=False):
        if full_v or not self.known_var:
            u = self._dot(v)
        else:
            u = self._dot(self.gt2data.dot(v))
            if add_y_var_diag:
                u += self.gt2data.dot(self.y_var_diag.dot(v))

        if not all_rows and self.known_var:
            u = self.gt2data.T.dot(u)
        return(u)
    
    @property
    def Kop(self):
        if not hasattr(self, '_Kop'):
            self._Kop = LinearOperator((self.n_obs, self.n_obs), matvec=self.dot)
        return(self._Kop)
    
    def inv_dot(self, v, show=False):
        res = minres(self.Kop, v, tol=1e-6, show=show)
        self.res = res[1]
        return(res[0])
    
    def inv_quad(self, v, show=False):
        u = self.inv_dot(v, show=show)
        return(np.sum(u * v))
