#!/usr/bin/env python
import numpy as np

from math import factorial
from itertools import combinations

from scipy.sparse.csr import csr_matrix
from scipy.sparse import triu
from scipy.linalg.decomp_svd import orth
from scipy.special._basic import comb

from gpmap.src.utils import (calc_cartesian_product, check_error,
                             calc_matrix_polynomial_dot, calc_tensor_product,
    calc_cartesian_product_dot)


class SeqLinOperator(object):
    def __init__(self, n_alleles, seq_length):
        self.alpha = n_alleles
        self.l = seq_length
        self.lp1 = seq_length + 1
        self.n = self.alpha ** self.l
        self.d = (self.alpha - 1) * self.l
        self.shape = (self.n, self.n)
    
    def quad(self, v):
        return(np.sum(v * self.dot(v)))


class LaplacianOperator(SeqLinOperator):
    def __init__(self, n_alleles, seq_length, save_memory=False, ps=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.save_memory = save_memory
    
        self.ps = ps
        self.calc_Kns(ps=ps)
            
        if save_memory:
            self.dot = self.dot1
        else:
            self.calc_L()
            self.dot = self.dot0
            
        self.calc_lambdas()
        self.calc_lambdas_multiplicity()
        self.calc_W_kd_matrix()
    
    def calc_lambdas(self):
        self.lambdas = np.arange(self.l + 1)
        if self.ps is None:
            self.lambdas *= self.alpha
    
    def calc_Kn(self, p=None):
        if p is None:
            p = np.ones(self.alpha)
        Kn = np.vstack([p] * p.shape[0])
        np.fill_diagonal(Kn, np.zeros(Kn.shape[0]))
        return(csr_matrix(Kn))
    
    def calc_Kns(self, ps=None):
        if ps is None:
            ps = [None] * self.l
        self.Kns = [self.calc_Kn(p) for p in ps]
    
    def calc_L(self):
        L = -calc_cartesian_product(self.Kns)
        L.setdiag(-L.sum(1).A1)
        self.L = L
    
    def dot0(self, v):
        return(self.L.dot(v))
    
    def dot1(self, v):
        return(self.d * v - calc_cartesian_product_dot(self.Kns, v))
    
    def get_Vj_basis(self, j):
        if not hasattr(self, 'vj'):
            site_L = self.alpha * np.eye(self.alpha) - np.ones((self.alpha, self.alpha))
            v0 = np.full((self.alpha, 1), 1 / np.sqrt(self.alpha)) 
            v1 = orth(site_L)
            
            msg = 'Basis for subspaces V0 and V1 are not orthonormal'
            check_error(np.allclose(v1.T.dot(v0), 0), msg)
            
            self.vj = {0: v0, 1: v1, (0,): v0, (1,): v1}

        if j not in self.vj:
#             self.vj[j] = np.tensordot(self.vj[j[0]], self.get_Vj(j[1:]), axes=0)
#             self.vj[j] = np.vstack([np.hstack([x * self.get_Vj_basis(j[1:]) for x in row])
#                                     for row in self.vj[j[0]]])
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
    
    def calc_lambdas_multiplicity(self):
        self.lambdas_multiplicity = [comb(self.l, k) * (self.alpha-1) ** k
                                     for k in range(self.lp1)]
    
    def calc_w(self, k, d):
        if self.ps is not None:
            raise ValueError('w method not implemented for unequal ps')
        
        """return value of the Krawtchouk polynomial for k, d"""
        l, a = self.l, self.alpha
        s = 0
        for q in range(l + 1):
            s += (-1)**q * (a - 1)**(k - q) * comb(d, q) * comb(l - d, k - q)
        return(1 / a**l * s)
    
    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.lp1, self.lp1])
        for k in range(self.lp1):
            for d in range(self.lp1):
                self.W_kd[k, d] = self.calc_w(k, d)
                

class LapDepOperator(SeqLinOperator):
    def __init__(self, n_alleles=None, seq_length=None, L=None, save_memory=False):
        if L is None:
            msg = 'either L or both seq_length and n_alleles must be given'
            check_error(n_alleles is not None and seq_length is not None, msg=msg)
            L = LaplacianOperator(n_alleles, seq_length, save_memory=save_memory)
        else:
            msg = 'either L or both seq_length and n_alleles must be given'
            check_error(L is not None, msg=msg)
            n_alleles, seq_length = L.alpha, L.l
            
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.L = L


class DeltaPOperator(LapDepOperator):
    def __init__(self, P, n_alleles=None, seq_length=None, L=None, save_memory=False):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L,
                         save_memory=save_memory)
        self.set_P(P)
        self.n_p_faces = self.calc_n_p_faces()
    
    def calc_n_p_faces(self):
        n_p_sites = comb(self.l, self.P)
        n_p_faces_per_sites = comb(self.alpha, 2) ** self.P
        allelic_comb_remaining_sites = self.alpha ** (self.l - self.P)
        return(n_p_sites * n_p_faces_per_sites * allelic_comb_remaining_sites)
        
    def set_P(self, P):
        self.P = P
        if self.P == (self.l + 1):
            msg = '"P" = l+1, the optimal density is equal to the empirical frequency.'
            raise ValueError(msg)
        elif not 1 <= self.P <= self.l:
            msg = '"P" not in the right range.'
            raise ValueError(msg)
        self.Pfactorial = factorial(self.P)
    
    def _L_dot(self, v, p=0):
        return(self.L.dot(v) - p * self.alpha * v)
    
    def dot(self, v):
        dotv = v.copy()
        for p in range(self.P):
            dotv = self._L_dot(dotv, p)
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
    
    
class ProjectionOperator(LapDepOperator):
    def __init__(self, n_alleles=None, seq_length=None, L=None, save_memory=False):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L,
                         save_memory=save_memory)
        self.calc_polynomial_coeffs()
    
    def calc_eig_vandermonde_matrix(self):
        V = np.vstack([self.L.lambdas ** i for i in range(self.seq_length + 1)]).T
        return(V)
    
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
            self.B = np.linalg.inv(self.calc_eig_vandermonde_matrix())

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
    
    def lambdas_to_coeffs(self, lambdas):
        return(self.B.dot(lambdas))
    
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
                    msg='"lambdas" must be define for projection')
        projection = calc_matrix_polynomial_dot(self.coeffs, self.L, v)
        return(projection)


class KernelAligner(object):
    def __init__(self, correlations, distances_n, seq_length, n_alleles, beta=0):
        self.seq_length = seq_length
        self.n_alleles = n_alleles
        self.set_beta(beta)
        self.calc_W_kd_matrix()
        
        self.correlations = correlations
        self.distances_n = distances_n
        self.construct_a(correlations, distances_n)
        self.construct_M(distances_n)
        self.M_inv = np.linalg.inv(self.M)
    
    def set_beta(self, beta):
        self.beta = beta
    
    def calc_second_order_diff_matrix(self):
        """Construct second order difference matrix for regularization"""
        Diff2 = np.zeros((self.seq_length - 2, self.seq_length))
        for i in range(Diff2.shape[0]):
            Diff2[i, i:i + 3] = [-1, 2, -1]
        self.second_order_diff_matrix = Diff2.T.dot(Diff2)
    
    def frobenius_norm(self, log_lambdas):
        """cost function for regularized least square method for inferring 
        lambdas"""
        lambdas = np.exp(log_lambdas)
        Frob1 = lambdas.dot(self.M).dot(lambdas)
        Frob2 = 2 * lambdas.dot(self.a)
        return(Frob1 - Frob2 + self.beta * log_lambdas[1:].dot(self.second_order_diff_matrix).dot(log_lambdas[1:]))
    
    def frobenius_norm_grad(self, log_lambdas):
        msg = 'gradient calculation only implemented for beta=0'
        check_error(self.beta > 0, msg=msg)
        lambdas = np.exp(log_lambdas)
        grad_Frob1 = 2 * self.M.dot(lambdas)
        grad_Frob2 = 2 * self.a
        return(grad_Frob1 - grad_Frob2)
    
    def construct_M(self, N_d):
        size = self.seq_length + 1
        M = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                for d in range(size):
                    M[i, j] += N_d[d] * self.W_kd[i, d] * self.W_kd[j, d]
        self.M = M
    
    def construct_a(self, rho_d, N_d):
        size = self.seq_length + 1
        a = np.zeros(size)
        for i in range(size):
            for d in range(size):
                a[i] += N_d[d] * self.W_kd[i, d] * rho_d[d]
        self.a = a
    
    def calc_w(self, k, d):
        """return value of the Krawtchouk polynomial for k, d"""
        l, a = self.seq_length, self.n_alleles
        s = 0
        for q in range(l + 1):
            s += (-1)**q * (a - 1)**(k - q) * comb(d, q) * comb(l - d, k - q)
        return(1 / a**l * s)
    
    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.seq_length + 1, self.seq_length + 1])
        for k in range(self.seq_length + 1):
            for d in range(self.seq_length + 1):
                self.W_kd[k, d] = self.calc_w(k, d)
