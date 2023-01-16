#!/usr/bin/env python
import numpy as np

from math import factorial, comb
from itertools import combinations

from scipy.sparse.csr import csr_matrix
from scipy.sparse import triu
from scipy.linalg.decomp_svd import orth

from gpmap.src.utils import (calc_cartesian_product, check_error,
                             calc_matrix_polynomial_dot, calc_tensor_product)


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
    def __init__(self, n_alleles, seq_length, save_memory=False):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.save_memory = save_memory
    
        self.calc_Kn()    
        if save_memory:
            self.calc_F()
            self.dot = self.dot3
        else:
            self.calc_L()
#             self.calc_A_triu()
            self.dot = self.dot0
            
        self.lambdas = np.arange(seq_length + 1)
    
    def calc_F(self):
        self.F = np.ones((self.alpha, self.alpha)) - 2 * np.eye(self.alpha)
    
    def calc_Kn(self):
        # TODO: figure out how to avoid ones
        Kn = np.ones((self.alpha, self.alpha))
        np.fill_diagonal(Kn, np.zeros(self.alpha))
        self.Kn = csr_matrix(Kn)
        self.Kn_triu = csr_matrix(triu(Kn))
    
    def calc_A_triu(self):
        self.triu_A = calc_cartesian_product([self.Kn_triu] * self.l)
    
    def calc_L(self):
        self.L = -calc_cartesian_product([self.Kn] * self.l)
        self.L.setdiag(self.d)
    
    def dot0(self, v):
        return(self.L.dot(v))
    
    def dot1(self, v):
        # TODO: figure out if this is saving us memory or not
        return(self.d * v - self.triu_A.dot(v) - self.triu_A.transpose(copy=False).dot(v))
    
    def dot2(self, v):
        if not hasattr(self, 'neighbors'):
            self.neighbors = np.vstack((self.triu_A + self.triu_A.T).asformat('lil').rows)
        return(self.d * v - v[self.neighbors].sum(1))
    
    def _dot3(self, v):
        size = v.shape[0]
        if size == self.alpha:
            return(v)
        else:
            s = size // self.alpha
            vs = np.vstack([self._dot3(v[i*s:(i+1)*s]) for i in range(self.alpha)])
            return(np.hstack(vs.sum(1).reshape((self.alpha, 1)) + self.F.dot(vs)))
    
    def dot3(self, v):
        return(self.d * v - self._dot3(v))
    
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
    
    def calc_lambdas_multiplicity(self):
        self.lambdas_multiplicity = [comb(self.l, k) * (self.alpha-1) ** k
                                     for k in range(self.lp1)]
    
    
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
    
    def dot(self, v):
        projection = calc_matrix_polynomial_dot(self.coeffs, self.L, v)
        return(projection)
