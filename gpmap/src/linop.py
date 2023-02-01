#!/usr/bin/env python
import numpy as np

from math import factorial
from itertools import combinations

from scipy.sparse.csr import csr_matrix
from scipy.linalg.decomp_svd import orth
from scipy.special._basic import comb

from gpmap.src.utils import (calc_cartesian_product, check_error,
                             calc_matrix_polynomial_dot, calc_tensor_product,
                             calc_cartesian_product_dot,
    calc_tensor_product_dot, calc_tensor_product_quad)


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
    def __init__(self, n_alleles, seq_length, ps=None, max_size=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.max_size = max_size
    
        self.ps = ps
        self.calc_Kns(ps=ps)
            
        if max_size is None:
            self.calc_L()
            self.dot = self.dot0
        else:
            self.dot = self.dot1
            
        self.calc_lambdas()
        self.calc_lambdas_multiplicity()
        if ps is None:
            self.calc_W_kd_matrix()
    
    def set_ps(self, ps):
        if ps is None:
            ps = [np.ones(self.alpha)] * self.l
        check_error(len(ps) == self.l)
        self.ps = ps
    
    def calc_lambdas(self):
        self.lambdas = np.arange(self.l + 1)
        if self.ps is None:
            self.lambdas *= self.alpha
    
    def calc_Kn(self, p):
        Kn = np.vstack([p] * p.shape[0])
        np.fill_diagonal(Kn, np.zeros(Kn.shape[0]))
        return(Kn)
    
    def guess_n_products(self):
        size = 1
        for k in range(self.l):
            size *= self.alpha
            if size >= self.max_size:
                break
        return(k)
    
    def calc_Kns(self):
        if self.max_size is None:            
            Kns = [csr_matrix(self.calc_Kn(p)) for p in self.ps]
        else:
            size = self.guess_n_products()
            Kns = [self.calc_Kn(p) for p in self.ps[:-size]]
            Kns.append(calc_cartesian_product([csr_matrix(self.calc_Kn(p)) for p in self.ps[-size:]]).tocsr())
        self.Kns = Kns
        self.Kns_shape = [x.shape for x in Kns]
    
    def calc_L(self):
        L = -calc_cartesian_product(self.Kns)
        L.setdiag(-L.sum(1).A1)
        self.L = L.tocsr()
    
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
    def __init__(self, n_alleles=None, seq_length=None, L=None, max_L_size=False):
        if L is None:
            msg = 'either L or both seq_length and n_alleles must be given'
            check_error(n_alleles is not None and seq_length is not None, msg=msg)
            L = LaplacianOperator(n_alleles, seq_length, max_L_size=max_L_size)
        else:
            msg = 'either L or both seq_length and n_alleles must be given'
            check_error(L is not None, msg=msg)
            n_alleles, seq_length = L.alpha, L.l
            
        super().__init__(n_alleles=n_alleles, seq_length=seq_length)
        self.L = L
    

class DeltaPOperator(LapDepOperator):
    def __init__(self, P, n_alleles=None, seq_length=None, L=None, max_L_size=False):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L,
                         max_L_size=max_L_size)
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


class VjProjectionOperator(LapDepOperator):
    def __init__(self, n_alleles=None, seq_length=None, L=None):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L,
                         max_L_size=True)
        self.calc_elementary_W()
    
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
    
    def set_j(self, positions):
        self.matrices = [self.W1[i] if i in positions else self.W0[i]
                         for i in range(self.l)]
    
    def dot(self, v):
        return(calc_tensor_product_dot(self.matrices, v))
    
    def dot_square_norm(self, v):
        '''
        Note: we are calculating the squared D_pi-norm of the projection to be 
        able to do it directly through recursive product
        '''
        return(calc_tensor_product_quad(self.matrices, v1=self.pi * v, v2=v))

    
class ProjectionOperator(LapDepOperator):
    def __init__(self, n_alleles=None, seq_length=None, L=None, max_L_size=False):
        super().__init__(n_alleles=n_alleles, seq_length=seq_length, L=L,
                         max_L_size=max_L_size)
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
    
    def inv_dot(self, v):
        lambdas = self.lambdas.copy()
        check_error(np.all(lambdas > 0), msg='All lambdas must be > 0')
        self.set_lambdas(1 / lambdas)
        u = self.dot(v)
        self.set_lambdas(lambdas)
        return(u)
