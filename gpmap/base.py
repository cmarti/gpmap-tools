#!/usr/bin/env python
from itertools import product
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csr import csr_matrix
from scipy.sparse.dia import dia_matrix
from numpy.linalg.linalg import matrix_power, norm
from scipy.special._basic import comb
from scipy.sparse.linalg.eigen.arpack.arpack import eigsh

from gpmap.utils import write_log
from gpmap.settings import (DNA_ALPHABET, RNA_ALPHABET, PROTEIN_ALPHABET,
                            ALPHABET, MAX_GENOTYPES, PROT_AMBIGUOUS_VALUES,
                            DNA_AMBIGUOUS_VALUES, RNA_AMBIGUOUS_VALUES)


def get_sparse_diag_matrix(values):
    n_genotypes = values.shape[0]
    m = dia_matrix((values, np.array([0])), shape=(n_genotypes, n_genotypes))
    return(m)


def extend_ambigous_seq(seq, mapping):
    return(list(map("".join, product(*map(mapping.get, seq)))))


class BaseGPMap(object):
    def set_alphabet_type(self, alphabet_type, n_alleles=None):
        self.alphabet_type = alphabet_type
        
        if isinstance(alphabet_type, list):
            self.alphabet = self.alphabet_type
            
        elif alphabet_type == 'dna':
            self.alphabet = [DNA_ALPHABET] * self.length
            self.complements = {'A': ['T'], 'T': ['A'],
                                'G': ['C'], 'C': ['G']}
            self.ambiguous_values = DNA_AMBIGUOUS_VALUES
            
        elif alphabet_type == 'rna':
            self.alphabet = [RNA_ALPHABET] * self.length
            self.complements = {'A': ['U'], 'U': ['A', 'G'],
                                'G': ['C', 'U'], 'C': ['G']}
            self.ambiguous_values = RNA_AMBIGUOUS_VALUES
            
        elif alphabet_type == 'protein':
            self.ambiguous_values = PROT_AMBIGUOUS_VALUES
            self.alphabet = [PROTEIN_ALPHABET] * self.length
            
        elif alphabet_type == 'custom':
            if n_alleles is None:
                raise ValueError('n_alleles must be provided for custom alphabet')
            n_alleles = [n_alleles] if isinstance(n_alleles, int) else n_alleles
            self.alphabet = [[ALPHABET[x] for x in range(a)] for a in n_alleles]
            self.ambiguous_values = [{'X': ''.join(a)} for a in self.alphabet]
            for i, alleles in enumerate(self.alphabet):
                self.ambiguous_values[i].update(dict(zip(alleles, alleles)))
                
        else:
            raise ValueError('alphabet_type not supported')
        
        if n_alleles is None:
            n_alleles = len(self.alphabet)
            
        if not isinstance(n_alleles, list):
            self.n_alleles = [n_alleles] * self.length
        else:
            self.n_alleles = n_alleles

    def extend_ambiguous_sequence(self, seq):
        """return list of all possible sequences given an ambiguous DNA input
        copied from https://www.biostars.org/p/260617/
        """
        return(extend_ambigous_seq(seq, self.ambiguous_values))
    
    def extend_ambiguous_sequences(self, seqs):
        extended = []
        for seq in seqs:
            extended.extend(self.extend_ambiguous_sequence(seq))
        return(extended)
    
    def report(self, msg):
        write_log(self.log, msg)


class SequenceSpace(BaseGPMap):
    def init(self, length, n_alleles=None, alphabet_type='dna', log=None):
        self.length = length
        
        self.set_alphabet_type(alphabet_type, n_alleles=n_alleles)
        self.n_genotypes = np.prod(self.n_alleles)
        
        if self.n_genotypes > MAX_GENOTYPES:
            raise ValueError('Sequence space is too big to handle')
        
        self.genotypes = self.get_genotypes()
        self.genotype_idxs = pd.Series(np.arange(self.n_genotypes),
                                       index=self.genotypes)

        # Matrices for site level elementary graphs
        self.site_I = [sp.identity(a) for a in self.n_alleles]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.site_Kn = [csr_matrix(np.ones((a, a))) - I
                            for a, I in zip(self.n_alleles, self.site_I)]
            
        self.log = log
    
    def get_seqs_from_alleles(self, alphabet):
        if not alphabet:
            yield('')
        else:
            for allele in alphabet[0]:
                for seq in self.get_seqs_from_alleles(alphabet[1:]):
                    yield(allele + seq)
    
    def get_genotypes(self):
        return(np.array([x for x in self.get_seqs_from_alleles(self.alphabet)]))    
        
    def get_m_k(self, k):
        return(self.m[k])
    
    def get_genotypes_idx(self, genotypes):
        return(self.genotype_idxs.loc[genotypes].values)
    
    def _get_A_csr(self):
        self.get_adjacency_matrix()
        if not hasattr(self, 'A_csr'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.A_csr = self.A.tocsr()
        return(self.A_csr)
    
    def get_neighborhood_idxs(self, seq, max_distance=1):
        idxs = np.array([self.genotype_idxs.loc[seq]]).flatten()
        a = self._get_A_csr()
        for _ in range(max_distance):
            idxs = np.append(idxs, a[idxs].indices)
        return(np.unique(idxs))
    
    def get_adjacency_matrix(self):
        if not hasattr(self, 'A'):
            self.calc_adjacency()
        return(self.A)
    
    def get_neighbor_pairs(self):
        if not hasattr(self, 'neighbor_pairs'):
            A = self.get_adjacency_matrix()
            try:
                self.neighbor_pairs = A.row, A.col
            except AttributeError:
                A = A.tocoo()
                self.neighbor_pairs = A.row, A.col
        return(self.neighbor_pairs)
    
    def _istack_matrices(self, m_diag, m_offdiag, pos):
        rows = []
        a = self.n_alleles[pos]
        for j in range(a):
            row = [m_offdiag] * j + [m_diag] + [m_offdiag] * (a - j - 1)
            rows.append(sp.hstack(row)) 
        m = sp.vstack(rows)
        return(m)

    def _calc_adjacency(self, m=None, pos=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if m is None:
                m = self.site_Kn[pos]
            
            if pos == (self.length-1):
                return(m)
    
            i = sp.identity(m.shape[0])
            m = self._istack_matrices(m, i, pos)
        return(self._calc_adjacency(m, pos+1))
    
    def calc_adjacency(self):
        self.A = self._calc_adjacency()

    def calc_generalized_laplacian(self, probability):
        self.calc_adjacency()
        seq1 = self.seqs[self.A.row, :]
        seq2 = self.seqs[self.A.col, :]
        diff = seq1 != seq2
        different_pos = np.argmax(diff, axis=1)
        
        values = np.array([np.sqrt(probability[s1[p], p] * probability[s2[p], p])
                           for s1, s2, p in zip(seq1, seq2, different_pos)])
        p = np.array([[probability[a, p] for p, a in enumerate(seq)] for seq in self.seqs])
        
        self.probability = np.exp(np.log(p).sum(1))
        L = self.A.copy()
        L.data = -values
        diag = (1 - p).sum(1)
        L.setdiag(diag)

        self.L = L

    def calc_laplacian(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.calc_adjacency()
            self.L = -self.A.tocsr()
        
        diag = -self.L.sum(1).A[:, 0]
        self.L.setdiag(diag)

    def calc_incidence_matrix(self):
        self.report('Calculating graph laplacian')
        
        row_ids, col_ids, values = [], [], []
        k = max(int(self.n_genotypes / 10), 1)

        edge_i = 0        
        for i in range(self.n_genotypes):
            for j in self.get_neighbors(i, only_j_higher_than_i=True):
                row_ids.append(edge_i)
                col_ids.append(i)
                values.append(1)
                
                row_ids.append(edge_i)
                col_ids.append(j)
                values.append(-1)
                
                edge_i += 1
            
            if i % k == 0:
                msg = '\t{:.0f}% completed ({})'
                self.report(msg.format((i)/self.n_genotypes*100, i))
        
        H = csr_matrix((values, (row_ids, col_ids)),
                        shape=(edge_i, self.n_genotypes))
        return(H)
    
    def calc_MAT(self):
        """Construct entries of powers of L. 
        Column: powers of L. 
        Row: Hamming distance"""
        l, a, s = self.length, self.n_alleles, self.length + 1
    
        # Construct C
        C = np.zeros([s, s])
        for i in range(s):
            for j in range(s):
                if i == j:
                    C[i, j] = i * (a - 2)
                if i == j + 1:
                    C[i, j] = i
                if i == j - 1:
                    C[i, j] = (l - j + 1) * (a - 1)
    
        # Construct D
        D = np.array(np.diag(l * (a - 1) * np.ones(s), 0))
    
        # Construct B
        B = D - C
    
        # Construct u
        u = np.zeros(s)
        u[0], u[1] = l * (a - 1), -1
    
        # Construct MAT column by column
        MAT = np.zeros([s, s])
        MAT[0, 0] = 1
        for j in range(1, s):
            MAT[:, j] = matrix_power(B, j-1).dot(u)
    
        # Invert MAT
        self.MAT_inv = np.linalg.inv(MAT)
        self.MAT = MAT
    
    def calc_w(self, k, d):
        """return value of the Krwatchouk polynomial for k, d"""
        l, a = self.length, self.n_alleles
        s = 0
        for q in range(l + 1):
            s += (-1)**q * (a - 1)**(k - q) * comb(d, q) * comb(l - d, k - q)
        return 1 / a**l * s
    
    def calc_W_kd_matrix(self):
        """return full matrix l+1 by l+1 Krawtchouk matrix"""
        self.W_kd = np.zeros([self.length + 1, self.length + 1])
        for k in range(self.length + 1):
            for d in range(self.length + 1):
                self.W_kd[k, d] = self.calc_w(k, d)
    
    def calc_eigenvalue_multiplicity(self):
        self.lambdas_multiplicity = np.array([comb(self.length, k) * (self.n_alleles - 1) ** k
                                              for k in range(self.length + 1)], dtype=int)
    
    def calc_collapsed_covariance_matrix(self, lambdas):
        return(np.dot(self.W_kd.T, lambdas))
    
    def calc_L_powers_coeffs(self, lambdas):
        covariance = self.calc_collapsed_covariance_matrix(lambdas)
        return(np.dot(self.MAT_inv, covariance))
    
    def calc_laplacian_powers(self, v):
        powers = [v]
        for _ in range(self.length):
            powers.append(self.L.dot(powers[-1]))
        return(np.vstack(powers).T)

    def calc_arnoldi_basis(self, v, ndim=None):
        if ndim is None:
            ndim = self.length + 1
        
        basis = [v / np.dot(v.T, v)]
        for _ in range(1, ndim):
            v_0 = basis[-1]
            v_i = self.L.dot(v_0)
            
            for v_j in basis:
                v_i = v_i - np.dot(v_i, v_j) * v_j
            
            print(len(basis), norm(v_i), norm(np.dot(np.vstack(basis), v_i)))
            v_i = v_i / norm(v_i)
            
            basis.append(v_i)
        return(np.vstack(basis).T)
    
    def calc_distance_matrix(self, i=None):
        if i == None:
            i = self.length - 1
        
        I = np.identity(self.n_alleles, dtype=int)
        if i == 0:
            D = np.ones((self.n_alleles, self.n_alleles), dtype=int) - I
        else:
            D_1 = self.calc_distance_matrix(i-1)
            D = self.istack_matrices(D_1, D_1 + 1, module=np)
        return(D)
    
    def calc_L_eigendecomposition(self):
        lambdas, q = eigsh(self.L, self.n_genotypes-1, which='LM')
        
        # Ensure eigenvalues
        for i in range(q.shape[1]):
            assert(np.allclose(self.L.dot(q[:, i]), lambdas[i] * q[:, i]))
        
        # Ensure orthonormal
        assert(np.allclose(np.identity(self.n_genotypes -1), np.dot(q.T, q)))
        
        self.L_lambdas = lambdas
        self.L_q = q
        return(lambdas, q)
