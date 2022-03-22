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
                            MAX_GENOTYPES)
from Bio.Data import IUPACData


def get_sparse_diag_matrix(values):
    n_genotypes = values.shape[0]
    m = dia_matrix((values, np.array([0])), shape=(n_genotypes, n_genotypes))
    return(m)


class BaseGPMap(object):
    def set_alphabet_type(self, alphabet_type, n_alleles=None):
        self.alphabet_type = alphabet_type
        if isinstance(alphabet_type, list):
            self.alphabet = self.alphabet_type
        elif alphabet_type == 'dna':
            self.alphabet = DNA_ALPHABET
            self.complements = {'A': ['T'], 'T': ['A'],
                                'G': ['C'], 'C': ['G']}
            self.ambiguous_values = IUPACData.ambiguous_dna_values
        elif alphabet_type == 'rna':
            self.alphabet = RNA_ALPHABET
            self.complements = {'A': ['U'], 'U': ['A', 'G'],
                                'G': ['C', 'U'], 'C': ['G']}
            self.ambiguous_values = IUPACData.ambiguous_rna_values
        elif alphabet_type == 'protein':
            self.alphabet = PROTEIN_ALPHABET
        elif alphabet_type == 'custom':
            if n_alleles is None:
                raise ValueError('n_alleles must be provided for custom alphabet')
            self.alphabet = [str(x) for x in range(n_alleles)]
            self.ambiguous_values = {'*': ''.join(self.alphabet),
                                     'N': ''.join(self.alphabet)}
            self.ambiguous_values.update(dict(zip(self.alphabet, self.alphabet)))
        else:
            raise ValueError('alphabet type not supported')
        self.n_alleles = len(self.alphabet)
    
    def extend_ambiguous_sequence(self, seq):
        """return list of all possible sequences given an ambiguous DNA input
        copied from https://www.biostars.org/p/260617/
        """
        return(list(map("".join, product(*map(self.ambiguous_values.get, seq)))))
    
    def extend_ambiguous_sequences(self, seqs):
        extended = []
        for seq in seqs:
            extended.extend(self.extend_ambiguous_sequence(seq))
        return(extended)
    
    def report(self, msg):
        write_log(self.log, msg)


class SequenceSpace(BaseGPMap):
    def init(self, length, n_alleles=None, alphabet_type='dna', log=None):
        self.set_alphabet_type(alphabet_type, n_alleles=n_alleles)
        self.length = length
        self.n_genotypes = self.n_alleles ** length
        if self.n_genotypes > MAX_GENOTYPES:
            raise ValueError('Sequence space is too big to handle')
        
        self.n_neighbors = length * (self.n_alleles - 1)
        self.bases = list(range(self.n_alleles))
        self.seq_to_pos_converter = np.flip(self.n_alleles**np.array(range(length)), axis=0)
        self.seqs = np.array(list(product(self.bases, repeat=length)))
        self.genotype_labels = self.get_genotype_labels()
        self.genotype_idxs = pd.Series(np.arange(self.n_genotypes),
                                       index=self.genotype_labels)
        self.subgraph_identity = sp.identity(self.n_alleles)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.complete_subgraph_A = csr_matrix(np.ones((self.n_alleles, self.n_alleles))) - self.subgraph_identity
        self.log = log
        
    def get_m_k(self, k):
        return(self.m[k])
    
    def get_genotype_labels(self):
        return(np.array([''.join([self.alphabet[a] for a in gt])
                         for gt in self.seqs]))
    
    def seq_to_pos(self, seq, coding_dict=None):
        if coding_dict is None:
            return int(np.sum(seq * self.seq_to_pos_converter))
        else:
            tmp = [coding_dict[letter] for letter in seq]
        return int(np.sum(tmp * self.seq_to_pos_converter))
    
    def get_seq_from_idx(self, idx):
        return(self.genotype_labels[idx])
    
    def get_A_csr(self):
        self.get_adjacency_matrix()
        if not hasattr(self, 'A_csr'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.A_csr = self.A.tocsr()
        return(self.A_csr)
    
    def get_neighborhood_idxs(self, seq, max_distance=1):
        idxs = np.array([self.genotype_idxs.loc[seq]]).flatten()
        a = self.get_A_csr()
        for _ in range(max_distance):
            idxs = np.append(idxs, a[idxs].indices)
        return(np.unique(idxs))
    
    def get_neighbors_old(self, i, only_j_higher_than_i=False):
        for site in range(self.length):
            for base in self.bases:
                seq_i = np.array(self.seqs[i])
                if base == seq_i[site]:
                    continue
                seq_i[site] = base
                j = self.seq_to_pos(seq_i)
                if only_j_higher_than_i and j < i:
                    continue
                yield(j)
    
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
    
    def istack_matrices(self, m_diag, m_offdiag, module=sp):
        rows = []
        for j in range(self.n_alleles):
            row = [m_offdiag] * j + [m_diag] + [m_offdiag] * (self.n_alleles - j - 1)
            rows.append(module.hstack(row)) 
        m = module.vstack(rows)
        return(m)

    def _calc_adjacency(self, m=None, l=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if m is None:
                i = sp.identity(self.n_alleles)
                m = csr_matrix(np.ones((self.n_alleles, self.n_alleles))) - i
            
            if l == (self.length-1):
                return(m)
    
            i = sp.identity(m.shape[0])
            m = self.istack_matrices(m, i)
        return(self._calc_adjacency(m, l+1))
    
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
