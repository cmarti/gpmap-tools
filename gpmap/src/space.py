#!/usr/bin/env python
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.sparse.csr import csr_matrix

from gpmap.src.seq import translate_seqs, guess_space_configuration
from gpmap.utils import check_error
from gpmap.src.settings import (DNA_ALPHABET, RNA_ALPHABET, PROTEIN_ALPHABET,
                                ALPHABET, MAX_STATES, PROT_AMBIGUOUS_VALUES,
                                DNA_AMBIGUOUS_VALUES, RNA_AMBIGUOUS_VALUES)


class DiscreteSpace(object):
    def __init__(self, adjacency_matrix, function=None, state_labels=None):
        self.init_space(adjacency_matrix, function=function, state_labels=state_labels)
    
    def _check_attributes(self, tol=1e-6):
        # TODO: check that the space is connected
        check_error(len(self.adjacency_matrix.shape) == 2,
                    msg='Ensure adjacency_matrix is a 2D array')
        check_error(self.adjacency_matrix.shape[0] == self.adjacency_matrix.shape[1],
                    msg='adjacencty_matrix should be square')
        check_error(np.all(self.adjacency_matrix.diagonal() == 0),
                    msg='loops are not allowed in the discrete space')
        check_error((abs(self.adjacency_matrix - self.adjacency_matrix .T)> tol).nnz == 0, 
                    msg='Adjacency matrix has to be symmetric')
        check_error((self.adjacency_matrix.data >= 0).all(), 
                    msg='Adjacency matrix entries have to be positive')
        check_error(self.adjacency_matrix.shape[0] == self.state_labels.shape[0],
                    msg='Size of adjacency_matrix and state_labels does not match')
    
    def _check_function(self):
        check_error(self.function.shape[0] == self.state_labels.shape[0],
                    msg='Size of adjacency_matrix and function does not match')
    
    def set_function(self, function):
        self.function = function
        self._check_function()
    
    def init_space(self, adjacency_matrix, function=None, state_labels=None):
        # State labels
        state_idxs = np.arange(adjacency_matrix.shape[0])
        if state_labels is None:
            state_labels = state_idxs.astype(str)
        
        if not isinstance(state_labels, np.ndarray):
            state_labels = np.array(state_labels)

        # States and their connectivity
        self.adjacency_matrix = adjacency_matrix
        self.state_labels = state_labels
        self.state_idxs = pd.Series(state_idxs, index=state_labels)
        self.n_states = state_labels.shape[0]
        self._check_attributes()
        
        # Function
        if function is not None:
            if not isinstance(function, np.ndarray):
                function = np.array(function)
            self.set_function(function)
    
    def get_state_idxs(self, states):
        return(self.state_idxs.loc[states])
    
    def get_neighbors(self, states, max_distance=1):
        idxs = self.get_state_idxs(states)
        adj_csr = self.adjacency_matrix.tocsr()
        for _ in range(max_distance):
            idxs = np.append(idxs, adj_csr[idxs].indices)
        return(self.state_labels[np.unique(idxs)])
    
    def get_neighbor_pairs(self):
        if not hasattr(self, 'neighbor_pairs'):
            A = self.adjacency_matrix
            try:
                self.neighbor_pairs = A.row, A.col
            except AttributeError:
                A = A.tocoo()
                self.neighbor_pairs = A.row, A.col
        return(self.neighbor_pairs)
    
    def get_edges_df(self):
        i, j = self.get_neighbor_pairs()
        edges_df = pd.DataFrame({'i': i, 'j': j})
        return(edges_df)
    
    def write_csv(self, fpath):
        df = pd.DataFrame({'function': self.function}, 
                          index=self.state_labels)
        df.to_csv(fpath)
    
    
class SequenceSpace(DiscreteSpace):
    def __init__(self, seq_length=None, n_alleles=None,
                 alphabet_type='dna', alphabet=None, function=None,
                 codon_table=None, stop_function=-10):
        self._init(seq_length=seq_length, n_alleles=n_alleles, 
                   alphabet_type=alphabet_type, alphabet=alphabet,
                   function=function, codon_table=codon_table,
                   stop_function=stop_function)
    
    def _init(self, seq_length=None, n_alleles=None,
              alphabet_type='dna', alphabet=None, function=None,
              codon_table=None, stop_function=-10):
        self.set_seq_length(seq_length, n_alleles, alphabet)
        self.set_alphabet_type(alphabet_type, n_alleles=n_alleles,
                               alphabet=alphabet)

        self.n_states = np.prod(self.n_alleles)
        check_error(self.n_states <= MAX_STATES,
                    msg='Sequence space is too big to handle ({})'.format(self.n_states))
        
        adjacency_matrix = self.calc_adjacency_matrix()
        state_labels = self.get_genotypes()
        
        self.init_space(adjacency_matrix, state_labels=state_labels)
        if function is not None:
            self.set_function(function, codon_table=codon_table,
                              stop_function=stop_function)
    
    @property
    def genotypes(self):
        return(self.state_labels)
    
    def set_function(self, function, codon_table=None, stop_function=-10):
        if codon_table is not None:
            check_error(self.seq_length % 3 == 0, 
                        'Only 3n-long sequences can be translated')
            prot = pd.Series(translate_seqs(self.genotypes, codon_table),
                             index=self.state_labels)
            function = function.reindex(prot).fillna(stop_function).values
            
        self.function = function
        self._check_function()
    
    def set_seq_length(self, seq_length=None, n_alleles=None, alphabet=None):
        if seq_length is None:
            check_error(n_alleles is not None or alphabet is not None,
                        'One of seq_length, n_alleles or alphabet is required')
            seq_length = len(n_alleles) if n_alleles is not None else len(alphabet)
        self.seq_length = seq_length
    
    def _calc_site_adjacency_matrices(self, n_alleles):
        site_I = [sp.identity(a) for a in n_alleles]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            site_Kn = [csr_matrix(np.ones((a, a))) - I
                       for a, I in zip(n_alleles, site_I)]
        
        self.site_I = site_I
        self.site_Kn = site_Kn
    
    def _istack_matrices(self, m_diag, m_offdiag, pos):
        rows = []
        a = self.n_alleles[pos]
        for j in range(a):
            row = [m_offdiag] * j + [m_diag] + [m_offdiag] * (a - j - 1)
            rows.append(sp.hstack(row)) 
        m = sp.vstack(rows)
        return(m)
    
    def _calc_adjacency_matrix(self, m=None, pos=None):
        if pos is None:
            pos = self.seq_length - 1
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if m is None:
                m = self.site_Kn[pos]
    
            if pos == 0:
                return(m)
    
            i = sp.identity(m.shape[0])
            m = self._istack_matrices(m, i, pos-1)
            
        return(self._calc_adjacency_matrix(m, pos-1))
    
    def calc_adjacency_matrix(self):
        self._calc_site_adjacency_matrices(self.n_alleles)
        return(self._calc_adjacency_matrix())
    
    def _check_alphabet(self, n_alleles, alphabet_type, alphabet):
        if alphabet is not None:
            msg = 'n_alleles cannot be specified when the alphabet is provided'
            check_error(n_alleles is None, msg=msg)
            msg = 'alphabet can only be provided for alphabet_type="custom"'
            check_error(alphabet_type == 'custom', msg=msg)
            
        elif alphabet_type == 'custom':
            msg = 'n_alleles must be provided for alphabet_type="custom"'
            check_error(n_alleles is not None, msg=msg)
            
        else:
            msg = 'n_alleles can only be specified for alphabet_type="custom"'
            check_error(n_alleles is None, msg=msg)
        
    def set_alphabet_type(self, alphabet_type, n_alleles=None, alphabet=None):
        self._check_alphabet(n_alleles, alphabet_type, alphabet)
        self.alphabet_type = alphabet_type
        
        if alphabet is not None:
            self.alphabet = alphabet
            
        elif alphabet_type == 'dna':
            self.alphabet = [DNA_ALPHABET] * self.seq_length
            self.complements = {'A': ['T'], 'T': ['A'],
                                'G': ['C'], 'C': ['G']}
            self.ambiguous_values = [DNA_AMBIGUOUS_VALUES] * self.seq_length
            
        elif alphabet_type == 'rna':
            self.alphabet = [RNA_ALPHABET] * self.seq_length
            self.complements = {'A': ['U'], 'U': ['A', 'G'],
                                'G': ['C', 'U'], 'C': ['G']}
            self.ambiguous_values = [RNA_AMBIGUOUS_VALUES] * self.seq_length
            
        elif alphabet_type == 'protein':
            self.ambiguous_values = [PROT_AMBIGUOUS_VALUES] * self.seq_length
            self.alphabet = [PROTEIN_ALPHABET] * self.seq_length
            
        elif alphabet_type == 'custom':
            n_alleles = [n_alleles] * self.seq_length if isinstance(n_alleles, int) else n_alleles
            self.alphabet = [[ALPHABET[x] for x in range(a)] for a in n_alleles]
            self.ambiguous_values = [{'X': ''.join(a)} for a in self.alphabet]
            for i, alleles in enumerate(self.alphabet):
                self.ambiguous_values[i].update(dict(zip(alleles, alleles)))
                
        else:
            alphabet_types = ['dna', 'rna', 'protein', 'custom']
            raise ValueError('alphabet_type can only be: {}'.format(alphabet_types))
        
        if n_alleles is None:
            n_alleles = [len(a) for a in self.alphabet]
        self.n_alleles = n_alleles
    
    def _get_seqs_from_alleles(self, alphabet):
        if not alphabet:
            yield('')
        else:
            for allele in alphabet[0]:
                for seq in self._get_seqs_from_alleles(alphabet[1:]):
                    yield(allele + seq)
    
    def get_genotypes(self):
        return(np.array([x for x in self._get_seqs_from_alleles(self.alphabet)]))
        
    
class CodonSpace(SequenceSpace):
    def __init__(self, allowed_aminoacids, codon_table='Standard',
                 add_variation=False, seed=None):
        if isinstance(allowed_aminoacids, str):
            allowed_aminoacids = np.array([allowed_aminoacids])
        
        if not isinstance(allowed_aminoacids, np.ndarray):
            allowed_aminoacids = np.array(allowed_aminoacids)
        
        function = pd.Series(np.ones(20), index=PROTEIN_ALPHABET)
        function.loc[allowed_aminoacids] = 2
        
        self._init(seq_length=3, alphabet_type='rna',
                   function=function, codon_table=codon_table,
                   stop_function=0)
        if add_variation:
            if seed is not None:
                np.random.seed(seed)
            self.function += 1 / 10 * np.random.normal(size=self.n_states)


def read_sequence_space_csv(fpath, function_col, seq_col=0, sort_seqs=True):
    df = pd.read_csv(fpath, index_col=seq_col)
    if sort_seqs:
        df.sort_index(inplace=True)
    
    function = df[function_col].values
    seqs = df.index.values
    config = guess_space_configuration(seqs)
    space = SequenceSpace(alphabet=config['alphabet'], function=function,
                          alphabet_type='custom')
    return(space)