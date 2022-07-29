#!/usr/bin/env python
import warnings
import numpy as np
import pandas as pd

from itertools import product
from _collections import defaultdict

from scipy.sparse.csr import csr_matrix
from scipy.sparse._matrix_io import save_npz
from scipy.sparse.extract import triu

from gpmap.src.seq import (translate_seqs, guess_space_configuration,
                           guess_alphabet_type)
from gpmap.src.utils import (get_sparse_diag_matrix, check_error,
                             calc_cartesian_product)
from gpmap.src.settings import (DNA_ALPHABET, RNA_ALPHABET, PROTEIN_ALPHABET,
                                ALPHABET, MAX_STATES, PROT_AMBIGUOUS_VALUES,
                                DNA_AMBIGUOUS_VALUES, RNA_AMBIGUOUS_VALUES)


class DiscreteSpace(object):
    '''
    Parameters
    ----------
    adjacency_matrix: scipy.sparse.csr_matrix of shape (n_states, n_states)
        Sparse matrix representing the adjacency relationships between
        states. The ij'th entry contains a 1 if the states `i` and `j`
        are connected and 0 otherwise
    
    y: array-like of shape (n_states,)
        Quantitative property associated to each state
    
    state_labels: array-like of shape (n_genotypes, )
        State labels in the sequence space
    
    Attributes
    ----------
    n_states: int
        Number of states in the discrete space
    
    state_labels: array-like of shape (n_genotypes, )
        State labels in the sequence space
    
    state_idxs: pd.Series of shape (n_genotypes, )
        pd.Series containing the index of each state. It has state_labels as
        index of the Series and can be used to quickly extract the index
        corresponding to a set of state labels
    
    is_regular: bool
        Boolean variable storing whether the resulting graph is regular
        or not, this is, whether each node has the same number of neighbors
        
    '''
    
    def __init__(self, adjacency_matrix, y=None, state_labels=None):
        self.init_space(adjacency_matrix, y=y, state_labels=state_labels)
    
    @property
    def is_regular(self):
        neighbors = np.unique(self.adjacency_matrix.sum(1))
        return(neighbors.shape[0] == 1)
    
    def calc_laplacian(self):
        D = get_sparse_diag_matrix(self.adjacency_matrix.sum(1).A1.flatten())
        self.laplacian = D - self.adjacency_matrix
    
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
    
    def _check_y(self):
        check_error(self.y.shape[0] == self.state_labels.shape[0],
                    msg='Size of adjacency_matrix and y does not match')
    
    def set_y(self, y):
        self.y = y
        self._check_y()
    
    def init_space(self, adjacency_matrix, y=None, state_labels=None):
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
        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            self.set_y(y)
    
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
    
    def write_edges_npz(self, fpath, triangular=True):
        if triangular:
            save_npz(fpath, triu(self.adjacency_matrix))
        else:
            save_npz(fpath, self.adjacency_matrix)
    
    def write_edges_csv(self, fpath):
        edges_df = self.get_edges_df()
        edges_df.to_csv(fpath, index=False)
    
    def write_csv(self, fpath):
        df = pd.DataFrame({'y': self.y}, 
                          index=self.state_labels)
        df.to_csv(fpath)
    
    
class SequenceSpace(DiscreteSpace):
    '''
    Class for creating a Sequence space characterized by having sequences as
    states. States are connected in the discrete space if they differ by a 
    single position in the sequence. It can be created in two different ways:
    
        - From a set of sequences and function values X, y
        - By specifying the properties of the sequence space (alphabet,
          sequence length, number of alleles per site and type of alphabet)
    
    Parameters
    ----------
    X: array-like of shape (n_genotypes,)
        Sequences to use as state labels of the discrete sequence space
    
    y: array-like of shape (n_genotypes,)
        Quantitative phenotype or fitness associated to each genotype
    
    seq_length: int (None)
        Length of the sequences in the sequence space. If not given, it will be
        guessed from `alphabet` or `n_alleles`
    
    n_alleles: list of size `seq_length` (None)
        List containing the number of alleles present in each of the sites
        of the sequence space. It can only be specified for 
        `alphabet_type=custom`
    
    alphabet_type: str ('dna')
        Sequence type: {'dna', 'rna', 'protein', 'custom'}
        
    alphabet: list of `seq_length' lists
        Every element of the list is itself a list containing the different
        alleles allowed in each site. Note that the number and type of alleles
        can be different for every site.
    
    Attributes
    ----------
    n_genotypes: int
        Number of states in the complete sequence space
    
    genotypes: array-like of shape (n_genotypes, )
        Genotype labels in the sequence space
        
    adjacency_matrix: scipy.sparse.csr_matrix of shape (n_genotypes, n_genotypes)
        Sparse matrix representing the adjacency relationships between
        genotypes. The ij'th entry contains a 1 if the genotypes `i` and `j`
        are separated by a single mutation and 0 otherwise
    
    y: array-like of shape (n_genotypes,)
        Quantitative phenotype or fitness associated to each genotype
        
    is_regular: bool
        Boolean variable storing whether the resulting Hamming graph is regular
        or not. In other words, whether every site has the same number of
        alleles
        
    Methods
    -------
    
    
    '''
    def __init__(self, X=None, y=None, seq_length=None, n_alleles=None,
                 alphabet_type='dna', alphabet=None):
        
        self._init(X=X, y=y, seq_length=seq_length, n_alleles=n_alleles, 
                   alphabet_type=alphabet_type, alphabet=alphabet)
    
    def _init(self, X=None, y=None,
              seq_length=None, n_alleles=None,
              alphabet_type='dna', alphabet=None):
        
        if X is not None and y is not None:
            config = guess_space_configuration(X, ensure_full_space=True)
            seq_length = config['length']
            alphabet_type = config['alphabet_type']
            alphabet = config['alphabet']
            y = pd.Series(y, index=X)
        
        self.set_seq_length(seq_length, n_alleles, alphabet)
        self.set_alphabet_type(alphabet_type, n_alleles=n_alleles,
                               alphabet=alphabet)
        self.n_states = np.prod(self.n_alleles)
        
        msg='Sequence space is too big to handle ({})'.format(self.n_states)
        check_error(self.n_states <= MAX_STATES, msg=msg)
        
        adjacency_matrix = self.calc_adjacency_matrix()
        state_labels = self.get_genotypes()
        self.init_space(adjacency_matrix, state_labels=state_labels)
        
        if y is not None:
            if X is None:
                X = self.genotypes
            self.set_y(X, y)
    
    @property
    def is_regular(self):
        return(np.unique(self.n_alleles).shape[0] == 1)
    
    @property
    def n_genotypes(self):
        return(self.n_states)
    
    @property
    def genotypes(self):
        return(self.state_labels)
    
    def set_y(self, X, y):
        y = pd.Series(y, index=X)
        y = y.reindex(self.genotypes).values
            
        self.y = y
        self._check_y()
        
    def to_nucleotide_space(self, codon_table='Standard', stop_y=None,
                            alphabet_type='dna'):
        '''
        Transforms a protein space into a nucleotide space using a codon table
        for translating the sequence
        
        Parameters
        ----------
        codon_table: str or Bio.Data.CodonTable
            NCBI code for an existing genetic code or a custom CodonTable 
            object to translate nucleotide sequences into protein
        
        stop_y: float (None)
            Value of the function given for every nucleotide sequence with an
            in-frame stop codon. If 'None', it will use the minimum
            value found across of the sequences, assumed to be equal to a 
            complete loss of function
        
        alphabet_type: str ('dna')
            Sequence type to use in the resulting nucleotide space
            It can only take one of the following values {'dna', 'rna'}
        
        Returns
        -------
        SequenceSpace
            Nucleotide sequence space with 4 alleles per site and 3 times 
            the number of sites of the current space
        '''
        
        
        msg = 'Only protein spaces can be transformed to nucleotide space'
        msg += ' through a codon model: {} not allowed'.format(self.alphabet_type)
        check_error(self.alphabet_type == 'protein', msg)
        
        msg = '`alphabet_type` must be one of ["dna", "rna"]'
        check_error(alphabet_type in ['dna', 'rna'], msg)
            
        
        
        nc_space = SequenceSpace(seq_length=3*self.seq_length,
                                 alphabet_type=alphabet_type)
        prot = pd.Series(translate_seqs(nc_space.genotypes, codon_table),
                         index=nc_space.genotypes)

        if stop_y is None:
            stop_y = self.y.min()
        y = pd.Series(self.y, index=self.genotypes)
        y = y.reindex(prot).fillna(stop_y).values
        nc_space.set_y(nc_space.genotypes, y)
        return(nc_space)
    
    def remove_codon_incompatible_transitions(self, codon_table='Standard'):
        '''
        Recalculates the adjacency matrix of the discrete space to only allow
        transitions that are compatible with the specified codon table
        
        Parameters
        ----------
        codon_table: str or Bio.Data.CodonTable
            NCBI code for an existing genetic code or a custom CodonTable 
            object to translate nucleotide sequences into protein

        '''
        msg = 'alphabet must be at least a subset of the protein alphabet'
        check_error(guess_alphabet_type(self.alphabet) == 'protein', msg)
        self.adjacency_matrix = self.calc_adjacency_matrix(codon_table=codon_table)
        
    def calc_transitions(self, codon_table):
        seqs = [''.join(x) for x in product(DNA_ALPHABET, repeat=3)]
        
        transitions = defaultdict(lambda: defaultdict(lambda: 0))        
        for codon1, codon2 in product(seqs, repeat=2):
            d = np.sum([x != y for x, y in zip(codon1, codon2)])
            if d != 1:
                continue
            aa1, aa2 = translate_seqs([codon1, codon2], codon_table)
            transitions[aa1][aa2] += 1
        transitions = pd.DataFrame(transitions).fillna(0).astype(int)
        transitions = transitions.loc[PROTEIN_ALPHABET, PROTEIN_ALPHABET]
        return(transitions)
    
    def set_seq_length(self, seq_length=None, n_alleles=None, alphabet=None):
        if seq_length is None:
            check_error(n_alleles is not None or alphabet is not None,
                        'One of seq_length, n_alleles or alphabet is required')
            seq_length = len(n_alleles) if n_alleles is not None else len(alphabet)
        self.seq_length = seq_length
    
    def _calc_site_matrix(self, alleles, transitions=None):
        n_alleles = len(alleles)
        if transitions is None:
            m = np.ones((n_alleles, n_alleles))
        else:
            m = transitions.loc[alleles, alleles].values
        np.fill_diagonal(m, np.zeros(n_alleles))
        return(csr_matrix(m))
    
    def _calc_site_adjacency_matrices(self, alleles, codon_table=None):
        if codon_table is None:
            transitions = None
        else:
            transitions = self.calc_transitions(codon_table=codon_table)
            # TODO: fix this to generalize to having multiple ways of going
            # from one aminoacid to another
            transitions = (transitions > 0).astype(int)
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            site_Kn = [self._calc_site_matrix(a, transitions) for a in alleles]
        return(site_Kn)
    
    def calc_adjacency_matrix(self, codon_table=None):
        if self.alphabet_type not in ['protein', 'custom']:
            codon_table = None 
        sites_A = self._calc_site_adjacency_matrices(self.alphabet, codon_table=codon_table)
        adjacency_matrix = calc_cartesian_product(sites_A)
        return(adjacency_matrix)
    
    def _check_alphabet(self, n_alleles, alphabet_type, alphabet):
        if alphabet is not None:
            msg = 'n_alleles cannot be specified when the alphabet is provided'
            check_error(n_alleles is None, msg=msg)
            
            if alphabet_type != 'custom':
                atype = guess_alphabet_type(alphabet)
                msg = 'The provided alphabet is not compatible with the'
                msg += ' alphabet_type {}'.format(alphabet_type)
                check_error(alphabet_type == atype, msg=msg)
            
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
        
    
def CodonSpace(allowed_aminoacids, codon_table='Standard',
               add_variation=False, seed=None):
    
    if isinstance(allowed_aminoacids, str):
        allowed_aminoacids = np.array([allowed_aminoacids])
    
    if not isinstance(allowed_aminoacids, np.ndarray):
        allowed_aminoacids = np.array(allowed_aminoacids)
    
    y = pd.Series(np.ones(20), index=PROTEIN_ALPHABET)
    y.loc[allowed_aminoacids] = 2
    
    prot_space = SequenceSpace(seq_length=1, alphabet_type='protein', y=y)
    nuc_space = prot_space.to_nucleotide_space(codon_table=codon_table, stop_y=0)
    
    if add_variation:
        if seed is not None:
            np.random.seed(seed)
        nuc_space.y += 1 / 10 * np.random.normal(size=nuc_space.n_genotypes)
    return(nuc_space)


def read_sequence_space_csv(fpath, y_col, seq_col=0, sort_seqs=True):
    df = pd.read_csv(fpath, index_col=seq_col)
    if sort_seqs:
        df.sort_index(inplace=True)
    
    y = df[y_col].values
    seqs = df.index.values
    config = guess_space_configuration(seqs, ensure_full_space=True)
    space = SequenceSpace(alphabet=config['alphabet'], y=y,
                          alphabet_type='custom')
    return(space)
