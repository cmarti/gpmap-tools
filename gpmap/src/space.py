#!/usr/bin/env python
import warnings
import numpy as np
import pandas as pd
import networkx as nx

from itertools import product, combinations
from collections import defaultdict

from scipy.sparse.csr import csr_matrix
from scipy.special import logsumexp

from gpmap.src.seq import (translate_seqs, guess_space_configuration,
                           guess_alphabet_type, get_seqs_from_alleles,
                           get_product_states, hamming_distance)
from gpmap.src.utils import check_error, write_edges
from gpmap.src.matrix import calc_cartesian_product
from gpmap.src.settings import (DNA_ALPHABET, RNA_ALPHABET, PROTEIN_ALPHABET,
                                ALPHABET, MAX_STATES, PROT_AMBIGUOUS_VALUES,
                                DNA_AMBIGUOUS_VALUES, RNA_AMBIGUOUS_VALUES)
from gpmap.src.graph import calc_max_min_path


class DiscreteSpace(object):
    '''
    Class to define an arbitrary discrete space characterized uniquely by the
    connectivity between the different states and optionally by the function
    e.g. fitness or energy at each state of the discrete space
    
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
    
    def format_values(self, values):
        if values.dtype == float:
            v = ['{:.2f}'.format(x) for x in values]
        else:
            v = ['{}'.format(x) for x in values] 
        return(v)
    
    def format_list_ends(self, values, k=3):
        if values.shape[0] > 2*k:
            v1, v2 = self.format_values(values[:k]), self.format_values(values[-k:])
            label = '[{},...,{}]'.format(','.join(v1), ','.join(v2))
        else:
            v = self.format_values(values)
            label = '[{}]'.format(','.join(v))  
        return(label)
    
    def __str__(self):
        s = 'Discrete Space:\n\tNumber of states: {}\n'.format(self.n_states)
        s += '\tState labels: {}\n'.format(self.format_list_ends(self.state_labels))
        if hasattr(self, 'y'):
            s += '\tStates function values: {}\n'.format(self.format_list_ends(self.y))
        else:
            s += '\tStates function values: undefined\n'
        s += '\tNumber of edges: {}'.format(self.n_edges)
        return(s)

    @property
    def n_edges(self):
        return(self.adjacency_matrix.sum().sum())
    
    @property
    def is_regular(self):
        '''
        Attribute characterizing whether the space is regular, this is, every
        state has the same number of neighbors
        '''
        if not hasattr(self, '_is_regular'):
            neighbors = np.unique(self.adjacency_matrix.sum(1))
            self._is_regular = neighbors.shape[0] == 1
        return(self._is_regular)
    
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
    
    def get_y(self, state_labels=None):
        if state_labels is None:
            return(self.y)
        idxs = self.get_state_idxs(state_labels)
        return(self.y[idxs])
    
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
        '''
        Returns the indexes for the provided state labels
        '''
        
        return(self.state_idxs.loc[states])
    
    def get_neighbors(self, states, max_distance=1):
        '''
        Returns the unique state labels corresponding to the d-neighbors of the
        provided states, where the distance is specified by `max_distance`
        
        Parameters
        ----------
        states : array-like of shape (state_number,)
            np.array or list of states from which to select the neighbors
        
        max_distance : int (1)
            The maximal distance at which neighbors from the provided states
            will be returned
            
        Returns
        -------
        neighbor_states : np.array
            Array containing the state labels in the d-neighborhood of `states
        
        '''
        idxs = self.get_state_idxs(states)
        adj_csr = self.adjacency_matrix.tocsr()
        for _ in range(max_distance):
            idxs = np.append(idxs, adj_csr[idxs].indices)
        return(self.state_labels[np.unique(idxs)])
    
    def get_neighbor_pairs(self):
        '''
        Returns a tuple with two arrays of indexes corresponding to the states
        that are connected to each other in the DiscreteSpace
        '''
        
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
    
    def write_edges(self, fpath, triangular=True):
        write_edges(self.adjacency_matrix, fpath, triangular=triangular)
    
    def write_csv(self, fpath):
        df = pd.DataFrame({'y': self.y}, 
                          index=self.state_labels)
        df.to_csv(fpath)


class GeneralSequenceSpace(DiscreteSpace):
    @property
    def n_genotypes(self):
        return(self.n_states)
    
    @property
    def genotypes(self):
        return(self.state_labels)
    
    def set_y(self, X, y, stop_y=None):
        y = pd.Series(y, index=X)
        y = y.reindex(self.genotypes)
        if stop_y is not None:
            idx = np.array(['*' in x for x in self.genotypes])
            y[idx] = stop_y
        y = y.values
        
        if np.any(np.isnan(y)):
            msg = 'Make sure to include all required genotypes'
            raise ValueError(msg.format(self.d, self.X0))
            
        self.y = y
        self._check_y()
    
    def set_seq_length(self, seq_length=None, n_alleles=None, alphabet=None):
        if seq_length is None:
            check_error(n_alleles is not None or alphabet is not None,
                        'One of seq_length, n_alleles or alphabet is required')
            seq_length = len(n_alleles) if n_alleles is not None else len(alphabet)
        self.seq_length = seq_length
        
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
    
    def set_alphabet_type(self, alphabet_type, n_alleles=None, alphabet=None, 
                          add_stop=False):
        if add_stop and alphabet_type != 'protein':
            raise ValueError('add_stop is only valid in protein spaces')
        
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
        
        if add_stop and alphabet_type == 'protein':
            self.alphabet = [a + ['*'] for a in self.alphabet]
        
        if n_alleles is None:
            n_alleles = [len(a) for a in self.alphabet]
        self.n_alleles = n_alleles
    
    def calc_adjacency_matrix(self, genotypes=None):
        if genotypes is None:
            genotypes = self.genotypes
        n_genotypes = genotypes.shape[0]
            
        gts1, gts2 = [], []
        for i, seq1 in enumerate(genotypes):
            for j, seq2 in enumerate(genotypes[i+1:]):
                j += i+1
                if hamming_distance(seq1, seq2) == 1:
                    gts1.extend([i, j]) 
                    gts2.extend([j, i])
        gts1 = np.array(gts1, dtype=int)
        gts2 = np.array(gts2, dtype=int)
        data = np.ones(gts1.shape[0])
        adjacency_matrix = csr_matrix((data, (gts1, gts2)),
                                      shape=(n_genotypes, n_genotypes)) 
        return(adjacency_matrix)
    
    def _get_edges(self, start, end, allow_bypasses, monotonic=False):
        i, j = self.get_neighbor_pairs()
        
        if monotonic:
            df = self.y[j] - self.y[i]
            idxs = df >= 0
            i, j = i[idxs], j[idxs]
        
        states_i, states_j = self.state_labels[i], self.state_labels[j]
        for node1, node2 in zip(states_i, states_j):
            d11 = hamming_distance(node1, start)
            d21 = hamming_distance(node2, start)
            
            d12 = hamming_distance(node1, end)
            d22 = hamming_distance(node2, end)
            
            if allow_bypasses and d21 >= d11 and d22 <= d12:
                yield(node1, node2)
            elif d21 > d11 and d22 < d12:
                yield(node1, node2)
    
    def calc_graph(self, start, end, allow_bypasses, monotonic=False):
        graph = nx.DiGraph()
        graph.add_edges_from(self._get_edges(start, end, allow_bypasses,
                                             monotonic=monotonic))
        nx.set_node_attributes(graph, {node: {'weight': w}
                                       for node, w in zip(self.state_labels,
                                                          self.y)})
        return(graph)
    
    def calc_max_min_path(self, start, end, allow_bypasses=True, monotonic=False):
        graph = self.calc_graph(start, end, allow_bypasses, monotonic)
        path = calc_max_min_path(graph, [start], [end])[0]
        return(path)
    
    def calc_n_paths(self, start, end, allow_bypasses=True, monotonic=False,
                     max_length=None):
        n = 0
        graph = self.calc_graph(start, end, allow_bypasses, monotonic) 
        for _ in nx.all_simple_paths(graph, start, end, cutoff=max_length):
            n += 1
        return(n)
        

class HammingBallSpace(GeneralSequenceSpace):
    '''
    Class for the space representing the Hamming ball around a target sequence
    up to a certain number of mutations from it. 
    
    Parameters
    ----------
    X0: str
        Focal sequence around which to build the Hamming ball space
    
    X: array-like of shape (n_genotypes,)
        Sequences to use as state labels of the discrete sequence space
    
    y: array-like of shape (n_genotypes,)
        Quantitative phenotype or fitness associated to each genotype
    
    d: int (None)
        Maximum distance from the focal sequence to include in the space
    
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
    
    '''
    def __init__(self, X0, X=None, y=None,
                 d=None, n_alleles=None,
                 alphabet_type='dna', alphabet=None):
        
        if X is not None and y is not None:
            config = guess_space_configuration(X, ensure_full_space=False,
                                               force_regular=False,
                                               force_regular_alleles=False)
            alphabet_type = config['alphabet_type']
            alphabet = config['alphabet']
            d = np.max([hamming_distance(X0, x) for x in X])
            y = pd.Series(y, index=X)
        
        self.X0 = X0
        self.d = d
        
        self.set_seq_length(len(X0), n_alleles, alphabet)
        self.set_alphabet_type(alphabet_type, n_alleles=n_alleles,
                               alphabet=alphabet)
        
        genotypes = self.get_genotypes()
        adjacency_matrix = self.calc_adjacency_matrix(genotypes)
        self.init_space(adjacency_matrix, state_labels=genotypes)
        
        if y is not None:
            if X is None:
                X = self.genotypes
            self.set_y(X, y)
    
    @property
    def is_regular(self):
        return(False)
        
    @property
    def n_genotypes(self):
        return(self.n_states)
    
    @property
    def genotypes(self):
        return(self.state_labels)
    
    def get_genotypes(self):
        positions = np.arange(self.seq_length)
        pos_alleles = []
        for j in np.arange(self.seq_length):
            j_alleles = self.alphabet[j][:]
            j_alleles.remove(self.X0[j])
            pos_alleles.append(j_alleles)

        genotypes = [self.X0]
        X0_list = [x for x in self.X0] 
        for i in range(1, self.d + 1):
            for pos in combinations(positions, i):
                pos_alphabet = [pos_alleles[j] for j in pos]
                for alleles in get_seqs_from_alleles(pos_alphabet):
                    X_i = X0_list.copy()
                    for k, a in zip(pos, alleles):
                        X_i[k] = a
                    genotypes.append(''.join(X_i))
                    if len(genotypes) > MAX_STATES:
                        raise ValueError('Sequence space too big')
        genotypes = np.array(genotypes)
        return(genotypes)
    
    
class ProductSpace(DiscreteSpace):
    '''
    General class for spaces that can be built as cartesian products
    of smaller subspaces characterized by a set of elementary graphs
    
    Parameters
    ----------
    elementary_graphs: csr_matrices
        List csr_matrices for the adjacency matrices from which to build the
        product space
    
    y: None or array-like of shape (n,)
        np.array containing the phenotypic values associated to each combination
        of states in the resulting space. If `y=None`, no phenotypic values will
        be stored
        
    state_labels: None or list
        List with the labels associated to each of the possible states
        `a` in each of the `l` elements of the product space. If 
        `state_labels=None`, numeric labels will be given by default.
    
    '''
    def __init__(self, elementary_graphs, y=None, state_labels=None):
        self.set_dim_sizes(elementary_graphs)
        adjacency_matrix = self.calc_adjacency_matrix(elementary_graphs)
        
        self.states = self.calc_states(state_labels=state_labels)
        state_labels = np.array(['-'.join([str(x) for x in seq])
                                 for seq in self.states])
         
        self.init_space(adjacency_matrix, y=y, state_labels=state_labels)
    
    def set_dim_sizes(self, elementary_graphs):
        self.graph_sizes = [adj_m.shape[0] for adj_m in elementary_graphs]

    def calc_states(self, state_labels=None):
        if state_labels is None:
            state_labels = [list(range(s)) for s in self.graph_sizes]
            
        state_labels = np.array([x for x in get_product_states(state_labels)])
        return(state_labels)
    
    def calc_adjacency_matrix(self, elementary_graphs):
        adjacency_matrix = calc_cartesian_product(elementary_graphs)
        return(adjacency_matrix)


class GridSpace(ProductSpace):
    """
    Class for creating an N-dimensional grid discrete space
    
    Parameters
    ----------
    length: int or array-like
        Number of states across each dimension of the grid. If an integer is 
        provided, all dimensions of the grid will have the same length. If 
        a series of lengths is provided, they will be used to form a grid of 
        dimensions with the specified lengths and the `ndim` argument will be
        ignored
    
    ndim: int
        Number of dimensions in the grid with a single `length` value.
        
    y: array-like of shape (length ** ndim,) or None
        Phenotypic values associated to each possible state
    
    """
    def __init__(self, length, y=None, ndim=2):
        self.length = length
        self.ndim = ndim
        
        if np.array(length).shape[0] > 1:
            elementary_graphs = [self.calc_elementary_graph(l) for l in length]
        else:
            elementary_graphs = [self.calc_elementary_graph(length)] * ndim
        super().__init__(elementary_graphs, y=y)
    
    def calc_elementary_graph(self, length):
        states = np.arange(length)
        i = np.append(states[:-1], states[1:])
        j = np.append(states[1:], states[:-1])
        data = np.ones(i.shape[0])
        m = csr_matrix((data, (i, j)))
        return(m)
    
    def set_peaks(self, positions, sigma=1):
        distances = np.array([np.abs(self.states - pos).sum(1) for pos in positions]).T
        y = np.exp(logsumexp(-distances / sigma, axis=1))
        self.set_y(y)
    
    @property
    def nodes_df(self):
        nodes_df = pd.DataFrame(self.states,
                                columns=[str(i+1) for i in range(self.ndim)],
                                index=self.state_labels)
        if hasattr(self, 'y'):
            nodes_df['function'] = self.y
        return(nodes_df)
        

class SequenceSpace(GeneralSequenceSpace, ProductSpace):
    """
    Class for creating a Sequence space characterized by having sequences as
    states. States are connected in the discrete space if they differ by a 
    single position in the sequence. It can be created in two different ways:
    
        - From a set of sequences and function values X, y
        - By specifying the properties of the sequence space (alphabet,
          sequence length, number of alleles per site and type of alphabet).
    
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
    
    stop_y: float (None)
        Value of the function given for protein sequence with an
        in-frame stop codon. If given, it will increase the protein 
        alphabet to incorporate `*` for stops
    
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
    
    """
    def __init__(self, X=None, y=None, seq_length=None, n_alleles=None,
                 alphabet_type='dna', alphabet=None, stop_y=None):
        
        self._init(X=X, y=y, seq_length=seq_length, n_alleles=n_alleles, 
                   alphabet_type=alphabet_type, alphabet=alphabet,
                   stop_y=stop_y)
    
    def __str__(self):
        s = 'Sequence Space:\n'
        s += '\tType: {}\n'.format(self.alphabet_type)
        s += '\tSequence length: {}\n'.format(self.seq_length)
        s += '\tNumber of alleles per site: {}\n'.format(self.n_alleles)
        s += '\tGenotypes: {}\n'.format(self.format_list_ends(self.genotypes))
        if hasattr(self, 'y'):
            s += '\tFunction y: {}'.format(self.format_list_ends(self.y))
        else:
            s += '\tFunction y: undefined'
        return(s)
    
    def _init(self, X=None, y=None,
              seq_length=None, n_alleles=None,
              alphabet_type='dna', alphabet=None, stop_y=None):
        
        if X is not None and y is not None:
            config = guess_space_configuration(X, ensure_full_space=True)
            seq_length = config['length']
            alphabet_type = config['alphabet_type']
            alphabet = config['alphabet']
            y = pd.Series(y, index=X)
        
        self.set_seq_length(seq_length, n_alleles, alphabet)
        self.set_alphabet_type(alphabet_type, n_alleles=n_alleles,
                               alphabet=alphabet, add_stop=stop_y is not None)
        self.n_states = np.prod(self.n_alleles)
        
        msg='Sequence space is too big to handle ({})'.format(self.n_states)
        check_error(self.n_states <= MAX_STATES, msg=msg)
        
        adjacency_matrix = self.calc_adjacency_matrix()
        state_labels = self.get_genotypes()
        self.init_space(adjacency_matrix, state_labels=state_labels)
        
        if y is not None:
            if X is None:
                X = self.genotypes
            self.set_y(X, y, stop_y=stop_y)
    
    @property
    def is_regular(self):
        return(np.unique(self.n_alleles).shape[0] == 1)
    
    def get_single_mutant_matrix(self, sequence, center=False):
        '''
        Returns the effects of single point mutations from a focal sequences
        
        Parameters
        ----------
        sequence: str
            String encoding the sequence from which to report all single point
            mutant effects
        
        center: bool (False)
            If True, results will be centered by position, so that the mean
            of allelic effects is 0. If False, the focal sequence will have 0
            and values would represent mutational effects from it
        
        Returns
        -------
        output: pd.DataFrame of shape (seq_length, total_alleles)
            pd.DataFrame containin the mutational or allelic effects
            for each allele across all sequence positions 
        '''
        seqy = self.get_y([sequence])
        data = [] 
        for i in range(self.seq_length):
            alleles = self.alphabet[i]
            mutants = [sequence[:i] + a + sequence[i+1:] for a in alleles]
            dy = self.get_y(mutants) - seqy
            if center:
                dy = dy - dy.mean()
            data.append(dict(zip(alleles, dy)))
        data = pd.DataFrame(data)
        return(data)
        
    def to_nucleotide_space(self, codon_table='Standard', alphabet_type='dna'):
        '''
        Transforms a protein space into a nucleotide space using a codon table
        for translating the sequence
        
        Parameters
        ----------
        codon_table: str or Bio.Data.CodonTable
            NCBI code for an existing genetic code or a custom CodonTable 
            object to translate nucleotide sequences into protein
        
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
            
        
        
        nc_space = SequenceSpace(seq_length=3 * self.seq_length,
                                 alphabet_type=alphabet_type)
        prot = pd.Series(translate_seqs(nc_space.genotypes, codon_table),
                         index=nc_space.genotypes)
        nc_space.protein_seqs = prot.values
        y = pd.Series(self.y, index=self.genotypes)
        y = y.reindex(prot).values
        
        if np.any(np.isnan(y)):
            msg = 'Make sure to include all protein sequences including stops'
            raise ValueError(msg)
        
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
    
    def get_genotypes(self):
        return(np.array([x for x in get_seqs_from_alleles(self.alphabet)]))
        
    
def CodonSpace(allowed_aminoacids, codon_table='Standard',
               add_variation=False, seed=None):
    
    if isinstance(allowed_aminoacids, str):
        allowed_aminoacids = np.array([allowed_aminoacids])
    
    if not isinstance(allowed_aminoacids, np.ndarray):
        allowed_aminoacids = np.array(allowed_aminoacids)
    
    y = pd.Series(np.ones(20), index=PROTEIN_ALPHABET)
    y.loc[allowed_aminoacids] = 2
    
    prot_space = SequenceSpace(seq_length=1, alphabet_type='protein', y=y,
                               stop_y=0)
    nuc_space = prot_space.to_nucleotide_space(codon_table=codon_table)
    
    if add_variation:
        if seed is not None:
            np.random.seed(seed)
        nuc_space.y += 1 / 10 * np.random.normal(size=nuc_space.n_genotypes)
    return(nuc_space)
