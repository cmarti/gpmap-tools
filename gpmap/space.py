#!/usr/bin/env python
import warnings
import numpy as np
import pandas as pd
import networkx as nx

from itertools import product, combinations
from collections import defaultdict

from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from gpmap.seq import (
    translate_seqs,
    guess_space_configuration,
    guess_alphabet_type,
    get_seqs_from_alleles,
    get_product_states,
    hamming_distance,
)
from gpmap.utils import check_error, write_edges
from gpmap.matrix import calc_cartesian_product
from gpmap.settings import (
    DNA_ALPHABET,
    RNA_ALPHABET,
    PROTEIN_ALPHABET,
    ALPHABET,
    MAX_STATES,
    PROT_AMBIGUOUS_VALUES,
    DNA_AMBIGUOUS_VALUES,
    RNA_AMBIGUOUS_VALUES,
)
from gpmap.graph import calc_max_min_path


class DiscreteSpace(object):
    """
    Class to define an arbitrary discrete space characterized by the
    connectivity between different states and optionally by a scalar value
    (e.g. fitness or energy) associated with each state.

    Parameters
    ----------
    adjacency_matrix : scipy.sparse.csr_matrix of shape (n_states, n_states)
        Sparse matrix representing the adjacency relationships between
        states. The (i, j) entry contains a 1 if states `i` and `j`
        are connected, and 0 otherwise.

    y : array-like of shape (n_states,), optional
        Function value associated with each state.

    state_labels : array-like of shape (n_states,), optional
        Labels for the states in the discrete space.

    Attributes
    ----------
    n_states : int
        Number of states in the discrete space.

    state_labels : array-like of shape (n_states,)
        Labels for the states in the discrete space.

    state_idxs : pd.Series of shape (n_states,)
        A pandas Series mapping state labels to their corresponding indices.
        The index of the Series is `state_labels`, allowing quick lookup
        of indices for a given set of state labels.

    is_regular : bool
        Indicates whether the graph is regular, i.e., whether each node
        has the same number of neighbors.
    """

    def __init__(self, adjacency_matrix, y=None, state_labels=None):
        self.init_space(adjacency_matrix, y=y, state_labels=state_labels)

    def format_values(self, values):
        if values.dtype == float:
            v = ["{:.2f}".format(x) for x in values]
        else:
            v = ["{}".format(x) for x in values]
        return v

    def format_list_ends(self, values, k=3):
        if values.shape[0] > 2 * k:
            v1, v2 = (
                self.format_values(values[:k]),
                self.format_values(values[-k:]),
            )
            label = "[{},...,{}]".format(",".join(v1), ",".join(v2))
        else:
            v = self.format_values(values)
            label = "[{}]".format(",".join(v))
        return label

    def __str__(self):
        s = "Discrete Space:\n\tNumber of states: {}\n".format(self.n_states)
        s += "\tState labels: {}\n".format(
            self.format_list_ends(self.state_labels)
        )
        if hasattr(self, "y"):
            s += "\tStates function values: {}\n".format(
                self.format_list_ends(self.y)
            )
        else:
            s += "\tStates function values: undefined\n"
        s += "\tNumber of edges: {}".format(self.n_edges)
        return s

    @property
    def n_edges(self):
        return self.adjacency_matrix.sum().sum()

    @property
    def is_regular(self):
        """
        Attribute characterizing whether the space is regular, this is, every
        state has the same number of neighbors
        """
        if not hasattr(self, "_is_regular"):
            neighbors = np.unique(self.adjacency_matrix.sum(1))
            self._is_regular = neighbors.shape[0] == 1
        return self._is_regular

    def _check_attributes(self, tol=1e-6):
        # TODO: check that the space is connected
        check_error(
            len(self.adjacency_matrix.shape) == 2,
            msg="Ensure adjacency_matrix is a 2D array",
        )
        check_error(
            self.adjacency_matrix.shape[0] == self.adjacency_matrix.shape[1],
            msg="adjacencty_matrix should be square",
        )
        check_error(
            np.all(self.adjacency_matrix.diagonal() == 0),
            msg="loops are not allowed in the discrete space",
        )
        check_error(
            (abs(self.adjacency_matrix - self.adjacency_matrix.T) > tol).nnz
            == 0,
            msg="Adjacency matrix has to be symmetric",
        )
        check_error(
            (self.adjacency_matrix.data >= 0).all(),
            msg="Adjacency matrix entries have to be positive",
        )
        check_error(
            self.adjacency_matrix.shape[0] == self.state_labels.shape[0],
            msg="Size of adjacency_matrix and state_labels does not match",
        )

    def _check_y(self):
        check_error(
            self.y.shape[0] == self.state_labels.shape[0],
            msg="Size of adjacency_matrix and y does not match",
        )

    def set_y(self, y):
        self.y = y
        self._check_y()

    def get_y(self, state_labels=None):
        if state_labels is None:
            return self.y
        idxs = self.get_state_idxs(state_labels)
        return self.y[idxs]

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
        """
        Get the indexes for the provided state labels.

        Parameters
        ----------
        states : array-like
            A list or array of state labels for which the indexes are to be retrieved.

        Returns
        -------
        pandas.Series
            A pandas Series containing the indexes corresponding to the provided
            state labels.
        """

        return self.state_idxs.loc[states]

    def get_neighbors(self, states, max_distance=1):
        """
        Retrieve the unique state labels corresponding to the neighbors of the
        provided states within a specified maximum distance.

        Parameters
        ----------
        states : array-like of shape (state_number,)
            A list or numpy array of state labels from which to find neighbors.

        max_distance : int, optional, default=1
            The maximum distance within which neighbors of the provided states
            will be included.

        Returns
        -------
        neighbor_states : np.array
            An array containing the state labels of all unique neighbors within
            the specified distance from the input states.
        """
        idxs = self.get_state_idxs(states)
        adj_csr = self.adjacency_matrix.tocsr()
        for _ in range(max_distance):
            idxs = np.append(idxs, adj_csr[idxs].indices)
        return self.state_labels[np.unique(idxs)]

    def get_neighbor_pairs(self):
        """
        Retrieve pairs of indices representing connected states in the DiscreteSpace.

        Returns
        -------
        tuple of np.ndarray
            Two arrays of indices, where the first array contains the source
            indices and the second array contains the target indices of the
            connections.
        """

        if not hasattr(self, "neighbor_pairs"):
            A = self.adjacency_matrix
            try:
                self.neighbor_pairs = A.row, A.col
            except AttributeError:
                A = A.tocoo()
                self.neighbor_pairs = A.row, A.col
        return self.neighbor_pairs

    def get_edges_df(self):
        """
        Generate a DataFrame representing the edges of the adjacency graph.

        This method retrieves pairs of neighboring nodes from the adjacency matrix
        and constructs a DataFrame where each row represents an edge between two nodes.

        Returns
        -------
        edges_df : pd.DataFrame
            A DataFrame with two columns:
            - 'i': The source node of the edge.
            - 'j': The target node of the edge.
        """
        i, j = self.get_neighbor_pairs()
        edges_df = pd.DataFrame({"i": i, "j": j})
        return edges_df

    def write_edges(self, fpath, triangular=True):
        write_edges(self.adjacency_matrix, fpath, triangular=triangular)

    def write_csv(self, fpath):
        df = pd.DataFrame({"y": self.y}, index=self.state_labels)
        df.to_csv(fpath)


class GeneralSequenceSpace(DiscreteSpace):
    @property
    def n_genotypes(self):
        return self.n_states

    @property
    def genotypes(self):
        return self.state_labels

    def set_y(self, X, y, stop_y=None):
        y = pd.Series(y, index=X)
        y = y.reindex(self.genotypes)
        if stop_y is not None:
            idx = np.array(["*" in x for x in self.genotypes])
            y[idx] = stop_y
        y = y.values

        if np.any(np.isnan(y)):
            msg = "Make sure to include all required genotypes"
            raise ValueError(msg.format(self.d, self.X0))

        self.y = y
        self._check_y()

    def set_seq_length(self, seq_length=None, n_alleles=None, alphabet=None):
        if seq_length is None:
            check_error(
                n_alleles is not None or alphabet is not None,
                "One of n_alleles, seq_length or alphabet is required",
            )
            seq_length = (
                len(n_alleles) if n_alleles is not None else len(alphabet)
            )
        self.seq_length = seq_length

    def _check_alphabet(self, n_alleles, alphabet_type, alphabet):
        if alphabet is not None:
            msg = "n_alleles cannot be specified when the alphabet is provided"
            check_error(n_alleles is None, msg=msg)

            if alphabet_type != "custom":
                atype = guess_alphabet_type(alphabet)
                msg = "The provided alphabet is not compatible with the"
                msg += " alphabet_type {}".format(alphabet_type)
                check_error(alphabet_type == atype, msg=msg)

        elif alphabet_type == "custom":
            msg = 'n_alleles must be provided for alphabet_type="custom"'
            check_error(n_alleles is not None, msg=msg)

        else:
            msg = 'n_alleles can only be specified for alphabet_type="custom"'
            check_error(n_alleles is None, msg=msg)

    def set_alphabet_type(
        self, alphabet_type, n_alleles=None, alphabet=None, add_stop=False
    ):
        if add_stop and alphabet_type != "protein":
            raise ValueError("add_stop is only valid in protein spaces")

        self._check_alphabet(n_alleles, alphabet_type, alphabet)
        self.alphabet_type = alphabet_type

        if alphabet is not None:
            self.alphabet = alphabet

        elif alphabet_type == "dna":
            self.alphabet = [DNA_ALPHABET] * self.seq_length
            self.complements = {"A": ["T"], "T": ["A"], "G": ["C"], "C": ["G"]}
            self.ambiguous_values = [DNA_AMBIGUOUS_VALUES] * self.seq_length

        elif alphabet_type == "rna":
            self.alphabet = [RNA_ALPHABET] * self.seq_length
            self.complements = {
                "A": ["U"],
                "U": ["A", "G"],
                "G": ["C", "U"],
                "C": ["G"],
            }
            self.ambiguous_values = [RNA_AMBIGUOUS_VALUES] * self.seq_length

        elif alphabet_type == "protein":
            self.ambiguous_values = [PROT_AMBIGUOUS_VALUES] * self.seq_length
            self.alphabet = [PROTEIN_ALPHABET] * self.seq_length

        elif alphabet_type == "custom":
            n_alleles = (
                [n_alleles] * self.seq_length
                if isinstance(n_alleles, int)
                else n_alleles
            )
            self.alphabet = [[ALPHABET[x] for x in range(a)] for a in n_alleles]
            self.ambiguous_values = [{"X": "".join(a)} for a in self.alphabet]
            for i, alleles in enumerate(self.alphabet):
                self.ambiguous_values[i].update(dict(zip(alleles, alleles)))

        else:
            alphabet_types = ["dna", "rna", "protein", "custom"]
            raise ValueError(
                "alphabet_type can only be: {}".format(alphabet_types)
            )

        if add_stop and alphabet_type == "protein":
            self.alphabet = [a + ["*"] for a in self.alphabet]

        if n_alleles is None:
            n_alleles = [len(a) for a in self.alphabet]
        self.n_alleles = n_alleles

    def calc_adjacency_matrix(self, genotypes=None):
        if genotypes is None:
            genotypes = self.genotypes
        n_genotypes = genotypes.shape[0]

        gts1, gts2 = [], []
        for i, seq1 in enumerate(genotypes):
            for j, seq2 in enumerate(genotypes[i + 1 :]):
                j += i + 1
                if hamming_distance(seq1, seq2) == 1:
                    gts1.extend([i, j])
                    gts2.extend([j, i])
        gts1 = np.array(gts1, dtype=int)
        gts2 = np.array(gts2, dtype=int)
        data = np.ones(gts1.shape[0])
        adjacency_matrix = csr_matrix(
            (data, (gts1, gts2)), shape=(n_genotypes, n_genotypes)
        )
        return adjacency_matrix

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
                yield (node1, node2)
            elif d21 > d11 and d22 < d12:
                yield (node1, node2)

    def calc_graph(self, start, end, allow_bypasses, monotonic=False):
        graph = nx.DiGraph()
        graph.add_edges_from(
            self._get_edges(start, end, allow_bypasses, monotonic=monotonic)
        )
        nx.set_node_attributes(
            graph,
            {node: {"weight": w} for node, w in zip(self.state_labels, self.y)},
        )
        return graph

    def calc_max_min_path(
        self, start, end, allow_bypasses=True, monotonic=False
    ):
        graph = self.calc_graph(start, end, allow_bypasses, monotonic)
        path = calc_max_min_path(graph, [start], [end])[0]
        return path

    def calc_n_paths(
        self, start, end, allow_bypasses=True, monotonic=False, max_length=None
    ):
        n = 0
        graph = self.calc_graph(start, end, allow_bypasses, monotonic)
        for _ in nx.all_simple_paths(graph, start, end, cutoff=max_length):
            n += 1
        return n


class HammingBallSpace(GeneralSequenceSpace):
    """
    Discrete space representing a Hamming ball space around a target
    sequence, including all sequences within a specified maximum Hamming distance.

    Parameters
    ----------
    X0 : str
        The focal sequence around which the Hamming ball space is constructed.

    X : array-like of shape (n_genotypes,), optional
        Sequences to use as state labels for the discrete sequence space.

    y : array-like of shape (n_genotypes,), optional
        Quantitative phenotype or fitness values associated with each genotype.

    d : int, optional
        The maximum Hamming distance from the focal sequence to include in the space.

    n_alleles : list of int, optional
        A list specifying the number of alleles present at each site of the sequence.
        This can only be specified for `alphabet_type='custom'`.

    alphabet_type : str, default='dna'
        The type of sequence. Options are {'dna', 'rna', 'protein', 'custom'}.

    alphabet : list of lists, optional
        A list where each element is itself a list containing the different alleles
        allowed at each site. The number and type of alleles can vary across sites.

    Attributes
    ----------
    n_genotypes : int
        The total number of states (genotypes) in the Hamming ball space.

    genotypes : array-like of shape (n_genotypes,)
        The genotype labels in the Hamming ball space.

    adjacency_matrix : scipy.sparse.csr_matrix of shape (n_genotypes, n_genotypes)
        A sparse matrix representing the adjacency relationships between genotypes.
        The (i, j) entry contains a 1 if genotypes `i` and `j` differ by a single
        mutation, and 0 otherwise.

    y : array-like of shape (n_genotypes,), optional
        Quantitative phenotype or fitness values associated with each genotype.

    is_regular : bool
        Indicates whether the resulting Hamming graph is regular, i.e., whether
        every site has the same number of alleles.
    """

    def __init__(
        self,
        X0,
        X=None,
        y=None,
        d=None,
        n_alleles=None,
        alphabet_type="dna",
        alphabet=None,
    ):
        if X is not None and y is not None:
            config = guess_space_configuration(
                X,
                ensure_full_space=False,
                force_regular=False,
                force_regular_alleles=False,
            )
            alphabet_type = config["alphabet_type"]
            alphabet = config["alphabet"]
            d = np.max([hamming_distance(X0, x) for x in X])
            y = pd.Series(y, index=X)

        self.X0 = X0
        self.d = d

        self.set_seq_length(len(X0), n_alleles, alphabet)
        self.set_alphabet_type(
            alphabet_type, n_alleles=n_alleles, alphabet=alphabet
        )

        genotypes = self.get_genotypes()
        adjacency_matrix = self.calc_adjacency_matrix(genotypes)
        self.init_space(adjacency_matrix, state_labels=genotypes)

        if y is not None:
            if X is None:
                X = self.genotypes
            self.set_y(X, y)

    @property
    def is_regular(self):
        return False

    @property
    def n_genotypes(self):
        return self.n_states

    @property
    def genotypes(self):
        return self.state_labels

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
                    genotypes.append("".join(X_i))
                    if len(genotypes) > MAX_STATES:
                        raise ValueError("Sequence space too big")
        genotypes = np.array(genotypes)
        return genotypes


class ProductSpace(DiscreteSpace):
    """
    General class for constructing spaces as Cartesian products
    of smaller subspaces, each characterized by a set of elementary graphs.

    Parameters
    ----------
    elementary_graphs : list of scipy.sparse.csr_matrix
        List of adjacency matrices (in CSR format) representing the elementary
        graphs that define the subspaces.

    y : array-like of shape (n,), optional
        Array containing the phenotypic values associated with each
        combination of states in the resulting space. If `y` is None,
        no phenotypic values will be stored.

    state_labels : list, optional
        List of labels associated with each possible state in the product space.
        If `state_labels` is None, numeric labels will be assigned by default.

    """

    def __init__(self, elementary_graphs, y=None, state_labels=None):
        self.set_dim_sizes(elementary_graphs)
        adjacency_matrix = self.calc_adjacency_matrix(elementary_graphs)

        self.states = self.calc_states(state_labels=state_labels)
        state_labels = np.array(
            ["-".join([str(x) for x in seq]) for seq in self.states]
        )

        self.init_space(adjacency_matrix, y=y, state_labels=state_labels)

    def set_dim_sizes(self, elementary_graphs):
        self.graph_sizes = [adj_m.shape[0] for adj_m in elementary_graphs]

    def calc_states(self, state_labels=None):
        if state_labels is None:
            state_labels = [list(range(s)) for s in self.graph_sizes]

        state_labels = np.array([x for x in get_product_states(state_labels)])
        return state_labels

    def calc_adjacency_matrix(self, elementary_graphs):
        adjacency_matrix = calc_cartesian_product(elementary_graphs)
        return adjacency_matrix


class GridSpace(ProductSpace):
    """
    N-dimensional grid discrete space.

    A discrete space formed by the Cartesian product of one-dimensional
    spaces of ordered n-states, represented by a line graph.

    Parameters
    ----------
    length: int or array-like
        The number of states across each dimension of the grid. If an integer
        is provided, all dimensions of the grid will have the same length. If
        an array-like of lengths is provided, they will be used to form a grid
        with the specified dimensions, and the `ndim` argument will be ignored.

    ndim: int
        The number of dimensions in the grid when a single `length` value is provided.

    y: array-like of shape (length ** ndim,) or None
        Phenotypic values associated with each possible state.

    """

    def __init__(self, length, y=None, ndim=2):
        self.length = length
        self.ndim = ndim

        if isinstance(length, int):
            elementary_graphs = [self.calc_elementary_graph(length)] * ndim
        else:
            elementary_graphs = [
                self.calc_elementary_graph(sl) for sl in length
            ]
        super().__init__(elementary_graphs, y=y)

    def calc_elementary_graph(self, length):
        states = np.arange(length)
        i = np.append(states[:-1], states[1:])
        j = np.append(states[1:], states[:-1])
        data = np.ones(i.shape[0])
        m = csr_matrix((data, (i, j)))
        return m

    def set_peaks(self, positions, sigma=1):
        """
        Set peaks in the grid space by assigning function values based on
        distances from specified positions.

        Parameters
        ----------
        positions : array-like of shape (n_peaks, ndim)
            Coordinates of the peaks in the grid space. Each row represents
            the position of a peak in the n-dimensional space.

        sigma : float, optional, default=1
            Controls the spread of the peaks. Smaller values result in sharper
            peaks, while larger values create broader peaks.
        """
        distances = np.array(
            [np.abs(self.states - pos).sum(1) for pos in positions]
        ).T
        y = np.exp(logsumexp(-distances / sigma, axis=1))
        self.set_y(y)

    @property
    def nodes_df(self):
        nodes_df = pd.DataFrame(
            self.states,
            columns=[str(i + 1) for i in range(self.ndim)],
            index=self.state_labels,
        )
        if hasattr(self, "y"):
            nodes_df["function"] = self.y
        return nodes_df


class SequenceSpace(GeneralSequenceSpace, ProductSpace):
    """
    Space of all possible sequences of certain length.

    Class for creating a Sequence space characterized by having sequences as
    states. States are connected in the discrete space if they differ by a
    single position in the sequence. It can be created in two different ways:

    1. From a set of sequences and function values (`X`, `y`).
    2. By specifying the properties of the sequence space (alphabet, sequence
       length, number of alleles per site, and type of alphabet).

    Parameters
    ----------
    X : array-like of shape (n_genotypes,), optional
        Sequences to use as state labels of the discrete sequence space.

    y : array-like of shape (n_genotypes,), optional
        Quantitative phenotype or fitness associated with each genotype.

    seq_length : int, optional
        Length of the sequences in the sequence space. If not provided, it will
        be inferred from `alphabet` or `n_alleles`.

    n_alleles : list of int, optional
        List containing the number of alleles present at each site in the
        sequence space. This can only be specified for `alphabet_type='custom'`.

    alphabet_type : str, default='dna'
        Type of sequence. Options are {'dna', 'rna', 'protein', 'custom'}.

    alphabet : list of lists, optional
        A list where each element is itself a list containing the different
        alleles allowed at each site. The number and type of alleles can vary
        across sites.

    stop_y : float, optional
        Value of the function assigned to protein sequences with an in-frame
        stop codon. If provided, the protein alphabet will be extended to
        include `*` for stop codons.

    Attributes
    ----------
    n_genotypes : int
        Number of states in the complete sequence space.

    genotypes : array-like of shape (n_genotypes,)
        Genotype labels in the sequence space.

    adjacency_matrix : scipy.sparse.csr_matrix of shape (n_genotypes, n_genotypes)
        Sparse matrix representing the adjacency relationships between genotypes.
        The (i, j) entry contains a 1 if genotypes `i` and `j` differ by a single
        mutation, and 0 otherwise.

    y : array-like of shape (n_genotypes,), optional
        Quantitative phenotype or fitness associated with each genotype.

    is_regular : bool
        Indicates whether the resulting Hamming graph is regular, i.e., whether
        every site has the same number of alleles.
    """

    def __init__(
        self,
        X=None,
        y=None,
        seq_length=None,
        n_alleles=None,
        alphabet_type="dna",
        alphabet=None,
        stop_y=None,
    ):
        self._init(
            X=X,
            y=y,
            seq_length=seq_length,
            n_alleles=n_alleles,
            alphabet_type=alphabet_type,
            alphabet=alphabet,
            stop_y=stop_y,
        )

    def __str__(self):
        s = "Sequence Space:\n"
        s += "\tType: {}\n".format(self.alphabet_type)
        s += "\tSequence length: {}\n".format(self.seq_length)
        s += "\tNumber of alleles per site: {}\n".format(self.n_alleles)
        s += "\tGenotypes: {}\n".format(self.format_list_ends(self.genotypes))
        if hasattr(self, "y"):
            s += "\tFunction y: {}".format(self.format_list_ends(self.y))
        else:
            s += "\tFunction y: undefined"
        return s

    def _init(
        self,
        X=None,
        y=None,
        seq_length=None,
        n_alleles=None,
        alphabet_type="dna",
        alphabet=None,
        stop_y=None,
    ):
        if X is not None and y is not None:
            config = guess_space_configuration(X, ensure_full_space=True)
            seq_length = config["length"]
            alphabet_type = config["alphabet_type"]
            alphabet = config["alphabet"]
            y = pd.Series(y, index=X)

        self.set_seq_length(seq_length, n_alleles, alphabet)
        self.set_alphabet_type(
            alphabet_type,
            n_alleles=n_alleles,
            alphabet=alphabet,
            add_stop=stop_y is not None,
        )
        self.n_states = np.prod(self.n_alleles)

        msg = "Sequence space is too big to handle ({})".format(self.n_states)
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
        return np.unique(self.n_alleles).shape[0] == 1

    def get_single_mutant_matrix(self, sequence, center=False):
        """
        Calculate the effects of single point mutations from a focal sequence.

        Parameters
        ----------
        sequence : str
            The sequence from which to compute all single point mutant effects.

        center : bool, optional, default=False
            If True, the results will be centered by position, ensuring that the
            mean of allelic effects at each position is 0. If False, the focal
            sequence will have a value of 0, and the results will represent
            mutational effects relative to it.

        Returns
        -------
        output : pd.DataFrame of shape (seq_length, total_alleles)
            A DataFrame containing the mutational or allelic effects for each
            allele across all sequence positions.
        """
        seqy = self.get_y([sequence])
        data = []
        for i in range(self.seq_length):
            alleles = self.alphabet[i]
            mutants = [sequence[:i] + a + sequence[i + 1 :] for a in alleles]
            dy = self.get_y(mutants) - seqy
            if center:
                dy = dy - dy.mean()
            data.append(dict(zip(alleles, dy)))
        data = pd.DataFrame(data)
        return data

    def to_nucleotide_space(self, codon_table="Standard", alphabet_type="dna"):
        """
        Convert a protein sequence space into a nucleotide sequence space.

        This method transforms a protein sequence space into a nucleotide sequence
        space using a specified codon table for translation. The resulting nucleotide
        space will have 4 alleles per site and 3 times the number of sites as the
        original protein space. It assumes that the function associated with each
        nucleotide sequence depends only on the protein sequence it encodes.

        Parameters
        ----------
        codon_table : str or Bio.Data.CodonTable
            The NCBI code for an existing genetic code or a custom CodonTable
            object used to translate nucleotide sequences into proteins.

        alphabet_type : str, optional, default='dna'
            The type of nucleotide sequence to use in the resulting space.
            Must be one of {'dna', 'rna'}.

        Returns
        -------
        SequenceSpace
            A nucleotide sequence space with the specified properties.
        """

        msg = "Only protein spaces can be transformed to nucleotide space"
        msg += " through a codon model: {} not allowed".format(
            self.alphabet_type
        )
        check_error(self.alphabet_type == "protein", msg)

        msg = '`alphabet_type` must be one of ["dna", "rna"]'
        check_error(alphabet_type in ["dna", "rna"], msg)

        nc_space = SequenceSpace(
            seq_length=3 * self.seq_length, alphabet_type=alphabet_type
        )
        prot = pd.Series(
            translate_seqs(nc_space.genotypes, codon_table),
            index=nc_space.genotypes,
        )
        nc_space.protein_seqs = prot.values
        y = pd.Series(self.y, index=self.genotypes)
        y = y.reindex(prot).values

        if np.any(np.isnan(y)):
            msg = "Make sure to include all protein sequences including stops"
            raise ValueError(msg)

        nc_space.set_y(nc_space.genotypes, y)
        return nc_space

    def remove_codon_incompatible_transitions(self, codon_table="Standard"):
        """
        Recalculate the adjacency matrix to allow only codon-compatible transitions
        in a protein sequence space.

        This method updates the adjacency matrix of the sequence space to ensure
        that transitions between states are compatible with the specified codon table.
        Only transitions that result in valid amino acid substitutions according to
        the codon table will be allowed.

        Parameters
        ----------
        codon_table : str or Bio.Data.CodonTable
            The NCBI code for an existing genetic code or a custom CodonTable
            object used to translate nucleotide sequences into proteins.
        """
        msg = "alphabet must be at least a subset of the protein alphabet"
        check_error(guess_alphabet_type(self.alphabet) == "protein", msg)
        self.adjacency_matrix = self.calc_adjacency_matrix(
            codon_table=codon_table
        )

    def calc_transitions(self, codon_table):
        seqs = ["".join(x) for x in product(DNA_ALPHABET, repeat=3)]

        transitions = defaultdict(lambda: defaultdict(lambda: 0))
        for codon1, codon2 in product(seqs, repeat=2):
            d = np.sum([x != y for x, y in zip(codon1, codon2)])
            if d != 1:
                continue
            aa1, aa2 = translate_seqs([codon1, codon2], codon_table)
            transitions[aa1][aa2] += 1
        transitions = pd.DataFrame(transitions).fillna(0).astype(int)
        transitions = transitions.loc[PROTEIN_ALPHABET, PROTEIN_ALPHABET]
        return transitions

    def _calc_site_matrix(self, alleles, transitions=None):
        n_alleles = len(alleles)
        if transitions is None:
            m = np.ones((n_alleles, n_alleles))
        else:
            m = transitions.loc[alleles, alleles].values
        np.fill_diagonal(m, np.zeros(n_alleles))
        return csr_matrix(m)

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
        return site_Kn

    def calc_adjacency_matrix(self, codon_table=None):
        if self.alphabet_type not in ["protein", "custom"]:
            codon_table = None
        sites_A = self._calc_site_adjacency_matrices(
            self.alphabet, codon_table=codon_table
        )
        adjacency_matrix = calc_cartesian_product(sites_A)
        return adjacency_matrix

    def get_genotypes(self):
        return np.array([x for x in get_seqs_from_alleles(self.alphabet)])


class CodonSpace(SequenceSpace):
    """
    Generate a 3-nucleotide sequence space based on allowed amino acids.

    This class creates a nucleotide sequence space corresponding to the
    provided amino acid constraints using a codon table. Optionally, random
    variation can be added to the nucleotide space.

    Parameters
    ----------
    allowed_aminoacids : str or array-like
        A single amino acid (as a string) or a list/array of allowed amino acids.
    codon_table : str, optional
        The codon table to use for mapping amino acids to nucleotides.
        Default is "Standard".
    add_variation : bool, optional
        If True, adds random variation to the nucleotide space. Default is False.
    seed : int, optional
        Seed for the random number generator, used when `add_variation` is True.
        Default is None.

    """

    def __init__(
        self,
        allowed_aminoacids,
        codon_table="Standard",
        add_variation=False,
        seed=None,
    ):
        super().__init__(alphabet_type="dna", seq_length=3)
        if isinstance(allowed_aminoacids, str):
            allowed_aminoacids = np.array([allowed_aminoacids])

        if not isinstance(allowed_aminoacids, np.ndarray):
            allowed_aminoacids = np.array(allowed_aminoacids)

        prot = pd.Series(
            translate_seqs(self.genotypes, codon_table), index=self.genotypes
        )
        protein_y = np.append(np.ones(20), [0])
        protein_y = pd.Series(protein_y, index=PROTEIN_ALPHABET + ["*"])
        protein_y.loc[allowed_aminoacids] = 2
        y = protein_y.reindex(prot).values

        if add_variation:
            if seed is not None:
                np.random.seed(seed)
            y += 1 / 10 * np.random.normal(size=self.n_genotypes)

        if np.any(np.isnan(y)):
            msg = "Make sure to include all protein sequences including stops"
            raise ValueError(msg)
        self.set_y(self.genotypes, y)
