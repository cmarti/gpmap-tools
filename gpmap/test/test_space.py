#!/usr/bin/env python
import unittest

import pandas as pd
import numpy as np

from gpmap.src.space import SequenceSpace, DiscreteSpace, CodonSpace,\
    ProductSpace, GridSpace
from scipy.sparse.csr import csr_matrix
from gpmap.src.settings import TEST_DATA_DIR
from os.path import join
from tempfile import NamedTemporaryFile


class SpaceTests(unittest.TestCase):
    def test_discrete_space_errors(self):
        # Fail when adjacency matrix is not well formed
        wrong_matrices = [np.array([0, 1]), 
                          np.array([[0, -1], [1, 0]]),
                          np.array([[1, 1], [1, 1]]),
                          np.array([[0, 0], [1, 0]])]
        for adjacency_matrix in wrong_matrices:
            try:
                DiscreteSpace(csr_matrix(adjacency_matrix))
                self.fail('DiscreteSpace did not capture erroneous matrix')
            except ValueError:
                pass
    
    def test_discrete_space_line(self):    
        adjacency_matrix = csr_matrix(np.array([[0, 1, 0],
                                                [1, 0, 1],
                                                [0, 1, 0]]))
        space = DiscreteSpace(csr_matrix(adjacency_matrix))
        assert(space.n_states == 3)
        assert(np.all(space.state_labels == ['0', '1', '2']))
    
    def test_discrete_space_str(self):
        adjacency_matrix = csr_matrix(np.array([[0, 1, 0],
                                                [1, 0, 1],
                                                [0, 1, 0]]))
        space = DiscreteSpace(csr_matrix(adjacency_matrix))
        expected_str = 'Discrete Space:\n\tNumber of states: 3\n\tState labels: '
        expected_str += '[0,1,2]\n\tStates function values: undefined\n\t'
        expected_str += 'Number of edges: 4'
        assert(space.__str__() == expected_str)
        
    def test_discrete_space_line_function(self):    
        adjacency_matrix = csr_matrix(np.array([[0, 1, 0],
                                                [1, 0, 1],
                                                [0, 1, 0]]))
        y = np.array([1, 0, 1])
        space = DiscreteSpace(csr_matrix(adjacency_matrix),
                              y=y)
        assert(np.all(y == space.y))
    
    def test_discrete_space_line_function_error(self):
        adjacency_matrix = csr_matrix(np.array([[0, 1, 0],
                                                [1, 0, 1],
                                                [0, 1, 0]]))    
        wrong_functions = [np.array([0, 1]),
                           np.array([0, 1, 1, 1])]
        for function in wrong_functions:
            try:
                DiscreteSpace(csr_matrix(adjacency_matrix), y=function)
                self.fail('DiscreteSpace did not capture erroneous function')
            except ValueError:
                pass
    
    def test_product_space(self):
        m = csr_matrix(np.array([[0, 1, 0],  [1, 0, 1], [0, 1, 0]]))
        ms = [m, m]
        space = ProductSpace(ms)
        assert(space.n_states == 9)
        assert(space.n_edges == 24)
    
    def test_grid_space(self):
        space = GridSpace(length=3)
        assert(space.n_states == 9)
        assert(space.n_edges == 24)
        
        peaks = np.array([[1, 1], [0, 0]])
        space.set_peaks(peaks)
        
        y = np.array([1.13533528, 0.73575888, 0.27067057,
                      0.73575888, 1.13533528, 0.41766651,
                      0.27067057, 0.41766651, 0.15365092])
        assert(np.allclose(space.y, y))
        assert(np.all(space.nodes_df.columns == ['1', '2', 'function']))
    
    def test_seq_space_str(self):
        space = SequenceSpace(seq_length=1, alphabet_type='protein')
        expected_str = 'Sequence Space:\n\tType: protein\n\tSequence length: 1\n\t'
        expected_str += 'Number of alleles per site: [20]\n\tGenotypes: '
        expected_str += '[A,C,D,...,V,W,Y]\n\tFunction y: undefined'
        assert(space.__str__() == expected_str)
    
    def test_calc_transitions(self):
        space = SequenceSpace(seq_length=1, alphabet_type='protein')
        transitions = space.calc_transitions(codon_table='Standard')
        n_neighbors = transitions.sum(0)
        
        assert(np.all(transitions.T == transitions))
        assert(n_neighbors['W'] == 7)
        assert(n_neighbors['Y'] == 14)
        assert(n_neighbors['A'] == 36)
        assert(n_neighbors['I'] == 27)
        assert(n_neighbors['M'] == 9)
        assert(n_neighbors['S'] == 51)
        assert(n_neighbors['L'] == 51)
        assert(n_neighbors['R'] == 52)
    
    def test_calc_laplacian(self):
        space = SequenceSpace(seq_length=1, alphabet_type='dna')
        space.calc_laplacian()
        assert(np.all(np.diag(space.laplacian.todense()) == 3))
        assert(np.all(space.laplacian.sum(1) == 0))
        
        space = SequenceSpace(seq_length=2, alphabet_type='dna')
        space.calc_laplacian()
        assert(np.all(np.diag(space.laplacian.todense()) == 6))
        assert(np.all(space.laplacian.sum(1) == 0))
    
    def test_calc_site_adjacency_matrices(self):
        space = SequenceSpace(seq_length=1, alphabet_type='protein')
        
        # Without taking into account genetic code
        sites_A = space._calc_site_adjacency_matrices(alleles=[['T', 'V']])
        m = sites_A[0]
        assert(np.all(m == np.array([[0, 1], [1, 0]])))
        assert(np.all(m.data == 1))
        
        # Inaccessible aminoacids
        sites_A = space._calc_site_adjacency_matrices(alleles=[['T', 'V']], codon_table='Standard')
        m = sites_A[0]
        assert(np.all(m.todense() == 0))
        assert(np.all(m.data == 1))
        
        # Accessible aminoacids
        sites_A = space._calc_site_adjacency_matrices(alleles=[['A', 'V']], codon_table='Standard')
        m = sites_A[0]
        assert(np.all(m == np.array([[0, 1], [1, 0]])))
        assert(np.all(m.data == 1))
        
        # Indirect T-V connection
        sites_A = space._calc_site_adjacency_matrices(alleles=[['A', 'V', 'T']], codon_table='Standard')
        m = sites_A[0]
        assert(np.all(m == np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])))
        assert(np.all(m.data == 1))
    
    def test_protein_space_codon_restricted(self):
        alphabet = [['A', 'V'],
                    ['A', 'V', 'T']]
        space = SequenceSpace(alphabet=alphabet, alphabet_type='custom')
        space.remove_codon_incompatible_transitions(codon_table='Standard')
        assert(np.all(space.genotypes == ['AA', 'AV', 'AT', 'VA', 'VV', 'VT']))
        
        m = space.adjacency_matrix.tocsr()
        
        # Ensure V-T transitions are not allowed in either direction
        assert(m[1, 2] == 0)
        assert(m[2, 1] == 0)
        assert(m[4, 5] == 0)
        assert(m[5, 4] == 0)

    def test_codon_space(self):
        s = CodonSpace(['S'], add_variation=True, seed=0)
        assert(s.n_states == 64)
        assert(s.state_labels[0] == 'AAA')
        codons = ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT']
        assert(np.all(s.state_labels[s.y > 1.5] == codons))
        
        s = CodonSpace(['K'], add_variation=True, seed=0)
        assert(np.all(s.state_labels[s.y > 1.5] == ['AAA', 'AAG']))
        
    def test_n_alleles(self):
        s = SequenceSpace(seq_length=2, alphabet_type='dna')
        assert(s.n_alleles == [4, 4])
        
        s = SequenceSpace(seq_length=2, alphabet_type='protein')
        assert(s.n_alleles == [20, 20])
        
        s = SequenceSpace(seq_length=2, n_alleles=2, alphabet_type='custom')
        assert(s.n_alleles == [2, 2])

        # Raise error when n_alleles is specified for non custom alphabets        
        try:
            s = SequenceSpace(seq_length=2, n_alleles=4)
            self.fail()
        except ValueError:
            pass
        
        # Try variable number of alleles per site
        s = SequenceSpace(seq_length=4, n_alleles=[2, 4, 2, 2], alphabet_type='custom')
        assert(s.n_states == 32)
        assert(s.genotypes.shape[0] == 32)
    
    def test_alphabet(self):
        alphabet = [['A', 'B'], 
                    ['C', 'D', 'E']]
        
        s = SequenceSpace(alphabet=alphabet, alphabet_type='custom')
        states = ['AC', 'AD', 'AE', 'BC', 'BD', 'BE']
        assert(np.all(s.state_labels == states))
        
        A = s.adjacency_matrix.todense()
        expected_A = np.array([[0, 1, 1, 1, 0, 0],
                               [1, 0, 1, 0, 1, 0],
                               [1, 1, 0, 0, 0, 1],
                               [1, 0, 0, 0, 1, 1],
                               [0, 1, 0, 1, 0, 1],
                               [0, 0, 1, 1, 1, 0]])
        assert(np.all(A == expected_A))
        
        # Switch order of sites
        alphabet = [['C', 'D', 'E'],
                    ['A', 'B']]
        
        s = SequenceSpace(alphabet=alphabet, alphabet_type='custom')
        states = ['CA', 'CB', 'DA', 'DB', 'EA', 'EB']
        assert(np.all(s.state_labels == states))
        
        A = s.adjacency_matrix.todense()
        expected_A = np.array([[0, 1, 1, 0, 1, 0],
                               [1, 0, 0, 1, 0, 1],
                               [1, 0, 0, 1, 1, 0],
                               [0, 1, 1, 0, 0, 1],
                               [1, 0, 1, 0, 0, 1],
                               [0, 1, 0, 1, 1, 0]])
        assert(np.all(A == expected_A))
    
    def test_adjacency_matrix(self):
        s = SequenceSpace(seq_length=1, alphabet_type='dna')
        A = s.adjacency_matrix.todense()
        assert(np.all(np.diag(A) == 0))
        assert(np.all(A + np.eye(4) == 1))
        
        s = SequenceSpace(seq_length=2, n_alleles=2, alphabet_type='custom')
        A = s.adjacency_matrix.todense()
        assert(np.all(np.diag(A) == 0))
        assert(np.all(A + np.eye(4) + np.fliplr(np.eye(4)) == 1))
    
    def test_get_neighbors(self):
        s = SequenceSpace(seq_length=3, alphabet_type='rna')
        seq = 'AAA'
        
        seqs = s.get_neighbors(seq, max_distance=1)
        assert(np.all(seqs == ['AAA', 'AAC', 'AAG', 'AAU', 'ACA',
                               'AGA', 'AUA', 'CAA', 'GAA', 'UAA']))
        
        seqs = s.get_neighbors(seq, max_distance=2)
        for seq in seqs:
            assert('A' in seq)
    
    def test_write_space(self):
        space = CodonSpace(['S'], add_variation=True, seed=0)
        fpath = join(TEST_DATA_DIR, 'serine.csv')
        space.write_csv(fpath)
        
        df = pd.read_csv(fpath, index_col=0)
        codons = ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT']
        assert(np.all(df[df['y'] > 1.5].index.values == codons))
    
    def test_build_space_from_sequences(self):
        fpath = join(TEST_DATA_DIR, 'serine.csv')
        data = pd.read_csv(fpath, index_col=0)
        codons = ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT']
        
        s = SequenceSpace(X=data.index.values, y=data['y'].values)
        assert(np.all(s.state_labels[s.y > 1.5] == codons))

        # Shuffle sequences before
        labels = data.index.values.copy()
        np.random.shuffle(labels)
        data = data.loc[labels, :]
        
        s = SequenceSpace(X=data.index.values, y=data['y'].values)
        assert(np.all(sorted(s.state_labels[s.y > 1.5]) == sorted(codons)))
    
    def test_calculate_variance_components(self):
        data = pd.read_csv(join(TEST_DATA_DIR, 'gb1.csv')).set_index('seq')
        space = SequenceSpace(X=data.index.values, y=data.log_binding.values)
        lambdas = space.calc_variance_components()
        assert(np.all(lambdas > 0))
    
    def test_to_codon_space(self):
        fpath = join(TEST_DATA_DIR, 'serine.protein.csv')
        data = pd.read_csv(fpath, index_col=0)
        
        s = SequenceSpace(X=data.index.values, y=data['function'].values)
        s = s.to_nucleotide_space(codon_table='Standard', stop_y=0)
        assert(s.n_genotypes == 64)
        assert(hasattr(s,'protein_seqs'))
        assert(np.unique(s.protein_seqs).shape[0] == 21)
    
    def test_write_edges_npz(self):
        s = SequenceSpace(seq_length=6, alphabet_type='dna')
        with NamedTemporaryFile('w') as fhand:
            s.write_edges_npz(fhand.name)
    
    def test_write_edges_csv(self):
        s = SequenceSpace(seq_length=6, alphabet_type='dna')
        with NamedTemporaryFile('w') as fhand:
            s.write_edges_csv(fhand.name)

        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'SpaceTests']
    unittest.main()
