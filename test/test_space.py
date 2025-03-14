#!/usr/bin/env python
import unittest
import pandas as pd
import numpy as np

from tempfile import NamedTemporaryFile
from scipy.sparse import csr_matrix
from scipy.special import comb

from gpmap.seq import hamming_distance, translate_seqs
from gpmap.datasets import DataSet
from gpmap.space import (SequenceSpace, DiscreteSpace, CodonSpace,
                             ProductSpace, GridSpace, HammingBallSpace)


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
    
    def test_hamming_ball_space(self):
        X0 = 'ATG'
        alphabet = [['A', 'C', 'G', 'T']] * len(X0)
        space = HammingBallSpace(X0=X0, alphabet=alphabet, d=2)
        
        # Ensure it contains the right number of genotypes
        n_genotypes = np.sum([comb(3, k) * 3 ** k for k in range(3)])
        assert(space.n_genotypes == n_genotypes)
        
        # Ensure all sequences within the riht distance
        for x in space.genotypes:
            assert(hamming_distance(x, X0) <= 2)
            
        # Build space from list of sequences
        genotypes = space.genotypes.copy()
        np.random.shuffle(genotypes)
        y = np.random.normal(size=genotypes.shape[0])
        space = HammingBallSpace(X0=X0, X=genotypes, y=y)

        assert(space.n_genotypes == n_genotypes)
        for x in space.genotypes:
            assert(hamming_distance(x, X0) <= 2)
        
        # Ensure right adjacency matrix
        space = HammingBallSpace(X0=X0, alphabet=alphabet, d=1)
        degrees = space.adjacency_matrix.sum(axis=0)
        assert(degrees[0, 0] == 9)
        assert(np.all(degrees[0, 1:] == 3))
        
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
        
        with NamedTemporaryFile() as fhand:
            fpath = '{}.csv'.format(fhand.name)
            space.write_csv(fpath)
            
            df = pd.read_csv(fpath, index_col=0)
            codons = ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT']
            assert(np.all(df[df['y'] > 1.5].index.values == codons))
    
    def test_build_space_from_sequences(self):
        data = DataSet('serine').landscape
        codons = ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT']
        s = SequenceSpace(X=data.index.values, y=data.y.values)
        assert(np.all(s.state_labels[s.y > 1.5] == codons))

        # Shuffle sequences before
        labels = data.index.values.copy()
        np.random.shuffle(labels)
        data = data.loc[labels, :]
        
        s = SequenceSpace(X=data.index.values, y=data['y'].values)
        assert(np.all(sorted(s.state_labels[s.y > 1.5]) == sorted(codons)))
    
    def test_get_single_mutant_matrix(self):
        space = DataSet('gb1').to_sequence_space()
        m = space.get_single_mutant_matrix('WWLG', center=False)
        assert(m.loc[0, 'W'] == 0)
        assert(m.loc[1, 'W'] == 0)
        assert(m.loc[2, 'L'] == 0)
        assert(m.loc[3, 'G'] == 0)
        
        m = space.get_single_mutant_matrix('WWLG', center=True)
        assert(m.loc[0, 'W'] != 0)
        assert(m.loc[1, 'W'] != 0)
        assert(m.loc[2, 'L'] != 0)
        assert(m.loc[3, 'G'] != 0)
        assert(np.allclose(m.mean(1), 0))
    
    def test_to_codon_space(self):
        data = DataSet('serine').landscape
        data['protein'] = translate_seqs(data.index.values)
        data = data.groupby(['protein']).mean().drop('*', axis=0)
        
        # Fail if stop_y is not provided
        try:
            s = SequenceSpace(X=data.index.values, y=data.y.values)
            s = s.to_nucleotide_space(codon_table='Standard')
            self.fail()
        except ValueError:
            pass
        
        # Fail if there are missing protein sequences
        try:
            s = SequenceSpace(X=data.index.values[:-1],
                              y=data.y.values[:-1], stop_y=0)
            s = s.to_nucleotide_space(codon_table='Standard')
            self.fail()
        except ValueError:
            pass
        
        # Correct run        
        s = SequenceSpace(X=data.index.values, y=data.y.values, stop_y=0)
        s = s.to_nucleotide_space(codon_table='Standard')
        assert(s.n_genotypes == 64)
        assert(hasattr(s,'protein_seqs'))
        assert(np.unique(s.protein_seqs).shape[0] == 21)
    
    def test_write_edges(self):
        s = SequenceSpace(seq_length=6, alphabet_type='dna')
        with NamedTemporaryFile('w') as fhand:
            s.write_edges(fhand.name + '.npz')
            s.write_edges(fhand.name + '.csv')
            
    def test_calc_path(self):
        X = np.array(['AAA', 'AAB', 'ABA', 'ABB',
                      'BAA', 'BAB', 'BBA', 'BBB'])
        y = np.array([3, 2, 1, 2, 1, 1, 1, 3])
        s = SequenceSpace(X=X, y=y)
        path = s.calc_max_min_path('AAA', 'BBB', allow_bypasses=False)
        assert(np.all(path == ['AAA', 'AAB', 'ABB', 'BBB']))
        
        y = np.array([3, 2, 1, 2, 3, 1, 3, 3])
        s = SequenceSpace(X=X, y=y)
        path = s.calc_max_min_path('AAA', 'BBB', allow_bypasses=False)
        assert(np.all(path == ['AAA', 'BAA', 'BBA', 'BBB']))
        
        X = np.array(['AA', 'AB', 'AC',
                      'BA', 'BB', 'BC',
                      'CA', 'CB', 'CC'])
        y = np.array([3, 2, 2,
                      3, 1, 3,
                      2, 3, 3])
        s = SequenceSpace(X=X, y=y)
        path = s.calc_max_min_path('AA', 'CC', allow_bypasses=False)
        assert(np.all(path == ['AA', 'AC', 'CC']))
        
        path = s.calc_max_min_path('AA', 'CC', allow_bypasses=True)
        assert(np.all(path == ['AA', 'BA', 'BC', 'CC']))
        

if __name__ == '__main__':
    import sys;sys.argv = ['', 'SpaceTests']
    unittest.main()
