#!/usr/bin/env python
import unittest

import pandas as pd
import numpy as np

from gpmap.src.space import SequenceSpace, DiscreteSpace, CodonSpace,\
    read_sequence_space_csv
from scipy.sparse.csr import csr_matrix
from gpmap.src.settings import TEST_DATA_DIR
from os.path import join


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
        
    def test_discrete_space_line_function(self):    
        adjacency_matrix = csr_matrix(np.array([[0, 1, 0],
                                                [1, 0, 1],
                                                [0, 1, 0]]))
        function = np.array([1, 0, 1])
        space = DiscreteSpace(csr_matrix(adjacency_matrix),
                              function=function)
        assert(np.all(function == space.function))
    
    def test_discrete_space_line_function_error(self):
        adjacency_matrix = csr_matrix(np.array([[0, 1, 0],
                                                [1, 0, 1],
                                                [0, 1, 0]]))    
        wrong_functions = [np.array([0, 1]),
                           np.array([0, 1, 1, 1])]
        for function in wrong_functions:
            try:
                DiscreteSpace(csr_matrix(adjacency_matrix), function=function)
                self.fail('DiscreteSpace did not capture erroneous function')
            except ValueError:
                pass
    
    def test_codon_space(self):
        s = CodonSpace(['S'], add_variation=True, seed=0)
        assert(s.n_states == 64)
        assert(s.state_labels[0] == 'AAA')
        codons = ['AGC', 'AGU', 'UCA', 'UCC', 'UCG', 'UCU']
        assert(np.all(s.state_labels[s.function > 1.5] == codons))
        
        s = CodonSpace(['K'], add_variation=True, seed=0)
        assert(np.all(s.state_labels[s.function > 1.5] == ['AAA', 'AAG']))
        
    def test_n_alleles(self):
        s = SequenceSpace(seq_length=2, alphabet_type='dna')
        assert(s.n_alleles == [4, 4])
        
        s = SequenceSpace(2, alphabet_type='protein')
        assert(s.n_alleles == [20, 20])
        
        s = SequenceSpace(2, n_alleles=2, alphabet_type='custom')
        assert(s.n_alleles == [2, 2])

        # Raise error when n_alleles is specified for non custom alphabets        
        try:
            s = SequenceSpace(2, n_alleles=4)
            self.fail()
        except ValueError:
            pass
        
        # Try variable number of alleles per site
        s = SequenceSpace(4, n_alleles=[2, 4, 2, 2], alphabet_type='custom')
        assert(s.n_states == 32)
        assert(s.genotypes.shape[0] == 32)
    
    def test_adjacency_matrix(self):
        s = SequenceSpace(1, alphabet_type='dna')
        A = s.adjacency_matrix.todense()
        assert(np.all(np.diag(A) == 0))
        assert(np.all(A + np.eye(4) == 1))
        
        s = SequenceSpace(2, 2, alphabet_type='custom')
        A = s.adjacency_matrix.todense()
        assert(np.all(np.diag(A) == 0))
        assert(np.all(A + np.eye(4) + np.fliplr(np.eye(4)) == 1))
    
    def test_get_neighbors(self):
        s = SequenceSpace(3, alphabet_type='rna')
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
        codons = ['AGC', 'AGU', 'UCA', 'UCC', 'UCG', 'UCU']
        assert(np.all(df[df['function'] > 1.5].index.values == codons))
    
    def test_read_space(self):
        fpath = join(TEST_DATA_DIR, 'serine.csv')
        s = read_sequence_space_csv(fpath, function_col='function')
        
        codons = ['AGC', 'AGU', 'UCA', 'UCC', 'UCG', 'UCU']
        assert(np.all(s.state_labels[s.function > 1.5] == codons))

        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'SpaceTests']
    unittest.main()
