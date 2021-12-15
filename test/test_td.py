#!/usr/bin/env python
import unittest

import numpy as np

from gpmap.td import AdditiveConvolutionalModel


class TDTests(unittest.TestCase):
    def test_sim_seqs(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel(filter_size=4)
        
        seqs = m.simulate_random_seqs(length=5, n_seqs=10)
        assert(len(seqs) == 10)
        for seq in seqs:
            assert(len(seq) == 5)
        
        wt = 'AGGAGG'
        seqs = list(m.simulate_random_mutants(seq=wt, n_seqs=10, p_mut=0.1))
        assert(len(seqs) == 10)
        for seq in seqs:
            assert(seq != wt)
        
        singles = list(m.get_single_mutants(seq))
        assert(len(singles) == 18)
        
        doubles = list(m.get_single_and_double_mutants(seq))
        assert(len(doubles) == 154)
        
        mutants = list(m.simulate_sequences(length=5, seq='AGGAG', p_mut=0.2,
                                            mode='error_pcr', n_seqs=10))
        assert(len(mutants) == 10)
        assert(len(mutants[0]) == 5)
        
        mutants = m.simulate_sequences(length=4, mode='combinatorial')
        assert(len(mutants) == 4**4)
        assert(len(mutants[0]) == 4)
        
        mutants = m.add_flanking_seqs(mutants, n_backgrounds=2)
        assert(len(mutants) == 2*4**4)
        assert(len(mutants[0]) == 10)
    
    def test_sim_data(self):
        np.random.seed(0)
        m = AdditiveConvolutionalModel(filter_size=4)
        seqs = m.simulate_random_seqs(length=5, n_seqs=10)
        seqs = m.add_flanking_seqs(seqs, n_backgrounds=2)
        data = m.simulate_data(seqs)
        assert(data['L'] == 11)
        assert(data['F'] == 13)
        assert(data['C'] == 4)
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'TDTests']
    unittest.main()
