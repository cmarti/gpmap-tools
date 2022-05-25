#!/usr/bin/env python
import unittest
from os.path import join

import numpy as np
import pandas as pd

from gpmap.utils import LogTrack
from gpmap.inference import VCregression
from gpmap.src.settings import TEST_DATA_DIR, BIN_DIR
from subprocess import check_call
from gpmap.src.space import SequenceSpace
from gpmap.src.seq import translate_seqs, guess_alphabet_type, guess_space_configuration


class SeqTests(unittest.TestCase):
    def test_guess_alphabet(self):
        alphabet_type = guess_alphabet_type(['A', 'C', 'G'])
        assert(alphabet_type == 'dna')
        
        alphabet_type = guess_alphabet_type(['A', 'C', 'G', 'T'])
        assert(alphabet_type == 'dna')
        
        alphabet_type = guess_alphabet_type([['A', 'C', 'G', 'T'],
                                             ['A', 'C', 'G', 'T']])
        assert(alphabet_type == 'dna')
        
        alphabet_type = guess_alphabet_type(['A', 'C', 'G', 'U'])
        assert(alphabet_type == 'rna')
        
        alphabet_type = guess_alphabet_type(['A', 'K', 'G', 'R'])
        assert(alphabet_type == 'protein')
        
        alphabet_type = guess_alphabet_type([['A', 'K'], ['G', 'R']])
        assert(alphabet_type == 'protein')
        
        alphabet_type = guess_alphabet_type(['0', '1'])
        assert(alphabet_type == 'custom')
        
    def test_guess_configuration(self):
        # Test with a simple scenario
        seqs = np.array(['AA', 'AB', 'AC',
                         'BA', 'BB', 'BC'])
        config = guess_space_configuration(seqs)
        assert(config['length'] == 2)
        assert(config['n_alleles'] == [2, 3])
        assert(config['alphabet_type'] == 'custom')
        
        # Fail when sequence space is not complete
        seqs = np.array(['AA', 'AB', 'AC',
                         'BA', 'BB'])
        try:
            config = guess_space_configuration(seqs)
            self.fail('guess_configuration did not fail with incomplete space')
        except ValueError:
            pass
        
        # With a real file
        fpath = join(TEST_DATA_DIR, 'gfp.short.csv')
        data = pd.read_csv(fpath).sort_values('pseudo_prot').set_index('pseudo_prot')
        config = guess_space_configuration(data.index.values)
        assert(config['length'] == 13)
        assert(config['n_alleles'] == [2, 1, 8, 1, 2, 2, 6, 1, 1, 1, 2, 1, 2])
        assert(config['alphabet_type'] == 'protein')
    
        # With another real file
        fpath = join(TEST_DATA_DIR, 'test.csv')
        data = pd.read_csv(fpath, index_col=0)
        config = guess_space_configuration(data.index.values)
        assert(config['length'] == 7)
        assert(config['n_alleles'] == [2, 3, 2, 2, 3, 2, 3])
        assert(config['alphabet_type'] == 'protein')
        
    def test_translate(self):
        dna = np.array(['ATGUGA', 'ATGUGAATG'])
        
        # Standard codon table
        protein = translate_seqs(dna, codon_table='Standard')
        assert(np.all(protein == ['M*', 'M*M']))
        
        # Test other codon tables with NCBI identifiers
        protein = translate_seqs(dna, codon_table=11)
        assert(np.all(protein == ['M*', 'M*M']))
    
    def xtest_get_neighbors(self):
        v = Visualization(3, alphabet_type='rna')
        seq = 'AAA'
        
        idxs = v.get_neighborhood_idxs(seq, max_distance=1)
        seqs = v.genotypes[idxs]
        assert(np.all(seqs == ['AAA', 'AAC', 'AAG', 'AAU', 'ACA',
                               'AGA', 'AUA', 'CAA', 'GAA', 'UAA']))
        
        idxs = v.get_neighborhood_idxs(seq, max_distance=2)
        seqs = v.genotypes[idxs]
        for seq in seqs:
            assert('A' in seq)
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'SeqTests']
    unittest.main()
