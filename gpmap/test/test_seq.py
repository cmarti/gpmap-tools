#!/usr/bin/env python
import unittest

from os.path import join

import numpy as np
import pandas as pd

from gpmap.src.settings import TEST_DATA_DIR
from gpmap.src.seq import (translate_seqs, guess_alphabet_type,
                           guess_space_configuration, get_custom_codon_table)


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
        dna = np.array(['ATGTGA', 'ATGTGAATG'])
        
        # Standard codon table
        protein = translate_seqs(dna, codon_table='Standard')
        assert(np.all(protein == ['M*', 'M*M']))
        
        # Test other codon tables with NCBI identifiers
        protein = translate_seqs(dna, codon_table=11)
        assert(np.all(protein == ['M*', 'M*M']))
    
    def test_get_custom_codon_table(self):
        fpath = join(TEST_DATA_DIR, 'code_6037.csv')
        aa_mapping = pd.read_csv(fpath)
        codon_table = get_custom_codon_table(aa_mapping)
        
        assert(codon_table.__str__())
        assert(codon_table.stop_codons == ['TAA', 'TAG', 'TGA'])

        # Ensure that translation works with the new code        
        dna = np.array(['ATGTGA', 'ATGTGATGG'])
        protein = translate_seqs(dna, codon_table=codon_table)
        assert(np.all(protein == ['S*', 'S*M']))
    
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'SeqTests']
    unittest.main()
