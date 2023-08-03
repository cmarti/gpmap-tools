#!/usr/bin/env python
import unittest
import numpy as np
import pandas as pd

from os.path import join
from gpmap.src.settings import TEST_DATA_DIR
from gpmap.src.seq import (translate_seqs, guess_alphabet_type,
                           guess_space_configuration, get_custom_codon_table,
                           get_seqs_from_alleles, get_one_hot_from_alleles,
                           generate_freq_reduced_code, transcribe_seqs,
                           msa_to_counts)


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
        
        # Incomplete sequence space
        config = guess_space_configuration(seqs, ensure_full_space=False)
        assert(config['n_alleles'] == [2, 3])
        assert(config['alphabet'] == [['A', 'B'], ['A', 'B', 'C']])
        
        # Make complete landscape with all alleles
        config = guess_space_configuration(seqs, ensure_full_space=False, 
                                           force_regular=True)
        assert(config['n_alleles'] == [3, 3])
        assert(config['alphabet'] == [['0', 'A', 'B'], ['A', 'B', 'C']])
        
        # With different alleles per site but same number
        seqs = np.array(['AA', 'AC',
                         'BA', 'BC'])
        config = guess_space_configuration(seqs, ensure_full_space=False, 
                                           force_regular=True,
                                           force_regular_alleles=False)
        assert(config['n_alleles'] == [2, 2])
        assert(config['alphabet'] == [['A', 'B'], ['A', 'C']])
        
        # Make complete landscape with all alleles
        config = guess_space_configuration(seqs, ensure_full_space=False, 
                                           force_regular=True,
                                           force_regular_alleles=True)
        assert(config['n_alleles'] == [3, 3])
        assert(config['alphabet'] == [['A', 'B', 'C'], ['A', 'B', 'C']])
        
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
    
    def test_get_seqs_from_alleles(self):
        alleles = [['a', 'b'], ['0', '1']]
        seqs = list(get_seqs_from_alleles(alleles))
        assert(np.all(seqs == ['a0', 'a1', 'b0', 'b1']))
    
    def test_get_one_hot_from_alleles(self):
        alphabet = [['A', 'T'], ['C', 'G']]
        onehot = get_one_hot_from_alleles(alphabet)
        exp_onehot = np.array([[1, 0, 1, 0],
                               [1, 0, 0, 1],
                               [0, 1, 1, 0],
                               [0, 1, 0, 1]])
        assert(np.allclose(onehot.todense(), exp_onehot))
        
        alphabet = [['A', 'T', 'C'], ['C', 'G']]
        onehot = get_one_hot_from_alleles(alphabet)
        exp_onehot = np.array([[1, 0, 0, 1, 0],
                               [1, 0, 0, 0, 1],
                               [0, 1, 0, 1, 0],
                               [0, 1, 0, 0, 1],
                               [0, 0, 1, 1, 0],
                               [0, 0, 1, 0, 1]])
        assert(np.allclose(onehot.todense(), exp_onehot))
    
    def test_get_freq_reduced_dict(self):
        seqs = np.array(['AG', 'AA', 'CG', 'CA', 'TG', 'GC'])
        
        code = generate_freq_reduced_code(seqs, 2)
        assert(code[0]['A'] == 'A')
        assert(code[0]['C'] == 'C')
        assert(code[0]['G'] == 'X')
        assert(code[1]['A'] == 'A')
        assert(code[1]['G'] == 'G')
        assert(code[1]['C'] == 'X')
        
        # Changing allele names by new alphabet across all sites
        code = generate_freq_reduced_code(seqs, 2, keep_allele_names=False)
        assert(code[0]['A'] == 'A')
        assert(code[0]['C'] == 'B')
        assert(code[0]['G'] == 'X')
        assert(code[1]['A'] == 'B')
        assert(code[1]['G'] == 'A')
        assert(code[1]['C'] == 'X')
        
        # Using counts
        counts = np.array([1, 1, 1, 1, 5, 5])
        code = generate_freq_reduced_code(seqs, 2, counts=counts)
        assert(code[0]['A'] == 'X')
        assert(code[0]['C'] == 'X')
        assert(code[0]['G'] == 'G')
        assert(code[0]['T'] == 'T')
        assert(code[1]['A'] == 'X')
        assert(code[1]['G'] == 'G')
        assert(code[1]['C'] == 'C')
    
    def test_transcribe_seqs(self):
        seqs = np.array(['AG', 'AA', 'CG', 'CA', 'TG', 'GC'])
        code = generate_freq_reduced_code(seqs, 2)
        
        new_seqs = transcribe_seqs(seqs, code)
        assert(np.all(new_seqs == ['AG', 'AA', 'CG', 'CA', 'XG', 'XX']))
    
    def test_msa_to_counts(self):
        msa = np.array(['AGTGG',
                        'AGTGT',
                        'GGTGG',
                        'ATACC',
                        'ATACC',
                        'ATACG'])
        
        # Regular counting
        X, y = msa_to_counts(msa)
        assert(np.all(X == np.array(['AGTGG', 'AGTGT', 'ATACC', 'ATACG', 'GGTGG'])))
        assert(np.all(y == np.array([1, 1, 2, 1, 1])))
        
        # Count subsequences at some positions
        X, y = msa_to_counts(msa, positions=[1, 2])
        assert(np.all(X == np.array(['GT', 'TA'])))
        assert(np.all(y == np.array([3, 3])))
        
        # Count subsequences at some positions with phylogenetic correction
        X, y = msa_to_counts(msa, positions=[1, 2], phylo_correction=True, max_dist=0.15)
        assert(np.all(X == np.array(['GT', 'TA'])))
        assert(np.all(y == np.array([3., 2.])))
        
        X, y = msa_to_counts(msa, positions=[1, 2], phylo_correction=True, max_dist=0.25)
        assert(np.all(X == np.array(['GT', 'TA'])))
        assert(np.all(y == np.array([4/3., 1.])))
    
        
if __name__ == '__main__':
    import sys;sys.argv = ['', 'SeqTests']
    unittest.main()
